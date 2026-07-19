# Phase 2 — Warp GPU 선형 정적 탄성 솔버 (구현 스펙)

> Claude Code 사용법: 이 파일과 `bench/` 의 기존 세 모듈을 **먼저 전부 읽고**,
> 아래 수용 기준(Acceptance Criteria)이 **전부 green** 이 될 때까지 GPU pod 터미널에서
> 실제로 실행·반복하며 구현한다. 컴파일 성공은 완료가 아니다. **수치로 증명**해야 한다.

---

## 0. 배경 & 기존 자산 (읽을 것)

- `bench/instrument.py` — `measure()` 컨텍스트로 구간의 벽시계·CPU%·GPU util 수집.
  - `result.cpu_under(20)` → CPU 제약 판정
  - `compare_walltime(gpu_res, cpu_res)` → 속도 비교 + speedup
  - `save_results(path, **named)` → JSON 저장
- `bench/gen_inp.py` — C3D8 외팔보 `.inp`를 메시 크기 스윕으로 생성 (`sweep()`).
  `solver="iterative"` 로 CG 데크도 생성. 규칙: **NFIX = x=0 전 DOF 고정**,
  **NLOAD = x=L 면에 z방향 CLOAD**, 재료 `E=210000, nu=0.3`.
- `bench/run_ccx.py` — ccx를 전 코어로 실행·측정하고 `.dat` 변위를 파싱 (`run_ccx()`).

## 1. 목표

`bench/warp_solve.py` 를 새로 작성. **NVIDIA Warp의 `warp.fem`** 으로 3D 선형 정적
탄성을 **GPU에서** 푼다. `gen_inp.py` 가 만든 `.inp` 를 입력받아 **ccx와 동일한 문제**를
풀고, 변위를 반환한다.

## 2. 하드 제약 = 수용 기준 (4개 모두 통과해야 완료)

1. **GPU-only.** 모든 Warp 배열·커널은 `device="cuda:0"` 상주. solve 루프 안에서
   host↔device 복사·`.numpy()` 금지. solve 구간을 `measure("warp_solve")` 로 감싸고
   **`cpu_avg < 20%` 를 자동 검증**. 넘으면 실패로 처리하고 **어떤 연산이 CPU를 쓰는지 보고**.
2. **성능 하한.** 동일 문제를 `run_ccx` 로 전 코어(100%) 실행한 벽시계보다
   Warp(GPU) 벽시계가 **작거나 같아야** 함. `compare_walltime` 로 판정, speedup 출력.
   공정 비교를 위해 **`solver="iterative"` 데크로 ccx도 CG로** 함께 측정.
   → 단, **§5의 crossover 규정**을 반드시 따를 것 (모든 크기에서 이길 필요 없음).
3. **정확도.** `wp.float64` 로 통일(fp32 금지). 결과 변위를 ccx `.dat` 과 비교해
   **상대오차 1e-3 이내**. 비교 스칼라는 하중단 최대 |u| 및 노드별 L2 상대오차.
4. **효율.** `bsr_cg` 반복 솔버를 **CUDA graph 로 캡처**해 런치 오버헤드·CPU 부하 최소화.

## 3. 구현 힌트

- **입력 파싱:** `meshio` 로 노드/요소를 읽되, `meshio` 가 CalculiX 의 `*BOUNDARY`/`*CLOAD`
  를 못 읽으면 `.inp` 를 직접 파싱하거나 `gen_inp.py` 의 생성 규칙(NFIX/NLOAD/E/nu)을
  그대로 재사용해 BC·하중을 재구성한다.
- **warp.fem 파이프라인:**
  - explicit `Hexmesh` geometry 를 `device="cuda:0"` 에 구성
  - `make_polynomial_space(degree=1, dtype=wp.vec3d)`
  - **변위 기반 선형탄성 bilinear form** 을 직접 정의 (Lamé λ, μ):
    `a(u,v) = ∫ [ 2μ ε(u):ε(v) + λ tr(ε(u)) tr(ε(v)) ] dΩ`,  `ε = sym(∇u)`
    ⚠️ 저장소 예제 `example_mixed_elasticity.py` 는 **mixed/비선형**이라 그대로 못 씀.
    변위 기반 form 은 직접 유도할 것. (참고: `example_elastic_shape_optimization.py`)
  - `integrate(...)` 로 BSR 강성행렬 조립 → 하중벡터 조립 → **Dirichlet 프로젝터** 적용
  - `fem_example_utils.bsr_cg` 로 solve
- **미설치 시:** `pip install warp-lang meshio matplotlib`

## 4. 산출물

- `bench/warp_solve.py` — 위 솔버 (함수: `solve_inp(inp_path) -> (MeasureResult, disp_dict)`).
- `bench/bench_driver.py` — 전체 스윕 실행:
  각 크기에서 (a) `run_ccx` 측정 (b) `warp_solve` 측정 (c) 오차·CPU%·speedup 수집
  → **`results.json`** + **crossover 곡선 PNG**(x=DOF, y=벽시계, ccx vs warp; matplotlib)
  + **parity 표 CSV**(크기별 상대오차·CPU%·speedup·pass/fail).

## 5. ⚠️ 핵심 주의 — 시간을 아끼는 규칙

- **정확도 먼저, 성능 나중.** 작은 메시(20×4×4)에서 제약 3을 **완전히 닫은 뒤에만**
  크기를 키운다. 큰 메시에서 물리 버그를 잡으려 하면 반복당 대기가 길어 지옥이 된다.
- **`bsr_cg` 는 전처리 없는 CG.** 큰 DOF(수십만) 탄성 문제에서 수렴이 매우 느려
  `max_iters` 안에 1e-3 에 못 닿을 수 있다. 이 경우 **최소한 Jacobi(대각) 전처리**를 추가.
  (수렴 실패를 성능 문제로 오인하지 말 것.)
- **제약 2 = crossover 곡선이 산출물.** 작은 문제에선 ccx(CPU)가 이기고, 어떤 DOF
  이상에서 GPU가 이기는 **교차점을 정직하게 보여주는 것**이 합격 결과다. **모든 크기에서
  이길 필요는 없다.** "N DOF 부터 GPU 우위" 를 곡선으로 제시하면 제약 2 충족으로 간주.
- **warp 커널 캐시**를 쓰기 가능한 영구 경로로: `export WARP_CACHE_PATH=/path/writable`
  (매 실행 재컴파일 방지). solve 성능 측정 시 **1회 워밍업 후** 측정할 것(JIT 제외).
- **pod 코어 수 주의:** k8s CPU limit 이 걸려 있으면 `os.cpu_count()` 가 노드 전체를
  볼 수 있음. ccx 스레드 수는 **파드에 실제 할당된 코어**로 맞춘다.

## 6. 작업 방식

작은 메시로 정확도(§2-3)부터 통과 → 크기를 키우며 crossover 와 CPU<20% 튜닝 →
각 단계에서 **GPU pod 터미널로 실제 실행해 수치 확인** → 네 제약이 전부 green 이 될
때까지 반복. 각 마일스톤마다 `results.json` 갱신.
