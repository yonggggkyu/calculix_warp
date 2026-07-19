# Phase 3 — Warp FEA 솔버: LLM Judge용 CalculiX 대체 (구현 스펙)

> Claude Code 사용법: 이 파일 + `bench/` 기존 모듈(instrument.py, gen_inp.py, run_ccx.py) +
> Phase 2 결과물(`warp_solve.py`)을 **먼저 전부 읽고**, §1의 **최종 형태(계약)** 를 목표로,
> §3 수용 기준이 **전부 green** 이 될 때까지 GPU pod에서 실제 실행·반복하며 구현한다.
> 컴파일 성공은 완료가 아니다. **ccx와 measured 값이 일치함을 수치로 증명**해야 한다.

---

## 0. 맥락 — 왜 만드는가

이 솔버는 **LLM Judge System(LLM_Judge_System_Design.md)의 Stage 3a — CalculiX Runner**를
GPU로 대체/가속하기 위한 것이다. Judge는 CalculiX 전체가 아니라 **좁은 표면만** 쓴다:

| Judge가 쓰는 것 (§4 매핑표) | 물리량 | 근거 카드 | 이번 주 |
|---|---|---|---|
| 항복 체크 | **max von Mises 응력** ⭐ | `*STATIC` → `.frd` S | ✅ 커밋 |
| 과도 변위 체크 | **max 변위** | `*STATIC` → `.frd` D | ✅ 커밋 |
| 공진 체크 | 고유진동수 | `*FREQUENCY` | 🔶 스트레치 |
| 좌굴 체크 | 좌굴 하중계수 | `*BUCKLE` eig | 🔶 스트레치 |

- 요소: gmsh가 만든 **비정형 2차 사면체(C3D10)** = solid 경로 기본.
- shell(S3/S4)·CFD는 Judge 문서 스스로 MVP에서 제외 → 이번 주 범위 밖.
- **모달·좌굴은 정적/von Mises 경로(M1~M4)가 일찍 끝나면** 착수(M5). 아니면 후속 스펙으로.

---

## 1. ★ 최종 형태 (인터페이스 계약) — 이게 목표다

Judge의 Stage 3a는 `(mesh, load_case) → measured 값 dict + solver_status` 를 기대하고,
Stage 4(판정)는 그 measured 값을 threshold와 **결정적 비교**하며, Stage 5(blueprint)는
실패 지점의 **위치(location)** 로 기하 처방을 만든다. 따라서 포팅본은 아래를 **정확히** 내야 한다.

### 1.1 주 진입점 (Judge가 부르는 함수)

Judge의 Stage 3a는 **gmsh 메시(`.msh`)를 ccx에 직접 넘긴다**(별도 `.inp` 파일을 안 만든다).
따라서 **주 진입점은 `.msh` + load_case 를 직접 받는 `solve_structural`** 이다. `.inp` 경로는
**검증 전용 보조**(같은 문제를 ccx로 돌려 대조)로만 둔다.

```python
# warp_fea/solver.py

# ★ 주 진입점 — Judge가 이걸 부른다. gmsh .msh(경로 또는 meshio 객체) + Stage 2 load_case.
def solve_structural(
    mesh: "str | meshio.Mesh",   # gmsh .msh 경로 또는 파싱된 meshio 객체
    load_case: dict,             # Stage 2 structural load_case 스키마(supports/loads/material)
    *, device: str = "cuda:0", tol: float = 1e-8,
) -> FEAResult: ...

# 검증 보조 — 같은 (mesh, load_case)를 .inp로 굽고 ccx와 대조할 때만 사용.
def solve_inp(inp_path: str, *, device="cuda:0", tol=1e-8) -> FEAResult: ...
```

**계약 핵심:** Judge는 `.msh` 파일 경로(또는 meshio 객체)와 load_case JSON 하나를 넘기고
FEAResult 를 돌려받는다. **region 이름 해석은 gmsh physical group** 을 통한다 — Stage 2 의
`"region":"mounting_holes"` 는 `.msh` 안의 물리 그룹명과 매칭되어야 하며, 솔버는 meshio 가
읽은 `cell_sets`/`point_sets`(=physical groups)로 노드·면 집합을 확정한다. 즉 **BC/하중 정보는
`.msh` 가 아니라 load_case 에서 오고, 위치(어느 면/노드)만 physical group 으로 해석**한다.

### 1.2 반환 계약 (FEAResult)

```json
{
  "measured": {
    "max_von_mises_stress": {
      "value": 2.31e8, "unit": "Pa",
      "location": {"coords": [x, y, z], "node_id": 4123, "region": "inlet_fillet"}
    },
    "max_displacement": {
      "value": 8.3e-4, "unit": "m",
      "location": {"coords": [x, y, z], "node_id": 991}
    }
    // eigenfrequencies: [...], buckling_load_factor: ...  ← M5에서 채움(키는 예약)
  },
  "solver_status": {
    "regime": "solid", "backend": "warp-gpu", "meshed": true,
    "converged": true, "n_dof": 226875, "n_iters": 412,
    "wall_time_s": 0.093, "cpu_avg_pct": 6.2, "gpu_util_avg_pct": 97.5,
    "device": "NVIDIA H200"
  },
  "fields": { "von_mises": "<gpu-array>", "displacement": "<gpu-array>" }  // 요청 시에만 .numpy()
}
```

**계약의 불변식 (반드시 지킬 것):**
- **SI 단위 고정.** load_case는 SI(E=193e9 Pa, pressure 1e6 Pa, 좌표 m)를 쓴다. 솔버 내부·출력
  전부 **Pa·m·N**. (Phase 2의 MPa/mm 관습을 그대로 쓰면 threshold 비교가 전부 틀어진다.)
- **`converged` 는 정직해야 한다.** CG가 tol 안에 못 들면 `converged=false`. Judge 원칙상
  false-pass는 치명적이므로, **미수렴을 성공인 척 measured로 내지 말 것.**
- **location 필수.** max von Mises의 좌표 + 그것이 속한 **named region**(=.inp의 ELSET/NSET명)
  을 리턴한다. Stage 5가 "inlet_fillet에서 항복"→"fillet 반경↑"로 번역하는 근거다.
- measured의 키 이름은 Stage 2 checks의 `quantity` 문자열과 **일치**시킨다
  (`max_von_mises_stress`, `max_displacement`).

---

## 2. Phase 2에서 이어받는 것 / 새로 만드는 것

이어받음(재사용): GPU 선형정적 solve 골격, `bsr_cg`(+Jacobi 전처리, CUDA graph),
`instrument.measure`(CPU%/GPU%/시간), `run_ccx`(검증 오라클), crossover/parity 인프라.

새로: ① 비정형 **tet(C3D10)** geometry, ② **von Mises 응력 복원**,
③ **압력하중(*DLOAD)·일반 고정BC(region 기반)**, ④ **§1 계약 래퍼**.

---

## 3. 수용 기준 (전부 green 이어야 완료)

1. **measured 일치(핵심).** 동일 문제(같은 gmsh 메시 + load_case)에서 `solve_structural` 의
   measured 가 ccx 추출값과 일치. 검증은 그 (mesh, load_case)를 `.inp`로 구워 `run_ccx` 로 돌린
   결과를 오라클로 쓴다(같은 메시·같은 BC 보장):
   - **max 변위: 상대오차 ≤ 1e-3** (주요 미지수라 타이트).
   - **max von Mises: 상대오차 ≤ 3%** (응력은 후처리 파생량이라 코드 간 외삽/평균 차이 존재 →
     완화 허용. 단 **비교 방식**은 §4.2대로 통일할 것). 최대 위치의 region 도 일치.
2. **SI 단위 계약.** 입력 SI → 출력 Pa·m. 단위 라운드트립 테스트 통과.
3. **GPU-only.** solve 구간 `measure()` 로 `cpu_avg < 20%` 자동 검증(Phase 2와 동일). fp64.
4. **미수렴 처리.** 일부러 tol/iter를 빡세게 준 케이스에서 `converged=false` 가 정확히 뜨고
   measured 를 신뢰값으로 내지 않음.
5. **계약 형태.** 반환 dict 가 §1.2 스키마와 정확히 일치(키·단위·location·solver_status).
6. **(M5, 여유 시) 모달** — 고유진동수가 ccx `*FREQUENCY` 와 상대오차 ≤ 2%.

---

## 4. 구현 노트 & 함정

**4.1 tet 요소.** meshio로 gmsh `.msh`/`.inp` 읽기(C3D10→tetra10). `warp.fem` Tetmesh geometry,
`make_polynomial_space(degree=2, dtype=wp.vec3d)` = 2차 tet. degree=1(C3D4)로 먼저 관통시킨 뒤
degree=2로 올린다. BC/하중/재료는 Phase 2의 변위기반 선형탄성 bilinear form 재사용.

**4.2 von Mises 복원 — 비교 방식을 반드시 통일.** FEM 응력은 요소 경계에서 불연속이고 ccx는
적분점→노드 외삽 후 평균한다. 코드 간 exact match 는 기대하지 말 것.
→ **1차 비교는 적분점(quadrature) 최대 von Mises 로**(외삽/평균 차이 배제), 3% 허용.
σ = 2μ ε + λ tr(ε) I, von Mises = sqrt(3/2 s:s), s=deviatoric. `warp.fem` 로 해의 ∇u 를
적분점에서 평가해 계산.

**4.3 SI 단위 함정(재강조).** load_case 예: E=193e9, ν=0.29, pressure=1e6, yield=215e6 (전부 Pa),
좌표 m. Phase 2 데크(E=210000)를 재사용하지 말고 **SI 케이스로 새로** 검증할 것.

**4.4 region 해석.** load_case 의 supports/loads 는 `"region":"mounting_holes"` 처럼 **이름**으로
온다. gmsh physical group → meshio point_sets/cell_sets(=.inp NSET/ELSET)로 노드/면 집합에
매핑. 압력하중은 지정 **표면(면 집합)** 에 traction 적분으로 조립(*DLOAD 대응).

**4.5 성능(Phase 2에서 이어짐).** 큰 tet 메시에서 무전처리 CG 수렴 느림 → **Jacobi 전처리 필수**.
JIT **워밍업 1회 후** 측정. `WARP_CACHE_PATH` 영구경로. ccx 스레드 수는 **cgroup 실제 코어**로.

---

## 5. 산출물 (파일 구조)

```
warp_fea/
  solver.py       # solve_structural(.msh, load_case) ← Judge 주 진입점(§1) / solve_inp = 검증 보조
  mesh_io.py      # gmsh/.inp → Tetmesh + region set 해석
  elasticity.py   # bilinear form + 응력/von Mises 복원
  results.py      # FEAResult / measured / solver_status 데이터클래스(§1.2 스키마)
  validate.py     # measured-dict parity vs ccx (케이스 세트 자동 대조)
bench/            # 재사용: instrument.py, run_ccx.py, gen_inp.py
```

## 6. 검증 전략

Phase 2 하네스 그대로. 케이스별로 (a) `solve_structural(.msh, load_case)` 로 Warp measured
(b) 같은 (mesh, load_case)를 `.inp`로 구워 `run_ccx` 로 ccx measured 추출(오라클)
(c) §3-1 기준으로 대조 → **measured-parity 표 CSV** + solve 시간/CPU%/GPU% 로그.
검증 케이스는 SI 단위로: 외팔보(점하중), 내압 실린더/박스(압력하중), 구멍 뚫린 판(응력집중).
각 케이스는 gmsh physical group 으로 region(고정면·하중면)을 라벨링해 두 경로가 같은 집합을 쓰게 한다.

## 7. 마일스톤

- **M1** hex→tet 교체. gmsh tet 메시 → Tetmesh, degree1→2. 변위 ccx 대조(≤1e-3).
- **M2** von Mises 복원(§4.2). 적분점 최대 응력 ccx 대조(≤3%) + 위치/ region.
- **M3** 압력하중(*DLOAD)·region 고정BC. SI 내압 케이스 검증.
- **M4** §1 계약 래퍼(`solve_structural(.msh, load_case)`→FEAResult) + region=physical group 해석.
  H-CCX류 1케이스를 Warp(주 경로)와 ccx(같은 문제를 .inp로 구운 오라클) 양측 통과, measured dict 일치.
- **M5(여유 시)** 모달: 질량행렬 조립 + LOBPCG로 최저 고유진동수 몇 개, ccx `*FREQUENCY` 대조(≤2%).

## 8. 작업 방식

작은 tet 메시로 변위(M1)→응력(M2) 정확도부터 닫고, 그다음 하중/BC(M3)·계약(M4). 각 단계
**GPU pod에서 실제 ccx와 대조**하며 수치 확인. M1~M4 green 이면 M5 착수, 아니면 M5는 후속
스펙으로 문서화하고 종료. 각 마일스톤마다 measured-parity 표 갱신.
