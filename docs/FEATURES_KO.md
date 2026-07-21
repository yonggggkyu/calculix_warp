# calculix_warp — 구현 기능 정리 (보고용)

> GPU(NVIDIA Warp) 기반 구조해석 솔버. LLM Judge System의 Stage 3a(CalculiX Runner)를
> 대체·가속하는 것이 목적. **모든 기능은 동일 메시·동일 하중조건으로 CalculiX(ccx)와
> 대조 검증**되었으며, 검증되지 않은 입력은 계산하지 않고 거부한다.

---

## 1. 한눈에 보기

| 구분 | 항목 | 상태 |
|---|---|---|
| **해석 종류** | 선형 정적 / 모달(고유진동수) / 선형 좌굴 | ✅ 검증 완료 |
| **재료** | 등방 탄성 / 직교이방성(9상수) | ✅ 검증 완료 |
| **구성** | 단일 파트 / 다중 재료 조립체(공유절점) | ✅ 검증 완료 |
| **하중** | 집중력 · 압력 · 표면응력 · 중력(자중) | ✅ 검증 완료 |
| **요소** | 2차 사면체 C3D10, 1차 C3D4 | ✅ 검증 완료 |
| **안전장치** | 범위 밖 입력 자동 거부 + 사유 코드 | ✅ 동작 확인 |
| **미구현** | 접촉(비선형), 완전이방성(21상수), 소성, hex/shell | ⛔ 거부 처리 |

정밀도: **fp64**, 단위: **SI 고정**(Pa · m · N)

---

## 2. 검증 결과 — ccx 대비 상대오차

동일 메시 + 동일 하중을 (a) 본 솔버 GPU (b) ccx 로 각각 풀어 비교.

### 2.1 정적 해석 (7케이스 × 2요소차수 = 14조합 전부 통과)

| 검증 케이스 | 검증 대상 | 변위 오차 | von Mises 오차 |
|---|---|---|---|
| 외팔보 (집중하중) | 기본 정적 | 3.3e-08 | 0.00% |
| 내압 실린더 | 압력하중 `*DLOAD` | 1.6e-07 | 0.00% |
| 구멍 뚫린 판 | 응력집중 (24.7만 DOF) | 5.3e-08 | 0.00% |
| 볼트 브라켓 | H-CCX형 실제 형상 | 4.3e-08 | 0.00% |
| **자중 보** | **중력하중 `GRAV`** | **7.6e-08** | 0.00% |
| **직교이방성 보** | **CFRP형 이방성** | **1.4e-07** | 0.00% |
| **2재료 조립체** | **강+알루미늄 결합** | **1.9e-07** | 0.00% |

> 판정 기준: 변위 ≤1e-3, von Mises ≤3% → **실측은 기준보다 4~5자릿수 여유**

### 2.2 모달 / 좌굴

| 해석 | ccx 대조 | 최대 상대오차 |
|---|---|---|
| 고유진동수 (최저 10모드) | `*FREQUENCY` | **1.1e-05** |
| 좌굴 하중계수 (최저 4개) | `*BUCKLE` | **1.1e-07** |

좌굴은 해석해(Euler 기둥 BLF ≈ 44.1)와도 일치(계산값 44.13).

### 2.3 성능 (Phase 2, 육면체 메시 기준)

| DOF | ccx (CPU 28코어) | 본 솔버 (H200) | 배속 |
|---:|---:|---:|---:|
| 9,963 | 0.36 s | 0.015 s | 24× |
| 70,227 | 3.8 s | 0.035 s | 109× |
| 525,987 | 54.2 s | 0.25 s | **216×** |

약 300 DOF부터 GPU 우위. 최대 케이스에서 GPU 사용률 97%, CPU 사용률 20% 미만 유지.
*(측정 범위는 solve 구간 기준)*

---

## 3. 기능 상세

### 3.1 해석 종류
- **선형 정적** — 변위·응력. 반환값: 최대 von Mises 응력, 최대 변위 (+위치·소속 region)
- **모달** — 최저 고유진동수 n개 [Hz]. GPU 부분공간 반복법
- **선형 좌굴** — 좌굴 하중계수(BLF). 참조하중 정적해 → 기하강성 → 고유치

### 3.2 재료
- **등방 탄성** — `E`, `nu`, `density`
- **직교이방성** — `E1,E2,E3 / nu12,nu13,nu23 / G12,G13,G23` (ccx `*ELASTIC, TYPE=ORTHO` 대응)
  - 물성 변환(D행렬)을 솔버와 ccx deck 생성기가 **공유** → 변환 오류가 솔버 오류로 위장 불가

### 3.3 조립체
- 부피 region별로 재료 지정, **경계면 절점 공유**(완전 결합)
- 강성·체적력·응력복원을 재료 영역별로 각각 조립
- ccx의 다중 `*SOLID SECTION`과 대응

### 3.4 하중 / 경계조건
| 하중 | 설명 | ccx 대응 |
|---|---|---|
| `force` | 절점집합에 총 하중[N] 분배 | `*CLOAD` |
| `pressure` | 면에 압력[Pa] | `*DLOAD` |
| `traction` | 면에 응력벡터[Pa] | (등가절점력) |
| `gravity` | 중력가속도[m/s²]×밀도 = 자중 | `*DLOAD ... GRAV` |

경계조건: 완전구속(fixed). 위치 지정은 **gmsh physical group** 이름으로.

---

## 4. 안전장치 — "조용히 틀리지 않기"

Judge에서 가장 위험한 실패는 **범위 밖 문제를 그럴듯한 숫자로 계산해 통과시키는 것**(false-pass).
따라서 지원하지 않는 입력은 **계산하지 않고 거부**하며, 사유 코드를 반환한다.

**거부 항목 (예시)**
- 접촉 정의 → `contact:defined`
- 완전이방성/소성/초탄성 → `material:anisotropic`, `material:plastic` 등
- 사면체 아닌 요소(hex/shell/beam) → `element:hexahedron` 등
- 중력인데 밀도 없음 → `load_malformed:gravity_no_density` *(없으면 조용히 0하중이 됨)*
- 이방성 상수 불완전/비물리(강성 비양정부호) → `material:orthotropic_incomplete`, `..._not_spd`
- 조립체 region 누락·중복 → `material:part_no_region`, `..._duplicate_region`
- 구속 부족(강체운동 잔존) → `constraint:none`

**미수렴 정직성** — 반복 솔버가 허용오차에 못 들면 `converged=false`로 표시하고
**measured 값을 비움**. 신뢰할 수 없는 수치를 Judge에 넘기지 않는다.

거부 사유는 `rejection_counts.json`에 누적 집계 → "실제로 어떤 미지원 기능이 얼마나 필요했는지" 추적 가능.

---

## 5. 사용법 (Judge 연동 인터페이스)

```python
from warp_fea import solve_structural

load_case = {
    "material": {"E": 193e9, "nu": 0.29, "density": 7900.0},   # SI
    "supports": [{"region": "mounting_holes", "type": "fixed"}],
    "loads":    [{"region": "payload_face", "type": "force", "vector": [0,0,-3924]}],
}
r = solve_structural("part.msh", load_case)

r.measured["max_von_mises_stress"].value      # [Pa]
r.measured["max_von_mises_stress"].location   # 좌표 + 절점번호 + region 이름
r.measured["max_displacement"].value          # [m]
r.solver_status.converged                     # False면 measured 신뢰 금지
r.rejected, r.reject_codes                    # 범위 밖 입력이면 True + 사유
```

재현: 저장소 루트에서 `python -m warp_fea.acceptance`
→ 메시 생성 → GPU 해석 → ccx 대조 → 수용기준 7개 전부 자동 검증

---

## 6. 미구현 / 다음 단계

| 항목 | 사유 | 규모 |
|---|---|---|
| **접촉(contact)** | 접촉 면적이 해에 의존하는 **비선형** 문제. 현재 구조(강성 1회 조립 + CG 1회)에 활성집합/페널티 반복 루프를 새로 얹어야 함 | 큼 (별도 Phase) |
| tie 구속 | 불일치 메시 결합. **선형**이라 조립체 확장으로 비교적 저렴 | 중간 |
| 완전이방성(21상수) | 구현은 직교이방성 확장이나 검증 케이스 필요 | 작음~중간 |
| 소성·대변형 | 비선형 | 큼 |

---

## 7. 발견한 이슈 (중요)

**CalculiX 2.17 멀티스레드 실행 시 출력이 간헐적으로 손상됨.**
`Using up to N cpu(s) for the stress calculation` 경로에서 적분점 응력이 1% 가량 오염되며,
변위까지 어긋나는 경우도 관측. 파일 형식은 정상이라 **육안으로 구분 불가**.

| ccx 스레드 | 손상 발생 |
|---:|---|
| 28 | 30회 중 1회 |
| 8 | 30회 중 4회 |
| **1** | **30회 중 0회** |

손상 시 최대 von Mises가 실제의 **15~80배**로 보고됨 → 항복 판정이 뒤집힐 수 있음.
→ **ccx로 응력을 뽑을 땐 반드시 `OMP_NUM_THREADS=1`**. 본 저장소의 검증 하네스는 단일
스레드 실행 + 재현성 재확인으로 처리하고 있음.

---

## 8. 환경

- GPU: NVIDIA H200 (sm_90) / CUDA 13.0 · Warp 1.15.0
- CalculiX 2.17 (검증 오라클 전용, 솔버 자체는 불필요)
- Python 3.11, 정밀도 fp64
- 저장소: https://github.com/yonggggkyu/calculix_warp
