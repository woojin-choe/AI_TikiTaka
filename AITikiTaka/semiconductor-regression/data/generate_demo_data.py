"""
Utility script to (re)generate the demo process_data.csv.

실제 반도체 공정에 가까운 데이터를 생성합니다:
  - 비선형 관계 (아레니우스 온도 의존성, 멱함수 압력 의존성)
  - 변수 간 교호작용 (RF_Power × Pressure → 플라즈마 밀도)
  - 공정 드리프트 (장비 에이징에 따른 시간 의존 편향)
  - 배치(Lot) 효과 (같은 Lot 내 상관성)
  - 이상 공정 포함 (파티클 오염 등 돌발 이벤트)
  - 비정규 노이즈 (가끔 튀는 값)

Run: python data/generate_demo_data.py
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_ROWS = 300          # 행 수를 늘려서 현실감 있게
N_LOTS = 20           # 배치(Lot) 수
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "process_data.csv")


def generate_demo_data(
    n_rows: int = N_ROWS,
    n_lots: int = N_LOTS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    실제 반도체 공정 특성을 반영한 데모 데이터를 생성합니다.

    Parameters
    ----------
    n_rows : int
        생성할 총 행 수
    n_lots : int
        배치(Lot) 수. 같은 Lot 내 데이터는 서로 상관성을 가집니다.
    random_state : int
        재현성을 위한 랜덤 시드

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # 1. 입력 변수 생성
    # ------------------------------------------------------------------
    rf_power     = rng.uniform(250, 400, n_rows)   # W
    pressure     = rng.uniform(8, 15,  n_rows)     # mTorr
    temperature  = rng.uniform(200, 300, n_rows)   # °C
    gas_flow     = rng.uniform(30, 80,  n_rows)    # sccm

    # ------------------------------------------------------------------
    # 2. 공정 드리프트 (장비 에이징)
    #    시간이 지날수록(행 순서) 챔버 오염으로 두께가 서서히 감소
    # ------------------------------------------------------------------
    time_index = np.linspace(0, 1, n_rows)           # 0 → 1 (정규화된 시간)
    drift = -4.0 * time_index                         # 최대 -4nm 드리프트

    # ------------------------------------------------------------------
    # 3. 배치(Lot) 효과
    #    같은 Lot에서 나온 데이터는 공통 오프셋을 공유
    # ------------------------------------------------------------------
    lot_ids = rng.integers(0, n_lots, n_rows)
    lot_thickness_offset  = rng.normal(0, 2.0, n_lots)   # Lot마다 ±2nm 편차
    lot_uniformity_offset = rng.normal(0, 0.3, n_lots)   # Lot마다 ±0.3% 편차

    # ------------------------------------------------------------------
    # 4. Thickness 계산
    #
    #    핵심 메커니즘:
    #      - 플라즈마 밀도 ∝ RF_Power / Pressure  (교호작용)
    #      - 온도 의존성: 아레니우스 exp(-Ea/kT) 형태
    #        → 여기서는 exp(0.003 × T)로 근사
    #      - 압력 의존성: 멱함수 Pressure^(-0.4)
    # ------------------------------------------------------------------
    Ea_k = 0.003   # 활성화 에너지 / 볼츠만 상수 (근사 스케일)

    plasma_density = rf_power / pressure              # 교호작용 항
    temp_factor    = np.exp(Ea_k * temperature)       # 아레니우스 근사
    pressure_factor = pressure ** (-0.4)              # 멱함수 압력 의존성

    # 베이스 두께 (스케일 맞추기 위해 정규화)
    thickness_base = (
        0.25 * plasma_density          # 플라즈마 밀도 (RF/P 교호작용)
        + 15.0 * temp_factor           # 온도 (비선형)
        + 120.0 * pressure_factor      # 압력 (멱함수)
        + 0.04 * gas_flow              # 가스 유량 (약한 선형)
    )

    # 정규분포 노이즈 + 가끔 크게 튀는 값 (비정규 노이즈)
    noise_thickness = rng.normal(0, 2.0, n_rows)
    spike_mask = rng.random(n_rows) < 0.04            # 4% 확률로 스파이크
    noise_thickness += spike_mask * rng.choice([-8, 8], n_rows) * rng.random(n_rows)

    thickness = (
        thickness_base
        + drift
        + lot_thickness_offset[lot_ids]
        + noise_thickness
    )

    # 현실적인 범위로 클리핑 (85~145 nm)
    thickness = np.clip(thickness, 85, 145)

    # ------------------------------------------------------------------
    # 5. Uniformity 계산
    #
    #    핵심 메커니즘:
    #      - 압력이 높으면 가스 분자 충돌 증가 → 불균일
    #      - RF/P 교호작용: 플라즈마 불균일성
    #      - 가스 유량이 높으면 균일도 약간 개선
    #      - 온도 구배가 클수록 불균일 (온도^2 항)
    # ------------------------------------------------------------------
    uniformity_base = (
        102.0
        - 0.018 * rf_power                        # RF 증가 → 불균일
        - 0.35  * pressure                        # 압력 증가 → 불균일
        - 0.0008 * (temperature - 250) ** 2       # 최적 온도(250°C) 벗어날수록 불균일 (2차)
        + 0.010 * gas_flow                        # 가스 유량 증가 → 약간 균일
        - 0.0015 * (rf_power / pressure - 25)     # 플라즈마 밀도가 과도하면 불균일
    )

    noise_uniformity = rng.normal(0, 0.4, n_rows)
    spike_mask_u = rng.random(n_rows) < 0.03      # 3% 확률로 스파이크
    noise_uniformity += spike_mask_u * rng.choice([-2, 2], n_rows) * rng.random(n_rows)

    uniformity = (
        uniformity_base
        + lot_uniformity_offset[lot_ids]
        + noise_uniformity
    )

    # 현실적인 범위로 클리핑 (90~99.5%)
    uniformity = np.clip(uniformity, 90.0, 99.5)

    # ------------------------------------------------------------------
    # 6. DataFrame 구성
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "Lot_ID":        lot_ids,
        "RF_Power":      rf_power,
        "Pressure":      pressure,
        "Temperature":   temperature,
        "Gas_Flow_Rate": gas_flow,
        "Thickness":     thickness,
        "Uniformity":    uniformity,
    })

    # ------------------------------------------------------------------
    # 7. 결측치 주입 (~5%)
    #    입력 변수에만 결측치 발생 (출력은 측정 실패 시 행 자체를 제거하는 게 현실적)
    # ------------------------------------------------------------------
    input_cols = ["RF_Power", "Pressure", "Temperature", "Gas_Flow_Rate"]
    n_missing = int(n_rows * 0.05)
    for col in input_cols:
        missing_idx = rng.choice(n_rows, size=n_missing, replace=False)
        df.loc[missing_idx, col] = np.nan

    # ------------------------------------------------------------------
    # 8. 중복 행 추가 (동일 레시피 재실행 상황)
    # ------------------------------------------------------------------
    n_dupes = 6
    dupe_rows = df.dropna().sample(n=n_dupes, random_state=random_state)
    df = pd.concat([df, dupe_rows], ignore_index=True)

    return df


if __name__ == "__main__":
    df = generate_demo_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Demo data written to {OUTPUT_PATH}  ({len(df)} rows)")
    print(f"\nThickness  — mean: {df['Thickness'].mean():.1f}, std: {df['Thickness'].std():.1f}, "
          f"range: [{df['Thickness'].min():.1f}, {df['Thickness'].max():.1f}]")
    print(f"Uniformity — mean: {df['Uniformity'].mean():.1f}, std: {df['Uniformity'].std():.1f}, "
          f"range: [{df['Uniformity'].min():.1f}, {df['Uniformity'].max():.1f}]")
