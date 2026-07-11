# 반도체 공정 데이터 기반 다중출력 회귀 모델 개발 명세

## 1. 프로젝트 목표

반도체 공정 조건 데이터를 입력받아 다음 두 개의 품질 지표를 동시에 예측하는 다중출력 회귀 모델을 개발한다.

### 입력값 X

- RF Power
- Pressure
- Temperature
- Gas Flow Rate

### 출력값 Y

- Thickness: 박막 두께
- Uniformity: 박막 균일도

최종 목표는 새로운 공정 조건이 입력되었을 때 Thickness와 Uniformity를 동시에 예측하는 것이다.

---

## 2. 개발 환경

- Python 3.11 이상
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- joblib

필요 시 추가 라이브러리를 설치할 수 있도록 `requirements.txt`를 작성한다.

---

## 3. 데이터 입력 형식

기본 입력 파일은 CSV 형식으로 가정한다.

예시 컬럼:

```csv
RF_Power,Pressure,Temperature,Gas_Flow_Rate,Thickness,Uniformity
300,10.2,250,50,102.4,97.8
320,11.1,260,55,108.7,96.9
```

실제 컬럼명이 다를 수 있으므로 `config.py` 또는 설정 딕셔너리에서 컬럼명을 쉽게 변경할 수 있게 구현한다.

필수 컬럼:

```python
FEATURE_COLUMNS = [
    "RF_Power",
    "Pressure",
    "Temperature",
    "Gas_Flow_Rate",
]

TARGET_COLUMNS = [
    "Thickness",
    "Uniformity",
]
```

---

## 4. 데이터 전처리

다음 과정을 구현한다.

1. CSV 파일 로드
2. 필수 컬럼 존재 여부 확인
3. 문자열 및 잘못된 값 처리
4. 결측치 개수 출력
5. 결측치가 적으면 해당 행 제거
6. 결측치가 많으면 median 기반 대체
7. 중복 행 제거
8. 각 변수의 기술통계량 출력
9. 이상치 분포 확인
10. 입력값과 출력값 상관관계 확인

데이터 누수를 막기 위해 전처리 객체는 반드시 train 데이터에만 fit한다.

---

## 5. 데이터 분할

기본 분할 비율:

```text
Train: 70%
Validation: 15%
Test: 15%
```

데이터 수가 적으면 다음 방식도 허용한다.

```text
Train: 80%
Test: 20%
```

기본 랜덤 시드는 다음으로 통일한다.

```python
RANDOM_STATE = 42
```

만약 데이터에 다음 컬럼이 존재하면 랜덤 분할보다 그룹 기반 분할을 우선한다.

- Lot ID
- Wafer ID
- Run ID
- Recipe ID
- Timestamp

같은 Lot, Wafer 또는 Run의 데이터가 train과 test에 동시에 들어가지 않도록 주의한다.

---

## 6. 모델 구성

다중출력 회귀 방식으로 Thickness와 Uniformity를 동시에 예측한다.

다음 모델을 구현하고 성능을 비교한다.

### Baseline

```python
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
```

### Random Forest

```python
from sklearn.ensemble import RandomForestRegressor
```

RandomForestRegressor는 다중출력을 직접 지원하므로 별도 wrapper 없이 사용한다.

### XGBoost

```python
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
```

XGBoost는 다음처럼 구성한다.

```python
model = MultiOutputRegressor(
    XGBRegressor(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
    )
)
```

---

## 7. 학습 파이프라인

다음 구조로 구현한다.

```text
데이터 로드
→ 전처리
→ train/validation/test 분할
→ 모델 학습
→ validation 성능 비교
→ 최종 모델 선택
→ test 성능 평가
→ 모델 저장
```

입력값 스케일링은 Linear Regression 계열에는 적용하고, Random Forest와 XGBoost에는 필수가 아니므로 Pipeline을 분리한다.

예시:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MultiOutputRegressor(LinearRegression())),
])
```

---

## 8. 평가 지표

Thickness와 Uniformity를 각각 평가한다.

필수 평가 지표:

- MAE
- RMSE
- R²

출력 예시:

```text
[Thickness]
MAE: 2.41
RMSE: 3.18
R²: 0.91

[Uniformity]
MAE: 0.72
RMSE: 0.95
R²: 0.87
```

전체 평균 지표도 함께 계산한다.

단, Thickness와 Uniformity는 단위와 범위가 다를 수 있으므로 단순 평균만으로 모델을 판단하지 않는다.

---

## 9. 시각화

다음 그래프를 각각 생성하고 `outputs/plots/` 폴더에 저장한다.

1. 실제값 vs 예측값 산점도
   - Thickness
   - Uniformity

2. 잔차 분포
   - Thickness
   - Uniformity

3. Feature Importance
   - Random Forest
   - XGBoost

4. 입력변수 상관관계 히트맵

각 그래프는 한 Figure에 하나씩 생성한다.

---

## 10. 모델 선택 기준

Validation 데이터 기준으로 다음 조건을 종합해 최종 모델을 선택한다.

1. Thickness MAE가 낮은가
2. Uniformity MAE가 낮은가
3. 두 출력의 R²가 모두 안정적인가
4. 특정 출력만 지나치게 잘 맞고 다른 출력은 무너지지 않는가
5. Test 데이터에서도 성능이 유지되는가

최종 모델 선택 근거를 텍스트 파일 또는 JSON으로 저장한다.

예시:

```json
{
  "selected_model": "XGBoost_MultiOutput",
  "reason": "Thickness와 Uniformity 모두에서 가장 낮은 validation MAE 기록",
  "validation_metrics": {
    "Thickness_MAE": 2.41,
    "Uniformity_MAE": 0.72
  }
}
```

---

## 11. 모델 저장

최종 모델과 전처리 객체를 저장한다.

```text
artifacts/
├── model.joblib
├── preprocessor.joblib
├── feature_columns.json
├── target_columns.json
└── metrics.json
```

가능하면 전처리와 모델을 하나의 Pipeline으로 묶어서 저장한다.

---

## 12. 예측 인터페이스

새로운 입력값을 받아 두 출력을 동시에 예측하는 함수를 작성한다.

```python
def predict_process_result(
    rf_power: float,
    pressure: float,
    temperature: float,
    gas_flow_rate: float,
) -> dict:
    ...
```

반환 예시:

```python
{
    "Thickness": 102.4,
    "Uniformity": 97.8
}
```

잘못된 값이 들어오면 명확한 예외 메시지를 반환한다.

예:

```text
RF Power must be a numeric value.
Pressure column is missing.
Input contains NaN.
```

---

## 13. CLI 실행 방식

다음 명령어로 학습할 수 있게 구현한다.

```bash
python train.py --data data/process_data.csv
```

예측은 다음처럼 실행한다.

```bash
python predict.py \
  --rf-power 300 \
  --pressure 10.2 \
  --temperature 250 \
  --gas-flow-rate 50
```

출력 예시:

```text
Predicted Thickness: 102.4
Predicted Uniformity: 97.8
```

---

## 14. 권장 프로젝트 구조

```text
semiconductor-regression/
├── data/
│   └── process_data.csv
├── artifacts/
├── outputs/
│   └── plots/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 15. 필수 구현 조건

- 함수 단위로 코드를 분리한다.
- 하드코딩을 최소화한다.
- 주요 함수에 type hint를 작성한다.
- 주요 로직에 간단한 docstring을 작성한다.
- 데이터 누수가 발생하지 않도록 한다.
- 에러 처리를 구현한다.
- 학습 결과를 콘솔과 파일에 모두 저장한다.
- 동일한 random seed에서 결과가 재현되도록 한다.
- test 데이터는 최종 평가 전에 사용하지 않는다.

---

## 16. 최종 산출물

다음 파일이 실행 가능한 상태로 제출되어야 한다.

1. `train.py`
2. `predict.py`
3. `requirements.txt`
4. `README.md`
5. 학습된 모델 파일
6. 평가 결과 JSON
7. 실제값-예측값 그래프
8. Feature Importance 그래프
9. 샘플 데이터 형식
10. 실행 방법 문서

---

## 17. 1차 개발 범위

1차 버전에서는 아래까지만 구현한다.

- CSV 데이터 로드
- 전처리
- 다중출력 회귀 모델 학습
- Linear Regression, Random Forest, XGBoost 비교
- MAE, RMSE, R² 평가
- 최종 모델 저장
- 단일 입력 예측 기능
- 기본 시각화

웹 대시보드, API 서버, 실시간 공정 연동은 1차 범위에서 제외한다.
