# Logistic Regression 

## 1. Logistic Regression
**이진 분류(Binary Classification)** 를 위한 대표적인 머신러닝 알고리즘
**연속값이 아닌 클래스(Class)를 예측**

예)
- 암 진단 (양성 / 음성)
- 스팸 메일 (스팸 / 정상)
- 합격 여부 (합격 / 불합격)
- 구매 여부 (구매 / 미구매)
---

# 2. Linear Regression과의 차이

| Linear Regression | Logistic Regression |
|-------------------|---------------------|
| 연속값 예측 | 클래스 예측 |
| 출력 범위 제한 없음 | 0~1 확률 출력 |
| 회귀 문제 | 분류 문제 |
Logistic Regression은 **예측 확률**을 계산한 후, 
Threshold를 기준으로 클래스를 결정

---

# 3. Sigmoid Function

Sigmoid 함수는 선형 모델의 출력을 **0~1 사이의 확률**로 변환

```text
Linear Output
      ↓
 Sigmoid Function
      ↓
Probability (0~1)
```

기본 Threshold가 `0.5`라면
```text
p < 0.5  → class 0
p ≥ 0.5 → class 1
```

---

# 4. 기본 흐름

```text
데이터 불러오기
        ↓
Train / Test Split
        ↓
LogisticRegression()
        ↓
fit()
        ↓
predict()
        ↓
Accuracy
        ↓
Confusion Matrix
        ↓
Coefficient
```

---

# 5. 주요 코드

| 함수 | 역할 |
|------|------|
| load_breast_cancer() | 유방암 데이터 불러오기 |
| train_test_split() | Train/Test 데이터 분리 |
| LogisticRegression() | Logistic Regression 모델 생성 |
| fit() | 모델 학습 |
| predict() | 클래스 예측 |
| predict_proba() | 클래스별 예측 확률 |
| accuracy_score() | 정확도 계산 |
| confusion_matrix() | 혼동행렬 계산 |
| ConfusionMatrixDisplay() | 혼동행렬 시각화 |

---

# 6. 결과 해석

### Accuracy
전체 데이터 중 올바르게 분류한 비율
**1에 가까울수록 성능이 좋다.**

### Confusion Matrix
실제값과 예측값을 비교하여 **오분류를 확인**

### predict_proba()
- 각 클래스에 속할 **확률**
- 일반적으로 확률이 높은 클래스를 최종 예측값으로 선택

### Coefficient
- 각 Feature가 예측에 미치는 영향
- **양수(+)** → class 1 방향으로 영향
- **음수(-)** → class 0 방향으로 영향

---
