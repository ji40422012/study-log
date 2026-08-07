# Polynomial Regression (Practical)

## 1. Polynomial Regression(다항회귀)
독립변수의 제곱, 세제곱 등의 다항식을 이용하여
비선형 관계를 모델링

예를 들어
Linear Regression
y = b + wx
↓
Polynomial Regression
y = b + w₁x + w₂x² + w₃x³

그래프는 곡선처럼 보이지만
모델은 Feature들의 선형결합을 학습하므로
여전히 **Linear Model 계열**

---

## 2. 언제 사용하는가?
Linear Regression으로는 설명하기 어려운
곡선 형태의 데이터를 모델링할 때

예시
- 집값 예측
- 생산량 예측
- 매출 예측
- 의료 데이터
- 환경 데이터

---

## 3. 데이터
scikit-learn from sklearn.datasets import load_diabetes

| 내용 | 값 |
|------|----|
| Samples | 442 |
| Features | 10 |
| Target | Disease Progression |

Target은 당뇨병 진행 정도

---

## 4. 분석 흐름

```
데이터 불러오기
        ↓
BMI 하나 선택
        ↓
Polynomial Regression
        ↓
degree 비교
        ↓
Train / Test
        ↓
Cross Validation
        ↓
전체 Feature 적용
        ↓
Linear vs Polynomial 비교
        ↓
Polynomial + Ridge
        ↓
Residual 분석
```

---

# 5. 주요 코드

## (1) PolynomialFeatures

```python
PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

옵션

| 옵션 | 설명 |
|------|------|
| degree | 다항식 차수 |
| include_bias | 상수항 생성 여부 |

---

## (2) make_pipeline

```python
make_pipeline(
    PolynomialFeatures(...),
    LinearRegression()
)
```
Polynomial Feature 생성 - Linear Regression 수행

---

## (3) degree
다항식의 최고 차수
- **degree=1** → Linear Regression
- **degree=2** → x² 추가(곡선 표현)
- **degree=3** → x³ 추가(더 복잡한 곡선)
- **degree↑** → Feature↑, 모델 복잡도↑, Overfitting 가능성↑

---

## (4) Train/Test Split

```python
train_test_split()
```
학습 데이터와 테스트 데이터로 분리하여
새로운 데이터에서도 잘 동작하는지 확인

---

## (5) Cross Validation

```python
cross_val_score()
```
5개의 Fold로 반복 학습해서 평균 성능 계산
- 한 번의 Train/Test보다 일반화 성능을 더 신뢰

---

## (6) Ridge Regression

```python
Ridge(alpha=1.0)
```
Polynomial Feature가 많아지면 계수가 매우 커질 수 있으므로
L2 규제로 계수를 줄여 과적합을 완화

---

# 6. Polynomial Feature 증가
`PolynomialFeatures()`는 기존 Feature에서 **제곱항과 상호작용항**을 자동으로 생성

| 변환 전 | 변환 후 (degree=2) |
|---------|-------------------|
| x₁, x₂ | x₁, x₂, x₁², x₁x₂, x₂² |

```
원본 Feature : 10개
↓
Polynomial(degree=2)
↓
65개 Feature
```
Feature가 증가할수록 **모델의 표현력은 높아지지만, 과적합 가능성도 함께 증가한다.**
---

# 7. 성능 평가

| 지표 | 의미 | 좋은 값 |
|------|------|---------|
| R² | 모델의 설명력 | 높을수록 좋음(1에 가까울수록) |
| RMSE | 평균 예측 오차 | 낮을수록 좋음 |
| Cross Validation | 여러 번 평가한 평균 성능 | 높고 일정할수록 좋음 |

# 8. 결과 해석

- **degree 증가** → 모델이 더 복잡한 패턴을 학습
- **Train 성능↑** → 학습 데이터에는 더 잘 맞음
- **Test 성능↓** → 과적합(Overfitting) 가능성
- **Ridge 적용** → 과적합을 완화하여 일반화 성능 향상

# 9. 실무에서는?

실무에서는 **높은 degree를 사용하는 것이 목표가 아니라, 가장 일반화 성능이 좋은 모델을 찾는 것이 목표**이다.

```
PolynomialFeatures
        ↓
degree 후보(2, 3, 5 ...)
        ↓
Cross Validation
        ↓
Ridge / Lasso 적용
        ↓
최종 모델 선택
```

---

# 10. 이번 실습에서 배운 함수

| 함수 | 역할 |
|------|------|
| load_diabetes() | 내장 데이터 |
| train_test_split() | 학습/테스트 분리 |
| PolynomialFeatures() | 다항 Feature 생성 |
| make_pipeline() | 전처리 + 모델 연결 |
| LinearRegression() | 선형회귀 |
| Ridge() | L2 규제 |
| fit() | 모델 학습 |
| predict() | 예측 |
| r2_score() | R² 계산 |
| mean_squared_error() | MSE 계산 |
| np.sqrt() | RMSE 계산 |
| cross_val_score() | 교차검증 |
| plt.scatter() | 산점도 |
| plt.plot() | 회귀곡선 |
| plt.grid() | 격자 |
| plt.legend() | 범례 |
| plt.tight_layout() | 레이아웃 자동 조정 |

---

# 11. 핵심 정리

- Polynomial Regression은 곡선을 표현하는 회귀 모델
  (그래프는 곡선이지만 모델은 Linear Model 계열)
- degree가 증가하면 Feature 수가 크게 증가 :복잡한 모델일수록 Overfitting 가능성 고려
- Test 성능과 Cross Validation으로 degree를 선택
- Ridge/Lasso를 함께 사용하면 일반화 성능이 향상
