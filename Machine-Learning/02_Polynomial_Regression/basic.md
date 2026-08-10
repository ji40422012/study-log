# 다항회귀(Polynomial Regression)

직선으로 표현하기 어려운 비선형 관계를 **다항 특성으로 변환하여 회귀 모델로 학습**

- 다항회귀의 기본 원리
- 다항식 차수(Degree)
- 과소적합 / 과적합
- `np.polyfit()`을 이용한 다항식 적합
- `PolynomialFeatures`를 이용한 다항 특성 생성
- `Pipeline`을 이용한 전처리와 모델 연결
- MSE / RMSE / R² 평가
- 잔차 분석
- 높은 차수의 문제와 규제
- 외삽(Extrapolation)

---

## 1. 다항회귀

선형회귀가 입력과 출력의 관계를 직선으로 표현한다면, 
다항회귀는 `x²`, `x³` 등의 특성을 추가하여 곡선 관계를 표현
```text
선형회귀 : y = w₁x + b
다항회귀 : y = w₁x + w₂x² + w₃x³ + b
```

입력은 하나여도 여러 다항 특성으로 확장된다.
```text
x → x, x², x³, ..., xⁿ
```

---

## 2. 다항회귀도 선형 모델인 이유

그래프는 곡선이지만 
학습하는 계수 `w`에 대해서는 선형 결합
```text
ŷ = w₁x + w₂x² + w₃x³ + b
```

`x`, `x²`, `x³`을 각각 별도의 Feature로 보면
```text
ŷ = w₁X₁ + w₂X₂ + w₃X₃ + b
```
이므로 다항 특성을 만든 뒤 `LinearRegression`으로 계수를 학습

---

## 3. 다항식 차수(Degree)

`degree`는 다항식에서 사용하는 가장 높은 차수

| degree | 생성 특성 | 특징 |
|---|---|---|
| 1 | x | 직선 |
| 2 | x, x² | 곡선 표현 |
| 3 | x, x², x³ | 더 복잡한 곡선 |
| ↑ | Feature 증가 | Overfitting 가능성 증가 |

너무 낮으면 **과소적합**, 너무 높으면 **과적합**될 수 있으므로 
검증 성능을 기준으로 선택

---

## 4. 과소적합과 과적합

### 과소적합(Underfitting)
모델이 너무 단순하여 데이터의 주요 패턴을 충분히 학습하지 못한 상태
- Train 오차가 큼
- Test 오차도 큼
- 데이터의 곡선 패턴을 충분히 표현하지 못함

### 과적합(Overfitting)
모델이 실제 패턴뿐 아니라 학습 데이터의 잡음까지 학습
- Train 성능은 매우 높을 수 있음
- Test 성능은 떨어질 수 있음
- 예측 곡선이 불필요하게 복잡해질 수 있음

따라서 **학습 데이터에 가장 잘 맞는 degree가 아니라 
새로운 데이터에서도 잘 예측하는 degree를 선택**

---

## 5. `np.linspace()`

```python
x = np.linspace(-3, 3, 400)
```
지정한 범위에서 일정한 간격의 값을 생성한다.

```python
np.linspace(start, stop, num)
```

| 옵션 | 의미 |
|---|---|
| `start` | 시작값 |
| `stop` | 마지막 값 |
| `num` | 생성할 값의 개수 |

예측 곡선을 그릴 때 정렬된 연속 값을 만들기 위해 자주 사용한다.

```python
x_plot = np.linspace(x.min(), x.max(),400)
```

---

## 6. `np.polyfit()`

```python
coefficients = np.polyfit(x,y,deg=3)
```

주어진 데이터에 가장 잘 맞는 다항식 계수를 **최소제곱법**으로 계산한다.

| 옵션 | 의미 |
|---|---|
| `x` | 입력 데이터 |
| `y` | 목표 데이터 |
| `deg` | 다항식 차수 |

3차 다항식의 계수는 다음 순서로 반환된다.

```text
[a, b, c, d]
ax³ + bx² + cx + d
```

---

## 7. `np.poly1d()`

```python
polynomial = np.poly1d(coefficients)
```
다항식 계수 배열을 함수처럼 사용할 수 있는 객체로 변환

```python
y_pred = polynomial(x_plot)
```
다항식 자체도 출력하여 확인

```python
print(polynomial)
```

---

## 8. `PolynomialFeatures`

```python
PolynomialFeatures( degree=2, include_bias=False)
```
기존 Feature에서 **제곱항과 상호작용항**을 자동으로 생성한다.

```text
x1, x2
↓
x1, x2, x1², x1x2, x2²
```

| 옵션 | 의미 |
|---|---|
| `degree` | 생성할 최고 차수 |
| `include_bias` | 상수항 1 포함 여부 |
| `interaction_only` | 상호작용항만 생성할지 여부 |

`LinearRegression`이 기본적으로 절편을 학습하므로 일반적으로
```python
include_bias=False
```

---

## 9. `LinearRegression`

```python
LinearRegression()
```
다항 특성으로 변환된 데이터를 이용하여 각 Feature의 계수를 학습

```python
model.coef_       # Feature별 계수
model.intercept_  # 절편
```

---

## 10. `make_pipeline()`

```python
model = make_pipeline(
    PolynomialFeatures(
        degree=5,
        include_bias=False
    ),
    LinearRegression()
)
```

전처리와 모델 학습을 하나의 흐름으로 연결한다.

```text
원본 데이터
    ↓
PolynomialFeatures
    ↓
LinearRegression
    ↓
예측
```

Pipeline을 사용하면 전처리와 모델을 함께 관리할 수 있고, 
예측이나 교차검증에서도 같은 전처리가 자동으로 적용

---

## 11. `reshape(-1, 1)`

Scikit-learn은 입력 Feature를 기본적으로 2차원 형태로 받는다.

```python
x.reshape(-1, 1)
```

```text
변환 전 : [1, 2, 3]

변환 후 :
[[1],
 [2],
 [3]]
```

- `-1` → 데이터 개수에 맞게 행 수 자동 계산
- `1` → Feature가 한 개라는 의미

---

## 12. 회귀 모델 평가

| 지표 | 의미 | 좋은 값 |
|---|---|---|
| MSE | 제곱 오차의 평균 | 낮을수록 좋음 |
| RMSE | MSE의 제곱근 | 낮을수록 좋음 |
| R² | 모델 설명력 | 1에 가까울수록 좋음 |

### MSE

```python
mse = mean_squared_error(
    y_true,
    y_pred
)
```

큰 오차에 더 큰 패널티

### RMSE

```python
rmse = np.sqrt(mse)
```

Target과 같은 단위이므로 실제 오차 크기를 해석하기 용이

### R²

```python
r2 = r2_score(
    y_true,
    y_pred
)
```

모델이 Target의 변동을 얼마나 설명하는지를 나타낸다.

단, **Train R²만 높다고 좋은 모델은 아니므로 
Test 또는 Cross Validation 성능도 함께 확인해야 한다.**

---

## 13. 잔차(Residual)

잔차 == 실제값과 예측값의 차이
```text
Residual = 실제값 - 예측값
```

```python
residuals = y - y_pred
```

좋은 회귀 모델이라면 잔차가 **0 주변에 특별한 패턴 없이 분포**하는 것이 이상적
곡선이나 깔때기 형태가 나타난다면 모델 형태, 분산, 이상치 등을 추가로 확인해야 한다.

---

## 14. 높은 차수의 문제와 규제

degree가 높아질수록 Feature 수와 값의 크기가 증가한다.

예를 들어
```text
x = 100
x⁵ = 10,000,000,000
```

이처럼 값의 크기가 크게 달라지면
- 계수 계산이 불안정해질 수 있음
- 작은 데이터 변화에도 모델이 민감해질 수 있음
- 과적합 가능성이 증가함

이를 완화하기 위해 다음 방법을 사용할 수 있다.

- `StandardScaler`
- Ridge / Lasso
- Cross Validation
- 지나치게 높은 degree 제한

예:

```python
from sklearn.linear_model import Ridge

model = make_pipeline(
    PolynomialFeatures(
        degree=5,
        include_bias=False
    ),
    Ridge(alpha=1.0)
)
```

`alpha`가 커질수록 계수에 더 강한 제한을 적용한다.

---

## 15. 외삽(Extrapolation)

다항회귀는 **학습 데이터 범위 밖의 예측에 특히 주의**해야 한다.

```text
학습 범위 : 0 ≤ x ≤ 10
예측 대상 : x = 100
```

최고차항의 영향이 커지면서 학습 범위 밖에서는 예측값이 급격히 증가하거나 감소할 수 있다.
따라서 다항회귀는 
**학습 범위 내부의 패턴 분석에 더 적합하고, 먼 범위의 외삽은 신중하게 사용**해야

---

## 16. 주요 코드

```python
# 다항 특성 변환 + 선형회귀 모델 생성
model = make_pipeline(
    PolynomialFeatures(
        degree=5,             # 5차까지 다항 특성 생성
        include_bias=False    # 상수항 1은 생성하지 않음
    ),
    LinearRegression()        # 생성된 다항 특성으로 선형회귀 적용
)

# 모델 학습
model.fit(
    x.reshape(-1, 1),         # x를 Scikit-learn이 요구하는 2차원 형태로 변환
    y                        # 실제 정답값
)
```

예측용 데이터 생성:

```python
# 예측 곡선을 그리기 위한 연속적인 x값 생성
x_plot = np.linspace(
    x.min(),                  # x의 최솟값부터
    x.max(),                  # x의 최댓값까지
    400                       # 400개의 값을 일정한 간격으로 생성
)

# 학습한 다항회귀 모델로 예측
y_pred = model.predict(
    x_plot.reshape(-1, 1)     # 예측용 x도 2차원 형태로 변환
)
```

make_pipeline() → 다항회귀 모델 구성
fit()           → 모델 학습
np.linspace()   → 그래프용 연속 x값 생성
predict()       → 학습한 모델로 예측


---

## 17. 결과 그래프

### 다항식 차수별 비교

- 낮은 degree → 데이터의 곡선 패턴을 충분히 표현하지 못할 수 있음
- 높은 degree → 학습 데이터의 잡음까지 따라갈 수 있음
- 적절한 degree → Test 또는 Cross Validation 성능을 기준으로 선택

### 다항회귀 결과

- 점 → 실제 데이터
- 곡선 → 학습된 다항회귀 모델의 예측값
- 잔차 → 실제값과 예측값의 차이

---

## 18. Matplotlib에서 사용한 주요 코드

```python
plt.scatter(x, y)                 # 실제 데이터
plt.plot(x_plot, y_pred)          # 예측 곡선
plt.grid(linestyle="--")          # 격자
plt.legend()                      # 범례
plt.tight_layout()                # 여백 자동 조정
plt.show()                        # 그래프 출력
```

예측 곡선은 입력된 순서대로 점을 연결하므로,
`x_plot`처럼 **정렬된 입력값을 사용**하는 것이 안전하다.

---

## 19. 정리

- 다항회귀는 입력값을 다항 특성으로 확장하여 비선형 관계를 표현한다.
- 다항 특성을 생성한 뒤 `LinearRegression`으로 각 항의 계수를 학습한다.
- degree가 낮으면 과소적합, 너무 높으면 과적합될 수 있다.
- 모델 선택은 Train 성능보다 Test 또는 Cross Validation 성능을 기준으로 한다.
- 높은 차수에서는 Scaling과 Ridge/Lasso 규제를 고려한다.
- 다항회귀는 학습 범위 밖의 외삽에 특히 주의해야 한다.
