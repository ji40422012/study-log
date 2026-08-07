# 실무형 다중선형회귀(Multiple Linear Regression)

## 학습 내용
(with scikit-learn.load_diabetes)
다중선형회귀 모델을 실무형 흐름으로 구현

- Multiple Linear Regression
- MAE / MSE / RMSE / R²
- 실제값 vs 예측값
- 회귀계수 해석
- Residual(잔차) 분석
- Feature 상관관계
- VIF와 다중공선성
- 5-Fold Cross Validation
- Ridge Regression
- Lasso Regression
- Lasso의 `alpha` 변화
- Linear / Ridge / Lasso 성능 비교

---

# 1. 데이터
```python
from sklearn.datasets import load_diabetes

data = load_diabetes()

X = data.data
y = data.target
feature_names = data.feature_names
```

데이터 구성:
```text
X shape: (442, 10)
y shape: (442,)
```

Feature:
```text
age
sex
bmi
bp
s1
s2
s3
s4
s5
s6
```
`target`은 **질병 진행 정도를 나타내는 연속형 값**
`load_diabetes()`의 Feature는 이미 평균 중심화 및 스케일 조정된 값
---

# 2. 다중선형회귀
다중선형회귀 == 여러 Feature를 동시에 사용하여 하나의 연속형 값을 예측

```text
y = b + w1x1 + w2x2 + ... + wnxn
```
- `x` : 입력 Feature
- `w` : 각 Feature의 회귀계수
- `b` : 절편
- `y` : 예측값

```text
질병 진행 정도
=
b
+ age 계수
+ bmi 계수
+ bp 계수
+ ...
```
각 회귀계수는 
**다른 Feature가 동일하다는 가정하에 
해당 Feature 변화와 Target 사이의 방향**

```text
coef > 0 → Feature 증가 시 Target 증가 방향
coef < 0 → Feature 증가 시 Target 감소 방향
```
단, 회귀계수는 인과관계를 의미하지 않음

---

# 3. Train / Test 분리

```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
```
- `test_size=0.2` → 전체 데이터의 20%를 Test로 사용
- `random_state=42` → 데이터 분할 결과 고정
- 관례상 대문자X → Feature(입력 데이터, 행렬), 소문자y → Target(정답 데이터, 벡터)

결과:
```text
Train: (353, 10)
Test : (89, 10)
```
Train 데이터로 모델을 학습하고 
Test 데이터는 새로운 데이터에 대한 성능 평가에 사용

---

# 4. 모델 학습과 예측

```python
model = LinearRegression()
model.fit(X_train, y_train)    #→ 모델 학습
y_pred = model.predict(X_test) #→ 학습된 모델로 예측
```
`fit()` 과정에서 각 Feature의 최적 회귀계수와 절편을 계산한다.

---

# 5. 회귀 모델 평가 지표
```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```
## MAE(Mean Absolute Error)
- 실제값과 예측값의 절대 오차 평균
```text
MAE ↓ → 평균적인 예측 오차가 작음
```
---
## MSE(Mean Squared Error)
- 오차를 제곱하여 평균낸 값
- 큰 오차에 더 큰 패널티를 준다.
```text
MSE ↓ → 좋음
```
---
## RMSE(√MSE)
- MSE에 제곱근을 적용하여 Target과 같은 단위로 다시 변환
```text
RMSE ↓ → 예측 오차가 작음
```
---
## R²
- 결정계수(Coefficient of Determination): 모델의 설명력
```text
R² ↑ → 모델 설명력이 높음
```
`R² = 0.45`라고 해서 정확도가 45%라는 의미는 아니다.
Target의 변동 중 약 45%를 현재 선형모델이 설명한다는 의미로 해석

---

# 6. Linear Regression 결과
실행 결과:

```text
MAE : 약 42.8
MSE : 약 2900
RMSE: 약 53.9
R²  : 약 0.45  → 약 절반 정도의 변동은 현재 Feature와 단순 선형관계만으로 설명되지 않는다.
```
모델은 전체적인 관계는 학습했지만 Target 변동 전체를 충분히 설명하지는 못했다.

가능한 원인:
- Feature가 부족할 수 있음
- 비선형 관계가 존재할 수 있음
- 데이터 자체의 변동성이 클 수 있음
- 선형모델의 표현력에 한계가 있을 수 있음

---

# 7. 실제값 vs 예측값
```python
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
```
- 점이 대각선에 가까울수록 예측이 정확

### 결과 해석
전체적으로 증가 방향은 따라가지만 대각선에서 떨어진 점도 상당
- 즉 모델이 전체적인 경향은 학습했지만 개별 데이터의 정확한 값을 모두 맞추지는 못했다.

---

# 8. 회귀계수
```python
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_
})
```

각 Feature의 계수를 확인하여 모델이 어느 방향으로 Target을 예측하는지 확인
```text
coef > 0 → Target 증가 방향
coef < 0 → Target 감소 방향
```
계수의 절댓값이 크면 모델에서 상대적으로 큰 영향을 가지지만, 
**다중공선성과 Feature Scale도 함께 고려해야**

---

# 9. Residual(잔차)
- 잔차는 실제값과 예측값의 차이
```python
residual = y_test - y_pred
```
```text
Residual = Actual - Predicted
```

잔차 그래프:
```python
plt.scatter(y_pred, residual, alpha=0.7)
plt.axhline(0, linestyle="--")
```

- 좋은 선형회귀라면 
- 잔차가 0을 중심으로 특별한 패턴 없이 분포하는 것이 이상적

```text
무작위 분포 → 선형모델이 비교적 적절
곡선 패턴 → 비선형 관계 가능
깔때기 모양 → 오차 분산이 일정하지 않을 가능성
```
따라서 단순히 R²만 보는 것보다 잔차를 함께 확인 필요

---

# 10. Feature 상관관계
```python
corr = df.drop(columns="target").corr()
```
Feature 간 Pearson 상관계수 연산

```text
상관계수 ≈ 1  → 강한 양의 관계
상관계수 ≈ -1 → 강한 음의 관계
상관계수 ≈ 0  → 선형관계가 약함
```
상관관계가 높은 Feature가 여러 개 존재하면 다중공선성 문제
---

# 11. 다중공선성(Multicollinearity)
다중공선성은 서로 비슷한 정보를 가진 Feature가 존재
예를 들면 집 전체 면적, 거실 면적,방 개수 (서로 강한 상관관계)

다중공선성이 심하면:
```text
예측 성능은 괜찮더라도 회귀계수가 불안정해질 수 있음
```
즉 데이터가 조금만 달라져도 특정 Feature의 계수가 크게 변할 수 있다.

---

# 12. VIF(Variance Inflation Factor)는 다중공선성을 확인
보통은 아래같이 하지만
```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["Feature"] = df.columns
vif["VIF"] = [
    variance_inflation_factor(df.values, i)
    for i in range(df.shape[1])
]
print(vif)
```
각 Feature를 나머지 Feature로 예측하여 VIF를 직접 계산해서 기본부터 짚어본다
```text
VIF = 1 / (1 - R²)
```

```python
X_target = X[:, i]
X_others = np.delete(X, i, axis=1)
vif_model.fit(X_others, X_target)
r2_i = vif_model.score(X_others, X_target)
vif = 1 / (1 - r2_i)
```

일반적인 기준(`5` 또는 `10`이 절대적인 제거 기준은 아님)
```text
VIF ≈ 1  → 다중공선성 거의 없음
VIF > 5  → 확인 필요
VIF > 10 → 높은 다중공선성 의심
```
---

# 13. 5-Fold Cross Validation

Train/Test를 한 번만 나누면 특정 데이터 분할에 따라 성능이 달라질 수 있으니
5-Fold Cross Validation 활용, 
전체 데이터를 5개 Fold로 나누고 각각을 한 번씩 검증 데이터로 사용한다.

```text
1회: Fold1 → Test
2회: Fold2 → Test
3회: Fold3 → Test
4회: Fold4 → Test
5회: Fold5 → Test
```

```python
cv_scores = cross_val_score(
    LinearRegression(),
    X,
    y,
    cv=5,
    scoring="r2"
)
```

결과 예:
```text
각 Fold R²:
0.430
0.523
0.483
0.426
0.550

평균 R² ≈ 0.48
표준편차 ≈ 0.05
```
- 평균 R² → 여러 데이터 분할에서의 평균 성능
- 표준편차 → Fold별 성능 변동 정도

```text
평균 ↑ + 표준편차 ↓
→ 안정적인 모델
```
한 번의 Test 결과보다 여러 Fold에서 비슷한 성능이 나오는지를 확인하는 것이 중요

---

# 14. Ridge Regression

Ridge는 Linear Regression에 **L2 규제**를 추가한 모델이다.
```text
Loss = MSE + α × Σ(w²)
```
Linear Regression은 예측 오차를 줄이는 데 집중하지만 
Ridge는 회귀계수가 지나치게 커지는 것도 함께 제한한다.

```python
ridge = Ridge(alpha=0.1)

ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
```

`alpha`는 규제 강도
```text
alpha = 0 → 일반 Linear Regression과 유사
alpha ↑   → 규제 강해짐
```

### 특징
- 큰 계수를 줄여 모델을 안정화
- 과적합 완화 가능
- 다중공선성 문제에서 유용
- 계수를 거의 0에 가깝게 만들 수 있지만 일반적으로 정확히 0으로 만들지는 않음

---

# 15. Lasso Regression
Lasso는 **L1 규제**를 사용

```text
Loss = MSE + α × Σ|w|
```
코드:
```python
lasso = Lasso(alpha=0.1, max_iter=10000)

lasso.fit(X_train, y_train)              # Lasso 모델 학습
lasso_pred = lasso.predict(X_test)       # 학습한 모델로 예측
lasso_r2 = r2_score(y_test, lasso_pred)  # 모델 설명력(R²) 계산
lasso_rmse = np.sqrt(
    mean_squared_error(y_test, lasso_pred)
)                                       # 예측 오차(RMSE) 계산
```

### 특징
Lasso는 일부 Feature의 계수를 정확히 `0`으로 만들 수 있다.

```text
coef = 0
→ 해당 Feature를 모델이 사실상 사용하지 않음
```
따라서 **Feature Selection 효과**

---

# 16. Ridge와 Lasso 차이

| 구분 | Ridge | Lasso |
|---|---|---|
| 규제 | L2 | L1 |
| 계수 | 전체적으로 축소 | 축소 + 일부 0 |
| Feature 제거 | 거의 없음 | 가능 |
| 다중공선성 | 유용 | 유용 |
| Feature Selection | X | O |

```text
Ridge → 다 같이 조금씩 줄인다
Lasso → 필요 없는 Feature는 0으로 만든다
```

---

# 17. Linear / Ridge / Lasso 비교
```python
comparison = pd.DataFrame({
    "Model": ["Linear", "Ridge", "Lasso"],
    "R²": [r2, ridge_r2, lasso_r2],
    "RMSE": [rmse, ridge_rmse, lasso_rmse]
})
```

평가 기준:
```text
R²   → 높을수록 좋음
RMSE → 낮을수록 좋음
```

규제를 적용했다고 항상 Linear Regression보다 좋은 성능이 나오는 것은 아니다.
규제의 목적은 단순히 Test 점수를 무조건 올리는 것보다 
**계수의 복잡도를 줄이고 새로운 데이터에서의 안정성을 높이는 것**

---

# 18. Lasso alpha 비교
Lasso의 규제 강도를 여러 값으로 변경하여 성능과 Feature 선택 변화 확인

```python
for alpha in [0.001, 0.01, 0.1, 1, 10]:

    model = Lasso(alpha=alpha, max_iter=10000)

    model.fit(X_train, y_train)                # 현재 alpha로 모델 학습
    y_pred_alpha = model.predict(X_test)       # Test 데이터 예측

    r2_alpha = r2_score(y_test, y_pred_alpha)  # R² 계산
    rmse_alpha = np.sqrt(
        mean_squared_error(y_test, y_pred_alpha)
    )                                         # RMSE 계산

    zero_coef = np.sum(model.coef_ == 0)       # 0이 된 Feature 개수
```

```text
alpha ↓
→ 규제가 약함
→ Linear Regression과 비슷

alpha 적당
→ 불필요한 계수 감소
→ 성능 유지 가능

alpha 너무 큼
→ 많은 Feature가 0
→ 모델이 너무 단순
→ Underfitting 가능
```
따라서 `alpha`는 성능과 모델 복잡도의 균형을 고려해 결정
---

# 19. Cross Validation 모델 비교

```python
ridge_cv = cross_val_score(
    Ridge(alpha=0.1),
    X,
    y,
    cv=5,
    scoring="r2"
)

lasso_cv = cross_val_score(
    Lasso(alpha=0.1),
    X,
    y,
    cv=5,
    scoring="r2"
)
```

```text
cross_val_score()
→ 여러 Fold에서 모델 성능 반복 평가
```

Linear / Ridge / Lasso의 평균 R²를 비교하여 특정 Train/Test 분할이 아니라 
여러 데이터 구간에서도 성능이 유지되는지 확인

```text
평균 R²가 높음 + Fold별 변동이 작음
→ 일반화 성능이 비교적 안정적
```

---

# 20. 실습에서 사용한 주요 코드

### 데이터 분리
```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

### Linear Regression
```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 회귀 평가
```python
mean_absolute_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)
```

### 잔차
```python
residual = y_test - y_pred
```

### 상관관계
```python
df.corr()
```

### Cross Validation
```python
cross_val_score(model, X, y, cv=5, scoring="r2")
```

### Ridge
```python
Ridge(alpha=0.1)
```

### Lasso
```python
Lasso(alpha=0.1, max_iter=10000)
```

### Lasso Feature Selection 확인
```python
np.sum(model.coef_ == 0)
```

---

# 21. 분석 결과
이번 분석에서 Linear Regression의 Test R²는 약 `0.45` 수준
== 모델이 일부 관계는 학습했지만 데이터의 전체 변동을 선형식만으로 충분히 설명하지 못한다는 의미

5-Fold Cross Validation에서도 평균 R²가 약 `0.48` 수준
== 특정 Test 분할에서만 우연히 나온 결과는 아님

Feature 간 상관관계와 VIF를 확인
== 다중회귀에서는 예측 성능뿐 아니라 **Feature끼리 서로 중복된 정보를 가지고 있는지** 확인 필요

Ridge와 Lasso 적용
== 규제가 회귀계수를 어떻게 제어하는지 비교
- Lasso는 `alpha`가 증가할수록 계수가 0이 되는 Feature가 증가하여 Feature Selection 효과
- 다만 규제가 너무 강하면 중요한 Feature까지 제거되어 성능이 떨어지는 Underfitting이 발생

---

# 정리

```text
데이터 확인
    ↓
Train / Test 분리
    ↓
Linear Regression
    ↓
MAE / MSE / RMSE / R²
    ↓
실제값 vs 예측값
    ↓
회귀계수 분석
    ↓
Residual 분석
    ↓
상관관계 / VIF
    ↓
Cross Validation
    ↓
Ridge
    ↓
Lasso
    ↓
alpha 변화
    ↓
모델 비교
```
