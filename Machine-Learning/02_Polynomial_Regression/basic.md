# 다항회귀(Polynomial Regression)

직선으로 표현하기 어려운 비선형 데이터를 
다항 특성으로 변환하여 회귀 모델 aksemfrl

- 다항함수의 기본 형태
- `np.linspace()`를 이용한 연속 데이터 생성
- `np.polyfit()`을 이용한 차수별 다항식 적합
- 다항식 차수에 따른 과소적합과 과적합 비교
- `PolynomialFeatures`를 이용한 다항 특성 생성
- `Pipeline`을 이용한 전처리와 모델 학습 연결
- `LinearRegression`을 이용한 계수 학습
- MSE, RMSE, R²를 이용한 모델 평가
- 예측 곡선과 잔차 시각화

---

## 다항회귀
선형회귀는 입력과 출력의 관계가 직선

```text
y = w₁x + b
```

다항회귀는 입력값의 제곱, 세제곱 등의 항을 추가하여 
곡선 형태의 관계

```text
y = w₁x + w₂x² + w₃x³ + b
```
입력 변수는 하나이지만 다음처럼 여러 특성으로 확장
```text
원래 입력값: x
변환된 특성:
x, x², x³, ..., xⁿ
```

---

## 다항회귀도 선형 모델인 이유
다항회귀는 입력값 `x`에 대해서는 곡선을 만들지만, 
학습하는 가중치에 대해서는 선형 결합

```text
ŷ = w₁x + w₂x² + w₃x³ + b
```

`x`, `x²`, `x³`을 각각 별도의 입력 특성으로 생각하면

```text
ŷ = w₁X₁ + w₂X₂ + w₃X₃ + b
```
따라서 다항 특성을 생성한 후 `LinearRegression`으로 계수를 학습

---

## 다항식 차수(Degree)
차수는 다항식에서 사용하는 가장 높은 지수

```python
PolynomialFeatures(degree=5)
```
5차 다항회귀라면

```text
x, x², x³, x⁴, x⁵
```

### 차수가 낮은 경우
- 모델이 단순하다.
- 계산량이 적다.
- 복잡한 데이터의 패턴을 충분히 표현하지 못할 수 있다.
- 과소적합이 발생할 수 있다.

### 차수가 높은 경우
- 복잡한 곡선을 표현할 수 있다.
- 학습 데이터의 작은 변화와 잡음까지 학습할 수 있다.
- 과적합 가능성이 높아진다.
- 큰 거듭제곱 값으로 인해 수치적으로 불안정할 수 있다.

---

## 과소적합과 과적합

### 과소적합(Underfitting)

모델이 너무 단순하여 데이터의 중요한 패턴을 충분히 학습하지 못한 상태
예를 들어 곡선 형태의 데이터를 1차 직선으로 학습하면
- 학습 데이터 오차가 큼
- 테스트 데이터 오차도 큼
- 주요 변화 패턴을 표현하지 못함

### 과적합(Overfitting)
모델이 학습 데이터의 실제 패턴뿐 아니라 잡음까지 학습한 상태
- 학습 데이터 오차는 매우 작음
- 새로운 데이터에서 오차가 커질 수 있음
- 예측 곡선이 불필요하게 많이 흔들릴 수 있음

### 적절한 차수 선택
학습 데이터에 가장 잘 맞는 차수가 아니라, 
**새로운 데이터에서도 잘 예측하는 차수**를 선택

이를 위해
- 학습 데이터와 테스트 데이터 분리
- 검증 데이터 사용
- 교차검증
- 규제 적용
- 검증 오차를 기준으로 차수 선택

---

## `np.linspace()`

```python
x = np.linspace(-3, 3, 400)
```
지정한 구간에서 동일한 간격의 숫자를 생성한다.

### 주요 옵션

```python
np.linspace(start, stop, num)
```

| 옵션 | 설명 |
|---|---|
| `start` | 시작값 |
| `stop` | 마지막 값 |
| `num` | 생성할 값의 개수 |
| `endpoint` | 마지막 값을 포함할지 여부 |
| `dtype` | 생성할 배열의 자료형 |

### 예시

```python
np.linspace(0, 10, 5)
```

결과:
```text
[0.0, 2.5, 5.0, 7.5, 10.0]
```
`num`을 크게 설정하면 그래프의 곡선이 더 부드럽게

---

## `np.polyfit()`

```python
coefficients = np.polyfit(
    x,
    y,
    deg=3
)
```
주어진 데이터에 가장 잘 맞는 다항식 계수를 최소제곱법으로 계산

### 주요 옵션

| 옵션 | 설명 |
|---|---|
| `x` | 입력 데이터 |
| `y` | 목표 데이터 |
| `deg` | 다항식 차수 |
| `rcond` | 작은 특이값을 무시하는 기준 |
| `full` | 추가 진단 정보를 반환할지 여부 |
| `cov` | 계수의 공분산 행렬 반환 여부 |

3차 다항식이라면 다음 순서로 계수를 반환한다.

```text
[a, b, c, d]
```
이는

```text
ax³ + bx² + cx + d
```

---

## `np.poly1d()`

```python
polynomial = np.poly1d(coefficients)
```
다항식 계수 배열을 실제 함수처럼 사용할 수 있는 객체로 변환

```python
y_pred = polynomial(x_plot)
```
또한 다항식을 출력해 확인

```python
print(polynomial)
```

---

## `PolynomialFeatures`

```python
PolynomialFeatures(
    degree=5,
    include_bias=False
)
```

기존 입력값을 다항 특성으로 확장

### 주요 옵션

| 옵션 | 설명 |
|---|---|
| `degree` | 생성할 다항 특성의 최고 차수 |
| `include_bias` | 값이 항상 1인 편향 특성을 포함할지 여부 |
| `interaction_only` | 각 변수의 거듭제곱 없이 변수 간 상호작용만 생성할지 여부 |
| `order` | 출력 배열의 메모리 저장 순서 |

### `include_bias=False`

`include_bias=True`이면 값이 항상 1인 특성이 추가

```text
[1, x, x², x³]
```
하지만 `LinearRegression`이 기본적으로 절편을 학습하므로 
중복을 피하기 위해 이렇게 설정

```python
include_bias=False
```

### 다변수 입력 예시
입력 특성이 `x1`, `x2`이고 차수가 2이면

```text
x1, x2, x1², x1x2, x2²
```
즉, 다항 특성뿐 아니라 변수 간 상호작용도 표현 가능

---

## `LinearRegression`

```python
LinearRegression(
    fit_intercept=True
)
```
다항 특성으로 변환된 데이터를 선형 결합하여 예측

### 주요 옵션

| 옵션 | 설명 |
|---|---|
| `fit_intercept` | 절편을 학습할지 여부 |
| `copy_X` | 입력 데이터를 복사할지 여부 |
| `n_jobs` | 병렬 계산에 사용할 작업 수 |
| `positive` | 계수를 양수로 제한할지 여부 |

학습 후 주요 속성:

```python
model.coef_
```

각 특성의 가중치

```python
model.intercept_
```
학습된 절편

---

## `make_pipeline()`

```python
model = make_pipeline(
    PolynomialFeatures(
        degree=5,
        include_bias=False
    ),
    LinearRegression()
)
```
전처리와 모델 학습 단계를 순서대로 연결

```text
원본 x
   ↓
다항 특성 변환
   ↓
선형회귀 학습
   ↓
예측값 출력
```

### 장점
- 전처리와 모델 학습을 함께 관리할 수 있다.
- 예측할 때도 같은 전처리가 자동으로 적용된다.
- 교차검증 시 데이터 누수를 방지하기 쉽다.
- 코드가 간결해진다.

---

## `reshape(-1, 1)`
Scikit-learn은 입력값을 2차원 형태로 받는다.

```python
x.reshape(-1, 1)
```

변환 전:

```text
[1, 2, 3]
```

변환 후:

```text
[[1],
 [2],
 [3]]
```

- `-1`: 데이터 개수에 맞게 행 수를 자동 계산
- `1`: 입력 특성이 한 개라는 의미

---

## 회귀 모델 평가

### MSE

평균제곱오차는 실제값과 예측값 차이의 제곱을 평균한 값
```python
mse = mean_squared_error(
    y_true,
    y_pred
)
```

```text
MSE = 평균((실제값 - 예측값)²)
```

- 값이 작을수록 예측 오차가 작다.
- 큰 오차에 더 큰 패널티를 준다.
- 단위가 목표 변수 단위의 제곱이다.

### RMSE

MSE에 제곱근을 적용한 값

```python
rmse = np.sqrt(mse)
```
- 목표 변수와 같은 단위를 갖는다.
- 실제 평균 오차 크기를 이해하기 쉽다.

### R²

```python
r2 = r2_score(
    y_true,
    y_pred
)
```
모델이 목표값의 변동을 얼마나 설명하는지 
- `1`에 가까움: 데이터 변동을 잘 설명
- `0`에 가까움: 평균값으로 예측하는 수준과 비슷함
- 음수: 평균값으로 예측하는 것보다 성능이 나쁠 수 있음

R²가 높다고 해서 반드시 과적합이 없는 것은 아니다.
학습 데이터의 R²만으로 모델을 선택해서는 안 된다.

---

## 잔차(Residual)

잔차는 실제값과 예측값의 차이

```text
잔차 = 실제값 - 예측값
```

```python
residuals = y - y_pred
```
좋은 회귀 모델이라면 잔차가 0을 중심으로 특별한 패턴 없이 분포하는 것이 이상적

잔차 그래프에서 곡선이나 깔때기 모양이 나타난다면
- 모델의 형태가 데이터에 적합하지 않음
- 분산이 일정하지 않음
- 이상치 존재
- 중요한 변수가 누락됨

![다항회귀 잔차](./images/polynomial_regression_residuals.png)

---

## Matplotlib 그래프 옵션

### `plt.figure()`

```python
plt.figure(
    figsize=(9, 5)
)
```

| 옵션 | 설명 |
|---|---|
| `figsize` | 그래프의 가로·세로 크기 |
| `dpi` | 화면에 표시되는 해상도 |
| `facecolor` | 그래프 전체 배경색 |
| `layout` | 그래프 배치 방식 |

---

### `plt.scatter()`

```python
plt.scatter(
    x,
    y,
    s=30,
    alpha=0.8,
    label="실제 데이터"
)
```

| 옵션 | 설명 |
|---|---|
| `x`, `y` | 점의 좌표 |
| `s` | 점의 크기 |
| `alpha` | 투명도 |
| `marker` | 점의 모양 |
| `label` | 범례 이름 |
| `edgecolors` | 점 테두리 색 |
| `linewidths` | 점 테두리 두께 |

---

### `plt.plot()`

```python
plt.plot(
    x_plot,
    y_pred,
    linewidth=2.5,
    label="예측 곡선"
)
```

| 옵션 | 설명 |
|---|---|
| `linewidth` | 선 두께 |
| `linestyle` | 선 모양 |
| `label` | 범례 이름 |
| `marker` | 각 데이터 위치에 표시할 마커 |
| `alpha` | 투명도 |

`plot()`은 입력된 순서대로 점을 연결한다.
따라서 `x`가 정렬되어 있지 않으면 선이 지그재그

예측 곡선을 그릴 때는 정렬된 데이터를 사용하는 것이 안전!
```python
x_plot = np.linspace(
    x.min(),
    x.max(),
    400
)
```

---

### `plt.grid()`

```python
plt.grid(
    visible=True,
    linestyle="--",
    alpha=0.4
)
```

| 옵션 | 설명 |
|---|---|
| `visible` | 격자 표시 여부 |
| `linestyle` | 격자선 모양 |
| `linewidth` | 격자선 두께 |
| `alpha` | 격자선 투명도 |
| `axis` | `x`, `y`, `both` 중 적용 축 |

---

### `plt.legend()`

```python
plt.legend()
```
`plot()`이나 `scatter()`의 `label`을 범례로 표시

주요 옵션:
```python
plt.legend(
    loc="best",
    fontsize=10,
    frameon=True
)
```

| 옵션 | 설명 |
|---|---|
| `loc` | 범례 위치 |
| `fontsize` | 글자 크기 |
| `frameon` | 범례 테두리 표시 여부 |
| `ncol` | 범례 열 개수 |

---

### `plt.subplot()`

```python
plt.subplot(
    2,
    2,
    index + 1
)
```

의미:
```text
2행 × 2열 그래프 중 index + 1번째 위치
```

예:
```python
plt.subplot(2, 2, 1)
```
== 2행 2열 중 첫 번째 그래프

---

### `plt.tight_layout()`

```python
plt.tight_layout()
```
제목, 축 이름, 범례 등이 겹치거나 잘리지 않도록 여백을 자동으로 조정

---

### `plt.savefig()`

```python
plt.savefig(
    "images/result.png",
    dpi=150,
    bbox_inches="tight"
)
```

| 옵션 | 설명 |
|---|---|
| `dpi` | 저장 이미지 해상도 |
| `bbox_inches="tight"` | 불필요한 외부 여백 제거 |
| `transparent` | 배경을 투명하게 저장할지 여부 |
| `facecolor` | 저장 이미지 배경색 |

`savefig()`는 보통 `show()`보다 먼저 호출하는 것이 안전

---

## 다항 특성의 문제점

차수가 높아질수록 입력값의 거듭제곱이 매우 커질 수 있다.

예를 들어 `x = 100`이고 차수가 5라면:

```text
x⁵ = 10,000,000,000
```

특성 간 크기 차이가 커지면
- 회귀계수 계산이 불안정해짐
- 작은 데이터 변화에 계수가 크게 변함
- 과적합 가능성 증가

이를 완화하기 위해서는
- `StandardScaler`로 특성 스케일링
- Ridge 또는 Lasso 규제
- 너무 높은 차수 사용 제한
- 교차검증으로 차수 선택

---

## 규제(Regularization)
높은 차수에서 계수가 지나치게 커지는 것을 방지하기 위해 규제를 사용할 수 있다.
예: Ridge 회귀

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

`alpha`가 커질수록 가중치에 더 강한 제한
(다만 너무 크게 설정하면 모델이 단순해져 과소적합)

---

## 외삽(Extrapolation)
다항회귀는 학습 데이터 범위 안에서는 자연스러운 곡선을 만들 수 있지만, 
범위 밖 예측에서는 값이 급격하게 증가하거나 감소
예를 들어 
학습 범위가

```text
0 ≤ x ≤ 10
```

`x = 100`과 같은 값의 예측은 신뢰하기 어렵다.
다항식은 최고차항의 영향을 크게 받기 때문에 학습 범위 밖에서 매우 빠르게 변할 수 있다

---

## 주요 코드

```python
model = make_pipeline(
    PolynomialFeatures(
        degree=5,
        include_bias=False
    ),
    LinearRegression()
)

model.fit(
    x.reshape(-1, 1),
    y
)
```

```python
x_plot = np.linspace(
    x.min(),
    x.max(),
    400
)

y_pred = model.predict(
    x_plot.reshape(-1, 1)
)
```

---

## 결과 그래프

### 다항식 차수별 비교

![다항식 차수별 비교](./images/polynomial_degree_comparison.png)

- 낮은 차수는 복잡한 패턴을 표현하지 못할 수 있다.
- 높은 차수는 데이터의 잡음까지 따라갈 수 있다.
- 적절한 차수는 테스트 또는 검증 성능을 기준으로 결정해야 

### 다항회귀 결과

![다항회귀 예측 결과](./images/polynomial_regression_result.png)

- 점은 실제 데이터
- 곡선은 학습된 다항회귀 모델의 예측값

---

## 추가로 학습할 내용

- `train_test_split()`을 이용한 학습·테스트 데이터 분리
- 교차검증을 이용한 최적 차수 선택
- `StandardScaler`를 이용한 다항 특성 스케일링
- Ridge와 Lasso 규제
- 학습 오차와 검증 오차 비교
- 잔차 분석
- Pipeline과 데이터 누수
- 모델의 외삽 위험성

---

## 정리

- 다항회귀는 입력값을 거듭제곱 특성으로 확장하여 곡선 관계를 표현한다
- 다항 특성을 만든 후에는 선형회귀로 각 항의 계수를 학습한다
- 차수가 낮으면 과소적합, 너무 높으면 과적합이 발생할 수 있다
- `PolynomialFeatures`는 다항 특성과 상호작용 특성을 자동으로 생성한다
- `Pipeline`은 전처리와 모델 학습을 하나의 과정으로 묶는다
- 예측 곡선을 그릴 때는 정렬된 입력값을 사용해야 한다
- 학습 데이터 성능만으로 차수를 선택하면 안 된다
- 높은 차수에서는 스케일링과 규제를 고려해야 한다
- 다항회귀는 학습 범위 밖의 예측에 특히 주의
