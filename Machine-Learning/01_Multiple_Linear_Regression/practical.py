import matplotlib.pyplot as plt  # 그래프 시각화
import numpy as np               # 수치 계산
import pandas as pd              # 데이터프레임

from sklearn.datasets import load_diabetes       # 당뇨병 회귀 내장 데이터
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 회귀 모델
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 회귀 평가
from sklearn.model_selection import train_test_split, cross_val_score  # 데이터 분리 / 교차검증

# =========================================================
# 1. 데이터 불러오기 : load_diabetes(sklearn.datasets 내장된 Regression 활용(연속형) 데이터)
# =========================================================
data = load_diabetes()
X = data.data                  # 10개의 입력 Feature
y = data.target                # 질병 진행 정도를 나타내는 연속값
feature_names = data.feature_names

print("X shape:", X.shape)              # r: (442, 10)
print("y shape:", y.shape)              # r: (442,)
print("Features:", feature_names)       # r: ['age', 'sex', 'bmi', 'bp', 's1', ... 's6']
# 입력 Feature들은 이미 평균 중심화 및 스케일 조정되어 제공

# =========================================================
# 2. DataFrame으로 데이터 확인
# =========================================================
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("\n===== 데이터 확인 =====")
print(df.head())
print("\n결측치:", df.isnull().sum().sum())  # r: 0

# =========================================================
# 3. Train / Test 분리
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,                  # 입력 데이터
    y,                  # Target
    test_size=0.2,      # Test 20%
    random_state=42     # 분할 결과 고정
)

print("\nTrain:", X_train.shape)  # r: (353, 10)
print("Test :", X_test.shape)     # r: (89, 10)

# =========================================================
# 4. Multiple Linear Regression
# =========================================================
model = LinearRegression()
model.fit(
    X_train,    # 학습 Feature
    y_train     # 실제 Target
)
y_pred = model.predict(X_test)

print("\n절편:", model.intercept_)  # r: 약 151.346

# =========================================================
# 5. 모델 평가
# =========================================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== 모델 평가 =====")
print(f"MAE : {mae:.3f}")    # r: 42.794
print(f"MSE : {mse:.3f}")    # r: 2900.194
print(f"RMSE: {rmse:.3f}")   # r: 53.853
print(f"R²  : {r2:.3f}")     # r: 0.453
# MAE  → 실제값과 예측값이 평균적으로 약 42.8 차이
# RMSE → 큰 오차까지 고려하면 약 53.9
# R²   → Test Target 변동의 약 45%를 모델이 설명

# =========================================================
# 6. 실제값 vs 예측값
# =========================================================
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.7)

# 실제값 = 예측값인 기준선
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())
plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
plt.xlabel("Actual") ; plt.ylabel("Predicted") ; plt.title("Actual vs Predicted")
plt.grid(linestyle="--", alpha=0.3)
plt.tight_layout() ; plt.show()
# 점이 대각선에 가까울수록 예측이 정확함
# 현재 모델은 전체적인 경향은 잡지만 오차도 상당히 존재

# =========================================================
# 7. 회귀계수 확인
# =========================================================
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_
})

coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

print("\n===== 회귀계수 =====")
print(coef_df[["Feature", "Coefficient"]])
# coef > 0 → Feature 증가 시 예측 Target 증가 방향
# coef < 0 → Feature 증가 시 예측 Target 감소 방향
# 다른 Feature가 동일하다는 모델 가정하에서 해석

# =========================================================
# 8. 회귀계수 시각화
# =========================================================
plot_coef = coef_df.sort_values("Coefficient")

plt.figure(figsize=(8, 5))
plt.barh(plot_coef["Feature"], plot_coef["Coefficient"])
plt.axvline(0, linestyle="--")  # 양수/음수 기준
plt.xlabel("Coefficient")
plt.title("Multiple Linear Regression Coefficients")
plt.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout(); plt.show()
# 계수 절댓값이 크다고 무조건 중요한 Feature라는 뜻은 아님
# 다중공선성이나 Feature 단위도 함께 고려해야 함

# =========================================================
# 9. Residual(잔차) 분석
# =========================================================
residual = y_test - y_pred  # 실제값 - 예측값
print("\n===== Residual =====")
print(f"잔차 평균: {residual.mean():.3f}")  # r: 0에 가까운 값이 이상적

plt.figure(figsize=(6, 5))
plt.scatter(
    y_pred,         # x축: 예측값
    residual,       # y축: 잔차
    alpha=0.7
)

plt.axhline(0, linestyle="--")  # 오차 0 기준선
plt.xlabel("Predicted") ; plt.ylabel("Residual"); plt.title("Residual Plot")
plt.grid(linestyle="--", alpha=0.3)
plt.tight_layout() ; plt.show()
# 좋은 선형모델 → 잔차가 0을 중심으로 특별한 패턴 없이 분포
# 곡선/깔때기 패턴 → 비선형성 또는 분산 문제를 의심

# =========================================================
# 10. Feature 상관관계
# =========================================================
corr = df.drop(columns="target").corr()

plt.figure(figsize=(8, 7))
plt.imshow(
    corr,             # 상관계수 행렬
    vmin=-1,          # 최소값
    vmax=1,           # 최대값
    cmap="coolwarm"   # 음수~양수 색상
)

plt.colorbar(label="Correlation")
plt.xticks(range(len(feature_names)), feature_names, rotation=45)
plt.yticks(range(len(feature_names)), feature_names)
plt.title("Feature Correlation")
plt.tight_layout(); plt.show()
# |상관계수|가 1에 가까울수록 두 Feature의 선형관계가 강함
# Feature끼리 지나치게 강한 관계 → 다중공선성 가능

# =========================================================
# 11. VIF로 다중공선성 확인
#from statsmodels.stats.outliers_influence import variance_inflation_factor
#statsmodels.stats.outliers_influence.variance_inflation_factor()
# =========================================================
def calculate_vif(X, feature_names):
    """각 Feature를 나머지 Feature로 예측해 VIF 계산"""
    vif_values = []
    for i in range(X.shape[1]):
        X_target = X[:, i]                    # 현재 검사할 Feature
        X_others = np.delete(X, i, axis=1)    # 나머지 Feature

        # 현재 Feature를 나머지 Feature로 회귀
        vif_model = LinearRegression()
        vif_model.fit(X_others, X_target)

        # 현재 Feature의 결정계수(R²)
        r2_i = vif_model.score(X_others, X_target)
        vif = 1 / (1 - r2_i)                  # VIF 공식
        # 계산된 VIF 저장
        vif_values.append(vif)

    # VIF가 큰 순서대로 정렬
    return pd.DataFrame({
        "Feature": feature_names,
        "VIF": vif_values
    }).sort_values("VIF", ascending=False)

# Feature별 VIF 계산
vif_df = calculate_vif(X_train, feature_names)

print("\n===== VIF =====")
print(vif_df)

# 일반적인 참고:
# VIF ≈ 1  → 다중공선성 거의 없음
# VIF > 5  → 주의해서 확인
# VIF > 10 → 높은 다중공선성을 의심
# 절대적인 제거 기준은 아니며 데이터와 목적에 따라 판단

# =========================================================
# 12. 5-Fold Cross Validation
#Train/Test를 한 번만 분리하면 데이터 구성에 따라 성능이 달라질 수 있으니
#데이터를 5개의 Fold로 나누고, 각 Fold를 한 번씩 Test 데이터로 사용, 총 5번 모델을 평가
# =========================================================
cv_scores = cross_val_score(
    LinearRegression(),  # 평가 모델
    X,                   # 전체 Feature
    y,                   # Target
    cv=5,                # 5-Fold
    scoring="r2"         # R² 기준
)

print("\n===== 5-Fold Cross Validation =====")
print("각 Fold R²:", np.round(cv_scores, 3))
# r: [0.430 0.523 0.483 0.426 0.550]

print(f"평균 R²: {cv_scores.mean():.3f}")   # r: 0.482 -모델의 평균 성능
print(f"표준편차: {cv_scores.std():.3f}")   # r: 0.049 - Fold마다 성능 변동 정도

# 한 번의 Train/Test 결과만 보지 않고 여러 분할에서 성능 확인
# 평균은 높고 표준편차는 작을수록 안정적인 모델


# =========================================================
# 13. Ridge Regression : Ridge는 계수가 너무 커지는 것을 억제
#기존 선형회귀는 LOSS=MSE
#Ridge는 Loss = MSE+계수패널티 를 최소화
## alpha: 클수록 계수를 더 강하게 축소(alpha=0.1 약한규제 / 10 강한규제 / 1000 계수를 거의 0으로 만듬)
# =========================================================
ridge = Ridge(alpha=0.1)

ridge.fit(X_train, y_train) #linear보다 계수를 조금 줄여 더 안정적인 모델
ridge_pred = ridge.predict(X_test)

ridge_r2 = r2_score(y_test, ridge_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

print("\n===== Ridge =====")
print(f"R²  : {ridge_r2:.3f}")
print(f"RMSE: {ridge_rmse:.3f}")
# Linear보다 Ridge값이 커지면 "규제를 넣어 새 데이터에도 더 잘 맞음(큰계수를 줄여 모델을 안정화)"
# Linear보다 Ridge값이 작아지면 "규제기 큰 도움이 되지 않는다"
# 다중공선성이 있을 때 유용

#Linear Regression → 예측만 잘하면 된다.
#Ridge Regression → 예측도 잘해야 하고,계수도 너무 커지면 안 된다

# =========================================================
# 14. Lasso Regression : 일부 계수를 정확히 0으로 만들 수 있다!
# =========================================================

# 기본선형회귀 : Loss = MSE
# Ridge(L2규제): Loss = MSE + α × (계수²의 합)
# Lasso(L1규제): Loss = MSE + α × (|계수|의 합) : Feature Selection 효과
lasso = Lasso(alpha=0.1, max_iter=10000)

lasso.fit(X_train, y_train)  #Lasso 모델을 학습
lasso_pred = lasso.predict(X_test)  # 학습된 모델로 Test 데이터를 예측

lasso_r2 = r2_score(y_test, lasso_pred) # 모델 설명력(R²) 계산
#모델의 설명력(R², 결정계수) 연산
##실제값과 예측값이 얼마나 잘 맞는지
##1에 가까울수록 좋음

lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred)) # 평균 예측 오차(RMSE) 계산
#평균 예측 오차(RMSE) 연산
##실제값과 예측값이 평균적으로 얼마나 차이 나는지 계산
##0에 가까울수록 좋음

print("\n===== Lasso =====")
print(f"R²  : {lasso_r2:.3f}")
print(f"RMSE: {lasso_rmse:.3f}")

# Lasso → L1 규제
# 필요성이 낮은 Feature의 계수를 0으로 만들 수 있어 Feature 선택 효과
# Feature 10개 - Lasso - 중요성이 낮다고 판단한 계수 = 0 - 실제로 사용하는 Feature 감소

# =========================================================
# 15. Linear / Ridge / Lasso 성능 비교
# =========================================================

# 세 모델의 R²와 RMSE를 하나의 표로 정리
comparison = pd.DataFrame({
    "Model": ["Linear", "Ridge", "Lasso"],
    "R²": [
        r2,          # Linear 설명력
        ridge_r2,    # Ridge 설명력
        lasso_r2     # Lasso 설명력
    ],
    "RMSE": [
        rmse,         # Linear 예측 오차
        ridge_rmse,   # Ridge 예측 오차
        lasso_rmse    # Lasso 예측 오차
    ]
})

print("\n===== 모델 비교 =====")
print(comparison.round(3))  # 소수점 3자리로 모델별 성능 출력

# R²는 높을수록 좋고 RMSE는 낮을수록 좋음
# 규제를 추가했다고 항상 성능이 좋아지는 것은 아님


# =========================================================
# 16. Lasso alpha(규제 강도) 비교
# =========================================================

print("\n===== Lasso alpha 비교 =====")

for alpha in [0.001, 0.01, 0.1, 1, 10]:

    # 서로 다른 규제 강도의 Lasso 모델 생성
    model = Lasso(
        alpha=alpha,      # 규제 강도
        max_iter=10000    # 최대 반복 횟수
    )

    model.fit(X_train, y_train)                # 현재 alpha로 Lasso 모델 학습
    y_pred_alpha = model.predict(X_test)       # 학습한 모델로 Test 데이터 예측

    r2_alpha = r2_score(y_test, y_pred_alpha)  # 현재 alpha 모델의 설명력(R²) 계산
    rmse_alpha = np.sqrt(
        mean_squared_error(y_test, y_pred_alpha)
    )                                         # 현재 alpha 모델의 예측 오차(RMSE) 계산

    zero_coef = np.sum(model.coef_ == 0)       # 계수가 0이 된 Feature 개수 계산

    print(
        f"alpha={alpha:<5} | "                 # 규제 강도
        f"R²={r2_alpha:.3f} | "                # 설명력
        f"RMSE={rmse_alpha:.3f} | "            # 예측 오차
        f"Zero Coef={zero_coef}"               # 제거된 Feature 수
    )

# alpha가 커질수록 더 많은 계수가 0이 될 수 있음
# 너무 강한 규제는 중요한 Feature까지 제거해 Underfitting 가능


# =========================================================
# 17. Ridge / Lasso Cross Validation 비교
# =========================================================

ridge_cv = cross_val_score(
    Ridge(alpha=0.1),   # 평가할 Ridge 모델
    X,                  # Feature
    y,                  # Target
    cv=5,               # 데이터를 5개 Fold로 나눠 반복 평가
    scoring="r2"        # R²를 평가 기준으로 사용
)                       # Ridge의 5-Fold R² 계산

lasso_cv = cross_val_score(
    Lasso(alpha=0.1),   # 평가할 Lasso 모델
    X,
    y,
    cv=5,
    scoring="r2"
)                       # Lasso의 5-Fold R² 계산

print("\n===== Cross Validation 모델 비교 =====")

print(f"Linear 평균 R²: {cv_scores.mean():.3f}")  # Linear의 평균 교차검증 성능
print(f"Ridge 평균 R² : {ridge_cv.mean():.3f}")   # Ridge의 평균 교차검증 성능
print(f"Lasso 평균 R² : {lasso_cv.mean():.3f}")   # Lasso의 평균 교차검증 성능

# 여러 데이터 분할에서도 비슷한 성능인지 확인
# 평균 R²가 높고 Fold별 편차가 작을수록 안정적
