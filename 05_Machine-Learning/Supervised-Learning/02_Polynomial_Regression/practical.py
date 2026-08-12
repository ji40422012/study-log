import matplotlib.pyplot as plt  # 그래프 시각화
import numpy as np               # 수치 계산
import pandas as pd              # 결과 표 정리

from sklearn.datasets import load_diabetes  # 회귀 내장 데이터
from sklearn.linear_model import LinearRegression, Ridge  # 선형회귀 / Ridge
from sklearn.metrics import mean_squared_error, r2_score  # 회귀 평가
from sklearn.model_selection import train_test_split, cross_val_score  # 데이터 분리 / 교차검증
from sklearn.pipeline import make_pipeline  # 전처리 + 모델 연결
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # 다항 Feature 생성 / 표준화

# =========================================================
# 1. 데이터 불러오기
# =========================================================

data = load_diabetes()
X = data.data                  # 전체 Feature
y = data.target                # 질병 진행 정도
feature_names = data.feature_names

print("X shape:", X.shape)    # r: (442, 10)
print("y shape:", y.shape)    # r: (442,)
print("Features:", feature_names)
# load_diabetes는 연속형 Target을 예측하는 Regression 데이터

# =========================================================
# 2. BMI Feature: Polynomial Regression 이해
# =========================================================
bmi_index = feature_names.index("bmi")
X_bmi = X[:, bmi_index].reshape(-1, 1)  # bmi만 추출하여 2차원 형태로 변환
print("\nBMI shape:", X_bmi.shape)       # r: (442, 1)

# =========================================================
# 3. BMI Train / Test 분리
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_bmi,              # BMI Feature
    y,                  # Target
    test_size=0.2,      # Test 20%
    random_state=42     # 분할 결과 고정
)

# =========================================================
# 4. degree별 Polynomial Regression 비교
# =========================================================
degrees = [1, 2, 3, 5, 10]
results = []
print("\n===== BMI Polynomial Regression =====")

for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(
            degree=degree,          # 다항식 차수
            include_bias=False      # 상수항 중복 생성 방지
        ),
        LinearRegression()          # 생성된 다항 Feature로 회귀
    )

    model.fit(X_train, y_train)              # 현재 degree 모델 학습
    y_pred = model.predict(X_test)           # Test 데이터 예측

    r2 = r2_score(y_test, y_pred)            # 모델 설명력(R²) 계산
    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )                                       # 예측 오차(RMSE) 계산

    results.append({
        "Degree": degree,
        "R2": r2,
        "RMSE": rmse
    })

    print(
        f"degree={degree:<2} | "
        f"R²={r2:.3f} | "
        f"RMSE={rmse:.3f}"
    )
# degree가 증가한다고 항상 Test 성능이 좋아지는 것은 아님
# 너무 높은 degree는 학습 데이터에 과하게 맞아 Overfitting 가능

# =========================================================
# 5. degree별 성능 표
# =========================================================
result_df = pd.DataFrame(results)
print("\n===== Degree 성능 비교 =====")
print(result_df.round(3))
# R²는 높을수록 좋음
# RMSE는 낮을수록 좋음


# =========================================================
# 6. degree별 곡선 시각화
# =========================================================
x_plot = np.linspace(
    X_bmi.min(),
    X_bmi.max(),
    300
).reshape(-1, 1)
plt.figure(figsize=(12, 8))

for i, degree in enumerate([1, 2, 3, 5], start=1):
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    model.fit(X_train, y_train)          # degree별 모델 학습
    y_plot = model.predict(x_plot)       # 부드러운 곡선 예측

    plt.subplot(2, 2, i)

    plt.scatter(
        X_train,
        y_train,
        alpha=0.5,
        label="Train"
    )

    plt.plot(
        x_plot,
        y_plot,
        linewidth=2,
        label=f"Degree {degree}"
    )

    plt.xlabel("BMI")
    plt.ylabel("Disease Progression")
    plt.title(f"Polynomial Degree {degree}")
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()
# degree가 높아질수록 곡선이 복잡
## 복잡한 곡선이 반드시 좋은 예측 모델을 의미하지는 않음

# =========================================================
# 7. 5-Fold Cross Validation으로 degree 비교
# =========================================================
cv_results = []

print("\n===== Degree Cross Validation =====")

for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        LinearRegression()
    )

    cv_scores = cross_val_score(
        model,           # 평가할 Polynomial 모델
        X_bmi,           # BMI Feature
        y,               # Target
        cv=5,            # 5-Fold
        scoring="r2"     # R² 기준 평가
    )                    # 여러 Fold에서 모델 성능 반복 평가

    cv_results.append({
        "Degree": degree,
        "CV_R2_Mean": cv_scores.mean(),
        "CV_R2_STD": cv_scores.std()
    })

    print(
        f"degree={degree:<2} | "
        f"평균 R²={cv_scores.mean():.3f} | "
        f"표준편차={cv_scores.std():.3f}"
    )
# 평균 R²가 높고 표준편차가 작을수록 안정적인 degree

# =========================================================
# 8. 전체 10개 Feature 사용
# =========================================================
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X,                  # 전체 10개 Feature
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# 9. PolynomialFeatures로 Feature 증가 확인
# =========================================================
poly = PolynomialFeatures(
    degree=2,           # 2차항 + 상호작용항 생성
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train_all)
X_test_poly = poly.transform(X_test_all)

print("\n===== Polynomial Feature 증가 =====")
print("원본 Feature 수:", X_train_all.shape[1])   # r: 10
print("변환 후 Feature 수:", X_train_poly.shape[1])  # r: 65
# x1, x2뿐 아니라 x1², x1*x2, x2² 등의 새로운 Feature 생성

# =========================================================
# 10. 전체 Feature - Linear Regression
# =========================================================
linear_model = LinearRegression()

linear_model.fit(X_train_all, y_train_all)       # 기본 Linear 모델 학습
linear_pred = linear_model.predict(X_test_all)   # Test 데이터 예측

linear_r2 = r2_score(y_test_all, linear_pred)    # Linear 설명력 계산
linear_rmse = np.sqrt(
    mean_squared_error(y_test_all, linear_pred)
)                                               # Linear 예측 오차 계산
print("\n===== Linear Regression =====")
print(f"R²  : {linear_r2:.3f}")       # r: 약 0.45
print(f"RMSE: {linear_rmse:.3f}")     # r: 약 53~54

# =========================================================
# 11. 전체 Feature - Polynomial Regression
# =========================================================
poly_model = make_pipeline(
    PolynomialFeatures(
        degree=2,               # 2차 Polynomial Feature 생성
        include_bias=False
    ),
    LinearRegression()
)

poly_model.fit(X_train_all, y_train_all)         # 2차 Polynomial 모델 학습
poly_pred = poly_model.predict(X_test_all)       # Test 데이터 예측

poly_r2 = r2_score(y_test_all, poly_pred)        # Polynomial 설명력 계산
poly_rmse = np.sqrt(
    mean_squared_error(y_test_all, poly_pred)
)                                               # Polynomial 예측 오차 계산

print("\n===== Polynomial Regression degree=2 =====")
print(f"R²  : {poly_r2:.3f}")
print(f"RMSE: {poly_rmse:.3f}")
# Feature가 많아졌지만 Test 성능이 반드시 좋아지는 것은 아님

# =========================================================
# 12. Linear vs Polynomial 비교
# =========================================================
comparison = pd.DataFrame({
    "Model": [
        "Linear",
        "Polynomial degree=2"
    ],
    "R2": [
        linear_r2,
        poly_r2
    ],
    "RMSE": [
        linear_rmse,
        poly_rmse
    ]
})

print("\n===== Linear vs Polynomial =====")
print(comparison.round(3))
# Polynomial이 더 복잡하지만 성능 향상이 없으면 복잡도 증가의 이점이 적음

# =========================================================
# 13. Polynomial + Ridge Regression
# =========================================================
ridge_poly = make_pipeline(
    PolynomialFeatures(
        degree=2,
        include_bias=False
    ),
    StandardScaler(),           # Polynomial Feature들의 Scale 조정
    Ridge(
        alpha=1.0               # L2 규제로 큰 계수 억제
    )
)

ridge_poly.fit(X_train_all, y_train_all)         # Polynomial + Ridge 모델 학습
ridge_pred = ridge_poly.predict(X_test_all)      # Test 데이터 예측

ridge_r2 = r2_score(y_test_all, ridge_pred)      # Ridge 설명력 계산
ridge_rmse = np.sqrt(
    mean_squared_error(y_test_all, ridge_pred)
)                                               # Ridge 예측 오차 계산

print("\n===== Polynomial + Ridge =====")
print(f"R²  : {ridge_r2:.3f}")
print(f"RMSE: {ridge_rmse:.3f}")
# 고차 Polynomial Feature로 계수가 커지는 문제를 Ridge 규제로 완화

# =========================================================
# 14. Linear / Polynomial / Ridge 비교
# =========================================================
final_comparison = pd.DataFrame({
    "Model": [
        "Linear",
        "Polynomial degree=2",
        "Polynomial + Ridge"
    ],
    "R2": [
        linear_r2,
        poly_r2,
        ridge_r2
    ],
    "RMSE": [
        linear_rmse,
        poly_rmse,
        ridge_rmse
    ]
})

print("\n===== 최종 모델 비교 =====")
print(final_comparison.round(3))

# R² ↑ → 설명력 증가
# RMSE ↓ → 예측 오차 감소


# =========================================================
# 15. 전체 Feature Cross Validation 비교
# =========================================================
linear_cv = cross_val_score(
    LinearRegression(),
    X,
    y,
    cv=5,
    scoring="r2"
)                                           # Linear의 5-Fold R² 계산

poly_cv = cross_val_score(
    make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression()
    ),
    X,
    y,
    cv=5,
    scoring="r2"
)                                           # Polynomial의 5-Fold R² 계산

ridge_cv = cross_val_score(
    make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1.0)
    ),
    X,
    y,
    cv=5,
    scoring="r2"
)                                           # Polynomial + Ridge의 5-Fold R² 계산


print("\n===== Cross Validation 모델 비교 =====")
print(f"Linear 평균 R²            : {linear_cv.mean():.3f}")
print(f"Polynomial 평균 R²        : {poly_cv.mean():.3f}")
print(f"Polynomial + Ridge 평균 R²: {ridge_cv.mean():.3f}")
# Test 한 번의 결과가 아니라 여러 Fold에서 일반화 성능 비교

# =========================================================
# 16. Residual 비교
# =========================================================
linear_residual = y_test_all - linear_pred   # Linear 잔차
poly_residual = y_test_all - poly_pred       # Polynomial 잔차
ridge_residual = y_test_all - ridge_pred     # Polynomial + Ridge 잔차

plt.figure(figsize=(7, 5))

plt.scatter(
    linear_pred,
    linear_residual,
    alpha=0.6,
    label="Linear"
)

plt.scatter(
    ridge_pred,
    ridge_residual,
    alpha=0.6,
    label="Polynomial + Ridge"
)

plt.axhline(0, linestyle="--")  # 오차 0 기준선
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Comparison")
plt.grid(linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 잔차가 0 주변에 특별한 패턴 없이 분포할수록 이상적


# =========================================================
# 17. 결론
# =========================================================
print("\n===== Conclusion =====")
best_model = final_comparison.loc[
    final_comparison["RMSE"].idxmin()
]

print("가장 낮은 RMSE 모델:", best_model["Model"])
print(f"R²  : {best_model['R2']:.3f}")
print(f"RMSE: {best_model['RMSE']:.3f}")

# Polynomial Regression은 degree를 높인다고 항상 성능이 좋아지지 않음
# Feature 수 증가 → 모델 복잡도 증가 → Overfitting 가능
# Cross Validation으로 degree와 모델 구조를 검증하는 것이 중요
# Ridge와 같은 규제를 통해 복잡도를 제어할 수 있음
