# =========================================================
# K-Nearest Neighbors (KNN) - Practical
# Wine Dataset
# =========================================================

import joblib                    # 학습 모델 저장 / 불러오기
import matplotlib.pyplot as plt  # 그래프 시각화
import numpy as np               # 수치 및 배열 연산
import pandas as pd              # 데이터 확인

from sklearn.datasets import load_wine  # Wine 내장 데이터
from sklearn.metrics import (
    accuracy_score,          # 정확도
    classification_report,   # Precision / Recall / F1
    confusion_matrix,        # 혼동행렬 계산
    ConfusionMatrixDisplay   # 혼동행렬 시각화
)
from sklearn.model_selection import (
    cross_val_score,         # Cross Validation
    StratifiedKFold,         # 클래스 비율을 유지하는 Fold
    train_test_split         # Train / Test 분리
)
from sklearn.neighbors import KNeighborsClassifier  # KNN 분류 모델
from sklearn.pipeline import make_pipeline           # 전처리 + 모델 연결
from sklearn.preprocessing import StandardScaler     # Feature 표준화


# =========================================================
# 1. Wine 데이터 불러오기
# =========================================================

wine = load_wine()

X = wine.data                    # 입력 Feature
y = wine.target                  # Wine 종류(0, 1, 2)
feature_names = wine.feature_names
target_names = wine.target_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("데이터 크기:", X.shape)           # r: (178, 13)
print("Feature 수:", len(feature_names)) # r: 13
print("클래스:", target_names)           # r: ['class_0' 'class_1' 'class_2']
print("클래스별 개수:", np.bincount(y))  # r: [59 71 48]


# =========================================================
# 2. Feature 기본 통계 확인
# =========================================================

print("\n===== Feature 통계 =====")
print(df[feature_names].describe().round(2).T[["mean", "std", "min", "max"]])

# Feature별 값의 범위가 크게 다름
# KNN은 거리 기반 모델이므로 Scale 차이의 영향을 크게 받음


# =========================================================
# 3. Feature Scale 시각화
# =========================================================

plt.figure(figsize=(11, 5))
df[feature_names].boxplot(rot=90)  # Feature별 값 범위 비교

plt.title("Wine Feature Scale")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig("feature_scale.png", dpi=150, bbox_inches="tight")  # MD용 결과 이미지 저장
plt.show()


# =========================================================
# 4. Train / Test 데이터 분리
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,       # 전체의 20%를 최종 Test로 사용
    random_state=42,     # 분할 결과 고정
    stratify=y           # 클래스 비율 유지
)

print("\nTrain:", X_train.shape)  # r: (142, 13)
print("Test :", X_test.shape)     # r: (36, 13)


# =========================================================
# 5. Scaling 전 KNN
# =========================================================

knn_unscaled = KNeighborsClassifier(n_neighbors=5)  # k=5 KNN
knn_unscaled.fit(X_train, y_train)                   # 원본 데이터로 학습
unscaled_pred = knn_unscaled.predict(X_test)         # Test 예측
unscaled_accuracy = accuracy_score(y_test, unscaled_pred)  # Accuracy 계산

print("\n===== Scaling 전 =====")
print(f"Accuracy: {unscaled_accuracy:.3f}")


# =========================================================
# 6. StandardScaler 적용
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # Train 기준 계산 + Scaling
X_test_scaled = scaler.transform(X_test)        # 같은 기준으로 Test Scaling

# fit_transform() : 기준 계산 + 변환
# transform()     : 계산된 기준으로 변환
# Test에는 fit_transform()을 사용하지 않음


# =========================================================
# 7. Scaling 후 KNN
# =========================================================

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)            # Scaling된 Train으로 학습
scaled_pred = knn_scaled.predict(X_test_scaled)    # Scaling된 Test 예측
scaled_accuracy = accuracy_score(y_test, scaled_pred)  # Accuracy 계산

print("\n===== Scaling 후 =====")
print(f"Accuracy: {scaled_accuracy:.3f}")


# =========================================================
# 8. Scaling 전 / 후 비교
# =========================================================

comparison = pd.DataFrame({
    "Model": ["Without Scaling", "With Scaling"],
    "Accuracy": [unscaled_accuracy, scaled_accuracy]
})

print("\n===== Scaling 성능 비교 =====")
print(comparison.round(3))

# KNN은 거리 계산을 사용하므로 Feature Scaling이 중요


# =========================================================
# 9. Pipeline
# =========================================================

model = make_pipeline(
    StandardScaler(),                    # Feature 표준화
    KNeighborsClassifier(n_neighbors=5)  # KNN 분류
)

model.fit(X_train, y_train)                        # Scaling + KNN 학습
pipeline_accuracy = model.score(X_test, y_test)   # Test Accuracy

print("\n===== Pipeline =====")
print(f"Accuracy: {pipeline_accuracy:.3f}")

# Pipeline으로 전처리와 모델을 하나의 과정으로 관리


# =========================================================
# 10. 5-Fold Cross Validation으로 k 비교
# =========================================================

k_values = range(1, 31)
cv_means = []
cv_stds = []

cv = StratifiedKFold(
    n_splits=5,          # 5-Fold
    shuffle=True,        # 데이터 순서 섞기
    random_state=42      # Fold 구성 고정
)

print("\n===== 5-Fold Cross Validation =====")

for k in k_values:
    model = make_pipeline(
        StandardScaler(),                   # Fold 내부에서 Scaling
        KNeighborsClassifier(n_neighbors=k) # 현재 k 적용
    )

    cv_scores = cross_val_score(
        model,
        X_train,              # Test를 제외한 Train에서만 k 선택
        y_train,
        cv=cv,                # Stratified 5-Fold
        scoring="accuracy"    # Accuracy 기준
    )

    cv_means.append(cv_scores.mean())  # 평균 성능
    cv_stds.append(cv_scores.std())    # Fold별 성능 변동

    print(
        f"k={k:>2} | "
        f"CV Accuracy={cv_scores.mean():.3f} | "
        f"STD={cv_scores.std():.3f}"
    )


# =========================================================
# 11. 최적 k 선택
# =========================================================

best_index = np.argmax(cv_means)      # 평균 CV Accuracy가 가장 높은 위치
best_k = list(k_values)[best_index]   # 해당 위치의 k
best_score = cv_means[best_index]     # 최고 평균 CV Accuracy

print("\n===== Best k =====")
print(f"Best k            : {best_k}")
print(f"Mean CV Accuracy  : {best_score:.3f}")

# Test 데이터를 사용하지 않고 Train 내부 CV 결과로 k 선택


# =========================================================
# 12. k별 Cross Validation 성능 시각화
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    k_values,          # x축: k
    cv_means,          # y축: 평균 CV Accuracy
    marker="o"
)

plt.axvline(
    best_k,                         # 최적 k 위치
    linestyle="--",
    label=f"Best k = {best_k}"
)

plt.xlabel("k")
plt.ylabel("Mean CV Accuracy")
plt.title("Cross Validation Accuracy according to k")
plt.xticks(range(1, 31, 2))
plt.grid(linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("knn_cv_accuracy.png", dpi=150, bbox_inches="tight")  # MD용 이미지
plt.show()

# k가 너무 작거나 너무 크면 일반화 성능이 떨어질 수 있음


# =========================================================
# 13. 최종 KNN 모델
# =========================================================

final_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=best_k)
)

final_model.fit(X_train, y_train)       # 최적 k로 최종 학습
y_pred = final_model.predict(X_test)    # 남겨둔 Test 데이터 예측
final_accuracy = accuracy_score(y_test, y_pred)  # 최종 Accuracy

print("\n===== Final Test =====")
print(f"Accuracy: {final_accuracy:.3f}")


# =========================================================
# 14. Classification Report
# =========================================================

print("\n===== Classification Report =====")
print(
    classification_report(
        y_test,                    # 실제값
        y_pred,                    # 예측값
        target_names=target_names, # 클래스 이름
        digits=3                   # 소수점 자리
    )
)

# Precision : 해당 클래스로 예측한 것 중 실제 정답 비율
# Recall    : 실제 해당 클래스 중 모델이 찾아낸 비율
# F1 Score  : Precision과 Recall의 균형


# =========================================================
# 15. Confusion Matrix
# =========================================================

cm = confusion_matrix(y_test, y_pred)  # 실제값과 예측값 비교

print("\n===== Confusion Matrix =====")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names
).plot(
    cmap="Blues",       # 색상
    values_format="d"   # 정수 표시
)

plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.savefig("knn_confusion_matrix.png", dpi=150, bbox_inches="tight")  # MD용 이미지
plt.show()


# =========================================================
# 16. 최종 모델 저장
# =========================================================

joblib.dump(final_model, "wine_knn.pkl")  # Pipeline 전체 저장
print("\n모델 저장 완료: wine_knn.pkl")

# StandardScaler와 KNN이 Pipeline에 함께 저장됨


# =========================================================
# 17. 저장 모델 불러오기
# =========================================================

loaded_model = joblib.load("wine_knn.pkl")          # 저장 모델 불러오기
loaded_accuracy = loaded_model.score(X_test, y_test)  # 정상 동작 확인

print(f"불러온 모델 Accuracy: {loaded_accuracy:.3f}")


# =========================================================
# 18. Conclusion
# =========================================================

print("\n===== Conclusion =====")

print(f"Scaling 전 Accuracy : {unscaled_accuracy:.3f}")
print(f"Scaling 후 Accuracy : {scaled_accuracy:.3f}")
print(f"Best k              : {best_k}")
print(f"CV Accuracy         : {best_score:.3f}")
print(f"Final Test Accuracy : {final_accuracy:.3f}")

# KNN은 거리 기반 모델이므로 Feature Scaling이 중요
# Cross Validation으로 k를 선택하고 Test는 마지막 평가에 사용
# Pipeline으로 전처리와 모델을 함께 관리
