# =========================================================
# K-Nearest Neighbors (KNN) - Basic
# Iris Dataset
# =========================================================

import matplotlib.pyplot as plt  # 그래프 시각화
import numpy as np               # 배열 및 거리 계산
import pandas as pd              # 데이터 확인

from sklearn.datasets import load_iris              # Iris 내장 데이터
from sklearn.metrics import accuracy_score, confusion_matrix  # 기본 평가
from sklearn.model_selection import train_test_split  # Train/Test 분리
from sklearn.neighbors import KNeighborsClassifier   # KNN 분류 모델

# =========================================================
# 1. Iris 데이터 불러오기
# =========================================================

iris = load_iris()

X = iris.data                  # 꽃의 Feature
y = iris.target                # 꽃 종류(0, 1, 2)
target_names = iris.target_names
feature_names = iris.feature_names

df = pd.DataFrame(
    X,
    columns=feature_names
)

df["target"] = y

print("데이터 크기:", X.shape)              # r: (150, 4)
print("Feature 수:", len(feature_names))    # r: 4
print("클래스:", target_names)              # r: ['setosa' 'versicolor' 'virginica']
# class 0 = setosa / class 1 = versicolor / class 2 = virginica

# =========================================================
# 2. Iris 데이터 시각화
# =========================================================

plt.figure(figsize=(8, 6))

markers = ["o", "^", "s"]

for i, marker in enumerate(markers):
    plt.scatter(
        X[y == i, 0],             # sepal length
        X[y == i, 1],             # sepal width
        marker=marker,            # 클래스별 점 모양
        label=target_names[i]     # 꽃 품종
    )
plt.xlabel("Sepal Length"); plt.ylabel("Sepal Width"); plt.title("Iris Dataset")
plt.grid(linestyle="--", alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
# 가까운 위치에 있는 데이터끼리 같은 품종일 가능성이 높음을 확인

# =========================================================
# 3. L1 / L2 Distance
# =========================================================

# 두 점
a = np.array([1, 1])
b = np.array([5, 4])

# Manhattan Distance(L1): 각 좌표 차이의 절댓값 합
manhattan = np.linalg.norm(a - b, ord=1)

# Euclidean Distance(L2): 두 점 사이의 직선 거리
euclidean = np.linalg.norm(a - b, ord=2)
print("\n===== Distance =====")
print(f"Manhattan Distance: {manhattan:.2f}")  # r: 7.00
print(f"Euclidean Distance: {euclidean:.2f}")  # r: 5.00

# =========================================================
# 4. L1 / L2 Distance 시각화
# =========================================================

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(a[0], a[1], s=70, label="Point A")
ax.scatter(b[0], b[1], s=70, label="Point B")

# Euclidean: 두 점을 직선으로 연결
ax.plot(
    [a[0], b[0]],
    [a[1], b[1]],
    linewidth=2,
    label=f"Euclidean = {euclidean:.2f}"
)

# Manhattan: 가로 + 세로 이동
ax.plot([a[0], b[0]], [a[1], a[1]], linestyle="--")
ax.plot(
    [b[0], b[0]],
    [a[1], b[1]],
    linestyle="--",
    label=f"Manhattan = {manhattan:.2f}"
)

ax.set_xlim(0, 6); ax.set_ylim(0, 5)
ax.set_xticks(range(0, 7)); ax.set_yticks(range(0, 6))
ax.set_title("L1 vs L2 Distance"); ax.set_aspect("equal")
ax.grid(True); ax.legend(); plt.tight_layout(); plt.show()
# KNN은 데이터 사이의 거리를 이용해 가까운 이웃을 찾는 알고리즘

# =========================================================
# 5. Train / Test 데이터 분리
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,          # 전체의 20%를 Test 데이터로 사용
    random_state=42,        # 분할 결과 고정
    stratify=y              # 각 클래스 비율 유지
)
print("\nTrain:", X_train.shape)  # r: (120, 4)
print("Test :", X_test.shape)     # r: (30, 4)

# =========================================================
# 6. 기본 KNN 모델
# =========================================================

knn = KNeighborsClassifier(
    n_neighbors=5           # 가장 가까운 이웃 5개를 이용
)

knn.fit(X_train, y_train)         # Train 데이터로 KNN 모델 학습
y_pred = knn.predict(X_test)      # Test 데이터의 꽃 품종 예측

accuracy = accuracy_score(
    y_test,
    y_pred
)                                # 전체 분류 정확도 계산

print("\n===== KNN k=5 =====")
print(f"Accuracy: {accuracy:.3f}")
# n_neighbors = k
# 예측할 데이터 주변의 k개 이웃을 확인하여 다수결로 클래스 결정

# =========================================================
# 7. Confusion Matrix
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)                                # 실제값과 예측값 비교

print("\n===== Confusion Matrix =====")
print(cm)
# 3개의 Iris 클래스이므로 3×3 혼동행렬 생성

# =========================================================
# 8. k 값에 따른 Accuracy 비교
# =========================================================
k_values = range(1, 16)
scores = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k            # 현재 k값으로 KNN 생성
    )

    model.fit(X_train, y_train)   # 현재 k값으로 모델 학습
    score = model.score(
        X_test,
        y_test
    )                            # Test Accuracy 계산

    scores.append(score)

print("\n===== k별 Accuracy =====")

for k, score in zip(k_values, scores):
    print(
        f"k={k:>2} | "
        f"Accuracy={score:.3f}"
    )

# =========================================================
# 9. k별 Accuracy 시각화
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(
    k_values,
    scores,
    marker="o"
)

plt.xlabel("k"); plt.ylabel("Accuracy")
plt.title("Accuracy according to k"); plt.xticks(k_values)
plt.grid(linestyle="--", alpha=0.4); plt.tight_layout() ;plt.show()
# k가 너무 작으면 주변 데이터에 민감해질 수 있음
# k가 너무 크면 서로 다른 클래스까지 많이 포함해 모델이 단순해질 수 있음

# =========================================================
# 10. 가장 높은 Accuracy의 k 확인
# =========================================================

best_score = max(scores)                  # 가장 높은 Accuracy
best_k = scores.index(best_score) + 1     # 해당 Accuracy의 k값

print("\n===== Best k =====")
print("최적 k:", best_k)                  # r: 3
print(f"최고 Accuracy: {best_score:.3f}") # r: 1.000
# 실무에서는 Cross Validation으로 k를 선택

# =========================================================
# 11. 최종 KNN 모델
# =========================================================
best_model = KNeighborsClassifier(
    n_neighbors=best_k
)

best_model.fit(   # 선택한 k로 최종 모델 학습
    X_train,
    y_train
)

# =========================================================
# 12. 새로운 꽃 예측
# =========================================================

new_flower = [[
    5.0,    # sepal length
    3.5,    # sepal width
    1.5,    # petal length
    0.2     # petal width
]]

prediction = best_model.predict(  # 새로운 꽃 품종 예측
    new_flower
)

pred_class = prediction[0]

print("\n===== 새로운 꽃 예측 =====")
print("예측 Class:", pred_class)
print("예측 품종:", target_names[pred_class])
# KNN은 새로운 데이터와 기존 학습 데이터 사이의 거리를 계산한 뒤
# 가장 가까운 k개의 이웃을 이용하여 클래스를 결정

# =========================================================
# 13. Summary
# =========================================================

print("\n===== Summary =====")
print(f"Best k   : {best_k}")
print(f"Accuracy : {best_score:.3f}")

# KNN            : 가까운 이웃을 이용한 분류 알고리즘
# n_neighbors    : 사용할 이웃의 수(k)
# fit()          : 학습 데이터 저장
# predict()      : 가까운 이웃을 이용해 클래스 예측
# score()        : 모델 Accuracy 계산
# L1 Distance    : Manhattan Distance
# L2 Distance    : Euclidean Distance