# =========================================================
# Decision Tree - Basic
# Playing Golf + Iris Dataset
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
# 한글 폰트 설정
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

# 1. Playing Golf 데이터 불러오기
golf = pd.read_csv("../datasets/decision/playing_golf.csv")
print("===== Playing Golf Dataset =====")
print(golf.head())

# 2. One-Hot Encoding
# Decision Tree가 처리할 수 있도록 범주형 문자 데이터를 숫자로 변환
encoded = pd.get_dummies(golf,dtype=int)
print("\n===== One-Hot Encoding =====")
print(encoded.head())

# 3. Feature / Target 분리
X = encoded.drop(columns=["play_no", "play_yes"]) # 1 : 골프 침, 0 : 골프 안 침
y = encoded["play_yes"]
print("\n입력 Feature:")
print(list(X.columns))

# 4. Decision Tree 모델 학습
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X,y)
# Decision Tree는 Feature를 기준으로 데이터를 반복적으로 분할하여 규칙을 생성

# 5. Playing Golf Decision Tree 시각화
plt.figure(figsize=(14, 8))
plot_tree(
    tree,
    feature_names=list(X.columns),  # Feature 이름
    class_names=["안침", "침"],      # Class 이름
    filled=True,                    # Class에 따라 노드 색상 표시
    rounded=True,                   # 노드 모서리를 둥글게 표시
    fontsize=9                      # 글자 크기
)
plt.title("Playing Golf Decision Tree"); plt.tight_layout(); plt.show()

# 6. Iris Dataset
iris = load_iris()
X = iris.data; y = iris.target
print("\n===== Iris Dataset =====")
print("데이터 크기:", X.shape)
print("Feature:", iris.feature_names)
print("Class:", iris.target_names)

# 7. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,       # 전체 데이터의 30%를 Test로 사용
    random_state=2       # 분할 결과 고정
)
print("\nTrain:", X_train.shape)
print("Test :", X_test.shape)

# 8. 깊이 제한 없는 Decision Tree
deep_tree = DecisionTreeClassifier(random_state=3)
deep_tree.fit(X_train,y_train)
train_accuracy = deep_tree.score(X_train,y_train)
test_accuracy = deep_tree.score(X_test,y_test)
print("\n===== 깊이 제한 없음 =====")
print(f"훈련 정확도 : {train_accuracy:.3f}")
print(f"테스트 정확도: {test_accuracy:.3f}")

# 9. 깊이 제한 없는 Tree 시각화
plt.figure(figsize=(14, 8))
plot_tree(
    deep_tree,
    feature_names=iris.feature_names,
    class_names=list(iris.target_names),
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree - No Depth Limit")
plt.tight_layout(); plt.show()
# Tree가 지나치게 깊어지면
# Train 데이터를 세밀하게 학습하여 과적합 가능성이 증가

# 10. max_depth에 따른 정확도 비교
depths = range(1, 8)
train_scores = []
test_scores = []
for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,      # Tree의 최대 깊이 제한
        random_state=3
    )
    model.fit(
        X_train,
        y_train
    )
    train_scores.append(
        model.score(X_train, y_train)
    )
    test_scores.append(
        model.score(X_test, y_test)
    )

# 11. Tree Depth와 Accuracy 시각화

plt.figure(figsize=(9, 5))
plt.plot(list(depths),train_scores, marker="o", label="훈련 정확도")
plt.plot(list(depths),test_scores, marker="^", label="테스트 정확도")
plt.xlabel("트리 최대 깊이"); plt.ylabel("정확도"); plt.title("Accuracy according to Tree Depth")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
# Depth가 증가하면 Train Accuracy는 높아지는 경향
# 하지만 Test Accuracy는 어느 시점부터 낮아질 수 있음
# → 과적합 가능성

# 12. 가지치기된 Decision Tree
pruned_tree = DecisionTreeClassifier(
    max_depth=3,         # 최대 깊이를 3으로 제한
    random_state=2
)
pruned_tree.fit( X_train, y_train)
train_accuracy_pruned = pruned_tree.score(X_train,y_train)
test_accuracy_pruned = pruned_tree.score(X_test,y_test)
print("\n===== max_depth=3 =====")
print(f"훈련 정확도 : {train_accuracy_pruned:.3f}")
print(f"테스트 정확도: {test_accuracy_pruned:.3f}")

#13. 가지치기된 Tree 시각화
plt.figure(figsize=(13, 7))
plot_tree(
    pruned_tree,
    feature_names=iris.feature_names,       # Feature 이름
    class_names=list(iris.target_names),    # Iris 품종
    filled=True,                            # 노드 색상
    rounded=True,                           # 둥근 노드
    fontsize=9
)
plt.title("Decision Tree - max_depth=3") ; plt.tight_layout(); plt.show()

# 14. Feature Importance
importance = pd.Series(
    pruned_tree.feature_importances_,
    index=iris.feature_names
).sort_values()

print("\n===== Feature Importance =====")
print(importance)

#15. Feature Importance 시각화
plt.figure(figsize=(8, 4.5))
importance.plot(kind="barh")
plt.xlabel("Feature Importance"); plt.title("Decision Tree Feature Importance")
plt.tight_layout(); plt.show()
# 값이 클수록 Tree의 데이터 분할에
# 상대적으로 많이 사용된 Feature

# 16.Summary
print("\n===== Summary =====")
print(f"깊이 제한 없음 Train : {train_accuracy:.3f}")
print(f"깊이 제한 없음 Test  : {test_accuracy:.3f}")
print(f"max_depth=3 Train    : {train_accuracy_pruned:.3f}")
print(f"max_depth=3 Test     : {test_accuracy_pruned:.3f}")

# Decision Tree → Feature를 기준으로 데이터를 반복적으로 분할
# max_depth → Tree의 최대 깊이를 제한
# Tree가 너무 깊음 → Train 데이터에 과적합될 가능성 증가
# Feature Importance → Tree의 분할 과정에서 각 Feature가 기여한 상대적 중요도
