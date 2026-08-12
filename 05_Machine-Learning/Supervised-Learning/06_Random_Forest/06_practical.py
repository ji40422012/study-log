# =========================================================
# Random Forest - Practical
# Pima Indians Diabetes Dataset
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.tree import DecisionTreeClassifier


# 1. 데이터 불러오기
cols = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv("../datasets/pima-indians-diabetes.csv", names=cols)
print("===== Pima Indians Diabetes Dataset =====")
print(df.head()); print("\n데이터 크기:", df.shape)

# 2. Feature / Target 분리
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
print("\nTarget 분포:")
print(y.value_counts())
# Outcome: 0 → 정상,  1 → 당뇨

# 3. Target Class 분포 시각화
target_counts = y.value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(["Normal", "Diabetes"], target_counts.values)
plt.title("Target Distribution"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig("rf_target_distribution.png", dpi=150, bbox_inches="tight"); plt.show()
# Outcome=0과 Outcome=1의 개수 차이를 확인
# Class Imbalance가 있는지 확인

# 4. Feature 분포 확인
df.drop(columns=["Outcome"]).hist(figsize=(12, 10),bins=20,grid=True)
plt.suptitle("Histogram of Features", y=1.02, fontsize=14)
plt.tight_layout(); plt.savefig("rf_feature_histogram.png", dpi=150, bbox_inches="tight")
plt.show()

# 5. 주요 Feature와 Outcome 관계 확인
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df.boxplot(column="Glucose", by="Outcome")
plt.title("Glucose by Outcome")
plt.suptitle("")
plt.subplot(1, 2, 2)
df.boxplot(column="BMI", by="Outcome")
plt.title("BMI by Outcome")
plt.suptitle("")
plt.tight_layout()
plt.savefig("rf_glucose_bmi.png", dpi=150, bbox_inches="tight")
plt.show()
# Outcome=1 그룹에서 Glucose와 BMI가 상대적으로 높은지 확인

# 6. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
print("\nTrain:", X_train.shape); print("Test :", X_test.shape)

# 7. 기준 모델 - Decision Tree
tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=1
)

tree.fit(X_train, y_train)
tree_train_accuracy = tree.score(X_train, y_train)
tree_test_accuracy = tree.score(X_test, y_test)

print("\n===== Decision Tree =====")
print(f"Train Accuracy: {tree_train_accuracy:.3f}")
print(f"Test Accuracy : {tree_test_accuracy:.3f}")
# Random Forest 성능을 비교하기 위한 기준 모델

# 8. Random Forest
forest = RandomForestClassifier(
    n_estimators=200,      # Decision Tree 200개 생성
    max_depth=7,           # 각 Tree의 최대 깊이
    max_samples=0.7,       # 각 Tree가 Train 데이터의 70%를 Bootstrap Sampling
    max_features="sqrt",   # 각 Node에서 일부 Feature만 무작위 후보로 사용
    random_state=42
)

forest.fit(X_train, y_train)
rf_train_accuracy = forest.score(X_train, y_train)
rf_test_accuracy = forest.score(X_test, y_test)
print("\n===== Random Forest =====")
print(f"Train Accuracy: {rf_train_accuracy:.3f}")
print(f"Test Accuracy : {rf_test_accuracy:.3f}")

# Random Forest
# → Data Randomness + Feature Randomness
# → 서로 다른 여러 Decision Tree 생성
# → 여러 Tree의 예측을 종합하여 최종 분류


# 9. Random Forest 예측 / 기본 평가

y_pred = forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n===== Classification Report =====")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Diabetes"],
        digits=3
    )
)
print(f"Accuracy: {accuracy:.3f}")

# 10. 단일 Train/Test 분할에 따른 성능 변화
print("\n===== Decision Tree - random_state 비교 =====")
for rs in [0, 1, 2, 3, 4]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=0.25,
        random_state=rs,
        stratify=y
    )

    tree_rs = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=1
    )
    tree_rs.fit(X_tr, y_tr)

    print(
        f"random_state={rs} | "
        f"Accuracy={tree_rs.score(X_te, y_te):.3f}"
    )
# 한 번의 Train/Test Split 결과만 보면 데이터 분할에 따라 성능이 달라질 수 있음

# 11. Stratified 5-Fold Cross Validation
cv = StratifiedKFold(
    n_splits=5,       # 5-Fold
    shuffle=True,     # 데이터를 섞은 뒤 Fold 생성
    random_state=1
)

tree_cv = cross_val_score(
    DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=1
    ),
    X,
    y,
    cv=cv,
    scoring="accuracy"
)
print("\n===== Decision Tree 5-Fold CV =====")
print("Fold Scores:", np.round(tree_cv, 3))
print(f"Mean Accuracy: {tree_cv.mean():.3f}")

# 12. Random Forest Cross Validation

rf_cv = cross_val_score(
    RandomForestClassifier(
        n_estimators=300,
        random_state=1
    ),
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n===== Random Forest 5-Fold CV =====")
print("Fold Scores:", np.round(rf_cv, 3))
print(f"Mean Accuracy: {rf_cv.mean():.3f}")
# 여러 Fold의 평균으로 일반화 성능을 더 안정적으로 평가

# 13. Decision Tree / Random Forest CV 비교
cv_result = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "CV Accuracy": [tree_cv.mean(), rf_cv.mean()]
})
print("\n===== CV 성능 비교 =====")
print(cv_result.round(3))

# 14. Random Forest GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [1, 3, 5, 7]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=1),
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
rf_grid.fit(X, y)
print("\n===== GridSearchCV =====")
print(f"Best CV Accuracy: {rf_grid.best_score_:.3f}")
print("Best Parameters:", rf_grid.best_params_)
# 4 × 4 × 4 = 64개 조합
# 각 조합을 5-Fold로 평가

# 15. 기본 Random Forest / 튜닝 Random Forest 비교
print("\n===== Random Forest Tuning =====")
print(f"기본 RF 평균 Accuracy : {rf_cv.mean():.3f}")
print(f"튜닝 RF 평균 Accuracy : {rf_grid.best_score_:.3f}")
# GridSearchCV를 이용하여 더 좋은 Hyperparameter 조합 탐색

# 16. 최적 Random Forest 모델
best_rf = rf_grid.best_estimator_
# Train 데이터로 최종 학습
best_rf.fit(X_train, y_train)
# 남겨둔 Test 데이터 예측
best_pred = best_rf.predict(X_test)
best_accuracy = accuracy_score(
    y_test,
    best_pred
)
print("\n===== Best Random Forest =====")
print(f"Test Accuracy: {best_accuracy:.3f}")

# 17. Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
tn, fp, fn, tp = cm.ravel()
print("\n===== Confusion Matrix =====")
print(cm)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Diabetes"]
).plot(
    cmap="Blues",
    values_format="d"
)
plt.title("Random Forest - Confusion Matrix"); plt.tight_layout()
plt.savefig("rf_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# 18. 최종 Classification Report
print("\n===== Final Classification Report =====")
print(
    classification_report(
        y_test,
        best_pred,
        target_names=["Normal", "Diabetes"],
        digits=3
    )
)
# FN → 실제 당뇨 환자를 정상으로 잘못 예측한 경우

# 19. Feature Importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)
print("\n===== Feature Importance =====")
print(importance_df)

# 20. Feature Importance 시각화
importance_plot = importance_df.sort_values(by="Importance",ascending=True)
plt.figure(figsize=(8, 5))
plt.barh(
    importance_plot["Feature"],
    importance_plot["Importance"]
)

plt.xlabel("Importance"); plt.title("Random Forest Feature Importance")
plt.tight_layout(); plt.savefig("rf_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# 21. Summary
print("\n===== Summary =====")
print(f"Decision Tree CV Accuracy : {tree_cv.mean():.3f}")
print(f"Random Forest CV Accuracy : {rf_cv.mean():.3f}")
print(f"Tuned RF CV Accuracy      : {rf_grid.best_score_:.3f}")
print(f"Final Test Accuracy       : {best_accuracy:.3f}")
print(f"False Negative            : {fn}")

# Decision Tree → 하나의 Tree에 의존
# Random Forest → 여러 Tree를 결합하여 예측
# Cross Validation → 여러 데이터 분할에서 모델 성능을 반복 평가
# GridSearchCV → 여러 Hyperparameter 조합을 Cross Validation으로 비교
# FN(False Negative) → 실제 당뇨 환자를 정상으로 놓친 경우
