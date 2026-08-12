# =========================================================
# Gradient Boosting - Practical
# Pima Indians Diabetes Dataset
# Hyperparameter / Model Evaluation
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)

# 1. 데이터 불러오기
cols = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv("../datasets/pima-indians-diabetes.csv", names=cols)
X = df.drop(columns=["Outcome"]); y = df["Outcome"]
print("===== Pima Indians Diabetes Dataset =====") ; print(df.head()); print("데이터 크기:", df.shape)

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=1,stratify=y)
print("\nTrain:", X_train.shape); print("Test :", X_test.shape)

# 3. learning_rate에 따른 성능 비교
learning_rates = [0.01, 0.05, 0.1, 0.2]
train_scores = []; test_scores = []

for rate in learning_rates:
    model = GradientBoostingClassifier(
        n_estimators=100,   # Tree 개수 고정
        learning_rate=rate,
        max_depth=3,
        random_state=1
    )

    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))
print("\n===== Learning Rate Comparison =====")

for rate, train_score, test_score in zip(learning_rates, train_scores, test_scores):
    print(f"learning_rate={rate:<4} " f"Train={train_score:.3f} " f"Test={test_score:.3f}")

# 4. learning_rate 성능 시각화
plt.figure(figsize=(8, 4.5))
plt.plot(learning_rates, train_scores, marker="o", label="Train Accuracy")
plt.plot(learning_rates, test_scores, marker="^", label="Test Accuracy")

plt.xlabel("Learning Rate"); plt.ylabel("Accuracy"); plt.title("Accuracy according to Learning Rate")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
# learning_rate가 커질수록 Train Accuracy가 높아질 수 있지만
# Test Accuracy까지 계속 좋아진다는 의미는 아님 → Train/Test 차이가 커지면 과적합 가능성 확인

# 5. n_estimators에 따른 성능 비교
n_estimators_list = [50, 100, 200, 300]
train_scores = []; test_scores = []

for n in n_estimators_list:
    model = GradientBoostingClassifier(
        n_estimators=n,
        learning_rate=0.05,   # learning_rate 고정
        max_depth=3,
        random_state=1
    )
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

print("\n===== n_estimators Comparison =====")

for n, train_score, test_score in zip(
    n_estimators_list, train_scores, test_scores
):
    print(
        f"n_estimators={n:<3} "
        f"Train={train_score:.3f} "
        f"Test={test_score:.3f}"
    )

# 6. n_estimators 성능 시각화
plt.figure(figsize=(8, 4.5))
plt.plot(n_estimators_list, train_scores, marker="o", label="Train Accuracy")
plt.plot(n_estimators_list, test_scores, marker="^", label="Test Accuracy")

plt.xlabel("n_estimators"); plt.ylabel("Accuracy"); plt.title("Accuracy according to n_estimators")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
# Tree 개수가 많아질수록 학습 성능은 높아질 수 있음 : 하지만 Test 성능이 함께 좋아지는지는 별도로 확인해야 함

# 7. GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [1, 2, 3]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=1),
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

# Test 데이터는 최종 평가를 위해 남겨두고 Train 데이터 안에서만 Cross Validation + Hyperparameter 탐색
grid.fit(X_train, y_train)

print("\n===== GridSearchCV ====="); print(f"Best CV Accuracy: {grid.best_score_:.3f}")
print("Best Parameters :", grid.best_params_)

# 8. 최적 모델로 Test 데이터 예측
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("\n===== Final Test ====="); print(f"Test Accuracy: {test_accuracy:.3f}")

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\n===== Confusion Matrix =====")
print(cm)
print(f"TN={tn}"); print(f"FP={fp}"); print(f"FN={fn}"); print(f"TP={tp}")
# FN(False Negative) → 실제 당뇨인데 정상으로 예측

# 10. Classification Report
print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred,target_names=["정상", "당뇨"]))
# Precision → 당뇨라고 예측한 데이터 중 실제 당뇨 비율
# Recall → 실제 당뇨 중 모델이 당뇨라고 찾아낸 비율
# F1-score → Precision과 Recall을 함께 고려한 지표

# 11. Feature Importance
importance = pd.Series(best_model.feature_importances_,index=X.columns).sort_values()
print("\n===== Feature Importance =====")
print(importance.sort_values(ascending=False))

plt.figure(figsize=(8, 4.5))
importance.plot(kind="barh")
plt.xlabel("Feature Importance")
plt.title("Gradient Boosting Feature Importance")
plt.tight_layout(); plt.show()

# 12. Summary
print("\n===== Summary =====")
print("Best Parameters:", grid.best_params_)
print(f"Best CV Accuracy : {grid.best_score_:.3f}")
print(f"Final Test       : {test_accuracy:.3f}")
print(f"False Negative   : {fn}")

# Gradient Boosting → Tree를 순차적으로 추가하며 모델을 개선
# learning_rate → 각 Tree의 기여도
# n_estimators → 순차적으로 학습할 Tree 개수
# Train Accuracy ↑ / Test Accuracy 정체 또는 ↓ : → 과적합 가능성
# GridSearchCV → Train 데이터 안에서 최적 Hyperparameter 탐색
# Test 데이터 → 최종 모델의 일반화 성능 평가

# 의료 분류 문제에서는 Accuracy뿐 아니라 Recall과 FN도 함께 확인