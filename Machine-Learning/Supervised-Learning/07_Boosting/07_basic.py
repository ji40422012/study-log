# =========================================================
# Gradient Boosting - Basic
# Pima Indians Diabetes Dataset
# =========================================================

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1. 데이터 불러오기 : 피마인디언 데이터셋
cols = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv("../datasets/pima-indians-diabetes.csv", names=cols)
X = df.drop(columns=["Outcome"]); y = df["Outcome"]
print("===== Pima Indians Diabetes Dataset =====")
print(df.head()); print("데이터 크기:", df.shape)

# 2. Gradient Boosting 모델
model = GradientBoostingClassifier(
    n_estimators=100,  # 순차적으로 학습할 Tree 개수
    learning_rate=0.1,  # 각 Tree의 보정 결과를 반영하는 정도
    max_depth=3,  # 개별 Tree의 최대 깊이
    random_state=1
)
# Random Forest → 여러 Tree를 독립적으로 학습한 후 결과를 종합
# Gradient Boosting → Tree를 순차적으로 학습 → 이전 모델의 부족한 부분을 다음 Tree가 보완

# 3. Cross Validation
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)
scores = cross_val_score(model,X,y,cv=cv,scoring="accuracy")
print("\n===== Gradient Boosting =====")
print("5-Fold Accuracy:", np.round(scores, 3))
print(f"평균 Accuracy: {scores.mean():.3f}")

# 4. learning_rate 비교
learning_rates = [0.01, 0.05, 0.1, 0.2]
print("\n===== Learning Rate Comparison =====")
for rate in learning_rates:
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=rate,
        max_depth=3,
        random_state=1
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(
        f"learning_rate={rate:<4} "
        f"평균 Accuracy={scores.mean():.3f}"
    )

# 5. Summary
print("\n===== Summary =====")
print("Random Forest     → Tree를 독립적으로 학습")
print("Gradient Boosting → Tree를 순차적으로 학습")
print("learning_rate     → 각 Tree의 보정 정도")
print("n_estimators      → 순차적으로 학습할 Tree 개수")
