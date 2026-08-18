import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

# Wine 데이터
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y  # 각 품종의 비율을 유지하면서 Train/Test 분리
)

print(f"훈련: {X_train.shape}, 테스트: {X_test.shape}")
print(f"품종: {list(wine.target_names)}")

# 표준화와 SVM을 하나의 Pipeline으로 구성
pipe = Pipeline([
    ("scaler", StandardScaler()),  # Feature를 평균 0, 표준편차 1로 표준화
    ("svc", SVC())                 # Support Vector Classifier
])

# SVM 하이퍼파라미터 후보
param_grid = {
    "svc__C": [0.1, 1, 10, 100],            # 작을수록 오차를 더 허용하고 Margin을 넓게 유지
    "svc__gamma": ["scale", 0.01, 0.1, 1],  # RBF Kernel에서 각 데이터의 영향 범위
    "svc__kernel": ["linear", "rbf"]         # Linear / RBF Kernel 비교
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,               # 5-Fold Cross Validation
    scoring="accuracy", # Accuracy를 기준으로 최적 모델 선택
    n_jobs=-1
)

grid.fit(X_train, y_train)

# 가장 성능이 좋은 모델로 테스트 데이터 예측
y_pred = grid.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)  # Accuracy = 전체 데이터 중 맞춘 비율

print("\n=== SVM Result ===")
print("가장 좋은 설정:", grid.best_params_)
print("교차검증 정확도:", round(grid.best_score_, 3))
print("테스트 정확도:", round(accuracy, 3))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=wine.target_names))