import matplotlib.pyplot as plt  # 그래프 시각화

from sklearn.datasets import load_breast_cancer  # 유방암 데이터셋
from sklearn.linear_model import LogisticRegression  # Logistic Regression 모델

from sklearn.metrics import (
    accuracy_score,          # 정확도
    confusion_matrix,        # 혼동행렬 계산
    ConfusionMatrixDisplay   # 혼동행렬 시각화
)
from sklearn.model_selection import train_test_split  # Train/Test 분리

# =========================================================
# 1. 데이터 불러오기
# =========================================================
data = load_breast_cancer()
X = data.data                  # Feature
y = data.target                # Target
feature_names = data.feature_names

print("데이터 크기 :", X.shape)            # r: (569, 30)
print("클래스 :", data.target_names)      # r: ['malignant' 'benign']
print("Feature 수 :", len(feature_names)) # r: 30
# X : 입력 데이터
# y : 정답(Label)

# =========================================================
# 2. Train / Test Split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,          # Test 20%
    random_state=42         # 결과 고정
)

print("\nTrain :", X_train.shape)
print("Test :", X_test.shape)

# r:
# Train : (455, 30)
# Test : (114, 30)

# =========================================================
# 3. Logistic Regression 모델 생성
# =========================================================
model = LogisticRegression(
    max_iter=1000           # 반복횟수 증가
)
# max_iter : 반복 학습 횟수(데이터가 복잡하면 늘려주는 것이 좋음)

# =========================================================
# 4. 모델 학습
# =========================================================
model.fit(
    X_train,
    y_train
)
# fit(): Train 데이터를 이용하여 Logistic Regression 모델 학습

# =========================================================
# 5. 예측
# =========================================================
y_pred = model.predict(
    X_test
)
# predict() : 학습한 모델로 클래스를 예측

# =========================================================
# 6. 확률 확인
# =========================================================
y_prob = model.predict_proba(
    X_test
)
print("\n예측 확률(앞 5개)")
print(y_prob[:5])

# r: [[0.99 0.01]
##  [0.02 0.98]
##  ...]
# 첫 번째 값 : class 0 확률
# 두 번째 값 : class 1 확률

# =========================================================
# 7. Accuracy
# =========================================================
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy :", round(accuracy,3))
# r: 약 0.96 ~ 0.98
# Accuracy: 전체 데이터 중 맞춘 비율

# =========================================================
# 8. Confusion Matrix
# =========================================================
cm = confusion_matrix(
    y_test,
    y_pred
)
print("\nConfusion Matrix")
print(cm)

# r: [[39  4]
##  [ 1 70]]
# 행 : 실제값
# 열 : 예측값

# =========================================================
# 9. Confusion Matrix 시각화
# =========================================================
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)
plt.figure(figsize=(5,5))             # Figure 크기
disp.plot(
    cmap="Blues",                     # 색상
    values_format="d"                 # 정수 출력
)

plt.title("Confusion Matrix")         # 제목
plt.tight_layout()                    # 여백 자동 조정
plt.show()

# =========================================================
# 10. Coefficient 확인
# =========================================================
coef = model.coef_[0]
print("\n===== Coefficient =====")

for name, value in zip(feature_names, coef):
    print(
        f"{name:25s} : {value:.3f}"
    )
# 양수 : class 1 방향으로 예측 증가
# 음수 : class 0 방향으로 예측 증가

# =========================================================
# 11. Summary
# =========================================================
print("\n===== Summary =====")
print(f"Accuracy : {accuracy:.3f}")
print("\nLogistic Regression 학습 완료")

# Accuracy : 모델의 전체 분류 성능
# Confusion Matrix : 어떤 클래스를 잘못 분류했는지 확인
# predict_proba() : 각 클래스의 예측 확률 반환
# predict() : 최종 클래스를 반환
# coef_ : # Feature가 예측에 미치는 영향
