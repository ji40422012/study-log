# =========================================================
# Random Forest - Basic
# Loan Approval Dataset
# =========================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# 1. Loan 데이터 불러오기
loan_data = pd.read_csv("../datasets/decision/train_loan_80.csv")
print("===== Loan Dataset =====")
print(loan_data.head())
print("\n데이터 크기:", loan_data.shape)

# 2. One-Hot Encoding
# 범주형 데이터를 숫자형 Feature로 변환
loan_encoded = pd.get_dummies(loan_data, drop_first=True, dtype=int)
print("\n===== One-Hot Encoding =====")
print(loan_encoded.head())

# 3. Feature / Target 분리
X = loan_encoded.drop(columns=["Loan_Status_Y"])
y = loan_encoded["Loan_Status_Y"]
print("\nFeature 수:", X.shape[1])
print("Target 분포:")
print(y.value_counts())

# 4. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 전체의 20%를 Test로 사용
    random_state=42,     # 분할 결과 고정
    stratify=y           # 승인 / 거절 비율 유지
)
print("\nTrain:", X_train.shape)
print("Test :", X_test.shape)

# 5. Random Forest 모델 생성
model = RandomForestClassifier(
    n_estimators=200,     # Decision Tree 200개 생성
    max_depth=7,          # 각 Tree의 최대 깊이 제한
    max_samples=0.7,      # 각 Tree가 Train 데이터의 70%를 Bootstrap Sampling
    max_features="sqrt",  # 각 분할에서 전체 Feature의 제곱근 개수만 무작위 후보로 사용
    random_state=42
)
# Random Forest → 여러 Decision Tree를 서로 다르게 만들어 결과를 종합하는 Ensemble 모델
## max_samples → 각 Tree가 학습할 데이터를 무작위 복원추출
## max_features → 각 Node에서 분할 후보로 사용할 Feature도 무작위 선택

# 6. 모델 학습
model.fit(X_train, y_train)  # 여러 Decision Tree를 학습

# 7. Train / Test Accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print("\n===== Accuracy =====")
print(f"Train Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy : {test_accuracy:.3f}")
# Train >> Test이면 과적합 가능성 확인

# 8. Test 데이터 예측
y_pred = model.predict(X_test)  # 여러 Tree의 예측을 종합하여 최종 Class 결정
accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Accuracy:", round(accuracy, 3))

# 9. Classification Report
print("\n===== Classification Report =====")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Denied", "Approved"],
        digits=3
    )
)
# Precision : 해당 Class로 예측한 것 중 실제 정답 비율
# Recall    : 실제 해당 Class 중 모델이 찾아낸 비율
# F1 Score  : Precision과 Recall의 균형

# 10. Feature Importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n===== Feature Importance =====")
print(importance_df)

# 11. Feature Importance 시각화
importance_plot = importance_df.sort_values(by="Importance", ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(
    importance_plot["Feature"],
    importance_plot["Importance"]
)
plt.xlabel("Importance"); plt.ylabel("Feature")
plt.title("Random Forest Feature Importance"); plt.tight_layout(); plt.show()
# 값이 클수록 Random Forest의 예측에 상대적으로 많이 기여한 Feature

# 12. Summary
print("\n===== Summary =====")
print(f"Train Accuracy : {train_accuracy:.3f}")
print(f"Test Accuracy  : {test_accuracy:.3f}")
print(f"Final Accuracy : {accuracy:.3f}")
print("\nTop Feature:")
print(importance_df.head(5))

# Random Forest → 여러 Decision Tree를 만들고 예측 결과를 종합
# n_estimators → 생성할 Tree 개수
# max_samples → 각 Tree가 사용할 학습 데이터 비율
# max_features → 각 분할에서 사용할 Feature 후보 수
# Feature Importance → 여러 Tree에서 각 Feature가 분할에 기여한 상대적 중요도