#실제 CSV → One-Hot Encoding → Train/Test
# → Entropy 기반 Decision Tree → 예측
# → Accuracy / Classification Report / Confusion Matrix
# → Feature Importance → 모델 저장

# =========================================================
# Decision Tree - Practical
# Loan Approval Dataset
# =========================================================

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 한글 폰트 설정
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False


# 1. Loan 데이터 불러오기
loan_data = pd.read_csv("../datasets/decision/train_loan_80.csv")
print("===== Loan Dataset =====")
print(loan_data.head())
print("\n데이터 크기:", loan_data.shape)
print("\n데이터 타입:")
print(loan_data.dtypes)

# 2. 범주형 데이터 One-Hot Encoding
loan_encoded = pd.get_dummies(
    loan_data,
    drop_first=True,     # 각 범주의 첫 번째 항목 제거
    dtype=int
)

print("\n===== One-Hot Encoding =====")
print(loan_encoded.head())
# 문자열 형태의 범주형 Feature를 숫자형 Feature로 변환

# 3. Feature / Target 분리
X = loan_encoded.drop(columns=["Loan_Status_Y"])
y = loan_encoded["Loan_Status_Y"]
print("\nFeature 수:", X.shape[1])
print("Target 분포:")
print(y.value_counts())
# X : 대출 승인 여부 예측에 사용할 입력 Feature
# y : Loan_Status_Y → 1 승인 / 0 거절

# 4. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,       # 전체 데이터의 20%를 Test로 사용
    random_state=42,     # 분할 결과 고정
    stratify=y           # 승인 / 거절 비율 유지
)
print("\nTrain:", X_train.shape)
print("Test :", X_test.shape)

# 5. Decision Tree 모델 생성
model = DecisionTreeClassifier(
    criterion="entropy",  # Entropy를 기준으로 분할
    max_depth=4,          # Tree 깊이 제한
    random_state=42
)
# criterion="entropy" → Entropy를 줄이고 Information Gain이 큰 분할을 선택
# max_depth → Tree가 지나치게 깊어지는 것을 제한하여 과적합 완화

# 6. 모델 학습
model.fit(X_train, y_train)  # Train 데이터로 Decision Tree 학습

# 7. Train / Test Accuracy 비교
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print("\n===== Accuracy =====")
print(f"Train Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy : {test_accuracy:.3f}")
# Train >> Test이면 과적합 가능성 확인

# 8. Test 데이터 예측
y_pred = model.predict(X_test)  # 학습한 모델로 대출 승인 여부 예측
accuracy = accuracy_score(y_test, y_pred)  # 전체 분류 정확도 계산
print("\nFinal Accuracy:", round(accuracy, 3))

# 9. Classification Report
print("\n===== Classification Report =====")
report = classification_report(
    y_test,
    y_pred,
    target_names=["Denied", "Approved"],
    digits=3
)
print(report)
# Precision : 승인/거절로 예측한 것 중 실제 정답 비율
# Recall    : 실제 승인/거절 중 모델이 찾아낸 비율
# F1 Score  : Precision과 Recall의 균형

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n===== Confusion Matrix =====")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Denied", "Approved"]
).plot(cmap="Blues", values_format="d")
plt.title("Loan Decision Tree - Confusion Matrix"); plt.tight_layout()
plt.savefig("decision_tree_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# 11. Decision Tree 시각화
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,                 # Feature 이름
    class_names=["Denied", "Approved"],      # Class 이름
    filled=True,                             # Class별 노드 색상
    rounded=True,                            # 둥근 노드
    fontsize=7
)
plt.title("Loan Approval Decision Tree")
plt.tight_layout()
plt.savefig("decision_tree_loan.png", dpi=150, bbox_inches="tight")
plt.show()

# 12. Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("\n===== Feature Importance =====")
print(importance.head(10))

# 13. Feature Importance 시각화

top_importance = importance.head(10).sort_values()
plt.figure(figsize=(9, 5))
top_importance.plot(kind="barh")

plt.xlabel("Feature Importance"); plt.title("Top 10 Feature Importance")
plt.tight_layout(); plt.savefig("decision_tree_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
# 값이 클수록 Tree 분할에 상대적으로 많이 기여한 Feature


# 14. 모델 저장

joblib.dump(model, "loan_model.pkl")
joblib.dump(list(X.columns), "model_features.pkl")
print("\n모델 저장 완료")
print("- loan_model.pkl")
print("- model_features.pkl")
# 모델뿐 아니라 학습 당시 Feature 순서도 함께 저장
# 새로운 데이터 예측 시 같은 Feature 구성을 사용해야 함

# 15. 저장된 모델 불러오기
loaded_model = joblib.load("loan_model.pkl")
loaded_features = joblib.load("model_features.pkl")
loaded_accuracy = loaded_model.score(X_test, y_test)
print("\n===== Loaded Model =====")
print(f"Accuracy: {loaded_accuracy:.3f}")
print("Feature 수:", len(loaded_features))

# 16. Conclusion
print("\n===== Conclusion =====")
print(f"Train Accuracy : {train_accuracy:.3f}")
print(f"Test Accuracy  : {test_accuracy:.3f}")
print(f"Final Accuracy : {accuracy:.3f}")
# 실제 범주형 데이터를 One-Hot Encoding 후 Decision Tree에 적용
# Entropy를 기준으로 분할
# max_depth로 Tree 복잡도 제한
# Accuracy뿐 아니라 Classification Report와 Confusion Matrix 확인
# Feature Importance로 주요 변수 확인
# 모델과 Feature 정보를 저장하여 재사용