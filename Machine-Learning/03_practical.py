
import matplotlib.pyplot as plt  # 그래프 시각화
import numpy as np               # 수치 및 배열 연산

from sklearn.datasets import load_breast_cancer  # 유방암 이진 분류 예제 데이터
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 모델

from sklearn.metrics import (
    accuracy_score,              # 정확도
    classification_report,       # 분류 평가 지표 요약
    confusion_matrix,            # 혼동행렬
    ConfusionMatrixDisplay,      # 혼동행렬 시각화
    f1_score,                    # F1 Score
    precision_score,             # Precision
    recall_score,                # Recall
    roc_auc_score,               # ROC-AUC
    roc_curve,                   # ROC Curve
    precision_recall_curve,      # PR Curve
    average_precision_score      # Average Precision
)

from sklearn.model_selection import train_test_split  # train/test 분리
from sklearn.pipeline import make_pipeline            # 전처리 + 모델 연결
from sklearn.preprocessing import StandardScaler      # 특성 표준화

# =========================================================
# 1. Sigmoid 함수
# =========================================================

def sigmoid(z):  #입력값을 0~1 사이의 확률값으로 변화
    return 1 / (1 + np.exp(-z))


z = np.linspace(
    -10,       # 시작값
    10,        # 끝값
    400        # 데이터 개수
)
probability = sigmoid(z)
plt.figure(figsize=(7, 4))  # 그래프 크기

plt.plot(
    z,                    # x축
    probability,          # y축
    linewidth=2,          # 선 두께
    label="Sigmoid"       # 범례
)

plt.axhline(
    y=0.5,                        # 확률 기준값
    linestyle="--",               # 점선
    label="Threshold = 0.5"
)

plt.axvline(
    x=0,              # z=0 기준선
    linestyle=":"
)

plt.xlabel("z = wx + b")
plt.ylabel("Probability")
plt.title("Sigmoid Function")
plt.ylim(-0.05, 1.05)

plt.grid(
    linestyle="--",   # 격자 모양
    alpha=0.4         # 투명도
)

plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# 2. 데이터 불러오기
# =========================================================
data = load_breast_cancer()
X = data.data              # 입력 Feature
y = data.target            # 0=악성, 1=양성
feature_names = data.feature_names

print("전체 데이터 크기:", X.shape)  # r: (569, 30)
print("클래스 이름:", data.target_names) # r: 클래스 이름:['malignant' 'benign']
print("Feature 개수:", len(feature_names)) # r: Feature 개수: 30

# =========================================================
# 3. Train / Test 데이터 분리
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,                  # 입력 데이터
    y,                  # 정답
    test_size=0.2,      # Test 20%
    random_state=42,    # 분할 결과 고정
    stratify=y          # 클래스 비율 유지
)
print("학습 데이터:", X_train.shape)
print("테스트 데이터:", X_test.shape)

# =========================================================
# 4. 모델 생성 및 학습
# =========================================================
model = make_pipeline(
    StandardScaler(),   # 평균 0, 표준편차 1로 표준화
    LogisticRegression(
        penalty="l2",   # L2 규제
        C=1.0,          # 작을수록 규제가 강함
        solver="lbfgs", # 최적화 알고리즘
        max_iter=1000   # 최대 반복 횟수
    )
)

model.fit(
    X_train,    # 학습 Feature
    y_train     # 학습 정답
)

# =========================================================
# 5. 예측
# =========================================================
# 최종 클래스 0/1
y_pred = model.predict(X_test)
# class 1(benign)의 예측 확률
y_prob = model.predict_proba(X_test)[:, 1]

# =========================================================
# 6. 모델 성능 평가
# =========================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n===== 모델 평가 =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

# Accuracy  : 전체 정답 비율
# Precision : 1이라고 예측한 것 중 실제 1
# Recall    : 실제 1 중 찾아낸 비율
# F1        : Precision과 Recall의 균형
# ROC-AUC   : 클래스 구분 능력

# =========================================================
# 7. Classification Report
# =========================================================
print("\n===== Classification Report =====")
print(
    classification_report(
        y_test,                         # 실제값
        y_pred,                         # 예측값
        target_names=data.target_names, # 클래스 이름
        digits=4                        # 소수점 자리
    )
)

# =========================================================
# 8. Confusion Matrix
# =========================================================
cm = confusion_matrix(
    y_test,    # 실제값
    y_pred     # 예측값
)

print("\n===== Confusion Matrix =====")
print(cm)

# [[TN, FP],
#  [FN, TP]]

ConfusionMatrixDisplay(
    confusion_matrix=cm,                          # 혼동행렬
    display_labels=["malignant(0)", "benign(1)"] # 클래스명
).plot(
    cmap="Blues",       # 색상맵
    values_format="d"   # 정수로 표시
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# =========================================================
# 9. ROC Curve
# =========================================================
fpr, tpr, roc_thresholds = roc_curve(
    y_test,    # 실제값
    y_prob     # class 1 확률
)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,                                       # False Positive Rate
    tpr,                                       # True Positive Rate
    linewidth=2,                               # 선 두께
    label=f"Logistic Regression (AUC={auc:.3f})"
)

# 랜덤 분류기 기준선
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.grid(linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# 10. Precision-Recall Curve : 클래스 불균형 문제에서 특히 유용
# =========================================================

pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
    y_test,    # 실제값
    y_prob     # class 1 확률
)

ap = average_precision_score(
    y_test,    # 실제값
    y_prob     # 예측 확률
)

plt.figure(figsize=(6, 5))

plt.plot(
    pr_recall,          # x축: Recall
    pr_precision,       # y축: Precision
    linewidth=2,        # 선 두께
    label=f"AP={ap:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
print(f"\nAverage Precision(AP): {ap:.4f}")

# =========================================================
# 11. Threshold 변화 비교
# =========================================================
print("\n===== Threshold 비교 =====")

for threshold in [0.3, 0.5, 0.7]:
    # 예측 확률이 threshold 이상이면 class 1로 분류
    y_pred_threshold = (y_prob >= threshold).astype(int)
    # class 1이라고 예측한 것 중 실제 class 1의 비율
    threshold_precision = precision_score(y_test,y_pred_threshold)
    # 실제 class 1 중 모델이 찾아낸 비율
    threshold_recall = recall_score(y_test,y_pred_threshold)
    # Precision과 Recall의 조화평균
    threshold_f1 = f1_score(y_test, y_pred_threshold)
    print(
        f"threshold={threshold:.1f} | "          # 현재 분류 기준값
        f"precision={threshold_precision:.3f} | "  # 정밀도
        f"recall={threshold_recall:.3f} | "        # 재현율
        f"f1={threshold_f1:.3f}"                   # F1 Score
    )

# r: 예시
# threshold=0.3 | precision=0.973 | recall=1.000 | f1=0.986
# threshold=0.5 | precision=0.986 | recall=0.986 | f1=0.986
# threshold=0.7 | precision=0.985 | recall=0.931 | f1=0.957

# 해석:
# threshold가 낮아지면 class 1로 분류하는 범위가 넓어져 Recall이 높아질 수 있음
# threshold가 높아지면 class 1 판정이 엄격해져 Recall이 낮아질 수 있음

# =========================================================
# 12. Train / Test ROC 비교
# =========================================================

# Train/Test 데이터의 class 1 예측 확률
train_prob = model.predict_proba(X_train)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]

# ROC Curve 계산
train_fpr, train_tpr, _ = roc_curve(y_train, train_prob)
test_fpr, test_tpr, _ = roc_curve(y_test, test_prob)

# ROC-AUC 계산
train_auc = roc_auc_score(y_train, train_prob)
test_auc = roc_auc_score(y_test, test_prob)

plt.figure(figsize=(6, 5))

plt.plot(
    train_fpr,                         # Train FPR
    train_tpr,                         # Train TPR
    linewidth=2,                       # 선 두께
    label=f"Train AUC={train_auc:.3f}" # 범례
)

plt.plot(
    test_fpr,                          # Test FPR
    test_tpr,                          # Test TPR
    linewidth=2,                       # 선 두께
    label=f"Test AUC={test_auc:.3f}"   # 범례
)

# 랜덤 분류기 기준선
plt.plot(
    [0, 1],                            # x축
    [0, 1],                            # y축
    linestyle="--"                     # 점선
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Train vs Test ROC")
plt.grid(linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

print("\n===== Train / Test ROC-AUC =====")
print(f"Train AUC: {train_auc:.4f}")  # r: 0.9975
print(f"Test AUC : {test_auc:.4f}")   # r: 0.9954

# Train과 Test의 AUC가 비슷하면 일반화 성능이 좋음
# Train만 높고 Test가 낮으면 과적합을 의심


# =========================================================
# 13. 로지스틱 회귀 계수 해석
# =========================================================

# Pipeline 내부 LogisticRegression 모델
logistic_model = model.named_steps["logisticregression"]

# 각 Feature의 학습된 계수
coefficients = logistic_model.coef_[0]

# 계수 절댓값이 큰 순서로 정렬
feature_coef = sorted(
    zip(feature_names, coefficients), # Feature와 계수 묶기
    key=lambda item: abs(item[1]),    # 절댓값 기준
    reverse=True                      # 내림차순
)

print("\n===== 계수 절댓값 상위 10개 =====")

for name, coef in feature_coef[:10]:
    # 계수를 Odds Ratio로 변환
    odds_ratio = np.exp(coef)
    print(
        f"{name:<30} "
        f"coef={coef:>7.3f} | "
        f"odds ratio={odds_ratio:>7.3f}"
    )

# coef > 0 → class 1(benign) 방향
# coef < 0 → class 0(malignant) 방향
# 절댓값이 클수록 모델 판단에 상대적으로 큰 영향
# 계수는 인과관계를 의미하지 않음

# =========================================================
# 14. 주요 계수 시각화
# =========================================================

selected = feature_coef[:10][::-1]

names = [item[0] for item in selected]
values = [item[1] for item in selected]

plt.figure(figsize=(8, 6))

plt.barh(
    names,     # y축: Feature 이름
    values     # x축: 회귀계수
)

plt.axvline(
    x=0,              # 양수/음수 기준
    linestyle="--",
    linewidth=1
)

plt.xlabel("Coefficient")
plt.title("Top 10 Logistic Regression Coefficients")

plt.grid(
    axis="x",          # x축 기준 격자
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()
plt.show()
