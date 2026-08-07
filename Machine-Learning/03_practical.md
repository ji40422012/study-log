# 로지스틱 회귀(Logistic Regression)

## 학습 내용
scikit-learn의 내장 데이터셋인 `Breast Cancer Wisconsin Diagnostic Dataset` 사용
로지스틱 회귀 이진 분류
- Sigmoid와 확률
- Odds / Logit
- Threshold와 Decision Boundary
- Log Loss
- StandardScaler + Pipeline
- Accuracy / Precision / Recall / F1
- Confusion Matrix
- ROC / AUC
- Precision-Recall Curve / AP
- Threshold 변화
- Train/Test ROC 비교
- 회귀계수와 Odds Ratio

---

# 1. 로지스틱 회귀
 `Regression`이지만 주로 **분류(Classification)** 
선형회귀가 연속적인 값을 예측한다면 
 로지스틱 회귀는 특정 클래스에 속할 **확률**을 예측
```text
Feature → 선형결합(wx+b) → Sigmoid → 확률 → Threshold → Class
```

```text
악성 / 양성
합격 / 불합격
스팸 / 정상메일

이번에는 
0 = malignant(악성) / 1 = benign(양성)
```

---

# 2. Sigmoid 함수

선형결합 결과는 음수부터 양수까지 제한 없이 나올 수 있으므로
```text
z = w1x1 + w2x2 + ... + b
```
이를 `0~1` 사이의 값으로 바꾸기 위해 Sigmoid 사용

```text
σ(z) = 1 / (1 + e^(-z))
```

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

특징:
```text
z << 0 → 0에 가까움
z = 0  → 0.5
z >> 0 → 1에 가까움
```
즉 Sigmoid 출력값을 class 1의 확률로 해석

---

# 3. Threshold와 Decision Boundary
모델은 먼저 확률을 계산하고 Threshold를 기준으로 최종 클래스를 결정

기본 Threshold가 `0.5`라면:
```text
p < 0.5  → class 0
p >= 0.5 → class 1
```
Threshold는 반드시 0.5일 필요는 없으며 
문제의 목적에 따라 조절 가능
Threshold ↓ → class 1 판정 증가 → Recall 증가 가능
Threshold ↑ → class 1 판정 감소 → Precision 증가 가능

예를들어
**질병 검사처럼 Positive를 놓치면 위험한 경우**
```text
Threshold ↓ (예: 0.5 → 0.3)
→ Positive 판정 증가 → FN 감소 가능 → Recall 증가
```
**정상 데이터를 Positive로 잘못 판단하면 문제가 큰 경우**
```text
Threshold ↑ (예: 0.5 → 0.7)
→ Positive 판정 기준이 엄격 → FP 감소 가능 → Precision 증가
```

---

# 4. Odds와 Logit
확률 `p`를 Odds로 표현하면:

```text
Odds = p / (1-p)
```
예를 들어 class 1 확률이 0.8이라면:
```text
Odds = 0.8 / 0.2 = 4
```
즉 class 1이 될 가능성이 class 0보다 `4:1`

Odds에 로그를 적용한 것이 Logit이다.
```text
Logit = log(p / (1-p))
```

로지스틱 회귀는 이 Log-Odds를 Feature의 선형결합으로 학습
```text
log(p / (1-p)) = b + w1x1 + w2x2 + ...
```

---

# 5. 손실 함수: Log Loss
선형회귀에서는 MSE를 많이 사용하지만 
이진 분류에서는 주로 **Log Loss(Binary Cross Entropy)** 

```text
Loss = -[y log(p) + (1-y) log(1-p)]
```

- 실제값이 1인데 `p≈1` → 손실 작음
- 실제값이 1인데 `p≈0` → 손실 매우 큼
즉 모델이 틀린 답을 높은 확률로 확신할수록 큰 패널티를 준다.

---

# 6. 데이터

```python
data = load_breast_cancer()
X = data.data
y = data.target
```

```text
전체 데이터: 569개
Feature: 30개
Class: 2개
```

```text
0 = malignant
1 = benign
```

---

# 7. Train / Test 분리

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

주요 옵션:

| 옵션 | 의미 |
|---|---|
| `test_size=0.2` | 전체의 20%를 Test로 사용 |
| `random_state=42` | 분할 결과 고정 |
| `stratify=y` | 클래스 비율 유지 |

결과:

```text
Train: (455, 30)
Test : (114, 30)
```

---

# 8. StandardScaler + Pipeline

Feature마다 단위와 크기가 다르기 때문에 표준화
```python
model = make_pipeline(
    StandardScaler(),  # Feature를 평균 0, 표준편차 1로 표준화

    LogisticRegression(
        penalty="l2",    # L2 규제: 계수가 너무 커지는 것을 제한
        C=1.0,           # 규제 강도의 역수: 작을수록 규제가 강함
        solver="lbfgs",  # 최적의 계수를 찾는 알고리즘
        max_iter=1000    # 최대 학습 반복 횟수
    )
)
```

`StandardScaler`:
```text
평균 ≈ 0
표준편차 ≈ 1
```

`Pipeline` 처리 흐름: 원본 데이터 → StandardScaler → LogisticRegression → 예측

### LogisticRegression 주요 옵션

| 옵션 | 의미 |
|---|---|
| `penalty="l2"` | L2 규제로 큰 계수 제한 |
| `C=1.0` | 규제 강도의 역수 |
| `solver="lbfgs"` | 최적화 알고리즘 |
| `max_iter=1000` | 최대 반복 횟수 |

`C`는 작을수록 규제가 강하고 클수록 규제가 약하다.

---

# 9. 모델 학습과 예측

```python
model.fit(X_train, y_train)
```

최종 클래스 예측:
```python
y_pred = model.predict(X_test)
```

확률 예측:
```python
y_prob = model.predict_proba(X_test)[:, 1]
```

차이:
```text
predict()       → 0 또는 1
predict_proba() → 각 클래스의 확률
```
`[:, 1]`은 이번 데이터에서 `benign(class 1)`일 확률을 가져온다.

ROC, AUC, PR Curve 계산에는 `y_pred`가 아니라 **`y_prob`**

---

# 10. 주요 평가 코드

```python
accuracy = accuracy_score(y_test, y_pred)    #Accuracy  = 전체 중 맞춘 비율
precision = precision_score(y_test, y_pred)  #Precision = 1이라고 예측한 것 중 실제 1
recall = recall_score(y_test, y_pred)        #Recall    = 실제 1 중 찾아낸 비율
f1 = f1_score(y_test, y_pred)                #F1        = Precision과 Recall의 조화평균
auc = roc_auc_score(y_test, y_prob)          #ROC-AUC   = 여러 Threshold에서의 클래스 구분 능력      
```

---

# 11. 모델 평가 결과

```text
Accuracy : 0.9825
Precision: 0.9861
Recall   : 0.9861
F1 Score : 0.9861
ROC-AUC  : 0.9954
```
Accuracy가 약 `98.3%`로 Test 데이터 대부분을 정확하게 분류했다.
Precision과 Recall이 모두 약 `98.6%`이므로 class 1에 대한 오탐과 미탐 모두 적었다.
ROC-AUC가 `0.9954`로 1에 매우 가까워 악성과 양성을 구분하는 능력이 매우 높게 나타났다.
---

# 12. Confusion Matrix

주요 코드:
```python
cm = confusion_matrix(y_test, y_pred)
```
결과:

```text
[[41  1]
 [ 1 71]]
```
구조:
```text
              예측 0   예측 1
실제 0          TN       FP
실제 1          FN       TP
```

따라서:
```text
TN = 41
FP = 1  #실제 악성을 양성으로 잘못 예측: 1개
FN = 1  #실제 양성을 악성으로 잘못 예측: 1개
TP = 71
```
`ConfusionMatrixDisplay` : 결과를 2×2 그래프로 

---

# 13. Precision과 Recall

### Precision

```text
Precision = TP / (TP + FP)
```
모델이 Positive라고 예측한 것 중 실제 Positive의 비율

### Recall

```text
Recall = TP / (TP + FN)
```
실제 Positive 중 모델이 놓치지 않고 찾아낸 비율


```text
정상 메일을 스팸으로 보내면 문제 → Precision 중요
질병 환자를 놓치면 문제 → Recall 중요
```

---

# 14. F1 Score
```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```
Precision과 Recall 중 하나만 높아서는 좋은 F1이 나오기 어렵다.

이번 결과:

```text
F1 = 0.9861
```
Precision과 Recall이 모두 높아 F1도 높게 나타났다.

---

# 15. ROC Curve와 AUC

주요 코드:
```python
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
```

ROC Curve:
```text
x축 = FPR
y축 = TPR
```

```text
TPR = Recall
FPR = FP / (FP + TN)
```

이번 결과:
```text
AUC = 0.9954
```
ROC Curve가 왼쪽 위에 가깝게 나타났고 
AUC도 거의 1이므로 클래스 구분 성능이 매우 높다.

---

# 16. Precision-Recall Curve

주요 코드:
```python
pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
```

```text
x축 = Recall
y축 = Precision
```

결과:
```text
AP ≈ 0.997
```
높은 Recall에서도 Precision을 잘 유지했다.
PR Curve는 특히 Positive 클래스가 적은 **불균형 데이터**에서 ROC-AUC와 함께 확인 필요

---

# 17. Threshold 변화

실습 코드:
```python
for threshold in [0.3, 0.5, 0.7]:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    threshold_precision = precision_score(y_test, y_pred_threshold)
    threshold_recall = recall_score(y_test, y_pred_threshold)
    threshold_f1 = f1_score(y_test, y_pred_threshold)
```

결과:
```text
threshold=0.3 | precision=0.973 | recall=1.000 | f1=0.986
threshold=0.5 | precision=0.986 | recall=0.986 | f1=0.986
threshold=0.7 | precision=0.985 | recall=0.931 | f1=0.957
```
`Threshold=0.3`에서는 Recall이 `1.0`으로 실제 class 1을 모두 찾아냈다.
반대로 `Threshold=0.7`에서는 판정 기준이 엄격해지면서 Recall이 `0.931`로 감소
따라서 Threshold는 단순히 0.5로 고정하는 값이 아니라 **목적에 맞게 결정해야 하는 값**

---

# 18. Train / Test ROC 비교

주요 코드:
```python
train_prob = model.predict_proba(X_train)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, train_prob)
test_auc = roc_auc_score(y_test, test_prob)
```

결과:
```text
Train AUC: 0.9975
Test AUC : 0.9954
```
Train과 Test 성능이 모두 높고 차이도 매우 작다.

```text
Train ≈ Test → 일반화 성능이 비교적 안정적
Train >> Test → 과적합 가능성
```
(이 모델에서는 눈에 띄는 과적합 없음)

---

# 19. 로지스틱 회귀 계수
Pipeline에서 학습된 모델을 가져온다.

```python
logistic_model = model.named_steps["logisticregression"]
coefficients = logistic_model.coef_[0]
```

이번 Target:
```text
0 = malignant
1 = benign
```

따라서:
```text
coef > 0 → benign 방향
coef < 0 → malignant 방향
```

계수의 절댓값이 클수록 모델 판단에 상대적으로 크게 작용
단, **계수가 크다고 해서 해당 Feature가 결과의 원인이라는 뜻은 아니다.**

---

# 20. Odds Ratio

계수에 지수함수를 적용하면 Odds Ratio로 변환할 수 있다.

```python
odds_ratio = np.exp(coef)
```

```text
Odds Ratio > 1 → class 1 Odds 증가
Odds Ratio < 1 → class 1 Odds 감소
```
`StandardScaler`를 사용했기 때문에 
여기서는 대략 Feature가 1 표준편차 변했을 때의 Odds 변화 방향으로 해석

---

# 21. 실습에서 사용한 주요 코드

### 데이터 분리

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 모델

```python
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)
)
```

### 학습 / 예측

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
```

### 모델 평가

```python
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
roc_auc_score(y_test, y_prob)
```

### 혼동행렬

```python
confusion_matrix(y_test, y_pred)
```

### ROC

```python
fpr, tpr, _ = roc_curve(y_test, y_prob)
```

### PR Curve

```python
precision, recall, _ = precision_recall_curve(y_test, y_prob)
```

---

# 22. 분석 결과 정리
분석 결과
Test 데이터 114개 중 112개를 바르게 분류

```text
Accuracy  = 0.9825
F1        = 0.9861
ROC-AUC   = 0.9954
```

Confusion Matrix에서도 FP와 FN이 각각 1개로 오분류가 적음
Threshold를 바꾸면서 Precision과 Recall이 달라지는 것을 확인했고, 
특히 Threshold를 0.3으로 낮추면 Recall이 1.0까지 증가했다.
그러나 실제라면? 
**새로운 데이터에서도 오분류가 적도록 일반화 성능을 높여야 할것**

Train AUC와 Test AUC도 거의 동일해 현재 설정에서는 과적합이 크게 나타나지 않았다.

계수 분석을 통해 로지스틱 회귀가 단순히 결과만 내는 모델이 아니라 
각 Feature가 어느 클래스 방향으로 작용했는지 해석할 수 있는 모델
---

# 정리

```text
데이터
  ↓
Train/Test 분리
  ↓
표준화
  ↓
Logistic Regression
  ↓
확률 예측
  ↓
Threshold로 Class 결정
  ↓
Accuracy / Precision / Recall / F1
  ↓
Confusion Matrix
  ↓
ROC-AUC / PR Curve
  ↓
Threshold 조절
  ↓
Train/Test 성능 비교
  ↓
계수 및 Odds Ratio 해석
```

로지스틱 회귀에서는 단순히 `Accuracy가 몇 %인가`보다 
**확률 → Threshold → 오분류 형태 → ROC/PR → 모델 계수**의 확인 흐름 중요

