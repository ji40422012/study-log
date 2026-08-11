# Random Forest (Basic)
여러 개의 Decision Tree를 만들고 각 Tree의 예측 결과를 종합하여  
최종 결과를 예측하는 **Ensemble Learning 모델**

Decision Tree에서 사용했던 `Loan Approval Dataset` 이용  
Random Forest의 기본 구조와 주요 Hyperparameter 학습

- Random Forest
- Ensemble Learning
- Bootstrap Sampling
- `n_estimators`
- `max_depth`
- `max_samples`
- `max_features`
- Train / Test Accuracy
- Classification Report
- Feature Importance

---

## 1. Random Forest
Decision Tree는 하나의 Tree를 만들어 예측한다
Random Forest는 여러 개의 Decision Tree를 만들고  
각 Tree의 예측 결과를 종합하여 최종 결과를 결정한다.

```text
Data 
 ↓
Decision Tree
 ↓
Prediction
```

```text
                 ┌→ Tree 1 ─→ Prediction
Training Data ───┼→ Tree 2 ─→ Prediction
                 ├→ Tree 3 ─→ Prediction
                 └→ ...
                         ↓
                    결과 종합
                         ↓
                  Final Prediction
```

분류 문제에서는 여러 Tree의 예측 결과를 바탕으로 최종 Class를 결정한다.

---

## 2. Decision Tree와 Random Forest

| Decision Tree | Random Forest |
|---|---|
| 하나의 Tree | 여러 개의 Tree |
| 전체 데이터에 크게 의존 | 서로 다른 데이터로 Tree 생성 |
| 특정 Feature에 의존할 수 있음 | Feature도 무작위로 선택 |
| 데이터 변화에 민감할 수 있음 | 여러 Tree를 결합하여 변동성 감소 |
| 구조를 직접 해석하기 쉬움 | 개별 Tree 해석은 상대적으로 어려움 |

Random Forest의 핵심은
Decision Tree 하나의 예측에 의존하지 않고  
서로 조금씩 다른 Tree의 결과를 종합

---

## 3. Ensemble Learning
여러 개의 모델을 결합하여 하나의 최종 모델을 만드는 방법

```text
Model 1 ─┐
Model 2 ─┤
Model 3 ─┼→ Ensemble → Final Prediction
Model 4 ─┤
Model 5 ─┘
```
---

## 4. Loan Dataset

```python
loan_data = pd.read_csv("../datasets/decision/train_loan_80.csv")
```

```text
Feature
  ↓
Random Forest
  ↓
Loan Status

0 → Denied / 1 → Approved
```
Decision Tree에서 사용했던 데이터와 같은 데이터를 사용하여  
단일 Tree와 여러 Tree를 사용하는 방식의 차이를 확인

---

## 5. One-Hot Encoding

```python
loan_encoded = pd.get_dummies(
    loan_data,
    drop_first=True,
    dtype=int
)
```
범주형 Feature를 모델이 처리할 수 있는 숫자 형태로 변환
`drop_first=True`를 사용하여 각 범주의 첫 번째 항목을 제거한다
(의사결정나무에서 했던 동일 작업)

---

## 6. Feature / Target

```python
X = loan_encoded.drop(columns=["Loan_Status_Y"])
y = loan_encoded["Loan_Status_Y"]
```

```text
X → 대출 승인 여부를 판단할 Feature / y → 실제 대출 승인 여부
```

---

## 7. Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```
(의사결정나무 동일)

| 옵션 | 의미 |
|---|---|
| `test_size=0.2` | 전체 데이터의 20%를 Test로 사용 |
| `random_state=42` | 분할 결과 고정 |
| `stratify=y` | 승인 / 거절 클래스 비율 유지 |

```text
전체 데이터
   ↓
Train 80% / Test  20%
```

---

## 8. Random Forest 모델

```python
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    max_samples=0.7,
    max_features="sqrt",
    random_state=42
)
```
주요 Hyperparameter:

| 옵션 | 의미 |
|---|---|
| `n_estimators=200` | Decision Tree 200개 생성 |
| `max_depth=7` | 각 Tree의 최대 깊이 |
| `max_samples=0.7` | 각 Tree가 사용할 학습 데이터 크기 |
| `max_features="sqrt"` | 각 분할에서 사용할 Feature 후보 수 |
| `random_state=42` | 난수 고정 |

---

## 9. n_estimators

```python
n_estimators=200
```
Random Forest에서 생성할 Decision Tree의 개수

```text
n_estimators=200

Tree 1
Tree 2
Tree 3
...
Tree 200
   ↓
예측 결과 종합
```

Tree 수를 늘리면 일반적으로 예측 결과가 안정될 수 있지만  
학습 시간과 계산량도 증가한다.

---

## 10. Bootstrap Sampling

Random Forest는 모든 Tree가 완전히 동일한 데이터를 학습하지 않도록  
학습 데이터를 무작위로 복원추출하여 사용

```text
Original Training Data
        ↓
Bootstrap Sample 1 → Tree 1
Bootstrap Sample 2 → Tree 2
Bootstrap Sample 3 → Tree 3
        ...
```

**복원추출**이므로 한 데이터가 여러 번 선택될 수도 있고  
어떤 데이터는 선택되지 않을 수도 있다.

이렇게 서로 다른 데이터를 이용하여 Tree를 만들면  
각 Tree가 조금씩 다른 판단 규칙을 학습하게 된다.

---

## 11. max_samples

```python
max_samples=0.7
```

각 Tree가 학습할 때 사용할 데이터의 크기를 설정한다.

이번 코드에서는:

```text
Training Data의 70%
        ↓
Bootstrap Sampling
        ↓
각 Decision Tree 학습
```

각 Tree가 동일한 전체 데이터를 그대로 사용하는 것이 아니라  
무작위로 추출된 데이터를 사용하도록 한다.

---

## 12. max_features

```python
max_features="sqrt"
```
`max_samples`와 `max_features`는 서로 다른 역할을 한다.

```text
max_samples
→ 학습할 Data Sample을 무작위 선택

max_features
→ Node 분할에 사용할 Feature 후보를 무작위 선택
```

`sqrt`는 각 Node에서 전체 Feature 수의 제곱근 정도를  
분할 후보 Feature로 사용한다.

예를 들어 Feature가 16개라면:

```text
sqrt(16) = 4
```

각 분할에서 약 4개의 Feature를 후보로 선택하여  
그중 좋은 분할 기준을 찾는다.

---

## 13. 왜 데이터와 Feature를 무작위로 선택할까?

모든 Tree가 같은 데이터와 같은 Feature를 사용하면  
서로 비슷한 Tree가 만들어질 가능성이 높다.

Random Forest는

```text
Data Randomness
    +
Feature Randomness
    ↓
서로 다른 Decision Tree
    ↓
예측 결과 종합
```

방식을 사용한다.

즉,

```text
max_samples
→ Tree마다 학습 데이터에 차이를 만듦

max_features
→ Tree의 분할 과정에도 차이를 만듦
```

이를 통해 특정 하나의 Tree에 지나치게 의존하는 문제를 줄인다.

---

## 14. max_depth

```python
max_depth=7
```

각 Decision Tree가 성장할 수 있는 최대 깊이를 제한한다.

```text
max_depth 작음
→ Tree 단순
→ Underfitting 가능

max_depth 큼
→ Tree 복잡
→ Train 데이터에 과도하게 적합될 가능성
```

Random Forest도 내부적으로 Decision Tree를 사용하기 때문에  
Tree의 깊이를 조절할 수 있다.

---

## 15. 모델 학습

```python
model.fit(X_train, y_train)
```

Random Forest는 내부적으로 여러 Decision Tree를 생성하여 학습한다.

```text
X_train / y_train
       ↓
Bootstrap Sampling
       ↓
여러 Decision Tree 학습
       ↓
Random Forest
```

---

## 16. Train / Test Accuracy

```python
train_accuracy = model.score(X_train,y_train)
test_accuracy = model.score(X_test,y_test)
```
Train과 Test 정확도를 함께 확인한다.

```text
Train ≈ Test   → 일반화 성능이 비교적 안정적
Train >> Test  → Overfitting 가능성 확인
```

Train Accuracy가 높다는 이유만으로 좋은 모델이라고 판단하면 안 된다.
새로운 데이터인 Test에서도 좋은 성능을 보이는지 확인해야 한다.

---

## 17. predict()

```python
y_pred = model.predict(X_test)
```
Test 데이터의 Class를 예측한다.

```text
Test Data
   ↓
Tree 1 → Class
Tree 2 → Class
Tree 3 → Class
...
   ↓
예측 결과 종합
   ↓
Final Class
```

---

## 18. Accuracy

```python
accuracy = accuracy_score(y_test,y_pred)
```
전체 Test 데이터 중 올바르게 예측한 비율이다.

```text
Accuracy = 올바르게 예측한 데이터 / 전체 데이터
```
Accuracy가 높을수록 전체적인 분류 정확도가 높다는 의미다.

하지만 클래스 분포나 오분류 형태에 따라  
Accuracy만으로 모델을 평가하는 것은 충분하지 않을 수 있다.

---

## 19. Classification Report

```python
classification_report(
    y_test,
    y_pred,
    target_names=["Denied", "Approved"]
)
```

주요 평가 지표:

| 지표 | 의미 |
|---|---|
| Precision | 해당 Class라고 예측한 것 중 실제 정답 비율 |
| Recall | 실제 해당 Class 중 올바르게 찾아낸 비율 |
| F1 Score | Precision과 Recall의 조화평균 |
| Support | 실제 해당 Class의 데이터 개수 |

Accuracy뿐 아니라 각 Class의 분류 성능을 확인할 수 있다.

---

## 20. Feature Importance

Random Forest에서도 각 Feature의 상대적 중요도를 확인할 수 있다.

```python
model.feature_importances_
```

Feature 이름과 결합하면:

```python
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
```

중요도가 높은 순서로 정렬:

```python
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)
```

```text
Importance 높음 → 여러 Tree의 분할 과정에서 상대적으로 많이 기여
Importance 낮음 → 상대적으로 적게 기여
```
이 역시
Feature Importance가 높다고 해서  
해당 Feature가 결과의 원인이라는 의미는 아니다.

---

## 21. Feature Importance 시각화

```python
plt.barh(importance_plot["Feature"],importance_plot["Importance"])
```

Feature별 중요도를 막대그래프로 비교한다.

```text
Feature A  ███████████
Feature B  ███████
Feature C  ████
Feature D  ██
```

그래프를 통해 어떤 Feature가 Random Forest의 예측에  
상대적으로 많이 사용되었는지 확인할 수 있다.

---

## 22. Random Forest의 핵심 구조

```text
Training Data
      ↓
Bootstrap Sampling
      ↓
┌─────────────────────────────┐
│ Tree 1 → 일부 Data + Feature │
│ Tree 2 → 일부 Data + Feature │
│ Tree 3 → 일부 Data + Feature │
│ ...                         │
│ Tree N → 일부 Data + Feature │
└─────────────────────────────┘
      ↓
각 Tree 예측
      ↓
결과 종합
      ↓
Final Prediction
```
Random Forest의 핵심은 
**여러 개의 서로 다른 Decision Tree를 만든다는 것**

---

## 23. 주요 코드

### 데이터 분리

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Random Forest

```python
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    max_samples=0.7,
    max_features="sqrt",
    random_state=42
)
```

### 학습 / 예측

```python
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### 평가

```python
accuracy_score(
    y_test,
    y_pred
)

classification_report(
    y_test,
    y_pred
)
```

### Feature Importance

```python
model.feature_importances_
```

---

## 24. Decision Tree → Random Forest

```text
Decision Tree
     ↓
하나의 Tree
     ↓
데이터 변화에 민감할 수 있음
     ↓
여러 Tree를 만들어보자
     ↓
Random Forest
     ↓
Data 무작위 선택
+
Feature 무작위 선택
     ↓
서로 다른 여러 Tree
     ↓
예측 결과 종합
```

Random Forest는 Decision Tree의 구조를 이용하면서  
여러 Tree를 결합하여 보다 안정적인 예측을 목표로 한다.

---

## 정리

- Random Forest는 여러 Decision Tree를 결합하는 Ensemble 모델이다.
- 각 Tree는 Bootstrap Sampling을 통해 서로 다른 데이터를 학습한다.
- `max_features`를 이용하여 분할에 사용할 Feature도 무작위로 선택한다.
- `n_estimators`는 생성할 Decision Tree의 개수를 결정한다.
- `max_depth`는 각 Tree의 복잡도를 제한한다.
- 여러 Tree의 결과를 종합하여 최종 Class를 예측한다.
- Train/Test Accuracy를 함께 비교하여 일반화 성능을 확인한다.
- Classification Report를 이용하여 Class별 성능을 확인할 수 있다.
- Feature Importance를 이용하여 예측에 상대적으로 많이 기여한 Feature를 확인할 수 있다.