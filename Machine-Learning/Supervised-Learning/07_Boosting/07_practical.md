# Gradient Boosting Practical

Pima Indians Diabetes Dataset을 이용하여  
Gradient Boosting의 주요 Hyperparameter에 따른 성능 변화를 비교하고 최적 모델을 평가

- `learning_rate` 비교
- `n_estimators` 비교
- Train / Test Accuracy
- GridSearchCV
- Confusion Matrix
- Precision / Recall / F1-score
- False Negative
- Feature Importance

---

## 데이터 분리

```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=1,stratify=y)
```
`stratify=y`를 사용하여 Train/Test 데이터에서 정상과 당뇨 클래스의 비율을 유지

---

## `learning_rate` 비교

```python
learning_rates = [0.01, 0.05, 0.1, 0.2]
```

`n_estimators=100`, `max_depth=3`으로 고정하고  
`learning_rate`만 변경하여 Train/Test Accuracy를 비교

```text
learning_rate 작음→ 각 Tree의 영향이 작음→ 천천히 학습
learning_rate 큼→ 각 Tree의 영향이 큼→ 빠르게 학습→ 과적합 가능성 확인
```

Train Accuracy만 높아지는 것이 아니라  
**Test Accuracy도 함께 확인하는 것이 중요**

---

## `n_estimators` 비교

```python
n_estimators_list = [50, 100, 200, 300]
```

이번에는 `learning_rate=0.05`로 고정하고 Tree 개수를 변경

```text
n_estimators 증가→ 더 많은 Tree를 순차적으로 학습→ 모델의 학습 능력 증가→ 계산량 증가→ 과적합 가능성 확인
```
Train Accuracy가 계속 증가하더라도  
Test Accuracy가 정체되거나 감소하면 과적합을 의심할 수 있다.

---

## Train / Test Accuracy

Hyperparameter에 따른 성능을 그래프로 비교

```python
plt.plot(learning_rates, train_scores, marker="o", label="Train Accuracy")
plt.plot(learning_rates, test_scores, marker="^", label="Test Accuracy")
```
```text
Train ↑
Test  ↑
→ 학습 성능과 일반화 성능 모두 개선

Train ↑
Test  정체 또는 ↓
→ 과적합 가능성
```

---

## GridSearchCV
`learning_rate`, `n_estimators`, `max_depth`의 여러 조합을 Cross Validation으로 비교

```python
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [1, 2, 3]
}
```
총 조합: 3 × 3 × 3= 27개
각 조합을 5-Fold Cross Validation으로 평가하여 최적 설정을 찾는다.

```python
grid.fit(X_train, y_train)
```
중요한 점은 **전체 데이터가 아니라 Train 데이터에서만 Hyperparameter를 탐색**한다는 것

```text
Train
  ↓
GridSearchCV
  ↓
Best Model
  ↓
Test
  ↓
Final Evaluation
```

---

## 최적 모델

```python
best_model = grid.best_estimator_
```

```python
grid.best_score_
grid.best_params_
```
- `best_score_` : 가장 높은 평균 Cross Validation 성능
- `best_params_` : 최적 Hyperparameter 조합
- `best_estimator_` : 최적 설정이 적용된 모델

최적 모델을 이용하여 Test 데이터를 최종 평가

```python
y_pred = best_model.predict(X_test)
```

---

## Confusion Matrix

```python
cm = confusion_matrix(y_test,y_pred)
tn, fp, fn, tp = cm.ravel()
```

```text
TN : 실제 정상 → 정상 예측
FP : 실제 정상 → 당뇨 예측
FN : 실제 당뇨 → 정상 예측
TP : 실제 당뇨 → 당뇨 예측
```
당뇨 예측에서는 특히 **FN(False Negative)**을 주의해서 확인

---

## Classification Report

```python
classification_report(y_test, y_pred,target_names=["정상", "당뇨"])
```

| 지표 | 의미 |
|---|---|
| Precision | 당뇨라고 예측한 데이터 중 실제 당뇨 비율 |
| Recall | 실제 당뇨 중 모델이 찾아낸 비율 |
| F1-score | Precision과 Recall을 함께 고려 |
| Support | 실제 데이터 개수 |

당뇨 클래스의 Recall이 낮다면 실제 당뇨 환자를 정상으로 판단하는 FN이 많다는 의미

---

## Feature Importance

```python
importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values()
```
Gradient Boosting에서 각 Feature가 Tree의 분할 과정에  
상대적으로 얼마나 기여했는지 확인

```text
Importance 높음→ 모델 예측에 상대적으로 많이 사용된 Feature
Importance 낮음→ 상대적으로 적게 사용된 Feature
```

---

## Practical 학습 흐름

```text
Pima Indians Diabetes
        ↓
Train / Test Split
        ↓
learning_rate 비교
        ↓
n_estimators 비교
        ↓
Train / Test Accuracy
        ↓
GridSearchCV
        ↓
Best Gradient Boosting
        ↓
Final Test
        ↓
Confusion Matrix
        ↓
Recall / FN
        ↓
Feature Importance
```

---

## 정리

- `learning_rate`는 각 Tree의 기여도를 조절한다.
- `n_estimators`는 순차적으로 학습할 Tree 개수를 결정한다.
- 두 Hyperparameter의 변화에 따라 Train/Test 성능이 달라진다.
- Train 성능만 높고 Test 성능이 개선되지 않으면 과적합을 확인해야 한다.
- GridSearchCV는 **Train 데이터 안에서** 수행하고 Test 데이터는 최종 평가에 사용한다.
- 분류 모델은 Accuracy뿐 아니라 Precision, Recall, F1-score를 함께 확인한다.
- 당뇨 예측에서는 실제 당뇨 환자를 놓치는 **FN과 Recall을 특히 주의해서 확인**한다.
- Feature Importance를 통해 모델에서 상대적으로 중요한 Feature를 확인할 수 있다.

---
