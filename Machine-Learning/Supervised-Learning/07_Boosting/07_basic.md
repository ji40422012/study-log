# Gradient Boosting

여러 Decision Tree를 **순차적으로 학습**하면서  
이전 모델의 부족한 부분을 다음 Tree가 보완하는 Ensemble 모델

- Gradient Boosting
- Sequential Learning
- `GradientBoostingClassifier`
- `n_estimators`
- `learning_rate`
- `max_depth`
- Cross Validation

---

## Gradient Boosting

Random Forest와 Gradient Boosting은 
여러 Decision Tree를 사용하는 Ensemble 모델이지만 학습 방식이 다르다.

```text
Random Forest
여러 Tree를 독립적으로 학습 → 결과를 종합 → Bagging

Gradient Boosting
Tree를 순차적으로 학습 → 이전 모델의 부족한 부분을 다음 Tree가 보완→ Boosting
```

Gradient Boosting의 흐름:

```text
Tree 1
  ↓
부족한 부분 확인
  ↓
Tree 2
  ↓
보완
  ↓
Tree 3
  ↓
...
  ↓
Final Prediction
```

---

## `GradientBoostingClassifier`

```python
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=1
)
```

| 옵션 | 설명 |
|---|---|
| `n_estimators` | 순차적으로 학습할 Tree 개수 |
| `learning_rate` | 각 Tree의 기여도를 조절 |
| `max_depth` | 개별 Tree의 최대 깊이 |
| `random_state` | 난수 결과 고정 |

### `n_estimators`와 `learning_rate`

두 값은 함께 고려하는 것이 중요하다.

```text
learning_rate 작음 → 각 Tree의 영향이 작음 → 더 많은 Tree가 필요할 수 있음
learning_rate 큼 → 각 Tree의 영향이 큼 → 빠르게 학습하지만 과적합에 주의
```
`max_depth`가 커질수록 개별 Tree가 복잡해지므로 과적합 가능성이 증가할 수 있다.

---

## Cross Validation
Pima Indians Diabetes Dataset을 이용하여  
5-Fold Cross Validation으로 Gradient Boosting의 성능을 확인

```python
scores = cross_val_score(model,X,y,cv=cv,scoring="accuracy")
print("5-Fold Accuracy:", np.round(scores, 3))
print(f"평균 Accuracy: {scores.mean():.3f}")
```
---

## `learning_rate` 비교
`n_estimators`와 `max_depth`를 고정하고 `learning_rate`만 변경하여 성능 변화를 확인

```python
learning_rates = [0.01, 0.05, 0.1, 0.2]

for rate in learning_rates:
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=rate,
        max_depth=3,
        random_state=1
    )
    scores = cross_val_score(model,X,y,cv=cv,scoring="accuracy")
    print(f"learning_rate={rate:<4} " f"평균 Accuracy={scores.mean():.3f}")
```
단순히 값이 클수록 좋은 것이 아니라  
Cross Validation 성능을 기준으로 적절한 값을 선택

---

## 정리

- Gradient Boosting은 여러 Decision Tree를 **순차적으로 학습**한다.
- 이전 모델의 부족한 부분을 다음 Tree가 보완한다.
- Random Forest는 **Bagging**, Gradient Boosting은 **Boosting** 계열이다.
- `n_estimators`는 Tree 개수, `learning_rate`는 각 Tree의 기여도를 조절한다.
- `learning_rate`와 `n_estimators`는 서로 연관되어 있다.
- `max_depth`는 개별 Tree의 복잡도를 조절한다.
- Cross Validation으로 모델의 일반화 성능을 확인한다.
