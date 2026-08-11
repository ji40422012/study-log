# Decision Tree (Basic)
Feature를 기준으로 데이터를 반복적으로 분할하여  
최종적으로 Class를 예측하는 **Tree 기반 지도학습 모델**

`Playing Golf` 데이터와 scikit-learn의 `Iris Dataset`을 이용하여  
Decision Tree의 기본 구조, 과적합, 가지치기, Feature Importance 학습

---

## 1. Decision Tree

Decision Tree는 
특정 Feature와 조건을 기준으로 데이터를 반복적으로 분할

```text
전체 데이터
    ↓
Feature 조건으로 분할
    ↓
하위 데이터 분할
    ↓
반복
    ↓
최종 Class 예측
```

예를 들어:

```text
날씨가 맑은가?
   ├── Yes → 습도가 높은가?
   │          ├── Yes → 골프 안 침
   │          └── No  → 골프 침
   │
   └── No  → ...
```
복잡한 수식보다 **조건을 이용한 의사결정 규칙**으로 결과를 해석할 수 있다는 장점

---

## 2. Tree 구조
Decision Tree는 주로 다음 요소로 구성된다.

| 용어 | 의미 |
|---|---|
| Root Node | 가장 처음 분할되는 노드 |
| Internal Node | 중간에서 조건을 이용해 데이터를 분할하는 노드 |
| Branch | 노드 사이의 연결 |
| Leaf Node | 최종 예측 결과를 나타내는 노드 |
| Depth | Root에서 Leaf까지 내려가는 깊이 |

```text
           Root
          /    \
      Node      Node
      /  \      /  \
   Leaf Leaf  Leaf Leaf
```

Tree가 깊어질수록 더 복잡한 규칙을 만들 수 있다.

---

## 3. Playing Golf Dataset

```python
golf = pd.read_csv("../datasets/decision/playing_golf.csv")
```
날씨 등의 조건을 이용하여 골프를 칠지 여부를 예측하는 데이터

Target:

```text
play_yes = 1 → 골프 침
play_yes = 0 → 골프 안 침
```

---

## 4. One-Hot Encoding
데이터에 문자열 형태의 범주형 변수가 있으므로 숫자 형태로 변환

```python
encoded = pd.get_dummies(golf, dtype=int)
```

예를 들어:

```text
outlook
  ↓
outlook_overcast
outlook_rainy
outlook_sunny
```
각 범주를 별도의 `0 / 1` Feature로 변환

---

## 5. Feature / Target

```python
X = encoded.drop(columns=["play_no", "play_yes"])
y = encoded["play_yes"]
```

```text
X → 예측에 사용하는 Feature
y → 모델이 예측해야 하는 Target
```
이번 모델의 Target은 `play_yes`이다.

---

## 6. DecisionTreeClassifier

```python
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X, y)
```
`DecisionTreeClassifier` == 분류 문제에 사용하는 Decision Tree 모델
`fit()`을 실행하면 
모델은 Feature를 이용하여 데이터를 효과적으로 나눌 수 있는 분할 조건을 반복적으로 찾는다

### 주요 옵션

| 옵션 | 의미 |
|---|---|
| `criterion` | 데이터를 나누는 기준 |
| `max_depth` | Tree 최대 깊이 |
| `min_samples_split` | 노드를 분할하기 위한 최소 데이터 수 |
| `min_samples_leaf` | Leaf에 필요한 최소 데이터 수 |
| `random_state` | 난수 결과 고정 |

---

## 7. Tree 시각화

```python
plot_tree(
    tree,
    feature_names=list(X.columns),
    class_names=["안침", "침"],
    filled=True,
    rounded=True
)
```

```text
상위 Node
    ↓
분할 조건
    ↓
True / False
    ↓
하위 Node
    ↓
최종 Class
```
**모델이 어떤 조건을 이용하여 판단했는지 해석하기 쉽다**

---

## 8. Iris Dataset
scikit-learn의 Iris Dataset

```python
iris = load_iris()
X = iris.data; y = iris.target
```
---

## 9. Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=2
)
```
전체 데이터를 학습 데이터와 테스트 데이터로 나눈다.

```text
Train → 모델 학습
Test  → 새로운 데이터에 대한 성능 평가
```
`test_size=0.3`은 전체 데이터의 30%를 Test 데이터로 사용한다는 의미이다.

---

## 10. Tree Depth와 과적합
깊이에 제한을 두지 않고 Tree를 생성해보기

```python
deep_tree = DecisionTreeClassifier(random_state=3)
deep_tree.fit(X_train, y_train)
```

```text
Tree Depth 증가
      ↓
복잡한 규칙 학습
      ↓
Train 데이터에 매우 잘 맞음
      ↓
새로운 데이터에서 성능 저하 가능
      ↓
Overfitting
```

**Train Accuracy가 높다고 반드시 좋은 모델은 아니다.**
Train과 Test 성능을 함께 확인

---

## 11. max_depth

```python
DecisionTreeClassifier(
    max_depth=3,
    random_state=2
)
```

`max_depth`는 Tree가 내려갈 수 있는 최대 깊이를 제한

```text
max_depth 작음
→ 단순한 Tree
→ 과소적합 가능

max_depth 적절
→ 중요한 패턴 학습
→ 일반화 성능 향상 가능

max_depth 너무 큼
→ 복잡한 Tree
→ 과적합 가능
```

---

## 12. 가지치기(Pruning)
Tree가 지나치게 복잡해지는 것을 제한하는 것

```python
pruned_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=2
)
```

목표는 Train 데이터를 완벽하게 맞추는 것이 아니라 
**새로운 데이터에서도 안정적으로 예측하는 모델**을 만드는 것

---

## 13. Feature Importance
학습 후 각 Feature가 분할에 얼마나 기여했는지 확인 

```python
pruned_tree.feature_importances_
```

```python
importance = pd.Series(
    pruned_tree.feature_importances_,
    index=iris.feature_names
).sort_values()
```

```text
중요도 높음→ Tree의 분할 과정에서 상대적으로 많이 기여
중요도 낮음→ 상대적으로 적게 기여
```
단, **Feature Importance가 높다고 
해당 Feature가 결과의 원인이라는 의미는 아니다.**

---

## 14. 학습 흐름

```text
Playing Golf Dataset
        ↓
One-Hot Encoding
        ↓
Decision Tree 학습
        ↓
Tree 구조 시각화
        ↓
Iris Dataset
        ↓
Train / Test Split
        ↓
깊이 제한 없는 Tree
        ↓
Train / Test Accuracy 비교
        ↓
max_depth 비교
        ↓
Pruning
        ↓
Feature Importance
```

---

## 15. 주요 코드

### 모델 생성 / 학습

```python
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X, y)
```

### Tree 시각화

```python
plot_tree(
    tree,
    feature_names=list(X.columns),
    class_names=["안침", "침"],
    filled=True,
    rounded=True
)
```

### 깊이 제한

```python
pruned_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=2
)
```

### 정확도

```python
train_accuracy = pruned_tree.score(X_train, y_train)
test_accuracy = pruned_tree.score(X_test, y_test)
```

### Feature Importance

```python
pruned_tree.feature_importances_
```

---

## 정리

- Decision Tree는 **Feature 조건을 기준으로 데이터를 반복적으로 분할**한다.
- Tree 구조를 시각화할 수 있어 모델의 판단 과정을 비교적 쉽게 확인할 수 있다.
- 범주형 데이터는 One-Hot Encoding을 이용하여 숫자로 변환할 수 있다.
- Tree가 너무 깊어지면 Train 데이터에 **과적합**될 수 있다.
- `max_depth`를 이용하여 Tree의 복잡도를 제한할 수 있다.
- Train/Test Accuracy를 함께 비교하여 일반화 성능을 확인한다.
- `feature_importances_`를 이용하여 각 Feature의 상대적 중요도를 확인할 수 있다.
