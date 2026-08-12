## 1. K-Nearest Neighbors (KNN)

```text
새로운 데이터 → 가까운 K개 탐색 → 다수결(Voting) → Class 결정
```
KNN은 별도의 회귀계수를 학습하지 않고, 
학습 데이터를 저장한 뒤 새로운 데이터가 들어오면 거리를 계산해 예측하는 **거리 기반 알고리즘**

---

## 2. Iris Dataset
scikit-learn의 `load_iris()` 데이터 사용

| 항목 | 내용 |
|---|---|
| 데이터 수 | 150 |
| Feature | 4 |
| Class | 3 |
| Class 0 | setosa |
| Class 1 | versicolor |
| Class 2 | virginica |

```text
데이터 크기 : (150, 4)
Feature 수 : 4
클래스 : ['setosa' 'versicolor' 'virginica']
```

### Iris 데이터 분포

![Iris Scatter Plot](iris_scatter.png)

꽃받침 길이와 너비를 기준으로 보면 
`setosa`는 비교적 잘 구분되며, `versicolor`와 `virginica`는 일부 영역에서 겹침

---

## 3. 거리(Distance)
KNN은 데이터 사이의 거리를 계산하여 
가까운 이웃(Nearest Neighbors)을 찾는다

| 거리 | 의미 |
|---|---|
| L1 / Manhattan | 각 좌표 차이의 절댓값 합 |
| L2 / Euclidean | 두 점 사이의 직선거리 |

이번 예제의 두 점이 `A=(1,1)`, `B=(5,4)`일 때:

```text
Manhattan Distance : 7.00
Euclidean Distance : 5.00
```

![L1 vs L2 Distance](distance.png)

scikit-learn의 KNN은 기본적으로 **Minkowski 거리에서 `p=2`**, 즉 Euclidean Distance를 사용

---

## 4. Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

| 옵션 | 의미 |
|---|---|
| `test_size=0.2` | 전체 데이터의 20%를 Test로 사용 |
| `random_state=42` | 데이터 분할 결과 고정 |
| `stratify=y` | 클래스 비율 유지 |

```text
Train : (120, 4)
Test  : (30, 4)
```

---

## 5. KNN 모델

```python
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```
`n_neighbors`  == 예측할 때 참고할 **가까운 이웃의 수(k)** 

`fit()`은 다른 모델처럼 회귀계수를 계산하기보다는 예측에 사용할 학습 데이터를 저장하고, 
`predict()` 단계에서 새로운 데이터와의 거리를 계산

---

## 6. k 값

| k | 특징 |
|---|---|
| 작음 | 주변 데이터에 민감 → Overfitting 가능 |
| 적절함 | 지역적인 패턴을 적절히 반영 |
| 큼 | 너무 많은 이웃 참고 → Underfitting 가능 |

따라서 여러 `k`를 비교하여 적절한 값을 선택

### k별 Accuracy

![Accuracy according to k](k_accuracy.png)

예제 실행 결과:

```text
===== Best k =====
최적 k : 3
최고 Accuracy : 1.000
```

이번 데이터 분할에서는 `k=3`에서 높은 성능이 나타났지만, 
**한 번의 Test 결과만 보고 최적 k를 결정하는 것은 적절하지 않다.**
보통은 Cross Validation을 이용하여 더 안정적으로 `k`를 선택

---

## 7. Accuracy와 Confusion Matrix

### Accuracy

전체 데이터 중 올바르게 분류한 비율로 **높을수록 좋다.**

```python
accuracy_score(y_test, y_pred)
```

### Confusion Matrix
실제 클래스와 예측 클래스를 비교하여 **어떤 품종을 잘못 분류했는지 확인**

```python
confusion_matrix(y_test, y_pred)
```
Iris는 클래스가 3개이므로 결과도 `3 × 3` 행렬

---

## 8. 새로운 꽃 예측

```python
new_flower = [[5.0, 3.5, 1.5, 0.2]]

prediction = best_model.predict(new_flower)
```
새로운 꽃의 네 가지 Feature를 입력하면 
가장 가까운 이웃들을 확인하여 품종을 예측

```text
예측 Class : 0
예측 품종 : setosa
```

---

## 9. 주요 코드

| 코드 | 역할 |
|---|---|
| `load_iris()` | Iris 데이터 불러오기 |
| `train_test_split()` | Train/Test 분리 |
| `np.linalg.norm()` | 거리 계산 |
| `KNeighborsClassifier()` | KNN 모델 생성 |
| `fit()` | 학습 데이터 저장 |
| `predict()` | 클래스 예측 |
| `score()` | Accuracy 계산 |
| `accuracy_score()` | 정확도 계산 |
| `confusion_matrix()` | 오분류 확인 |

---

## 10. 핵심 정리

- KNN은 **가까운 이웃을 기준으로 분류**
- 예측 결과는 `k` 값에 영향
- L1, L2 등의 거리 기준을 사용할 수 있다.
- KNN은 거리 기반 모델이므로 **Feature Scale의 영향을 크게 받는다**
