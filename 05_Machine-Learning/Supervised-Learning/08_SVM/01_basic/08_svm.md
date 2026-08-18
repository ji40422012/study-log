# Support Vector Machine

SVM(Support Vector Machine)은 클래스 사이의 **결정경계(Decision Boundary)**를 찾는 지도학습 알고리즘

Wine 데이터를 이용하여 **표준화 → SVM 학습 → 하이퍼파라미터 튜닝 → 모델 평가** 과정을 학습

## 1. SVM

SVM은 데이터를 분류하는 결정경계를 찾을 때 클래스 사이의 **Margin을 최대화하는 방향**으로 경계를 결정한다.

* **Decision Boundary** : 클래스를 구분하는 결정경계
* **Margin** : 결정경계와 가까운 데이터 사이의 여백
* **Support Vector** : 결정경계에 가까우며 경계를 결정하는 데 중요한 데이터

Margin이 넓은 결정경계를 찾는 것이 SVM의 핵심 컨셉

## 2. Wine Dataset(Scikit-learn 내장) 

```python
from sklearn.datasets import load_wine

wine = load_wine()
X, y = wine.data, wine.target
```

Wine 데이터는 여러 Feature를 이용해 와인의 품종을 분류하는 다중 클래스 분류 데이터이다.
Train/Test 데이터에서도 각 품종의 비율이 유지되도록 `stratify=y`를 사용했다.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y  # 각 품종의 비율을 유지하면서 Train/Test 분리
)
```

## 3. StandardScaler
SVM은 Feature의 크기 차이에 영향을 받을 수 있기 때문에 표준화를 적용했다.

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),  # Feature를 평균 0, 표준편차 1로 표준화
    ("svc", SVC())                 # Support Vector Classifier
])
```

`Pipeline`을 사용하면 표준화와 SVM 학습 과정을 하나로 연결할 수 있다.

```text
Data
 ↓
StandardScaler
 ↓
SVC
```

## 4. Hyperparameter

SVM에서 주요 하이퍼파라미터인 `C`, `gamma`, `kernel`을 비교

### C

`C`는 오분류에 얼마나 강하게 페널티를 줄 것인지 조절

* `C`가 작음 → 일부 오분류를 허용하고 넓은 Margin을 선호
* `C`가 큼 → 훈련 데이터를 더 정확하게 분류하려는 경향

`C`가 지나치게 크면 모델이 훈련 데이터에 과도하게 맞춰질 가능성이 있다.

### Kernel

```python
"svc__kernel": ["linear", "rbf"]
```

* `linear` : 선형 결정경계를 사용
* `rbf` : 비선형 데이터에서도 복잡한 결정경계를 만들 수 있음

Linear와 RBF의 결정경계 차이는 `02_Decision_Boundary`에서 추가로 확인한다.

### gamma

`gamma`는 RBF Kernel에서 각 데이터가 결정경계에 영향을 미치는 범위를 조절한다.

* `gamma`가 작음 → 영향 범위가 넓어 비교적 완만한 결정경계
* `gamma`가 큼 → 영향 범위가 좁아 복잡한 결정경계가 만들어질 수 있음

## 5. GridSearchCV

여러 하이퍼파라미터 조합을 직접 하나씩 학습하지 않고 `GridSearchCV`를 이용해 최적의 조합을 탐색

```python
param_grid = {
    "svc__C": [0.1, 1, 10, 100],            # 작을수록 오차를 더 허용하고 Margin을 넓게 유지
    "svc__gamma": ["scale", 0.01, 0.1, 1],  # RBF Kernel에서 각 데이터의 영향 범위
    "svc__kernel": ["linear", "rbf"]         # Linear / RBF Kernel 비교
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,                # 5-Fold Cross Validation
    scoring="accuracy",  # Accuracy를 기준으로 최적 모델 선택
    n_jobs=-1
)
```

`cv=5`를 사용하여 5-Fold Cross Validation으로 각 하이퍼파라미터 조합의 성능을 비교했다.

학습 후 다음 값으로 최적 모델을 확인할 수 있다.

```python
print("가장 좋은 설정:", grid.best_params_)
print("교차검증 정확도:", round(grid.best_score_, 3))
```

## 6. Model Evaluation

GridSearchCV에서 선택된 최적 모델로 Test 데이터를 예측했다.

```python
y_pred = grid.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)  # Accuracy = 전체 데이터 중 맞춘 비율
```

Accuracy뿐만 아니라 Confusion Matrix와 Classification Report를 이용해 클래스별 분류 결과도 확인했다.

```python
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=wine.target_names))
```

### Confusion Matrix

Confusion Matrix를 이용하면 실제 클래스와 모델이 예측한 클래스를 비교하여 **어떤 품종을 서로 잘못 분류했는지** 확인할 수 있다.

### Classification Report

Classification Report에서는 클래스별로 다음 평가 지표를 확인할 수 있다.

* **Precision** : 해당 클래스라고 예측한 데이터 중 실제 해당 클래스인 비율
* **Recall** : 실제 해당 클래스 데이터 중 모델이 찾아낸 비율
* **F1-score** : Precision과 Recall의 조화평균

## 정리

* SVM은 클래스 사이의 **Margin을 최대화하는 결정경계**를 찾는다.
* 결정경계에 가까운 중요한 데이터를 **Support Vector**라고 한다.
* SVM은 Feature 크기에 영향을 받을 수 있어 `StandardScaler`를 함께 사용할 수 있다.
* `C`는 오분류 허용 정도와 Margin에 영향을 준다.
* `kernel`을 이용해 Linear / RBF 방식을 선택할 수 있다.
* `gamma`는 RBF Kernel의 결정경계 형태에 영향을 준다.
* `GridSearchCV`를 이용해 여러 하이퍼파라미터 조합을 비교할 수 있다.
* 최종 모델은 Test 데이터와 Confusion Matrix, Classification Report를 이용해 평가한다.
