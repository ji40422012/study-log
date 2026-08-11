# Machine Learning : Learning Roadmap
---
모델의 원리, 전처리, 평가 방법을 학습
**지도학습, 비지도학습, 추천 시스템 등**(진행중)

## 01. Supervised Learning
정답(Label)이 있는 데이터를 이용하여  
Feature와 Target의 관계를 학습하는 **지도학습**

### Regression
- Multiple Linear Regression
- Polynomial Regression

### Classification
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree (예정)
- Random Forest (예정)

```text
Supervised-Learning/
├── 01_Multiple_Linear_Regression/
├── 02_Polynomial_Regression/
├── 03_Logistic_Regression/
├── 04_KNN/
├── 05_Decision_Tree/        # 예정
└── 06_Random_Forest/        # 예정
```

#### ✅ 01. Multiple Linear Regression
- Multiple Linear Regression
- Gradient Descent
- Cost Function
- Learning Rate
- Train / Test Split
- VIF
- Cross Validation
- Ridge Regression
- Lasso Regression

---

#### ✅ 02. Polynomial Regression
- Polynomial Regression
- PolynomialFeatures
- Degree
- Underfitting / Overfitting
- Cross Validation
- Ridge Regression

---

#### ✅ 03. Logistic Regression

- Logistic Regression
- Sigmoid / Probability
- Threshold
- StandardScaler / Pipeline
- Precision / Recall / F1
- Confusion Matrix
- ROC / AUC
- Precision-Recall Curve
- Odds / Odds Ratio

---

#### ✅ 04. K-Nearest Neighbors

- K-Nearest Neighbors
- L1 / L2 Distance
- K 값에 따른 성능 변화
- StandardScaler
- Pipeline
- Cross Validation
- Confusion Matrix
- Classification Report
- Model Save / Load

---

#### ⏳ 05. Decision Tree

- Decision Tree
- Node / Branch / Leaf
- Gini Impurity
- Entropy
- Tree Depth
- Overfitting
- Feature Importance
- Pruning

---

#### ⏳ 06. Random Forest

- Random Forest
- Ensemble Learning
- Bootstrap Sampling
- Multiple Decision Trees
- Feature Randomness
- Feature Importance
- Overfitting 완화
- Hyperparameter Tuning

---

## 02. Unsupervised Learning

정답(Label) 없이 데이터에 존재하는  
패턴이나 구조를 찾는 **비지도학습**

### Clustering

- K-Means Clustering
- Cluster
- Centroid
- Distance
- Elbow Method
- Silhouette Score
- Cluster Visualization

```text
Unsupervised-Learning/
└── 01_Clustering/
```

#### ✅ 01. Clustering

- K-Means
- Cluster / Centroid
- 거리 기반 군집화
- 적절한 Cluster 수 선택
- Elbow Method
- Silhouette Score
- 군집 결과 시각화
- 군집별 특징 해석

---

## 03. Recommender System

사용자 또는 아이템의 정보를 이용하여  
사용자에게 적절한 항목을 추천하는 **추천 시스템**

### Content-Based Filtering

아이템의 특징을 이용하여  
사용자가 선호한 아이템과 유사한 아이템을 추천

```text
Recommender-System/
└── 01_Content_Based_Filtering/
```

#### ✅ 01. Content-Based Filtering

- Content-Based Filtering
- Item Feature
- Text Vectorization
- TF-IDF
- Cosine Similarity
- Similarity Matrix
- 유사 아이템 탐색
- Recommendation

---

## 구성

```text
Machine-Learning/
│
├── Supervised-Learning/
│   ├── 01_Multiple_Linear_Regression/
│   ├── 02_Polynomial_Regression/
│   ├── 03_Logistic_Regression/
│   ├── 04_KNN/
│   ├── 05_Decision_Tree/        # 예정
│   └── 06_Random_Forest/        # 예정
│
├── Unsupervised-Learning/
│   └── 01_Clustering/
│
└── Recommender-System/
    └── 01_Content_Based_Filtering/
```

---

## Learning Progress

| Category | Algorithm | Status |
|---|---|---|
| Supervised Learning | Multiple Linear Regression | ✅ 완료 |
| Supervised Learning | Polynomial Regression | ✅ 완료 |
| Supervised Learning | Logistic Regression | ✅ 완료 |
| Supervised Learning | K-Nearest Neighbors | ✅ 완료 |
| Supervised Learning | Decision Tree | ⏳ 예정 |
| Supervised Learning | Random Forest | ⏳ 예정 |
| Unsupervised Learning | Clustering | ✅ 학습 |
| Recommender System | Content-Based Filtering | ✅ 학습 |

---

## Skills

`Python` `NumPy` `Pandas` `Matplotlib` `Scikit-learn`

`Regression` `Classification` `Clustering` `Recommendation`

`Feature Scaling` `Pipeline` `Cross Validation` `Model Evaluation`
