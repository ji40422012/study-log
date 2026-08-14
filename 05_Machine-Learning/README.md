# Machine Learning : Learning Roadmap
---
데이터의 패턴을 학습하여 예측하거나 구조를 발견하는 **Machine Learning**

```text
Machine Learning
├── Supervised Learning
│   ├── Regression
│   ├── Classification
│   └── Model Evaluation
│
└── Unsupervised Learning
    ├── Clustering
    └── Association Rule
    ├── Anomaly Detection
    └── Dimensionality Reduction

## 01. Supervised Learning
정답(Label)이 있는 데이터를 이용하여  
Feature와 Target의 관계를 학습하는 **지도학습**

### Regression
- Multiple Linear Regression
- Polynomial Regression

### Classification

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting

### Model Evaluation

- Cross Validation
- Confusion Matrix
- Precision / Recall / F1-score
- Hyperparameter Tuning
- Model Comparison

```text
Supervised-Learning/
├── 01_Multiple_Linear_Regression/
├── 02_Polynomial_Regression/
├── 03_Logistic_Regression/
├── 04_KNN/
├── 05_Decision_Tree/
├── 06_Random_Forest/
├── 07_Boosting/
└── 08_Model_Comparison/
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

#### ✅ 05. Decision Tree

- Decision Tree
- Node / Branch / Leaf
- Gini Impurity
- Entropy
- Tree Depth
- Overfitting
- Feature Importance
- Pruning

---

#### ✅ 06. Random Forest

- Random Forest
- Ensemble Learning
- Bootstrap Sampling
- Multiple Decision Trees
- Feature Randomness
- Feature Importance
- Overfitting 완화
- Hyperparameter Tuning

---

#### ✅ 07. Boosting

- Gradient Boosting
- Sequential Learning
- Weak Learner
- Learning Rate
- n_estimators
- Random Forest와의 차이

---

#### ✅ 08. Model Comparison

- Decision Tree
- Random Forest
- Gradient Boosting
- Tuned Random Forest
- Cross Validation
- Confusion Matrix
- False Negative
- Classification Report
- 모델 성능 비교

---

## 02. Unsupervised Learning

정답(Label) 없이 데이터에 존재하는  
패턴이나 구조를 찾는 **비지도학습**

```text
Unsupervised-Learning/
├── 01_Clustering/
└── 02_Association_Rule/
├── Anomaly Detection       
└── Dimensionality Reduction 
```

#### ✅ 01. Clustering

- K-Means Clustering
- Cluster / Centroid
- Inertia
- Elbow Method
- Cluster Visualization
- StandardScaler
- DBSCAN
- eps / min_samples
- Noise Detection

---


#### ✅ 02. Association Rule

- Transaction Data
- Apriori Algorithm
- Frequent Itemsets
- Association Rules
- Support
- Confidence
- Lift
- 구매 패턴 분석

---

#### ⏳ 03. Anomaly Detection



---

#### ⏳ 04. Dimensionality Reduction



---


## Skills

`Python` `NumPy` `Pandas` `Matplotlib` `Scikit-learn`  
`Regression` `Classification` `Ensemble Learning` `Clustering`  
`Cross Validation` `Model Evaluation` `Hyperparameter Tuning`
