# Decision Tree

Feature를 기준으로 데이터를 반복적으로 분할하여  
Class를 예측하는 **Tree 기반 지도학습 모델**

Decision Tree의 기본 구조부터 Tree Depth, 과적합, 가지치기와  
실제 Loan Dataset을 이용한 분류 모델 학습

---

## 학습 내용

### Basic
`Playing Golf Dataset`과 `Iris Dataset`을 이용하여 Decision Tree의 기본 원리 학습

- Decision Tree 구조
- One-Hot Encoding
- `DecisionTreeClassifier`
- Tree 시각화
- Tree Depth
- Underfitting / Overfitting
- Pruning
- Feature Importance

### Practical
`Loan Approval Dataset`을 이용하여 실제 대출 승인 분류 모델

- 범주형 데이터 전처리
- Train / Test Split
- Entropy / Information Gain
- `max_depth`
- Accuracy
- Precision / Recall / F1
- Classification Report
- Confusion Matrix
- Feature Importance
- Model Save / Load

---

## 구성

```text
05_Decision_Tree/
├── README.md
├── 05_basic.ipynb
├── 05_basic.py
├── 05_basic.md
├── 05_practical.py
├── 05_practical.md
├── decision_tree_confusion_matrix.png
├── decision_tree_loan.png
└── decision_tree_feature_importance.png
```

---

## 핵심 흐름

```text
데이터
  ↓
범주형 데이터 전처리
  ↓
Train / Test Split
  ↓
Decision Tree
  ↓
Feature 기준 데이터 분할
  ↓
Tree Depth 조절
  ↓
모델 평가
  ↓
Feature Importance
```

---

## Skills

`Python` `Pandas` `Matplotlib` `Scikit-learn` `Joblib`

`Decision Tree` `One-Hot Encoding` `Entropy` `Information Gain` `Pruning` `Feature Importance` `Model Evaluation`
