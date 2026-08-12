# Supervised Learning
정답(Label)이 있는 데이터를 이용하여 
Feature와 Target의 관계를 학습하는 **지도학습(Supervised Learning)** 

지도학습 == **회귀(Regression)** + **분류(Classification)** 

- **Regression** : 연속적인 값 예측
- **Classification** : 데이터의 클래스 예측

---

## 01. Multiple Linear Regression
여러 Feature를 이용하여 연속적인 값을 예측하는 다중선형회귀
- Linear Regression
- 회귀계수와 절편
- MSE / RMSE / R²
- 모델 결과 해석

## 02. Polynomial Regression
다항 특성을 이용하여 비선형 관계를 표현하는 회귀 모델
- Polynomial Features
- Degree
- Underfitting / Overfitting
- Pipeline
- Residual
- Regularization

## 03. Logistic Regression
확률을 이용하여 클래스를 예측하는 분류 모델
- Sigmoid / Probability
- Threshold
- StandardScaler / Pipeline
- Precision / Recall / F1
- Confusion Matrix
- ROC-AUC / PR Curve

## 04. K-Nearest Neighbors
가까운 데이터의 클래스를 이용하여 예측하는 거리 기반 모델
- KNN / Distance
- k
- Feature Scaling
- Pipeline
- Cross Validation
- Classification Report
- Confusion Matrix

## 05. Decision Tree
> 예정
Feature를 기준으로 데이터를 반복적으로 분할하여 예측하는 Tree 기반 모델

## 06. Random Forest
> 예정
여러 Decision Tree를 결합하여 예측하는 Ensemble 모델

---

## 구성

```text
Basic
  ↓
알고리즘 기본 원리
  ↓
모델 학습 / 예측
  ↓
Practical
  ↓
전처리 / 모델 평가 / 개선
```

---

## Learning Progress

| Algorithm | Type | Status |
|---|---|---|
| Multiple Linear Regression | Regression | 완료 |
| Polynomial Regression | Regression | 완료 |
| Logistic Regression | Classification | 완료 |
| K-Nearest Neighbors | Classification | 완료 |
| Decision Tree | Classification / Regression | 학습 예정 |
| Random Forest | Classification / Regression | 학습 예정 |

---

## Skills

`Python` `NumPy` `Pandas` `Matplotlib` `Scikit-learn`

`Regression` `Classification` `Feature Scaling` `Pipeline` `Cross Validation` `Model Evaluation`
