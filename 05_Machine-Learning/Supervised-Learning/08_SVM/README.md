# Support Vector Machine

클래스 사이의 **Margin을 최대화하는 결정경계** 찾기
(Margin을 최대화하는 결정경계를 이용한 분류)

## Learning

### Basic

* Support Vector / Margin
* StandardScaler
* Linear / RBF Kernel
* C / gamma
* GridSearchCV
* Model Evaluation

### Decision Boundary

* Linear SVM
* Non-linear Data
* Feature Mapping
* RBF Kernel
* Kernel Trick
* Linear vs RBF

## Structure

```text
08_SVM/
├── 01_Basic/
│   ├── 08_svm.py
│   └── 08_svm.md
│
├── 02_Decision_Boundary/
│   ├── 08_svm_boundary.py
│   ├── 08_svm_boundary.md
│   ├── 08_linear_boundary.png
│   ├── 08_feature_mapping_3d.png
│   ├── 08_circular_boundary.png
│   ├── 08_rbf_boundary.png
│   └── 08_moons_kernel_comparison.png
│
└── README.md
```

* **Kernel** : 선형적으로 분리하기 어려운 데이터의 비선형 분류
* **GridSearchCV** : `C`, `gamma`, `kernel`의 최적 조합 탐색
