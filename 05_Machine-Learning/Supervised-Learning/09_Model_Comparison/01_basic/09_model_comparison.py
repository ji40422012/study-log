# =========================================================
# Model Comparison - Basic
# Pima Indians Diabetes Dataset
# Decision Tree / Random Forest / Gradient Boosting
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (train_test_split,StratifiedKFold,cross_val_score, GridSearchCV)
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report)

# 1. 데이터 불러오기
cols = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv("../datasets/pima-indians-diabetes.csv", names=cols)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]
print("===== Pima Indians Diabetes Dataset =====")
print(df.head()); print("데이터 크기:", df.shape)

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=1,stratify=y)
print("\nTrain:", X_train.shape); print("Test :", X_test.shape)

# 3. Cross Validation 설정
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)

# 4. Decision Tree
dt = DecisionTreeClassifier(max_depth=5,min_samples_split=10,min_samples_leaf=5,random_state=1)
dt_cv = cross_val_score(dt,X,y,cv=cv,scoring="accuracy")
print("\n===== Decision Tree =====");print("CV Scores:", np.round(dt_cv, 3))
print(f"Mean Accuracy: {dt_cv.mean():.3f}")

# 5. Random Forest
rf = RandomForestClassifier(n_estimators=300, random_state=1)
rf_cv = cross_val_score(rf,X,y,cv=cv,scoring="accuracy")
print("\n===== Random Forest =====")
print("CV Scores:", np.round(rf_cv, 3))
print(f"Mean Accuracy: {rf_cv.mean():.3f}")

# 6. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3, random_state=1)
gb_cv = cross_val_score(gb,X,y,cv=cv,scoring="accuracy")
print("\n===== Gradient Boosting =====")
print("CV Scores:", np.round(gb_cv, 3))
print(f"Mean Accuracy: {gb_cv.mean():.3f}")

# 7. Random Forest 튜닝
param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [1, 3, 5, 7]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=1),
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
print("\n===== Tuned Random Forest =====")
print(f"Best CV Accuracy: {rf_grid.best_score_:.3f}")
print("Best Parameters:", rf_grid.best_params_)
best_rf = rf_grid.best_estimator_

# 8. CV Accuracy 비교
cv_result = {"DT": dt_cv.mean(),"RF": rf_cv.mean(), "GB": gb_cv.mean(),"RF(t)": rf_grid.best_score_}
print("\n===== CV Accuracy Comparison =====")
for name, score in cv_result.items():
    print(f"{name:5s}: {score:.3f}")


# 9. CV Accuracy 시각화

plt.figure(figsize=(8, 4.5))
bars = plt.bar(list(cv_result.keys()),list(cv_result.values()))
plt.ylabel("Accuracy"); plt.title("Model Accuracy Comparison")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.005,
        f"{height:.3f}",
        ha="center"
    )
plt.tight_layout()
plt.savefig("08_model_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()

# 10. 비교할 모델 정의

models = {
    "DT": DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=1
    ),

    "RF": RandomForestClassifier(
        n_estimators=300,
        random_state=1
    ),

    "GB": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=1
    ),

    "RF(t)": best_rf
}

# 11. Test 데이터 기준 모델 비교

result_rows = []
fn_result = {}

print("\n===== Test Evaluation =====")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)

    tn, fp, fn, tp = cm.ravel()

    fn_result[name] = fn

    result_rows.append({
        "Model": name,
        "Accuracy": accuracy,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    print(f"\n{name}")
    print("Confusion Matrix")
    print(cm)
    print(f"Accuracy={accuracy:.3f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# 12. 결과 DataFrame
result_df = pd.DataFrame(result_rows)

print("\n===== Model Result =====")
print(result_df.round(3))


# 13. FN 비교
best_fn_model = min(fn_result,key=fn_result.get)
print("\n===== False Negative Comparison =====")
print("모델별 FN:", fn_result)

print(
    f"FN이 가장 낮은 모델: {best_fn_model}, "
    f"FN={fn_result[best_fn_model]}"
)
# FN → 실제 당뇨 환자를 정상으로 잘못 예측한 경우

# 14. FN 시각화
plt.figure(figsize=(8, 4.5))
bars = plt.bar(list(fn_result.keys()), list(fn_result.values()))
plt.ylabel("False Negative"); plt.title("False Negative Comparison")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.3,
        f"{int(height)}",
        ha="center"
    )
plt.tight_layout()
plt.savefig("08_false_negative.png", dpi=150, bbox_inches="tight")
plt.show()

# 15. 모델별 Classification Report

print("\n===== Classification Report =====")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n----- {name} -----")
    print(
        classification_report(
            y_test,
            pred,
            target_names=["정상", "당뇨"],
            digits=3
        )
    )

# 16. Summary

best_accuracy_model = result_df.loc[
    result_df["Accuracy"].idxmax(),
    "Model"
]
print("\n===== Summary =====")
print(f"Test Accuracy 최고 모델 : {best_accuracy_model}")
print(f"FN 최소 모델            : {best_fn_model}")

print("\nCV Accuracy")
for name, score in cv_result.items():
    print(f"{name:5s}: {score:.3f}")

print("\nFalse Negative")
for name, fn in fn_result.items():
    print(f"{name:5s}: {fn}")

# 모델 선택은 Accuracy 하나만으로 결정하지 않음
# 문제의 목적에 따라 Precision / Recall / F1 / FN 등을 함께 확인

# 특히 당뇨 예측에서는
# 실제 당뇨 환자를 정상으로 놓치는 FN을 중요하게 봐야 함
