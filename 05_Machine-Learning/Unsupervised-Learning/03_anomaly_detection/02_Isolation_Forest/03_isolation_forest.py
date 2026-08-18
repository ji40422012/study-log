#Isolation Forest는
# 무작위 Feature와 분할값으로 관측치를 고립시키고,
# 더 적은 분할로 고립되는 데이터를 이상치로 판단하는 방식
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

# California Housing 데이터
housing = pd.read_csv("../datasets/housing/housing.csv")

# 처음에는 2개 Feature만 사용하여 Isolation Forest 결과를 시각적으로 확인
features = ["median_income", "median_house_value"]
X = housing[features].copy()

model = IsolationForest(
    contamination=0.02,  # 전체 데이터 중 이상치로 판단할 비율
    random_state=42      # 실행할 때마다 같은 결과가 나오도록 설정
)

X["anomaly"] = model.fit_predict(X[features])       # 정상=1, 이상치=-1
X["anomaly_score"] = model.decision_function(X[features])  # 값이 작을수록 상대적으로 이상함

normal = X[X["anomaly"] == 1]
anomaly = X[X["anomaly"] == -1]

print("=== Isolation Forest ===")
print(f"전체 데이터 수: {len(X)}")
print(f"정상 데이터 수: {len(normal)}")
print(f"이상치 수: {len(anomaly)}")

# anomaly score가 작은 데이터부터 확인
print("\nAnomaly Score가 낮은 상위 10개")
print(X.sort_values("anomaly_score").head(10))

# median_income × median_house_value에서 정상/이상치 비교
plt.figure(figsize=(8, 5))
plt.scatter(normal["median_income"], normal["median_house_value"],
            s=10, alpha=0.4, label="Normal")
plt.scatter(anomaly["median_income"], anomaly["median_house_value"],
            s=25, label="Anomaly")

plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.legend()
plt.tight_layout()
plt.savefig("03_isolation_forest.png", dpi=150, bbox_inches="tight")  # md에서 사용할 이미지
plt.show()