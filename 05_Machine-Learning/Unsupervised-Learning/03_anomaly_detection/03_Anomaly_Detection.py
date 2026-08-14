import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

# Mall Customers 데이터
mall = pd.read_csv("../datasets/unsupervised/Mall_Customers.csv")
income = mall["Annual Income"]

# IQR = Q3 - Q1
q1 = income.quantile(0.25)
q3 = income.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr   # IQR 기준 정상 범위의 하한
upper = q3 + 1.5 * iqr   # IQR 기준 정상 범위의 상한

outliers = mall[(income < lower) | (income > upper)]

print("=== Mall Customers : IQR ===")
print(f"Q1: {q1}"); print(f"Q3: {q3}"); print(f"IQR: {iqr}")
print(f"정상 범위: {lower:.0f} ~ {upper:.0f}")
print(f"이상치 수: {len(outliers)}명")

# Box Plot: IQR 기준 이상치 위치 확인
plt.figure(figsize=(8, 3)); plt.boxplot(income, vert=False); plt.title("Annual Income Box Plot")
plt.xlabel("Annual Income (k$)"); plt.tight_layout()
plt.savefig("03_income_boxplot.png", dpi=150, bbox_inches="tight")  # md에서 사용할 이미지
plt.show()

# Z-score: 평균에서 표준편차 몇 배만큼 떨어져 있는지 계산
z = (income - income.mean()) / income.std()
mall["income_z"] = z
print("\n=== Mall Customers : Z-score ===")
print("Z-score 절대값이 3보다 큰 데이터 수:", (z.abs() > 3).sum())

# 절대 Z-score가 큰 상위 5명 확인
top5_index = z.abs().sort_values(ascending=False).index[:5]

print("\nZ-score 절대값 상위 5명")
print(mall.loc[top5_index, ["Annual Income", "income_z"]])

# California Housing 데이터
housing = pd.read_csv("../datasets/housing/housing.csv")
house_value = housing["median_house_value"]

# median_house_value의 IQR 계산
q1 = house_value.quantile(0.25)
q3 = house_value.quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr   # IQR 기준 정상 범위의 하한
upper = q3 + 1.5 * iqr   # IQR 기준 정상 범위의 상한

# 각 행이 이상치인지 나타내는 Boolean Series
outlier_mask = (house_value < lower) | (house_value > upper)

print("\n=== California Housing : IQR ===")
print(f"Q1: {q1}"); print(f"Q3: {q3}"); print(f"IQR: {iqr}")
print(f"하한: {lower}"); print(f"상한: {upper}")
print(f"이상치 개수: {outlier_mask.sum()}")  # True=1이므로 sum()으로 이상치 개수 계산

# Box Plot: median_house_value의 이상치 확인
plt.figure(figsize=(8, 3)); plt.boxplot(house_value, vert=False)
plt.title("Median House Value Box Plot"); plt.xlabel("Median House Value")
plt.tight_layout(); plt.savefig("03_housing_boxplot.png", dpi=150, bbox_inches="tight")  # md에서 사용할 이미지
plt.show()

# 이상치를 제외한 정상 데이터
normal = housing[~outlier_mask]

print(f"이상치 제거 전 데이터 수: {len(housing)}")
print(f"이상치 제거 후 데이터 수: {len(normal)}")

# 이상치 제거 후 표준편차
std = normal["median_house_value"].std()
std_trunc = np.trunc(std * 100) / 100  # 소수점 셋째 자리에서 버림

print(f"이상치 제거 후 표준편차: {std_trunc}")