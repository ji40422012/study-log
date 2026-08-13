# =========================================================
# Clustering / K-Means + DBSCAN
# Mall Customers Dataset
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
# 한글 폰트 설정
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

# 1. 데이터 불러오기
mall = pd.read_csv("../datasets/unsupervised/Mall_Customers.csv")

print("===== Mall Customers Dataset =====")
print(mall.head())
print("\n데이터 크기:", mall.shape)

# 군집화에 사용할 Feature
X = mall[["Annual Income", "Spending Score"]]

# 2. 고객 분포 확인
plt.figure(figsize=(7, 5))
plt.scatter(mall["Annual Income"],mall["Spending Score"],s=45)

plt.xlabel("연소득(k$)"); plt.ylabel("지출점수(1~100)"); plt.title("쇼핑몰 고객 분포")
plt.tight_layout(); plt.savefig("01_customer_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# 3. Elbow Method
# 적절한 군집 개수 k를 찾기 위해 k=1~10까지 Inertia 비교

inertias = []
k_range = range(1, 11)

for k in k_range:
    model = KMeans(
        n_clusters=k,     # 생성할 군집 개수
        random_state=42,  # 실행 결과 고정
        n_init=10         # 서로 다른 초기 중심점으로 10번 학습 후 가장 좋은 결과 선택
    )

    model.fit(X)
    # inertia_ → 각 데이터와 자신이 속한 Centroid 사이 거리의 제곱합
    inertias.append(model.inertia_)

# 4. Elbow Method 시각화
plt.figure(figsize=(8, 5))
plt.plot(list(k_range),inertias,marker="o")
plt.xlabel("그룹 수 k"); plt.ylabel("Inertia"); plt.title("Elbow Method")
plt.grid(True); plt.tight_layout();
plt.savefig("01_elbow_method.png", dpi=150, bbox_inches="tight")
plt.show()
# 그래프의 감소 폭이 완만해지는 지점을 확인
# 이번 데이터에서는 약 k=4~5 부근에서 꺾이는 형태
# 수업에서는 k=5로 설정

# 5. K-Means 군집화
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# fit_predict() → 모델 학습 + 각 데이터가 속한 Cluster Label 반환
labels = kmeans.fit_predict(X)
mall["group"] = labels

# 각 Cluster의 중심점
centers = kmeans.cluster_centers_
print("\n===== K-Means =====")
print("Cluster 수:", kmeans.n_clusters)
print("Centroid:")
print(centers)

# 6. K-Means 결과 시각화
plt.figure(figsize=(8, 6))
for group in range(5):
    points = mall[mall["group"] == group]
    plt.scatter(
        points["Annual Income"],
        points["Spending Score"],
        s=50,
        label=f"그룹 {group}"
    )

# Centroid 표시
plt.scatter(centers[:, 0],centers[:, 1],s=120, marker="^",label="Centroid")
plt.xlabel("연소득(k$)"); plt.ylabel("지출점수(1~100)"); plt.title("K-Means Clustering")
plt.legend(); plt.tight_layout()
plt.savefig("01_kmeans_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

# 7. 새로운 고객의 Cluster 예측
new_customer = pd.DataFrame([[70, 50]],columns=["Annual Income", "Spending Score"])
group = kmeans.predict(new_customer)[0]
print("\n===== New Customer =====")
print("연소득: 70k"); print("지출점수: 50"); print(f"예측 Cluster: 그룹 {group}")

# 8. DBSCAN을 위한 Scaling
# DBSCAN은 거리 기반 알고리즘이므로 Feature Scale의 영향을 크게 받음
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 9. DBSCAN
dbscan = DBSCAN(
    eps=0.4,          # 이웃으로 인정할 거리 반경
    min_samples=5     # 핵심점이 되기 위한 최소 Sample 수
)
dbscan_labels = dbscan.fit_predict(X_scaled)

# DBSCAN에서 -1은 어느 Cluster에도 속하지 않는 Noise
cluster_count = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
noise_count = int((dbscan_labels == -1).sum())

print("\n===== DBSCAN =====")
print(f"Cluster 수: {cluster_count}")
print(f"Noise 수  : {noise_count}")

# 10. Summary
print("\n===== Summary =====")
print("K-Means");
print("→ 군집 개수 k를 미리 지정"); print("→ Centroid를 기준으로 데이터 그룹화"); print("→ Elbow Method로 적절한 k 탐색")
print("\nDBSCAN")
print("→ 군집 개수를 미리 지정하지 않음"); print("→ 데이터 밀도를 기준으로 군집화"); print("→ Noise를 -1로 구분")