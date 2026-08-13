# =========================================================
# Association Rule Analysis
# Apriori + Association Rules
# Grocery Basket Dataset
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
# 한글 폰트 설정
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

# 1. 데이터 불러오기
basket = pd.read_csv("../datasets/recommend/market/groceries_basket.csv")

print("===== Grocery Basket Dataset =====")
print("거래(장바구니) 수:", len(basket))
print("상품 수:", basket.shape[1])
print("\n데이터 타입:")
print(basket.dtypes)
print("\n데이터 일부:")
print(basket.head())
# 각 열은 상품, 값은 해당 장바구니에서 상품 구매 여부


# 2. 상품 구매 빈도 확인
item_freq = basket.sum().sort_values(ascending=False)

print("\n===== Top 10 Items =====")
print(item_freq.head(10))

plt.figure(figsize=(9, 5)); item_freq.head(10)[::-1].plot(kind="barh")
plt.xlabel("구매 장바구니 수"); plt.title("가장 많이 팔린 상품 Top 10")
plt.tight_layout(); plt.savefig("02_item_frequency.png", dpi=150, bbox_inches="tight")
plt.show()

# 3. Frequent Itemsets
# Apriori를 이용하여 자주 함께 등장하는 상품 조합 탐색
frequent = apriori(
    basket,
    min_support=0.02,  # 전체 거래의 2% 이상 등장한 조합만 선택
    use_colnames=True  # 열 번호 대신 실제 상품명 사용
)
# 각 Itemset에 포함된 상품 개수
frequent["n_items"] = frequent["itemsets"].apply(len)
print("\n===== Frequent Itemsets ====="); print("빈발 Itemset 수:", len(frequent))

# 4. 2개 상품 조합 중 Support가 높은 조합 확인
pairs = (frequent[frequent["n_items"] == 2].sort_values("support", ascending=False))
print("\n===== Top Frequent Pairs =====")

for _, row in pairs.head(6).iterrows():
    print(
        f"{set(row['itemsets'])} "
        f"Support={row['support']:.3f}"
    )


# 5. Association Rules 생성
rules = association_rules(
    frequent,
    metric="lift",       # Lift 기준으로 규칙 생성
    min_threshold=1.0    # Lift가 1 이상인 규칙만 선택
)
# Confidence가 30% 이상인 규칙만 선택
rules = (rules[rules["confidence"] >= 0.3].sort_values("lift", ascending=False))
print("\n연관규칙 수:", len(rules))

# 6. 결과를 보기 좋은 형태로 변환
view = rules.copy()
view["A(산 것)"] = view["antecedents"].apply(lambda items: ", ".join(sorted(items)))
view["B(같이 삼)"] = view["consequents"].apply(lambda items: ", ".join(sorted(items)))
result_columns = ["A(산 것)","B(같이 삼)","support","confidence","lift"]

print("\n===== Association Rules =====")
print(view[result_columns].head(8).round(3).to_string(index=False))

# 7. 가장 높은 Lift 규칙 확인
if not view.empty:
    best_rule = view.iloc[0]

    print("\n===== Highest Lift Rule =====")
    print(f"A          : {best_rule['A(산 것)']}")
    print(f"B          : {best_rule['B(같이 삼)']}")
    print(f"Support    : {best_rule['support']:.3f}")
    print(f"Confidence : {best_rule['confidence']:.3f}")
    print(f"Lift       : {best_rule['lift']:.3f}")


# 8. Summary
print("\n===== Summary =====")
print("Support")
print("→ 전체 거래에서 특정 상품 조합이 함께 등장한 비율")

print("\nConfidence")
print("→ A를 구매했을 때 B도 함께 구매한 비율")

print("\nLift")
print("→ A와 B의 관계가 우연한 동시 구매보다 얼마나 강한지 나타내는 지표")
print("→ Lift > 1이면 양의 연관성이 있다고 해석")

print("\nAssociation Rule")
print("→ 함께 구매되는 상품 패턴을 찾아 구매 관계를 분석")
