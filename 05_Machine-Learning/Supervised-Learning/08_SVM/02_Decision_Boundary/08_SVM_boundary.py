import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles, make_moons
from sklearn.svm import SVC

plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

# 비선형 분류 확인을 위한 원형 데이터
X, y = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=0)

def plot_boundary(ax, clf, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=30)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


# Linear SVM: 원형 데이터는 직선으로 분리하기 어려움
lin = SVC(kernel="linear").fit(X, y)

fig, ax = plt.subplots(figsize=(6, 5.5))
plot_boundary(ax, lin, X, y, f"Linear SVM (Accuracy {lin.score(X, y):.2f})")
plt.tight_layout()
plt.savefig("08_linear_boundary.png", dpi=150, bbox_inches="tight")
plt.show()


# 2차원 데이터를 3차원으로 Feature Mapping
def phi(P):
    return np.column_stack([
        P[:, 0],
        P[:, 1],
        P[:, 0] ** 2 + P[:, 1] ** 2  # 원점에서 멀수록 값이 커지는 새로운 Feature
    ])

Xm = phi(X)

z_in = Xm[y == 1, 2].mean()
z_out = Xm[y == 0, 2].mean()

print(f"안쪽 원 평균 높이 z = {z_in:.2f}")
print(f"바깥쪽 원 평균 높이 z = {z_out:.2f}")


# Feature Mapping 결과를 3차원으로 시각화
fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    Xm[:, 0], Xm[:, 1], Xm[:, 2],
    c=y, cmap="coolwarm", edgecolor="k", s=30
)

xx, yy = np.meshgrid(
    np.linspace(-1.2, 1.2, 10),
    np.linspace(-1.2, 1.2, 10)
)

# 두 클래스를 나누는 평면
plane_z = (z_in + z_out) / 2
ax.plot_surface(
    xx, yy,
    np.full_like(xx, plane_z),
    alpha=0.3,
    color="green"
)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x1² + x2²")
ax.set_title("Feature Mapping to 3D")

plt.tight_layout()
plt.savefig("08_feature_mapping_3d.png", dpi=150, bbox_inches="tight")
plt.show()


# 직접 Feature Mapping한 뒤 Linear SVM 적용
class PhiSVM:
    def __init__(self):
        self.model = SVC(kernel="linear")  # 고차원 공간에서는 Linear SVM 사용

    def fit(self, X, y):
        self.model.fit(phi(X), y)
        return self

    def predict(self, X):
        return self.model.predict(phi(X))


phi_svm = PhiSVM().fit(X, y)

fig, ax = plt.subplots(figsize=(6, 5.5))
plot_boundary(ax, phi_svm, X, y, "Feature Mapping + Linear SVM")
plt.tight_layout()
plt.savefig("08_circular_boundary.png", dpi=150, bbox_inches="tight")
plt.show()


# RBF Kernel: 직접 Feature Mapping하지 않고 비선형 결정경계 생성
rbf = SVC(
    kernel="rbf",
    gamma=1  # 클수록 각 데이터의 영향 범위가 좁아져 경계가 복잡해질 수 있음
).fit(X, y)

fig, ax = plt.subplots(figsize=(6, 5.5))
plot_boundary(ax, rbf, X, y, f"RBF Kernel SVM (Accuracy {rbf.score(X, y):.2f})")
plt.tight_layout()
plt.savefig("08_rbf_boundary.png", dpi=150, bbox_inches="tight")
plt.show()

print("Linear Kernel 정확도:", round(lin.score(X, y), 2))
print("RBF Kernel 정확도:", round(rbf.score(X, y), 2))


# Polynomial Kernel Trick 확인
def phi_poly(v):
    x1, x2 = v
    return np.array([
        1,
        np.sqrt(2) * x1,
        np.sqrt(2) * x2,
        x1 ** 2,
        np.sqrt(2) * x1 * x2,
        x2 ** 2
    ])

a = np.array([2, 3])
b = np.array([1, 4])

explicit = phi_poly(a) @ phi_poly(b)  # 실제로 고차원으로 변환한 뒤 내적
kernel = (a @ b + 1) ** 2             # Kernel Function으로 같은 내적 결과 계산

print("\n=== Kernel Trick ===")
print("φ(a) · φ(b) =", explicit)
print("(a · b + 1)² =", kernel)
print("두 값이 같음 → 실제 고차원 좌표를 만들지 않고도 고차원 내적 계산 가능")


# 초승달 데이터에서 Linear / RBF Kernel 비교
X_moon, y_moon = make_moons(n_samples=200, noise=0.15, random_state=0)

linear_moon = SVC(kernel="linear").fit(X_moon, y_moon)
rbf_moon = SVC(kernel="rbf").fit(X_moon, y_moon)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

plot_boundary(
    ax[0],
    linear_moon,
    X_moon,
    y_moon,
    f"Linear Kernel ({linear_moon.score(X_moon, y_moon):.2f})"
)

plot_boundary(
    ax[1],
    rbf_moon,
    X_moon,
    y_moon,
    f"RBF Kernel ({rbf_moon.score(X_moon, y_moon):.2f})"
)

plt.tight_layout()
plt.savefig("08_moons_kernel_comparison.png", dpi=150, bbox_inches="tight")
plt.show()