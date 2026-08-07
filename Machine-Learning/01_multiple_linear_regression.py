import os

import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 및 음수 기호 설정
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

# 공부 시간(x1), 과외 횟수(x2), 성적(y)
x1 = np.array([2, 4, 6, 8], dtype=np.float64)
x2 = np.array([0, 4, 2, 3], dtype=np.float64)
y = np.array([81, 93, 91, 97], dtype=np.float64)

# 실제 데이터 확인
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")

ax.scatter3D(x1, x2, y, color="red", s=60)
ax.set_xlabel("공부 시간(x1)")
ax.set_ylabel("과외 횟수(x2)")
ax.set_zlabel("성적(y)")

plt.tight_layout()
plt.show()

# 모델 파라미터 초기화
w1 = 0.0
w2 = 0.0
b = 0.0

learning_rate = 0.01
epochs = 2001
n = len(y)

# 학습 과정 기록
history_w1 = []
history_w2 = []
history_b = []
history_loss = []

for epoch in range(epochs):
    # 예측값: y_hat = w1*x1 + w2*x2 + b
    y_hat = w1 * x1 + w2 * x2 + b

    # 오차와 평균제곱오차(MSE)
    error = y - y_hat
    loss = np.mean(error**2)

    # MSE를 각 파라미터로 편미분한 기울기
    w1_gradient = (-2 / n) * np.sum(x1 * error)
    w2_gradient = (-2 / n) * np.sum(x2 * error)
    b_gradient = (-2 / n) * np.sum(error)

    # 경사하강법으로 파라미터 갱신
    w1 -= learning_rate * w1_gradient
    w2 -= learning_rate * w2_gradient
    b -= learning_rate * b_gradient

    # 학습 과정 기록
    history_w1.append(w1)
    history_w2.append(w2)
    history_b.append(b)
    history_loss.append(loss)

    if epoch % 200 == 0:
        print(
            f"epoch {epoch:>4}: "
            f"w1={w1:.3f}, "
            f"w2={w2:.3f}, "
            f"b={b:.3f}, "
            f"mse={loss:.3f}"
        )

print(
    f"\n최종: w1={w1:.3f}, "
    f"w2={w2:.3f}, "
    f"b={b:.3f}"
)

# 실제 데이터와 회귀평면 시각화
fig = plt.figure(figsize=(6.5, 5))
ax = fig.add_subplot(111, projection="3d")

ax.scatter3D(
    x1,
    x2,
    y,
    color="red",
    s=40,
    label="실제 데이터"
)

# 회귀평면을 만들기 위한 격자
x1_grid, x2_grid = np.meshgrid(
    np.linspace(x1.min(), x1.max(), 20),
    np.linspace(x2.min(), x2.max(), 20)
)

# 각 격자 좌표의 예측값
y_grid = w1 * x1_grid + w2 * x2_grid + b

ax.plot_surface(
    x1_grid,
    x2_grid,
    y_grid,
    alpha=0.4,
    cmap="viridis"
)

ax.set_xlabel("공부 시간")
ax.set_ylabel("과외 횟수")
ax.set_zlabel("성적")
ax.legend()

plt.tight_layout()

# GitHub README/Markdown에서 사용할 이미지 저장
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(
    os.path.join(output_dir, "multiple_linear_regression_plane.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# 손실 함수 변화 시각화
plt.figure(figsize=(7, 4))
plt.plot(history_loss)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("학습에 따른 손실 변화")
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "multiple_linear_regression_loss.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()