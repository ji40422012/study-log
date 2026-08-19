import numpy as np
import matplotlib.pyplot as plt

plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False


class Perceptron:
    def __init__(self, input_size):
        self.w = np.zeros(input_size + 1)  # w[0]=bias, w[1:]=각 입력에 대한 가중치

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.w[1:]) + self.w[0]  # 입력×가중치의 합 + 편향
        return Perceptron.step_function(weighted_sum)

    @staticmethod
    def step_function(weighted_sum):
        return 1 if weighted_sum > 0 else 0  # 가중합이 0보다 크면 1, 아니면 0

    def train(self, train_inputs, labels, lr=0.01, epochs=100):
        for _ in range(epochs):  # 전체 데이터를 epochs만큼 반복 학습
            for inputs, label in zip(train_inputs, labels):
                pred = self.predict(inputs)
                error = label - pred  # 실제값 - 예측값

                self.w[1:] += lr * error * inputs  # 가중치 = 기존 가중치 + 학습률 × 오차 × 입력
                self.w[0] += lr * error            # 편향 = 기존 편향 + 학습률 × 오차


# AND Gate 학습 데이터
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

model = Perceptron(input_size=2)
model.train(X, y, lr=0.1, epochs=100)

print("=== AND Perceptron ===")

for inputs, label in zip(X, y):
    pred = model.predict(inputs)
    print(f"입력: {inputs}, 예측: {pred}, 정답: {label}")

print(f"\n학습된 가중치: {model.w[1:]}")
print(f"학습된 편향: {model.w[0]}")


# 학습된 Perceptron의 결정경계 시각화
plt.figure(figsize=(6, 5))

for label in np.unique(y):
    points = X[y == label]
    plt.scatter(points[:, 0], points[:, 1], s=100, label=f"Class {label}")

# w1*x1 + w2*x2 + bias = 0을 x2에 대해 정리
w1, w2 = model.w[1:]
bias = model.w[0]

x1 = np.linspace(-0.5, 1.5, 100)

if w2 != 0:
    x2 = -(w1 * x1 + bias) / w2
    plt.plot(x1, x2, label="Decision Boundary")

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron - AND Gate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("01_perceptron_and.png", dpi=150, bbox_inches="tight")  # md에서 사용할 Figure
plt.show()
