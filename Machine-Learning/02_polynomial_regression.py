# pip install numpy pandas matplotlib scikit-learn
## numpy: 배열 계산, linspace, 다항식 계산
## pandas: CSV 파일 불러오기
## matplotlib: 그래프 출력 및 이미지 저장
## scikit-learn: PolynomialFeatures, LinearRegression, 평가 지표 사용

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# ---------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------
# 결과 그래프를 저장할 폴더 : Path를 사용하면 운영체제에 맞는 경로를 안전하게 만들 수 있다.
OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 다항회귀에 사용할 다항식 차수
POLYNOMIAL_DEGREE = 5

# CSV 파일 경로 :실제 파일 위치에 맞게 수정
DATA_PATH = Path(
    r"C:\workspace\workspace_python\datasets\nonlinear_data.csv"
)

# ---------------------------------------------------------
# 1. 기본 함수 그래프 확인
# ---------------------------------------------------------

def cubic_function(x: np.ndarray) -> np.ndarray:
    """
    3차 다항함수 값을 계산한다.

    Parameters
    ----------
    x : np.ndarray
        함수에 입력할 x값 배열

    Returns
    -------
    np.ndarray
        각 x값에 대응하는 y값
    """
    return 4 * x**3 - 3 * x**2 + 2 * x + 1

def plot_basic_polynomial_function() -> None:
    """3차 다항함수의 기본 형태를 그래프로 출력한다."""
    # -3부터 3까지 균일한 간격으로 400개의 값을 생성한다.
    # 데이터 개수를 늘리면 곡선이 더 부드럽게 표현된다.
    x = np.linspace(-3, 3, 400)

    # 각 x값에 대한 함수값을 계산한다.
    y = cubic_function(x)

    # figsize=(가로, 세로), 단위는 인치
    plt.figure(figsize=(7, 4))

    # linewidth: 선 두께
    # label: 범례에 표시할 이름
    plt.plot(
        x,
        y,
        linewidth=2,
        label=r"$y=4x^3-3x^2+2x+1$"
    )

    plt.title("3차 다항함수")
    plt.xlabel("x")
    plt.ylabel("y")

    # alpha: 격자선의 투명도
    # linestyle: 격자선 형태
    plt.grid(
        visible=True,
        linestyle="--",
        alpha=0.5
    )

    # plot()에서 지정한 label을 범례로 표시한다.
    plt.legend()

    # 그래프 요소가 잘리지 않도록 여백을 자동 조정한다.
    plt.tight_layout()

    # dpi: 저장 이미지의 해상도
    # bbox_inches="tight": 그래프 주변의 불필요한 여백 제거
    plt.savefig(
        OUTPUT_DIR / "basic_polynomial_function.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()


# ---------------------------------------------------------
# 2. 다항식 차수별 적합 비교
# ---------------------------------------------------------

def create_noisy_sine_data(
    sample_count: int = 20,
    noise_scale: float = 0.2,
    random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    sin 함수에 정규분포 잡음을 추가한 데이터를 생성

    Parameters
    ----------
    sample_count : int
        생성할 데이터 개수

    noise_scale : float
        정규분포 잡음의 표준편차
        값이 클수록 데이터가 실제 함수에서 더 많이 벗어난다.

    random_seed : int
        난수 생성 결과를 고정하기 위한 값

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x값과 잡음이 포함된 y값
    """
    rng = np.random.default_rng(random_seed)

    # 0부터 2π까지 sample_count개의 데이터를 생성한다.
    x = np.linspace(0, 2 * np.pi, sample_count)

    # 평균 0, 표준편차 noise_scale인 정규분포 잡음 생성
    noise = rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=sample_count
    )

    y = np.sin(x) + noise

    return x, y


def plot_degree_comparison() -> None:
    """1차, 2차, 3차, 9차 다항식의 데이터 적합 결과를 비교한다."""

    x, y = create_noisy_sine_data()

    # 예측 곡선을 부드럽게 그리기 위한 연속적인 x값
    x_plot = np.linspace(
        x.min(),
        x.max(),
        300
    )

    # 비교할 다항식 차수
    degrees = [1, 2, 3, 9]

    plt.figure(figsize=(14, 9))

    for index, degree in enumerate(degrees):
        # np.polyfit(x, y, deg)
        # x, y 데이터에 가장 잘 맞는 지정 차수의 다항식 계수를 계산한다.
        coefficients = np.polyfit(
            x,
            y,
            deg=degree
        )

        # np.poly1d()는 계산된 계수로 실제 다항함수 객체를 만든다.
        polynomial = np.poly1d(coefficients)

        # 2행 2열 중 index + 1번째 위치에 그래프를 배치한다.
        plt.subplot(2, 2, index + 1)

        # 실제 학습 데이터 표시
        # s: 점의 크기
        # alpha: 점의 투명도
        plt.scatter(
            x,
            y,
            s=35,
            alpha=0.8,
            label="데이터"
        )

        # 학습한 다항곡선 표시
        plt.plot(
            x_plot,
            polynomial(x_plot),
            linewidth=2,
            label=f"{degree}차 다항식"
        )

        plt.title(f"다항식 차수: {degree}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(
            visible=True,
            linestyle="--",
            alpha=0.4
        )
        plt.legend()

    # 여러 subplot 사이의 간격을 자동 조정한다.
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "polynomial_degree_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


# ---------------------------------------------------------
# 3. CSV 데이터를 이용한 다항회귀
# ---------------------------------------------------------

def load_dataset(
    file_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """
    CSV 파일에서 입력 변수 X와 목표 변수 y를 불러온다.

    Parameters
    ----------
    file_path : Path
        CSV 파일 경로

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        입력 변수 x와 목표 변수 y

    Raises
    ------
    FileNotFoundError
        파일이 존재하지 않는 경우

    ValueError
        CSV에 X 또는 y 열이 없는 경우
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {"X", "y"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV 파일에는 'X'와 'y' 열이 필요합니다."
        )

    # 결측값이 있는 행은 제거한다.
    df = df.dropna(subset=["X", "y"])

    # to_numpy(dtype=...)로 숫자 배열로 변환한다.
    x = df["X"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)

    return x, y


def create_polynomial_model(
    degree: int
):
    """
    다항 특성 변환과 선형회귀를 연결한 Pipeline을 생성한다.

    Parameters
    ----------
    degree : int
        생성할 다항 특성의 최고 차수

    Returns
    -------
    Pipeline
        PolynomialFeatures와 LinearRegression이 연결된 모델
    """
    model = make_pipeline(
        PolynomialFeatures(
            degree=degree,

            # False이면 값이 항상 1인 특성을 생성하지 않는다.
            # 절편은 LinearRegression의 fit_intercept가 처리한다.
            include_bias=False
        ),
        LinearRegression(
            # True이면 절편 b를 학습한다.
            fit_intercept=True
        )
    )

    return model


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict[str, float]:
    """
    회귀 모델의 평가 지표를 계산한다.

    Parameters
    ----------
    y_true : np.ndarray
        실제값

    y_pred : np.ndarray
        예측값

    Returns
    -------
    dict[str, float]
        MSE, RMSE, R²
    """
    mse = mean_squared_error(y_true, y_pred)

    # RMSE는 MSE에 제곱근을 적용한 값이다.
    # 원래 목표 변수와 단위가 같아 해석이 더 쉽다.
    rmse = np.sqrt(mse)

    # R²는 모델이 데이터의 변동을 얼마나 설명하는지 나타낸다.
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }


def print_polynomial_equation(
    model
) -> None:
    """학습한 다항회귀 모델의 계수와 절편을 출력한다."""

    # Pipeline에서 각 단계의 객체를 이름으로 가져온다.
    polynomial_features = model.named_steps[
        "polynomialfeatures"
    ]
    linear_regression = model.named_steps[
        "linearregression"
    ]

    # 생성된 특성 이름 확인
    # 입력 변수가 하나이면 x, x², x³ 등의 이름이 생성된다.
    feature_names = polynomial_features.get_feature_names_out(
        ["x"]
    )

    coefficients = linear_regression.coef_
    intercept = linear_regression.intercept_

    print("\n학습된 회귀식")
    print(f"절편: {intercept:.4f}")

    for feature_name, coefficient in zip(
        feature_names,
        coefficients
    ):
        print(
            f"{feature_name:>5}: "
            f"{coefficient:.4f}"
        )


def fit_and_plot_polynomial_regression(
    x: np.ndarray,
    y: np.ndarray,
    degree: int
) -> None:
    """다항회귀 모델을 학습하고 평가 및 시각화를 수행한다."""

    # scikit-learn은 입력 X를 2차원 배열로 받는다.
    #
    # 변환 전:
    # [1, 2, 3]
    #
    # 변환 후:
    # [[1],
    #  [2],
    #  [3]]
    x_2d = x.reshape(-1, 1)

    model = create_polynomial_model(degree)

    # 다항 특성 생성과 선형회귀 학습이 순서대로 실행된다.
    model.fit(x_2d, y)

    # 학습 데이터에 대한 예측
    y_train_pred = model.predict(x_2d)

    metrics = evaluate_model(
        y_true=y,
        y_pred=y_train_pred
    )

    print(f"\n다항식 차수: {degree}")
    print(f"MSE : {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²  : {metrics['r2']:.4f}")

    print_polynomial_equation(model)

    # 기존 x가 정렬되어 있지 않으면 plot()이 점을 잘못된 순서로
    # 연결할 수 있다. 따라서 연속적인 x_plot을 별도로 생성한다.
    x_plot = np.linspace(
        x.min(),
        x.max(),
        400
    )

    y_plot_pred = model.predict(
        x_plot.reshape(-1, 1)
    )

    plt.figure(figsize=(9, 5))

    # 실제 데이터
    plt.scatter(
        x,
        y,
        s=30,
        alpha=0.8,
        label="실제 데이터"
    )

    # 학습된 예측 곡선
    plt.plot(
        x_plot,
        y_plot_pred,
        linewidth=2.5,
        label=f"{degree}차 다항회귀"
    )

    plt.title(
        f"다항회귀 결과 — degree={degree}"
    )
    plt.xlabel("X")
    plt.ylabel("y")

    plt.grid(
        visible=True,
        linestyle="--",
        alpha=0.4
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "polynomial_regression_result.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    # 잔차 = 실제값 - 예측값
    residuals = y - y_train_pred

    plt.figure(figsize=(8, 4))

    # 기준선 y=0
    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    plt.scatter(
        y_train_pred,
        residuals,
        s=30,
        alpha=0.8
    )

    plt.title("잔차 그래프")
    plt.xlabel("예측값")
    plt.ylabel("잔차(실제값 - 예측값)")
    plt.grid(
        visible=True,
        linestyle="--",
        alpha=0.4
    )
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "polynomial_regression_residuals.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


# ---------------------------------------------------------
# 실행 영역
# ---------------------------------------------------------

def main() -> None:
    """프로그램의 전체 실행 순서를 관리한다."""

    # 다항함수의 기본 모양 확인
    plot_basic_polynomial_function()

    # 차수별 적합 결과 비교
    plot_degree_comparison()

    # CSV 데이터 불러오기
    x, y = load_dataset(DATA_PATH)

    # 다항회귀 학습 및 결과 출력
    fit_and_plot_polynomial_regression(
        x=x,
        y=y,
        degree=POLYNOMIAL_DEGREE
    )


# 이 파일을 직접 실행했을 때만 main()을 호출한다.
# 다른 Python 파일에서 import할 때는 자동 실행되지 않는다.
if __name__ == "__main__":
    main()