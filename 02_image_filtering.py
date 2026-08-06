import cv2
import numpy as np
import matplotlib.pyplot as plt

# 필터 커널 생성
##커널(Kernel)은 이미지의 각 픽셀을 계산하는 작은 행렬
##cv2.filter2D()는 이 커널을 이용해 이미지를 변환
FILTER = {
    #3×3 평균 필터-더 강하게 하려면 커널 크기를 키워
    ##'blur': np.ones((5, 5), dtype=np.float32) / 25
    ##'blur': np.ones((7, 7), dtype=np.float32) / 49
    'blur': np.ones((3, 3), dtype=np.float32) / 9,

    # 커널-윤곽선 조정: 중앙 값을 크게 하고 주변을 음수로 설정하여 윤곽을 더 강조
#    # 'edge': np.array([
#    #     [-1, -1, -1],
#    #     [-1,  8, -1],
#    #     [-1, -1, -1]
#    # ], dtype=np.float32)
    'edge': np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32),

    #Sharpen(선명화) 효과: 중앙 값(5)을 7, 9, 13 등으로 증가시켜 선명도를 높인다.
#   # 'sharpen': np.array([
#   #     [-1, -1, -1],
#   #     [-1, 13, -1],
#   #      [-1, -1, -1]
#   # ], dtype=np.float32)
    'sharpen': np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ], dtype=np.float32)
}

# 이미지 읽기
img = cv2.imread('ex_opencv/imageNet/cat1.jpg')

if img is None:
    print("이미지를 불러오지 못했습니다.")
else:
    # BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
    #필터적용
    ##Blur : 평균값을 사용하여 이미지를 부드럽게(노이즈 감소)
    blur_img = cv2.filter2D(img_rgb, -1, FILTER['blur'])
    ##Edge : 경계(윤곽선)만 강조
    edge_img = cv2.filter2D(img_rgb, -1, FILTER['edge'])
    ##Sharpen : 경계를 강조하여 이미지를 선명하게 표시
    sharpen_img = cv2.filter2D(img_rgb, -1, FILTER['sharpen'])
    #결과출력
    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(blur_img)
    plt.title("Blur")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(edge_img, cmap='gray')
    plt.title("Edge")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(sharpen_img)
    plt.title("Sharpen")
    plt.axis('off')

    plt.tight_layout()
    plt.show()