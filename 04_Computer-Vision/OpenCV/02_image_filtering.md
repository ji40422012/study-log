# 이미지 필터링

## 학습 내용

OpenCV의 `cv2.filter2D()`로 이미지 필터 적용

- **Blur(평균 필터)** : 주변 픽셀의 평균값을 적용, 이미지를 부드럽게 만들고 노이즈를 감소
- **Edge 필터** : 픽셀의 밝기 차이를 이용, 이미지의 경계(윤곽선)를 검출
- **Sharpen 필터** : 경계 부분을 강조, 이미지를 더욱 선명하게 
---

## Kernel(커널)

커널(Kernel)은 이미지의 각 픽셀을 계산하기 위한 작은 행렬
`cv2.filter2D()`는 커널를 이미지 전체에 적용하여 새로운 이미지를 생성

### Blur 커널

```python
np.ones((3, 3), dtype=np.float32) / 9
```

- 주변 3×3 픽셀의 평균값을 계산, 노이즈를 줄이고 이미지를 부드럽게

### Edge 커널

```python
np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])
```

- 픽셀 간 밝기 차이를 계산, 윤곽선을 강조

### Sharpen 커널

```python
np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
```

- 중앙 픽셀의 값을 강조, 이미지가 더욱 선명하게

---

## 주요 코드

```python
# Blur
blur_img = cv2.filter2D(img_rgb, -1, FILTER['blur'])

# Edge
edge_img = cv2.filter2D(img_rgb, -1, FILTER['edge'])

# Sharpen
sharpen_img = cv2.filter2D(img_rgb, -1, FILTER['sharpen'])
```

---
- **Kernel**은 이미지의 각 픽셀을 계산하기 위한 작은 행렬
- 커널의 값에 따라 다양한 영상 처리 효과 구현 가능!
- `cv2.filter2D()`를 사용하면 원하는 커널를 이미지에 적용
- **Blur**는 노이즈 제거와 부드러운 이미지 생성
- **Edge**는 물체의 윤곽선이나 경계를 검출
- **Sharpen**은 흐린 이미지를 더욱 선명하게
