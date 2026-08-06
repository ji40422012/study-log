# OpenCV를 이용한 이미지 처리의 기본 과정과 얼굴 검출

## 주요 코드

```python
# BGR → RGB 변환
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 다운샘플링
down2 = img_rgb[::2, ::2]

# Otsu 이진화
ret, thresh = cv2.threshold(
    img_gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 얼굴 검출
faces = face_cascade.detectMultiScale(
    img_face_gray,
    scaleFactor=1.01,
    minNeighbors=6,
    minSize=(20, 20)
)
```

## 정리

- OpenCV는 BGR, Matplotlib는 RGB를 사용
- 이미지는 NumPy 배열(ndarray)
- 다운샘플링으로 이미지 크기를 줄임
- Otsu 알고리즘은 최적의 임계값을 자동으로 계산한다.
- Haar Cascade는 미리 학습된 모델(XML)로 얼굴을 검출

### 다운샘플링의 장점

- 이미지 크기를 줄여 메모리 사용량을 감소시킨다.
- 연산해야 하는 픽셀 수가 줄어 처리 속도가 빨라진다.
- 실시간 영상 처리에서 성능을 향상시킬 수 있다.
- 너무 큰 이미지를 다룰 때 계산 비용을 줄일 수 있다.

### 얼굴 검출
detectMultiScale()의 scaleFactor, minNeighbors, minSize가 얼굴 검출 성능을 결정
