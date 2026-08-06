# 자동차 이미지 윤곽선 검출(Contour Detection) with openCV

## 학습 내용
자동차 이미지에서 윤곽선(Contour)을 찾고, 
일정 면적 이상인 큰 윤곽선만 선택하여 표시하기
- Grayscale 변환
- Gaussian Blur를 이용한 노이즈 제거
- Canny Edge를 이용한 경계 검출
- `findContours()`를 이용한 윤곽선 좌표 추출
- `contourArea()`를 이용한 작은 윤곽선 제거
- `drawContours()`를 이용한 결과 표시

---

## 처리 과정

```text
자동차 이미지
      ↓
Gray 변환
      ↓
Gaussian Blur
      ↓
Canny Edge
      ↓
findContours
      ↓
면적 800 이상 선택
      ↓
drawContours
      ↓
결과 출력
```

---

## Grayscale 변환

```python
gray_car = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
```
윤곽선 검출에서는 색상보다 밝기 변화가 중요하므로 이미지를 흑백으로 변환

---

## Gaussian Blur

```python
blur = cv2.GaussianBlur(gray_car, (5, 5), 0)
```
이미지의 작은 노이즈 줄이기
(노이즈가 많으면 작은 점이나 무늬도 윤곽선으로 잘못 검출!)

- 커널 크기가 클수록 더 부드러워짐
- 너무 크게 설정하면 중요한 경계도 흐려질 수 있음

---

## Canny Edge 검출

```python
edge = cv2.Canny(blur, 80, 180)
```
밝기 변화가 큰 부분을 경계로 검출한다.

- `80` : 낮은 임계값
- `180` : 높은 임계값
- 값이 낮을수록 더 많은 경계를 검출
- 값이 높을수록 강한 경계만 남김

---

## 윤곽선 찾기

```python
contours, _ = cv2.findContours(
    edge,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
```
- `cv2.RETR_EXTERNAL` : 가장 바깥쪽 윤곽선만 찾음
- `cv2.CHAIN_APPROX_SIMPLE` : 불필요한 좌표를 줄여 메모리를 절약

---

## 큰 윤곽선만 선택

```python
large_contours = [
    contour
    for contour in contours
    if cv2.contourArea(contour) > 800
]
```
윤곽선 면적이 800보다 큰 것만 선택
: 작은 점이나 잡음으로 생성된 윤곽선을 제거하는 역할

### 결과

```text
전체 윤곽선: 35개
면적 800 이상 윤곽선: 6개
```
작은 윤곽선 29개는 제거되고, 비교적 큰 물체의 윤곽선만 남는다.

---

## 윤곽선 그리기

```python
cv2.drawContours( car, large_contours, -1,(0, 255, 0),2)
```
- `-1` : 선택된 모든 윤곽선 표시
- `(0, 255, 0)` : 초록색
- `2` : 선 두께

---

## 결과 출력

```python
car_rgb = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 5))
plt.imshow(car_rgb)
plt.axis("off")
plt.show()
```
OpenCV는 BGR, Matplotlib는 RGB를 사용하므로 색상 변환 후 출력

---

## 주요 코드

```python
gray_car = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_car, (5, 5), 0)
edge = cv2.Canny(blur, 80, 180)

contours, _ = cv2.findContours(
    edge,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

large_contours = [
    contour
    for contour in contours
    if cv2.contourArea(contour) > 800
]

cv2.drawContours(car, large_contours, -1, (0, 255, 0), 2)
```

---

## 정리
- Grayscale : 색상 정보 제거, 밝기만 사용
- Gaussian Blur : 작은 노이즈를 줄임
- Canny Edge : 밝기 변화가 큰 경계를 검출
- `findContours()`: 경계선을 좌표 목록으로 변환
- `contourArea()` : 작은 윤곽선을 제거
- `drawContours()`: 검출된 윤곽선을 이미지에 표시
  (면적 기준값 `800`은 이미지 크기와 대상에 따라 조정)