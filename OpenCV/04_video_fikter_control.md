# 동영상 필터 제어(Video Filter Control) with openCV
동영상을 실시간으로 재생하고, 키보드 입력에 따라 Blur 필터를 적용 및 해제

- `VideoCapture()`를 이용한 동영상 재생
- `blur()`를 이용한 실시간 Blur 적용
- `waitKey()`를 이용한 키보드 입력 처리
- `putText()`를 이용한 현재 모드 표시
- 영상이 끝나면 처음부터 다시 재생

---

## 동작 과정

```text
동영상 열기
      ↓
프레임 읽기
      ↓
영상 끝?
 ├─ 예 → 처음 프레임으로 이동
 └─ 아니오
      ↓
현재 mode 확인
 ├─ 0 → 원본 출력
 └─ 1 → Blur 적용
      ↓
영상 출력
      ↓
키 입력
 ├─ 0 → 원본
 ├─ 1 → Blur
 └─ q → 종료
```

---

## 주요 함수

### `cv2.VideoCapture()`
동영상 파일을 열고 프레임을 읽기 위한 객체를 생성

```python
cap = cv2.VideoCapture("Air_Force_One.mp4")
```

---

### `cv2.blur()`
평균 필터를 이용, 이미지를 부드럽게

```python
frame = cv2.blur(frame, (15, 15))
```
- `(15, 15)` == 커널 크기: 커널 클수록 Blur 강

---

### `cv2.putText()`
영상 위에 현재 모드와 같은 문자열 출력

```python
cv2.putText(frame, "mode: blur", (200, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
```
---

### `cv2.waitKey()`
키보드 입력을 받아 프로그램을 제어

```python
key = cv2.waitKey(30) & 0xFF
```
- `0` : 원본 영상
- `1` : Blur 적용
- `q` : 프로그램 종료

---

## 예시

프로그램 실행 중

|입력 키|동작|
|---|---|
|`0`|원본 영상 출력|
|`1`|Blur 필터 적용|
|`q`|프로그램 종료|

영상이 끝까지 재생되면 자동으로 처음부터 다시 재생된다.

---

## 주요 코드

```python
if mode == 1:
    frame = cv2.blur(frame, (15, 15))
```
Blur 필터

```python
key = cv2.waitKey(30) & 0xFF
```
30ms 동안 키 입력 대기

```python
if key == ord('q'):
    break
elif key == ord('0'):
    mode = 0
elif key == ord('1'):
    mode = 1
```
키보드 입력에 따라 모드를 변경

---

## 정리

- `VideoCapture()` : 동영상을 프레임 단위로 읽기
- `blur()` : 실시간으로 Blur 적용 (점 = 노이즈) 
- `waitKey()` : 키보드 입력을 처리
- `putText()` : 영상 위에 현재 상태를 표시 가능
- 영상이 끝나면 프레임 위치를 0으로 설정하여 반복 재생