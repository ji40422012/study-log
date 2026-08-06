# 동영상 프레임 추출(Frame Extraction)
동영상을 프레임(Frame) 단위로 읽고, 일정 간격마다 이미지를 저장
일정 간격으로 프레임을 저장하여 영상 분석이나 데이터셋 생성에 활용

- OpenCV `VideoCapture()` 

- `VideoCapture()`를 이용하여 동영상 파일 열기
- `read()`로 프레임을 순차적으로 읽기
- 일정 간격(10프레임마다)으로 이미지 저장
- `imwrite()`를 이용하여 JPG 파일 생성
- `os.makedirs()`를 이용하여 저장 폴더 자동 생성


## 처리 과정

```text
동영상 열기
      ↓
프레임 읽기
      ↓
읽기 성공?
 ├─ 아니오 → 종료
 └─ 예
      ↓
10프레임마다 저장?
 ├─ 예 → JPG 저장
 └─ 아니오
      ↓
다음 프레임 읽기
```

---

## 주요 코드

```python
cap = cv2.VideoCapture("sample.mp4")
```
동영상 파일 열기

```python
ret, frame = cap.read()
```
프레임을 하나씩 읽는다.
- `ret` : 읽기 성공 여부(True / False)
- `frame` : 현재 프레임 이미지

```python
if count % interval == 0:
```
10프레임마다 이미지를 저장.

```python
cv2.imwrite(save, frame)
```
현재 프레임을 JPG 파일로 저장

---

## 정리

- 동영상은 여러 장의 **프레임(Frame)** 으로 이루어져 있다.
- `VideoCapture()` : 프레임을 순차적으로 읽기
- `read()` : 프레임과 읽기 성공 여부 반환
- `imwrite()` : 원하는 프레임을 이미지 파일로 저장
