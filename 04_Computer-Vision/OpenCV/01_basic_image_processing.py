#이미지 처리의 기본 과정+얼굴 찾기

#pip install opencv-python
#pip install opencv-contrib-python

import cv2
print(cv2.__version__); print(hasattr(cv2, 'CascadeClassifier')) #cv2 안에 CascadeClassifier 있는지? r: True
import matplotlib.pyplot as plt
import os

# 이미지 불러오기
img = cv2.imread('ex_opencv/video/p.png')
if img is None:
    print("이미지를 불러오지 못했습니다.")
else:
    # BGR → RGB 변환 : OpenCV는 BGR Matplotlib는 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #확인
    print(f'데이터 타입: {type(img_rgb)}') # r: <class 'numpy.ndarray'> -이미지는 NumPy 배열
    print(f'shape: {img_rgb.shape}')   # r: (726,485,3) -높이, 너비, RGB채널
    print(f'size: {img_rgb.size}')     # 전체 원소 갯수(높이 × 너비 × 채널)
    print(f'픽셀 타입: {img_rgb.dtype}') #픽셀 자료형: uint8
    # 원본 이미지 출력
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

    # -------------------------
    # 다운샘플링
    # -------------------------
    down2 = img_rgb[::2, ::2]  #1/2: 2칸마다 하나씩 가져오기
    down4 = img_rgb[::4, ::4]  #1/4
    down8 = img_rgb[::8, ::8]  #1/8

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1); plt.imshow(img_rgb); plt.title('Original'); plt.axis('off')
    plt.subplot(1, 4, 2); plt.imshow(down2); plt.title('1/2'); plt.axis('off')
    plt.subplot(1, 4, 3); plt.imshow(down4); plt.title('1/4'); plt.axis('off')
    plt.subplot(1, 4, 4); plt.imshow(down8); plt.title('1/8'); plt.axis('off')
    plt.tight_layout(); plt.show()

    # -------------------------
    # 그레이스케일 및 이진화
    # -------------------------
    img_gray = cv2.imread(
        'ex_opencv/video/p.png',
        cv2.IMREAD_GRAYSCALE  #흑백으로 읽기
    )

    # Otsu 방식으로 자동 임계값 계산
    ret, thresh = cv2.threshold(img_gray,0,255,         #0부터 255까지
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU  #자동으로 가장 적절한 임계값을 계산
    )
    print(f'Otsu 임계값: {ret}')
    cv2.imshow('Thresholding', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#얼굴 탐지용 Gray
img_face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Haar 모델 불러오기 : OpenCV가 제공하는 얼굴 탐지 모델(XML 파일)의 경로
xml_path = os.path.join(
    cv2.data.haarcascades,
    'haarcascade_frontalface_default.xml'
)
#CascadeClassifier (얼굴 탐지 모델)을 메모리에 불러
face_cascade = cv2.CascadeClassifier(xml_path)

faces = face_cascade.detectMultiScale(    #이미지 안에서 얼굴을 찾아
    img_face_gray,
    scaleFactor=1.01,   #이미지 크기를 조금씩 줄여가며: 1.01(아주촘촘)-느리지만 세밀 (일반적으로 1.1)
    minNeighbors=6,     #얼굴로 인정하기 위한 최소 이웃 수: 작으면 오탐지 증가, 크면 얼굴 놓칠 수도
    minSize=(20, 20)    #20×20보다 작은 얼굴은 무시
)
print(f'발견된 얼굴 수: {len(faces)}')
print(faces)

#결과 그리기
for (x, y, w, h) in faces:   #발견된 얼굴에 표시 : rectangle(사각형)
    cv2.rectangle(img,
                  (x, y),         #시작점
                  (x + w, y + h), #끝점
                  (255, 0, 0),    #파랑(BGR)
                  2)              # 두께2

#Matplotlib에서 정상적인 색상으로 출력하기 위해 RGB로 변환
face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(face)
plt.show()
