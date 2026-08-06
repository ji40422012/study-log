#자동차 이미지에서 윤곽선(Contour)을 찾아 큰 윤곽만 추출하여 표시하는 프로그램

# 자동차 이미지
#       ▼
# Gray 변환
#       ▼
# Gaussian Blur(노이즈 제거)
#       ▼
# Canny Edge (윤곽 검출)
#       ▼
# findContours (윤곽선 좌표 찾기)
#       ▼
# 면적 800 이상만 선택
#       ▼
# drawContours (초록색으로 윤곽 표시)
#       ▼
# 결과 출력




import cv2
import numpy as np
import matplotlib.pyplot as plt

car = cv2.imread('../../datasets/car.jpg')
#1. edge -> 2. coutours

#컬러 이미지를 흑백으로 변환합니다.
## Edge 검출, 윤곽선 검출은 색상보다 밝기 정보만 필요하기 때문
gray_car = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)

#노이즈를 제거: 커널 크기 (5,5)가 클수록 더 부드럽게
##노이즈가 많으면 작은 점들도 Edge로 인식
blur = cv2.GaussianBlur(gray_car, (5,5), 0)

#윤곽선만 추출 작은 값 =많은 edge, 큰 값: 강한 edge 만 남김
edge = cv2.Canny(blur, 80, 180)

plt.figure(figsize = (8,5))
plt.imshow(edge, cmap = 'gray')
plt.axis('off')
plt.show()


#윤곽선 찾기
#findContours()는 흰색 영역의 경계
edgeline, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##반환값 edge line = 윤곽선 1, 윤곽선 2, ...각 윤곽은 수백개의 좌표로 구성
## cv2.RETR_EXTERNAL : 가장 바깥 윤곽만
##CHAIN_APPROX_SIMPLE: 윤곽 좌표를 압축 저장(모든 좌표 저장이 아니라 시작과 끝만(메모리 절약)

#윤곽선 내부 픽셀 수 800보다 큰
#리스트 컴프리헨션
l_edgeline = [
    c
    for c in edgeline        #윤곽선을 하나씩 꺼내
    if cv2.contourArea(c)    #윤곽의 면적을 계산
       > 800]                #그 중 800이상인 것만 저장 (=작은 노이즈가 제거됨)
print(
    '전체 윤곽선:',
    len(edgeline),
    '면적 큰 윤곽:',
    len(l_edgeline)
)
cv2.drawContours(car,        #그릴것
                 l_edgeline, #그릴 윤곽 목록
                 -1,         #모든 윤곽 그리기
                 (0,255,0), #초록색으로
                 2       #두께 2
                 )
plt.figure(figsize = (8,5))
plt.imshow(car)   #현재 car는 BGR 형식이므로 색상이 이상하게 보일 수
#OpenCV의 색상 형식(BGR)을 Matplotlib의 색상 형식(RGB)으로 변환한 후 이미지를 출력
plt.imshow(cv2.cvtColor(car, cv2.COLOR_BGR2RGB))
plt.show()