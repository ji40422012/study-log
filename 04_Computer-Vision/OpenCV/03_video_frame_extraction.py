#동영상을 프레임(frame) 단위로 읽어서 일정 간격마다 이미지를 저장하는 프로그램
# 동영상 열기
#       ▼
# 프레임 읽기
#       ▼
# 읽기 성공?
#  ├── 아니오 → 종료
#  └── 예
#       ▼
# 10프레임마다?
#  ├── 예 → JPG 저장
#  └── 아니오 → 저장 안 함
#       ▼
# count + 1
#       ▼
# 다음 프레임 읽기


import cv2
import os
cap = cv2.VideoCapture('ex_opencv/video/sample.mp4')
count = 0
interval = 10 #10프레임마다 1장 저장
OUTPUT = os.path.join('ex_opencv/video', 'frames')

# makedirs() : 폴더 생성
# exist_ok=True : 이미 폴더가 있어도 오류를 내지 않음
os.makedirs(OUTPUT, exist_ok=True)

while True:  #동영상이 끝날 때까지 계속 반복
    ret, frame = cap.read() #성공여부, 이미지 데이터
            # ret: True → 프레임을 정상적으로 읽음, False → 영상 끝 또는 읽기 실패
    if not ret:  #if not ret=더 이상 읽을 프레임이 없으면 : 반복문을 종료
        break    #영상 끝
    if count % interval == 0:   #나머지가 0일 때만 저장합니다.
        f_nm = f'frame{count:04d}.jpg'  #파일이름: 4자리 정수로 표시하고 부족한 자리는 0으로 채운다
        save = os.path.join(OUTPUT, f_nm)  #저장경로
        cv2.imwrite(save, frame)  #현재 프레임을 JPG 파일로 저장
        print(f'saved {f_nm}')    #저장 메시지 출력
    count += 1
cap.release()  #동영상 파일을 닫고 메모리를 해제
print('완료')
