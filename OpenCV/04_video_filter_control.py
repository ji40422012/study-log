#동영상을 재생하면서 키보드 입력에 따라 영상에 필터(Blur)를 적용하는 프로그램

#동영상 열기
#       ▼
# 프레임 읽기
#       ▼
# 영상 끝?
#  ├── 예 → 처음 프레임으로 이동
#  └── 아니오
#       ▼
# mode 확인
#  ├── 0 → 원본 출력
#  └── 1 → Blur 적용
#       ▼
# 화면 출력
#       ▼
# 키 입력
#  ├── 0 → 원본
#  ├── 1 → Blur
#  └── q → 종료

import cv2
cap = cv2.VideoCapture('ex_opencv/video/Air_Force_One.mp4')

mode = 0   # 0: none(원본영상), 1: blur

while True:  #동영상을 계속 재생하기 위해 무한 반복
    ret, frame = cap.read()

    if not ret:  #영상 끝이면 처음부터 다시:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 현재 프레임 위치를 0으로 바꾸면 처음부터 다시 재생
        continue

    #mode가 1이면 블러 적용
    if mode == 1:
        frame = cv2.blur(frame, (15, 15))  #커널 크기 15,15
        cv2.putText(
            frame,
            'mode: blur',   #'mode: blur' 라는 글자 넣기
            (200, 30),      #글자위치
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),  #흰색
            2
        )
    else:
        cv2.putText(  #이미지에 글씨
            frame,
            'mode: none',
            (200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),  #흰색
            2
        )

    cv2.imshow('video', frame)

    #30ms 동안 키 입력을 기다립니다.
    key = cv2.waitKey(30) & 0xFF

    #q를 누르면 종료
    if key == ord('q'):
        break

    #0을 누르면 원본 영상
    elif key == ord('0'):
        mode = 0

    #1을 누르면 Blur 효과
    elif key == ord('1'):
        mode = 1

cap.release()  #동영상 파일을 닫습니다
cv2.destroyAllWindows() #OpenCV 창을 모두 닫습니다