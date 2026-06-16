# opencv 로 컴퓨터 카메라 연결

import cv2
import sys

# system 기본 카메라로 부터 cv2.VideoCapture 객체 생성
cap = cv2.VideoCapture(0)

if not cap.isOpened(): # 카메라 열기에 실패했다면 (다른 앱이 카메라를 사용 중인 경우)
    print('Camera Open Failed') # 카메라가 없거나, 카메라 드라이버 미설정
    sys.exit()

# 카메라 프레임 해상도 출력
print('Frame width :', round(cap.get(cv2.CAP_PROP_FRAME_WIDTH )))
print('Frame height :', round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT )))

# 매 프레임 처리 및 화면 출력 (영상 출력)
while True:
    ret, frame = cap.read()
    # frame : 카메라로 부터 읽은 프레임(화면) 정보 저장
    # ret : 읽기 성공 여부 저장 (True | False)

    if not ret: # ret 가 False 이면 (읽기 실패 의미)
        print('Read Failed')
        break

    # 읽은 영상을 화면에 출력 처리
    cv2.imshow('frame', frame) # 윈도우 창이 열리면서 영상이 출력
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 읽은 영상에서 테두리 (경계, edge) 선으로 변환해서 출력
    edge = cv2.Canny(frame, 50, 150)
    cv2.imshow('edge', edge) # 윈도우 창이 열리면서 경계선 처리 영상 출력됨

    if cv2.waitKey(1) == 27: # esc 키의 유니코드 (영상 윈도우 창에서)
        break
# while end ------------------------------------------------

cap.release()
cv2.destroyAllWindows()
