import cv2
import sys
import numpy as np

# 이미지 합성 처리용 함수 작성 ----------------------------
def overlay(frame, cat, pos):
    if pos[0] < 0 or pos[1] < 0:
        return

    if pos[0] + cat.shape[1] > frame.shape[1] or pos[1] + cat.shape[0] > frame.shape[0]:
        return

    sx = pos[0]
    ex = pos[0] + cat.shape[1]
    sy = pos[1]
    ey = pos[1] + cat.shape[0]

    img1 = frame[sy:ey, sx:ex]  # shape=(h, w, 3)
    img2 = cat[:, :, 0:3]  # shape=(h, w, 3)
    alpha = 1.0 - (cat[:, :, 3] / 255.0)  # shape=(h, w)
    #ww = np.stack((alpha,) * 3, axis=-1)

    img1[:, :, 0] = (img1[:, :, 0] * alpha + img2[:, :, 0] * (1. - alpha)).astype(np.uint8)
    img1[:, :, 1] = (img1[:, :, 1] * alpha + img2[:, :, 1] * (1. - alpha)).astype(np.uint8)
    img1[:, :, 2] = (img1[:, :, 2] * alpha + img2[:, :, 2] * (1. - alpha)).astype(np.uint8)

# ---------------------------------------------------------

model = 'opencv_face_detector_uint8.pb'
config = 'opencv_face_detector.pbtxt'

cap = cv2.VideoCapture(0)  # 다른 앱에서 카메라 사용 중지시킴

if not cap.isOpened():
    print('Camera open failed!')
    exit()

net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net Model Load Failed!')
    exit()

# 합성할 고양이 귀 이미지 불러오기
cat = cv2.imread('cat.png', cv2.IMREAD_UNCHANGED)
if cat is None:
    print('Image Load Failed!')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:  # 리턴된 ret가 false이면
        break;

    # 읽은 프레임을 blob 로 바꿈
    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob) # 생략 가능
    # 모델 실행하고 결과 받음 : 얼굴 인식
    detect = net.forward()

    # 받은 결과를 4차원 배열로 바꿈
    detect = detect[0, 0, :, :]
    (h, w) = frame.shape[:2]

    # 영상에 얼굴에 사각형 박스와 글자 출력 표시
    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.5:  # 정확도가 50% 미만이면 출력 표시 안함
            break

        # 사각형 박스 표시를 위한 lefttop, rightbottom 좌표 계산
        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

        label = 'Face : %4.3f' % confidence
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        # 합성할 이미지 출력 위치 좌표와 face 크기와 맞추기 설정
        fx = (x2 - x1) / cat.shape[1]
        cat2 = cv2.resize(cat, (0, 0), fx=fx, fy=fx)
        pos = (x1, y1 - (y2 - y1) // 4)

        # 합성(중첩) 실행
        overlay(frame, cat2, pos)

        # for end ---------------------------------------------

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:  # Esc 키 누르면 루프 종료됨
        break

    # while end ------------------

cv2.destroyAllWindows()
