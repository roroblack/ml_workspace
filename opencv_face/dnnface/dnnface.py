import cv2

model = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = 'deploy.prototxt'
# model = 'opencv_face_detector_uint8.pb'
# config = 'opencv_face_detector.pbtxt'

cap = cv2.VideoCapture(0)  # 다른 앱에서 카메라 사용 중지시킴

if not cap.isOpened():
    print('Camera open failed!')
    exit()

net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net Model Load Failed!')
    exit()

while True:
    _, frame = cap.read()
    if frame is None:
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

        cv2.imshow('frame', frame)

        # for end ---------------------------------------------

    if cv2.waitKey(1) == 27:  # Esc 키 누르면 루프 종료됨
        break

    # while end ------------------

cv2.destroyAllWindows()
