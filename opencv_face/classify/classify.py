'''

이 코드는 openCV DNN 모듈과 사전 학습된 GooogleLeNet 모델을 활용해서
제공된 사진 이미지를 잘 분류하는지 확인하는 프로그램임

'''

# 프로그램 종료 (sys.exit) 사용
import sys

# 배열처리, 최대값 위치 탐색 (argmax) 으로 분류항목 클래스 위치 탐색
import numpy as np
import cv2

# filename = 'space_shuttle.jpg'
# filename = 'beagle.jpg'
# filename = 'cup.jpg'
# filename = 'pineapple.jpg'
# filename = 'scooter.jpg'
filename = 'fish_img.jpg'

if len(sys.argv) > 1:
    filename = sys.argv[1]
    print('sys.argv[1] = ', sys.argv[1], ' == ',  filename)
    # sys.exit(0)

img = cv2.imread(filename)

if img is None:
    print('Image load failed!')
    sys.exit()

# load network : 제공되는 dnn 학습모델과 구성을 다운받아서 사용
net = cv2.dnn.readNet('bvlc_googlenet.caffemodel', 'deploy.prototxt')

if net.empty():
    print('Network Model Load Failed!')
    exit()

# load class names
classNames = None
with open('classification_classes_ILSVRC2012.txt', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


print(type(classNames))
print(classNames)
print(len(classNames))

# Inference : 준비된 이미지를 모델에 적용해서 클래스항목으로 분류되는지 테스트
inputBlob = cv2.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))
net.setInput(inputBlob, 'data')  # 생략해도 됨
prob = net.forward()
# 모델을 통해서 나온 테스트 결과 확인
# print(prob.shape)
# print(type(prob))
# print(prob)

# check result & display
out = prob.flatten()
classId = np.argmax(out)
confidence = out[classId]

# 출력용 문장 만들기
text = '%s (%4.2f%%)' % (classNames[classId], confidence * 100)
print(text)
# 이미지에 출력 처리
cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
