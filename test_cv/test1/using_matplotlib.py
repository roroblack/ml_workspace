import matplotlib.pyplot as plt
import cv2

# 컬러 영상 출력
imgBGR = cv2.imread('../images/cat.bmp')
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

plt.axis('off')
# plt.imshow(imgRGB)
# plt.imshow(imgBGR)
# plt.show()


# 그레이스케일로 영상 출력
# imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
imgGray = cv2.imread('../images/cat.bmp', cv2.IMREAD_GRAYSCALE)
# print(type(imgGray), '')

plt.axis('off')
plt.imshow(imgGray, cmap='gray')
plt.show()

# 두 개의 이미지를 함께 출력
plt.subplot(121), plt.axis('off'), plt.imshow(imgRGB)
plt.subplot(122), plt.axis('off'), plt.imshow(imgGray, cmap='gray')
plt.show()
