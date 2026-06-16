import cv2
import sys

# 이미지 파일 불러오기
img = cv2.imread('../images/cat.bmp')
if img is None:
    print('Image not loaded')
    sys.exit()

print(type(img))
print(img.shape)

# 읽은 이미지 출력
cv2.namedWindow('imshow')
# cv2.imshow('cat', img)
cv2.imshow('imshow', img)
cv2.waitKey() # 키보드 입력이 있을 때까지 기다림
# cv2.waitKey(0)


# 창 닫기
cv2.destroyAllWindows()
