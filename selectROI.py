import cv2
import csv

IMAGE_SOURCE_PATH = "C9.png"

# cap = cv2.VideoCapture(VIDEO_SOURCE_PATH)
# suc, image = cap.read()

# cv2.imwrite("frame0.jpg", image)
# cv2.namedWindow("frame0", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("frame0", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
img = cv2.imread(IMAGE_SOURCE_PATH)
img = cv2.resize(img, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)

r = cv2.selectROIs('ROI Selector', img, showCrosshair=False, fromCenter=False)

rlist = r.tolist()
for i in range(0, len(rlist)):
    rlist[i].append(i + 1)
print(rlist)

with open('ROI/rois.csv', 'w', newline='') as outf:
    csvw = csv.writer(outf)
    csvw.writerows(rlist)

cv2.destroyAllWindows()