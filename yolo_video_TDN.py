import time
import cv2
import numpy as np
import os
from math import sqrt

image_path = '1.mp4'  ######path to image
config_path = 'yoloFile/yolov4.cfg'  #####path to config file (.cfg)
weight_path = 'yoloFile/yolov4.weights'  #####path to weights file
class_name_path = 'yoloFile/yolov3.txt'  #####path_to_class_name
spaceScale = 1.1


####note: Nếu dùng yolov4.cfg thì phải dùng yolov4.weights và ngược lại, class name thì giữ nguyên vì 2 phiên bản đều có class name giống nhau

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def checkPosition1(x, y):
    return y < (0.10455764075067267 * x + 651.3994638069704)


def checkPosition2(x, y):
    return y > (0.07929749866950715 * x + 562.2245875465672)


def draw_prediction(img, class_id, index, lenght, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) + str(index + 1) + ' ' + str(round(lenght, 2))
    distance = str(round(lenght, 2))

    color = (0, 255, 0)

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    # cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(img, distance, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(img, str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def best_fit_line(x):
    return 1920 - int(-0.3288 * x + 3.0116)


net = cv2.dnn.readNet(weight_path, config_path)
with open(class_name_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

coord_tree = []
with open("Coord_tree/Coord_tree_cam_2.txt", "r") as f:
    for line in f.readlines():
        x, y = int(line.split(",")[0]), int(line.split(",")[1])
        coord_tree.append((x, y))
# print(coord_tree)

tree_distance = dict()
for i in range(len(coord_tree) - 1):
    x1, y1 = coord_tree[i]
    x2, y2 = coord_tree[i + 1]
    tree_distance[i + 1] = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
# print(tree_distance)

tree_real_distance = {1: 6.52, 2: 6.76, 3: 7.8, 4: 5, 5: 6.79}
cap = cv2.VideoCapture("1.mp4")
# frame = 1
imageIndex = 0
# with open("CSV/center_point.csv", "w+") as f:
#     f.write('Name,Coord\n')

# while imageIndex <= len(os.listdir("TDN cam 2")):
frame = 0
while True:
    frame += 1
    if frame % 100 != 1:
        continue
    # image = cv2.imread(f"TDN cam 2/{os.listdir('TDN cam 2')[imageIndex]}")
    ret, image = cap.read()
    image_copy = image.copy()
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 1 / 255

    ##########

    blob = cv2.dnn.blobFromImage(image, scale, (608, 608), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))
    ###############################
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.3
    nms_threshold = 0.4

    # Thực hiện xác định bằng HOG và SVM
    start = time.time()

    for coord in coord_tree:
        cv2.circle(image, coord, 3, (0, 0, 255), 5)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id in [2, 7]:
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    if checkPosition1(center_x, center_y) == False or checkPosition2(center_x, center_y) == False:
                        continue
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # print(indices)
    coordition = dict()
    for index, i in enumerate(indices):
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        coordition[center_x] = [x, y, w, h, center_x, center_y, i]
        cv2.imwrite(f"Crop/Crop_{index}.jpg", image_copy[int(y):int(y + h), int(x):int(x + w)])
    # with open("CSV/center_point.csv", "a+") as f:
    #     f.write(f"TDN cam 2/{os.listdir('TDN cam 2')[imageIndex]}")
    for index, key in enumerate(sorted(coordition)):
        x, y, w, h, center_x, center_y, i = coordition[sorted(coordition)[index]]
        cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), 5)


        # with open("CSV/center_point.csv", "a+") as f:
        #     f.write(f",{center_x} {center_y}")
        # with open("CSV/center_point.csv", "a+") as f:
        #     f.write(f"\n")

        lenght = 0
        for dis in range(len(coord_tree) - 1):
            x1_tree = coord_tree[dis][0]
            x2_tree = coord_tree[dis + 1][0]
            if x1_tree <= x <= x2_tree:
                lenght = w / tree_distance[dis + 1] * tree_real_distance[dis + 1]
        draw_prediction(image, class_ids[i], index, lenght, confidences[i], round(x), round(y), round(x + w),
                        round(y + h))
        cv2.imwrite("cam2_TDN_process.jpg", image)

        if index != len(sorted(coordition)) - 1:
            x1, y1, w1, h1, center_x1, center_y1, i1 = coordition[sorted(coordition)[index + 1]]
            space = int((x1 - x - w) / (spaceScale * (w + w1) / 2))
            if space >= 1:
                cv2.putText(image, f"{space} spaces!",
                            (int(x + w + (x1 - x - w) / 2 - 70), int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                for spaceBox in range(space):
                    cv2.rectangle(image, (
                        int(x + w * spaceScale + (w + w1) / 2 * spaceBox), int(y * (1.01 + 0.01 * spaceBox))), (
                                      int(x + w * spaceScale + (w + w1) / 2 + (w + w1) / 2 * spaceBox),
                                      int((y * (1.01 + 0.01 * spaceBox) * 2 + h + h1) / 2)), (0, 0, 255),
                                  2)
        if index == 0 and (x / (w * spaceScale)) >= 1:
            space = int((x / (w * spaceScale)))
            cv2.putText(image, f"{space} spaces!",
                        (int(x / 2 - 70), int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            for spaceBox in range(space):
                cv2.rectangle(image, (
                    int(w * spaceScale + (w + 0) / 2 * spaceBox), int(y * (1.01 + 0.01 * spaceBox))), (
                                  int(w * spaceScale + (w + 0) / 2 + (w + 0) / 2 * spaceBox),
                                  int((y * (1.01 + 0.01 * spaceBox) * 2 + h + 0) / 2)), (0, 0, 255),
                              2)
        elif index == len(sorted(coordition)) - 1 and (Width - (x + w)) / (w * spaceScale) >= 1:
            space = int((Width - (x + w)) / (w * spaceScale))
            cv2.putText(image, f"{space} spaces!",
                        (int(x + w + (Width - x - w) / 2 - 70), int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            for spaceBox in range(space):
                cv2.rectangle(image, (
                    int(x + w * spaceScale + (w + Width) / 2 * spaceBox), int(y * (1.01 + 0.01 * spaceBox))), (
                                  int(x + w * spaceScale + (w + Width) / 2 + (w + Width) / 2 * spaceBox),
                                  int((y * (1.01 + 0.01 * spaceBox) * 2 + h + 0) / 2)), (0, 0, 255),
                              2)
    end = time.time()
    print("YOLO Execution time: " + str(end - start))
    image = cv2.resize(image, dsize=None, fx=0.7, fy=0.7)
    cv2.imshow("object detection", image)
    imageIndex += 1
    # cv2.imwrite("image_TDN cam 2.jpg", image)
    key = cv2.waitKey(1)
    # if key == ord('c'):
    #     imageIndex += 1
    #     continue
    # elif key == ord('b'):
    #     imageIndex -= 1
    if key == ord("q"):
        break
