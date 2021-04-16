import time
import cv2
import numpy as np
import os
from math import sqrt
from checkModule import check_car_slot

image_path = '1.mp4'  ######path to image
config_path = 'yoloFile/yolov4.cfg'  #####path to config file (.cfg)
weight_path = 'yoloFile/yolov4.weights'  #####path to weights file
class_name_path = 'yoloFile/yolov3.txt'  #####path_to_class_name
spaceScale = 1.1
disScale = 0.9


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

    color = (0, 0, 255)

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    # cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(img, distance, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # cv2.putText(img, str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


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

while imageIndex <= len(os.listdir("TDN cam 2")):
    # frame = 0
    # while True:
    #     frame += 1
    #     if frame % 100 != 1:
    #         continue
    image = cv2.imread(f"TDN cam 2/{os.listdir('TDN cam 2')[imageIndex]}")
    # ret, image = cap.read()
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

    # for coord in coord_tree:
    #     cv2.circle(image, coord, 3, (0, 0, 255), 5)

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
        # cv2.imwrite(f"Crop/Crop_{index}.jpg", image_copy[int(y):int(y + h), int(x):int(x + w)])
    # with open("CSV/center_point.csv", "a+") as f:
    #     f.write(f"TDN cam 2/{os.listdir('TDN cam 2')[imageIndex]}")

    for index, key in enumerate(sorted(coordition)):
        x, y, w, h, center_x, center_y, i = coordition[sorted(coordition)[index]]
        # cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), 5)

        lenght = 0
        for dis in range(len(coord_tree) - 1):
            x1_tree = coord_tree[dis][0]
            x2_tree = coord_tree[dis + 1][0]
            if x1_tree <= x <= x2_tree:
                lenght = w / tree_distance[dis + 1] * tree_real_distance[dis + 1] * disScale
            elif coord_tree[0][0] > x:
                lenght = w / tree_distance[1] * tree_real_distance[1] * disScale
            elif coord_tree[5][0] < x:
                lenght = w / tree_distance[5] * tree_real_distance[5] * disScale
        draw_prediction(image, class_ids[i], index, lenght, confidences[i], round(x), round(y), round(x + w),
                        round(y + h))

    for index, key in enumerate(sorted(coordition)):
        index += 1
        if index == 1:
            x, y, w, h, center_x, center_y, i = coordition[sorted(coordition)[index - 1]]
            if x > 5:
                lenght2 = sqrt(x ** 2 + (h / 4) ** 2) / tree_distance[1] * \
                          tree_real_distance[1] * disScale
                cv2.putText(image, str(round(lenght2, 2)), (20, int((y + h // 2) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
                real_distances, type_car = check_car_slot(lenght2)
                if not real_distances:
                    continue
                for i, real_distance in enumerate(real_distances):
                    print(type_car[i])
                    pix_distance = real_distance / 0.9 * tree_distance[1] / tree_real_distance[1]
                    x_slot = x - (pix_distance + 5) * (i + 1)
                    y_slot = y - 5 * (i + 1)
                    cv2.rectangle(image, (int(x_slot), int(y_slot)), (int(x_slot + pix_distance), int(y + h)),
                                  (0, 255, 0), 2)
                    cv2.putText(image, type_car[i] +" "+ str(real_distance), (int(x_slot + pix_distance / 2 -70), int(y_slot + h / 2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif index == len(coordition):
            x, y, w, h, center_x, center_y, i = coordition[sorted(coordition)[index - 1]]
            if x + w < image.shape[1] - 5:
                lenght2 = sqrt((x - image.shape[1] + w) ** 2 + (2 * h / 3) ** 2) / tree_distance[4] * \
                          tree_real_distance[4] * disScale
                cv2.putText(image, str(round(lenght2, 2)), (int((x + w + 20)), int((y + h // 2) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
                real_distances, type_car = check_car_slot(lenght2)
                if not real_distances:
                    continue
                for i, real_distance in enumerate(real_distances):
                    print(type_car[i])
                    pix_distance = real_distance / 0.9 * tree_distance[1] / tree_real_distance[1]
                    x_slot = x + (pix_distance) * i + w + 5
                    y_slot = y + 5 * (i + 1)
                    cv2.rectangle(image, (int(x_slot), int(y_slot + 3)), (int(x_slot + pix_distance), int(y + h + 3)),
                                  (0, 255, 0),
                                  2)
                    cv2.putText(image, type_car[i] + " " + str(real_distances), (int(x_slot + pix_distance / 2 - 70), int(y_slot + h / 2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if index > len(coordition) - 1:
            break
        x, y, w, h, center_x, center_y, i = coordition[sorted(coordition)[index - 1]]
        x2, y2, w2, h2, center_x2, center_y2, i2 = coordition[sorted(coordition)[index]]
        # cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), 5)

        lenght2 = 0
        for dis in range(len(coord_tree) - 1):
            x1_tree = coord_tree[dis][0]
            x2_tree = coord_tree[dis + 1][0]
            if x1_tree <= (x + w + x2) / 2 <= x2_tree:
                if x + w < x2:
                    lenght2 = sqrt((x2 - x - w) ** 2 + (y2 + h2 / 2 - y - h / 2) ** 2) / tree_distance[dis + 1] * \
                              tree_real_distance[dis + 1] * disScale
            elif coord_tree[0][0] > (x + w + x2) / 2:
                if x + w < x2:
                    lenght2 = sqrt((x2 - x - w) ** 2 + (y2 + h2 / 2 - y - h / 2) ** 2) / tree_distance[1] * \
                              tree_real_distance[1] * disScale
            elif (x + w + x2) / 2 > coord_tree[5][0]:
                if x + w < x2:
                    lenght2 = sqrt((x2 - x - w) ** 2 + (y2 + h2 / 2 - y - h / 2) ** 2) / tree_distance[5] * \
                              tree_real_distance[5] * disScale
        if lenght2 > 0:
            cv2.putText(image, str(round(lenght2, 2)), (int((x + x2 + w) // 2 - 20), int((y + (h2 - h) // 2) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            real_distances, type_car = check_car_slot(lenght2)
            if not real_distances:
                continue
            for i, real_distance in enumerate(real_distances):
                print(type_car[i])
                pix_distance = real_distance / disScale * tree_distance[dis + 1] / tree_real_distance[dis + 1] - 70
                x_slot = x + pix_distance * i + w + 5
                y_slot = y + 5 * (i + 1)
                # cv2.rectangle(image, (int(x_slot), int(y_slot + 3)), (int(x_slot + pix_distance), int(y + h + 3)),
                #               (0, 255, 0),
                #               2)
                cv2.putText(image, type_car[i] + " " + str(real_distance), (int(x_slot + pix_distance / 2 - 70), int(y_slot + h / 2 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # cv2.imwrite("cam2_TDN_process.jpg", image)
    end = time.time()
    print("YOLO Execution time: " + str(end - start))
    image = cv2.resize(image, dsize=None, fx=0.7, fy=0.7)
    cv2.imshow("object detection", image)
    imageIndex += 1
    # cv2.imwrite("image_TDN cam 2.jpg", image)
    key = cv2.waitKey()
    if key == ord('c'):
        imageIndex += 1
        continue
    elif key == ord('b'):
        imageIndex -= 1
    if key == ord("q"):
        break
