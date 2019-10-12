import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 'path to input image/video'
VIDEO = './videos/video1.mp4'

# 'path to yolo config file'
# download https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.cfg
CONFIG = './model3/yolov3-tiny.cfg'

# 'path to text file containing class names'
# download https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
CLASSES = './model1/yolov3.txt'

# 'path to yolo pre-trained weights'
# wget https://pjreddie.com/media/files/yolov3.weights
WEIGHTS = './model3/yolov3-tiny.weights'

print(os.path.exists(CLASSES))
print(os.path.exists(CONFIG))
print(os.path.exists(WEIGHTS))
print(os.path.exists(VIDEO))

# read class names from text file
with open(CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

scale = 0.00392
conf_threshold = 0.5
nms_threshold = 0.05

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3)).astype(int)
# print(f'all colors: {COLORS}')


# function to get the output layer names
# in the architecture
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # print(output_layers)
    return output_layers


# # function to draw bounding box on the detected object with class name
# def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
#     label = str(classes[class_id])
#     color = COLORS[class_id]
#     cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
#     cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def process_image(image, index):
    width = image.shape[1]
    height = image.shape[0]

    # read pre-trained model and config file
    # net = cv2.dnn.readNet(WEIGHTS, CONFIG)
    net = cv2.dnn.readNetFromDarknet(CONFIG, darknetModel=WEIGHTS)

    # create input blob
    # look into the scale variable (wtf is it doing???)
    # blob = cv2.dnn.blobFromImage(image, scale, (320, 320), (0, 0, 0), True, crop=False)
    blob = cv2.dnn.blobFromImage(image, scalefactor=scale, size=(320, 320), mean=(0, 0, 0), swapRB=True)
    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    # for each detection from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            # print(detection)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # original: 0.5
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                # class_ids.append(class_id)
                # confidences.append(float(confidence))
                # boxes.append([x, y, w, h])
                color_rgb = (int(COLORS[class_id][0]), int(COLORS[class_id][1]), int(COLORS[class_id][2]))
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color_rgb, thickness=2)
                cv2.circle(image, (center_x, center_y), 4, color_rgb, thickness=2)
                cv2.putText(image, classes[int(class_id)], (int(x - 10), int(y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, thickness=2)

    # # apply non-max suppression
    # indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    #
    # # go through the detections remaining
    # # after nms and draw bounding box
    # if len(indices) > 0:
    #     for i in indices:
    #         i = i[0]
    #         box = boxes[i]
    #         x = box[0]
    #         y = box[1]
    #         w = box[2]
    #         h = box[3]
    #
    #         draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    #
    #         label = str(classes[class_id])
    #         color = COLORS[class_id]
    #         cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    #         cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Frame", image)


# # open the video file
# cap = cv2.VideoCapture(0)
#
# index = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     # process_image(frame, index)
#     cv2.imshow("Frame", frame)
#     index = index + 1
#
# # release resources
# cv2.destroyAllWindows()qq


cap = cv2.VideoCapture(0)
index = 0
while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
    process_image(frame, index)
    # cv2.imshow("Frame", frame)

    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break
    index += 1

cap.release()
cv2.destroyAllWindows()
