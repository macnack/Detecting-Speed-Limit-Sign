import os
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import os
from matplotlib import pyplot as plt
import random

anno_test_path = "./test/annotations/"
img_test_path = "./test/images/"

anno_train_path = "./train/annotations/"
img_train_path = "./train/images/"

data_xml_path = Path("./dataset/annotations/")
data_img_path = Path("./dataset/images/")


# select object from a file to pandas row
def make_data_row(file_path):
    test = ET.parse(file_path).getroot()
    items = []
    for name in test.findall("./object"):
        filename_ = test.find("./filename").text
        width_ = test.find("./size/width").text
        heigh_ = test.find("./size/height").text
        class_ = name.find("./name").text
        xmin_ = name.find("./bndbox/xmin").text
        ymin_ = name.find("./bndbox/ymin").text
        xmax_ = name.find("./bndbox/xmax").text
        ymax_ = name.find("./bndbox/ymax").text
        items.append({'filename': filename_, 'width': width_,
                      'heigh': heigh_, 'class': class_,
                      'xmin': xmin_, 'ymin': ymin_,
                      'xmax': xmax_, 'ymax': ymax_})
    return items


# from folder create pandas data frame with objects

def make_frame(path):
    items = []
    for entry in os.listdir(path):
        file_path = os.path.join(path, entry)
        if os.path.isfile(file_path):
            if '.xml' in file_path:
                test = ET.parse(file_path).getroot()
                for name in test.findall("./object"):
                    filename_ = test.find("./filename").text
                    width_ = test.find("./size/width").text
                    heigh_ = test.find("./size/height").text
                    class_ = name.find("./name").text
                    xmin_ = name.find("./bndbox/xmin").text
                    ymin_ = name.find("./bndbox/ymin").text
                    xmax_ = name.find("./bndbox/xmax").text
                    ymax_ = name.find("./bndbox/ymax").text
                    items.append({'filename': filename_, 'width': width_,
                                  'heigh': heigh_, 'class': class_,
                                  'xmin': int(xmin_), 'ymin': int(ymin_),
                                  'xmax': int(xmax_), 'ymax': int(ymax_)})
                    # break turn off repetition
    return pd.DataFrame(items)


def class_change(frame):
    class_new_dict = {'trafficlight': 0, 'speedlimit': 1, 'stop': 0, 'crosswalk': 0}
    frame['class'] = frame['class'].apply(lambda x: class_new_dict[x])


# make train dataset
def fill_train_data(frame):  # only with no repetition
    for key in frame['class'].value_counts().keys().tolist():
        files = frame.loc[frame['class'] == key]
        random_20 = files.sample(int(len(files) * 0.2))
        file_name = random_20['filename'].value_counts().keys().tolist()
        for item in file_name:
            item = item[:-4]
            img_path = os.path.join(img_train_path, item.__add__('.png'))
            xml_path = os.path.join(anno_train_path, item.__add__('.xml'))
            Path(img_path).rename(Path(os.path.join(img_test_path, item.__add__('.png'))))
            Path(xml_path).rename(Path(os.path.join(anno_test_path, item.__add__('.xml'))))
    return 1


def learn_bovw(frame):
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()

    for sample in frame:
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)


data_frame_train = make_frame(anno_train_path)
class_change(data_frame_train)
data_frame_test = make_frame(anno_test_path)
class_change(data_frame_test)
img = cv2.imread(os.path.join(img_train_path, "road117.png"))
# print(data_frame_train['filename'])

i = 0
images = []
images1 =[]
images2 =[]
for _, row in data_frame_train.head(25).iterrows():
    img = cv2.imread(os.path.join(img_train_path, row[0]), cv2.IMREAD_COLOR)
    img = img[row[5]:row[7], row[4]:row[6]]
    images1.append(cv2.imread(os.path.join(img_train_path, row[0]), cv2.IMREAD_COLOR)[row[5]:row[7], row[4]:row[6]])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,3,5)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(th2,kernel,iterations=1)
    imgray=cv2.Canny(erosion,30,100)
    circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 10,
                               param1=50,param2=20,minRadius=1,maxRadius=40)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            #cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.rectangle(img, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]), (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    sift = cv2.SIFT_create()
    kpts = sift.detect(gray, None)
    images.append(cv2.drawKeypoints(gray, kpts, img))
for x in range(5):
    plt.figure(0)
    plt.subplot(5, 5, x * 5 + 1)
    plt.imshow(images[x * 5])
    plt.subplot(5, 5, x * 5 + 2)
    plt.imshow(images[x * 5 + 1])
    plt.subplot(5, 5, x * 5 + 3)
    plt.imshow(images[x * 5 + 2])
    plt.subplot(5, 5, x * 5 + 4)
    plt.imshow(images[x * 5 + 3])
    plt.subplot(5, 5, x * 5 + 5)
    plt.imshow(images[x * 5 + 4])
for x in range(5):
    plt.figure(1)
    plt.subplot(5, 5, x * 5 + 1)
    plt.imshow(images1[x * 5])
    plt.subplot(5, 5, x * 5 + 2)
    plt.imshow(images1[x * 5 + 1])
    plt.subplot(5, 5, x * 5 + 3)
    plt.imshow(images1[x * 5 + 2])
    plt.subplot(5, 5, x * 5 + 4)
    plt.imshow(images1[x * 5 + 3])
    plt.subplot(5, 5, x * 5 + 5)
    plt.imshow(images1[x * 5 + 4])
# images = []
# for _, row in data_frame_train.head(25).iterrows():
#     img = cv2.imread(os.path.join(img_train_path, row[0]), cv2.IMREAD_COLOR)
#     # img = img[row[5]:row[7], row[4]:row[6]]
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     circles = cv2.HoughCircles(th3, cv2.HOUGH_GRADIENT, 1, 10,
#                                param1=100, param2=20, minRadius=1, maxRadius=50)
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:
#             # draw the outer circle
#             cv2.rectangle(img, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]), (0, 255, 0), 2)
#             # draw the center of the circle
#             cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
#     sift = cv2.SIFT_create()
#     kpts = sift.detect(th3, None)
#     # img = cv2.drawKeypoints(th3, kpts, img)
#     images.append(img)
# for x in range(5):
#     plt.figure(2)
#     plt.subplot(5, 5, x * 5 + 1)
#     plt.imshow(images[x * 5])
#     plt.subplot(5, 5, x * 5 + 2)
#     plt.imshow(images[x * 5 + 1])
#     plt.subplot(5, 5, x * 5 + 3)
#     plt.imshow(images[x * 5 + 2])
#     plt.subplot(5, 5, x * 5 + 4)
#     plt.imshow(images[x * 5 + 3])
#     plt.subplot(5, 5, x * 5 + 5)
#     plt.imshow(images[x * 5 + 4])
print(data_frame_train.head(25))
plt.show()
