import os
import random
import numpy as np
import cv2
from sklearn.ensemble import AdaBoostClassifier
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
    my_bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for _,row in frame.iterrows():
        img = cv2.imread(os.path.join(img_train_path, row[0]), cv2.IMREAD_COLOR)
        img = img[row[5]:row[7], row[4]:row[6]]
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kpts = sift.detect(grayscale, None)
        kpts, descriptor = sift.compute(grayscale, kpts)
        if descriptor is not None:
            my_bow.add(descriptor)
    my_vocabulary = my_bow.cluster()
    np.save('my_voc.npy', my_vocabulary)

def extract_features(frame , path):
    sift = cv2.SIFT_create()
    flannBasedMatcher = cv2.FlannBasedMatcher_create()
    my_bow = cv2.BOWImgDescriptorExtractor(sift, flannBasedMatcher)
    my_vocabulary = np.load('my_voc.npy')
    my_bow.setVocabulary(my_vocabulary)
    imageDescriptors = []
    if path == 'train':
        path = img_train_path
    else:
        path = img_test_path
    for _,row in frame.iterrows():
        img = cv2.imread(os.path.join(path, row[0]), cv2.IMREAD_COLOR)
        # img = img[row[5]:row[7], row[4]:row[6]]
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kpts = sift.detect(grayscale, None)
        imageDescriptor = my_bow.compute(grayscale, kpts)
        imageDescriptors.append(imageDescriptor)
    frame['desc'] = imageDescriptors
    return frame

def train(frame):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    descriptors = frame['desc'].to_numpy()
    labels = frame['class'].to_list()
    x_matrix = np.empty((1, 128))
    y_vec = []
    for n_desc in range(len(descriptors)):
        if descriptors[n_desc] is not None:
            x_matrix = np.vstack((x_matrix, descriptors[n_desc]))
            y_vec.append(labels[n_desc])
    clf.fit(x_matrix[1:],y_vec)
    return clf

def predict( rf, frame ):
    class_predict = []
    descriptors = frame['desc'].to_numpy()
    for desc in descriptors:
        if desc is not None:
            class_predict.append(rf.predict(desc)[0])
        else:
            class_predict.append(0)
    frame['class_pred'] = class_predict
    return frame

def evaluate(frame):
    y_pred = frame['class_pred'].to_list()
    y_real = frame['class'].to_list()
    confusion = confusion_matrix(y_real, y_pred)
    tn, fp, fn, tp = confusion.ravel()
    print(confusion)
    accuracy = 100 * (tp + tn ) / ( tp + tn + fp + fn )
    print("accuracy =", round(accuracy, 2), "%")
    return


print("Read train file")
data_frame_train = make_frame(anno_train_path)
print(data_frame_train['class'].value_counts())
class_change(data_frame_train)
print("Read test file")
# data_frame_test = make_frame(anno_test_path)
# print(data_frame_test['class'].value_counts())
# class_change(data_frame_test)
print("learn bovw")
learn_bovw(data_frame_train)
print("extract features")
data_frame_train = extract_features(data_frame_train, 'train')
print("train")
rf = train(data_frame_train)

# data_frame_test = extract_features( data_frame_test, 'test' )
print("predict")
# data_frame_test = predict(rf, data_frame_test)

#evaluate( data_frame_test )
if 1:
    data_frame_train = predict(rf, data_frame_train)
    evaluate( data_frame_train )
#print(data_frame_test[['filename','class','class_pred']].tail(100))
# images = []
# images1 =[]
# images2 =[]
# for _, row in data_frame_train.tail(25).iterrows():
#     img = cv2.imread(os.path.join(img_train_path, row[0]), cv2.IMREAD_COLOR)
#     img = img[row[5]:row[7], row[4]:row[6]]
#     images1.append(cv2.imread(os.path.join(img_train_path, row[0]), cv2.IMREAD_COLOR)[row[5]:row[7], row[4]:row[6]])
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,
#                     cv2.THRESH_BINARY,3,5)
#     kernel = np.ones((5,5),np.uint8)
#     erosion = cv2.erode(th2,kernel,iterations=1)
#     imgray=cv2.Canny(erosion,30,100)
#     can=cv2.Canny(img,100,200)
#     # circles = cv2.HoughCircles(can, cv2.HOUGH_GRADIENT, 1, 30,
#     #                            param1=100,param2=20,minRadius=10,maxRadius=40)
#     circles = cv2.HoughCircles(can, cv2.HOUGH_GRADIENT, 1, 100,
#                                param1=100,param2=20,minRadius=10,maxRadius=100)
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:
#             # draw the outer circle
#             cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
#             #cv2.rectangle(img, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]), (0, 255, 0), 2)
#             # draw the center of the circle
#             cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
#     # sift = cv2.SIFT_create()
#     # kpts = sift.detect(gray, None)
#     # images.append(cv2.drawKeypoints(gray, kpts, img))
#     images2.append(img)
# # for x in range(5):
# #     plt.figure(0)
# #     plt.subplot(5, 5, x * 5 + 1)
# #     plt.imshow(images[x * 5])
# #     plt.subplot(5, 5, x * 5 + 2)
# #     plt.imshow(images[x * 5 + 1])
# #     plt.subplot(5, 5, x * 5 + 3)
# #     plt.imshow(images[x * 5 + 2])
# #     plt.subplot(5, 5, x * 5 + 4)
# #     plt.imshow(images[x * 5 + 3])
# #     plt.subplot(5, 5, x * 5 + 5)
# #     plt.imshow(images[x * 5 + 4])
# # for x in range(5):
# #     plt.figure(1)
# #     plt.subplot(5, 5, x * 5 + 1)
# #     plt.imshow(images1[x * 5])
# #     plt.subplot(5, 5, x * 5 + 2)
# #     plt.imshow(images1[x * 5 + 1])
# #     plt.subplot(5, 5, x * 5 + 3)
# #     plt.imshow(images1[x * 5 + 2])
# #     plt.subplot(5, 5, x * 5 + 4)
# #     plt.imshow(images1[x * 5 + 3])
# #     plt.subplot(5, 5, x * 5 + 5)
# #     plt.imshow(images1[x * 5 + 4])
# for x in range(5):
#     plt.figure(1)
#     plt.subplot(5, 5, x * 5 + 1)
#     plt.imshow(images2[x * 5])
#     plt.subplot(5, 5, x * 5 + 2)
#     plt.imshow(images2[x * 5 + 1])
#     plt.subplot(5, 5, x * 5 + 3)
#     plt.imshow(images2[x * 5 + 2])
#     plt.subplot(5, 5, x * 5 + 4)
#     plt.imshow(images2[x * 5 + 3])
#     plt.subplot(5, 5, x * 5 + 5)
#     plt.imshow(images2[x * 5 + 4])
# images = []
# for _, row in data_frame_train.tail(25).iterrows():
#     img = cv2.imread(os.path.join(img_train_path, row[0]), cv2.IMREAD_COLOR)
#     # img = img[row[5]:row[7], row[4]:row[6]]
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,
#                     cv2.THRESH_BINARY,3,5)
#     kernel = np.ones((5,5),np.uint8)
#     erosion = cv2.erode(th2,kernel,iterations=1)
#     imgray=cv2.Canny(erosion,100,50)
#     can=cv2.Canny(gray,100,200)
#     # circles = cv2.HoughCircles(th3, cv2.HOUGH_GRADIENT, 1, 30,
#     #                             param1=100, param2=20, minRadius=1, maxRadius=100)
#     # #circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 50,
#     # #                           param1=100,param2=20,minRadius=10,maxRadius=60)
#     circles = cv2.HoughCircles(can, cv2.HOUGH_GRADIENT, 1, 50,
#                                param1=100,param2=50,minRadius=10,maxRadius=100)
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:
#             # draw the outer circle
#             cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
#             #cv2.rectangle(img, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]), (0, 255, 0), 2)
#             # draw the center of the circle
#             cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
#     # sift = cv2.SIFT_create()
#     # kpts = sift.detect(th3, None)
#     # img = cv2.drawKeypoints(th3, kpts, img)
#     images.append(can)
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
# print(data_frame_train.head(25))
#
# plt.show()
