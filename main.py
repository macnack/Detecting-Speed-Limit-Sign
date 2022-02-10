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


#file paths
anno_test_path = "./test/annotations/"
img_test_path = "./test/images/"

anno_train_path = "./train/annotations/"
img_train_path = "./train/images/"

data_xml_path = Path("./dataset/annotations/")
data_img_path = Path("./dataset/images/")


# select object from a file to pandas dataframe
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


# import from xml file....
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

#classification other and speedlimit (0, 1)
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

# learn bovw
def learn_bovw(frame):
    dict_size = 128
    my_bow = cv2.BOWKMeansTrainer(dict_size)
    sift = cv2.SIFT_create()
    for _, row in frame.iterrows():
        img = cv2.imread(os.path.join(img_train_path, row[0]), cv2.IMREAD_COLOR)
        img = img[row[5]:row[7], row[4]:row[6]]
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kpts = sift.detect(grayscale, None)
        kpts, descriptor = sift.compute(grayscale, kpts)
        if descriptor is not None:
            my_bow.add(descriptor)
    my_vocabulary = my_bow.cluster()
    np.save('my_voc.npy', my_vocabulary) # vocabulary


def extract_features(frame, path):
    sift = cv2.SIFT_create()
    flannBasedMatcher = cv2.FlannBasedMatcher_create()
    my_bow = cv2.BOWImgDescriptorExtractor(cv2.SIFT_create(), cv2.FlannBasedMatcher_create())
    my_vocabulary = np.load('my_voc.npy')
    my_bow.setVocabulary(my_vocabulary)
    imageDescriptors = []
    if path == 'train':
        path = img_train_path
    else:
        path = img_test_path
    for _, row in frame.iterrows():
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
    clf.fit(x_matrix[1:], y_vec)
    return clf


def predict(rf, frame):
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
    accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)
    print("accuracy =", round(accuracy, 2), "%")
    return


def output(filename, n_objects, bounds):
    print(filename)
    print(n_objects)
    if n_objects != 0:
        for bound in bounds:
            print(bound[0][0], ' ', bound[1][0], ' ', bound[0][1], ' ', bound[1][1])
    return
#method classify from terminal
def classify():
    to_do_list = {}
    loop = input('number of files: ')
    for x in range(int(loop)):
        filename = input('filename: ')
        number_of_objects = input('number of objects: ')
        bounds_box = []
        for y in range(int(number_of_objects)):
            bounds = list(map(int, input('bounds').split()))
            bounds_box.append(((bounds[0], bounds[2]), (bounds[1], bounds[3])))
        to_do_list.update({filename : bounds_box})
    return to_do_list

print("Read train file")
data_frame_train = make_frame(anno_train_path)
print(data_frame_train['class'].value_counts())
class_change(data_frame_train)
print("Read test file")
data_frame_test = make_frame(anno_test_path)
print(data_frame_test['class'].value_counts())
class_change(data_frame_test)
print("learn bovw")
learn_bovw(data_frame_train)
print("extract features")
data_frame_train = extract_features(data_frame_train, 'train')
print("train")
rf = train(data_frame_train)

# data_frame_test = extract_features( data_frame_test, 'test' )
print("predict")
# data_frame_test = predict(rf, data_frame_test)

# evaluate( data_frame_test )
if 0:
    data_frame_train = predict(rf, data_frame_train)
    evaluate(data_frame_train)
# print(data_frame_test[['filename','class','class_pred']].tail(100))
images = []


# for _, row in data_frame_test.iloc[75:100].iterrows():
for _, row in data_frame_test.iloc[54:79].iterrows():
    img = cv2.imread(os.path.join(img_test_path, row[0]), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT_ALT, 1.5, 10,
                               param1=300, param2=0.85, minRadius=1, maxRadius=100)
    print("Name: ", row[0])
    if circles is not None:
        boxes = []
        circles = np.uint16(np.around(circles))
        count = 0
        for i in circles[0, :]:
            box = ((i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]))
            boxes.append(box)
            # draw the outer circle
            cv2.rectangle(img, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]), (255, 0, 0), 2)
            ## Is this speedlimit ?? detection
            sift = cv2.SIFT_create()
            flannBasedMatcher = cv2.FlannBasedMatcher_create()
            my_bow = cv2.BOWImgDescriptorExtractor(sift, flannBasedMatcher)
            my_vocabulary = np.load('my_voc.npy')
            my_bow.setVocabulary(my_vocabulary)
            img_gray_part = gray[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            kpts = sift.detect(img_gray_part, None)
            imageDescriptor = my_bow.compute(img_gray_part, kpts)
            if imageDescriptor is not None:
                count += 1

    images.append(img)
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
plt.figure(1)
box = boxes[1]
img = cv2.imread("wies_pl.jpeg", cv2.IMREAD_COLOR)
img = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
plt.imshow(img)

sift = cv2.SIFT_create()
flannBasedMatcher = cv2.FlannBasedMatcher_create()
my_bow = cv2.BOWImgDescriptorExtractor(sift, flannBasedMatcher)
my_vocabulary = np.load('my_voc.npy')
my_bow.setVocabulary(my_vocabulary)
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kpts = sift.detect(grayscale, None)
imageDescriptor = my_bow.compute(grayscale, kpts)
img = cv2.drawKeypoints(grayscale, kpts, img)
plt.figure(2)
plt.imshow(img)
print("predict")
print(rf.predict(imageDescriptor))
plt.show()
