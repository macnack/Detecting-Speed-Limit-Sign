import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import os
from matplotlib import pyplot as plt

# from sklearn.ensemble import AdaBoostClassifier

# TODO Jakość kodu i raport (4/4)


# TODO Skuteczność klasyfikacji 0.0 (0/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0

# stderr:
# Traceback (most recent call last):
#   File "main.py", line 429, in <module>
#     main()
#   File "main.py", line 423, in main
#     classify(rf)
#   File "main.py", line 372, in classify
#     classify_frame = input_classify()
#   File "main.py", line 252, in input_classify
#     bounds = list(map(int, input(bounds_str).split()))
# NameError: name 'bounds_str' is not defined

# TODO Skuteczność detekcji mAP = 0.0 (0/6) (0/2)

# TODO max(0, 0+0) = 0

# file paths
anno_test_path = "../test/annotations/"
img_test_path = "../test/images/"

anno_train_path = "../train/annotations/"
img_train_path = "../train/images/"

data_xml_path = Path("../dataset/annotations/")
data_img_path = Path("../dataset/images/")

# factorial of image
alpha = 1 / 10
# nnm threshold
overlapThresh = 0.3
# images to show
images = []
# draw the box
Draw = False
Output = False
# 'gui'
if Output is True:
    welcome_str = 'detect or classify ( repeat )'
    filename_str = 'filename: '
    number_files_str = 'number of files: '
    number_obj_str = 'number of objects: '
    bounds_str = 'bounds: '
else:
    welcome_str = ''
    filename_str = ''
    number_files_str = ''
    number_obj_str = ''
    bounds_str = ''


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
                    items.append({'filename': filename_, 'width': int(width_),
                                  'heigh': int(heigh_), 'class': class_,
                                  'xmin': int(xmin_), 'ymin': int(ymin_),
                                  'xmax': int(xmax_), 'ymax': int(ymax_)})
                    # break if turn off repetition
    return pd.DataFrame(items)


# to test classify method
def make_input_frame(path):
    items = []
    for entry in os.listdir(path):
        file_path = os.path.join(path, entry)
        if os.path.isfile(file_path):
            if '.xml' in file_path:
                test = ET.parse(file_path).getroot()
                for name in test.findall("./object"):
                    filename_ = test.find("./filename").text
                    # width_ = test.find("./size/width").text
                    # heigh_ = test.find("./size/height").text
                    class_ = name.find("./name").text
                    xmin_ = name.find("./bndbox/xmin").text
                    ymin_ = name.find("./bndbox/ymin").text
                    xmax_ = name.find("./bndbox/xmax").text
                    ymax_ = name.find("./bndbox/ymax").text
                    items.append({'filename': filename_, 'bounds': ((int(xmin_), int(xmax_)),
                                                                    (int(ymin_), int(ymax_))), 'class': class_})
                    # break if turn off repetition
    return pd.DataFrame(items)


# classification other and speedlimit (0, 1)
def class_change(frame):
    # TODO Przydalyby sie tez przyklady np. tla czy innych obiektow w klasie "other".
    class_new_dict = {'trafficlight': 0, 'speedlimit': 1, 'stop': 0, 'crosswalk': 0, 'other': 0, 'background': 0}
    frame['class'] = frame['class'].apply(lambda x: class_new_dict[x])


# make random shuffle train dataset
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
        _, descriptor = sift.compute(grayscale, sift.detect(grayscale, None))
        if descriptor is not None:
            my_bow.add(descriptor)
    my_vocabulary = my_bow.cluster()
    np.save('my_voc.npy', my_vocabulary)  # vocabulary


# extraxt deskriptor from frame
def extract_features(frame, path):
    sift = cv2.SIFT_create()
    my_bow = cv2.BOWImgDescriptorExtractor(sift, cv2.FlannBasedMatcher_create())
    my_vocabulary = np.load('my_voc.npy')
    my_bow.setVocabulary(my_vocabulary)
    imageDescriptors = []
    flag = False
    if path == 'train':
        path = img_train_path
    elif path == 'terminal':
        path = img_test_path
        flag = True
    else:
        path = img_test_path

    for _, row in frame.iterrows():
        img = cv2.imread(os.path.join(path, row[0]), cv2.IMREAD_COLOR)
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not flag:
            grayscale = grayscale[row[5]:row[7], row[4]:row[6]]
        if flag:
            ((xmin, xmax), (ymin, ymax)) = row[1]  # ( (xmin, xmax), (ymin, ymax) )
            grayscale = grayscale[ymin:ymax, xmin:xmax]
        imageDescriptor = my_bow.compute(grayscale, sift.detect(grayscale, None))
        imageDescriptors.append(imageDescriptor)
    frame['desc'] = imageDescriptors
    return frame


# train model
def train(frame):
    # clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf = RandomForestClassifier()
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


# predict value
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


# compute accuracy
def evaluate(frame):
    confusion = confusion_matrix(frame['class'].to_list(), frame['class_pred'].to_list())
    tn, fp, fn, tp = confusion.ravel()
    print(confusion)
    print("accuracy =", round(100 * (tp + tn) / (tp + tn + fp + fn), 2), "%")
    return


def output(filename, bounds):
    print(filename)
    print(len(bounds))
    if len(bounds) != 0:
        for bound in bounds:
            print(bound[0][0], bound[1][0], bound[0][1], bound[1][1])
    return


# output for frame
def output_(frame):
    filename = frame['filename'].to_list()
    bounds = frame['find_bounds'].to_list()
    for i in range(len(filename)):
        print(filename[i])
        print(len(bounds[i]))
        for j in range(len(bounds[i])):
            ((xmin, ymin), (xmax, ymax)) = bounds[i][j]
            print(xmin, xmax, ymin, ymax)
    return 1


# output for frame with bounds
def output_with_box(frame):
    filename = frame['filename'].to_list()
    bounds = frame['find_bounds'].to_list()
    class_pred = frame['class_pred'].to_list()
    for i in range(len(filename)):
        print(filename[i])
        print(sum(class_pred[i]))
        for j in range(len(class_pred[i])):
            if class_pred[i][j] == 1:
                ((xmin, ymin), (xmax, ymax)) = bounds[i][j]
                print(xmin, xmax, ymin, ymax)
    return 1


# method classify from terminal
def input_classify():
    to_do_list = []
    # 'number of files: '
    loop = input(number_files_str)
    for x in range(int(loop)):
        # text 'filename: '
        filename = input(filename_str)
        # text 'number of objects: '
        number_of_objects = input(number_obj_str)
        for y in range(int(number_of_objects)):
            # text 'bounds: '
            bounds = list(map(int, input(bounds_str).split()))
            tuple_ = ((bounds[0], bounds[2]), (bounds[1], bounds[3]))
            to_do_list.append({'filename': str(filename), 'bounds': tuple_})
    return pd.DataFrame(to_do_list)


# NMS algorithm
def non_max_suppression(x_min, x_max, y_min, y_max):
    choose = []
    area = (x_max - x_min + 1) * (y_max - y_min + 1)
    index = np.argsort(y_max)
    while len(index) > 0:
        last_index = len(index) - 1
        i = index[last_index]
        choose.append(i)
        suppression = [last_index]
        for pos in range(0, last_index):
            j = index[pos]
            width = max(0, min(x_max[i], x_max[j]) - max(x_min[i], x_min[j]) + 1)
            height = max(0, min(y_max[i], y_max[j]) - max(y_min[i], y_min[j]) + 1)
            overlap = float(width * height) / area[j]
            if overlap > overlapThresh:
                suppression.append(pos)
        index = np.delete(index, suppression)
    boxes = []
    for pick in choose:
        box = ((x_min[pick], y_min[pick]), (x_max[pick], y_max[pick]))
        boxes.append(box)
    return boxes


# used for detection
def extract_and_predict_loc(rf, frame):
    filename = frame['filename'].to_list()
    bounds = frame['find_bounds'].to_list()
    class_predict = []
    for i in range(len(filename)):
        img = cv2.imread(os.path.join(img_test_path, filename[i]), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        my_bow = cv2.BOWImgDescriptorExtractor(sift, cv2.FlannBasedMatcher_create())
        my_vocabulary = np.load('my_voc.npy')
        my_bow.setVocabulary(my_vocabulary)
        class_predict_ = []
        for ((xmin, ymin), (xmax, ymax)) in bounds[i]:
            img_gray_part = gray[ymin:ymax, xmin:xmax]
            imageDescriptor = my_bow.compute(img_gray_part, sift.detect(img_gray_part, None))
            if imageDescriptor is not None:
                class_predict_.append(rf.predict(imageDescriptor)[0])
            else:
                class_predict_.append(0)
        class_predict.append(class_predict_)
    frame['class_pred'] = class_predict
    return frame


# Hough Tranfrom for localize circle in img
def localization(frame):
    filename = 'road.png'
    new_frame = []
    for _, row in frame.iterrows():
        img = cv2.imread(os.path.join(img_test_path, row[0]), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT_ALT, 1.5, 10,
                                   param1=300, param2=0.85, minRadius=1, maxRadius=100)
        if filename is not row[0]:
            filename = row[0]
            xmin_ = np.array([0])
            ymin_ = np.array([0])
            xmax_ = np.array([0])
            ymax_ = np.array([0])
            if circles is not None:
                boxes = []
                count = 0
                for i in circles[0, :]:
                    xmin = np.array(np.uint16(np.around(max(i[0] - i[2], 0))))
                    ymin = np.array(np.uint16(np.around(max(i[1] - i[2], 0))))
                    xmax = np.array(np.uint16(np.around(min(i[0] + i[2], img.shape[1]))))
                    ymax = np.array(np.uint16(np.around(min(i[1] + i[2], img.shape[0]))))
                    ile = alpha * img.shape[1]
                    # TODO Tutaj chyba powinno byc "xmax - xmin >= alpha * img.shape[1]". - Checked MK
                    if xmax - xmin >= alpha * img.shape[1] and ymax - ymin >= alpha * img.shape[0]:
                        box = ((xmin, ymin), (xmax, ymax))
                        count += 1
                        boxes.append(box)
                        xmin_ = np.vstack((xmin_, xmin))
                        xmax_ = np.vstack((xmax_, xmax))
                        ymin_ = np.vstack((ymin_, ymin))
                        ymax_ = np.vstack((ymax_, ymax))
                    # draw the outer circle
                    if Draw:
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    images.append(img)
                if len(xmin_) != 1:
                    boxes = non_max_suppression(xmin_[1:, 0], xmax_[1:, 0], ymin_[1:, 0], ymax_[1:, 0])
            new_frame.append({'filename': filename, 'find_bounds': boxes})
            # output(filename, boxes)
    return pd.DataFrame(new_frame)


def output_predict(frame):
    class_predict = frame['class_pred']
    filename = frame['filename'].to_list()
    class_predict = frame['class_pred'].to_list()
    for i in range(len(filename)):
        print(filename[i])
        item = class_predict[i]
        if item == 1:
            print('speedlimit')
        else:
            print('other')
    # for item in class_predict:
    #     if item == 1:
    #         print('speedlimit')
    #     else:
    #         print('other')
    return 1


def classify(rf):
    # classify = [{'filename': 'road192.png', 'bounds': ((133, 205), (223, 300))},
    #             {'filename': 'road545.png', 'bounds': ((77, 198), (122, 241))},
    #             {'filename': 'road545.png', 'bounds': ((140, 245), (209, 314))},
    #             {'filename': 'road545.png', 'bounds': ((56, 315), (100, 359))},
    #             {'filename': 'road259.png', 'bounds': ((139, 65), (214, 138))},
    #             {'filename': 'road161.png',
    #              'bounds': ((178, 50), (230, 102))}]  # {'filename': 'road832.png', 'bounds': ((74, 42), (196, 161))},
    # classify = make_input_frame(anno_test_path)
    # classify_frame = pd.DataFrame(classify)
    # class_change(classify_frame)
    classify_frame = input_classify()
    classify_frame = extract_features(classify_frame, 'terminal')
    classify_frame = predict(rf, classify_frame)
    output_predict(classify_frame)
    return 1


def detect(rf, frame):
    located = localization(frame)
    located = extract_and_predict_loc(rf, located)
    output_with_box(located)
    #output_(located)


def plot(images_):
    for x in range(5):
        plt.figure(0)
        plt.subplot(5, 5, x * 5 + 1)
        plt.imshow(images_[x * 5])
        plt.subplot(5, 5, x * 5 + 2)
        plt.imshow(images_[x * 5 + 1])
        plt.subplot(5, 5, x * 5 + 3)
        plt.imshow(images_[x * 5 + 2])
        plt.subplot(5, 5, x * 5 + 4)
        plt.imshow(images_[x * 5 + 3])
        plt.subplot(5, 5, x * 5 + 5)
        plt.imshow(images_[x * 5 + 4])
    plt.show()


def main():
    # print("Read train file")
    data_frame_train = make_frame(anno_train_path)
    # print(data_frame_train['class'].value_counts())
    class_change(data_frame_train)
    # print("Read test file")
    data_frame_test = make_frame(anno_test_path)
    # print(data_frame_test['class'].value_counts())
    class_change(data_frame_test)
    # print("learn bovw")
    learn_bovw(data_frame_train)
    # print("extract features")
    data_frame_train = extract_features(data_frame_train, 'train')
    # print("train")
    rf = train(data_frame_train)
    # print("predict")
    data_frame_test = extract_features(data_frame_test, 'test')
    data_frame_test = predict(rf, data_frame_test)
    # evaluate(data_frame_test)

    # tekst "detect or classify ( repeat ) "
    if input(welcome_str) == 'classify':
        classify(rf)
    else:
        detect(rf, data_frame_test)


if __name__ == '__main__':
    main()
