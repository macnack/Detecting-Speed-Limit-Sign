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
                                  'xmin': xmin_, 'ymin': ymin_,
                                  'xmax': xmax_, 'ymax': ymax_})
                    #break turn off repetition
    return pd.DataFrame(items)
# make train dataset
def fill_train_data( frame ): # only with no repetition
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

data_frame_train = make_frame(anno_train_path)
data_frame_test = make_frame(anno_test_path)
# img_speedlimit = data_frame.loc[data_frame['class'] == 'speedlimit']
# xy = (img_speedlimit['filename'].value_counts().keys().tolist(),
#       img_speedlimit['filename'].value_counts().tolist())

print(data_frame_test['heigh'].value_counts())
print(data_frame_train['heigh'].value_counts())
