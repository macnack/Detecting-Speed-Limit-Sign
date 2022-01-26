import os
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import xml.etree.ElementTree as ET
import os
anno_test_path = "./test/annotations/road4.xml"
anno_test_path1 = "./test/annotations/road2.xml"
img_test_path = "./test/images/"
path = "./test/annotations/"
anno_train_path = "./train/annotations/"
img_train_path = "./train/images/"

# select object from a file to pandas row
def make_data_row( file_path ):
    test =ET.parse( file_path ).getroot()
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
                      'heigh': heigh_, 'class' : class_,
                      'xmin' : xmin_, 'ymin': ymin_,
                      'xmax': xmax_, 'ymax': ymax_})
    return items
# from folder create pandas data frame with objects
def make_frame( path ):
    frame = pd.DataFrame()
    items = []
    for entry in os.listdir(path):
        file_path = os.path.join(path, entry)
        if os.path.isfile(file_path):
            if '.xml' in file_path:
                    test =ET.parse( file_path ).getroot()
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
                                        'heigh': heigh_, 'class' : class_,
                                        'xmin' : xmin_, 'ymin': ymin_,
                                        'xmax': xmax_, 'ymax': ymax_})
    return pd.DataFrame(items)

df_train = make_frame(path)
print(df_train.sort_values(by='filename'))
