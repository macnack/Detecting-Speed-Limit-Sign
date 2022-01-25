import os
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
anno_test_path = "./test/annotations/road4.xml"
img_test_path = "./test/images/"

anno_train_path = "./train/annotations/"
img_train_path = "./train/images/"
images_path = Path('./test/images')
anno_path = Path('./test/annotations')

def make_data_row( file_path ):
    test =ET.parse( anno_test_path ).getroot()
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
                      'heigh_': heigh_, 'class' : class_,
                      'xmin' : xmin_, 'ymin': ymin_,
                      'xmax': xmax_, 'ymax': ymax_})
    return items
df_train = make_data_row(anno_test_path)
print(pd.DataFrame(df_train))
