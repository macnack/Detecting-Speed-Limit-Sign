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
path = "./train/annotations/"
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
#print( len((df_train.loc[df_train['class'] == 'speedlimit' ]['filename']).value_counts()) )
#print( len((df_train.loc[df_train['class'] == 'speedlimit' ]['filename'])) )
#print( df_train['filename'].value_counts().values)
#print(len((df_train['filename'].value_counts().axes).value))


# for filename in (df_train['filename'].value_counts().axes):
#      for x in filename.values:
#           print(x)
#      print(len(filename.values))
# for y in df_train['filename'].value_counts():
#     print(y)
# print( df_train['filename'].value_counts() )
img_speedlimit = df_train.loc[df_train['class'] == 'speedlimit']
xy = ( img_speedlimit['filename'].value_counts().keys().tolist(),  img_speedlimit['filename'].value_counts().tolist() )
# for ax in range(len(xy[0])):
#     print( xy[0][ax] )
#     print( xy[1][ax] )
# print(sum(xy[1])*0.2)

for ax in range(78):
    print( xy[0][ax] )
    print( xy[1][ax] )
#print(df_train.value_counts())
