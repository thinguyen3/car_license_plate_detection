# CAR LICENSE PLATE DETECTION

## Introduction
 
License Plate Recognition (LPR) is a problem aimed at identifying vehicles by detecting and recognizing its license plate. It has been broadly used in real life applications such as traffic monitoring systems which include unattended parking lots, automatic toll collection, and criminal pursuit.

The target of this project is to implement a vehicle retrieval system for a Indian surveillance camera, by detecting and recognizing India license plates. It will be useful for vehicle registration and identification, and therefore may further contribute to the possibility of vehicle tracking and vehicle activity analysis.

## Dataset

The dataset was download from Kaggle with the [Dataset Link](https://www.kaggle.com/andrewmvd/car-plate-detection). This dataset contains 433 images with bounding box annotations of the car license plates within the images. After downloading the dataset, I reduced the size of the dataset by manually deleting images from 433 images to 125 images.

## Preprocessing

First, I import the libraries.

```
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, MaxPool2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Input
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
from keras import backend as K
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import cv2
import os
import glob
```
## Import data by uploading the zip file and extract all.

```
from zipfile import ZipFile
file_name = './data125.zip'
with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('Done')
```

I create the variable X containing all the images of cars by resizing them to 200 * 200.

`IMAGE_SIZE = 200`
```
img_dir = "./images" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
files.sort() # Sort the images in alphabetical order to match them to the xml files containing the annotations of the bounding boxes
X=[]
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
    X.append(np.array(img))
```
    
I create the variable y containing all the bounding boxe annotations (label). Before that, I resize the annotations so that it fits the new size of the images (200*200).

```
from lxml import etree
def resizeannotation(f):
    tree = etree.parse(f)
    for dim in tree.xpath("size"):
        width = int(dim.xpath("width")[0].text)
        height = int(dim.xpath("height")[0].text)
    for dim in tree.xpath("object/bndbox"):
        xmin = int(dim.xpath("xmin")[0].text)/(width/IMAGE_SIZE)
        ymin = int(dim.xpath("ymin")[0].text)/(height/IMAGE_SIZE)
        xmax = int(dim.xpath("xmax")[0].text)/(width/IMAGE_SIZE)
        ymax = int(dim.xpath("ymax")[0].text)/(height/IMAGE_SIZE)
    return [int(xmax), int(ymax), int(xmin), int(ymin)]
```
    
Create a function resizeannotation.

```path = './annotations'
text_files = ['./annotations/'+f for f in sorted(os.listdir(path))]
y=[]
for i in text_files:
    y.append(resizeannotation(i))
resizeannotation("./annotations/Cars17.xml")
```

## Prepare the data for the CNN.

```
#Transforming in array
X=np.array(X)
y=np.array(y)
#Renormalisation
X = X / 255
y = y / 255
# Seed for reproducing same results
seed = 42
np.random.seed(seed)

## Split the dataset in two: training set/testing set.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
```

## Implemented many types of neural networks, various hyperparameters, gradient descent, activation functions, cost functions, using of karas tuner, using pre-trained CNNs (Resnet50, VGG16, VGG19), VGG16 models showed the best accuracy score with the recognition rate is 84%. 

### VGG16 pre-trained model

VGG-16 is a convolutional neural network that is 16 layers deep. We can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.

```
# Create the model VGG16
# Destroys the current TF graph and creates a new one. Useful to avoid clutter from old models / layers.
from keras import backend as K
K.clear_session()
model = Sequential()
model.add(VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="sigmoid"))
model.layers[-6].trainable = False
model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 vgg16 (Functional)          (None, 6, 6, 512)         14714688  
                                                                 
 flatten (Flatten)           (None, 18432)             0         
                                                                 
 dense (Dense)               (None, 128)               2359424   
                                                                 
 dense_1 (Dense)             (None, 128)               16512     
                                                                 
 dense_2 (Dense)             (None, 64)                8256      
                                                                 
 dense_3 (Dense)             (None, 4)                 260       
                                                                 
=================================================================
Total params: 17,099,140
Trainable params: 2,384,452
Non-trainable params: 14,714,688
```
```
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=20, verbose=1)
# evaluate the keras model
_,accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
Epoch 1/30
5/5 [==============================] - 51s 11s/step - loss: 0.0741 - accuracy: 0.3100 - val_loss: 0.0330 - val_accuracy: 0.4800
Epoch 2/30
5/5 [==============================] - 50s 11s/step - loss: 0.0265 - accuracy: 0.2700 - val_loss: 0.0187 - val_accuracy: 0.5600
Epoch 3/30
5/5 [==============================] - 51s 11s/step - loss: 0.0170 - accuracy: 0.4600 - val_loss: 0.0170 - val_accuracy: 0.6400
Epoch 4/30
5/5 [==============================] - 50s 10s/step - loss: 0.0108 - accuracy: 0.6600 - val_loss: 0.0125 - val_accuracy: 0.8000
Epoch 5/30
5/5 [==============================] - 50s 10s/step - loss: 0.0067 - accuracy: 0.6500 - val_loss: 0.0097 - val_accuracy: 0.7200
Epoch 6/30
5/5 [==============================] - 50s 11s/step - loss: 0.0043 - accuracy: 0.8200 - val_loss: 0.0100 - val_accuracy: 0.5200
Epoch 7/30
5/5 [==============================] - 49s 10s/step - loss: 0.0029 - accuracy: 0.7500 - val_loss: 0.0087 - val_accuracy: 0.6000
Epoch 8/30
5/5 [==============================] - 51s 11s/step - loss: 0.0020 - accuracy: 0.7600 - val_loss: 0.0080 - val_accuracy: 0.6800
Epoch 9/30
5/5 [==============================] - 51s 11s/step - loss: 0.0013 - accuracy: 0.8900 - val_loss: 0.0078 - val_accuracy: 0.8400
Epoch 10/30
5/5 [==============================] - 49s 10s/step - loss: 8.4695e-04 - accuracy: 0.8900 - val_loss: 0.0078 - val_accuracy: 0.8400
Epoch 11/30
5/5 [==============================] - 50s 10s/step - loss: 6.7843e-04 - accuracy: 0.9400 - val_loss: 0.0075 - val_accuracy: 0.7600
Epoch 12/30
5/5 [==============================] - 50s 11s/step - loss: 4.7564e-04 - accuracy: 0.9300 - val_loss: 0.0075 - val_accuracy: 0.7600
Epoch 13/30
5/5 [==============================] - 49s 10s/step - loss: 3.7235e-04 - accuracy: 0.9400 - val_loss: 0.0075 - val_accuracy: 0.7200
Epoch 14/30
5/5 [==============================] - 50s 10s/step - loss: 2.7135e-04 - accuracy: 0.9100 - val_loss: 0.0074 - val_accuracy: 0.8000
Epoch 15/30
5/5 [==============================] - 54s 12s/step - loss: 2.0763e-04 - accuracy: 0.9400 - val_loss: 0.0073 - val_accuracy: 0.8000
Epoch 16/30
5/5 [==============================] - 53s 11s/step - loss: 1.5414e-04 - accuracy: 0.9700 - val_loss: 0.0073 - val_accuracy: 0.7600
Epoch 17/30
5/5 [==============================] - 50s 11s/step - loss: 1.2715e-04 - accuracy: 0.9600 - val_loss: 0.0073 - val_accuracy: 0.8800
Epoch 18/30
5/5 [==============================] - 51s 11s/step - loss: 1.0480e-04 - accuracy: 0.9700 - val_loss: 0.0073 - val_accuracy: 0.8000
Epoch 19/30
5/5 [==============================] - 50s 11s/step - loss: 1.0818e-04 - accuracy: 0.9800 - val_loss: 0.0073 - val_accuracy: 0.8000
Epoch 20/30
5/5 [==============================] - 50s 10s/step - loss: 1.0203e-04 - accuracy: 0.9800 - val_loss: 0.0073 - val_accuracy: 0.7600
Epoch 21/30
5/5 [==============================] - 52s 11s/step - loss: 8.9072e-05 - accuracy: 0.9700 - val_loss: 0.0073 - val_accuracy: 0.8800
Epoch 22/30
5/5 [==============================] - 50s 11s/step - loss: 9.0992e-05 - accuracy: 0.9500 - val_loss: 0.0073 - val_accuracy: 0.7600
Epoch 23/30
5/5 [==============================] - 51s 11s/step - loss: 8.2724e-05 - accuracy: 0.9800 - val_loss: 0.0073 - val_accuracy: 0.8400
Epoch 24/30
5/5 [==============================] - 51s 11s/step - loss: 8.2961e-05 - accuracy: 0.9700 - val_loss: 0.0073 - val_accuracy: 0.8400
Epoch 25/30
5/5 [==============================] - 50s 11s/step - loss: 8.1690e-05 - accuracy: 0.9800 - val_loss: 0.0073 - val_accuracy: 0.8400
Epoch 26/30
5/5 [==============================] - 50s 10s/step - loss: 7.3216e-05 - accuracy: 0.9800 - val_loss: 0.0073 - val_accuracy: 0.8400
Epoch 27/30
5/5 [==============================] - 51s 11s/step - loss: 8.9932e-05 - accuracy: 0.9700 - val_loss: 0.0073 - val_accuracy: 0.8800
Epoch 28/30
5/5 [==============================] - 49s 10s/step - loss: 7.3488e-05 - accuracy: 0.9800 - val_loss: 0.0073 - val_accuracy: 0.8800
Epoch 29/30
5/5 [==============================] - 50s 11s/step - loss: 7.2524e-05 - accuracy: 0.9700 - val_loss: 0.0073 - val_accuracy: 0.8400
Epoch 30/30
5/5 [==============================] - 50s 11s/step - loss: 9.1677e-05 - accuracy: 0.9800 - val_loss: 0.0072 - val_accuracy: 0.8400
1/1 [==============================] - 10s 10s/step - loss: 0.0072 - accuracy: 0.8400
Accuracy: 84.00
```

    

    
    

