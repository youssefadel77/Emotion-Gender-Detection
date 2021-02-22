# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 01:35:15 2019

@author: youss
"""
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (192, 192, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 7, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset-Mirror/training_set',
                                                 target_size = (192, 192),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset-Mirror/test_set',
                                            target_size = (192, 192),
                                            batch_size = 32,
                                            class_mode = 'categorical')



classifier.fit_generator(training_set,
                         steps_per_epoch = 28730,
                         epochs = 3,
                         validation_data = test_set,
                         validation_steps = 7157)


##Prediction Part
import numpy as np
from keras.preprocessing import image

img_pred = image.load_img('C:/Users/youss/Downloads/test8.jpg', target_size = (192, 192))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)
rslt = classifier.predict(img_pred)
print(rslt)
ind = training_set.class_indices

check = np.where(rslt == np.amax(rslt)) 
print(check[1])

if check[1] == 0:
    prediction = "Angry"
elif check[1] == 1:
    prediction = "Disgust"
elif check[1] == 2:
    prediction = "Fear"
elif check[1] == 3:
    prediction = "Happy"
elif check[1] == 4:
    prediction = "Sad"
elif check[1] == 5:
    prediction = "Surprise"
elif check[1] == 6:
    prediction = "Neutral"   

print(prediction) 







##Prediction Part
"""import numpy as np
from keras.preprocessing import image

img_pred = image.load_img('cat_or_dog_5.jpg', target_size = (64, 64))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)
rslt = classifier.predict(img_pred)
ind = training_set.class_indices
if rslt[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"


##Save model to json
import os
from keras.models import model_from_json

clssf = classifier.to_json()
with open("CatOrDog.json", "w") as json_file:
    json_file.write(clssf)
classifier.save_weights("CorDweights.h5")
print("model saved to disk....")
"""