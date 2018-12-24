# -- coding: utf-8 --
"""
Created on Wed Oct  3 01:19:22 2018

@author: mdirs
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier=Sequential();

classifier.add(Convolution2D(64,(4,4),input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(64,(4,4),input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(64,(4,4),input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

 

classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid')) 




classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/mdirs/Downloads/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:/Users/mdirs/Downloads/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=215,
        epochs=25,
        validation_data=test_set,
        validation_steps=32)

classifier.save_weights('C:/Users/mdirs/Downloads/project/ninety5.h5')


# serialize model to JSON
model_json = classifier.to_json()
with open("C:/Users/mdirs/Downloads/project/ninety5_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("C:/Users/mdirs/Downloads/project/ninety5_1.h5")
print("Saved model to disk")

from keras.models import model_from_json




json_file = open('eighty5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("eighty5.h5")
print("Loaded model from disk")




json_file = open('C:/Users/mdirs/Downloads/project/eighty5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/Users/mdirs/Downloads/project/eighty5.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

from keras.preprocessing import image
import numpy as np

test_image = image.load_img('C:/Users/mdirs/Downloads/project/hh.jpg', target_size = (64, 64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#predict the result
result = loaded_model.predict(test_image)

print(result)

print(loaded_model)



































