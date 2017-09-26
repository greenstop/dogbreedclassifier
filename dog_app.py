from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

train_files, train_targets = load_dataset('dogImages.lnk/train')
valid_files, valid_targets = load_dataset('dogImages.lnk/valid')
test_files, test_targets = load_dataset('dogImages.lnk/test')

dog_names = [item[20:-1] for item in sorted(glob("dogImages.lnk/train/*/"))]

print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

np.savez_compressed("dog_breeds.npz", train_files=train_files, train_targets=train_targets, valid_files=valid_files, valid_targets=valid_targets, test_files=test_files, test_targets=test_targets)

data = np.load("dog_breeds.npz");
train_files=data["train_files"];
train_targets=data["train_targets"];
valid_files=data["valid_files"];
valid_targets=data["valid_targets"];
test_files=data["test_files"];
test_targets=data["test_targets"];

dog_names = [item[20:-1] for item in sorted(glob("dogImages.lnk/train/*/"))]

print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

import random
random.seed(8675309)

human_files = np.array(glob("lfw.lnk/*/*"))
random.shuffle(human_files)

print('There are %d total human images.' % len(human_files))

import cv2                
import matplotlib.pyplot as plt                        
get_ipython().magic('matplotlib inline')

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

img = cv2.imread(human_files[3])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray)

print('Number of faces detected:', len(faces))

for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(cv_rgb)
plt.show()

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

human_files_short = human_files[:100]
dog_files_short = train_files[:100]

print("Humans: ", sum([ face_detector(f) for f in human_files_short ]))
print("Dogs: ", sum([ face_detector(f) for f in dog_files_short ]))

from keras.applications.resnet50 import ResNet50

ResNet50_model = ResNet50(weights='imagenet')

from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

print("Humans: ", sum([ dog_detector(f) for f in human_files_short ]))
print("Dogs: ", sum([ dog_detector(f) for f in dog_files_short ]))

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

np.savez_compressed("dog_breeds_processed.npz", train_tensors=train_tensors, 
                    valid_tensors=valid_tensors, test_tensors=test_tensors)

data = np.load("dog_breeds_processed.npz");
train_tensors=data["train_tensors"];
valid_tensors=data["valid_tensors"];
test_tensors=data["test_tensors"];

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

K=64
model = Sequential()
model.add(Conv2D(filters=K*1, kernel_size=2, padding='same', activation='relu', input_shape=(224, 224, 3))); 
model.add(Conv2D(filters=K*1, kernel_size=2, padding='same', activation='relu')); 
model.add(MaxPooling2D(pool_size=4))

model.add(Conv2D(filters=K*2, kernel_size=2, padding='same', activation='relu')); model.add(Dropout(0.1));
model.add(Conv2D(filters=K*2, kernel_size=2, padding='same', activation='relu')); model.add(Dropout(0.1));
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=K*3, kernel_size=2, padding='same', activation='relu')); model.add(Dropout(0.1));
model.add(Conv2D(filters=K*3, kernel_size=2, padding='same', activation='relu')); model.add(Dropout(0.1));
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=K*4, kernel_size=2, padding='same', activation='relu')); model.add(Dropout(0.2));
model.add(Conv2D(filters=K*4, kernel_size=2, padding='same', activation='relu')); model.add(Dropout(0.2));
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=K*5, kernel_size=2, padding='same', activation='relu')); model.add(Dropout(0.2));
model.add(Conv2D(filters=K*5, kernel_size=2, padding='same', activation='relu')); model.add(Dropout(0.2));
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=K*6, kernel_size=2, padding='same', activation='relu')); model.add(Dropout(0.2));
model.add(Conv2D(filters=K*6, kernel_size=2, padding='same', activation='relu')); model.add(Dropout(0.2));
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(K*7, activation='relu')); model.add(Dropout(0.3))
model.add(Dense(K*8, activation='relu')); model.add(Dropout(0.4))
model.add(Dense(133, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint  

epochs = 10

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

model.load_weights('saved_models/weights.best.from_scratch.hdf5')

dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

bottleneck_features = np.load('bottleneck_features.lnk/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']

VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()

VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')

VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

bottleneck_features = np.load('bottleneck_features.lnk/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']

from keras.layers import Dense, Flatten
from keras.models import Sequential

model = Sequential()
model.add(Flatten(input_shape=(1, 1, 2048)))

model.add(Dense(1024, activation='relu')); model.add(Dropout(0.4))
model.add(Dense(512, activation='relu')); model.add(Dropout(0.5))
model.add(Dense(133, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.resnet50.h5',  verbose=1, save_best_only=True)

model.fit(train_Resnet50, train_targets,  validation_data=(valid_Resnet50, valid_targets),
          epochs=1000, batch_size=256, callbacks=[checkpointer], verbose=2)

model.load_weights('saved_models/weights.best.resnet50.h5')

score = model.evaluate(test_Resnet50,test_targets, verbose=0)
print("Test accuracy %.4f%s"%(score[1]*100,"%"))

from extract_bottleneck_features import *

def resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def breed_predictor(img_path):
    if face_detector(img_path):
        print("You are a human.");
        breed=resnet50_predict_breed(img_path);
        print("You resemble this breed: %s."%breed);
    elif dog_detector(img_path):
        print("You are a dog.");
        breed=resnet50_predict_breed(img_path);
        print("This is likely this breed: %s."%breed);
    else:
        print("You are neither");
    

img_paths = glob("six_images/*")

for img in img_paths:
    print("file: %s"%img)
    breed_predictor(img)
    print("="*50)






