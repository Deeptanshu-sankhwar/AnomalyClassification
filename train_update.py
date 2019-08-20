import os
import random
import cv2
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import time

maxFrames = 40
dims = 128
# folds = 5
seed = random.uniform(0, 1)

#given a openCV object refrencing the video, returns
def pullFrame(vidobj):
    framedVideo = []
    length = int(vidobj.get(cv2.CAP_PROP_FRAME_COUNT))
    success = 1
    count = 0
    flag = False
    if length < maxFrames:
        while True:
            success, image = vidobj.read()
            if success == 0 or count >= maxFrames:
                break
            resize = cv2.resize(image, (dims, dims))
            framedVideo.append(resize)
            count += 1
        while count < maxFrames:
            arr = np.zeros((dims,dims,3))
            framedVideo.append(arr)
            count += 1
    else:
        if length >= maxFrames and length <= 55:
            while True:
                success, image = vidobj.read()
                if success == 0 or count >= maxFrames:
                    break
                resize = cv2.resize(image, (dims, dims))
                framedVideo.append(resize)
                count += 1
        else:
            step = length // maxFrames
            while True:
                success, image = vidobj.read()
                if success == 0 or count >= length or len(framedVideo) >= maxFrames:
                    break
                if count % step == 0:
                    resize = cv2.resize(image, (dims, dims))
                    framedVideo.append(resize)
                count += 1
                
            while len(framedVideo) < maxFrames:
                flag = True
                arr = np.zeros((dims, dims, 3))
                framedVideo.append(arr)
                count += 1
            
    return (np.array(framedVideo))

  
def generate_data(path, batch_size, dataType):
    Dict = {}
    fileList = []
    folders = ['train', 'test']

    if dataType == 'train':
        listData = os.listdir(path + folders[0] + '/')
        random.seed(seed)
        random.shuffle(listData)
        i = 0
        for item in listData:
            # print(listData)
            if(listData[i][0] == "S"):
                Dict[item] = 1
                fileList.append(listData[i])
            else:
                Dict[item] = 0
                fileList.append(listData[i])
            i += 1
    # print(fileList[0])

    if dataType == 'test':
        listData = os.listdir(path + folders[1] + '/')
        random.seed(seed)
        random.shuffle(listData)
        i = 0
        for item in listData:
            if(listData[i][0] == "S"):
                Dict[item] = 1
                fileList.append(listData[i])
            else:
                Dict[item] = 0
                fileList.append(listData[i])
            i += 1

    
    i = 0
    random.shuffle(fileList)
    while True:
        output_x = []
        output_y = []
        for b in range(batch_size):
            if i == len(fileList):
                i = 0
                random.shuffle(fileList)
                
            vid = fileList[i]
            vidObj = cv2.VideoCapture(vid)
            framedVideo = pullFrame(vidObj)
            output_x.append(framedVideo)
            yLabel = Dict[vid]
            output_y.append(yLabel)
            i += 1

        output_x = np.array(output_x)
        output_x = output_x / 255.0  # min-max normalization
          
        output_y = np.array(output_y).reshape(-1, 1)
        yield (output_x, output_y)

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, TimeDistributed
from keras.layers import AveragePooling2D, Dense, GRU, Input, LSTM
from keras.models import Model
from keras import optimizers

path = ''                          #path where dataset resides as raw videos
batchSize = 4
no_of_epochs = 5
start = time.time()
cvscores = []


cnn = Sequential()
cnn.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(dims, dims,3), padding='same'))
cnn.add(MaxPooling2D(2))
cnn.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
cnn.add(MaxPooling2D(2))
cnn.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
cnn.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
cnn.add(MaxPooling2D(2))
cnn.add(Flatten())
    #cnn.summary()

rnn = Sequential()
rnn.add(GRU(64, return_sequences=True))
rnn.add(GRU(64))

dense = Sequential()
dense.add(Dense(64,activation='relu'))
dense.add(Dense(64,activation='relu'))
dense.add(Dense(1,activation='sigmoid'))

main_input = Input(shape = (maxFrames, dims, dims, 3))    #input a sequence of 40 images
model = TimeDistributed(cnn)(main_input)                  #this makes cnn run 40 times
model = rnn(model)
model = dense(model)

adm = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

final_model = Model(inputs = main_input, outputs = model)
final_model.compile(loss='binary_crossentropy', optimizer=adm, metrics=['accuracy'])
    #final_model.summary()
    
    
# print("\n\nFOLD : " + str(num+1))
history = final_model.fit_generator(generate_data(path, batchSize, 'train'), 
                                        steps_per_epoch = 33/batchSize,
                                        validation_data = generate_data(path, batchSize, 'test'),
                                        validation_steps= 19/batchSize,
                                        epochs=no_of_epochs, 
                                        verbose=1)
    # scores = final_model.evaluate_generator(generate_data(path, batchSize, num, 'test'), steps= testSteps, verbose=1)
cvscores.append(history.history.get('val_acc')[-1] * 100)
    #print(history.history.keys())
    
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
    
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
    
print('\n'+ ". accuracy : " + str(history.history.get('val_acc')[-1]*100) + ' %')
final_model.save('binary_shopliftModel' +  '.h5')  # creates a HDF5 file 
    
print("\n%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
end = time.time()
print('\nAvg. Execution time per fold: ' + str(((end - start)/60)/5) + ' mins')
print('\nTotal Execution time: ' + str((end - start)/60) + ' mins')
