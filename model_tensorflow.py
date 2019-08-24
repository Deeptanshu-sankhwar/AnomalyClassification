import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import time
import os
import cv2
import tflearn
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected, time_distributed
import random
import math
from tflearn.layers.conv import conv_2d, max_pool_2d
from tensorflow.contrib import rnn

maxFrames = 40
dims = 128
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
    folders = ['train_1', 'test_1']

    if dataType == 'train':
        listData = os.listdir(path + folders[0] + '/')
        random.seed(seed)
        random.shuffle(listData)
        i = 0
        for item in listData:
            # print(listData)
            if(listData[i][0:3] == "SEX"):  #flip to train on SCT
                Dict[item] = 1
                fileList.append(listData[i])
            elif (listData[i][0:3] != "SCT"):
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
            if(listData[i][0:3] == "SEX"):  #flip to test on SCT cases
                Dict[item] = 1
                fileList.append(listData[i])
            elif (listData[i][0:3] != "SCT"):
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


def flatten_layer(layer):
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


model_name = 'shop-bin-TH'
path = ''
batchSize = 8
no_of_epochs = 4
cvscores = []

tf.reset_default_graph()

convnet = input_data(shape=[dims, dims, 3])
print(convnet)
convnet = conv_2d(convnet, 64, 3, activation = 'relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 3, activation = 'sigmoid', padding = 'same')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 16, 3, activation = 'sigmoid', padding = 'same')
convnet = conv_2d(convnet, 16, 3, activation = 'sigmoid', padding = 'same')
convnet = max_pool_2d(convnet, 2)
print(convnet)
convnet = tflearn.layers.core.time_distributed(convnet, fully_connected,  [64])
print(convnet)

recnet = tflearn.layers.recurrent.gru(convnet, 64, return_seq=True, scope=None, name='GRU')
recnet = tflearn.layers.recurrent.gru(recnet, 64, scope=None, name='GRU')


dense = tf.layers.dense(recnet, 64, activation = 'relu')
dense = tf.layers.dense(dense, 64, activation = 'relu')
dense = tf.layers.dense(dense, 1, activation = 'sigmoid')

# main_input = tf.keras.Input(shape = (maxFrames, dims, dims, 3))
# model = tflearn.time_distributed(convnet, main_input)                  #this makes cnn run 40 times
# # model = rnn(model)
# model = dense(model)

adm = tflearn.optimizers.Adam(learning_rate=0.005, beta1=0.9, beta2=0.999, epsilon=None, use_locking = False)
# dense = fully_connected(dense, 1024, activation = 'relu')
# dense = dropout(dense, 0.8)
# dense = fully_connected(dense, 3, activation = 'softmax')
# final_model = Model(inputs = main_input, outputs = model)
dense = regression(dense, loss='binary_crossentropy', optimizer=adm, metric='accuracy')
    #final_model.summary()
print(dense)
model = tflearn.DNN(dense)    
# print("\n\nFOLD : " + str(num+1))
dense.fit(generate_data(path, batchSize, 'train'), 
                                        steps_per_epoch = 16,
                                        validation_data = generate_data(path, batchSize, 'test'),
                                        validation_steps= 16,
                                        epochs=no_of_epochs, 
                                        verbose=1)
    # scores = final_model.evaluate_generator(generate_data(path, batchSize, num, 'test'), steps= testSteps, verbose=1)
cvscores.append(history.history.get('val_acc')[-1] * 100)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('fooAcc_nodel.png')
