import os
import random
import cv2
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import add, Conv2D, Flatten, MaxPooling2D, TimeDistributed, Dropout, GlobalAveragePooling3D, Input, Dense, Activation, Conv3D, MaxPooling3D, SeparableConv2D, AveragePooling2D, AveragePooling3D, BatchNormalization, ReLU
from keras.layers import GRU, LSTM
from keras.optimizers import SGD
from keras.models import Model
from sklearn import metrics
from keras.applications.xception import Xception
from keras import optimizers


maxFrames = 90
dims = 128
clip_length = 90
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


path = ''                          #path where dataset resides as raw videos
batchSize = 5
no_of_epochs = 5
start = time.time()
cvscores = []



def bottleneck_residual(x, out_filter, strides, activate_before_residual=False, inflate=False):
    orig_x = x
    # a
    if inflate:
        x = Conv3D(out_filter // 4, (3, 1, 1), strides=strides, padding='same')(x)
    else:
        x = Conv3D(out_filter // 4, (1, 1, 1), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # b
    if orig_x.get_shape().as_list()[-1] != out_filter and out_filter != 256:
        x = Conv3D(out_filter // 4, (1, 3, 3), strides=(1, 2, 2), padding='same')(x)
    else:
        x = Conv3D(out_filter // 4, (1, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # c
    x = Conv3D(out_filter, (1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # when channels change, shortcut
    if orig_x.get_shape().as_list()[-1] != out_filter and out_filter != 256:
        orig_x = Conv3D(out_filter, (1, 1, 1), strides=(1, 2, 2), padding='same')(orig_x)
    else:
        orig_x = Conv3D(out_filter, (1, 1, 1), strides=(1, 1, 1), padding='same')(orig_x)
    orig_x = BatchNormalization()(orig_x)
    orig_x = ReLU()(orig_x)

    x = add([orig_x, x])
    x = ReLU()(x)

    return x

def build_3d_net(model_net, num_classes, clip_length, img_size):
    video_input = Input(shape=(clip_length, img_size, img_size, 3))
    encoded_frame_sequence = TimeDistributed(model_net)(video_input)
    print(encoded_frame_sequence.shape)
    resnet = Conv3D(64, (5, 7, 7), strides=(1, 1, 1), padding='same', name='3d_resnet_3a_1_3x3x3')(encoded_frame_sequence)
    resnet = BatchNormalization()(resnet)
    resnet = ReLU()(resnet)

    block_num = [3, 4, 6, 3]
    # res2
    resnet = bottleneck_residual(resnet, 256, strides=(1, 1, 1), inflate=True)
    for _ in range(1, block_num[0]):
        resnet = bottleneck_residual(resnet, 256, strides=(1, 1, 1), activate_before_residual=False, inflate=True)

    resnet = MaxPooling3D(pool_size=(3, 1, 1), strides=(2, 1, 1), padding='same')(resnet)

    # res3
    resnet = bottleneck_residual(resnet, 512, strides=(1, 1, 1), activate_before_residual=False, inflate=True)
    for _ in range(1, block_num[1]):
        if _ % 2:
            resnet = bottleneck_residual(resnet, 512, strides=(1, 1, 1), activate_before_residual=False, inflate=False)
        else:
            resnet = bottleneck_residual(resnet, 512, strides=(1, 1, 1), activate_before_residual=False, inflate=True)

    # res4
    resnet = bottleneck_residual(resnet, 1024, strides=(1, 1, 1), activate_before_residual=False, inflate=True)
    for _ in range(1, block_num[1]):
        if _ % 2:
            resnet = bottleneck_residual(resnet, 1024, strides=(1, 1, 1), activate_before_residual=False, inflate=False)
        else:
            resnet = bottleneck_residual(resnet, 1024, strides=(1, 1, 1), activate_before_residual=False, inflate=True)

    resnet = MaxPooling3D(pool_size=(3, 1, 1), strides=(2, 1, 1), padding='same')(resnet)

    # res5    全连接层原文中用的1024
    resnet = bottleneck_residual(resnet, 1024, strides=(1, 1, 1), activate_before_residual=False, inflate=True)
    for _ in range(1, block_num[1]):
        if _ % 2:
            resnet = bottleneck_residual(resnet, 1024, strides=(1, 1, 1), activate_before_residual=False, inflate=True)
        else:
            resnet = bottleneck_residual(resnet, 1024, strides=(1, 1, 1), activate_before_residual=False, inflate=False)

    print(resnet.shape)
    resnet = GlobalAveragePooling3D()(resnet)    # ECO作者用的GAP
    print(resnet.shape)
    resnet = Dropout(0.5)(resnet)
    # 354个类别
    print(resnet)
    predictions = Dense(1, activation='softmax')(resnet)
    print(predictions.shape)
    # 最终模型
    with tf.device('/cpu:0'):
        InceptionV3_Resnet3D = Model(inputs=video_input, outputs=predictions)
    print(InceptionV3_Resnet3D.summary())
    return InceptionV3_Resnet3D

def build_2d_net(img_size):
    # 输出是96*28*28
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))    # default input shape is (299, 299, 3)

    x = SeparableConv2D(96, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(base_model.get_layer('block4_sepconv1_act').output)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)

    model = Model(inputs=base_model.input, outputs=x)
    print(model.summary())
    return model

model_2d = build_2d_net(dims)
model_3d = build_3d_net(model_2d, 2, clip_length, dims)

optimizer = SGD(lr=0.01, momentum=0.9)

model_3d.compile(optimizer=optimizer, loss='binary_crossentropy',
                     metrics=['accuracy'])
history = model_3d.fit_generator(generate_data(path, batchSize, 'train'),
                           steps_per_epoch = 30,
                           validation_data = generate_data(path, batchSize, 'test'),
                           validation_steps = 15,
                           epochs=no_of_epochs,
                           # callbacks=[checkpoint],
                           workers=batchSize*10,
                           max_queue_size=batchSize*10,
                           use_multiprocessing=True,
                           shuffle=False,
                           # verbose = 1
                           )
# cnn = Sequential()
# cnn.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(dims, dims,3), padding='same'))
# cnn.add(MaxPooling2D(2))
# cnn.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
# cnn.add(MaxPooling2D(2))
# cnn.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
# cnn.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
# cnn.add(MaxPooling2D(2))
# cnn.add(Flatten())
#     #cnn.summary()

# rnn = Sequential()
# rnn.add(GRU(64, return_sequences=True))
# rnn.add(GRU(64))

# dense = Sequential()
# dense.add(Dense(64,activation='relu'))
# dense.add(Dense(64,activation='relu'))
# dense.add(Dense(1,activation='sigmoid'))

# main_input = Input(shape = (maxFrames, dims, dims, 3))    #input a sequence of 40 images
# model = TimeDistributed(cnn)(main_input)                  #this makes cnn run 40 times
# model = rnn(model)
# model = dense(model)

# adm = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# final_model = Model(inputs = main_input, outputs = model)
# final_model.compile(loss='binary_crossentropy', optimizer=adm, metrics=['accuracy'])
#     #final_model.summary()
    
    
# # print("\n\nFOLD : " + str(num+1))
# history = final_model.fit_generator(generate_data(path, batchSize, 'train'), 
#                                         steps_per_epoch = 33/batchSize,
#                                         validation_data = generate_data(path, batchSize, 'test'),
#                                         validation_steps= 19/batchSize,
#                                         epochs=no_of_epochs, 
#                                         verbose=1)
    # scores = final_model.evaluate_generator(generate_data(path, batchSize, num, 'test'), steps= testSteps, verbose=1)
cvscores.append(history.history.get('val_acc')[-1] * 100)
    #print(history.history.keys())
    
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
    
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("3d-resnet-fig-acc.png")
    
print('\n'+ ". accuracy : " + str(history.history.get('val_acc')[-1]*100) + ' %')
# model_3d.save('binary_shopliftModel' +  '.h5')  # creates a HDF5 file 
    
print("\n%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
end = time.time()
print('\nAvg. Execution time per fold: ' + str(((end - start)/60)/5) + ' mins')
print('\nTotal Execution time: ' + str((end - start)/60) + ' mins')