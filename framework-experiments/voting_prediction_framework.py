#Voting Framework
from keras.models import Model, load_model
import numpy as np
import os
import cv2
import tensorflow as tf
import random
import pandas as pd
from keras.layers import Reshape,Add
from pathlib import Path
import matplotlib.pyplot as plt

def get_video_frames(src, fpv, frame_height, frame_width):
    # print('reading video from', src)
    cap = cv2.VideoCapture(src)

    frames = []
    if not cap.isOpened():
        cap.open(src)
    ret = True
    while(True and ret):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    # print(src)
    rnd_idx = random.randint(5,len(frames)-5)
    rnd_frame = frames[rnd_idx]
    rnd_frame = cv2.resize(rnd_frame,(128,128)) #Needed for Densenet121-2d

    # Return fpv=10 frames
    step = len(frames)//fpv
    avg_frames = frames[::step]
    avg_frames = avg_frames[:fpv]
    avg_resized_frames = []
    i = 0
    for af in avg_frames:
        rsz_f = cv2.resize(af, (frame_width, frame_height))
        # cv2.imwrite('images/img' + str(i) + '.png', rsz_f)
        # i = i+1
        avg_resized_frames.append(rsz_f)
    return np.asarray(rnd_frame)/255.0,np.asarray(avg_resized_frames)


path = '/media/data/vca/dataset/Shoplifting/'

accuracy_sum = 0
accuracy_2A_sum = 0
accuracy_3A_sum = 0
accuracy_4A_sum = 0

x = [1, 2, 3, 4, 5]
y = []
y_2A = []
y_3A = []
y_4A = []   

for j in range(5):
    model_2A = load_model('2A-FRESH/model-2A-'+str(j+1)+'FRESH.h5', custom_objects={'tf':tf})
    model_3A = load_model('3A-FRESH/model-3A-'+str(j+1)+'FRESH.h5', custom_objects={'tf':tf})
    model_4A = load_model('4A-FRESH/model-4A-'+str(j+1)+'FRESH.h5', custom_objects={'tf':tf})

    listData_2A = os.listdir(path + 'test/mixed_2A')
    listData_3A = os.listdir(path + 'test/mixed_3A')
    listData_4A = os.listdir(path + 'test/mixed_4A')

    count = 0
    count_2A = 0
    count_3A = 0
    count_4A = 0

    for i in range(len(listData_2A)):
        print(listData_2A[i])
        print(listData_3A[i])
        print(listData_4A[i])

        vote2 = 0
        vote3 = 0
        vote4 = 0

        pvid_2A = path+'test/mixed_2A/' + listData_2A[i]
        pvidcv_2A = cv2.VideoCapture(pvid_2A)
        _,pframe_2A = get_video_frames(pvid_2A, 64, 128, 128)
        pframe_2A = pframe_2A.reshape(-1, 64, 128, 128, 3)
        res_2A = model_2A.predict(pframe_2A)[0]

        pvid_3A = path+'test/mixed_3A/' + listData_3A[i]
        pvidcv_3A = cv2.VideoCapture(pvid_3A)
        _,pframe_3A = get_video_frames(pvid_3A, 64, 128, 128)
        pframe_3A = pframe_3A.reshape(-1, 64, 128, 128, 3)
        res_3A = model_3A.predict(pframe_3A)[0]

        pvid_4A = path+'test/mixed_4A/' + listData_4A[i]
        pvidcv_4A = cv2.VideoCapture(pvid_4A)
        _,pframe_4A = get_video_frames(pvid_4A, 64, 128, 128)
        pframe_4A = pframe_4A.reshape(-1, 64, 128, 128, 3)
        res_4A = model_4A.predict(pframe_4A)[0]

        res1 = [res_2A[0], res_3A[0], res_4A[0]]
        res2 = [res_2A[1], res_3A[1], res_4A[1]]

        if res2[0] > 0.5:
            vote2 = 1
            if listData_2A[i][0] == 's':
                count_2A += 1
        else:
            vote2 = -1
            if listData_2A[i][0] == 'n':
                count_2A += 1

        if res2[1] > 0.5:
            vote3 = 1
            if listData_2A[i][0] == 's':
                count_3A += 1
        else:
            vote3 = -1
            if listData_3A[i][0] == 'n':
                count_3A += 1

        if res2[2] > 0.5:
            vote4 = 1
            if listData_2A[i][0] == 's':
                count_4A += 1
        else:
            vote4 = -1
            if listData_2A[i][0] == 'n':
                count_4A += 1

        vote = vote2 + vote3 + vote4

        if vote > 0:
            print("Shoplifting")
            
            if listData_2A[i][0] == 's':
                count += 1

            prediction = "Shoplifting"
            arg = np.argmax(res2)
            if arg == 0:
                res = res_2A
                print(res_2A)
            elif arg == 1:
                res = res_3A
                print(res_3A)
            else:
                res = res_4A
                print(res_4A)
        else:
            print("Don't Worry")

            if listData_2A[i][0] == 'n':
                count += 1

            prediction = "Non Shoplifting"
            arg = np.argmax(res1)
            if arg == 0:
                res = res_2A
                print(res_2A)
            elif arg == 1:
                res = res_3A
                print(res_3A)
            else:
                res = res_4A
                print(res_4A)

        db1 = pd.DataFrame({'Video' : listData_2A[i], '2A-prob-1' : str(res_2A[0]), '2A-prob-2' : str(res_2A[1]), '3A-prob-1' : str(res_3A[0]), '3A-prob-2' : str(res_3A[1]), '4A-prob-1' : str(res_4A[0]), '4A-prob-2' : str(res_4A[1]), 'FINAL-prob-1' : str(res[0]), 'FINAL-prob-2' : str(res[1]), 'Prediction' : prediction, 'Accuracy' : str(count/(i+1)), 'Accuracy_2A' : str(count_2A/(i+1)), 'Accuracy_3A' : str(count_3A/(i+1)), 'Accuracy_4A' : str(count_4A/(i+1))},  index = [i])
        if Path("framework-analysisVOTE.csv").is_file():
            with open ('framework-analysisVOTE.csv', 'a') as f:
                db1.to_csv(f, header = False)
        else:
            db1.to_csv('framework-analysisVOTE.csv')


    accuracy_sum += count/len(listData_2A)
    accuracy_2A_sum += count_2A/len(listData_2A)
    accuracy_3A_sum += count_3A/len(listData_2A)
    accuracy_4A_sum += count_4A/len(listData_2A)

    y.append(count/len(listData_2A))
    y_2A.append(count_2A/len(listData_2A))
    y_3A.append(count_3A/len(listData_2A))
    y_4A.append(count_4A/len(listData_2A))


plt.plot(x, y, label = 'final accuracy')
plt.plot(x, y_2A, label = '2A accuracy')
plt.plot(x, y_3A, label = '3A accuracy')
plt.plot(x, y_4A, label = '4A accuracy')
plt.xlabel('Iteration of model')
plt.ylabel('Acuuracy')
plt.title('Accuracy of differnet models on trained iterations')
plt.legend()
plt.savefig('framework-analysisVOTE.png')


db1 = pd.DataFrame({'Video' : '  ', '2A-prob-1' : '  ', '2A-prob-2' : '  ', '3A-prob-1' : '  ', '3A-prob-2' : '  ', '4A-prob-1' : '  ', '4A-prob-2' : '  ', 'FINAL-prob-1' : '  ', 'FINAL-prob-2' : '  ', 'Prediction' : '  ', 'Accuracy' : str(accuracy_sum/5), 'Accuracy_2A' : str(accuracy_2A_sum/5), 'Accuracy_3A' : str(accuracy_3A_sum/5), 'Accuracy_4A' : str(accuracy_4A_sum/5)},  index = [38])    
with open ('framework-analysisVOTE.csv', 'a') as f:
    db1.to_csv(f, header = False)