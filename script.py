import cv2
import numpy as np
import os, shutil

maxFrames = 80
dims = 128
def pullFrame(vidobj):
    framedVideo = []
    length = int(vidobj.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
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
                    cv2.imwrite("images/" + str(count) + ".png", resize)
                count += 1
                
            while len(framedVideo) < maxFrames:
                flag = True
                arr = np.zeros((dims, dims, 3))
                framedVideo.append(arr)
                count += 1
            
    return (np.array(framedVideo))

def del_content(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def convert(vid, folder, name):
    vid = folder + vid
    cap = cv2.VideoCapture(vid)
    framedVideo = pullFrame(cap)

    frame_array = []
    video_array = []
    files = os.listdir("images/")
    for item in files:
        video_array.append(int(item[:-4]))

    video_array.sort()
    # print(video_array)
    for i in range(len(video_array)):
        video_array[i] = 'images/' + str(video_array[i]) + '.png'


    for item in video_array:
        # print(item)
        img = cv2.imread(item)
        frame_array.append(img)

    out = cv2.VideoWriter("framedVideos/test/" + name +".avi", cv2.VideoWriter_fourcc(*'DIVX'), 7, (128, 128))

    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

    del_content('images/')


# folder = os.listdir("train/")
folder = os.listdir("test/")
for item in folder:
    if (item[0] == 'A'):
        convert(item, "test/",'Abuse'+item[-11:-9])
    if (item[0] == 'S'):
        convert(item, "test/", 'Shoplifting' + item[-11:-9])