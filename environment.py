# -*- coding: utf-8 -*-
''' Modules for installation -> torch, torchvision, numpy, keyboard, cv2, mss.
    Use pip3 install 'module'.
'''
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte
import torchvision.transforms as T
from PIL import Image
import numpy as np
import threading
import keyboard
import torch
import cv2
import time
import mss

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def histogram(env, image):
    ''' Histogram generate function
    '''
    cropped_img = image[int((2*env.h/5)):int(env.h-(2*env.h/5)),
                        int(2*env.w/5):int(env.w-(2*env.w/5))]
    hist = cv2.calcHist([cropped_img], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

class env():
    ''' Class for environment management
    '''
    def __init__(self, resolution):
        self.w, self.h = resolution
        self.movements = ["a","d"]

        imageCapture = threading.Thread(name = 'imageCapture', target = self.imageCapture)
        imageCapture.setDaemon(True)
        imageCapture.start()

        img = cv2.imread('media/gameover.jpg')
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.hist_restart = histogram(self, rgb_img)

    def imageCapture(self):
        monitor = {'left': 0, 'top': 0, 'width': self.w, 'height': self.h}
        with mss.mss() as sct:
            while True:
                sct_img = sct.grab(monitor)
                img_np = np.array(sct_img)
                cropped_img = img_np[int(self.h/5):int(self.h-(self.h/3)), int(self.w/3):int(self.w-(self.w/3))]
                gray_frame = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                threash_triangle = threshold_otsu(gray_frame)
                binary_triangle = gray_frame > threash_triangle
                binary_triangle = img_as_ubyte(binary_triangle)
                resized_bw_frame = cv2.resize(binary_triangle,(int(90),int(160)),
                                                interpolation=cv2.INTER_AREA)
                self.bw_frame = cv2.bitwise_not(resized_bw_frame)
                self.rgb_frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                _ = cv2.waitKey(1)

    def reset(self):
        time.sleep(0.5)
        keyboard.send("enter")
        time.sleep(3.0)

    def step(self, action):
        done = False
        keyboard.send(self.movements[action])
        time.sleep(0.08)
        hist = histogram(self, self.rgb_frame)
        comparation = cv2.compareHist(self.hist_restart, hist, cv2.HISTCMP_BHATTACHARYYA)
        if (comparation > 0.50):  
            rew = 1
        else:
            done = True
            rew = 0
        return [], rew, done, []

    def get_screen(self):
        return resize(self.bw_frame).unsqueeze(0).to(device)
