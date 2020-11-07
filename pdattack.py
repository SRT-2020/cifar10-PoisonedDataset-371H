import keras
import numpy as np
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import random


def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])
    
    # Copy the image n == len(xs) times so that we can 
    # create n new perturbed images
    tile = [len(xs)] + [1]*(xs.ndim+1)
    imgs = np.tile(img, tile)
    
    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)
    
    for x,img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb
    
    return imgs

def pdAttack(data, numP):
    (x_train,y_train),(x_test,y_test) = data
    if (numP == 1):
        for i in range(len(x_train)):
            x = random.randint(0,31)
            y = random.randint(0,31)
            r = random.randint(0,255)
            g = random.randint(0,255)
            b = random.randint(0,255)
            pixel = np.array([x,y,r,g,b])
            temp = x_train[i]
            perturb_image(pixel, temp)
            x_train[i] = temp
        data1 = (x_train, y_train),(x_test, y_test)
        return data1
            

    if(numP == 2):
        for i in range(len(x_train)):
            x1 = random.randint(0,31) # Get random values for two pixel coordiantes and colors
            x2 = random.randint(0,31)
            y1 = random.randint(0,31)
            y2 = random.randint(0,31)
            r1 = random.randint(0,255) 
            g1 = random.randint(0,255)
            b1 = random.randint(0,255)
            r2 = random.randint(0,255)
            g2 = random.randint(0,255)
            b2 = random.randint(0,255)
            pixel = np.array([])
            pixel = np.append([x1, y1, r1, g1, b1], [x2, y2, r2, g2, b2])
            temp = x_train[i]
            perturb_image(pixel, temp)
            x_train[i] = temp
        data2 = (x_train, y_train),(x_test, y_test)
        return data2
        
    
    if(numP == 3):
        for i in range(len(x_train)):
            x1 = random.randint(0,31) # Get random values for two pixel coordiantes and colors
            x2 = random.randint(0,31)
            x3 = random.randint(0,31)
            y3 = random.randint(0,31)
            y1 = random.randint(0,31)
            y2 = random.randint(0,31)
            r1 = random.randint(0,255) 
            g1 = random.randint(0,255)
            b1 = random.randint(0,255)
            r2 = random.randint(0,255)
            g2 = random.randint(0,255)
            b2 = random.randint(0,255)
            r3 = random.randint(0,255)
            g3 = random.randint(0,255)
            b3 = random.randint(0,255)
            pixel = np.array([])
            pixel = np.append([x1,y1,r1,g1,b1],[x2,y2,r2,g2,b2])
            pixel = np.append(pixel,[x3,y3,r3,g3,b3])
            temp = x_train[i]
            perturb_image(pixel, temp)
            x_train[i] = temp
        data3 = (x_train,y_train),(x_test,y_test)
        return data3



        

            





