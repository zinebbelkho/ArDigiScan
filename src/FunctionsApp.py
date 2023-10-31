import numpy as np
from PIL import Image

def sigmoid (z):
    return 1.0/(1.0+np.exp(-z))

def feedforward (a,w,b):
    for b1, w1 in zip(b,w):
        a = sigmoid(np.dot(w1, a)+b1)
    return a

def transform (x):
    return 1-x/255

#Bounding Box functions :
def invert_img(image):
    arr=np.array(image)
    arr=255-arr
    return Image.fromarray(arr)

def size_normalisation (img):
    img=invert_img(img)
    img=img.crop(img.getbbox())
    #Resizing :
    h=img.height    
    w=img.width
    if h>w:
        img=img.resize((int(np.floor(20*w/h)), 20))
    else:
        img=img.resize((20, int(np.floor(20*h/w))))
    return img
    
# Centring functions :
def invert_arr(arr):
    arr = 255-arr
    return arr

def massCenter(arr):
    arr = invert_arr(arr)
    totalMass = 0
    sumX = 0
    sumY = 0
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            totalMass += arr[y,x]
            sumX += arr[y,x] * (x+0.5)
            sumY += arr[y,x] * (y+0.5)
    centerX = sumX/totalMass
    centerY = sumY/totalMass
    return (centerX, centerY)

def transVector(arr):
    x, y = massCenter(arr)
    Cx, Cy = (arr.shape[1]/2, arr.shape[0]/2)
    return (int(np.round(Cx-x)), int(np.round(Cy-y)))

def center(img):
    arr = np.array(img)
    Tx, Ty = transVector(arr)
    arr = invert_arr(arr)
    newArr = np.zeros(arr.shape)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            if (arr[y, x] >= 0 and (0 <= y+Ty <=arr.shape[0]-1) and (0 <= x+Tx <= arr.shape[1]-1) ):
                newArr[y+Ty, x+Tx] = arr[y, x]
    newArr = invert_arr(newArr)
    img = Image.fromarray(newArr)
    return img