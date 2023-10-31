# Libraries and packages :
import os
import numpy as np
from PIL import Image

# Functions :
def Vectorized (i):
    vect=np.zeros((10,1))
    vect[i]=1.0
    return vect

def transform (x):
    return 1-x/255

def load_training_data ():
    main_folder = "C:/Users/ZAHAR AMINA/Desktop/ArDigiScan/Data_base/MAHDBase_TrainingSet"
    training_data=[]
    for root, dirs, files in os.walk(main_folder):
        for filename in files:
            if filename.endswith(".bmp"):            
                with Image.open(os.path.join(root, filename)) as img:
                    img=np.array(img)
                    img=transform(img)
                    img=np.reshape(img, (784,1))
                    n=np.int64(filename[-5])
                    vect=Vectorized(n)                     
                    training_data.append((img,vect))         
    return training_data

def load_test_data ():
    main_folder = "C:/Users/ZAHAR AMINA/Desktop/ArDigiScan/Data_base/MAHDBase_TestingSet"
    testing_data=[]
    for root, dirs, files in os.walk(main_folder):
        for filename in files:
            if filename.endswith(".bmp"):            
                with Image.open(os.path.join(root, filename)) as img:
                    img=np.array(img)
                    img=transform(img)
                    img=np.reshape(img,(784,1))
                    n=np.int64(filename[-5])                   
                    testing_data.append((img,n))                
    return testing_data
  
  
        