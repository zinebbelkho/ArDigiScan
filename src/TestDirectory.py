# Libraries and files :
import os
import numpy as np
from PIL import Image
import pickle
import FunctionsApp

# Bring Our Network :
f=open("Network.pickle","rb")
net=pickle.load(f)
f.close()

nTotal = 0

main_folder = "C:/Users/ZAHAR AMINA/Desktop/arabic digits"

for i in range(10):
    n=0
    for j in range(10):
        with Image.open(f"{main_folder}/{i}/{i}digit_0{j}.jpeg").convert("L") as img:
            img=FunctionsApp.size_normalisation(img)

            # Centring image in 28x28 px:
            newImg=np.zeros((28,28))
            newImg=Image.fromarray(newImg)
            Image.Image.paste(newImg,img)
            a=FunctionsApp.invert(newImg)
            a=np.array(a)
            a=FunctionsApp.transform(a)
            a=FunctionsApp.centring(a)
            
            # Result :
            a=np.reshape(a,(784,1))
            r=FunctionsApp.feedforward(a, net.weights, net.biases)
            digit=int (np.argmax(r))
            
            # Is The Result Correct Or No ?
            if digit == i:
                n+=1
    nTotal += n
    print ("La précision pour le digit",i,"est de :",n*100/10.,"%")
print("La précision totale est de :",nTotal,"%")