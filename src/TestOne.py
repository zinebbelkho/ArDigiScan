
'''      To test just one imagen, one digit !     '''

# Libraries and files :
import pickle
import numpy as np
from PIL import Image
import FunctionsApp

directory = "C:/Users/ZAHAR AMINA/Desktop/arabic digits/7/"

with Image.open(directory+"7digit_05.jpeg") as img :
    # Show image :
    img.show()

    # Bounding Box :
    img=FunctionsApp.invert(img)
    img=img.crop(img.getbbox())

    # Resizing :
    h=img.height    
    w=img.width
    if h>w:
        img=img.resize((int(np.floor(20*w/h)), 20))
    else:
         img=img.resize((20, int(np.floor(20*h/w))))

    # Centring image in 28x28 px:
    newImg=np.zeros((28,28))
    newImg=Image.fromarray(newImg)
    Image.Image.paste(newImg,img)
    a=FunctionsApp.invert(newImg)
    a=np.array(a)
    a=FunctionsApp.transform(a)
    a=FunctionsApp.centring(a)

    # Bring Our Network :
    f=open("Network.pickle","rb")
    net=pickle.load(f)
    f.close()

    # Result :
    a=np.reshape(a,(784,1))
    r=FunctionsApp.feedforward(a, net.weights, net.biases)
    print ("Digit Written is "+np.argmax(r))