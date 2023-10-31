from PIL import Image
import numpy as np
import FunctionsApp

directory = "C:/Users/ZAHAR AMINA/Desktop/"
img = Image.open(directory+"3_pasted.jpg").convert("L")
img = FunctionsApp.invert(img)
arr = np.array(img)
arr = FunctionsApp.centring(arr)
img = Image.fromarray(arr)
img = FunctionsApp.invert(img).convert("L")
img.show()
img.save(directory+"3_centered.jpg")