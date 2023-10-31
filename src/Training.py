#Libraries
import MAHDBase
import NeuralNet
import pickle

#Downloading Data
training_data=MAHDBase.load_training_data ()
test_data=MAHDBase.load_test_data()

#Creating the neural network
net = NeuralNet.Network([784,30,10])

#Training the neural network
net.SGD(training_data, 30, 10, 3, test_data=test_data)

#Saving the neural network
f=open('Network.pickle','wb')
pickle.dump(net,f)
f.close()
