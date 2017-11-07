#Importing the pandas library for python, used for reading data from the dataset.csv file.
import pandas as pd

#Importing the keras library used to build, compile and evaluate the Dense neural net.
import keras

#Importing backend.
from keras import backend as K

#Importing this so as to plot the model defined by us.
import keras.utils.vis_utils
from keras.utils import plot_model

#Starting by importing numpy and setting a seed for computer's pseudorandom number generator.
import numpy as np	

#Seed in reproducibility.
np.random.seed(561)

#Importing the sequential model type from Keras, which is simply a linear stack of neural network layers, aids in the feed forward Dense neural net.
from keras.models import Sequential

#Importing the core layers, used in almost any Neural Network, namely, Dense, Dropout, Activation and Flatten.
#Dense layer is a fully connected Neural Network
#Dropout layer helps prevent ovefitting
#Activation layer adds activation function to the output
from keras.layers import Dense
from keras.layers import Dropout 
from keras.layers import Activation
from keras.layers import Flatten

#Next importing the additional Layers, that help us train on large image data.
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D

#Importing utilites.
from keras.utils import np_utils

#Importing pickle for Python 2.7.10
import cPickle as pickle

#From sklearn.model_selection importing train_test_split so as to split the data set into test data set and training data set
from sklearn.model_selection import train_test_split

 
def load_data():

	'''
	Function to load the data from the csv dataset
	returns:
		-- datas, binarized character matrices
		-- labels, in the form of 26 length binary vectors
	'''

	df = pd.read_csv("dataset.csv", header=None)
	data = np.array(df[1][1:])
	label = np.array(df[2][1:])
	datas = np.empty([data.shape[0], 128], dtype=np.float32)
	labels = np.empty(data.shape[0], dtype=np.uint8)
	print datas
	for x in range(data.shape[0]):
		datas[x] =  (np.fromstring(data[x], 'u1') - ord('0')) #to convert from string to numpy vector
		labels[x] = ord(label[x]) - ord('a') #character to ASCII
	labels = keras.utils.to_categorical(labels, 26) # One hot encoding, to make the label into a vector, so that only the corresponding element will be 1 with everything else 0
	datas=np.reshape(datas, (datas.shape[0], 16, 8, 1)) #Reshape the vector into a proper 16x8 image
	return datas, labels

# defining function to build the different layers of the neural network
def make_model():
	model = Sequential() #For a sequential neural network
	model.add(Flatten())	
	model.add(Dense(288, activation='relu',input_shape=[16, 8, 1]))

	model.add(Dense(4096, activation='relu'))
	model.add(Dense(576, activation='relu'))
	model.add(Dense(3640, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(26, activation='softmax'))
	#Final predicted output
	
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
	#We're using ADAM(a variation of SGD) for optimazation, learning rate goes as a parameter ^ there
	
	return model


data, label = load_data()
#Training data is 80% of the entire data set
data_train = data[:40000]
label_train = label[:40000]
#Validation data is 20% of the entire data set
data_val = data[40000:50000]
label_val = label[40000:50000]
data_test = data[50000:]
label_test = label[50000:]

#Making the model using the function defined previously
model = make_model()

#Plotting the model made using keras library
plot_model(model, to_file='DenseNeuralNetwork.png',show_shapes=True,show_layer_names=True)

#training the model with the training data set, using epochs = 10, meaning it iterates through the entire data set 10 times)
#And, validation data is 20%, while 80% is training data.
model.fit(data_train, label_train, batch_size=16, epochs=50, validation_data=(data_val, label_val))

score = model.evaluate(data_test, label_test, verbose=0)

#Printing the results
print('Test loss:', score[0])

print('Test accuracy:', score[1])
