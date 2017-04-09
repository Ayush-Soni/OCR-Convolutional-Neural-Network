#Starting by importing numpy and setting a seed for computer's pseudorandom number generator.
import numpy as np	

#Seed in reproducibility.
np.random.seed(561)

#Importing the sequential model type from Keras, which is simply a linear stack of neural network layers, aids in the feed forward CNN.
from keras.models import Sequential

#Importing the core layers, used in almost any Neural Network, namely, Dense, Dropout, Activation and Flatten.
#Dense layer is a fully connected Neural Network
#Dropout layer helps prevent ovefitting
#Activation layer adds activation function to the output
from keras.layers import Dense
from keras.layers import Dropout 
from keras.layers import Activation
from keras.layers import Flatten

#Next importing the CNN Layers, that help us train on large image data.
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

#Importing utilites.
from keras.utils import np_utils

#Importing pickle for Python 2.7.10
import cPickle as pickle

#From sklearn.model_selection importing train_test_split so as to split the data set into test data set and training data set
from sklearn.model_selection import train_test_split

#Done importing, moving on to importing the pickle file, ocr_dataset.p, which has 52,000+ samples of small case alphabets.
#Each alphabet is represented by a 16*8 matrix, which is in from of 128 character long array.
#The extractData.py function	extracts all the data into a Pickle file, where in the data is stores in a data structure with two structures.
path_to_dataset = "../data_set/ocr_dataset.p";
data_dictionary = pickle.load(open(path_to_dataset, "rb"));

#Extracting data from the dictionary stored in the pickle
label_array = data_dictionary['label']
data_arrays = data_dictionary['data']

#Dividing data into training set and test set
data_array_train, data_array_test, label_array_train, label_array_test = train_test_split(data_arrays, label_array, test_size = 0.2, random_state=42);

#print data_array_train, data_array_test, label_array_train, label_array_test 

#Building the neural network using a sequential model
model = Sequential()

#input dimensions will be 26, because there are 26 classes
model.add(Dense(units=64, input_dim = 26))


