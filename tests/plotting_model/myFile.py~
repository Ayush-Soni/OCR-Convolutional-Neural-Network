from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import keras.utils.vis_utils
from keras.utils import plot_model

model = Sequential()
model.add(Dense(128, input_shape=(128,)))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(Dense(64))
model.add(Dense(26))
plot_model(model, to_file='model.png',show_shapes=True,show_layer_names=True)
