from PIL import Image
import numpy as np
import scipy.misc
import cPickle as pickle
import time

image_data = pickle.load(open('ocr_dataset.p','rb'))
label_array = image_data['label']
data_array = image_data['data']

for i in range(0, len(image_data["label"])):
    w, h = 8, 16
    data = data_array[i]
    data = np.array(data).reshape(16, 8)
    scipy.misc.imsave(str(label_array[i])+str(i)+".png", data)

