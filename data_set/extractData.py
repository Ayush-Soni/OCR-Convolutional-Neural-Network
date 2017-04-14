import re
import numpy as np
import cPickle as pickle
import time

filePointer = file("letter.data")
dataArr = np.array([0]*128)
labelArr = np.array([0])

while True:
    s = filePointer.readline()
    if not s: break
    arr = re.sub('[^\w]', ' ', s).split()
    data = arr[6:]
    label = ord(arr[1])-97
    dataArr = np.vstack([dataArr, data])
    labelArr = np.hstack([labelArr, label])
dataDict = {'data':dataArr, 'label':labelArr}
filePointer.close()

pickle.dump(dataDict, open("ocr_dataset.p", "wb"))
