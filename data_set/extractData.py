import re
import numpy as np
import cPickle as pickle

fo = file('letter.data')
dataArr = np.array([0]*128)
labelArr = np.array([0])

for x in xrange(0, 400):    #range stores the entire list in memory, xrange is lazy, so it evaluates the value at each iteration
    s = ''
    s = fo.readline()
    arr = re.sub('[^\w]', ' ', s).split()
    data = arr[6:]
    data = map(int, data)
    label = ord(arr[1])
    dataArr = np.vstack([dataArr, data])
    labelArr = np.hstack([labelArr, label])
dataDict = {'data':dataArr, 'label':labelArr}
fo.close()
pickle.dump(dataDict, open("ocr_dataset.p", "wb"))
