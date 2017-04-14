import numpy as np
import pandas as pd
np.set_printoptions(threshold='nan')
df = pd.read_csv("log.csv", header=None)
data = np.array(df[1][1:])
label = np.array(df[2][1:])
datas = np.empty([data.shape[0], 128], dtype=np.float32)
labels = np.empty(data.shape[0], dtype=np.uint8)

print labels
