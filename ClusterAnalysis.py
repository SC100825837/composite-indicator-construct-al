import numpy as np
import pandas as pd
from scipy.cluster.vq import *
from sklearn.cluster import KMeans
import sys
import json

# sys.argv[1]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

data = pd.DataFrame(json.loads(sys.argv[1]))
n_clusters = int(sys.argv[2])
# data = pd.DataFrame(json.loads("[[50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0], [53.52, 54.2, 54.88, 55.55, 56.22, 56.88, 57.53, 58.17, 58.82, 59.45, 60.08, 60.71, 61.33, 61.94, 62.55, 63.15], [56.88, 57.53, 58.17, 58.82, 59.45, 60.08, 60.71, 61.33, 61.94, 62.55, 63.15, 63.75, 64.34, 64.93, 65.52, 66.1], [60.08, 60.71, 61.33, 61.94, 62.55, 63.15, 63.75, 64.34, 64.93, 65.52, 66.1, 66.67, 67.24, 67.81, 68.37, 68.93], [63.15, 63.75, 64.34, 64.93, 65.52, 66.1, 66.67, 67.24, 67.81, 68.37, 68.93, 69.48, 70.03, 70.57, 71.11, 71.65], [66.1, 66.67, 67.24, 67.81, 68.37, 68.93, 69.48, 70.03, 70.57, 71.11, 71.65, 72.18, 72.71, 73.23, 73.75, 74.27]]"))
# n_clusters = int(2)

kmeans = KMeans(n_clusters=n_clusters).fit(data)
pred = kmeans.predict(data)
print(np.array(pred))
