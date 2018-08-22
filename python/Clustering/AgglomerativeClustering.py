
# Load the Relevant libraries
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.precision', 2)  
pd.set_option('display.max_colwidth',10000)

d = [[17], [28], [50], [60], [80], [89], [150], [167], [171], [189]]#np.random.random(13876).reshape(-1,1)

Z = linkage(d, 'single')

plt.figure(figsize=(10, 8))
plt.scatter(Z[:,0], Z[:,1])  # plot all points

plt.figure(figsize=(25,10))
plt.xlabel('x')
plt.ylabel('y')
dendrogram(Z, leaf_rotation=90, leaf_font_size=8)

Z = linkage(d, 'complete')

plt.figure(figsize=(10, 8))
plt.scatter(Z[:,0], Z[:,1])  # plot all points

plt.figure(figsize=(25,10))
plt.xlabel('x')
plt.ylabel('y')
dendrogram(Z, leaf_rotation=90, leaf_font_size=8)