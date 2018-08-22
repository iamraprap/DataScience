
# Load the Relevant libraries
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.precision', 2)  
pd.set_option('display.max_colwidth',10000)

d = [[17], [28], [50], [60], [80], [89], [150], [167], [171], [189]]#np.random.random(13876).reshape(-1,1)

km = KMeans(n_clusters = 3, max_iter=3 )
km.fit(d)
print(d)

d = pd.DataFrame(d)
d.assign(cluster=sr.values)