from keras.models import Model
from keras.layers import Input, LSTM, GRU
import numpy as np

timesteps = 8
num_features = 2
num_nodes = 3

data = np.random.randn(1, timesteps, num_features)
print(data)
print(data.shape)