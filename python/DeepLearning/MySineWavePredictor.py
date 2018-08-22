from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, GRU, Dropout
from keras.regularizers import l1_l2
from keras.optimizers import RMSprop
import math
import numpy as np
import matplotlib.pyplot as plt

MAKE_PLOTS = True
NUM_LSTM_NODES = 15
NUM_EPOCHS = 10

#Set up params for dataset.
NUM_FREQS = 2      # Number of sine waves to superimpose
SR = 200           # Digital sampling rate
MAX_FREQ = SR / 2  # Nyquist frequency
two_pi = 2 * math.pi

MAX_FREQ /= 4  # calm the frequencies down a little.

MAX_X_DATA = 5     # Set x-range from 0 to MAX_X_DATA

TRAIN_TEST_SPLIT = 0.7  # Percentage of data in training set

# Use this to try on random sine waves if you'd like.
RANDOMIZE_WAVES = False
if RANDOMIZE_WAVES:
    np.random.seed(42)
    freqs = np.random.rand(NUM_FREQS) * MAX_FREQ
    amplitudes = np.random.rand(NUM_FREQS)
    phases = np.random.rand(NUM_FREQS) * two_pi
    waves = list(zip(freqs, amplitudes, phases))
else:
    # Or just use fixed frequencies etc, to make it easier:
    waves = [(20, .5, 0), (3, .3, 3)]  # Each element is (freq, amplitude, phase)

# Make the sine wave data.
data = []
t_list = [float(i)/SR for i in range(SR * MAX_X_DATA)]
for t in t_list:
    total = 0
    for freq, A, phase in waves:
        total += A * np.sin(phase + t * freq * two_pi)
    data.append(total)

# normalize
max_data = abs(np.max(data))
data /= max_data

if MAKE_PLOTS == True:
    plt.figure(figsize=(14,5))
    plt.ylim(-1,1)
    plt.plot(t_list, data)
    plt.legend(loc='upper left')
    plt.show()

# Make input/ouput pairs: match datapoint (N) to each datapoint (N+1)
# data_pairs should be a list of data points like [(data_1, data_2), (data_2, data_3),...]
#data_pairs
data_pairs = [(data[i],data[i+1]) for i in range(0, len(data)-1)]

split_idx = int(len(data_pairs)*TRAIN_TEST_SPLIT )
train = data_pairs[:split_idx]
test = data_pairs[split_idx:]

train_x = np.array([x[0] for x in train])
train_y = np.array([x[1] for x in train])
train_x = np.reshape(train_x, (len(train_x), 1, 1))  # batch, #prev_context, #input nodes
train_y = np.reshape(train_y, (len(train_y), 1))     # batch, #prev_context, #output nodes

test_x = np.array([x[0] for x in test])
test_y = np.array([x[1] for x in test])
test_x = np.reshape(test_x, (len(test_x), 1, 1))  # batch, #prev_context, #input nodes
test_y = np.reshape(test_y, (len(test_y), 1))     # batch, #prev_context, #output nodes

# Build LSTM model.

# Just an example. TODO: modify to improve performance:
model = Sequential()

model.add(LSTM(NUM_LSTM_NODES, batch_input_shape=(1, 1, 1), stateful=True, activation='linear', inner_activation='linear', return_sequences=True))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

# compile model
model.compile(loss='mse', optimizer='adam')
'''
model.add(GRU(NUM_LSTM_NODES, dropout=0.1, recurrent_dropout=0.5, return_sequences=True, batch_input_shape=(1, 1, 1)))
model.add(GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5 ))
model.add(Dense(1, activation='softmax'))
model.summary()
model.compile(loss='mae', optimizer=RMSprop())
'''
# fit model
for i in range(NUM_EPOCHS):
    print("Epoch %d" % i)
    model.fit(train_x, train_y, batch_size=1, epochs=1, shuffle=False, verbose=1)
    model.reset_states()

# Evaluate accuracy
scores = model.evaluate(test_x, test_y, batch_size=1, verbose=1)
model.reset_states()
print("Model Accuracy: %.02f%%" % (scores*100))


if MAKE_PLOTS == True:
    result = []
    for i in range(len(test_x)):
        prediction = model.predict(test_x[i].reshape(1,1,1), verbose=0)
        result.append([prediction, test_y[i]])
    model.reset_states()  # Do this after giving an input sequence, to clear out hidden states.

    predictions = [[x[0][0][0], x[1][0]] for x in result]

    plt.plot(predictions[50:150])
    plt.legend(('pred', 'actual'))
    plt.show()

    errors = [abs(x[1] - x[0]) for x in predictions]

    plt.plot(errors)
    plt.show()