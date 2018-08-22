import math
import os
import string

import matplotlib.pyplot as plt

from IPython.display import SVG

from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, GRU, LSTM, Dense, Masking, Dropout, Embedding, Flatten, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l1_l2
from keras.utils import to_categorical
from keras.utils.vis_utils import model_to_dot
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1  # Start with 10% of the GPU RAM
config.gpu_options.allow_growth = True                    # Dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)                                         # Set this TensorFlow session as the default session for Keras

TRAIN_TEST_SPLIT = 0.7           # % of data in training set

NUM_LSTM_NODES = 256             # Num of intermediate LSTM nodes
CONTEXT_VECTOR_SIZE = 256        # Size of context vector (num of LSTM nodes in final LSTM layer)

EMBEDDING_DIM = 100              # Embedding layer size for input words

BATCH_SIZE = 64
NUM_EPOCHS = 500

NUM_DATA_EXAMPLES = 5000         # limit memory usage while experimenting

LR = 0.01
DROPOUT = 0.3

MAX_NUM_WORDS = 32767

def add_space_around_punctuation(s):
    result = ''
    for c in s:
        if c in string.punctuation and c != "'":  # Apostrophes are important
            result += ' %s ' % c
        else:
            result += c
    return result

def clean_sentence(s):
    s = s.strip()
    s = s.lower()
    s = add_space_around_punctuation(s)
    return s

def get_words_from_sentence(s, add_start_symbol=False, add_end_symbol=False, reverse=False):
    words = list(filter(None, s.split(' ')))
    if reverse:
        words = words[::-1]
    if add_start_symbol:
        words = ['<S>'] + words
    if add_end_symbol:
        words.append('</S>')
    return words

def get_word_list_from_sentence_string(s, add_start_symbol=False, add_end_symbol=False, reverse=False):
    return get_words_from_sentence(clean_sentence(s), add_start_symbol, add_end_symbol, reverse)    
    
def get_sentences(path, filename, add_start_symbol=False, add_end_symbol=False, reverse=False):
    with open(os.path.join(path, filename), 'r') as f:
        lines = f.readlines()
        return [get_word_list_from_sentence_string(s, add_start_symbol, add_end_symbol, reverse) 
                for s in lines]

def get_word_set(sentences):
    words = set()
    for s in sentences:
        for word in s:
            words.add(word)
    return words


# Store the input sentences (English) in s1
# Store the target senteces (French) in s2

# Consider reversing the input sentences to improve trianing.
# Add start and stop symbols for the decoder.
PATH = '../data/AsnLib/'
s1 = get_sentences(PATH, 'small_vocab_en.txt', ...)       # TODO
s2 = get_sentences(PATH, 'small_vocab_fr.txt', ...)       # TODO

# Restruct to a subset of the data
s1 = s1[:NUM_DATA_EXAMPLES]
s2 = s2[:NUM_DATA_EXAMPLES]

tokenizer_1 = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_1.fit_on_texts(s1)
sequences_1 = tokenizer_1.texts_to_sequences(s1)
word_index_1 = tokenizer_1.word_index
print('Found %s unique tokens @ s1' % len(word_index_1))

tokenizer_1 = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_1.fit_on_texts(s1)
sequences_1 = tokenizer_1.texts_to_sequences(s1)
word_index_1 = tokenizer_1.word_index
print('Found %s unique tokens @ s1' % len(word_index_1))

tokenizer_2 = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_2.fit_on_texts(s2)
sequences_2 = tokenizer_2.texts_to_sequences(s2)
word_index_2 = tokenizer_2.word_index
print('Found %s unique tokens @ s2' % len(word_index_2))
    
# Create two lists, w1 and w2, which hold the set of all words that show up in s1 and s2.
w1 = [k for k in word_index_1.keys()]
w2 = [k for k in word_index_2.keys()]
print("len(w1) = %d" % (len(w1)))
print("len(w2) = %d" % (len(w2)))

def get_word_to_index_dict(words):
    return {w: i+1 for i,w in enumerate(words)}  # use i+1 to reserve 0 for the mask index
def reverse_dict(d):
    return {v: k for k,v in d.items()}

word_to_index1 = get_word_to_index_dict(w1)
word_to_index2 = get_word_to_index_dict(w2)
index_to_word1 = reverse_dict(word_to_index1)
index_to_word2 = reverse_dict(word_to_index2)
index_to_word1[0] = '<MASK>'
index_to_word2[0] = '<MASK>'

def sentence_to_indices(s, word_to_index):
    """Input s is a sentence string. word_to_index is a dict mapping words to indices.
    
    This function should convert a sentence to a list of indices, such as [5, 2, 17, 3], and return the list."""
    return [word_to_index[k] for k in s if k in word_to_index.keys() ]

def indices_to_sentence(indices, index_to_word):
    """indices is a list of word indices. word_to_index is a dict mapping indices to words.
    
    This function should convert the indices list, such as [5, 2, 17, 3], to a list of word strings, and 
    return the list."""
    return [index_to_word[k] for k in indices if k in index_to_word.keys() ]    

# Record the number of words in the input and output data, respectively.
num_words_X = len(w1) + 1  # add 1 to reserve 0 for mask
num_words_y = len(w2) + 1  # add 1 to reserve 0 for mask

# Convert the input sentences in s1 to a list of sentences each represented as a list of integers.
# For example, the output list might look like [[5, 2, 17, 3], [1, 9, 85, 3, 22, 9], ...]
# Do the same for the output sentences.
inputs_as_indices = [ [word_to_index1[t] for t in s if t in word_to_index1.keys()] for s in s1 ]
outputs_as_indices = [ [word_to_index2[t] for t in s if t in word_to_index2.keys()] for s in s2  ]

print("len(inputs_as_indices) = %d" % (len(inputs_as_indices)))
print("len(outputs_as_indices) = %d" % (len(outputs_as_indices)))

# Now pad the input and output index sequences with a filler (index 0) so that all sequences for each LSTM have the 
# same length. Use the keras function pad_sequences to do this easily.
# Hint: For the inputs, padding should be on the left, like so: [[0, 0, 5, 2, 17, 3], ...]
#       For the outputs, padding should be on the right, like so: [[9, 7, 5, 4, 0, 0, 0], ...]
inputs = pad_sequences(inputs_as_indices, dtype = "int32", padding = "pre", truncating = "pre", value = 0)
outputs = pad_sequences(outputs_as_indices, dtype = "int32", padding = "post", truncating = "post", value = 0)
print("len(inputs) = %d" % (len(inputs)))
print("len(outputs) = %d" % (len(outputs)))
print(indices_to_sentence(inputs[0], index_to_word1))
print(indices_to_sentence(outputs[0], index_to_word2))

# compute the maximum sequence length of the inputs and outputs, just to see how they look.
max_seq_len_X = len(inputs[0])
max_seq_len_y = len(outputs[0])
max_seq_len_X, max_seq_len_y

# Just for convenience: define some more expressive variable names
max_input_seq_len = max_seq_len_X
max_output_seq_len = max_seq_len_y
num_input_words = num_words_X
num_output_words = num_words_y
num_words_X, num_words_y

# Create train and test sets.
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, 
                                                    test_size=1 - TRAIN_TEST_SPLIT,
                                                    random_state=42)

X_train_one_hot = to_categorical(X_train)
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_input_words))
encoder = LSTM(NUM_LSTM_NODES, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_output_words))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(NUM_LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_output_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

#encoder_input_data = X_train
encoder_input_data = np.zeros(X_train_one_hot.shape)
encoder_input_data[:,:-1] = X_train_one_hot[:,1:,:]

#decoder_input_data = y_train
decoder_input_data = np.zeros(y_train_one_hot.shape)
decoder_input_data[:,:-1] = y_train_one_hot[:,1:,:]

decoder_target_data = np.zeros(y_train_one_hot.shape)
decoder_target_data[:,:-1] = y_train_one_hot[:,1:,:]

decoder_target_data_test = np.zeros(y_test_one_hot.shape)
decoder_target_data_test[:,:-1] = y_test_one_hot[:,1:,:]

print("Shapes")
print(encoder_input_data.shape)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2)
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(NUM_LSTM_NODES,))
decoder_state_input_c = Input(shape=(NUM_LSTM_NODES,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_output_words))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, word_to_index2['<S>']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = index_to_word2[sampled_token_index]
        decoded_sentence += sampled_word

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '\n' or
           len(decoded_sentence) > max_output_seq_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_output_words))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    #print('Input sentence:', indices_to_sentence(input_seq, index_to_word2))
    print('Decoded sentence:', decoded_sentence)