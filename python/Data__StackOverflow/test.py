import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from ast import literal_eval
from nltk.tokenize import word_tokenize

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

train = read_data('train.tsv')
validation = read_data('validation.tsv')
tags = pd.DataFrame([ x for tag in train['tags'] for x in tag])[0].unique()
print(tags)