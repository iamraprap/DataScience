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
test = pd.read_csv('test.tsv', sep='\t')

train.head()

#Split the data into train/val/test
X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values

#We'll be working with regular expressions to clean the text data
import re


replace_re_by_space = re.compile('[/(){}\[\]\|@,;]')
delete_re_symbols = re.compile('[^0-9a-z #+_]')
stop_words =  set(stopwords.words('english'))


def text_processing(text):
    """
        Input text: string
        
        Output: modified text based on RE
    """
    text = text.lower() # add a function to convert text to lowercase
    text = re.sub(replace_re_by_space, ' ', text) # add a function that remove all symbols in replace_re_by_space symbols and replace them by space in text
    text = re.sub(delete_re_symbols, '', text) # add function that simply remove all symbols in delete_re_symbols from text
    token_word=word_tokenize(text)
    filtered_sentence = [w for w in token_word if not w in stop_words] # filtered_sentence contain all words that are not in stopwords dictionary
    lenght_of_string=len(filtered_sentence)
    text_new=""
    for w in filtered_sentence:
        if w!=filtered_sentence[lenght_of_string-1]:
             text_new=text_new+w+" " # when w is not the last word so separate by whitespace
        else:
            text_new=text_new+w
    text = text_new# remove stopwords from text, nothing to do here
    return text


def text_processing_test():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function", 
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_processing(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return "CONGRATS! ALL TESTS PASSED!"



#This should not throw an exception
print(text_processing_test())
'''
CONGRATS! ALL TESTS PASSED!
'''
X_train = [text_processing(x) for x in X_train]
X_val = [text_processing(x) for x in X_val]
X_test = [text_processing(x) for x in X_test]
print(X_train[:3])
'''
['draw stacked dotplot r', 'mysql select records datetime field less specified value', 'terminate windows phone 81 app']
'''
