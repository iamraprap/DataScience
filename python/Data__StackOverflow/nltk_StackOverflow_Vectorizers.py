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
labels = pd.DataFrame([ x for tag in train['tags'] for x in tag])[0].unique()

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



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def count_vectorizer_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with proper parameters choice, 
    # add token_pattern= '(\S+)' to the list of parameter,  '(\S+)'  means any non white space
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.fit_transform(X_val)
    X_test = vectorizer.fit_transform(X_test)
    
    return X_train, X_val, X_test

#Run this cell
X_train_vectorizer, X_val_vectorizer, X_test_vectorizer = count_vectorizer_features(X_train, X_val, X_test)

def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with proper parameters choice, 
    # add token_pattern= '(\S+)' to the list of parameter,  '(\S+)'  means any non white space
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.9, smooth_idf=True)
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.fit_transform(X_val)
    X_test = vectorizer.fit_transform(X_test)
    
    return X_train, X_val, X_test

#Run this cell
X_train_tfidf, X_val_tfidf, X_test_tfidf = tfidf_features(X_train, X_val, X_test)

print('X_test_tfidf ', X_test_tfidf.shape) 
print('X_val_tfidf ',X_val_tfidf.shape)
print('X_val_vectorizer ',X_val_vectorizer.shape)
'''
X_test_tfidf  (20000, 77)
X_val_tfidf  (30000, 78)
X_val_vectorizer  (30000, 16104)
'''



from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=sorted(labels))
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.pipeline import Pipeline

def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
        
    classifier = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression()))])

    return classifier.fit(X_train, y_train)

#model = train_classifier(X_train, y_train)

classifier_vectorizer = train_classifier(X_train_vectorizer, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)

y_val_predicted_labels_vectorizer = classifier_vectorizer.predict(X_val_vectorizer)
y_val_predicted_scores_vectorizer = classifier_vectorizer.decision_function(X_val_vectorizer)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(5):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))