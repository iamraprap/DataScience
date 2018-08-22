# Load the Relevant libraries
import sklearn as sk
from sklearn import datasets
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.precision', 2)  
pd.set_option('display.max_colwidth',10000)
# URL for the AAAI (UW Repository)
url = "./AAAI2014AcceptedPapers.csv"
def tidy_split(df, column, sep=None, eraseList=None, keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    if sep==None:
        return None
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = np.unique(sep.split(presplit))   
        values = [x for x in values if x not in eraseList] if (eraseList!=None and len(eraseList)>0) else values
        values = [x.strip() for x in values if len(x.strip())>0 ]
        if len(values)<1:
            continue
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df

#read data into pandas dataframe
dataAll = pd.read_csv(url, sep=',', header=0)
dataAll = dataAll.dropna()
#dataAll = dataAll.head(10)
print(dataAll.describe().transpose())
dataAll = tidy_split(dataAll, 'title', re.compile(r",| ")) # stem, cvf

def clean(x):
    return x.replace(" and ", ",")

#dataAll['authors'] = dataAll['authors'].apply(clean)
#dataAll = tidy_split(dataAll, 'authors', re.compile(r',')) # categorial/cvf/onehot
#dataAll = tidy_split(dataAll, 'groups', re.compile(r'\n'), None) # stem, cvf
#dataAll = tidy_split(dataAll, 'keywords', re.compile(r'\n'), None) # stem, cvf
#dataAll = tidy_split(dataAll, 'topics', re.compile(r'\n'), None) # stem, tfidf
#dataAll = tidy_split(dataAll, 'abstract', re.compile(r'\n'), None) # stem, tfidf

print(dataAll.describe().transpose())
print("\ncount:%d columns:[%s]" % (len(dataAll.columns), dataAll.columns))






class DataFrameGenericColumnExtractor(TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X[self.column]    

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

class ModelTransformer(TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X))
    
abstract_pipeline = Pipeline([
           ('extract_text', DataFrameGenericColumnExtractor('abstract'))
         , ('count_vec', HashingVectorizer(analyzer="word", stop_words='english', n_features=1000, binary=False))
         , ('tfidf_vec', TfidfTransformer() ) #TfidfVectorizer(analyzer="word", stop_words='english', max_df=0.5, max_features=1000, min_df=2, use_idf=True))
        ])

authors_pipeline = Pipeline([
           ('extract_text', DataFrameGenericColumnExtractor('keywords'))
         , ('count_vec', HashingVectorizer(analyzer="word", stop_words='english', binary=False))
         , ('tfidf_vec', TfidfTransformer() ) #TfidfVectorizer(analyzer="word", stop_words='english', max_df=0.5, max_features=1000, min_df=2, use_idf=True))
        ])

groups_pipeline = Pipeline([
           ('extract_text', DataFrameGenericColumnExtractor('groups'))
         , ('count_vec', HashingVectorizer(analyzer="word", stop_words='english', binary=False))
         , ('tfidf_vec', TfidfTransformer() ) #TfidfVectorizer(analyzer="word", stop_words='english', max_df=0.5, max_features=1000, min_df=2, use_idf=True))
        ])

keywords_pipeline = Pipeline([
           ('extract_text', DataFrameGenericColumnExtractor('keywords'))
         , ('count_vec', HashingVectorizer(analyzer="word", stop_words='english', binary=False))
         , ('tfidf_vec', TfidfTransformer() ) #TfidfVectorizer(analyzer="word", stop_words='english', max_df=0.5, max_features=1000, min_df=2, use_idf=True))
        ])

topics_pipeline = Pipeline([
           ('extract_text', DataFrameGenericColumnExtractor('groups'))
         , ('count_vec', HashingVectorizer(analyzer="word", stop_words='english', binary=False))
         , ('tfidf_vec', TfidfTransformer() ) #TfidfVectorizer(analyzer="word", stop_words='english', max_df=0.5, max_features=1000, min_df=2, use_idf=True))
        ])    





all_features = Pipeline([
           ('union', FeatureUnion([
                           ('abstract_pl', abstract_pipeline)
                         , ('authors_pl', authors_pipeline)
                         , ('groups_pl', groups_pipeline)
                         , ('keywords_pl', keywords_pipeline)
                         , ('topics_pl', topics_pipeline)
                        ])
           )
         , ('cluster', KMeans(n_clusters=2))
        ])    
    
X = dataAll#[['abstract','keywords','groups']]  #dataAll.drop('groups', axis=1)
#y = dataAll[['abstract','keywords']]

matrix = all_features.fit_transform(X)
print(matrix.shape)






print("Matrix")
pd.DataFrame(matrix).head()



plt.scatter(matrix[:,0],matrix[:,1], c=all_features.named_steps['cluster'].labels_, cmap='rainbow')  




n_clusters = 6
print("n_clusters=%d" % (n_clusters))
all_features = Pipeline([
           ('union', FeatureUnion([
                           ('abstract_pl', abstract_pipeline)
                         , ('authors_pl', authors_pipeline)
                         , ('groups_pl', groups_pipeline)
                         , ('keywords_pl', keywords_pipeline)
                         , ('topics_pl', topics_pipeline)
                        ])
           )
         , ('cluster', KMeans(n_clusters=n_clusters))
        ])  

X = dataAll#[['abstract','keywords','groups']]  #dataAll.drop('groups', axis=1)
#y = dataAll[['abstract','keywords']]

matrix = all_features.fit_transform(X)
print(matrix.shape)
plt.scatter(matrix[:,0],matrix[:,1], c=all_features.named_steps['cluster'].labels_, cmap='rainbow')  

print("columns: %s" % dataAll.columns)
gby = dataAll
s = pd.Series(all_features.named_steps['cluster'].labels_)
gby = gby.assign(cluster=s.values)
gby.cluster.apply(str)
#gbyo = gby.groupby('cluster', as_index=False)
print("SeriesValues: %s" % np.unique(s))
print(gby.columns)
print(gby.cluster.unique())



n_sample = 5
gbyo = gby.groupby(['cluster'], as_index=False).apply(lambda x: x.sample(n_sample)).reset_index()
gbyo


gbyo['deeplearning'] = gbyo.apply(lambda row: (1 if (row.abstract.lower().find('deep')>0) \
                                               or (row.keywords.lower().find('deep')>0) \
                                               or (row.groups.lower().find('deep')>0) \
                                               or (row.topics.lower().find('deep')>0) \
                                               else 0) \
                                               , axis=1)

gbyo['reinforcementlearning'] = gbyo.apply(lambda row: (1 if (row.abstract.lower().find('reinforce')>0) \
                                               or (row.keywords.lower().find('reinforce')>0) \
                                               or (row.groups.lower().find('reinforce')>0) \
                                                        or (row.topics.lower().find('reinforce')>0) \
                                               else 0) \
                                               , axis=1)
print(gbyo.cluster.unique())
gbyo.describe()

gbyo_case = gbyo.groupby(['cluster'], as_index=False).agg('sum')
gbyo_case.head()

# from the table above, we can assume that if deeplearning and reinforcementlearning has a score, its an intersect
# and means these two topics are existing in the paper if we use K=cluster#
gbyo[(gbyo.deeplearning==1) & (gbyo.reinforcementlearning==1)].head()