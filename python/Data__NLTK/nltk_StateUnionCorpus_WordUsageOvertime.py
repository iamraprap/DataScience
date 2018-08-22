import nltk
#nltk.download() #need to download these in specific paths that the tool will look for
from nltk.corpus import state_union

cfd = nltk.ConditionalFreqDist(
    (target, fileid.split('-')[0])
    for fileid in state_union.fileids()
    for word in state_union.words(fileid)
    for target in ['men', 'women', 'people']#, 
    if word.lower()==target)
cfd.plot(cumulative=True)

'''
Please check word_usage_overtime.png uploaded

The data shows that there is exponential growth of the usage of 'people overtime'
'''