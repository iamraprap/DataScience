import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

 
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.precision', 3)  
pd.set_option('display.max_colwidth',10000)

D = []
T = []

"""Compute candidate 1-itemset"""
C1 = {}
"""total number of transactions contained in the file"""
transactions = 0
with open("NISTraffic.csv", "r") as f:
    for line in f:
        T = []
        transactions += 1 
        if transactions <= 1155050:#Testing
            for word in line.split(","):
                if word != '' and word != '\n':
                    T.append(word)
                    if word not in C1.keys():
                        C1[word] = 1
                    else:
                        count = C1[word]
                        C1[word] = count + 1
            D.append(T)

print(len(D))

te = TransactionEncoder()
te_ary = te.fit(D).transform(D)
df = pd.DataFrame(te_ary, columns=te.columns_)

msupport     = [0.95, 0.96, 0.97, 0.98, 0.99, 0.999]
tconfidence  = [ 0.7, 0.75, 0.8,  0.85, 0.90, 0.95]

n_items = []
support = []
confidence = []

for m in msupport:
    for t in tconfidence:
        print("m=%s, t=%s" % (m, t))
        frequent_itemsets = apriori(df, min_support=m, use_colnames=False)
        #print(frequent_itemsets)
        ar = association_rules(frequent_itemsets, metric="confidence", min_threshold=t)
        #print(ar)
        n_items.append(len(ar))
        support.append(m)
        confidence.append(t)
        #plt.scatter(ar[['support']], ar[['confidence']])  # plot all points

sw = pd.DataFrame()
sw.loc[:,'support'] = pd.Series(support)
sw.loc[:,'confidence'] = pd.Series(confidence)
sw.loc[:,'n_items'] = pd.Series(n_items)
sw
#plt.plot(sw[['support']], sw[['confidence']], sw[['n_items']], linewidth=2.0)
#plt.scatter(sw[['support']], sw[['confidence']], c=sw[['n_items']], cmap='rainbow')
#plt.show()

#plt.plot(sw[['n_items']])
#plt.ylabel('some numbers')
#plt.show()