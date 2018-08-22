import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

 
import pandas as pd


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
frequent_itemsets = apriori(df, min_support=0.95, use_colnames=False)

frequent_itemsets
association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)