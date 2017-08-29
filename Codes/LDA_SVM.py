# Stumbleupon Evergreen Classification Challenge Dataset


#Writing boilerplate of each URL into separate file
#for train data
with open("train_boilerplate.txt", "r") as ins:
	i = 1
	for line in ins:
		name = "train" + str(i) + ".txt"
		f = open(name,"w")
		f.write(line)
		i += 1 
print "Conversion done for train data ..........."
#for test data
with open("test_boilerplate.txt", "r") as ins:
	i = 1
	for line in ins:
		name = "test" + str(i) + ".txt"
		f = open(name,"w")
		f.write(line)
		i += 1 
print "Conversion done for test data ..........."

# Importing the Required packages
from gensim import corpora
from sklearn.metrics import roc_auc_score
import numpy as np
from collections import defaultdict
from pprint import pprint  # pretty-printer
import logging, gensim
from sklearn import *
from sklearn.svm import *
import collections
from sklearn.feature_extraction import FeatureHasher

# Reading the training set of Documents
documents = []
for i in range(0, 7395):
    f1 = open("train" + str(i+1) + ".txt", "r")
    documents.append(f1.read())

# Two pre-processing steps discussed in the report
# Using ONIX stopword list and removing all the stopwords from the documents    
f = open("stopword.txt")
stoplist = set(f.read().split('\n'))
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# Remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
          for text in texts]

# Storing the pre-processed documents as a dictionary, for future reference
dictionary = corpora.Dictionary(texts)
dictionary.save('arf.dict') 

# Converting the texts into a corpus .mm format for gensims LDA implementation
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = gensim.models.TfidfModel(corpus) 
corpora.MmCorpus.serialize('arf.mm', corpus) 
id2word = corpora.Dictionary.load('arf.dict')
mm = corpora.MmCorpus('arf.mm')

# LDA function being called. Parameteres played with - (num_topics, i.e. number of topics and passes, i.e. the amount of learning for the topics)
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=10, update_every=0, passes=10)
X_train = [lda[text] for text in corpus]
siz = len(X_train)
X_train2 = [[0 for i in range(50)] for j in range(siz)]
count = -1
for row in X_train:
    count = count+1
    for(key, val) in row:
        X_train2[count][key] = val

Y_train=[]
for i in range(0, 7395):
    if i<=3601:
        Y_train.append(0)
    else:
        Y_train.append(1)

# Getting the top 10 topics generated using LDA    
lda.print_topics(10)
for i in range(0, lda.num_topics-1):
    print lda.print_topic(i)
    

################################ TESTING ######################################
documents = []
for i in range(0, 3171):
    f1 = open("test1" + str(i+1) + ".txt", "r")
    documents.append(f1.read())

texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
          for text in texts]

dictionary = corpora.Dictionary(texts)
dictionary.save('arf1.dict') 

corpus1 = [dictionary.doc2bow(text) for text in texts]
tfidf1 = gensim.models.TfidfModel(corpus1)
corpora.MmCorpus.serialize('arf1.mm', corpus1) 
id2word = corpora.Dictionary.load('arf1.dict')

mm = corpora.MmCorpus('arf1.mm')
lda = gensim.models.ldamodel.LdaModel(corpus=corpus1, id2word=id2word, num_topics=10, update_every=0, passes=10)
X_test = [lda[text] for text in corpus1]
siz = len(X_test)
X_test2 = [[0 for i in range(50)] for j in range(siz)]
count = -1
for row in X_test:
    count = count+1
    for(key, val) in row:
        X_test2[count][key] = val

# Training the SVM
clf = SVC()
params = dict(gamma=[0.001, 0.01,0.1, 0.2, 1, 10, 100],C=[1,10,100,1000], kernel=["rbf", "linear"])
clf = grid_search.GridSearchCV(clf,param_grid=params,cv=2, scoring='f1')
# SVM predicting
clf.fit(X_train2, Y_train)
a = clf.predict(X_test2)
print collections.Counter(a)

