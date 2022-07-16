# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:39:19 2021

"""

""" Importing Packages"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

""" Reading the Data """

data1 = pd.read_csv('A_Z_Handwritten_Data.csv')
data1.info()
print(max(data1))

""" Reducing Memory Usage"""
data = data1.astype('int16')
data.info()
d = data.iloc[:,1:785]
letter = d.values[13000]
letter_img = letter.reshape(28,28)
plt.imshow(letter_img, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')

""" Converting Numerical Data to Character Data """

ch = data['0']+65
s = ch.size
clist=[]
for i in range(s):
    a = chr(ch[i])
    clist.append(a)
c=pd.Series(clist)
   
""" Visual Representation of data """
l =[]
count = np.zeros(26)
for i in range(26):
    a = chr(i+65)
    l.append(a)
    j=0
    count[0] = 1
for i in range(1,s):
    if ch[i] != ch[i-1]:
        j = j+1
    count[j] = count[j]+1   
fig = plt.figure(figsize = (10, 5))
plt.bar(l,count,color='g',width=0.6)
plt.xlabel('Alphabets',fontweight='bold',color='blue')
plt.ylabel('Frequency of letters',fontweight='bold',color='blue')
plt.title('Number of alphabets present in the dataset',fontweight= 'bold',color='r')
plt.show()

""" Splitting Testing  and Training Data """

d_train, d_test = train_test_split(d,test_size=0.2,random_state=42)
c_train, c_test = train_test_split(c,test_size=0.2,random_state=42)

""" Building the Model to recognise the letter O """

c_train_O = (c_train == 'O')
c_test_O = (c_test == 'O')

""" Fitting the Logistic Regression Model"""

clf = LogisticRegression(tol=0.1,solver='liblinear',max_iter=1000)
clf.fit(d_train,c_train_O)

""" Testing Model """

c_predict = clf.predict(d_test)


""" Cross Validation """

score = cross_val_score(clf,d_train,c_train_O,cv=5,scoring='accuracy')
print('Accuracy score')
print(score)
print('Mean Accuracy')
print(score.mean())
c_train_predict = cross_val_predict(clf,d_train,c_train_O,cv=5)

"""Confusion Matrix and other scores """
print('Confusion Matrix')
print(confusion_matrix(c_train_O,c_train_predict))
print('Precision Score')
print(precision_score(c_train_O,c_train_predict))
print('Recall score')
print(recall_score(c_train_O,c_train_predict))
print('f1 Score')
print(f1_score(c_train_O,c_train_predict))
