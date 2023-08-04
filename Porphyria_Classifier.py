#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
pd.options.display.max_columns = 999


# In[34]:


df = pd.read_csv (r'/mnt/rudrapatnv_shared/Balu/Porphyria/Imbalance_350_final.csv')



# In[38]:


import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]




df=df.fillna(0.01)
X = df.drop(['Label'],axis=1)
y = df['Label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
cv=StratifiedKFold(n_splits=5,shuffle=False, random_state=None)


# In[41]:


logreg = LogisticRegression(solver = 'lbfgs')
logreg.fit(X_train,y_train)
y_pred_test = logreg.predict(X_test)
acc = accuracy_score(y_test,y_pred_test)
fscore=f1_score(y_test,y_pred_test)
print("Accuracy of LogisticRegression: %.2f%%" % (acc * 100.0))
print("Fscore of LogisticRegression: %.2f%%" % (fscore * 100.0))


# In[42]:


gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred_test1 = gnb.predict(X_test)

fscore=f1_score(y_test,y_pred_test1)
acc = accuracy_score(y_test,y_pred_test1)
print("Accuracy of GaussianNB: %.2f%%" % (acc * 100.0))
print("Fscore of GaussianNB: %.2f%%" % (fscore * 100.0))


# In[43]:


clf = KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree')

clf.fit(X_train,y_train)
y_pred3 = clf.predict(X_test)
acc3 =   accuracy_score(y_test,y_pred3)
fscore=f1_score(y_test,y_pred3,average='weighted')
print("Accuracy of KNN: %.2f%%" % (acc3 * 100.0))
print("Fscore of KNN: %.2f%%" % (fscore * 100.0))


# In[44]:


svc1 = SVC(C=50,kernel='rbf',gamma=1)     

svc1.fit(X_train,y_train)
y_pred4 = svc1.predict(X_test)

fscore=f1_score(y_test,y_pred4,average='weighted')
acc4=    accuracy_score(y_test,y_pred4)
print("Accuracy of SVM: %.2f%%" % (acc4 * 100.0))
print("Fscore of SVM: %.2f%%" % (fscore * 100.0))


# In[45]:


dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

y_pred67 = dt.predict(X_test)
fscore=f1_score(y_test,y_pred67)
acc2 = accuracy_score(y_test,y_pred67)
print("Accuracy of Decision Tree: %.2f%%" % (acc2 * 100.0))
print("Fscore of Decision Tree: %.2f%%" % (fscore * 100.0))


# In[46]:


rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred2 = rf.predict(X_test)

fscore=f1_score(y_test,y_pred2,average='weighted')
acc2 = accuracy_score(y_test,y_pred2)

print("Accuracy of Random Forest: %.2f%%" % (acc2 * 100.0))
print("Fscore of Random Forest: %.2f%%" % (fscore * 100.0))


# In[47]:


et = ExtraTreesClassifier()
et.fit(X_train,y_train)
y_pred21 = et.predict(X_test)

fscore=f1_score(y_test,y_pred21,average='weighted')
acc2 = accuracy_score(y_test,y_pred21)

print("Accuracy of Extra Tree: %.2f%%" % (acc2 * 100.0))
print("Fscore of Extra Tree: %.2f%%" % (fscore * 100.0))


# In[48]:


ab= AdaBoostClassifier()
ab.fit(X_train,y_train)
y_pred22 = ab.predict(X_test)

fscore2=f1_score(y_test,y_pred22,average='weighted')
acc22 = accuracy_score(y_test,y_pred22)

print("Accuracy of AdaBoost: %.2f%%" % (acc22 * 100.0))
print("Fscore of AdaBoost: %.2f%%" % (fscore2 * 100.0))


# In[49]:

#Save model to disk
import joblib
filename = 'RF_350_IB.sav'
joblib.dump(rf, filename)


filename = 'KNN_350_IB.sav'
joblib.dump(clf, filename)


# In[50]:


filename = 'SVM_350_IB.sav'
joblib.dump(svc1, filename)


# In[ ]:




