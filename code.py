#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""  This cell is importing all the necessary libraries we need to make this script run """ 

import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
from nltk.corpus import stopwords
import nltk.corpus
import pandas as pd

# Importing libraries for splitting and training the model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


# In[2]:


# reading in the training data
""" --- Make sure that this python script is in the same directory/folder as the input data --- """

data = pd.read_excel('NLP_Data1.xlsx')


# In[3]:


# Sanity check to make sure that we have the correct number of rows and columns PRIOR to pre-processing

print('Output: There are', data.shape[0], 'rows and', data.shape[1], 'columns')


# In[4]:


"""
This is the first part of the pre-processing.
We are removing ALL the call center notes and
just leaving in the app reviews
"""

data['Document']=data['Document'].str.lower()
data = data[data['Document'].str.contains('noob')==False]
data = data[data['Document'].str.contains('cx')==False]
data = data[data['Document'].str.startswith('rv')==False]
data = data[data['Document'].str.startswith('ur')==False]
data = data[data['Document'].str.startswith('ts')==False]
data = data[data['Document'].str.startswith('cx')==False]
data = data[data['Document'].str.contains('-mac:')==False]


# In[5]:


# after first set of pre-processing, we want to make sure we still have the 2 columns. We should see a reduced number of rows

print(' After removing the call center notes, there are' , data.shape[0], 'rows and', data.shape[1], 'columns')


# In[6]:


# We are calling out the training columns

Document = data.Document 
Documents = pd.Series(Document.values.tolist())
Symptom = data.Symptom
Symptoms = pd.Series(Symptom.values.tolist())


# In[7]:


# This function cleans basic aspects of the text present in the "Document" column
import re

def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """

    for i in range(len(text)):
        # remove HTML tags 
        text[i] = re.sub(r'<.*?>', '', text[i])
    
        # remove the characters [\], ['] and ["]
        text[i] = re.sub(r"\\", " ", text[i])    
        text[i] = re.sub(r"\'", " ", text[i])    
        text[i] = re.sub(r"\"", " ", text[i])
        text[i] = re.sub(r"\W", " ", text[i])
        text[i] = re.sub(r"app", " ", text[i])
    
        # convert text to lowercase
        text[i] = text[i].strip().lower()
    
        # replace punctuation characters with spaces
        filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        translate_dict = dict((c, " ") for c in filters)
        translate_map = str.maketrans(translate_dict)
        text[i] = text[i].translate(translate_map)

    return text


# In[8]:


# Running the pre-processing function on the "Document" column in the dataframe
Documents = clean_text(Documents)


# In[9]:


Documents


# In[10]:


Documents = list(Documents)
Symptoms = list(Symptom)


# In[11]:


Documents


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(Documents, Symptoms, test_size=0.2, random_state = 42)


# In[13]:


# Checking the size of the training set

len(X_train)


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer

#vectorizer = CountVectorizer(stop_words="english")
vectorizer = TfidfVectorizer()

training_features = vectorizer.fit_transform(X_train)
test_features = vectorizer.fit_transform(X_test)


# In[ ]:


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(training_features, y_train)
clf.out_activation_ = 'softmax'


# In[ ]:


predicted_results = clf.predict(test_features)


# In[ ]:


accuracy = (np.sum(predicted_results == y_test)/len(y_test))*100


# In[ ]:


accuracy

