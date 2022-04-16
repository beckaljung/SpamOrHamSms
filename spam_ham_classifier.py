'''Name: Alice Kåhlin and Rebecka Ljung
    ID: alika734 and reblj459
    Course: TNM108
    Year: HT 2021'''

# ------ Import -------
import pandas as pd
import numpy as np 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer

from collections import Counter
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#---- Read and clean the dataset----
# Read the csv file
mails = pd.read_csv('DatasetOfMessages.csv', encoding = 'latin-1')
# Do not need three of the columns in the data set, drop them 
mails = mails.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis=1)
# Rename the columns from v1 and v2 to more accosiateble headers 
mails = mails.rename(columns={'v1':'Spam/ham','v2':'Message'})
# If column that includes Spam or Ham has ham set to 0 and Spam set to 1
mails['Spam'] = mails['Spam/ham'].map({'ham': 0, 'spam': 1})
# Do not work with string, so drop Spam/ham, working with numbers 0 and 1 instead
mails.drop(['Spam/ham'], axis = 1, inplace = True)

# Get the total number of messages in the data set
totMails = mails['Message'].shape[0]

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    mess = mess.lower()

    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure', 'å']

    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return ' '.join([word for word in nopunc.split() if word not in STOPWORDS])

mails['Clean_Message'] = mails['Message'].apply(text_process)

# Same process for the spam words, Spam words is when column Spam contains 1
words_spam = mails[mails['Spam']==1].Clean_Message.apply(lambda x: [word.lower() for word in x.split()])
spam_words = Counter()

for msg in words_spam:
    spam_words.update(msg)

#----- Split the dataset into training and test sets -----
mess_train, mess_test, spam_train, spam_test = train_test_split(mails['Clean_Message'], mails['Spam'], test_size=0.25, random_state = 1, shuffle = True)

# ---- Pipline with NB ----
pipe = Pipeline([('vect', CountVectorizer()), 
                 ('tfidf', TfidfTransformer()),  
                 ('clf', MultinomialNB())])

# ---- Pipline with Logistic Regression ----
pipe_log = Pipeline([('vect', CountVectorizer()),
                     ('clf', LogisticRegression(solver='lbfgs', max_iter=1000))])

# ---- Pipline with SVM ----
pipe_svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
    ,max_iter=5, tol=None)),
])

# ---- Pipline with Random Forest ----
pipe_random_forest = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier(n_estimators=100,max_depth=None,n_jobs=-1)),
])

# ---- Grid Search ----
parameters={
    'vect__ngram_range':[(1,1),(1,2)],
    'tfidf__use_idf':(True, False),
    'clf__alpha':(1e-2,1e-3), 
}

parameters_log = {
    'clf__C' : np.logspace(-4, 4, 20),
}

parameters_rf={
    'vect__ngram_range':[(1,1),(1,2)],
    'tfidf__use_idf':(True, False),
}

gs_NB_clf = GridSearchCV(pipe, parameters, cv=5, n_jobs=-1)
gs_NB_clf = gs_NB_clf.fit(mess_train, spam_train)
gs_NB_pred = gs_NB_clf.predict(mess_test)
print("DONE GRIDSEARCH NB")

gs_log_clf = GridSearchCV(pipe_log, parameters_log, cv=5, n_jobs=-1)
gs_log_clf = gs_log_clf.fit(mess_train, spam_train)
gs_log_clf = gs_log_clf.predict(mess_test)
print("DONE GRIDSEARCH LOG")

gs_SVM_clf = GridSearchCV(pipe_svm, parameters, cv=5, n_jobs=-1)
gs_SVM_clf = gs_SVM_clf.fit(mess_train, spam_train)
gs_SVM_clf = gs_SVM_clf.predict(mess_test)
print("DONE GRIDSEARCH SVM")

gs_rf_clf = GridSearchCV(pipe_random_forest, parameters_rf, cv=5, n_jobs=-1)
gs_rf_clf = gs_rf_clf.fit(mess_train, spam_train)
gs_rf_clf = gs_rf_clf.predict(mess_test)
print("DONE GRIDSEARCH RANDOM FOREST")

#----- Print the results -----
print("---------NB---------")
print(metrics.accuracy_score(spam_test, gs_NB_pred))
print(metrics.confusion_matrix(spam_test, gs_NB_pred))

print("---------Log---------")
print(metrics.accuracy_score(spam_test, gs_log_clf))
print(metrics.confusion_matrix(spam_test, gs_log_clf))

print("---------SVM---------")
print(metrics.accuracy_score(spam_test, gs_SVM_clf))
print(metrics.confusion_matrix(spam_test, gs_SVM_clf))

print("---------Random Forest---------")
print(metrics.accuracy_score(spam_test, gs_rf_clf))
print(metrics.confusion_matrix(spam_test, gs_rf_clf))

print("---------Spam words--------")
print(spam_words.most_common(10))

#----- Word Clouds -----
STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure', 'å']
spam_words = ' '.join(list(mails[mails['Spam'] == 1]['Message']))
spam_wc = WordCloud(stopwords = STOPWORDS, width = 512, height = 512).generate(spam_words)
plt.figure(figsize=(10,8), facecolor='k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

spam_wc2 = WordCloud(stopwords = STOPWORDS, width = 512, height = 512, max_words=10).generate(spam_words)
plt.figure(figsize=(10,8), facecolor='k')
plt.imshow(spam_wc2)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
