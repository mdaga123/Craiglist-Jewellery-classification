####Unstructured Data Final Project


import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import *

df_raw=pd.read_excel('Sheet for classification.xlsx')

#Merge title and text columns
df_raw['Classification']=df_raw['Classification'].str.lower()
df_raw['title_text_combined']=df_raw['Title'].map(str)+df_raw['Text'].map(str)

#Split into train and test dataset
training_x=df_raw[0:340]['title_text_combined']
training_c=df_raw[0:340]['Classification']

testing_x=df_raw[341:]['title_text_combined']
testing_c=df_raw[341:]['Classification']

#Tokenization for training set
train_title_text_combined_list=[]
train_full_string_list=[]

for i in range(len(training_x)):
    tokens_combine=nltk.tokenize.word_tokenize(str(training_x.iloc[i]))
  
    #lower case all entries
    for i in range(len(tokens_combine)):
        tokens_combine[i]=tokens_combine[i].lower()
    
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens_combine if token.isalpha()]

    #remove stop words
    stop_words_removed = [token for token in lemmatized_tokens if token not 
                          in stopwords.words('english') if token.isalpha()]
    
    #combine into a large string
    full_string=''
    for i in range(len(stop_words_removed)):
        full_string+=' '+stop_words_removed[i]
    
    train_full_string_list.append(full_string[1:]) #remove space at the beginning

#TF-IDF
vectorizer2 = TfidfVectorizer(ngram_range=(1,2), min_df=3)
vectorizer2.fit(train_full_string_list)

v2 = vectorizer2.transform(train_full_string_list)
array_final=v2.toarray()

training_x=pd.DataFrame(array_final)

#Tokenization for test set
test_title_text_combined_list=[]
test_full_string_list=[]

for i in range(len(testing_x)):
    tokens_combine=nltk.tokenize.word_tokenize(str(testing_x.iloc[i]))
  
    #lower case all entries
    for i in range(len(tokens_combine)):
        tokens_combine[i]=tokens_combine[i].lower()
    
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens_combine if token.isalpha()]

    #remove stop words
    stop_words_removed = [token for token in lemmatized_tokens if token not 
                          in stopwords.words('english') if token.isalpha()]
    
    #combine into a large string
    full_string=''
    for i in range(len(stop_words_removed)):
        full_string+=' '+stop_words_removed[i]
    
    test_full_string_list.append(full_string[1:]) #remove space at the beginning

#TF-IDF
vectorizer2 = TfidfVectorizer(ngram_range=(1,2), min_df=3)
vectorizer2.fit(test_full_string_list)

v2 = vectorizer2.transform(test_full_string_list)
array_final=v2.toarray()

testing_x=pd.DataFrame(array_final)

## Naive Bayes
from sklearn.naive_bayes import MultinomialNB
NBmodel = MultinomialNB()
# training
NBmodel.fit(training_x, training_c)
y_pred_NB = NBmodel.predict(testing_x)
# evaluation
acc_NB = accuracy_score(testing_c, y_pred_NB)
print("Naive Bayes model Accuracy:: {:.2f}%".format(acc_NB*100))

#SVC
from sklearn.svm import LinearSVC
SVMmodel = LinearSVC()
# training
SVMmodel.fit(training_x, training_c)
y_pred_SVM = SVMmodel.predict(testing_x)
# evaluation
acc_SVM = accuracy_score(testing_c, y_pred_SVM)
print("SVM model Accuracy:
{:.2f}%".format(acc_SVM*100))
