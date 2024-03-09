#!/usr/bin/env python
# coding: utf-8

# In[1]:


#reading data from mongodb 
import pymongo
from pymongo import MongoClient
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
#connecting to mongo sever
client = MongoClient('localhost',27017)
db = client.amazon
data = db.amz_collection

#reading data through mongodb
df = pd.DataFrame(list(data.find()))
df


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 
import seaborn as sns
import string
import nltk
import warnings 
#nltk.download('stopwords')
#nltk.download('wordnet')
from wordcloud import WordCloud


# In[3]:


#showing inforamtion of columns
df.info()
df.columns


# In[4]:


pd.options.mode.chained_assignment = None  # default='warn'

df['reviewsrating']=df.reviewsrating.astype("float")


# In[5]:


# Creating a new column depend of reviewsrating
#Dividing review rating into two variable 1 and 0
#where 1 represents as  positvie revivew and 2 represent as negative review

df['label'] = df['reviewsrating'].apply(lambda x : 1 if x >= 4 else 0)


# In[6]:


#changing name of varibale for better understanding
df[['text', 'rating']] = df[['comb_review', 'label']]
df.head()


# In[7]:


review=pd.DataFrame(df.groupby('reviewsrating').size().sort_values(ascending=False).rename('No of Users').reset_index())
review.head()


# In[8]:


#To remove specific sting from text
def remove_pattern(text, pattern):
    
    # find all the pattern in the  text 
    r = re.findall(pattern, text)
    
    # replace the pattern with an empty space
    for i in r: text = re.sub(pattern, '', text)
    
    return text


# In[9]:


# lower case every word 
df['text'] = df['text'].str.lower()
df['text']=df['text'].astype(str)


# tokenize the text to search for any stop words to remove it
df['tokenized_text'] = df['text'].apply(lambda x : x.split())

# creating a set of stopwords
stopWords = set(nltk.corpus.stopwords.words('english'))

#removing stopwords from tokenized_text
df['tokenized_text'] = df['tokenized_text'].apply(lambda x : [word for word in x if not word in stopWords])

# create a word lemma

lemma = nltk.stem.WordNetLemmatizer()
pos = nltk.corpus.wordnet.VERB


df['tokenized_text'] = df['tokenized_text'].apply(lambda x : [lemma.lemmatize(word, pos) for word in x])

# removing punctuation

df['tokenized_text'] = df['tokenized_text'].apply(lambda x : [ remove_pattern(word,'\.') for word in x])

# rejoin the text again to get a cleaned text
df['cleaned_text'] = df['tokenized_text'].apply(lambda x : ' '.join(x))

#drop tokenized_text column
df.drop(labels=['tokenized_text'], axis=1, inplace=True)

df.head()


# In[10]:


df[['text','cleaned_text']]


# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer
#max_df means for occurred in too many documents
#min_df means occurred in too few documents (min_df)
#max_features were cut off by feature selection.
tfidf_Vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=1000, stop_words='english')

tfidf_features = tfidf_Vectorizer.fit_transform(df['cleaned_text'])

tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_Vectorizer.get_feature_names())

tfidf_df.head(5000)


# In[12]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(tfidf_df, df['label'], test_size=0.2, random_state=42)


# In[13]:



from sklearn.ensemble import RandomForestClassifier
clf_tfidf_RF=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf_tfidf_RF.fit(X_train,y_train)

pred_tfidf_RF=clf_tfidf_RF.predict(X_test)


from sklearn.metrics import classification_report, accuracy_score, f1_score,confusion_matrix

print("Random forest Classifier")
print("Accuracy Socre: ",(100 * accuracy_score(y_test, pred_tfidf_RF)))


# In[14]:


#predicted values
print(pred_tfidf_RF)


# In[15]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

clf_tfidf_ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.001)

clf_tfidf_ada.fit(X_train, y_train)


pred_tfidf_ada = clf_tfidf_ada.predict(X_test)


from sklearn.metrics import classification_report, accuracy_score, f1_score,confusion_matrix

print("AdaBoostClassifier ")
print("Accuracy Socre: ",(100 * accuracy_score(y_test, pred_tfidf_ada)))


# In[16]:


from sklearn.naive_bayes import BernoulliNB
clf_BNB = BernoulliNB()
clf_BNB.fit(X_train, y_train)
pred_tfidf_BNB = clf_BNB.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score, f1_score,confusion_matrix

print("BernoulliNB")
print("Accuracy Socre: ",(100 * accuracy_score(y_test, pred_tfidf_BNB)))


# In[17]:


from sklearn.naive_bayes import MultinomialNB
clf_MNB = MultinomialNB()
clf_MNB.fit(X_train, y_train)
pred_tfidf_MNB = clf_MNB.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score, f1_score,confusion_matrix

print("MultinomialNB")
print("Accuracy Socre: ",(100 * accuracy_score(y_test, pred_tfidf_MNB)))

 


# In[18]:


##Using textblob library , we show polarity of each product
from textblob import TextBlob
df['emotion']=df['cleaned_text'].apply(lambda x:TextBlob(x).sentiment.polarity)


# In[19]:


product_polarity=pd.DataFrame(df.groupby('name')['emotion'].mean().sort_values(ascending=True))

plt.figure(figsize=(18,20))
plt.xlabel('Emotion')
plt.ylabel('Products')
plt.title('Polarity of Product Reviews')
polarity_graph=plt.barh(np.arange(len(product_polarity.index)),product_polarity['emotion'],color='blue')


for bar,product in zip(polarity_graph,product_polarity.index):
  plt.text(0.005,bar.get_y()+bar.get_width(),'{}'.format(product),va='center',fontsize=11,color='white')

for bar,polarity in zip(polarity_graph,product_polarity['emotion']):
  plt.text(bar.get_width()+0.001,bar.get_y()+bar.get_width(),'%.3f'%polarity,va='center',fontsize=11,color='black')
  
plt.yticks([])
plt.show()


# In[20]:


#show wordCloud for reviews which are greater than 4 
from wordcloud import WordCloud
reviews_great = str(df['reviewstext'][df['reviewsrating']>=4])
greatcloud = WordCloud(width=1200,height=800).generate(reviews_great)
plt.imshow(greatcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[21]:


#show wordCloud for reviews which are less than 4

from wordcloud import WordCloud
reviews_great = str(df['reviewstext'][df['reviewsrating']<4])
greatcloud = WordCloud(width=1200,height=800).generate(reviews_great)
plt.imshow(greatcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[25]:


test = ['TERRIBLE DONT BUY i Bought this around black friday for $60 hoping it would be awesome... it failed so hard i tried multiple different micro SD cards none of which were recognized and YES i formated them with every format i could think of ... Fat32, NTFS, Fat, Xfat... i even tried to have the tablet do it... didnt work... to make matters worse half the apps i wanted to use werent in the app store and i came to find out that it isnt linked to the normal google play store this tablet has its own app store which is missing many common apps... the main reason i bought this was to play clash of clans and i cant because it wasnt on the app store... i tried to also use aftermarket play stores to play COC but it didnt work... launched and played 1 time but didnt work or update after that... needless to say i returned it and bought a $250 samsung galaxy tab A 10.1 (2016 version) with S-pen and its WAYYYYY better... bottom line you get what you pay for... also hint the s-pen version has an extra 1 GB of ram over the non pen version... so you should get that if you can afford the extra $50...']
test_vec = tfidf_Vectorizer.transform(test)
clf_BNB.predict(test_vec)


# In[29]:





# In[ ]:




