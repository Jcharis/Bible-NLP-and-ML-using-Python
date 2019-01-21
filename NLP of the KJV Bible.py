#!/usr/bin/env python
# coding: utf-8

# ### NLP of the KJV Bible
# + Sentiment Analysis
# + EDA
# + Summarization
# + Prediction of Verse
# 
# #### Data Sources
# + https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_kjv.csv

# In[1]:


# Load EDA Pkgs
import pandas as pd


# In[2]:


# Load Dataset
df = pd.read_csv("t_kjv.csv")


# In[3]:


# Columns
df.columns


# In[4]:


# Head
df.head()


# In[5]:


# Rename Books
df1 = df


# In[6]:


# Replacing with the correct name
df1.b.replace({1:"Genesis",
2:"Exodus",
3:"Leviticus",
4:"Numbers",
5:"Deuteronomy",
6:"Joshua",
7:"Judges",
8:"Ruth",
9:"1 Samuel (1 Kings)",
10:"2 Samuel (2 Kings)",
11:"1 Kings (3 Kings)",
12:"2 Kings (4 Kings)",
13:"1 Chronicles",
14:"2 Chronicles",
15:"Ezra",
16:"Nehemiah",
17:"Esther",
18:"Job",
19:"Psalms",
20:"Proverbs",
21:"Ecclesiastes",
22:"Song of Solomon (Canticles)",
23:"Isaiah",
24:"Jeremiah",
25:"Lamentations",
26:"Ezekiel",
27:"Daniel",
28:"Hosea",
29:"Joel",
30:"Amos",
31:"Obadiah",
32:"Jonah",
33:"Micah",
34:"Nahum",
35:"Habakkuk",
36:"Zephaniah",
37:"Haggai",
38:"Zechariah",
39:"Malachi",
40:"Matthew",
41:"Mark",
42:"Luke",
43:"John",
44:"Acts",
45:"Romans",
46:"1 Corinthians",
47:"2 Corinthians",
48:"Galatians",
49:"Ephesians",
50:"Philippians",
51:"Colossians",
52:"1 Thessalonians",
53:"2 Thessalonians",
54:"1 Timothy",
55:"2 Timothy",
56:"Titus",
57:"Philemon",
58:"Hebrews",
59:"James",
60:"1 Peter",
61:"2 Peter",
62:"1 John",
63:"2 John",
64:"3 John",
65:"Jude",
66:"Revelation"},inplace=True)


# In[7]:


df1.b


# In[8]:


df1.head()


# In[9]:


# Renaming Columns
df1.columns = ["id","book","chapter","verse","text"]


# In[10]:


df1.columns


# In[11]:


df1.to_csv("kjv_cleandata1.csv")


# #### EDA
# + Longest sentence
# + Shortest sentence
# 

# In[12]:


df1.text


# In[13]:


# Length of Each Sentence
df1["verse_length"] = df1.text.str.len()


# In[14]:


# Longest Sentence or Verse In the Bible
df1.text.str.len().max()


# In[15]:


# Location of text/verse
df1.text.str.len().idxmax()


# In[16]:


df1.iloc[12826]


# In[17]:


# Shortest Sentence or Verse in the Bible
df1.text.str.len().min()


# In[18]:


# Location of text
df1.text.str.len().idxmin()


# In[19]:


df1.iloc[30673]


# In[20]:


# Second Longest Verse
df1.text.str.len().nlargest(2)


# In[21]:


# Second shortest verse
df1.text.str.len().nsmallest(2)


# In[22]:


df1.iloc[26558]


# #### NLP
# + Tokens
# + Sentiment Analysis
# + NER

# In[23]:


# For Most of the NLP 
import spacy 
nlp = spacy.load('en')


# In[24]:


# For Sentiment Analysis
from textblob import TextBlob


# In[25]:


def get_sentiment(text):
    docx = TextBlob(text)
    sent = docx.sentiment
    return sent


# In[26]:


# Get the First Verse
df1.text.loc[0]


# In[27]:


get_sentiment(df1.text.loc[0])


# In[28]:


df1.text.apply(get_sentiment)


# In[29]:


# Let Make it individual
def get_polarity(text):
    docx = TextBlob(text)
    sent = docx.sentiment.polarity
    return sent

def get_subjectivity(text):
    docx = TextBlob(text)
    sent = docx.sentiment.subjectivity
    return sent


# In[30]:


df1['verse_polarity'] = df1.text.apply(get_polarity)
df1['verse_subjectivity'] = df1.text.apply(get_subjectivity)
df1['sentiment'] = df1.text.apply(get_sentiment)


# In[31]:


df1.head()


# In[32]:


# The Verse with the Highest Positive Sentiment
df1.verse_polarity.max()


# In[33]:


df1.verse_polarity.idxmax()


# In[34]:


df1.iloc[146]


# In[35]:


df1.verse_polarity.nlargest(100)


# #### Conclusion
# + There are over 100 positive sentiment with a polarity of 1.0

# In[37]:


df1.verse_polarity.nsmallest(100)


# In[38]:


df1.iloc[476].text


# In[39]:


#### NER of Each Sentence
def get_ner(text):
    docx = nlp(text)
    result = [(token.text,token.label_) for token in docx.ents]
    return result


# In[40]:


df1.text.iloc[0]


# In[41]:


get_ner(df1.text.iloc[0])


# In[42]:


df1['named_entities'] = df1.text.apply(get_ner)


# In[43]:


df1.head()


# In[44]:


df1.to_csv("kjv_final1.csv")


# In[ ]:


#### Predicting New and Old
+ NT 1
+ OT 0
+ Using Naive Bayes


# In[53]:


# Last Verse of Old Testament
df1.loc[23144]['text']


# In[54]:


# Last Verse of Old Testament
df1.loc[23144]['text']


# In[55]:


df2 = df1


# In[58]:


# Last verse of NT 7,957 
df2.loc[0:23144,'label'] = 0 


# In[59]:


df2.head()


# In[60]:


df2.loc[23145:,'label'] = 1


# In[61]:


df2.tail()


# In[65]:


# Model ML
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.cross_validation import train_test_split b17
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[70]:


Xfeatures = df2['text']
y = df2['label']


# In[71]:


# Feature Extraction 
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[73]:


# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[74]:


# Accuracy of our Model
print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")


# In[75]:


# Accuracy of our Model
print("Accuracy of Model",clf.score(X_train,y_train)*100,"%")


# In[ ]:


#### Predicting A Text
+ Whether therefore ye eat, or drink, or whatsoever ye do, do all to the glory of God.(1 Corinthians 10:31 )


# In[76]:


# Sample1 Prediction
sample_verse = ["Whether therefore ye eat, or drink, or whatsoever ye do, do all to the glory of God"]
vect = cv.transform(sample_verse).toarray()


# In[77]:


# Old Testament is 0, New Testament is 1
clf.predict(vect)


# In[ ]:


### Example
+ Isaiah 41:10
sample_verse2 = ["Fear thou not; for I am with thee: be not dismayed; for I am thy God: I will strengthen thee; yea, I will help thee; yea, I will uphold thee with the right hand of my righteousness."]


# In[79]:



sample_verse2 = ["Fear thou not; for I am with thee: be not dismayed; for I am thy God: I will strengthen thee; yea, I will help thee; yea, I will uphold thee with the right hand of my righteousness."]


# In[80]:


vect2 = cv.transform(sample_verse2).toarray()


# In[81]:


clf.predict(vect2)


# #### Saving the Model

# In[82]:


from sklearn.externals import joblib


# In[83]:



biblepredictionNV_model = open("biblepredictionNV_model.pkl","wb")

joblib.dump(clf,biblepredictionNV_model)


# In[84]:


biblepredictionNV_model.close()


# #### Conclusion
# + We Have been able to see the longest and shortest verse
# + Being able to build a model for predicting which part of the bible a particular verse belongs to

# In[85]:


# Thanks
# By Jesse JCharis
# Jesus Saves @ JCharisTech
# J-Secur1ty


# In[ ]:




