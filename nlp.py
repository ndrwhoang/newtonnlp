# Libraries
import numpy as np
import pandas as pd
import nltk

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation



# Import data
df = pd.read_csv('D:/work/pythonproject/newtonnlp/mainpage.csv', encoding = "utf-8")

# Overview
df.head()
df.shape

# Tokenization
first_text = df.Text.values[0]
print(first_text)
first_text_list = nltk.word_tokenize(first_text)
print(first_text_list)


# Stopword Removal
# Importing stop word list
stopwords = nltk.corpus.stopwords.words('english') #  Adjustments might be needed
print(stopwords)
# Filtering out stop words
first_text_list_cleaned = [i for i in first_text_list if i.lower() not in stopwords]
print(first_text_list_cleaned)
print(len(first_text_list), len(first_text_list_cleaned))


# Lemmatization
lemm = WordNetLemmatizer()


# Vectorizing
vectorizer = CountVectorizer(min_df = 0)


##### Sklearn vectorizing and lemmanizing
lemm = WordNetLemmatizer()
class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))


# Creating list of texts
text = list(df.Text.values)


# Applying above function to list
tf_vectorizer = LemmaCountVectorizer(max_df = 0.95,
                                     min_df = 2,
                                     stop_words = 'english',
                                     decode_error = 'ignore')
tf = tf_vectorizer.fit_transform(text)
tf2 = pd.DataFrame(tf.todense())
tf2
