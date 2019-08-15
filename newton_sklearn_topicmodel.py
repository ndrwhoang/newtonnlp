# Libraries
import base64
import pandas as pd
import numpy as np
import re


# Plotting libraries
import plotly.offline as py
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.tools as tls
from matplotlib import pyplot as plt
%matplotlib inline




# Import data
df = pd.read_csv('fulldataset.csv', encoding = "utf-8")
df = df.drop(df.columns[0], axis=1)
list(df)
df.fillna('NA', inplace = True)


# Subsetting English corpus
df_eng = df.loc[df.language1 == 'English']
# Subsetting Latin corpus
df_latin = df.loc[df.language1 == 'Latin']


from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import bigrams, word_tokenize, FreqDist
# Calling for lemmatizer as stemming can be unreliable
lemm = WordNetLemmatizer()
# Combining sklearn vectorizer with lemmatization
# Subclass the original TfidfCountVectorizer, change the build_analyzer method to have lemmatization
class LemmaCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(ianalyzer) for ianalyzer in analyzer(doc))
    
# Create a list of all the texts to be vectorized
text = list(df_eng.full_text)
# Calling the vectorizer function above
tf_vectorizer = LemmaCountVectorizer(max_df = 0.90,
                                     min_df = 2,
                                     stop_words = 'english',
                                     decode_error = 'ignore')
tf = tf_vectorizer.fit_transform(text)


# Term frequency graphs
# List of features / words
feature_names = tf_vectorizer.get_feature_names()
# List of count of each word
count_vec = np.asarray(tf.sum(axis = 0)).ravel()
# Combining 2 lists together
zipped = list(zip(feature_names, count_vec))
x, y = (list(i) for i in zip(*sorted(zipped, key = lambda i: i[1], reverse=True)))
# Plotting
data = [go.Bar(x = x[0:50], y = y[0:50],
               marker = dict(colorscale = 'Jet', color = y[0:50]),
               text = 'Word Count')]
layout = go.Layout(title = 'Word Frequency - Top 50')
fig = go.Figure(data = data, layout = layout)    
py.plot(fig, filename = 'basic-bar.html')


# Top and bottom 15 words in the corpus
bottom = np.concatenate([y[0:15], y[-16:-1]])
top = np.concatenate([x[0:15], x[-16:-1]])



from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
# Creating a topic model
# Applying sklearn implementation of LDA, genism's implementation is worth considering
# 10 topics, maximum 5 iterations, switch batch to online if data size too large, set seed for reproducability
lda = LatentDirichletAllocation(n_components = 3,
                                max_iter = 10,
                                learning_method = 'batch',
                                learning_offset = 50,
                                random_state = 0,
                                n_jobs = -1)
# Fitting LDA on vectorized text to generate topic models
lda.fit(tf)


# Diagnose model performance
# Log likelihood (higher the better)
lda.score(tf)


# =============================================================================
# from sklearn.model_selection import GridSearchCV
# # LDA model with tuning
# # Tuning parameters
# search_params = {'n_components': [3, 5, 7, 10],
#                  'learning_decay': [.5, .7, .9]}
# # Model initialization
# lda = LatentDirichletAllocation(random_state = 0)
# # Grid search with cross-validation (3 folds)
# model = GridSearchCV(lda, param_grid = search_params)
# model.fit(tf)
# 
# # Best model
# best_lda_model = model.best_estimator_
# # Parameters
# model.best_params_
# # Log likelihood score
# model.best_score_
# 
# =============================================================================

# List of topics and top words in each topic
# Function to print out top words
def top_words(model, feature, n_words):
    for index, topic in enumerate(model.components_):                                   # Loop through the topics
        message = '\nTopic #{}:'.format(index)                                          # For each topic, number it
        message += ' '.join([feature[i] for i in topic.argsort()[:-n_words - 1 :-1]])   # and take the top used n_words
        print(message)
        print('='*20)
tf_feature_names = tf_vectorizer.get_feature_names()
# Printing out 40 words
n_words = 40
top_words(lda, tf_feature_names, n_words)                                               


# Graphs for each topic model
# Access each topic
first_topic = best_lda_model.components_[0]
second_topic = best_lda_model.components_[1]
third_topic = best_lda_model.components_[2]
fourth_topic = best_lda_model.components_[3]
# Access the words of each topic
first_topic_words = [tf_feature_names[i] for i in first_topic.argsort()[:-50 - 1 :-1]]
second_topic_words = [tf_feature_names[i] for i in second_topic.argsort()[:-50 - 1 :-1]]
third_topic_words = [tf_feature_names[i] for i in third_topic.argsort()[:-50 - 1 :-1]]
fourth_topic_words = [tf_feature_names[i] for i in fourth_topic.argsort()[:-50 - 1 :-1]]



from wordcloud import WordCloud, STOPWORDS
# Words Clouds
# Create variable for which graph to plot for parameterization
w = second_topic_words
# Plotting
cloud = WordCloud(stopwords = STOPWORDS,
                  background_color = 'white',
                  width = 2500,
                  height = 1800).generate(' '.join(w))
plt.imshow(cloud)


# Dominant Topic and Distribution
# Document - Topic Matrix
lda_output = lda.transform(tf)
# Column names
topicnames = ['Topic ' + str(icomponents) for icomponents in range(lda.n_components)]
# Document index
docnames = ['Doc ' + str(itext) for itext in range(len(df))]
# Create a df of the 2
df_doc_topic = pd.DataFrame(np.round(lda_output, 2), columns = topicnames, index = docnames)
# Create dominant topic column for each document
df_doc_topic['dominant_topic'] = np.argmax(df_doc_topic.values, axis = 1)
df_doc_topic
# Distribution of documents by topics
df_topic_distribution = df_doc_topic['dominant_topic'].value_counts().reset_index(name = '# Documents')
df_topic_distribution.columns = ['Topic', '# Documents']
df_topic_distribution



import pyLDAvis
import pyLDAvis.sklearn
# pyLDAvis graphs
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer, mds = 'tsne')
pyLDAvis.show(panel)



from sklearn.cluster import KMeans
# Topic Clustering
# Initialize clusters
clusters = KMeans(n_clusters = 4, random_state = 0).fit_predict(lda_output)
# Singular value decomposition model to create graph
svd_model = TruncatedSVD(n_components = 2)
lda_output_svd = svd_model.fit_transform(lda_output)
# Create axis out of components from the decomposition
x = lda_output_svd[:, 0]
y = lda_output_svd[:, 1]
# Percentage of information explained by 2 components
print('% of Variance Explained: \n', np.round(svd_model.explained_variance_ratio_,2))
# Plotting
plt.figure(figsize = (15,15))
plt.scatter(x, y, c = clusters)
plt.xlabel('Component 2')
plt.ylabel('Component 1')
plt.title('Topic Clusters',)


# =============================================================================
# import networkx as nx
# from collections import Counter
# # Word co occurence network / directed word graph
# full_str = ' '.join(df_eng.full_text)
# tokens = word_tokenize(full_str)
# bgs_count = Counter(list(bigrams(tokens)))
# bgs_count.most_common(20)
# =============================================================================



