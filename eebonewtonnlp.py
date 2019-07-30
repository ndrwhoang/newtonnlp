import pandas as pd
import numpy as np
import gensim




df_eebo = pd.read_csv('D:/work/pythonproject/newtonnlp/csv/eebogravity.csv', encoding = 'utf-8')
df_newton = pd.read_csv('D:/work/pythonproject/newtonnlp/csv/fulldataset.csv', encoding = 'utf-8')




df_newton = df_newton.drop(df_newton.columns[0], axis=1)
df_eebo = df_eebo.drop(df_eebo.columns[0], axis = 1)


df_newton_eng = df_newton.loc[df_newton.language1 == 'English']


list(df_newton_eng)
list(df_eebo)


df_newton_eng = df_newton_eng[['title', 'time', 'full_text']].copy()
df_newton_eng.columns = ['title', 'date', 'full_text']


df = df_eebo.append(df_newton_eng, ignore_index = True).copy()
df = df_eebo.copy()
df = df_newton_eng.copy()
#
df = df.loc[df.language1 == 'English']
df['type'].value_counts()
df = df[['title', 'full_text', 'type']].copy()
list(df)

#### Pre processing with gensim
text = list(df.full_text)

# Tokenizing
from gensim.utils import simple_preprocess

def tokenizer(corpus):
    for i in corpus:
        yield(gensim.utils.simple_preprocess(str(i), deacc = True))
data_words = list(tokenizer(text))

# Removing stopwords
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
data_words = remove_stopwords(data_words)

# Lemmatization
import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


#### Creating topic models
# Create Dctionary
import gensim.corpora as corpora

id2word = corpora.Dictionary(data_lemmatized)
id2word.filter_extremes(no_below = 5, no_above = 0.9)

# Create Corpus
corpus = [id2word.doc2bow(i) for i in data_lemmatized]

# LDA model
lda = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                      id2word = id2word,
                                      num_topics = 16,
                                      random_state = 100,
                                      passes = 50,
                                      alpha = 'auto',
                                      per_word_topics = True)

# Viewing the topics
for i in range(0,16):
    print(lda.print_topics()[i])
# Perplexity
lda.log_perplexity(corpus)

#Coherence
from gensim.models import CoherenceModel

lda_coherence_model = CoherenceModel(model = lda,
                                     texts = data_lemmatized,
                                     dictionary = id2word,
                                     coherence = 'c_v')
lda_coherence = lda_coherence_model.get_coherence()
lda_coherence

import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
pyLDAvis.show(vis)



# =============================================================================
# # LDA with mallet
# import os
# os.environ.update({'MALLET_HOME': r'D:/work/pythonproject/newtonnlp/mallet-2.0.8/'})
# mallet_path = 'D:\\work\\pythonproject\\newtonnlp\\mallet-2.0.8\\bin\\mallet'
# 
# mallet_path = 'D:/work/pythonproject/newtonnlp/mallet-2.0.8/bin/mallet'
# ldamallet = gensim.models.wrappers.LdaMallet(mallet_path,
#                                              corpus = corpus,
#                                              num_topics = 7,
#                                              id2word = id2word)
# 
# print(ldamallet.show_topics(formatted = False))
# 
# coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
# print('\nCoherence Score: ', coherence_ldamallet)
# 
# ldamallet_converted = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)
# mallet_vis = pyLDAvis.gensim.prepare(ldamallet_converted, corpus, id2word)
# pyLDAvis.show(mallet_vis)
# =============================================================================


# Search for opimal number of topics
# Function performing a series of LDA models and computing coherence value for each one 
def compute_coherence_value(dictionary, corpus, texts, limit, start = 2, step = 2):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                                id2word = id2word,
                                                num_topics = num_topics,
                                                random_state = 100,
                                                alpha = 'auto',
                                                passes = 50,
                                                per_word_topics = True)
        model_list.append(model)
        coherence = CoherenceModel(model = model,
                                   texts = texts,
                                   dictionary = dictionary,
                                   coherence = 'c_v')
        coherence_values.append(coherence.get_coherence())
    return model_list, coherence_values    
    
model_list, coherence_values = compute_coherence_value(dictionary = id2word,
                                                       corpus = corpus,
                                                       texts = data_lemmatized,
                                                       limit = 30)



import matplotlib.pyplot as plt
%matplotlib inline

limit=30; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# =============================================================================
# def format_topics_sentences(ldamodel = lda,corpus = corpus, texts = text):
#     topics_df = pd.DataFrame()
#     
#     for i, row in enumerate(ldamodel[corpus]):
#         row = sorted(row[0], key = lambda x: (x[1]), reverse = True)
#         for j, (topic_num, prop_topic) in enumerate(row):
#             if j == 0:
#                 wp = ldamodel.show_topic(topic_num)
#                 topic_keywords = ', '.join([word for word, prop in wp])
#                 topics_df = topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),ignore_index = True)
#             else:
#                 break
#         topics_df.columns = ['Dominant Topic', '% Contribution', 'Keywords']
#         
#         contents = pd.Series(texts)
#         topics_df = pd.concat([topics_df, contents], axis = 1)
#         return(topics_df)
# 
# df_topic_keywords = format_topics_sentences(ldamodel = lda,
#                                             corpus = corpus,
#                                             texts = text)
# 
# df_dominant_topic = df_topic_keywords.reset_index()
# df_dominant_topic.columns = ['Doc #', 'Dominant Topic', '% Contribution', 'Keywords', 'Text']
# 
# df_dominant_topic[['Doc #', 'Dominant Topic', '% Contribution', 'Keywords']].head(10)
# =============================================================================
                   
# Dominant topic for each document                   
df_topics = pd.DataFrame()

for i,row in enumerate(lda[corpus]):
    row = sorted(row[0], key = lambda x: (x[1]), reverse = True)
    for j, (topic_num, prop_topic) in enumerate(row):
        if j == 0:
            keywords = ', '.join([word for word, prop in lda.show_topic(topic_num)])
            df_topics = df_topics.append(pd.Series([int(topic_num), round(prop_topic, 4), keywords]), ignore_index = True)

df_topics.columns = ['Dominant Topic', '% Contribution', 'Keywords']



# Topic frequency distribution
# Frequency
topic_counts = df_topics['Dominant Topic'].value_counts()
# Percentage distribution
topic_perc = round(topic_counts/topic_counts.sum(), 4)
# Keywords
keywords_list = []
for i in range(0,16):
    keywords = ', '.join([word for word, prop in lda.show_topic(i)])
    keywords_list.append(keywords)
keywords_list = pd.Series(keywords_list)
# Dataframe
df_dominant_topics = pd.concat([topic_counts, topic_perc, keywords_list], axis = 1)
df_dominant_topics.columns = ['Frequency Count', '% of Corpus', 'Keywords']
df_dominant_topics









df.fillna('NA', inplace = True)
df.isna().sum()
df = df.drop(df.columns[0], axis=1)
list(df)


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
# New preprocessing
lemm = WordNetLemmatizer()
# Combining sklearn vectorizer with lemmatization
# Subclass the original CountVectorizer, change the build_analyzer method to have lemmatization
class LemmaTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaTfidfVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(ianalyzer) for ianalyzer in analyzer(doc))
    
# Create a list of all the texts to be vectorized
text = list(df.full_text)
# Calling the vectorizer function above
tf_vectorizer = LemmaTfidfVectorizer(max_df = 0.8,
                                     min_df = 5,
                                     stop_words = stopwords.words('english'),
                                     decode_error = 'ignore')
tf = tf_vectorizer.fit_transform(text)



lda = LatentDirichletAllocation(n_components = 3,
                                max_iter = 10,
                                learning_method = 'batch',
                                learning_offset = 50,
                                random_state = 0)
# Fitting LDA on vectorized text to generate topic models
lda.fit(tf)


# Diagnose model performance
# Log likelihood (higher the better)
lda.score(tf)


def top_words(model, feature, n_words):
    for index, topic in enumerate(model.components_):                                   # Loop through the topics
        message = '\nTopic #{}:'.format(index)                                          # For each topic, number it
        message += ' '.join([feature[i] for i in topic.argsort()[:-n_words - 1 :-1]])   # and take the top used n_words
        print(message)
        print('='*20)
tf_feature_names = tf_vectorizer.get_feature_names()
# Printing out 40 words
n_words = 20
top_words(lda, tf_feature_names, n_words)                                               


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


# pyLDAvis graphs
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer, mds = 'tsne')
pyLDAvis.show(panel)


# Topic Clustering
# Initialize clusters
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
%matplotlib inline

clusters = KMeans(n_clusters = 3, random_state = 0).fit_predict(lda_output)
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