import pandas as pd
import numpy as np
import gensim


df_eebo1 = pd.read_csv('eebogravity.csv', encoding = 'utf-8')
df_eebo2 = pd.read_csv('eebogravity2.csv', encoding = 'utf-8')

df = pd.concat([df_eebo1, df_eebo2])

df = df.drop(df.columns[0], axis = 1)


##### Add mystery text
f = open('New Text Document.txt', 'r', encoding = 'utf-8').read()
df_ph = pd.DataFrame({'title' : 'De Gravitatione', 
                      'date' : 1860, 
                      'full_text' : f},
                    index = [0])

df = df.append(df_ph)



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
                                      num_topics = 8,
                                      random_state = 100,
                                      passes = 40,
                                      alpha = 'auto',
                                      per_word_topics = True)

# Viewing the topics
for i in range(0, 8):
    print('Topic ' + str(i))
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
                                                passes = 40,
                                                per_word_topics = True)
        model_list.append(model)
        coherence = CoherenceModel(model = model,
                                   texts = texts,
                                   dictionary = dictionary,
                                   coherence = 'c_v')
        coherence_values.append(coherence.get_coherence())
        print(num_topics)
    return model_list, coherence_values    
    
model_list, coherence_values = compute_coherence_value(dictionary = id2word,
                                                       corpus = corpus,
                                                       texts = data_lemmatized,
                                                       limit = 25)



import matplotlib.pyplot as plt
%matplotlib inline

limit=25; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# Dominant topic for each document                   
df_topics = pd.DataFrame()

for i,row in enumerate(lda[corpus]):
    row = sorted(row[0], key = lambda x: (x[1]), reverse = True)
    for j, (topic_num, prop_topic) in enumerate(row):
        if j == 0:
            keywords = ', '.join([word for word, prop in lda.show_topic(topic_num)])
            df_topics = df_topics.append(pd.Series([int(topic_num), round(prop_topic, 4), keywords]), ignore_index = True)

df_topics.columns = ['Dominant Topic', '% Contribution', 'Keywords']
df_topics['Dominant Topic'].value_counts()


# Topic frequency distribution
# Frequency
topic_counts = df_topics['Dominant Topic'].value_counts()
# Percentage distribution
topic_perc = round(topic_counts/topic_counts.sum(), 4)
# Keywords
keywords_list = []
for i in range(0,7):
    keywords = ', '.join([word for word, prop in lda.show_topic(i)])
    keywords_list.append(keywords)
keywords_list = pd.Series(keywords_list)
# Dataframe
df_dominant_topics = pd.concat([topic_counts, topic_perc, keywords_list], axis = 1)
df_dominant_topics.columns = ['Frequency Count', '% of Corpus', 'Keywords']
df_dominant_topics


# Similar texts to mystery text
txt_index = df_topics.index[df_topics['Dominant Topic'] == 3].tolist()
df_mys = df.iloc[txt_index]


# Export to csv
df_mys.to_csv('Similar Texts (8 topics model).csv', encoding = 'utf-8')
