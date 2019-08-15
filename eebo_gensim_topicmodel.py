import pandas as pd
import numpy as np
import gensim


# Importing csv
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

# Forming Bigram and Trigram
bigram = gensim.models.Phrases(text) 
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram = gensim.models.Phrases(bigram[text])  
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Bigram
data_words = [bigram_mod[doc] for doc in data_words]

# Trigram
data_words = [trigram_mod[bigram_mod[doc]] for doc in data_words]

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

# Coherence score
from gensim.models import CoherenceModel

lda_coherence_model = CoherenceModel(model = lda,
                                     texts = data_lemmatized,
                                     dictionary = id2word,
                                     coherence = 'c_v')
lda_coherence = lda_coherence_model.get_coherence()
lda_coherence


# Bubble graph with word ranking
import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
pyLDAvis.show(vis)

# Dominant topic for each document                   
df_topics = pd.DataFrame()

for i,row in enumerate(lda[corpus]):
    row = sorted(row[0], key = lambda x: (x[1]), reverse = True)
    for j, (topic_num, prop_topic) in enumerate(row):
        if j == 0:
            keywords = ', '.join([word for word, prop in lda.show_topic(topic_num)])
            df_topics = df_topics.append(pd.Series([int(topic_num), round(prop_topic, 4), keywords]), ignore_index = True)
    
contents = pd.Series(data_words)
df_topics = pd.concat([df_topics, contents], axis = 1)
    
df_topics.columns = ['Dominant Topic', '% Contribution', 'Keywords', 'Text']
df_topics['Dominant Topic'].value_counts()


# Topic frequency distribution
# Dataframe with most representative text for each topic
df_dominant_topics = pd.DataFrame()

for i, group in df_topics.groupby('Dominant Topic'):
    df_dominant_topics = pd.concat([df_dominant_topics, group.sort_values(['% Contribution'], ascending = False).head(1)], axis = 0)

# Distribution of Word Count
import matplotlib.pyplot as plt
%matplotlib inline
doc_lens = [len(d) for d in df_topics['Text']]
# Plot
plt.figure(figsize=(8, 3.5), dpi=160)
plt.hist(doc_lens, bins = 50, color='navy')
plt.text(32000, 30, "Mean   : " + str(round(np.mean(doc_lens))))
plt.text(32000, 28, "Median : " + str(round(np.median(doc_lens))))
plt.text(32000, 26, "Stdev   : " + str(round(np.std(doc_lens))))
plt.text(32000, 24, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
plt.text(32000, 22, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

plt.gca().set(xlim=(0, 42000), ylabel='Number of Documents', xlabel='Document Word Count')
plt.title('Distribution of Document Word Counts')
plt.show()

# Per Topic Word Cloud
from wordcloud import WordCloud
import matplotlib.colors as mcolors

# Selecting colors
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

# Creating word clouds
cloud = WordCloud(stopwords = stop_words,
                  background_color = 'white',
                  max_words = 15,
                  colormap = 'tab10',
                  color_func = lambda *args, **kwargs: cols[i],
                  prefer_horizontal = 1.0)

topics = lda.show_topics(formatted = False)

fig, axes = plt.subplots(4, 2, figsize = (10, 10), sharex = True, sharey = True)
for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i + 1))
    plt.gca().axis('off')
plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


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

# Calling function     
model_list, coherence_values = compute_coherence_value(dictionary = id2word,
                                                       corpus = corpus,
                                                       texts = data_lemmatized,
                                                       limit = 25)              # number of topic models, from 2 to limit shown here



# Plotting the coherence score
limit=25; start=2; step=2;                                                      # Change limit according to limit above
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))




# Similar texts to mystery text
txt_index = df_topics.index[df_topics['Dominant Topic'] == 3].tolist()
df_mys = df.iloc[txt_index]


# Export to csv
df_mys.to_csv('Similar Texts (8 topics model).csv', encoding = 'utf-8')
