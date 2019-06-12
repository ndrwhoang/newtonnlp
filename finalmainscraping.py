import pandas as pd
import numpy as np
from bs4 import BeautifulSoup 
import requests
import csv
import re


# Web page link
url = 'http://www.newtonproject.ox.ac.uk/texts/newtons-works/religious'
# Get Content
page_response = requests.get(url, timeout = 5)
mainsoup = BeautifulSoup(page_response.text, 'html.parser')
norm_text = mainsoup.find_all('tr')


# Which variables to get
# Key (number index)
key = []
# Main title
main_title = []
# Metadata
metadata = []
# Hyperlinks list
hyper_link = []



    
for i in norm_text:
    # Scraping index key
    key.append(i.find('td', {'class' : 'key'}).text)
    # Scraping main title
    main_title.append(i.find('p', {'class' : 'title'}).text)
    # Scraping metadata
    metadata.append(i.find('p', {'class' : 'metadata'}).text)
    # Scraping list of articles linked
    # This is handled differently from other elements because sometimes an article is divided into multiple small chapters
    # All articles linked within
    link = []
    # Looking for all normalized texts
    hyper = i.find_all('a', href = re.compile('normalized'))
    for j in hyper:
        link.append('http://www.newtonproject.ox.ac.uk' + j.get('href'))
    # Adding link(s) to main list
    hyper_link.append(link)
    

# Create a dataframe
df = pd.DataFrame({'key' : key,
                   'main_title' : main_title,
                   'metadata' : metadata,
                   'link_list' : hyper_link})
    
    
# Cleaning hyper_link column
# Splitting hyper_link column
# Source: https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows/40449726#40449726
lst_col = 'link_list'
x = df.assign(**{lst_col:df[lst_col].str.split(',')})
df = pd.DataFrame({
        col:np.repeat(x[col].values, x[lst_col].str.len())
        for col in x.columns.difference([lst_col])
}).assign(**{lst_col:np.concatenate(x[lst_col].values)})[x.columns.tolist()]
# Removing redundant redundant characters
df['link_list'] = df.link_list.apply(lambda x: x.replace('[', '').replace(']', '').replace(' ', '').replace("'", ""))


# Cleaning metadata column
# Removing 'Metadata:'
df.metadata = [x[10:] for x in df.metadata]
# Removing redundant characters
df['metadata'] = df.metadata.apply(lambda x: x.replace('\n', ''))
df.metadata = df.metadata.replace('\s+', ' ', regex = True)
# Splitting into multiple columns
#df[['time', 'language', 'no1', 'length']] = df['metadata'].str.split(',', expand = True)
pd.concat([df, df.metadata.str.get_dummies(sep = ', ')],1)

# Scraping full text for each article
# Title of each small chapters
title = []
# Body of text
full_text = []

# Scraping text for each article
for i in df.link_list:
    # Get web page content
    response = requests.get(i, timeout = 5)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get titles
    title.append(soup.find('h1').text)
    
    # Get texts
    paragraphs = soup.find('div', {'id' : 'tei'}).find_all('p')
    a = " ".join([ paragraph.text for paragraph in paragraphs])
    full_text.append(a)
    
    
# Adding new columns to df
df = df.assign(title = title)
df = df.assign(full_text = full_text)



# Export to csv
df.to_csv('fulldataset.csv', encoding = 'utf-8')



