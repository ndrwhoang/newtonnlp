import pandas as pd

# Import data
df = pd.read_csv('D:/work/pythonproject/newtonnlp/csv/fulldataset.csv', encoding = "utf-8")
df = df.drop(df.columns[0], axis=1)
list(df)



# Cleaning metadata
# Creating length column
df['length'] = df['metadata'].str.split(', ').str[-1]

# Word count column
df['word_count'] = df['metadata'].str.extract("([0-9]*\,*[0-9]*\swords[^\,]*)").fillna('na')

# Time column
df['time'] = df['metadata'].str.split(', ').str[0]

# Language column
df['language'] = df.metadata.str.split(', ').str[1]

# Create a column that categorizes each text into 1 language category
# 4 categories: English, Latin, Mixed (English and Latin), and Other
df.loc[df['language'].str.contains('English'), 'language1'] = 'English'
df.loc[df['language'].str.contains('Latin'), 'language1'] = 'Latin'
df.loc[(df['language'].str.contains('English')) & (df['language'].str.contains('Latin')), 'language1'] = 'Mixed'
df['language1'].fillna('Other', inplace = True)



# Export to csv
df.to_csv('fulldatasetcleaned.csv', encoding = 'utf-8')
