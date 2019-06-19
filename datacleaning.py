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

# Export to csv
df.to_csv('fulldatasetcleaned.csv', encoding = 'utf-8')