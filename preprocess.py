import numpy as np
import pandas as pd
from string import punctuation
import re
import nltk
import random
# nltk.download('stopwords')
import string   
from tensorflow.keras.preprocessing.text import Tokenizer                        
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
# from nltk.tokenize import TweetTokenizer
# from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, LSTM, Dense,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.feature_extraction.text import TfidfVectorizer
# import matplotlib.pyplot as plt
# import seaborn as sns
# from nltk.tokenize import word_tokenize
# nltk.download("punkt")
from tokenizer import tokenize

df = pd.read_csv('train-sample.csv')

def tok(str):

    maxlen = 512
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(str)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences([str])
    text = pad_sequences(sequences, maxlen=maxlen)
    return text

def stemming(word):
    stemmer = PorterStemmer() 
    list1=[]
    for i in word.split():
        list1.append(stemmer.stem(i))
    return ' '.join(list1)


def create_text_column(df, title_col, body_col, prefix=" "):  # Optional tags_cols and prefix arguments
  """
  Creates a new column named 'text' in the DataFrame by combining title, body, and optional tags.

  Args:
      df (pandas.DataFrame): The DataFrame to process.
      title_col (str): The name of the column containing the title text.
      body_col (str): The name of the column containing the body markdown text.
      tags_cols (list, optional): A list of column names containing tag text (defaults to None).
      prefix (str, optional): A prefix to add before the body text (defaults to a space).

  Returns:
      pandas.DataFrame: The modified DataFrame with the new 'text' column.
  """

  df['text'] = df.apply(lambda row: create_text_row(row[title_col], row[body_col], row, prefix), axis=1)
  return df

def create_text_row(title, body, row=None, prefix=" "):  # Helper function for row-wise processing
  """
  Constructs the text string for a single row.

  Args:
      title (str): The title text.
      body (str): The body markdown text.
      tags_cols (list, optional): A list of column names containing tag text (defaults to None).
      row (pandas.Series, optional): The entire row if tags_cols are not provided (defaults to None).
      prefix (str, optional): A prefix to add before the body text (defaults to a space).

  Returns:
      str: The combined text string.
  """

  text = f"Title: '{title}'"
  text += f"\n{prefix}Body: '{body}'"
  return text


# Assuming your DataFrame is named df
df = create_text_column(df, "Title", "BodyMarkdown", prefix="  ")  # Example with tags_cols and prefix
# Alternatively, if tags are in separate columns starting with "Tag"
df = create_text_column(df, "Title", "BodyMarkdown")


def pro(df):
   stopwords_english = stopwords.words('english')
   df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_english)]))
   df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
   stemmer = PorterStemmer() 
   def stemming(word):
    list1=[]
    for i in word.split():
        list1.append(stemmer.stem(i))
        return ' '.join(list1)
    df['text'] = df['text'].apply(lambda x:stemming(x)) 


def encode_labels(df, source_column, custom_labelling, default_value=None):
  """
  Encodes categorical labels in a DataFrame column using a custom labeling scheme.

  Args:
      df (pandas.DataFrame): The DataFrame containing the labels to encode.
      source_column (str): The name of the column containing the categorical labels.
      custom_labelling (dict): A dictionary mapping string labels to numerical codes.
      default_value (int, optional): The default value to assign for missing labels (defaults to None).

  Returns:
      list: A list of encoded labels corresponding to the rows in the DataFrame.
  """

  encoded_labels = []
  for label in df[source_column]:
    encoded_label = custom_labelling.get(label, default_value)
    if encoded_label is None:
      print(f"Warning: Label '{label}' not found in custom_labelling. Using default value: {default_value}")
    encoded_labels.append(encoded_label)
  return encoded_labels

# Example usage
custom_labelling = {
    'open': 0,
    'not a real question': 1,
    'not constructive': 2,
    'too localized': 3,
    'off topic': 4
}

encoded_labels = encode_labels(df, "OpenStatus", custom_labelling)
