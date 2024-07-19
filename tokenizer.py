import pandas as pd
import pandas as pd
# import numpy as np
# from string import punctuation
# import re
import nltk
# from nltk.corpus import twitter_samples
# import random
nltk.download('stopwords')
# import string   
from tensorflow.keras.preprocessing.text import Tokenizer                        
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
# from nltk.tokenize import TweetTokenizer
# from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, LSTM, Dense,Dropout
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.feature_extraction.text import TfidfVectorizer
# import matplotlib.pyplot as plt
# import seaborn as sns


def tokenize():
    df = pd.read_csv('train-sample.csv')
    df['BodyMarkdown'] = df['BodyMarkdown'].astype('str')
    df = df.dropna()
    stopwords_english = stopwords.words('english')
    df['BodyMarkdown'] = df['BodyMarkdown'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_english)]))

    stemmer = PorterStemmer() 

    def stemming(word):
        list1=[]
        for i in word.split():
            list1.append(stemmer.stem(i))
        return ' '.join(list1)
    
    df['BodyMarkdown'] = df['BodyMarkdown'].apply(lambda x:stemming(x))   
    tokenizer = Tokenizer()
    tokenizer.fit_on_text(df['BodyMarkdown']) 
    word_index = tokenizer.word_index
    return tokenizer 