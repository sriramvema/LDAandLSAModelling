import pandas as pd
import numpy as np
import re
import nltk
import gensim
import csv
nltk.download('stopwords')
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import NMF
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def clean_text(text):
    new_text = text[1]
    if isinstance(new_text, str):
        new_text = re.sub(r'@[A-Za-z0-9_]+', '', new_text)
        new_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', new_text)
        new_text = re.sub(r'[^\w\s]', '', new_text).lower()
        new_text = re.sub(r'\d+', '', new_text)
        new_text = re.sub(r'\s+', ' ', new_text).strip()
        tokens = [word for word in text if word not in STOPWORDS and len(word) > 3]
        text2 = ' '.join(tokens)
    return text2

def clean_data(data):
    for i in range(len(data)):
        data[i] = clean_text(data[i])
    return data

with open('C:/Users/srira/OneDrive/Desktop/Python Assignment/HW2/India_BTMSET.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    data1 = [line for line in reader]
data1 = clean_data(data1)


with open('C:/Users/srira/OneDrive/Desktop/Python Assignment/HW2/UK_BTMSET (1).csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    data2 = [line for line in reader]
data2 = clean_data(data2)


with open('C:/Users/srira/OneDrive/Desktop/Python Assignment/HW2/US_BTM.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    data3 = [line for line in reader]
data3 = clean_data(data3)

def apply_lsa(data, num_topics=10):
    # create a vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(data)

    # create an LSA model
    svd_model = TruncatedSVD(n_components=num_topics)

    # apply the LSA model to the TF-IDF matrix
    svd_matrix = svd_model.fit_transform(tfidf)

    # get the top words for each topic
    top_words = []
    terms = vectorizer.get_feature_names_out()
    for i, component in enumerate(svd_model.components_):
        word_idx = np.argsort(component)[::-1][:7]
        topic_words = [terms[idx] for idx in word_idx]
        top_words.append(topic_words)

    # print the top topics and top words
    for i, topic in enumerate(top_words):
        print(f"Topic {i}: {', '.join(topic)}")

print("TOPICS FOR DATA1:")
apply_lsa(data1)
print()

print("TOPICS FOR DATA2:")
apply_lsa(data2)
print()

print("TOPICS FOR DATA3:")
apply_lsa(data3)
print()
