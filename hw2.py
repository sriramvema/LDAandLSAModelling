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

vectorizer = CountVectorizer(max_features=5000)
dtm1 = vectorizer.fit_transform(data1)
dtm2 = vectorizer.fit_transform(data2)
dtm3 = vectorizer.fit_transform(data3)

lsa1 = TruncatedSVD(n_components=10, random_state=0)
lsa2 = TruncatedSVD(n_components=10, random_state=0)
lsa3 = TruncatedSVD(n_components=10, random_state=0)
lsa1.fit(dtm1)
lsa2.fit(dtm2)
lsa3.fit(dtm3)

print("done")


def find_optimal_num_topics_lsa(dtm):
    '''
    This function finds the optimum number of topics for an LSA model.

    Parameters:
        dtm (scipy.sparse matrix): Document-term matrix.

    Returns:
        num_topics (int): Optimum number of topics.
    '''

    # Calculate cosine similarity matrix for DTM
    cos_sim_matrix = cosine_similarity(dtm)

    # Find the range of possible number of topics
    num_topics_range = range(1, min(dtm.shape) - 1)

    # Calculate truncated SVD for each number of topics
    svd_scores = []
    for num_topics in num_topics_range:
        svd = TruncatedSVD(n_components=num_topics, random_state=0)
        svd.fit(dtm)
        svd_scores.append(np.mean(cosine_similarity(svd.transform(dtm))))

    # Plot the elbow graph
    plt.plot(num_topics_range, svd_scores)
    plt.xlabel('Number of topics')
    plt.ylabel('Cosine similarity score')
    plt.title('Elbow graph for LSA model')
    plt.show()

    # Find the optimal number of topics
    elbow_idx = np.argmax(np.abs(np.diff(svd_scores)))
    num_topics = num_topics_range[elbow_idx]

    return num_topics

# Find the coherence score of the LDA model for each file
coherence1 = find_optimal_num_topics_lsa(dtm1)


print(coherence1)

# Print the coherence score of the LDA model for each file
print(f'Coherence score for file 1: {coherence1}')




# Print the top 10 topics for each file
print('Top 10 topics for file 1:')
for i, topic in enumerate(lsa1.components_):
    print(f'Topic {i}:')
    print(', '.join(vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]))
    print()

print('Top 10 topics for file 2:')
for i, topic in enumerate(lsa2.components_):
    print(f'Topic {i}:')
    print(', '.join(vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]))
    print()

print('Top 10 topics for file 3:')
for i, topic in enumerate(lsa3.components_):
    print(f'Topic {i}:')
    print(', '.join(vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]))
    print()

