import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import DataBaseClass

import streamlit as st
st.header('Insight Recommendation')  # header for webapp
st.image('https://builtin.com/sites/www.builtin.com/files/styles/og/public/recommendation-system-machine-learning_1.jpg')
query = st.text_input('Ask your query:')

#export
def preprocess(title, body=None):
    """ Preprocess the input, i.e. lowercase, remove html tags, special character and digits."""
    text = ''
    if body is None:
        text = title
    else:
        text = title + body
    # to lower case
    text = text.lower()

    # remove tags
    text = re.sub("</?.*?>"," <> ", text)
    
    # remove special characters and digits
    text = re.sub("(\\d|\\W)+"," ", text).strip()
    return text
    
def create_tfidf_features(corpus, max_features=5000, max_df=0.95, min_df=2):
    """ Creates a tf-idf matrix for the `corpus` using sklearn. """
    tfidf_vectorizor = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word', 
                                       stop_words='english', ngram_range=(1, 1), max_features=max_features, 
                                       norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                                       max_df=max_df, min_df=min_df)
    X = tfidf_vectorizor.fit_transform(corpus)
    print('tfidf matrix successfully created.')
    return X, tfidf_vectorizor

def calculate_similarity(X, vectorizor, query, top_k=5):
    """ Vectorizes the `query` via `vectorizor` and calculates the cosine similarity of 
    the `query` and `X` (all the documents) and returns the `top_k` similar documents."""
    
    # Vectorize the query to the same length as documents
    query_vec = vectorizor.transform(query)
    # Compute the cosine similarity between query_vec and all the documents
    cosine_similarities = cosine_similarity(X,query_vec).flatten()
    # Sort the similar documents from the most similar to less similar and return the indices
    most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]
    return (most_similar_doc_indices, cosine_similarities)

def show_similar_documents(df, cosine_similarities, similar_doc_indices):
    """ Prints the most similar documents using indices in the `similar_doc_indices` vector."""
    counter = 1
    for index in similar_doc_indices:
        st.write('Top-{}, Similarity = {}'.format(counter, cosine_similarities[index]))
        st.write('body: {}, '.format(df[index]))
        st.write()
        counter += 1


#  Read Vanguard Dataframe
obj_data_base_class = DataBaseClass('guest', 'Aa12345', 'localhost', 5432, 'insights_db')

vanguard_command = """select * from insights_data"""

vanguard_data = obj_data_base_class.db_query(
    command=vanguard_command,
    read_query=True
)

vanguard_col_command = """select column_name from information_schema.columns where table_name='insights_data'"""

vanguard_col = obj_data_base_class.db_query(
    
    command=vanguard_col_command,
    read_query=True
)
vanguard_col = [tup[0] for tup in vanguard_col]
vanguard_df = pd.DataFrame(vanguard_data, columns=vanguard_col)

 # Preprocess the corpus
data = [preprocess(title, body) for title, body in zip(vanguard_df['article_title'], vanguard_df['content'])] 

print('creating tfidf matrix...')
# Learn vocabulary and idf, return term-document matrix
X,v = create_tfidf_features(data)
features = v.get_feature_names_out()
len(features)

user_question = [query]
# search_start = time.time()
sim_vecs, cosine_similarities = calculate_similarity(X, v, user_question)
# search_time = time.time() - search_start
# print("search time: {:.2f} ms".format(search_time * 1000))
# print()
show_similar_documents(data, cosine_similarities, sim_vecs)


