from utilities.tweets import clean_tweet_texts, get_tweet_embeddings, get_topics
from utilities.plots import visualise_embedding
from toxicity_tagger.toxicity import get_toxicity, filter_by_toxicity
import json
import streamlit as st

class EmbeddingPage:
    def __init__(self, tweets):
        self.texts = clean_tweet_texts(tweets)

    def visualise_embeddings(self, selected_toxicity=[]):
        if selected_toxicity:
            with st.spinner("Tagging toxicity..."):
                self.texts['toxicity'] = get_toxicity(self.texts['text'])

            # Filter by selected toxicity tags
            self.texts['selected_toxicity'] = self.texts.apply(filter_by_toxicity, axis=1, args=([selected_toxicity]))

            with st.spinner("Creating embedding matrix..."):
                scaled_embedding_matrix = get_tweet_embeddings(self.texts['text'])

            visualise_embedding(scaled_embedding_matrix, self.texts['selected_toxicity'],  colour_by='toxicity')

        else:
            with st.spinner("Creating embedding matrix..."):
                scaled_embedding_matrix = get_tweet_embeddings(self.texts['text'])
            with st.spinner("Clustering tweets..."):
                top_n_words_per_cluster, topic_cluster_sizes, label_column = get_topics(scaled_embedding_matrix,
                                                                                        self.texts['text'],
                                                                                        top_n_per_cluster=10)
                visualise_embedding(scaled_embedding_matrix, label_column, colour_by="cluster")

                st.write("Top N Words per Cluster: ")
                st.write(top_n_words_per_cluster)