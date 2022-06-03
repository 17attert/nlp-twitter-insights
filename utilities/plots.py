import umap.umap_ as umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import streamlit as st
import random


def word_sentiment_to_colour(row):
    """Map sentiment score to a colour, the darker the colour the more
    intense the sentiment
    Args:
        row: Row of a pandas.DataFrame that the function can be applied to.

    Returns:
        (str) A RGB string indicating the color the word should be plotted in.
    """

    score = row['sentiment_score']
    if score < 0:
        return "rgb(%d, 0, 0)" % (255 - (-score) * 255)
    else:
        return "rgb(0, %d, 0)" % (255 - score * 255)


def generate_colour_map(labels):
    """
    Generate a RGB tuple for each different label supplied in the labels argument.
    Args:
        labels: (pandas.Series) A series of labels that require colours.

    Returns:
        (dict) A dict of label: (R, G, B) pairs.
    """
    colour_map = {}
    for label in labels:
        colour_map[label] = (random.random(), random.random(), random.random())

    return colour_map


def visualise_embedding(scaled_embedding_matrix, label_column, colour_by):
    """
    A function to reduce and visualise an embedding space.
    Args:
        scaled_embedding_matrix: (pandas.Dataframe) Embedding matrix returned by utilities.tweets.get_tweet_embeddings.
        label_column: (pandas.Series) A column indicating the labels of the points in the plot.
        colour_by: (str) A string indicating whether the embeddings have been coloured by "cluster" or "toxicity".

    Returns:
        None
    """

    # Apply UMAP to map the embedding vectors to a 2-dimensional vector space
    embedding = umap.UMAP(n_neighbors=15,
                          n_components=2,
                          metric='cosine').fit_transform(scaled_embedding_matrix)

    plot_data = pd.DataFrame(embedding, columns=['x', 'y'])
    plot_data[colour_by] = label_column

    # Create colour map
    colour_map = generate_colour_map(plot_data[colour_by].unique())

    # Generate figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(plot_data['x'], plot_data['y'], c=plot_data[colour_by].map(colour_map))

    # Add a legend
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) for k, v in
               colour_map.items()]
    ax.legend(title=colour_by, handles=handles, bbox_to_anchor=(1.05, 1), loc='upper right')

    plt.show()
    st.pyplot(fig)
