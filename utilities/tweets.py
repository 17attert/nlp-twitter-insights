import re
import string
import json

import pandas as pd

from .twitterati import lookups

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import StandardScaler

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import umap.umap_ as umap
import hdbscan
from .topics import c_tf_idf, extract_top_n_words_per_topic_cluster, extract_topic_cluster_sizes

# Save stopwords into set for faster lookup
stops = set(stopwords.words('english'))


def get_wordnet_pos(tag):
    """
    Convert a POS tag from NLTK form to Wordnet form.
    Args:
       tag: POS tag in NLTK form.

    Returns:
        POS tag in Wordnet form.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def get_tokens(row):
    tokens = row['tokens']
    words = []
    for token in tokens:
        temp = {}
        temp['word'] = token
        words.append(temp)

    return pd.DataFrame(words)


def tweet_to_wordlist(tweet, pos=[], remove_stopwords=True):
    # Function converts text to a sequence of words,
    # Returns a list of words.
    lemmatizer = WordNetLemmatizer()

    # Get parts of speech of tweet
    tokens_pos = nltk.pos_tag(nltk.word_tokenize(tweet))

    # Filter by specified parts of speech
    if pos:
        words = []

        for token in tokens_pos:
            if token[1] in pos:
                words.append(token)
            else:
                continue
    else:
        words = tokens_pos

    # lemmatizing
    words = [lemmatizer.lemmatize(word_token[0], get_wordnet_pos(word_token[1])) for word_token in words]

    # Remove stop words
    if remove_stopwords:
        words = [w for w in words if not w in stops]

    return words


def create_word_freq_df(tweets_df):
    # Create mapping table
    mapping_table_multi_index = tweets_df.apply(get_tokens, axis=1, raw=False)
    mapping_table = pd.DataFrame()
    for i in mapping_table_multi_index.index:
        mapping_table = pd.concat([mapping_table, mapping_table_multi_index.iloc[i]], axis=0)
    mapping_table.reset_index(drop=True, inplace=True)
    mapping_table_agg = mapping_table.groupby(by='word').size().reset_index()

    wordcloud_df = mapping_table_agg.rename({
        0: "count"
    }, axis=1)

    return wordcloud_df


def get_tweets(hashtags, period=None, remove_retweets=True, remove_replies=True, remove_quote=True, lang='en',
               max_count=1000, save=False):
    """
    Retrieve tweets
    Args:
        hashtags: (list) List of hashtags to query the Twitter APIv2 with.
        period: (int) Number of days prior to today to query the API with. Maximum 6.
        remove_retweets: (boolean) Exclude retweets from thr query?
        remove_replies: (boolean) Exclude replied from thr query?
        remove_quote: (boolean) Exclude quotes from thr query?
        lang: (str) Language of tweets.
        max_count: (int) Maximum number of tweets that should be returned.
        save: (boolean) Save the tweets as a JSON?

    Returns:
        (list) List of tweets returned as a result of querying the Twitter APIv2 using the query parameters supplied.
    """
    # Create the search query
    if len(hashtags) > 1:
        query = hashtags[0]
        for hashtag in hashtags[1:]:
            query = query + " OR " + hashtag
    else:
        query = hashtags[0]

    # Filter out retweets
    if remove_retweets:
        query = query + " -is:retweet"

    # Filter out replies
    if remove_replies:
        query = query + " -is:reply"

    if remove_quote:
        query = query + " -is:quote"

    # Return lang tweets only
    query = query + f" lang:{lang}"

    # Perform the query
    search_result = lookups.recent_search_lookup(search_query=query, period=period, max_count=max_count)

    # Create a dict of the search result
    tweets = {}
    for i, result in enumerate(search_result):
        tweets[i] = result

    # Save to JSON
    if save:
        with open(f'data/tweets-{query}-{period}-{max_count}.json', 'w') as fp:
            json.dump(tweets, fp)

    return tweets


def get_tweet_embeddings(texts):
    """
     A function to embed a set of tweets into an embedding space.
    Args:
        texts: list
            List of tweets to embed.

    Returns:
        numpy.ndarray
            Scaled embedding matrix created from the supplied tweets.
    """

    # Load in the model to produce the embeddings
    model = SentenceTransformer(
        model_name_or_path="all-MiniLM-L6-v2")

    # Grab the tweet embeddings
    tweet_embeddings = model.encode(texts, show_progress_bar=True)

    # Stack the input list of embeddings to create a single matrix
    embedding_matrix = np.stack(tweet_embeddings)

    # Scale the matrix
    embedding_matrix = pd.DataFrame(embedding_matrix)
    scaled_embedding_matrix = StandardScaler().fit_transform(embedding_matrix)

    return scaled_embedding_matrix


def get_topics(scaled_embedding_matrix, texts, top_n_per_cluster=10):
    """
    A function to uncover the topics present in a text embedding.
    Args:
        scaled_embedding_matrix: numpy.ndarray
            Scaled tweet embedding matrix.

        texts: list
            List of tweets.

        top_n_per_cluster: int
            Number of topics to return per cluster.
    Returns:
        Tuple
            Top n words per cluster, Size of each cluster, Numeric label for each cluser.
    """

    # Apply UMAP to map the embedding vectors to a 2-dimensional vector space
    umap_embeddings = umap.UMAP(n_neighbors=15,
                                n_components=5,
                                metric='cosine').fit_transform(scaled_embedding_matrix)

    cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                              metric='euclidean',
                              cluster_selection_method='eom').fit(umap_embeddings)

    # Create documents per identified cluster
    docs_df = pd.DataFrame(texts)
    docs_df.columns = ['Doc']
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(texts))

    # topic -1 refers to all documents that did not have any topics assigned
    top_n_words_per_cluster = extract_top_n_words_per_topic_cluster(tf_idf, count, docs_per_topic, n=top_n_per_cluster)
    topic_cluster_sizes = extract_topic_cluster_sizes(docs_df)

    return top_n_words_per_cluster, topic_cluster_sizes, cluster.labels_

def de_emojify(text):
    """Removes emojis from a given string and returns the de-emojified string.

        Args:
            text (str): String to be de-emojified.

        Returns:
            str: De-emojified string.
        """
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def de_link(text):
    """Removes URLs from a given strinf and returns the string without hyperlinks.

    Args:
        text (str): String containing hyperlinks.

    Returns:
        str: String without hyperlinks.
    """

    text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)

    return text


def clean_tweet_texts(tweets):
    # Extract text from tweets and remove hyperlinks, emojis, and other useless characters
    texts = []
    for i in range(len(tweets)):
        temp_text = de_emojify(tweets[i]['text'])
        temp_text = de_link(temp_text)
        temp_text = temp_text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
        temp_text = temp_text.replace("@", "").replace("#", "").replace("\n", "")  # remove @, #
        texts.append(temp_text)

    return pd.DataFrame(texts, columns=["text"])
