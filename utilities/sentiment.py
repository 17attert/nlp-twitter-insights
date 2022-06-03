import pandas as pd
from .models import load_sentiment_classifier
from .data import batch_text, predictions_to_scores
import tensorflow as tf

def get_sentiment(texts, batch_size=64):
    """
    A function to predict the sentiment of a list of text strings.
    Args:
        texts: (list) Clean texts that require sentiment tagging.
        batch_size: (int) Size of each batch to be supplied to the sentiment classification model.

    Returns:
        (list) List of sentiment scores corresponding the tweets supplied as input.
    """

    # Load sentiment tokenizer and classifier
    tokenizer, model = load_sentiment_classifier(checkpoint="distilbert-base-uncased-finetuned-sst-2-english")

    # Preprocess text into batches
    batch_texts = batch_text(texts, batch_size)

    sentiment_scores = []
    for i, batch in enumerate(batch_texts):
        # Tokenize
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="tf")

        # Get preds
        output = model(**tokens)
        preds_array = tf.math.softmax(output.logits, axis=-1)

        # Get scores
        scores = predictions_to_scores(preds_array)

        sentiment_scores.extend(scores)

    return sentiment_scores


def create_sentiment_mapping(row):
    """
    Map a tweet's sentiment score to every word in the tweet.
    Args:
        row: (pd.Series) Row that the function should be applied to.

    Returns:
        (pd.DataFrame) DataFrame of tweet tokens and their corresponding sentiment scores.
    """
    tokens = row['tokens']
    score = row['sentiment_score']

    word_score = []
    for token in tokens:
        temp = {}
        temp['word'] = token
        temp['sentiment_score'] = score

        word_score.append(temp)

    return pd.DataFrame(word_score)


def score_dataset_with_sentiment(tweets_df, word_freq_df):
    """
    Score every word in a tweet corpus with the average sentiment of the tweets those words appear in.
    Args:
        tweets_df: (pandas.DataFrame) DataFrame of clean tweets.
        word_freq_df: (pandas.DataFrame) DataFrame consisting of every unique word in the clean tweet corpus and
                                         it's associated frequency in the corpus.

    Returns:
        (pandas.DataFrame) A DataFrame consisting of each unique word in the tweet corpus, scored by sentiment.
    """
    # Get sentiment for each tweet
    tweets_df['sentiment_score'] = get_sentiment(list(tweets_df['text'].values))

    # Create mapping table
    mapping_table_multi_index = tweets_df.apply(create_sentiment_mapping, axis=1, raw=False)
    mapping_table = pd.DataFrame()
    for i in mapping_table_multi_index.index:
        mapping_table = pd.concat([mapping_table, mapping_table_multi_index.iloc[i]], axis=0)
    mapping_table.reset_index(drop=True, inplace=True)

    # Group by word (average sentiment)
    mapping_table_agg = mapping_table.groupby(by='word').agg({'sentiment_score': 'mean'}).reset_index()

    sentiment_score_df = word_freq_df.merge(mapping_table_agg[['word', 'sentiment_score']], on='word',
                                                how='inner')

    return sentiment_score_df
