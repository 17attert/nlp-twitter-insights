
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    """
    Calculate the cluster-wide TF-IDF.

    Args:
        documents: (pandas.DataFrame) Documents composed of all tweets in an identified cluster joined together.
        m:
        ngram_range:

    Returns:

    """
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic_cluster(tf_idf, count, docs_per_topic, n=20):
        words = count.get_feature_names()
        labels = list(docs_per_topic.Topic)
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        return top_n_words

def extract_topic_cluster_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                    .Doc
                    .count()
                    .reset_index()
                    .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                    .sort_values("Size", ascending=False)).reset_index(drop=True)
    return topic_sizes