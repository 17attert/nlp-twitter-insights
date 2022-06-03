import numpy as np


def batch_text(texts, batch_size):
    """
    Batch tweets for inference by the Sentiment Classification model.
    Args:
        texts: (list) List of clean tweets.
        batch_size: Size of each batch of tweets to be passed to the model.

    Returns: (numpy.array) The original texts list of tweets, split into batches.

    """
    if len(texts) > batch_size:
        # Ensure array split will result in equal division
        if len(texts) % batch_size != 0:
            last_batch = texts[((len(texts) // batch_size) * batch_size):
                               (((len(texts) // batch_size) * batch_size) + len(texts) % batch_size)]
            texts = texts[:((len(texts) // batch_size) * batch_size)]
        else:
            last_batch = None

        texts = np.array(texts)
        batched_texts = np.split(texts, len(texts) / batch_size)

        # Convert back to list
        batched_texts = [list(batch) for batch in batched_texts]

        if last_batch is not None:
            batched_texts.append(last_batch)

    else:
        batched_texts = np.array(texts)

    return batched_texts


def predictions_to_scores(preds_array):
    """
    Convert predictions on the interval [0, 1] to a score on the interval [-1, 1].
    Args:
        preds_array: (list) Array of predictions from the Sentiment Classifier.

    Returns:
        (list) A list of sentiment scores on the interval [-1, 1].
    """
    # Convert prediction that text is positive into score
    scores = []
    for pred in preds_array:
        scores.append(round(pred.numpy()[1] * 2 + -1, 5))

    return scores
