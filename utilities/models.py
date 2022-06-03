import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

def load_sentiment_classifier(checkpoint):
    """
    Load a Sequence Classification model from the Hugging Face Model Hub using a checkpoint.
    Args:
        checkpoint: (str) String indicating the name of the model (as specified on the Hugging Face Model Hub).

    Returns:
        (tuple): The model's tokenizer, the model object.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

    return tokenizer, model