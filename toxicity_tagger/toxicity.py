from .toxic_comment_tagger import ToxicCommentTagger, LABEL_COLUMNS
from .utilities.constants import MODELS_CACHE_DIR
from transformers import BertTokenizerFast as BertTokenizer
from .train import BERT_MODEL_NAME, MAX_TOKEN_COUNT
import numpy as np
import streamlit as st

def get_toxicity(texts, threshold=0.5):
    """
    A function to predict the toxicity tags of a piece of text.
    Args:
        texts: (list) A list of clean texts (cleaned by utilities.tweets.clean_tweet_texts).
        threshold: (float) A probabilty threshold for assigning a particular predicted toxicity tag to a tweet.

    Returns:

    """
    # Load in model
    trained_model = ToxicCommentTagger.load_from_checkpoint(
        MODELS_CACHE_DIR.joinpath("best-toxic-tagger-checkpoint.ckpt"),
        n_classes=len(LABEL_COLUMNS),
        label_columns=LABEL_COLUMNS,
        bert_model_name=BERT_MODEL_NAME
    )
    trained_model.eval()
    trained_model.freeze()

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    toxic_tags = []

    st.write("Inferring toxicity tags...")
    progress_bar = st.progress(0.0)

    for i, text in enumerate(texts):
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_TOKEN_COUNT,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        )

        _, prediction = trained_model(encoding["input_ids"], encoding["attention_mask"])
        prediction = prediction.flatten().numpy()

        tags = list(np.array(LABEL_COLUMNS)[np.where(prediction > threshold)[0]])
        toxic_tags.append(tags)

        print(f"{round((i + 1)/len(texts) * 100, 2)}% complete")
        progress_bar.progress((i + 1)/len(texts))

    return toxic_tags

def filter_by_toxicity(row, selected_tags):
    """Check to determine if the predicted toxicity tags are a subset of the tags specified by the user.
    Args:
        row: Row of a pandas.Dataframe to apply the function to.
        selected_tags: The toxicity tags specified by the user.

    Returns:

    """
    toxicity_tags = set(row['toxicity'])
    selected_tags = set(selected_tags)

    if selected_tags.issubset(toxicity_tags):
        return "toxicity detected"
    else:
        return "no toxicity"


if __name__ == "__main__":
    get_toxicity(["You are such a loser! You'll regret everything you've done to me!", "Hello world", "I love pie", "Fuck you"])

