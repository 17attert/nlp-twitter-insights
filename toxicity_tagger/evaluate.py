"""Generate performance metrics for the Toxicity Tagger model on a test set.
"""

from toxic_comment_tagger import ToxicCommentTagger, LABEL_COLUMNS
from toxic_comment_dataset import ToxicCommentsDataset
from utilities.constants import MODELS_CACHE_DIR, DATA_DIR
from transformers import BertTokenizerFast as BertTokenizer
import pandas as pd
from tqdm.auto import tqdm
from torchmetrics.functional import auroc, accuracy
from train import BERT_MODEL_NAME, MAX_TOKEN_COUNT

import torch

if __name__ == "__main__":
    # Load in model
    trained_model = ToxicCommentTagger.load_from_checkpoint(
        MODELS_CACHE_DIR.joinpath("best-toxic-tagger-checkpoint.ckpt"),
        n_classes=len(LABEL_COLUMNS),
        label_columns=LABEL_COLUMNS,
        bert_model_name=BERT_MODEL_NAME
    )
    trained_model.eval()
    trained_model.freeze()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = trained_model.to(device)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    test_dataset = ToxicCommentsDataset(
        pd.read_csv(DATA_DIR.joinpath("test_with_labels.csv")),
        tokenizer,
        max_token_len=MAX_TOKEN_COUNT,
        label_columns=LABEL_COLUMNS
    )

    predictions = []
    labels = []
    for item in tqdm(test_dataset):
        _, prediction = trained_model(
            item["input_ids"].unsqueeze(dim=0).to(device),
            item["attention_mask"].unsqueeze(dim=0).to(device)
        )
        predictions.append(prediction.flatten())
        labels.append(item["labels"].int())
    predictions = torch.stack(predictions).detach().cpu()
    labels = torch.stack(labels).detach().cpu()

    predictions = torch.stack(predictions).detach().cpu()
    labels = torch.stack(labels).detach().cpu()

    print(f"Accuracy: {accuracy(predictions, labels, threshold=0.5)}")

    print("AUROC per tag")
    for i, name in enumerate(LABEL_COLUMNS):
        tag_auroc = auroc(predictions[:, i], labels[:, i], pos_label=1)
        print(f"{name}: {tag_auroc}")