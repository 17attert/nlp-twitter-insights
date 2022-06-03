"""Finetuning script for the Toxicity Tagger.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast as BertTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from .toxic_comment_dataset import ToxicCommentsDataset
from .toxic_comment_data_module import ToxicCommentDataModule
from .toxic_comment_tagger import ToxicCommentTagger, LABEL_COLUMNS
from .utilities.constants import DATA_DIR

from pytorch_lightning.loggers import TensorBoardLogger

BERT_MODEL_NAME = 'bert-base-cased'
MAX_TOKEN_COUNT = 512
RANDOM_SEED = 42

# Training parameters
N_EPOCHS = 10
BATCH_SIZE = 12

if __name__ == "__main__":
    # Set random seed
    pl.seed_everything(RANDOM_SEED)

    # Read in and split training data
    df = pd.read_csv(DATA_DIR.joinpath("train.csv"))
    train_df, val_df = train_test_split(df, test_size=0.25)

    # Downsample toxic comments in training data
    train_toxic = train_df[train_df[LABEL_COLUMNS].sum(axis=1) > 0]
    train_clean = train_df[train_df[LABEL_COLUMNS].sum(axis=1) == 0]
    train_df = pd.concat([
        train_toxic,
        train_clean.sample(int(0.1 * train_df.shape[0]))
    ])

    # Get tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # Create train_dataset
    train_dataset = ToxicCommentsDataset(
        train_df,
        tokenizer,
        LABEL_COLUMNS,
        max_token_len=MAX_TOKEN_COUNT
    )

    # Create training data module
    data_module = ToxicCommentDataModule(
        train_df,
        val_df,
        LABEL_COLUMNS,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_COUNT
    )

    # Create model
    steps_per_epoch = len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 5
    model = ToxicCommentTagger(
        n_classes=len(LABEL_COLUMNS),
        label_columns=LABEL_COLUMNS,
        bert_model_name=BERT_MODEL_NAME,
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps
    )

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="/content/drive/MyDrive/ToxicClassifier/models/cache",
        filename="best-toxic-tagger-checkpoint",
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger("lightning_logs", name="toxic-comments")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    progress_callback = TQDMProgressBar(30)

    # Create trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_callback],
        max_epochs=N_EPOCHS,
        gpus=1
    )

    # Train
    # trainer.fit(ckpt_path=MODELS_CACHE_DIR.joinpath("last.ckpt")
    trainer.fit(model, data_module)
