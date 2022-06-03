import pytorch_lightning as pl
from .toxic_comment_dataset import ToxicCommentsDataset
from torch.utils.data import DataLoader


class ToxicCommentDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, label_columns, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.label_columns = label_columns
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = ToxicCommentsDataset(
            self.train_df,
            self.tokenizer,
            self.label_columns,
            self.max_token_len
        )
        self.test_dataset = ToxicCommentsDataset(
            self.test_df,
            self.tokenizer,
            self.label_columns,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )
