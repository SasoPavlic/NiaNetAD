from typing import List, Optional, Union

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class Transformer_train(Dataset):
    def __init__(self, batch_size, x_train, y_train):

        train_size = 0
        if x_train.shape[0] % batch_size == 0:
            train_size = x_train.shape[0]
        else:
            train_size = x_train.shape[0] - (x_train.shape[0] % batch_size)

        df_train = x_train.head(train_size)
        self.y_train = y_train
        self.y_train = torch.tensor(self.y_train[:].values)
        self.x_train = torch.tensor(df_train[:].values)
        # https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor
        super(Transformer_train, self).__init__()

    def __len__(self):
        shape = self.x_train.shape[0]
        return shape

    def __getitem__(self, index):
        input, label = self.x_train[index], self.y_train[index]
        return input, label


class Transformer_val(Dataset):
    def __init__(self, batch_size, x_train, y_train):

        train_size = 0
        if x_train.shape[0] % batch_size == 0:
            train_size = x_train.shape[0]
        else:
            train_size = x_train.shape[0] - (x_train.shape[0] % batch_size)

        df_val = x_train.head(train_size)
        self.y_val = y_train
        self.y_val = torch.tensor(self.y_val[:].values)
        self.x_val = torch.tensor(df_val[:].values)
        super(Transformer_val, self).__init__()

    def __len__(self):
        shape = self.x_val.shape[0]
        return shape

    def __getitem__(self, index):
        input, label = self.x_val[index], self.y_val[index]
        return input, label


class Transformer_test(Dataset):
    def __init__(self, batch_size, x_test, y_test):
        test_size = 0
        if x_test.shape[0] % batch_size == 0:
            test_size = x_test.shape[0]
        else:
            test_size = x_test.shape[0] - (x_test.shape[0] % batch_size)

        df_test = x_test.head(test_size)
        self.y_test = y_test
        self.y_test = torch.tensor(self.y_test[:].values)
        self.x_test = torch.tensor(df_test[:].values)
        super(Transformer_test, self).__init__()

    def __len__(self):
        shape = self.x_test.shape[0]
        return shape

    def __getitem__(self, index):
        input, label = self.x_test[index], self.y_test[index]
        return input, label


class TabularDataset(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 1,
            num_workers: int = 16,
            pin_memory: bool = False,
            train_size: int = 80,
            test_size: int = 10,
            val_size: int = 10,
            anomaly_label: bool = True,
            **kwargs,
    ):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.anomaly_label = anomaly_label

    def setup(self, stage: Optional[str] = None) -> None:
        with open(self.data_path) as f:
            df_dataset = pd.read_csv(f, delimiter=";")

        df_target = df_dataset["Fault_lag"]
        df_data = df_dataset.astype('float32').drop("Fault_lag", axis=1)

        x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=self.test_size)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size)

        y_train = pd.Series([1 if yi == self.anomaly_label else 0 for yi in y_train])
        y_test = pd.Series([1 if yi == self.anomaly_label else 0 for yi in y_test])
        y_val = pd.Series([1 if yi == self.anomaly_label else 0 for yi in y_val])

        self.train_dataset = Transformer_train(self.batch_size, x_train, y_train)
        self.test_dataset = Transformer_test(self.batch_size, x_test, y_test)
        self.val_dataset = Transformer_val(self.batch_size, x_val, y_val)

    # TODO Implement re-usable datalaoder process
    # https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
    def train_dataloader(self) -> DataLoader:
        data = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            # persistent_workers=True
        )

        return data

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        data = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            # persistent_workers=True
        )
        return data

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        data = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            # persistent_workers=True
        )
        return data
