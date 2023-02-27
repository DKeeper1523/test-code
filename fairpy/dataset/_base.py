# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

""" Base class for all datasets """

import os
import sys
import gdown
import tarfile
import pandas as pd
import numpy as np
import numpy.typing as npt
from zipfile import ZipFile
from collections import OrderedDict
from typing import Sequence, Any, Callable, Optional, NamedTuple, Tuple, Dict, Union, Mapping
from abc import ABC, abstractmethod

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


class FeatIndex(NamedTuple):
    """ Feature names and indexes after one-hot encoding for tabular dataset """
    cat_feat: Tuple[str, ...]
    num_feat: Tuple[str, ...]
    sen_feat: Tuple[str, ...]
    cat_idx: Tuple[int, ...]
    num_idx: Tuple[int, ...]
    sen_idx: Tuple[int, ...]
    feat2idx: Mapping[str, Tuple[int, ...]]


class SplitData(NamedTuple):
    """ Training data, label, and sensitive attributes after train, validation, and test split """
    X_train: npt.NDArray[np.float64]
    y_train: np.ndarray
    s_train: npt.NDArray[np.int16]
    X_val: npt.NDArray[np.float64]
    y_val: np.ndarray
    s_val: npt.NDArray[np.int16]
    X_test: npt.NDArray[np.float64]
    y_test: np.ndarray
    s_test: npt.NDArray[np.int16]


class Dataset(ABC):
    """ Abstract base class for all datasets """

    def _check_and_download_data(
            self,
            dir_path: str,
            file2url: Mapping[str, str],
            download: bool,
    ) -> None:
        """
        Check local data and download if no data available

        Parameters
        ----------
        dir_path: str
            Directory of the data folder

        file2url: str -> str
            File name and corresponding url for downloading

        download: bool
            Set to True to enable automatic downloading

        use_gdown: bool
            Set to True to use gdown for downloading

        """

        def check_file_exist(dir_path: str, file2url: Mapping[str, str]) -> Mapping[str, bool]:
            """ Existence flag for each file """
            flag = {f: False for f in file2url.keys()}
            for file_name in file2url.keys():
                if os.path.isfile(os.path.join(dir_path, file_name)):
                    flag[file_name] = True
            return flag

        def raise_not_exist(exist_flag: Mapping[str, bool]) -> None:
            for file_name, exist in exist_flag.items():
                if not exist:
                    raise FileExistsError(
                        "%s does not exist in %s, consider set `download` to Ture" % (file_name, dir_path)
                    )

            return

        def download_data(
                dir_path: str,
                file2url: Mapping[str, str],
                exist_flag: Mapping[str, bool],
        ) -> None:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for file_name, url in file2url.items():
                if not exist_flag[file_name]:
                    print("FairPy => Download %s from %s" % (file_name, url))
                    gdown.download(url, os.path.join(dir_path, file_name), quiet=False)

                # TODO: include all compressed file
                if file_name.endswith(".zip"):
                    zf = ZipFile(os.path.join(dir_path, file_name), 'r')
                    zf.extractall(dir_path)
                    zf.close()
                elif file_name.endswith(".tar.gz"):
                    f = tarfile.open(os.path.join(dir_path, file_name))
                    f.extractall(dir_path)
                    f.close()

            return

        exist_flag = check_file_exist(dir_path, file2url)
        if not download:
            raise_not_exist(exist_flag)
        else:
            download_data(dir_path, file2url, exist_flag)

        return


class TabDataset(Dataset):
    """
    Base class for fair classification on tabular datasets

    Parameters
    ----------
    df: pd.DataFrame
        Tabular dataset in pandas dataframe

    label_name: str
        The name of target variable in dataframe

    sen_feat: str or Sequence[str]
        The name of sensitive feature(s), can be one or multiple features

    cat_feat: Sequence[str]
        The list of names of categorical features

    num_feat: Sequence[str]
        The list of names of numerical features

    dataset_name: str
        The name of the dataset

    custom_preprocessing: Callable: pd.DataFrame -> pd.DataFrame
        Customized preprocessing for dataframe, e.g., data selection or deletion

    """

    def __init__(
            self,
            df: pd.DataFrame,
            label_name: str,
            sen_feat: Union[str, Sequence[str]],
            cat_feat: Sequence[str],
            num_feat: Sequence[str],
            dataset_name: Optional[str] = None,
            custom_preprocessing: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
            na_values: Optional[Sequence[Any]] = None,
            drop_feat: Optional[Sequence[str]] = None,
            label_mapping: Optional[Sequence[Any]] = None,
    ):
        if isinstance(sen_feat, str):
            sen_feat = tuple([sen_feat])

        self.df = df
        self.label_name = label_name
        if isinstance(sen_feat, str):
            self._sen_feat = tuple([sen_feat])
        else:
            self._sen_feat = tuple(sen_feat)
        self._cat_feat = tuple(cat_feat)
        self._num_feat = tuple(num_feat)
        self.dataset_name = dataset_name
        self.na_values = na_values
        self.drop_feat = drop_feat
        self.label_mapping = label_mapping

        self._feat2idx = None
        self.label_encoder = LabelEncoder()
        self.sen_encoder = LabelEncoder()
        if self.label_mapping is not None:
            self.label_encoder.classes_ = self.label_mapping

        if custom_preprocessing is not None:
            self.df = custom_preprocessing(self.df)
        self._std_preprocessing()

        self.input_df = self.df[[feat for feat in self._num_feat] + [feat for feat in self._cat_feat]]
        self.target_df = self.df[self.label_name]
        self.sen_df = self.df[list(self._sen_feat)]

        # TODO: more encoding?
        self.encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.encoded_df = self.one_hot_encoding(self.df)

    def _std_preprocessing(self) -> None:
        """ Drop features and delete row with nan values """

        if self.drop_feat is not None:
            self.df.drop(columns=self.drop_feat, inplace=True)
        if self.na_values is not None:
            self.df.replace(self.na_values, np.nan, inplace=True)
            self.df.dropna(axis=0, inplace=True)
            self.df.reset_index(inplace=True)

        return

    def one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """ One-hot encoding for categorical features """

        df_cat = df[list(self._cat_feat)]
        df_num = df[list(self._num_feat)]

        if hasattr(self.encoder, "categories_"):
            df_cat_values = self.encoder.transform(df_cat.values)
        else:
            df_cat_values = self.encoder.fit_transform(df_cat.values)

            self._cat_col_name = ["%s.%s" % (name, val) for name, all_val in
                                  zip(self._cat_feat, self.encoder.categories_) for val in all_val]
            self._feat2idx = OrderedDict({feat: [i] for i, feat in enumerate(self._num_feat)})
            curr_idx = len(self._num_feat)
            for i, feat in enumerate(self._cat_feat):
                feat_idx = [j + curr_idx for j in range(len(self.encoder.categories_[i]))]
                self._feat2idx.update({feat: feat_idx})
                curr_idx += len(feat_idx)

            self._cat_idx, self._num_idx, self._sen_idx = [], [], []
            for feat in self._feat2idx.keys():
                if feat in self._cat_feat:
                    self._cat_idx.extend(self._feat2idx[feat])
                if feat in self._num_feat:
                    self._num_idx.extend(self._feat2idx[feat])
                if feat in self._sen_feat:
                    self._sen_idx.extend(self._feat2idx[feat])

            self._cat_idx = tuple(self._cat_idx)
            self._num_idx = tuple(self._num_idx)
            self._sen_idx = tuple(self._sen_idx)
            self._feat2idx = {key: tuple(list_) for key, list_ in self._feat2idx.items()}
            self._feat2idx = OrderedDict(sorted(self._feat2idx.items(), key=lambda x: x[1][0]))  # order by index

        df_cat = pd.DataFrame(df_cat_values, columns=self._cat_col_name)
        encoded_df = pd.concat([df_num, df_cat], axis=1)

        return encoded_df

    @property
    def feat_idx(self) -> FeatIndex:
        cat_idx = tuple([idx for feat in self._cat_feat for idx in self._feat2idx[feat]])
        num_idx = tuple([idx for feat in self._num_feat for idx in self._feat2idx[feat]])
        sen_idx = tuple([idx for feat in self._sen_feat for idx in self._feat2idx[feat]])

        return FeatIndex(self._cat_feat, self._num_feat, self._sen_feat, cat_idx, num_idx, sen_idx, self._feat2idx)

    def get_X_y_s(
            self,
            onehotify: bool = True,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int16], npt.NDArray[np.int16]]:
        """ Return training data, labels, and sensitive attributes """

        y = self.target_df.to_numpy()
        if hasattr(self.label_encoder, "classes_"):
            y = self.label_encoder.transform(y)
        else:
            y = self.label_encoder.fit_transform(y)

        # TODO: add support for multiple sensitive attribute
        s = self.sen_df.to_numpy().squeeze()
        s = self.sen_encoder.fit_transform(s)

        if onehotify:
            X = self.encoded_df.to_numpy()
            X, y, s = self.validate(X, y, s)
        else:
            X = self.input_df.to_numpy()

        return X, y, s

    def split(self, val_p: float = 0.15, test_p: float = 0.15, random_state: int = 42):
        assert val_p < 1. and test_p < 1., "Invalid partition of data"

        X, y, s = self.get_X_y_s()
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, s, test_size=test_p, random_state=random_state)
        if val_p != 0.:
            X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
                X_train, y_train, s_train, test_size=val_p / (1 - test_p), random_state=random_state)
        else:
            X_val = y_val = s_val = None

        return SplitData(X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test)

    def validate(self, X: np.ndarray, y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = X.astype(np.float64)
        y = y.astype(np.int16)
        s = s.astype(np.int16)

        assert len(np.unique(y)) > 1, "Only one class exists in the dataset"
        assert np.any(y) >= 0, "Class label should be an integer larger or equal to zero"
        if len(np.unique(s)) == 1:
            import warnings
            warnings.warn("Only one sensitive group in the dataset")

        return X, y, s

    def get_params(self):
        return

    @property
    def name(self):
        return self.dataset_name
