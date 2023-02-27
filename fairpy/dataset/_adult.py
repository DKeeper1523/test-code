# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import os
import sys
import pandas as pd
from typing import Sequence, Union

from ._base import TabDataset


class Adult(TabDataset):
    """ https://archive.ics.uci.edu/ml/datasets/adult """

    file2url = {
        "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    }

    column_names = (
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'income-per-year',
    )

    categorical_feat = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
    ]

    numerical_feat = [
        'age',
        'education-num',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
    ]

    drop_feat = ["fnlwgt"]
    na_values = ("?")

    label_name = "income-per-year"
    label_mapping = ("<=50K", ">50K")

    dataset_name = "adult"

    def __init__(
            self,
            sen_feat: Union[str, Sequence[str]] = "sex",
            dir_path: str = os.path.join(sys.path[1], "fairpy/data/adult"),
            download: bool = False,
    ):
        self._check_and_download_data(dir_path, self.file2url, download)

        train_df = pd.read_csv(
            os.path.join(dir_path, "adult.data"),
            header=None,
            names=self.column_names,
            skipinitialspace=True,
        )

        test_df = pd.read_csv(
            os.path.join(dir_path, "adult.test"),
            header=None,
            names=self.column_names,
            skipinitialspace=True,
        )
        for i, row in test_df.iterrows():
            if isinstance(row["income-per-year"], str):
                test_df.at[i, "income-per-year"] = row["income-per-year"][:-1]

        df = pd.concat([train_df, test_df], axis=0)
        df.reset_index(inplace=True)

        super(Adult, self).__init__(
            df=df,
            label_name=self.label_name,
            sen_feat=sen_feat,
            cat_feat=self.categorical_feat,
            num_feat=self.numerical_feat,
            dataset_name=self.dataset_name,
            custom_preprocessing=None,
            na_values=self.na_values,
            drop_feat=self.drop_feat,
            label_mapping=self.label_mapping,
        )
