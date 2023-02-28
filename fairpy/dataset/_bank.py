# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import os
import sys
import pandas as pd
from typing import Sequence, Union

from ._base import TabDataset


class Bank(TabDataset):
    """ https://archive.ics.uci.edu/ml/datasets/bank+marketing """

    file2url = {
        "bank-additional.zip": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
    }

    categorical_feat = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
    ]

    numerical_feat = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed"
    ]

    drop_feat = []
    na_values = ("unknown")

    label_name = "y"
    label_mapping = ("no", "yes")

    dataset_name = "bank"

    def __init__(
            self,
            sen_feat: Union[str, Sequence[str]] = "marital",
            dir_path: str = os.path.join(sys.path[1], "fairpy/data/bank"),
            download: bool = False,
    ):
        self._check_and_download_data(dir_path, self.file2url, download)
        df = pd.read_csv(os.path.join(dir_path, "bank-additional/bank-additional-full.csv"), sep=";")

        super(Bank, self).__init__(
            df=df,
            label_name=self.label_name,
            sen_feat=sen_feat,
            cat_feat=self.categorical_feat,
            num_feat=self.numerical_feat,
            dataset_name=self.dataset_name,
            na_values=self.na_values,
            drop_feat=self.drop_feat,
            label_mapping=self.label_mapping,
        )
