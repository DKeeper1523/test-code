# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import os
import sys
import pandas as pd
from typing import Sequence, Union

from ._base import TabDataset


class Credit(TabDataset):
    """ https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients """

    file2url = {
        "default of credit card clients.xls": "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
    }

    categorical_feat = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]

    numerical_feat = [
        "LIMIT_BAL",
        "AGE",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    drop_feat = []
    na_values = ()

    label_name = "default payment next month"
    label_mapping = (0, 1)

    dataset_name = "credit"

    def __init__(
            self,
            sen_feat: Union[str, Sequence[str]] = "SEX",
            dir_path: str = os.path.join(sys.path[1], "fairpy/data/credit"),
            download: bool = False,
    ):
        self._check_and_download_data(dir_path, self.file2url, download)
        df = pd.read_excel(os.path.join(dir_path, "default of credit card clients.xls"), header=1, index_col=0)

        super(Credit, self).__init__(
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
