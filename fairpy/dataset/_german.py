# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import os
import sys
import pandas as pd
from typing import Sequence, Union

from ._base import TabDataset


class German(TabDataset):
    """ https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29 """

    file2url = {
        "german.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
    }

    column_names = (
        "status",
        "month",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings",
        "employment",
        "investment_as_income_percentage",
        "personal_status",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "installment_plans",
        "housing",
        "number_of_credits",
        "skill_level",
        "people_liable_for",
        "telephone",
        "foreign_worker",
        "credit",
    )

    categorical_feat = [
        "status",
        "credit_history",
        "purpose",
        "savings",
        "employment",
        "other_debtors",
        "property",
        "installment_plans",
        "housing",
        "skill_level",
        "telephone",
        "foreign_worker",
        "sex",
        "marital-status",
    ]

    numerical_feat = [
        "month",
        "credit_amount",
        "investment_as_income_percentage",
        "residence_since",
        "age",
        "number_of_credits",
        "people_liable_for",
    ]

    drop_feat = ["personal_status"]
    na_values = ()

    label_name = "credit"
    label_mapping = ("2", "1")

    dataset_name = "german"

    def __init__(
            self,
            sen_feat: Union[str, Sequence[str]] = "sex",
            dir_path: str = os.path.join(sys.path[1], "fairpy/data/german"),
            download: bool = False,
    ):
        self._check_and_download_data(dir_path, self.file2url, download)

        df = pd.read_csv(
            os.path.join(dir_path, "german.data"),
            sep=" ",
            names=self.column_names,
        )

        super(German, self).__init__(
            df=df,
            label_name=self.label_name,
            sen_feat=sen_feat,
            cat_feat=self.categorical_feat,
            num_feat=self.numerical_feat,
            dataset_name=self.dataset_name,
            custom_preprocessing=self.default_preprocessing,
            na_values=self.na_values,
            drop_feat=self.drop_feat,
            label_mapping=self.label_mapping,
        )

    @staticmethod
    def default_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a derived sex attribute based on personal_status.
        https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/german_dataset.py
        """

        sex_map = {'A91': 'male', 'A93': 'male', 'A94': 'male', 'A92': 'female', 'A95': 'female'}
        marital_status_map = {'A91': 'married', 'A92': 'married', 'A93': 'single', 'A94': 'married', 'A95': 'single'}
        df['sex'] = df['personal_status'].replace(sex_map)
        df["marital-status"] = df["personal_status"].replace(marital_status_map)
        df["credit"] = df["credit"].astype(str)

        return df
