# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import os
import sys
import pandas as pd
from typing import Sequence, Union

from ._base import TabDataset


class Oulad(TabDataset):
    """ https://analyse.kmi.open.ac.uk/open_dataset """

    file2url = {
        "anonymisedData.zip": "https://analyse.kmi.open.ac.uk/open_dataset/download",
    }

    categorical_feat = [
        "code_module",
        "code_presentation",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ]

    numerical_feat = [
        "num_of_prev_attempts",
        "studied_credits",
    ]

    drop_feat = ["id_student"]
    na_values = (" ")

    label_name = "final_result"
    label_mapping = ("Fail", "Pass")

    dataset_name = "oulad"

    def __init__(
            self,
            sen_feat: Union[str, Sequence[str]] = "gender",
            dir_path: str = os.path.join(sys.path[1], "fairpy/data/oulad"),
            download: bool = False,
    ):
        self._check_and_download_data(dir_path, self.file2url, download)
        df = pd.read_csv(os.path.join(dir_path, "studentInfo.csv"))

        super(Oulad, self).__init__(
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
        def final_result(row):
            return 'Pass' if row['final_result'] == "Distinction" else row['final_result']

        df['final_result'] = df.apply(lambda row: final_result(row), axis=1)
        df = df[(df["final_result"] == "Pass") | (df["final_result"] == "Fail")]
        df.reset_index(inplace=True, drop=True)

        return df
