# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import os
import sys
import pandas as pd
from typing import Sequence, Union

from ._base import TabDataset


class Dutch(TabDataset):
    """ https://github.com/tailequy/fairness_dataset/blob/main/experiments/data/dutch.csv """

    file2url = {
        "dutch.csv": "https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/dutch.csv",
    }

    categorical_feat = [
        "sex",
        "age",
        "household_position",
        "household_size",
        "prev_residence_place",
        "citizenship",
        "country_birth",
        "edu_level",
        "economic_status",
        "cur_eco_activity",
        "marital_status",
    ]

    numerical_feat = [
    ]

    drop_feat = []
    na_values = ()

    label_name = "occupation"
    label_mapping = (0, 1)

    dataset_name = "dutch"

    def __init__(
            self,
            sen_feat: Union[str, Sequence[str]] = "sex",
            dir_path: str = os.path.join(sys.path[1], "fairpy/data/dutch"),
            download: bool = False,
    ):
        self._check_and_download_data(dir_path, self.file2url, download)
        df = pd.read_csv(os.path.join(dir_path, "dutch.csv"))

        super(Dutch, self).__init__(
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
