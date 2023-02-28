# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import os
import sys
import pandas as pd
from typing import Sequence, Union

from ._base import TabDataset


class Compas(TabDataset):
    """ https://github.com/propublica/compas-analysis """

    file2url = {
        "compas-scores-two-years.csv": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
    }

    features_to_keep = (
        "sex",
        "age",
        "age_cat",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_charge_degree",
        "c_charge_desc",
        "two_year_recid",
    )

    categorical_feat = [
        "sex",
        "age_cat",
        "race",
        "c_charge_degree",
        "c_charge_desc",
    ]

    numerical_feat = [
        "age",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
    ]

    drop_feat = []
    na_values = ()

    label_name = "two_year_recid"
    label_mapping = ("Did recid.", "No recid.")

    dataset_name = "compas"

    def __init__(
            self,
            sen_feat: Union[str, Sequence[str]] = "race",
            dir_path: str = os.path.join(sys.path[1], "fairpy/data/compas"),
            download: bool = False,
    ):
        self._check_and_download_data(dir_path, self.file2url, download)
        df = pd.read_csv(os.path.join(dir_path, "compas-scores-two-years.csv"), index_col='id')

        super(Compas, self).__init__(
            df=df,
            label_name=self.label_name,
            sen_feat=sen_feat,
            cat_feat=self.categorical_feat,
            num_feat=self.numerical_feat,
            dataset_name=self.dataset_name,
            custom_preprocessing=self.custom_preprocessing,
            na_values=self.na_values,
            drop_feat=self.drop_feat,
            label_mapping=self.label_mapping,
        )

    def custom_preprocessing(self, df):
        """
        Perform the same preprocessing as the original analysis:
        https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        """

        def two_year_recid(row):
            return 'Did recid.' if row['two_year_recid'] == 1 else 'No recid.'

        df['two_year_recid'] = df.apply(lambda row: two_year_recid(row), axis=1)
        df = df[
            (df.days_b_screening_arrest <= 30)
            & (df.days_b_screening_arrest >= -30)
            & (df.is_recid != -1)
            & (df.c_charge_degree != 'O')
            & (df.score_text != 'N/A')
            ]
        df = df[list(self.features_to_keep)]

        return df
