import pandas as pd

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_statistical import AlphaPrecision

from .utils import process_data_for_evaluation


def evaluate_highdensity(
        df_real: pd.DataFrame, 
        df_syn: pd.DataFrame, 
        num_col_name: list,
        cat_col_name: list
    ) -> dict:
    
    x_real, x_syn = process_data_for_evaluation(df_real, df_syn, num_col_name=num_col_name, cat_col_name=cat_col_name)

    loader_real = GenericDataLoader(x_real)
    loader_syn = GenericDataLoader(x_syn)

    alpha_prec = AlphaPrecision()

    report = alpha_prec.evaluate(loader_real, loader_syn)
    report = {
        k: v for (k, v) in report.items() if "naive" in k
    }
    return report
    