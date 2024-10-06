import numpy as np
import pandas as pd
from typing import Callable

from metrics.utils import process_data_for_mle, grid_search
from sklearn.preprocessing import LabelEncoder

def evaluate_ml_efficiency(
        real_train_df: pd.DataFrame, 
        real_test_df: pd.DataFrame, 
        syn_df: pd.DataFrame, 
        model: Callable, 
        info: dict, 
        param_grid: dict,
        evaluation_metrics: dict,
        n_folds: int = 5,
        score_fn: str = 'accuracy',
        seed: int = 222):
    
    target_name = info['target_name']

    x_real_train, y_train = real_train_df.drop(target_name, axis=1), real_train_df[target_name]
    x_real_test, y_test = real_test_df.drop(target_name, axis=1), real_test_df[target_name]
    x_syn, y_syn = syn_df.drop(target_name, axis=1), syn_df[target_name]


    # Generative model may not synthetize all labels, so we fit LabelEncoder for each data source.
    label_enc_train = LabelEncoder()
    label_enc_syn = LabelEncoder()
    label_enc_train.fit(y_train)
    label_enc_syn.fit(y_syn)

    y_train = label_enc_train.transform(y_train)
    y_test = label_enc_train.transform(y_test)
    y_syn = label_enc_syn.transform(y_syn)


    #assert np.unique(y_train).all() == np.unique(y_syn).all(), "synthethic data does not share the same classes as train."

    x_train, x_test, x_syn = process_data_for_mle(x_real_train, x_real_test, x_syn, info)

    best_params_syn = grid_search(
        x=x_syn, y=y_syn, model=model, 
        param_grid=param_grid, score_fn=score_fn, 
        seed=seed, n_splits=n_folds)

    syn_model = model.__class__(**best_params_syn)
    syn_model.fit(x_syn, y_syn)
    syn_preds = syn_model.predict(x_test)

    best_params_real = grid_search(
                x=x_train, y=y_train, model=model, 
                param_grid=param_grid, score_fn=score_fn, 
                seed=seed, n_splits=n_folds)
        
    real_model = model.__class__(**best_params_real)
    real_model.fit(x_train, y_train)
    test_preds = real_model.predict(x_test)

    # After training, we revert the labels back to ensure proper mapping
    syn_preds = label_enc_syn.inverse_transform(syn_preds)
    syn_preds = label_enc_train.transform(syn_preds)

    for metric, metric_call in evaluation_metrics.items():
        if metric == "utility": utility = metric_call(y_test, syn_preds)
        elif metric == "accuracy_train_test": acc_train_test = metric_call(y_test, test_preds)
        elif metric == "fidelity": fidelity = metric_call(test_preds, syn_preds)
        else: raise ValueError(f"Metric {metric} not studied.")

    report = {'utility': utility, 'acc_train_test': acc_train_test, 'fidelity': fidelity}

    return report