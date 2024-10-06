import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from metrics import evaluate_highdensity, evaluate_ml_efficiency, evaluate_quality_report

eval_metrics = {
    'utility': accuracy_score,
    'accuracy_train_test': accuracy_score,
    'fidelity': accuracy_score
    }


xgboost_hparams = {
    'n_estimators': [100, 200],
    'subsample': [0.7, 0.9, 1],
    'colsample_bytree': [0.7, 0.9, 1],
    }

def evaluate_metric(
    metric_name: str,
    model: str,
    df_train_real: pd.DataFrame,
    syn_df: pd.DataFrame,
    info: dict,
    conditional: bool,
    df_test_real: pd.DataFrame=None):

    dataset_name = info['name']
    
    columns_name = ['dataset_name', 'model']
    row_values = [dataset_name, model]
    
    num_col_name = info['num_col_name']
    cat_col_name = info['cat_col_name'] if conditional else info['cat_col_name'] + [info['target_name']]

    if metric_name == 'highdensity':
        report = evaluate_highdensity(
            df_real=df_train_real, 
            df_syn=syn_df, 
            num_col_name=num_col_name, 
            cat_col_name=cat_col_name
        )
        
        precision_alpha = report['delta_precision_alpha_naive']
        coverage_beta = report['delta_coverage_beta_naive']
        authenticity = report['authenticity_naive']

        columns_name = columns_name + ['alpha_precision', 'beta_recall', 'authenticity']
        row_values = row_values + [precision_alpha, coverage_beta, authenticity]
            
    elif metric_name == 'quality':
        report = evaluate_quality_report(
            df_real=df_train_real, 
            df_syn=syn_df, 
            num_col_name=num_col_name, 
            cat_col_name=cat_col_name
        )
        
        marginal = report['marginal']
        pairs_corr = report['pairs_correlation']
        data_val = report['data_validity']
        data_struct = report['data_struct']
        
        columns_name = columns_name + ['marginal', 'pairs-correlation', 'data-validity', 'data-struct']
        row_values = row_values + [marginal, pairs_corr, data_val, data_struct]

    elif metric_name == "ml_efficiency":
        
        report = evaluate_ml_efficiency(
            real_train_df=df_train_real,
            real_test_df=df_test_real,
            syn_df=syn_df,
            model=XGBClassifier(),
            info=info,
            param_grid=xgboost_hparams,
            evaluation_metrics=eval_metrics
        )
        utility = report['utility'] 
        acc_train_test = report['acc_train_test']
        fidelity = report['fidelity']
    
        columns_name = columns_name + ['utility', 'accuracy_train_test', 'fidelity']
        row_values = row_values + [utility, acc_train_test, fidelity]
    
    else:
        raise ValueError(f"Metric {metric_name} not studied.")
    
    row = {k: v for k, v in zip(columns_name, row_values)}
    return row