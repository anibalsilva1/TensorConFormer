import numpy as np
import pandas as pd

from typing import Dict, Tuple, List, Callable
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV


def info_for_sdmetrics(num_col_name: list,
                       cat_col_name: list):
    info_sdv = {}
    info_sdv['infoDATA_SPEC_VERSION'] = 'SINGLE_TABLE_V1'

    columns_cat = {k: {'sdtype': 'categorical'} for k in cat_col_name}
    columns_num = {k: {'sdtype': 'numerical'} for k in num_col_name}

    columns = columns_cat | columns_num
    
    info_sdv['columns'] = columns
    return info_sdv

def process_numerical_data(
        df_real: pd.DataFrame,
        df_syn: pd.DataFrame,
        num_col_name: List,
    ) -> Tuple[np.ndarray, np.ndarray]:

    numerical_transform = StandardScaler()
    real = df_real.copy()
    syn = df_syn.copy()

    numerical_transform.fit(real[num_col_name])
    x_real_num = numerical_transform.transform(real[num_col_name])
    x_syn_num = numerical_transform.transform(syn[num_col_name])

    return x_real_num, x_syn_num

def process_categorical_data(
        df_real: pd.DataFrame,
        df_syn: pd.DataFrame,
        cat_col_name: List,
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    real = df_real.copy()
    syn  = df_syn.copy()

    categorical_transform = OneHotEncoder(sparse_output=False,
                                          handle_unknown='ignore',
                                          drop=None)
    
    categorical_transform.fit(real[cat_col_name])

    
    x_syn_cat = categorical_transform.transform(syn[cat_col_name])

    x_real_cat = categorical_transform.transform(real[cat_col_name])

    return x_real_cat, x_syn_cat

def process_data_for_evaluation(
        df_real: pd.DataFrame, 
        df_syn: pd.DataFrame, 
        num_col_name: list,
        cat_col_name: list,
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    if num_col_name and cat_col_name:
        x_real_num, x_syn_num = process_numerical_data(df_real, df_syn, num_col_name)
        x_real_cat, x_syn_cat = process_categorical_data(df_real, df_syn, cat_col_name)

        x_syn = np.concatenate([x_syn_num, x_syn_cat], axis=1)
        x_real = np.concatenate([x_real_num, x_real_cat], axis=1)
    
    elif not num_col_name and cat_col_name:
        x_real, x_syn = process_categorical_data(df_real, df_syn, cat_col_name)
    
    elif num_col_name and not cat_col_name:
        x_real, x_syn = process_numerical_data(df_real, df_syn, num_col_name)
    
    return x_real, x_syn


def process_data_for_mle(
        x_real_train: pd.DataFrame, 
        x_real_test: pd.DataFrame, 
        x_syn: pd.DataFrame, 
        info: dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:


    num_col_name = info['num_col_name']
    cat_col_name = info['cat_col_name']

    if num_col_name and cat_col_name:
        numerical_transformation = StandardScaler()

        categorical_transform = OneHotEncoder(sparse_output=False,
                                              handle_unknown='ignore',
                                              drop=None)

        numerical_transformation.fit(x_real_train[num_col_name])
        categorical_transform.fit(x_real_train[cat_col_name])

        x_train_num = numerical_transformation.transform(x_real_train[num_col_name])
        x_test_num = numerical_transformation.transform(x_real_test[num_col_name])
        x_syn_num = numerical_transformation.transform(x_syn[num_col_name])

        x_syn_cat = categorical_transform.transform(x_syn[cat_col_name])
        x_train_cat = categorical_transform.transform(x_real_train[cat_col_name])
        x_test_cat = categorical_transform.transform(x_real_test[cat_col_name])

        x_train = np.concatenate([x_train_num, x_train_cat], axis=1)
        x_test = np.concatenate([x_test_num, x_test_cat], axis=1)
        x_syn = np.concatenate([x_syn_num, x_syn_cat], axis=1)
    
    elif not num_col_name and cat_col_name:
        categorical_transform = OneHotEncoder(sparse_output=False,
                                              handle_unknown='ignore',
                                              drop=None)
        
        categorical_transform.fit(x_real_train[cat_col_name])

        x_syn = categorical_transform.transform(x_syn[cat_col_name])
        x_train = categorical_transform.transform(x_real_train[cat_col_name])
        x_test = categorical_transform.transform(x_real_test[cat_col_name])
    
    elif num_col_name and not cat_col_name:

        numerical_transformation = StandardScaler()
        numerical_transformation.fit(x_real_train[num_col_name])
        
        x_train = numerical_transformation.transform(x_real_train[num_col_name])
        x_test = numerical_transformation.transform(x_real_test[num_col_name])
        x_syn = numerical_transformation.transform(x_syn[num_col_name])

    return x_train, x_test, x_syn


def grid_search(
        x: np.ndarray,
        y: np.ndarray,
        model: Callable,
        param_grid: Dict, 
        score_fn: str, 
        seed: int,
        n_splits: int
    ) -> Dict:

    gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=score_fn,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
            n_jobs=-1
        )

    gs.fit(x, y)
    best_params = gs.best_params_
    
    return best_params