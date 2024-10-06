import numpy as np
import pandas as pd

def process_output(dataset: np.ndarray, 
                   info: dict, 
                   transforms: dict,
                   conditional: bool):
    
    var_indices = info['variable_indices']
    num_col_name = info['num_col_name']
    cat_col_name = info['cat_col_name'] if conditional else info['cat_col_name'] + [info['target_name']]
    column_names = num_col_name + cat_col_name

    x_ = {}
    for var, idxs in var_indices.items():
        if np.size(idxs) == 1:
            x_[var] = dataset[:, idxs].squeeze()
        else:
            x_[var] = np.argmax(dataset[:, idxs], axis=-1)
    
    df = pd.DataFrame(x_)
    if num_col_name and cat_col_name:
        numerical_transform, ordinal_transform = transforms['numerical'], transforms['ordinal']
        df[num_col_name] = numerical_transform.inverse_transform(df[num_col_name])
        df[cat_col_name] = ordinal_transform.inverse_transform(df[cat_col_name])
        
    elif not num_col_name and cat_col_name:
        ordinal_transform = transforms['ordinal']
        df[cat_col_name] = ordinal_transform.inverse_transform(df[cat_col_name])
    
    elif num_col_name and not cat_col_name:
        numerical_transform = transforms['numerical']
        df[num_col_name] = numerical_transform.inverse_transform(df[num_col_name])

    else:
        raise ValueError("Scenario not expected.")
    
    df = df[column_names]
    
    return df