import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import QuantileTransformer, \
                                  StandardScaler, \
                                  MinMaxScaler, \
                                  OneHotEncoder, \
                                  OrdinalEncoder

class DataProcessor(object):
    def __init__(self,
                 num_transform: str,
                 cat_transform: str,
                 data_path: str = 'data',
                 conditional_model: bool = True,
                 as_numpy: bool = True,
                 ):
        
        self.num_transform = num_transform
        self.cat_transform = cat_transform
        self.data_path = data_path
        self.conditional_model = conditional_model
        self.as_numpy = as_numpy

        self.info_path = f'{data_path}/Info'
        
    def read_info(self, dataname):
        with open(f'{self.info_path}/{dataname}.json', 'r') as f:
            info = json.load(f)
        
        return info

    def read_data_from_path(self, info):
        train_path = info['data_path']
        test_path = info['test_path']

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        if 'val_path' in info:
            val_path = info['val_path']
            df_val = pd.read_csv(val_path)
            return df_train, df_test, df_val
        
        return df_train, df_test
    
    def get_numerical_transform(self, X_train, num_col_names):
        if self.num_transform == 'std':
            normalizer = StandardScaler()
        elif self.num_transform == "minmax":
            normalizer = MinMaxScaler()
        elif self.num_transform == 'quantile':
            # Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
            normalizer = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
                subsample=int(1e9),
                random_state=21)
        else:
            raise ValueError(f'Normalization type: {self.num_transform} not implemented. Possible normalizations: ["std", "minmax", "quantile"].')
        
        normalizer.fit(X_train[num_col_names])
        return normalizer
    
    def ordinal_encoder(self, datasets, cat_col_names):
        X = pd.concat(datasets, axis=0)
        ordinal_transform = OrdinalEncoder()
        ordinal_transform.fit(X[cat_col_names])
        return ordinal_transform
    
    def get_categorical_transform(self, datasets, cat_col_names):
        '''
        For categorical variables, transformation is applied to the whole dataset 
        because some categories from test / val might not be present in train.
        '''
        X = pd.concat(datasets, axis=0)
        if self.cat_transform == 'one_hot':
            categorical_transform = OneHotEncoder(sparse_output=False, 
                                                  drop=None,
                                                  handle_unknown='ignore')
        else:
            raise NotImplementedError(f'Categorical Transformation type: {self.cat_transform} not implemented. Possible categorical transformation: "one_hot".')

        categorical_transform.fit(X[cat_col_names])
    
        return categorical_transform
        

    def transform_numerical(self, datasets, normalizer, num_col_names):
        x_nums = []
        for dataset in datasets:
            x_num = normalizer.transform(dataset[num_col_names])
            x_num = pd.DataFrame(x_num, columns=num_col_names)
            x_nums.append(x_num)
        
        return x_nums
    
    def transform_categorical(self, datasets, cat_transform, cat_col_names):
        x_cats = []
        new_categories = cat_transform.get_feature_names_out()
        for dataset in datasets:
            x_cat = cat_transform.transform(dataset[cat_col_names])
            x_cat = pd.DataFrame(x_cat, columns=new_categories)
            x_cats.append(x_cat)

        return x_cats
    
    def transform_datasets(self, datasets, transforms, info):

        num_col_names = info['num_col_name']
        if self.conditional_model:
            cat_col_names = info['cat_col_name']
        else:
            cat_col_names = info['cat_col_name'] + [info['target_name']]
        
        train_df = datasets[0]
        if num_col_names:
            numerical_normalizer = self.get_numerical_transform(train_df, num_col_names)
            dfs_num = self.transform_numerical(datasets, numerical_normalizer, num_col_names)
            transforms['numerical'] = numerical_normalizer

        if cat_col_names:
            ordinal_transform = self.ordinal_encoder(datasets, cat_col_names)
            for dataset in datasets:
                dataset[cat_col_names] = ordinal_transform.transform(dataset[cat_col_names])
                
            categorical_transform = self.get_categorical_transform(datasets, cat_col_names)
            dfs_cat = self.transform_categorical(datasets, categorical_transform, cat_col_names)
            transforms['ordinal'] = ordinal_transform
            transforms['categorical'] = categorical_transform
        
        if num_col_names and cat_col_names:
            dfs = [pd.concat([df_num, df_cat], axis=1) for df_num, df_cat in zip(dfs_num, dfs_cat)]
            return dfs, transforms
        
        elif num_col_names and not cat_col_names:
            return dfs_num, transforms
        
        elif not num_col_names and cat_col_names:
            return dfs_cat, transforms
        
        else:
            raise ValueError("Combination not expected.")

    def process_variable_indices(self, transforms, info):
        '''
        Processes the variable indices from the new dataset.
        Numerical variables always come first, if present.
        '''
        num_col_names = info['num_col_name']
        if self.conditional_model:
            cat_col_names = info['cat_col_name']
        else:
            cat_col_names = info['cat_col_name'] + [info['target_name']]
            
        x_cat_start = 0
        if num_col_names and cat_col_names:
            num_transform, cat_transform = transforms['numerical'], transforms['categorical']
            feats_num = num_transform.feature_names_in_

            numerical_indices = [np.array([i]) for i in range(len(feats_num))]
            numerical_indices = {k: v for k, v in zip(feats_num, numerical_indices)}
            x_cat_start = len(numerical_indices)

            feats_cat = cat_transform.feature_names_in_
            categories_sizes = [np.size(s) for s in cat_transform.categories_]
            cat_offsets = x_cat_start + np.cumsum(categories_sizes)
            
            categorical_indices = [np.arange(x_cat_start, cat_offsets[i]) if i == 0 else np.arange(cat_offsets[i-1], cat_offsets[i]) for i in range(len(cat_offsets))]
            categorical_indices = {k: v for k, v in zip(feats_cat, categorical_indices)}

            variable_indices = numerical_indices | categorical_indices
            return variable_indices
        
        elif num_col_names and not cat_col_names:
            num_transform = transforms['numerical']
            
            feats_num = num_transform.feature_names_in_

            numerical_indices = [np.array([i]) for i in range(len(feats_num))]
            numerical_indices = {k: v for k, v in zip(feats_num, numerical_indices)}

            return numerical_indices
        
        elif not num_col_names and cat_col_names:
            cat_transform = transforms['categorical']
            feats_cat = cat_transform.feature_names_in_
            categories_sizes = [np.size(s) for s in cat_transform.categories_]
            cat_offsets = np.cumsum(categories_sizes)
            cat_offsets = x_cat_start + cat_offsets

            category_indices = [np.arange(x_cat_start, cat_offsets[i]) if i == 0 else np.arange(cat_offsets[i-1], cat_offsets[i]) for i in range(len(cat_offsets))]
            category_indices = {k: v for k, v in zip(feats_cat, category_indices)}
            return category_indices
        
        else:
            raise ValueError("Combination not expected.")
    
    def label_encoder(self, labels):
        y_train = labels[0]
        label_encoder = OneHotEncoder(sparse_output=False,
                                      drop=None,
                                      )
        label_encoder.fit(y_train.to_frame())
        labels_oh = []
        for label in labels:
            label_oh = label_encoder.transform(label.to_frame())
            labels_oh.append(label_oh)
        return labels_oh, label_encoder
    

    def process_data(self, dataname):
        info = self.read_info(dataname)
        datasets = self.read_data_from_path(info)

        target_name = info['target_name']
        transforms = {}

        labels = [ds[target_name].copy() for ds in datasets]
        labels, label_encoder = self.label_encoder(labels)
        transforms['labels'] = label_encoder

        if self.conditional_model:
            datasets = [ds.drop(target_name, axis=1).copy() for ds in datasets]

        datasets, transforms = self.transform_datasets(datasets=datasets,
                                                       transforms=transforms,
                                                       info=info)
        
        variable_indices = self.process_variable_indices(transforms=transforms,
                                                         info=info)

        info['variable_indices'] = variable_indices
        
        if self.as_numpy:
            datasets = [ds.to_numpy() for ds in datasets]
        
        return datasets, labels, transforms, info