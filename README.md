# Tabular data generation with tensor contraction layers and transformers

This repository provides instructions to train the models described in the paper: "Tabular data generation with tensor contraction layers and transformers" using your own dataset.

# Create environment and install dependencies via conda:

```
conda env create -f requirements.yml
```
then activate the environment:
```
conda activate tensorconformer 
```

# Jax + Flax

To install jax, you can install it via pip:

```
pip install -U jax
```

If you have CUDA 12 installed, you can also

```
pip install -U "jax[cuda12]"
```

For more details, please follow the official installation guide (https://jax.readthedocs.io/en/latest/installation.html)

To install flax

```
pip install flax
```

# Evaluating your own dataset

If you want to train a model using your own dataset `[DATANAME]`, start by creating `data` directory:

```
mkdir data
cd data
```

and then a directory to your datasets:

```
mkdir [DATANAME]
```

The datasets must be provided in the `.csv` format. It also expected that the datasets (train, test and validation) do **not** contain any missing values, and the first row denotes the **header** of the dataset. Then, create an `Info` directory:

```
mkdir Info
```

and provide a `.json` file with the following information

```
{
    'name': [DATANAME], # Dataset name
    'num_col_name': [LIST], # Numerical column names
    'cat_col_name': [LIST], # Categorical column names
    'target_name': [STRING], # Target variable name
    'data_path': data/[DATANAME]/[DATANAME].csv, # Path to the training data
    'test_path': data/[DATANAME]/[DATANAME]_test.csv, # Path to the test data
    'val_path': data/[DATANAME]/[DATANAME]_val.csv, # Path to validation data (optional).
}
```

and save it under `data/Info/[DATANAME].json`.

# Training a model

After all the needed information was created for your dataset, you can train a given model using `main.py`

```
python main.py --model=TensorConFormer --dataname=[DATANAME]
```

# Evaluating synthethic data quality

To evaluate synthetic data quality for the considered metrics, you can do so by installing the dependencies

```
conda env create -f requirements_eval.yml
```

activating the environment

```
conda activate evaluation
```

and then running the following

```
python main_eval.py --model=TensorConFormer --metric=[METRIC], --dataname=[DATANAME]
```

where `[METRIC] = [quality, highdensity, ml_efficiency]`.
