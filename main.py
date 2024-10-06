import os
import pandas as pd
import time
import pickle
import jax
from jax import random
import jax.numpy as jnp

from trainers import get_trainer

from data_utils import DataProcessor
from data_utils import create_data_loaders
from models import sample_from_model, get_model, get_model_parameters

from absl import flags
from absl import app
from absl import logging

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

# General flags
flags.DEFINE_string(name="dataname", default=None, help="Dataset name.")
flags.DEFINE_string(name="model", default="TensorConFormer", help="Model name.")

# Data processing
flags.DEFINE_string(name="datapath", default='data', help="Path of the dataname.")
flags.DEFINE_bool(name="conditional_model", default=False, help="Conditional genereration.")
flags.DEFINE_bool(name="as_numpy", default=True, help="Process datasets as numpy arrays.")
flags.DEFINE_string(name="num_transform", default="quantile", help="Numerical transformation.")
flags.DEFINE_string(name="cat_transform", default="one_hot", help="Categorical transformation.")

# Models Parameters
flags.DEFINE_integer(name="embed_dim", default=4, help="Embedding dimension for tokenization.")
flags.DEFINE_integer(name="latent_dim", default=32, help="Latent dimension.")
flags.DEFINE_integer(name="hidden_dim", default=96, help="Hidden dimension.")

# Transformer paramaters
flags.DEFINE_integer(name="num_layers", default=2, help="Number of layers in the transformer.")
flags.DEFINE_integer(name="num_heads", default=1, help="Number of heads for attention.")
flags.DEFINE_integer(name="dim_feedforward", default=128, help="Embeddings projection dimension.")
flags.DEFINE_float(name="dropout_prob", default=0.0, help="Dropout probability.")

# Training
flags.DEFINE_integer(name="print_every_n_epochs", default=15, help="Print evaluation loss every n epochs.")
flags.DEFINE_integer(name="num_epochs", default=500, help="Number of epochs to train.")
flags.DEFINE_boolean(name="save_models", default=False, help="Save models.")
flags.DEFINE_boolean(name="save_best_models_only", default=False, help="Save only best models.")
flags.DEFINE_integer(name="save_interval_steps", default=1, help="Save models between a given interval.")
flags.DEFINE_integer(name="batch_size", default=2048, help="Batch size of data loaders.")

# Optimizer
flags.DEFINE_string(name="optimizer", default="adam", help="Optimizer name.")
flags.DEFINE_float(name="lr", default=1e-3, help="Learning rate.")
flags.DEFINE_integer(name="warmup", default=0, help="Learning rate warmup.")

# Early stopping
flags.DEFINE_float(name="min_delta", default=1e-3, help="Improvement needed to not raise patience.")
flags.DEFINE_integer(name="patience", default=15, help="Patience.")
flags.DEFINE_string(name="best_metric", default="inf", help="Bound for best metric.")


def main(argv):

    syn_path = "synthetic"
    save_syn_path = f'{syn_path}/{FLAGS.dataname}'
    if not os.path.exists(save_syn_path):
        os.makedirs(save_syn_path)

    meta_path = "meta"
    save_meta_path = f'{meta_path}/{FLAGS.dataname}'

    if not os.path.exists(save_meta_path):
        os.makedirs(save_meta_path)

    trainer_class = get_trainer(FLAGS.model)
    processor = DataProcessor(
        num_transform=FLAGS.num_transform,
        cat_transform=FLAGS.cat_transform,
        data_path=FLAGS.datapath,
        conditional_model=FLAGS.conditional_model,
        as_numpy=FLAGS.as_numpy,
        )

    datasets, labels, transforms, info = processor.process_data(dataname=FLAGS.dataname)

    loaders = create_data_loaders(
        dataframes=datasets, 
        labels=labels, 
        is_train=[True, True, False] if len(datasets) > 2 else [True, True], 
        batch_size=FLAGS.batch_size)
    
    if len(loaders) > 2:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, val_loader = loaders
        test_loader = val_loader

    variable_indices = info['variable_indices']

    model_hparams = get_model_parameters(model=FLAGS.model, flags=FLAGS)
    
    if FLAGS.model != 'VAE':
        output_dim = len(variable_indices)
        model_hparams['output_dim'] = output_dim
        model_hparams['variable_indices'] = variable_indices
        model_hparams['conditional'] = FLAGS.conditional_model

    else:
        model_hparams['output_dim'] = datasets[0].shape[1]

    trainer = trainer_class(
        model_hparams=model_hparams,
        optimizer_name=FLAGS.optimizer,
        optimizer_hparams={'lr': FLAGS.lr,
                           'warmup': FLAGS.warmup},
        early_stopping_params={'min_delta': FLAGS.min_delta,
                               'patience': FLAGS.patience,
                               'best_metric': float(FLAGS.best_metric)},
        exmp_input=next(iter(train_loader)),
        dataset_name=FLAGS.dataname,
        checkpointer_params = {'save_interval_steps': FLAGS.save_interval_steps},
        save_models=FLAGS.save_models,
        print_every_n_epochs=FLAGS.print_every_n_epochs,
        indexes=variable_indices
    )

    start_train_time = time.time()
    metrics = trainer.train_model(train_loader=train_loader, 
                                  val_loader=val_loader, 
                                  test_loader=test_loader, 
                                  num_epochs=FLAGS.num_epochs, 
                                  save_best_models_only=FLAGS.save_best_models_only)
    end_train_time = time.time()
    
    start_sample_time = time.time()
    df_syn = sample_from_model(trainer=trainer, 
                               loader=train_loader, 
                               n_samples=datasets[0].shape[0],
                               seed=42,
                               info=info,
                               transforms=transforms)
    
    end_sample_time = time.time()

    model_class = get_model(FLAGS.model)
    m = model_class(**model_hparams)
    params = m.init({'params': random.key(0), 'eps': random.key(1)}, next(iter(train_loader))[0], next(iter(train_loader))[1])
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    meta = {
        'training_time': end_train_time - start_train_time,
        'sampling_time': end_sample_time - start_sample_time,
        'num_params': num_params,
        'metrics': metrics
    }

    with open(f'{save_meta_path}/{FLAGS.model}.pkl', 'wb') as f:
        pickle.dump(meta, f)
    

    df_syn.to_csv(f'{save_syn_path}/{FLAGS.model}.csv', index=False)

if __name__ == "__main__":
    app.run(main)        

