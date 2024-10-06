import jax.numpy as jnp
from jax import random
import jax
import numpy as np

from data_utils import process_output
from . import VAE, TensorContracted, TensorConFormer, TensorConFormerDec, TensorConFormerEnc, Transformed

def sample_from_model(
        trainer,
        loader,
        n_samples,
        seed,
        info,
        transforms,
        ):
    
    target_name = info['target_name']
    
    main_rng = random.key(seed)
    n_classes = loader.dataset.n_classes
    

    z_rng, eps_rng, y_rng = random.split(main_rng, num=3)
    if trainer.model.conditional:
        ys = loader.dataset.label
        y_ = jnp.argmax(ys, axis=-1)
        y_sampled = random.choice(key=y_rng, a=y_, shape=(n_samples, ))
        y_sampled_oh = jax.nn.one_hot(y_sampled, num_classes=n_classes)
    
    if trainer.model_class.__name__ == "VAE":
        z = random.normal(key=z_rng, shape=(n_samples, trainer.model.latent_dim))
    elif trainer.model_class.__name__ == "Transformed":
        z = random.normal(key=z_rng, shape=(n_samples, len(info['variable_indices'])+1, trainer.model.embed_dim))
    else:
        z = random.normal(key=z_rng, shape=(n_samples, trainer.model.latent_dim, trainer.model.embed_dim))
    
    x_recon = trainer.state.apply_fn(trainer.state.params, 
                                     rngs={'eps': eps_rng}, 
                                     z=z, 
                                     c=y_sampled_oh if trainer.model.conditional else None, 
                                     method="sample_from_latent")

    df_syn = process_output(x_recon, info, transforms, trainer.model.conditional)
    if trainer.model.conditional:
        label_encoder = transforms['labels']
        y_sampled_oh = np.array(y_sampled_oh)
        labels = label_encoder.inverse_transform(y_sampled_oh)
        df_syn[target_name] = labels.squeeze()
    
    return df_syn

def get_model(model_name):
    if model_name == "VAE":
        model_class = VAE
    elif model_name == "TensorContracted":
        model_class = TensorContracted
    elif model_name == "TensorConFormer":
        model_class = TensorConFormer
    elif model_name == "Transformed":
        model_class = Transformed
    elif model_name == "TensorConFormerEnc":
        model_class = TensorConFormerEnc
    elif model_name == "TensorConFormerDec":
        model_class = TensorConFormerDec
    else:
        raise ValueError(f"Model: {model_name} not found.")
    return model_class

def get_model_parameters(model, flags):
    if model in ['TensorConFormer', 'TensorConFormerEnc', 'TensorConFormerDec']:
        model_hparams = {
            'num_layers': flags.num_layers,
            'embed_dim': flags.embed_dim,
            'latent_dim': flags.latent_dim,
            'num_heads': flags.num_heads,
            'dim_feedforward': flags.dim_feedforward,
            'dropout_prob': flags.dropout_prob,
            'hidden_dim': flags.hidden_dim,
        }

    elif model == "VAE":
        model_hparams = {
            'hidden_dim': flags.hidden_dim,
            'latent_dim': flags.latent_dim
        }

    elif model == "TensorContracted":
        model_hparams = {
            'hidden_dim': flags.hidden_dim,
            'latent_dim': flags.latent_dim,
            'embed_dim': flags.embed_dim
        }

    elif model == "Transformed":
        model_hparams = {
            'num_layers': flags.num_layers,
            'num_heads': flags.num_heads,
            'dim_feedforward': flags.dim_feedforward,
            'dropout_prob': flags.dropout_prob,
        }
    
    return model_hparams




