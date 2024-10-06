import json
import os
import pandas as pd

from absl import flags
from absl import app
from absl import logging

from utils import evaluate_metric

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string(name='metric', default='highdensity', help='Name of the metric to evaluate.')
flags.DEFINE_string(name="dataname", default=None, help='Name of the dataset to evaluate.')
flags.DEFINE_string(name='model', default='TensorConFormer', help='Models to obtain results.')
flags.DEFINE_string(name='synthetic_path', default='synthetic', help='Synthetic data path.')
flags.DEFINE_string(name='data_path', default='data', help='Dataset path.')
flags.DEFINE_string(name='info_path', default='Info', help='Dataset information path.')
flags.DEFINE_boolean(name='conditional', default=False, help='If the model was conditionally trained.')
flags.DEFINE_string(name='save_path', default='eval_results', help='Model results save path.')

def main(argv):
    data_path = FLAGS.data_path
    dataname = FLAGS.dataname
    synthetic_path = FLAGS.synthetic_path
    info_path = FLAGS.info_path
    model = FLAGS.model
    metric = FLAGS.metric
    conditional = FLAGS.conditional
    save_path = FLAGS.save_path

    metric_save_path = f'{save_path}/{metric}/{dataname}'
    if not os.path.exists(metric_save_path):
        os.makedirs(metric_save_path)

    with open(f'{data_path}/{info_path}/{dataname}.json', 'r') as f:
        info = json.load(f)
    
    real_df_path = info['data_path']
    test_df_path = info['test_path']
    synthetic_df_path = f'{synthetic_path}/{dataname}/{model}.csv'

    real_df = pd.read_csv(real_df_path)
    test_df = pd.read_csv(test_df_path)
    syn_df = pd.read_csv(synthetic_df_path)
    
    row = evaluate_metric(metric_name=metric,
                          model=model,
                          df_train_real=real_df,
                          df_test_real=test_df,
                          syn_df=syn_df,
                          conditional=conditional,
                          info=info
                )
    
    res_df = pd.DataFrame([row])
    res_df.to_csv(f'{metric_save_path}/{model}.csv', index=False)


if __name__ == '__main__':
    app.run(main)