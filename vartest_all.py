import argparse
import numpy as np
from config import cfg, cfg_from_list
from pysqoe.models import get_qoe_model
from pysqoe.datasets import get_dataset
from pysqoe.evaluation.ftest import FTest
from pysqoe.evaluation import LogisticHamid, Logistic5


def parse_args():
    parser = argparse.ArgumentParser(description='PySQoE: A python library to benchmark objective QoE models for streaming videos')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    arg = parser.parse_args()
    return arg

def rescale(x):
    x_ = np.array(x)
    x_min = np.min(x_)
    x_max = np.max(x_)
    y_min = 0
    y_max = 100
    y = x_ * (y_max - y_min) / (x_max - x_min) + (x_max - y_max * x_min) / (x_max - x_min)
    y = y - np.min(y)
    return y

def compute_prediction_residual(model, dataset):
    sbj_score = []
    obj_score = []
    for i in range(len(dataset)):
        streaming_video, mos = dataset[i]
        sbj_score.append(mos)
        x = model(streaming_video)
        obj_score.append(x)
    fun = LogisticHamid(x=obj_score, y=sbj_score)
    y = fun(x=obj_score)
    res = y - sbj_score
    return res

def record(row, out, mode='a+'):
    if out is not None:
        with open(file=out, mode=mode) as f:
            f.write(row + '\n')

"""
In this experiment, we perform F-test on the prediction residuals in the aggregated database
(LIVE-NFLX-I, LIVE-NFLX-II, WaterlooSQoE-III, WaterlooSQoE-IV). To alleviate potential dist-
ribution mismatch among the datasets, we apply different non-linear mapping on each dataset,
and aggregate the prediction residuals.
"""
def main():
    input_list = cfg.INPUT_DIR.split(':')
    model_list = cfg.MODEL.split(':')
    models = [get_qoe_model(model_name) for model_name in model_list]
    # Note that the mos in LIVE-NFLX-I and LIVE-NFLX-II are transformed to z-score,
    # In order to make a meaningful evaluation, we need to firstly rescale the range of 
    # mos in LIVE-NFLX datasets to match the range of mos in WaterlooSQoE datasets.
    # LIVE-NFLX-I
    livenflx1 = get_dataset(dataset='LIVE-NFLX-I', root_dir=input_list[0], version=cfg.DATASET_VERSION)
    x = livenflx1.streaming_logs['mos']
    y = rescale(x)
    livenflx1.streaming_logs['mos'] = y
    res_dict = {}
    for model_name, model in zip(model_list, models):
        res = compute_prediction_residual(model, livenflx1)
        res_dict[model_name] = res
    # LIVE-NFLX-II
    livenflx2 = get_dataset(dataset='LIVE-NFLX-II', root_dir=input_list[1], version=cfg.DATASET_VERSION)
    x = livenflx2.streaming_logs['mos']
    y = rescale(x)
    livenflx2.streaming_logs['mos'] = y
    for model_name, model in zip(model_list, models):
        res = compute_prediction_residual(model, livenflx2)
        res_dict[model_name] = np.concatenate((res_dict[model_name], res), axis=0)
    # WaterlooSQoE-III
    waterloosqoe3 = get_dataset(dataset='WaterlooSQoE-III', root_dir=input_list[2], version=cfg.DATASET_VERSION)
    for model_name, model in zip(model_list, models):
        res = compute_prediction_residual(model, waterloosqoe3)
        res_dict[model_name] = np.concatenate((res_dict[model_name], res), axis=0)
    # # WaterlooSQoE-IV
    waterloosqoe4 = get_dataset(dataset='WaterlooSQoE-IV', root_dir=input_list[3], version=cfg.DATASET_VERSION)
    for model_name, model in zip(model_list, models):
        res = compute_prediction_residual(model, waterloosqoe4)
        res_dict[model_name] = np.concatenate((res_dict[model_name], res), axis=0)
    
    # sanity check
    for model, res in res_dict.items():
        print('%s: %.3f' % (model, np.sqrt(np.mean(res**2))))
    
    # perform F-Test and record results
    row = ['F-Test'] + [key for key, _ in res_dict.items()]
    row = ','.join(row)
    record(row=row, out='results/Concat/F-Test.csv', mode='w')
    for model1, x1 in res_dict.items():
        performance = [FTest.vartest(x1=x1, x2=x2) for _, x2 in res_dict.items()]
        row = [model1] + performance
        row = ','.join(row)
        record(row=row, out='results/Concat/F-Test.csv') 


if __name__ == "__main__":
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    main()