import argparse
from config import cfg, cfg_from_list
from pysqoe.models import train
from pysqoe.datasets import get_dataset
from pysqoe.experiment import Experiment

"""
PySQoE takes a list of streaming videos and the corresponding mean opinion scores, produces
objective QoE scores as specified by MODEL_PROFILES (dumped in the INPUT_DATASET csv), and
evaluates the performance of the models in terms of CRITERIA.

Sample usage:
python main.py --set MODE train MODEL Liu2012QoE:Yin2015QoE:KSQI
python main.py --set DATASET LIVE-NFLX-II INPUT_DIR data/LIVE-NFLX-II \
                     MODEL FTW:Mok2011QoE:Liu2012QoE:Xue2014QoE:Yin2015QoE:Spiteri2016QoE:Bentaleb2016QoE:SQI:P1203:VideoATLAS:KSQI
python main.py --set MODE parse DATASET WaterlooSQoE-I INPUT_DIR data/WaterlooSQoE-I
"""

def parse_args():
    parser = argparse.ArgumentParser(description='PySQoE: A python library to benchmark objective QoE models for streaming videos')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    arg = parser.parse_args()
    return arg

def main():
    if cfg.MODE == 'train':
        dataset_s = get_dataset(dataset=cfg.TRAINING_DATASET_S, root_dir=cfg.DATASET_S_ROOT, download=cfg.DOWNLOAD)
        dataset_a = get_dataset(dataset=cfg.TRAINING_DATASET_A, root_dir=cfg.DATASET_A_ROOT, download=cfg.DOWNLOAD)
        train(models=cfg.MODEL, dataset_s=dataset_s, dataset_a=dataset_a)
    elif cfg.MODE == 'test':
        dataset = get_dataset(dataset=cfg.DATASET, root_dir=cfg.INPUT_DIR, version=cfg.DATASET_VERSION)
        e = Experiment(models=cfg.MODEL, criteria=cfg.CRITERIA, model_comparison=cfg.MODEL_COMPARISON)
        e(dataset=dataset)
    elif cfg.MODE == 'parse':
        # insert features into the streaming logs
        dataset = get_dataset(dataset=cfg.DATASET, root_dir=cfg.INPUT_DIR, version='server_video')
        for i in range(len(dataset)):
            streaming_video, _ = dataset[i]
            streaming_video.dump2csv()
    else:
        raise ValueError('Invalid mode %s.' % cfg.MODE)

if __name__ == "__main__":
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    main()
