import os.path as osp
from easydict import EasyDict as edict


__C = edict()
# cfg is treated as a global variable. 
# Need add "from config import cfg" for each file to use cfg.
cfg = __C

# Mode
__C.MODE = 'test'
# Root directory
__C.ROOT_DIR = osp.abspath(osp.join(osp.abspath('')))
# Model directory
__C.MODEL_DIR = osp.abspath(osp.join(cfg.ROOT_DIR, 'models'))
# dataset parameters
__C.DATASET = ''
__C.DOWNLOAD = True
__C.DATASET_VERSION = 'feature'
# Dataset directory
__C.INPUT_DIR = ''
# Path to input dataset
__C.MODEL = ''
# FFmpeg directory
__C.CRITERIA = 'PLCC:SRCC:KRCC'
# Path to the video elements
__C.MODEL_COMPARISON = ''
# training dataset
__C.TRAINING_DATASET_S = 'WaterlooSQoE-I'
__C.TRAINING_DATASET_A = 'WaterlooSQoE-II'
__C.DATASET_S_ROOT = 'data/WaterlooSQoE-I'
__C.DATASET_A_ROOT = 'data/WaterlooSQoE-II'

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            # print(subkey)
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
