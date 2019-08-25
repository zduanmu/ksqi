import argparse
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from config import cfg, cfg_from_list
from pysqoe.datasets import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Plot Database Statistics')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    arg = parser.parse_args()
    return arg

def extract_stats(dataset):
    b = []
    a = []
    r = []
    for i in range(len(dataset)):
        streaming_video, _ = dataset[i]
        b.extend(streaming_video.data['vmaf'])
        a.extend(np.array(streaming_video.data['vmaf'][1:]) - 
                 np.array(streaming_video.data['vmaf'][:-1]))
        r.extend(streaming_video.data['rebuffering_duration'])
    b = np.asarray(b)
    a = np.asarray(a)
    r = np.asarray(r)
    return b, a, r

def main():
    del matplotlib.font_manager.weight_dict['roman']
    matplotlib.font_manager._rebuild()

    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams.update({'font.weight': 'light'})
    plt.rcParams.update({'axes.labelweight': 'light'})

    colors = [(126 / 256, 100 / 256, 158 / 256),
            (49  / 256, 133 / 256, 155 / 256),
            (120 / 256, 148 / 256, 64  / 256),
            (255 / 256, 217 / 256, 101 / 256),
            (234 / 256, 112 / 256, 13  / 256),
            (217 / 256, 149 / 256, 143 / 256),
            (255 / 256, 125 / 256, 223 / 256),
            (116 / 256, 252 / 256, 143 / 256),
            (139 / 256, 149 / 256, 138 / 256)]

    dashes_list = [(), (5,2,20,2), (1,1,1,1), (8,2), (4,2,1,2), (5,2),
                (), (5,2,20,2), (1,1,1,1)]
    # statistics for WaterlooSQoE-I
    dataset = get_dataset(dataset='WaterlooSQoE-I',   root_dir='/media/zduanmu/Database/Research/WaterlooSQoE-I')
    w1_b, w1_a, w1_r = extract_stats(dataset)
    # statistics for WaterlooSQoE-II
    dataset = get_dataset(dataset='WaterlooSQoE-II',  root_dir='/media/zduanmu/Database/Research/WaterlooSQoE-II')
    w2_b, w2_a, w2_r = extract_stats(dataset)
    # statistics for WaterlooSQoE-III
    dataset = get_dataset(dataset='WaterlooSQoE-III', root_dir='/media/zduanmu/Database/Research/WaterlooSQoE-III')
    w3_b, w3_a, w3_r = extract_stats(dataset)
    # statistics for WaterlooSQoE-IV
    dataset = get_dataset(dataset='WaterlooSQoE-IV',  root_dir='/media/zduanmu/Database/Research/WaterlooSQoE-IV')
    w4_b, w4_a, w4_r = extract_stats(dataset)
    # statistics for LIVE-NFLX-I
    dataset = get_dataset(dataset='LIVE-NFLX-I',      root_dir='/media/zduanmu/Database/Research/LIVE-NFLX-I')
    l1_b, l1_a, l1_r = extract_stats(dataset)
    # statistics for LIVE-NFLX-I
    dataset = get_dataset(dataset='LIVE-NFLX-II',     root_dir='/media/zduanmu/Database/Research/LIVE-NFLX-II')
    l2_b, l2_a, l2_r = extract_stats(dataset)

    sns.kdeplot(w1_b, color=colors[0], dashes=dashes_list[0],
			    linewidth=2, label='WaterlooSQoE-I')
    sns.kdeplot(w2_b, color=colors[1], dashes=dashes_list[1],
			    linewidth=2, label='WaterlooSQoE-II')
    sns.kdeplot(w3_b, color=colors[2], dashes=dashes_list[2],
			    linewidth=2, label='WaterlooSQoE-III')
    sns.kdeplot(w4_b, color=colors[3], dashes=dashes_list[3],
			    linewidth=2, label='WaterlooSQoE-IV')
    sns.kdeplot(l1_b, color=colors[4], dashes=dashes_list[4],
			    linewidth=2, label='LIVE-NFLX-I')
    ax = sns.kdeplot(l2_b, color=colors[5], dashes=dashes_list[5],
			    linewidth=2, label='LIVE-NFLX-II')
    plt.xlim((0, 100))
    plt.xlabel('VMAF')
    plt.ylabel('Probability density function')
    ax.legend(loc=1, frameon=False, fontsize=14)
    plt.tight_layout()
    plt.show()

    x = np.arange(0, 10, 0.1)
    y = np.zeros(x.size)
    y[0] = 1680 / 1800
    y[50] = 120 / 1800
    plt.plot(x, y, color=colors[0], dashes=dashes_list[0],
             linewidth=2, label='WaterlooSQoE-I')
    # sns.kdeplot(w1_r, color=colors[0], dashes=dashes_list[0],
	# 		    linewidth=2, bw=0.00002, label='Waterloo-SQoE-I')
    x = np.arange(0, 10, 0.1)
    y = np.zeros(x.size)
    y[0] = 1
    plt.plot(x, y, color=colors[1], dashes=dashes_list[1],
             linewidth=2, label='WaterlooSQoE-II')
    # sns.kdeplot(w2_r, color=colors[1], dashes=dashes_list[1],
	# 		    linewidth=2, bw=0.00002, label='Waterloo-SQoE-II')
    sns.kdeplot(w3_r, color=colors[2], dashes=dashes_list[2],
			    linewidth=2, bw=0.02, label='WaterlooSQoE-III')
    sns.kdeplot(w4_r, color=colors[3], dashes=dashes_list[3],
			    linewidth=2, bw=0.02, label='WaterlooSQoE-IV')
    sns.kdeplot(l1_r, color=colors[4], dashes=dashes_list[4],
			    linewidth=2, bw=0.02, label='LIVE-NFLX-I')
    ax = sns.kdeplot(l2_r, color=colors[5], dashes=dashes_list[5],
			    linewidth=2, bw=0.02, label='LIVE-NFLX-II')
    plt.xlim((0, 10))
    plt.xlabel('Rebuffering duration')
    plt.ylabel('Probability density function')
    ax.legend(loc=1, frameon=False, fontsize=14)
    plt.tight_layout()
    plt.show()

    sns.kdeplot(w1_a, color=colors[0], dashes=dashes_list[0],
			    linewidth=2, label='WaterlooSQoE-I')
    sns.kdeplot(w2_a, color=colors[1], dashes=dashes_list[1],
			    linewidth=2, label='WaterlooSQoE-II')
    sns.kdeplot(w3_a, color=colors[2], dashes=dashes_list[2],
			    linewidth=2, label='WaterlooSQoE-III')
    sns.kdeplot(w4_a, color=colors[3], dashes=dashes_list[3],
			    linewidth=2, label='WaterlooSQoE-IV')
    sns.kdeplot(l1_a, color=colors[4], dashes=dashes_list[4],
			    linewidth=2, label='LIVE-NFLX-I')
    ax = sns.kdeplot(l2_a, color=colors[5], dashes=dashes_list[5],
			    linewidth=2, label='LIVE-NFLX-II')
    plt.xlim((-100, 100))
    plt.xlabel('VMAF adaptation magnitude')
    plt.ylabel('Probability density function')
    ax.legend(loc=1, frameon=False, fontsize=14)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    main()