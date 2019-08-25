import numpy as np
from itertools import combinations
from pysqoe.evaluation import ModelComparison


class gMAD(ModelComparison):
    def __init__(self):
        self.name = 'gMAD'

    def __call__(self, score_dict, sbj_score, out):
        """ This is an implementation of the model comparison algorithm described in [R1].
        We made two modifications to the original algorithms.
            1. We set k = 1
            2. Instead of searching for the most dissimilar pairs (by attacker) on a constant quality
               contour (by defender), we find the pair on which the two models mostly disagree with each
               other in the sense of KL-divergence.

        Arguments:
            score_dict {dict} -- a dictionary stores quality scores of all QoE models
            sbj_score {nparray} -- a numpy array stores mean opinion scores, not used in this class
            out {str} -- output file stores gMAD pairs. Row: attacker. Column: defender.
        """
        s_dict = score_dict.copy()
        s_dict.pop('streaming_log', None)
        row = [self.name] + [key for key, _ in s_dict.items()]
        row = ','.join(row)
        self._record(row=row, out=out, mode='w')
        for model1, x1 in s_dict.items(): 
            performance = []
            for _, x2 in s_dict.items():
                y1, y2 = gMAD.pair_comparison(x1=np.array(x1), x2=np.array(x2))
                if y1 == -1:
                    performance += ['None:None']
                else:
                    performance += ['%s:%s' % (score_dict['streaming_log'][y1], score_dict['streaming_log'][y2])]
            row = [model1] + performance
            row = ','.join(row)
            self._record(row=row, out=out) 

    @staticmethod
    def pair_comparison(x1, x2):
        """
        Arguments:
            x1 {nparray} -- attacker scores
            x2 {nparray} -- defender scores
        
        Returns:
            output {str} -- '%d:%d' % (video_index_1, video_index_2)
        """
        if np.array_equal(x1, x2):
            return -1, -1
        
        kld_max = -1
        idx1 = -1
        idx2 = -1
        for ((i, x1_1), (j, x1_2)), x2p in zip(combinations(enumerate(x1), 2), combinations(x2, 2)):
            x1p = np.array([x1_1, x1_2])
            ps = np.exp(x1p) / np.sum(np.exp(x1p), axis=0)
            qs = np.exp(x2p) / np.sum(np.exp(x2p), axis=0)
            kld = np.sum(np.where(ps != 0, ps * np.log(ps / qs), 0))
            if kld > kld_max:
                kld_max = kld
                idx1 = i
                idx2 = j
        print(idx1, idx2, kld)
        return idx1, idx2

# unit test
if __name__ == "__main__":
    x0 = ['1.csv', '2.csv', '3.csv', '4.csv', '5.csv']
    x1 = np.array([50, 60, 70, 80, 50])
    x2 = np.array([40, 30, 60, 10, 50])
    score_dict = {'x1': x1, 'x2': x2, 'streaming_log': x0}
    gmad = gMAD()
    gmad(score_dict=score_dict, sbj_score=None, out='tmp1.csv')
    print("========================")
    y1 = x2
    y2 = x1
    score_dict = {'y1': y1, 'y2': y2, 'streaming_log': x0}
    gmad(score_dict=score_dict, sbj_score=None, out='tmp2.csv')
