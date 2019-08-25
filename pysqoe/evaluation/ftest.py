import numpy as np
from scipy.stats import f
from pysqoe.evaluation import ModelComparison, LogisticHamid, Logistic5


class FTest(ModelComparison):
    def __init__(self):
        self.name = 'F-Test'

    def __call__(self, score_dict, sbj_score, out):
        s_dict = score_dict.copy()
        s_dict.pop('streaming_log', None)
        row = [self.name] + [key for key, _ in s_dict.items()]
        row = ','.join(row)
        self._record(row=row, out=out, mode='w')
        for model1, x1 in s_dict.items():
            performance = [FTest.pair_comparison(x1=np.array(x1), x2=np.array(x2), sbj_score=sbj_score)
                                for _, x2 in s_dict.items()]
            row = [model1] + performance
            row = ','.join(row)
            self._record(row=row, out=out) 
        
    @staticmethod
    def pair_comparison(x1, x2, sbj_score):
        fun1 = Logistic5(x=x1, y=sbj_score)
        fun2 = Logistic5(x=x2, y=sbj_score)
        y1 = fun1(x=x1)
        y2 = fun2(x=x2)
        res1 = y1 - sbj_score
        res2 = y2 - sbj_score
        h = FTest.vartest(x1=res1, x2=res2)
        return h

    @staticmethod
    def vartest(x1, x2):
        var_y1 = np.var(x1)
        var_y2 = np.var(x2)
        df1 = len(x1) - 1
        df2 = len(x2) - 1
        alpha = 0.05
        flag = var_y1 > var_y2

        if flag:
            F = var_y1 / var_y2
            p_value = 1 - f.cdf(F, df1, df2)
            if p_value >= alpha:
                h = '-'
            else:
                h = '0'
        else:
            F = var_y2 / var_y1
            p_value = 1 - f.cdf(F, df2, df1)
            if p_value >= alpha:
                h = '-'
            else:
                h = '1'
        return h
