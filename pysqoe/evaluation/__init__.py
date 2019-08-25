import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


def get_criterion(criterion_name, version='default'):
    if criterion_name == 'PLCC':
        criterion = PLCC(version=version)
    elif criterion_name == 'SRCC':
        criterion = SRCC(version=version)
    elif criterion_name == 'KRCC':
        criterion = KRCC(version=version)
    else:
        raise NotImplementedError('Invalid criterion %s' % criterion_name)

    return criterion

class Criterion(object):
    __valid_versions = ['default', 'logistic', 'hamid']
    def __init__(self, version='default'):
        assert version in self.__valid_versions
        self.version = version

    def __call__(self, obj_score, sbj_score):
        pass

    def _nonlinear_mapping(self, obj_score, sbj_score):
        if self.version is 'default':
            scaled_score = obj_score
        elif self.version is 'logistic':
            func = Logistic5(x=obj_score, y=sbj_score)
            scaled_score = func(x=obj_score)
        elif self.version is 'hamid':
            func = LogisticHamid(x=obj_score, y=sbj_score)
            scaled_score = func(x=obj_score)

        return scaled_score

def get_comparison_method(comparison_method):
    if comparison_method == 'F-Test':
        cm = FTest()
    elif comparison_method == 'gMAD':
        cm = gMAD()
    else:
        raise NotImplementedError('Invalid criterion %s' % comparison_method)

    return cm

class ModelComparison(object):
    def __init__(self):
        pass

    def __call__(self, score_dict, out):
        pass

    def _record(self, row, out, mode='a+'):
        if out is not None:
            with open(file=out, mode=mode) as f:
                f.write(row + '\n')

def logistic5(x, a, b, c, s):
    y = (a - b) / (1 + np.exp(-((x - c) / abs(s)))) + b
    return y

class Logistic5():
    # train
    def __init__(self, x, y):
        try:
            popt, _ = curve_fit(logistic5, x, y)
            self.a = popt[0]
            self.b = popt[1]
            self.c = popt[2]
            self.s = popt[3]
        except RuntimeError:
            self.a = np.max(y)
            self.b = np.min(y)
            self.c = np.mean(x)
            self.s = 0.5

    # inference
    def __call__(self, x):
        y = logistic5(x, self.a, self.b, self.c, self.s)
        return y

def logistic_hamid(x, a, b, c, d, e):
    x_ = np.array(x)
    tmp = 0.5 - 1 / (1 + np.exp(b * (x_ - c)))
    y = a * tmp + d + e * x_
    return y

class LogisticHamid():
    # train
    def __init__(self, x, y):
        try:
            popt, _ = curve_fit(logistic_hamid, x, y)
            self.a = popt[0]
            self.b = popt[1]
            self.c = popt[2]
            self.d = popt[3]
            self.e = popt[4]
        except RuntimeError:
            corr, _ = pearsonr(x, y)
            if corr > 0:
                self.c = np.mean(x)
                self.a = np.abs(np.max(y) - np.min(y))
                self.d = np.mean(y)
                self.b = 1 / np.std(x)
                self.e = 1
            else:
                self.c = np.mean(x)
                self.a = - np.abs(np.max(y) - np.min(y))
                self.d = np.mean(y)
                self.b = 1 / np.std(x)
                self.e = 1

    # inference
    def __call__(self, x):
        y = logistic_hamid(x, self.a, self.b, self.c, self.d, self.e)
        return y

from pysqoe.evaluation.plcc import PLCC
from pysqoe.evaluation.srcc import SRCC
from pysqoe.evaluation.krcc import KRCC
from pysqoe.evaluation.ftest import FTest
from pysqoe.evaluation.gmad import gMAD
