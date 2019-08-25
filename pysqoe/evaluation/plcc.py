from pysqoe.evaluation import Criterion
from scipy.stats import pearsonr


class PLCC(Criterion):
    def __init__(self, version='default'):
        super().__init__(version=version)

    def __call__(self, obj_score, sbj_score):
        obj_score = self._nonlinear_mapping(obj_score, sbj_score)
        corr, _ = pearsonr(obj_score, sbj_score)
        return corr
