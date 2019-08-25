import os
import numpy as np
import pandas as pd
from pysqoe.models import get_qoe_model
from pysqoe.evaluation import get_criterion, get_comparison_method


class Experiment(object):
    def __init__(self, models, criteria, model_comparison=None, plot=True):
        self.model_names = models.split(':')
        self.models = [get_qoe_model(model_name) for model_name in self.model_names]
        self.result_dir = os.path.abspath(os.path.join(os.path.abspath(''), 'results'))
        self.crit_names = [] if criteria == '' else criteria.split(':')
        self.criteria = [get_criterion(crit_name) for crit_name in self.crit_names]
        self.comparison_names = [] if model_comparison == '' else model_comparison.split(':')
        self.comparison_methods = [get_comparison_method(mc_method) for mc_method in self.comparison_names]

    def __call__(self, dataset):
        # create output folders
        result_dir = os.path.join(self.result_dir, dataset.name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        score_out = os.path.join(result_dir, 'scores.csv')
        criteria_out = os.path.join(result_dir, 'performance.csv')
        mc_out = [os.path.join(result_dir, '%s.csv' % n) for n in self.comparison_names]
        
        # perform objective QoE assessment
        row = ['streaming_log'] + self.model_names
        row = ','.join(row)
        self._record(row=row, out=score_out, mode='w')
        sbj_score = []
        for i in range(len(dataset)):
            streaming_video, mos = dataset[i]
            sbj_score.append(mos)
            # compute objective QoE
            obj_score = [str(np.around(model(streaming_video), decimals=3)) for model in self.models]
            # record results to csv
            row = [streaming_video.get_video_name()] + obj_score
            row = ','.join(row)
            self._record(row=row, out=score_out)
        print('The objective QoE scores are recorded in %s.' % score_out)
        df = pd.read_csv(score_out)
        score_dict = df.to_dict(orient='list')

        # perform objective QoE model evaluation
        row = ['qoe_model'] + self.crit_names
        row = ','.join(row)
        self._record(row=row, out=criteria_out, mode='w')
        for model in self.model_names:
            performance = [str(np.around(criterion(obj_score=score_dict[model], sbj_score=sbj_score),
                           decimals=3)) for criterion in self.criteria]
            row = [model] + performance
            row = ','.join(row)
            self._record(row=row, out=criteria_out)
        if self.criteria:
            print('The performance of the objective QoE models is recorded in %s.' % criteria_out)

        # perform model comparison
        for mc_method, mc_file in zip(self.comparison_methods, mc_out):
            mc_method(score_dict=score_dict, sbj_score=sbj_score, out=mc_file)
        if self.comparison_methods:
            print('The model comparison results are recorded in %s.' % result_dir)

        print('Testing is completed.')

    def _record(self, row, out, mode='a+'):
        if out is not None:
            with open(file=out, mode=mode) as f:
                f.write(row + '\n')
