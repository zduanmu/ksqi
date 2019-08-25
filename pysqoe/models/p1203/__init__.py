import os
import numpy as np
from pysqoe.models import QoeModel


class P1203(QoeModel):
    def __init__(self):
        """
        This is an implementation of the objective QoE model described in [R1].
        For more information, please look at [R2].

        Note that we do not implement the training module for P.1203 because 1) the
        training method is unclear, and 2) there is no publicly available dataset
        that is suitable to train the model.

        [R1]:  W. Robitza, M.-N. Garcia, and A. Raake, ``A modular HTTP adaptive
               streaming QoE model-Candidate for ITU-T P.1203 (`P. NATS'),'' in
               Proc. IEEE Int. Conf. Qual. Multimedia Exp., Erfurt, Germany, 2017,
               pp. 1-6.
        [R2]:  https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.1203.3-201612-S!!PDF-E&type=items
        """
        # P1203.1 parameters
        # 8.1.1 parameters
        self.q1_video = 4.66
        self.q2_video = -0.07
        self.q3_video = 4.06
        # 8.1.2 parameters
        self.u1_video = 72.61
        self.u2_video = 0.32
        # 8.1.3 parameters
        self.t1_video = 30.98
        self.t2_video = 1.29
        self.t3_video = 64.65
        # device parameters
        self.htv1_video = -0.60293
        self.htv2_video = 2.12382
        self.htv3_video = -0.36936
        self.htv4_video = 0.03409
        # mode0 parameters
        self.a1_video = 11.99835
        self.a2_video = -2.99992
        self.a3_video = 41.24751
        self.a4_video = 0.13183
        # P1203.2 parameters
        self.a1_audio = 100
        self.a2_audio_dict = {
            'mpeg_l2': -0.02,
            'ac3': -0.03,
            'aac_lc': -0.05,
            'he_aac': -0.11,
            'default': -0.03
        }
        self.a3_audio_dict = {
            'mpeg_l2': 15.48,
            'ac3': 15.70,
            'aac_lc': 14.60,
            'he_aac': 20.06,
            'default': 15.70
        }
        self.a2_audio = None
        self.a3_audio = None
        self.mos_max = 4.9
        self.mos_min = 1.04
        # P1203 parameters
        # weights for w_buffi
        self.c_ref7 = 0.48412879
        self.c_ref8 = 10.0
        # weights for neg_bias
        self.c1_nb = 1.87403625
        self.c2_nb = 7.85416481
        self.c23_nb = 0.01853820
        # weights for O.34
        self.av1_o34 = -0.00069084
        self.av2_o34 = 0.15374283
        self.av3_o34 = 0.97153861
        self.av4_o34 = 0.02461776
        # weights for O.35
        self.t1_o35 = 0.00666620027943848
        self.t2_o35 = 0.0000404018840273729
        self.t3_o35 = 0.156497800436237
        self.t4_o35 = 0.143179744942738
        self.t5_o35 = 0.0238641564518876
        self.c1_o35 = 0.67756080
        self.c2_o35 = -8.05533303
        self.c3_o35 = 0.17332553
        self.c4_o35 = -0.01035647
        # weights for O.46
        self.s1_o46 = 9.35158684
        self.s2_o46 = 0.91890815
        self.s3_o46 = 11.0567558
        # load random forest
        self._load_randomforest()

    def __call__(self, streaming_video):
        o_21 = self._compute_p12032(streaming_video)
        o_22 = self._compute_p12031(streaming_video)
        # compute P.1203.3
        # total duration
        chunk_dur = np.array(streaming_video.data['chunk_duration'])
        rb_dur = np.array(streaming_video.data['rebuffering_duration'])
        T = np.sum(chunk_dur)
        t = np.cumsum(chunk_dur)
        t_rb_start = t - chunk_dur[0]
        # numStalls in [R2]
        num_stalls = np.count_nonzero(rb_dur)
        # totalBuffLen in [R2]
        w_buff = self.c_ref7 + (1 - self.c_ref7) * np.exp(t_rb_start * np.log(0.5) / (-self.c_ref8))
        total_buff_len = np.sum(rb_dur * w_buff)
        # avgBuffLen in [R2] (the name is confusing, it actually represents for
        # average interval between stalling events)
        avg_buff_len = 0
        if num_stalls > 1:
            t_rb_s = np.delete(t_rb_start, np.where(rb_dur == 0))
            avg_buff_len = np.mean(t_rb_s[1:] - t_rb_s[:-1])
        # O.34
        o_34 = np.maximum(np.minimum(self.av1_o34 + self.av2_o34 * o_21 + \
                          self.av3_o34 * o_22 + self.av4_o34 * o_21 * o_22, 5), 1)
        # 8.1.2
        # negativeBias in [R2]
        w_diff = self.c1_nb + (1 - self.c1_nb) * np.exp(-(T - t) * (np.log(0.5) / (-self.c2_nb)))
        o_34_diff = o_34 * w_diff
        # negPerc in [R2]
        neg_perc = np.percentile(o_34_diff, 10)
        # negBias in [R2]
        neg_bias = np.maximum(-neg_perc, 0) * self.c23_nb
        # vidQualSpread in [R2]
        vid_qual_spread = np.max(o_22) - np.min(o_22)
        # vidQualChangeRate in [R2]
        vid_qual_change_rate = np.sum(np.abs(o_22[1:] - o_22[:-1]) > 0.2) / T
        # qDirChangesTot in [R2]
        ma_filter = np.ones(5) / 5
        o_22_pad = np.pad(o_22, 2, 'edge')
        o_22_ma = np.convolve(o_22_pad, ma_filter, 'valid')
        q = 3
        qc = []
        for q in range(3, o_22_ma.size, 3):
            dif = o_22_ma[q - 3] - o_22_ma[q]
            if dif > 0.2:
                qc.append(1)
            elif dif > -0.2 and dif <= 0.2:
                qc.append(0)
            else:
                qc.append(-1)
        qc_zero_rm = list(filter(lambda a: a != 0, qc))
        q_dir_changes_tot = 0
        if qc_zero_rm:
            q_dir_changes_tot += 1
        for i in range(1, len(qc_zero_rm)):
            if qc_zero_rm[i] != qc_zero_rm[i - 1]:
                q_dir_changes_tot += 1
        # qDirChangesLongest in [R2]
        qc_len = []
        distances = []
        for i, qc_i in enumerate(qc):
            if qc_i != 0:
                if qc_len:
                    if qc_len[-1][1] != qc_i:
                        qc_len.append([i, qc_i])
                else:
                    qc_len.append([i, qc_i])
        if qc_len and len(qc) > 1:
            qc_len.insert(0, [0, 0])
            qc_len.append([len(qc), 0])
            for i in range(1, len(qc)):
                distances.append(qc[i] - qc[i - 1])
            q_dir_changes_longest = max(distances) * 3.0
        else:
            q_dir_changes_longest = len(o_22)

        # 8.1.3 parameters related to machine learning module
        media_length = T
        rebuff_count = np.count_nonzero(rb_dur[1:])
        init_buff_dur = rb_dur[0]
        stall_dur = np.sum(rb_dur[1:]) + init_buff_dur / 3.0
        rebuff_freq = rebuff_count / media_length
        stall_ratio = stall_dur / media_length
        if num_stalls > 0:
            time_last_rebuff_to_end = media_length - t_rb_start[-1]
        else:
            time_last_rebuff_to_end = 0

        average_pv_score_one = np.mean(o_22[:round(o_22.size / 3)])
        average_pv_score_two = np.mean(o_22[round(o_22.size / 3):(2 * round(o_22.size / 3))])
        average_pv_score_three = np.mean(o_22[(2 * round(o_22.size / 3)):])
        one_percentile_pv_score = np.percentile(o_22, 1)
        five_percentile_pv_score = np.percentile(o_22, 5)
        ten_percentile_pv_score = np.percentile(o_22, 10)
        average_pa_score_one = np.mean(o_21[:round(o_21.size / 2)])
        average_pa_score_two = np.mean(o_21[round(o_21.size / 2):])

        # O.35 and its dependencies 8-2 -- 8-11 in [R2]
        # O.35 baseline
        w1 = self.t1_o35 + self.t2_o35 * np.exp(t / (T / self.t3_o35))
        w2 = self.t4_o35 - self.t5_o35 * o_34
        o_35_baseline = np.sum(w1 * w2 * o_34) / np.sum(w1 * w2)
        # negBias has been computed
        # oscComp in [R2]
        q_diff = np.maximum(0.1 + np.log10(vid_qual_spread + 0.01), 0)
        osc_test = (q_dir_changes_tot / T) < 0.25 and q_dir_changes_longest < 30
        if osc_test:
            osc_comp = q_diff * np.exp(np.minimum(self.c1_o35 * q_dir_changes_tot + self.c2_o35, 1.5))
        else:
            osc_comp = 0

        # adaptComp
        adapt_test = (q_dir_changes_tot / T) < 0.25
        if adapt_test:
            adapt_comp = self.c3_o35 * vid_qual_spread * vid_qual_change_rate + self.c4_o35
        else:
            adapt_comp = 0

        # O.35
        o_35 = o_35_baseline - neg_bias - osc_comp - adapt_comp
        # eq. 8-13
        si = np.exp(-num_stalls / self.s1_o46) * np.exp(-(total_buff_len / T) / self.s2_o46) * \
             np.exp(-(avg_buff_len / T) / self.s3_o46)
        input_feature = np.array([rebuff_count, stall_dur, rebuff_freq, stall_ratio,
                                  time_last_rebuff_to_end, average_pv_score_one, average_pv_score_two,
                                  average_pv_score_three, one_percentile_pv_score, five_percentile_pv_score,
                                  ten_percentile_pv_score, average_pa_score_one, average_pa_score_two, media_length])
        rf_prediction = self._run_randomforest(input_feature)
        # eq. 8-12
        o_46 = 0.75 * (1 - (o_35 - 1) * si) + 0.25 * rf_prediction
        # eq. 8-14
        y = 0.02833052 + 0.98117059 * o_46
        return y

    # P.1203.1 module
    def _compute_p12031(self, streaming_video):
        vb = np.array(streaming_video.data['video_bitrate'])
        # coding_resolution, display_resolution, fps
        fps = np.array(streaming_video.data['framerate'])
        coding_resolution = np.array(streaming_video.data['width']) * \
                            np.array(streaming_video.data['height'])
        display_width, display_height = streaming_video.get_display_resolution()
        display_resolution = np.ones(coding_resolution.size) * (display_width * display_height)
        y = np.zeros(vb.size)
        quant = self._mode0(vb, coding_resolution, fps)
        mos_q = self.q1_video + self.q2_video * np.exp(self.q3_video * quant)
        mos_q = np.maximum(np.minimum(mos_q, 5), 1)
        dq = 100 - self._get_r_from_mos(mos_q)
        dq = np.maximum(np.minimum(dq, 100), 0)
        # 8.1.2 upscaling degradation
        scale_factor = np.maximum(display_resolution / coding_resolution, 1)
        du = self.u1_video * np.log10(self.u2_video * (scale_factor - 1) + 1)
        du = np.maximum(np.minimum(du, 100), 0)
        # 8.1.3 temporal degradation
        dt1 = 100 * (self.t1_video - self.t2_video * fps) / (self.t3_video + fps)
        dt2 = dq * (self.t1_video - self.t2_video * fps) / (self.t3_video + fps)
        dt3 = du * (self.t1_video - self.t2_video * fps) / (self.t3_video + fps)
        dt = np.zeros(dq.size)
        dt[fps < 24] = dt1[fps < 24] - dt2[fps < 24] - dt3[fps < 24]
        dt = np.maximum(np.minimum(dt, 100), 0)
        # 8.1.4 integration
        d = np.maximum(np.minimum(dq + du + dt, 100), 0)
        q = 100 - d
        y = self._get_mos_from_r(q)
        for i in range(dq.size):
            if du[i] == 0 and dt[i] == 0:
                y[i] = mos_q[i]

        if streaming_video.device == 'phone':
            mos_qh = self.htv1_video + self.htv2_video * y + self.htv3_video * y ** 2 + self.htv4_video * y ** 3
            y = np.maximum(np.minimum(mos_qh, 5), 1)
        return y

    def _mode0(self, bitrate, coding_resolution, fps):
        bpp = bitrate / (coding_resolution * fps)
        quant = self.a1_video + self.a2_video * np.log(self.a3_video + \
                np.log(bitrate) + np.log(bitrate * bpp + self.a4_video))
        return quant

    def _get_r_from_mos(self, mos):
        x = np.minimum(mos, 4.5)
        q = np.zeros(mos.size)
        h = np.zeros(mos.size)

        for i, m in enumerate(x):
            if m > 2.7505:
                h[i] = (1.0 / 3.0) * (np.pi - np.arctan(15.0 * np.sqrt(-903522.0 + \
                        1113960.0 * m - 202500.0 * m * m) / (6750.0 * m - 18566.0)))
            else:
                h[i] = (1.0 / 3.0) * (np.arctan(15.0 * np.sqrt(-903522.0 + \
                        1113960.0 * m - 202500.0 * m * m) / (6750.0 * m - 18566.0)))
        
        q = 20.0 * (8.0 - np.sqrt(226.0) * np.cos(h + np.pi / 3.0)) / 3.0
        return q

    # P.1203.2 module
    def _compute_p12032(self, streaming_video):
        K = len(streaming_video.data['representation_index'])
        if 'audio_bitrate' not in streaming_video.data:
            ab = np.zeros(K)
        else:
            ab = np.array(streaming_video.data['audio_bitrate'])

        if 'audio_codec' not in streaming_video.data:
            self.a2_audio = self.a2_audio_dict['default']
            self.a3_audio = self.a3_audio_dict['default']
        else:
            self.a2_audio = np.zeros(K)
            self.a3_audio = np.zeros(K)
            for k in range(K):
                self.a2_audio[k] = self.a2_audio_dict[streaming_video.data['audio_codec'][k]]
                self.a3_audio[k] = self.a3_audio_dict[streaming_video.data['audio_codec'][k]]

        q_cod_a = self.a1_audio * np.exp(self.a2_audio * ab) + self.a3_audio
        q_a = 100 - q_cod_a
        q21 = self._get_mos_from_r(q_a)
        return q21

    def _get_mos_from_r(self, qs):
        mos = np.zeros(qs.size)
        for i, q in enumerate(qs):
            if q > 0 and q < 100:
                mos[i] = self.mos_min + (self.mos_max - self.mos_min) / 100 * q + q * (q - 60) * (100 - q) * 7e-6
            elif q >= 100:
                mos[i] = self.mos_max
            else:
                mos[i] = self.mos_min
        return mos

    # P.1203 module
    def _load_randomforest(self):
        model_dir = os.path.dirname(os.path.realpath(__file__))
        files = [os.path.join(model_dir, 'tree%d.csv' % i) for i in range(1, 21)]
        self.forest = [np.loadtxt(f, delimiter=',') for f in files]

    def _run_randomforest(self, x):
        """random forest regression model in P.1203.3
        
        Arguments:
            x {14x1 nparray} -- feature vector
        """
        ys = np.zeros(len(self.forest))
        for i, tree in enumerate(self.forest):
            cur_node = 0
            while True:
                if tree[cur_node, 1] == -1:
                    break

                cur_value = x[int(tree[cur_node, 1])]
                cur_threshold = tree[cur_node, 2]
                if cur_value < cur_threshold:
                    cur_node = int(tree[cur_node, 3])
                else:
                    cur_node = int(tree[cur_node, 4])

            ys[i] = tree[cur_node, 2]
        
        y = np.mean(ys)
        return y
