import os
import numpy as np
from osqp import OSQP
from scipy.sparse import vstack
from scipy.sparse import csc_matrix
from pysqoe.models import QoeModel


def osqp_solve_qp(P, q, G=None, h=None, B=None, c=None, initvals=None):
    """
    Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            C * x == b
    using OSQP <https://github.com/oxfordcontrol/osqp>.
    Parameters
    ----------
    P : scipy.sparse.csc_matrix
        Symmetric quadratic-cost matrix.
    q : numpy.array
        Quadratic cost vector.
    G : scipy.sparse.csc_matrix
        Linear inequality constraint matrix.
    h : numpy.array
        Linear inequality constraint vector.
    B : scipy.sparse.csc_matrix, optional
        Linear equality constraint matrix.
    c : numpy.array, optional
        Linear equality constraint vector.
    initvals : numpy.array, optional
        Warm-start guess vector.
    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    Note
    ----
    OSQP requires `P` to be symmetric, and won't check for errors otherwise.
    Check out for this point if you e.g. `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.
    """
    l = -np.inf * np.ones(len(h))
    if B is not None:
        qp_A = vstack([G, B]).tocsc()
        qp_l = np.hstack([l, c])
        qp_u = np.hstack([h, c])
    else:  # no equality constraint
        qp_A = G
        qp_l = l
        qp_u = h
    osqp = OSQP()
    osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False)
    if initvals is not None:
        osqp.warm_start(x=initvals)
    res = osqp.solve()
    if res.info.status_val != osqp.constant('OSQP_SOLVED'):
        print("OSQP exited with status '%s'" % res.info.status)
    return res.x

class KSQI(QoeModel):
    r"""
    This is an implementation of the objective QoE model described in [R1].

    [R1]: Z. Duanmu, K. Zeng, K. Ma, A. Rehman, and Z. Wang, ``A knowledge-driven
          quality-of-experience model for adaptive streaming videos,'' submitted to
          IEEE Trans. Image Processing, 2019.
    [R2]: Z. Duanmu, K. Zeng, K. Ma, A. Rehman, and Z. Wang, ``A quality-of-experience
          index for streaming video,'' IEEE J. Sel. Topics Signal Process., vol. 11,
          no. 1, pp. 154-166, Feb. 2017.
    """
    def __init__(self, num_t=10, num_p=10, lbd=1):
        model_dir = os.path.dirname(os.path.realpath(__file__))
        self.s_model_txt = os.path.join(model_dir, 's_model.txt')
        self.a_model_txt = os.path.join(model_dir, 'a_model.txt')
        self.lbd = lbd
        self.eps = 1e-3
        self.p_init = 80
        self.p_range = 100

        if os.path.isfile(self.s_model_txt):
            self.s_model = np.loadtxt(self.s_model_txt)
        else:
            self.s_model = None

        if os.path.isfile(self.a_model_txt):
            self.a_model = np.loadtxt(self.a_model_txt)
        else:
            self.a_model = None
        
        self.num_p = num_p
        self.num_t = num_t
        self.N_s = self.num_t * self.num_p
        self.N_a = self.num_p * self.num_p
        # bin size of p
        self.p_size = self.p_range / self.num_p
        self.M_s = None
        self.M_a = None

    def __call__(self, streaming_video):
        assert self.s_model is not None, 'Model weights do not exist.'
        assert self.a_model is not None, 'Model weights do not exist.'
        assert self.a_model.shape[0] == self.s_model.shape[0]
        assert self.a_model.shape[0] == self.a_model.shape[1]
        self.num_p = self.s_model.shape[0]
        self.num_t = self.s_model.shape[1]
        self.p_size = self.p_range / self.num_p

        q = []
        p = np.array(streaming_video.data['vmaf'])
        t = np.array(streaming_video.data['rebuffering_duration'])
        p_pre = p[0]
        for i, p_cur in enumerate(p):
            q_i = self._compute_segment_qoe(p_pre, p_cur, t[i], i==0)
            q.append(q_i)
            p_pre = p_cur
        qoe = np.mean(np.asarray(q))
        qoe = np.maximum(np.minimum(qoe, 100), 0)
        return qoe

    def get_s_idx(self, p, t):
        # -eps to quantize 100 into the last bin
        si = int(max((p - self.eps), 0) / self.p_size)
        sj = int(min(max(round(t - self.eps), 0), self.num_t)) - 1
        return si, sj

    def get_a_idx(self, p, delta_p):
        # -eps to quantize 100 into the last bin
        ai = int(max((p - self.eps), 0) / self.p_size)
        # handle \Delta p = 95~100 and -100~-95
        aj = int(min(max(round(delta_p / self.p_size), -self.num_p + 1), self.num_p - 1))
        aj += ai
        aj = min(max(aj, 0), self.num_p - 1)
        return ai, aj

    def _compute_segment_qoe(self, p_pre, p_cur, t, is_first):
        delta_p = p_cur - p_pre
        if is_first:
            # we discount the impact of initial buffering by a factor of 9 according to [R2]
            si, sj = self.get_s_idx(self.p_init, t / 9)
            a = 0
        else:
            si, sj = self.get_s_idx(p_pre, t)
            ai, aj = self.get_a_idx(p_pre, delta_p)
            a = self.a_model[ai, aj]

        # handle stalling duration = 0
        if sj == -1:
            s = 0
        else:
            s = self.s_model[si, sj]

        q = p_cur + s + a
        return q

    # model training
    def train(self, dataset_s, dataset_a):
        """obtain optimal model parameters on dataset_s and dataset_a.
        
        Arguments:
            dataset_s {StreamingDatabase} -- A streaming database used to train rebuffering QoE submodule
            dataset_a {StreamingDatabase} -- A streaming database used to train quality adaptation QoE submodule
        """
        print('Training KSQI...')
        self.s_model = np.zeros((self.num_p, self.num_t))
        self.a_model = np.zeros((self.num_p, self.num_p))
        self._train_s(dataset=dataset_s)
        self._train_a(dataset=dataset_a)

    def _train_s(self, dataset):
        s = self._solve_s(dataset)
        np.savetxt(self.s_model_txt, s, fmt='%03.2f')

    def _solve_s(self, dataset):
        ''' Construct input for quadratic programming and solve for s'''
        P, q = self._construct_obj_s(dataset)
        G, h = self._construct_ineq_s()
        P = csc_matrix(P)
        G = csc_matrix(G)
        s_opt = osqp_solve_qp(P, q, G=G, h=h)
        # prune s
        s_opt[np.abs(s_opt) < 1e-4] = 0
        s = s_opt.reshape((self.num_p, self.num_t))
        return s

    def _construct_obj_s(self, dataset):
        # fidelity term
        self.M_s = len(dataset)
        PF = np.zeros((self.N_s, self.N_s))
        qF = np.zeros(self.N_s)
        for m in range(self.M_s):
            streaming_video, mos = dataset[m]
            # number of chunks
            C = len(streaming_video.data['representation_index'])
            # index in \textbf{s}
            s_idx = []
            # row index and col index in matrix S
            t = streaming_video.data['rebuffering_duration'][0]
            si_mc, sj_mc = self.get_s_idx(self.p_init, t)
            if sj_mc != -1:
                s_idx.append(si_mc * self.num_t + sj_mc)
            # construct qF
            L = C * mos - np.sum(streaming_video.data['mos'])
            for c in range(1, C):
                p_pre = streaming_video.data['mos'][c - 1]
                p_cur = streaming_video.data['mos'][c]
                t = streaming_video.data['rebuffering_duration'][c]
                delta_p = p_cur - p_pre
                # -eps to quantize 100 into the last bin
                ai_mc = int(max((p_pre - self.eps), 0) / self.p_size)
                # handle \Delta p = 95~100 and -100~-95
                aj_mc = int(min(max(round(delta_p / self.p_size), -self.num_p + 1), self.num_p - 1))
                aj_mc += ai_mc
                aj_mc = min(max(aj_mc, 0), self.num_p - 1)
                L -= self.a_model[ai_mc, aj_mc]

                # locate the position of each stalling event in \textbf{s}
                si_mc, sj_mc = self.get_s_idx(p_pre, t)
                if sj_mc != -1:
                    s_idx.append(si_mc * self.num_t + sj_mc)

            # construct the combination of pairs
            for si in s_idx:
                qF[si] -= ((2 * L) / (C ** 2))
                for sj in s_idx:
                    PF[si, sj] += (1 / (C ** 2))

        PF /= (self.M_s * 0.5)
        qF /= self.M_s

        # smoothness term
        PS = np.zeros((self.N_s, self.N_s))
        # it is easier to use index in the matrix form of S
        # index in S with boundaries clipped
        s_idx = [si * self.num_t + sj for si in range(1, self.num_p - 1)
                 for sj in range(1, self.num_t - 1)]
        for s in s_idx:
            PS[s, s] += 8
            PS[s - self.num_t, s - self.num_t] += 1
            PS[s + self.num_t, s + self.num_t] += 1
            PS[s - 1, s - 1] += 1
            PS[s + 1, s + 1] += 1
            PS[s - self.num_t, s + self.num_t] += 1
            PS[s + self.num_t, s - self.num_t] += 1
            PS[s - 1, s + 1] += 1
            PS[s + 1, s - 1] += 1
            PS[s - self.num_t, s] -= 2
            PS[s, s - self.num_t] -= 2
            PS[s + self.num_t, s] -= 2
            PS[s, s + self.num_t] -= 2
            PS[s - 1, s] -= 2
            PS[s, s - 1] -= 2
            PS[s + 1, s] -= 2
            PS[s, s + 1] -= 2

        PS /= (0.5 * (self.num_t - 2) * (self.num_p - 2))
        # combine them to get objective function
        P = PF + self.lbd * PS
        q = qF
        return P, q

    def _construct_ineq_s(self):
        # G0: negativity induced from equality constraints
        G0 = np.zeros((self.num_p, self.N_s))
        for i in range(self.num_p):
            G0[i, i * self.num_t] = 1
        h0 = np.zeros(self.num_p)
        # G1 and h1: monotonicity w.r.t time
        G1_t1 = np.diag(np.ones(self.N_s))
        G1_t2 = np.ones(self.N_s - 1) * (-1)  # Left elements of diagonal are -1.
        G1_t3 = np.diag(G1_t2, k=-1)  # Diagonal matrix shifted to left.
        G1_t4 = G1_t1 + G1_t3
        idx_rm = np.array(np.arange(0, self.N_s))[np.s_[::self.num_t]]
        G1 = np.delete(G1_t4, idx_rm, 0)
        h1 = np.zeros(G1.shape[0])
        # G2 and h2: monotonicity w.r.t quality
        G2_t1 = np.diag(np.ones(self.N_s)) * (-1)
        G2_t2 = np.ones(self.N_s - self.num_t)
        G2_t3 = np.diag(G2_t2, k=self.num_t)
        G2_t4 = G2_t1 + G2_t3
        G2 = G2_t4[:-self.num_t, :]
        h2 = np.zeros(G2.shape[0])
        # G3 and h3: quality preservation
        G3 = -G2
        h3 = np.ones(G2.shape[0]) * self.p_size
        # G4 and h4: monotonicity w.r.t stalling frequency
        an = self.num_t - 1
        s_num = np.arange(an, -0.5, -2)
        sn = int(np.sum(s_num))  # number of constraint for each quality level
        G4_t1 = np.zeros((sn, self.N_s))
        G4_t2 = np.zeros((sn, self.N_s))
        G4_t3 = np.zeros((sn, self.N_s))
        G4 = np.zeros((sn * self.num_p, self.N_s))
        count = 0
        for i in range(1, int(np.floor(self.num_t / 2) + 1)):
            for j in range(i, self.num_t - i + 1):
                G4_t1[count, i - 1] = 1
                G4_t2[count, j - 1] = 1
                G4_t3[count, i + j - 1] = -1
                count += 1
        G4_t4 = G4_t1 + G4_t2 + G4_t3

        for i in range(self.num_p):
            G4[(i * sn):(i + 1) * sn, :] = np.roll(G4_t4, i * self.num_t, axis=1)
        h4 = np.zeros(sn * self.num_p)
        # combine all constraints
        G = np.concatenate((G0, G1, G2, G3, G4), axis=0)
        h = np.concatenate((h0, h1, h2, h3, h4), axis=0)
        return G, h

    def _train_a(self, dataset):
        a = self._solve_a(dataset)
        a = a / 4 # normalize by segment size
        np.savetxt(self.a_model_txt, a, fmt='%03.2f')

    def _solve_a(self, dataset):
        ''' Construct input for quadratic programming and solve for s'''
        P, q = self._construct_obj_a(dataset)
        G, h = self._construct_ineq_a()
        B, c = self._construct_eq_a()
        P = csc_matrix(P)
        G = csc_matrix(G)
        B = csc_matrix(B)
        a_opt = osqp_solve_qp(P, q, G=G, h=h, B=B, c=c)
        # prune a
        a_opt[np.abs(a_opt) < 1e-4] = 0
        a = a_opt.reshape((self.num_p, self.num_p))
        return a

    def _construct_obj_a(self, dataset):
        self.M_a = len(dataset)
        # fidelity term
        PF = np.zeros((self.N_a, self.N_a))
        qF = np.zeros(self.N_a)
        # there is no stalling events in Waterloo SQoE-II dataset
        # we will ignore S hereafter
        for m in range(self.M_a):
            streaming_video, mos = dataset[m]
            # number of chunks
            C = len(streaming_video.data['representation_index'])
            # construct qF
            L = C * mos - np.sum(streaming_video.data['mos'])
            # -eps to quantize 100 into the last bin
            p = streaming_video.data['mos'][0]
            for c in range(1, C):
                delta_p = streaming_video.data['mos'][c] - streaming_video.data['mos'][c - 1]
                ai_m, aj_m = self.get_a_idx(p, delta_p)
                PF[ai_m * self.num_p + aj_m, ai_m * self.num_p + aj_m] += (1 / (C ** 2))
                qF[ai_m * self.num_p + aj_m] += ((-2 * L) / (C ** 2))

        PF /= (self.M_a * 0.5)
        qF /= self.M_a

        # smoothness term
        PS = np.zeros((self.N_a, self.N_a))
        for i in range(1, self.num_p - 1):
            for j in range(1, self.num_p - 1):
                s = i * self.num_p + j
                PS[s, s] += 8
                PS[s - self.num_p - 1, s - self.num_p - 1] += 1
                PS[s + self.num_p + 1, s + self.num_p + 1] += 1
                PS[s - 1, s - 1] += 1
                PS[s + 1, s + 1] += 1
                PS[s - self.num_p - 1, s + self.num_p + 1] += 1
                PS[s + self.num_p + 1, s - self.num_p - 1] += 1
                PS[s - 1, s + 1] += 1
                PS[s + 1, s - 1] += 1
                PS[s - self.num_p - 1, s] -= 2
                PS[s, s - self.num_p - 1] -= 2
                PS[s + self.num_p + 1, s] -= 2
                PS[s, s + self.num_p + 1] -= 2
                PS[s - 1, s] -= 2
                PS[s, s - 1] -= 2
                PS[s + 1, s] -= 2
                PS[s, s + 1] -= 2

        PS /= (0.5 * ((self.num_p - 2) ** 2))
        # combine them to get objective function
        P = PF + self.lbd * PS
        q = qF
        return P, q

    def _construct_ineq_a(self):
        # sign of negative adaptation
        G1 = np.zeros((int((self.N_a - self.num_p) / 2), self.N_a))
        h1 = np.zeros(int((self.N_a - self.num_p) / 2))
        count = 0
        for i in range(self.num_p):
            for j in range(i):
                G1[count, i * self.num_p + j] = 1
                count += 1

        # sign of positive adaptation
        G2 = np.zeros((int((self.N_a - self.num_p) / 2), self.N_a))
        h2 = np.zeros(int((self.N_a - self.num_p) / 2))
        count = 0
        for i in range(self.num_p):
            for j in range(i + 1, self.num_p):
                G2[count, i * self.num_p + j] = -1
                count += 1

        # Weber's law
        num_const = np.arange(1, self.num_p - 1).sum() * 2
        G3 = np.zeros((num_const, self.N_a))
        h3 = np.zeros(num_const)
        count = 0
        for i in range(self.num_p - 1):
            for j in range(self.num_p - 1):
                # the constraints for Delta p = 0 is redundant
                if i != j:
                    G3[count, i * self.num_p + j] = -1
                    G3[count, (i + 1) * self.num_p + (j + 1)] = 1
                    count += 1

        # Monotonicity of adaptation magnitude
        num_const = (self.num_p - 1) * self.num_p
        G4 = np.zeros((num_const, self.N_a))
        h4 = np.zeros(num_const)
        count = 0
        for i in range(self.num_p):
            for j in range(self.num_p - 1):
                G4[count, i * self.num_p + j] = 1
                G4[count, i * self.num_p + (j + 1)] = -1
                count += 1

        # negativity effect
        num_const = np.arange(1, self.num_p).sum()
        G5 = np.zeros((num_const, self.N_a))
        h5 = np.zeros(num_const)
        count = 0
        for i in range(self.num_p):
            for j in range(i + 1, self.num_p):
                G5[count, i * self.num_p + j] = 1
                G5[count, j * self.num_p + i] = 1
                count += 1

        # combine all constraints
        G = np.concatenate((G1, G2, G3, G4, G5), axis=0)
        h = np.hstack([h1, h2, h3, h4, h5])

        # G = np.concatenate((G1, G2, G4, G3), axis=0)
        # h = np.hstack([h1, h2, h4, h3])
        return G, h

    def _construct_eq_a(self):
        # no adaptation
        B = np.zeros((self.num_p, self.N_a))
        for i in range(self.num_p):
            B[i, i + i * self.num_p] = 1
        c = np.zeros(self.num_p)
        return B, c
