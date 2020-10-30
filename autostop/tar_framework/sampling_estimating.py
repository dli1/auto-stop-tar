# coding=utf-8

import math
import datetime
import itertools
import numpy as np
from collections import defaultdict
from autostop.tar_framework.utils import *


def constant_one():
    return itertools.repeat(1).__next__()


class HorvitzThompson(object):
    """
    With-replacement sample, Horvitz-Thompson estimatation.
    """
    def __init__(self):
        self.N = 0
        self.dtype = np.float128
        self.complete_dids = None
        self.complete_labels = None
        self.cumulated_prod_1st_order = None
        self.cumulated_prod_2nd_order = None

        self.dist = None

    def init(self, complete_dids, complete_labels):
        self.N = len(complete_dids)

        self.complete_dids = complete_dids
        self.complete_labels = np.array(complete_labels).reshape((1, self.N))
        self.cumulated_prod_1st_order = np.zeros((1, self.N), dtype=self.dtype)
        self.cumulated_prod_2nd_order = np.zeros((self.N, self.N), dtype=self.dtype)
        return

    def update_distribution(self, **kwargs):
        """Make sampling distribution"""
        raise NotImplementedError

    def _reorder_dist(self, ranked_dids, dist):
        did_prob_dct = dict(zip(ranked_dids, dist))
        new_dist = []
        for did in self.complete_dids:
            new_dist.append(did_prob_dct[did])
        new_dist = np.array(new_dist, dtype=self.dtype).reshape((1, -1))
        return new_dist

    def _mask_sampled_dids(self, sampled_state):
        first_order_mask = []
        for i, did in enumerate(self.complete_dids):
            if sampled_state[did] is True:
                first_order_mask.append(1)
            else:
                first_order_mask.append(0)

        # first_order_mask = np.zeros(self.N)  # this is slower
        # for did in self.sampled_dids:
        #     i = self.complete_dids.index(did)
        #     first_order_mask[i] = 1

        first_order_mask = np.array(first_order_mask).reshape((1, -1))
        M = np.tile(first_order_mask, (self.N, 1))
        MT = M.T
        second_order_mask = M*MT

        return first_order_mask, second_order_mask

    def _sample(self, ranked_dids, dist, n, replace):
        """Sample n items"""
        assert len(ranked_dids) == len(dist), 'dids len != dist len ({} != {})'.format(len(ranked_dids), len(dist))
        return np.random.choice(a=ranked_dids, size=n, replace=replace, p=dist)  # returned type is the same with ranked_dids

    def _update(self, sampled_dids, ranked_dids, dist, stopping_condition):
        if sampled_dids is None:
            return
        
        n = len(sampled_dids)

        # update first order and second order inclusion probability
        dist = self._reorder_dist(ranked_dids, dist)
        self.cumulated_prod_1st_order += n * np.log(1.0 - dist)
        if stopping_condition == 'strict1' or stopping_condition == 'strict2':
            M = np.tile(dist, (self.N, 1))
            MT = M.T
            temp = 1.0 - M - MT
            np.fill_diagonal(temp, 1)  # set diagonal values to 1 to make sure log calculation is valid
            self.cumulated_prod_2nd_order += n * np.log(temp)

        return

    def sample(self, _, ranked_dids, n, stopping_condition):
        dist = self.dist
        sampled_dids = self._sample(ranked_dids, dist, n, replace=True)
        self._update(sampled_dids, ranked_dids, dist, stopping_condition)
        return sampled_dids  # note: it may contain duplicated dids

    def estimate(self, t, stopping_condition, sampled_state):
        # calculate inclusion probabilities
        first_order_inclusion_probabilities = 1.0 - np.exp(self.cumulated_prod_1st_order)
        assert np.min(first_order_inclusion_probabilities) > 0
        assert np.max(first_order_inclusion_probabilities) <= 1.0

        # total
        first_order_mask, second_order_mask = self._mask_sampled_dids(sampled_state)
        total = np.sum(first_order_mask * (self.complete_labels / first_order_inclusion_probabilities))

        if t > 1:
            variance1, variance2 = -1, -1  # no need to calculate variance if stopping condition is loose

            if stopping_condition == 'strict1' or stopping_condition == 'strict2':
                M = np.tile(first_order_inclusion_probabilities, (self.N, 1))
                MT = M.T
                second_order_inclusion_probabilities = M + MT - (1.0 - np.exp(self.cumulated_prod_2nd_order))
                assert np.min(second_order_inclusion_probabilities) > 0.0

                if stopping_condition == 'strict1':
                    # variance 1

                    # 1/pi_i**2 - 1/pi_i
                    part1 = 1.0 / first_order_inclusion_probabilities**2 - 1.0 / first_order_inclusion_probabilities
                    # y_i**2
                    yi_2 = self.complete_labels**2

                    # 1/(pi_i*pi_j) - 1/pi_ij
                    M = np.tile(first_order_inclusion_probabilities, (self.N, 1))
                    MT = M.T
                    part2 = 1.0 / (M * MT) - 1.0 / second_order_inclusion_probabilities
                    np.fill_diagonal(part2, 0.0)  # set diagonal values to zero, because summing part2 do not include diagonal values

                    #  y_i * y_j
                    M = np.tile(self.complete_labels, (self.N, 1))
                    MT = M.T
                    yi_yj = M*MT

                    variance1 = np.sum(first_order_mask * part1*yi_2) + np.sum(second_order_mask*part2*yi_yj)

                if stopping_condition == 'strict2':
                    v = len([did for did in sampled_state if sampled_state[did] is True])
                    if v == 1:
                        variance2 = 0
                    else:
                        # (v * y_i / pi_i - total)**2
                        variance2 = (v * self.complete_labels / first_order_inclusion_probabilities - total) ** 2
                        variance2 = (self.N - v) / self.N / v / (v - 1) * np.sum(first_order_mask*variance2)

        else:  # t=1 second order pi matrix is zero. no need to calcualate variance.
            variance1, variance2 = 0, 0

        # LOGGER.info('WRSampler.estimate is done.')

        return total, variance1, variance2


class HTUniformSampler(HorvitzThompson):
    """
    Sample from mixture uniform distribution with replacement.
    """
    def __init__(self):
        super(HTUniformSampler, self).__init__()

    def update_distribution(self):
        dist = [1 / self.N] * self.N
        summ = sum(dist)
        dist = [item/summ for item in dist]
        self.dist = dist
        return


class HTMixtureUniformSampler(HorvitzThompson):
    """
    Sample from mixture uniform distribution with replacement.
    """
    def __init__(self):
        super(HTMixtureUniformSampler, self).__init__()
        self.dist = None

    def update_distribution(self, epsilon, alpha):
        alpha = min(alpha, self.N)  # make sure alpha is smaller than N
        dist = [1.0 - epsilon / alpha] * alpha + [epsilon / (self.N-alpha)] * (self.N - alpha)
        summ = sum(dist)
        dist = [item/summ for item in dist]
        self.dist = dist
        return


class HTPowerLawSampler(HorvitzThompson):
    def __init__(self):
        super(HTPowerLawSampler, self).__init__()

    def update_distribution(self, beta):
        dist = [(i + 1) ** (beta) for i in np.arange(self.N)]
        summ = sum(dist)
        dist = [item / summ for item in dist]
        self.dist = dist
        return

    def init(self, beta, complete_dids, complete_labels):
        super(HTPowerLawSampler, self).init(complete_dids, complete_labels)
        if beta <= -4.0:
            self.dtype = np.float128
        else:
            self.dtype = np.float64


class HTAPPriorSampler(HorvitzThompson):
    def __init__(self):
        super(HTAPPriorSampler, self).__init__()

    def update_distribution(self):
        dist = [math.log((self.N+1) / (i + 1)) for i in np.arange(self.N)]  # i+1 from 1 to N
        summ = sum(dist)
        dist = [item / summ for item in dist]
        self.dist = dist
        return


class HansenHurwitz(object):
    def __init__(self):
        self.batch_data = defaultdict(list)  # key: batch, value: list of (did, selection_prob ) tuples
        self.dist = None
        self.N = 0

    def init(self, total_num, did2label):
        self.N = total_num
        self.did2label = did2label

    def update_distribution(self, **kwargs):
        """Make sampling distribution"""
        raise NotImplementedError

    def _sample(self, ranked_dids, dist, n, replace):
        """Sample n items"""
        assert len(ranked_dids) == len(dist), 'dids len != dist len ({} != {})'.format(len(ranked_dids), len(dist))
        return np.random.choice(a=ranked_dids, size=n, replace=replace, p=dist)

    def _update(self, t, ranked_dids, sampled_dids, dist):
        for did in sampled_dids:
            idx = ranked_dids.index(did)
            prob = dist[idx]
            self.batch_data[t].append((did, prob))   # with duplicates
        return

    def sample(self, t, ranked_dids, n, _):
        dist = self.dist
        sampled_dids = self._sample(ranked_dids, dist, n, replace=True)
        self._update(t, ranked_dids, sampled_dids, dist)
        return sampled_dids  # may contain duplicated dids

    def clear(self):
        self.batch_data = defaultdict(list)
        return

    def estimate(self, t, stopping_condition, sampled_state):
        # total
        totals = [self.did2label[did] / selection_p for batch in self.batch_data.keys()
                  for did, selection_p in self.batch_data[batch]]
        total = np.mean(totals)

        if stopping_condition == 'strict1':
            # variance

            variances = [(self.did2label[did] / selection_p - total)**2 for batch in self.batch_data.keys()
                         for did, selection_p in self.batch_data[batch]]
            if len(variances) == 1 or len(variances) == 0:
                variance = 0
            else:
                variance = np.sum(variances) / len(variances) / (len(variances) - 1)
        else:
            variance = -1
        LOGGER.info('HansenHurwitz.estimate is done.')
        return total, variance, -1


class HHUniformSampler(HansenHurwitz):
    def __init__(self, ):
        super(HHUniformSampler, self).__init__()

    def update_distribution(self):
        dist = [1.0 / self.N] * self.N
        summ = sum(dist)
        dist = [item/summ for item in dist]
        self.dist = dist
        return


class HHMixtureUniformSampler(HansenHurwitz):
    def __init__(self):
        super(HHMixtureUniformSampler, self).__init__()

    def update_distribution(self, epsilon, alpha):
        alpha = min(alpha, self.N)  # make sure b is smaller than N
        dist = [1.0 - epsilon / alpha] * alpha + [epsilon / (self.N - alpha)] * (self.N - alpha)
        summ = sum(dist)
        dist = [item / summ for item in dist]
        self.dist = dist
        return


class HHPowerLawSampler(HansenHurwitz):
    def __init__(self):
        super(HHPowerLawSampler, self).__init__()

    def update_distribution(self, beta):
        dist = [(i + 1) ** (beta) for i in np.arange(self.N)]
        summ = sum(dist)
        dist = [item / summ for item in dist]
        self.dist = dist
        return


class HHAPPriorSampler(HansenHurwitz):
    def __init__(self):
        super(HHAPPriorSampler, self).__init__()

    def update_distribution(self):
        dist = [math.log((self.N+1) / (i + 1)) for i in np.arange(self.N)]  # i+1 from 1 to N
        summ = sum(dist)
        dist = [item / summ for item in dist]
        self.dist = dist
        return


class SCALSampler(object):
    """
    Sample from non overlapped bucket
    """
    def __init__(self):
        self.buckted_dids = set()
        self.sampled_dids = set()
        self.buckted_states = defaultdict(lambda: False)

    def get_bucketed_dids(self):
        return self.buckted_dids

    def get_sampled_dids(self):
        return self.sampled_dids

    def sample(self, ranked_dids, n, B, did2label):
        """

        :param ranked_dids:
        :param n: sub-sample size
        :param B: batch size
        :param did2label:
        :return:
        """
        rest_dids = [did for did in ranked_dids if self.buckted_states[did] == False]

        if len(rest_dids) == 0:  # U is exhausted, return
            sampled_dids = []
            batch_esti_r = 0
            return sampled_dids, batch_esti_r

        B = min(len(rest_dids), B)  # new B
        b = min(n, B)  # real sample size
        if b < 1:
            b = 1
        bucketed_dids = rest_dids[: B]
        # dist = [1.0 / B] * B

        sampled_dids = np.random.choice(a=bucketed_dids, size=b, replace=False)
        summ = sum([did2label[did] for did in sampled_dids])
        batch_esti_r = summ * B / b

        self.sampled_dids = self.sampled_dids.union(set(sampled_dids))
        self.buckted_dids = self.buckted_dids.union(set(bucketed_dids))
        for did in bucketed_dids:
            self.buckted_states[did] = True

        return bucketed_dids, sampled_dids, batch_esti_r


class StratifiedSampler(object):
    """
    Sample from non overlapped bucket
    """
    def __init__(self):
        self.inclusion_probs = defaultdict(float)  # key: did, value: inclusion probability
        self.buckted_dids = set()
        self.sampled_dids = set()
        self.buckted_states = defaultdict(lambda: False)

    def _update(self, bucketed_dids, sampled_dids, prob):
        for did in sampled_dids:
            self.inclusion_probs[did] = prob  # inclusion probability
            self.sampled_dids.add(did)

        for did in bucketed_dids:
            self.buckted_dids.add(did)
            self.buckted_states[did] = True
        return

    def sample(self, t, ranked_dids, n, b):
        rest_dids = [did for did in ranked_dids if self.buckted_states[did] == False]
        if len(rest_dids) <= b:
            n = min(b, n, len(rest_dids))
            bucketed_dids = rest_dids
            inclusion_prob = min(n / len(rest_dids), 1)
            dist = [1 / len(rest_dids)] * len(rest_dids)
        else:
            bucketed_dids = rest_dids[: b]
            inclusion_prob = n / b
            dist = [1 / b] * b
        sampled_dids = np.random.choice(a=bucketed_dids, size=n, replace=True, p=dist)
        self._update(bucketed_dids, sampled_dids, inclusion_prob)

        return sampled_dids

    def estimate(self, complete_data):
        population_total = 0
        for did in self.sampled_dids:
            rel = complete_data[did]['rel']
            inclusion_p = self.inclusion_probs[did]
            population_total += rel/inclusion_p
        return population_total


if __name__ == '__main__':

    np.random.seed(0)

    N = 50000
    n = 100
    b = 100

    dids = [i for i in range(N)]
    labels = np.random.choice(a=[0, 1], size=N, p=[0.9, 0.1])
    true_r = labels.sum()

    sampler = HTAPPriorSampler()
    sampler.init(dids, labels)
    sampler.update_distribution()

    esti_rs = []
    for i in range(10):
        print(i)
        s1 = datetime.datetime.now()
        sampler.sample(None, dids, n, 'loose')
        s2 = datetime.datetime.now()
        print('sample {}s'.format((s2 - s1).seconds))
        total, variance1, variance2 = sampler.estimate(i, 'loose')
        s3 = datetime.datetime.now()
        print('estimate {}s'.format((s3 - s2).seconds))
        esti_rs.append(total)

    esti_rs = np.array(esti_rs)
    print('true_r = {}, esti_r mean={} std={}'.format(true_r, esti_rs.mean(), esti_rs.std()))

    pass