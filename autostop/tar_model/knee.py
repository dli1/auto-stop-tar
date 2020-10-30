# coding=utf-8

"""
The implementation is based on
[1] Satopaa, Ville, et al. "Finding a" kneedle" in a haystack: Detecting knee points in system behavior." 2011 31st international conference on distributed computing systems workshops. IEEE, 2011.
[2] Gordon V. Cormack and Maura R. Grossman. 2016. Engineering Quality and Reliability in Technology-Assisted Review. In Proceedings of the 39th International ACM SIGIR Conference on Research and Development in Information Retrieval (Pisa, Italy) (SIGIR ’16). ACM, New York, NY, USA, 75–84.
"""

import csv
import math
import numpy as np
from operator import itemgetter
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from autostop.tar_framework.assessing import Assessor
from autostop.tar_framework.ranking import Ranker
from autostop.tar_model.utils import *
from autostop.tar_framework.utils import *


def detect_knee(data, window_size=1, s=10):
    """
    Detect the so-called knee in the data.

    The implementation is based on paper [1] and code here (https://github.com/jagandecapri/kneedle).

    @param data: The 2d data to find an knee in.
    @param window_size: The data is smoothed using Gaussian kernel average smoother, this parameter is the window used for averaging (higher values mean more smoothing, try 3 to begin with).
    @param s: How many "flat" points to require before we consider it a knee.
    @return: The knee values.
    """

    data_size = len(data)
    data = np.array(data)

    if data_size == 1:
        return None

    # smooth
    smoothed_data = []
    for i in range(data_size):

        if 0 < i - window_size:
            start_index = i - window_size
        else:
            start_index = 0
        if i + window_size > data_size - 1:
            end_index = data_size - 1
        else:
            end_index = i + window_size

        sum_x_weight = 0
        sum_y_weight = 0
        sum_index_weight = 0
        for j in range(start_index, end_index):
            index_weight = norm.pdf(abs(j-i)/window_size, 0, 1)
            sum_index_weight += index_weight
            sum_x_weight += index_weight * data[j][0]
            sum_y_weight += index_weight * data[j][1]

        smoothed_x = sum_x_weight / sum_index_weight
        smoothed_y = sum_y_weight / sum_index_weight

        smoothed_data.append((smoothed_x, smoothed_y))

    smoothed_data = np.array(smoothed_data)

    # normalize
    normalized_data = MinMaxScaler().fit_transform(smoothed_data)

    # difference
    differed_data = [(x, y-x) for x, y in normalized_data]

    # find indices for local maximums
    candidate_indices = []
    for i in range(1, data_size-1):
        if (differed_data[i-1][1] < differed_data[i][1]) and (differed_data[i][1] > differed_data[i+1][1]):
            candidate_indices.append(i)

    # threshold
    step = s * (normalized_data[-1][0] - data[0][0]) / (data_size - 1)

    # knees
    knee_indices = []
    for i in range(len(candidate_indices)):
        candidate_index = candidate_indices[i]

        if i+1 < len(candidate_indices):  # not last second
            end_index = candidate_indices[i+1]
        else:
            end_index = data_size

        threshold = differed_data[candidate_index][1] - step

        for j in range(candidate_index, end_index):
            if differed_data[j][1] < threshold:
                knee_indices.append(candidate_index)
                break

    if knee_indices != []:
        return knee_indices #data[knee_indices]
    else:
        return None


def test_detect_knee():
    # data with knee at [0.2, 0.75]
    print('First example.')
    data = [[0, 0],
            [0.1, 0.55],
            [0.2, 0.75],
            [0.35, 0.825],
            [0.45, 0.875],
            [0.55, 0.9],
            [0.675, 0.925],
            [0.775, 0.95],
            [0.875, 0.975],
            [1, 1]]

    knees = detect_knee(data, window_size=1, s=1)
    for knee in knees:
        print(data[knee])

    # data with knee at [0.45  0.1  ], [0.775 0.2  ]
    print('Second example.')
    data = [[0, 0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.35, 0.1],
            [0.45, 0.1],
            [0.55, 0.1],
            [0.675, 0.2],
            [0.775, 0.2],
            [0.875, 0.2],
            [1, 1]]
    knees = detect_knee(data, window_size=1, s=1)
    for knee in knees:
        print(data[knee])


def knee_method(data_name, topic_set, topic_id,
                query_file, qrel_file, doc_id_file, doc_text_file,  # data parameters
                stopping_beta=100, stopping_percentage=1.0, stopping_recall=None,  # autostop parameters
                rho='dynamic',
                random_state=0):
    """
    Implementation of the Knee method.
    See
    @param data_name:
    @param topic_set:
    @param topic_id:
    @param stopping_beta: stopping_beta: only stop TAR process until at least beta documents had been screen
    @param stopping_percentage:
    @param stopping_recall:
    @param rho:
    @param random_state:
    @return:
    """
    np.random.seed(random_state)

    # model named with its configuration
    model_name = 'knee' + '-'
    model_name += 'sb' + str(stopping_beta) + '-'
    model_name += 'sp' + str(stopping_percentage) + '-'
    model_name += 'sr' + str(stopping_recall) + '-'

    model_name += 'rho' + str(rho)
    LOGGER.info('Model configuration: {}.'.format(model_name))

    # loading data
    assessor = Assessor(query_file, qrel_file, doc_id_file, doc_text_file)
    complete_dids = assessor.get_complete_dids()
    complete_pseudo_dids = assessor.get_complete_pseudo_dids()
    complete_pseudo_texts = assessor.get_complete_pseudo_texts()
    did2label = assessor.get_did2label()
    total_true_r = assessor.get_total_rel_num()
    total_num = assessor.get_total_doc_num()

    # preparing document features
    ranker = Ranker()
    ranker.set_did_2_feature(dids=complete_pseudo_dids, texts=complete_pseudo_texts, corpus_texts=complete_pseudo_texts)
    ranker.set_features_by_name('complete_dids', complete_dids)

    # local parameters
    stopping = False
    t = 0
    batch_size = 1
    temp_doc_num = 100
    knee_data = []

    # starting the TAR process
    interaction_file = name_interaction_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                             exp_id=random_state, topic_id=topic_id)
    with open(interaction_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f)
        while not stopping:
            t += 1
            LOGGER.info('TAR: iteration={}'.format(t))
            train_dids, train_labels = assessor.get_training_data(temp_doc_num)
            train_features = ranker.get_feature_by_did(train_dids)
            ranker.train(train_features, train_labels)

            test_features = ranker.get_features_by_name('complete_dids')
            scores = ranker.predict(test_features)

            zipped = sorted(zip(complete_dids, scores), key=itemgetter(1), reverse=True)
            ranked_dids, _ = zip(*zipped)

            # cutting off instead of sampling
            selected_dids = assessor.get_top_assessed_dids(ranked_dids, batch_size)
            assessor.update_assess(selected_dids)

            # statistics
            sampled_num = assessor.get_assessed_num()
            sampled_percentage = sampled_num/total_num
            running_true_r = assessor.get_assessed_rel_num()
            running_true_recall = running_true_r / float(total_true_r)
            ap = calculate_ap(did2label, ranked_dids)

            # update parameters
            batch_size += math.ceil(batch_size / 10)

            # debug: writing values
            csvwriter.writerow(
                (t, batch_size, total_num, sampled_num, total_true_r, running_true_r, ap, running_true_recall))

            # detect knee
            knee_data.append((sampled_num, running_true_r))
            knee_indice = detect_knee(knee_data)  # x: sampled_percentage, y: running_true_r
            if knee_indice is not None:

                knee_index = knee_indice[-1]
                rank1, r1 = knee_data[knee_index]
                rank2, r2 = knee_data[-1]

                try:
                    current_rho = float(r1 / rank1) / float((r2 - r1 + 1) / (rank2 - rank1))
                except:
                    print('(rank1, r1) = ({} {}), (rank2, r2) = ({} {})'.format(rank1, r1, rank2, r2))
                    current_rho = 0  # do not stop

                if rho == 'dynamic':
                    rho = 156 - min(running_true_r, 150)   # rho is in [6, 156], see [1]
                else:
                    rho = float(rho)

                if current_rho > rho:
                    if sampled_num > stopping_beta:
                        stopping = True

            # debug: stop early
            if stopping_recall:
                if running_true_recall >= stopping_recall:
                    stopping = True
            if stopping_percentage:
                if sampled_num >= int(total_num * stopping_percentage):
                    stopping = True

    shown_dids = assessor.get_assessed_dids()
    check_func = assessor.assess_state_check_func()
    tar_run_file = name_tar_run_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                     exp_id=random_state, topic_id=topic_id)
    with open(tar_run_file, 'w', encoding='utf8') as f:
        write_tar_run_file(f=f, topic_id=topic_id, check_func=check_func, shown_dids=shown_dids)

    LOGGER.info('TAR is finished.')

    return

if __name__ == '__main__':
    # test_detect_knee()

    data_name = 'clef2017'
    topic_id = 'CD008081'
    topic_set = 'test'
    query_file = os.path.join(PARENT_DIR, 'data', data_name, 'topics', topic_id)
    qrel_file = os.path.join(PARENT_DIR, 'data', data_name, 'qrels', topic_id)
    doc_id_file = os.path.join(PARENT_DIR, 'data', data_name, 'docids', topic_id)
    doc_text_file = os.path.join(PARENT_DIR, 'data', data_name, 'doctexts', topic_id)

    knee_method(data_name, topic_id, topic_set,query_file, qrel_file, doc_id_file, doc_text_file)