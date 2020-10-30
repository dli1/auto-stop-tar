# coding=utf-8

"""
The implementation is based on
[1] Noah Hollmann and Carsten Eickhoff. 2017. Ranking and Feedback-based Stopping for Recall-Centric Document Retrieval. In CLEF (Working Notes).
"""


import bisect
import pickle
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


from autostop.tar_framework.assessing import Assessor, DataLoader
from autostop.tar_framework.ranking import bm25_okapi_rank
from autostop.tar_framework.utils import *


def fit_power_law(ranked_dids, ranked_labels):
    def f(x, a, b):
        return a * x^b - 1
    popt, pcov = curve_fit(f=f, xdata=range(len(ranked_dids)), ydata=ranked_labels)
    return popt


def score_distribion_training_fitting(data_name, topic_set, topic_id,
                                      query_file, qrel_file, doc_id_file, doc_text_file,  # data parameters
                                      training_query_files, training_qrel_files,
                                      training_doc_id_files, training_doc_text_files,  # training data
                                      target_recall=0.99,  # if target_recall=1.0, score_bound will be -inf
                                      random_state=0):
    """
    Implementation of the score distribution model -- training-fitting.
    :param data_name:
    :param topic_id:
    :param exp_id:
    :param training_topics:
    :return:
    """
    np.random.seed(random_state)

    # model named with its configuration
    model_name = 'sdtf' + '-'
    model_name += 'tr' + str(target_recall)
    LOGGER.info('Model configuration: {}.'.format(model_name))

    # collecting rel scores from tuning_topic_set

    training_rel_scores = []
    for tquery_file, tqrel_file, tdoc_id_file, tdoc_text_file in zip(training_query_files,
                                                                     training_qrel_files,
                                                                     training_doc_id_files,
                                                                     training_doc_text_files):
        tquery = DataLoader.read_title(tquery_file)
        did2label = DataLoader.read_qrels(tqrel_file)
        complete_dids = DataLoader.read_doc_ids(tdoc_id_file)
        did2text = DataLoader.read_doc_texts(tdoc_text_file)
        complete_texts = [did2text[did] for did in complete_dids]

        # ranked dids, ranking scores
        ranked_dids, ranked_scores = bm25_okapi_rank(complete_dids, complete_texts, tquery)

        # normalized scores
        scaler = StandardScaler()
        ranked_scores = np.array(ranked_scores).reshape(-1, 1)
        norm_scores = list(scaler.fit_transform(ranked_scores).flatten())

        # rel scores
        rel_scores = [score for did, score in zip(ranked_dids, norm_scores) if did2label[did] == REL]
        training_rel_scores.extend(rel_scores)


    # ranking dids, ranking scores
    assessor = Assessor(query_file, qrel_file, doc_id_file, doc_text_file)
    complete_dids = assessor.get_complete_dids()
    complete_texts = assessor.get_complete_texts()
    query = assessor.get_title()
    ranked_dids, ranked_scores = bm25_okapi_rank(complete_dids, complete_texts, query)

    # normalizing scores
    scaler = StandardScaler()
    ranked_scores = np.array(ranked_scores).reshape(-1, 1)
    norm_scores = list(scaler.fit_transform(ranked_scores).flatten())

    # calculating cutoff
    loc, scale = norm.fit(training_rel_scores)
    score_bound = norm.ppf(1 - target_recall, loc=loc, scale=scale)
    small2big_ranked_scores = list(reversed(norm_scores))
    cutoff = bisect.bisect_left(small2big_ranked_scores, score_bound)
    cutoff = max(1, cutoff)
    screen_dids = ranked_dids[:-cutoff]

    # output run file
    check_func = assessor.assess_state_check_func()
    tar_run_file = name_tar_run_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                     exp_id=random_state, topic_id=topic_id)
    with open(tar_run_file, 'w', encoding='utf8') as f:
        write_tar_run_file(f=f, topic_id=topic_id, check_func=check_func, shown_dids=screen_dids)

    LOGGER.info('TAR is finished.')
    return


def score_distribion_feedback_uniform(data_name, topic_set, topic_id,
                                      query_file, qrel_file, doc_id_file, doc_text_file,  # data parameters
                                      sample_percentage=0.1, target_recall=0.99,
                                      random_state=0):
    np.random.seed(random_state)

    # model named with its configuration
    model_name = 'sdfu' + '-'
    model_name += 'smp' + str(sample_percentage) + '-'
    model_name += 'tr' + str(target_recall)
    LOGGER.info('Model configuration: {}.'.format(model_name))

    # loading data
    assessor = Assessor(query_file, qrel_file, doc_id_file, doc_text_file)
    complete_dids = assessor.get_complete_dids()
    complete_texts = assessor.get_complete_texts()
    query = assessor.get_title()
    total_num = assessor.get_total_doc_num()

    # ranking dids, ranking scores
    ranked_dids, ranked_scores = bm25_okapi_rank(complete_dids, complete_texts, query)

    # normalizing scores
    scaler = StandardScaler()
    ranked_scores = np.array(ranked_scores).reshape(-1, 1)
    norm_scores = list(scaler.fit_transform(ranked_scores).flatten())

    # uniformly sampling some documents to fit Gaussian
    sample_num = int(sample_percentage * total_num)
    sampled_dids = list(np.random.choice(a=complete_dids, size=sample_num, replace=False).flatten())
    sampled_rel_scores = [norm_scores[ranked_dids.index(did)] for did in sampled_dids if assessor.get_rel_label(did) == REL]
    if sampled_rel_scores == []:
        sampled_rel_scores = [0]  # standard normal distribution

    # calculating cutoff
    mean, std = norm.fit(sampled_rel_scores)
    try:
        score_bound = norm.ppf(1-target_recall, loc=mean, scale=std)
    except:
        score_bound = norm.ppf(1-target_recall)

    cutoff = bisect.bisect_left(list(reversed(norm_scores)), score_bound)
    cutoff = max(1, cutoff)
    screen_dids = ranked_dids[:-cutoff]

    # output run file
    # NO INTERACTION: only apply ranked_dids order
    check_func = assessor.assess_state_check_func()
    shown_dids = screen_dids
    for did in ranked_dids[-cutoff:]:
        if check_func(did) is True:
            screen_dids.append(did)

    tar_run_file = name_tar_run_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                     exp_id=random_state, topic_id=topic_id)
    with open(tar_run_file, 'w', encoding='utf8') as f:
        write_tar_run_file(f=f, topic_id=topic_id, check_func=check_func, shown_dids=shown_dids)

    LOGGER.info('TAR is finished.')
    return


if __name__ == '__main__':
    data_name = 'clef2017'
    topic_id = 'CD008081'
    topic_set = 'test'
    query_file = os.path.join(PARENT_DIR, 'data', data_name, 'topics', topic_id)
    qrel_file = os.path.join(PARENT_DIR, 'data', data_name, 'qrels', topic_id)
    doc_id_file = os.path.join(PARENT_DIR, 'data', data_name, 'docids', topic_id)
    doc_text_file = os.path.join(PARENT_DIR, 'data', data_name, 'doctexts', topic_id)

    training_query_files, training_qrel_files, training_doc_id_files, training_doc_text_files = [], [], [], []
    for topic_id in ['CD009135', 'CD009786']:
        training_query_files.append(os.path.join(PARENT_DIR, 'data', data_name, 'topics', topic_id))
        training_qrel_files.append(os.path.join(PARENT_DIR, 'data', data_name, 'qrels', topic_id))
        training_doc_id_files.append(os.path.join(PARENT_DIR, 'data', data_name, 'docids', topic_id))
        training_doc_text_files.append(os.path.join(PARENT_DIR, 'data', data_name, 'doctexts', topic_id))

    score_distribion_training_fitting(data_name, topic_id, topic_set,
                                      query_file, qrel_file, doc_id_file, doc_text_file,
                                      training_query_files, training_qrel_files,
                                      training_doc_id_files, training_doc_text_files)