# coding=utf-8

"""
The implementation is based on:
[1] Gordon V. Cormack and Maura R. Grossman. 2016. Engineering Quality and Reliability in Technology-Assisted Review. In Proceedings of the 39th International ACM SIGIR Conference on Research and Development in Information Retrieval(Pisa, Italy) (SIGIR ’16). ACM, New York, NY, USA, 75–84.
"""
import csv
import math
import numpy as np
from operator import itemgetter
from autostop.tar_framework.assessing import Assessor
from autostop.tar_framework.ranking import Ranker
from autostop.tar_framework.utils import *
from autostop.tar_model.utils import *


def target_method(data_name, topic_set, topic_id,
                  query_file, qrel_file, doc_id_file, doc_text_file,  # data parameters
                   stopping_percentage=None, stopping_recall=None,  # autostop parameters
                   target_rel_num=10,  # target parameter
                   random_state=0):

    np.random.seed(random_state)

    # model named with its configuration
    model_name = 'target' + '-'
    model_name += 'sp' + str(stopping_percentage) + '-'
    model_name += 'sr' + str(stopping_recall) + '-'
    model_name += 'trn' + str(target_rel_num)
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

    # sample target set: the target set should not affect the interaction process.
    population = complete_dids
    target_set = set()
    sample_size = 100
    sample_rel_num = 0
    while not(len(population) == 0 or sample_rel_num >= target_rel_num):
        population = list(set(complete_dids).difference(target_set))
        sample_size = min(len(population), sample_size)
        sampled_dids = set(np.random.choice(a=population, size=sample_size, replace=False))  # unique random elements
        sample_rel_num += len([did for did in sampled_dids if assessor.get_rel_label(did) == 1])
        target_set = target_set.union(sampled_dids)

    target_rel_set = set([did for did in target_set if assessor.get_rel_label(did) == 1])

    # starting the TAR process
    stopping = False
    t = 0
    batch_size = 100
    temp_doc_num = 100

    interaction_file = name_interaction_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                             exp_id=random_state, topic_id=topic_id)
    with open(interaction_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f)
        while not stopping:
            t += 1

            # train
            train_dids, train_labels = assessor.get_training_data(temp_doc_num)
            train_features = ranker.get_feature_by_did(train_dids)
            ranker.train(train_features, train_labels)

            # predict
            complete_features = ranker.get_features_by_name('complete_dids')
            scores = ranker.predict(complete_features)
            zipped = sorted(zip(complete_dids, scores), key=itemgetter(1), reverse=True)
            ranked_dids, scores = zip(*zipped)

            # cutting off instead of sampling
            selected_dids = assessor.get_top_assessed_dids(ranked_dids, batch_size)
            assessor.update_assess(selected_dids)

            # statistics
            sampled_dids = assessor.get_assessed_dids()
            sampled_num = len(set(sampled_dids).union(target_set))
            running_true_r = assessor.get_assessed_rel_num()
            running_true_recall = running_true_r / float(total_true_r)
            ap = calculate_ap(did2label, ranked_dids)

            # update parameters
            batch_size += math.ceil(batch_size / 10)

            # debug: writing values
            csvwriter.writerow(
                (t, batch_size, total_num, sampled_num, total_true_r,
                 running_true_r, ap, running_true_recall))

            sampled_rel_set = set(assessor.get_assessed_rel_dids())

            if set(target_rel_set).issubset(sampled_rel_set):
                stopping = True

            if sampled_num >= total_num:  # dids are exhausted
                stopping = True

            # stop early
            if stopping_recall:
                if running_true_recall >= stopping_recall:
                    stopping = True
            if stopping_percentage:
                if sampled_num >= int(total_num * stopping_percentage):
                    stopping = True

    # INTERACTION METHOD: apply ranked_dids order.
    sampled_dids = assessor.get_assessed_dids()
    shown_dids = list(target_set.union(set(sampled_dids)))

    shown_features = ranker.get_feature_by_did(shown_dids)
    shown_scores = ranker.predict(shown_features)
    zipped = sorted(zip(shown_dids, shown_scores), key=itemgetter(1), reverse=True)
    shown_dids, scores = zip(*zipped)

    check_func = assessor.assess_state_check_func()
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

    target_method(data_name, topic_id, topic_set, query_file, qrel_file, doc_id_file, doc_text_file)