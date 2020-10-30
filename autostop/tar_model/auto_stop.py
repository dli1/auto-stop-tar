# coding=utf-8
import csv
import math
import numpy as np
from operator import itemgetter

from autostop.tar_framework.assessing import Assessor
from autostop.tar_framework.ranking import Ranker
from autostop.tar_framework.sampling_estimating import HTUniformSampler, HTMixtureUniformSampler, HTPowerLawSampler, HTAPPriorSampler, \
    HHMixtureUniformSampler, HHPowerLawSampler, HHAPPriorSampler

from autostop.tar_framework.utils import *
from autostop.tar_model.utils import *

def autostop_method(data_name, topic_set, topic_id,
                    query_file, qrel_file, doc_id_file, doc_text_file,  # data parameters
                    sampler_type='HTAPPriorSampler', epsilon=0.5, beta=-0.1,
                    stopping_percentage=None, stopping_recall=None, target_recall=1.0, stopping_condition='loose', # autostop parameters
                    random_state=0):

    np.random.seed(random_state)

    # loading data
    assessor = Assessor(query_file, qrel_file, doc_id_file, doc_text_file)
    complete_dids = assessor.get_complete_dids()
    complete_labels = assessor.get_complete_labels()
    complete_pseudo_dids = assessor.get_complete_pseudo_dids()
    complete_pseudo_texts = assessor.get_complete_pseudo_texts()
    did2label = assessor.get_did2label()
    total_true_r = assessor.get_total_rel_num()
    total_num = assessor.get_total_doc_num()

    # preparing document features
    ranker = Ranker()
    ranker.set_did_2_feature(dids=complete_pseudo_dids, texts=complete_pseudo_texts, corpus_texts=complete_pseudo_texts)
    ranker.set_features_by_name('complete_dids', complete_dids)

    # sampler
    model_name = 'autostop' +'-'
    model_name += 'sp' + str(stopping_percentage) + '-'
    model_name += 'sr' + str(stopping_recall) + '-'
    model_name += 'smp' + str(sampler_type) + '-'

    if sampler_type == 'HTMixtureUniformSampler':
        model_name += 'epsilon' + str(epsilon) + '-'
        sampler = HTMixtureUniformSampler()
        sampler.init(complete_dids, complete_labels)
    elif sampler_type == 'HTUniformSampler':
        sampler = HTUniformSampler()
        sampler.init(complete_dids, complete_labels)
        sampler.update_distribution()
    elif sampler_type == 'HTPowerLawSampler':
        model_name += 'beta' + str(beta) + '-'
        sampler = HTPowerLawSampler()
        sampler.init(beta, complete_dids, complete_labels)
        sampler.update_distribution(beta=beta)
    elif sampler_type == 'HTAPPriorSampler':
        sampler = HTAPPriorSampler()
        sampler.init(complete_dids, complete_labels)
        sampler.update_distribution()
    elif sampler_type == 'HHMixtureUniformSampler':
        model_name += 'epsilon' + str(epsilon) + '-'
        sampler = HHMixtureUniformSampler()
        sampler.init(total_num, did2label)
    elif sampler_type == 'HHPowerLawSampler':
        model_name += 'beta' + str(beta) + '-'
        sampler = HHPowerLawSampler()
        sampler.init(total_num, did2label)
        sampler.update_distribution(beta=beta)
    elif sampler_type == 'HHAPPriorSampler':
        sampler = HHAPPriorSampler()
        sampler.init(total_num, did2label)
        sampler.update_distribution()
    else:
        raise TypeError

    model_name += 'tr' + str(target_recall) + '-'
    model_name += 'sc' + stopping_condition

    # local parameters
    stopping = False
    t = 0
    batch_size = 1
    temp_doc_num = 100

    # starting the TAR process
    interaction_file = name_interaction_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                             exp_id=random_state, topic_id=topic_id)
    with open(interaction_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f)
        while not stopping:
            t += 1
            train_dids, train_labels = assessor.get_training_data(temp_doc_num)
            train_features = ranker.get_feature_by_did(train_dids)
            ranker.train(train_features, train_labels)

            test_features = ranker.get_features_by_name('complete_dids')
            scores = ranker.predict(test_features)

            zipped = sorted(zip(complete_dids, scores), key=itemgetter(1), reverse=True)
            ranked_dids, _ = zip(*zipped)

            if sampler_type == 'HHMixtureUniformSampler' or sampler_type == 'HTMixtureUniformSampler':
                sampler.update_distribution(epsilon=epsilon, alpha=batch_size)

            sampled_dids = sampler.sample(t, ranked_dids, batch_size, stopping_condition)

            assessor.update_assess(sampled_dids)

            sampled_state = assessor.get_assessed_state()

            total_esti_r, var1, var2 = sampler.estimate(t, stopping_condition, sampled_state)

            # statistics
            sampled_num = assessor.get_assessed_num()
            running_true_r = assessor.get_assessed_rel_num()

            if total_esti_r != 0:
                running_esti_recall = running_true_r / float(total_esti_r)
            else:
                running_esti_recall = 0
            if total_true_r != 0:
                running_true_recall = running_true_r / float(total_true_r)
            else:
                running_true_recall = 0
            ap = calculate_ap(did2label, ranked_dids)

            # update parameters
            batch_size += math.ceil(batch_size / 10)

            # debug: writing values
            csvwriter.writerow(
                (t, batch_size, total_num, sampled_num, total_true_r, total_esti_r, var1, var2,
                 running_true_r, ap, running_esti_recall, running_true_recall))

            # autostop
            if running_true_r > 0:
                if stopping_condition == 'loose':
                    if running_true_r >= target_recall * total_esti_r:
                        stopping = True
                elif stopping_condition == 'strict1':
                    if running_true_r >= target_recall * (total_esti_r + np.sqrt(var1)):
                        stopping = True
                elif stopping_condition == 'strict2':
                    if running_true_r >= target_recall * (total_esti_r + np.sqrt(var2)):
                        stopping = True
                else:
                    raise NotImplementedError

            # debug: stop early
            if stopping_recall:
                if running_true_recall >= stopping_recall:
                    stopping = True
            if stopping_percentage:
                if sampled_num >= int(total_num * stopping_percentage):
                    stopping = True

    # writing
    sampled_dids = assessor.get_assessed_dids()
    shown_features = ranker.get_feature_by_did(sampled_dids)
    shown_scores = ranker.predict(shown_features)
    zipped = sorted(zip(sampled_dids, shown_scores), key=itemgetter(1), reverse=True)
    shown_dids, scores = zip(*zipped)

    check_func = assessor.assess_state_check_func()
    tar_run_file = name_tar_run_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                     exp_id=random_state, topic_id=topic_id)
    with open(tar_run_file, 'w', encoding='utf8') as f:
        write_tar_run_file(f=f, topic_id=topic_id, check_func=check_func, shown_dids=shown_dids)

    LOGGER.info('TAR is finished.')
    return


def autostop_for_large_collection(data_name, topic_set, topic_id,
        query_files, qrel_files, doc_id_files, doc_text_files,  # data parameters
        sampler_type='HTAPPriorSampler', epsilon=0.5, beta=-0.1,
        stopping_percentage=None, stopping_recall=None, target_recall=1.0, stopping_condition='strict1', # autostop parameters
        random_state=0):

    # sampler
    model_name = 'autostop' + '-'
    model_name += 'sp' + str(stopping_percentage) + '-'
    model_name += 'sr' + str(stopping_recall) + '-'
    model_name += 'smp' + str(sampler_type) + '-'

    if sampler_type == 'HTMixtureUniformSampler':
        model_name += 'epsilon' + str(epsilon) + '-'
    elif sampler_type == 'HTPowerLawSampler':
        model_name += 'beta' + str(beta) + '-'
    elif sampler_type == 'HHMixtureUniformSampler':
        model_name += 'epsilon' + str(epsilon) + '-'
    elif sampler_type == 'HHPowerLawSampler':
        model_name += 'beta' + str(beta) + '-'
    else:
        raise NotImplementedError

    model_name += 'tr' + str(target_recall) + '-'
    model_name += 'sc' + stopping_condition
    LOGGER.info('Model configuration: {}.'.format(model_name))

    # starting the TAR process after splitting the complete collection into small blocks
    total_shown_dids = []
    for query_file, qrel_file, doc_id_file, doc_text_file in zip(query_files, qrel_files, doc_id_files, doc_text_files):

        # loading data
        assessor = Assessor(query_file, qrel_file, doc_id_file, doc_text_file)
        complete_dids = assessor.get_complete_dids()
        complete_labels = assessor.get_complete_labels()
        complete_pseudo_dids = assessor.get_complete_pseudo_dids()
        complete_pseudo_texts = assessor.get_complete_pseudo_texts()
        did2label = assessor.get_did2label()
        total_true_r = assessor.get_total_rel_num()
        total_num = assessor.get_total_doc_num()

        # ranker
        ranker = Ranker()
        ranker.set_did_2_feature(dids=complete_pseudo_dids, texts=complete_pseudo_texts, corpus_texts=complete_pseudo_texts)
        ranker.set_features_by_name('complete_dids', complete_dids)

        if sampler_type == 'HTMixtureUniformSampler':
            sampler = HTMixtureUniformSampler()
            sampler.init(complete_dids, complete_labels)
        elif sampler_type == 'HTUniformSampler':
            sampler = HTUniformSampler()
            sampler.init(complete_dids, complete_labels)
            sampler.update_distribution()
        elif sampler_type == 'HTPowerLawSampler':
            sampler = HTPowerLawSampler()
            sampler.init(beta, complete_dids, complete_labels)
            sampler.update_distribution(beta=beta)
        elif sampler_type == 'HTAPPriorSampler':
            sampler = HTAPPriorSampler()
            sampler.init(complete_dids, complete_labels)
            sampler.update_distribution()
        elif sampler_type == 'HHMixtureUniformSampler':
            sampler = HHMixtureUniformSampler()
            sampler.init(total_num, did2label)
        elif sampler_type == 'HHPowerLawSampler':
            sampler = HHPowerLawSampler()
            sampler.init(total_num, did2label)
            sampler.update_distribution(beta=beta)
        elif sampler_type == 'HHAPPriorSampler':
            sampler = HHAPPriorSampler()
            sampler.init(total_num, did2label)
            sampler.update_distribution()
        else:
            raise TypeError

        # local parameters
        stopping = False
        t = 0
        batch_size = 1
        temp_doc_num = 100

        while not stopping:
            t += 1
            train_dids, train_labels = assessor.get_training_data(temp_doc_num)
            train_features = ranker.get_feature_by_did(train_dids)
            ranker.train(train_features, train_labels)

            test_features = ranker.get_features_by_name('complete_dids')
            scores = ranker.predict(test_features)

            zipped = sorted(zip(complete_dids, scores), key=itemgetter(1), reverse=True)
            ranked_dids, _ = zip(*zipped)

            if sampler_type == 'HHMixtureUniformSampler' or sampler_type == 'HTMixtureUniformSampler':
                sampler.update_distribution(epsilon=epsilon, alpha=batch_size)

            sampled_dids = sampler.sample(t, ranked_dids, batch_size, stopping_condition)
            assessor.update_assess(sampled_dids)

            sampled_state = assessor.get_assessed_state()
            total_esti_r, var1, var2 = sampler.estimate(t, stopping_condition, sampled_state)

            # statistics
            sampled_num = assessor.get_assessed_num()
            running_true_r = assessor.get_assessed_rel_num()

            if total_true_r != 0:
                running_true_recall = running_true_r / float(total_true_r)
            else:
                running_true_recall = 0

            # update parameters
            batch_size += math.ceil(batch_size / 10)

            # autostop
            if running_true_r > 0:
                if stopping_condition == 'loose':
                    if running_true_r >= target_recall * total_esti_r:
                        stopping = True
                elif stopping_condition == 'strict1':
                    if running_true_r >= target_recall * (total_esti_r + np.sqrt(var1)):
                        stopping = True
                elif stopping_condition == 'strict2':
                    if running_true_r >= target_recall * (total_esti_r + np.sqrt(var2)):
                        stopping = True
                else:
                    raise NotImplementedError

            # stop early
            if stopping_recall:
                if running_true_recall >= stopping_recall:
                    stopping = True

            if stopping_percentage:
                if sampled_num >= int(total_num * stopping_percentage):
                    stopping = True

        sampled_dids = assessor.get_assessed_dids()
        shown_features = ranker.get_feature_by_did(sampled_dids)
        shown_scores = ranker.predict(shown_features)
        zipped = sorted(zip(sampled_dids, shown_scores), key=itemgetter(1), reverse=True)
        shown_dids, scores = zip(*zipped)

        total_shown_dids.extend(shown_dids)

    tar_run_file = name_tar_run_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                     exp_id=random_state, topic_id=topic_id)
    with open(tar_run_file, 'w', encoding='utf8') as f:
        write_tar_run_file(f=f, topic_id=topic_id, check_func=lambda x: True, shown_dids=total_shown_dids)

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

    autostop_method(data_name, topic_id, topic_set, query_file, qrel_file, doc_id_file, doc_text_file)