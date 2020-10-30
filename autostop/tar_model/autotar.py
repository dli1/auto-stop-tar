# coding=utf-8
"""
The implementation is based on the following paper:
[1] Gordon V. Cormack and Maura R. Grossman. 2015. Autonomy and Reliability of Continuous Active Learning for
Technology-Assisted Review. CoRR abs/1504.06868 (2015). arXiv:1504.06868 http://arxiv.org/abs/1504.06868

"""
import csv
import math
import numpy as np
from operator import itemgetter
from autostop.tar_framework.assessing import DataLoader, Assessor
from autostop.tar_framework.ranking import Ranker
from autostop.tar_model.utils import *
from autostop.tar_framework.utils import *


def autotar_method(data_name, topic_set, topic_id,
                   query_file, qrel_file, doc_id_file, doc_text_file,  # data parameters
                   stopping_percentage=1.0, stopping_recall=None,  # autostop parameters, for debug
                   ranker_tfidf_corpus_files=[], classifier='lr', min_df=2, C=1.0,  # ranker parameters
                   random_state=0):
    """
    Implementation of the TAR process.
    @param data_name: dataset name
    @param topic_set: parameter-tuning set or test set
    @param topic_id: topic id
    @param stopping_percentage: stop TAR when x percentage of documents have been screened
    @param stopping_recall: stop TAR when x recall is achieved
    @param corpus_type: indicates what corpus to use when building features, see Ranker
    @param min_df: parameter of Ranker
    @param C: parameter of Ranker
    @param save_did2feature: save the did2feature dict as a pickle to fasten experiments
    @param random_state: random seed
    @return:
    """
    np.random.seed(random_state)

    # model named with its configuration
    model_name = 'autotar' + '-'
    model_name += 'sp' + str(stopping_percentage) + '-'
    model_name += 'sr' + str(stopping_recall) + '-'
    model_name += 'ct' + str(ranker_tfidf_corpus_files) + '-'
    model_name += 'csf' + classifier + '-'
    model_name += 'md' + str(min_df) + '-'
    model_name += 'c' + str(C)
    LOGGER.info('Model configuration: {}.'.format(model_name))

    # loading data
    assessor = Assessor(query_file, qrel_file, doc_id_file, doc_text_file)
    complete_dids = assessor.get_complete_dids()
    complete_pseudo_dids = assessor.get_complete_pseudo_dids()
    complete_pseudo_texts = assessor.get_complete_pseudo_texts()

    if ranker_tfidf_corpus_files == []:
        corpus_texts = assessor.get_complete_pseudo_texts()
    else:  # this branch is only available for autotar, to study the effect of ranker
        def read_temp(files):
            for file in files:
                for text in DataLoader.read_doc_texts_2_list(file):
                    yield text

        corpus_texts = read_temp(ranker_tfidf_corpus_files)
    did2label = assessor.get_did2label()
    total_true_r = assessor.get_total_rel_num()
    total_num = assessor.get_total_doc_num()

    # preparing document features
    ranker = Ranker(model_type=classifier, random_state=random_state, min_df=min_df, C=C)
    ranker.set_did_2_feature(dids=complete_pseudo_dids, texts=complete_pseudo_texts, corpus_texts=corpus_texts)
    ranker.set_features_by_name('complete_dids', complete_dids)

    # local parameters are set according to [1]
    stopping = False
    t = 0
    batch_size = 1
    temp_doc_num = 100

    interaction_file = name_interaction_file(data_name=data_name, model_name=model_name, topic_set=topic_set, exp_id=random_state, topic_id=topic_id)
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
            ranked_dids, scores = zip(*zipped)

            # cutting off instead of sampling
            selected_dids = assessor.get_top_assessed_dids(ranked_dids, batch_size)
            assessor.update_assess(selected_dids)

            # statistics
            sampled_num = assessor.get_assessed_num()
            running_true_r = assessor.get_assessed_rel_num()
            running_true_recall = running_true_r / float(total_true_r)
            ap = calculate_ap(did2label, ranked_dids)

            # update parameters
            batch_size += math.ceil(batch_size / 10)

            # debug: writing values
            csvwriter.writerow((t, batch_size, total_num, sampled_num, total_true_r, running_true_r, ap, running_true_recall))

            # debug: stop early
            if stopping_recall:
                if running_true_recall >= stopping_recall:
                    stopping = True
            if stopping_percentage:
                if sampled_num >= int(total_num * stopping_percentage):
                    stopping = True

    # tar run file
    shown_dids = assessor.get_assessed_dids()
    check_func = assessor.assess_state_check_func()
    tar_run_file = name_tar_run_file(data_name=data_name, model_name=model_name, topic_set=topic_set, exp_id=random_state, topic_id=topic_id)
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

    autotar_method(data_name, topic_id, topic_set,query_file, qrel_file, doc_id_file, doc_text_file)