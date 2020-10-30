# coding=utf-8

import datetime
import argparse
import numpy as np
from autostop.tar_model.autotar import autotar_method
from autostop.tar_model.knee import knee_method
from autostop.tar_model.scal import scal_method
from autostop.tar_model.score_distribution import score_distribion_training_fitting, score_distribion_feedback_uniform
from autostop.tar_model.target import target_method
from autostop.tar_model.auto_stop import autostop_method, autostop_for_large_collection


FLAGS = None


def main():
    # print all parameter settings
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

    model = FLAGS.model

    # data
    data_name = FLAGS.data_name
    topic_set = FLAGS.topic_set
    topic_id = FLAGS.topic_id

    query_file = FLAGS.query_file
    qrel_file = FLAGS.qrel_file
    doc_id_file = FLAGS.doc_id_file
    doc_text_file = FLAGS.doc_text_file

    stopping_percentage = FLAGS.stopping_percentage
    stopping_recall = FLAGS.stopping_recall
    stopping_condition = FLAGS.stopping_condition
    target_recall = FLAGS.target_recall

    # ranker
    ranker_tfidf_corpus_files = FLAGS.ranker_tfidf_corpus_files
    classifier = FLAGS.classifier
    min_df = FLAGS.min_df
    C = FLAGS.C

    # knee
    rho = FLAGS.rho
    stopping_beta = FLAGS.stopping_beta

    # target
    sample_percentage = FLAGS.sample_percentage
    target_rel_num = FLAGS.target_rel_num

    # score distribution
    training_query_files = FLAGS.training_query_files
    training_qrel_files = FLAGS.training_qrel_files
    training_doc_id_files = FLAGS.training_doc_id_files
    training_doc_text_files = FLAGS.training_doc_text_files

    # scal
    sub_percentage = FLAGS.sub_percentage
    bound_bt = FLAGS.bound_bt
    max_or_min = FLAGS.max_or_min
    bucket_type = FLAGS.bucket_type
    ita = FLAGS.ita

    # autostop
    sampler_type = FLAGS.sampler_type
    epsilon = FLAGS.epsilon
    beta = FLAGS.beta

    # autostop_large
    query_files = FLAGS.query_files
    qrel_files = FLAGS.qrel_files
    doc_id_files = FLAGS.doc_id_files
    doc_text_files = FLAGS.doc_text_files

    random_state = FLAGS.random_state

    np.random.seed(random_state)

    if 'autotar' == model:
        autotar_method(data_name=data_name, topic_set=topic_set, topic_id=topic_id,
                       query_file=query_file, qrel_file=qrel_file, doc_id_file=doc_id_file, doc_text_file=doc_text_file,
                       stopping_percentage=stopping_percentage, stopping_recall=stopping_recall,
                       ranker_tfidf_corpus_files=ranker_tfidf_corpus_files, classifier=classifier, min_df=min_df, C=C,
                       random_state=random_state)

    elif 'knee' == model:
        knee_method(data_name=data_name, topic_set=topic_set, topic_id=topic_id,
                    query_file=query_file, qrel_file=qrel_file, doc_id_file=doc_id_file, doc_text_file=doc_text_file,
                    stopping_beta=stopping_beta, stopping_percentage=stopping_percentage, stopping_recall=stopping_recall,
                    rho=rho,
                    random_state=random_state)

    elif 'scal' == model:
        scal_method(data_name=data_name, topic_set=topic_set, topic_id=topic_id,
                    query_file=query_file, qrel_file=qrel_file, doc_id_file=doc_id_file, doc_text_file=doc_text_file,
                    stopping_percentage=stopping_percentage, stopping_recall=stopping_recall, target_recall=target_recall,
                    sub_percentage=sub_percentage, bound_bt=bound_bt, max_or_min=max_or_min, bucket_type=bucket_type, ita=ita,
                    random_state=random_state)

    elif 'sdtf' == model:
        score_distribion_training_fitting(
            data_name=data_name, topic_set=topic_set, topic_id=topic_id,
            query_file=query_file, qrel_file=qrel_file, doc_id_file=doc_id_file, doc_text_file=doc_text_file,
            training_query_files=training_query_files, training_qrel_files=training_qrel_files,
            training_doc_id_files=training_doc_id_files, training_doc_text_files=training_doc_text_files,
            target_recall=target_recall,
            random_state=random_state)

    elif 'sdfu' == model:
        score_distribion_feedback_uniform(
            data_name=data_name, topic_set=topic_set, topic_id=topic_id,
            query_file=query_file, qrel_file=qrel_file, doc_id_file=doc_id_file, doc_text_file=doc_text_file,
            sample_percentage=sample_percentage, target_recall=target_recall,
            random_state=random_state)

    elif 'target' == model:
        target_method(data_name=data_name, topic_set=topic_set, topic_id=topic_id,
            query_file=query_file, qrel_file=qrel_file, doc_id_file=doc_id_file, doc_text_file=doc_text_file,
            stopping_percentage=stopping_percentage, stopping_recall=stopping_recall, target_rel_num=target_rel_num,
            random_state=random_state)

    elif 'autostop' == model:
        autostop_method(data_name=data_name, topic_set=topic_set, topic_id=topic_id,
            query_file=query_file, qrel_file=qrel_file, doc_id_file=doc_id_file, doc_text_file=doc_text_file,
            sampler_type=sampler_type, epsilon=epsilon, beta=beta,
            stopping_percentage=stopping_percentage, stopping_recall=stopping_recall, target_recall=target_recall, stopping_condition=stopping_condition,
            random_state=random_state)

    elif 'autostoplarge' == model:
        autostop_for_large_collection(data_name=data_name, topic_set=topic_set, topic_id=topic_id,
            query_files=query_files, qrel_files=qrel_files, doc_id_files=doc_id_files, doc_text_files=doc_text_files,
            sampler_type=sampler_type, epsilon=epsilon, beta=beta,
            stopping_percentage=stopping_percentage, stopping_recall=stopping_recall, target_recall=target_recall, stopping_condition=stopping_condition,
            random_state=random_state)

    else:
        print('model {} is not supported!'.format(model))
        return

    return



if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='', help='autostop, knee, target, autotar, scal, sdtf, sdfu.')
    parser.add_argument('--data_name', type=str, default='clef2017', help='clef2017, clef2018, clef2019, tr, legal.')
    parser.add_argument('--topic_set', type=str, default='', help='indicating which topic set or test set.')
    parser.add_argument('--topic_id', type=str, default='', help='')

    parser.add_argument('--query_file', type=str, default='', help='')
    parser.add_argument('--qrel_file', type=str, default='', help='')
    parser.add_argument('--doc_id_file', type=str, default='', help='')
    parser.add_argument('--doc_text_file', type=str, default='', help='')

    # parameter for TAR
    parser.add_argument('--stopping_percentage', type=float, default=1.0, help='stop the TAR process when x percentage of documents have been screened.')
    parser.add_argument('--stopping_recall', type=float, default=None, help='for debug.')
    parser.add_argument('--stopping_condition', type=str, default=None, help='parameter for the autostop method.')
    parser.add_argument('--target_recall', type=float, default=None)

    # parameter for the ranking module
    parser.add_argument('--ranker_tfidf_corpus_files', type=list, default=[],
                        help='Corpus used to construct document tfidf features')
    parser.add_argument('--classifier', type=str, default='lr', help='classifier')
    parser.add_argument('--min_df', type=int, default=2)
    parser.add_argument('--C', type=float, default=1.0)

    # parameter for knee
    parser.add_argument('--rho', type=str, default='', help='')
    parser.add_argument('--stopping_beta', type=float, default=100)

    # parameter for target
    parser.add_argument('--target_rel_num', type=int, default=10, help='')

    # parameter for score distribution
    parser.add_argument('--training_topics', type=str, default='', help='')
    parser.add_argument('--sample_percentage', type=float, default=0.1, help='')
    parser.add_argument('--sdtfclassic_alpha', type=float, default=1, help='')
    parser.add_argument('--which_collection', type=str, default='', help='')

    # parameter for scal
    parser.add_argument('--sub_percentage', type=float, default=1.0, help='')
    parser.add_argument('--bound_bt', type=int, default=30, help='')
    parser.add_argument('--max_or_min', type=str, default='min', help='')
    parser.add_argument('--bucket_type', type=str, default='samplerel', help='')
    parser.add_argument('--ita', type=float, default=1.05, help='')

    # parameter for sampling distribution
    parser.add_argument('--epsilon', type=float, default=0.5, help='mixture uniform, weight for top documents: 0.1, ..., 0.9')
    parser.add_argument('--beta', type=float, default=-0.1, help='power law: 0.01, 0.1, 1, 2')

    parser.add_argument('--random_state', type=int, default=0, help='random seed')
    FLAGS, unparsed = parser.parse_known_args()

    main()

    pass
