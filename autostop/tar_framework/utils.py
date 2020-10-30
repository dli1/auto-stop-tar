# coding=utf-8

import os
import pandas as pd

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RET_DIR = os.path.join(PARENT_DIR, 'ret')
DATA_DIR = os.path.join(PARENT_DIR, 'data')

INF = 1e+10

REL = 1
NONREL = 0



# 12 topics
CLEF_2017_TRAINING_TOPICS = [
'CD008686',
'CD009593',
'CD011548',
'CD009372',
'CD008803',
'CD009323',
'CD008691',
'CD010542',
'CD009944',
'CD008760',
'CD009185',
'CD009925'
]

# 30 topics
CLEF_2017_TEST_TOPICS = [
'CD008081',
'CD007394',
'CD007427',
'CD008054',
'CD008643',
'CD008782',
'CD009020',
'CD009135',
'CD009519',
'CD009551',
'CD009579',
'CD009591',
'CD009647',
'CD009786',
'CD010023',
'CD010173',
'CD010276',
'CD010339',
'CD010386',
'CD010409',
'CD010438',
'CD010632',
'CD010633',
'CD010653',
'CD010705',
'CD011134',
'CD011549',
'CD011975',
'CD011984',
'CD012019',
]


CLEF_2018_TOPICS = [
'CD008122',
'CD008587',
'CD008759',
'CD008892',
'CD009175',
'CD009263',
'CD009694',
'CD010213',
'CD010296',
'CD010502',
'CD010657',
'CD010680',
'CD010864',
'CD011053',
'CD011126',
'CD011420',
'CD011431',
'CD011515',
'CD011602',
'CD011686',
'CD011912',
'CD011926',
'CD012009',
'CD012010',
'CD012083',
'CD012165',
'CD012179',
'CD012216',
'CD012281',
'CD012599'
]

CLEF_2019_TOPICS = [
'CD006468',
'CD000996',
'CD001261',
'CD004414',
'CD007867',
'CD009069',
'CD009642',
'CD010038',
'CD010239',
'CD010558',
'CD010753',
'CD011140',
'CD011571',
'CD011768',
'CD011977',
'CD012069',
'CD012164',
'CD012342',
'CD012455',
'CD012551',
'CD012661',
'CD011558',
'CD011787',
'CD008874',
'CD009044',
'CD011686',
'CD012080',
'CD012233',
'CD012567',
'CD012669',
'CD012768',
]


LEGAL_TRAINING_TOPICS = [
'301',
'302',
]

LEGAL_TEST_TOPICS = [
'303',
'304'
]

TR_TRAINING_TOPICS = [
"athome100",
"athome101",
"athome102",
"athome103",
"athome104",
"athome105",
"athome106",
"athome107",
"athome108",
"athome109",
]

TR_TEST_TOPICS = [
'401',
'402',
'403',
'404',
'405',
'406',
'407',
'408',
'409',
'410',
'411',
'412',
'413',
'414',
'415',
'416',
'417',
'418',
'419',
'420',
'421',
'422',
'423',
'424',
'425',
'426',
'427',
'428',
'429',
'430',
'431',
'432',
'433',
'434',
]


def get_file_ids(path):
    """Get all file names in path."""
    file_ids = []
    for root, dirs, files in os.walk(path):
        file_ids.extend(files)
    file_ids = [f for f in file_ids if not f.startswith('.')]

    return file_ids


def check_path(mdir):
    if not os.path.exists(mdir):
        os.makedirs(mdir)


def name_tar_run_file(data_name, model_name, topic_set, exp_id, topic_id):
    mdir = os.path.join(RET_DIR, data_name, 'tar_run', model_name, topic_set, str(exp_id))
    check_path(mdir)
    fname = topic_id + '.run'
    return os.path.join(mdir, fname)


def write_tar_run_file(f, topic_id, check_func, shown_dids):
    """

    "Three types of interactions are supported:
    # INTERACTION = AF, relevance feedback is used by the system to compute the ranking of subsequent documents
    # INTERACTION = NF, relevance feedback is not being used by the system
    # INTERACTION = NS, the document is not shown to the user (these documents can be excluded from the output"

    See CLEF-TAR 2017 (https://sites.google.com/site/clefehealth2017/task-2).

    @param f:
    @param topic_id:
    @param check_func:
    @param shown_dids:
    @return:
    """
    for i, did in enumerate(shown_dids):

        if check_func(did) is True:
            screen = 'AF'
        else:
            screen = 'NF'
        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(topic_id, screen, did, i + 1, -i, 'mrun'))
    return


def name_interaction_file(data_name, model_name, topic_set, exp_id, topic_id):
    mdir = os.path.join(RET_DIR, data_name, 'interaction', model_name, topic_set, str(exp_id))
    check_path(mdir)
    fname = topic_id + '.csv'
    return os.path.join(mdir, fname)


def read_interaction_file(data_name, model_name, topic_set, exp_id, topic_id):
    mdir = os.path.join(RET_DIR, data_name, 'interaction', model_name, exp_id, topic_set, str(exp_id))
    filename = topic_id + '.csv'
    df = pd.read_csv(os.path.join(mdir, filename))
    if len(df) == 0:
        print('Empty df', os.path.join(mdir, filename))

    return df
