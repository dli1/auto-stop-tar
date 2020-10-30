# coding=utf-8

import json
import copy
import numpy as np
from collections import defaultdict
from autostop.tar_framework.utils import *


class DataLoader(object):
    def __init__(self, query_file, qrel_file, doc_id_file, doc_text_file):
        """
        Load data.
        @param query_file: e.g. {"id": 1, "query": , "title": }
        @param qrel_file: TREC qrel format, each line is an example, e.g. qid 0 did 1
        @param doc_id_file: stores the list of document ids from which to run the TAR process, each line is an id
        @param doc_text_file: stores the texts of document ids in doc_id_file, each line is a json format string, e.g. {"id": 1, "title": , "content": }
        """
        self.title = self.read_title(query_file)
        self.did2label = self.read_qrels(qrel_file)
        self.dids = self.read_doc_ids(doc_id_file)
        self.did2text = self.read_doc_texts(doc_text_file)

        self.pseudo_did = 'pseudo_did'
        self.pseudo_text = self.title
        self.pseudo_label = REL

        LOGGER.info('{} DataLoader.__init__ is done.'.format(os.path.basename(query_file)))

    @staticmethod
    def read_title(query_file):
        with open(query_file, 'r', encoding='utf8') as f:
            entry = json.loads(f.read())
            return entry['title']

    @staticmethod
    def read_qrels(qrel_file):
        dct = {}
        with open(qrel_file, 'r', encoding='utf8') as f:
            for line in f:
                if len(line.split()) != 4:
                    continue
                topic_id, dummy, doc_id, rel = line.split()
                dct[doc_id] = int(rel)
        return dct

    @staticmethod
    def read_doc_ids(doc_id_file):
        dids =[]
        with open(doc_id_file, 'r', encoding='utf8') as f:
            for line in f:
                dids.append(line.strip())
        return dids

    @staticmethod
    def read_doc_texts(doc_text_file):
        dct = {}
        with open(doc_text_file, 'r', encoding='utf8') as f:
            for line in f:
                # entry = {'id': doc_id, 'title': subject, 'content': content}
                entry = json.loads(line)
                doc_id = entry['id']
                text = entry['title'] + ' ' + entry['content']
                dct[doc_id] = text
        return dct

    @staticmethod
    def read_doc_texts_2_list(doc_text_file):
        """
        Only used in autotar_method.
        @param doc_text_file:
        @return:
        """
        with open(doc_text_file, 'r', encoding='utf8') as f:
            for line in f:
                entry = json.loads(line)
                text = entry['title'] + ' ' + entry['content']
                yield text

    def get_title(self):
        return self.title

    def get_did2label(self):
        return self.did2label

    def get_complete_pseudo_dids(self):
        return self.dids + [self.pseudo_did]

    def get_complete_pseudo_texts(self):
        return [self.did2text[did] for did in self.dids] + [self.pseudo_text]

    def get_complete_dids(self):
        return self.dids

    def get_complete_texts(self):
        return [self.did2text[did] for did in self.dids]

    def get_complete_labels(self):
        return [self.did2label[did] for did in self.dids]

    def get_rel_label(self, did):
        return self.did2label[did]

    def get_total_doc_num(self):
        return len(self.dids)

    def get_total_rel_num(self):
        return len(list(filter(lambda did: self.did2label[did] == REL, self.dids)))


class Assessor(DataLoader):
    """
    Manager the assessment module of the TAR framework.
    """
    def __init__(self,query_file, qrel_file, doc_id_file, doc_text_file):
        super().__init__(query_file, qrel_file, doc_id_file, doc_text_file)

        self.assessed_dids = []
        self.unassessed_dids = copy.copy(self.did2label)
        self.assess_state = defaultdict(lambda: False)

    def get_training_data(self, temp_doc_num):
        """
        Provide training data for training ranker
        :param type:
        :return:
        """
        asdids = self.get_assessed_dids()

        population = self.get_unassessed_dids()
        temp_doc_num = min(len(population), temp_doc_num)
        temp_dids = list(np.random.choice(a=population, size=temp_doc_num, replace=False))  # unique random elements

        dids = [self.pseudo_did] + asdids + temp_dids
        labels = [self.pseudo_label] + [self.did2label[did] for did in asdids] + len(temp_dids)*[0]

        assert len(dids) == len(labels)
        return dids, labels

    def update_assess(self, dids):

        for did in dids:
            if self.assess_state[did] is False:
                self.assessed_dids.append(did)
                self.unassessed_dids.pop(did)
                self.assess_state[did] = True
        return

    def assess_state_check_func(self):
        def func(did):
            return self.assess_state[did]
        return func

    def get_top_assessed_dids(self, ranked_dids, b):
        cnt = 0
        top_dids = []
        for did in ranked_dids:
            if self.assess_state[did] is False:
                top_dids.append(did)
                cnt += 1
            if cnt >= b:
                break
        return top_dids

    def get_assessed_dids(self):
        return self.assessed_dids

    def get_assessed_num(self):
        return len(self.assessed_dids)

    def get_assessed_rel_dids(self):
        as_dids = self.get_assessed_dids()
        return list(filter(lambda did: self.did2label[did] == REL, as_dids))

    def get_assessed_rel_num(self):
        as_dids = self.get_assessed_dids()
        return len(list(filter(lambda did: self.did2label[did] == REL, as_dids)))

    def get_unassessed_dids(self):
        return list(self.unassessed_dids.keys())

    def get_unassessed_num(self):
        return len(self.unassessed_dids.keys())

    def get_assessed_state(self):
        return self.assess_state


if __name__ == '__main__':
    topic_id = 'CD007394'
    mdir = '/Users/danli/Documents/Project/autostop/release_version/data/clef2017'
    query_file = os.path.join(mdir, 'topics', topic_id)
    qrel_file = os.path.join(mdir, 'qrels', topic_id)
    doc_id_file = os.path.join(mdir, 'docids', topic_id)
    doc_text_file = os.path.join(mdir, 'doctexts', topic_id)

    data = DataLoader(query_file, qrel_file, doc_id_file, doc_text_file)
    pass