from abc import ABC
from collections import defaultdict
import logging

import numpy as np
import stanza

class LinguisticFeatures(ABC):
    def __init__(self):
        self.nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse,constituency', use_gpu=False, tokenize_pretokenized=True)

    @staticmethod
    def _get_feat(word, feat):
        if word is None:
            return None
        if word.feats is None:
            return None
        feats = {feat.split("=")[0]: feat.split("=")[1] for feat in word.feats.split("|")}
        if feat not in feats:
            return None
        return feats[feat]

    @staticmethod
    def _get_word_depth(parsed_sent):
        tree = defaultdict(list)
        word_depth = [0 for i in range(len(parsed_sent.words) + 1)]
        for word in parsed_sent.words:
            tree[word.head].append(word.id)
        word_depth[tree[0][0]] = 0
        def count_words_depth(root_id):
            for word in tree[root_id]:
                word_depth[word] = word_depth[root_id] + 1
                count_words_depth(word)
        count_words_depth(tree[0][0])
        return word_depth

    @staticmethod
    def _get_subj_number(parsed_sent, word_depth):
        potential_subj = []
        for word in parsed_sent.words:
            if word.deprel == "nsubj":
                potential_subj.append(word)
        if len(potential_subj) == 0:
            return None
        subj = sorted(potential_subj, key=lambda subj: word_depth[subj.id])[0]
        return LinguisticFeatures._get_feat(subj, "Number")

    @staticmethod
    def _get_obj_number(parsed_sent, word_depth):
        potential_obj = []
        for word in parsed_sent.words:
            if word.deprel == "obj":
                potential_obj.append(word)
        if len(potential_obj) == 0:
            return None
        obj = sorted(potential_obj, key=lambda obj: word_depth[obj.id])[0]
        return LinguisticFeatures._get_feat(obj, "Number")

    @staticmethod
    def _get_verb_tense(parsed_sent, word_depth):
        potential_verbs = []
        for word in parsed_sent.words:
            feat = LinguisticFeatures._get_feat(word, "Tense")
            if feat is not None:
                potential_verbs.append(word)
        if len(potential_verbs) == 0:
            return None
        verb = sorted(potential_verbs, key=lambda vb: word_depth[vb.id])[0]
        return LinguisticFeatures._get_feat(verb, "Tense")

    @staticmethod
    def _get_20_class(consts):
        top_const = ("ADVP_NP_VP_.", "CC_ADVP_NP_VP_.", "CC_NP_VP_.", "IN_NP_VP_.", "NP_ADVP_VP_.", "NP_NP_VP_.", "NP_PP_.",
                                           "NP_VP_.", "PP_NP_VP_.", "RB_NP_VP_.", "SBAR_NP_VP_.", "SBAR_VP_.", "S_CC_S_.", "S_NP_VP_",
                                           "S_VP_.", "VBD_NP_VP_.", "VP_.", "WHADVP_SQ_.", "WHNP_SQ_.")
        if "_".join(consts) not in top_const:
            logging.warning(f"{consts}, {'_'.join(consts)} not in the list of top constituents")
            return len(top_const)
        else:
            return {c : i for i, c in enumerate(top_const)}["_".join(consts)]


    @staticmethod
    def get_features_for_sent(parsed_sent):
        word_depth =  LinguisticFeatures._get_word_depth(parsed_sent)
        subj_num = LinguisticFeatures._get_subj_number(parsed_sent, word_depth)
        obj_num = LinguisticFeatures._get_obj_number(parsed_sent, word_depth)
        vb_tense = LinguisticFeatures._get_verb_tense(parsed_sent, word_depth)

        return {
            "random": np.random.randint(0, 2),
            "sentence_length": len(parsed_sent.words),
            "tree_depth": max(word_depth),
            "top_constituents": LinguisticFeatures._get_20_class(tuple([str(child.label) for child in parsed_sent.constituency.children[0].children])),
            "tense": 0 if str(vb_tense) == "Past" else (1 if str(vb_tense) == "Pres" else None),
            "subject_number": 0 if str(subj_num) == "Sing" else (1 if str(subj_num) == "Plur" else None),
            "object_number": 0 if str(obj_num) == "Sing" else (1 if str(obj_num) == "Plur" else None),
        }
    
    def get_features_per_sent(self, dataset):
        text_tokenized = [item["tokens"] for item in dataset]
        parsed_text = self.nlp(text_tokenized)
        features_list = []
        for sent in parsed_text.sentences:
            features_list.append(LinguisticFeatures.get_features_for_sent(sent))
        return features_list
        

