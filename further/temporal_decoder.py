from sklearn.metrics.pairwise import cosine_distances
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import json
import re
import pickle
# import optuna
import torch
from operator import itemgetter
import heapq
import math
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader
import nltk
import evaluate
import datasets
from datasets import load_dataset, Dataset, DatasetDict,load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq,TFMT5ForConditionalGeneration, MT5Tokenizer, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, TrainingArguments, Trainer, T5Tokenizer, T5ForConditionalGeneration
import os
import sklearn
from spacy.matcher import Matcher
from spacy.util import filter_spans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import textacy
import seaborn as sns
import hdbscan
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from TemporalGraph import *
from collections import Counter

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

class TemporalDecoder:
    def __init__(self, model, tokenizer, embedder_model, clusters: dict, temporal_graph):
        self.model = model  # main model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder_model = embedder_model
        self.prefix = ""  # TODO: RACHIT, add this!
        self.midfix = ""
        self.verb2cluster = clusters["verb2cluster"]
        self.cluster2verb = cluster["cluster2verb"]
        self.centroids = clusters["centroids"]
        self.temporal_graph = temporal_graph

    def extract_clean_verb_phrases(self, doc)->list[str]:
        """
        Given a spaCy Doc, returns a list of (verb_text, verb_phrase) tuples,
        where verb_phrase includes auxiliaries, direct objects, and prep-objects.
        
        :param doc: spaCy Doc
        :return: list of pairs of (verb, verb_phrase) present in the argument sentence 
        """

        results = []
        for verb in doc:
            if verb.pos_ !="VERB":
                continue

            # 1) auxiliaries (aux, auxpass)
            aux_tokens = [child for child in verb.children if child.dep_ in ("aux", "auxpass")]
            # 2) direct objects (dobj, obj)
            dobj_tokens = [child for child in verb.children if child.dep_ in ("dobj", "obj")]
            # 3) prepositional objects: find prep → pobj
            pobj_tokens = []
            for prep in [c for c in verb.children if c.dep_ == "prep"]:
                pobj_tokens.extend([c for c in prep.children if c.dep_ == "pobj"])

            # 4) assemble and sort by document order
            tokens = aux_tokens + [verb] + dobj_tokens + pobj_tokens
            tokens = sorted(set(tokens), key=lambda t: t.i)

            # 5) create phrase string
            phrase = " ".join(t.text for t in tokens)

            if len(phrase)>1 and len(verb.text)>1:  # filter out apostrophe-words
                results.append((verb.text, phrase))

        return results

    def get_cluster(self, verb: str, returnCluster=False):
        """
            If returnCluster is True then return cluster in which the string is closest to
            Else, return all of the cluster contents
            
            :param verb: input verb
            :return Optional[Int, List[Str]]
        """
        
        new_embedding = self.embedder_model.encode(verb)  # shape (1, embedding_dim)
        assigned_cluster = int(np.argmin(cosine_distances(new_embedding.reshape(1, -1), self.centroids)))  # find the cluster of the verb
        return assigned_cluster if returnCluster else self.cluster2verb[str(assigned_cluster)]  # pick all the verbs in the cluster

    def include_next_verbs(self, sentence: str)->list[str]:
        """
            Give an verb from a dialogue, returns the next possible verbs (events) to respond with
            
            :param sentence: input dialogue
            :return: List[Str]
        """

        if not isinstance(sentence, str):
            raise Exception("Input has to be a str")

        doc = self.nlp(sentence)  # tokenize
        vp_pairs = self.extract_clean_verb_phrases(doc)  # collect the verb and phrase
        verbs, phrases = zip(*vp_pairs)

        next_possible_verbs = []
        for i in range(len(verbs)):  # for each verb in the sentence
            phrase_embed = embedder_model.encode(phrases[i])
            similar_verbs = self.get_cluster(verbs[i])  # get the cluster for the verb

            for verb in similar_verbs:  # for each similar verb
                try:
                    nodes = self.temporal_graph.get_event_nodes(verb)  #   find its nodes
                except Exception:  # given event doesn't exist in the graph
                    continue
                else:
                    #   use their avg embedding and compare them with phrase embedding
                    #   find top1 wrt cosine similarity
                    best_node = nodes[np.argmin([cosine_similarity(phrase_embed, node.avg_embedding) for node in nodes])]

                    # get the next verbs for the top1 node
                    next_verbs = self.temporal_graph.get_next_event(best_node)
                    # we have multiple verbs now

                    # find the cluster from which max no of verbs exist
                    common_cluster = most_frequent([self.get_cluster(next_verb.event, returnCluster=True) for next_verb in next_verbs])

                    # pick the top 5 from the cluster
                    next_possible_verbs += list(zip(*heapq.nlargest(5, enumerate(self.cluster2verb[str(common_cluster)]), key=itemgetter(1))))[1]
        return next_possible_verbs

    def preprocess(self, batch):
        """
            Preprocessing of the Dataset before training
            :param batch: set of data
            :return: tokenized input to the model
        """
        
        inputs = []
        for doc in batch['Input']:
            dialogue = eval(doc)  # doc is in str format, due to read from csv
            total_dialogue = "\n".join(dialogue)
            next_verbs = self.include_next_verbs(dialogue[-1])  # include only the last sentence in the dialogue for next events
            inputs.append(self.prefix + total_dialogue + self.midfix + ",".join(next_verbs))
        
        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True,padding='max_length')
        labels = self.tokenizer(text_target=batch["Outputs"], max_length=512, truncation=True, padding='max_length')
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prompt(self, user_prompt:str)->str:
        """
            Talk with the model
            :param user_prompt: user input
            :return: model response
        """
        
        if not isinstance(user_prompt, str):
            raise Exception("User input has to be string type")
        
        next_verbs = self.include_next_verbs(user_prompt)
        input_prompt = self.prefix + user_prompt + self.midfix + ",".join(next_verbs)
        inputs = self.tokenizer(input_prompt, return_tensors="pt")  # Tokenize the text and generate output
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the output tokens to text
        return response

