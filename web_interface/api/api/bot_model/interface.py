# Imports
import numpy as np  # linear algebra
import numpy.random
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import shutil
import json
import re
import torch
import datetime
import nltk
import evaluate
import shutil
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, TFMT5ForConditionalGeneration, MT5Tokenizer, \
    AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
import timexy
import os
from timexy import Timexy, rule
from timexy.languages import en

nlp = spacy.load("en_core_web_sm")

# Optionally add config if varying from default values
config = {
    "kb_id_type": "timex3",  # possible values: 'timex3'(default), 'timestamp'
    "label": "timexy",  # default: 'timexy'
    "overwrite": False  # default: False
}
nlp.add_pipe("timexy", config=config, before="ner")

from datetime import datetime

import sys
class Interface:
    def __init__(self):
        for path in sys.path:
            print(path)
        self.num = numpy.random.random()
        self.file_name = f"/Users/rachitsandeepjain/PycharmProjects/temporal_chatbot_backend/api/api/bot_model/temp_files/temp_{numpy.random.random()}.txt"
        print("pranav", self.file_name)
        self.f = open(self.file_name, 'w')
        self.f.write('Generate: ')
        self.f.close()

        self.f = open(self.file_name, 'r+')
        self.f.seek(0, 2)
        self.model = T5ForConditionalGeneration.from_pretrained("/Users/rachitsandeepjain/PycharmProjects/temporal_chatbot_backend/api/api/bot_model/model")
        self.tokenizer = T5Tokenizer.from_pretrained("/Users/rachitsandeepjain/PycharmProjects/temporal_chatbot_backend/api/api/bot_model/tokenizer")

    def closeInterface(self):
        self.f.close()
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        else:
            print("The file does not exist")

    # api/api/bot_model/temp_files
    # api/api/views.py
    def printTranscript(self):
        self.f.close()

        news = f"/Users/rachitsandeepjain/PycharmProjects/temporal_chatbot_backend/api/api/bot_model/printed_files/Conversation_{numpy.random.random()}.txt"
        file = open(news, 'w')
        file.close()

        shutil.copy(self.file_name, news)
        self.f = open(self.file_name, 'r+')

    def userInput(self, inputLine):
        input = inputLine + '\n'
        self.f.write(input)

        # Tokenize the text and generate output
        self.f.seek(0)
        mod_input = self.f.read() + '\n'
        inputs = self.tokenizer(mod_input, return_tensors="pt")
        outputs = self.model.generate(**inputs)

        # Decode the output tokens to text
        model_result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = model_result + '\n'
        self.f.write(result)
        self.f.seek(0,2)
        return result

    def displayTranscript(self):
        self.f.seek(0)
        l = self.f.readlines()
        for i in l:
            print(i)

    def clearTranscript(self):
        self.f.close()
        with open(self.file_name, 'w') as _:
            pass
        self.f = open(self.file_name, "r+")
