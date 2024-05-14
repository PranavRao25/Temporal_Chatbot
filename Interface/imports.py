import spacy
import shutil
import torch
from datetime import datetime
import shutil
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration
from transformers import GenerationConfig
import timexy
import os
from timexy import Timexy,rule
from timexy.languages import en

# Imports

nlp = spacy.load("en_core_web_sm")

# Optionally add config if varying from default values
config = {
    "kb_id_type": "timex3",  # possible values: 'timex3'(default), 'timestamp'
    "label": "timexy",       # default: 'timexy'
    "overwrite": False       # default: False
}
nlp.add_pipe("timexy", config=config, before="ner")
