
import transformers
from transformers import GPTJConfig, AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM, GPT2Tokenizer
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import random
random.seed(42)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction

import pickle

#Load dataset
with open('/data/lowercase4gptfinetune.pkl', 'rb') as file:
    bpdict = pickle.load(file)

#Initialize model
device = torch.device("cuda")
tokenizer = GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("stanford-crfm/BioMedLM").to(device)

preds = []
probs = []
#Evaluation using prompt 0
for i in range (len(bpdict['test']['p0']['prompts'])):
  input_ids = tokenizer.encode(
      bpdict['test']['p0']['prompts'].tolist()[i], return_tensors="pt"
  ).to(device)

  output = model.generate(input_ids, max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True)

  new_token = torch.argmax(output.scores[0][0], dim = 0)
  a_token_id = tokenizer.encode(' a', add_special_tokens=False)[0]
  b_token_id = tokenizer.encode(' b', add_special_tokens=False)[0]
  c_token_id = tokenizer.encode(' c', add_special_tokens=False)[0]

  abc_logits = [output.scores[0][0][a_token_id],output.scores[0][0][b_token_id],output.scores[0][0][c_token_id]]
  if new_token == a_token_id:
    pred = 0
  elif new_token == b_token_id:
    pred = 1
  elif new_token == c_token_id:
    pred = 2
  else:
    pred = -1

  preds.append(pred)
  probs.append(abc_logits)

bpdict['test']['p0']['preds'] = preds
bpdict['test']['p0']['logits_abc'] = probs

def multi_label_metrics(predictions, labels, prob):
    f1_macro_average = f1_score(y_true=labels, y_pred=predictions, average='macro')
    f1_micro_average = f1_score(y_true=labels, y_pred=predictions, average='micro')
    roc_auc_weighted = roc_auc_score(y_true = labels, y_score = prob, multi_class='ovo', average = "weighted")
    roc_auc_macro = roc_auc_score(y_true = labels, y_score = prob, multi_class='ovo', average = "macro")
    accuracy = accuracy_score(labels, predictions)
    percision_macro = precision_score(y_true=labels, y_pred=predictions, average='macro')
    percision_micro = precision_score(y_true=labels, y_pred=predictions, average='micro')
    recall_macro = recall_score(y_true=labels, y_pred=predictions, average='macro')
    recall_micro = recall_score(y_true=labels, y_pred=predictions, average='micro')
    # return as dictionary
    metrics = {'accuracy': accuracy,
               'roc_auc_weighted': roc_auc_weighted,
               'roc_auc_macro': roc_auc_macro,
               'f1_micro': f1_micro_average,
               'f1_macro': f1_macro_average,
               'percision_macro': percision_macro,
               'percision_micro': percision_micro,
               'recall_macro': recall_macro,
               'recall_micro': recall_micro,
               }
    return metrics

multi_label_metrics(bpdict['test']['p0']['preds'], bpdict['test']['p0']['labels_n'], F.softmax(torch.tensor(bpdict['test']['p0']['logits_abc'])))


