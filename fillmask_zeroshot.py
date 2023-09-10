
import random
import numpy as np
import pandas as pd
from scipy.special import softmax
random.seed(42)


# Load raw data
sd = pd.read_csv("/data/test.csv")

sd = sd.rename(columns = {'s':'Systolic BP(mmHg)','d':'Diastolic BP(mmHg)','labels':'label'})

#Generate prompts
def bp_generate(input_arr, label_arr, label_n, systolic, diastolic, r):
  systolic = str(systolic)
  diastolic = str(diastolic)
  label = int(label_n)
  r = int(r)
  bp_masked = ["Blood pressure of "+systolic+"/"+diastolic+" mmHg is [MASK].",
               "In the ED, the patient's initial blood pressure was "+systolic+"/"+diastolic+", which was [MASK].",
               "Blood pressure(mm Hg):" +systolic+"/"+diastolic+", [MASK]."]
  if label == 0:
    low_bp_label = ["Blood pressure of "+systolic+"/"+diastolic+" mmHg is low.",
                    "In the ED, the patient's initial blood pressure was "+systolic+"/"+diastolic+", which was low.",
                    "Blood pressure(mm Hg):" +systolic+"/"+diastolic+", low."]
    input_arr.append(bp_masked[r])
    label_arr.append(low_bp_label[r])

  if label == 1:
    normal_bp_label = ["Blood pressure of "+systolic+"/"+diastolic+" mmHg is normal.",
                        "In the ED, the patient's initial blood pressure was "+systolic+"/"+diastolic+", which was normal.",
                        "Blood pressure(mm Hg):" +systolic+"/"+diastolic+", normal."]
    input_arr.append(bp_masked[r])
    label_arr.append(normal_bp_label[r])


  if label == 2:
    high_bp_label = ["Blood pressure of "+systolic+"/"+diastolic+" mmHg is high.",
                        "In the ED, the patient's initial blood pressure was "+systolic+"/"+diastolic+", which was high.",
                        "Blood pressure(mm Hg):" +systolic+"/"+diastolic+", high."]
    input_arr.append(bp_masked[r])
    label_arr.append(high_bp_label[r])

label= sd['labels']
label_n = sd['labels_n']
systolic = sd['s']
diastolic = sd['d']
input_arr =[]
label_arr= []
for i in range(len(label)):
  #Generate the dataset with prompt 0
  bp_generate(input_arr, label_arr, label_n[i], systolic[i], diastolic[i], 0)

sd['prompts']= input_arr
sd['labels'] = label_arr


"""## Initiate models"""


import torch
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, DataCollatorWithPadding
import sys
sys.path.insert(0,'/content/BertForMaskedClassification.py')
from BertForMaskedClassification import BertForMaskedClassification
import accelerate
import transformers
from transformers import TrainingArguments
from transformers import Trainer
#import evaluate

#create a PyTorch dataset from our data.
class BP_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction
import torch


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



def prepare_compute_metrics(val_labels_n):
  def compute_metrics(eval_preds):
    nonlocal val_labels_n
    logits, label = eval_preds
    torch.save(logits, "logits.pt")
    predictions = np.argmax(logits, axis=-1)
    log = torch.from_numpy(logits)
    class_prob = log.softmax(dim=-1)
    result = multi_label_metrics(
        prob= class_prob,
        predictions=predictions,
        labels=val_labels_n)
    return result
  return compute_metrics

def BERT_fillmask_pretrain(weights, outdic, val_inputs, val_labels, val_labels_n):
  random.seed(42)
  #Initiate tokenizer and model
  config = BertConfig.from_pretrained(weights)
  tokenizer = BertTokenizer.from_pretrained(weights)
  model = BertForMaskedClassification(config, tokenizer)
  data_collator =DataCollatorWithPadding (tokenizer=tokenizer)

  inputs_val = tokenizer(
    val_inputs,
    return_tensors='pt',
    max_length = 512,
    truncation = True,
    padding = True)

  labels_val = tokenizer(
    val_labels,
    return_tensors='pt',
    max_length = 512,
    truncation = True,
    padding = True)

  max_seq_length_val = max(inputs_val["input_ids"].size(1), labels_val["input_ids"].size(1))
  # Pad inputs_val
  if inputs_val["input_ids"].size(1) < max_seq_length_val:
      inputs_val["input_ids"] = torch.nn.functional.pad(inputs_val["input_ids"], (0, max_seq_length_val - inputs_val["input_ids"].size(1)), mode='constant', value=0)
      inputs_val["attention_mask"] = torch.nn.functional.pad(inputs_val["attention_mask"], (0, max_seq_length_val - inputs_val["attention_mask"].size(1)), mode='constant', value=0)
      inputs_val["token_type_ids"] = torch.nn.functional.pad(inputs_val["token_type_ids"], (0, max_seq_length_val - inputs_val["token_type_ids"].size(1)), mode='constant', value=0)


  inputs_val['labels'] = torch.where(inputs_val.input_ids == tokenizer.mask_token_id, labels_val["input_ids"], -100)
  print(inputs_val['labels'])
  val_dataset = BP_Dataset(inputs_val, inputs_val['labels'])


  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model.to(device)


  args = TrainingArguments(
      output_dir=outdic,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=1,
      learning_rate=2e-5,
      weight_decay=0.001,
      num_train_epochs=4,
      seed=42,
      data_seed=42,
      evaluation_strategy="epoch"
  )

  trainer = Trainer(
      model,
      data_collator=data_collator,
      tokenizer= tokenizer,
      eval_dataset=val_dataset,
      compute_metrics=prepare_compute_metrics(val_labels_n),
      args = args
  )

  return trainer.evaluate()

val_inputs = sd["prompts"].to_list()
val_labels = sd["labels"].to_list()
val_labels_n = sd["labels_n"].to_list()
out_dir = "biobert_zeroshot"
model_result=BERT_fillmask_pretrain("dmis-lab/biobert-base-cased-v1.2", out_dir ,val_inputs, val_labels, val_labels_n)
print(model_result)


