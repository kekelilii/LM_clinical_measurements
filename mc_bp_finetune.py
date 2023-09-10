
#Define model name(or path for BP finetuned models), and output dir
model_path = "dmis-lab/biobert-base-cased-v1.2" 
out_dir="biobert_bp_finetune"

#Load packages
import torch
import transformers
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer,AutoConfig,AutoModelForMultipleChoice
import evaluate
import accelerate
import random
import numpy as np
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
from collections import Counter
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction
from scipy.special import softmax

import sys
sys.path.insert(0, '..')
from custom_modeling_gpt2 import GPT2ForMultipleChoice

# Data preprocess
### YOU CAN CAHNGE THE PROMPT TO YOUR OWN STYLE
def process_df(fpath):
  datasets = pd.read_csv(fpath)
  datasets['ending0']='low'
  datasets['ending1']='normal'
  datasets['ending2']='high'
  datasets['sent1'] = ""
  datasets['sent2'] = ""
  for i in range(len(datasets['labels_n'])):
    s_str = str(datasets['s'][i])  # Convert to string
    d_str = str(datasets['d'][i])  # Convert to string
    ######## CHANGE PROMPT HERE
    datasets['sent1'][i] = "Blood pressure(mm Hg):"+ s_str + "/" + d_str + "."
    datasets['sent2'][i] = "The patient's blood pressure is"
  datasets = datasets.drop(columns=['Unnamed: 0'])
  datasets = datasets.drop(columns=['labels'])
  datasets= datasets.rename(columns={'labels_n': 'label'})
  datasets = datasets.drop(columns=['s'])
  datasets = datasets.drop(columns=['d'])
  return datasets

from datasets import Dataset
def con_base(dataset):
  datasets = Dataset.from_pandas(dataset)

  # Print information about the created dataset
  return datasets

train = process_df('data/bp/train.csv')
test = process_df('data/bp/test.csv')
validation = process_df('/data/bp/val.csv')

from collections import Counter

value_counts = Counter(train['label'])

for value, count in value_counts.items():
    print(f"{value}: {count}")

from datasets import DatasetDict
dataset_dict = DatasetDict({
    'train': con_base(train),
    'validation': con_base(validation),
    'test': con_base(test)
})



#Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)

if config.model_type == "gpt2":
    model_class = GPT2ForMultipleChoice
else:
    model_class = AutoModelForMultipleChoice

model = model_class.from_pretrained(model_path)

#Added pad_token_id
if tokenizer.pad_token_id is None:
    print('Adding [PAD] token to tokenizer and model word embeddings.')
    num_added_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'})
    embedding_layer = model.resize_token_embeddings(len(tokenizer))
    config.pad_token_id = tokenizer.pad_token_id

ending_names = ["ending0", "ending1", "ending2"]

def preprocess_function(examples):
    # Repeat each first sentence 3 times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 3 for context in examples["sent1"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["sent2"]
    second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    return {k: [v[i:i+3] for i in range(0, len(v), 3)] for k, v in tokenized_examples.items()}

tokenized_dic = dataset_dict.map(preprocess_function, batched=True)


#DataCollatorForMultipleChoice
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

"""## evaluate function"""
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

from scipy.special import softmax


def compute_metrics(eval_preds):
  logits, label = eval_preds
  torch.save(logits, "logits.pt")
  torch.save(label, "label.pt")
  predictions = np.argmax(logits, axis=1)
  log = torch.from_numpy(logits)
  class_prob = log.softmax(dim=-1)
  result = multi_label_metrics(
        prob= class_prob,
        predictions=predictions,
        labels=label)
  return result


"""## finetune"""
training_args = TrainingArguments(
    output_dir=out_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    seed=42,
    data_seed=42,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dic["train"],
    eval_dataset=tokenized_dic["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

"""## Save model and evaluate
### change save file name everytime
#### eg. my_biobert_model_after_bp_p0
"""
### change saved model name
#trainer.save_model("my_model_after_bp_p0")

random.seed(42)
print(trainer.evaluate(eval_dataset = tokenized_dic["test"] ))