
#Define data path, model name(path)
data_path = "/data/medqa/general_QAoriginal.csv" #IMPORTANT: For fine-tuning on QA1 and QA2, please follow instructions in the "Preprocess" section to make modifications
model_path = "some fine-tuned model"

#Load packages
import torch
import transformers
from transformers import TrainingArguments, Trainer, AutoModelForMultipleChoice
from transformers import AutoTokenizer, AutoConfig
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


## load the data
genral_QA = pd.read_csv(data_path)

#Preprocess
import re
def extract_last_sentence_and_modify(row):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', row['question']) #For QA1/QA2, row['question'] -> row['replaced_question']
    if len(sentences) > 0:
        last_sentence = sentences[-1]
        modified_question = ' '.join(sentences[:-1])  # Join all sentences except the last one
    else:
        last_sentence = ""
        modified_question = row['question'] #For QA1/QA2, row['question'] -> row['replaced_question']

    return pd.Series([modified_question, last_sentence])

# Apply the function and create new columns
genral_QA[['question', 'last_sentence']] = genral_QA.apply(extract_last_sentence_and_modify, axis=1) #For QA1/QA2, genral_QA[['question', 'last_sentence']] -> genral_QA[['replaced_question', 'last_sentence']]

import ast
def extract_and_parse_options(options):
  options = str(options)
  options_dict = ast.literal_eval(options)
  keys = list(options_dict.keys())
  values = list(options_dict.values())
  return keys, values

# Apply the function
genral_QA['keys'], genral_QA['values'] = zip(*genral_QA['options'].apply(extract_and_parse_options))

# Create new columns for options
for i in range(4):
    genral_QA['ending' + str(i)] = genral_QA['values'].apply(lambda values: values[i] if i < len(values) else '')

# Function to assign labels based on answer matching ending values
def assign_label(row):
    for i in range(4):
        if row['answer'] == row['ending' + str(i)]:
            return i
    return -1  # Return -1 if no match is found

# Apply the label assignment function
genral_QA['label'] = genral_QA.apply(assign_label, axis=1)

# Columns to drop
columns_to_drop = ['answer', 'options','meta_info','answer_idx','metamap_phrases', 'keys','values'] #For QA1/QA2, add "question", "replacement_made" to the columns

# Drop the specified columns
genral_QA = genral_QA.drop(columns=columns_to_drop, axis=1)

genral_QA =  genral_QA.rename(columns={'question': 'sent1'}) #For QA1/QA2, columns={'question': 'sent1'} -> columns={'replaced_question': 'sent1'}
genral_QA =  genral_QA.rename(columns={'last_sentence': 'sent2'})

#Split train/test/val
from sklearn.model_selection import train_test_split
train_general_w, temp_data = train_test_split(genral_QA, test_size=0.2, random_state=42)
validation_general_w, test_general_w = train_test_split(temp_data, test_size=0.5, random_state=42)

#### IF NEED TO SAVE, RUN
#train_general_w.to_csv("/home/ka/Downloads/train_general_w.csv", index=False)
#validation_general_w.to_csv("/home/ka/Downloads/validation_general_w.csv", index=False)
#test_general_w.to_csv("/home/ka/Downloads/test_general_w.csv", index=False)

"""## preprocess the data"""

def con_base(dataset):
  datasets = Dataset.from_pandas(dataset)
  return datasets

dataset_dict_general_w = DatasetDict({
    'train': con_base(train_general_w),
    'validation': con_base(validation_general_w),
    'test': con_base(test_general_w)
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

ending_names = ["ending0", "ending1", "ending2", "ending3"]

def preprocess_function(examples):
    # Repeat each first sentence 3 times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["sent2"]
    second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, padding = 'max_length', max_length = 512)
    # Un-flatten
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

tokenized_dic_general_w = dataset_dict_general_w.map(preprocess_function, batched=True)

#Define training_args for evaluation
training_args = TrainingArguments(
    #### out dir
    output_dir="model_after_QA_finetune",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    seed=42,
    data_seed=42,
    weight_decay=0.01,
    #fp16=True,
)


trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset=tokenized_dic_general_w["train"],
    eval_dataset=tokenized_dic_general_w["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

random.seed(42)
print(trainer.evaluate(eval_dataset = tokenized_dic_general_w["test"] ))
