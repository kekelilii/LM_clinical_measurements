# LM_clinical_measurements
Code and datasets for paper "Finetuning on Blood Pressure Classification Helps Language Models Answer Medical Questions"

```
pip install -r requirements.txt
```

## Pre-Finetuning Evaluation
For BERT and BioBERT:
```
fillmask_zeroshot.py
```
For BioMedLM:
```
textgeneration_zeroshot.py
```
We used the [OPENAI Completions API](https://platform.openai.com/docs/api-reference/completions) for the pre-fine-tuning evalution on GPT3.5 Davinci-002

## Finetune
Finetuning on the BP Dataset
```
mc_bp_finetune.py
```
Finetuning on the MedQA Dataset and Testing
```
mc_medqa_finetune.py
mc_medqa_test.py
```

We used the ```GPT2ForMultipleChoice``` class in ```custom_modeling_gpt2.py``` for the finetuning on [BioMedLM](https://github.com/stanford-crfm/BioMedLM/tree/main). Please refer to their repo for more details.


