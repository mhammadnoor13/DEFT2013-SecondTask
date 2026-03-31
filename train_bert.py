import pandas as pd
import numpy as np
import torch
import random

from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)



df = pd.read_csv("processed.csv")

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

train_dataset = dataset["train"]
test_dataset = dataset["test"]



tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)



model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1": f1_score(labels, preds, average="macro")
    }



training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",          
    greater_is_better=True,               
    logging_dir="./logs"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)



trainer.train()
import pandas as pd
import numpy as np
import torch
import random

from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)


df = pd.read_csv("processed.csv")

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

train_dataset = dataset["train"]
test_dataset = dataset["test"]



tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)



model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1": f1_score(labels, preds, average="macro")
    }



training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",          
    greater_is_better=True,               
    logging_dir="./logs"
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)



trainer.train()
trainer.save_model("best_model")


results = trainer.evaluate()
print(results)


results = trainer.evaluate()
print(results)