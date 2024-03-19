import logging

from datasets import load_dataset
import torch
import numpy as np

from linguistic_features import LinguisticFeatures


class Task:
    def __init__(self):
        self.tasks_names = ["random", "sentence_length", "tree_depth", "top_constituents", "tense", "subject_number", "object_number"]
        self.dataset = load_dataset(path="universal_dependencies", name="en_ewt")
        lf = LinguisticFeatures()

        features_train = lf.get_features_per_sent(self.dataset["train"])
        features_train = {feature: [item[feature] for item in features_train] for feature in self.tasks_names}
        for feature in self.tasks_names:
            self.dataset["train"] = self.dataset["train"].add_column(feature, features_train[feature])
        features_test = lf.get_features_per_sent(self.dataset["test"])
        features_test = {feature: [item[feature] for item in features_test] for feature in self.tasks_names}
        for feature in self.tasks_names:
            self.dataset["test"] = self.dataset["test"].add_column(feature, features_test[feature])
        features_val = lf.get_features_per_sent(self.dataset["validation"])
        features_val = {feature: [item[feature] for item in features_val] for feature in self.tasks_names}
        for feature in self.tasks_names:
            self.dataset["validation"] = self.dataset["validation"].add_column(feature, features_val[feature])

    def create_datasets_and_dataloaders(self, task, tokenizer, batch_size):
        logging.info("len of datasets before filtering", len(self.dataset["train"]), len(self.dataset["test"]), len(self.dataset["validation"]))

        self.dataset_train = self.dataset["train"].filter(lambda example: example[task] is not None)
        self.dataset_test = self.dataset["test"].filter(lambda example: example[task] is not None)
        self.dataset_val = self.dataset["validation"].filter(lambda example: example[task] is not None)

        self.num_classes = np.array(self.dataset_train[task]).max() + 1
        self.chance_performance = max([sum(self.dataset_train[task] == label) / len(self.dataset_train[task]) for label in np.unique(self.dataset_train[task])])
        self.name = task

        logging.info("len of datasets after filtering", len(self.dataset_train), len(self.dataset_test), len(self.dataset_val))

        self.dataset_train = self.dataset_train.map(lambda e: self._tokenize(task, e, tokenizer, "train"), batched=True, batch_size=batch_size)
        self.dataset_test = self.dataset_test.map(lambda e: self._tokenize(task, e, tokenizer, "test"), batched=True, batch_size=batch_size)
        self.dataset_val = self.dataset_val.map(lambda e: self._tokenize(task, e, tokenizer, "val"), batched=True, batch_size=batch_size)

        self.dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        self.dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        self.dataset_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Create dataloaders.
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=batch_size, num_workers=0)
        self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=batch_size, num_workers=0)
        self.dataloader_val = torch.utils.data.DataLoader(self.dataset_val, batch_size=batch_size, num_workers=0)
        
    def _tokenize(self, task, examples, tokenizer, split):
        tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, padding=True)
        tokenized_inputs["labels"] = examples[task]
        return tokenized_inputs