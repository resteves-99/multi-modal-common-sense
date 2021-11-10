import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def batch_processor(dataset, split, batch_size=0, attributes="weight", dataset_size=20):
    if dataset == "verb":
        return VerbProcessor(attributes, dataset_size, split, batch_size)
    elif dataset == "prost":
        print(attributes)
        return ProstProcessor(split, batch_size, )
    elif dataset == "doq":
        return DoqProcessor(split)

class DoqProcessor():
    def __init__(self, split):
        self.data = pd.read_csv(f"data/doq/noun/{split}.csv")
        self.data = self.data.values #get into numpy array
        self.verbs = {"big": "bigger", "heavy": "heavier", "fast": "faster"}


    def forward(self):
        labels = []
        inputs = []
        for row in self.data:
            str = f"A {row[0]} is {self.verbs[row[2]]} than a {row[1]}"
            inputs.append(str)
            labels.append(row[3])
        labels = torch.tensor(labels)

        return labels, None, inputs, None

class ProstProcessor():
    def __init__(self, split, batch_size, attribute="bouncing"):
        # attributes can be one of:
        # 'bouncing' 'breaking' 'circumference' 'directions'
        #  'grasping' 'height' 'mass' 'rolling' 'sliding' 'stacking'
        self.dataset = load_dataset('corypaik/prost', split='test')
        self.dataset = self.dataset.train_test_split(test_size=0.1)[split]

        if split == "train":
            end_idx = int(len(self.dataset)/2)
            self.dataset.select(range(end_idx))
        else:
            start_idx = int(len(self.dataset)/2)
            end_idx = int(len(self.dataset))
            self.dataset.select(range(start_idx, end_idx))

        if attribute is not "all":
            self.dataset = self.dataset.filter(lambda row: row["group"] == attribute)
        if split == "train":
            self.dataset = self.dataset.select(range(500))

        self.idx = 0
        self.batch_size = batch_size

    def forward(self):
        label_to_char = {0: 'A', 1:'B', 2:'C', 3:'D'}
        inputs = []
        labels = []
        for row in self.dataset:
            # assumes bert special tokens
            curr_input = row['question'] + " [SEP] " + row['context']
            for idx in range(4):
                curr_answer = row[label_to_char[idx]]
                tmp_curr_input = curr_input.replace("[MASK]", curr_answer)
                curr_label = idx == row['label']
                inputs.append(tmp_curr_input)
                labels.append(curr_label)
        return labels, None, inputs, None

class VerbProcessor():
    def __init__(self, attributes, dataset_size, split, batch_size):
        # validate inputs and store them
        self.batch_size = batch_size
        self.attributes = attributes
        self.epochs = 0
        self.curr_idx = 0
        verbs = {"size": "bigger", "weight": "heavier", "strength": "stronger", "rigidness": "more rigid", "speed": "faster"}
        self.verbs = verbs[attributes]

        # initialize database
        file = f"data/verbphysics/objects/train-{dataset_size}/{split}.csv"
        database = pd.read_csv(file)
        self.database = self.preprocess(database)
        self.num_rows = self.database.shape[0]


    # removes unnecesary columns and unlabelled data
    def preprocess(self, database):
        # remove unncesary columns
        cols = ["obj1", "obj2"]
        cols.append(f"{self.attributes}-agree")
        cols.append(f"{self.attributes}-maj")
        # for curr_attr in self.attributes:
        #     cols.append(f"{curr_attr}-agree")
        #     cols.append(f"{curr_attr}-maj")
        database = database.loc[:, cols]

        # remove unlabelled data
        database = database[database[f"{self.attributes}-maj"] != -42]
        # database = database[database[f"{self.attributes}-maj"] != 0] # ambigious questions
        database = database[database[f"{self.attributes}-agree"] != 1]
        # for curr_attr in self.attributes:
        #     database = database[database[f"{curr_attr}-maj"] != -42]
        database = database.reset_index()

        return database

    def check_valid_params(self, attributes, dataset_size, split):
        valid_attributes = ["size", "weight", "strength", "rigidness", "speed"]
        # if not all(elem in valid_attributes  for elem in attributes):
        if attributes not in valid_attributes:
            print(f"The available attributes are one of {valid_attributes}")
            return False

        valid_sizes = [5, 20]
        if dataset_size not in valid_sizes:
            print(f"The available dataset sizes are {valid_sizes}")
            return False

        valid_splits = ["train", "dev", "test"]
        if split not in valid_splits:
            print(f"The splits are {valid_splits}")
            return False

        return True

    def forward(self):
        if self.batch_size == 0:
            idxs = range(0, self.num_rows)
        else:
            idxs = range(self.curr_idx, self.curr_idx+self.batch_size)
            self.curr_idx = self.curr_idx+self.batch_size
            if self.curr_idx >= self.num_rows:
                self.curr_idx = 0
                self.epochs += 1

        database_batch = self.database.iloc[idxs]
        curr_batch = database_batch.loc[:, f"{self.attributes}-maj"]
        curr_batch = torch.tensor(curr_batch.values)

        agreement = database_batch.loc[:, f"{self.attributes}-agree"]
        agreement = torch.tensor(agreement.values)

        statements = []
        for _, row in database_batch.iterrows():
            curr_statement = "A " + str(row["obj1"]) + f" is {self.verbs} than a " + str(row["obj2"])
            statements.append(curr_statement)

        return curr_batch, agreement, statements, self.epochs

if __name__ == '__main__':
    prc = batch_processor(dataset="doq", split='test', batch_size=0)
    # data, agreement, statements, curr_epochs = prc.forward()
