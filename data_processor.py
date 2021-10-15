import numpy as np
import pandas as pd
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class BatchProcessor():
    def __init__(self, model, dataset_size, split="train", attributes="weight", batch_size=20):
        # validate inputs and store them
        if not self.check_valid_params(model, attributes, dataset_size, split):
            raise Exception("invalid batch processor inputs")
        self.batch_size = batch_size
        self.attributes = attributes
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
        # for curr_attr in self.attributes:
        #     database = database[database[f"{curr_attr}-maj"] != -42]
        database = database.reset_index()

        return database

    def check_valid_params(self, model, attributes, dataset_size, split):
        valid_models = ["roberta", "clip"]
        if model not in valid_models:
            print(f"The available models are {valid_models}")
            return False

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
        idxs = np.random.random_integers(low=0, high=self.num_rows-1, size=self.batch_size)
        database_batch = self.database.iloc[idxs]
        curr_batch = database_batch.loc[:, f"{self.attributes}-maj"]
        curr_batch = torch.tensor(curr_batch.values)

        agreement = database_batch.loc[:, f"{self.attributes}-agree"]

        statements = []
        for _, row in database_batch.iterrows():
            curr_statement = "A " + str(row["obj1"]) + f" is {self.verbs} than a " + str(row["obj2"])
            statements.append(curr_statement)

        return curr_batch, agreement, statements

if __name__ == '__main__':
    prc = BatchProcessor(model="roberta", dataset_size=5)
    data, agreement, statements = prc.forward()

    print(statements)
