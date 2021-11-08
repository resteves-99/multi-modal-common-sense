import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import CLIPTokenizer, RobertaTokenizer, BertTokenizer, LxmertTokenizer, XLNetTokenizer, AutoTokenizer, T5Tokenizer
from transformers import CLIPTextModel, RobertaModel, VisualBertModel, LxmertModel, XLNetModel, AutoModel

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from data_processor import batch_processor

import argparse
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "roberta": "roberta-base", 
    "roberta_small": "klue/roberta-small", 
    "visualbert": 'uclanlp/visualbert-vqa-coco-pre',  # dif tokenizer ??
    "lxmert": "unc-nlp/lxmert-base-uncased", 
    "uniter": "visualjoyce/transformers4vl-uniter-base",
}


class WrappedModel:
    def __init__(self, attributes="weight", model="clip", use_pooled=False, batch_size=0, split="train", dataset_size=20, verbose=False, epochs=10, dataset="vocab"):
        if model not in MODELS:
            raise Exception("Model not recognized.")
        elif model == 'clip':  # special case
            self.model = CLIPTextModel.from_pretrained(MODELS[model]) 
        else:
            self.model = AutoModel.from_pretrained(MODELS[model])
            if verbose:
                print(self.model)
        
        if model == "visualbert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif model == "numbert":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        else:
            # if i am initializing the special tokens here will the transformer recognize them?
            self.tokenizer = AutoTokenizer.from_pretrained(MODELS[model])
        
        assert(self.tokenizer is not None)
        assert(self.model is not None)
        
        self.classifier = LogisticRegression(random_state=0, max_iter=1000)

        self.verbose = verbose
        self.use_pooled = use_pooled
        self.batch_size = batch_size
        # self.max_epochs = args.epochs
        self.train_loader = batch_processor(dataset, split, batch_size, attributes, dataset_size, self.tokenizer)
        
        split = "test"
        self.batch_size = 0
        self.test_loader = batch_processor(dataset, split, batch_size, attributes, dataset_size, self.tokenizer)

    def get_features(self, processor):
        if self.model.base_model_prefix == 'lxmert':
            features, labels, statements = self.get_lxmert_features(processor)
        else:
            labels, _, statements, _ = processor.forward()  # not used: cur batch, epoch
            inputs = self.tokenizer(statements, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            if self.use_pooled:
                features = outputs.pooler_output
            else:
                """ TODO: identify better fix for non-symetric tensors """
                features = outputs.last_hidden_state[:, :9, :].reshape(len(statements), -1)
                    
        if len(statements) == self.batch_size:
            features = torch.cat(features).cpu().numpy()
            labels = torch.cat(labels).cpu().numpy()
        
        return features, labels, statements

    def get_lxmert_features(self, processor):
        """ gross special handler for passing in bogus image data """
        all_features = []
        all_labels = []
        all_statements = []

        if self.batch_size == 0:
            self.epochs = 1

        with torch.no_grad():
            for epoch in self.epochs:
                labels, _, statements, _ = processor.forward()  # not used: cur batch, epoch
                num_ex = self.batch_size
                if num_ex == 0:
                    num_ex = len(statements)

                if self.verbose:
                    print(self.batch_size * len(all_labels))

                if len(labels) == 0:
                    return all_features

                inputs = self.tokenizer(statements, return_tensors="pt", padding=True, truncation=True)
                visual_feats = torch.rand(num_ex, 768, 2048)
                bounding_boxes = torch.rand(num_ex, 768, 4)

                outputs = self.model(**inputs, visual_feats=visual_feats, visual_pos=bounding_boxes)

                if self.use_pooled:
                    all_features += [outputs.pooled_output]
                else:
                    all_features += [outputs.language_output[:, :9, :].reshape(len(statements), -1)]

                all_statements += statements
                all_labels += [labels]

            return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy(), all_statements

    def train(self):
        features, labels, _ = self.get_features(self.train_loader)
        if self.verbose:
            print(f"training on [{len(labels)}] samples")

        with torch.no_grad():
            print(features, len(features))
            print(labels, len(labels))
            self.classifier.fit(features, labels)

    def test(self):
        features, labels, raw = self.get_features(self.test_loader)
        if self.verbose:
            print(f"testing on [{len(labels)}] samples")
        
        with torch.no_grad():
            predictions = np.array(self.classifier.predict(features))
        
        labels = np.array(labels)
        # probs = self.classifier.predict_proba(features)
        accuracy = (100. * len(labels[labels == predictions])) / len(raw)
        print(classification_report(labels, predictions))
        
        return accuracy

    def forward(self):
        self.train()
        return self.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Flip some bits!')
    parser.add_argument('--dataset_size', type=int, default=20,
                        help='which dataset size to use, 5 or 20')
    parser.add_argument('--split', type=str, default="train",
                        help='')
    parser.add_argument('--attributes', type=str, default="weight",
                        help='')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='')
    parser.add_argument('--epochs', type=int, default=2,
                        help='')
    parser.add_argument('--use_pooled', type=bool, default=False,
                        help='Pool last hidden state?')
    parser.add_argument('--model', type=str, default="roberta",
                        help='')
    parser.add_argument('--dataset', type=str, default="verb")
    
    args = parser.parse_args()

    m = WrappedModel(
        attributes=args.attributes,
        model=args.model, 
        use_pooled=args.use_pooled, 
        batch_size=args.batch_size, 
        split=args.split,
        verbose=False,
        dataset=args.dataset,
    )
    m.forward()