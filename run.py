import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPTokenizer, RobertaTokenizer, BertTokenizer, LxmertTokenizer, XLNetTokenizer, AutoTokenizer
from transformers import CLIPTextModel, RobertaModel, VisualBertModel, LxmertModel, XLNetModel, AutoModel


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from data_processor import BatchProcessor

import argparse
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "roberta": "roberta-base", 
    "roberta_small": "klue/roberta-small", 
    "visualbert": 'uclanlp/visualbert-vqa-coco-pre',  # dif tokenizer ??
    "lxmert": "unc-nlp/lxmert-base-uncased", 
    # "xlnet": "xlnet-base-cased", 
}


class WrappedModel:
    def __init__(self, attributes="weight", model="clip", use_pooled=False, batch_size=0, split="train", dataset_size=20, verbose=False):
        """TODO: explore initation strategies
        can we call one function to load any model?"""
        if model not in MODELS:
            raise Exception("Model not recognized.")
        elif model == 'clip':  # special case
            self.model = CLIPTextModel.from_pretrained(MODELS[model]) 
        elif model == 'lxmert':
            """ TODO: input black image """
            # self.tokenizer = LxmertTokenizer.from_pretrained(MODELS[args.model])
            # self.model = LxmertModel.from_pretrained(MODELS[args.model])
            raise Exception("Not implemented yet")
        else:
            self.model = AutoModel.from_pretrained(MODELS[model])
        
        if model == "visualbert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(MODELS[model])
            
        self.classifier = LogisticRegression(random_state=0, max_iter=1000)
        
        self.verbose = verbose
        self.use_pooled = use_pooled
        self.batch_size = batch_size
        # self.max_epochs = args.epochs
        self.train_loader = BatchProcessor(attributes, dataset_size, split, batch_size)
        
        split = "test"
        self.batch_size = 0
        self.test_loader = BatchProcessor(attributes, dataset_size, split, batch_size)

    def get_features(self, processor):
        # print(f"getting features")
        labels, _, statements, _ = processor.forward()  # not used: cur batch, epoch
        inputs = self.tokenizer(statements, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        
        features = None
        if self.use_pooled:
            features = outputs.pooler_output
        else:
            """ TODO: identify better fix for non-symetric tensors """
            features = outputs.last_hidden_state[:, :9, :].reshape(len(statements), -1)
                    
        if len(statements) == self.batch_size:
            features = torch.cat(features).cpu().numpy()
            labels = torch.cat(labels).cpu().numpy()
        
        return features, labels, statements

    def train(self):
        features, labels, _ = self.get_features(self.train_loader)
        if self.verbose:
            print(f"training on [{len(labels)}] samples")

        with torch.no_grad():
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
        if self.verbose:
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
    parser.add_argument('--model', type=str, default="clip",
                        help='')
    
    args = parser.parse_args()

    m = WrappedModel(
        attributes=args.attributes,
        model=args.model, 
        use_pooled=args.use_pooled, 
        batch_size=args.batch_size, 
        split=args.split,
        verbose=True,
    )
    m.forward()