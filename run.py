import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel, RobertaTokenizer, RobertaModel, VisualBertModel, BertTokenizer


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from data_processor import BatchProcessor
import argparse

class WrappedModel:
    def __init__(self, args):
        """TODO: explore initation strategies
        can we call one function to load any model?"""
        if args.model == "clip":
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        elif args.model == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.model = RobertaModel.from_pretrained("roberta-base")
        elif args.model == 'visualbert':
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
        else:
            raise Exception("model not recognized")

        self.classifier = LogisticRegression(random_state=0, max_iter=1000)
        
        self.use_pooled = args.use_pooled
        self.batch_size = args.batch_size
        # self.max_epochs = args.epochs
        self.train_loader = BatchProcessor(args)
        
        args.split = "test"
        args.batch_size = 0
        self.batch_size = 0
        self.test_loader = BatchProcessor(args)

    def get_features(self, processor):
        # print(f"getting features")
        _, labels, statements, _ = processor.forward()  # not used: cur batch, epoch
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
        print(f"training on [{len(labels)}] samples")
        
        with torch.no_grad():
            self.classifier.fit(features, labels)

    def test(self):
        features, labels, raw = self.get_features(self.test_loader)
        print(f"testing on [{len(labels)}] samples")
        
        with torch.no_grad():
            predictions = np.array(self.classifier.predict(features))
        
        labels = np.array(labels)
        # probs = self.classifier.predict_proba(features)
        accuracy = (100. * len(labels[labels == predictions])) / len(raw)
        print(classification_report(labels, predictions))

    def forward(self):
        self.train()
        self.test()

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

    m = WrappedModel(args)
    m.forward()