import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from sklearn.linear_model import LogisticRegression
from data_processor import BatchProcessor
import argparse

class WrappedModel(nn.module):
    def __init__(self, args):
        """TODO: explore initation strategies
        can we call one function to load any model?"""
        if args.model == "clip":
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            raise Exception("model not recognized")

        self.classifier = LogisticRegression(random_state=0, verbose=0)

        self.max_epochs = args.epochs
        self.train_loader = BatchProcessor(args)
        args.batch_size = 0
        args.split = "test"
        self.test_loader = BatchProcessor(args)

    def process_inputs(self, processor):
        """TODO: write this function"""
        pass

    def train(self):
        while True:
            inputs, labels, epochs = self.process_inputs(self.train_loader)
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = outputs.pooled_output
            self.classifier.fit(last_hidden_state, labels)
            if epochs >= self.max_epochs:
                break

    def test(self):
        """TODO: log results"""
        inputs, labels, _ = self.process_features(self.test_loader)
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooled_output
        predictions = self.classifier.predict(last_hidden_state)
        probs = self.classifier.predict_proba(last_hidden_state)

    def forward(self):
        self.train()
        self.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Flip some bits!')
    parser.add_argument('--dataset_size', type=int, default=20,
                        help='which dataset size to use, 5 or 20')
    parser.add_argument('--split', type=str, default="train",
                        help='')
    parser.add_argument('--attribute', type=str, default="weight",
                        help='')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='')
    parser.add_argument('--epochs', type=int, default=2,
                        help='')

    parser.add_argument('--model', type=str, default="clip",
                        help='')
    args = parser.parse_args()