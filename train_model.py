import torch
import numpy as np
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, VisualBertModel
from transformers import CLIPTextModel, T5Tokenizer, BertTokenizer


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# local
from models.uniter.vqa_model import VQAModel

logging.getLogger("transformers").setLevel(logging.ERROR)


MODELS = {
    'clip': {
        'model': CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32"),
        'tokenizer': AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32"),
    },
    'roberta': {
        'model': AutoModel.from_pretrained("roberta-base"),
        "tokenizer": AutoTokenizer.from_pretrained("roberta-base"),
    },
    'roberta_small': {
        'model': AutoModel.from_pretrained("klue/roberta-small"),
        "tokenizer": AutoTokenizer.from_pretrained("klue/roberta-small"),
    },
    'visualbert': {
        'model': VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre'),
        'tokenizer': BertTokenizer.from_pretrained("bert-base-uncased"),
    },
    't5': {
        'model': AutoModel.from_pretrained("t5-base"),
        'tokenizer': T5Tokenizer.from_pretrained("t5-base"),
    },
    'lxmert': {
        'model': AutoModel.from_pretrained("unc-nlp/lxmert-base-uncased"),
        'tokenizer': AutoTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased"),
    },
    'uniter': {
        'model': VQAModel(num_answers=2, model = 'uniter'),
        'tokenizer': BertTokenizer.from_pretrained("bert-base-cased"),
    }
}

class WordModel:
    def __init__(self, model, try_encoder=False, verbose=False):
        if model not in MODELS:
            raise Exception("Model not recognized.")
        
        self.try_encoder = try_encoder
        self.verbose = verbose
        
        self.tokenizer = MODELS[model]['tokenizer']
        self.model = MODELS[model]['model']
        # print(self.model)
        self.embedding = self._set_word_embeddings()
        
        if self.verbose:
            print(f"Model: {self.model.base_model_prefix} | Tokenizer: {self.tokenizer.name_or_path}")
        
    def _set_word_embeddings(self):
        """ Return word embedding layer from model """
        embeds = None
        if self.model.base_model_prefix == 'uniter':
            embeds = self.model.encoder.model.uniter.embeddings.word_embeddings
        elif self.model.base_model_prefix == 'clip':
            embeds = self.model.text_model.embeddings.token_embedding
        elif self.model.base_model_prefix == 'transformer':
            embeds = self.model.encoder.embed_tokens
        else:
            embeds = self.model.embeddings.word_embeddings
            
        if self.verbose:
            print(f"Loaded embedding layer: {embeds}")
        
        return embeds
    
    def get_features(self, input_sentences, pooled=False):
        all_features = []
        iters = int(len(input_sentences)/20)
        if len(input_sentences) % 20 != 0:
            iters += 1

        for i in range(iters):  # tqdm
            batch = input_sentences[i*20:(i+1)*20]
            inputs = self.tokenizer(batch, return_tensors="pt", padding='max_length', truncation=True, max_length=100)
            if self.try_encoder:
                if self.model.base_model_prefix in ['roberta', 'roberta_small', 'visualbert', 'visual_bert', 'uniter', 'clip']:
                    encoded = self.model(**inputs)
                    all_features += [encoded.last_hidden_state.reshape(len(batch), -1)]
                else:
                    raise Exception("Model doesn't have native encoder support, use embeddings")
            else:
                if pooled:
                    all_features += [self.embedding(inputs['input_ids']).mean(axis=1)]
                else:
                    all_features += [self.embedding(inputs['input_ids']).reshape(len(batch), -1)]
        
        with torch.no_grad():
            out = torch.cat(all_features).cpu().numpy()
        
        if self.verbose:
            print(f"Created features from {len(batch)} examples with shape: {out.shape}")
            
        return out
    

def train_model(features, labels):
    # print(f"Training on [{len(labels)}] samples")
    # lr = LogisticRegression(random_state=0, max_iter=1000)
    lr = make_pipeline(StandardScaler(), SGDClassifier(random_state=0, max_iter=1000, loss="log", n_jobs=-1))
    
    with torch.no_grad():
        lr.fit(features, labels)
    
    return lr 


def test_model(lr, test_features, test_labels, return_probs=False, verbose=False):
    # print(f"Testing on [{len(test_labels)}] samples")
    with torch.no_grad():
        predictions = np.array(lr.predict(test_features))

    labels = np.array(test_labels)
    
    accuracy = (100. * len(labels[labels == predictions])) / len(test_labels)
    if verbose:
        print(classification_report(test_labels, predictions))

    errors = []
    for idx in range(len(labels)):
        if labels[idx] != predictions[idx]:
            errors.append(idx)
    if return_probs:
        with torch.no_grad():
            probs = np.array(lr.predict_proba(test_features))
        return accuracy, probs, errors
    else:
        return accuracy, errors


def run_model(model_name, labels, sentences, test_labels, test_sentences, verbose=False, try_encoder=False):
    # instantiate
    m = WordModel(model_name, verbose=verbose, try_encoder=try_encoder)
    
    # train
    train_features = m.get_features(sentences)
    lr = train_model(train_features, labels)
    
    # test
    test_features = m.get_features(test_sentences)
    acc, errors = test_model(lr, test_features, test_labels, verbose=verbose)
    
    return acc, errors
