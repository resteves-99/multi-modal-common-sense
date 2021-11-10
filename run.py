from data_processor import batch_processor
from train_model import run_model
import argparse

def run(args):
    train_processor = batch_processor(args.dataset, "train", attributes=args.attributes)
    train_labels, _, train_sentences, _ = train_processor.forward()
    # print(train_sentences)

    test_processor = batch_processor(args.dataset, "test", attributes=args.attributes)
    test_labels, _, test_sentences, _ = test_processor.forward()

    accuracy = run_model(args.model, train_labels, train_sentences, test_labels, test_sentences)
    print(f"The model achieved {accuracy}% accuracy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Flip some bits!')
    parser.add_argument('--attributes', type=str, default="weight",
                        help='')
    parser.add_argument('--model', type=str, default="roberta",
                        help='')
    parser.add_argument('--dataset', type=str, default="verb")

    args = parser.parse_args()
    run(args)