from data_processor import batch_processor
from train_model import run_model
import argparse
import os

def run(args):
    # train data
    train_processor = batch_processor(args.dataset, "train", attributes=args.attributes)
    train_labels, _, train_sentences, _ = train_processor.forward()
    # print(train_sentences)

    # test data
    test_processor = batch_processor(args.dataset, "test", attributes=args.attributes)
    test_labels, _, test_sentences, _ = test_processor.forward()

    # run model
    accuracy, error_idxs = run_model(args.model, train_labels, train_sentences, test_labels, test_sentences, try_encoder=args.try_encoder, verbose=args.verbose)
    print(f"The model achieved {accuracy}% accuracy")

    # logging
    curr_directory = os.path.dirname(__file__)
    extensions = ["./log", f"/{args.model}", f"/{args.dataset}", f"/{args.attributes}", f"/encoder_{args.try_encoder}"]
    for curr_dir in extensions:
        curr_directory = curr_directory + curr_dir
        if not os.path.isdir(curr_directory):
            os.mkdir(curr_directory)
    file = open(curr_directory + "/log.txt", "w")
    file.write(f"The model achieved {accuracy}% accuracy\n")
    file.write(str(error_idxs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Flip some bits!')
    parser.add_argument('--attributes', type=str, default="weight",
                        help='')
    parser.add_argument('--model', type=str, default="roberta",
                        help='')
    parser.add_argument('--dataset', type=str, default="verb")
    parser.add_argument('--try_encoder', type=bool, default=True)
    parser.add_argument('--verbose', type=bool, default=False)

    args = parser.parse_args()
    run(args)