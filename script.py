import subprocess

if __name__ == '__main__':
    all_models = ['roberta', 'roberta_small', 'visualbert', 'clip'] #, 'uniter']
    # all_models = ['uniter']
    all_datasets = ["doq", "verb"] #, "prost"]
    dataset_attributes = {"verb": ['size', 'weight', 'strength', 'rigidness', 'speed'],
                          "prost":['all', 'bouncing', 'breaking', 'circumference', 'directions', 'grasping',
                                   'height', 'mass', 'rolling', 'sliding', 'stacking'],
                          "doq": ["NA"]}
    for dataset in all_datasets:
        for attr in dataset_attributes[dataset]:
            for model in all_models:
                print(f"{model} model on {dataset} dataset with attribute {attr} with encodings!!!!!!!!!!!!")
                command = ["python", "run.py", "--dataset", f"{dataset}", "--model", f"{model}", "--attributes", f"{attr}", "--try_encoder"]
                subprocess.run(command)


    all_models = ['roberta', 'roberta_small', 'visualbert', 't5', 'lxmert', 'clip'] # , 'uniter']
    all_datasets = ["doq", "verb"] #, "prost"]
    for model in all_models:
        for dataset in all_datasets:
            for attr in dataset_attributes[dataset]:
                print(f"{model} model on {dataset} dataset with attribute {attr} with embeddings!!!!!!!!!!!!")
                command = ["python", "run.py", "--dataset", f"{dataset}", "--model", f"{model}", "--attributes", f"{attr}"]
                subprocess.run(command)
