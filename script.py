import subprocess

if __name__ == '__main__':
    all_models = ["roberta", "clip"]
    for model in all_models:
        print(f"{model} model!!!!!!!!!!!!")
        command = ["python", "run.py", "--dataset", "doq", "--model", f"{model}"]
        subprocess.run(command)
