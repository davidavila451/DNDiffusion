# main.py

import argparse
import subprocess
import os

def run_train(config_path):
    subprocess.run(["accelerate", "launch", "training/train.py", "--config", config_path])

def run_generate(config_path):
    subprocess.run(["python", "generation/generate.py", "--config", config_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI tasks")
    parser.add_argument("--mode", choices=["train", "generate"], required=True, help="Which task to run")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to config file")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args.config)
    elif args.mode == "generate":
        run_generate(args.config)
