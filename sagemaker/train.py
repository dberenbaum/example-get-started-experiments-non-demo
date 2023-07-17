from box import ConfigBox
from ruamel.yaml import YAML
from sagemaker.pytorch.estimator import PyTorch

import boto3
import sagemaker
from sagemaker.estimator import Estimator


def train():
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    print(role)

    yaml = YAML(typ="safe")
    params = ConfigBox(yaml.load(open("../params.yaml", encoding="utf-8")))
    args = params["base"] | params["train"]
    args["fine_tune_args.epochs"] = args["fine_tune_args"]["epochs"]
    args["fine_tune_args.base_lr"] = args["fine_tune_args"]["base_lr"]
    del args["fine_tune_args"]
    print(args)

    pytorch_estimator = PyTorch("src/train.py",
                                instance_type="ml.m5.large",
                                instance_count=1,
                                framework_version="2.0.0",
                                py_version="py310",
                                hyperparameters = args,
                                role=role,
                                source_dir="..",
                                output_path="s3://example-get-started-experiments-non-demo/outputs"
                               )
    pytorch_estimator.fit({"train": "s3://example-get-started-experiments-non-demo/data/train_data"})


if __name__ == "__main__":
    train()