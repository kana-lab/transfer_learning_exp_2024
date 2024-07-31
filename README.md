# Experiment of Transfer Learning

## Setup

```shell
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install -r requirements.txt
```

## Dataset Preparation

You must have your kaggle account.

```shell
(.venv) $ kaggle competitions download -c dogs-vs-cats
```

Please extend downloaded zip files.


## Run

```shell
(.venv) $ python train.py
```

