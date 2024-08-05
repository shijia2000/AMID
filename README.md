# AMID

## Introduction

This repository hosts the code for our paper "Open-World Semi-Supervised Learning under Compound Distribution Shifts" accepted by BMVC2024.



## Preparation

### Required Packages

We suggest first creating a conda environment:

```sh
conda create --name amid python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```



## Usage

Here is an example to train AMID on the PACS benchmark.

```
python main.py --c config/openSSL/openSSL_pacs_a35_0.yaml
```

