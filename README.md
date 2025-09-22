# FACT
Code release of paper "FACT: Fine-grained Across-variable Convolution for multivariate Time-series forecasting"

## Get Started

1. Install Python 3.8
```
pip install -r requirements.txt
```

2. Data. All benchmark datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1MKugRwUKN2u9tIBgES-n3-QOT5l6unLl/view?usp=drive_link). All datasets should be placed under folder `./dataset`, such as `./dataset/electricity/electricity.csv`.

3. You can reproduce all the experiment results as the following examples
```
sh FACT.sh
