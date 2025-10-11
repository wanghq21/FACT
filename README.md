# FACT
Code release of paper "FACT: Fine-grained Across-variable Convolution for multivariate Time-series forecasting"

## Get Started

1. Install Python 3.8
```
pip install -r requirements.txt
```

2. Data. All benchmark datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1MKugRwUKN2u9tIBgES-n3-QOT5l6unLl/view?usp=drive_link). All datasets should be placed under folder `./dataset`, such as `./dataset/electricity/electricity.csv`.

3. You can reproduce all the experiment results by running:
```
sh FACT.sh
```

## Architecture
The architecture of FACT is:
<p align="center">
<img src=".\figure\architecture.png" height = "450" alt="" align=center />
<br><br>
</p>

## Experiments
### The overall performance on different datasets
<p align="center">
<img src=".\figure\performance.png" height = "650" alt="" align=center />
<br><br>
</p>

### Compared with attention
<p align="center">
<img src=".\figure\compare_attention.png" height = "250" alt="" align=center />
<br><br>
</p>
