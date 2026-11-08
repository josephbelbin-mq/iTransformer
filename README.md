# iTransformer

The repo is a fork of the official implementation for the paper: [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625). [[Slides]](https://cloud.tsinghua.edu.cn/f/175ff98f7e2d44fbbe8e/), [[Poster]](https://cloud.tsinghua.edu.cn/f/36a2ae6c132d44c0bd8c/), [[Intro (CN)]](https://mp.weixin.qq.com/s/-pvBnA1_NSloNxa6TYXTSg).
.

## Introduction

This repository contains the code, datasets, experimental scripts, and results produced for the COMP8240 project at Macquarie University.

The project reproduces the main findings of iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (Liu et al., 2024) and investigates the conditions under which iTransformer provides advantages over competing forecasting approaches. In addition to reproducing published benchmark results, the project introduces reconstructed and synthetic datasets designed to examine the effects of nonlinear cross-variate interactions and latent state dynamics on forecasting performance.

The study evaluates iTransformer, PatchTST, and linear forecasting baselines across real-world and synthetic datasets. Results suggest that strong nonlinear interactions between variables are an important factor contributing to iTransformer's effectiveness, while also demonstrating that simpler methods remain highly competitive on datasets with predominantly linear structure.

The accompanying project report is available in analysis/COMP8240_Final_Report.pdf.

## Usage 

1. Install Pytorch and the necessary dependencies.

```
pip install -r requirements.txt
```

1. The datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Baidu Cloud](https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr).

2. Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

```
# Multivariate forecasting with iTransformer
bash ./scripts/multivariate_forecasting/Traffic/iTransformer.sh

# Generate beijing dataset:
Download https://zenodo.org/records/3902671/files/BeijingPM25Quality_TRAIN.ts?download=1
python ./dataset_generation/beijing/load_beijing_data_multistation.py


# Generate space_weather dataset:
python ./dataset_generation/space_weather/load_space_weather.py

# Generate synthetic dataset:
python dataset/synthetic/generate.py --max-lag 128 --outdir ../dataset/synthetic/  --num-sequences 100 --sequence-length 1000

# Sweep over synthetic data:
python ./sweep.py --dataset_dir <data_location> --model iTransformer
```

Please refer to iTransformer repo README for more ablation studies and details.

## Citation

If you find this repo helpful, please cite the original paper. 

```
@article{liu2023itransformer,
  title={iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author={Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  journal={arXiv preprint arXiv:2310.06625},
  year={2023}
}
```

## iTransformer Acknowledgement

iTransformer authors appreciate the following GitHub repos a lot for their valuable code and efforts.
- Reformer (https://github.com/lucidrains/reformer-pytorch)
- Informer (https://github.com/zhouhaoyi/Informer2020)
- FlashAttention (https://github.com/shreyansh26/FlashAttention-PyTorch)
- Autoformer (https://github.com/thuml/Autoformer)
- Stationary (https://github.com/thuml/Nonstationary_Transformers)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- lucidrains (https://github.com/lucidrains/iTransformer)

iTransformer work was supported by Ant Group through the CCF-Ant Research Fund and awarded as [Outstanding Projects of CCF Fund](https://mp.weixin.qq.com/s/PDLNbibZD3kqhcUoNejLfA).

## Contact

If you have any questions regarding iTransformer, feel free to contact:
* Yong Liu (liuyong21@mails.tsinghua.edu.cn)
* Haoran Zhang (z-hr20@mails.tsinghua.edu.cn)
* Tengge Hu (htg21@mails.tsinghua.edu.cn)

If you have any questions regarding the reproduction please contact:
* Joseph Belbin (joseph.belbin@students.mq.edu.au)
