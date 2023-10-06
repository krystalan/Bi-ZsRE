
<div align="center">
<h1>
Bi-ZsRE
</h1>
</div>

This repository contains the data and codes for our paper "[Cross-Lingual Knowledge Editing in Large Language Models](https://arxiv.org/abs/2309.08952)".

### Quick Links
- [1. Overview](#1-overview)
- [2. Bi-ZsRE](#2-bi-zsre)
- [3. Codes](#3-codes)
    - [3.1 ROME](#31-rome)
    - [3.2 MEMIT](#32-memit)
    - [3.3 KN](#33-kn)
    - [3.4 Others](#34-others)
- [4. Evaluation](#4-evaluation)
    - [4.1 How to Evaluate](#41-how-to-evaluate)
    - [4.2 Results](#42-results)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

### 1. Overview

In this work, we explore the effect of source language editing on a different target language. For example, when we use English samples to edit a multi-lingual large language model, can the model reflect consistent behaviors when faced with a different target language?

<p align="center">
    <br>
    <img src="./figs/intro.png" width="600"/>
    <br>
</p>

### 2. Bi-ZsRE

**Summary**: Bi-ZsRE translates the editing samples of ZsRE from English into Chinese. 

**Detail**: The orginal ZsRE dataset ([Levy et al., 2017](https://aclanthology.org/K17-1034/)) is a Question Answering (QA) dataset whose queries require models to answer the questions based on the information within the queries. More recently, based on ZsRE, [Yao et al. (2023)](https://arxiv.org/abs/2305.13172) provide a test set with 1,037 samples for a more comprehensive evaluation of knowledge editing, where each test sample additionally contains a QA pair to assess LLMs’ portability to reason based on the edited fact. Here, we translate the ZsRE samples as well as the portability QA pairs provided by Yao et al. (2023), and follow the data splitting of Yao et al. (2023).

**Resource**: You can download Bi-ZsRE data from the [share link](https://drive.google.com/file/d/1Dw2q-oIUWoV1CvUl5q3CBaD0JIV9jtn0). All files are described as follows:

- `zsre_mend_train_10000.json`: The original ZsRE English training samples
- `zsre_mend_train_10000_chinese.json`: The translated Chinese samples
- `zsre_mend_eval.json`: The original ZsRE English dev samples
- `zsre_mend_eval_chinese.json`: The translated Chinese samples.
- `bizsre_test.json`: The original English samples as well as translated Chinese samples.

Among them, `zsre_mend_train_10000.json`, `zsre_mend_eval.json` and the English part of `bizsre_test.json` are originally provided by [EasyEdit](https://github.com/zjunlp/EasyEdit#current-implementation). 

## 3. Codes

Our codes are based on [EasyEdit](https://github.com/zjunlp/EasyEdit), and slightly change some functions to adapt to the cross-lingual evaluation.

Before running the following codes, make sure to set MODEL PATH in the corresponding config files (every `.yaml` file in `hparams` folder).

#### 3.1 ROME
Editing Chinese LLaMA with English samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=ROME --hparams_dir=hparams/ROME/llama-7b --data_dir=data --source_lang en
```
Editing Chinese LLaMA with Chinese samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=ROME --hparams_dir=hparams/ROME/llama-7b --data_dir=data --source_lang zh
```
Editing Chinese LLaMA2 with English samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=ROME --hparams_dir=hparams/ROME/llama2-7b --data_dir=data --source_lang en --backbone chinese_llama2_7b
```
Editing Chinese LLaMA2 with Chinese samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=ROME --hparams_dir=hparams/ROME/llama2-7b --data_dir=data --source_lang zh --backbone chinese_llama2_7b
```
Editing BaiChuan with English samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=ROME --hparams_dir=hparams/ROME/baichuan-7b --data_dir=data --source_lang en --backbone baichuan7b
```
Editing BaiChuan with Chinese samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=ROME --hparams_dir=hparams/ROME/baichuan-7b --data_dir=data --source_lang zh --backbone baichuan7b
```

#### 3.2 MEMIT

Editing Chinese LLaMA with English samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=MEMIT --hparams_dir=hparams/MEMIT/llama-7b --data_dir=data --source_lang en
```
Editing Chinese LLaMA with Chinese samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=MEMIT --hparams_dir=hparams/MEMIT/llama-7b --data_dir=data --source_lang zh
```
Editing Chinese LLaMA2 with English samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=MEMIT --hparams_dir=hparams/MEMIT/llama2-7b --data_dir=data --source_lang en --backbone chinese_llama2_7b
```
Editing Chinese LLaMA2 with Chinese samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=MEMIT --hparams_dir=hparams/MEMIT/llama2-7b --data_dir=data --source_lang zh --backbone chinese_llama2_7b
```
Editing BaiChuan with English samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=MEMIT --hparams_dir=hparams/MEMIT/baichuan-7b --data_dir=data --source_lang en --backbone baichuan7b
```
Editing BaiChuan with Chinese samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=MEMIT --hparams_dir=hparams/MEMIT/baichuan-7b --data_dir=data --source_lang zh --backbone baichuan7b
```


#### 3.3 KN
Editing Chinese LLaMA with English samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=KN --hparams_dir=hparams/KN/llama-7b --data_dir=data --source_lang en
```
Editing Chinese LLaMA with Chinese samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=KN --hparams_dir=hparams/KN/llama-7b --data_dir=data --source_lang zh
```
Editing Chinese LLaMA2 with English samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=KN --hparams_dir=hparams/KN/llama2-7b --data_dir=data --source_lang en --backbone chinese_llama2_7b
```
Editing Chinese LLaMA2 with Chinese samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=KN --hparams_dir=hparams/KN/llama2-7b --data_dir=data --source_lang zh --backbone chinese_llama2_7b
```
Editing BaiChuan with English samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=KN --hparams_dir=hparams/KN/baichuan-7b --data_dir=data --source_lang en --backbone baichuan7b
```
Editing BaiChuan with Chinese samples:
```
CUDA_VISIBLE_DEVICES=<gpu> python run_bizsre.py --editing_method=KN --hparams_dir=hparams/KN/baichuan-7b --data_dir=data --source_lang zh --backbone baichuan7b
```

#### 3.4 Others
ROME, MEMIT and KN are three locate-then-edit knowledge editing methods that do not need additional training. For other methods, you should first train the model with `zsre_mend_train_10000.json` or `zsre_mend_train_10000_chinese.json`, and then perform the model on the test set.

For the training codes, please refer to `train.py`.


## 4. Evaluation

#### 4.1 How to Evaluate?
After running the codes, model results will be generated in the `results/` folder in the format of `{BACKBONE_NAME}_{METHOD_NAME}_{DIRECTION}_results.json`.

- BACKBONE_NAME: `baichuan7b`, `cinese_llama7b` or `chinese_llama2_7b`
- METHOD_NAME: `ROME`, `MEMIT`, `KN`, `SERAC`, `IKE`, `MEND`
- DIRECTION: `en_zh` (the model is edited in English samples and test on all samples) or `zh_en` (the model is edited in Chinese samples and test on all)

For our evaluation code, please refer to `evaluate.py`:
```python
from evaluate import calculate_metrics
calculate_metrics("results/baichuan7b_ROME_en_zh_results.json")
```
(using results of English-edited BaiChuan-7B via ROME as an example)


#### 4.2 Results
The following tables show our experimental results. Note that the results are slightly different from [our paper V1](https://arxiv.org/abs/2309.08952v1) due to the data updating and bug fixing.

The results (F1/EM score) of using [Chinese-LLaMA-7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca) as backbone:
| Method | Reli. | Gene. (En) | Gene. (Zh) | Loca. (En) | Loca. (Zh) | Port. (En) | Port. (Zh) |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| FT (En) | 20.46 / 0.77 | 18.36 / 0.19 | 22.08 / 0.10 | 87.49 / 70.11 | 79.52 / 47.44 | 6.30 / 0.00 | 24.63 / 0.00 |
FT (Zh) | 9.54 / 0.19 | 12.38 / 0.00 | 35.91 / 0.96 | 87.81 / 73.29 | 57.78 / 15.24 | 6.77 / 0.10 | 21.09 / 0.19 |
SERAC (En) | 73.84 / 56.03 | 50.86 / 27.10 | 19.26 / 0.29 | 100.00 / 100.00 | 99.97 / 99.90 | 6.52 / 0.00 | 15.63 / 0.00 |
SERAC (Zh) | 27.05 / 12.83 | 14.67 / 0.00 | 67.41 / 37.32 | 100.00 / 100.00 | 94.42 / 85.82 | 6.56 / 0.00 | 20.65 / 0.00 |
IKE (En) | 99.90 / 99.90 | 99.24 / 98.36 | 92.95 / 69.72 | 62.79 / 36.26 | 36.16 / 6.75 | 50.86 / 17.84 | 33.84 / 4.24 |
IKE (Zh) | 99.90 / 99.71 | 85.39 / 77.24 | 97.31 / 95.37 | 64.14 / 37.32 | 52.46 / 17.36 | 40.07 / 5.01 | 38.39 / 7.52 |
MEND (En) | 37.57 / 2.22 | 33.24 / 1.35 | 17.08 / 0.00 | 88.96 / 74.25 | 91.75 / 75.89 | 6.56 / 0.10 | 16.91 / 0.00 |
MEND (Zh) | 15.47 / 0.48 | 14.39 / 0.00 | 44.32 / 0.68 | 89.19 / 73.87 | 78.17 / 46.00 | 6.72 / 0.10 | 22.94 / 0.19 |
KN (En) | 4.63 / 0.00 | 4.54 / 0.00 | 6.66 / 0.00 | 42.25 / 29.12 | 36.75 / 19.67 | 3.53 / 0.00 | 8.46 / 0.00 |
KN (Zh) | 3.09 / 0.00 | 4.74 / 0.00 | 5.08 / 0.00 | 29.82 / 16.39 | 20.08 / 8.78 | 2.87 / 0.00 | 5.56 / 0.00 |
ROME (En) | 98.98 / 97.20 | 94.58 / 87.85 | 26.65 / 6.75 | 92.49 / 81.49 | 89.08 / 67.60 | 8.48 / 0.00 | 17.07 / 0.00 |
ROME (Zh) | 36.63 / 20.15 | 24.24 / 8.87 | 81.83 / 39.92 | 89.21 / 74.73 | 86.44 / 63.74 | 6.52 / 0.00 | 21.33 / 0.10 |
MEMIT (En) | 96.19 / 92.48 | 90.66 / 81.97 | 28.26 / 6.75 | 98.31 / 94.70 | 97.31 / 91.51 | 8.27 / 0.00 | 17.96 / 0.00 |
MEMIT (Zh) | 35.54 / 19.19 | 22.88 / 8.97 | 81.11 / 39.34 | 98.13 / 94.12 | 95.84 / 86.98 | 6.88 / 0.00 | 23.29 / 0.00 |

(Reli.: Reliability; Gene.: Generalization; Loca.: Locality; Port.: Portability)

The results (F1/EM score) of using [Chinese-LLaMA2-7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) as backbone:
| Method | Reli. | Gene. (En) | Gene. (Zh) | Loca. (En) | Loca. (Zh) | Port. (En) | Port. (Zh) |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
FT (En) | 36.62 / 5.98 | 35.01 / 7.52 | 20.24 / 0.10 | 81.90 / 55.06 | 72.95 / 32.11 | 7.33 / 0.00 | 17.91 / 0.00
FT (Zh) | 13.03 / 1.16 | 16.30 / 1.06 | 36.01 / 0.77 | 76.68 / 48.02 | 59.70 / 16.59 | 7.07 / 0.00 | 19.25 / 0.00
SERAC (En) | 98.78 / 97.01 | 89.62 / 82.64 | 21.92 / 2.60 | 100.00 / 100.00 | 97.67 / 93.44 | 8.75 / 0.00 | 17.30 / 0.00
SERAC (Zh) | 26.76 / 20.44 | 19.87 / 2.31 | 71.76 / 49.37 | 100.00 / 100.00 | 77.85 / 56.89 | 8.14 / 0.00 | 23.67 / 2.03
IKE (En) | 100.00 / 100.00 | 99.69 / 99.32 | 92.28 / 77.72 | 56.35 / 30.76 | 41.59 / 4.63 | 45.72 / 11.76 | 37.04 / 4.82
IKE (Zh) | 99.95 / 99.90 | 94.40 / 91.22 | 99.40 / 98.94 | 51.42 / 23.43 | 52.23 / 14.66 | 40.75 / 5.40 | 45.05 / 13.69
MEND (En) | 56.57 / 0.00 | 49.33 / 0.00 | 20.66 / 0.00 | 95.46 / 86.79 | 95.25 / 86.21 | 7.62 / 0.00 | 17.34 / 0.00
MEND (Zh) | 20.65 / 0.00 | 20.40 / 0.00 | 47.04 / 0.00 | 96.45 / 89.87 | 90.13 / 70.11 | 7.06 / 0.00 | 22.62 / 0.00
KN (En) | 10.94 / 0.00 | 10.96 / 0.00 | 12.30 / 0.00 | 49.28 / 6.85 | 43.65 / 9.35 | 5.75 / 0.00 | 14.39 / 0.00
KN (Zh) | 8.40 / 0.00 | 10.55 / 0.00 | 12.19 / 0.00 | 45.10 / 4.44 | 37.47 / 3.95 | 5.88 / 0.00 | 14.02 / 0.00
ROME (En) | 77.65 / 67.98 | 72.27 / 55.06 | 23.27 / 3.28 | 93.67 / 81.58 | 95.55 / 84.96 | 7.48 / 0.10 | 17.88 / 0.00
ROME (Zh) | 24.88 / 8.29 | 20.17 / 2.51 | 60.44 / 12.83 | 93.75 / 82.45 | 94.75 / 83.70 | 7.06 / 0.00 | 24.75 / 2.12
MEMIT (En) | 83.01 / 74.64 | 77.63 / 61.43 | 23.91 / 3.95 | 98.45 / 95.37 | 98.13 / 93.54 | 8.08 / 0.10 | 17.22 / 0.00
MEMIT (Zh) | 25.84 / 9.06 | 20.41 / 2.12 | 64.16 / 13.31 | 98.67 / 95.76 | 96.75 / 89.49 | 7.29 / 0.00 | 26.10 / 2.31

The results (F1/EM score) of using [BaiChuan-7B](https://github.com/baichuan-inc/Baichuan-7B) as backbone:
| Method | Reli. | Gene. (En) | Gene. (Zh) | Loca. (En) | Loca. (Zh) | Port. (En) | Port. (Zh) |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
FT (En) | 33.33 / 13.11 | 27.09 / 7.43 | 20.79 / 0.19 | 91.71 / 83.12 | 87.36 / 64.71 | 9.21 / 0.19 | 30.77 / 0.10
FT (Zh) | 13.08 / 1.45 | 13.39 / 0.58 | 28.71 / 4.34 | 95.18 / 90.26 | 53.83 / 16.88 | 9.28 / 0.29 | 27.76 / 0.29
KN (En) | 10.77 / 0.00 | 10.32 / 0.00 | 19.69 / 0.00 | 71.28 / 55.74 | 93.32 / 80.71 | 8.96 / 0.19 | 31.74 / 0.00
KN (Zh) | 10.22 / 0.00 | 10.49 / 0.00 | 19.52 / 0.00 | 73.43 / 58.24 | 84.62 / 59.98 | 9.04 / 0.29 | 31.64 / 0.00
ROME (En) | 68.97 / 52.36 | 60.45 / 42.53 | 24.45 / 1.45 | 98.31 / 96.43 | 98.71 / 95.85 | 9.65 / 0.29 | 31.61 / 0.19
ROME (Zh) | 24.04 / 6.36 | 16.05 / 1.93 | 68.74 / 12.63 | 98.06 / 95.66 | 97.96 / 93.15 | 9.40 / 0.29 | 27.98 / 0.68
MEMIT (En) | 71.20 / 54.97 | 66.47 / 49.66 | 26.19 / 2.51 | 98.60 / 96.72 | 98.82 / 95.56 | 9.43 / 0.10 | 30.53 / 0.29
MEMIT (Zh) | 23.95 / 6.27 | 19.11 / 5.59 | 72.29 / 14.75 | 98.47 / 96.53 | 96.87 / 90.55 | 9.05 / 0.19 | 24.49 / 0.48

### Acknowledgement
- Our codes are based on [EasyEdit](https://github.com/zjunlp/EasyEdit), and we thank their outstanding open-source contributions.
- Our data is based on vanilla ZsRE dataset ([Levy et al., 2017](https://aclanthology.org/K17-1034/)) and the portability QA pairs collect by [Yao et al. (2023)](https://arxiv.org/abs/2305.13172).
    - [Zero-Shot Relation Extraction via Reading Comprehension](https://aclanthology.org/K17-1034/) (CoNLL 2017)
    - [Editing Large Language Models: Problems, Methods, and Opportunities](https://arxiv.org/abs/2305.13172) (arXiv preprint 2023)

### Citation
If you find this work is useful or use the data in your work, please consider cite our paper:
```
@article{wang2023cross,
  title={Cross-Lingual Knowledge Editing in Large Language Models},
  author={Wang, Jiaan and Liang, Yunlong and Sun, Zengkui and Cao, Yuxuan and Xu, Jiarong},
  journal={arXiv preprint arXiv:2309.08952},
  year={2023}
}
```
We also recommend citing the vanilla ZsRE dataset and Yao et al. (2023):
```
@inproceedings{levy-etal-2017-zero,
    title = "Zero-Shot Relation Extraction via Reading Comprehension",
    author = "Levy, Omer  and
      Seo, Minjoon  and
      Choi, Eunsol  and
      Zettlemoyer, Luke",
    booktitle = "Proceedings of the 21st Conference on Computational Natural Language Learning ({C}o{NLL} 2017)",
    month = aug,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/K17-1034",
    doi = "10.18653/v1/K17-1034",
    pages = "333--342"
}

@article{yao2023editing,
  title={Editing Large Language Models: Problems, Methods, and Opportunities},
  author={Yao, Yunzhi and Wang, Peng and Tian, Bozhong and Cheng, Siyuan and Li, Zhoubo and Deng, Shumin and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2305.13172},
  year={2023}
}
```