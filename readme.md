# Offical Code Implementation for EndoGSLAM

> EndoGLSAM: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries using Gaussian Splatting.
> Kailing Wang*, Chen Yang*, Yuehao Wang, Sikuang Li, Yan Wang, Qi Dou, Xiaokang Yang, Wei Shen†


<a href="https://github.com/Loping151/EndoGSLAM"> <img alt="Github Repository" src="https://img.shields.io/badge/Github-Repository-blue?logo=github&logoColor=blue"> </a>
<a href="https://arxiv.org/abs/2403.15124"> <img alt="Paper" src="https://img.shields.io/badge/Arxiv-Paper-red?logo=arxiv&logoColor=red"> </a>
<a href="https://loping151.github.io/endogslam"> <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-green?logo=pagekit&logoColor=white"> </a>

## 🛠️ Requirements

You can install them following the instructions below.

  
```bash
conda create -n endogslam python=3.10 # recommended
conda activate endogslam
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu118
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # alternatively
pip install -r requirements.txt
```

Latest version is recommended for all the packages unless specified, but make sure that your CUDA version is compatible with your `pytorch`.

Tested machines: Ubuntu22.04+RTX4090, Ubuntu22.04+RTX2080Ti, Windows10+RTX2080.

## ⚓ Preparation
We use the [C3VD](https://durrlab.github.io/C3VD/) dataset. You can use the scripts in `data/prepeocess_c3vd` to preprocess the dataset. We also provide the preprocessed dataset [here]().

After you get prepared, the data structure should be like this:

```
- data/
  |- C3VD/
    |- cecum_t1_b/
      |- color/
      |- depth/
      |- pose.txt
    |- cecum_t3_a/
- scripts/
  |- main.py
- utils/
- other_folders/
- readme.md
```

If you want to use your own dataset, you can modify the dataloader or organize your data in the same structure.

## 🚀 Training and 💯 Evaluation

Training arguments can be found in `scripts/main.py`. To use the default setting:

```bash
python scripts/main.py configs/c3vd/c3vd_base.py
```

To evaluate on a single scene:
```bash 
python scripts/calc_metrics.py --gt data/C3VD/sigmoid_t3_a --render experiments/C3VD_base/sigmoid_t3_a --test_single
```

We use the same visualization scripts as [SplaTAM](https://github.com/spla-tam/SplaTAM) for debug only.

## 🏗️ Todo
- [ ] Release reconstruction results for comparison
- [] Release preprocessed dataset
- [x] Release code
- [x] Release paper


## Acknowledgements
We would like to acknowledge the following inspiring work:
- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) (Bernhard Kerbl et al.)
- [SplaTAM](https://github.com/spla-tam/SplaTAM) (Nikhil Keetha et al.)

## Citation

If you find this code useful for your research, please use the following BibTeX entries:

```
    @article{wang2024endogslam,
        title={EndoGSLAM: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries using Gaussian Splatting},
        author={Kailing Wang and Chen Yang and Yuehao Wang and Sikuang Li and Yan Wang and Qi Dou and Xiaokang Yang and Wei Shen},
        journal={arXiv preprint arXiv:2403.15124},
        year={2024}
    }
```