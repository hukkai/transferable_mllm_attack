# Transferable Adversarial Attacks for Multimodal Large Language Models

Official implementation of **Transferable Visual Adversarial Attacks for Proprietary Multimodal Large Language Models**.

📄 [Read the paper on arXiv](https://arxiv.org/abs/2505.01050)

This repository contains the codebase used in our experiments.  
Below is a simple instruction for how to run our code. Detailed instructions for setup and usage will be provided by Oct 30, 2025.

## Quick start

1. Prepare the dataset: create a `data` folder under this repo and run the following command under the `data` folder:
```
import os, kagglehub
path = kagglehub.dataset_download("google-brain/nips-2017-adversarial-learning-development-set")
os.system(f"mv {path} ./nips2017_adv_dev/")
```

2. Link the ImageNet dataset (only the validation set is needed) to the `data` folder. Extract features for ImageNet validation set images:
```
python3 utils/extract_feat.py --mdoel_id 0
```
We will also provide our extracted features soon.

3. Optimize the attack:
```
bash run.sh
```
Generated attacks will be saved at `results/saved_folder/`, where "saved_folder" is specified in `batch_attack.py`, for example `s299_x9_eps8`. Images with file name starting with "ema_" are the final outputs. 

This [google drive](https://drive.google.com/drive/folders/1jCC2Nu8_miZQEf_LbKHUejRYzhhOgVQ6?usp=sharing) contains some generated images from our method.

