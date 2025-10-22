# Transferable Adversarial Attacks for Multimodal Large Language Models

Official implementation of **Transferable Visual Adversarial Attacks for Proprietary Multimodal Large Language Models**.

📄 [Read the paper on arXiv](https://arxiv.org/abs/2505.01050)

This repository contains the codebase used in our experiments.  
Below is a simple instruction for how to run our code. Detailed instructions for setup and usage will be provided by Oct 30, 2025.

## Quick start

1. Prepare the dataset: create a `data` folder under this repo and run the following command under the `data` folder:import kagglehub

```
path = kagglehub.dataset_download("google-brain/nips-2017-adversarial-learning-development-set")
print("Path to dataset files:", path)
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



