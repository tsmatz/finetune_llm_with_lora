# Fine-tuning LLM with LoRA (Low-Rank Adaptation)

This example shows you [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) implementation with step-by-step explanation in notebook.<br>
This example is also made runnable in the mainstream computing with small footprint - such as, in a signle GPU of Tesla T4 or consumer GPU (NVIDIA RTX) - so that you can soon run and check results.

> Note : For more large language models, please run training on multiple devices with large-scale model-parallel techniques. (See [here](https://tsmatz.wordpress.com/2023/09/21/model-parallelism/).)

Today LoRA can be easily implemented with ```PEFT``` package, but this example implements LoRA from scratch (manually) in a step-by-step manner.

To focus on LoRA implementation, here I download pre-trained model from Hugging Face, unlike [official example](https://github.com/microsoft/LoRA). (But I'll use PyTorch training loop.)

## 1. Set-up and Install

To run this example, please install prerequisite's software and setup your environment as follows.<br>
In the following setting, I have used a GPU-utilized virtual machine (VM) with "Ubuntu Server 20.04 LTS" image in Microsoft Azure.

### Install GPU driver (CUDA)

Install CUDA (NVIDIA GPU driver) as follows.

```
# compilers and development settings
sudo apt-get update
sudo apt install -y gcc
sudo apt-get install -y make

# install CUDA
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
sudo sh cuda_12.2.2_535.104.05_linux.run
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64" >> ~/.bashrc
source ~/.bashrc
```

### Install packages

Install PyTorch, Hugging Face transformer, and other libraries as follows.

```
# install and upgrade pip
sudo apt-get install -y python3-pip
sudo -H pip3 install --upgrade pip
# install packages
pip3 install torch transformers pandas matplotlib
# install jupyter for running notebook
pip3 install jupyter
```

## 2. Fine-tune (Train)

Run jupyter notebook.

```
jupyter notebook
```

Open jupyter notebook in browser, and run the following examples.

| Example                                                              | Description                                       |
| -------------------------------------------------------------------- | ------------------------------------------------- |
| [01-finetune-opt-with-lora.ipynb](01-finetune-opt-with-lora.ipynb)   | Fine-tuning Meta's OPT-125M with LoRA             |
| [02-finetune-gpt2-with-lora.ipynb](02-finetune-gpt2-with-lora.ipynb) | Fine-tuning OpenAI's GPT-2 small (125M) with LoRA |
