# Fine-tuning LLM with LoRA (Low-Rank Adaptation)

LoRA (Low-Rank Adaptation) is one of mostly used parameter-efficient fine-tuning (PEFT) methods today.

This example shows you [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) implementation from scratch (manually) in a step-by-step manner (without ```PEFT``` package), and also shows you clear ideas behind this implementation in IPython notebook.

This is also runnable in the mainstream hardware with small footprint - such as, a signle GPU of Tesla T4, consumer GPUs (NVIDIA RTX), etc - for you to try this code easily.

| Example                                                              | Description                                                             |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [01-finetune-opt-with-lora.ipynb](01-finetune-opt-with-lora.ipynb)   | Fine-tuning Meta's OPT-125M with LoRA<br>(Also, explaining LoRA method) |
| [02-finetune-gpt2-with-lora.ipynb](02-finetune-gpt2-with-lora.ipynb) | Fine-tuning OpenAI's GPT-2 small (124M) with LoRA                       |

Unlike examples in [official repository](https://github.com/microsoft/LoRA), here I download pre-trained models to focus on LoRA implementation.

> Note : In this repository, Hugging Face API is used to download pre-trained models and I then apply regular PyTorch training loop for fine-tuning. (I don't use blackboxed ```Trainer``` class in Hugging Face API.)

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

Download this repository.

```
git clone https://github.com/tsmatz/finetune_llm_with_lora
```

Run jupyter notebook.

```
jupyter notebook
```

Open jupyter notebook in browser, and run examples in this repository.
