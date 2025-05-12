#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p vip_gpu_ailab_low
#SBATCH -A ailab
module load compilers/cuda/12.2 cudnn/8.9.5.29_cuda12.x compilers/gcc/11.3.0
conda activate py310cjy
cd /home/bingxing2/ailab/caijinyu/image_captioning/

python main.py train_evaluate --config_file configs/resnet101_attention.yaml

#using: sbatch -N 1 --gres=gpu:1 -p vip_gpu_ailab -A ai4neuro run.sh

