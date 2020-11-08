# Future work
1. Hyperparameter optimization for several parameter to find the optimal solution.
2. Experiment with different data augmentation strategies.
    * Apply random (fc, bw) bandpass filter.
    * Apply random power threshold cutoff.


# Training

```shell

python train.py \
    --output_dir=./project_dir \
    --dataset_dir=./data/intermediate/ \
    --batch_size=32
```

The training scripts follows the model implementation provided in the paper (Learning Sleep Stages from Radio Signals: A Conditional Adversarial Architecture) with slight modification to the training optimization steps.

# Predition

```shell

python pred.py \
    --input_dir=./project_dir
```
