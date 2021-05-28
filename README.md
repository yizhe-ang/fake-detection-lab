# fake-detection-lab
Media Forensics / Fake Image Detection experiments in PyTorch.

# Installation
We use `conda` for managing Python and CUDA versions, and `pip-tools` for managing Python package dependencies.
1. Specify the appropriate `cudatoolkit` and `cudnn` versions to install on your machine in the `environment.yml` file.
2. To create the `conda` environment, run: `conda env create`
3. Activate the environment: `conda activate fake-detection-lab`
4. Install all necessary packages: `pip-sync requirements/prod.txt`

# Model Artifacts
All model artifacts can be accessed and downloaded [here](https://drive.google.com/drive/folders/1Qm1WUUithm0dE1qnJXGfoCbMG37jq3mW?usp=sharing).
- `exif_sc.npy`: EXIF-SC model weights

# Project Structure
```
├── artifacts
│   └── exif_sc.npy      <-- Store model weights here
├── assets
├── configs              <-- Configuration files for scripts
├── data
│   ├── downloaded       <-- To store downloaded data
│   └── raw              <-- Dataset metadata
├── notebooks
├── requirements
├── src
│   ├── attacks          <-- Implementation of adversarial attacks
│   ├── datasets         <-- Data loading classes
│   ├── evaluation       <-- Evaluation classes and utilities
│   ├── models           <-- Implementation of detection models
│   ├── trainers         <-- Classes for model training
│   ├── structures.py
│   └── utils.py
├── evaluate.py          <-- Main entry point for evaluation
├── non_adv_evaluate.py  <-- Main entry point for evaluation
├── train.py             <-- Main entry point for training
└── ...
```

# Usage

## Training
```
python train.py \
    --config configs/train/exif_sc.yaml \
    --checkpoints_dir checkpoints \
    --gpu 0
```
Runs training on a dataset, based on the settings specified in the configuration file. Weights are saved as a torch `.ckpt` file in the specified directory.

More [info](src/trainers/README.md).

## Evaluation
More info [here](src/models/exif_sc/README.md) and [here](src/attacks/README.md)
### Without Adversarial Attack
```
python non_adv_evaluate.py \
    --config configs/evaluate/non_adv.yaml \
    --weights_path path/to/weights.{npy, ckpt}
```
Runs the evaluation on a dataset, based on the settings specified in the configuration file.

### With Adversarial Attack
```
python evaluate.py \
    --config configs/evaluate/adv.yaml \
    --weights_path path/to/weights.{npy, ckpt}
```
Runs the evaluation on a clean dataset, and also on the dataset after it has been adversarially perturbed, based on the settings specified in the configuration file.

# Datasets
All metadata for the datasets used can be found [here](data/raw).

# Resources
### Model Conversion
- Microsoft's [MMdnn](https://github.com/microsoft/MMdnn)
- [ONNX](https://github.com/onnx/onnx)

### Survey Papers
- Media Forensics and DeepFakes: an overview ([Luisa Verdoliva, 2020](https://arxiv.org/abs/2001.06564))
- A Survey of Machine Learning Techniques in Adversarial Image Forensics ([Nowroozia et al., 2020](https://arxiv.org/abs/2010.09680))

### Fake Detectors
- Fighting Fake News: Image Splice Detection via Learned Self-Consistency ([Huh et al., ECCV 2018](https://minyoungg.github.io/selfconsistency/))

### Adversarial Machine Learning
- Adversarial Attack on Deep Learning-Based Splice Localization ([Rozsa et al., 2020](https://arxiv.org/abs/2004.08443))
