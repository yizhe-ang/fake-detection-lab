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
│   └── exif_sc.npy     <-- Store model weights here
├── assets
├── configs             <-- Configuration files for scripts
├── data                <-- To store downloaded data
├── notebooks
├── requirements
├── src
│   ├── attacks         <-- Implementation of adversarial attacks
│   ├── datasets        <-- Data loading classes
│   ├── evaluation      <-- Evaluation classes and utilities
│   ├── models          <-- Implementation of detection models
│   ├── structures.py
│   └── utils.py
├── evaluate.py         <-- Main entry point for evaluation
└── ...
```

# Usage
```
python evaluate.py --config configs/evaluate/config.yaml
```
The above command runs the evaluation on a clean dataset, and also on the dataset after it has been adversarially perturbed, based on the settings specified in the configuration file.

# Resources
### Model Conversion
- Microsoft's [MMdnn](https://github.com/microsoft/MMdnn)
- [ONNX](https://github.com/onnx/onnx)

### Survey Papers
- Media Forensics and DeepFakes: an overview ([Luisa Verdoliva, 2020](https://arxiv.org/abs/2001.06564))

### Fake Detectors
- Fighting Fake News: Image Splice Detection via Learned Self-Consistency ([Huh et al., ECCV 2018](https://minyoungg.github.io/selfconsistency/))

### Adversarial Machine Learning
- Adversarial Attack on Deep Learning-Based Splice Localization ([Rozsa et al., 2020](https://arxiv.org/abs/2004.08443))

# Datasets
- [Columbia Image Splicing Detection Evaluation Dataset](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)
- [Realistic Tampering Dataset](http://pkorus.pl/downloads/dataset-realistic-tampering)
- [In-the-Wild Dataset](https://minyoungg.github.io/selfconsistency/)
- [Scene Completion Using Millions of Photographs - Dataset](http://graphics.cs.cmu.edu/projects/scene-completion/)
- [DSO-1 Dataset](https://recodbr.wordpress.com/code-n-data/#dso1_dsi1)
