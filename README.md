# fake-detection-lab
Media Forensics / Fake Image Detection experiments in PyTorch.

# Installation
1. Install Python 3.7 and [PyTorch](https://pytorch.org/).
2. ```pip install -r requirements.txt```

# Model Artifacts
All model artifacts can be accessed and downloaded [here](https://drive.google.com/drive/folders/1Qm1WUUithm0dE1qnJXGfoCbMG37jq3mW?usp=sharing).
- `exif_sc.npy`: EXIF-SC model weights

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
