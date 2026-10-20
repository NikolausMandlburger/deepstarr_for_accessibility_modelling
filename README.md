## Sequence to accessibility modelling with DeepSTARR

This directory contains a tutorial showcasing how to do sequence to accessibility modelling with using DeepSTARR type convolutional neural networks.

The tutorial ipython notebooks containing python and bash code can be found in the src subfolder, they can be opened using jupyter lab.

### Setup

All software needed to run the tutorial (PyTorch, tangermeme, modisco-lite, bedtools, MACS3, ...) is specified in `environment.yml` and can be installed in one step with conda/mamba:

```
mamba env create -f environment.yml
conda activate deepstarr_accessibility_tutorial
```

By default this installs a CUDA-enabled PyTorch build.

The tutorial cover
- Input data processing and region selection
- Training DeepSTARR accessibility models
- Interpreting what models have learned 
