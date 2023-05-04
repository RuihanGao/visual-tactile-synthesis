# Controllable Visual-Tactile Synthesis

### [Project Page](https://visual-tactile-syn.github.io/) | [Paper](https://arxiv.org/abs/)

**tl;dr** Content creation beyond visual output: We present Visual-Tactile-Syn, a conditional image-to-image approach to synthesize visual appearance and tactile geometry of different materials, given a handcrafted or DALL⋅E 2 sketch. The visual and tactile outputs can be rendered on a surface haptic device like TanvasTouch® where users can slide on the screen to feel the rendered textures. Turn the audio ON to hear the sound of the rendering.

[![Teaser video](https://img.youtube.com/vi/TdwPfwsGX3I/default.jpg)](https://youtu.be/TdwPfwsGX3I)

***Controllable Visual-Tactile Synthesis*** <br>
[Ruihan Gao](https://ruihangao.com/), [Wenzhen Yuan](http://robotouch.ri.cmu.edu/yuanwz/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)<br>
Carnegie Mellon University<br>
arXiv, 2023

## Updates 
* [05/04] We release the [visual-tactile-syn arxiv](https://arxiv.org/abs/), which synthesizes synchronized visual-tactile output from a sketch/text input and renders the multi-modal output on a surface haptic device TanvasTouch®.

## Timeline
We are plan to release our code and dataset in the following steps:
- [x] Inference and Evaluation code 
- [x] Preprocessed data of all 20 garments in our <i>TouchClothing</i> dataset
- [x] Pretrained model (ours & baselines) on the <i>TouchClothing</i> dataset
- [ ] Training code
- [ ] Data preprocessing code for camera and GelSight R1.5 data

## Setup
This code was tested with Python 3.8 and [Pytorch](https://pytorch.org/) 1.11.0.
```
git clone https://github.com/RuihanGao/Visual-Tactile-Syn.git
conda env create -f environment.yml
conda activate VTS
```

### Dataset
We provide the preprocessed data for our <i>TouchClothing</i> dataset, which contains 20 pieces of garments of various shapes and textures. <br>
20 Objects included in <i>TouchClothing</i> dataset:
<br>
<img src="assets/figure2_dataset.png" alt=“figure2_dataset” width="400">
<br>
Example of preprocessed data:
<br>
<img src="assets/figure4_sample_data.png" alt=“figure4_sample_data” width="400">
<br>

Use the following commands to download and unzip the dataset. <br>
(1) Download the preprocessed data from Google Drive via the following command: 
(install `gdown` if you haven't done so) <br>
Total size: 580M, which takes about 20s to download. <br>
```
pip install gdown
gdown "https://drive.google.com/uc?export=download&id=1VlgYpDSxQP70sYpFERHuzKnTNIH4Gf4s"
unzip TouchClothing_dataset.zip
```
(2) Put the unzipped folder named `datasets` in the code repo.

### Pre-trained models
We provide the pretrained models for 1. our method 2. baselines included in our paper. For each method, we provide 20 models, one for each object in our <i>TouchClothing</i> dataset.
See the Google drive folder [here](https://drive.google.com/drive/folders/1ewamRPEyKir3jiPoeS7NBmZPZbFqxEoB?usp=sharing). To use them, <br>
(1) download the checkpoints <br>
* checkpoints for our method (124M): `gdown https://drive.google.com/uc?export=download&id=11y2jP2vT7CtBIaEDcjROZ5hupHsYWG8D`
* checkpoints for baselines (21.5G): `gdown https://drive.google.com/uc?export=download&id=1vBd7awJJml5wDEp5gQGeyefpvilTOByw`

(2) After unzipping, put all pre-trained models in the folder `checkpoints` to load them properly in the testing code.

## Usage
In general, our pipeline contains two steps. We first feed the sketch input to our model to synthesize synchronized visual and tactile output that imitates the appearance and geometry of the training object. Then we convert the tactile output to a friction map required by TanvasTouch and render the multi-modal output on the surface haptic device, where you can <i>see</i> and <i>feel</i> the object simultaneously.

### Train our model
```
material_idx=0
python -m experiments SingleG_AllMaterials_baseline_ours train $material_idx
```
where the material_idx set which object in the dataset to use. Replace the material_idx by any other number within the length of material list or use 'all' to run multiple experiments at once.
The list of the material can be found in the launcher file `experiments/SingleG_AllMaterials_baseline_ours_launcher.py` 

Note: Loading the dataset to cache before training may take up to 20-30 mins and the training takes about 16h on a single A5000 GPU. Please be patient. <br>
For a proof-of-concept training, set `data_len` in `SingleG_AllMaterials_baseline_ours_launcher` and `verbose_freq` in `models/sinskitG_model.py` to a smaller number, e.g. 3 or 10.

### Test our model
```
material_idx=0
python -m experiments SingleG_AllMaterials_baseline_ours test $material_idx
```
The results will be stored in the `results` directory.

### Visual-Tactile Synthesis (baseline comparison)
<img src="assets/figure6_baseline.png" alt=figure6_baseline width="800">
<br>

### Swapping different sketches & materials
<img src="assets/figure7_swap_sketch.png" alt=figure7_swap_sketch width="800">
<br>

### Text-guided visual-tactile synthesis
<img src="assets/figure8_DALLE_sketch.png" alt=figure8_DALLE_sketch width="800">
<br>

**Please see our website and paper for more interactive and comprehensive results**

## Citation
``` bibtex
@article{xxxxx,
  title={Controllable Visual-Tactile Synthesis},
  author={Gao, Ruihan and Yuan, Wenzhen and Zhu, Jun-Yan},
  journa={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023},
}
```

## Acknowledgement
We thank Sheng-Yu Wang, Kangle Deng, Muyang Li, Aniruddha Mahapatra, and Daohan Lu for proofreading the draft. We are also grateful to Sheng-Yu Wang, Nupur Kumari, Gaurav Parmar, George Cazenavette, and Arpit Agrawal for their helpful comments and discus- sion. Additionally, we thank Yichen Li, Xiaofeng Guo, and Fujun Ruan for their help with the hardware setup. Ruihan Gao is supported by A*STAR National Science Scholarship (Ph.D.).

Our code base is built upon [Contrastive Unpaired Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation).
