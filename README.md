# 3D Brain Tumor Segmentation (BraTS 2019) – Transformer-based Network

DIP (Digital Image Processing) project. A 3D encoder–decoder segmentation network built with PyTorch that uses
multi-scale attention blocks (TMMM / TMSM modules) to segment brain tumors from multi-modal MRI scans.

Everything lives in a single notebook: [dip-project.ipynb](dip-project.ipynb).

## Dataset

[BraTS 2019 Training Data (HGG)](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019)
— each patient folder has four MRI modalities (`flair`, `t1`, `t1ce`, `t2`) and a segmentation mask (`seg`) in NIfTI format.

Preprocessing (in `BraTSDataset`):
- Load the 4 modalities and stack them into a 4-channel volume
- Label 4 is remapped to 3 (classes: background, necrotic/non-enhancing core, edema, enhancing tumor)
- Crop to the non-zero tumor bounding box
- Per-channel z-score normalisation
- Resize to `64 x 64 x 64` (trilinear for image, nearest for mask)

## Model

- **Embedding**: 4x4x4 patch embedding (4 input channels -> `C[0]`)
- **Encoder**: 4 stages (channels `32, 64, 128, 256`), each with 3 `TMSM` blocks; later stages are downsampled with 2x2x2 convs
- **TMSM**: multi-head attention where keys/values come from depthwise-conv reduced (multi-scale) features
- **Decoder**: transposed-conv upsampling with skip connections, outputs at 4 resolutions (deep supervision)
- **Loss**: multi-class Dice loss with deep supervision weights `[0.125, 0.25, 0.5, 1.0]`
- **Metrics**: per-class Dice, mean Dice, mean IoU, Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET) Dice, accuracy

The notebook also contains an `OptimizedInference` class (FP16, `torch.compile`, batched inference) for fast prediction.

## Project structure

```
.
├── dip-project.ipynb   # data loading, model, training, evaluation, inference
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Install a CUDA build of PyTorch from <https://pytorch.org> if you want GPU training.

## Usage

1. Download the BraTS 2019 dataset and set the `ROOT` path in the dataset cell of the notebook
   (it is currently the Kaggle path `/kaggle/input/datasets/aryashah2k/brain-tumor-segmentation-brats-2019/...`).
2. Open `dip-project.ipynb` in Jupyter / VS Code / Kaggle and run the cells in order.
3. Training config (defaults in the notebook): 200 epochs, batch size 32, lr `1e-4`, seed 42, 80/20 train/val split.
   Multi-GPU is supported through `nn.DataParallel` (originally trained on 2x Tesla T4 on Kaggle).
4. The trained model is saved as `segmentation_model.pt`; the last cells plot loss/Dice curves and show
   slice-wise predictions with an interactive slider.

## Tech stack

Python, PyTorch, NumPy, NiBabel, pandas, Matplotlib, tqdm, ipywidgets
