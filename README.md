# KimiaNet Colorectal Cancer Histology Classifier

This project fine-tunes **KimiaNet** (DenseNet-121 backbone) to classify 8 colorectal histology texture classes from the Kather et al. (2016) dataset.

## Dataset

**Dataset:** *Collection of textures in colorectal cancer histology* (DOI: [10.5281/zenodo.53169](https://doi.org/10.5281/zenodo.53169))

### Classes (8)
- Tumor epithelium (01_TUMOR)
- Simple stroma (02_STROMA)
- Complex stroma (03_COMPLEX)
- Immune cells (04_LYMPHO)
- Debris (05_DEBRIS)
- Normal mucosal glands (06_MUCOSA)
- Adipose tissue (07_ADIPOSE)
- Background (08_EMPTY)

## Project Structure

```
kather_kimianet_workspace/
├── data/
│   ├── raw/                            # Raw downloaded zip files
│   ├── Kather_texture_2016_image_tiles_5000/  # 5000 image tiles dataset
│   └── Kather_texture_2016_larger_images_10/  # 10 larger images dataset
├── weights/
│   └── KimiaNetKerasWeights.h5         # Pre-trained KimiaNet weights
└── artifacts/                          # Saved model outputs
    ├── kimianet_kather_classifier.keras  # Trained model
    └── class_to_idx.json               # Class mapping
```

## Model Architecture

- **Backbone:** DenseNet-121 pre-trained with KimiaNet weights
- **Modifications:**
  - Global Average Pooling
  - Batch Normalization
  - Dropout (0.3)
  - Dense output layer (8 classes with softmax activation)

## Training Approach

The model is trained in two phases:
1. **Phase A (Warm-up):** Train only the classification head with frozen backbone
   - Learning rate: 1e-3
   - Optimizer: AdamW with weight decay 1e-4
   - Epochs: 5 (with early stopping)

2. **Phase B (Fine-tuning):** Gradually unfreeze and train the last dense block
   - Learning rate: 1e-4
   - Optimizer: AdamW with weight decay 1e-4
   - Epochs: 15 (with early stopping)
   - Unfreezes: conv5 layers, dense_block4, and related batch norm layers

## Data Preprocessing & Augmentation

- Resize images to 224×224 (DenseNet-121 input size)
- Normalize pixel values to [0,1]
- **Augmentations:**
  - Random horizontal and vertical flips
  - 90-degree rotations
  - Random brightness adjustments
  - Random contrast adjustments

## Dataset Split

- Training: 70%
- Validation: 15%
- Testing: 15%
- Stratified by class to maintain class distribution

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- Pillow

## Usage

1. Run the Jupyter notebook `Kather_KimiaNet_Classifier.ipynb` to:
   - Download the dataset
   - Preprocess images
   - Train the model
   - Evaluate performance
   - Save the trained model

2. The trained model and class mapping are saved in the `artifacts` directory.

## Citation

If you use this code or model, please cite:

```
@article{kather2016multi,
  title={Multi-class texture analysis in colorectal cancer histology},
  author={Kather, Jakob Nikolas and Weis, Cleo-Aron and Bianconi, Francesco and Melchers, Susanne M and Schad, Lothar R and Gaiser, Timo and Marx, Alexander and Z{\"o}llner, Frank Gerrit},
  journal={Scientific reports},
  volume={6},
  number={1},
  pages={27988},
  year={2016},
  publisher={Nature Publishing Group UK London}
}

@article{riasatian2021fine,
  title={Fine-tuning and training of densenet for histopathology image representation using TCGA diagnostic slides},
  author={Riasatian, Abtin and Babaie, Morteza and Maleki, Danial and Kalra, Shivam and Valipour, Mojdeh and Hemati, Sobhan and Zaveri, Manit and Kraus, Jacob and Sheikhzadeh, Fahime and Tizhoosh, Hamid R and others},
  journal={Medical Image Analysis},
  volume={70},
  pages={102032},
  year={2021},
  publisher={Elsevier}
}
```

## License

This project is available under the terms specified in the dataset's original license from Zenodo.