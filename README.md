# xMINT: A Multimodal Integration Transformer for Xenium Gene Imputation

## Getting Started

This repository contains the implementation of the paper available on [OpenReview](https://openreview.net/forum?id=hnYLq2lwOv).

To get started, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/jichunxie/xMINT_manu_support
    cd xMINT_manu_support
    ```

2. Create a Conda environment with Python 3.8:
    ```bash
    conda create -n xmint_env python=3.8
    conda activate xmint_env
    ```

3. Install the required packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Input Data Format

You will need paired Xenium datasets from the same tissue to get started.

For each dataset, the input data should include:

1. **Xenium Output Bundle**: The data generated by Xenium.
2. **Post-Xenium H&E Image**: An OME-TIFF file of the H&E stained image corresponding to the Xenium output.

Unzip the Xenium Output Bundle folder and place the H&E image into the output folder.

Our paper utilized the following datasets:
- [FFPE Human Breast with Pre-designed Panel 1 Standard](https://www.10xgenomics.com/datasets/ffpe-human-breast-with-pre-designed-panel-1-standard)
- [Human Tonsil Data: Xenium Human Multi-Tissue and Cancer Panel 1 Standard](https://www.10xgenomics.com/datasets/human-tonsil-data-xenium-human-multi-tissue-and-cancer-panel-1-standard)

## Pipeline

To run the pipeline, follow these steps:

1. Run the preprocessing script:
    ```bash
    python preprocess_to_sequence.py
    ```

2. Run the main xMINT script:
    ```bash
    python xmint_main.py
    ```

Before running these scripts, ensure you update the sample names and other customized parameters as needed.