# Venus-MAXWELL
Source code of Venus-MAXWELL: Efficient Learning of Protein-Mutation Stability Landscapes using Protein Language Models

#### If you find this work useful, please star the repo! (Click the star button in the top right corner of the page)
Thank you! (づ｡◕‿‿◕｡)づ

## Recommended environment
It is highly recommended to start a new conda environment from scratch due to potential CUDA compatability issues between pytorch and the pytorch-geometric package required for the inverse folding model.

To set up a new conda environment with required packages, run the following commands:

```bash
conda create -n maxwell python=3.10
conda activate maxwell

# Install pytorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install lightning==2.4
pip install numpy==1.26.0
pip install biopython==1.85
pip install pandas==2.1.0

# Install esm-inverse-folding dependencies
pip install biotite==0.40.0
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
pip install fair-esm

# Install esm-2 and prosst dependencies (optional)
pip install transformers
```

## Clone the repository
```bash
git clone https://github.com/ai4protein/Venus-MAXWELL.git
cd Venus-MAXWELL
```

## Generate ddG predictions of all possible single mutations of a protein from PDB file with Venus-MAXWELL(ESM-IF)

Download the weights from [here](https://drive.google.com/file/d/1kB1IweY43yNLoIkOovS7GezXS-3Ro0eA/view?usp=drive_link) and put them in the `weights` folder.
```bash
ls weights
```
should output:
```bash
esmif-maxwell.ckpt
```

```bash
python predict_ddg.py --pdb_file example_data/fireprotdb_1AG2_ddG.pdb --ckpt_path weights/esmif-maxwell.ckpt --output_file example_data/fireprotdb_1AG2_ddG.csv --device cpu
```

## Train and Test Venus-MAXWELL (ESM-2) on the ddg_datasets
```bash
python check_dataset.py --dataset_path example_datasets/ddg_datasets/train
python check_dataset.py --dataset_path example_datasets/ddg_datasets/valid
python check_dataset.py --dataset_path example_datasets/ddg_datasets/test
python -m maxwell.esm2.train --train_folder example_datasets/train --valid_folder example_datasets/valid --test_folder example_datasets/test
```

### Citation
If you find this work useful, please cite:
```bibtex
@article {Yu2025.05.30.656964,
	author = {Yu, Yuanxi and Jiang, Fan and Ma, XinZhu and Zhang, Liang and Zhong, Bozitao and Ouyang, Wanli and Fan, Guisheng and Yu, Huiqun and Hong, Liang and Li, Mingchen},
	title = {Venus-MAXWELL: Efficient Learning of Protein-Mutation Stability Landscapes using Protein Language Models},
	elocation-id = {2025.05.30.656964},
	year = {2025},
	doi = {10.1101/2025.05.30.656964},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/06/02/2025.05.30.656964},
	eprint = {https://www.biorxiv.org/content/early/2025/06/02/2025.05.30.656964.full.pdf},
	journal = {bioRxiv}
}
```
