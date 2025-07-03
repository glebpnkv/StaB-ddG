## StaB-ddG: Predicting mutational effects on protein binding from folding energy

StaB-ddG predicts mutational effects on binding interaction energies ($\Delta \Delta G$) given the 3D structure of a reference (i.e. wild type) interface.  This is a companion repository for [our paper](https://icml.cc/virtual/2025/poster/45926). 

We provide
* [Installation instructions](#setup) 
* [An example demonstrating how to make predictions](#predicting-binding-ddg), and
* [Training and inference code to reproduce the results of our paper](#training-and-evaluation).

## Setup
We use conda to manage required dependencies. The are few required packages (see `environment.yaml`); alternatively to creating a new environment, you can run StaB-ddG with an existing environment with PyTorch, after `pip install tqdm scipy wandb pandas`.
```
conda env create -f environment.yaml
conda activate stabddg
```

## Predicting binding $\Delta \Delta G$
StaB-ddG requires as input PDB files and a csv file with mutations of interest, and predicts the $\Delta \Delta G$ in the unit of kilocalories per mole (kcal/mol). We use the convention that a negative value ($\Delta \Delta G < 0$) represents a destabilizing mutation.

### Predicting $\Delta \Delta G$ for one mutant
For a single ddG prediction, StaB-ddG requires a pdb file, the chains specifying the two binders, and a mutation string.
* `--pdb_path` path to the wild type structure.
* `--mutation` a single mutation or multiple mutations separated by commas. For example, `YH103H,QC7R` denotes a double mutant. The first character of a mutation string denotes the wild type amino acid, the second character the chain, followed by the position in the chain, and lastly the amino acid to mutate to. For example, `YH103H` denotes a mutation from Y to H at position 103 of chain H. 
* `--chains` chains specifying what interface the $\Delta \Delta G$ is computed over for a multichain complex, separated by an underscore. For example, `ABC_DE` denotes that the energy is computed between the interface of chains `ABC` and `DE`. 
* `--mc_samples` the number of Monte Carlo samples to average over for variance reduction. This is set to 20 by default.

#### Example
We provide an example that predicts the effect of the mutation `EA63Q,QD30V,KA66A` on `1AO7`.
```
python run_stabddg.py --pdb_path examples/one_mutation/1AO7.pdb \
    --mutation EA63Q,QD30V,KA66A --chains ABC_DE
```
This will create a directory `1AO7/` with the following files:
* `output.csv` contains the predicted `$\Delta \Delta G$` in column `pred_1`
* `1AO7.pdb` the wild type structure
* `1AO7_ABC.pdb` the wild type structure with only chains `ABC`
* `1AO7_DE.pdb` the wild type structure with only chains `DE`
* `1AO7_ABC_DE_cache.pkl` cache of intermediate output
* `input.csv` intermediate input file

#### Model checkpoints
We provide the following checkpoints. The final checkpoint, `stabddg.pt`, should be used for inference.
```
./model_ckpts/
├── proteinmpnn.pt # soluble ProteinMPNN weights (v_48_020.pt) from https://github.com/dauparas/ProteinMPNN
├── stability_finetuned.pt # model weights after stability fine-tuning
├── stabddg.pt # final model fine-tuned on both stability and binding data.
```

### Predicting $\Delta \Delta G$ for a list of mutants
A list of mutations across different complexes can be provided in the form of a mutation csv file (`--csv_path`). The mutation csv file should contain two columns, `#Pdb` and `mutation`. The `#Pdb` column should contain the name of the complex PDB file concatenated with an underscore and the chains without the `.pdb` suffix (e.g. `1AO7_ABC_DE`). The `mutation` column contains the mutations of interest with the same format as described above. In addition, a `--pdb_dir` should be specified that contains the wild type structures of the mutations of interest.
#### Example
An example is provided in `examples/list_of_mutations`. Predictions can be generated using the following command:
```
python run_stabddg.py --pdb_dir examples/list_of_mutations/pdbs \
    --csv_path examples/list_of_mutations/mutations.csv
```
By default, the output will be saved in the same directory as the mutant csv file, with similar intermediate outputs saved in `--pdb_dir` as in the one mutant case.

## Training and evaluation
The two fine-tuning sections map onto the two fine-tuning steps described in the paper. First, we fine-tune on the Megascale protein folding stability dataset, then fine-tune on SKEMPI. The fine-tuning runs can be optionally tracked on Wandb with the flag `--wandb`. 

### Fine-tuning on Megascale protein folding stability data
First download the data from https://zenodo.org/records/7992926. Specifically, the files needed are `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` and `AlphaFold_model_PDBs.zip`.   This takes several minutes.
```
data_dir=<destination_for_files>
cd $data_dir
wget https://zenodo.org/records/7992926/files/AlphaFold_model_PDBs.zip 
wget https://zenodo.org/records/7992926/files/Processed_K50_dG_datasets.zip 
unzip AlphaFold_model_PDBs.zip 
unzip Processed_K50_dG_datasets.zip 
```

Then from the repo directory, launch stabiity finetuning as
```
python stability_finetune.py \
    --stability_data $data_dir/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
    --pdb_dir $data_dir/AlphaFold_model_PDBs
```
The above script requires torch to access a single gpu with `torch.device='cuda'`.

The model checkpoints will be saved in `cache/stability_finetuned` by default.

### Fine-tuning on SKEMPI binding energy data
A filtered version of the SKEMPI csv is provided in `data/SKEMPI/filtered_skempi.csv`. The PDB files can be downloaded from https://life.bsc.es/pid/skempi2/database/index.
```
cd $data_dir
wget  https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz 
tar -xvzf SKEMPI2_PDBs.tgz # Data now in $data_dir/PDBs/
```

After the download, finetune from the repo directory.
```
python skempi_finetune.py --train_split_path data/SKEMPI/train_pdb.pkl \
    --run_name RUN_NAME \
    --checkpoint model_ckpts/stability_finetuned.pt \
    --skempi_pdb_dir $data_dir/PDBs \
    --skempi_path data/SKEMPI/filtered_skempi.csv
```
The model checkpoints will be saved in `cache/skempi_finetuned` by default.

#### Train/test splits
Training and evaluation on the SKEMPI binding energy data involves several files that we provide in `./data/SKEMPI/`.
```
./data/SKEMPI/
├── skempi_v2.csv # unfiltered SKEMPI csv file from https://life.bsc.es/pid/skempi2/database/index
├── quality_filtering.ipynb # filtering script corresponding to the criteria described in our paper
├── skempi_splits.ipynb # splitting script corresponding to the interface-homology cluster splitting described in our paper
├── filtered_skempi.csv # the output of quality_filtering.ipynb (containing a subset of the rows in `skempi_v2.csv`).  An additional column indicates the cluster assignment associated the train/test split.
├── test_clusters.(pkl/txt) # test split
├── test_pdb.pkl # test split pdbs
├── train_clusters.(pkl/txt) # train split
├── train_pdb.pkl # train split pdbs
```

### Running evaluation on SKEMPI
To reproduce results from the paper, run the following command.
```
python skempi_eval.py --run_name EVAL \
    --batch_size 10000 \
    --checkpoint "model_ckpts/stabddg.pt" \
    --skempi_path data/SKEMPI/filtered_skempi.csv \
    --skempi_pdb_dir data/SKEMPI_v2/PDBs \
    --skempi_pdb_cache_path cache/skempi_full_mask_pdb_dict.pkl \
    --skempi_split_path "data/SKEMPI/test_pdb.pkl"
```

## Acknowledgements
* Code built upon [ProteinMPNN](https://github.com/dauparas/ProteinMPNN/blob/main/protein_mpnn_utils.py) and [Graph-Based Protein Design](https://github.com/jingraham/neurips19-graph-protein-design), specifically [./stabddb/mpnn_utils.py](https://github.com/dauparas/ProteinMPNN/training/model_utils.py). 
* Folding stability data collected by [Tsuboyama et al.](https://www.nature.com/articles/s41586-023-06328-6). And train/test splits by Dieckhaus et al., specifically [`rocklin/mega_splits.pkl`](https://github.com/Kuhlman-Lab/ThermoMPNN/blob/main/dataset_splits/mega_splits.pkl)(https://github.com/Kuhlman-Lab/ThermoMPNN)
* Binding energy data curated from SKEMPIV2 by [Jankauskaite et al.](https://academic.oup.com/bioinformatics/article/35/3/462/5055583)
