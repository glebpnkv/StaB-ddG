# StaB-ddG

# Setup 
```
conda create --name protddg
conda activate protddg
# depending on your CUDA driver version, can update to newer torch versions. 
# Here we provide an example for 11.8.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm wandb scipy pandas
```
# Predicting binding ddG
StaB-ddG requires as input PDB files and a csv file with mutations of interest.

## PDB files
For a single ddG prediction, StaB-ddG requires 3 PDB files --- one for each binder and one for the entire complex. Specifically, the naming of the PDB files should follow:
* Complex: `NAME.pdb`
* Binder 1: `NAME_binder1chains.pdb`
* Binder 2: `NAME_binder2chains.pdb`. 

The chains specify what interface the ddG is computed over for a multichain complex. For example, to predict the binding ddG of the complex 7STF for the interface between chains HL and ABC, we should provide the following three PDB files: `7STF.pdb`, `7STF_HL.pdb`, and `7STF_ABC.pdb`.

## Mutation csv file
The mutation csv file should contain two columns, `#Pdb` and `Mutation(s)_cleaned`. The `#Pdb` column should contain the name of the complex PDB file without the `.pdb` suffix (e.g. `7STF`). 

The `Mutation(s)_cleaned` contains the mutations of interest. Each entry can have multiple mutations separated by commas. For example, `YH103H,QC7R` denotes a double mutant. The first character of a mutation string denotes the wild type amino acid, the second character the chain, followed by the position in the chain, and lastly the amino acid to mutate to. For example, `YH103H` denotes a mutation from Y to H at position 103 of chain H.
## Example
An example is provided for TCR mimics in `example/tcrm_case_study`. Predictions can be generated using the following command:

```
python run_stabddg.py --run_name test --csv_path example/tcrm_case_study/tcrm_case_study_spr.csv --pdb_dir example/tcrm_case_study/spr_pdbs/ --pdb_cache_path cache/tcrm_case_study_spr --fix_perm --fix_noise --batch_size 10000 --checkpoint "model_ckpts/stabddg.pt" 
```

# Repository overview

* `run_stabddg.py` using StaB-ddG for binding ddG prediction
* `megascale_finetune.py` fine-tuning ProteinMPNN on folding stability data
* `skempi_finetune.py` fine-tuning StaB-ddG on binding data from SKEMPI
* `skempi_eval.py` evaluating StaB-ddG on SKEMPI test split
* `protddg/`
    * `model.py` StaB-ddG model
    * `ppi_dataset.py` dataset classes for binding data
    * `mpnn_utils.py` code copied over from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
* `model_ckpts` we provide the following checkpoints
    * `v_48_020.pt` soluble ProteinMPNN weights from https://github.com/dauparas/ProteinMPNN
    * `megascale_finetuned.pt` model weights after megascale fine-tuning
    * `stabddg.pt` final model fine-tuned on both stability and binding data. This is the checkpoint that should be used for inference.
* `data/`
    * `rocklin/mega_splits.pkl` megascale dataset splits from https://github.com/Kuhlman-Lab/ThermoMPNN
    * `SKEMPI/`
        * `skempi_v2.csv` unfiltered SKEMPI csv file from https://life.bsc.es/pid/skempi2/database/index
        * `filtered_skempi.csv` filtered SKEMPI csv file
        * `train_pdb.pkl, train_clusters.pkl` training split
        * `test_pdb.pkl, test_clusters.pkl` test split
        * `quality_filtering.ipynb` filtering script
        * `skempi_splits.ipynb` script creating train/test cluster-based splits

# Model training

## Fine-tuning on Megascale
The Megascale stability dataset can be downloaded from https://zenodo.org/records/7992926. Specifically, the files needed are `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` and `AlphaFold_model_PDBs.zip`. 

```
python megascale_finetune.py --run_name RUN_NAME --rocklin_csv DIR/Tsuboyama2023_Dataset2_Dataset3_20230416.csv --pdb_dir DIR/AlphaFold_model_PDBs --fix_noise --fix_perm --num_epochs 70 --wandb --checkpoint model_ckpts/v_48_020.pt
```

## Fine-tuning on SKEMPI
A filtered version of the SKEMPI csv is provided in `data/SKEMPI/filtered_skempi.csv`. The PDB files can be downloaded from https://life.bsc.es/pid/skempi2/database/index. After the download, the files should be split and renamed to match the required input format described above. 
```
python skempi_finetune.py --train_split_path data/SKEMPI/train_pdb.pkl --epochs 200 --wandb --lr 1e-6 --run_name RUN_NAME --fix_noise --single_batch_train --fix_perm --checkpoint model_ckpts/megascale_finetuned.pt --skempi_pdb_dir data/SKEMPI_v2/PDBs --skempi_path data/SKEMPI/filtered_skempi.csv
```

## Running evaluation on SKEMPI
To reproduce results from the paper, run the following command.
```
python skempi_eval.py --run_name EVAL --fix_perm --fix_noise --batch_size 10000 --trials 1 --ensemble 20 --checkpoint "model_ckpts/stabddg.pt" --skempi_path data/SKEMPI/filtered_skempi.csv --skempi_pdb_dir data/SKEMPI_v2/PDBs --skempi_pdb_cache_path cache/skempi_full_mask_pdb_dict.pkl --skempi_split_path "data/SKEMPI/test_pdb.pkl"
```