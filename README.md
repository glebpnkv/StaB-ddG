# binding-ddG

# Setup 
```
conda create --name protddg
conda activate protddg
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tqdm wandb scipy pandas
```
# Predicting binding ddG
StaB-ddG requires as input PDB files and a csv file with mutations of interest.

## PDB files
For a single ddG prediction, StaB-ddG requires 3 PDB files --- one for each binder and one for the entire complex. Specifically, the naming of the PDB files should follow:
* Complex: `NAME_binder1chains_binder2chains.pdb`
* Binder 1: `NAME_binder1chains.pdb`
* Binder 2: `NAME_binder2chains.pdb`. 

The chains specify what interface the ddG is computed over for a multichain complex. For example, to predict the binding ddG of the complex 7STF for the interface between chains HL and ABC, we should provide the following three PDB files: `7STF_HL_ABC.pdb`, `7STF_HL.pdb`, and `7STF_ABC.pdb`.

## Mutation csv file
The mutation csv file should contain two columns, `#Pdb` and `Mutation(s)_cleaned`. The `#Pdb` column should contain the name of the complex PDB file without the `.pdb` suffix (e.g. `7STF_HL_ABC`). 

The `Mutation(s)_cleaned` contains the mutations of interest. Each entry can have multiple mutations separated by commas. For example, `YH103H,QC7R` denotes a double mutant. The first character of a mutation string denotes the wild type amino acid, the second character the chain, followed by the position in the chain, and lastly the amino acid to mutate to. For example, `YH103H` denotes a mutation from Y to H at position 103 of chain H.
## Example
TODO

# Model training

## Fine-tuning on Megascale

```
CUDA_VISIBLE_DEVICES=3 python megascale_finetune.py --run_name RUN_NAME --seed 0 --lr 1e-6 --fix_noise --dropout 0.0 --fix_perm --num_epochs 70 --wandb --val_freq 30 --batch_size 10000 --noise_level 0.2
```

## Fine-tuning on SKEMPI
```
python skempi_finetune.py --train_split_path data/SKEMPI/train_pdb.pkl --epochs 200 --wandb --lr 1e-6 --run_name RUN_NAME --seed 1 --val_ensembles 20 --fix_noise --val_trials 1 --batch_size 15000 --single_batch_train --fix_perm --checkpoint model_ckpts/soluble_train_s0_epoch69.pt
```

## Running evaluation on SKEMPI
```
python skempi_eval.py --run_name refactor_test --fix_perm --fix_noise --batch_size 10000 --trials 1 --ensemble 20 --checkpoint "model_ckpts/nolinear1e-6_nofixmonomer_200.pt" --skempi_path data/SKEMPI/filtered_skempi.csv --skempi_pdb_dir /home/exx/arthur/data/SKEMPI_v2/PDBs --skempi_pdb_cache_path cache/skempi_full_mask_pdb_dict.pkl --skempi_split_path "data/SKEMPI/test_pdb.pkl"
```