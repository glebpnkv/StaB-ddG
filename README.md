# binding-ddG

# Setup 
Use environment in ProteinMPNN. 

TODO: list additional packages to install.

# Evaluate binding ddG
## Formatting input
TODO
## Example
TODO
## Running evaluation on SKEMPI
```
CUDA_VISIBLE_DEVICES=2 python skempi_eval_updated.py --run_name refactor_test --fix_perm --fix_noise --batch_size 10000 --trials 1 --ensemble 20 --checkpoint "new_ckpts/nolinear1e-6_nofixmonomer_200.pt" --skempi_path SKEMPI/filtered_skempi.csv --skempi_pdb_dir /home/exx/arthur/data/SKEMPI_v2/PDBs --skempi_pdb_cache_path cache/skempi_full_mask_pdb_dict.pkl --skempi_split_path "SKEMPI/test_pdb.pkl"
```
# Fine-tuning on SKEMPI
```
CUDA_VISIBLE_DEVICES=2 python skempi_finetune_updated.py --train_split_path SKEMPI/train_pdb.pkl --epochs 200 --wandb --lr 1e-6 --run_name RUN_NAME --seed 1 --val_ensembles 20 --fix_noise --val_trials 1 --batch_size 15000 --single_batch_train --fix_perm --checkpoint new_ckpts/soluble_train_s0_epoch69.pt
```
# Fine-tuning on Megascale

```
CUDA_VISIBLE_DEVICES=3 python megascale_finetune.py --run_name RUN_NAME --seed 0 --lr 1e-6 --fix_noise --dropout 0.0 --fix_perm --num_epochs 70 --wandb --val_freq 30 --batch_size 10000 --noise_level 0.2
```