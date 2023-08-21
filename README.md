This repository contains code for generating datasets of parcel model runs using [PySDM](https://github.com/open-atmos/PySDM), and using them to train machine learning emulators of aerosol activation, using approaches based on the work of [Silva et al.](https://doi.org/10.5194/gmd-14-3067-2021)

To generate a dataset on Caltech's HPC cluster, run the following:
```bash
sbatch generate_parcel_data.sh [dataset_name] [num_simulations] [num_modes]
```
And then to see the output logs displayed,
```bash
tail -f slurm.out
```
For instance,
```bash
sbatch generate_parcel_data.sh my_2modal_dataset 20000 2
```
will perform 20000 runs of simulations with 2-modal aerosol populations, creating the files `datasets/my_2modal_dataset_train.csv`, `datasets/my_2modal_dataset_test.csv`, and `datasets/my_2modal_dataset_fail.csv` - which are the generated train dataset, test dataset, and initial conditions of the runs that failed (due to a condensation solver failure or timeout).

