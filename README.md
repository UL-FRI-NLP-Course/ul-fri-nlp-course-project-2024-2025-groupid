# Natural Language Processing Course  
**Automatic Generation of Slovenian Traffic News for RTV Slovenija**

## Running inference on HPC

1. Change to the project directory:
   ```bash
   cd /d/hpc/projects/onj_fri/groupid

2. Prepare input data by modifying or adding entries in the
   `./src/inputs.json` file

3. Submit the inference job:
    ```bash
   sbatch ./jobs/inference.sh

5. Outputs will be saved (relative to the directoy you ran from) in the file: `./logs/inference-<job_id>.out` 

## Finetuning on HPC

1. Change to the project directory:
   ```bash
   cd /d/hpc/projects/onj_fri/groupid

3. Submit the finetuning job:
    ```bash
   sbatch ./jobs/finetune.sh

5. (LORA) Adapter will be saved in the folder: `./models/` 
