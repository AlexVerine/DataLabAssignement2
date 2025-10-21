# DataLabAssignement2

## Using GPUs with MesoNet

### 1. Access MesoNet
To use the provided GPUs, you must first:
- Create a MesoNet account and request access to the project.
- Once accepted, add your SSH key to MesoNet for secure access. Follow the instructions [here](https://www.mesonet.fr/documentation/user-documentation/code_form/juliet/connexion).

### 2. Set Up Your Environment
On Juliet (MesoNet's cluster), you need to:

1. Create a virtual environment for Python:
   ```bash
   python -m venv venv
   ```

2. Activate the environment:
   ```bash
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Run Training or Generation
- Use the provided shell scripts (`scripts/train.sh` or `scripts/generate.sh`) to launch jobs.
- These scripts will automatically activate the virtual environment (`venv`).
- If you don't use Juliet cluster, please provide a data_path argument for the training scripts. Otherwise, everything should be automated.

### 4. Hardware Compatibility
The code is compatible with:
- **NVIDIA GPUs (CUDA)**
- **Apple Silicon (MPS, for M1/M2/M3/M4 chips)**
- **CPU-only mode** (not recommended for performance)

We recommend using CUDA (for NVIDIA GPUs) or MPS (for Apple Silicon).

### 5. Slurm Resources
- If you need help with Slurm (MesoNet's job scheduler), there will be a Slurm lecture on 29/10/2025 or refer to the official documentation: [Slurm Documentation](https://slurm.schedmd.com/documentation.html).

### 6. Example Commands
To submit a job to MesoNet:
```bash
# Make the script executable
chmod +x generate.sh

# Submit the job to Slurm
scripts/train.sh
```

## generate.py
Use the file *generate.py* to generate 10000 samples of MNIST in the folder samples. 
Example:
  > python3 generate.py --bacth_size 64

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt


## Checkpoints
Push the minimal amount of models in the folder *checkpoints*.

