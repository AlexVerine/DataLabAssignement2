#!/bin/bash
name="test_train_datalab"
outdir="outputs"
n_gpu=1
    echo "Launching test for $name"
    
    sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:${n_gpu}
#SBATCH --time=00:20:00
#SBATCH --mem=256G
#SBATCH --account=m25146        # Your project account 
#SBATCH --job-name=gan_train      # Job name
#SBATCH --output=${outdir}/%x_%j.out  # Standard output and error log
#SBATCH --error=${outdir}/%x_%j.err  # Error log

source venv/bin/activate
# Run your training script
python train.py --epochs 100 --lr 0.0002 --batch_size 64 --gpus ${n_gpu}
EOT

