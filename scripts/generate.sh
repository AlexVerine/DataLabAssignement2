#!/bin/bash
name="generate_images"
outdir="outputs"
    echo "Launching test for $name"
    
    sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=256G
#SBATCH --account=m25146        # Your project account
#SBATCH --job-name=generation      # Job name
#SBATCH --output=${outdir}/%x_%j.out  # Standard output and error log
#SBATCH --error=${outdir}/%x_%j.err  # Error log

source venv/bin/activate
# Run your training script
python generate.py
EOT