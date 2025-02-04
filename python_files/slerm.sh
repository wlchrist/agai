#!/bin/bash
#SBATCH --job-name=swe_training        # Name of your job
#SBATCH --output=training_output.log    # Log file for standard output
#SBATCH --error=training_error.log      # Log file for errors
#SBATCH --time=02:00:00                 # Set maximum run time (hh:mm:ss)
#SBATCH --account=hackathon             # Hackathon account
#SBATCH --reservation=hackathon_gpu     # Reservation for hackathon
#SBATCH --partition=kamiak     # ✅ Set your desired partition
#SBATCH --gres=gpu:2                    # Request 1 GPU
#SBATCH --mem=32G                        # Request 16GB of RAM
#SBATCH --cpus-per-task=8                # Request 4 CPU cores

# Load necessary modules

# Activate virtual environment (optional, if you created one)
source team5env/bin/activate

# Run the Python script
python swemodel.py
