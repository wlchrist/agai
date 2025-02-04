#!/bin/bash
#SBATCH --job-name=tmax_training        # Name of your job
#SBATCH --output=training_output.log    # Log file for standard output
#SBATCH --error=training_error.log      # Log file for errors
#SBATCH --time=02:00:00                 # Set maximum run time (hh:mm:ss)
#SBATCH --account=hackathon             # Hackathon account
#SBATCH --reservation=hackathon_gpu     # Reservation for hackathon
#SBATCH --partition=kamiak     # ✅ Set your desired partition
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --mem=16G                        # Request 16GB of RAM
#SBATCH --cpus-per-task=4                # Request 4 CPU cores

# Load necessary modules

# Activate virtual environment (optional, if you created one)
source ../python_files/team5env/bin/activate

# Run the Python script
python ../python_files/swemodel.py
