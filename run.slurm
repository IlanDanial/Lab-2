#!/bin/bash -l
#SBATCH --job-name=gpu_lab1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --account=eel4930
#SBATCH --qos=eel4930

# 1. Load modules on HiPerGator
module purge
module load gcc cuda

echo "Job ID:     $SLURM_JOB_ID"
echo "Host:       $(hostname)"
echo "Date:       $(date)"

# Environment for the report
echo -e "\n=== Environment ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvcc --version | tail -2
gcc --version | head -1

# 2. Build lab1
echo -e "\n=== Build ==="
make clean
make

# 3. Compute Sanitizer on the irregular case
echo -e "\n=== Compute Sanitizer ==="
compute-sanitizer --tool memcheck ./lab2 31 17 50

# 4. Tests
echo -e "\n=== Tests ==="
echo -e "\n=== Test 1 ==="
./lab2 1 1 1
echo -e "\n=== Test 2 ==="
./lab2 3 5 2
echo -e "\n=== Test 3 ==="
./lab2 16 16 16
echo -e "\n=== Test 4 ==="
./lab2 31 17 50
echo -e "\n=== Test 5 ==="
./lab2 65 33 129


echo -e "
=== Timing 1024x1024x1024 ==="
./lab2 1024 1024 1024
