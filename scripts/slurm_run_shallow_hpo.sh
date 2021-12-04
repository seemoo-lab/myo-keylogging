# Code for the "Inferring Keystrokes from Myo Armband Sensors" project
#
# Copyright (C) 2019-2021  Matthias Gazzari, Annemarie Mattmann
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# project and job names
#SBATCH --account=project42
#SBATCH --job-name=run

# sdtout and stderr output files
#SBATCH --output=slurm_job_%A_task_%a.out

# send mail (BEGIN, END or ALL)
##SBATCH --mail-type=ALL

# node configuration (16 cores, 24 GB total memory, 2 Nvidia K20Xm GPUs)
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1625M
#SBATCH --constraint=nvd2

# maximum runtime in minutes (one may use the hh:mm:ss format, 1430 minutes = 23:50:00)
# send SIGUSR1 300 s (5 minutes) before hitting the walltime
#SBATCH --time=1430
#SBATCH --signal=SIGUSR1@300

# task array --> $SLURM_ARRAY_TASK_ID
#SBATCH --array=1-16

echo ---------- MAIN SCRIPT ----------
echo hostname=$(hostname)
echo nproc=$(nproc)
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo ---------------------------------

# purge loaded modules and load the relevant ones
module purge
module load intel/2019.4
module load python/3.6.8
module load cuda/9.2
echo ---------------------------------

# alternative way: use ntasks=2 (SLURM_PROCID) instead of steps (SLURM_STEP_ID)
#SRUN_PARAMS="-l --exclusive --cpus-per-task=8 --mem-per-cpu=1625M --ntasks=2 --nodes=1"
#srun ${SRUN_PARAMS} step.sh &

# run configurations dependent on SLURM_ARRAY_TASK_ID
DEBUG='echo hostname=$(hostname) nproc=$(nproc) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python3 -c "import torch; torch.randn(1000,1000,100).cuda(); import time; time.sleep(5)"'
SRUN_PARAMS='--exclusive --cpus-per-task=8 --mem-per-cpu=1625M --ntasks=1 --nodes=1'
ML_PARAMS="--func shallow_hpo --n_iter 1000 --epochs 500 --patience 20 --target press"
BINARY_PARAMS="--encoding binary --step 10 --resample_once "
MULTICLASS_PARAMS="--encoding multiclass --step 1"
declare -A CMD_ARRAY
CMD_ARRAY=(
	['0']=${DEBUG}
	['1']="python3  -m ml crnn     ${ML_PARAMS} ${BINARY_PARAMS} --seg_width 35"
	['2']="python3  -m ml wavenet  ${ML_PARAMS} ${BINARY_PARAMS} --seg_width 35"
	['3']="python3  -m ml resnet   ${ML_PARAMS} ${BINARY_PARAMS} --seg_width 35"
	['4']="python3  -m ml resnet11 ${ML_PARAMS} ${BINARY_PARAMS} --seg_width 35"
	['5']="python3  -m ml crnn     ${ML_PARAMS} ${MULTICLASS_PARAMS} --seg_width 35"
	['6']="python3  -m ml wavenet  ${ML_PARAMS} ${MULTICLASS_PARAMS} --seg_width 35"
	['7']="python3  -m ml resnet   ${ML_PARAMS} ${MULTICLASS_PARAMS} --seg_width 35"
	['8']="python3  -m ml resnet11 ${ML_PARAMS} ${MULTICLASS_PARAMS} --seg_width 35"
	['9']="python3  -m ml crnn     ${ML_PARAMS} ${BINARY_PARAMS} --seg_width 45"
	['10']="python3 -m ml wavenet  ${ML_PARAMS} ${BINARY_PARAMS} --seg_width 45"
	['11']="python3 -m ml resnet   ${ML_PARAMS} ${BINARY_PARAMS} --seg_width 45"
	['12']="python3 -m ml resnet11 ${ML_PARAMS} ${BINARY_PARAMS} --seg_width 45"
	['13']="python3 -m ml crnn     ${ML_PARAMS} ${MULTICLASS_PARAMS} --seg_width 45"
	['14']="python3 -m ml wavenet  ${ML_PARAMS} ${MULTICLASS_PARAMS} --seg_width 45"
	['15']="python3 -m ml resnet   ${ML_PARAMS} ${MULTICLASS_PARAMS} --seg_width 45"
	['16']="python3 -m ml resnet11 ${ML_PARAMS} ${MULTICLASS_PARAMS} --seg_width 45"
)

# execute on both GPUs in parallel
for i in {0..1}
do
	CMD="${CMD_ARRAY[${SLURM_ARRAY_TASK_ID}]} --uid ${SLURM_ARRAY_TASK_ID}_${i}"
	echo ${CMD}
	CUDA_VISIBLE_DEVICES=${i} srun ${SRUN_PARAMS} bash -c "${CMD}" 2>&1 | sed -u 's/^/GPU '${i}': /' &
done

# verify the use of both GPUs when debugging
if [ ${SLURM_ARRAY_TASK_ID} -eq '0' ]
then
	sleep 8
	ps -aux | grep python3
	nvidia-smi
fi

# wait for both processes to finish
wait
echo "Execution done"
