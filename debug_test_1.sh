#$ -cwd
#$ -N debug_test_1
#$ -o debug_test_1.txt
#$ -j y
#$ -S /bin/bash
#$ -pe mpi 4
#$ -l h_rt=240:00:00
#$ -l h_vmem=512g
#$ -l h="compute-0-5|compute-0-6"

for rep in {0..4}; do mpirun -np 4 python biPASS_sync.py "debug_test_1" "debug_test_1_config" "$rep"; done