#!/bin/bash
# Array job: one VASP single point per phonopy displacement directory.
# Submit from the phonopy directory after `phunfold setup-disp`:
#   N=$(ls -d disp-* | wc -l); sbatch --array=0-$N run_displacements.slurm
# Index 0 runs perfect/ (undisplaced supercell, needed for --fz).
#SBATCH --job-name=dw-phon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%A_%a.out

module load vasp            # adapt to your cluster
VASP=${VASP:-vasp_std}

DIRS=(perfect $(ls -d disp-* | sort -V))
D=${DIRS[$SLURM_ARRAY_TASK_ID]}
cd "$D" || exit 1
if grep -q "General timing" OUTCAR 2>/dev/null; then
    echo "$D already finished"; exit 0
fi
srun $VASP > vasp.out
