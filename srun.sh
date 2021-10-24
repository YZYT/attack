# srun -N 1 -n 5 -c 1 --pty -p p-RTX2080  --nodelist=pgpu04  --gres=gpu:1 bash 
srun -N 1 -n 5 -c 1 --pty -p p-RTX2080   --gres=gpu:1 bash 
srun -N 1 -n 5 -c 1 --pty -p p-V100  --gres=gpu:1 bash