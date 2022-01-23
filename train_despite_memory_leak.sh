#!/bin/bash

#If memory usage exceeds 95% of available memory (which it will eventually because of a memory leak...) the script will save and terminate. We can then run the script again and load in the saved policies to continue training.

#python RL/robust_train.py --load-from-checkpoint --load-file-path current_best.pt
python RL/continue_training.py

while true; do
pkill -9 python
python RL/continue_training.py --load-from-checkpoint --load-file-path current.pt
done

