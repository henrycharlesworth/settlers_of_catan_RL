#!/bin/bash

#If memory usage exceeds 95% of available memory (which it will eventually because of a memory leak...) the script will save and terminate. We can then run the script again and load in the saved policies to continue training.

python RL/robust_train.py

while true; do
python RL/robust_train.py --load-from-checkpoint --load-file-path current.pt
done

