#!/bin/bash

echo "#################################################"
echo "Running point training on model 2: batch_size = 32"
echo "#################################################"
python3 train_model.py --type 'point'

echo "#################################################"
echo "Running mesh training on model 2: batch_size = 32 "
echo "#################################################"
python3 train_model.py --type 'mesh'


echo "#################################################"
echo "Running Vox training on Implicit: batch_size = 4"
echo "#################################################"
python3 train_model.py --batch_size 4 --type 'vox'

