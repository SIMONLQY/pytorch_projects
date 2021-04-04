#! /bin/bash

python Main.py --dataset RV --trans rotat > danRVrotat.txt
python Main.py --dataset FM --trans rotat > danFMrotat.txt
python Main.py --dataset RV --trans rotat2 > danRVtotat2.txt
python Main.py --dataset FM --trans rotat2 > danFMrotat2.txt