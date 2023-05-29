#!/bin/bash

# Make Logs dir if it doesn't exist
mkdir -p Logs

# Normal training
./train.py IP 2>\dev\null | tee Logs/IP_train_log.txt
./train.py KSC 2>\dev\null | tee Logs/KSC_train_log.txt
./train.py PU 2>\dev\null | tee Logs/PU_train_log.txt
./train.py SV 2>\dev\null | tee Logs/SV_train_log.txt

# Test
./test.py 2>\dev\null

# Map test
./test_map.py IP 2>\dev\null
./test_map.py KSC 2>\dev\null
./test_map.py PU 2>\dev\null
./test_map.py SV 2>\dev\null

