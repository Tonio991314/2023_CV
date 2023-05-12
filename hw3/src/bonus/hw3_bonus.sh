#!/usr/bin/env bash
STARTTIME=$(date +%s)
time python3 main.py -t orb
ENDTIME=$(date +%s)
echo "Time elpased $(($ENDTIME - $STARTTIME)) seconds"