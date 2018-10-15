#!/bin/bash
python3 multiple_rnn.py -d 0 -l 0.00005 -a AUDUSD -n 2 -g 0 &
python3 multiple_rnn.py -d 0 -l 0.00005 -a AUDUSD -n 5 -g 0 &
python3 multiple_rnn.py -d 0 -l 0.00005 -a AUDUSD -n 8 -g 0 &
python3 multiple_rnn.py -d 0 -l 0.00005 -a AUDUSD -n 12 -g 1 
# python3 multiple_rnn.py -d 0 -l 0.00005 -a AUDUSD -n 15 -g 1 &
# python3 multiple_rnn.py -t 5 &
# python3 multiple_rnn.py -t 10 &
# python3 multiple_rnn.py -t 7 &
# wait
# python3 multiple_rnn.py -t 3
