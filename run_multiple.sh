#!/bin/bash
python3 multiple_rnn.py -a SPY -e 500 -l 0.00001 -n 5  -g 0 -p 1 -r 500000 &
python3 multiple_rnn.py -a SPY -e 500 -l 0.00001 -n 8  -g 1 -p 0 &
python3 multiple_rnn.py -a SPY -e 500 -l 0.00001 -n 10 -g 0 -p 0 &
python3 multiple_rnn.py -a SPY -e 500 -l 0.00001 -n 12 -g 1 -p 0 &
python3 multiple_rnn.py -a SPY -e 500 -l 0.00001 -n 15 -g 1 -p 0 &
wait
python3 multiple_rnn.py -a SPY -e 500 -l 0.00001 -n 18  -g 0 -p 0 -r 500000 &
python3 multiple_rnn.py -a SPY -e 500 -l 0.00001 -n 21  -g 1 -p 0 &
python3 multiple_rnn.py -a SPY -e 500 -l 0.00001 -n 24 -g 0 -p 0 &
python3 multiple_rnn.py -a SPY -e 500 -l 0.00001 -n 27 -g 1 -p 0 &
python3 multiple_rnn.py -a SPY -e 500 -l 0.00001 -n 30 -g 1 -p 0 &
