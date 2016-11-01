#! /bin/bash
#BSUB -W 6000
#BSUB -n 4
#BSUB -o ./out.%J
#BSUB -e ./err.%J
rm err/*
rm out/*

bsub -W 100 -n 12 -q long -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J /share/rkrish11/miniconda/bin/python2.7 par_exec.py
