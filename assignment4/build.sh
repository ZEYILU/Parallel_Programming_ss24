#!/bin/bash

gcc -std=c99 Quicksort_seq.c -o quicksort_sequential

gcc -std=c99 Quicksort_par.c -o quicksort_parallel
