#!/bin/bash

KMP_AFFINITY=scatter,granularity=fine,0,1  OMP_NUM_THREADS=44 th benchmark_mkldnn.lua
