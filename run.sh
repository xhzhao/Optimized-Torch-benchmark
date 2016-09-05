#!/bin/bash
source /home/xiaohui/.bashrc


. /home/xiaohui/test/Optimized-Torch/install/bin/torch-activate
KMP_AFFINITY=scatter,granularity=fine,0,1  OMP_NUM_THREADS=44 th benchmark_mkldnn.lua
