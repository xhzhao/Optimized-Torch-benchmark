#!/bin/bash
source ~/.bashrc

#. /home/xiaohui/Optimized-Torch/install/bin/torch-activate

which th
KMP_AFFINITY=scatter,granularity=fine  OMP_NUM_THREADS=68 th  benchmark_mkldnn.lua
