#!/bin/bash
source /home/xiaohui/.bashrc

. /home/xiaohui/torch_icc/install/bin/torch-activate

KMP_AFFINITY=scatter,granularity=fine,0,1  OMP_NUM_TiHREADS=44 th benchmark_cudnn.lua 
