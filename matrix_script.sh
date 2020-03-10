#!/usr/bin/env bash


filelist=`ls ../csp-benchmark`
for file in $filelist
do
    echo $file
#    python matrix_main.py cpu "../csp-benchmark/"$file 50000
    python matrix_main.py cuda "../csp-benchmark/"$file 50000
done