#!/usr/bin/env bash


filelist=`ls ../csp-benchmark`
for file in $filelist
do
    echo $file
    python cpu_main.py "../csp-benchmark/"$file 50000
done