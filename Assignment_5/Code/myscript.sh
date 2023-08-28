#!/bin/bash


for file in /home/tenet/Desktop/CS22Z121/Assignment_5/obs_seq_test/*.seq
do

	 sed -i 's/[][,]//g' $file

done
