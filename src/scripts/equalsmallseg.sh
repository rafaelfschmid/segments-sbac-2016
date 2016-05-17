#!/bin/bash
dir=$1
c=1048576
while [ $c -le 134217728 ]
do
	#echo $c 
	d=1
	while [ $d -le 1048576 ] 
	do
		((x=$c/$d))
		./equal.exe $d $x > $dir/$d"_"$c.in
		((d=$d*2))
		sleep 0.5
	done
	((c=$c*2))
done
