#!/bin/bash
dir=$1
c=32768
while [ $c -le 67108864 ]
do
	#echo $c 
	d=1
	while [ $d -le  32768 ] 
	do
		((x=$c/$d))
		./equal.exe $d $x > $dir/$d"_"$c.in
		((d=$d*2))
		sleep 0.5
	done
	((c=$c*2))
done
