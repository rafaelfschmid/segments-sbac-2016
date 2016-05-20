in=$1
out=$2

./scripts/exec.sh mergeseg.exe $in > $out/mergeseg.time
./scripts/exec.sh radixseg.exe $in > $out/radixseg.time
./scripts/exec.sh fixcub.exe $in > $out/fixcub.time
./scripts/exec.sh fixsort/bitonic/bitonicsort.exe $in > $out/bitonicsort.time
./scripts/exec.sh fixsort/mergesort/mergesort.exe $in > $out/mergesort.time
./scripts/exec.sh fixsort/oddevensort/oddevensort.exe $in > $out/oddevensort.time
./scripts/exec.sh fixsort/quicksort/quicksort.exe $in > $out/quicksort.time
./scripts/exec.sh fixpreprocess.exe $in > $out/fixpreprocess.time


