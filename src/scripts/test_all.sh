in=$1 # dir input files
result=$2 # correct result files
error=$3 # dir errors

./scripts/test.sh fixcub.exe $in $result $error/fixcub
./scripts/test.sh fixmergemgpu.exe $in $result $error/fixmergemgpu
./scripts/test.sh fixthrust.exe $in $result $error/fixthrust
./scripts/test.sh mergeseg.exe $in $result $error/mergeseg
./scripts/test.sh radixseg.exe $in $result $error/radixseg
./scripts/test.sh fixsort/bitonic/bitonicsort.exe $in $result $error/fixbitonic
./scripts/test.sh fixsort/bitonicseg/bitonicseg.exe $in $result $error/bitonicseg
./scripts/test.sh fixsort/mergesort/mergesort.exe $in $result $error/fixmerge
./scripts/test.sh fixsort/oddevensort/oddevensort.exe $in $result $error/fixoddeven
./scripts/test.sh fixsort/quicksort/quicksort.exe $in $result $error/fixquick

