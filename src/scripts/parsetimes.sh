dir=$1

./parser.exe $dir/fixbitonic.time $dir/00fixbitonic.time
./parser.exe $dir/fixpreprocess.time $dir/00fixpreprocess.time
./parser.exe $dir/fixcub.time $dir/00fixcub.time
./parser.exe $dir/fixmerge.time $dir/00fixmerge.time
./parser.exe $dir/fixoddeven.time $dir/00fixoddeven.time
./parser.exe $dir/fixquick.time $dir/00fixquick.time
./parser.exe $dir/mergeseg.time $dir/00mergeseg.time
./parser.exe $dir/radixseg.time $dir/00radixseg.time

