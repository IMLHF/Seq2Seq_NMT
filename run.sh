echo $1
mkdir exp/$1
python3 -m $1.nmt_main | tee exp/$1.log && echo ' '
