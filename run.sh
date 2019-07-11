echo $1
if [ -d "exp/$1" ]; then
date=`date`
echo 'dir exist, mv to "'$1_$date'"'
mv exp/$1 "exp/$1_$date"
fi
mkdir exp/$1
python3 -m $1.nmt_main 2>&1 | tee exp/$1/$1.log && echo ' '
