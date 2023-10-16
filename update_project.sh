root=~/project/StreamDynamic
rsync -avzz $root remote-s1:/home/xiang.huang/project/ --exclude-from=$root/exclude.txt
