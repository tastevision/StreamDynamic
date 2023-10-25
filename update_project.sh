root=~/project/StreamDynamic
rsync -avzz $root s1:/home/xiang.huang/project/ --exclude-from=$root/exclude.txt
