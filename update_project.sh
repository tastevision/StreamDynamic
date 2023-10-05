root=~/project/StreamDynamic
rsync -avzz $root yb-s11:/home/xiang.huang/project/ --exclude-from=$root/exclude.txt
