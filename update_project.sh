root=~/project/StreamDynamic
rsync -avzz $root lab_pc:/home/xiang.huang/project/ --exclude-from=$root/exclude.txt
