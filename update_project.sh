root=~/project/StreamDynamic
rsync -avzz $root lab_pc:/home/taster/project/ --exclude-from=$root/exclude.txt
