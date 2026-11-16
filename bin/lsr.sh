#!/bin/bash

# Terminal ANSI Color codes
NONE='\033[00m'
RED='\033[00;31m'
GREEN='\033[00;32m'
YELLOW='\033[00;33m'
PURPLE='\033[00;35m'
CYAN='\033[00;36m'
WHITE='\033[00;37m'
BOLD='\033[1m'
UNDERLINE='\033[4m'

LINE="--------------------------------------------------------------------------"

COUNT=0; 
echo $LINE
echo "[# LINES] => [FILE]"
echo $LINE
for f in $(find . -type f  ! -name '*~' ! -name '.DS_Store'); do 
	echo "$f";
	COUNT=$[COUNT+1]; 
done
echo $LINE
echo -e "${GREEN}Total number of files is $COUNT${NONE}"
echo $LINE
