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

LINE="----------------------------------------------------------------------------------"

COUNT=0; 
echo $LINE
echo "[# LINES] => [FILE]"
echo $LINE
for f in $(find . -type f -name '*.cpp' -name '*.CPP' -name '*.c' -name '*.C' -name '*.h' -name '*.H' -name '*.hpp' -name '*.HPP'); do 
	tmp=$(wc -l < $f); 
	echo "$tmp => $f";
	COUNT=$[COUNT+tmp]; 
done
echo $LINE
echo -e "${GREEN}Total number of lines of code is $COUNT${NONE}"
echo $LINE
