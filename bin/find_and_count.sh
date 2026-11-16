#!/bin/bash

# using grep, searches all *.cpp files for the phrase given as the command line argument

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
for f in $(find . -name '*.cpp' ); do
	tmp=$(grep $1 < $f | wc -l ); 
	echo "$tmp => $f";
	COUNT=$[COUNT+tmp]; 
done
echo $LINE
echo -e "${GREEN}Total number of times that we found the phrase is $COUNT${NONE}"
echo $LINE
echo "$COUNT,$1" >>stats.csv

