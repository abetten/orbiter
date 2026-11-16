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
for f in $(find . -name '*.cpp' -o \
		-name '*.CPP' -o \
		-name '*.c' -o \
		-name '*.C' -o \
		-name '*.h' -o \
		-name '*.H' -o \
		-name '*.hpp' -o \
		-name '*.HPP' -o \
		-path "./src/lib/layer1_foundations/Eigen_interface/Eigen" -prune -false); do
	tmp=$(wc -l < $f); 
	echo "$tmp => $f";
	COUNT=$[COUNT+tmp]; 
done
echo $LINE
echo -e "${GREEN}Total number of lines of code is $COUNT${NONE}"
echo $LINE
