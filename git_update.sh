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

# find all .a files and delete them before uploading the code to github
echo $LINE
echo -e "${GREEN}Deleting the following .a files${NONE}"
echo $LINE
for f in $(find . -type f -name "*.a"); do
    rm -rfv $f
done
echo $LINE

# find all .o files and delete them before uploading the code to github
echo $LINE
echo -e "${GREEN}Deleting the following .o files${NONE}"
echo $LINE
for f in $(find . -type f -name "*.o"); do
    rm -rfv $f
done
echo $LINE

# find all '~' files and delete them before uploading the code to github
echo $LINE
echo -e "${GREEN}Deleting the following '~' files${NONE}"
echo $LINE
for f in $(find . -type f -name "*.o"); do
    rm -rfv $f
done
echo $LINE

# find all '.out' files and delete them before uploading the code to github
echo $LINE
echo -e "${GREEN}Deleting the following '.out' files${NONE}"
echo $LINE
for f in $(find . -type f -name "*.o"); do
    rm -rfv $f
done
echo $LINE

# update the code on github
echo $LINE
echo -e "${GREEN}Preparing to upload code to github...${NONE}"
echo $LINE
git add .
git commit -m "version of $(date)"
git push origin master
echo $LINE

#echo $LINE
