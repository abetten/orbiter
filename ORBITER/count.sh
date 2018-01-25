#!/bin/bash
COUNT=0; for f in $(find . -type f ! -name '*.o' ! -name '*~' ! -name '*.out' ! -name '*.a' ! -name '.DS_Store'); do tmp=$(wc -l < $f); COUNT=$[COUNT+tmp]; done
echo "The number of lines of code is $COUNT"

