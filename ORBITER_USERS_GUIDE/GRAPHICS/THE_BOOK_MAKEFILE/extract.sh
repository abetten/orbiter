#!/bin/bash

$1orbiter.out -v 3 \
	-extract_from_file_with_tail $4 $6 $3 $5/$6.txt
echo "$2a2tex.out -text_width 60 <$5/$6.txt >$5/$6.tex"
$2a2tex.out -text_width 60 <$5/$6.txt >$5/$6.tex
