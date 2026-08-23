

s:
	sort ls_before.txt >ls_before_sorted.txt
	sort ls_after.txt >ls_after_sorted.txt

tex1:
	~/bin/a2tex.out -numbers <symmetrica.txt >symmetrica.tex
	~/bin/a2tex.out -numbers <ls_before_sorted.txt >ls_before_sorted.tex
	~/bin/a2tex.out -numbers <ls_after_sorted.txt >ls_after_sorted.tex
	~/bin/a2tex.out -numbers <directory_structure_sorted.txt \
		>directory_structure_sorted.tex
	~/bin/a2tex.out  <log >executables.tex
	~/bin/a2tex.out -numbers <MAKEFILE_EASY/makefile >MAKEFILE_EASY/makefile.tex

