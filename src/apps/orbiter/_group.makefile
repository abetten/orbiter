
:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define gens -vector -file _gens.csv -end \
		-define G -permutation_group \
		-bsgs  "" 0 1 "" 0 gens -end \
