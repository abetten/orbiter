
complete_1:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define gens -vector -file complete_1_gens.csv -end \
		-define G -permutation_group \
		-bsgs complete_1 "K_{1}" 1 1 "" 0 gens -end \
