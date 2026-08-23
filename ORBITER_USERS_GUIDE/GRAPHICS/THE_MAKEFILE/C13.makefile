
C13:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define gens -vector -file C13_gens.csv -end \
		-define G -permutation_group \
		-bsgs C13 "C_{13}" 13 13 "0" 1 gens -end \
