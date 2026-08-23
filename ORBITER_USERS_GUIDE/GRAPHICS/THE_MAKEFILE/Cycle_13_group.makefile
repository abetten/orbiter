
Cycle_13:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define gens -vector -file Cycle_13_gens.csv -end \
		-define G -permutation_group \
		-bsgs Cycle_13 "Cycle\_13" 13 26 "0,5" 2 gens -end \
