
Cycle_3:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define gens -vector -file Cycle_3_gens.csv -end \
		-define G -permutation_group \
		-bsgs Cycle_3 "Cycle\_3" 3 6 "0,1" 2 gens -end \
