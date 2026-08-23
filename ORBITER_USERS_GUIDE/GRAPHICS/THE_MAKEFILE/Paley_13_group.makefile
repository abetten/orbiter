
Paley_13:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define gens -vector -file Paley_13_gens.csv -end \
		-define G -permutation_group \
		-bsgs Paley_13 "Paley\_13" 13 78 "0,1" 3 gens -end \
