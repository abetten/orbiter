
Hamming_7_2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define gens -vector -file Hamming_7_2_gens.csv -end \
		-define G -permutation_group \
		-bsgs Hamming_7_2 "{\rm Hamming\_7\_2}" 128 645120 "0,31,47,115,117" 7 gens -end \
