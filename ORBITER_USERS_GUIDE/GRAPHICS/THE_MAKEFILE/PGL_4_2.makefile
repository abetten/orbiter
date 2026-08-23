
PGL_4_2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define gens -vector -file PGL_4_2_gens.csv -end \
		-define G -permutation_group \
		-bsgs PGL_4_2 "{\rm PGL}(4,2)" 15 20160 "0,1,2,3" 6 gens -end \
