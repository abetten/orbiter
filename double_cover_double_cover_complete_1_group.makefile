
double_cover_double_cover_complete_1:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define gens -vector -file double_cover_double_cover_complete_1_gens.csv -end \
		-define G -permutation_group \
		-bsgs double_cover_double_cover_complete_1 "{\rm double\_cover\_{double_cover_complete_1}}" 7 4 "1,4" 2 gens -end \
