ORBITER=src/apps/orbiter/orbiter.out

problem2: problem2_q9 problem2_q11 problem2_q13 problem2_q17

problem2_q9:
	$(ORBITER) -v 1 \
		-define F -finite_field -q 9 -end \
		-define P -projective_space -n 2 -field F -v 0 -end \
		-define C -quartic_curve -space P -by_normal_form '6,3,4,6,2,5' -end \
		-with C -do -quartic_curve_activity -draw_bitmap -end

problem2_q11:
	$(ORBITER) -v 1 \
		-define F -finite_field -q 11 -end \
		-define P -projective_space -n 2 -field F -v 0 -end \
		-define C -quartic_curve -space P -by_normal_form '8,5,3,2,10,7' -end \
		-with C -do -quartic_curve_activity -draw_bitmap -end

problem2_q13:
	$(ORBITER) -v 1 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space -n 2 -field F -v 0 -end \
		-define C -quartic_curve -space P -by_normal_form '3,2,10,4,2,3' -end \
		-with C -do -quartic_curve_activity -draw_bitmap -end

problem2_q17:
	$(ORBITER) -v 1 \
		-define F -finite_field -q 17 -end \
		-define P -projective_space -n 2 -field F -v 0 -end \
		-define C -quartic_curve -space P -by_normal_form '2,7,16,3,9,4' -end \
		-with C -do -quartic_curve_activity -draw_bitmap -end

clean_bitmaps:
	rm -f quartic_curve_by_normal_form_q*_incma_draw.bmp
