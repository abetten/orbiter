// classify_double_sixes.C
// 
// Anton Betten
//
// October 10, 2017
//
// based on surface_classify_wedge.C started September 2, 2016
// 
//
//

#include "orbiter.h"


classify_double_sixes::classify_double_sixes()
{
	null();
}

classify_double_sixes::~classify_double_sixes()
{
	freeself();
}

void classify_double_sixes::null()
{
	q = 0;
	F = NULL;
	Surf_A = NULL;
	Surf = NULL;

	LG = NULL;

	A2 = NULL;
	AW = NULL;
	Elt0 = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
	Surf = NULL;
	SG_line_stab = NULL;
	Neighbors = NULL;
	Neighbor_to_line = NULL;
	Neighbor_to_klein = NULL;
	Line_to_neighbor = NULL;
	Stab = NULL;
	stab_gens = NULL;
	orbit = NULL;
	line_to_orbit = NULL;
	orbit_to_line = NULL;
	Pts_klein = NULL;
	Pts_wedge = NULL;
	Pts_wedge_to_line = NULL;
	line_to_pts_wedge = NULL;
	A_on_neighbors = NULL;
	Five_plus_one = NULL;
	u = NULL;
	v = NULL;
	w = NULL;
	u1 = NULL;
	v1 = NULL;
	len = 0;
	Idx = NULL;
	nb = 0;
	Po = NULL;

	Flag_orbits = NULL;

	Double_sixes = NULL;

}

void classify_double_sixes::freeself()
{
	if (Elt0) {
		FREE_INT(Elt0);
		}
	if (Elt1) {
		FREE_INT(Elt1);
		}
	if (Elt2) {
		FREE_INT(Elt2);
		}
	if (Elt3) {
		FREE_INT(Elt3);
		}
	if (Elt4) {
		FREE_INT(Elt4);
		}
	if (Idx) {
		FREE_INT(Idx);
		}
	if (Po) {
		FREE_INT(Po);
		}
	if (Flag_orbits) {
		delete Flag_orbits;
		}
	if (Double_sixes) {
		delete Double_sixes;
		}
	null();
}

void classify_double_sixes::init(surface_with_action *Surf_A, linear_group *LG, 
	int argc, const char **argv, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_double_sixes::init" << endl;
		}
	classify_double_sixes::Surf_A = Surf_A;
	classify_double_sixes::LG = LG;
	F = Surf_A->F;
	q = F->q;
	A = Surf_A->A;
	Surf = Surf_A->Surf;
	
	
	u = NEW_INT(6);
	v = NEW_INT(6);
	w = NEW_INT(6);
	u1 = NEW_INT(6);
	v1 = NEW_INT(6);
	


	A = LG->A_linear;
	A2 = LG->A2;

	if (A2->type_G != action_on_wedge_product_t) {
		cout << "classify_double_sixes::init group must act in wedge action" << endl;
		exit(1);
		}

	AW = A2->G.AW;

	Elt0 = NEW_INT(A->elt_size_in_INT);
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	Elt4 = NEW_INT(A->elt_size_in_INT);

	pt0_line = 0; // pt0 = the line spanned by 1000, 0100 
		// (we call it point because it is a point on the Klein quadric)
	pt0_wedge = 0; // in wedge coordinates 100000
	pt0_klein = 0; // in klein coordinates 100000


	if (f_v) {
		cout << "classify_double_sixes::init before SG_line_stab->generators_for_parabolic_subgroup" << endl;
		}


	SG_line_stab = new strong_generators;
	SG_line_stab->generators_for_parabolic_subgroup(A, 
		A->G.matrix_grp, 2, verbose_level - 1);

	if (f_v) {
		cout << "classify_double_sixes::init after SG_line_stab->generators_for_parabolic_subgroup" << endl;
		}



	if (f_v) {
		cout << "classify_double_sixes::init before compute_neighbors" << endl;
		}
	compute_neighbors(verbose_level - 1);
	{
	spreadsheet *Sp;
	make_spreadsheet_of_neighbors(Sp, 0 /* verbose_level */);
	delete Sp;
	}
	if (f_v) {
		cout << "classify_double_sixes::init after compute_neighbors nb_neighbors = " << nb_neighbors << endl;
		cout << "Neighbors=";
		INT_vec_print(cout, Neighbors, nb_neighbors);
		cout << endl;
		}





	if (f_v) {
		cout << "classify_double_sixes::init computing restricted action on neighbors" << endl;
		}

	A_on_neighbors = new action;
	A_on_neighbors->induced_action_by_restriction(*A2, 
		FALSE /* f_induce_action */, NULL, 
		nb_neighbors, Neighbors, 
		0 /* verbose_level */);

	if (f_v) {
		cout << "classify_double_sixes::init restricted action on neighbors has been computed" << endl;
		}


	Five_plus_one = new generator;
	Five_plus_one->read_arguments(argc, argv, 0);



	cout << "classify_double_sixes::init before Five_plus_one->init" << endl;
	Five_plus_one->init(A, A_on_neighbors, 
		SG_line_stab,
		5 /* sz */, 
		verbose_level - 1 + 5);
	cout << "classify_double_sixes::init after Five_plus_one->init" << endl;


	Five_plus_one->init_check_func(callback_partial_ovoid_test, 
		(void *)this /* candidate_check_data */);


	//Five_plus_one->f_print_function = TRUE;
	//Five_plus_one->print_function = callback_print_set;
	//Five_plus_one->print_function_data = (void *) this;

	BYTE fname_base[1000];
	sprintf(fname_base, "fiveplusone_q%ld", q);
	
	Five_plus_one->depth = 5;
	strcpy(Five_plus_one->fname_base, fname_base);

	INT nb_oracle_nodes = ONE_MILLION;
	
	if (f_v) {
		cout << "classify_double_sixes::init calling init_oracle with " << nb_oracle_nodes << " nodes" << endl;
		}
	
	Five_plus_one->init_oracle(nb_oracle_nodes, verbose_level - 1);

	if (f_v) {
		cout << "classify_double_sixes::init after calling init_root_node" << endl;
		}
	
	Five_plus_one->root[0].init_root_node(Five_plus_one, Five_plus_one->verbose_level);

	if (f_v) {
		cout << "classify_double_sixes::init done" << endl;
		}
}


void classify_double_sixes::compute_neighbors(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, a, b, c;

	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors" << endl;
		}

	nb_neighbors = (q + 1) * q * (q + 1);
	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors nb_neighbors = " << nb_neighbors << endl;
		}
	Neighbors = NEW_INT(nb_neighbors);
	Neighbor_to_line = NEW_INT(nb_neighbors);
	Neighbor_to_klein = NEW_INT(nb_neighbors);
	
	INT sz;

	// At first, we get the neighbors as points on the Klein quadric:
	// Later, we will change them to wedge ranks:

	Surf->O->perp(0, Neighbors, sz, 0 /* verbose_level*/);
	
	if (sz != nb_neighbors) {
		cout << "classify_double_sixes::compute_neighbors sz != nb_neighbors" << endl;
		exit(1);
		}
	cout << "classify_double_sixes::compute_neighbors nb_neighbors = " << nb_neighbors << endl;
	
	Line_to_neighbor = NEW_INT(Surf->nb_lines_PG_3);
	for (i = 0; i < Surf->nb_lines_PG_3; i++) {
		Line_to_neighbor[i] = -1;
		}

	// Convert Neighbors from points on the Klein quadric to wedge points:
	Surf->klein_to_wedge_vec(Neighbors, Neighbors, nb_neighbors);

	// Sort the set Neighbors:
	INT_vec_heapsort(Neighbors, nb_neighbors);




	// Establish the bijection between Neighbors and Lines in PG(3,q) 
	// by going through the Klein correspondence.
	// It is important that this be done after we sort Neighbors.
	for (i = 0; i < nb_neighbors; i++) {
		a = Neighbors[i];
		AW->unrank_point(w, a);
		Surf->wedge_to_klein(w, v);
		b = Surf->O->rank_point(v, 1, 0 /* verbose_level*/);
		c = Surf->Klein->Point_on_quadric_to_line[b];
		Neighbor_to_line[i] = c;
		Line_to_neighbor[c] = i;
		}

	INT_vec_apply(Neighbor_to_line, Surf->Klein->Line_to_point_on_quadric, Neighbor_to_klein, nb_neighbors);



	if (f_v) {
		cout << "classify_double_sixes::compute_neighbors done" << endl;
		}
}

void classify_double_sixes::make_spreadsheet_of_neighbors(spreadsheet *&Sp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname_csv[1000];

	if (f_v) {
		cout << "classify_double_sixes::make_spreadsheet_of_neighbors" << endl;
		}

	sprintf(fname_csv, "neighbors_%ld.csv", q);
	

	Surf->make_spreadsheet_of_lines_in_three_kinds(Sp, 
		Neighbors, Neighbor_to_line, Neighbor_to_klein, nb_neighbors, 0 /* verbose_level */);

	cout << "before Sp->save " << fname_csv << endl;
	Sp->save(fname_csv, verbose_level);
	cout << "after Sp->save " << fname_csv << endl;





	if (f_v) {
		cout << "classify_double_sixes::make_spreadsheet_of_neighbors done" << endl;
		}
}

void classify_double_sixes::classify_partial_ovoids(INT f_draw_poset, 
	INT f_draw_poset_full, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT schreier_depth = 10000;
	INT f_use_invariant_subset_if_available = TRUE;
	INT f_debug = FALSE;
	INT t0 = os_ticks();


	if (f_v) {
		cout << "classify_double_sixes::classify_partial_ovoids" << endl;
		}
	if (f_v) {
		cout << "classify_double_sixes::classify_partial_ovoids nb_neighbors = " << nb_neighbors << endl;
		cout << "Neighbors=";
		INT_vec_print(cout, Neighbors, nb_neighbors);
		cout << endl;
		}
	if (f_v) {
		cout << "classify_double_sixes::classify_partial_ovoids classifying starter" << endl;
		}
	Five_plus_one->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_debug, 
		verbose_level + 3);
	if (f_v) {
		cout << "classify_double_sixes::classify_partial_ovoids classifying starter done" << endl;
		}
	
	if (f_draw_poset) {
		{
		BYTE fname_poset[1000];
		sprintf(fname_poset, "fiveplusone_q%ld", q);
		Five_plus_one->draw_poset(fname_poset, 5 /*depth*/, 0 /* data1 */, TRUE /* f_embedded */, FALSE /* f_sideways */, 0 /* verbose_level */);
		}
		}
	if (f_draw_poset_full) {
		{
		BYTE fname_poset[1000];
		sprintf(fname_poset, "fiveplusone_q%ld", q);
		Five_plus_one->draw_poset(fname_poset, 5 /*depth*/, 0 /* data1 */, TRUE /* f_embedded */, FALSE /* f_sideways */, 0 /* verbose_level */);
		}
		}
	if (q < 20) {
		{
		spreadsheet *Sp;
		Five_plus_one->make_spreadsheet_of_orbit_reps(Sp, 5);
		BYTE fname_csv[1000];
		sprintf(fname_csv, "fiveplusone_%ld.csv", q);
		Sp->save(fname_csv, verbose_level);
		delete Sp;
		}
		}
	if (f_v) {
		cout << "classify_double_sixes::classify_partial_ovoids done" << endl;
		}
}



INT classify_double_sixes::partial_ovoid_test(INT *S, INT len, INT verbose_level)
// Make sure that the set of lines is pairwise disjoint.
// No other tests are performed (as these other tests would be expensive).
{
	//verbose_level = 2;
	INT i, x, y;
	INT f_OK = TRUE;
	INT fxy;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "classify_double_sixes::partial_ovoid_test" << endl;
		}
	if (f_vv) {
		cout << "testing S=";
		INT_vec_print(cout, S, len);
		cout << endl;
		for (i = 0; i < len; i++) {
			AW->unrank_point(u, Neighbors[S[i]]);
			Surf->wedge_to_klein(u, u1);
			cout << i << " : " << S[i] << " : " << Neighbors[S[i]] << " : ";
			cout << " wedge: ";
			INT_vec_print(cout, u, 6);
			cout << " klein: ";
			INT_vec_print(cout, u1, 6);
			cout << endl;
			}
		}
	y = Neighbors[S[len - 1]];
	AW->unrank_point(v, y);
	Surf->wedge_to_klein(v, v1);
	
	for (i = 0; i < len - 1; i++) {
		x = Neighbors[S[i]];
		AW->unrank_point(u, x);
		Surf->wedge_to_klein(u, u1);

		fxy = Surf->O->evaluate_bilinear_form(u1, v1, 1);
		
		if (fxy == 0) {
			f_OK = FALSE;
			if (f_vv) {
				cout << "not OK; ";
				cout << "{x,y}={" << x << "," << y << "} are collinear" << endl;
				INT_vec_print(cout, u1, 6);
				cout << endl;
				INT_vec_print(cout, v1, 6);
				cout << endl;
				cout << "fxy=" << fxy << endl;
				}
			break;
			}
		}


#if 0
	if (f_OK && len >= 4) {
		INT *Rk;
		INT nb_subsets, a;
		INT *lines;
		
		lines = NEW_INT(len);
		for (i = 0; i < len; i++) {
			a = S[i];
			lines[i] = Neighbor_to_line[a];
			}
		if (!Surf->compute_rank_of_any_four(Rk, nb_subsets, lines, len, 0 /* verbose_level */)) {
			f_OK = FALSE;
			}
		if (f_v) {
			cout << "Ranks of 4 subsets of lines:" << endl;
			INT_vec_print(cout, Rk, nb_subsets);
			cout << endl;
			}
		FREE_INT(Rk);
		FREE_INT(lines);
		}



	if (f_OK && len >= 4) {
		INT *Trans;
		INT *lines;
		INT nb_subsets, a, j, l;
		
		lines = NEW_INT(len);
		for (i = 0; i < len; i++) {
			a = S[i];
			lines[i] = Neighbor_to_line[a];
			}

		if (!Surf->compute_transversals_of_any_four(Trans, nb_subsets, lines, len, 0 /* verbose_level */)) {
			f_OK = FALSE;
			}
		if (f_v) {
			cout << "Transversals:" << endl;
			INT_matrix_print(Trans, nb_subsets, 2);
			}


		INT *Trans2;
		Trans2 = NEW_INT(nb_subsets * 2);
		j = 0;
		for (i = 0; i < nb_subsets * 2; i++) {
			a = Trans[i];
			if (a >= 0) {
				Trans2[j++] = a;
				}
			}
		l = j;
		INT_vec_heapsort(Trans2, l);
		j = 0;
		for (i = 0; i < l; i++) {
			a = Trans2[i];
			if (i > 0) {
				if (a == Trans2[i - 1]) {
					continue;
					}
				}
			Trans2[j++] = a;
			}
		l = j;

		if (f_vv) {
			cout << "The transversals are:" << endl;
			Surf->Gr->print_set(Trans2, l);
			}

		if (l - 1 < nb_subsets) {
			f_OK = FALSE;
			}


		FREE_INT(Trans2);
		FREE_INT(Trans);
		FREE_INT(lines);
		}
#endif


#if 0
	if (f_OK) {

		INT S2[10];
		INT a, c, r;
	

		S2[0] = pt0_line;
		for (i = 0; i < len; i++) {
			a = S[i];
			c = Neighbor_to_line[a];
			S2[i + 1] = c;
			}
		
		if (FALSE) {
			cout << "line set: ";
			INT_vec_print(cout, S2, len + 1);
			cout << endl;
			}

		if (FALSE) {
			Gr->print_set(S2, len + 1);
			}

		r = Surf->compute_system_in_RREF(len + 1, S2, 0 /*verbose_level*/);
		if (r < 4 + 3 * len) {
			f_OK = FALSE;
			}
		}
#endif
	
	if (f_v) {
		if (!f_OK) {
			cout << "classify_double_sixes::partial_ovoid_test fails" << endl;
			}
		else {
			cout << "classify_double_sixes::partial_ovoid_test is OK" << endl;
			}
		}
	return f_OK;
}

void classify_double_sixes::test_orbits(INT verbose_level)
{
	//verbose_level += 2;
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE; // (verbose_level >= 2);
	INT i, r;
	INT S[5];
	INT S2[6];
	
	if (f_v) {
		cout << "classify_double_sixes::test_orbits" << endl;
		}
	len = Five_plus_one->nb_orbits_at_level(5);

	if (f_v) {
		cout << "classify_double_sixes::test_orbits testing " << len << " orbits of 5-sets of lines:" << endl;
		}
	nb = 0;
	Idx = NEW_INT(len);
	for (i = 0; i < len; i++) {
		if (f_vv || ((i % 1000) == 0)) {
			cout << "classify_double_sixes::test_orbits orbit " << i << " / " << len << ":" << endl;
			}
		Five_plus_one->get_set_by_level(5, i, S);
		if (f_vv) {
			cout << "set: ";
			INT_vec_print(cout, S, 5);
			cout << endl;
			}

		if (partial_ovoid_test(S, 5, 0 /*verbose_level - 2*/)) {
			if (f_vv) {
				cout << "classify_double_sixes::test_orbits orbit " << i << " / " << len << " passes the partial_ovoid_test" << endl;
				}
			}
		else {
			cout << "classify_double_sixes::test_orbits Does not pass the surface test, this should not happen" << endl;
			exit(1);
			}


		INT_vec_apply(S, Neighbor_to_line, S2, 5);
		S2[5] = pt0_line;
		if (f_vv) {
			cout << "5+1 lines = ";
			INT_vec_print(cout, S2, 6);
			cout << endl;
			}

#if 1
		if (f_vv) {
			Surf->Gr->print_set(S2, 6);
			}
#endif

		r = Surf->compute_system_in_RREF(6, S2, 0 /*verbose_level*/);
		if (f_vv) {
			cout << "classify_double_sixes::test_orbits orbit " << i << " / " << len << " has rank = " << r << endl;
			}
		if (r == 19) {
			Idx[nb++] = i;
			}
		}

	if (f_v) {
		cout << "classify_double_sixes::test_orbits we found " << nb << " / " << len << " orbits where the rank is 19" << endl;
		cout << "Idx=";
		INT_vec_print(cout, Idx, nb);
		cout << endl;
		}
	if (f_v) {
		cout << "classify_double_sixes::test_orbits done" << endl;
		}
}

void classify_double_sixes::make_spreadsheet_of_fiveplusone_configurations(spreadsheet *&Sp, 
	/*INT f_with_fusion, INT *Fusion, vector_ge *transporter,*/ 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT nb_orbits;
	INT i, k;
	INT *Stab_order;
	INT *Len;
	BYTE **Transporter;
	BYTE **Text;
	INT *rep;
	INT *lines;
	INT *data;
	longinteger_object go;
	longinteger_object len;
	BYTE fname_csv[1000];
	BYTE str[1000];
	

	if (f_v) {
		cout << "classify_double_sixes::make_spreadsheet_of_fiveplusone_configurations" << endl;
		}
	sprintf(fname_csv, "fiveplusone19_%ld.csv", q);

	k = 5;

	//nb_orbits = Five_plus_one->nb_orbits_at_level(k);
	rep = NEW_INT(k);
	lines = NEW_INT(k);
	Stab_order = NEW_INT(nb);
	Len = NEW_INT(nb);
	Transporter = NEW_PBYTE(nb);
	Text = NEW_PBYTE(nb);
	data = NEW_INT(A->make_element_size);

	for (i = 0; i < nb; i++) {
		Five_plus_one->get_set_by_level(k, Idx[i], rep);
		INT_vec_apply(rep, Neighbor_to_line, lines, k);
		Five_plus_one->get_stabilizer_order(k, Idx[i], go);
		Five_plus_one->orbit_length(Idx[i], k, len);
		Stab_order[i] = go.as_INT();
		Len[i] = len.as_INT();
		}
	for (i = 0; i < nb; i++) {
		Five_plus_one->get_set_by_level(k, Idx[i], rep);
		INT_vec_apply(rep, Neighbor_to_line, lines, k);

		INT_vec_print_to_str(str, lines, k);

		Text[i] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text[i], str);
		}

#if 0
	if (f_with_fusion) {
		for (i = 0; i < nb; i++) {
			if (Fusion[i] == -2) {
				str[0] = 0;
				strcat(str, "\"N/A\"");
				}
			else {
				A->element_code_for_make_element(transporter->ith(i), data);


				INT_vec_print_to_str(str, data, A->make_element_size);

				}
			Transporter[i] = NEW_BYTE(strlen(str) + 1);
			strcpy(Transporter[i], str);
			}
		}
#endif


	Sp = new spreadsheet;
#if 0
	if (f_with_fusion) {
		Sp->init_empty_table(nb + 1, 7);
		}
	else {
		Sp->init_empty_table(nb + 1, 5);
		}
#endif
	Sp->init_empty_table(nb + 1, 5);
	Sp->fill_column_with_row_index(0, "Orbit");
	Sp->fill_column_with_INT(1, Idx, "Idx");
	Sp->fill_column_with_text(2, (const BYTE **) Text, "Rep");
	Sp->fill_column_with_INT(3, Stab_order, "Stab_order");
	Sp->fill_column_with_INT(4, Len, "Orbit_length");
#if 0
	if (f_with_fusion) {
		Sp->fill_column_with_INT(5, Fusion, "Fusion");
		Sp->fill_column_with_text(6, (const BYTE **) Transporter, "Transporter");
		}
#endif
	cout << "before Sp->save " << fname_csv << endl;
	Sp->save(fname_csv, verbose_level);
	cout << "after Sp->save " << fname_csv << endl;

	FREE_INT(rep);
	FREE_INT(lines);
	FREE_INT(Stab_order);
	FREE_INT(Len);
	for (i = 0; i < nb; i++) {
		FREE_BYTE(Text[i]);
		}
	FREE_PBYTE(Text);
	for (i = 0; i < nb; i++) {
		FREE_BYTE(Transporter[i]);
		}
	FREE_PBYTE(Transporter);
	FREE_INT(data);
	if (f_v) {
		cout << "classify_double_sixes::make_spreadsheet_of_fiveplusone_configurations done" << endl;
		}
}


void classify_double_sixes::identify_five_plus_one(INT *five_lines, INT transversal_line, 
	INT *five_lines_out_as_neighbors, INT &orbit_index, 
	INT *transporter, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT W1[5];
	INT W2[5];
	INT N1[5];

	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one" << endl;
		}

	
	Surf->line_to_wedge_vec(five_lines, W1, 5);


	A->make_element_which_moves_a_line_in_PG3q(Surf->Gr, transversal_line, Elt0, 0 /* verbose_level */);


	A2->map_a_set(W1, W2, 5, Elt0, 0 /* verbose_level */);

	INT_vec_search_vec(Neighbors, nb_neighbors, W2, 5, N1);

	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one tracing the set ";
		INT_vec_print(cout, N1, 5);
		cout << endl;
		}
	orbit_index = Five_plus_one->trace_set(N1, 5, 5, five_lines_out_as_neighbors, Elt1, 0 /* verbose_level */);
	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one orbit_index = " << orbit_index << endl;
		}

	
	A->element_mult(Elt0, Elt1, transporter, 0);
	if (f_v) {
		cout << "classify_double_sixes::identify_five_plus_one done" << endl;
		}
}

void classify_double_sixes::classify(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classify_double_sixes::classify" << endl;
		}

	if (f_v) {
		cout << "classify_double_sixes::classify before downstep" << endl;
		}
	downstep(verbose_level);
	if (f_v) {
		cout << "classify_double_sixes::classify after downstep" << endl;
		cout << "we found " << Flag_orbits->nb_flag_orbits << " flag orbits out of " << Five_plus_one->nb_orbits_at_level(5) << " orbits" << endl;
		}

	if (f_v) {
		cout << "classify_double_sixes::classify before upstep" << endl;
		}
	upstep(verbose_level);
	if (f_v) {
		cout << "classify_double_sixes::classify after upstep" << endl;
		cout << "we found " << Double_sixes->nb_orbits << " double sixes out of " << Flag_orbits->nb_flag_orbits << " flag orbits" << endl;
		}

	if (f_v) {
		cout << "classify_double_sixes::classify done" << endl;
		}
}

void classify_double_sixes::downstep(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT f, i, nb_orbits, nb_flag_orbits, c;

	if (f_v) {
		cout << "classify_double_sixes::downstep" << endl;
		}

	if (f_v) {
		cout << "classify_double_sixes::downstep before test_orbits" << endl;
		}
	test_orbits(verbose_level - 1);
	if (f_v) {
		cout << "classify_double_sixes::downstep after test_orbits" << endl;
		cout << "Idx=";
		INT_vec_print(cout, Idx, nb);
		cout << endl;
		}



	nb_orbits = Five_plus_one->nb_orbits_at_level(5);
	
	Flag_orbits = new flag_orbits;
	Flag_orbits->init(A, A2, nb_orbits /* nb_primary_orbits_lower */, 
		5 + 6 + 12 /* pt_representation_sz */, nb, 
		verbose_level);

	if (f_v) {
		cout << "classify_double_sixes::downstep initializing flag orbits" << endl;
		}

	nb_flag_orbits = 0;
	for (f = 0; f < nb; f++) {

		i = Idx[f];
		if (f_v) {
			cout << "orbit " << f << " / " << nb << " with rank = 19 is orbit " << i << " / " << nb_orbits << endl;
			}

		set_and_stabilizer *R;
		longinteger_object ol;
		longinteger_object go;
		INT dataset[23];

		R = Five_plus_one->get_set_and_stabilizer(5 /* level */, i /* orbit_at_level */, 0 /* verbose_level */);

		Five_plus_one->orbit_length(i /* node */, 5 /* level */, ol);

		R->Strong_gens->group_order(go);

		INT_vec_copy(R->data, dataset, 5);

		INT_vec_apply(dataset, Neighbor_to_line, dataset + 5, 5);
		
		dataset[10] = pt0_line;

		INT double_six[12];
		if (f_vv) {
			cout << "5+1 lines = ";
			INT_vec_print(cout, dataset + 5, 6);
			cout << endl;
			}

		if (f_vv) {
			cout << "classify_double_sixes::downstep before create_double_six_from_five_lines_with_a_common_transversal" << endl;
			}

		c = Surf_A->create_double_six_from_five_lines_with_a_common_transversal(dataset + 5, pt0_line, double_six, 0 /*verbose_level*/);
		//c = Surf_A->create_double_six_safely(dataset + 5, pt0_line, double_six, verbose_level);
		//c = Surf->create_double_six_from_five_lines_with_a_common_transversal(dataset + 5, double_six, 0 /*verbose_level*/);
		
		if (c) {

			if (f_vv) {
				cout << "The starter configuration is good, a double six has been computed:" << endl;
				INT_matrix_print(double_six, 2, 6);
				}

			INT_vec_copy(double_six, dataset + 11, 12);


			Flag_orbits->Flag_orbit_node[nb_flag_orbits].init(Flag_orbits, nb_flag_orbits /* flag_orbit_index */, 
				i /* downstep_primary_orbit */, 0 /* downstep_secondary_orbit */, 
				ol.as_INT() /* downstep_orbit_len */, FALSE /* f_long_orbit */, 
				dataset /* INT *pt_representation */, R->Strong_gens, 
				verbose_level - 2);
			R->Strong_gens = NULL;

			if (f_vv) {
				cout << "orbit " << f << " / " << nb << " with rank = 19 is orbit " << i << " / " << nb_orbits << ", stab order " << go << endl;
				}
			nb_flag_orbits++;
			}
		else {
			if (f_vv) {
				cout << "orbit " << f << " / " << nb << " with rank = 19 does not yield a double six, skipping" << endl;
				}
			}


		delete R;
		}

	Flag_orbits->nb_flag_orbits = nb_flag_orbits;


	Po = NEW_INT(nb_flag_orbits);
	for (f = 0; f < nb_flag_orbits; f++) {
		Po[f] = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
		}
	if (f_v) {
		cout << "classify_double_sixes::downstep we found " << nb_flag_orbits << " flag orbits out of " << nb_orbits << " orbits" << endl;
		}
	if (f_v) {
		cout << "classify_double_sixes::downstep initializing flag orbits done" << endl;
		}
}


void classify_double_sixes::upstep(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, h, k, i0;
	INT f, po, so;
	INT *f_processed;
	INT nb_processed;

	if (f_v) {
		cout << "classify_double_sixes::upstep" << endl;
		}


	f_processed = NEW_INT(Flag_orbits->nb_flag_orbits);
	INT_vec_zero(f_processed, Flag_orbits->nb_flag_orbits);
	nb_processed = 0;

	Double_sixes = new classification;

	longinteger_object go;
	A->group_order(go);

	Double_sixes->init(A, A2, Flag_orbits->nb_flag_orbits, 12, go, verbose_level);


	if (f_v) {
		cout << "flag orbit : downstep_primary_orbit" << endl;
		cout << "f : po" << endl;
		for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {
			po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
			cout << f << " : " << po << endl;
			}
		}
	for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {

		double progress;
		INT dataset[23];
		
		if (f_processed[f]) {
			continue;
			}

		progress = ((double)nb_processed * 100. ) / (double) Flag_orbits->nb_flag_orbits;

		if (f_v) {
			cout << "Defining new orbit " << Flag_orbits->nb_primary_orbits_upper << " from flag orbit " << f << " / " << Flag_orbits->nb_flag_orbits << " progress=" << progress << "%" << endl;
			}
		Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit = Flag_orbits->nb_primary_orbits_upper;
		

		if (Flag_orbits->pt_representation_sz != 23) {
			cout << "Flag_orbits->pt_representation_sz != 23" << endl;
			exit(1);
			}
		po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
		so = Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
		if (f_v) {
			cout << "po=" << po << " so=" << so << endl;
			}
		INT_vec_copy(Flag_orbits->Pt + f * 23, dataset, 23);




		vector_ge *coset_reps;
		INT nb_coset_reps;
		
		coset_reps = new vector_ge;
		coset_reps->init(Surf_A->A);
		coset_reps->allocate(12);


		strong_generators *S;
		longinteger_object go;
		INT double_six[12];

		INT_vec_copy(dataset + 11, double_six, 12);

		if (f_v) {
			cout << "double six:";
			INT_vec_print(cout, double_six, 12);
			cout << endl;
			}
		S = Flag_orbits->Flag_orbit_node[f].gens->create_copy();
		S->group_order(go);
		if (f_v) {
			cout << "po=" << po << " so=" << so << " go=" << go << endl;
			}

		nb_coset_reps = 0;
		for (i = 0; i < 2; i++) {
			for (j = 0; j < 6; j++) {
			
				if (f_v) {
					cout << "i=" << i << " j=" << j << endl;
					}
				INT transversal_line;
				INT five_lines[5];
				//INT five_lines_in_wedge[5];
				INT five_lines_out_as_neighbors[5];
				INT orbit_index;
				INT f2;
				
				transversal_line = double_six[i * 6 + j];
				i0 = 1 - i;
				k = 0;
				for (h = 0; h < 6; h++) {
					if (h == j) {
						continue;
						}
					five_lines[k++] = double_six[i0 * 6 + h];
					}

				//INT_vec_apply(five_lines, Line_to_neighbor, five_lines_in_wedge, 5);
				
				if (f_v) {
					cout << "transversal_line = " << transversal_line << " five_lines=";
					INT_vec_print(cout, five_lines, 5);
					cout << endl;
					}
				identify_five_plus_one(five_lines, transversal_line, 
					five_lines_out_as_neighbors, orbit_index, 
					Elt3 /* transporter */, verbose_level - 2);

				if (f_v) {
					cout << "We found a transporter:" << endl;
					A->element_print_quick(Elt3, cout);
					}

				if (!INT_vec_search(Po, Flag_orbits->nb_flag_orbits, orbit_index, f2)) {
					cout << "cannot find orbit " << orbit_index << " in Po" << endl;
					cout << "Po=";
					INT_vec_print(cout, Po, Flag_orbits->nb_flag_orbits);
					cout << endl;
					exit(1);
					}

				if (Flag_orbits->Flag_orbit_node[f2].downstep_primary_orbit != orbit_index) {
					cout << "Flag_orbits->Flag_orbit_node[f2].downstep_primary_orbit != orbit_index" << endl;
					exit(1);
					}





		
				if (f2 == f) {
					if (f_v) {
						cout << "We found an automorphism of the double six:" << endl;
						A->element_print_quick(Elt3, cout);
						cout << endl;
						}
					A->element_move(Elt3, coset_reps->ith(nb_coset_reps), 0);
					nb_coset_reps++;
					//S->add_single_generator(Elt3, 2 /* group_index */, verbose_level - 2);
					}
				else {
					if (f_v) {
						cout << "We are identifying flag orbit " << f2 << " with flag orbit " << f << endl;
						}
					if (!f_processed[f2]) {
						Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit = Flag_orbits->nb_primary_orbits_upper;
						Flag_orbits->Flag_orbit_node[f2].f_fusion_node = TRUE;
						Flag_orbits->Flag_orbit_node[f2].fusion_with = f;
						Flag_orbits->Flag_orbit_node[f2].fusion_elt = NEW_INT(A->elt_size_in_INT);
						A->element_invert(Elt3, Flag_orbits->Flag_orbit_node[f2].fusion_elt, 0);
						f_processed[f2] = TRUE;
						nb_processed++;
						}
					else {
						cout << "Flag orbit " << f2 << " has already been identified with flag orbit " << f << endl;
						if (Flag_orbits->Flag_orbit_node[f2].fusion_with != f) {
							cout << "Flag_orbits->Flag_orbit_node[f2].fusion_with != f" << endl;
							exit(1);
							}
						}
					}
				} // next j
			} // next i


		coset_reps->reallocate(nb_coset_reps);

		strong_generators *Aut_gens;

		{
		longinteger_object ago;
		
		if (f_v) {
			cout << "classify_double_sixes::upstep Extending the group by a factor of " << nb_coset_reps << endl;
			}
		Aut_gens = new strong_generators;
		Aut_gens->init_group_extension(S, coset_reps, nb_coset_reps, verbose_level - 2);

		Aut_gens->group_order(ago);


		if (f_v) {
			cout << "the double six has a stabilizer of order " << ago << endl;
			cout << "The double six stabilizer is:" << endl;
			Aut_gens->print_generators_tex(cout);
			}
		}



		Double_sixes->Orbit[Flag_orbits->nb_primary_orbits_upper].init(Double_sixes, 
			Flag_orbits->nb_primary_orbits_upper, 
			Aut_gens, dataset + 11, verbose_level);

		delete coset_reps;
		//delete Aut_gens;
		delete S;
		
		f_processed[f] = TRUE;
		nb_processed++;
		Flag_orbits->nb_primary_orbits_upper++;
		} // next f


	if (nb_processed != Flag_orbits->nb_flag_orbits) {
		cout << "nb_processed != Flag_orbits->nb_flag_orbits" << endl;
		cout << "nb_processed = " << nb_processed << endl;
		cout << "Flag_orbits->nb_flag_orbits = " << Flag_orbits->nb_flag_orbits << endl;
		exit(1);
		}

	Double_sixes->nb_orbits = Flag_orbits->nb_primary_orbits_upper;
	
	if (f_v) {
		cout << "We found " << Flag_orbits->nb_primary_orbits_upper << " orbits of double sixes" << endl;
		}
	
	FREE_INT(f_processed);


	if (f_v) {
		cout << "classify_double_sixes::upstep done" << endl;
		}
}

void classify_double_sixes::print_five_plus_ones(ostream &ost)
{
	INT f, i, l;

	l = Five_plus_one->nb_orbits_at_level(5);

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Classification of $5+1$ Configurations in $\\PG(3," << q << ")$}" << endl;



	{
	longinteger_object go;
	A->Strong_gens->group_order(go);

	ost << "The order of the group is ";
	go.print_not_scientific(ost);
	ost << "\\\\" << endl;

	ost << "\\bigskip" << endl;
	}



	longinteger_domain D;
	longinteger_object ol, Ol;
	Ol.create(0);

	ost << "The group has " 
		<< l 
		<< " orbits on five plus one configurations in $\\PG(3," << q << ").$" << endl << endl;

	ost << "Of these, " << nb << " impose 19 conditions." << endl << endl;


	ost << "Of these, " << Flag_orbits->nb_flag_orbits << " are associated with double sixes. They are:" << endl << endl;


	for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {

		i = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;


		set_and_stabilizer *R;

		R = Five_plus_one->get_set_and_stabilizer(5 /* level */, i /* orbit_at_level */, 0 /* verbose_level */);
		Five_plus_one->orbit_length(i /* node */, 5 /* level */, ol);
		D.add_in_place(Ol, ol);
		
		ost << "$" << f << " / " << Flag_orbits->nb_flag_orbits << "$ is orbit $" << i << " / " << l << "$ $" << endl;
		R->print_set_tex(ost);
		ost << "$ orbit length $";
		ol.print_not_scientific(ost);
		ost << "$\\\\" << endl;

		delete R;
		}

	ost << "The overall number of five plus one configurations associated with double sixes in $\\PG(3," << q << ")$ is: " << Ol << "\\\\" << endl;


	//Double_sixes->print_latex(ost, "Classification of Double Sixes");
}

void classify_double_sixes::identify_double_six(INT *double_six, 
	INT *transporter, INT &orbit_index, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f, f2;
	INT *Elt1;
	INT *Elt2;
	INT transversal_line;
	INT five_lines[5];
	INT five_lines_out_as_neighbors[5];
	INT po;

	if (f_v) {
		cout << "classify_double_sixes::identify_double_six" << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	
	if (f_v) {
		cout << "classify_double_sixes::identify_double_six identifying the five lines a_1,...,a_5 with transversal b_6" << endl;
		}
	transversal_line = double_six[11];
	INT_vec_copy(double_six, five_lines, 5);
	
	identify_five_plus_one(five_lines, transversal_line, 
		five_lines_out_as_neighbors, po, 
		Elt1 /* transporter */, 0 /* verbose_level */);

	if (f_vv) {
		cout << "po=" << po << endl;
		cout << "Elt1=" << endl;
		A->element_print_quick(Elt1, cout);
		}

	
	if (!INT_vec_search(Po, Flag_orbits->nb_flag_orbits, po, f)) {
		cout << "classify_double_sixes::identify_double_six did not find po in Po" << endl;
		exit(1);
		}
	
	if (f_vv) {
		cout << "po=" << po << " f=" << f << endl;
		}

	if (Flag_orbits->Flag_orbit_node[f].f_fusion_node) {
		A->element_mult(Elt1, Flag_orbits->Flag_orbit_node[f].fusion_elt, Elt2, 0);
		f2 = Flag_orbits->Flag_orbit_node[f].fusion_with;
		orbit_index = Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit;
		}
	else {
		f2 = -1;
		A->element_move(Elt1, Elt2, 0);
		orbit_index = Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit;
		}
	if (f_v) {
		cout << "classify_double_sixes::identify_double_six f=" << f << " f2=" << f2 << " orbit_index=" << orbit_index << endl;
		}
	A->element_move(Elt2, transporter, 0);
	if (f_vv) {
		cout << "transporter=" << endl;
		A->element_print_quick(transporter, cout);
		}
	
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	if (f_v) {
		cout << "classify_double_sixes::identify_double_six done" << endl;
		}
}

void classify_double_sixes::write_file(ofstream &fp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "classify_double_sixes::write_file" << endl;
		}
	fp.write((char *) &q, sizeof(INT));
	fp.write((char *) &nb_neighbors, sizeof(INT));
	fp.write((char *) &len, sizeof(INT));
	fp.write((char *) &nb, sizeof(INT));
	fp.write((char *) &Flag_orbits->nb_flag_orbits, sizeof(INT));

	for (i = 0; i < nb; i++) {
		fp.write((char *) &Idx[i], sizeof(INT));
		}
	for (i = 0; i < Flag_orbits->nb_flag_orbits; i++) {
		fp.write((char *) &Po[i], sizeof(INT));
		}


	Five_plus_one->write_file(fp, 5 /* depth_completed */, verbose_level);


	Flag_orbits->write_file(fp, verbose_level);

	Double_sixes->write_file(fp, verbose_level);

	if (f_v) {
		cout << "classify_double_sixes::write_file finished" << endl;
		}
}

void classify_double_sixes::read_file(ifstream &fp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, nb_flag_orbits;
	
	if (f_v) {
		cout << "classify_double_sixes::read_file" << endl;
		}
	fp.read((char *) &q, sizeof(INT));
	fp.read((char *) &nb_neighbors, sizeof(INT));
	fp.read((char *) &len, sizeof(INT));
	fp.read((char *) &nb, sizeof(INT));
	fp.read((char *) &nb_flag_orbits, sizeof(INT));

	Idx = NEW_INT(nb);
	for (i = 0; i < nb; i++) {
		fp.read((char *) &Idx[i], sizeof(INT));
		}

	Po = NEW_INT(nb_flag_orbits);
	for (i = 0; i < nb_flag_orbits; i++) {
		fp.read((char *) &Po[i], sizeof(INT));
		}


	INT depth_completed;

	Five_plus_one->read_file(fp, depth_completed, verbose_level);
	if (depth_completed != 5) {
		cout << "classify_double_sixes::read_file depth_completed != 5" << endl;
		exit(1);
		}


	Flag_orbits = new flag_orbits;
	Flag_orbits->A = A;
	Flag_orbits->A2 = A;
	Flag_orbits->read_file(fp, verbose_level);

	Double_sixes = new classification;
	Double_sixes->A = A;
	Double_sixes->A2 = A2;

	A->group_order(Double_sixes->go);

	Double_sixes->read_file(fp, verbose_level);

	if (f_v) {
		cout << "classify_double_sixes::read_file finished" << endl;
		}
}


