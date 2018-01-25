// surface_classify.C
// 
// Anton Betten
// August 28, 2016
//
// 
//
//

#include "orbiter.h"

#if 0
surface_classify::surface_classify()
{
	null();
}

surface_classify::~surface_classify()
{
	freeself();
}

void surface_classify::null()
{
	Klein = NULL;
	Surf = NULL;
	Sch = NULL;
	Stab = NULL;
	stab_gens = NULL;
	Pts = NULL;
	A_on_neighbors = NULL;
}

void surface_classify::freeself()
{
	if (Klein) {
		delete Klein;
		}
	if (Surf) {
		delete Surf;
		}
	if (Sch) {
		delete Sch;
		}
	if (Stab) {
		delete Stab;
		}
	if (stab_gens) {
		delete stab_gens;
		}
	if (Pts) {
		FREE_INT(Pts);
		}
	if (A_on_neighbors) {
		delete A_on_neighbors;
		}
	null();
}

void surface_classify::init(finite_field *F, action *A, orthogonal *O, 
	generator *gen, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	
	
	if (f_v) {
		cout << "surface_classify::init" << endl;
		}
	surface_classify::F = F;
	q = F->q;
	surface_classify::A = A;
	surface_classify::O = O;
	surface_classify::gen = gen;

	u = NEW_INT(6);
	v = NEW_INT(6);

	if (f_v) {
		cout << "surface_classify::init initializing Klein correspondence" << endl;
		}
	Klein = new klein_correspondence;
	Klein->init(F, O, verbose_level);
	if (f_v) {
		cout << "surface_classify::init initializing Klein correspondence done" << endl;
		}


	Surf = new surface;

	if (f_v) {
		cout << "surface_classify::init initializing surface" << endl;
		}
	Surf->init(F, verbose_level);
	if (f_v) {
		cout << "surface_classify::init initializing surface done" << endl;
		}



	strong_generators *even_gens;
	longinteger_object go;

	even_gens = new strong_generators;

	if (f_v) {
		cout << "surface_classify::init before even_gens->init" << endl;
		}
	even_gens->init(A, verbose_level);

	if (f_v) {
		cout << "surface_classify::init before gens->even_subgroup" << endl;
		}
	even_gens->even_subgroup(verbose_level);

	even_gens->group_order(go);
	if (f_v) {
		cout << "surface_classify::init created generators for the even subgroup of order " << go << endl;
		}



#if 1
#if 1
	A->point_stabilizer_any_point_with_given_group(even_gens, 
		pt, 
		Sch, Stab, stab_gens, 
		verbose_level);
#else
	A->point_stabilizer_any_point(pt, 
		Sch, Stab, stab_gens, 
		verbose_level);
#endif
	Stab->group_order(go);

	cout << "The special point is " << pt << endl;

#else
	INT f, len;

	if (f_v) {
		cout << "surface_classify::init computing all point orbits:" << endl;
		}
	Sch = A->Strong_gens->orbits_on_points_schreier(A, 0 /* verbose_level */);
	//A->compute_all_point_orbits(Sch, *A->Strong_gens->gens, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_classify::init computing all point orbits done, found " << Sch->nb_orbits << " orbits" << endl;
		}


	f = Sch->orbit_first[0];
	len = Sch->orbit_len[0];
	pt = Sch->orbit[f];

	if (f_v) {
		cout << "surface_classify::init orbit rep = " << pt << endl;
		}

	A->group_order(go);
	if (f_v) {
		cout << "surface_classify::init computing point stabilizer:" << endl;
		}
	Sch->point_stabilizer(A, go, 
		Stab, 0 /* orbit_no */, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_classify::init computing point stabilizer done:" << endl;
		}
	Stab->group_order(go);
	if (f_v) {
		cout << "surface_classify::init point stabilizer is a group of order " << go << endl;
		}

	if (f_v) {
		cout << "surface_classify::init computing strong generators for the point stabilizer:" << endl;
		}
	stab_gens = new strong_generators;
	stab_gens->init_from_sims(Stab, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_classify::init strong generators for the point stabilizer have been computed" << endl;
		}
#endif


	INT i, j;
	
	if (f_v) {
		cout << "surface_classify::init computing lines on the special point:" << endl;
		}
	INT *line_pencil_line_ranks;


	line_pencil_line_ranks = NEW_INT(O->nb_lines);
	O->lines_on_point_by_line_rank(pt, line_pencil_line_ranks, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_classify::init line_pencil_line_ranks=";
		INT_vec_print(cout, line_pencil_line_ranks, O->alpha);
		cout << endl;
		}

	Pts = NEW_INT(O->alpha * (q + 1));
	for (i = 0; i < O->alpha; i++) {
		O->points_on_line_by_line_rank(line_pencil_line_ranks[i], Pts + i * (q + 1), 0 /* verbose_level */);
		}

	if (f_v) {
		cout << "surface_classify::init Points collinear with pt " << pt << ":" << endl;
		INT_matrix_print(Pts, O->alpha, q + 1);
		}

	INT_vec_heapsort(Pts, O->alpha * (q + 1));
	if (f_v) {
		cout << "surface_classify::init after sorting:" << endl;
		INT_vec_print(cout, Pts, O->alpha * (q + 1));
		cout << endl;
		}

	j = 0;
	for (i = 0; i < O->alpha * (q + 1); i++) {
		if (Pts[i] != pt) {
			Pts[j++] = Pts[i];
			}
		}
	nb_pts = j;
	INT_vec_heapsort(Pts, nb_pts);
	if (f_v) {
		cout << "surface_classify::init after removing pt and sorting:" << endl;
		INT_vec_print(cout, Pts, nb_pts);
		cout << endl;
		cout << "surface_classify::init nb_pts=" << nb_pts << endl;
		}

	if (f_v) {
		cout << "surface_classify::init computing the restricted action on the neighbors:" << endl;
		}
	A_on_neighbors = new action;
	A_on_neighbors->induced_action_by_restriction(*A, 
		FALSE /* f_induce_action */, NULL, 
		nb_pts, Pts, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_classify::init restricted action on neighbors has been computed" << endl;
		}

	FREE_INT(line_pencil_line_ranks);

	if (f_v) {
		cout << "surface_classify::init done" << endl;
		}

}

INT surface_classify::surface_test(INT *S, INT len, INT verbose_level)
{
	INT i, x, y;
	INT f_OK = TRUE;
	INT fxy;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "surface_classify::surface_test" << endl;
		}
	if (f_vv) {
		for (i = 0; i < len; i++) {
			O->unrank_point(O->v1, 1, Pts[S[i]], 0);
			INT_vec_print(cout, u, 6);
			cout << endl;
			}
		}
	y = Pts[S[len - 1]];
	O->unrank_point(v, 1, y, 0);
	
	for (i = 0; i < len - 1; i++) {
		x = Pts[S[i]];
		O->unrank_point(u, 1, x, 0);

		fxy = O->evaluate_bilinear_form(u, v, 1);
		
		if (fxy == 0) {
			f_OK = FALSE;
			if (f_vv) {
				cout << "not OK; ";
				cout << "{x,y}={" << x << "," << y << "} are collinear" << endl;
				INT_vec_print(cout, u, 6);
				cout << endl;
				INT_vec_print(cout, v, 6);
				cout << endl;
				cout << "fxy=" << fxy << endl;
				}
			break;
			}
		}
	
	if (f_v) {
		if (!f_OK) {
			cout << "surface test fails" << endl;
			}
		}
	return f_OK;
}

void surface_classify::process_surfaces(INT nb_identify, 
	BYTE **Identify_label, 
	INT **Identify_coeff, 
	INT **Identify_monomial, 
	INT *Identify_length, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT iso_type, orb;

	if (f_v) {
		cout << "surface_classify::process_surfaces" << endl;
		}

	test_orbits(verbose_level);

	allocate_data();


	if (f_v) {
		cout << "before classify_surfaces" << endl;
		}
	classify_surfaces(verbose_level);
	if (f_v) {
		cout << "after classify_surfaces" << endl;
		}


	if (f_v) {
		cout << "The classification is complete, nb_iso=" << nb_iso << endl;
		}
	for (iso_type = 0; iso_type < nb_iso; iso_type++) {
		orb = Orb[iso_type];
		cout << "iso type " << iso_type << " is orbit " << orb << " ago=" << Ago[iso_type] << "=" << Ago0[iso_type] << " * " << Ago_nb_cosets[iso_type] << " nb_points=" << Nb_points[iso_type] << " nb_lines=" << Nb_lines[iso_type] << " nb of Eckardt points " << Nb_E[iso_type] << " nb_double_six=" << Nb_double_six[iso_type];
		cout << endl;
		cout << " equation: ";
		Surf->print_equation(cout, Coeffs[iso_type]);
		cout << endl;
		
		}
	cout << "We found " << nb_iso << " isomorphism types of surfaces" << endl;


	if (nb_identify) {
		INT **Label;
		INT *nb_Labels;
		INT w, i;
		
		identify(nb_identify, 
			Identify_label, 
			Identify_coeff, 
			Identify_monomial, 
			Identify_length, 
			Label, 
			nb_Labels, 
			verbose_level);
		
		for (iso_type = 0; iso_type < nb_iso; iso_type++) {
			cout << "iso_type " << iso_type << " / " << nb_iso << ":" << endl;
			if (nb_Labels[iso_type] == 0) {
				continue;
				}
			cout << "Iso type " << iso_type << " is isomorphic to ";
			for (i = 0; i < nb_Labels[iso_type]; i++) {
				w = Label[iso_type][i];
				cout << Identify_label[w] << " ";
				}
			cout << endl;
			}
		
		for (iso_type = 0; iso_type < nb_iso; iso_type++) {
			FREE_INT(Label[iso_type]);
			}
		FREE_PINT(Label);
		FREE_INT(nb_Labels);
		}
	
	free_data();

}

void surface_classify::test_orbits(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, b, c, r;
	INT S[5];
	INT S2[6];
	
	if (f_v) {
		cout << "surface_classify::test_orbits" << endl;
		}
	len = gen->nb_orbits_at_level(5);

	if (f_v) {
		cout << "surface_classify::test_orbits testing " << len << " orbits of 5-sets of lines:" << endl;
		}
	nb = 0;
	Idx = NEW_INT(len);
	for (i = 0; i < len; i++) {
		if (f_v) {
			cout << "orbit " << i << " / " << len << ":" << endl;
			}
		gen->get_set_by_level(5, i, S);
		for (j = 0; j < 5; j++) {
			a = S[j];
			b = Pts[a];
			c = Klein->Point_on_quadric_to_line[b];
			S2[j] = c;
			}
		S2[5] = Klein->Point_on_quadric_to_line[pt];
		r = Surf->compute_system_in_RREF(6, S2, 0 /*verbose_level*/);
		if (r == 19) {
			Idx[nb++] = i;
			}
		}

	if (f_v) {
		cout << "surface_classify::test_orbits we found " << nb << " / " << len << " orbits where the rank is 19" << endl;
		}
	if (f_v) {
		cout << "surface_classify::test_orbits done" << endl;
		}
}

void surface_classify::classify_surfaces(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT orb, iso_type;
	
	if (f_v) {
		cout << "surface_classify::classify_surfaces" << endl;
		}

	for (orb = 0; orb < nb; orb++) {
		is_isomorphic_to[orb] = -1;
		}

	iso_type = 0;

	for (orb = 0; orb < nb; orb++) {

		cout << "orb=" << orb << " is_isomorphic_to[orb]=" << is_isomorphic_to[orb] << endl;
		
		if (is_isomorphic_to[orb] >= 0) {
			if (f_v) {
				cout << "surface_classify::classify_surfaces orbit " << orb << " / " << nb << " is isomorphic to " << is_isomorphic_to[orb] << ", skipping" << endl;
				}
			continue;
			}

		if (!new_surface(orb, iso_type, verbose_level)) {
			cout << "Failed surface" << endl;
			continue;
			}

		iso_type++;		
		}
	nb_iso = iso_type;

	if (f_v) {
		cout << "surface_classify::classify_surfaces performing isomorphism testing nb_iso = " << nb_iso << endl;
		}

#if 0
	{
	classify C;

	C.init(Nb_points, nb, FALSE, 0);
	cout << "The distribution of the number of points on these surfaces is: ";
	C.print_naked(TRUE);
	cout << endl;
	}
#endif


	if (f_v) {
		cout << "surface_classify::classify_surfaces done" << endl;
		}
}

INT surface_classify::new_surface(INT orb, INT iso_type, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT idx;
	
	if (f_v) {
		cout << "surface_classify::new_surface" << endl;
		}
	

	if (f_v) {
		cout << "surface_classify::new_surface new isomorphism type " << iso_type << " is defined via orbit " << orb << " / " << nb << ":" << endl;
		}

	idx = Idx[orb];

	Orb[iso_type] = orb;
	is_isomorphic_to[orb] = iso_type;

	surface_data *D;

	D = new surface_data;

	cout << "surface_classify does not work any longer" << endl;
	exit(1);
#if 0
#if 0
	if (!D->init(this, orb, iso_type, verbose_level)) {
		cout << "this surface is bad, skipping" << endl;
		delete D;
		return FALSE;
		}
#endif

	cout << "surface with iso_type = " << iso_type << " has been defined" << endl;
	
	Nb_points[iso_type] = D->nb_points_on_surface;
	Nb_lines[iso_type] = D->nb_lines;
	Nb_E[iso_type] = D->nb_E;
	Nb_double_six[iso_type] = D->nb_double_sixes;
	Coeffs[iso_type] = NEW_INT(Surf->nb_monomials);
	INT_vec_copy(D->coeff, Coeffs[iso_type], Surf->nb_monomials);
	Ago0[iso_type] = D->ago0;
	Ago_nb_cosets[iso_type] = D->nb_aut;
	Ago[iso_type] = D->ago0 * D->nb_aut;

	The_surface[iso_type] = D;

	if (f_v) {
		cout << "surface_classify::new_surface done" << endl;
		}
	return TRUE;
#endif
}


void surface_classify::identify(INT nb_identify, 
	BYTE **Identify_label, 
	INT **Identify_coeff, 
	INT **Identify_monomial, 
	INT *Identify_length, 
	INT **&Label, 
	INT *&nb_Labels, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	INT iso_type, orb2, idx2;
	
	if (f_v) {
		cout << "surface_classify::identify" << endl;
		}


	cout << "Performing " << nb_identify << " identifications:" << endl;

	Label = NEW_PINT(nb_iso);
	nb_Labels = NEW_INT(nb_iso);
	INT_vec_zero(nb_Labels, nb_iso);
	for (iso_type = 0; iso_type < nb_iso; iso_type++) {
		Label[iso_type] = NEW_INT(nb_identify);
		}

	INT w;

	for (w = 0; w < nb_identify; w++) {

		cout << "identifying surface " << w << " / " << nb_identify << endl;
	
		INT *my_coeff;
		INT *my_Surface;
		INT nb_points_on_my_surface;
		INT *my_Lines;
		INT nb_my_lines;

		my_coeff = NEW_INT(Surf->nb_monomials);
		INT_vec_zero(my_coeff, Surf->nb_monomials);
		for (i = 0; i < Identify_length[w]; i++) {
			INT c, m;
			c = Identify_coeff[w][i];
			m = Identify_monomial[w][i];
			my_coeff[m] = c;
			}
		if (f_v) {
			cout << "identifying the surface ";
			INT_vec_print(cout, my_coeff, Surf->nb_monomials);
			cout << endl;
			}


		my_Surface = NEW_INT(Surf->P->N_points);
		my_Lines = NEW_INT(Surf->P->N_lines);

		Surf->enumerate_points(my_coeff, my_Surface, nb_points_on_my_surface, 0 /* verbose_level */);
		if (f_v) {
			cout << "The surface " << w << " to be identified has " << nb_points_on_my_surface << " points" << endl;
			}

		Surf->P->find_lines_which_are_contained(my_Surface, nb_points_on_my_surface, my_Lines, nb_my_lines, Surf->P->N_lines /* max_lines */, 0 /* verbose_level */);

		if (f_v) {
			cout << "The surface " << w << " has " << nb_my_lines << " lines" << endl;
			}
		if (nb_my_lines != 27 && nb_my_lines != 21) {
			cout << "something is wrong with the input surface, skipping" << endl;
			goto skip;
			}
	
		INT *my_Adj;
		INT j1, j2, a1, a2;

		my_Adj = NEW_INT(nb_my_lines * nb_my_lines);
		INT_vec_zero(my_Adj, nb_my_lines * nb_my_lines);
		for (j1 = 0; j1 < nb_my_lines; j1++) {
			a1 = my_Lines[j1];
			for (j2 = j1 + 1; j2 < nb_my_lines; j2++) {
				a2 = my_Lines[j2];
				if (!Surf->P->test_if_lines_are_disjoint_from_scratch(a1, a2)) {
					my_Adj[j1 * nb_my_lines + j2] = 1;
					my_Adj[j2 * nb_my_lines + j1] = 1;
					}
				}
			}
	

		set_of_sets *my_SoS;
		INT *my_Table;
		INT my_N;
	
		my_SoS = new set_of_sets;

		my_SoS->init_from_adjacency_matrix(nb_my_lines, my_Adj, 0 /* verbose_level */);

		make_table_of_double_sixes(my_Lines, nb_my_lines, 
			my_SoS, my_Table, my_N, verbose_level);

		INT subset[5];
		INT subset2[5];
		INT S3[6];
		INT K1[6];
		INT K2[6];
		INT K3[6];
		INT K4[6];
		INT h, l, nCk;
		INT pt1, pt1b, coset, f;

		if (my_N == 0) {
			cout << "my_N == 0" << endl;
			exit(1);
			}
		l = 0;
		i = my_Table[l * 2 + 0];
		j = my_Table[l * 2 + 1];
		nCk = INT_n_choose_k(my_SoS->Set_size[i], 5);
		unrank_k_subset(j, subset, my_SoS->Set_size[i], 5);
		for (h = 0; h < 5; h++) {
			subset2[h] = my_SoS->Sets[i][subset[h]];
			S3[h] = my_Lines[subset2[h]];
			}
		S3[5] = my_Lines[i];
		for (h = 0; h < 6; h++) {
			K1[h] = Klein->Line_to_point_on_quadric[S3[h]];
			}
		pt1 = K1[5];
		for (h = 0; h < 5; h++) {
			f = O->evaluate_bilinear_form_by_rank(K1[h], K1[5]);
			if (f) {
				cout << "K1[" << h << "] and K1[5] are not collinear" << endl;
				exit(1);
				}
			}
		if (f_v) {
			cout << "pt1 = " << pt1 << endl;
			}
		coset = Sch->orbit_inv[pt1];
		Sch->coset_rep(coset);
		// coset rep now in cosetrep
		A->element_invert(Sch->cosetrep, Elt0, 0);
		pt1b = A->element_image_of(pt1, Elt0, 0);
		if (pt1b != pt) {
			cout << "pt1b != pt" << endl;
			exit(1);
			}
		for (h = 0; h < 5; h++) {
			K2[h] = A->element_image_of(K1[h], Elt0, 0);
			}
		for (h = 0; h < 5; h++) {
			f = O->evaluate_bilinear_form_by_rank(K2[h], pt);
			if (f) {
				cout << "K2[" << h << "] and pt are not collinear" << endl;
				exit(1);
				}
			}
		for (h = 0; h < 5; h++) {
			if (!INT_vec_search(Pts, nb_pts, K2[h], K3[h])) {
				cout << "cannot find K2[h] in Pts[]" << endl;
				cout << "K2[h]=" << K2[h] << endl;
				exit(1);
				}
	
			}
		if (f_v) {
			cout << "down coset " << l << " / " << my_N << " tracing the set ";
			INT_vec_print(cout, K3, 5);
			cout << endl;
			}
		idx2 = gen->trace_set(K3, 5, 5, 
			K4, Elt1, 
			0 /* verbose_level */);

		if (!INT_vec_search(Idx, nb, idx2, orb2)) {
			cout << "cannot find orbit in Idx" << endl;
			exit(1);
			}
		iso_type = is_isomorphic_to[orb2];

		if (f_v) {
			cout << "The surface is isomorphic to surface " << iso_type << endl;
			}
		Label[iso_type][nb_Labels[iso_type]] = w;
		nb_Labels[iso_type]++;

		delete my_SoS;
		FREE_INT(my_Table);
		FREE_INT(my_Adj);

skip:
		FREE_INT(my_coeff);
		FREE_INT(my_Surface);
		FREE_INT(my_Lines);
		} // next w
	if (f_v) {
		cout << "surface_classify::identify done" << endl;
		}
}

void surface_classify::allocate_data()
{
	The_surface = new p_surface_data[nb];
	Nb_points = NEW_INT(nb);
	Nb_lines = NEW_INT(nb);
	Nb_double_six = NEW_INT(nb);
	Nb_E = NEW_INT(nb);
	is_isomorphic_to = NEW_INT(len);
	Ago0 = NEW_INT(nb);
	Ago_nb_cosets = NEW_INT(nb);
	Ago = NEW_INT(nb);
	Orb = NEW_INT(nb);
	Coeffs = NEW_PINT(nb);
	Elt0 = NEW_INT(A->elt_size_in_INT);
	Elt1 = NEW_INT(A->elt_size_in_INT);
}

void surface_classify::free_data()
{
	INT iso_type;
	
	
	FREE_INT(Nb_points);
	FREE_INT(Nb_lines);
	FREE_INT(Nb_double_six);
	FREE_INT(Nb_E);
	FREE_INT(is_isomorphic_to);
	FREE_INT(Ago0);
	FREE_INT(Ago_nb_cosets);
	FREE_INT(Ago);
	FREE_INT(Orb);
	for (iso_type = 0; iso_type < nb_iso; iso_type++) {
		FREE_INT(Coeffs[iso_type]);
		delete The_surface[iso_type];
		}
	delete [] The_surface;
	FREE_PINT(Coeffs);
	FREE_INT(Elt0);
	FREE_INT(Elt1);
}

void surface_classify::make_table_of_double_sixes(INT *Lines, INT nb_lines, 
	set_of_sets *SoS, INT *&Table, INT &N, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT subset[5];
	INT subset2[5];
	INT S3[6];
	INT N1, nCk, h;
	INT i, j, r;
	
	if (f_v) {
		cout << "surface_classify::make_table_of_double_sixes" << endl;
		}

	N = 0;
	for (i = 0; i < nb_lines; i++) {
		if (SoS->Set_size[i] < 5) {
			continue;
			}
		nCk = INT_n_choose_k(SoS->Set_size[i], 5);
		for (j = 0; j < nCk; j++) {
			unrank_k_subset(j, subset, SoS->Set_size[i], 5);
			for (h = 0; h < 5; h++) {
				subset2[h] = SoS->Sets[i][subset[h]];
				S3[h] = Lines[subset2[h]];
				}
			S3[5] = Lines[i];
			r = Surf->compute_system_in_RREF(6, S3, 0 /*verbose_level*/);
			if (r == 19) {
				N++;
				}
			}
		}
	if (f_v) {
		cout << "We found " << N << " single sixes on this surface" << endl;
		}
	Table = NEW_INT(N * 2);
	N1 = 0;
	for (i = 0; i < nb_lines; i++) {
		if (SoS->Set_size[i] < 5) {
			continue;
			}
		nCk = INT_n_choose_k(SoS->Set_size[i], 5);
		for (j = 0; j < nCk; j++) {
			unrank_k_subset(j, subset, SoS->Set_size[i], 5);
			for (h = 0; h < 5; h++) {
				subset2[h] = SoS->Sets[i][subset[h]];
				S3[h] = Lines[subset2[h]];
				}
			S3[5] = Lines[i];
			r = Surf->compute_system_in_RREF(6, S3, 0 /*verbose_level*/);
			if (r == 19) {
				Table[N1 * 2 + 0] = i;
				Table[N1 * 2 + 1] = j;
				N1++;
				}
			}
		}
	if (N1 != N) {
		cout << "N1 != N" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface_classify::make_table_of_double_sixes done" << endl;
		}
}
#endif



