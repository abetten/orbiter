// surface_data.C
// 
// Anton Betten
// August 30, 2016
//
// 
//
//

#include "orbiter.h"


surface_data::surface_data()
{
	null();
}

surface_data::~surface_data()
{
	freeself();
}

void surface_data::null()
{
	stab_gens = NULL;
	f_data_allocated = FALSE;
}

void surface_data::freeself()
{
	if (f_data_allocated) {
		free_data();
		}
	if (stab_gens) {
		delete stab_gens;
		}
	null();
}

void surface_data::allocate_data()
{
	f_data_allocated = TRUE;
	D = NEW_INT(6 * 8);
	coeff = NEW_INT(SC->Surf->nb_monomials);
	Surface = NEW_INT(SC->Surf->P->N_points);
	Lines = NEW_INT(27);
	Lines_wedge = NEW_INT(27);
	Lines_klein = NEW_INT(27);
	Elt0 = NEW_INT(A->elt_size_in_INT);
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Aut_cosets = new vector_ge;
	Aut_cosets->init(SC->A);
}

void surface_data::free_data()
{
	FREE_INT(D);
	FREE_INT(coeff);
	FREE_INT(Surface);
	FREE_INT(Lines);
	FREE_INT(Lines_wedge);
	FREE_INT(Lines_klein);
	FREE_INT(Elt0);
	FREE_INT(Elt1);
	FREE_INT(Elt2);

	delete IS;
	delete PStack;
	delete Aut_cosets;
}

INT surface_data::init(surface_classify_wedge *SC, INT orb, INT iso_type, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "surface_data::init" << endl;
		}

	surface_data::SC = SC;
	surface_data::orb = orb;
	surface_data::iso_type = iso_type;
	A = SC->A;
	F = SC->F;
	q = F->q;
	nb_monomials = SC->Surf->nb_monomials;

	//pt = SC->pt0_wedge;
	idx = SC->Classify_double_sixes->Idx[orb];



	allocate_data();



	SC->Classify_double_sixes->Five_plus_one->get_set_by_level(5, idx, S);
	INT_vec_apply(S, SC->Classify_double_sixes->Neighbor_to_line, S2, 5);
	S2[5] = SC->Classify_double_sixes->pt0_line;

	SC->Surf->unrank_lines(D, S2, 6);

	if (f_v) {
		cout << "surface_data::init single six a_1,a_2,a_3,a_4,a_5,b_0:" << endl;
		INT_matrix_print(D, 6, 8);
		}


	if (f_v) {
		cout << "surface_data::init the set of lines is: ";
		INT_vec_print(cout, S2, 6);
		cout << endl;
		}
	
	INT double_six[12];
	INT Line_coords[27 * 8];

	if (f_v) {
		cout << "surface_data::init before create_double_six_from_five_lines_with_a_common_transversal ";
		}
	if (SC->Surf->create_double_six_from_five_lines_with_a_common_transversal(S2, double_six, verbose_level)) {
		cout << "The starter configuration is good, a double six has been computed:" << endl;
		INT_matrix_print(double_six, 2, 6);
		}
	else {
		cout << "The starter configuration is bad, there is no double six" << endl;
		f_data_allocated = FALSE;
		return FALSE;
		}


	nb_lines = 27;
	INT_vec_copy(double_six, Lines, 12);
	SC->Surf->create_remaining_fifteen_lines(double_six, Lines + 12, 0 /* verbose_level */);

	SC->Surf->line_to_wedge_vec(Lines, Lines_wedge, 27);
	
	SC->Surf->line_to_klein_vec(Lines, Lines_klein, 27);

	SC->Surf->unrank_lines(Line_coords, Lines, 27);

	if (f_v) {
		cout << "surface_data::init the 27 lines are:" << endl;
		INT_matrix_print(Line_coords, 27, 8);
		//SC->Surf->print_lines_tex(cout, Lines);
		}

	{
	BYTE fname_csv[1000];
	
	sprintf(fname_csv, "surface_%ld_%ld_lines.csv", q, iso_type);
	

	SC->Surf->save_lines_in_three_kinds(fname_csv, Lines_wedge, Lines, Lines_klein, 27);
	}


	SC->Surf->build_cubic_surface_from_lines(6, S2, coeff, 0 /* verbose_level */);
	PG_element_normalize_from_front(*SC->F, coeff, 1, SC->Surf->nb_monomials);
	
	if (f_v) {
		cout << "surface_data::init the coefficient vector of the cubic surface " << orb << " / " << SC->Classify_double_sixes->nb << " is : ";
		INT_vec_print(cout, coeff, nb_monomials);
		cout << " = ";
		SC->Surf->print_equation(cout, coeff);
		cout << endl;
		}


	SC->Surf->enumerate_points(coeff, Surface, nb_points_on_surface, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_data::init the surface has " << nb_points_on_surface << " points" << endl;
		}

	INT_vec_heapsort(Surface, nb_points_on_surface);

#if 0
	if (f_v) {
		cout << "surface_data::init determining all lines on the surface:" << endl;
		}
	SC->Surf->P->find_lines_which_are_contained(Surface, nb_points_on_surface, 
		Lines, nb_lines, 27 /* max_lines */, 0 /* verbose_level */);

	if (f_v) {
		cout << "surface_data::init nb_lines = " << nb_lines << endl;
		}

	if (nb_lines == 21) {
		f_data_allocated = FALSE;
		return FALSE;
		}
	if (nb_lines != 21 && nb_lines != 27) {
		cout << "surface_data::init nb_lines = " << nb_lines << endl;
		exit(1);
		}
#endif	

	if (f_v) {
		cout << "surface_data::init the surface is isomorphism type = " << iso_type << endl;
		}


	if (f_v) {
		cout << "surface_data::init the surface contains " << nb_lines << " lines. They are : ";
		INT_vec_print(cout, Lines, nb_lines);
		cout << endl;
		}

	if (f_v) {
		cout << "surface_data::init computing the incidence structure induced on the points covered by lines" << endl;
		}

	//compute_points_on_lines(verbose_level);

	SC->Surf->compute_points_on_lines(Surface, nb_points_on_surface, 
		Lines, nb_lines, 
		pts_on_lines, 
		verbose_level);

	compute_decomposition_schemes(verbose_level);

	nb_E = pts_on_lines->number_of_eckardt_points(verbose_level);
	if (f_v) {
		cout << "surface_data::init the surface contains " << nb_E << " Eckardt points" << endl;
		}



	compute_adjacency_matrix_and_line_intersections(verbose_level);



	if (f_v) {
		cout << "surface_data::init computing the starter configurations" << endl;
		}
	SC->Surf->list_starter_configurations(Lines, 27 /*nb_lines*/, 
		line_intersections, Table, nb_starter_configurations, verbose_level);

	if (f_v) {
		cout << "surface_data::init computing the starter configurations done, we found " << nb_starter_configurations << " starter configurations" << endl;
		}


	if (f_v) {
		cout << "surface_data::init before automorphisms_and_isomorphisms" << endl;
		}
	//automorphisms_and_isomorphisms(0 /* verbose_level */);

	spreadsheet *Sp;

	automorphisms_and_isomorphisms_with_spreadsheet(Sp, 0 /* verbose_level */);

	delete Sp;
	if (f_v) {
		cout << "surface_data::init after automorphisms_and_isomorphisms" << endl;
		}

	
	//SC->gen->get_stabilizer_order(5, idx, Ago0_long);

	strong_generators *stab_gens0;
	sims *Stab;
	longinteger_object go0;

	if (f_v) {
		cout << "surface_data::init creating the stabilizer" << endl;
		}
	
	
	SC->Classify_double_sixes->Five_plus_one->get_stabilizer_generators(stab_gens0,  
		5, idx, 0 /* verbose_level */);



	if (f_v) {
		cout << "surface_data::init creating stabilizer of the configuration of 6 lines:" << endl;
		}
	Stab = stab_gens0->create_sims(verbose_level);

	Stab->group_order(go0);
	if (f_v) {
		cout << "surface_data::init go0=" << go0 << " orbit_len = " << nb_aut << endl;
		}
	
	delete stab_gens0;


	Stab->transitive_extension_using_coset_representatives(
			Aut_cosets->data, Aut_cosets->len, 
			0 /*verbose_level*/);

	nb_cosets = Aut_cosets->len;
	
	stab_gens = new strong_generators;
	stab_gens->init_from_sims(Stab, 0 /* verbose_level */);
	Stab->group_order(ago);
	if (f_v) {
		cout << "surface_data::init created stabilizer group of order ago=" << ago << endl;
		cout << "generators:" << endl;
		stab_gens->print_generators();
		}


	action *Ar;

	// A2 is the action on the wedge product, not on lines
	Ar = SC->A2->restricted_action(Lines_wedge, nb_lines, verbose_level);
	if (f_v) {
		cout << "surface_data::init created restricted action" << endl;
		cout << "generators are:" << endl;
		stab_gens->print_with_given_action(cout, Ar);
		}

	delete Ar;
	//delete stab_gens;
	delete Stab;

	if (f_v) {
		cout << "surface_data::init done" << endl;
		}
	return TRUE;
}


void surface_data::automorphisms_and_isomorphisms(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "surface_data::automorphisms_and_isomorphisms" << endl;
		}

	longinteger_object Ago0_long;
	
	SC->Classify_double_sixes->Five_plus_one->get_stabilizer_order(5, idx, Ago0_long);
	ago0 = Ago0_long.as_INT();



	INT S3[6];
	INT K1[6];
	INT W1[6];
	INT W2[6];
	INT W3[6];
	INT W4[6];
	INT l, f;
	INT orb2, idx2;
	INT line_idx, subset_idx, h;


	if (f_v) {
		cout << "surface_data::automorphisms_and_isomorphisms performing downstep for iso_type " << iso_type << endl;
		}

	nb_aut = 0;

	for (l = 0; l < nb_starter_configurations; l++) {
		if (f_v) {
			cout << "surface_data::automorphisms_and_isomorphisms performing isomorphism testing iso_type " << iso_type << " downstep case " << l << " / " << nb_starter_configurations << endl;
			}

		line_idx = Table[l * 2 + 0];
		subset_idx = Table[l * 2 + 1];

		SC->Surf->create_starter_configuration(line_idx, subset_idx, line_intersections, Lines, S3, 0 /* verbose_level */);

		if (f_vv) {
			cout << "line_idx=" << line_idx << " subset_idx=" << subset_idx << " line=" << S3[5] << " neighbors S3[]= ";
			INT_vec_print(cout, S3, 6);
			cout << endl;
			}


		INT_vec_apply(S3, SC->Surf->Klein->Line_to_point_on_quadric, K1, 6);

		for (h = 0; h < 5; h++) {
			f = SC->Surf->O->evaluate_bilinear_form_by_rank(K1[h], K1[5]);
			if (f) {
				cout << "surface_data::automorphisms_and_isomorphisms K1[" << h << "] and K1[5] are not collinear" << endl;
				exit(1);
				}
			}
		if (f_vv) {
			for (h = 0; h < 6; h++) {
				cout << "line S3[" << h << "]=" << S3[h] << " is generated by" << endl;
				SC->Surf->Gr->unrank_INT(S3[h], 0);
				INT_matrix_print(SC->Surf->Gr->M, 2, 4);
				}
			}

		SC->Surf->line_to_wedge_vec(S3, W1, 6);

		if (f_vv) {
			cout << "W1: ";
			INT_vec_print(cout, W1, 6);
			cout << endl;
			}


		A->make_element_which_moves_a_line_in_PG3q(SC->Surf->Gr, S3[5], Elt0, 0 /* verbose_level */);

		if (f_vv) {
			cout << "line " << S3[5] << ":" << endl;
			SC->Surf->Gr->unrank_INT(S3[5], 0);
			INT_matrix_print(SC->Surf->Gr->M, 2, 4);
			cout << "Elt0=" << endl;
			A->element_print_quick(Elt0, cout);
			}

		SC->A2->map_a_set(W1, W2, 6, Elt0, 0 /* verbose_level */);

		if (f_v) {
			cout << "W2: ";
			INT_vec_print(cout, W2, 6);
			cout << endl;
			}

		INT_vec_search_vec(SC->Classify_double_sixes->Neighbors, SC->Classify_double_sixes->nb_neighbors, W2, 5, W3);




		if (f_vv) {
			cout << "surface_data::automorphisms_and_isomorphisms down coset " << l << " / " << nb_starter_configurations << " tracing the set ";
			INT_vec_print(cout, W3, 5);
			cout << endl;
			}
		idx2 = SC->Classify_double_sixes->Five_plus_one->trace_set(W3, 5, 5, W4, Elt1, 0 /* verbose_level */);

		if (!INT_vec_search(SC->Classify_double_sixes->Idx, SC->Classify_double_sixes->nb, idx2, orb2)) {
			cout << "surface_data::automorphisms_and_isomorphisms cannot find orbit in Idx" << endl;
			exit(1);
			}
		A->element_mult(Elt0, Elt1, Elt2, 0);
		if (orb2 == orb) {
			if (f_v) {
				cout << "surface_data::automorphisms_and_isomorphisms down coset " << l << " / " << nb_starter_configurations << " found an automorphism" << endl;
				}
			if (f_v) {
				A->element_print_quick(Elt2, cout);
				}
			store_automorphism_coset(Elt2, verbose_level);
			nb_aut++;
			}
		else {
			if (f_v) {
				cout << "surface_data::automorphisms_and_isomorphisms down coset " << l << " / " << nb_starter_configurations << " orbit " << orb << " is isomorphic to " << orb2 << endl;
				}
			SC->is_isomorphic_to[orb2] = iso_type;
			SC->store_isomorphism(Elt2, orb2, verbose_level);

			//cout << "SC->is_isomorphic_to[" << orb2 << "] = " << iso_type << endl;
			}
		}


	if (f_v) {
		cout << "surface_data::automorphisms_and_isomorphisms done" << endl;
		}
}

void surface_data::automorphisms_and_isomorphisms_with_spreadsheet(spreadsheet *&Sp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "surface_data::automorphisms_and_isomorphisms_with_spreadsheet" << endl;
		}

	Sp = new spreadsheet;
	
	longinteger_object Ago0_long;
	
	SC->Classify_double_sixes->Five_plus_one->get_stabilizer_order(5, idx, Ago0_long);
	ago0 = Ago0_long.as_INT();



	INT S3[6];
	INT K1[6];
	INT W1[6];
	INT W2[6];
	INT W3[6];
	INT W4[6];
	INT l, f;
	INT orb2, idx2;
	INT line_idx, subset_idx, h;

	BYTE str[1000];
	INT *Line_idx;
	INT *Subset_idx;
	INT *Orb2;
	BYTE **Text_S3;
	BYTE **Text_K1;
	BYTE **Text_W1;
	BYTE **Text_W2;
	BYTE **Text_W3;
	BYTE **Text_W4;
	BYTE **Text_Elt0;
	BYTE **Text_Elt1;
	BYTE **Text_Elt2;
	INT *Elt0_data;
	INT *Elt1_data;
	INT *Elt2_data;


	Elt0_data = NEW_INT(A->make_element_size);
	Elt1_data = NEW_INT(A->make_element_size);
	Elt2_data = NEW_INT(A->make_element_size);
	Line_idx = NEW_INT(nb_starter_configurations);
	Subset_idx = NEW_INT(nb_starter_configurations);
	Text_S3 = NEW_PBYTE(nb_starter_configurations);
	Text_K1 = NEW_PBYTE(nb_starter_configurations);
	Text_W1 = NEW_PBYTE(nb_starter_configurations);
	Text_W2 = NEW_PBYTE(nb_starter_configurations);
	Text_W3 = NEW_PBYTE(nb_starter_configurations);
	Text_W4 = NEW_PBYTE(nb_starter_configurations);
	Text_Elt0 = NEW_PBYTE(nb_starter_configurations);
	Text_Elt1 = NEW_PBYTE(nb_starter_configurations);
	Text_Elt2 = NEW_PBYTE(nb_starter_configurations);
	Orb2 = NEW_INT(nb_starter_configurations);

	if (f_v) {
		cout << "surface_data::automorphisms_and_isomorphisms_with_spreadsheet performing downstep for iso_type " << iso_type << endl;
		}

	nb_aut = 0;

	for (l = 0; l < nb_starter_configurations; l++) {
		if (f_v) {
			cout << "surface_data::automorphisms_and_isomorphisms_with_spreadsheet performing isomorphism testing iso_type " << iso_type << " downstep case " << l << " / " << nb_starter_configurations << endl;
			}

		line_idx = Table[l * 2 + 0];
		subset_idx = Table[l * 2 + 1];
		Line_idx[l] = line_idx;
		Subset_idx[l] = subset_idx;

		SC->Surf->create_starter_configuration(line_idx, subset_idx, line_intersections, Lines, S3, 0 /* verbose_level */);

		if (f_vv) {
			cout << "line_idx=" << line_idx << " subset_idx=" << subset_idx << " line=" << S3[5] << " neighbors S3[]= ";
			INT_vec_print(cout, S3, 6);
			cout << endl;
			}

		INT_vec_print_to_str(str, S3, 6);
		Text_S3[l] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_S3[l], str);


		INT_vec_apply(S3, SC->Surf->Klein->Line_to_point_on_quadric, K1, 6);

		INT_vec_print_to_str(str, K1, 6);
		Text_K1[l] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_K1[l], str);

		for (h = 0; h < 5; h++) {
			f = SC->Surf->O->evaluate_bilinear_form_by_rank(K1[h], K1[5]);
			if (f) {
				cout << "surface_data::automorphisms_and_isomorphisms_with_spreadsheet K1[" << h << "] and K1[5] are not collinear" << endl;
				exit(1);
				}
			}
		if (f_vv) {
			for (h = 0; h < 6; h++) {
				cout << "line S3[" << h << "]=" << S3[h] << " is generated by" << endl;
				SC->Surf->Gr->unrank_INT(S3[h], 0);
				INT_matrix_print(SC->Surf->Gr->M, 2, 4);
				}
			}

		SC->Surf->line_to_wedge_vec(S3, W1, 6);

		if (f_vv) {
			cout << "W1: ";
			INT_vec_print(cout, W1, 6);
			cout << endl;
			}

		INT_vec_print_to_str(str, W1, 6);
		Text_W1[l] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_W1[l], str);



		A->make_element_which_moves_a_line_in_PG3q(SC->Surf->Gr, S3[5], Elt0, 0 /* verbose_level */);

		if (f_vv) {
			cout << "line " << S3[5] << ":" << endl;
			SC->Surf->Gr->unrank_INT(S3[5], 0);
			INT_matrix_print(SC->Surf->Gr->M, 2, 4);
			cout << "Elt0=" << endl;
			A->element_print_quick(Elt0, cout);
			}

		SC->A2->map_a_set(W1, W2, 6, Elt0, 0 /* verbose_level */);

		if (f_v) {
			cout << "W2: ";
			INT_vec_print(cout, W2, 6);
			cout << endl;
			}

		INT_vec_print_to_str(str, W2, 6);
		Text_W2[l] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_W2[l], str);


		INT_vec_search_vec(SC->Classify_double_sixes->Neighbors, SC->Classify_double_sixes->nb_neighbors, W2, 5, W3);


		INT_vec_print_to_str(str, W3, 5);
		Text_W3[l] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_W3[l], str);


		if (f_vv) {
			cout << "surface_data::automorphisms_and_isomorphisms_with_spreadsheet down coset " << l << " / " << nb_starter_configurations << " tracing the set ";
			INT_vec_print(cout, W3, 5);
			cout << endl;
			}
		idx2 = SC->Classify_double_sixes->Five_plus_one->trace_set(W3, 5, 5, W4, Elt1, 0 /* verbose_level */);

		INT_vec_print_to_str(str, W4, 5);
		Text_W4[l] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_W4[l], str);


		if (!INT_vec_search(SC->Classify_double_sixes->Idx, SC->Classify_double_sixes->nb, idx2, orb2)) {
			cout << "surface_data::automorphisms_and_isomorphisms_with_spreadsheet cannot find orbit in Idx" << endl;
			exit(1);
			}

		Orb2[l] = orb2;

		
		A->element_mult(Elt0, Elt1, Elt2, 0);

		A->element_code_for_make_element(Elt0, Elt0_data);
		INT_vec_print_to_str(str, Elt0_data, A->make_element_size);
		Text_Elt0[l] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_Elt0[l], str);
		A->element_code_for_make_element(Elt1, Elt1_data);
		INT_vec_print_to_str(str, Elt1_data, A->make_element_size);
		Text_Elt1[l] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_Elt1[l], str);
		A->element_code_for_make_element(Elt2, Elt2_data);
		INT_vec_print_to_str(str, Elt2_data, A->make_element_size);
		Text_Elt2[l] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_Elt2[l], str);


		if (orb2 == orb) {
			if (f_v) {
				cout << "surface_data::automorphisms_and_isomorphisms_with_spreadsheet down coset " << l << " / " << nb_starter_configurations << " found an automorphism" << endl;
				}
			if (f_v) {
				A->element_print_quick(Elt2, cout);
				}
			store_automorphism_coset(Elt2, verbose_level);
			nb_aut++;
			}
		else {
			if (f_v) {
				cout << "surface_data::automorphisms_and_isomorphisms_with_spreadsheet down coset " << l << " / " << nb_starter_configurations << " orbit " << orb << " is isomorphic to " << orb2 << endl;
				}
			SC->is_isomorphic_to[orb2] = iso_type;
			SC->store_isomorphism(Elt2, orb2, verbose_level);

			//cout << "SC->is_isomorphic_to[" << orb2 << "] = " << iso_type << endl;
			}
		}

	BYTE fname_csv[1000];
	sprintf(fname_csv, "downstep_%ld_%ld.csv", q, iso_type);
	
	Sp->init_empty_table(nb_starter_configurations + 1, 13);
	Sp->fill_column_with_row_index(0, "Down_coset");
	Sp->fill_column_with_INT(1, Line_idx, "Line_idx");
	Sp->fill_column_with_INT(2, Subset_idx, "Subset_idx");
	Sp->fill_column_with_text(3, (const BYTE **) Text_S3, "S3");
	Sp->fill_column_with_text(4, (const BYTE **) Text_K1, "K1");
	Sp->fill_column_with_text(5, (const BYTE **) Text_W1, "W1");
	Sp->fill_column_with_text(6, (const BYTE **) Text_W2, "W2");
	Sp->fill_column_with_text(7, (const BYTE **) Text_W3, "N1");
	Sp->fill_column_with_text(8, (const BYTE **) Text_W4, "N2");
	Sp->fill_column_with_INT(9, Orb2, "Orb");
	Sp->fill_column_with_text(10, (const BYTE **) Text_Elt0, "Elt0");
	Sp->fill_column_with_text(11, (const BYTE **) Text_Elt1, "Elt1");
	Sp->fill_column_with_text(12, (const BYTE **) Text_Elt2, "Elt2");
	//Sp->fill_column_with_INT(3, Stab_order, "Stab_order");
	//Sp->fill_column_with_INT(4, Len, "Orbit_length");
	cout << "before Sp->save " << fname_csv << endl;
	Sp->save(fname_csv, verbose_level);
	cout << "after Sp->save " << fname_csv << endl;


	FREE_INT(Elt0_data);
	FREE_INT(Elt1_data);
	FREE_INT(Elt2_data);
	FREE_INT(Line_idx);
	FREE_INT(Subset_idx);
	for (l = 0; l < nb_starter_configurations; l++) {
		FREE_BYTE(Text_S3[l]);
		}
	FREE_PBYTE(Text_S3);
	for (l = 0; l < nb_starter_configurations; l++) {
		FREE_BYTE(Text_K1[l]);
		}
	FREE_PBYTE(Text_K1);
	for (l = 0; l < nb_starter_configurations; l++) {
		FREE_BYTE(Text_W1[l]);
		}
	FREE_PBYTE(Text_W1);
	for (l = 0; l < nb_starter_configurations; l++) {
		FREE_BYTE(Text_W2[l]);
		}
	FREE_PBYTE(Text_W2);
	for (l = 0; l < nb_starter_configurations; l++) {
		FREE_BYTE(Text_W3[l]);
		}
	FREE_PBYTE(Text_W3);
	for (l = 0; l < nb_starter_configurations; l++) {
		FREE_BYTE(Text_W4[l]);
		}
	FREE_PBYTE(Text_W4);
	for (l = 0; l < nb_starter_configurations; l++) {
		FREE_BYTE(Text_Elt0[l]);
		}
	FREE_PBYTE(Text_Elt0);
	for (l = 0; l < nb_starter_configurations; l++) {
		FREE_BYTE(Text_Elt1[l]);
		}
	FREE_PBYTE(Text_Elt1);
	for (l = 0; l < nb_starter_configurations; l++) {
		FREE_BYTE(Text_Elt2[l]);
		}
	FREE_PBYTE(Text_Elt2);
	FREE_INT(Orb2);

	if (f_v) {
		cout << "surface_data::automorphisms_and_isomorphisms_with_spreadsheet done" << endl;
		}
}

void surface_data::compute_decomposition_schemes(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_data::compute_decomposition_schemes" << endl;
		}

	IS = new incidence_structure;

	IS->init_by_set_of_sets(pts_on_lines, FALSE);

	PStack = new partitionstack;
	PStack->allocate(nb_lines + nb_points_on_surface, 0 /* verbose_level */);
	PStack->subset_continguous(nb_lines, nb_points_on_surface);
	PStack->split_cell(0 /* verbose_level */);
	
	IS->compute_TDO_safe(*PStack, 1 /*nb_lines + nb_points_on_surface*/ /* depth */, 0 /* verbose_level */);


	{
	IS->get_and_print_row_tactical_decomposition_scheme_tex(
		cout /*fp_row_scheme */, FALSE /* f_enter_math */, TRUE /* f_print_subscripts */, *PStack);
	IS->get_and_print_column_tactical_decomposition_scheme_tex(
		cout /*fp_col_scheme */, FALSE /* f_enter_math */, TRUE /* f_print_subscripts */, *PStack);
	}

	if (f_v) {
		cout << "surface_data::compute_decomposition_schemes done" << endl;
		}
}

void surface_data::compute_adjacency_matrix_and_line_intersections(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_data::compute_adjacency_matrix_and_line_intersections" << endl;
		}

	SC->Surf->compute_adjacency_matrix_of_line_intersection_graph(Adj, Lines, nb_lines, verbose_level);

#if 0
	cout << "Adjacency matrix:" << endl;
	INT_matrix_print(Adj, nb_lines, nb_lines);
	cout << endl;
#endif
	
	line_intersections = new set_of_sets;

	line_intersections->init_from_adjacency_matrix(nb_lines, Adj, 0 /* verbose_level */);

	if (f_v) {
		cout << "surface_data::compute_adjacency_matrix_and_line_intersections done" << endl;
		}
}

void surface_data::print(INT iso_type)
{
	cout << "iso type " << iso_type << " is orbit " << orb << " ago=" << ago << " = " << ago0 << " * " << nb_cosets << " nb_points=" << nb_points_on_surface << " nb_lines=" << nb_lines << " nb of Eckardt points " << nb_E;
	cout << endl;
	cout << " equation: ";
	SC->Surf->print_equation(cout, coeff);
#if 0
	cout << " set of points: ";
	INT_vec_print(cout, Surface, nb_points_on_surface);
#endif
	cout << endl;
	cout << "Generators for the automorphism group:" << endl;
	stab_gens->print_generators();
	
}

void surface_data::store_automorphism_coset(INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_data::store_automorphism_coset" << endl;
		}
	Aut_cosets->append(Elt);
	if (f_v) {
		cout << "surface_data::store_automorphism_coset done" << endl;
		}
}


