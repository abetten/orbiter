// projective_space_with_action.C
// 
// Anton Betten
//
// December 22, 2017
//
//
// 
//
//

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"


projective_space_with_action::projective_space_with_action()
{
	null();
}

projective_space_with_action::~projective_space_with_action()
{
	freeself();
}

void projective_space_with_action::null()
{
	q = 0;
	F = NULL;
	P = NULL;
	A = NULL;
	A_on_lines = NULL;
	S = NULL;
	Elt1 = NULL;
}

void projective_space_with_action::freeself()
{
	if (P) {
		FREE_OBJECT(P);
		}
	if (A) {
		FREE_OBJECT(A);
		}
	if (A_on_lines) {
		FREE_OBJECT(A_on_lines);
		}
	if (S) {
		FREE_OBJECT(S);
		}
	if (Elt1) {
		FREE_INT(Elt1);
		}
	null();
}

void projective_space_with_action::init(
	finite_field *F, INT n, INT f_semilinear,
	INT f_init_incidence_structure, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::init" << endl;
		}
	projective_space_with_action::f_init_incidence_structure
		= f_init_incidence_structure;
	projective_space_with_action::n = n;
	d = n + 1;
	projective_space_with_action::F = F;
	q = F->q;
	projective_space_with_action::f_semilinear = f_semilinear;
	
	P = NEW_OBJECT(projective_space);
	P->init(n, F, 
		f_init_incidence_structure, 
		verbose_level);
	
	init_group(f_semilinear, verbose_level);
	
	Elt1 = NEW_INT(A->elt_size_in_INT);


	if (f_v) {
		cout << "projective_space_with_action::init done" << endl;
		}
}

void projective_space_with_action::init_group(
		INT f_semilinear, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "projective_space_with_action::init_group" << endl;
		}
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating linear group" << endl;
		}
	create_linear_group(S, A, 
		F, d, 
		TRUE /*f_projective*/,
		FALSE /* f_general*/,
		FALSE /* f_affine */,
		f_semilinear,
		FALSE /* f_special */,
		0 /* verbose_level*/);
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating linear group done" << endl;
		}


	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating action on lines" << endl;
		}
	A_on_lines = A->induced_action_on_grassmannian(2, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating action on lines done" << endl;
		}


	if (f_v) {
		cout << "projective_space_with_action::init_group done" << endl;
		}
}

strong_generators *projective_space_with_action::set_stabilizer(
	INT *set, INT set_size, INT &canonical_pt, 
	INT *canonical_set_or_NULL, 
	INT f_save_incma_in_and_out,
	const char *save_incma_in_and_out_prefix,
	INT f_compute_canonical_form,
	uchar *&canonical_form,
	INT &canonical_form_len,
	INT verbose_level)
// December 22, 2017, based on earlier work:
// added to action_global.C on 2/28/2011, called from analyze.C
// November 17, 2014 moved here from TOP_LEVEL/extra.C
// December 31, 2014, moved here from projective_space.C
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);

	action *A_linear;
	INT *Incma;
	INT *partition;
	INT *labeling;
	INT nb_rows, nb_cols;
	INT *Aut, Aut_counter;
	INT *Base, Base_length;
	INT *Transversal_length, Ago;
	INT N, i, j, h, a, L;

	A_linear = A;

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		cout << "set_size = " << set_size << endl;
		}
	if (P->incidence_bitvec == NULL) {
		cout << "projective_space_with_action::set_stabilizer P->incidence_bitvec == NULL" << endl;
		exit(1);
		}

	if (f_vv) {
		cout << "computing the type of the set" << endl;
		}

	classify C;

	C.init(set, set_size, TRUE, 0);
	if (C.second_nb_types > 1) {
		cout << "projective_space_with_action::set_stabilizer: The set is a multiset:" << endl;
		C.print(FALSE /*f_backwards*/);
		}

	if (f_vv) {
		cout << "The type of the set is:" << endl;
		C.print(FALSE /*f_backwards*/);
		cout << "C.second_nb_types = " << C.second_nb_types << endl;
		}
	if (f_vv) {
		cout << "allocating data" << endl;
		}
	nb_rows = P->N_points + 1;
	nb_cols = P->N_lines + C.second_nb_types;
	Incma = NEW_INT(nb_rows * nb_cols);
	partition = NEW_INT(nb_rows + nb_cols);
	labeling = NEW_INT(nb_rows + nb_cols);

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer Initializing Incma" << endl;
		}

	for (i = 0; i < P->N_points; i++) {
		for (j = 0; j < P->N_lines; j++) {
			Incma[i * nb_cols + j] = P->is_incident(i, j);
			}
		}
	// last columns, make zero:
	for (j = 0; j < C.second_nb_types; j++) {
		for (i = 0; i < P->N_points; i++) {
			Incma[i * nb_cols + P->N_lines + j] = 0;
			}
		}
	
	// last row, make zero:
	for (j = 0; j < nb_cols; j++) {
		Incma[P->N_points * nb_cols + j] = 0;
		}

	// last columns:
	for (j = 0; j < C.second_nb_types; j++) {
		INT f2, l2, m, idx, f, l;

		f2 = C.second_type_first[j];
		l2 = C.second_type_len[j];
		m = C.second_data_sorted[f2 + 0];
		if (f_vvv) {
			cout << "j=" << j << " f2=" << f2
					<< " l2=" << l2 << " multiplicity=" << m << endl;
			}
		for (h = 0; h < l2; h++) {
			idx = C.second_sorting_perm_inv[f2 + h];
			f = C.type_first[idx];
			l = C.type_len[idx];
			i = C.data_sorted[f + 0];
			if (f_vvv) {
				cout << "h=" << h << " idx=" << idx
						<< " f=" << f << " l=" << l << " i=" << i << endl;
				}
			Incma[i * nb_cols + P->N_lines + j] = 1;
			}	
#if 0
		for (h = 0; h < set_size; h++) {
			i = set[h];
			Incma[i * nb_cols + N_lines + j] = 1;
			}
#endif
		}
	// bottom right entries:
	for (j = 0; j < C.second_nb_types; j++) {
		Incma[P->N_points * nb_cols + P->N_lines + j] = 1;
		}

	if (f_save_incma_in_and_out) {
		cout << "projective_space_with_action::set_stabilizer "
				"Incma:" << endl;
		if (nb_rows < 10) {
			print_integer_matrix_width(cout,
					Incma, nb_rows, nb_cols, nb_cols, 1);
			}
		else {
			cout << "too large to print" << endl;
			}

		char fname_csv[1000];
		char fname_bin[1000];

		sprintf(fname_csv, "%sIncma_in_%ld_%ld.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_bin, "%sIncma_in_%ld_%ld.bin",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		INT_matrix_write_csv(fname_csv, Incma, nb_rows, nb_cols);

		for (i = 0; i < nb_rows + nb_cols; i++) {
			labeling[i] = i;
			}

		colored_graph *CG;
		create_Levi_graph_from_incidence_matrix(CG,
				Incma, nb_rows, nb_cols, TRUE, labeling, verbose_level);
		CG->save(fname_bin, verbose_level);
		//FREE_INT(Incma);
		delete CG;
		}

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer "
				"initializing partition" << endl;
		}
	N = nb_rows + nb_cols;
	L = nb_rows * nb_cols;
	for (i = 0; i < N; i++) {
		partition[i] = 1;
		}
	partition[P->N_points - 1] = 0;
	partition[P->N_points] = 0;
	partition[nb_rows + P->N_lines - 1] = 0;
	for (j = 0; j < C.second_nb_types; j++) {
		partition[nb_rows + P->N_lines + j] = 0;
		}
	if (f_vvv) {
		cout << "projective_space_with_action::set_stabilizer "
				"partition:" << endl;
		for (i = 0; i < N; i++) {
			//cout << i << " : " << partition[i] << endl;
			cout << partition[i];
			}
		cout << endl;
		}

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer "
				"initializing Aut, Base, Transversal_length" << endl;
		}
	Aut = NEW_INT(N * N);
	Base = NEW_INT(N);
	Transversal_length = NEW_INT(N);
	
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer "
				"calling nauty_interface_matrix_INT" << endl;
		}
	nauty_interface_matrix_INT(Incma, nb_rows, nb_cols, 
		labeling, partition, 
		Aut, Aut_counter, 
		Base, Base_length, 
		Transversal_length, Ago, verbose_level - 3);
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer "
				"done with nauty_interface_matrix_INT, Ago=" << Ago << endl;
		}

	INT *Incma_out;
	INT ii, jj;
	if (f_vvv) {
		cout << "projective_space_with_action::set_stabilizer "
				"labeling:" << endl;
		INT_vec_print(cout, labeling, nb_rows + nb_cols);
		cout << endl;
		}

	Incma_out = NEW_INT(L);
	for (i = 0; i < nb_rows; i++) {
		ii = labeling[i];
		for (j = 0; j < nb_cols; j++) {
			jj = labeling[nb_rows + j] - nb_rows;
			//cout << "i=" << i << " j=" << j
			//<< " ii=" << ii << " jj=" << jj << endl;
			Incma_out[i * nb_cols + j] = Incma[ii * nb_cols + jj];
			}
		}
	if (f_vvv) {
		cout << "projective_space_with_action::set_stabilizer "
				"Incma Out:" << endl;
		if (nb_rows < 20) {
			print_integer_matrix_width(cout,
					Incma_out, nb_rows, nb_cols, nb_cols, 1);
			}
		else {
			cout << "projective_space_with_action::set_stabilizer "
					"too large to print" << endl;
			}
		}

	canonical_pt = -1;
	if (C.second_nb_types == 1) {
		for (i = 0; i < P->N_points; i++) {
			if (Incma[i * nb_cols + P->N_lines + 0] == 1) {
				ii = labeling[i];
				canonical_pt = ii;
				break;
				}
			}
		}
	else {
		// cannot compute the canonical point
		}


	if (f_compute_canonical_form) {
		

		canonical_form = bitvector_allocate_and_coded_length(
				L, canonical_form_len);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (Incma_out[i * nb_cols + j]) {
					a = i * nb_cols + j;
					bitvector_set_bit(canonical_form, a);
					}
				}
			}

		}

	if (canonical_set_or_NULL) {
		h = 0;
		for (i = 0; i < P->N_points; i++) {
			if (Incma[labeling[i] * nb_cols + P->N_lines + 0] == 1) {
				canonical_set_or_NULL[h++] = i;
				}
			}
		if (h != set_size) {
			cout << "projective_space_with_action::set_stabilizer "
					"h != set_size" << endl;
			exit(1);
			}
		}

	if (f_save_incma_in_and_out) {
		char fname_labeling[1000];
		char fname_csv[1000];
		char fname_bin[1000];

		sprintf(fname_labeling, "%slabeling_%ld_%ld.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_csv, "%sIncma_out_%ld_%ld.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_bin, "%sIncma_out_%ld_%ld.bin",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		
		INT_vec_write_csv(labeling, N, fname_labeling, "canonical labeling");
		INT_matrix_write_csv(fname_csv, Incma_out, nb_rows, nb_cols);

		
		colored_graph *CG;
		create_Levi_graph_from_incidence_matrix(CG,
				Incma_out, nb_rows, nb_cols, TRUE, labeling, verbose_level);
		CG->save(fname_bin, verbose_level);
		FREE_OBJECT(CG);
		}

	FREE_INT(Incma_out);

	action *A_perm;
	longinteger_object ago;


	A_perm = NEW_OBJECT(action);

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer "
				"before init_permutation_group_from_generators" << endl;
		}
	ago.create(Ago);
	A_perm->init_permutation_group_from_generators(N, 
		TRUE, ago, 
		Aut_counter, Aut, 
		Base_length, Base,
		0 /*verbose_level - 2 */);

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer "
				"create_automorphism_group_of_incidence_structure: "
				"created action ";
		A_perm->print_info();
		cout << endl;
		}

	//action *A_linear;

	//A_linear = A;

	if (A_linear == NULL) {
		cout << "projective_space_with_action::set_stabilizer "
				"A_linear == NULL" << endl;
		exit(1);
		}

	vector_ge *gens; // permutations from nauty
	vector_ge *gens1; // matrices
	INT g, frobenius, pos;
	INT *Mtx;
	
	gens = A_perm->Strong_gens->gens;

	gens1 = NEW_OBJECT(vector_ge);
	gens1->init(A_linear);
	gens1->allocate(gens->len);
	
	Mtx = NEW_INT(d * d + 1); // leave space for frobenius
	
	pos = 0;
	for (g = 0; g < gens->len; g++) {
		if (f_vv) {
			cout << "projective_space_with_action::set_stabilizer "
					"strong generator " << g << ":" << endl;
			A_perm->element_print(gens->ith(g), cout);
			cout << endl;
			}
		
		if (reverse_engineer_semilinear_map(A_perm, P, 
			gens->ith(g), Mtx, frobenius, 
			0 /*verbose_level - 2*/)) {

			Mtx[d * d] = frobenius;
			A_linear->make_element(Elt1, Mtx, 0 /*verbose_level - 2*/);
			if (f_vv) {
				cout << "projective_space_with_action::set_stabilizer "
						"semi-linear group element:" << endl;
				A_linear->element_print(Elt1, cout);
				}
			A_linear->element_move(Elt1, gens1->ith(pos), 0);
		

			pos++;
			}
		else {
			if (f_vv) {
				cout << "projective_space_with_action::set_stabilizer "
						"generator " << g << " does not correspond "
								"to a semilinear mapping" << endl;
				}
			}
		}
	gens1->reallocate(pos);
	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer "
				"we found " << gens1->len << " generators" << endl;
		}

	if (f_vvv) {
		gens1->print(cout);
		}
	

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer "
				"we are now testing the generators:" << endl;
		}
	INT j1, j2;
	
	for (g = 0; g < gens1->len; g++) {
		if (f_vv) {
			cout << "generator " << g << ":" << endl;
			}
		//A_linear->element_print(gens1->ith(g), cout);
		for (i = 0; i < P->N_points; i++) {
			j1 = A_linear->element_image_of(i, gens1->ith(g), 0);
			j2 = A_perm->element_image_of(i, gens->ith(g), 0);
			if (j1 != j2) {
				cout << "projective_space_with_action::set_stabilizer "
						"problem with generator: j1 != j2" << endl;
				cout << "i=" << i << endl;
				cout << "j1=" << j1 << endl;
				cout << "j2=" << j2 << endl;
				cout << endl;
				exit(1);
				}
			}
		}
	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer "
				"the generators are OK" << endl;
		}



	sims *S;
	longinteger_object go;

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer "
				"we are now creating the group" << endl;
		}

	S = create_sims_from_generators_with_target_group_order(A_linear, 
		gens1, ago, 0 /*verbose_level*/);
#if 0
	S = create_sims_from_generators_without_target_group_order(A_linear, 
		gens1, 0 /*verbose_level - 4*/);
#endif
	
	S->group_order(go);

	
	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer "
				"Found a group of order " << go << endl;
		}
	if (f_vvv) {
		cout << "projective_space_with_action::set_stabilizer "
				"strong generators are:" << endl;
		S->print_generators();
		cout << "projective_space_with_action::set_stabilizer "
				"strong generators are (in tex):" << endl;
		S->print_generators_tex(cout);
		}


	longinteger_domain D;
	
	if (D.compare_unsigned(ago, go)) {
		cout << "projective_space_with_action::set_stabilizer "
				"the group order does not match" << endl;
		cout << "ago = " << ago << endl;
		cout << "go = " << go << endl;
		exit(1);
		}

	FREE_INT(Aut);
	FREE_INT(Base);
	FREE_INT(Transversal_length);
	FREE_INT(Incma);
	FREE_INT(partition);
	FREE_INT(labeling);
	FREE_OBJECT(A_perm);
	FREE_OBJECT(gens1);
	FREE_INT(Mtx);


	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer "
				"before initializing strong generators" << endl;
		}
	strong_generators *SG;

	SG = NEW_OBJECT(strong_generators);
	SG->init_from_sims(S, 0 /* verbose_level*/);
	delete S;
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer "
				"after initializing strong generators" << endl;
		}

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer "
				"done" << endl;
		}
	return SG;
}


strong_generators
*projective_space_with_action::set_stabilizer_of_object(
	object_in_projective_space *OiP, 
	INT f_save_incma_in_and_out,
	const char *save_incma_in_and_out_prefix,
	INT f_compute_canonical_form,
	uchar *&canonical_form,
	INT &canonical_form_len,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);

	action *A_linear;
	INT *Incma;
	INT *partition;
	INT *labeling;
	INT nb_rows, nb_cols;
	INT *Aut, Aut_counter;
	INT *Base, Base_length;
	INT *Transversal_length, Ago;
	INT N, i, j, a, L;

	A_linear = A;

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	if (P->incidence_bitvec == NULL) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object P->incidence_bitvec == NULL" << endl;
		exit(1);
		}


	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object before OiP->encode_incma" << endl;
		}
	OiP->encode_incma(Incma, nb_rows, nb_cols, partition, verbose_level - 1);
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object after OiP->encode_incma" << endl;
		}

	labeling = NEW_INT(nb_rows + nb_cols);
	for (i = 0; i < nb_rows + nb_cols; i++) {
		labeling[i] = i;
		}


	if (f_save_incma_in_and_out) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object Incma:" << endl;
		if (nb_rows < 10) {
			print_integer_matrix_width(cout,
					Incma, nb_rows, nb_cols, nb_cols, 1);
			}
		else {
			cout << "too large to print" << endl;
			}

		char fname_csv[1000];
		char fname_bin[1000];

		sprintf(fname_csv, "%sIncma_in_%ld_%ld.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_bin, "%sIncma_in_%ld_%ld.bin",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		INT_matrix_write_csv(fname_csv, Incma, nb_rows, nb_cols);

		colored_graph *CG;
		create_Levi_graph_from_incidence_matrix(CG,
				Incma, nb_rows, nb_cols, TRUE, labeling, verbose_level);
		CG->save(fname_bin, verbose_level);
		//FREE_INT(Incma);
		delete CG;
		}

	N = nb_rows + nb_cols;
	L = nb_rows * nb_cols;

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object initializing Aut, Base, Transversal_length" << endl;
		}
	Aut = NEW_INT(N * N);
	Base = NEW_INT(N);
	Transversal_length = NEW_INT(N);
	
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object calling nauty_interface_matrix_INT" << endl;
		}
	nauty_interface_matrix_INT(Incma, nb_rows, nb_cols, 
		labeling, partition, 
		Aut, Aut_counter, 
		Base, Base_length, 
		Transversal_length, Ago, verbose_level - 3);
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object done with nauty_interface_matrix_INT, "
				"Ago=" << Ago << endl;
		}

	INT *Incma_out;
	INT ii, jj;
	if (f_vvv) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object labeling:" << endl;
		INT_vec_print(cout, labeling, nb_rows + nb_cols);
		cout << endl;
		}

	Incma_out = NEW_INT(L);
	for (i = 0; i < nb_rows; i++) {
		ii = labeling[i];
		for (j = 0; j < nb_cols; j++) {
			jj = labeling[nb_rows + j] - nb_rows;
			//cout << "i=" << i << " j=" << j << " ii=" << ii
			//<< " jj=" << jj << endl;
			Incma_out[i * nb_cols + j] = Incma[ii * nb_cols + jj];
			}
		}
	if (f_vvv) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object Incma Out:" << endl;
		if (nb_rows < 20) {
			print_integer_matrix_width(cout,
					Incma_out, nb_rows, nb_cols, nb_cols, 1);
			}
		else {
			cout << "projective_space_with_action::set_stabilizer_"
					"of_object too large to print" << endl;
			}
		}



	if (f_compute_canonical_form) {
		

		canonical_form = bitvector_allocate_and_coded_length(
				L, canonical_form_len);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (Incma_out[i * nb_cols + j]) {
					a = i * nb_cols + j;
					bitvector_set_bit(canonical_form, a);
					}
				}
			}

		}


	if (f_save_incma_in_and_out) {
		char fname_labeling[1000];
		char fname_csv[1000];
		char fname_bin[1000];

		sprintf(fname_labeling, "%slabeling_%ld_%ld.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_csv, "%sIncma_out_%ld_%ld.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_bin, "%sIncma_out_%ld_%ld.bin",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		
		cout << "labeling:" << endl;
		INT_vec_print_as_matrix(cout,
				labeling, N, 10 /* width */, TRUE /* f_tex */);

		INT_vec_write_csv(labeling, N,
				fname_labeling, "canonical labeling");
		INT_matrix_write_csv(fname_csv, Incma_out, nb_rows, nb_cols);

		
		colored_graph *CG;
		create_Levi_graph_from_incidence_matrix(CG,
				Incma_out, nb_rows, nb_cols, TRUE, labeling, verbose_level);
		CG->save(fname_bin, verbose_level);
		FREE_OBJECT(CG);
		}

	FREE_INT(Incma_out);

	action *A_perm;
	longinteger_object ago;


	A_perm = NEW_OBJECT(action);

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object before init_permutation_group_"
				"from_generators" << endl;
		}
	ago.create(Ago);
	A_perm->init_permutation_group_from_generators(N, 
		TRUE, ago, 
		Aut_counter, Aut, 
		Base_length, Base,
		0 /*verbose_level - 2 */);

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object create_automorphism_group_of_"
				"incidence_structure: created action ";
		A_perm->print_info();
		cout << endl;
		}

	//action *A_linear;

	//A_linear = A;

	if (A_linear == NULL) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object A_linear == NULL" << endl;
		exit(1);
		}

	vector_ge *gens; // permutations from nauty
	vector_ge *gens1; // matrices
	INT g, frobenius, pos;
	INT *Mtx;
	
	gens = A_perm->Strong_gens->gens;

	gens1 = NEW_OBJECT(vector_ge);
	gens1->init(A_linear);
	gens1->allocate(gens->len);
	
	Mtx = NEW_INT(d * d + 1); // leave space for frobenius
	
	pos = 0;
	for (g = 0; g < gens->len; g++) {
		if (f_vv) {
			cout << "projective_space_with_action::set_stabilizer_"
					"of_object strong generator " << g << ":" << endl;
			A_perm->element_print(gens->ith(g), cout);
			cout << endl;
			}
		
		if (reverse_engineer_semilinear_map(A_perm, P, 
			gens->ith(g), Mtx, frobenius, 
			0 /*verbose_level - 2*/)) {

			Mtx[d * d] = frobenius;
			A_linear->make_element(Elt1, Mtx, 0 /*verbose_level - 2*/);
			if (f_vv) {
				cout << "projective_space_with_action::set_stabilizer_"
						"of_object semi-linear group element:" << endl;
				A_linear->element_print(Elt1, cout);
				}
			A_linear->element_move(Elt1, gens1->ith(pos), 0);
		

			pos++;
			}
		else {
			if (f_vv) {
				cout << "projective_space_with_action::set_stabilizer_"
						"of_object generator " << g << " does not "
								"correspond to a semilinear mapping" << endl;
				}
			}
		}
	gens1->reallocate(pos);
	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer_of_object "
				"we found " << gens1->len << " generators" << endl;
		}

	if (f_vvv) {
		gens1->print(cout);
		}
	

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer_of_object "
				"we are now testing the generators:" << endl;
		}
	INT j1, j2;
	
	for (g = 0; g < gens1->len; g++) {
		if (f_vv) {
			cout << "generator " << g << ":" << endl;
			}
		//A_linear->element_print(gens1->ith(g), cout);
		for (i = 0; i < P->N_points; i++) {
			j1 = A_linear->element_image_of(i, gens1->ith(g), 0);
			j2 = A_perm->element_image_of(i, gens->ith(g), 0);
			if (j1 != j2) {
				cout << "projective_space_with_action::set_stabilizer_"
						"of_object problem with generator: j1 != j2" << endl;
				cout << "i=" << i << endl;
				cout << "j1=" << j1 << endl;
				cout << "j2=" << j2 << endl;
				cout << endl;
				exit(1);
				}
			}
		}
	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer_of_"
				"object the generators are OK" << endl;
		}



	sims *S;
	longinteger_object go;

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object we are now creating the group" << endl;
		}

	S = create_sims_from_generators_with_target_group_order(A_linear, 
		gens1, ago, 0 /*verbose_level*/);
	
	S->group_order(go);

	
	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object Found a group of order " << go << endl;
		}
	if (f_vvv) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object strong generators are:" << endl;
		S->print_generators();
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object strong generators are (in tex):" << endl;
		S->print_generators_tex(cout);
		}


	longinteger_domain D;
	
	if (D.compare_unsigned(ago, go)) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object the group order does not match" << endl;
		cout << "ago = " << ago << endl;
		cout << "go = " << go << endl;
		exit(1);
		}

	FREE_INT(Aut);
	FREE_INT(Base);
	FREE_INT(Transversal_length);
	FREE_INT(Incma);
	FREE_INT(partition);
	FREE_INT(labeling);
	FREE_OBJECT(A_perm);
	FREE_OBJECT(gens);
	FREE_INT(Mtx);


	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object before initializing strong generators" << endl;
		}
	strong_generators *SG;

	SG = NEW_OBJECT(strong_generators);
	SG->init_from_sims(S, 0 /* verbose_level*/);
	FREE_OBJECT(S);
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object after initializing strong generators" << endl;
		}

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object done" << endl;
		}
	return SG;
}


void projective_space_with_action::report_fixed_objects_in_PG_3_tex(
	INT *Elt, ostream &ost, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_fixed_"
				"objects_in_PG_3_tex" << endl;
		}

	if (P->n != 3) {
		cout << "projective_space_with_action::report_fixed_"
				"objects_in_PG_3_tex P->n != 3" << endl;
		exit(1);
		}
	projective_space *P3;
	INT i, j, cnt;
	INT v[4];

	P3 = P;
	
	ost << "Fixed Objects:\\\\" << endl;



	ost << "The element" << endl;
	ost << "$$" << endl;
	A->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following fixed objects:\\\\" << endl;


	ost << "Fixed points:\\\\" << endl;

	cnt = 0;
	for (i = 0; i < P3->N_points; i++) {
		j = A->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
			}
		}

	ost << "There are " << cnt << " fixed points, they are: \\\\" << endl;
	for (i = 0; i < P3->N_points; i++) {
		j = A->element_image_of(i, Elt, 0 /* verbose_level */);
		PG_element_unrank_modified(*F, v, 1, 4, i);
		if (j == i) {
			ost << i << " : ";
			INT_vec_print(ost, v, 4);
			ost << "\\\\" << endl;
			cnt++;
			}
		}

	ost << "Fixed Lines:\\\\" << endl;

	{
	action *A2;
	
	A2 = A->induced_action_on_grassmannian(2, 0 /* verbose_level*/);

	cnt = 0;
	for (i = 0; i < A2->degree; i++) {
		j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
			}
		}

	ost << "There are " << cnt << " fixed lines, they are: \\\\" << endl;
	cnt = 0;
	for (i = 0; i < A2->degree; i++) {
		j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			ost << i << " : $\\left[";
			A2->G.AG->G->print_single_generator_matrix_tex(ost, i);
			ost << "\\right]$\\\\" << endl;
			cnt++;
			}
		}

	FREE_OBJECT(A2);
	}

	ost << "Fixed Planes:\\\\" << endl;

	{
	action *A2;
	
	A2 = A->induced_action_on_grassmannian(3, 0 /* verbose_level*/);

	cnt = 0;
	for (i = 0; i < A2->degree; i++) {
		j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
			}
		}

	ost << "There are " << cnt << " fixed planes, they are: \\\\" << endl;
	cnt = 0;
	for (i = 0; i < A2->degree; i++) {
		j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			ost << i << " : $\\left[";
			A2->G.AG->G->print_single_generator_matrix_tex(ost, i);
			ost << "\\right]$\\\\" << endl;
			cnt++;
			}
		}

	FREE_OBJECT(A2);
	}

	if (f_v) {
		cout << "projective_space_with_action::report_"
				"fixed_objects_in_PG_3_tex done" << endl;
		}
}


