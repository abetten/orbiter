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

namespace orbiter {


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
		FREE_int(Elt1);
		}
	null();
}

void projective_space_with_action::init(
	finite_field *F, int n, int f_semilinear,
	int f_init_incidence_structure,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

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
	
	Elt1 = NEW_int(A->elt_size_in_int);


	if (f_v) {
		cout << "projective_space_with_action::init done" << endl;
		}
}

void projective_space_with_action::init_group(
		int f_semilinear, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
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
	int *set, int set_size, int &canonical_pt, 
	int *canonical_set_or_NULL, 
	int f_save_incma_in_and_out,
	const char *save_incma_in_and_out_prefix,
	int f_compute_canonical_form,
	uchar *&canonical_form,
	int &canonical_form_len,
	int verbose_level)
// December 22, 2017, based on earlier work:
// added to action_global.C on 2/28/2011, called from analyze.C
// November 17, 2014 moved here from TOP_LEVEL/extra.C
// December 31, 2014, moved here from projective_space.C
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	action *A_linear;
	int *Incma;
	int *partition;
	int *labeling;
	int nb_rows, nb_cols;
	int *Aut, Aut_counter;
	int *Base, Base_length;
	int *Transversal_length, Ago;
	int N, i, j, h, a, L;

	A_linear = A;

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		cout << "set_size = " << set_size << endl;
		}
	if (P->incidence_bitvec == NULL) {
		cout << "projective_space_with_action::set_stabilizer "
				"P->incidence_bitvec == NULL" << endl;
		exit(1);
		}

	if (f_vv) {
		cout << "computing the type of the set" << endl;
		}

	classify C;

	C.init(set, set_size, TRUE, 0);
	if (C.second_nb_types > 1) {
		cout << "projective_space_with_action::set_stabilizer: "
				"The set is a multiset:" << endl;
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
	Incma = NEW_int(nb_rows * nb_cols);
	partition = NEW_int(nb_rows + nb_cols);
	labeling = NEW_int(nb_rows + nb_cols);

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer "
				"Initializing Incma" << endl;
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
		int f2, l2, m, idx, f, l;

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

		sprintf(fname_csv, "%sIncma_in_%d_%d.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_bin, "%sIncma_in_%d_%d.bin",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		int_matrix_write_csv(fname_csv, Incma, nb_rows, nb_cols);

		for (i = 0; i < nb_rows + nb_cols; i++) {
			labeling[i] = i;
			}

		colored_graph *CG;
		create_Levi_graph_from_incidence_matrix(CG,
				Incma, nb_rows, nb_cols, TRUE, labeling, verbose_level);
		CG->save(fname_bin, verbose_level);
		//FREE_int(Incma);
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
	Aut = NEW_int(N * N);
	Base = NEW_int(N);
	Transversal_length = NEW_int(N);
	
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer "
				"calling nauty_interface_matrix_int" << endl;
		}
	nauty_interface_matrix_int(Incma, nb_rows, nb_cols, 
		labeling, partition, 
		Aut, Aut_counter, 
		Base, Base_length, 
		Transversal_length, Ago, verbose_level - 3);
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer "
				"done with nauty_interface_matrix_int, "
				"Ago=" << Ago << endl;
		}

	int *Incma_out;
	int ii, jj;
	if (f_vvv) {
		cout << "projective_space_with_action::set_stabilizer "
				"labeling:" << endl;
		int_vec_print(cout, labeling, nb_rows + nb_cols);
		cout << endl;
		}

	Incma_out = NEW_int(L);
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

		sprintf(fname_labeling, "%slabeling_%d_%d.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_csv, "%sIncma_out_%d_%d.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_bin, "%sIncma_out_%d_%d.bin",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		
		int_vec_write_csv(labeling, N,
				fname_labeling, "canonical labeling");
		int_matrix_write_csv(fname_csv, Incma_out, nb_rows, nb_cols);

		
		colored_graph *CG;
		create_Levi_graph_from_incidence_matrix(CG,
				Incma_out, nb_rows, nb_cols,
				TRUE, labeling, verbose_level);
		CG->save(fname_bin, verbose_level);
		FREE_OBJECT(CG);
		}

	FREE_int(Incma_out);

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
	int g, frobenius, pos;
	int *Mtx;
	
	gens = A_perm->Strong_gens->gens;

	gens1 = NEW_OBJECT(vector_ge);
	gens1->init(A_linear);
	gens1->allocate(gens->len);
	
	Mtx = NEW_int(d * d + 1); // leave space for frobenius
	
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
	int j1, j2;
	
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

	FREE_int(Aut);
	FREE_int(Base);
	FREE_int(Transversal_length);
	FREE_int(Incma);
	FREE_int(partition);
	FREE_int(labeling);
	FREE_OBJECT(A_perm);
	FREE_OBJECT(gens1);
	FREE_int(Mtx);


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
	int f_save_incma_in_and_out,
	const char *save_incma_in_and_out_prefix,
	int f_compute_canonical_form,
	uchar *&canonical_form,
	int &canonical_form_len,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	action *A_linear;
	int *Incma;
	int *partition;
	int *labeling;
	int nb_rows, nb_cols;
	int *Aut, Aut_counter;
	int *Base, Base_length;
	int *Transversal_length, Ago;
	int N, i, j, a, L;

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
	OiP->encode_incma(Incma, nb_rows, nb_cols,
			partition, verbose_level - 1);
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object after OiP->encode_incma" << endl;
		}

	labeling = NEW_int(nb_rows + nb_cols);
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

		sprintf(fname_csv, "%sIncma_in_%d_%d.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_bin, "%sIncma_in_%d_%d.bin",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		int_matrix_write_csv(fname_csv, Incma, nb_rows, nb_cols);

		colored_graph *CG;
		create_Levi_graph_from_incidence_matrix(CG,
				Incma, nb_rows, nb_cols,
				TRUE, labeling, verbose_level);
		CG->save(fname_bin, verbose_level);
		//FREE_int(Incma);
		delete CG;
		}

	N = nb_rows + nb_cols;
	L = nb_rows * nb_cols;

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object initializing Aut, Base, "
				"Transversal_length" << endl;
		}
	Aut = NEW_int(N * N);
	Base = NEW_int(N);
	Transversal_length = NEW_int(N);
	
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object calling nauty_interface_matrix_int" << endl;
		}
	nauty_interface_matrix_int(
		Incma, nb_rows, nb_cols,
		labeling, partition, 
		Aut, Aut_counter, 
		Base, Base_length, 
		Transversal_length, Ago, verbose_level - 3);
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object done with nauty_interface_matrix_int, "
				"Ago=" << Ago << endl;
		}

	int *Incma_out;
	int ii, jj;
	if (f_vvv) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object labeling:" << endl;
		int_vec_print(cout, labeling, nb_rows + nb_cols);
		cout << endl;
		}

	Incma_out = NEW_int(L);
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

		sprintf(fname_labeling, "%slabeling_%d_%d.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_csv, "%sIncma_out_%d_%d.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_bin, "%sIncma_out_%d_%d.bin",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		
		cout << "labeling:" << endl;
		int_vec_print_as_matrix(cout,
				labeling, N, 10 /* width */, TRUE /* f_tex */);

		int_vec_write_csv(labeling, N,
				fname_labeling, "canonical labeling");
		int_matrix_write_csv(fname_csv, Incma_out, nb_rows, nb_cols);

		
		colored_graph *CG;
		create_Levi_graph_from_incidence_matrix(CG,
				Incma_out, nb_rows, nb_cols,
				TRUE, labeling, verbose_level);
		CG->save(fname_bin, verbose_level);
		FREE_OBJECT(CG);
		}

	FREE_int(Incma_out);

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
	int g, frobenius, pos;
	int *Mtx;
	
	gens = A_perm->Strong_gens->gens;

	gens1 = NEW_OBJECT(vector_ge);
	gens1->init(A_linear);
	gens1->allocate(gens->len);
	
	Mtx = NEW_int(d * d + 1); // leave space for frobenius
	
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
	int j1, j2;
	
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
						"of_object problem with generator: "
						"j1 != j2" << endl;
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

	if (f_v) {
		cout << "before freeing Aut" << endl;
	}
	FREE_int(Aut);
	if (f_v) {
		cout << "before freeing Base" << endl;
	}
	FREE_int(Base);
	if (f_v) {
		cout << "before freeing Transversal_length" << endl;
	}
	FREE_int(Transversal_length);
	if (f_v) {
		cout << "before freeing Incma" << endl;
	}
	FREE_int(Incma);
	if (f_v) {
		cout << "before freeing partition" << endl;
	}
	FREE_int(partition);
	if (f_v) {
		cout << "before freeing labeling" << endl;
	}
	FREE_int(labeling);
	if (f_v) {
		cout << "before freeing A_perm" << endl;
	}
	FREE_OBJECT(A_perm);
	if (f_v) {
		cout << "not freeing gens" << endl;
	}
	//FREE_OBJECT(gens);
	if (f_v) {
		cout << "before freeing Mtx" << endl;
	}
	FREE_int(Mtx);


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
	int *Elt, ostream &ost, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

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
	int i, j, cnt;
	int v[4];

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
		F->PG_element_unrank_modified(v, 1, 4, i);
		if (j == i) {
			ost << i << " : ";
			int_vec_print(ost, v, 4);
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

void projective_space_with_action::report_orbits_in_PG_3_tex(
	int *Elt, ostream &ost,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::"
				"report_orbits_in_PG_3_tex" << endl;
		}

	if (P->n != 3) {
		cout << "projective_space_with_action::"
				"report_orbits_in_PG_3_tex P->n != 3" << endl;
		exit(1);
		}
	//projective_space *P3;
	int order;

	longinteger_object full_group_order;
	order = A->element_order(Elt);

	full_group_order.create(order);

	//P3 = P;

	ost << "Fixed Objects:\\\\" << endl;



	ost << "The group generated by the element" << endl;
	ost << "$$" << endl;
	A->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following orbits:\\\\" << endl;

	ost << "Orbits on points:\\\\" << endl;


	schreier *Sch;

	Sch = NEW_OBJECT(schreier);
	A->all_point_orbits_from_single_generator(*Sch,
			Elt,
			verbose_level);
	Sch->print_orbit_lengths_tex(ost);


	FREE_OBJECT(Sch);

	ost << "Orbits on lines:\\\\" << endl;

	{
	action *A2;
	schreier *Sch;

	A2 = A->induced_action_on_grassmannian(2, 0 /* verbose_level*/);

	Sch = NEW_OBJECT(schreier);
	A2->all_point_orbits_from_single_generator(*Sch,
			Elt,
			verbose_level);
	Sch->print_orbit_lengths_tex(ost);


	FREE_OBJECT(Sch);
	FREE_OBJECT(A2);
	}

	ost << "Orbits on planes:\\\\" << endl;

	{
	action *A2;
	schreier *Sch;


	A2 = A->induced_action_on_grassmannian(3, 0 /* verbose_level*/);

	Sch = NEW_OBJECT(schreier);
	A2->all_point_orbits_from_single_generator(*Sch,
			Elt,
			verbose_level);
	Sch->print_orbit_lengths_tex(ost);


	FREE_OBJECT(Sch);
	FREE_OBJECT(A2);
	}

	if (f_v) {
		cout << "projective_space_with_action::"
				"report_orbits_in_PG_3_tex done" << endl;
		}
}

void projective_space_with_action::report_decomposition_by_single_automorphism(
	int *Elt, ostream &ost,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_"
				"decomposition_by_single_automorphism" << endl;
		}

	if (P->n != 3) {
		cout << "projective_space_with_action::report_"
				"decomposition_by_single_automorphism P->n != 3" << endl;
		exit(1);
		}
	//projective_space *P3;
	int order;

	longinteger_object full_group_order;
	order = A->element_order(Elt);

	full_group_order.create(order);

	//P3 = P;

	//ost << "Fixed Objects:\\\\" << endl;


#if 0
	ost << "The group generated by the element" << endl;
	ost << "$$" << endl;
	A->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following orbits:\\\\" << endl;

	ost << "Orbits on points:\\\\" << endl;
#endif

	schreier *Sch1;
	schreier *Sch2;
	incidence_structure *Inc;
	partitionstack *Stack;
	partitionstack S1;
	partitionstack S2;

	Sch1 = NEW_OBJECT(schreier);
	Sch2 = NEW_OBJECT(schreier);
	A->all_point_orbits_from_single_generator(*Sch1,
			Elt,
			verbose_level);
	Sch1->print_orbit_lengths_tex(ost);


	//ost << "Orbits on lines:\\\\" << endl;

	Sch2 = NEW_OBJECT(schreier);
	A_on_lines->all_point_orbits_from_single_generator(*Sch2,
			Elt,
			verbose_level);
	//Sch->print_orbit_lengths_tex(ost);

	if (f_v) {
		cout << "projective_space_with_action::report_"
				"decomposition_by_single_automorphism "
				"before incidence_and_stack_for_type_ij" << endl;
		}
	P->incidence_and_stack_for_type_ij(
		1 /* row_type */, 2 /* col_type */,
		Inc,
		Stack,
		verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::report_"
				"decomposition_by_single_automorphism "
				"after incidence_and_stack_for_type_ij" << endl;
		}

	if (f_v) {
		cout << "projective_space_with_action::report_"
				"decomposition_by_single_automorphism "
				"before S1.allocate" << endl;
		}
	S1.allocate(A->degree, 0 /* verbose_level */);
	S2.allocate(A_on_lines->degree, 0 /* verbose_level */);

	if (f_v) {
		cout << "projective_space_with_action::report_"
				"decomposition_by_single_automorphism "
				"before Sch1->get_orbit_partition" << endl;
		}
	Sch1->get_orbit_partition(S1,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::report_"
				"decomposition_by_single_automorphism "
				"before Sch2->get_orbit_partition" << endl;
		}
	Sch2->get_orbit_partition(S2,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::report_"
				"decomposition_by_single_automorphism "
				"after Sch2->get_orbit_partition" << endl;
		}
	int i, j, sz;

	for (i = 1; i < S1.ht; i++) {
		if (f_v) {
			cout << "projective_space_with_action::report_"
					"decomposition_by_single_automorphism "
					"before Stack->split_cell (S1) i=" << i << endl;
			}
		Stack->split_cell(S1.pointList + S1.startCell[i], S1.cellSize[i], verbose_level);
	}
	int *set;
	set = NEW_int(A_on_lines->degree);
	for (i = 1; i < S2.ht; i++) {
		sz = S2.cellSize[i];
		int_vec_copy(S2.pointList + S2.startCell[i], set, sz);
		for (j = 0; j < sz; j++) {
			set[j] += A->degree;
		}
		if (f_v) {
			cout << "projective_space_with_action::report_"
					"decomposition_by_single_automorphism "
					"before Stack->split_cell (S2) i=" << i << endl;
			}
		Stack->split_cell(set, sz, verbose_level);
	}
	FREE_int(set);

	int f_print_subscripts = FALSE;
	ost << "Row scheme under cyclic group:\\\\" << endl;
	Inc->get_and_print_row_tactical_decomposition_scheme_tex(
		ost, TRUE /* f_enter_math */,
		f_print_subscripts, *Stack);
	ost << "Column scheme under cyclic group:\\\\" << endl;
	Inc->get_and_print_column_tactical_decomposition_scheme_tex(
		ost, TRUE /* f_enter_math */,
		f_print_subscripts, *Stack);


#if 0
	// data structure for the partition stack,
	// following Leon:
		int n;
		int ht;
		int ht0;

		int *pointList, *invPointList;
		int *cellNumber;

		int *startCell;
		int *cellSize;
		int *parent;
#endif


	FREE_OBJECT(Sch1);
	FREE_OBJECT(Sch2);
	FREE_OBJECT(Inc);
	FREE_OBJECT(Stack);

	if (f_v) {
		cout << "projective_space_with_action::report_"
				"decomposition_by_single_automorphism done" << endl;
		}
}


object_in_projective_space *projective_space_with_action::create_object_from_string(
	int type, const char *set_as_string, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::create_object_from_string" << endl;
		cout << "type=" << type << endl;
		}


	int *the_set_in;
	int set_size_in;
	object_in_projective_space *OiP;

	int_vec_scan(set_as_string, the_set_in, set_size_in);


	if (f_v) {
		cout << "The input set has size " << set_size_in << ":" << endl;
		cout << "The input set is:" << endl;
		int_vec_print(cout, the_set_in, set_size_in);
		cout << endl;
		cout << "The type is: ";
		if (type == t_PTS) {
			cout << "t_PTS" << endl;
			}
		else if (type == t_LNS) {
			cout << "t_LNS" << endl;
			}
		else if (type == t_PAC) {
			cout << "t_PAC" << endl;
			}
		}


	OiP = NEW_OBJECT(object_in_projective_space);

	if (type == t_PTS) {
		OiP->init_point_set(P,
				the_set_in, set_size_in, verbose_level - 1);
		}
	else if (type == t_LNS) {
		OiP->init_line_set(P,
				the_set_in, set_size_in, verbose_level - 1);
		}
	else if (type == t_PAC) {
		OiP->init_packing_from_set(P,
				the_set_in, set_size_in, verbose_level - 1);
		}
	else {
		cout << "create_object_from_string unknown type" << endl;
		exit(1);
		}

	FREE_int(the_set_in);

	if (f_v) {
		cout << "projective_space_with_action::create_object_from_string"
				" done" << endl;
		}
	return OiP;
}

int projective_space_with_action::process_object(
	classify_bitvectors *CB,
	object_in_projective_space *OiP,
	int f_save_incma_in_and_out, const char *prefix,
	int nb_objects_to_test,
	strong_generators *&SG,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "projective_space_with_action::process_object "
				"n=" << CB->n << endl;
		}

	longinteger_object go;
	//int *Extra_data;
	char save_incma_in_and_out_prefix[1000];

	if (f_save_incma_in_and_out) {
		sprintf(save_incma_in_and_out_prefix, "%s_%d_", prefix, CB->n);
		}


	uchar *canonical_form;
	int canonical_form_len;


	if (f_v) {
		cout << "projective_space_with_action::process_object "
				"before PA->set_stabilizer_of_object" << endl;
		}

	SG = set_stabilizer_of_object(
		OiP,
		f_save_incma_in_and_out, save_incma_in_and_out_prefix,
		TRUE /* f_compute_canonical_form */,
		canonical_form, canonical_form_len,
		verbose_level - 2);

	if (f_v) {
		cout << "projective_space_with_action::process_object "
				"after PA->set_stabilizer_of_object" << endl;
		}


	SG->group_order(go);

	//cout << "object:" << endl;
	//OiP->print(cout);
	//cout << "go=" << go << endl;
#if 0
	cout << "projective_space_with_action::process_object canonical form: ";
	for (i = 0; i < canonical_form_len; i++) {
		cout << (int)canonical_form[i];
		if (i < canonical_form_len - 1) {
			cout << ", ";
			}
		}
#endif
	//cout << endl;

#if 0
	Extra_data = NEW_int(OiP->sz);
	int_vec_copy(OiP->set, Extra_data, OiP->sz);

	if (CB->n == 0) {
		CB->init(nb_objects_to_test, canonical_form_len, verbose_level);
		sz = OiP->sz;
		}
	else {
		if (OiP->sz != sz) {
			cout << "projective_space_with_action::process_object "
					"OiP->sz != sz" << endl;
			exit(1);
			}
		}
	if (!CB->add(canonical_form, Extra_data, verbose_level)) {
		FREE_int(Extra_data);
		}
#endif
	if (CB->n == 0) {
		CB->init(nb_objects_to_test, canonical_form_len, verbose_level);
		}
	ret = CB->add(canonical_form, OiP, verbose_level);


	//delete SG;

	if (f_v) {
		cout << "projective_space_with_action::process_object done" << endl;
		}
	return ret;
}

void projective_space_with_action::classify_objects_using_nauty(
	data_input_stream *Data,
	int nb_objects_to_test,
	classify_bitvectors *CB,
	int f_save_incma_in_and_out, const char *prefix,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int input_idx, ret;

	if (f_v) {
		cout << "classify_objects_using_nauty" << endl;
		}

	for (input_idx = 0; input_idx < Data->nb_inputs; input_idx++) {
		cout << "input " << input_idx << " / " << Data->nb_inputs
			<< " is:" << endl;

		if (Data->input_type[input_idx] == INPUT_TYPE_SET_OF_POINTS) {
			cout << "input set of points "
				<< Data->input_string[input_idx] << ":" << endl;

			object_in_projective_space *OiP;
			strong_generators *SG;

			OiP = create_object_from_string(t_PTS,
					Data->input_string[input_idx], verbose_level);

			if (f_v) {
				cout << "classify_objects_using_nauty "
						"before process_object" << endl;
				}

			ret = process_object(CB, OiP,
					f_save_incma_in_and_out, prefix,
					nb_objects_to_test,
					SG,
					verbose_level);

			if (f_v) {
				cout << "classify_objects_using_nauty "
						"after process_object, ret=" << ret << endl;
				}


			if (!ret) {
				FREE_OBJECT(SG);
				FREE_OBJECT(OiP);
				}
			else {
				cout << "New isomorphism type! The n e w number of "
					"isomorphism types is " << CB->nb_types << endl;
				int idx;

				object_in_projective_space_with_action *OiPA;

				OiPA = NEW_OBJECT(object_in_projective_space_with_action);

				OiPA->init(OiP, SG, verbose_level);
				idx = CB->type_of[CB->n - 1];
				CB->Type_extra_data[idx] = OiPA;

				compute_and_print_ago_distribution(cout,
						CB, verbose_level);
				}
			}
		else if (Data->input_type[input_idx] == INPUT_TYPE_SET_OF_LINES) {
			cout << "input set of lines " << Data->input_string[input_idx]
				<< ":" << endl;

			object_in_projective_space *OiP;
			strong_generators *SG;

			OiP = create_object_from_string(t_LNS,
					Data->input_string[input_idx], verbose_level);
			if (!process_object(CB, OiP,
				f_save_incma_in_and_out, prefix,
				nb_objects_to_test,
				SG,
				verbose_level)) {

				FREE_OBJECT(SG);
				FREE_OBJECT(OiP);
				}
			else {
				cout << "New isomorphism type! The n e w number of "
						"isomorphism types is " << CB->nb_types << endl;
				int idx;

				object_in_projective_space_with_action *OiPA;

				OiPA = NEW_OBJECT(object_in_projective_space_with_action);

				OiPA->init(OiP, SG, verbose_level);
				idx = CB->type_of[CB->n - 1];
				CB->Type_extra_data[idx] = OiPA;

				compute_and_print_ago_distribution(cout,
					CB, verbose_level);
				}
			}
		else if (Data->input_type[input_idx] == INPUT_TYPE_SET_OF_PACKING) {
			cout << "input set of packing "
				<< Data->input_string[input_idx] << ":" << endl;

			object_in_projective_space *OiP;
			strong_generators *SG;

			OiP = create_object_from_string(t_PAC,
					Data->input_string[input_idx], verbose_level);
			if (!process_object(CB, OiP,
				f_save_incma_in_and_out, prefix,
				nb_objects_to_test,
				SG,
				verbose_level)) {

				FREE_OBJECT(SG);
				FREE_OBJECT(OiP);
				}
			else {
				cout << "New isomorphism type! The n e w number of "
					"isomorphism types is " << CB->nb_types << endl;
				int idx;

				object_in_projective_space_with_action *OiPA;

				OiPA = NEW_OBJECT(object_in_projective_space_with_action);

				OiPA->init(OiP, SG, verbose_level);
				idx = CB->type_of[CB->n - 1];
				CB->Type_extra_data[idx] = OiPA;

				compute_and_print_ago_distribution(cout,
						CB, verbose_level);
				}
			}
		else if (Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINTS ||
				Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_LINES ||
				Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_PACKINGS ||
				Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
			cout << "input from file " << Data->input_string[input_idx]
				<< ":" << endl;

			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			cout << "Reading the file " << Data->input_string[input_idx] << endl;
			SoS->init_from_file(
					P->N_points /* underlying_set_size */,
					Data->input_string[input_idx], verbose_level);
			cout << "Read the file " << Data->input_string[input_idx] << endl;

			int h;


			// for use if INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE
			int *Spread_table;
			int nb_spreads;
			int spread_size;

			if (Data->input_type[input_idx] ==
					INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
				cout << "Reading spread table from file "
					<< Data->input_string2[input_idx] << endl;
				int_matrix_read_csv(Data->input_string2[input_idx],
						Spread_table, nb_spreads, spread_size,
						0 /* verbose_level */);
				cout << "Reading spread table from file "
						<< Data->input_string2[input_idx] << " done" << endl;
				cout << "The spread table contains " << nb_spreads
						<< " spreads" << endl;
				}

			cout << "processing " << SoS->nb_sets << " objects" << endl;

			for (h = 0; h < SoS->nb_sets; h++) {


				int *the_set_in;
				int set_size_in;
				object_in_projective_space *OiP;


				set_size_in = SoS->Set_size[h];
				the_set_in = SoS->Sets[h];

				if (f_vv || ((h % 1024) == 0)) {
					cout << "The input set " << h << " / " << SoS->nb_sets
						<< " has size " << set_size_in << ":" << endl;
					}

				if (f_vvv) {
					cout << "The input set is:" << endl;
					int_vec_print(cout, the_set_in, set_size_in);
					cout << endl;
					}

				OiP = NEW_OBJECT(object_in_projective_space);

				if (Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_POINTS) {
					OiP->init_point_set(P, the_set_in, set_size_in,
							0 /* verbose_level*/);
					}
				else if (Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_LINES) {
					OiP->init_line_set(P, the_set_in, set_size_in,
							0 /* verbose_level*/);
					}
				else if (Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_PACKINGS) {
					OiP->init_packing_from_set(P,
							the_set_in, set_size_in, verbose_level);
					}
				else if (Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
					OiP->init_packing_from_spread_table(P, the_set_in,
						Spread_table, nb_spreads, spread_size,
						verbose_level);
					}
				else {
					cout << "unknown type" << endl;
					exit(1);
					}
				strong_generators *SG;
				if (!process_object(CB, OiP,
					f_save_incma_in_and_out, prefix,
					nb_objects_to_test,
					SG,
					verbose_level - 3)) {

					FREE_OBJECT(OiP);
					FREE_OBJECT(SG);
					}
				else {
					cout << "New isomorphism type! The n e w number of "
							"isomorphism types is " << CB->nb_types << endl;

					int idx;

					object_in_projective_space_with_action *OiPA;

					OiPA = NEW_OBJECT(object_in_projective_space_with_action);

					OiPA->init(OiP, SG, verbose_level);
					idx = CB->type_of[CB->n - 1];
					CB->Type_extra_data[idx] = OiPA;


					compute_and_print_ago_distribution(cout,
							CB, verbose_level);
					}

				if (f_vv) {
					cout << "after input set " << h << " / "
							<< SoS->nb_sets
							<< ", we have " << CB->nb_types
							<< " isomorphism types of objects" << endl;
					}

				}
			if (Data->input_type[input_idx] ==
					INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
				FREE_int(Spread_table);
				}
			FREE_OBJECT(SoS);
			}
		else {
			cout << "unknown input type" << endl;
			exit(1);
			}
		}

	CB->finalize(verbose_level); // computes C_type_of and perm

	if (f_v) {
		cout << "projective_space_with_action::classify_objects_using_nauty done" << endl;
		}
}







// #############################################################################
// globals:
// #############################################################################



void OiPA_encode(void *extra_data,
		int *&encoding, int &encoding_sz, void *global_data)
{
	//cout << "OiPA_encode" << endl;
	object_in_projective_space_with_action *OiPA;
	object_in_projective_space *OiP;

	OiPA = (object_in_projective_space_with_action *) extra_data;
	OiP = OiPA->OiP;
	//OiP->print(cout);
	OiP->encode_object(encoding, encoding_sz, 1 /* verbose_level*/);
	//cout << "OiPA_encode done" << endl;

}

void OiPA_group_order(void *extra_data,
		longinteger_object &go, void *global_data)
{
	//cout << "OiPA_group_order" << endl;
	object_in_projective_space_with_action *OiPA;
	//object_in_projective_space *OiP;

	OiPA = (object_in_projective_space_with_action *) extra_data;
	//OiP = OiPA->OiP;
	OiPA->Aut_gens->group_order(go);
	//cout << "OiPA_group_order done" << endl;

}

void print_summary_table_entry(int *Table,
		int m, int n, int i, int j, int val, char *output, void *data)
{
	classify_bitvectors *CB;
	object_in_projective_space_with_action *OiPA;
	void *extra_data;
	longinteger_object go;
	int h;

	CB = (classify_bitvectors *) data;

	if (i == -1) {
		if (j == -1) {
			sprintf(output, "\\mbox{Orbit}");
			}
		else if (j == 0) {
			sprintf(output, "\\mbox{Rep}");
			}
		else if (j == 1) {
			sprintf(output, "\\#");
			}
		else if (j == 2) {
			sprintf(output, "\\mbox{Ago}");
			}
		else if (j == 3) {
			sprintf(output, "\\mbox{Objects}");
			}
		}
	else {
		//cout << "print_summary_table_entry i=" << i << " j=" << j << endl;
		if (j == -1) {
			sprintf(output, "%d", i);
			}
		else if (j == 2) {
			extra_data = CB->Type_extra_data[CB->perm[i]];

			OiPA = (object_in_projective_space_with_action *) extra_data;
			OiPA->Aut_gens->group_order(go);
			go.print_to_string(output);
			}
		else if (j == 3) {


			int *Input_objects;
			int nb_input_objects;
			CB->C_type_of->get_class_by_value(Input_objects,
				nb_input_objects, CB->perm[i], 0 /*verbose_level */);
			int_vec_heapsort(Input_objects, nb_input_objects);

			output[0] = 0;
			for (h = 0; h < nb_input_objects; h++) {
				sprintf(output + strlen(output), "%d", Input_objects[h]);
				if (h < nb_input_objects - 1) {
					strcat(output, ", ");
					}
				if (h == 10) {
					strcat(output, "\\ldots");
					break;
					}
				}

			FREE_int(Input_objects);
			}
		else {
			sprintf(output, "%d", val);
			}
		}
}


void compute_ago_distribution(
	classify_bitvectors *CB, classify *&C_ago, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_ago_distribution" << endl;
		}
	int *Ago;
	int i;

	Ago = NEW_int(CB->nb_types);
	for (i = 0; i < CB->nb_types; i++) {
		object_in_projective_space_with_action *OiPA;

		OiPA = (object_in_projective_space_with_action *)
				CB->Type_extra_data[i];
		Ago[i] = OiPA->Aut_gens->group_order_as_int();
		}
	C_ago = NEW_OBJECT(classify);
	C_ago->init(Ago, CB->nb_types, FALSE, 0);
	FREE_int(Ago);
	if (f_v) {
		cout << "compute_ago_distribution done" << endl;
		}
}

void compute_ago_distribution_permuted(
	classify_bitvectors *CB, classify *&C_ago, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_ago_distribution_permuted" << endl;
		}
	int *Ago;
	int i;

	Ago = NEW_int(CB->nb_types);
	for (i = 0; i < CB->nb_types; i++) {
		object_in_projective_space_with_action *OiPA;

		OiPA = (object_in_projective_space_with_action *)
				CB->Type_extra_data[CB->perm[i]];
		Ago[i] = OiPA->Aut_gens->group_order_as_int();
		}
	C_ago = NEW_OBJECT(classify);
	C_ago->init(Ago, CB->nb_types, FALSE, 0);
	FREE_int(Ago);
	if (f_v) {
		cout << "compute_ago_distribution_permuted done" << endl;
		}
}

void compute_and_print_ago_distribution(ostream &ost,
	classify_bitvectors *CB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_and_print_ago_distribution" << endl;
		}
	classify *C_ago;
	compute_ago_distribution(CB, C_ago, verbose_level);
	ost << "ago distribution: " << endl;
	ost << "$$" << endl;
	C_ago->print_naked_tex(ost, TRUE /* f_backwards */);
	ost << endl;
	ost << "$$" << endl;
	FREE_OBJECT(C_ago);
}

void compute_and_print_ago_distribution_with_classes(ostream &ost,
	classify_bitvectors *CB, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "compute_and_print_ago_distribution_with_classes" << endl;
		}
	classify *C_ago;
	compute_ago_distribution_permuted(CB, C_ago, verbose_level);
	ost << "Ago distribution: " << endl;
	ost << "$$" << endl;
	C_ago->print_naked_tex(ost, TRUE /* f_backwards */);
	ost << endl;
	ost << "$$" << endl;
	set_of_sets *SoS;
	int *types;
	int nb_types;

	SoS = C_ago->get_set_partition_and_types(types,
			nb_types, verbose_level);


	// go backwards to show large group orders first:
	for (i = SoS->nb_sets - 1; i >= 0; i--) {
		ost << "Group order $" << types[i]
			<< "$ appears for the following $" << SoS->Set_size[i]
			<< "$ classes: $" << endl;
		int_set_print_tex(ost, SoS->Sets[i], SoS->Set_size[i]);
		ost << "$\\\\" << endl;
		//int_vec_print_as_matrix(ost, SoS->Sets[i],
		//SoS->Set_size[i], 10 /* width */, TRUE /* f_tex */);
		//ost << "$$" << endl;

		}

	FREE_int(types);
	FREE_OBJECT(SoS);
	FREE_OBJECT(C_ago);
}

}
