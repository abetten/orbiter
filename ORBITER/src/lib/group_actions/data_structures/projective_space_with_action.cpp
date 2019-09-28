// projective_space_with_action.cpp
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
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {


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

	vector_ge *nice_gens;

	A = NEW_OBJECT(action);
	A->init_linear_group(S,
		F, d, 
		TRUE /*f_projective*/,
		FALSE /* f_general*/,
		FALSE /* f_affine */,
		f_semilinear,
		FALSE /* f_special */,
		nice_gens,
		0 /* verbose_level*/);
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating linear group done" << endl;
		}
	FREE_OBJECT(nice_gens);


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
// added to action_global.cpp on 2/28/2011, called from analyze.cpp
// November 17, 2014 moved here from TOP_LEVEL/extra.cpp
// December 31, 2014, moved here from projective_space.cpp
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
	file_io Fio;

	A_linear = A;

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		cout << "set_size = " << set_size << endl;
		}
#if 0
	if (P->incidence_bitvec == NULL) {
		cout << "projective_space_with_action::set_stabilizer "
				"P->incidence_bitvec == NULL" << endl;
		exit(1);
		}
#endif

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer computing the type of the set" << endl;
		}

	classify C;

	C.init(set, set_size, TRUE, 0);
	if (C.second_nb_types > 1) {
		cout << "projective_space_with_action::set_stabilizer: "
				"The set is a multiset:" << endl;
		C.print(FALSE /*f_backwards*/);
		}

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer The type of the set is:" << endl;
		C.print(FALSE /*f_backwards*/);
		cout << "C.second_nb_types = " << C.second_nb_types << endl;
		}
	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer allocating data" << endl;
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
			cout << "projective_space_with_action::set_stabilizer j=" << j << " f2=" << f2
					<< " l2=" << l2 << " multiplicity=" << m << endl;
			}
		for (h = 0; h < l2; h++) {
			idx = C.second_sorting_perm_inv[f2 + h];
			f = C.type_first[idx];
			l = C.type_len[idx];
			i = C.data_sorted[f + 0];
			if (f_vvv) {
				cout << "projective_space_with_action::set_stabilizer h=" << h << " idx=" << idx
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
			cout << "projective_space_with_action::set_stabilizer too large to print" << endl;
			}

		char fname_csv[1000];
		char fname_bin[1000];

		sprintf(fname_csv, "%sIncma_in_%d_%d.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_bin, "%sIncma_in_%d_%d.bin",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		Fio.int_matrix_write_csv(fname_csv, Incma, nb_rows, nb_cols);

		for (i = 0; i < nb_rows + nb_cols; i++) {
			labeling[i] = i;
			}

		colored_graph *CG;

		CG = NEW_OBJECT(colored_graph);

		CG->create_Levi_graph_from_incidence_matrix(
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

	int t0, t1, dt, tps;
	double delta_t_in_sec;

	tps = os_ticks_per_second();
	t0 = os_ticks();

	nauty_interface_matrix_int(Incma, nb_rows, nb_cols, 
		labeling, partition, 
		Aut, Aut_counter, 
		Base, Base_length, 
		Transversal_length, Ago, verbose_level - 3);

	t1 = os_ticks();
	dt = t1 - t0;
	delta_t_in_sec = (double) t1 / (double) dt;

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer "
				"done with nauty_interface_matrix_int, "
				"Ago=" << Ago << " dt=" << dt
				<< " delta_t_in_sec=" << delta_t_in_sec << endl;
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
		
		Fio.int_vec_write_csv(labeling, N,
				fname_labeling, "canonical labeling");
		Fio.int_matrix_write_csv(fname_csv, Incma_out, nb_rows, nb_cols);

		
		colored_graph *CG;

		CG = NEW_OBJECT(colored_graph);

		CG->create_Levi_graph_from_incidence_matrix(
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
		verbose_level - 2);

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
	gens1->init(A_linear, verbose_level - 2);
	gens1->allocate(gens->len, verbose_level - 2);
	
	Mtx = NEW_int(d * d + 1); // leave space for frobenius
	
	pos = 0;
	for (g = 0; g < gens->len; g++) {
		if (f_vv) {
			cout << "projective_space_with_action::set_stabilizer "
					"strong generator " << g << ":" << endl;
			A_perm->element_print(gens->ith(g), cout);
			cout << endl;
			}
		
		if (A_perm->reverse_engineer_semilinear_map(P,
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
	gens1->reallocate(pos, verbose_level - 2);
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
				"we are now creating the group; before A_linear->create_sims_from_generators_with_target_group_order" << endl;
		}

	S = A_linear->create_sims_from_generators_with_target_group_order(
		gens1, ago, verbose_level - 5);
#if 0
	S = A_linear->create_sims_from_generators_without_target_group_order(
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


void projective_space_with_action::canonical_labeling(
	object_in_projective_space *OiP,
	int *canonical_labeling,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	action *A_linear;
	int *Incma;
	int *partition;
	int nb_rows, nb_cols;
	int *Aut, Aut_counter;
	int *Base, Base_length;
	int *Transversal_length, Ago;
	int N, i, L;

	A_linear = A;

	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling"
				<< endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
#if 0
	if (P->incidence_bitvec == NULL) {
		cout << "projective_space_with_action::canonical_labeling "
				"P->incidence_bitvec == NULL" << endl;
		exit(1);
		}
#endif


	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"before OiP->encode_incma" << endl;
		}
	OiP->encode_incma(Incma, nb_rows, nb_cols,
			partition, verbose_level - 1);
	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"after OiP->encode_incma" << endl;
		}
	if (verbose_level > 5) {
		cout << "projective_space_with_action::canonical_labeling "
				"Incma:" << endl;
		int_matrix_print_tight(Incma, nb_rows, nb_cols);
	}

	//canonical_labeling = NEW_int(nb_rows + nb_cols);
	for (i = 0; i < nb_rows + nb_cols; i++) {
		canonical_labeling[i] = i;
		}


	N = nb_rows + nb_cols;
	L = nb_rows * nb_cols;

	if (f_vv) {
		cout << "projective_space_with_action::canonical_labeling "
				"initializing Aut, Base, "
				"Transversal_length" << endl;
		}
	Aut = NEW_int(N * N);
	Base = NEW_int(N);
	Transversal_length = NEW_int(N);

	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"calling nauty_interface_matrix_int" << endl;
		}


	int t0, t1, dt, tps;
	double delta_t_in_sec;

	tps = os_ticks_per_second();
	t0 = os_ticks();

	nauty_interface_matrix_int(
		Incma, nb_rows, nb_cols,
		canonical_labeling, partition,
		Aut, Aut_counter,
		Base, Base_length,
		Transversal_length, Ago, verbose_level - 3);

	t1 = os_ticks();
	dt = t1 - t0;
	delta_t_in_sec = (double) t1 / (double) dt;

	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"done with nauty_interface_matrix_int, "
				"Ago=" << Ago << " dt=" << dt
				<< " delta_t_in_sec=" << delta_t_in_sec << endl;
		}


	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"done with nauty_interface_matrix_int, "
				"Ago=" << Ago << endl;
		}
	FREE_int(Aut);
	FREE_int(Base);
	FREE_int(Transversal_length);
	FREE_int(Incma);
	FREE_int(partition);
	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling done"
				<< endl;
		}
}

strong_generators
*projective_space_with_action::set_stabilizer_of_object(
	object_in_projective_space *OiP, 
	int f_save_incma_in_and_out,
	const char *save_incma_in_and_out_prefix,
	int f_compute_canonical_form,
	uchar *&canonical_form,
	int &canonical_form_len,
	int *canonical_labeling,
	int verbose_level)
// canonical_labeling[nb_rows, nb_cols]
// where nb_rows and nb_cols is the encoding size ,
// which can be computed using
// object_in_projective_space::encoding_size(
//   int &nb_rows, int &nb_cols,
//   int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	action *A_linear;
	int *Incma;
	int *partition;
	//int *labeling;
	int nb_rows, nb_cols;
	int *Aut, Aut_counter;
	int *Base, Base_length;
	int *Transversal_length, Ago;
	int N, i, j, a, L;
	combinatorics_domain Combi;
	file_io Fio;

	A_linear = A;

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
#if 0
	if (P->incidence_bitvec == NULL) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object P->incidence_bitvec == NULL" << endl;
		exit(1);
		}
#endif


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
	if (verbose_level > 5) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object Incma:" << endl;
		int_matrix_print_tight(Incma, nb_rows, nb_cols);
	}

	//canonical_labeling = NEW_int(nb_rows + nb_cols);
	for (i = 0; i < nb_rows + nb_cols; i++) {
		canonical_labeling[i] = i;
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
		Fio.int_matrix_write_csv(fname_csv, Incma, nb_rows, nb_cols);

		colored_graph *CG;

		CG = NEW_OBJECT(colored_graph);

		CG->create_Levi_graph_from_incidence_matrix(
				Incma, nb_rows, nb_cols,
				TRUE, canonical_labeling, verbose_level);
		CG->save(fname_bin, verbose_level);
		//FREE_int(Incma);
		FREE_OBJECT(CG);
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
	int t0, t1, dt, tps;
	double delta_t_in_sec;

	tps = os_ticks_per_second();
	t0 = os_ticks();

	nauty_interface_matrix_int(
		Incma, nb_rows, nb_cols,
		canonical_labeling, partition,
		Aut, Aut_counter, 
		Base, Base_length, 
		Transversal_length, Ago, verbose_level - 3);

	t1 = os_ticks();
	dt = t1 - t0;
	delta_t_in_sec = (double) dt / (double) tps;

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object done with nauty_interface_matrix_int, "
				"Ago=" << Ago << " dt=" << dt
				<< " delta_t_in_sec=" << delta_t_in_sec << endl;
		}
	if (verbose_level > 5) {
		int h;
		int degree = nb_rows +  nb_cols;

		for (h = 0; h < Aut_counter; h++) {
			cout << "aut generator " << h << " / "
					<< Aut_counter << " : " << endl;
			Combi.perm_print(cout, Aut + h * degree, degree);
			cout << endl;
		}
	}

	int *Incma_out;
	int ii, jj;
	if (f_vvv) {
		cout << "projective_space_with_action::set_stabilizer_"
				"of_object labeling:" << endl;
		int_vec_print(cout, canonical_labeling, nb_rows + nb_cols);
		cout << endl;
		}

	Incma_out = NEW_int(L);
	for (i = 0; i < nb_rows; i++) {
		ii = canonical_labeling[i];
		for (j = 0; j < nb_cols; j++) {
			jj = canonical_labeling[nb_rows + j] - nb_rows;
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
		latex_interface L;

		sprintf(fname_labeling, "%slabeling_%d_%d.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_csv, "%sIncma_out_%d_%d.csv",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		sprintf(fname_bin, "%sIncma_out_%d_%d.bin",
				save_incma_in_and_out_prefix, nb_rows, nb_cols);
		
		cout << "labeling:" << endl;
		L.int_vec_print_as_matrix(cout,
				canonical_labeling, N, 10 /* width */, TRUE /* f_tex */);

		Fio.int_vec_write_csv(canonical_labeling, N,
				fname_labeling, "canonical labeling");
		Fio.int_matrix_write_csv(fname_csv, Incma_out, nb_rows, nb_cols);

		
		colored_graph *CG;

		CG = NEW_OBJECT(colored_graph);

		CG->create_Levi_graph_from_incidence_matrix(
				Incma_out, nb_rows, nb_cols,
				TRUE, canonical_labeling, verbose_level);
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
		verbose_level);

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
	gens1->init(A_linear, verbose_level - 2);
	gens1->allocate(gens->len, verbose_level - 2);
	
	Mtx = NEW_int(d * d + 1); // leave space for frobenius
	
	pos = 0;
	for (g = 0; g < gens->len; g++) {
		if (f_vv) {
			cout << "projective_space_with_action::set_stabilizer_"
					"of_object strong generator " << g << ":" << endl;
			//A_perm->element_print(gens->ith(g), cout);
			cout << endl;
			}
		
		if (A_perm->reverse_engineer_semilinear_map(P,
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
	gens1->reallocate(pos, verbose_level - 2);
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

	S = A_linear->create_sims_from_generators_with_target_group_order(
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
#if 0
	if (f_v) {
		cout << "before freeing labeling" << endl;
	}
	FREE_int(labeling);
#endif
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
			0 /*verbose_level*/);
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
			0 /*verbose_level*/);
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
			0 /*verbose_level*/);
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
			0 /*verbose_level*/);
	Sch1->print_orbit_lengths_tex(ost);


	//ost << "Orbits on lines:\\\\" << endl;

	Sch2 = NEW_OBJECT(schreier);
	A_on_lines->all_point_orbits_from_single_generator(*Sch2,
			Elt,
			0 /*verbose_level*/);
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
		0 /*verbose_level*/);
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
	Sch1->get_orbit_partition(S1, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_with_action::report_"
				"decomposition_by_single_automorphism "
				"before Sch2->get_orbit_partition" << endl;
		}
	Sch2->get_orbit_partition(S2, 0 /*verbose_level*/);
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
		Stack->split_cell(
				S1.pointList + S1.startCell[i],
				S1.cellSize[i], verbose_level);
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
		Stack->split_cell(set, sz, 0 /*verbose_level*/);
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


object_in_projective_space *
projective_space_with_action::create_object_from_string(
	int type, const char *input_fname, int input_idx,
	const char *set_as_string, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::create_object_from_string" << endl;
		cout << "type=" << type << endl;
		}


	object_in_projective_space *OiP;

	OiP = NEW_OBJECT(object_in_projective_space);

	OiP->init_object_from_string(P,
			type, input_fname, input_idx,
			set_as_string, verbose_level);


	if (f_v) {
		cout << "projective_space_with_action::create_object_from_string"
				" done" << endl;
		}
	return OiP;
}

object_in_projective_space *
projective_space_with_action::create_object_from_int_vec(
	int type, const char *input_fname, int input_idx,
	int *the_set, int set_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::create_object_from_int_vec" << endl;
		cout << "type=" << type << endl;
		}


	object_in_projective_space *OiP;

	OiP = NEW_OBJECT(object_in_projective_space);

	OiP->init_object_from_int_vec(P,
			type, input_fname, input_idx,
			the_set, set_sz, verbose_level);


	if (f_v) {
		cout << "projective_space_with_action::create_object_from_int_vec"
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
	int *canonical_labeling,
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
		canonical_labeling,
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
	classify_bitvectors *CB,
	int f_save_incma_in_and_out, const char *prefix,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int input_idx, ret;
	int t0, t1, dt;
	file_io Fio;

	if (f_v) {
		cout << "projective_space_with_action::classify_objects_using_nauty" << endl;
		}

	int nb_objects_to_test;

	if (f_v) {
		cout << "projective_space_with_action::classify_objects_using_nauty "
				"before count_number_of_objects_to_test" << endl;
		}
	nb_objects_to_test = Data->count_number_of_objects_to_test(
		verbose_level - 1);

	t0 = os_ticks();

	for (input_idx = 0; input_idx < Data->nb_inputs; input_idx++) {
		cout << "projective_space_with_action::classify_objects_using_nauty input " << input_idx << " / " << Data->nb_inputs
			<< " is:" << endl;

		if (Data->input_type[input_idx] == INPUT_TYPE_SET_OF_POINTS) {
			cout << "projective_space_with_action::classify_objects_using_nauty input set of points "
				<< Data->input_string[input_idx] << ":" << endl;

			object_in_projective_space *OiP;
			strong_generators *SG;

			OiP = create_object_from_string(t_PTS,
					"command_line", CB->n,
					Data->input_string[input_idx], verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::classify_objects_using_nauty "
						"before process_object" << endl;
				}
			int nb_rows, nb_cols;
			int *canonical_labeling;

			OiP->encoding_size(
					nb_rows, nb_cols,
					verbose_level);
			canonical_labeling = NEW_int(nb_rows + nb_cols);

			ret = process_object(CB, OiP,
					f_save_incma_in_and_out, prefix,
					nb_objects_to_test,
					SG,
					canonical_labeling,
					verbose_level);

			FREE_int(canonical_labeling);

			if (f_v) {
				cout << "projective_space_with_action::classify_objects_using_nauty "
						"after process_object INPUT_TYPE_SET_OF_POINTS, ret=" << ret << endl;
				}


			if (!ret) {
				cout << "before FREE_OBJECT(SG)" << endl;
				FREE_OBJECT(SG);
				cout << "before FREE_OBJECT(OiP)" << endl;
				FREE_OBJECT(OiP);
				//cout << "before FREE_OBJECT(canonical_labeling)" << endl;
				//FREE_int(canonical_labeling);
				//cout << "after FREE_OBJECT(canonical_labeling)" << endl;
				}
			else {
				cout << "projective_space_with_action::classify_objects_using_nauty New isomorphism type! The n e w number of "
					"isomorphism types is " << CB->nb_types << endl;
				int idx;

				object_in_projective_space_with_action *OiPA;

				OiPA = NEW_OBJECT(object_in_projective_space_with_action);

				OiPA->init(OiP, SG, nb_rows, nb_cols,
						canonical_labeling, verbose_level);
				idx = CB->type_of[CB->n - 1];
				CB->Type_extra_data[idx] = OiPA;

				//compute_and_print_ago_distribution(cout,
				//		CB, verbose_level);
				}
			}
		else if (Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINT_SET) {
			cout << "projective_space_with_action::classify_objects_using_nauty input set of points from file "
				<< Data->input_string[input_idx] << ":" << endl;

			object_in_projective_space *OiP;
			strong_generators *SG;
			int *the_set;
			int set_size;

			Fio.read_set_from_file(Data->input_string[input_idx],
				the_set, set_size, verbose_level);

			OiP = create_object_from_int_vec(t_PTS,
					Data->input_string[input_idx], CB->n,
					the_set, set_size, verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::classify_objects_using_nauty "
						"before encoding_size" << endl;
				}
			int nb_rows, nb_cols;
			int *canonical_labeling;

			OiP->encoding_size(
					nb_rows, nb_cols,
					verbose_level);
			canonical_labeling = NEW_int(nb_rows + nb_cols);

			if (f_v) {
				cout << "projective_space_with_action::classify_objects_using_nauty "
						"before process_object" << endl;
				}
			ret = process_object(CB, OiP,
					f_save_incma_in_and_out, prefix,
					nb_objects_to_test,
					SG,
					canonical_labeling,
					verbose_level);

			FREE_int(canonical_labeling);

			if (f_v) {
				cout << "projective_space_with_action::classify_objects_using_nauty "
						"after process_object INPUT_TYPE_FILE_OF_POINT_SET, ret=" << ret << endl;
				}


			if (!ret) {
				FREE_OBJECT(SG);
				FREE_OBJECT(OiP);
				//FREE_int(canonical_labeling);
				}
			else {
				cout << "projective_space_with_action::classify_objects_using_nauty New isomorphism type! The n e w number of "
					"isomorphism types is " << CB->nb_types << endl;
				int idx;

				object_in_projective_space_with_action *OiPA;

				OiPA = NEW_OBJECT(object_in_projective_space_with_action);

				OiPA->init(OiP, SG, nb_rows, nb_cols,
						canonical_labeling, verbose_level);
				idx = CB->type_of[CB->n - 1];
				CB->Type_extra_data[idx] = OiPA;

				//compute_and_print_ago_distribution(cout,
				//		CB, verbose_level);
				}
			}
		else if (Data->input_type[input_idx] == INPUT_TYPE_SET_OF_LINES) {
			cout << "projective_space_with_action::classify_objects_using_nauty input set of lines " << Data->input_string[input_idx]
				<< ":" << endl;

			object_in_projective_space *OiP;
			strong_generators *SG;

			OiP = create_object_from_string(t_LNS,
					"command_line", CB->n,
					Data->input_string[input_idx], verbose_level);

			int nb_rows, nb_cols;
			int *canonical_labeling;

			OiP->encoding_size(
					nb_rows, nb_cols,
					verbose_level);
			canonical_labeling = NEW_int(nb_rows + nb_cols);


			if (!process_object(CB, OiP,
				f_save_incma_in_and_out, prefix,
				nb_objects_to_test,
				SG,
				canonical_labeling,
				verbose_level)) {

				FREE_OBJECT(SG);
				FREE_OBJECT(OiP);
				FREE_int(canonical_labeling);
				}
			else {
				cout << "projective_space_with_action::classify_objects_using_nauty New isomorphism type! The n e w number of "
						"isomorphism types is " << CB->nb_types << endl;
				int idx;

				object_in_projective_space_with_action *OiPA;

				OiPA = NEW_OBJECT(object_in_projective_space_with_action);

				OiPA->init(OiP, SG, nb_rows, nb_cols,
						canonical_labeling, verbose_level);
				idx = CB->type_of[CB->n - 1];
				CB->Type_extra_data[idx] = OiPA;

				//compute_and_print_ago_distribution(cout,
				//	CB, verbose_level);
				}
			}
		else if (Data->input_type[input_idx] == INPUT_TYPE_SET_OF_PACKING) {
			cout << "projective_space_with_action::classify_objects_using_nauty input set of packing "
				<< Data->input_string[input_idx] << ":" << endl;

			object_in_projective_space *OiP;
			strong_generators *SG;

			OiP = create_object_from_string(t_PAC,
					"command_line", CB->n,
					Data->input_string[input_idx], verbose_level);

			int nb_rows, nb_cols;
			int *canonical_labeling;

			OiP->encoding_size(
					nb_rows, nb_cols,
					verbose_level);
			canonical_labeling = NEW_int(nb_rows + nb_cols);

			if (!process_object(CB, OiP,
				f_save_incma_in_and_out, prefix,
				nb_objects_to_test,
				SG,
				canonical_labeling,
				verbose_level)) {

				FREE_OBJECT(SG);
				FREE_OBJECT(OiP);
				FREE_int(canonical_labeling);
				}
			else {
				cout << "projective_space_with_action::classify_objects_using_nauty New isomorphism type! The n e w number of "
					"isomorphism types is " << CB->nb_types << endl;
				int idx;

				object_in_projective_space_with_action *OiPA;

				OiPA = NEW_OBJECT(object_in_projective_space_with_action);

				OiPA->init(OiP, SG, nb_rows, nb_cols,
						canonical_labeling, verbose_level);
				idx = CB->type_of[CB->n - 1];
				CB->Type_extra_data[idx] = OiPA;

				//compute_and_print_ago_distribution(cout,
				//		CB, verbose_level);
				}
			}
		else if (Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINTS ||
				Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_LINES ||
				Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_PACKINGS ||
				Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
			cout << "projective_space_with_action::classify_objects_using_nauty input from file " << Data->input_string[input_idx]
				<< ":" << endl;

			set_of_sets *SoS;

			SoS = NEW_OBJECT(set_of_sets);

			cout << "projective_space_with_action::classify_objects_using_nauty Reading the file " << Data->input_string[input_idx] << endl;
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
				cout << "projective_space_with_action::classify_objects_using_nauty Reading spread table from file "
					<< Data->input_string2[input_idx] << endl;
				Fio.int_matrix_read_csv(Data->input_string2[input_idx],
						Spread_table, nb_spreads, spread_size,
						0 /* verbose_level */);
				cout << "Reading spread table from file "
						<< Data->input_string2[input_idx] << " done" << endl;
				cout << "The spread table contains " << nb_spreads
						<< " spreads" << endl;
				}

			cout << "projective_space_with_action::classify_objects_using_nauty processing " << SoS->nb_sets << " objects" << endl;

			for (h = 0; h < SoS->nb_sets; h++) {


				int *the_set_in;
				int set_size_in;
				object_in_projective_space *OiP;


				set_size_in = SoS->Set_size[h];
				the_set_in = SoS->Sets[h];

				if (f_vv || ((h % 1024) == 0)) {
					cout << "projective_space_with_action::classify_objects_using_nauty The input set " << h << " / " << SoS->nb_sets
						<< " has size " << set_size_in << ":" << endl;
					}

				if (f_vvv) {
					cout << "projective_space_with_action::classify_objects_using_nauty The input set is:" << endl;
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
					cout << "projective_space_with_action::classify_objects_using_nauty unknown type" << endl;
					exit(1);
					}

				stringstream sstr;
				//sstr << set_size_in;
				for (int k = 0; k < set_size_in; k++) {
					sstr << the_set_in[k];
					if (k < set_size_in - 1) {
						sstr << ",";
					}
				}
				string s = sstr.str();
				//cout << s << endl;
				//Ago_text[clique_no] = NEW_char(strlen(s.c_str()) + 1);
				//strcpy(Ago_text[clique_no], s.c_str());

				OiP->input_fname = Data->input_string[input_idx];
				OiP->input_idx = h;

				int l = strlen(s.c_str());

				OiP->set_as_string = NEW_char(l + 1);
				strcpy(OiP->set_as_string, s.c_str());



				strong_generators *SG;


				int nb_rows, nb_cols;
				int *canonical_labeling;

				OiP->encoding_size(
						nb_rows, nb_cols,
						verbose_level);
				canonical_labeling = NEW_int(nb_rows + nb_cols);


				if (!process_object(CB, OiP,
					f_save_incma_in_and_out, prefix,
					nb_objects_to_test,
					SG,
					canonical_labeling,
					verbose_level - 3)) {

					FREE_OBJECT(OiP);
					FREE_OBJECT(SG);
					FREE_int(canonical_labeling);
					}
				else {
					t1 = os_ticks();
					//cout << "poset_classification::print_level_info t0=" << t0 << endl;
					//cout << "poset_classification::print_level_info t1=" << t1 << endl;
					dt = t1 - t0;
					//cout << "poset_classification::print_level_info dt=" << dt << endl;

					cout << "Time ";
					time_check_delta(cout, dt);
					cout << " --- New isomorphism type! input set " << h
							<< " / " << SoS->nb_sets << " The n e w number of "
							"isomorphism types is " << CB->nb_types << endl;

					int idx;

					object_in_projective_space_with_action *OiPA;

					OiPA = NEW_OBJECT(object_in_projective_space_with_action);

					OiPA->init(OiP, SG, nb_rows, nb_cols,
							canonical_labeling, verbose_level);
					idx = CB->type_of[CB->n - 1];
					CB->Type_extra_data[idx] = OiPA;


					}

				if (f_vv) {
					cout << "projective_space_with_action::classify_objects_using_nauty after input set " << h << " / "
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
			cout << "projective_space_with_action::classify_objects_using_nauty unknown input type" << endl;
			exit(1);
			}
		}

	if (f_v) {
		cout << "projective_space_with_action::classify_objects_using_nauty before compute_and_print_ago_distribution" << endl;
	}

	compute_and_print_ago_distribution(cout,
			CB, verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::classify_objects_using_nauty after compute_and_print_ago_distribution" << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::classify_objects_using_nauty before CB->finalize" << endl;
	}

	CB->finalize(verbose_level); // computes C_type_of and perm

	if (f_v) {
		cout << "projective_space_with_action::classify_objects_using_nauty done" << endl;
		}
}


void projective_space_with_action::save(
		const char *output_prefix,
		classify_bitvectors *CB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname[1000];
	file_io Fio;

	if (f_v) {
		cout << "projective_space_with_action::save" << endl;
	}
	sprintf(fname, "%s_classified.cvs", output_prefix);

	{
		ofstream fp(fname);
		int i, j;

		fp << "rep,ago,original_file,input_idx,input_set,"
				"nb_rows,nb_cols,canonical_form" << endl;
		for (i = 0; i < CB->nb_types; i++) {

			object_in_projective_space_with_action *OiPA;
			object_in_projective_space *OiP;

			//cout << i << " / " << CB->nb_types << " is "
			//	<< CB->Type_rep[i] << " : " << CB->Type_mult[i] << " : ";
			OiPA = (object_in_projective_space_with_action *)
					CB->Type_extra_data[i];
			OiP = OiPA->OiP;
			if (OiP == NULL) {
				cout << "OiP == NULL" << endl;
				exit(1);
			}
			if (OiP->type != t_PAC) {
				//OiP->print(cout);
				}
			//OiP->print(cout);

	#if 0
			for (j = 0; j < rep_len; j++) {
				cout << (int) Type_data[i][j];
				if (j < rep_len - 1) {
					cout << ", ";
					}
				}
	#endif
			//cout << "before writing OiP->set_as_string:" << endl;
			const char *p = "";

			if (OiP->set_as_string) {
				p = OiP->set_as_string;
			}

			int ago;

			if (OiP->f_has_known_ago) {
				ago = OiP->known_ago;
			}
			else {
				ago = OiPA->Aut_gens->group_order_as_int();
			}
			fp << i << "," << ago
					<< "," << OiP->input_fname
					<< "," << OiP->input_idx
					<< ",\"" << p << "\",";
			//cout << "before writing OiPA->nb_rows:" << endl;
			fp << OiPA->nb_rows << "," << OiPA->nb_cols<< ",";

			//cout << "before writing canonical labeling:" << endl;
			fp << "\"";
			for (j = 0; j < OiPA->nb_rows + OiPA->nb_cols; j++) {
				fp << OiPA->canonical_labeling[j];
				if (j < OiPA->nb_rows + OiPA->nb_cols - 1) {
					fp << ",";
				}
			}
			fp << "\"";
			fp << endl;
			}
		fp << "END" << endl;
	}
	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "projective_space_with_action::save done" << endl;
	}
}

void projective_space_with_action::merge_packings(
		const char **fnames, int nb_files,
		const char *file_of_spreads,
		classify_bitvectors *&CB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "projective_space_with_action::merge_packings" << endl;
	}

	CB = NEW_OBJECT(classify_bitvectors);


	// for use if INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE
	int *Spread_table;
	int nb_spreads;
	int spread_size;

	if (f_v) {
		cout << "projective_space_with_action::merge_packings "
				"Reading spread table from file "
				<< file_of_spreads << endl;
	}
	Fio.int_matrix_read_csv(file_of_spreads,
			Spread_table, nb_spreads, spread_size,
			0 /* verbose_level */);
	if (f_v) {
		cout << "Reading spread table from file "
				<< file_of_spreads << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
	}

	int f, g, N, table_length, nb_reject = 0;

	N = 0;

	if (f_v) {
		cout << "projective_space_with_action::merge_packings "
				"counting the overall number of input packings" << endl;
	}

	for (f = 0; f < nb_files; f++) {

		if (f_v) {
			cout << "projective_space_with_action::merge_packings file "
					<< f << " / " << nb_files << " : " << fnames[f] << endl;
		}

		spreadsheet *S;

		S = NEW_OBJECT(spreadsheet);

		S->read_spreadsheet(fnames[f], 0 /*verbose_level*/);

		table_length = S->nb_rows - 1;
		N += table_length;



		FREE_OBJECT(S);

	}

	if (f_v) {
		cout << "projective_space_with_action::merge_packings file "
				<< "we have " << N << " packings in "
				<< nb_files << " files" << endl;
	}

	for (f = 0; f < nb_files; f++) {

		if (f_v) {
			cout << "projective_space_with_action::merge_packings file "
					<< f << " / " << nb_files << " : " << fnames[f] << endl;
		}

		spreadsheet *S;

		S = NEW_OBJECT(spreadsheet);

		S->read_spreadsheet(fnames[f], 0 /*verbose_level*/);
		if (FALSE /*f_v3*/) {
			S->print_table(cout, FALSE);
			}

		int ago_idx, original_file_idx, input_idx_idx, input_set_idx;
		int nb_rows_idx, nb_cols_idx, canonical_form_idx;

		ago_idx = S->find_by_column("ago");
		original_file_idx = S->find_by_column("original_file");
		input_idx_idx = S->find_by_column("input_idx");
		input_set_idx = S->find_by_column("input_set");
		nb_rows_idx = S->find_by_column("nb_rows");
		nb_cols_idx = S->find_by_column("nb_cols");
		canonical_form_idx = S->find_by_column("canonical_form");

		table_length = S->nb_rows - 1;

		//rep,ago,original_file,input_idx,input_set,nb_rows,nb_cols,canonical_form


		for (g = 0; g < table_length; g++) {

			int ago;
			char *text;
			int *the_set_in;
			int set_size_in;
			int *canonical_labeling;
			int canonical_labeling_sz;
			int nb_rows, nb_cols;
			object_in_projective_space *OiP;


			ago = S->get_int(g + 1, ago_idx);
			nb_rows = S->get_int(g + 1, nb_rows_idx);
			nb_cols = S->get_int(g + 1, nb_cols_idx);

			text = S->get_string(g + 1, input_set_idx);
			int_vec_scan(text, the_set_in, set_size_in);


			if (f_v) {
				cout << "File " << f << " / " << nb_files
						<< ", input set " << g << " / "
						<< table_length << endl;
				//int_vec_print(cout, the_set_in, set_size_in);
				//cout << endl;
				}

			if (FALSE) {
				cout << "canonical_form_idx=" << canonical_form_idx << endl;
			}
			text = S->get_string(g + 1, canonical_form_idx);
			if (FALSE) {
				cout << "text=" << text << endl;
			}
			int_vec_scan(text, canonical_labeling, canonical_labeling_sz);
			if (FALSE) {
				cout << "File " << f << " / " << nb_files
						<< ", input set " << g << " / "
						<< table_length << " canonical_labeling = ";
				int_vec_print(cout, canonical_labeling, canonical_labeling_sz);
				cout << endl;
				}

			if (canonical_labeling_sz != nb_rows + nb_cols) {
				cout << "projective_space_with_action::merge_packings "
						"canonical_labeling_sz != nb_rows + nb_cols" << endl;
				exit(1);
			}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (FALSE) {
				cout << "projective_space_with_action::merge_packings "
						"before init_packing_from_spread_table" << endl;
			}
			OiP->init_packing_from_spread_table(P, the_set_in,
				Spread_table, nb_spreads, spread_size,
				0 /*verbose_level*/);
			if (FALSE) {
				cout << "projective_space_with_action::merge_packings "
						"after init_packing_from_spread_table" << endl;
			}
			OiP->f_has_known_ago = TRUE;
			OiP->known_ago = ago;

			int *Incma_in;
			int *Incma_out;
			int nb_rows1, nb_cols1;
			int *partition;
			uchar *canonical_form;
			int canonical_form_len;


			if (FALSE) {
				cout << "projective_space_with_action::merge_packings "
						"before encode_incma" << endl;
			}
			OiP->encode_incma(Incma_in, nb_rows1, nb_cols1,
					partition, 0 /*verbose_level - 1*/);
			if (FALSE) {
				cout << "projective_space_with_action::merge_packings "
						"after encode_incma" << endl;
			}
			if (nb_rows1 != nb_rows) {
				cout << "projective_space_with_action::merge_packings "
						"nb_rows1 != nb_rows" << endl;
				exit(1);
			}
			if (nb_cols1 != nb_cols) {
				cout << "projective_space_with_action::merge_packings "
						"nb_cols1 != nb_cols" << endl;
				exit(1);
			}

			OiP->input_fname = S->get_string(g + 1, original_file_idx);
			OiP->input_idx = S->get_int(g + 1, input_idx_idx);

			text = S->get_string(g + 1, input_set_idx);
			int l = strlen(text);

			OiP->set_as_string = NEW_char(l + 1);
			strcpy(OiP->set_as_string, text);

			int i, j, ii, jj, a, ret;
			int L = nb_rows * nb_cols;

			Incma_out = NEW_int(L);
			for (i = 0; i < nb_rows; i++) {
				ii = canonical_labeling[i];
				for (j = 0; j < nb_cols; j++) {
					jj = canonical_labeling[nb_rows + j] - nb_rows;
					//cout << "i=" << i << " j=" << j << " ii=" << ii
					//<< " jj=" << jj << endl;
					Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
					}
				}
			if (FALSE) {
				cout << "projective_space_with_action::merge_packings "
						"before bitvector_allocate_and_coded_length" << endl;
			}
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

			if (CB->n == 0) {
				if (f_v) {
					cout << "projective_space_with_action::merge_packings "
							"before CB->init" << endl;
				}
				CB->init(N, canonical_form_len, verbose_level);
				}
			if (f_v) {
				cout << "projective_space_with_action::merge_packings "
						"before CB->add" << endl;
			}
			ret = CB->add(canonical_form, OiP, 0 /*verbose_level*/);
			if (ret == 0) {
				nb_reject++;
			}
			if (f_v) {
				cout << "projective_space_with_action::merge_packings "
						"CB->add returns " << ret
						<< " nb iso = " << CB->nb_types
						<< " nb_reject=" << nb_reject << endl;
			}


			int idx;

			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init_known_ago(OiP, ago, nb_rows, nb_cols,
					canonical_labeling, 0 /*verbose_level*/);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;


			FREE_int(the_set_in);
			//FREE_int(canonical_labeling);
			FREE_int(Incma_in);
			FREE_int(Incma_out);
			FREE_int(partition);
			//FREE_uchar(canonical_form);

		} // next g



	} // next f

	if (f_v) {
		cout << "projective_space_with_action::merge_packings done, "
				"we found " << CB->nb_types << " isomorphism types "
				"of packings" << endl;
		}


	//FREE_OBJECT(CB);
	FREE_int(Spread_table);

	if (f_v) {
		cout << "projective_space_with_action::merge_packings done" << endl;
	}
}

void projective_space_with_action::select_packings(
		const char *fname,
		const char *file_of_spreads_original,
		spread_tables *Spread_tables,
		int f_self_polar,
		int f_ago, int select_ago,
		classify_bitvectors *&CB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_accept = 0;
	file_io Fio;

	if (f_v) {
		cout << "projective_space_with_action::select_packings" << endl;
	}

	CB = NEW_OBJECT(classify_bitvectors);



	int *Spread_table;
	int nb_spreads;
	int spread_size;
	int packing_size;
	int a, b;

	if (f_v) {
		cout << "projective_space_with_action::select_packings "
				"Reading spread table from file "
				<< file_of_spreads_original << endl;
	}
	Fio.int_matrix_read_csv(file_of_spreads_original,
			Spread_table, nb_spreads, spread_size,
			0 /* verbose_level */);
	if (nb_spreads != Spread_tables->nb_spreads) {
		cout << "projective_space_with_action::select_packings "
				"nb_spreads != Spread_tables->nb_spreads" << endl;
		exit(1);
	}
	if (spread_size != Spread_tables->spread_size) {
		cout << "projective_space_with_action::select_packings "
				"spread_size != Spread_tables->spread_size" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Reading spread table from file "
				<< file_of_spreads_original << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
	}



	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s" << endl;
	}

	int *s2l, *l2s;
	int i, idx;
	int *set;
	int extra_data[1];
	sorting Sorting;

	extra_data[0] = spread_size;

	set = NEW_int(spread_size);
	s2l = NEW_int(nb_spreads);
	l2s = NEW_int(nb_spreads);
	for (i = 0; i < nb_spreads; i++) {
		int_vec_copy(Spread_tables->spread_table +
				i * spread_size, set, spread_size);
		Sorting.int_vec_heapsort(set, spread_size);
		if (!Sorting.search_general(Spread_tables->spread_table,
				nb_spreads, set, idx,
				table_of_sets_compare_func,
				extra_data, 0 /*verbose_level*/)) {
			cout << "projective_space_with_action::select_packings "
					"cannot find spread " << i << " = ";
			int_vec_print(cout, set, spread_size);
			cout << endl;
			exit(1);
		}
		s2l[i] = idx;
		l2s[idx] = i;
	}
	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s done" << endl;
	}

	int g, table_length, nb_reject = 0;


	if (f_v) {
		cout << "projective_space_with_action::select_packings file "
				<< fname << endl;
	}

	spreadsheet *S;

	S = NEW_OBJECT(spreadsheet);

	S->read_spreadsheet(fname, 0 /*verbose_level*/);
	if (FALSE /*f_v3*/) {
		S->print_table(cout, FALSE);
		}

	int ago_idx, original_file_idx, input_idx_idx, input_set_idx;
	int nb_rows_idx, nb_cols_idx, canonical_form_idx;

	ago_idx = S->find_by_column("ago");
	original_file_idx = S->find_by_column("original_file");
	input_idx_idx = S->find_by_column("input_idx");
	input_set_idx = S->find_by_column("input_set");
	nb_rows_idx = S->find_by_column("nb_rows");
	nb_cols_idx = S->find_by_column("nb_cols");
	canonical_form_idx = S->find_by_column("canonical_form");

	table_length = S->nb_rows - 1;

	//rep,ago,original_file,input_idx,
	//input_set,nb_rows,nb_cols,canonical_form
	int f_first = TRUE;


	for (g = 0; g < table_length; g++) {

		int ago;
		char *text;
		int *the_set_in;
		int set_size_in;
		int *canonical_labeling;
		int canonical_labeling_sz;
		int nb_rows, nb_cols;
		object_in_projective_space *OiP;
		int f_accept;
		int *set1;
		int *set2;

		ago = S->get_int(g + 1, ago_idx);
		nb_rows = S->get_int(g + 1, nb_rows_idx);
		nb_cols = S->get_int(g + 1, nb_cols_idx);

		text = S->get_string(g + 1, input_set_idx);
		int_vec_scan(text, the_set_in, set_size_in);

		packing_size = set_size_in;

		if (f_v && (g % 1000) == 0) {
			cout << "File " << fname
					<< ", input set " << g << " / "
					<< table_length << endl;
			//int_vec_print(cout, the_set_in, set_size_in);
			//cout << endl;
			}


		if (f_self_polar) {
			set1 = NEW_int(packing_size);
			set2 = NEW_int(packing_size);

			// test if self-polar:
			for (i = 0; i < packing_size; i++) {
				a = the_set_in[i];
				b = s2l[a];
				set1[i] = b;
			}
			Sorting.int_vec_heapsort(set1, packing_size);
			for (i = 0; i < packing_size; i++) {
				a = set1[i];
				b = Spread_tables->dual_spread_idx[a];
				set2[i] = b;
			}
			Sorting.int_vec_heapsort(set2, packing_size);

#if 0
			cout << "set1: ";
			int_vec_print(cout, set1, packing_size);
			cout << endl;
			cout << "set2: ";
			int_vec_print(cout, set2, packing_size);
			cout << endl;
#endif
			if (int_vec_compare(set1, set2, packing_size) == 0) {
				cout << "The packing is self-polar" << endl;
				f_accept = TRUE;
			}
			else {
				f_accept = FALSE;
			}
			FREE_int(set1);
			FREE_int(set2);
		}
		if (f_ago) {
			if (ago == select_ago) {
				f_accept = TRUE;
			}
			else {
				f_accept = FALSE;
			}
		}



		if (f_accept) {

			nb_accept++;


			if (FALSE) {
				cout << "canonical_form_idx=" << canonical_form_idx << endl;
			}
			text = S->get_string(g + 1, canonical_form_idx);
			if (FALSE) {
				cout << "text=" << text << endl;
			}
			int_vec_scan(text, canonical_labeling, canonical_labeling_sz);
			if (FALSE) {
				cout << "File " << fname
						<< ", input set " << g << " / "
						<< table_length << " canonical_labeling = ";
				int_vec_print(cout, canonical_labeling, canonical_labeling_sz);
				cout << endl;
				}

			if (canonical_labeling_sz != nb_rows + nb_cols) {
				cout << "projective_space_with_action::select_packings "
						"canonical_labeling_sz != nb_rows + nb_cols" << endl;
				exit(1);
			}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (FALSE) {
				cout << "projective_space_with_action::select_packings "
						"before init_packing_from_spread_table" << endl;
			}
			OiP->init_packing_from_spread_table(P, the_set_in,
				Spread_table, nb_spreads, spread_size,
				0 /*verbose_level*/);
			if (FALSE) {
				cout << "projective_space_with_action::merge_packings "
						"after init_packing_from_spread_table" << endl;
			}
			OiP->f_has_known_ago = TRUE;
			OiP->known_ago = ago;

			int *Incma_in;
			int *Incma_out;
			int nb_rows1, nb_cols1;
			int *partition;
			uchar *canonical_form;
			int canonical_form_len;


			if (FALSE) {
				cout << "projective_space_with_action::select_packings "
						"before encode_incma" << endl;
			}
			OiP->encode_incma(Incma_in, nb_rows1, nb_cols1,
					partition, 0 /*verbose_level - 1*/);
			if (FALSE) {
				cout << "projective_space_with_action::select_packings "
						"after encode_incma" << endl;
			}
			if (nb_rows1 != nb_rows) {
				cout << "projective_space_with_action::select_packings "
						"nb_rows1 != nb_rows" << endl;
				exit(1);
			}
			if (nb_cols1 != nb_cols) {
				cout << "projective_space_with_action::select_packings "
						"nb_cols1 != nb_cols" << endl;
				exit(1);
			}

			OiP->input_fname = S->get_string(g + 1, original_file_idx);
			OiP->input_idx = S->get_int(g + 1, input_idx_idx);

			text = S->get_string(g + 1, input_set_idx);
			int l = strlen(text);

			OiP->set_as_string = NEW_char(l + 1);
			strcpy(OiP->set_as_string, text);

			int i, j, ii, jj, a, ret;
			int L = nb_rows * nb_cols;

			Incma_out = NEW_int(L);
			for (i = 0; i < nb_rows; i++) {
				ii = canonical_labeling[i];
				for (j = 0; j < nb_cols; j++) {
					jj = canonical_labeling[nb_rows + j] - nb_rows;
					//cout << "i=" << i << " j=" << j << " ii=" << ii
					//<< " jj=" << jj << endl;
					Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
					}
				}
			if (FALSE) {
				cout << "projective_space_with_action::select_packings "
						"before bitvector_allocate_and_coded_length" << endl;
			}
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

			if (f_first) {
				if (f_v) {
					cout << "projective_space_with_action::select_packings "
							"before CB->init" << endl;
				}
				CB->init(table_length, canonical_form_len, verbose_level);
				f_first = FALSE;
			}


			if (f_v) {
				cout << "projective_space_with_action::select_packings "
						"before CB->add" << endl;
			}

			ret = CB->add(canonical_form, OiP, verbose_level);
			if (ret == 0) {
				cout << "reject" << endl;
				nb_reject++;
			}
			if (f_v) {
				cout << "projective_space_with_action::select_packings "
						"CB->add returns " << ret
						<< " nb iso = " << CB->nb_types
						<< " nb_reject=" << nb_reject
						<< " nb_accept=" << nb_accept
						<< " CB->n=" << CB->n
						<< " CB->nb_types=" << CB->nb_types
						<< endl;
			}


			int idx;

			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init_known_ago(OiP, ago, nb_rows, nb_cols,
					canonical_labeling, 0 /*verbose_level*/);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;

			FREE_int(Incma_in);
			FREE_int(Incma_out);
			FREE_int(partition);
			//FREE_int(canonical_labeling);
			//FREE_uchar(canonical_form);
		} // if (f_accept)



		FREE_int(the_set_in);

	} // next g




	if (f_v) {
		cout << "projective_space_with_action::select_packings done, "
				"we found " << CB->nb_types << " isomorphism types "
				"of packings. nb_accept = " << nb_accept
				<< " CB->n = " << CB->n
				<< " CB->nb_types = " << CB->nb_types
				<< endl;
		}


	//FREE_OBJECT(CB);
	FREE_int(Spread_table);

	if (f_v) {
		cout << "projective_space_with_action::select_packings done" << endl;
	}
}



void projective_space_with_action::select_packings_self_dual(
		const char *fname,
		const char *file_of_spreads_original,
		int f_split, int split_r, int split_m,
		spread_tables *Spread_tables,
		classify_bitvectors *&CB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_accept = 0;
	file_io Fio;

	if (f_v) {
		cout << "projective_space_with_action::"
				"select_packings_self_dual" << endl;
	}

	CB = NEW_OBJECT(classify_bitvectors);



	int *Spread_table_original;
	int nb_spreads;
	int spread_size;
	int packing_size;
	int a, b;

	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"Reading spread table from file "
				<< file_of_spreads_original << endl;
	}
	Fio.int_matrix_read_csv(file_of_spreads_original,
			Spread_table_original, nb_spreads, spread_size,
			0 /* verbose_level */);
	if (nb_spreads != Spread_tables->nb_spreads) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"nb_spreads != Spread_tables->nb_spreads" << endl;
		exit(1);
	}
	if (spread_size != Spread_tables->spread_size) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"spread_size != Spread_tables->spread_size" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Reading spread table from file "
				<< file_of_spreads_original << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
	}



	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s" << endl;
	}

	int *s2l, *l2s;
	int i, idx;
	int *set;
	int extra_data[1];
	sorting Sorting;

	extra_data[0] = spread_size;

	set = NEW_int(spread_size);
	s2l = NEW_int(nb_spreads);
	l2s = NEW_int(nb_spreads);
	for (i = 0; i < nb_spreads; i++) {
		int_vec_copy(Spread_table_original +
				i * spread_size, set, spread_size);
		Sorting.int_vec_heapsort(set, spread_size);
		if (!Sorting.search_general(Spread_tables->spread_table,
				nb_spreads, set, idx,
				table_of_sets_compare_func,
				extra_data, 0 /*verbose_level*/)) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"cannot find spread " << i << " = ";
			int_vec_print(cout, set, spread_size);
			cout << endl;
			exit(1);
		}
		s2l[i] = idx;
		l2s[idx] = i;
	}
	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s done" << endl;
	}

	int g, table_length, nb_reject = 0;


	if (f_v) {
		cout << "projective_space_with_action::"
				"select_packings_self_dual file "
				<< fname << endl;
	}

	spreadsheet *S;

	S = NEW_OBJECT(spreadsheet);

	S->read_spreadsheet(fname, 0 /*verbose_level*/);
	if (FALSE /*f_v3*/) {
		S->print_table(cout, FALSE);
		}

	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"read file " << fname << endl;
	}


	int ago_idx, original_file_idx, input_idx_idx, input_set_idx;
	int nb_rows_idx, nb_cols_idx, canonical_form_idx;

	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"finding column indices" << endl;
	}

	ago_idx = S->find_by_column("ago");
	original_file_idx = S->find_by_column("original_file");
	input_idx_idx = S->find_by_column("input_idx");
	input_set_idx = S->find_by_column("input_set");
	nb_rows_idx = S->find_by_column("nb_rows");
	nb_cols_idx = S->find_by_column("nb_cols");
	canonical_form_idx = S->find_by_column("canonical_form");

	table_length = S->nb_rows - 1;

	//rep,ago,original_file,input_idx,
	//input_set,nb_rows,nb_cols,canonical_form
	int f_first = TRUE;


	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"first pass, table_length=" << table_length << endl;
	}

	// first pass: build up the database:

	for (g = 0; g < table_length; g++) {

		int ago;
		char *text;
		int *the_set_in;
		int set_size_in;
		int *canonical_labeling;
		int canonical_labeling_sz;
		int nb_rows, nb_cols;
		object_in_projective_space *OiP;
		int f_accept;

		ago = S->get_int(g + 1, ago_idx);
		nb_rows = S->get_int(g + 1, nb_rows_idx);
		nb_cols = S->get_int(g + 1, nb_cols_idx);

		text = S->get_string(g + 1, input_set_idx);
		int_vec_scan(text, the_set_in, set_size_in);

		packing_size = set_size_in;

		if (f_v && (g % 1000) == 0) {
			cout << "File " << fname
					<< ", input set " << g << " / "
					<< table_length << endl;
			//int_vec_print(cout, the_set_in, set_size_in);
			//cout << endl;
			}


		f_accept = TRUE;



		if (f_accept) {

			nb_accept++;


			if (FALSE) {
				cout << "canonical_form_idx=" << canonical_form_idx << endl;
			}
			text = S->get_string(g + 1, canonical_form_idx);
			if (FALSE) {
				cout << "text=" << text << endl;
			}
			int_vec_scan(text, canonical_labeling, canonical_labeling_sz);
			if (FALSE) {
				cout << "File " << fname
						<< ", input set " << g << " / "
						<< table_length << " canonical_labeling = ";
				int_vec_print(cout,
						canonical_labeling, canonical_labeling_sz);
				cout << endl;
				}

			if (canonical_labeling_sz != nb_rows + nb_cols) {
				cout << "projective_space_with_action::"
						"select_packings_self_dual "
						"canonical_labeling_sz != nb_rows + nb_cols" << endl;
				exit(1);
			}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (FALSE) {
				cout << "projective_space_with_action::"
						"select_packings_self_dual "
						"before init_packing_from_spread_table" << endl;
			}
			OiP->init_packing_from_spread_table(P, the_set_in,
					Spread_table_original, nb_spreads, spread_size,
				0 /*verbose_level*/);
			if (FALSE) {
				cout << "projective_space_with_action::"
						"select_packings_self_dual "
						"after init_packing_from_spread_table" << endl;
			}
			OiP->f_has_known_ago = TRUE;
			OiP->known_ago = ago;

			int *Incma_in;
			int *Incma_out;
			int nb_rows1, nb_cols1;
			int *partition;
			uchar *canonical_form;
			int canonical_form_len;


			if (FALSE) {
				cout << "projective_space_with_action::"
						"select_packings_self_dual "
						"before encode_incma" << endl;
			}
			OiP->encode_incma(Incma_in, nb_rows1, nb_cols1,
					partition, 0 /*verbose_level - 1*/);
			if (FALSE) {
				cout << "projective_space_with_action::"
						"select_packings_self_dual "
						"after encode_incma" << endl;
			}
			if (nb_rows1 != nb_rows) {
				cout << "projective_space_with_action::"
						"select_packings_self_dual "
						"nb_rows1 != nb_rows" << endl;
				exit(1);
			}
			if (nb_cols1 != nb_cols) {
				cout << "projective_space_with_action::"
						"select_packings_self_dual "
						"nb_cols1 != nb_cols" << endl;
				exit(1);
			}

			OiP->input_fname = S->get_string(g + 1, original_file_idx);
			OiP->input_idx = S->get_int(g + 1, input_idx_idx);

			text = S->get_string(g + 1, input_set_idx);
			int l = strlen(text);

			OiP->set_as_string = NEW_char(l + 1);
			strcpy(OiP->set_as_string, text);

			int i, j, ii, jj, a, ret;
			int L = nb_rows * nb_cols;

			Incma_out = NEW_int(L);
			for (i = 0; i < nb_rows; i++) {
				ii = canonical_labeling[i];
				for (j = 0; j < nb_cols; j++) {
					jj = canonical_labeling[nb_rows + j] - nb_rows;
					//cout << "i=" << i << " j=" << j << " ii=" << ii
					//<< " jj=" << jj << endl;
					Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
					}
				}
			if (FALSE) {
				cout << "projective_space_with_action::"
						"select_packings_self_dual "
						"before bitvector_allocate_and_coded_length" << endl;
			}
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

			if (f_first) {
				if (f_v) {
					cout << "projective_space_with_action::"
							"select_packings_self_dual "
							"before CB->init" << endl;
				}
				CB->init(table_length, canonical_form_len, verbose_level);
				f_first = FALSE;
			}


			if (FALSE) {
				cout << "projective_space_with_action::"
						"select_packings_self_dual "
						"before CB->add" << endl;
			}

			ret = CB->add(canonical_form, OiP, 0 /*verbose_level*/);
			if (ret == 0) {
				cout << "reject" << endl;
				nb_reject++;
			}
			if (FALSE) {
				cout << "projective_space_with_action::"
						"select_packings_self_dual "
						"CB->add returns " << ret
						<< " nb iso = " << CB->nb_types
						<< " nb_reject=" << nb_reject
						<< " nb_accept=" << nb_accept
						<< " CB->n=" << CB->n
						<< " CB->nb_types=" << CB->nb_types
						<< endl;
			}


			int idx;

			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init_known_ago(OiP, ago, nb_rows, nb_cols,
					canonical_labeling, 0 /*verbose_level*/);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;

			FREE_int(Incma_in);
			FREE_int(Incma_out);
			FREE_int(partition);
			//FREE_int(canonical_labeling);
			//FREE_uchar(canonical_form);
		} // if (f_accept)



		FREE_int(the_set_in);

	} // next g




	if (f_v) {
		cout << "projective_space_with_action::"
				"select_packings_self_dual done, "
				"we found " << CB->nb_types << " isomorphism types "
				"of packings. nb_accept = " << nb_accept
				<< " CB->n = " << CB->n
				<< " CB->nb_types = " << CB->nb_types
				<< endl;
		}


	// second pass:

	int nb_self_dual = 0;
	int g1 = 0;
	int *self_dual_cases;
	int nb_self_dual_cases = 0;


	self_dual_cases = NEW_int(table_length);


	if (f_v) {
		cout << "projective_space_with_action::"
				"select_packings_self_dual "
				"second pass, table_length="
				<< table_length << endl;
	}


	for (g = 0; g < table_length; g++) {

		int ago;
		char *text;
		int *the_set_in;
		int set_size_in;
		int *canonical_labeling1;
		int *canonical_labeling2;
		//int canonical_labeling_sz;
		int nb_rows, nb_cols;
		object_in_projective_space *OiP1;
		object_in_projective_space *OiP2;
		int *set1;
		int *set2;

		ago = S->get_int(g + 1, ago_idx);
		nb_rows = S->get_int(g + 1, nb_rows_idx);
		nb_cols = S->get_int(g + 1, nb_cols_idx);

		text = S->get_string(g + 1, input_set_idx);
		int_vec_scan(text, the_set_in, set_size_in);

		packing_size = set_size_in;


		if (f_split) {
			if ((g % split_m) != split_r) {
				continue;
			}
		}
		g1++;
		if (f_v && (g1 % 100) == 0) {
			cout << "File " << fname
					<< ", case " << g1 << " input set " << g << " / "
					<< table_length
					<< " nb_self_dual=" << nb_self_dual << endl;
			//int_vec_print(cout, the_set_in, set_size_in);
			//cout << endl;
			}


		set1 = NEW_int(packing_size);
		set2 = NEW_int(packing_size);

		for (i = 0; i < packing_size; i++) {
			a = the_set_in[i];
			b = s2l[a];
			set1[i] = b;
		}
		Sorting.int_vec_heapsort(set1, packing_size);
		for (i = 0; i < packing_size; i++) {
			a = set1[i];
			b = Spread_tables->dual_spread_idx[a];
			set2[i] = l2s[b];
		}
		for (i = 0; i < packing_size; i++) {
			a = set1[i];
			b = l2s[a];
			set1[i] = b;
		}
		Sorting.int_vec_heapsort(set1, packing_size);
		Sorting.int_vec_heapsort(set2, packing_size);

#if 0
		cout << "set1: ";
		int_vec_print(cout, set1, packing_size);
		cout << endl;
		cout << "set2: ";
		int_vec_print(cout, set2, packing_size);
		cout << endl;
#endif




		OiP1 = NEW_OBJECT(object_in_projective_space);
		OiP2 = NEW_OBJECT(object_in_projective_space);

		if (FALSE) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"before init_packing_from_spread_table" << endl;
		}
		OiP1->init_packing_from_spread_table(P, set1,
				Spread_table_original, nb_spreads, spread_size,
				0 /*verbose_level*/);
		OiP2->init_packing_from_spread_table(P, set2,
				Spread_table_original, nb_spreads, spread_size,
				0 /*verbose_level*/);
		if (FALSE) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"after init_packing_from_spread_table" << endl;
		}
		OiP1->f_has_known_ago = TRUE;
		OiP1->known_ago = ago;



		uchar *canonical_form1;
		uchar *canonical_form2;
		int canonical_form_len;



		int *Incma_in1;
		int *Incma_out1;
		int *Incma_in2;
		int *Incma_out2;
		int nb_rows1, nb_cols1;
		int *partition;
		//uchar *canonical_form1;
		//uchar *canonical_form2;
		//int canonical_form_len;


		if (FALSE) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"before encode_incma" << endl;
		}
		OiP1->encode_incma(Incma_in1, nb_rows1, nb_cols1,
				partition, 0 /*verbose_level - 1*/);
		OiP2->encode_incma(Incma_in2, nb_rows1, nb_cols1,
				partition, 0 /*verbose_level - 1*/);
		if (FALSE) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"after encode_incma" << endl;
		}
		if (nb_rows1 != nb_rows) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"nb_rows1 != nb_rows" << endl;
			exit(1);
		}
		if (nb_cols1 != nb_cols) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"nb_cols1 != nb_cols" << endl;
			exit(1);
		}


		if (FALSE) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"before PA->set_stabilizer_of_object" << endl;
			}


		canonical_labeling1 = NEW_int(nb_rows * nb_cols);
		canonical_labeling2 = NEW_int(nb_rows * nb_cols);

		canonical_labeling(
				OiP1,
				canonical_labeling1,
				0 /*verbose_level - 2*/);
		canonical_labeling(
				OiP2,
				canonical_labeling2,
				0 /*verbose_level - 2*/);


		OiP1->input_fname = S->get_string(g + 1, original_file_idx);
		OiP1->input_idx = S->get_int(g + 1, input_idx_idx);
		OiP2->input_fname = S->get_string(g + 1, original_file_idx);
		OiP2->input_idx = S->get_int(g + 1, input_idx_idx);

		text = S->get_string(g + 1, input_set_idx);
		int l = strlen(text);

		OiP1->set_as_string = NEW_char(l + 1);
		strcpy(OiP1->set_as_string, text);

		OiP2->set_as_string = NEW_char(l + 1);
		strcpy(OiP2->set_as_string, text);

		int i, j, ii, jj, a, ret;
		int L = nb_rows * nb_cols;

		Incma_out1 = NEW_int(L);
		Incma_out2 = NEW_int(L);
		for (i = 0; i < nb_rows; i++) {
			ii = canonical_labeling1[i];
			for (j = 0; j < nb_cols; j++) {
				jj = canonical_labeling1[nb_rows + j] - nb_rows;
				//cout << "i=" << i << " j=" << j << " ii=" << ii
				//<< " jj=" << jj << endl;
				Incma_out1[i * nb_cols + j] = Incma_in1[ii * nb_cols + jj];
				}
			}
		for (i = 0; i < nb_rows; i++) {
			ii = canonical_labeling2[i];
			for (j = 0; j < nb_cols; j++) {
				jj = canonical_labeling2[nb_rows + j] - nb_rows;
				//cout << "i=" << i << " j=" << j << " ii=" << ii
				//<< " jj=" << jj << endl;
				Incma_out2[i * nb_cols + j] = Incma_in2[ii * nb_cols + jj];
				}
			}
		if (FALSE) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"before bitvector_allocate_and_coded_length" << endl;
		}
		canonical_form1 = bitvector_allocate_and_coded_length(
				L, canonical_form_len);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (Incma_out1[i * nb_cols + j]) {
					a = i * nb_cols + j;
					bitvector_set_bit(canonical_form1, a);
					}
				}
			}
		canonical_form2 = bitvector_allocate_and_coded_length(
				L, canonical_form_len);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (Incma_out2[i * nb_cols + j]) {
					a = i * nb_cols + j;
					bitvector_set_bit(canonical_form2, a);
					}
				}
			}


		if (FALSE) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"before CB->search" << endl;
		}

		int idx1, idx2;

		ret = CB->search(canonical_form1, idx1, 0 /*verbose_level*/);

		if (ret == FALSE) {
			cout << "cannot find the dual packing, "
					"something is wrong" << endl;
			ret = CB->search(canonical_form1, idx1, 5 /* verbose_level*/);
#if 0
			cout << "CB:" << endl;
			CB->print_table();
			cout << "canonical form1: ";
			for (int j = 0; j < canonical_form_len; j++) {
				cout << (int) canonical_form1[j];
				if (j < canonical_form_len - 1) {
					cout << ", ";
					}
				}
			cout << endl;
#endif
			exit(1);
		}
		if (FALSE) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"CB->search returns idx1=" << idx1 << endl;
		}
		ret = CB->search(canonical_form2, idx2, 0 /*verbose_level*/);

		if (ret == FALSE) {
			cout << "cannot find the dual packing, "
					"something is wrong" << endl;
			ret = CB->search(canonical_form2, idx2, 5 /* verbose_level*/);
#if 0
			cout << "CB:" << endl;
			CB->print_table();
			cout << "canonical form2: ";
			for (int j = 0; j < canonical_form_len; j++) {
				cout << (int) canonical_form2[j];
				if (j < canonical_form_len - 1) {
					cout << ", ";
					}
				}
#endif
			exit(1);
		}
		if (FALSE) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"CB->search returns idx2=" << idx2 << endl;
		}

		FREE_int(Incma_in1);
		FREE_int(Incma_out1);
		FREE_int(Incma_in2);
		FREE_int(Incma_out2);
		FREE_int(partition);
		FREE_int(canonical_labeling1);
		FREE_int(canonical_labeling2);
		FREE_uchar(canonical_form1);
		FREE_uchar(canonical_form2);

		FREE_int(set1);
		FREE_int(set2);

		if (idx1 == idx2) {
			cout << "self-dual" << endl;
			nb_self_dual++;
			self_dual_cases[nb_self_dual_cases++] = g;
		}

		FREE_int(the_set_in);

	} // next g

	char fname_base[1000];
	char fname_self_dual[1000];

	strcpy(fname_base, fname);
	chop_off_extension(fname_base);
	if (f_split) {
		sprintf(fname_self_dual, "%s_self_dual_r%d_m%d.csv",
				fname_base, split_r, split_m);
	}
	else {
		sprintf(fname_self_dual, "%s_self_dual.csv", fname_base);
	}
	cout << "saving self_dual_cases to file " << fname_self_dual << endl;
	Fio.int_vec_write_csv(self_dual_cases, nb_self_dual_cases,
			fname_self_dual, "self_dual_idx");
	cout << "written file " << fname_self_dual
			<< " of size " << Fio.file_size(fname_self_dual) << endl;



	//FREE_OBJECT(CB);
	FREE_int(Spread_table_original);

	if (f_v) {
		cout << "projective_space_with_action::"
				"select_packings_self_dual "
				"done, nb_self_dual = " << nb_self_dual << endl;
	}
}




void projective_space_with_action::latex_report(const char *fname,
		const char *prefix,
		classify_bitvectors *CB,
		int f_save_incma_in_and_out,
		int fixed_structure_order_list_sz,
		int *fixed_structure_order_list,
		int max_TDO_depth,
		int verbose_level)
{
	int i, j;
	int f_v = (verbose_level >= 1);
	sorting Sorting;
	file_io Fio;
	latex_interface L;

	if (f_v) {
		cout << "projective_space_with_action::latex_report" << endl;
	}
	{
	ofstream fp(fname);
	latex_interface L;

	L.head_easy(fp);

	int *Table;
	int width = 4;
	int *row_labels;
	int *col_labels;
	int row_part_first[2], row_part_len[1];
	int nb_row_parts = 1;
	int col_part_first[2], col_part_len[1];
	int nb_col_parts = 1;



	row_part_first[0] = 0;
	row_part_first[1] = CB->nb_types;
	row_part_len[0] = CB->nb_types;

	col_part_first[0] = 0;
	col_part_first[1] = width;
	col_part_len[0] = width;

	Table = NEW_int(CB->nb_types * width);
	int_vec_zero(Table, CB->nb_types * width);

	row_labels = NEW_int(CB->nb_types);
	col_labels = NEW_int(width);
	for (i = 0; i < CB->nb_types; i++) {
		row_labels[i] = i;
		}
	for (j = 0; j < width; j++) {
		col_labels[j] = j;
		}

	for (i = 0; i < CB->nb_types; i++) {

		j = CB->perm[i];
		Table[i * width + 0] = CB->Type_rep[j];
		Table[i * width + 1] = CB->Type_mult[j];
		Table[i * width + 2] = 0; // group order
		Table[i * width + 3] = 0; // object list
		}

	fp << "\\section{Summary of Orbits}" << endl;
	fp << "$$" << endl;
	L.int_matrix_print_with_labels_and_partition(fp,
			Table, CB->nb_types, 4,
		row_labels, col_labels,
		row_part_first, row_part_len, nb_row_parts,
		col_part_first, col_part_len, nb_col_parts,
		print_summary_table_entry,
		CB /*void *data*/,
		TRUE /* f_tex */);
	fp << "$$" << endl;

	compute_and_print_ago_distribution_with_classes(fp,
			CB, verbose_level);

	for (i = 0; i < CB->nb_types; i++) {

		j = CB->perm[i];
		object_in_projective_space_with_action *OiPA;
		object_in_projective_space *OiP;

		cout << "###################################################"
				"#############################" << endl;
		cout << "Orbit " << i << " / " << CB->nb_types
				<< " is canonical form no " << j
				<< ", original object no " << CB->Type_rep[j]
				<< ", frequency " << CB->Type_mult[j]
				<< " : " << endl;


		{
		int *Input_objects;
		int nb_input_objects;
		CB->C_type_of->get_class_by_value(Input_objects,
			nb_input_objects, j, 0 /*verbose_level */);

		cout << "This isomorphism type appears " << nb_input_objects
				<< " times, namely for the following "
						"input objects:" << endl;
		L.int_vec_print_as_matrix(cout, Input_objects,
				nb_input_objects, 10 /* width */,
				FALSE /* f_tex */);

		FREE_int(Input_objects);
		}

		OiPA = (object_in_projective_space_with_action *)
				CB->Type_extra_data[j];
		OiP = OiPA->OiP;
		if (OiP->type != t_PAC) {
			OiP->print(cout);
			}

		//OiP->init_point_set(PA->P, (int *)CB->Type_extra_data[j],
		//sz, 0 /* verbose_level*/);



		strong_generators *SG;
		longinteger_object go;
		char save_incma_in_and_out_prefix[1000];

		if (f_save_incma_in_and_out) {
			sprintf(save_incma_in_and_out_prefix,
					"%s_iso_%d_%d", prefix, i, j);
			}


		uchar *canonical_form;
		int canonical_form_len;

		int nb_r, nb_c;
		int *canonical_labeling;

		OiP->encoding_size(
				nb_r, nb_c,
				verbose_level);
		canonical_labeling = NEW_int(nb_r + nb_c);


		SG = set_stabilizer_of_object(
			OiP,
			f_save_incma_in_and_out, save_incma_in_and_out_prefix,
			TRUE /* f_compute_canonical_form */,
			canonical_form, canonical_form_len,
			canonical_labeling,
			0 /* verbose_level */);

		FREE_int(canonical_labeling);

		SG->group_order(go);

		fp << "\\section*{Orbit " << i << " / "
			<< CB->nb_types << "}" << endl;
		fp << "Orbit " << i << " / " << CB->nb_types <<  " stored at "
			<< j << " is represented by input object "
			<< CB->Type_rep[j] << " and appears "
			<< CB->Type_mult[j] << " times: \\\\" << endl;
		if (OiP->type != t_PAC) {
			OiP->print(fp);
			fp << "\\\\" << endl;
			}
		//int_vec_print(fp, OiP->set, OiP->sz);
		fp << "Group order " << go << "\\\\" << endl;

		fp << "Stabilizer:" << endl;
		SG->print_generators_tex(fp);

		{
		int *Input_objects;
		int nb_input_objects;
		CB->C_type_of->get_class_by_value(Input_objects,
				nb_input_objects, j, 0 /*verbose_level */);
		Sorting.int_vec_heapsort(Input_objects, nb_input_objects);

		fp << "This isomorphism type appears " << nb_input_objects
				<< " times, namely for the following "
				<< nb_input_objects << " input objects: " << endl;
		if (nb_input_objects < 10) {
			fp << "$" << endl;
			L.int_set_print_tex(fp, Input_objects, nb_input_objects);
			fp << "$\\\\" << endl;
			}
		else {
			fp << "$$" << endl;
			L.int_vec_print_as_matrix(fp, Input_objects,
				nb_input_objects, 10 /* width */, TRUE /* f_tex */);
			fp << "$$" << endl;
			}

		FREE_int(Input_objects);
		}


		int *Incma;
		int nb_rows, nb_cols;
		int *partition;
		incidence_structure *Inc;
		partitionstack *Stack;


		OiP->encode_incma_and_make_decomposition(
			Incma, nb_rows, nb_cols, partition,
			Inc,
			Stack,
			verbose_level);
		FREE_int(Incma);
		FREE_int(partition);
	#if 0
		cout << "set ";
		int_vec_print(cout, OiP->set, OiP->sz);
		cout << " go=" << go << endl;

		cout << "Stabilizer:" << endl;
		SG->print_generators_tex(cout);


		incidence_structure *Inc;
		partitionstack *Stack;

		int Sz[1];
		int *Subsets[1];

		Sz[0] = OiP->sz;
		Subsets[0] = OiP->set;

		cout << "computing decomposition:" << endl;
		PA->P->decomposition(1 /* nb_subsets */, Sz, Subsets,
			Inc,
			Stack,
			verbose_level);

	#if 0
		cout << "the decomposition is:" << endl;
		Inc->get_and_print_decomposition_schemes(*Stack);
		Stack->print_classes(cout);
	#endif




	#if 0
		fp << "canonical form: ";
		for (i = 0; i < canonical_form_len; i++) {
			fp << (int)canonical_form[i];
			if (i < canonical_form_len - 1) {
				fp << ", ";
				}
			}
		fp << "\\\\" << endl;
	#endif
	#endif


		Inc->get_and_print_row_tactical_decomposition_scheme_tex(
			fp, TRUE /* f_enter_math */,
			TRUE /* f_print_subscripts */, *Stack);

	#if 0
		Inc->get_and_print_tactical_decomposition_scheme_tex(
			fp, TRUE /* f_enter_math */,
			*Stack);
	#endif



		int f_refine_prev, f_refine, h;
		int f_print_subscripts = TRUE;

		f_refine_prev = TRUE;
		for (h = 0; h < max_TDO_depth; h++) {
			if (EVEN(h)) {
				f_refine = Inc->refine_column_partition_safe(
						*Stack, verbose_level - 3);
				}
			else {
				f_refine = Inc->refine_row_partition_safe(
						*Stack, verbose_level - 3);
				}

			if (f_v) {
				cout << "incidence_structure::compute_TDO_safe "
						"h=" << h << " after refine" << endl;
				}
			if (EVEN(h)) {
				//int f_list_incidences = FALSE;
				Inc->get_and_print_column_tactical_decomposition_scheme_tex(
					fp, TRUE /* f_enter_math */,
					f_print_subscripts, *Stack);
				//get_and_print_col_decomposition_scheme(
				//PStack, f_list_incidences, FALSE);
				//PStack.print_classes_points_and_lines(cout);
				}
			else {
				//int f_list_incidences = FALSE;
				Inc->get_and_print_row_tactical_decomposition_scheme_tex(
					fp, TRUE /* f_enter_math */,
					f_print_subscripts, *Stack);
				//get_and_print_row_decomposition_scheme(
				//PStack, f_list_incidences, FALSE);
				//PStack.print_classes_points_and_lines(cout);
				}

			if (!f_refine_prev && !f_refine) {
				break;
				}
			f_refine_prev = f_refine;
			}

		cout << "Classes of the partition:\\\\" << endl;
		Stack->print_classes_tex(fp);



		OiP->klein(verbose_level);


		sims *Stab;
		int *Elt;
		int nb_trials;
		int max_trials = 100;

		Stab = SG->create_sims(verbose_level);
		Elt = NEW_int(A->elt_size_in_int);

		for (h = 0; h < fixed_structure_order_list_sz; h++) {
			if (Stab->find_element_of_given_order_int(Elt,
					fixed_structure_order_list[h], nb_trials, max_trials,
					verbose_level)) {
				fp << "We found an element of order "
						<< fixed_structure_order_list[h] << ", which is:" << endl;
				fp << "$$" << endl;
				A->element_print_latex(Elt, fp);
				fp << "$$" << endl;
				report_fixed_objects_in_PG_3_tex(
					Elt, fp,
					verbose_level);
				}
			else {
				fp << "We could not find an element of order "
					<< fixed_structure_order_list[h] << "\\\\" << endl;
				}
			}


		FREE_int(Elt);
		FREE_OBJECT(Stack);
		FREE_OBJECT(Inc);
		FREE_OBJECT(SG);

		}


	L.foot(fp);
	}

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	//FREE_int(perm);
	//FREE_int(v);
	if (f_v) {
		cout << "projective_space_with_action::latex_report done" << endl;
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
	sorting Sorting;

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
			Sorting.int_vec_heapsort(Input_objects, nb_input_objects);

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
	latex_interface L;

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
		L.int_set_print_tex(ost, SoS->Sets[i], SoS->Set_size[i]);
		ost << "$\\\\" << endl;
		//int_vec_print_as_matrix(ost, SoS->Sets[i],
		//SoS->Set_size[i], 10 /* width */, TRUE /* f_tex */);
		//ost << "$$" << endl;

		}

	FREE_int(types);
	FREE_OBJECT(SoS);
	FREE_OBJECT(C_ago);
}


int table_of_sets_compare_func(void *data, int i,
		int *search_object,
		void *extra_data)
{
	int *Data = (int *) data;
	int *p = (int *) extra_data;
	int len = p[0];
	int ret;

	ret = int_vec_compare(Data + i * len, search_object, len);
	return ret;
}



}}
