/*
 * action_projective.cpp
 *
 *  Created on: Feb 18, 2019
 *      Author: betten
 */



#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {

strong_generators *action::set_stabilizer_in_projective_space(
	projective_space *P,
	long int *set, int set_size, //int &canonical_pt,
	int *canonical_set_or_NULL,
	int verbose_level)
// assuming we are in a linear action.
// added 2/28/2011, called from analyze.cpp
// November 17, 2014 moved here from TOP_LEVEL/extra.cpp
// December 31, 2014, moved here from projective_space.cpp
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	object_in_projective_space *OiP;
	nauty_interface_with_group Nau;

#if 0
	int *Incma;
	int *partition;
	int *labeling;
	long int *vertex_labeling;
	int nb_rows, nb_cols;
	int *Aut, Aut_counter;
	int *Base, Base_length;
	long int *Base_lint;
	int *Transversal_length;
	longinteger_object Ago;
	int N, i, j, h;
	file_io Fio;
#endif

	if (f_v) {
		cout << "action::set_stabilizer_in_projective_space" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		cout << "set_size = " << set_size << endl;
	}


	OiP = NEW_OBJECT(object_in_projective_space);

	OiP->init_point_set(P, set, set_size, verbose_level);

	int nb_rows, nb_cols;
	//long int *canonical_labeling;
	//int canonical_labeling_len;
	bitvector *Canonical_form = NULL;

	OiP->encoding_size(
			nb_rows, nb_cols,
			verbose_level);
	//canonical_labeling = NEW_lint(nb_rows + nb_cols);


	strong_generators *SG;
	nauty_output *NO;


	NO = NEW_OBJECT(nauty_output);
	NO->allocate(nb_rows + nb_cols, 0 /* verbose_level */);

	if (f_v) {
		cout << "action::set_stabilizer_in_projective_space before Nau.set_stabilizer_of_object" << endl;
	}
	SG = Nau.set_stabilizer_of_object(
		OiP,
		this /* A_linear */,
		FALSE /* f_compute_canonical_form */, Canonical_form,
		//canonical_labeling, canonical_labeling_len,
		NO,
		verbose_level - 2);
	if (f_v) {
		cout << "action::set_stabilizer_in_projective_space after Nau.set_stabilizer_of_object" << endl;
	}

	long int nb_backtrack1, nb_backtrack2;

	nb_backtrack1 = NO->nb_firstpathnode;
	nb_backtrack2 = NO->nb_processnode;

	if (f_v) {
		cout << "canonical_form_nauty::quartic_curve "
				"go = " << *NO->Ago << endl;

		cout << "canonical_form_nauty::quartic_curve "
				"nb_backtrack1 = " << nb_backtrack1 << endl;
		cout << "canonical_form_nauty::quartic_curve "
				"nb_backtrack2 = " << nb_backtrack2 << endl;


	}


	FREE_OBJECT(NO);

#if 0

	if (f_vv) {
		cout << "action::set_stabilizer_in_projective_space "
				"computing the type of the set" << endl;
	}

	tally C;

	C.init_lint(set, set_size, TRUE, 0);
	if (C.second_nb_types > 1) {
		cout << "action::set_stabilizer_in_projective_space: "
				"The set is a multiset:" << endl;
		C.print(FALSE /*f_backwards*/);
	}

	if (f_vv) {
		cout << "action::set_stabilizer_in_projective_space "
				"The type of the set is:" << endl;
		C.print(FALSE /*f_backwards*/);
		cout << "C.second_nb_types = " << C.second_nb_types << endl;
	}
	if (f_vv) {
		cout << "action::set_stabilizer_in_projective_space "
				"allocating data" << endl;
	}
	nb_rows = P->N_points + 1;
	nb_cols = P->N_lines + C.second_nb_types;
	Incma = NEW_int(nb_rows * nb_cols);
	partition = NEW_int(nb_rows + nb_cols);
	labeling = NEW_int(nb_rows + nb_cols);
	vertex_labeling = NEW_lint(nb_rows + nb_cols);

	if (f_vv) {
		cout << "action::set_stabilizer_in_projective_space "
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
			cout << "j=" << j << " f2=" << f2 << " l2=" << l2
					<< " multiplicity=" << m << endl;
		}
		for (h = 0; h < l2; h++) {
			idx = C.second_sorting_perm_inv[f2 + h];
			f = C.type_first[idx];
			l = C.type_len[idx];
			i = C.data_sorted[f + 0];
			if (f_vvv) {
				cout << "h=" << h << " idx=" << idx << " f=" << f
						<< " l=" << l << " i=" << i << endl;
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

	if (f_vvv) {
		cout << "action::set_stabilizer_in_projective_space Incma:" << endl;
		//int_matrix_print(Incma, nb_rows, nb_cols);
	}


	if (f_vv) {
		cout << "action::set_stabilizer_in_projective_space "
				"initializing partition" << endl;
	}
	N = nb_rows + nb_cols;
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
		cout << "partition:" << endl;
		for (i = 0; i < N; i++) {
			//cout << i << " : " << partition[i] << endl;
			cout << partition[i];
		}
		cout << endl;
	}

	if (f_vv) {
		cout << "action::set_stabilizer_in_projective_space "
				"initializing Aut, Base, Transversal_length" << endl;
	}
	Aut = NEW_int(N * N);
	Base = NEW_int(N);
	Base_lint = NEW_lint(N);
	Transversal_length = NEW_int(N);
	nauty_interface Nau;

	if (f_v) {
		cout << "action::set_stabilizer_in_projective_space, "
				"calling nauty_interface_matrix_int" << endl;
	}
	Nau.nauty_interface_matrix_int(Incma, nb_rows, nb_cols,
		labeling, partition,
		Aut, Aut_counter,
		Base, Base_length,
		Transversal_length, Ago, verbose_level - 3);
	if (f_v) {
		cout << "action::set_stabilizer_in_projective_space, "
				"done with nauty_interface_matrix_int, Ago=" << Ago << endl;
	}

	Orbiter->Int_vec.copy_to_lint(Base, Base_lint, Base_length);

	int *Incma_out;
	int ii, jj;
	if (f_vvv) {
		cout << "action::set_stabilizer_in_projective_space labeling:" << endl;
		//int_vec_print(cout, labeling, nb_rows + nb_cols);
		cout << endl;
	}

	Incma_out = NEW_int(nb_rows * nb_cols);
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
		cout << "action::set_stabilizer_in_projective_space Incma_out:" << endl;
		//int_matrix_print(Incma_out, nb_rows, nb_cols);
	}


	canonical_pt = -1;
	if (set_size) {
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
		if (canonical_set_or_NULL) {
			h = 0;
			for (i = 0; i < P->N_points; i++) {
				if (Incma_out[i * nb_cols + P->N_lines + 0] == 1) {
					canonical_set_or_NULL[h++] = labeling[i];
				}
			}
			if (h != set_size) {
				cout << "action::set_stabilizer_in_projective_space "
						"h != set_size" << endl;
				cout << "h=" << h << endl;
				cout << "set_size=" << set_size << endl;
				exit(1);
			}
		}
	}


	FREE_int(Incma_out);

	action *A_perm;
	longinteger_object ago;


	A_perm = NEW_OBJECT(action);

	if (f_v) {
		cout << "action::set_stabilizer_in_projective_space "
				"before init_permutation_group_from_generators" << endl;
	}
	Ago.assign_to(ago);
	//ago.create(Ago, __FILE__, __LINE__);
	A_perm->init_permutation_group_from_generators(N,
		TRUE, ago,
		Aut_counter, Aut,
		Base_length, Base_lint,
		0 /*verbose_level - 2 */);

	if (f_vv) {
		cout << "action::set_stabilizer_in_projective_space: "
				"create_automorphism_group_of_incidence_structure: "
				"created action ";
		A_perm->print_info();
		cout << endl;
	}

	//action *A_linear;

	//A_linear = A;

#if 0
	if (A_linear == NULL) {
		cout << "set_stabilizer_in_projective_space: "
				"A_linear == NULL" << endl;
		exit(1);
	}
#endif

	vector_ge *gens; // permutations from nauty
	vector_ge *gens1; // matrices
	int d, g, frobenius, pos;
	int *Mtx;
	int *Elt1;

	Elt1 = NEW_int(elt_size_in_int);

	d = P->n + 1;

	gens = A_perm->Strong_gens->gens;
	//gens = A->strong_generators;

	gens1 = NEW_OBJECT(vector_ge);
	gens1->init(this, verbose_level - 2);
	gens1->allocate(gens->len, verbose_level - 2);

	Mtx = NEW_int(d * d + 1); // leave space for frobenius

	pos = 0;
	for (g = 0; g < gens->len; g++) {
		if (f_vv) {
			cout << "action::set_stabilizer_in_projective_space: "
					"strong generator " << g << ":" << endl;
			//A_perm->element_print(gens->ith(g), cout);
			cout << endl;
		}

		if (A_perm->reverse_engineer_semilinear_map(P,
			gens->ith(g), Mtx, frobenius,
			0 /*verbose_level - 2*/)) {

			Mtx[d * d] = frobenius;
			make_element(Elt1, Mtx, 0 /*verbose_level - 2*/);
			if (f_vv) {
				cout << "semi-linear group element:" << endl;
				//element_print(Elt1, cout);
			}
			element_move(Elt1, gens1->ith(pos), 0);


			pos++;
		}
		else {
			if (f_vv) {
				cout << "action::set_stabilizer_in_projective_space: "
						"generator " << g
						<< " does not correspond to a semilinear mapping"
						<< endl;
			}
		}
	}
	gens1->reallocate(pos, verbose_level - 2);
	if (f_vv) {
		cout << "action::set_stabilizer_in_projective_space "
				"we found " << gens1->len << " generators" << endl;
	}

	if (f_vvv) {
		//gens1->print(cout);
	}


	if (f_vv) {
		cout << "action::set_stabilizer_in_projective_space: "
				"we are now testing the generators:" << endl;
	}
	int j1, j2;

	for (g = 0; g < gens1->len; g++) {
		if (f_vv) {
			cout << "generator " << g << ":" << endl;
		}
		//A_linear->element_print(gens1->ith(g), cout);
		for (i = 0; i < P->N_points; i++) {
			j1 = element_image_of(i, gens1->ith(g), 0);
			j2 = A_perm->element_image_of(i, gens->ith(g), 0);
			if (j1 != j2) {
				cout << "action::set_stabilizer_in_projective_space "
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
		cout << "action::set_stabilizer_in_projective_space: "
				"the generators are OK" << endl;
	}



	sims *S;
	longinteger_object go;

	if (f_vv) {
		cout << "action::set_stabilizer_in_projective_space: "
				"we are now creating the group" << endl;
	}

	S = create_sims_from_generators_with_target_group_order(
		gens1, ago, 0 /*verbose_level*/);

	S->group_order(go);


	if (f_vv) {
		cout << "action::set_stabilizer_in_projective_space: "
				"Found a group of order " << go << endl;
	}
	if (f_vvv) {
		cout << "action::set_stabilizer_in_projective_space: "
				"strong generators are:" << endl;
		//S->print_generators();
		cout << "set_stabilizer_in_projective_space: "
				"strong generators are (in tex):" << endl;
		//S->print_generators_tex(cout);
	}


	longinteger_domain D;

	if (D.compare_unsigned(ago, go)) {
		cout << "action::set_stabilizer_in_projective_space: "
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
	FREE_lint(vertex_labeling);
	FREE_OBJECT(A_perm);
	FREE_OBJECT(gens1);
	FREE_int(Mtx);
	FREE_int(Elt1);


	strong_generators *SG;

	SG = NEW_OBJECT(strong_generators);

	SG->init_from_sims(S, 0);
	FREE_OBJECT(S);

	if (f_v) {
		cout << "action::set_stabilizer_in_projective_space done" << endl;
	}
#endif

	//FREE_lint(canonical_labeling);
	FREE_OBJECT(OiP);

	return SG;
}

int action::reverse_engineer_semilinear_map(
	projective_space *P,
	int *Elt, int *Mtx, int &frobenius,
	int verbose_level)
// uses the function A->element_image_of
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "action::reverse_engineer_semilinear_map, before P->reverse_engineer_semilinear_map" << endl;
	}
	ret = P->reverse_engineer_semilinear_map(Elt, Mtx, frobenius, verbose_level);
	if (f_v) {
		cout << "action::reverse_engineer_semilinear_map, after P->reverse_engineer_semilinear_map" << endl;
	}
	if (f_v) {
		cout << "action::reverse_engineer_semilinear_map done" << endl;
	}
	return ret;
}


void action::report_fixed_objects_in_P3(ostream &ost,
	projective_space *P3,
	int *Elt,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, cnt;
	int v[4];
	finite_field *F;

	if (f_v) {
		cout << "action::report_fixed_objects_in_P3" << endl;
	}
	//ost << "\\section{Fixed Objects}" << endl;

	F = P3->F;

	ost << "\\bigskip" << endl;

	ost << "The element" << endl;
	ost << "$$" << endl;
	element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following fixed objects:\\\\" << endl;


	ost << "\\bigskip" << endl;
	//ost << "Fixed Points:\\" << endl;

	cnt = 0;
	for (i = 0; i < P3->N_points; i++) {
		j = element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
		}
	}

	ost << "There are " << cnt << " fixed points, they are: \\\\" << endl;
	for (i = 0; i < P3->N_points; i++) {
		j = element_image_of(i, Elt, 0 /* verbose_level */);
		F->PG_element_unrank_modified(v, 1, 4, i);
		if (j == i) {
			ost << i << " : ";
			Orbiter->Int_vec.print(ost, v, 4);
			ost << "\\\\" << endl;
			cnt++;
		}
	}

	ost << "\\bigskip" << endl;
	//ost << "Fixed Lines\\\\" << endl;

	{
		action *A2;

		A2 = induced_action_on_grassmannian(2, 0 /* verbose_level*/);

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
				ost << i << " : $";
				A2->G.AG->G->print_single_generator_matrix_tex(ost, i);
				ost << "$\\\\" << endl;
				cnt++;
			}
		}

		FREE_OBJECT(A2);
	}

	ost << "\\bigskip" << endl;
	//ost << "Fixed Planes\\\\" << endl;

	{
		action *A3;

		A3 = induced_action_on_grassmannian(3, 0 /* verbose_level*/);

		cnt = 0;
		for (i = 0; i < A3->degree; i++) {
			j = A3->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
			}
		}

		ost << "There are " << cnt << " fixed planes, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A3->degree; i++) {
			j = A3->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				ost << i << " : $";
				A3->G.AG->G->print_single_generator_matrix_tex(ost, i);
				ost << "$\\\\" << endl;
				cnt++;
			}
		}

		FREE_OBJECT(A3);
	}
	if (f_v) {
		cout << "action::report_fixed_objects_in_P3 done" << endl;
	}
}


}}

