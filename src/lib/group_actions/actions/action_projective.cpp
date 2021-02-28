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
	long int *set, int set_size, int &canonical_pt,
	int *canonical_set_or_NULL,
	//int f_save_incma_in_and_out, const char *save_incma_in_and_out_prefix,
	int verbose_level)
// assuming we are in a linear action.
// added 2/28/2011, called from analyze.cpp
// November 17, 2014 moved here from TOP_LEVEL/extra.cpp
// December 31, 2014, moved here from projective_space.cpp
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
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

	if (f_v) {
		cout << "action::set_stabilizer_in_projective_space" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		cout << "set_size = " << set_size << endl;
		}

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

#if 0
	if (f_save_incma_in_and_out) {
		if (f_vv) {
			cout << "Incma in:" << endl;
			if (nb_rows < 10) {
				print_integer_matrix_width(cout,
						Incma, nb_rows, nb_cols, nb_cols, 1);
				}
			else {
				cout << "too large to print" << endl;
				}
			}
		string fname_csv;
		string fname_bin;
		char str[1000];

		sprintf(str, "Incma_in_%d_%d", nb_rows, nb_cols);

		fname_csv.assign(save_incma_in_and_out_prefix);
		fname_csv.append(str);
		fname_csv.append(".csv");
		fname_bin.assign(save_incma_in_and_out_prefix);
		fname_bin.append(str);
		fname_bin.append(".bin");
		//sprintf(fname_csv, "%sIncma_in_%d_%d.csv",
		//		save_incma_in_and_out_prefix, nb_rows, nb_cols);
		//sprintf(fname_bin, "%sIncma_in_%d_%d.bin",
		//		save_incma_in_and_out_prefix, nb_rows, nb_cols);
		Fio.int_matrix_write_csv(fname_csv, Incma, nb_rows, nb_cols);

		for (i = 0; i < nb_rows + nb_cols; i++) {
			vertex_labeling[i] = i;
			}

		colored_graph *CG;

		CG = NEW_OBJECT(colored_graph);

		CG->create_Levi_graph_from_incidence_matrix(
				Incma, nb_rows, nb_cols, TRUE, vertex_labeling, verbose_level);
		CG->save(fname_bin, verbose_level);
		//FREE_int(Incma);
		FREE_OBJECT(CG);
		}
#endif

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

#if 0
	if (f_save_incma_in_and_out) {
		if (f_vv) {
			cout << "Incma Out:" << endl;
			if (nb_rows < 20) {
				print_integer_matrix_width(cout,
						Incma_out, nb_rows, nb_cols, nb_cols, 1);
				}
			else {
				cout << "too large to print" << endl;
				}
			}
		string fname_csv;
		string fname_bin;
		char str[1000];

		sprintf(str, "Incma_out_%d_%d", nb_rows, nb_cols);

		fname_csv.assign(save_incma_in_and_out_prefix);
		fname_csv.append(str);
		fname_csv.append(".csv");
		fname_bin.assign(save_incma_in_and_out_prefix);
		fname_bin.append(str);
		fname_bin.append(".bin");

		//sprintf(fname_csv, "%sIncma_out_%d_%d.csv",
		//		save_incma_in_and_out_prefix, nb_rows, nb_cols);
		//sprintf(fname_bin, "%sIncma_out_%d_%d.bin",
		//		save_incma_in_and_out_prefix, nb_rows, nb_cols);
		Fio.int_matrix_write_csv(fname_csv, Incma_out, nb_rows, nb_cols);


		colored_graph *CG;

		CG = NEW_OBJECT(colored_graph);

		CG->create_Levi_graph_from_incidence_matrix(
				Incma_out, nb_rows, nb_cols, TRUE, vertex_labeling,
				verbose_level);
		CG->save(fname_bin, verbose_level);
		FREE_OBJECT(CG);
		}
#endif

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
	return SG;
}

int action::reverse_engineer_semilinear_map(
	projective_space *P,
	int *Elt, int *Mtx, int &frobenius,
	int verbose_level)
// uses the function A->element_image_of
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	finite_field *F;
	int d = P->n + 1;
	int *v1, *v2, *v1_save;
	int *w1, *w2, *w1_save;
	int /*q,*/ h, hh, i, j, l, e, frobenius_inv, lambda, rk, c, cv;
	int *system;
	int *base_cols;
	number_theory_domain NT;


	if (f_v) {
		cout << "action::reverse_engineer_semilinear_map" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "d=" << d << endl;
		}
	F = P->F;
	//q = F->q;

	v1 = NEW_int(d);
	v2 = NEW_int(d);
	v1_save = NEW_int(d);
	w1 = NEW_int(d);
	w2 = NEW_int(d);
	w1_save = NEW_int(d);



	if (f_v) {
		cout << "action::reverse_engineer_semilinear_map "
				"maping unit vectors" << endl;
		}
	for (e = 0; e < d; e++) {
		// map the unit vector e_e
		// (with a one in position e and zeros elsewhere):
		for (h = 0; h < d; h++) {
			if (h == e) {
				v1[h] = 1;
				}
			else {
				v1[h] = 0;
				}
			}
		Orbiter->Int_vec.copy(v1, v1_save, d);
		i = P->rank_point(v1);
			// Now, the value of i should be equal to e.
		j = element_image_of(i, Elt, 0);
		P->unrank_point(v2, j);

#if 0
		if (f_vv) {
			print_from_to(d, i, j, v1_save, v2);
			}
#endif


		Orbiter->Int_vec.copy(v2, Mtx + e * d, d);
		}

	if (f_vv) {
		cout << "Mtx (before scaling):" << endl;
		print_integer_matrix_width(cout, Mtx, d, d, d, F->log10_of_q);
		cout << endl;
		}

	// map the vector (1,1,...,1):
	if (f_v) {
		cout << "action::reverse_engineer_semilinear_map "
				"mapping the all-one vector"
				<< endl;
		}
	for (h = 0; h < d; h++) {
		v1[h] = 1;
		}
	Orbiter->Int_vec.copy(v1, v1_save, d);
	i = P->rank_point(v1);
	j = element_image_of(i, Elt, 0);
	P->unrank_point(v2, j);

#if 0
	if (f_vv) {
		print_from_to(d, i, j, v1_save, v2);
		}
#endif

	system = NEW_int(d * (d + 1));
	base_cols = NEW_int(d + 1);
	// coefficient matrix:
	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			system[i * (d + 1) + j] = Mtx[j * d + i];
			}
		}
	// RHS:
	for (i = 0; i < d; i++) {
		system[i * (d + 1) + d] = v2[i];
		}
	rk = F->Gauss_simple(system, d, d + 1, base_cols, verbose_level - 4);
	if (rk != d) {
		cout << "rk != d, fatal" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "after Gauss_simple:" << endl;
		print_integer_matrix_width(cout, system,
				d, d + 1, d + 1, F->log10_of_q);
		cout << endl;
		}
	for (i = 0; i < d; i++) {
		c = system[i * (d + 1) + d];
		for (j = 0; j < d; j++) {
			Mtx[i * d + j] = F->mult(c, Mtx[i * d + j]);
			}
		}

	if (f_vv) {
		cout << "Mtx (after scaling):" << endl;
		print_integer_matrix_width(cout, Mtx, d, d, d, F->log10_of_q);
		cout << endl;
		}


	// figure out the frobenius:
	if (f_v) {
		cout << "action::reverse_engineer_semilinear_map "
				"figuring out the frobenius" << endl;
		}

	frobenius = 0;
	if (F->q != F->p) {

		// create the vector (1,p,0,...,0)

		for (h = 0; h < d; h++) {
			if (h == 0) {
				v1[h] = 1;
				}
			else if (h == 1) {
				v1[h] = F->p;
				}
			else {
				v1[h] = 0;
				}
			}
		Orbiter->Int_vec.copy(v1, v1_save, d);
		i = P->rank_point(v1);
		j = element_image_of(i, Elt, 0);
		P->unrank_point(v2, j);


#if 0
		if (f_vv) {
			print_from_to(d, i, j, v1_save, v2);
			}
#endif


		// coefficient matrix:
		for (i = 0; i < d; i++) {
			for (j = 0; j < 2; j++) {
				system[i * 3 + j] = Mtx[j * d + i];
				}
			}
		// RHS:
		for (i = 0; i < d; i++) {
			system[i * 3 + 2] = v2[i];
			}
		rk = F->Gauss_simple(system,
				d, 3, base_cols, verbose_level - 4);
		if (rk != 2) {
			cout << "rk != 2, fatal" << endl;
			exit(1);
			}
		if (f_vv) {
			cout << "after Gauss_simple:" << endl;
			print_integer_matrix_width(cout,
					system, 2, 3, 3, F->log10_of_q);
			cout << endl;
			}

		c = system[0 * 3 + 2];
		if (c != 1) {
			cv = F->inverse(c);
			for (hh = 0; hh < 2; hh++) {
				system[hh * 3 + 2] = F->mult(cv, system[hh * 3 + 2]);
				}
			}
		if (f_vv) {
			cout << "after scaling the last column:" << endl;
			print_integer_matrix_width(cout,
					system, 2, 3, 3, F->log10_of_q);
			cout << endl;
			}
		lambda = system[1 * 3 + 2];
		if (f_vv) {
			cout << "lambda=" << lambda << endl;
			}


		l = F->log_alpha(lambda);
		if (f_vv) {
			cout << "l=" << l << endl;
			}
		for (i = 0; i < F->e; i++) {
			if (NT.i_power_j(F->p, i) == l) {
				frobenius = i;
				break;
				}
			}
		if (i == F->e) {
			cout << "action::reverse_engineer_semilinear_map "
					"problem figuring out the Frobenius" << endl;
			exit(1);
			}

		}
	else {
		frobenius = 0;
		}

	frobenius_inv = (F->e - frobenius) % F->e;
	if (f_vv) {
		cout << "frobenius = " << frobenius << endl;
		cout << "frobenius_inv = " << frobenius_inv << endl;
		}


	for (hh = 0; hh < d * d; hh++) {
		Mtx[hh] = F->frobenius_power(Mtx[hh], frobenius_inv);
		}
	if (f_v) {
		cout << "action::reverse_engineer_semilinear_map "
				"done, we found the following map" << endl;
		cout << "Mtx:" << endl;
		print_integer_matrix_width(cout,
				Mtx, d, d, d, F->log10_of_q);
		cout << endl;
		cout << "frobenius = " << frobenius << endl;
		}



	FREE_int(v1);
	FREE_int(v2);
	FREE_int(v1_save);
	FREE_int(w1);
	FREE_int(w2);
	FREE_int(w1_save);
	FREE_int(system);
	FREE_int(base_cols);


	return TRUE;
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
	ost << "has the following fixed objects:\\" << endl;


	ost << "\\bigskip" << endl;
	ost << "Fixed Points:\\" << endl;

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
	ost << "Fixed Lines\\\\" << endl;

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
	ost << "Fixed Planes\\\\" << endl;

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

