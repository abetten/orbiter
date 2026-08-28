/*
 * vector_activity.cpp
 *
 *  Created on: May 18, 2026
 *      Author: betten
 */






#include "orbiter_user_interface.h"

using namespace std;


namespace orbiter {
namespace layer6_user_interface {
namespace activities_layer5 {


vector_activity::vector_activity()
{
	Record_birth();
	Descr = NULL;

	nb_objects = 0;
	with_labels = NULL;
	pVB = NULL;

	nb_output = 0;
	Output = 0;

}


vector_activity::~vector_activity()
{
	Record_death();

}

void vector_activity::init(
		vector_activity_description *Descr,
		other::data_structures::vector_builder **pVB,
		int nb_objects,
		std::vector<std::string> &with_labels,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_activity::init" << endl;
	}


	vector_activity::Descr = Descr;
	vector_activity::pVB = pVB;
	vector_activity::nb_objects = nb_objects;
	vector_activity::with_labels = &with_labels;

	if (f_v) {
		cout << "vector_ge_activity::init done" << endl;
	}
}

void vector_activity::perform_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_activity::perform_activity" << endl;
	}


	if (Descr->f_report) {

		if (f_v) {
			cout << "vector_activity::perform_activity f_report" << endl;
		}


		pVB[0]->print(
				cout);

		if (f_v) {
			cout << "vector_activity::perform_activity "
					"f_report done" << endl;
		}


	}
	else if (Descr->f_missing_sets) {

		if (f_v) {
			cout << "vector_activity::perform_activity f_missing_sets" << endl;
			cout << "vector_activity::perform_activity n = " << Descr->missing_sets_n << endl;
		}

		int universal_set_size;

		universal_set_size = Descr->missing_sets_n;

		long int *v;
		long int *Ranks;
		int m, n;

		combinatorics::other_combinatorics::combinatorics_domain Combi;

		v = pVB[0]->v;
		m = pVB[0]->k;
		n = pVB[0]->len / m;

		pVB[0]->print(
				cout);

		int i;
		int *v1;
		int *v2;

		Ranks = NEW_lint(m);
		v1 = NEW_int(n);

		for (i = 0; i < m; i++) {

			Lint_vec_copy_to_int(v + i * n, v1, n);

			Ranks[i] = Combi.rank_k_subset(
					v1, universal_set_size, n);

		}

		Lint_vec_heapsort(Ranks, m);

		long int *complement;
		int size_complement;
		long int N;

		N = Combi.int_n_choose_k(universal_set_size, n);

		complement = NEW_lint(N);

		Combi.set_complement_lint(
				Ranks, m,
				complement, size_complement,
				N);
		// subset must be in increasing order


		v2 = NEW_int(size_complement * n);

		for (i = 0; i < size_complement; i++) {

			Combi.unrank_k_subset(
					complement[i], v2 + i * n, universal_set_size, n);

		}

		cout << "The number of missing sets is " << size_complement << endl;
		cout << "The missing sets are:" << endl;
		Int_matrix_print(v2, size_complement, n);


		string fname;

		fname = (*with_labels)[0] + "_m.csv";

		other::orbiter_kernel_system::file_io Fio;
		std::string col_heading;

		col_heading = "Vector";

		Fio.Csv_file_support->int_matrix_write_csv_tabulated(
				fname, col_heading,
				v2, size_complement, n,
				verbose_level - 1);

		if (f_v) {
			cout << "vector_activity::perform_activity "
					"written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}

		FREE_lint(complement);
		FREE_lint(Ranks);
		FREE_int(v1);
		FREE_int(v2);

		if (f_v) {
			cout << "vector_activity::perform_activity "
					"f_missing_sets done" << endl;
		}


	}
	else if (Descr->f_apply_permutation) {

		if (f_v) {
			cout << "vector_activity::perform_activity f_apply_permutation" << endl;
		}


		long int *perm;
		int N;

		Get_vector_or_set(Descr->apply_permutation_perm, perm, N);


		long int *v;
		int m, n;

		combinatorics::other_combinatorics::combinatorics_domain Combi;

		v = pVB[0]->v;
		m = pVB[0]->k;
		n = pVB[0]->len / m;

		//pVB[0]->print(cout);

		if (f_v) {
			cout << "vector_activity::perform_activity matrix before" << endl;
			Lint_matrix_print(v, m, n);
		}


		combinatorics::special_functions::permutations Permutations;

		long int *v1;

		v1 = NEW_lint(m * n);
		Lint_vec_zero(v1, m * n);

		Permutations.compute_canonical_form_of_incidence_matrix_lint(
				perm /*long int *canonical_labeling*/,
				m /* nb_rows */, n /* nb_cols */,
				v /* long int *Incma_in */, v1 /* int *Incma_out */, verbose_level - 2);


#if 0
		int i, j, a, ii, jj;


		for (i = 0; i < m; i++) {

			ii = perm[i];

			for (j = 0; j < n; j++) {

				a = v[i * n + j];

				jj = perm[m + j] - m;

				v1[ii * n + jj] = a;
			}

		}
#endif

		if (f_v) {
			cout << "vector_activity::perform_activity matrix after" << endl;
			Lint_matrix_print(v1, m, n);
		}
		Lint_vec_copy(v1, v, m * n);

		FREE_lint(v1);
		FREE_lint(perm);


	}
	else if (Descr->f_apply_permutation_inverse) {

		if (f_v) {
			cout << "vector_activity::perform_activity f_apply_permutation_inverse" << endl;
		}


		long int *perm;
		long int *perm_inv;
		int N;

		Get_vector_or_set(Descr->apply_permutation_inverse_perm, perm, N);



		long int *v;
		int m, n;

		combinatorics::other_combinatorics::combinatorics_domain Combi;

		v = pVB[0]->v;
		m = pVB[0]->k;
		n = pVB[0]->len / m;

		//pVB[0]->print(cout);

		if (f_v) {
			cout << "vector_activity::perform_activity matrix before" << endl;
			Lint_matrix_print(v, m, n);
		}

		combinatorics::special_functions::permutations Permutations;

		perm_inv = NEW_lint(N);
		Permutations.perm_inverse_lint(
				perm, perm_inv, N);

		long int *v1;

		v1 = NEW_lint(m * n);
		Lint_vec_zero(v1, m * n);

		Permutations.compute_canonical_form_of_incidence_matrix_lint(
				perm_inv /*long int *canonical_labeling*/,
				m /* nb_rows */, n /* nb_cols */,
				v /* long int *Incma_in */, v1 /* int *Incma_out */, verbose_level - 2);


#if 0
		int i, j, a, ii, jj;


		for (i = 0; i < m; i++) {

			ii = perm[i];

			for (j = 0; j < n; j++) {

				a = v[i * n + j];

				jj = perm[m + j] - m;

				v1[ii * n + jj] = a;
			}

		}
#endif

		if (f_v) {
			cout << "vector_activity::perform_activity matrix after" << endl;
			Lint_matrix_print(v1, m, n);
		}
		Lint_vec_copy(v1, v, m * n);

		FREE_lint(v1);
		FREE_lint(perm);
		FREE_lint(perm_inv);


	}




	if (f_v) {
		cout << "vector_activity::perform_activity done" << endl;
	}

}




}}}


