/*
 * design_create.cpp
 *
 *  Created on: Sep 19, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


design_create::design_create()
{
	Descr = NULL;

	//std::string prefix;
	//std::string label_txt;
	//std::string label_tex;

	q = 0;
	F = NULL;
	k = 0;

	A = NULL;
	A2 = NULL;
	Aut = NULL;
	Aut_on_lines = NULL;

	degree = 0;

	f_has_set = FALSE;
	set = NULL;
	sz = 0;

	f_has_group = FALSE;
	Sg = NULL;

	PA = NULL;
	P = NULL;
	block = NULL;

	v = 0;
	b = 0;
	nb_inc = 0;
	f_has_incma = FALSE;
	incma = NULL;
}

design_create::~design_create()
{
	if (F) {
		FREE_OBJECT(F);
	}
	if (set) {
		FREE_lint(set);
	}
	if (Sg) {
		FREE_OBJECT(Sg);
	}
	if (PA) {
		FREE_OBJECT(PA);
	}
	if (block) {
		FREE_int(block);
	}
	if (incma) {
		FREE_int(incma);
	}
}

void design_create::init(apps_combinatorics::design_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "design_create::init" << endl;
	}
	design_create::Descr = Descr;

	if (Descr->f_field) {

		F = Get_object_of_type_finite_field(Descr->field_label);
		q = F->q;

		if (f_v) {
			cout << "design_create::init q = " << q << endl;
		}
	}

	if (Descr->f_family) {
		if (f_v) {
			cout << "design_create::init "
					"family_name=" << Descr->family_name << endl;
		}
		if (ST.stringcmp(Descr->family_name, "PG_2_q") == 0) {
			if (f_v) {
				cout << "design_create::init PG(2," << q << ")" << endl;
			}
			if (!Descr->f_field) {
				cout << "please use option -field <label> to specify the field" << endl;
				exit(1);
			}
			create_design_PG_2_q(F, set, sz, k, verbose_level);

			f_has_set = TRUE;

			v = q * q + q + 1;
			b = v;

			char str[1000];

			snprintf(str, sizeof(str), "PG_2_q%d", q);
			prefix.assign(str);

			snprintf(str, sizeof(str), "PG_2_%d", q);
			label_txt.assign(str);

			snprintf(str, sizeof(str), "PG\\_2\\_%d", q);
			label_tex.assign(str);
		}
		else {
			cout << "design_create::init family name is not recognized" << endl;
			exit(1);
		}


		if (f_v) {
			cout << "design_create::init before compute_incidence_matrix" << endl;
		}

		compute_incidence_matrix(verbose_level);

		if (f_v) {
			cout << "design_create::init after compute_incidence_matrix" << endl;
		}


	}
	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "design_create::init "
					"from catalogue not yet implemented" << endl;
		}
		//int nb_iso;
		//knowledge_base K;

		exit(1);

		}
	else if (Descr->f_list_of_blocks_coded) {

		if (f_v) {
			cout << "design_create::init "
					"list of blocks_coded" << endl;
		}

		degree = Descr->list_of_blocks_coded_v;
		k = Descr->list_of_blocks_coded_k;




		Get_lint_vector_from_label(Descr->list_of_blocks_coded_label, set, sz, 0 /* verbose_level */);
		//Lint_vec_scan(Descr->list_of_blocks_text, set, sz);

		f_has_set = TRUE;
		v = degree;
		b = sz;

		char str[1000];

		snprintf(str, sizeof(str), "blocks_v%d_k%d", degree, k);
		prefix.assign(str);

		snprintf(str, sizeof(str), "blocks_v%d_k%d", degree, k);
		label_txt.assign(str);

		snprintf(str, sizeof(str), "blocks\\_v%d\\_k%d", degree, k);
		label_tex.assign(str);

		A = NEW_OBJECT(actions::action);

		int f_no_base = FALSE;

		if (Descr->f_no_group) {
			f_no_base = TRUE;
		}
		A->init_symmetric_group(degree, f_no_base, verbose_level);

		A2 = NEW_OBJECT(actions::action);
		A2->induced_action_on_k_subsets(*A, k, verbose_level);

		Aut = NULL;
		Aut_on_lines = NULL;
		f_has_group = FALSE;
		Sg = NULL;

		if (f_v) {
			cout << "design_create::init before compute_incidence_matrix" << endl;
		}

		compute_incidence_matrix(verbose_level);

		if (f_v) {
			cout << "design_create::init after compute_incidence_matrix" << endl;
		}

	}
	else if (Descr->f_list_of_sets_coded) {

		if (f_v) {
			cout << "design_create::init "
					"list of sets" << endl;
		}

		degree = Descr->list_of_sets_coded_v;

		Get_lint_vector_from_label(Descr->list_of_sets_coded_label, set, sz, 0 /* verbose_level */);
		//Lint_vec_scan(Descr->list_of_blocks_text, set, sz);

		f_has_set = TRUE;
		v = degree;
		b = sz;

		char str[1000];

		snprintf(str, sizeof(str), "sets_v%d", degree);
		prefix.assign(str);

		snprintf(str, sizeof(str), "sets_v%d", degree);
		label_txt.assign(str);

		snprintf(str, sizeof(str), "sets\\_v%d", degree);
		label_tex.assign(str);

		A = NEW_OBJECT(actions::action);

		int f_no_base = FALSE;

		if (Descr->f_no_group) {
			f_no_base = TRUE;
		}
		A->init_symmetric_group(degree, f_no_base, verbose_level);

		//A2 = NEW_OBJECT(actions::action);
		//A2->induced_action_on_k_subsets(*A, k, verbose_level);

		Aut = NULL;
		Aut_on_lines = NULL;
		f_has_group = FALSE;
		Sg = NULL;


		if (f_v) {
			cout << "design_create::init before compute_incidence_matrix" << endl;
		}

		compute_incidence_matrix(verbose_level);

		if (f_v) {
			cout << "design_create::init after compute_incidence_matrix" << endl;
		}


	}
	else if (Descr->f_list_of_blocks_coded_from_file) {

		if (f_v) {
			cout << "design_create::init "
					"list of blocks from file " << Descr->list_of_blocks_coded_from_file_fname << endl;
		}

		degree = Descr->list_of_blocks_coded_v;
		k = Descr->list_of_blocks_coded_k;

		orbiter_kernel_system::file_io Fio;
		int m, n;

		Fio.lint_matrix_read_csv(Descr->list_of_blocks_coded_from_file_fname,
				set, m, n, verbose_level);


		if (n != 1) {
			cout << "design_create::init f_list_of_blocks_from_file n != 1" << endl;
			exit(1);
		}
		sz = m;

		f_has_set = TRUE;
		v = degree;
		b = sz;

		char str[1000];

		snprintf(str, sizeof(str), "blocks_v%d_k%d", degree, k);
		prefix.assign(str);

		snprintf(str, sizeof(str), "blocks_v%d_k%d", degree, k);
		label_txt.assign(str);

		snprintf(str, sizeof(str), "blocks\\_v%d\\_k%d", degree, k);
		label_tex.assign(str);


		A = NEW_OBJECT(actions::action);

		int f_no_base = FALSE;

		if (Descr->f_no_group) {
			f_no_base = TRUE;
		}

		A->init_symmetric_group(degree, f_no_base, verbose_level);

		A2 = NEW_OBJECT(actions::action);
		A2->induced_action_on_k_subsets(*A, k, verbose_level);

		Aut = NULL;
		Aut_on_lines = NULL;
		f_has_group = FALSE;
		Sg = NULL;


		if (f_v) {
			cout << "design_create::init before compute_incidence_matrix" << endl;
		}

		compute_incidence_matrix(verbose_level);

		if (f_v) {
			cout << "design_create::init after compute_incidence_matrix" << endl;
		}


	}


	else if (Descr->f_list_of_blocks_from_file) {

		if (f_v) {
			cout << "design_create::init "
					"f_list_of_blocks_from_file " << Descr->list_of_blocks_from_file_fname << endl;
		}

		degree = Descr->list_of_blocks_from_file_v;

		orbiter_kernel_system::file_io Fio;
		int m, k;
		int *blocks;

		Fio.int_matrix_read_csv(
				Descr->list_of_blocks_from_file_fname,
				blocks, m, k, verbose_level);



		f_has_set = FALSE;
		v = degree;
		b = sz;

		char str[1000];

		snprintf(str, sizeof(str), "blocks_v%d_k%d", degree, k);
		prefix.assign(str);

		snprintf(str, sizeof(str), "blocks_v%d_k%d", degree, k);
		label_txt.assign(str);

		snprintf(str, sizeof(str), "blocks\\_v%d\\_k%d", degree, k);
		label_tex.assign(str);


		A = NEW_OBJECT(actions::action);

		int f_no_base = FALSE;

		if (Descr->f_no_group) {
			f_no_base = TRUE;
		}

		A->init_symmetric_group(degree, f_no_base, verbose_level);

		A2 = NEW_OBJECT(actions::action);
		A2->induced_action_on_k_subsets(*A, k, verbose_level);

		Aut = NULL;
		Aut_on_lines = NULL;
		f_has_group = FALSE;
		Sg = NULL;


		if (f_v) {
			cout << "design_create::init before compute_incidence_matrix_from_blocks" << endl;
		}

		compute_incidence_matrix_from_blocks(blocks, b, k, verbose_level);

		if (f_v) {
			cout << "design_create::init after compute_incidence_matrix_from_blocks" << endl;
		}


	}


	else if (Descr->f_wreath_product_designs) {

		if (f_v) {
			cout << "design_create::init "
					"f_wreath_product_designs" << endl;
		}

		int n;

		n = Descr->wreath_product_designs_n;
		k = Descr->wreath_product_designs_k;


		//Orbiter->Lint_vec.scan(Descr->list_of_blocks_text, set, sz);

		degree = 2 * n;

		combinatorics::combinatorics_domain Combi;
		long int nb_blocks;


		Combi.create_wreath_product_design(n, k,
				set, nb_blocks, verbose_level);

		if (f_v) {
			cout << "design_create::init "
					"f_wreath_product_designs nb_blocks=" << nb_blocks << endl;
		}

		sz = nb_blocks;
		//Orbiter->Lint_vec.scan(Descr->list_of_blocks_text, set, sz);

		f_has_set = TRUE;
		v = degree;
		b = sz;

		char str[1000];

		snprintf(str, sizeof(str), "wreath_product_designs_n%d_k%d", n, k);
		prefix.assign(str);

		snprintf(str, sizeof(str), "wreath_product_designs_n%d_k%d", n, k);
		label_txt.assign(str);

		snprintf(str, sizeof(str), "wreath\\_product\\_designs\\_n%d\\_k%d", n, k);
		label_tex.assign(str);

		A = NEW_OBJECT(actions::action);

		int f_no_base = FALSE;

		if (Descr->f_no_group) {
			f_no_base = TRUE;
		}

		A->init_symmetric_group(degree, f_no_base, verbose_level);

		A2 = NEW_OBJECT(actions::action);
		A2->induced_action_on_k_subsets(*A, k, verbose_level);

		Aut = NULL;
		Aut_on_lines = NULL;
		f_has_group = FALSE;
		Sg = NULL;


		if (f_v) {
			cout << "design_create::init before compute_incidence_matrix" << endl;
		}

		compute_incidence_matrix(verbose_level);

		if (f_v) {
			cout << "design_create::init after compute_incidence_matrix" << endl;
		}



	}
	else {
		cout << "design_create::init no design created" << endl;
		sz = 0;
		f_has_group = FALSE;


		//exit(1);
	}



	if (f_has_group) {
		cout << "design_create::init the stabilizer is:" << endl;
		Sg->print_generators_tex(cout);
	}
	else {
		cout << "design_create::init stabilizer is not available" << endl;
	}



	if (f_v) {
		cout << "design_create::init done" << endl;
	}
}

void design_create::create_design_PG_2_q(field_theory::finite_field *F,
		long int *&set, int &sz, int &k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_create::create_design_PG_2_q" << endl;
	}

	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;
	int j;
	int f_semilinear;
	//int *block;

	if (F->e > 1) {
		f_semilinear = TRUE;
	}
	else {
		f_semilinear = FALSE;
	}
	PA = NEW_OBJECT(projective_geometry::projective_space_with_action);
	PA->init(F, 2 /* n */, f_semilinear,
			TRUE /*f_init_incidence_structure*/, verbose_level);

	P = PA->P;


	design_create::k = q + 1;
	k = q + 1;
	degree = P->N_points;

	block = NEW_int(k);
	sz = P->N_lines;
	set = NEW_lint(sz);
	for (j = 0; j < sz; j++) {
		Int_vec_copy(P->Implementation->Lines + j * k, block, k);
		Sorting.int_vec_heapsort(block, k);
		set[j] = Combi.rank_k_subset(block, P->N_points, k);
		if (f_v) {
			cout << "block " << j << " / " << sz << " : ";
			Int_vec_print(cout, block, k);
			cout << " : " << set[j] << endl;
		}
	}
	Sorting.lint_vec_heapsort(set, sz);
	if (f_v) {
		cout << "design : ";
		Lint_vec_print(cout, set, sz);
		cout << endl;
	}

	if (f_v) {
		cout << "design_create::create_design_PG_2_q creating actions" << endl;
	}
	A = NEW_OBJECT(actions::action);

	int f_no_base = FALSE;

	A->init_symmetric_group(degree, f_no_base, verbose_level);

	A2 = NEW_OBJECT(actions::action);
	A2->induced_action_on_k_subsets(*A, k, verbose_level);

	Aut = PA->A;
	Aut_on_lines = PA->A_on_lines;
	f_has_group = TRUE;
	Sg = Aut->Strong_gens;


	//Aonk = A2->G.on_k_subsets;

	if (f_v) {
		cout << "design_create::create_design_PG_2_q creating actions done" << endl;
	}

	if (f_v) {
		cout << "design_create::create_design_PG_2_q done" << endl;
	}
}

void design_create::unrank_block_in_PG_2_q(int *block,
		int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_create::unrank_block_in_PG_2_q rk=" << rk
				<< " P->N_points=" << P->N_points << " k=" << k << endl;
	}
	combinatorics::combinatorics_domain Combi;

	Combi.unrank_k_subset(rk, block, P->N_points, k);
	if (f_v) {
		cout << "design_create::unrank_block_in_PG_2_q block = ";
		Int_vec_print(cout, block, k);
		cout << endl;
	}
	if (f_v) {
		cout << "design_create::unrank_block_in_PG_2_q done" << endl;
	}
}

int design_create::rank_block_in_PG_2_q(int *block,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk;

	if (f_v) {
		cout << "design_create::rank_block_in_PG_2_q" << endl;
	}
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	Sorting.int_vec_heapsort(block, k);
	rk = Combi.rank_k_subset(block, P->N_points, k);
	if (f_v) {
		cout << "design_create::rank_block_in_PG_2_q done" << endl;
	}
	return rk;
}

int design_create::get_nb_colors_as_two_design(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::combinatorics_domain Combi;
	int nb_c;

	if (f_v) {
		cout << "design_create::get_nb_colors_as_two_design" << endl;
	}
	nb_c = Combi.binomial2(P->N_points - 2);
	if (f_v) {
		cout << "design_create::get_nb_colors_as_two_design done" << endl;
	}
	return nb_c;
}

int design_create::get_color_as_two_design_assume_sorted(long int *design, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c, i;

	if (f_v) {
		cout << "design_create::get_color_as_two_design_assume_sorted" << endl;
	}
	combinatorics::combinatorics_domain Combi;

	Combi.unrank_k_subset(design[0], block, P->N_points, k);
	if (block[0] != 0) {
		cout << "block[0] != 0" << endl;
		exit(1);
	}
	if (block[1] != 1) {
		cout << "block[1] != 1" << endl;
		exit(1);
	}
	for (i = 2; i < k; i++) {
		block[i] -= 2;
	}
	c = Combi.rank_k_subset(block + 2, P->N_points - 2, k - 2);
	if (f_v) {
		cout << "design_create::get_color_as_two_design_assume_sorted done" << endl;
	}
	return c;
}

void design_create::compute_incidence_matrix(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_create::compute_incidence_matrix" << endl;
	}

	if (f_v) {
		cout << "design_create::compute_incidence_matrix set = ";
		Lint_vec_print(cout, set, sz);
		cout << endl;
	}

	combinatorics::combinatorics_domain Combi;


	if (f_has_set) {

		if (Descr->f_list_of_sets_coded) {

			int h;

			Combi.compute_incidence_matrix_from_sets(
						v, b, set,
						incma,
						verbose_level);

			nb_inc = 0;
			for (h = 0; h < v * b; h++) {
				if (incma[h]) {
					nb_inc++;
				}
			}

		}
		else {

			nb_inc = k * b;
			Combi.compute_incidence_matrix(v, b, k, set,
					incma, verbose_level);

		}
		f_has_incma = TRUE;
	}
	else {
		cout << "design_create::compute_incidence_matrix please give a set" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "design_create::compute_incidence_matrix The incidence matrix is:" << endl;
		Int_matrix_print(incma, v, b);
	}

	if (f_v) {
		cout << "design_create::compute_incidence_matrix done" << endl;
	}

}


void design_create::compute_incidence_matrix_from_blocks(int *blocks, int nb_blocks, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_create::compute_incidence_matrix_from_blocks" << endl;
	}

	if (f_v) {
		cout << "design_create::compute_incidence_matrix_from_blocks blocks = ";
		Int_matrix_print(blocks, nb_blocks, k);
		cout << endl;
	}

	b = nb_blocks;
	int i, j, h;

	incma = NEW_int(v * b);
	Int_vec_zero(incma, v * b);

	for (j = 0; j < nb_blocks; j++) {
		for (h = 0; h < k; h++) {
			i = blocks[j * k + h];
			incma[i * b + j] = 1;
		}
	}

	f_has_incma = TRUE;

	if (f_v) {
		cout << "design_create::compute_incidence_matrix_from_blocks The incidence matrix is:" << endl;
		Int_matrix_print(incma, v, b);
	}

	if (f_v) {
		cout << "design_create::compute_incidence_matrix_from_blocks done" << endl;
	}

}


}}}



