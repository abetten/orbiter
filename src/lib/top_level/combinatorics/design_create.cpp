/*
 * design_create.cpp
 *
 *  Created on: Sep 19, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


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

	set = NULL;
	sz = 0;

	f_has_group = FALSE;
	Sg = NULL;

	PA = NULL;
	P = NULL;
	block = NULL;

	//null();
}

design_create::~design_create()
{
	freeself();
}

void design_create::null()
{
}

void design_create::freeself()
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
	null();
}

void design_create::init(design_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string_tools ST;

	if (f_v) {
		cout << "design_create::init" << endl;
	}
	design_create::Descr = Descr;

	if (Descr->f_q) {

		q = Descr->q;

		if (f_v) {
			cout << "design_create::init q = " << q << endl;
			//cout << "design_create::init k = " << k << endl;
		}
		F = NEW_OBJECT(finite_field);
		F->finite_field_init(q, FALSE /* f_without_tables */, 0);
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
			if (!Descr->f_q) {
				cout << "please use option -q <q> to specify the field order" << endl;
				exit(1);
			}
			create_design_PG_2_q(F, set, sz, k, verbose_level);

			char str[1000];

			sprintf(str, "PG_2_q%d", q);
			prefix.assign(str);

			sprintf(str, "PG_2_%d", q);
			label_txt.assign(str);

			sprintf(str, "PG\\_2\\_%d", q);
			label_tex.assign(str);
		}

	}
	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "design_create::init "
					"from catalogue" << endl;
		}
		//int nb_iso;
		//knowledge_base K;

		exit(1);

		}
	else if (Descr->f_list_of_blocks) {

		if (f_v) {
			cout << "design_create::init "
					"list of blocks" << endl;
		}

		degree = Descr->list_of_blocks_v;
		k = Descr->list_of_blocks_k;
		Orbiter->Lint_vec.scan(Descr->list_of_blocks_text, set, sz);

		char str[1000];

		sprintf(str, "blocks_v%d_k%d", degree, k);
		prefix.assign(str);

		sprintf(str, "blocks_v%d_k%d", degree, k);
		label_txt.assign(str);

		sprintf(str, "blocks\\_v%d\\_k%d", degree, k);
		label_tex.assign(str);

		A = NEW_OBJECT(action);

		int f_no_base = FALSE;

		if (Descr->f_no_group) {
			f_no_base = TRUE;
		}
		A->init_symmetric_group(degree, f_no_base, verbose_level);

		A2 = NEW_OBJECT(action);
		A2->induced_action_on_k_subsets(*A, k, verbose_level);

		Aut = NULL;
		Aut_on_lines = NULL;
		f_has_group = FALSE;
		Sg = NULL;

	}
	else if (Descr->f_list_of_blocks_from_file) {

		if (f_v) {
			cout << "design_create::init "
					"list of blocks from file " << Descr->list_of_blocks_from_file_fname << endl;
		}

		degree = Descr->list_of_blocks_v;
		k = Descr->list_of_blocks_k;

		file_io Fio;
		int m, n;

		Fio.lint_matrix_read_csv(Descr->list_of_blocks_from_file_fname,
				set, m, n, verbose_level);


		if (n != 1) {
			cout << "design_create::init f_list_of_blocks_from_file n != 1" << endl;
			exit(1);
		}
		sz = m;

		//Orbiter->Lint_vec.scan(Descr->list_of_blocks_text, set, sz);

		char str[1000];

		sprintf(str, "blocks_v%d_k%d", degree, k);
		prefix.assign(str);

		sprintf(str, "blocks_v%d_k%d", degree, k);
		label_txt.assign(str);

		sprintf(str, "blocks\\_v%d\\_k%d", degree, k);
		label_tex.assign(str);

		A = NEW_OBJECT(action);

		int f_no_base = FALSE;

		if (Descr->f_no_group) {
			f_no_base = TRUE;
		}

		A->init_symmetric_group(degree, f_no_base, verbose_level);

		A2 = NEW_OBJECT(action);
		A2->induced_action_on_k_subsets(*A, k, verbose_level);

		Aut = NULL;
		Aut_on_lines = NULL;
		f_has_group = FALSE;
		Sg = NULL;

	}
	else {
		cout << "design_create::init no design created" << endl;
		sz = 0;
		f_has_group = FALSE;
		//exit(1);
	}


	if (f_v) {
		cout << "design_create::init set = ";
		Orbiter->Lint_vec.print(cout, set, sz);
		cout << endl;
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

void design_create::create_design_PG_2_q(finite_field *F,
		long int *&set, int &sz, int &k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_create::create_design_PG_2_q" << endl;
	}

	combinatorics_domain Combi;
	sorting Sorting;
	int j;
	int f_semilinear;
	//int *block;

	if (F->e > 1) {
		f_semilinear = TRUE;
	}
	else {
		f_semilinear = FALSE;
	}
	PA = NEW_OBJECT(projective_space_with_action);
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
		Orbiter->Int_vec.copy(P->Implementation->Lines + j * k, block, k);
		Sorting.int_vec_heapsort(block, k);
		set[j] = Combi.rank_k_subset(block, P->N_points, k);
		if (f_v) {
			cout << "block " << j << " / " << sz << " : ";
			Orbiter->Int_vec.print(cout, block, k);
			cout << " : " << set[j] << endl;
		}
	}
	Sorting.lint_vec_heapsort(set, sz);
	if (f_v) {
		cout << "design : ";
		Orbiter->Lint_vec.print(cout, set, sz);
		cout << endl;
	}

	if (f_v) {
		cout << "design_create::create_design_PG_2_q creating actions" << endl;
	}
	A = NEW_OBJECT(action);

	int f_no_base = FALSE;

	A->init_symmetric_group(degree, f_no_base, verbose_level);

	A2 = NEW_OBJECT(action);
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
	combinatorics_domain Combi;

	Combi.unrank_k_subset(rk, block, P->N_points, k);
	if (f_v) {
		cout << "design_create::unrank_block_in_PG_2_q block = ";
		Orbiter->Int_vec.print(cout, block, k);
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
	combinatorics_domain Combi;
	sorting Sorting;

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
	combinatorics_domain Combi;
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
	combinatorics_domain Combi;

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

}}


