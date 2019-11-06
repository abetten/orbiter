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

	//char prefix[1000];
	//char label_txt[1000];
	//char label_tex[1000];

	q = 0;
	F = NULL;
	k = 0;

	A = NULL;
	A2 = NULL;

	degree = 0;

	set = NULL;
	sz = 0;

	f_has_group = FALSE;
	Sg = NULL;

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
	if (P) {
		FREE_OBJECT(P);
	}
	if (block) {
		FREE_int(block);
	}
	null();
}

void design_create::init(design_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_create::init" << endl;
	}
	design_create::Descr = Descr;
	if (!Descr->f_q) {
		cout << "design_create::init !Descr->f_q" << endl;
		exit(1);
	}
	q = Descr->q;
#if 0
	if (!Descr->f_k) {
		cout << "design_create::init !Descr->f_k" << endl;
		exit(1);
	}
	k = Descr->k;
#endif
	if (f_v) {
		cout << "design_create::init q = " << q << endl;
		//cout << "design_create::init k = " << k << endl;
	}
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	if (Descr->f_family) {
		if (f_v) {
			cout << "design_create::init "
					"family_name=" << Descr->family_name << endl;
		}
		if (strcmp(Descr->family_name, "PG_2_q") == 0) {
			if (f_v) {
				cout << "design_create::init PG(2," << q << ")" << endl;
			}
			create_design_PG_2_q(F, set, sz, k, verbose_level);

			sprintf(prefix, "PG_2_q%d", q);
			sprintf(label_txt, "PG_2_%d", q);
			sprintf(label_tex, "PG\\_2\\_%d", q);
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

#if 0
		nb_iso = K.Spread_nb_reps(q, k);
		if (Descr->iso >= nb_iso) {
			cout << "design_create::init "
					"iso >= nb_iso, this spread does not exist" << endl;
			exit(1);
		}

		int *rep;

		rep = K.Spread_representative(q, k, Descr->iso, sz);
		set = NEW_int(sz);
		int_vec_copy(rep, set, sz);

		Sg = NEW_OBJECT(strong_generators);

		if (f_v) {
			cout << "design_create::init "
					"before Sg->BLT_set_from_catalogue_stabilizer" << endl;
		}

		Sg->stabilizer_of_spread_from_catalogue(A,
			q, k, Descr->iso,
			verbose_level);
		f_has_group = TRUE;

		sprintf(prefix, "catalogue_q%d_k%d_%d",
				q, k, Descr->iso);
		sprintf(label_txt, "catalogue_q%d_k%d_%d",
				q, k, Descr->iso);
		sprintf(label_tex, "catalogue\\_q%d\\_k%d\\_%d",
				q, k, Descr->iso);
		if (f_v) {
			cout << "design_create::init "
					"after Sg->BLT_set_from_catalogue_stabilizer" << endl;
		}
#endif
		}
	else {
		cout << "design_create::init we do not "
				"recognize the type of spread" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "design_create::init set = ";
		lint_vec_print(cout, set, sz);
		cout << endl;
	}

	if (f_has_group) {
		cout << "design_create::init the stabilizer is:" << endl;
		Sg->print_generators_tex(cout);
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
	//int *block;

	design_create::k = q + 1;
	k = q + 1;
	P = NEW_OBJECT(projective_space);
	P->init(2, F,
			TRUE /* f_init_incidence_structure */,
			verbose_level);
	degree = P->N_points;

	block = NEW_int(k);
	sz = P->N_lines;
	set = NEW_lint(sz);
	for (j = 0; j < sz; j++) {
		int_vec_copy(P->Lines + j * k, block, k);
		Sorting.int_vec_heapsort(block, k);
		set[j] = Combi.rank_k_subset(block, P->N_points, k);
		if (f_v) {
			cout << "block " << j << " / " << sz << " : ";
			int_vec_print(cout, block, k);
			cout << " : " << set[j] << endl;
		}
	}
	Sorting.lint_vec_heapsort(set, sz);
	if (f_v) {
		cout << "design : ";
		lint_vec_print(cout, set, sz);
		cout << endl;
	}

	if (f_v) {
		cout << "design_create::create_design_PG_2_q creating actions" << endl;
	}
	A = NEW_OBJECT(action);
	A->init_symmetric_group(degree, verbose_level);

	A2 = NEW_OBJECT(action);
	A2->induced_action_on_k_subsets(*A, k, verbose_level);

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
		int_vec_print(cout, block, k);
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


