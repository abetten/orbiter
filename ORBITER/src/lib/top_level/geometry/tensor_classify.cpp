/*
 * tensor_classify.cpp
 *
 *  Created on: Sep 14, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace top_level {


tensor_classify::tensor_classify()
{
	argc = 0;
	argv = NULL;
	nb_factors = 0;
	vector_space_dimension = 0;
	v = NULL;
	n = 0;
	q = 0;
	SG = NULL;
	F = NULL;
	A = NULL;
	A0 = NULL;
	Ar = NULL;
	nb_points = 0;
	points = NULL;
	W = NULL;
	VS = NULL;
	Poset = NULL;
	Gen = NULL;
	t0 = 0;
}

tensor_classify::~tensor_classify()
{

}

void tensor_classify::init(int argc, const char **argv,
		int nb_factors, int n, int q, int depth,
		int f_permutations, int f_orbits, int f_tensor_ranks,
		int f_orbits_restricted, const char *orbits_restricted_fname,
		int f_orbits_restricted_compute,
		int f_report,
		int f_poset_classify, int poset_classify_depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;
	os_interface Os;

	if (f_v) {
		cout << "tensor_classify::init" << endl;
	}
	t0 = Os.os_ticks();

	tensor_classify::argc = argc;
	tensor_classify::argv = argv;
	tensor_classify::nb_factors = nb_factors;
	tensor_classify::n = n;
	tensor_classify::q = q;

	A = NEW_OBJECT(action);



	F = NEW_OBJECT(finite_field);

	F->init(q, 0);

#if 0
	cout << "tensor_classify::init before "
			"A->init_wreath_product_group_and_restrict" << endl;
	A->init_wreath_product_group_and_restrict(nb_factors, n,
			F, f_tensor_ranks,
			verbose_level);
	cout << "tensor_classify::init after "
			"A->init_wreath_product_group_and_restrict" << endl;

	if (!A->f_has_subaction) {
		cout << "tensor_classify::init action "
				"A does not have a subaction" << endl;
		exit(1);
	}
	A0 = A->subaction;

	W = A0->G.wreath_product_group;
#else
	cout << "tensor_classify::init before "
			"A->init_wreath_product_group" << endl;
	A->init_wreath_product_group(nb_factors, n,
			F, f_tensor_ranks,
			verbose_level);
	cout << "tensor_classify::init after "
			"A->init_wreath_product_group" << endl;

	A0 = A;
	W = A0->G.wreath_product_group;

#if 0
	int nb_points;
	int *points;
	action *Awr;

	cout << "W->degree_of_tensor_action=" << W->degree_of_tensor_action << endl;
	nb_points = W->degree_of_tensor_action;
	points = NEW_int(nb_points);
	for (i = 0; i < nb_points; i++) {
		points[i] = W->perm_offset_i[nb_factors] + i;
	}

	if (f_v) {
		cout << "tensor_classify "
				"before A_wreath->restricted_action" << endl;
	}
	Awr = A->restricted_action(points, nb_points,
			verbose_level);
	Awr->f_is_linear = TRUE;
#endif

#endif

	vector_space_dimension = W->dimension_of_tensor_action;


	if (!A0->f_has_strong_generators) {
		cout << "tensor_classify::init action A0 does not "
				"have strong generators" << endl;
		exit(1);
		}

	v = NEW_int(vector_space_dimension);

	SG = A0->Strong_gens;
	SG->group_order(go);

	cout << "tensor_classify::init The group " << A->label
			<< " has order " << go
			<< " and permutation degree " << A->degree << endl;




#if 0
	i = SG->gens->len - 1;
	cout << "generator " << i << " is: " << endl;


	int h;

	cout << "computing image of 2:" << endl;
	h = A->element_image_of(2,
			SG->gens->ith(i), 10 /*verbose_level - 2*/);


	for (j = 0; j < A->degree; j++) {
		h = A->element_image_of(j,
				SG->gens->ith(i), verbose_level - 2);
		cout << j << " -> " << h << endl;
	}

		A->element_print_as_permutation(SG->gens->ith(i), cout);
	cout << endl;
#endif

	cout << "tensor_classify::init Generators are:" << endl;
	for (i = 0; i < SG->gens->len; i++) {
		cout << "generator " << i << " / "
				<< SG->gens->len << " is: " << endl;
		A->element_print_quick(SG->gens->ith(i), cout);
		cout << "as permutation: " << endl;
		if (A->degree < 400) {
			A->element_print_as_permutation_with_offset(
					SG->gens->ith(i), cout,
					0 /* offset*/,
					TRUE /* f_do_it_anyway_even_for_big_degree*/,
					TRUE /* f_print_cycles_of_length_one*/,
					0 /* verbose_level*/);
			//A->element_print_as_permutation(SG->gens->ith(i), cout);
			cout << endl;
		} else {
			cout << "too big to print" << endl;
		}
	}
	cout << "tensor_classify::init Generators as permutations are:" << endl;



	if (A->degree < 400) {
		for (i = 0; i < SG->gens->len; i++) {
			A->element_print_as_permutation(SG->gens->ith(i), cout);
			cout << endl;
		}
	}
	else {
		cout << "too big to print" << endl;
	}

#if 0
	cout << "tensor_classify::init Generators in ASCII format are:" << endl;
		cout << SG->gens->len << endl;
		for (i = 0; i < SG->gens->len; i++) {
			A->element_print_for_make_element(
					SG->gens->ith(i), cout);
				cout << endl;
		}
		cout << -1 << endl;
#endif

	cout << "tensor_classify::init Generators in GAP format are:" << endl;
	if (A->degree < 200) {
		cout << "G := Group([";
		for (i = 0; i < SG->gens->len; i++) {
			A->element_print_as_permutation_with_offset(
					SG->gens->ith(i), cout,
					1 /*offset*/,
					TRUE /* f_do_it_anyway_even_for_big_degree */,
					FALSE /* f_print_cycles_of_length_one */,
					0 /* verbose_level*/);
			if (i < SG->gens->len - 1) {
				cout << ", " << endl;
			}
		}
		cout << "]);" << endl;
	}
	else {
		cout << "too big to print" << endl;
	}
	cout << "tensor_classify::init "
			"Generators in compact permutation form are:" << endl;
	if (A->degree < 200) {
		cout << SG->gens->len << " " << A->degree << endl;
		for (i = 0; i < SG->gens->len; i++) {
			for (j = 0; j < A->degree; j++) {
				a = A->element_image_of(j,
						SG->gens->ith(i), 0 /* verbose_level */);
				cout << a << " ";
				}
			cout << endl;
			}
		cout << "-1" << endl;
	}
	else {
		cout << "too big to print" << endl;
	}

	if (f_poset_classify) {
		classify_poset(poset_classify_depth, verbose_level + 10);
	}

	if (f_report) {
		cout << "report:" << endl;


		file_io Fio;
		latex_interface L;

		{
		char fname[1000];
		char title[1000];
		char author[1000];
		//int f_with_stabilizers = TRUE;

		sprintf(title, "Wreath product $%s$", W->label_tex);
		sprintf(author, "Orbiter");
		sprintf(fname, "WreathProduct_q%d_n%d.tex", W->q, W->nb_factors);

			{
			ofstream fp(fname);
			latex_interface L;

			//latex_head_easy(fp);
			L.head(fp,
				FALSE /* f_book */,
				TRUE /* f_title */,
				title, author,
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				TRUE /*f_enlarged_page */,
				TRUE /* f_pagenumbers*/,
				NULL /* extra_praeamble */);

			fp << "\\section{The field of order " << q << "}" << endl;
			fp << "\\noindent The field ${\\mathbb F}_{"
					<< W->q
					<< "}$ :\\\\" << endl;
			W->F->cheat_sheet(fp, verbose_level);


			W->report(fp, verbose_level);

			fp << "\\section{Generators}" << endl;
			for (i = 0; i < SG->gens->len; i++) {
				fp << "$$" << endl;
				A->element_print_latex(SG->gens->ith(i), fp);
				if (i < SG->gens->len - 1) {
					fp << ", " << endl;
				}
				fp << "$$" << endl;
			}


			fp << "\\section{The Group}" << endl;
			A->report(fp, A->f_has_sims, A->Sims, A->f_has_strong_generators, A->Strong_gens, verbose_level);


			if (f_poset_classify) {


				{
				char fname_poset[1000];

				Gen->draw_poset_fname_base_poset_lvl(fname_poset, poset_classify_depth);
				Gen->draw_poset(fname_poset,
						poset_classify_depth /*depth*/,
						0 /* data1 */,
						FALSE /* f_embedded */,
						FALSE /* f_sideways */,
						verbose_level);
				}


				fp << endl;
				fp << "\\section{Poset Classification}" << endl;
				fp << endl;


				Gen->report(fp);
				fp << "\\subsection*{Orbits at level " << poset_classify_depth << "}" << endl;
				int nb_orbits, orbit_idx;

				nb_orbits = Gen->nb_orbits_at_level(poset_classify_depth);
				for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
					fp << "\\subsubsection*{Orbit " << orbit_idx << " / " << nb_orbits << "}" << endl;

					long int *Orbit; // orbit_length * depth
					int orbit_length;

					cout << "before get_whole_orbit orbit_idx=" << orbit_idx << endl;

					Gen->get_whole_orbit(
							poset_classify_depth, orbit_idx,
							Orbit, orbit_length, verbose_level);

					long int *data;

					data = NEW_lint(orbit_length);

					for (i = 0; i < orbit_length; i++) {

						fp << "set " << i << " / " << orbit_length << " is: ";


						uint32_t a, b;

						a = 0;
						for (j = 0; j < poset_classify_depth; j++) {
							b = W->rank_one_tensors[Orbit[i * poset_classify_depth + j]];
							a ^= b;
						}

						for (j = 0; j < poset_classify_depth; j++) {
							fp << Orbit[i * poset_classify_depth + j];
							if (j < poset_classify_depth - 1) {
								fp << ", ";
							}
						}
						fp << "= ";
						for (j = 0; j < poset_classify_depth; j++) {
							b = W->rank_one_tensors[Orbit[i * poset_classify_depth + j]];
							fp << b;
							if (j < poset_classify_depth - 1) {
								fp << ", ";
							}
						}
						fp << " = " << a;
						data[i] = a;
						fp << "\\\\" << endl;
					}
					sorting Sorting;

					Sorting.lint_vec_heapsort(data, orbit_length);

					fp << "$$" << endl;
					L.print_lint_matrix_tex(fp, data, (orbit_length + 9)/ 10, 10);
					fp << "$$" << endl;

					classify C;

					C.init_lint(data, orbit_length, TRUE, 0);
					fp << "$$";
					C.print_naked_tex(fp, TRUE /* f_backwards */);
					fp << "$$";
					FREE_lint(data);
				}
			}

			L.foot(fp);
			}
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}


		cout << "report done" << endl;
	}





	int *result = NULL;

	cout << "time check: ";
	Os.time_check(cout, t0);
	cout << endl;

	cout << "tensor_classify::init "
			"before wreath_product_orbits_CUDA:" << endl;
	cout << __FILE__ << ":" << __LINE__ << endl;

	int nb_gens, degree;

	if (f_permutations) {
		W->compute_permutations(SG, A, result, nb_gens, degree, nb_factors, verbose_level);
	}
	//wreath_product_orbits_CUDA(W, SG, A, result, nb_gens, degree, nb_factors, verbose_level);

	if (f_orbits) {
		W->orbits(SG, A, result, nb_gens, degree, nb_factors, verbose_level);
	}
	if (f_orbits_restricted) {
		W->orbits_restricted(SG, A, result, nb_gens, degree, nb_factors, orbits_restricted_fname, verbose_level);

	}
	if (f_orbits_restricted_compute) {
		W->orbits_restricted_compute(SG, A, result, nb_gens, degree, nb_factors, orbits_restricted_fname, verbose_level);

	}

	cout << "time check: ";
	Os.time_check(cout, t0);
	cout << endl;

	cout << "tensor_classify::init "
			"after wreath_product_orbits_CUDA:" << endl;
	cout << __FILE__ << ":" << __LINE__ << endl;
	cout << "we found " << nb_gens << " generators of degree " << degree << endl;



//	schreier *Sch;
//
//	Sch = NEW_OBJECT(schreier);
//
//	cout << "before Sch->init_images_only" << endl;
//	Sch->init_images_only(nb_gens,
//			degree, result, verbose_level);
//
//	cout << "nb_gens: " << nb_gens << endl;
//
//	cout << "computing point orbits from image table:" << endl;
//	Sch->compute_all_point_orbits(0);
//
//	Sch->print_orbit_lengths(cout);
//
//	cout << "time check: ";
//	time_check(cout, t0);
//	cout << endl;
//
//
//	cout << "computing point orbits from image table done" << endl;
//	cout << "We found " << Sch->nb_orbits << " orbits" << endl;
//
//
//#if 0
//	A->perform_tests(SG, verbose_level);
//#endif
//
//	exit(0);
//
//
}


void tensor_classify::classify_poset(int depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tensor_classify::classify_poset" << endl;
	}
	Gen = NEW_OBJECT(poset_classification);

	Gen->read_arguments(argc, argv, 0);

	//Gen->prefix[0] = 0;
	sprintf(Gen->fname_base, "wreath_%d_%d_%d", nb_factors, n, q);

	Gen->f_max_depth = TRUE;
	Gen->max_depth = depth;
	Gen->depth = depth;

	if (f_v) {
		cout << "tensor_classify::classify_poset before create_restricted_action_on_rank_one_tensors" << endl;
	}
	create_restricted_action_on_rank_one_tensors(verbose_level);
	if (f_v) {
		cout << "tensor_classify::classify_poset after create_restricted_action_on_rank_one_tensors" << endl;
	}

#if 0
	VS = NEW_OBJECT(vector_space);
	VS->init(F,
			vector_space_dimension /* dimension */,
			verbose_level - 1);
	VS->init_rank_functions(
			wreath_rank_point_func,
			wreath_unrank_point_func,
			this,
			verbose_level - 1);


	Poset = NEW_OBJECT(poset);
	Poset->init_subspace_lattice(
			A0, A,
			SG,
			VS,
			verbose_level);
#else
	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, Ar,
			SG,
			verbose_level);

	if (f_v) {
		cout << "tensor_classify::classify_poset before "
				"Poset->add_testing_without_group" << endl;
		}
	Poset->add_testing_without_group(
			wreath_product_rank_one_early_test_func_callback,
			this /* void *data */,
			verbose_level);
#endif

	if (f_v) {
		cout << "tensor_classify::classify_poset before Gen->init" << endl;
		}
	Gen->init(Poset, depth /* sz */, verbose_level);
	if (f_v) {
		cout << "tensor_classify::classify_poset after Gen->init" << endl;
		}


	Gen->f_print_function = TRUE;
	Gen->print_function = wreath_product_print_set;
	Gen->print_function_data = this;

	int nb_nodes = 1000;

	if (f_v) {
		cout << "tensor_classify::classify_poset "
				"before Gen->init_poset_orbit_node" << endl;
		}
	Gen->init_poset_orbit_node(nb_nodes, verbose_level - 1);
	if (f_v) {
		cout << "tensor_classify::classify_poset "
				"calling Gen->init_root_node" << endl;
		}
	Gen->root[0].init_root_node(Gen, verbose_level - 1);

	//int schreier_depth;
	int f_use_invariant_subset_if_available;
	int f_debug;

	//schreier_depth = Gen->depth;
	f_use_invariant_subset_if_available = TRUE;
	f_debug = FALSE;

	//int t0 = os_ticks();

	if (f_v) {
		cout << "tensor_classify::classify_poset before Gen->main" << endl;
		cout << "A=";
		A->print_info();
		cout << "A0=";
		A0->print_info();
		}


	//Gen->f_allowed_to_show_group_elements = TRUE;

	if (f_v) {
		cout << "tensor_classify::classify_poset "
				"before Gen->main, verbose_level=" << verbose_level << endl;
		}
	Gen->main(t0,
		depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level);
	if (f_v) {
		cout << "tensor_classify::classify_poset "
				"after Gen->main" << endl;
		}
	if (f_v) {
		cout << "tensor_classify::classify_poset done" << endl;
	}
}

void tensor_classify::create_restricted_action_on_rank_one_tensors(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "tensor_classify::create_restricted_action_on_rank_one_tensors" << endl;
	}

	nb_points = W->nb_rank_one_tensors;
	points = NEW_lint(nb_points);
	for (i = 0; i < nb_points; i++) {
		uint32_t a, b;

		a = W->rank_one_tensors[i];
		b = W->affine_rank_to_PG_rank(a);

		points[i] = W->perm_offset_i[nb_factors] + b;
	}

	if (f_v) {
		cout << "tensor_classify::create_restricted_action_on_rank_one_tensors "
				"before A->restricted_action" << endl;
	}
	Ar = A->restricted_action(points, nb_points,
			verbose_level);
	Ar->f_is_linear = TRUE;
	if (f_v) {
		cout << "tensor_classify::create_restricted_action_on_rank_one_tensors "
				"after A->restricted_action" << endl;
	}
	if (f_v) {
		cout << "tensor_classify::create_restricted_action_on_rank_one_tensors done" << endl;
	}
}


void tensor_classify::early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_OK;
	int i, j, c;

	if (f_v) {
		cout << "tensor_classify::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}


	if (len == 0) {
		lint_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
		}
	else {
		nb_good_candidates = 0;

		if (f_vv) {
			cout << "tensor_classify::early_test_func before testing" << endl;
			}
		for (j = 0; j < nb_candidates; j++) {


			if (f_vv) {
				cout << "tensor_classify::early_test_func "
						"testing " << j << " / "
						<< nb_candidates << endl;
				}

			f_OK = TRUE;
			c = candidates[j];

			for (i = 0; i < len; i++) {
				if (S[i] == c) {
					f_OK = FALSE;
					break;
				}
			}



			if (f_OK) {
				good_candidates[nb_good_candidates++] =
						candidates[j];
				}
			} // next j
		} // else
	if (f_v) {
		cout << "tensor_classify::early_test_func done" << endl;
	}
}


int wreath_rank_point_func(int *v, void *data)
{
	tensor_classify *T;
	int rk;

	T = (tensor_classify *) data;
	//AG_element_rank(LS->Fq->q, v, 1, LS->vector_space_dimension, rk);
	//T->F->PG_element_rank_modified(v, 1, T->vector_space_dimension, rk);
	rk = T->W->tensor_PG_rank(v);

	//uint32_t tensor_PG_rank(int *tensor);

	return rk;
}

void wreath_unrank_point_func(int *v, int rk, void *data)
{
	tensor_classify *T;

	T = (tensor_classify *) data;
	//AG_element_unrank(LS->Fq->q, v, 1, LS->vector_space_dimension, rk);
	//T->F->PG_element_unrank_modified(v, 1, T->vector_space_dimension, rk);
	T->W->tensor_PG_unrank(v, rk);

	//void tensor_PG_unrank(int *tensor, uint32_t PG_rk);


}


void wreath_product_print_set(ostream &ost, int len, long int *S, void *data)
{
	tensor_classify *T;
	int i;

	T = (tensor_classify *) data;
	cout << "set: ";
	lint_vec_print(cout, S, len);
	cout << endl;
	for (i = 0; i < len; i++) {
		T->F->PG_element_unrank_modified(T->v,
				1, T->vector_space_dimension, S[i]);
		cout << S[i] << " : ";
		int_vec_print(cout, T->v, T->vector_space_dimension);
		cout << endl;
	}
}




void wreath_product_rank_one_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	tensor_classify *T = (tensor_classify *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "wreath_product_rank_one_early_test_func_callback for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	T->early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "wreath_product_rank_one_early_test_func_callback done" << endl;
		}
}


}}


