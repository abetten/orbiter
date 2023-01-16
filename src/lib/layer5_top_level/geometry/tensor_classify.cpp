/*
 * tensor_classify.cpp
 *
 *  Created on: Sep 14, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace layer5_applications {
namespace apps_geometry {


//static int wreath_rank_point_func(int *v, void *data);
//static void wreath_unrank_point_func(int *v, int rk, void *data);
static void wreath_product_print_set(std::ostream &ost, int len, long int *S, void *data);
static void wreath_product_rank_one_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);


tensor_classify::tensor_classify()
{
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
	Control = NULL;
	Poset = NULL;
	Gen = NULL;
	t0 = 0;
}

tensor_classify::~tensor_classify()
{

}

void tensor_classify::init(
		field_theory::finite_field *F, groups::linear_group *LG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "tensor_classify::init" << endl;
	}
	t0 = Os.os_ticks();

	q = F->q;

	//A = NEW_OBJECT(action);
	//A = LG->A_linear;
	A = LG->A2;


#if 0
	F = NEW_OBJECT(finite_field);

	F->init(q, 0);

	if (f_v) {
		cout << "tensor_classify::init before "
				"A->init_wreath_product_group" << endl;
	}
	A->init_wreath_product_group(nb_factors, n, F,
			verbose_level);
	if (f_v) {
		cout << "tensor_classify::init after "
				"A->init_wreath_product_group" << endl;
	}
#endif


	A0 = LG->A_linear;
	W = A0->G.wreath_product_group;

	vector_space_dimension = W->dimension_of_tensor_action;
	nb_factors = W->nb_factors;
	n = W->dimension_of_matrix_group;


	if (!A0->f_has_strong_generators) {
		cout << "tensor_classify::init action A0 does not "
				"have strong generators" << endl;
		exit(1);
	}

	v = NEW_int(vector_space_dimension);

	SG = A0->Strong_gens;
	SG->group_order(go);

	if (f_v) {
		cout << "tensor_classify::init The group " << A->label
				<< " has order " << go
				<< " and permutation degree " << A->degree << endl;
	}


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

	if (f_v) {
		SG->print_generators(cout);
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

	if (f_v) {
		SG->print_generators_gap(cout);
	}



	if (f_v) {
		cout << "tensor_classify::init done" << endl;
	}

}


void tensor_classify::classify_poset(int depth,
		poset_classification::poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tensor_classify::classify_poset" << endl;
	}


	if (f_v) {
		cout << "tensor_classify::classify_poset "
				"before create_restricted_action_on_rank_one_tensors" << endl;
	}
	create_restricted_action_on_rank_one_tensors(verbose_level);
	if (f_v) {
		cout << "tensor_classify::classify_poset "
				"after create_restricted_action_on_rank_one_tensors" << endl;
	}

	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
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

	Poset->f_print_function = TRUE;
	Poset->print_function = wreath_product_print_set;
	Poset->print_function_data = this;

	Control->f_depth = TRUE;
	Control->depth = depth;

	if (f_v) {
		cout << "tensor_classify::classify_poset before "
				"Gen->initialize_and_allocate_root_node" << endl;
		}
	Gen = NEW_OBJECT(poset_classification::poset_classification);

	Gen->initialize_and_allocate_root_node(
			Control, Poset,
			depth /* sz */, verbose_level);
	if (f_v) {
		cout << "tensor_classify::classify_poset after "
				"Gen->initialize_and_allocate_root_node" << endl;
	}


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

		points[i] = /*W->perm_offset_i[nb_factors] +*/ b;
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
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}


	if (len == 0) {
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
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



void tensor_classify::report(int f_poset_classify, int poset_classify_depth,
		graphics::layered_graph_draw_options *draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_OK;
	int i, j; //, c;

	if (f_v) {
		cout << "tensor_classify::report" << endl;
	}



	orbiter_kernel_system::file_io Fio;
	orbiter_kernel_system::latex_interface L;

	string fname, title, author, extra_praeamble;
	char str[1000];
	//int f_with_stabilizers = TRUE;

	snprintf(str, 1000, "Wreath product $%s$", W->label_tex.c_str());
	title.assign(str);

	author.assign("Orbiter");
	snprintf(str, 1000, "WreathProduct_q%d_n%d.tex", W->q, W->nb_factors);
	fname.assign(str);

	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;

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
			extra_praeamble /* extra_praeamble */);

		fp << "\\section{The field of order " << q << "}" << endl;
		fp << "\\noindent The field ${\\mathbb F}_{"
				<< W->q
				<< "}$ :\\\\" << endl;
		W->F->cheat_sheet(fp, verbose_level);

		if (f_v) {
			cout << "tensor_classify::report before W->report" << endl;
		}

		W->report(fp, verbose_level);

		if (f_v) {
			cout << "tensor_classify::report after W->report" << endl;
		}

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
		A->report(fp, A->f_has_sims, A->Sims, A->f_has_strong_generators, A->Strong_gens,
				draw_options,
				verbose_level);

		if (f_v) {
			cout << "tensor_classify::report after A->report" << endl;
		}


		if (f_poset_classify) {



#if 0
			{
			char fname_poset[1000];

			Gen->draw_poset_fname_base_poset_lvl(fname_poset, poset_classify_depth);
			Gen->draw_poset(fname_poset,
					poset_classify_depth /*depth*/,
					0 /* data1 */,
					FALSE /* f_embedded */,
					FALSE /* f_sideways */,
					100 /* rad */,
					verbose_level);
			}
#endif


			fp << endl;
			fp << "\\section{Poset Classification}" << endl;
			fp << endl;

			poset_classification::poset_classification_report_options Opt;

			Gen->report2(fp, &Opt, verbose_level);
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
				data_structures::sorting Sorting;

				Sorting.lint_vec_heapsort(data, orbit_length);

				fp << "$$" << endl;
				L.print_lint_matrix_tex(fp, data, (orbit_length + 9)/ 10, 10);
				fp << "$$" << endl;

				data_structures::tally C;

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

	if (f_v) {
		cout << "tensor_classify::report done" << endl;
	}
}




#if 0
static int wreath_rank_point_func(int *v, void *data)
{
	tensor_classify *T;
	int rk;

	T = (tensor_classify *) data;

	rk = T->W->tensor_PG_rank(v);

	return rk;
}

static void wreath_unrank_point_func(int *v, int rk, void *data)
{
	tensor_classify *T;

	T = (tensor_classify *) data;

	T->W->tensor_PG_unrank(v, rk);

}
#endif


static void wreath_product_print_set(std::ostream &ost, int len, long int *S, void *data)
{
	tensor_classify *T;
	int i;

	T = (tensor_classify *) data;
	cout << "set: ";
	Lint_vec_print(cout, S, len);
	cout << endl;
	for (i = 0; i < len; i++) {
		T->F->PG_element_unrank_modified(T->v,
				1, T->vector_space_dimension, S[i]);
		cout << S[i] << " : ";
		Int_vec_print(cout, T->v, T->vector_space_dimension);
		cout << endl;
	}
}




static void wreath_product_rank_one_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	tensor_classify *T = (tensor_classify *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "wreath_product_rank_one_early_test_func_callback for set ";
		Lint_vec_print(cout, S, len);
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


}}}


