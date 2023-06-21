/*
 * classify_five_plus_one.cpp
 *
 *  Created on: Jan 29, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_double_sixes {


static void callback_partial_ovoid_test_early(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);


classify_five_plus_one::classify_five_plus_one()
{
	q = 0;
	F = NULL;
	A = NULL;
	Surf_A = NULL;
	Surf = NULL;


	A2 = NULL;
	AW = NULL;
	Elt0 = NULL;
	Elt1 = NULL;

	SG_line_stab = NULL;

	Linear_complex = NULL;

	//Line_to_neighbor = NULL;

	//ring_theory::longinteger_object go, stab_go;

	Stab = NULL;
	stab_gens = NULL;
	orbit = NULL;
	orbit_len = 0;

	pt0_idx_in_orbit = 0;

	A_on_neighbors = NULL;
	Control = NULL;
	Poset = NULL;
	Five_plus_one = NULL;

	Pts_for_partial_ovoid_test = NULL;

}

classify_five_plus_one::~classify_five_plus_one()
{
	if (Elt0) {
		FREE_int(Elt0);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Pts_for_partial_ovoid_test) {
		FREE_int(Pts_for_partial_ovoid_test);
	}
}


void classify_five_plus_one::init(
		cubic_surfaces_in_general::surface_with_action
			*Surf_A,
	poset_classification::poset_classification_control
		*Control,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_five_plus_one::init" << endl;
		}
	classify_five_plus_one::Surf_A = Surf_A;
	F = Surf_A->PA->F;
	q = F->q;
	Surf = Surf_A->Surf;



	A = Surf_A->A;
	A2 = Surf_A->A_wedge;

	if (A2->type_G != action_on_wedge_product_t) {
		cout << "classify_five_plus_one::init group must "
				"act in wedge action" << endl;
		exit(1);
	}

	AW = A2->G.AW;

	Elt0 = NEW_int(A->elt_size_in_int);
	Elt1 = NEW_int(A->elt_size_in_int);



	if (f_v) {
		cout << "classify_five_plus_one::init before "
				"SG_line_stab->generators_for_parabolic_subgroup" << endl;
	}


	SG_line_stab = NEW_OBJECT(groups::strong_generators);
	SG_line_stab->generators_for_parabolic_subgroup(
			A,
		A->G.matrix_grp, 2, verbose_level - 1);

	if (f_v) {
		cout << "classify_five_plus_one::init after "
				"SG_line_stab->generators_for_parabolic_subgroup" << endl;
	}



	Linear_complex = NEW_OBJECT(orthogonal_geometry::linear_complex);


	if (f_v) {
		cout << "classify_five_plus_one::init before "
				"Linear_complex->init" << endl;
	}
	Linear_complex->init(Surf, verbose_level - 1);
	if (f_v) {
		cout << "classify_five_plus_one::init after "
				"Linear_complex->init" << endl;
	}


	std::string label_of_set;

	label_of_set.assign("linear_complex");


	if (f_v) {
		cout << "classify_five_plus_one::init "
				"computing restricted action on neighbors" << endl;
	}

	A_on_neighbors = NEW_OBJECT(actions::action);
	A_on_neighbors = A2->Induced_action->create_induced_action_by_restriction(
		NULL,
		Linear_complex->nb_neighbors, Linear_complex->Neighbors, label_of_set,
		false /* f_induce_action */,
		0 /* verbose_level */);

	if (f_v) {
		cout << "classify_five_plus_one::init "
				"restricted action on neighbors "
				"has been computed" << endl;
	}


	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subset_lattice(A, A_on_neighbors,
			SG_line_stab,
			verbose_level);

	if (f_v) {
		cout << "classify_five_plus_one::init before "
				"Poset->add_testing_without_group" << endl;
	}
	Pts_for_partial_ovoid_test = NEW_int(5 * 6);
	Poset->add_testing_without_group(
			callback_partial_ovoid_test_early,
				this /* void *data */,
				verbose_level);

	Control->f_depth = true;
	Control->depth = 5;


	if (f_v) {
		cout << "classify_five_plus_one::init "
				"before Five_plus_one->init" << endl;
	}
	Five_plus_one = NEW_OBJECT(poset_classification::poset_classification);

	Five_plus_one->initialize_and_allocate_root_node(
			Control, Poset,
		5 /* sz */,
		verbose_level - 1);
	if (f_v) {
		cout << "classify_five_plus_one::init "
				"after Five_plus_one->init" << endl;
	}


	if (f_v) {
		cout << "classify_five_plus_one::init done" << endl;
	}
}



void classify_five_plus_one::classify_partial_ovoids(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth = 10000;
	int f_use_invariant_subset_if_available = true;
	int f_debug = false;
	orbiter_kernel_system::os_interface Os;
	int t0 = Os.os_ticks();


	if (f_v) {
		cout << "classify_five_plus_one::classify_partial_ovoids" << endl;
	}
	if (f_v) {
		cout << "classify_five_plus_one::classify_partial_ovoids "
				"nb_neighbors = " << Linear_complex->nb_neighbors << endl;
		cout << "Neighbors=";
		Lint_vec_print(
				cout,
				Linear_complex->Neighbors,
				Linear_complex->nb_neighbors);
		cout << endl;
	}
	if (f_v) {
		cout << "classify_five_plus_one::classify_partial_ovoids "
				"before Five_plus_one->main" << endl;
	}
	Five_plus_one->main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level - 2);
	if (f_v) {
		cout << "classify_five_plus_one::classify_partial_ovoids "
				"after Five_plus_one->main" << endl;
	}

#if 0
	if (q < 20) {
		{
			data_structures::spreadsheet *Sp;
			Five_plus_one->make_spreadsheet_of_orbit_reps(Sp, 5);
			string fname_csv;
			fname_csv = "fiveplusone_" + std::to_string(q) + ".csv";
			Sp->save(fname_csv, verbose_level);
			FREE_OBJECT(Sp);
		}
	}
#endif
	if (f_v) {
		cout << "classify_five_plus_one::classify_partial_ovoids done" << endl;
	}
}

int classify_five_plus_one::line_to_neighbor(
		long int line_rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;
	long int point_rk;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "classify_five_plus_one::line_to_neighbor" << endl;
	}

	point_rk = Surf->Klein->line_to_point_on_quadric(
			line_rk, 0 /* verbose_level*/);

	if (!Sorting.lint_vec_search(
			Linear_complex->Neighbors, Linear_complex->nb_neighbors, point_rk,
			idx, 0 /* verbose_level */)) {
		cout << "classify_five_plus_one::line_to_neighbor line " << line_rk
				<< " = point " << point_rk << " not found in Neighbors[]" << endl;
		exit(1);
	}
	return idx;
}


void classify_five_plus_one::partial_ovoid_test_early(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int i, j;
	int u[6];
	int v[6];
	int fxy;
	int f_OK;

	if (f_v) {
		cout << "classify_five_plus_one::partial_ovoid_test_early "
				"checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}

	if (len > 5) {
		cout << "classify_five_plus_one::partial_ovoid_test_early "
				"len > 5" << endl;
		exit(1);
	}
	for (i = 0; i < len; i++) {
		AW->unrank_point(u, Linear_complex->Neighbors[S[i]]);
		Surf->wedge_to_klein(u, Pts_for_partial_ovoid_test + i * 6);
	}

	if (len == 0) {
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
	}
	else {
		nb_good_candidates = 0;

		if (f_vv) {
			cout << "classify_five_plus_one::partial_ovoid_test_early "
					"before testing" << endl;
		}
		for (j = 0; j < nb_candidates; j++) {


			if (f_vv) {
				cout << "classify_five_plus_one::partial_ovoid_test_early "
						"testing " << j << " / "
						<< nb_candidates << endl;
			}

			AW->unrank_point(u, Linear_complex->Neighbors[candidates[j]]);
			Surf->wedge_to_klein(u, v);

			f_OK = true;
			for (i = 0; i < len; i++) {
				fxy = Surf->O->Quadratic_form->evaluate_bilinear_form(
						Pts_for_partial_ovoid_test + i * 6, v, 1);

				if (fxy == 0) {
					f_OK = false;
					break;
				}
			}
			if (f_OK) {
				good_candidates[nb_good_candidates++] =
						candidates[j];
			}
		} // next j
	} // else
}



void classify_five_plus_one::identify_five_plus_one(
	long int *five_lines,
	long int transversal_line,
	long int *five_lines_out_as_neighbors,
	int &orbit_index,
	int *transporter, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int W1[5];
	long int W2[5];
	long int N1[5];
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "classify_five_plus_one::identify_five_plus_one" << endl;
		cout << "classify_five_plus_one::identify_five_plus_one "
				"transversal_line=" << transversal_line << endl;
		cout << "classify_five_plus_one::identify_five_plus_one "
				"five_lines=";
		Lint_vec_print(cout, five_lines, 5);
		cout << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	Surf->line_to_wedge_vec(five_lines, W1, 5);
	if (f_v) {
		cout << "classify_five_plus_one::identify_five_plus_one W1=";
		Lint_vec_print(cout, W1, 5);
		cout << endl;
	}


	actions::action_global AG;


	AG.make_element_which_moves_a_line_in_PG3q(A,
		Surf->P->Solid,
		//Surf->Gr,
		transversal_line,
		Elt0,
		0 /* verbose_level */);
	if (f_v) {
		cout << "classify_five_plus_one::identify_five_plus_one "
				"element which moves transversal line:" << endl;
		A->Group_element->element_print(Elt0, cout);
	}


	A2->Group_element->map_a_set(
			W1,
			W2,
			5,
			Elt0, 0 /* verbose_level */);
	if (f_v) {
		cout << "classify_five_plus_one::identify_five_plus_one W2=";
		Lint_vec_print(cout, W2, 5);
		cout << endl;
	}

	Sorting.lint_vec_search_vec(
			Linear_complex->Neighbors, Linear_complex->nb_neighbors,
			W2, 5, N1);

	if (f_v) {
		cout << "classify_five_plus_one::identify_five_plus_one "
				"tracing the set N1=";
		Lint_vec_print(cout, N1, 5);
		cout << endl;
	}
	orbit_index = Five_plus_one->trace_set(
			N1, 5, 5,
			five_lines_out_as_neighbors,
			Elt1,
			0/*verbose_level - 2*/);
	if (f_v) {
		cout << "classify_five_plus_one::identify_five_plus_one "
				"orbit_index = " << orbit_index << endl;
	}
	if (f_v) {
		cout << "classify_five_plus_one::identify_five_plus_one "
				"element which moves neighbor set:" << endl;
		A->Group_element->element_print(Elt1, cout);
	}


	A->Group_element->element_mult(Elt0, Elt1, transporter, 0);
	if (f_v) {
		cout << "classify_five_plus_one::identify_five_plus_one "
				"element which moves five_plus_one:" << endl;
		A->Group_element->element_print(transporter, cout);
	}
	if (f_v) {
		cout << "classify_five_plus_one::identify_five_plus_one "
				"done" << endl;
	}
}

void classify_five_plus_one::report(
		std::ostream &ost,
		graphics::layered_graph_draw_options
			*draw_options,
		poset_classification::poset_classification_report_options
			*Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classify_five_plus_one::report" << endl;
	}


	if (f_v) {
		cout << "classify_five_plus_one::report reporting groups" << endl;
	}
	//ost << "\\section*{The groups}" << endl;
	ost << "\\section*{The semilinear group}" << endl;
	A->report(ost,
			A->f_has_sims,
			A->Sims,
			A->f_has_strong_generators,
			A->Strong_gens,
			draw_options, verbose_level);
	A->latex_all_points(ost);

	if (f_v) {
		cout << "classify_five_plus_one::report reporting orthogonal group" << endl;
	}
	ost << "\\section*{The orthogonal group}" << endl;
	A2->report(ost,
			A2->f_has_sims,
			A2->Sims,
			A2->f_has_strong_generators,
			A2->Strong_gens,
			draw_options, verbose_level);
	if (A2->degree < 100) {
		A2->latex_all_points(ost);
	}

	if (f_v) {
		cout << "classify_five_plus_one::report reporting line stabilizer" << endl;
	}
	ost << "\\section*{The group stabilizing the fixed line}" << endl;
	A_on_neighbors->report(ost,
			A_on_neighbors->f_has_sims,
			A_on_neighbors->Sims,
			A_on_neighbors->f_has_strong_generators,
			A_on_neighbors->Strong_gens,
			draw_options, verbose_level);
	A_on_neighbors->latex_all_points(ost);

	ost << "{\\small\\arraycolsep=2pt" << endl;
	SG_line_stab->print_generators_tex(ost);
	ost << "}" << endl;

	if (f_v) {
		cout << "classify_five_plus_one::report before Five_plus_one->report" << endl;
	}
	ost << "\\section*{The classification of five-plus-ones}" << endl;
	Five_plus_one->report2(ost, Opt, verbose_level);
	if (f_v) {
		cout << "classify_five_plus_one::report after Five_plus_one->report" << endl;
	}


	if (f_v) {
		cout << "classify_five_plus_one::report done" << endl;
	}
}



// #############################################################################



static void callback_partial_ovoid_test_early(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	classify_five_plus_one *Classify = (classify_five_plus_one *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "callback_partial_ovoid_test_early for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
	}
	Classify->partial_ovoid_test_early(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);
	if (f_v) {
		cout << "callback_partial_ovoid_test_early done" << endl;
	}
}




}}}}

