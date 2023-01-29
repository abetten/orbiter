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

	nb_neighbors = 0;
	Neighbors = NULL;
	Neighbor_to_line = NULL;
	Neighbor_to_klein = NULL;
	//Line_to_neighbor = NULL;

	//ring_theory::longinteger_object go, stab_go;

	Stab = NULL;
	stab_gens = NULL;
	orbit = NULL;
	orbit_len = 0;

	pt0_idx_in_orbit = 0;
	pt0_wedge = 0;
	pt0_line = 0;
	pt0_klein = 0;

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

	pt0_line = 0; // pt0 = the line spanned by 1000, 0100
		// (we call it point because it is a point on the Klein quadric)
	pt0_wedge = 0; // in wedge coordinates 100000
	pt0_klein = 0; // in klein coordinates 100000


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



	if (f_v) {
		cout << "classify_five_plus_one::init "
				"before compute_neighbors" << endl;
	}
	compute_neighbors(verbose_level);
	if (f_v) {
		cout << "classify_five_plus_one::init "
				"after compute_neighbors" << endl;
	}
	{
		data_structures::spreadsheet *Sp;
		if (f_v) {
			cout << "classify_five_plus_one::init "
					"before make_spreadsheet_of_neighbors" << endl;
		}
		make_spreadsheet_of_neighbors(Sp, 0 /* verbose_level */);
		if (f_v) {
			cout << "classify_five_plus_one::init "
					"after make_spreadsheet_of_neighbors" << endl;
		}
		FREE_OBJECT(Sp);
	}
	if (f_v) {
		cout << "classify_five_plus_one::init "
				"after compute_neighbors "
				"nb_neighbors = " << nb_neighbors << endl;
		cout << "Neighbors=";
		Lint_vec_print(cout, Neighbors, nb_neighbors);
		cout << endl;
	}





	if (f_v) {
		cout << "classify_five_plus_one::init "
				"computing restricted action on neighbors" << endl;
	}

	A_on_neighbors = NEW_OBJECT(actions::action);
	A_on_neighbors = A2->create_induced_action_by_restriction(
		NULL,
		nb_neighbors, Neighbors,
		FALSE /* f_induce_action */,
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

	Control->f_depth = TRUE;
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


	//Five_plus_one->init_check_func(callback_partial_ovoid_test,
	//	(void *)this /* candidate_check_data */);


	//Five_plus_one->f_print_function = TRUE;
	//Five_plus_one->print_function = callback_print_set;
	//Five_plus_one->print_function_data = (void *) this;


	if (f_v) {
		cout << "classify_five_plus_one::init done" << endl;
	}
}


void classify_five_plus_one::compute_neighbors(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, a, b, c;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "classify_five_plus_one::compute_neighbors" << endl;
	}

	nb_neighbors = (long int) (q + 1) * q * (q + 1);
	if (f_v) {
		cout << "classify_five_plus_one::compute_neighbors "
				"nb_neighbors = " << nb_neighbors << endl;
	}
	Neighbors = NEW_lint(nb_neighbors);
	Neighbor_to_line = NEW_lint(nb_neighbors);
	Neighbor_to_klein = NEW_lint(nb_neighbors);

	int sz;

	// At first, we get the neighbors
	// as points on the Klein quadric:
	// Later, we will change them to wedge ranks:

	if (f_v) {
		cout << "classify_five_plus_one::compute_neighbors "
				"before Surf->O->perp" << endl;
		}
	Surf->O->perp(0, Neighbors, sz, 0 /*verbose_level - 3*/);
	if (f_v) {
		cout << "classify_five_plus_one::compute_neighbors "
				"after Surf->O->perp" << endl;

		//cout << "Neighbors:" << endl;
		//lint_matrix_print(Neighbors, (sz + 9) / 10, 10);
	}

	if (sz != nb_neighbors) {
		cout << "classify_five_plus_one::compute_neighbors "
				"sz != nb_neighbors" << endl;
		cout << "sz = " << sz << endl;
		cout << "nb_neighbors = " << nb_neighbors << endl;
		exit(1);
	}
	if (f_v) {
		cout << "classify_five_plus_one::compute_neighbors "
				"nb_neighbors = " << nb_neighbors << endl;
	}

	if (f_v) {
		cout << "classify_five_plus_one::compute_neighbors "
				"allocating Line_to_neighbor, "
				"Surf->nb_lines_PG_3=" << Surf->nb_lines_PG_3 << endl;
	}

#if 0
	Line_to_neighbor = NEW_lint(Surf->nb_lines_PG_3);
	for (i = 0; i < Surf->nb_lines_PG_3; i++) {
		Line_to_neighbor[i] = -1;
	}
#endif


	// Convert Neighbors[] from points
	// on the Klein quadric to wedge points:
	if (f_v) {
		cout << "classify_five_plus_one::compute_neighbors "
				"before Surf->klein_to_wedge_vec" << endl;
	}
	Surf->klein_to_wedge_vec(Neighbors, Neighbors, nb_neighbors);

	// Sort the set Neighbors:
	Sorting.lint_vec_heapsort(Neighbors, nb_neighbors);




	// Establish the bijection between Neighbors and Lines in PG(3,q)
	// by going through the Klein correspondence.
	// It is important that this be done after we sort Neighbors.
	if (f_v) {
		cout << "classify_five_plus_one::compute_neighbors "
				"Establish the bijection between Neighbors and Lines in "
				"PG(3,q), nb_neighbors=" << nb_neighbors << endl;
	}
	int N100;
	int w[6];
	int v[6];

	N100 = nb_neighbors / 100 + 1;

	for (i = 0; i < nb_neighbors; i++) {
		if ((i % N100) == 0) {
			cout << "classify_five_plus_one::compute_neighbors i=" << i << " / "
					<< nb_neighbors << " at "
					<< (double)i * 100. / nb_neighbors << "%" << endl;
		}
		a = Neighbors[i];
		AW->unrank_point(w, a);
		Surf->wedge_to_klein(w, v);
		if (FALSE) {
			cout << i << " : ";
			Int_vec_print(cout, v, 6);
			cout << endl;
		}
		b = Surf->O->Hyperbolic_pair->rank_point(
				v, 1, 0 /* verbose_level*/);
		if (FALSE) {
			cout << " : " << b;
			cout << endl;
		}
		c = Surf->Klein->point_on_quadric_to_line(
				b, 0 /* verbose_level*/);
		if (FALSE) {
			cout << " : " << c << endl;
			cout << endl;
		}
		Neighbor_to_line[i] = c;
		//Line_to_neighbor[c] = i;
		}

	if (f_v) {
		cout << "classify_five_plus_one::compute_neighbors "
				"before int_vec_apply" << endl;
	}
	for (i = 0; i < nb_neighbors; i++) {
		Neighbor_to_klein[i] = Surf->Klein->line_to_point_on_quadric(
				Neighbor_to_line[i], 0 /* verbose_level*/);
	}
#if 0
	lint_vec_apply(Neighbor_to_line,
			Surf->Klein->Line_to_point_on_quadric,
			Neighbor_to_klein, nb_neighbors);
#endif


	if (f_v) {
		cout << "classify_five_plus_one::compute_neighbors done" << endl;
	}
}

void classify_five_plus_one::make_spreadsheet_of_neighbors(
	data_structures::spreadsheet *&Sp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char str[1000];
	string fname_csv;

	if (f_v) {
		cout << "classify_five_plus_one::make_spreadsheet_of_neighbors" << endl;
	}

	snprintf(str, sizeof(str), "neighbors_%d.csv", q);
	fname_csv.assign(str);


	Surf->make_spreadsheet_of_lines_in_three_kinds(Sp,
		Neighbors, Neighbor_to_line,
		Neighbor_to_klein, nb_neighbors, 0 /* verbose_level */);

	if (f_v) {
		cout << "before Sp->save " << fname_csv << endl;
	}
	Sp->save(fname_csv, verbose_level);
	if (f_v) {
		cout << "after Sp->save " << fname_csv << endl;
	}





	if (f_v) {
		cout << "classify_five_plus_one::make_spreadsheet_of_neighbors done" << endl;
	}
}

void classify_five_plus_one::classify_partial_ovoids(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int schreier_depth = 10000;
	int f_use_invariant_subset_if_available = TRUE;
	int f_debug = FALSE;
	orbiter_kernel_system::os_interface Os;
	int t0 = Os.os_ticks();


	if (f_v) {
		cout << "classify_five_plus_one::classify_partial_ovoids" << endl;
	}
	if (f_v) {
		cout << "classify_five_plus_one::classify_partial_ovoids "
				"nb_neighbors = " << nb_neighbors << endl;
		cout << "Neighbors=";
		Lint_vec_print(cout, Neighbors, nb_neighbors);
		cout << endl;
	}
	if (f_v) {
		cout << "classify_five_plus_one::classify_partial_ovoids "
				"classifying starter" << endl;
	}
	Five_plus_one->main(t0,
		schreier_depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level);
	if (f_v) {
		cout << "classify_five_plus_one::classify_partial_ovoids "
				"classifying starter done" << endl;
	}

#if 0
	if (q < 20) {
		{
			data_structures::spreadsheet *Sp;
			Five_plus_one->make_spreadsheet_of_orbit_reps(Sp, 5);
			char str[1000];
			string fname_csv;
			snprintf(str, sizeof(str), "fiveplusone_%d.csv", q);
			fname_csv.assign(str);
			Sp->save(fname_csv, verbose_level);
			FREE_OBJECT(Sp);
		}
	}
#endif
	if (f_v) {
		cout << "classify_five_plus_one::classify_partial_ovoids done" << endl;
	}
}

int classify_five_plus_one::line_to_neighbor(long int line_rk, int verbose_level)
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
			Neighbors, nb_neighbors, point_rk,
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
	int f_vv = FALSE; //(verbose_level >= 2);
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
		AW->unrank_point(u, Neighbors[S[i]]);
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

			AW->unrank_point(u, Neighbors[candidates[j]]);
			Surf->wedge_to_klein(u, v);

			f_OK = TRUE;
			for (i = 0; i < len; i++) {
				fxy = Surf->O->Quadratic_form->evaluate_bilinear_form(
						Pts_for_partial_ovoid_test + i * 6, v, 1);

				if (fxy == 0) {
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
}



void classify_five_plus_one::identify_five_plus_one(
	long int *five_lines,
	long int transversal_line,
	long int *five_lines_out_as_neighbors, int &orbit_index,
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


	A->make_element_which_moves_a_line_in_PG3q(
		Surf->Gr,
		transversal_line,
		Elt0,
		0 /* verbose_level */);
	if (f_v) {
		cout << "classify_five_plus_one::identify_five_plus_one "
				"element which moves transversal line:" << endl;
		A->element_print(Elt0, cout);
	}


	A2->map_a_set(
			W1,
			W2,
			5,
			Elt0, 0 /* verbose_level */);
	if (f_v) {
		cout << "classify_five_plus_one::identify_five_plus_one W2=";
		Lint_vec_print(cout, W2, 5);
		cout << endl;
	}

	Sorting.lint_vec_search_vec(Neighbors, nb_neighbors,
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
		A->element_print(Elt1, cout);
	}


	A->element_mult(Elt0, Elt1, transporter, 0);
	if (f_v) {
		cout << "classify_five_plus_one::identify_five_plus_one "
				"element which moves five_plus_one:" << endl;
		A->element_print(transporter, cout);
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

