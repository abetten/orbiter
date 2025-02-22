/*
 * classify_cubic_curves.cpp
 *
 *  Created on: Mar 7, 2019
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_geometry {


classify_cubic_curves::classify_cubic_curves()
{
	Record_birth();
	q = 0;
	F = NULL;
	A = NULL; // do not free

	CCA = NULL; // do not free
	CC = NULL; // do not free

	Arc_gen = NULL;

	nb_orbits_on_sets = 0;
	nb = 0;
	Idx = NULL;


	Flag_orbits = NULL;

	Po = NULL;

	nb_orbits_on_curves = 0;

	Curves = NULL;
}

classify_cubic_curves::~classify_cubic_curves()
{
	Record_death();
	if (Arc_gen) {
		FREE_OBJECT(Arc_gen);
	}
	if (Idx) {
		FREE_int(Idx);
	}
 	if (Flag_orbits) {
		FREE_OBJECT(Flag_orbits);
	}
 	if (Po) {
 		FREE_int(Po);
 	}
	if (Curves) {
		FREE_OBJECT(Curves);
	}
}

void classify_cubic_curves::init(
		projective_geometry::projective_space_with_action *PA,
		cubic_curve_with_action *CCA,
		arc_generator_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_cubic_curves::init" << endl;
	}
	classify_cubic_curves::CCA = CCA;
	F = CCA->F;
	q = F->q;
	A = CCA->A;
	CC = CCA->CC;

	Arc_gen = NEW_OBJECT(arc_generator);


	if (f_v) {
		cout << "classify_cubic_curves::init before Arc_gen->init" << endl;
	}



	// ToDo

	Arc_gen->init(
			Descr,
			PA,
			A->Strong_gens,
			verbose_level);

#if 0
	Arc_gen->init(GTA,
			F,
			A, A->Strong_gens,
			9 /* starter_size */,
			false /* f_conic_test */,
			Control,
			verbose_level);
#endif


	if (f_v) {
		cout << "classify_cubic_curves::init after Arc_gen->init" << endl;
	}


	if (f_v) {
		cout << "classify_cubic_curves::init done" << endl;
	}
}

void classify_cubic_curves::compute_starter(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classify_cubic_curves::compute_starter" << endl;
	}
	Arc_gen->compute_starter(verbose_level);
	if (f_v) {
		cout << "classify_cubic_curves::compute_starter done" << endl;
	}
}


void classify_cubic_curves::test_orbits(
		int verbose_level)
{
	//verbose_level += 2;
	int f_v = (verbose_level >= 1);
	int f_vv = false; // (verbose_level >= 2);
	int i, r;
	long int S[9];
	//long int *Pts_on_curve;
	//long int *singular_Pts;
	int *type;
	//int nb_pts_on_curve; //, nb_singular_pts;

	if (f_v) {
		cout << "classify_cubic_curves::test_orbits" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}
	nb_orbits_on_sets = Arc_gen->gen->nb_orbits_at_level(9);

	//Pts_on_curve = NEW_lint(CC->P->N_points);
	//singular_Pts = NEW_lint(CC->P->N_points);
	type = NEW_int(CC->P->Subspaces->N_lines);

	if (f_v) {
		cout << "classify_cubic_curves::test_orbits testing "
				<< nb_orbits_on_sets << " orbits of 9-sets of points:" << endl;
	}
	nb = 0;
	Idx = NEW_int(nb_orbits_on_sets);
	for (i = 0; i < nb_orbits_on_sets; i++) {
		if (f_vv || ((i % 1000) == 0)) {
			cout << "classify_cubic_curves::test_orbits orbit "
				<< i << " / " << nb_orbits_on_sets << ":" << endl;
		}
		Arc_gen->gen->get_set_by_level(9, i, S);
		if (f_vv) {
			cout << "set: ";
			Lint_vec_print(cout, S, 5);
			cout << endl;
		}




#if 1
		if (f_vv) {
			CC->P->Reporting->print_set(S, 9);
		}
#endif

		r = CC->compute_system_in_RREF(9, S, 0 /*verbose_level*/);
		if (f_vv) {
			cout << "classify_cubic_curves::test_orbits orbit "
					<< i << " / " << nb_orbits_on_sets
					<< " has rank = " << r << endl;
		}
		if (r == 9) {

			// second test:
			// the curve should not contain lines:
			int eqn[10];
			int idx;

			CC->P->Plane->determine_cubic_in_plane(
					CC->Poly,
					9 /* nb_pts */, S /* int *Pts */, eqn,
					verbose_level - 5);

			{
				long int *Pts;
				int nb_pts;

				{
					vector<long int> Pts_on_curve;

					CC->Poly->enumerate_points(eqn,
							Pts_on_curve,
							0 /*verbose_level - 4*/);

					int h;

					nb_pts = Pts_on_curve.size();
					Pts = NEW_lint(nb_pts);
					for (h = 0; h < nb_pts; h++) {
						Pts[h] = Pts_on_curve[h];
					}
				}
				CC->P->Subspaces->line_intersection_type(
						Pts, nb_pts /* set_size */,
						type, 0 /*verbose_level*/);

				FREE_lint(Pts);
			}

			other::data_structures::tally Cl;

			Cl.init(type, CC->P->Subspaces->N_lines, false, 0);
			idx = Cl.determine_class_by_value(q + 1);

			if (idx == -1) {

#if 0
				// third test: the curve should have no singular points:

				CC->compute_singular_points(
						eqn,
						Pts_on_curve, nb_pts_on_curve,
						singular_Pts, nb_singular_pts,
						0 /*verbose_level*/);

				if (nb_singular_pts == 0) {
					Idx[nb++] = i;
				}
#else
				Idx[nb++] = i;
#endif
			}
		}
	} // next i

	if (f_v) {
		cout << "classify_cubic_curves::test_orbits we found "
				<< nb << " / " << nb_orbits_on_sets
				<< " orbits where the rank is 9" << endl;
		cout << "Idx=";
		Int_vec_print(cout, Idx, nb);
		cout << endl;
	}

	//FREE_lint(Pts_on_curve);
	//FREE_lint(singular_Pts);
	FREE_int(type);

	if (f_v) {
		cout << "classify_cubic_curves::test_orbits done" << endl;
	}
}


void classify_cubic_curves::downstep(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; // (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int f, i, nb_flag_orbits;

	if (f_v) {
		cout << "classify_cubic_curves::downstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	if (f_v) {
		cout << "classify_cubic_curves::downstep "
				"before test_orbits" << endl;
	}
	test_orbits(verbose_level - 1);
	if (f_v) {
		cout << "classify_cubic_curves::downstep "
				"after test_orbits" << endl;
		cout << "Idx=";
		Int_vec_print(cout, Idx, nb);
		cout << endl;
	}



	Flag_orbits = NEW_OBJECT(invariant_relations::flag_orbits);
	Flag_orbits->init(A, A,
		nb_orbits_on_sets /* nb_primary_orbits_lower */,
		9 + 10 /* pt_representation_sz */,
		nb,
		1 /* upper_bound_for_number_of_traces */, // ToDo
		NULL /* void (*func_to_free_received_trace)(void *trace_result, void *data, int verbose_level) */,
		NULL /* void (*func_latex_report_trace)(std::ostream &ost, void *trace_result, void *data, int verbose_level)*/,
		NULL /* void *free_received_trace_data */,
		verbose_level);

	if (f_v) {
		cout << "classify_cubic_curves::downstep "
				"initializing flag orbits" << endl;
	}

	nb_flag_orbits = 0;
	for (f = 0; f < nb; f++) {

		i = Idx[f];
		if (f_v) {
			if ((f % 1000) == 0) {
				cout << "classify_cubic_curves::downstep "
						"orbit " << f << " / " << nb
						<< " with rank = 9 is orbit "
						<< i << " / " << nb_orbits_on_sets << endl;
			}
		}

		data_structures_groups::set_and_stabilizer *R;
		algebra::ring_theory::longinteger_object ol;
		algebra::ring_theory::longinteger_object go;
		long int dataset[19];

		R = Arc_gen->gen->get_set_and_stabilizer(
				9 /* level */,
				i /* orbit_at_level */,
				0 /* verbose_level */);

		Arc_gen->gen->orbit_length(
				i /* node */, 9 /* level */, ol);

		R->Strong_gens->group_order(go);

		Lint_vec_copy(R->data, dataset, 9);

		int eqn[10];
		if (f_vv) {
			cout << "9 points = ";
			Lint_vec_print(cout, dataset, 9);
			cout << endl;
		}

		if (f_vv) {
			cout << "classify_cubic_curves::downstep before "
					"determine_cubic_in_plane" << endl;
		}

		CC->P->Plane->determine_cubic_in_plane(
				CC->Poly,
				9 /* nb_pts */, dataset /* int *Pts */, eqn,
				0 /*verbose_level - 5*/);
		//c = Surf_A->create_double_six_from_five_lines_with_a_common_transversal(
		//		dataset + 5, pt0_line, double_six,
		//		0 /*verbose_level*/);

		if (f_vv) {
			cout << "The starter configuration is good, "
					"a cubic has been computed:" << endl;
			Int_vec_print(cout, eqn, 10);
		}

		Int_vec_copy_to_lint(eqn, dataset + 9, 10);


		Flag_orbits->Flag_orbit_node[nb_flag_orbits].init(
			Flag_orbits,
			nb_flag_orbits /* flag_orbit_index */,
			i /* downstep_primary_orbit */,
			0 /* downstep_secondary_orbit */,
			ol.as_int() /* downstep_orbit_len */,
			false /* f_long_orbit */,
			dataset /* int *pt_representation */,
			R->Strong_gens,
			0 /*verbose_level - 2*/);
		R->Strong_gens = NULL;

		if (f_vv) {
			cout << "orbit " << f << " / " << nb
				<< " with rank = 9 is orbit " << i
				<< " / " << nb_orbits_on_sets << ", stab order "
				<< go << endl;
		}
		nb_flag_orbits++;

		FREE_OBJECT(R);
	} // next f

	Flag_orbits->nb_flag_orbits = nb_flag_orbits;


	Po = NEW_int(nb_flag_orbits);
	for (f = 0; f < nb_flag_orbits; f++) {
		Po[f] = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
	}
	if (f_v) {
		cout << "classify_cubic_curves::downstep we found "
			<< nb_flag_orbits << " flag orbits out of "
			<< nb_orbits_on_sets << " orbits" << endl;
	}
	if (f_v) {
		cout << "classify_cubic_curves::downstep "
				"initializing flag orbits done" << endl;
	}
}


void classify_cubic_curves::upstep(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, r;
	int f, po, so;
	int *f_processed;
	int nb_processed;
	int *Elt;
	int idx_set[9];
	long int set[9];
	long int canonical_set[9];
	long int *Pts;
	int *type;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "classify_cubic_curves::upstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	Elt = NEW_int(A->elt_size_in_int);
	Pts = NEW_lint(CCA->CC->P->Subspaces->N_points);
	type = NEW_int(CCA->CC->P->Subspaces->N_lines);

	f_processed = NEW_int(Flag_orbits->nb_flag_orbits);
	Int_vec_zero(f_processed, Flag_orbits->nb_flag_orbits);
	nb_processed = 0;

	Curves = NEW_OBJECT(invariant_relations::classification_step);

	algebra::ring_theory::longinteger_object go;
	A->group_order(go);

	Curves->init(A, A,
			Flag_orbits->nb_flag_orbits, 19, go,
			verbose_level);


	if (f_v) {
		cout << "flag orbit : downstep_primary_orbit" << endl;
		if (Flag_orbits->nb_flag_orbits < 50) {
			cout << "f : po" << endl;
			for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {
				po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
				cout << f << " : " << po << endl;
			}
		}
		else {
			cout << "classify_cubic_curves::upstep "
					"too many flag orbits to print" << endl;
		}
	}
	for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {

		double progress;
		long int dataset[19];

		if (f_processed[f]) {
			continue;
		}

		progress = ((double)nb_processed * 100. ) /
				(double) Flag_orbits->nb_flag_orbits;

		if (f_v) {
			cout << "Defining new orbit "
				<< Flag_orbits->nb_primary_orbits_upper
				<< " from flag orbit " << f << " / "
				<< Flag_orbits->nb_flag_orbits
				<< " progress=" << progress << "%" << endl;
		}
		Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit
			= Flag_orbits->nb_primary_orbits_upper;


		if (Flag_orbits->pt_representation_sz != 19) {
			cout << "Flag_orbits->pt_representation_sz != 19" << endl;
			exit(1);
		}
		po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
		so = Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
		if (f_v) {
			cout << "po=" << po << " so=" << so << endl;
		}
		Lint_vec_copy(Flag_orbits->Pt + f * 19, dataset, 19);




		data_structures_groups::vector_ge *coset_reps;
		int nb_coset_reps;


		groups::strong_generators *S;
		algebra::ring_theory::longinteger_object go;
		int eqn[10];

		Lint_vec_copy_to_int(dataset + 9, eqn, 10);

		if (f_v) {
			cout << "equation:";
			Int_vec_print(cout, eqn, 10);
			cout << endl;
		}
		S = Flag_orbits->Flag_orbit_node[f].gens->create_copy(verbose_level - 2);
		S->group_order(go);
		if (f_v) {
			cout << "po=" << po << " so=" << so
					<< " go=" << go << endl;
		}

		nb_coset_reps = 0;

		int nb_pts;
		int N;
		int orbit_index;
		int f2;
		int h;

		{
			vector<long int> Points;
			CCA->CC->Poly->enumerate_points(eqn, Points,
					0 /*verbose_level - 4*/);

			nb_pts = Points.size();

			for (h = 0; h < nb_pts; h++) {
				Pts[h] = Points[h];
			}
		}
		if (f_v) {
			cout << "po=" << po << " so=" << so
					<< " we found a curve with " << nb_pts
					<< " points" << endl;
		}

		N = Combi.int_n_choose_k(nb_pts, 9);

		coset_reps = NEW_OBJECT(data_structures_groups::vector_ge);
		coset_reps->init(CCA->A, verbose_level - 2);
		coset_reps->allocate(N, verbose_level - 2);


		for (i = 0; i < N; i++) {
			if (false) {
				cout << "po=" << po << " so=" << so
						<< " i=" << i << " / " << N << endl;
			}

			Combi.unrank_k_subset(i, idx_set, nb_pts, 9);
			for (j = 0; j < 9; j++) {
				set[j] = Pts[idx_set[j]];
			}

			r = CC->compute_system_in_RREF(9, set, 0 /*verbose_level*/);

			if (r < 9) {
				continue;
			}

			CCA->CC->P->Subspaces->line_intersection_type(
				set, 9 /* set_size */, type, 0 /*verbose_level*/);
			// type[N_lines]

			for (j = 0; j < CCA->CC->P->Subspaces->N_lines; j++) {
				if (type[j] > 3) {
					break;
				}
			}
			if (j < CCA->CC->P->Subspaces->N_lines) {
				continue;
			}


			orbit_index = Arc_gen->gen->trace_set(
					set, 9, 9,
					canonical_set,
					Elt,
					0 /*verbose_level - 2*/);

			if (!Sorting.int_vec_search(Po, Flag_orbits->nb_flag_orbits,
					orbit_index, f2)) {
				cout << "cannot find orbit " << orbit_index
						<< " in Po" << endl;
				cout << "Po=";
				Int_vec_print(cout, Po, Flag_orbits->nb_flag_orbits);
				cout << endl;
				exit(1);
			}

			if (Flag_orbits->Flag_orbit_node[f2].downstep_primary_orbit
					!= orbit_index) {
				cout << "Flag_orbits->Flag_orbit_node[f2].downstep_"
						"primary_orbit != orbit_index" << endl;
				exit(1);
			}






			if (f2 == f) {
				if (f_v) {
					cout << "We found an automorphism of "
							"the curve with " << nb_pts << " points:" << endl;
					A->Group_element->element_print_quick(Elt, cout);
					cout << endl;
				}
				A->Group_element->element_move(Elt, coset_reps->ith(nb_coset_reps), 0);
				nb_coset_reps++;
				//S->add_single_generator(Elt3,
				//2 /* group_index */, verbose_level - 2);
			}
			else {
				if (false) {
					cout << "We are identifying flag orbit "
							<< f2 << " with flag orbit " << f << endl;
				}
				if (!f_processed[f2]) {
					Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit
						= Flag_orbits->nb_primary_orbits_upper;
					Flag_orbits->Flag_orbit_node[f2].f_fusion_node
						= true;
					Flag_orbits->Flag_orbit_node[f2].fusion_with
						= f;
					Flag_orbits->Flag_orbit_node[f2].fusion_elt
						= NEW_int(A->elt_size_in_int);
					A->Group_element->element_invert(Elt,
							Flag_orbits->Flag_orbit_node[f2].fusion_elt,
							0);
					f_processed[f2] = true;
					nb_processed++;
				}
				else {
					if (false) {
						cout << "Flag orbit " << f2 << " has already been "
								"identified with flag orbit " << f << endl;
					}
					if (Flag_orbits->Flag_orbit_node[f2].fusion_with != f) {
						cout << "Flag_orbits->Flag_orbit_node[f2]."
								"fusion_with != f" << endl;
						exit(1);
					}
				}
			}

		}

		coset_reps->reallocate(nb_coset_reps, verbose_level - 2);

		groups::strong_generators *Aut_gens;

		{
			algebra::ring_theory::longinteger_object ago;

			if (f_v) {
				cout << "classify_cubic_curves::upstep "
						"Extending the group by a factor of "
						<< nb_coset_reps << endl;
			}
			Aut_gens = NEW_OBJECT(groups::strong_generators);
			Aut_gens->init_group_extension(S,
					coset_reps, nb_coset_reps,
					verbose_level - 2);
			if (f_v) {
				cout << "classify_cubic_curves::upstep "
						"Aut_gens tl = ";
				Int_vec_print(cout,
						Aut_gens->tl, Aut_gens->A->base_len());
				cout << endl;
			}

			Aut_gens->group_order(ago);


			if (f_v) {
				cout << "the double six has a stabilizer of order "
						<< ago << endl;
				cout << "The new stabilizer is:" << endl;
				Aut_gens->print_generators_tex(cout);
			}
		}



		Curves->Orbit[Flag_orbits->nb_primary_orbits_upper].init(
				Curves,
			Flag_orbits->nb_primary_orbits_upper,
			f,
			Aut_gens, dataset, NULL /* extra_data */, verbose_level);

		FREE_OBJECT(coset_reps);
		FREE_OBJECT(S);

		f_processed[f] = true;
		nb_processed++;
		Flag_orbits->nb_primary_orbits_upper++;
	} // next f


	if (nb_processed != Flag_orbits->nb_flag_orbits) {
		cout << "nb_processed != Flag_orbits->nb_flag_orbits" << endl;
		cout << "nb_processed = " << nb_processed << endl;
		cout << "Flag_orbits->nb_flag_orbits = "
				<< Flag_orbits->nb_flag_orbits << endl;
		exit(1);
	}

	Curves->nb_orbits = Flag_orbits->nb_primary_orbits_upper;

	if (f_v) {
		cout << "We found " << Flag_orbits->nb_primary_orbits_upper
				<< " orbits of curves" << endl;
	}

	FREE_int(Elt);
	FREE_int(f_processed);
	FREE_lint(Pts);
	FREE_int(type);


	if (f_v) {
		cout << "classify_cubic_curves::upstep done" << endl;
	}
}


void classify_cubic_curves::do_classify(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classify_cubic_curves::do_classify" << endl;
	}

	if (f_v) {
		cout << "classify_cubic_curves::do_classify "
				"before downstep" << endl;
	}
	downstep(verbose_level);
	if (f_v) {
		cout << "classify_cubic_curves::do_classify "
				"after downstep" << endl;
		cout << "we found " << Flag_orbits->nb_flag_orbits
				<< " flag orbits out of "
				<< Arc_gen->gen->nb_orbits_at_level(9)
				<< " orbits" << endl;
	}

	if (f_v) {
		cout << "classify_cubic_curves::do_classify "
				"before upstep" << endl;
	}
	upstep(verbose_level);
	if (f_v) {
		cout << "classify_cubic_curves::do_classify "
				"after upstep" << endl;
		cout << "we found " << Curves->nb_orbits
				<< " cubic curves out of "
				<< Flag_orbits->nb_flag_orbits
				<< " flag orbits" << endl;
	}

	if (f_v) {
		cout << "classify_cubic_curves::do_classify done" << endl;
	}
}


int classify_cubic_curves::recognize(
		int *eqn_in,
		int *Elt, int &iso_type, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, r;
	int idx_set[9];
	long int set[9];
	long int canonical_set[9];
	int *Elt1;
	long int *Pts_on_curve;
	long int *singular_Pts;
	int *type;
	int ret;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "classify_cubic_curves::recognize" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	Elt1 = NEW_int(A->elt_size_in_int);
	Pts_on_curve = NEW_lint(CCA->CC->P->Subspaces->N_points);
	singular_Pts = NEW_lint(CCA->CC->P->Subspaces->N_points);
	type = NEW_int(CCA->CC->P->Subspaces->N_lines);


	int nb_pts_on_curve; //, nb_singular_pts;
	int N;
	int orbit_index;
	int f2;
	int h;


	{
		vector<long int> Points;

		CCA->CC->Poly->enumerate_points(eqn_in, Points,
				verbose_level - 4);


		nb_pts_on_curve = Points.size();

		for (h = 0; h < nb_pts_on_curve; h++) {
			Pts_on_curve[h] = Points[h];
		}
	}

	if (f_v) {
		cout << "classify_cubic_curves::recognize"
				<< " we found a curve with " << nb_pts_on_curve
				<< " points" << endl;
	}
	CCA->CC->P->Subspaces->line_intersection_type(
			Pts_on_curve, nb_pts_on_curve /* set_size */, type, 0 /*verbose_level*/);
	// type[N_lines]

	ret = true;
	for (j = 0; j < CCA->CC->P->Subspaces->N_lines; j++) {
		if (type[j] > 3) {
			ret = false;
			break;
		}
	}

#if 0
	if (ret) {
		CCA->CC->compute_singular_points(
				eqn_in,
				Pts_on_curve, nb_pts_on_curve,
				singular_Pts, nb_singular_pts,
				0 /*verbose_level*/);

		if (nb_singular_pts) {
			ret = false;
		}
	}
#endif

	if (ret) {
		N = Combi.int_n_choose_k(nb_pts_on_curve, 9);


		for (i = 0; i < N; i++) {
			if (f_v) {
				cout << "classify_cubic_curves::recognize"
						<< " i=" << i << " / " << N << endl;
				}

			Combi.unrank_k_subset(i, idx_set, nb_pts_on_curve, 9);
			for (j = 0; j < 9; j++) {
				set[j] = Pts_on_curve[idx_set[j]];
			}

			r = CC->compute_system_in_RREF(9,
					set, 0 /*verbose_level*/);

			if (r < 9) {
				continue;
			}

			CCA->CC->P->Subspaces->line_intersection_type(
				set, 9 /* set_size */, type, 0 /*verbose_level*/);
			// type[N_lines]

			for (j = 0; j < CCA->CC->P->Subspaces->N_lines; j++) {
				if (type[j] > 3) {
					break;
				}
			}
			if (j < CCA->CC->P->Subspaces->N_lines) {
				continue;
			}


			if (f_v) {
				cout << "classify_cubic_curves::recognize"
						<< " i=" << i << " / " << N
						<< " before trace_set" << endl;
				}


			orbit_index = Arc_gen->gen->trace_set(
					set, 9, 9,
					canonical_set,
					Elt,
					verbose_level - 1);


			if (f_v) {
				cout << "classify_cubic_curves::recognize"
						<< " i=" << i << " / " << N
						<< " after trace_set, "
						"orbit_index=" << orbit_index << endl;
				}

			if (!Sorting.int_vec_search(Po, Flag_orbits->nb_flag_orbits,
					orbit_index, f2)) {

				continue;
#if 0
				cout << "classify_cubic_curves::recognize "
						"cannot find orbit " << orbit_index
						<< " in Po" << endl;
				cout << "Po=";
				int_vec_print(cout, Po, Flag_orbits->nb_flag_orbits);
				cout << endl;
				exit(1);
#endif
			}

			if (f_v) {
				cout << "classify_cubic_curves::recognize"
						<< " i=" << i << " / " << N
						<< " after trace_set, "
						"f2=" << f2 << endl;
				}

			iso_type = Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit;

			if (f_v) {
				cout << "classify_cubic_curves::recognize"
						<< " i=" << i << " / " << N
						<< " after trace_set, "
						"iso_type=" << iso_type << endl;
			}

			if (Flag_orbits->Flag_orbit_node[f2].f_fusion_node) {
				A->Group_element->element_mult(Elt,
								Flag_orbits->Flag_orbit_node[f2].fusion_elt,
								Elt1,
								0);
				A->Group_element->element_move(Elt1,
								Elt,
								0);
			}
			break;

		}
		if (i == N) {
			cout << "classify_cubic_curves::recognize "
					"could not identify the curve" << endl;
			ret = false;
		}
		else {
			ret = true;
		}
	}


	FREE_int(Elt1);
	FREE_lint(Pts_on_curve);
	FREE_lint(singular_Pts);
	FREE_int(type);

	if (f_v) {
		cout << "classify_cubic_curves::recognize done" << endl;
	}
	return ret;
}

void classify_cubic_curves::family1_recognize(
		int *Iso_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	int eqn[10];
	int e, iso_type;

	if (f_v) {
		cout << "classify_cubic_curves::family1_recognize" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);

	for (e = 0; e < F->q; e++) {

#if 1
		Int_vec_zero(eqn, 10);
		// 0 = x0x1(x0 + x1) + ex2^3
		// 0 = x0^2x1 + x0x1^2 + ex2^3
		// 0 = X^2Y + XY^2 + eZ^3

		eqn[3] = 1;
		eqn[5] = 1;
		eqn[2] = e;
//0 & X^3 & ( 3, 0, 0 )
//1 & Y^3 & ( 0, 3, 0 )
//2 & Z^3 & ( 0, 0, 3 )
//3 & X^2Y & ( 2, 1, 0 )
//4 & X^2Z & ( 2, 0, 1 )
//5 & XY^2 & ( 1, 2, 0 )
//6 & Y^2Z & ( 0, 2, 1 )
//7 & XZ^2 & ( 1, 0, 2 )
//8 & YZ^2 & ( 0, 1, 2 )
//9 & XYZ & ( 1, 1, 1 )
#endif
		if (recognize(eqn,
				Elt, iso_type, verbose_level)) {
			Iso_type[e] = iso_type;
		}
		else {
			Iso_type[e] = -1;
		}

	}

	FREE_int(Elt);
}

void classify_cubic_curves::family2_recognize(
		int *Iso_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	int eqn[10];
	int e, iso_type;

	if (f_v) {
		cout << "classify_cubic_curves::family2_recognize" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);

	for (e = 0; e < F->q; e++) {

#if 1
		Int_vec_zero(eqn, 10);
		// 0 = x0x1(x0 + x1 + x2) + ex2^3
		// 0 = x0^2x1 + x0x1^2 + x1x2x3 + ex2^3
		// 0 = X^2Y + XY^2 + XYZ + eZ^3

		eqn[3] = 1;
		eqn[5] = 1;
		eqn[2] = e;
		eqn[9] = 1;
//0 & X^3 & ( 3, 0, 0 )
//1 & Y^3 & ( 0, 3, 0 )
//2 & Z^3 & ( 0, 0, 3 )
//3 & X^2Y & ( 2, 1, 0 )
//4 & X^2Z & ( 2, 0, 1 )
//5 & XY^2 & ( 1, 2, 0 )
//6 & Y^2Z & ( 0, 2, 1 )
//7 & XZ^2 & ( 1, 0, 2 )
//8 & YZ^2 & ( 0, 1, 2 )
//9 & XYZ & ( 1, 1, 1 )
#endif
		if (recognize(eqn,
				Elt, iso_type, verbose_level)) {
			Iso_type[e] = iso_type;
		}
		else {
			Iso_type[e] = -1;
		}

	}

	FREE_int(Elt);
}

void classify_cubic_curves::family3_recognize(
		int *Iso_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	int eqn[10];
	int e, iso_type;
	int three, six;
	int three_e, six_e_plus_one;

	if (f_v) {
		cout << "classify_cubic_curves::family3_recognize" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);

	for (e = 0; e < F->q; e++) {

#if 1
		Int_vec_zero(eqn, 10);
		// 0 = x0x1x2 + e(x0 + x1 + x2)
		// 0 = e(x0^3 + x1^3 + x2^3)
		// + 3e(x0^2x1 + x0^2x2 + x1^2x0 + x1^2x2 + x2^2x0 + x2^2x1)
		// + (6e+1)x0x1x2
		three = F->Z_embedding(3);
		six = F->Z_embedding(6);
		three_e = F->mult(three, e);
		six_e_plus_one = F->add(F->mult(six, e), 1);
		eqn[0] = e;
		eqn[1] = e;
		eqn[2] = e;
		eqn[3] = three_e;
		eqn[4] = three_e;
		eqn[5] = three_e;
		eqn[6] = three_e;
		eqn[7] = three_e;
		eqn[8] = three_e;
		eqn[9] = six_e_plus_one;
//0 & X^3 & ( 3, 0, 0 )
//1 & Y^3 & ( 0, 3, 0 )
//2 & Z^3 & ( 0, 0, 3 )
//3 & X^2Y & ( 2, 1, 0 )
//4 & X^2Z & ( 2, 0, 1 )
//5 & XY^2 & ( 1, 2, 0 )
//6 & Y^2Z & ( 0, 2, 1 )
//7 & XZ^2 & ( 1, 0, 2 )
//8 & YZ^2 & ( 0, 1, 2 )
//9 & XYZ & ( 1, 1, 1 )
#endif
		if (recognize(eqn,
				Elt, iso_type, verbose_level)) {
			Iso_type[e] = iso_type;
		}
		else {
			Iso_type[e] = -1;
		}

	}

	FREE_int(Elt);
}

void classify_cubic_curves::familyE_recognize(
		int *Iso_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	int eqn[10];
	int d, iso_type;

	if (f_v) {
		cout << "classify_cubic_curves::familyE_recognize" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);

	for (d = 0; d < F->q; d++) {

#if 1
		Int_vec_zero(eqn, 10);
		// 0 = x2^2x1 + x0^3 - dx1^3
		// 0 = Z^2Y + X^3 - dY^3

		eqn[0] = 1;
		eqn[1] = F->negate(d);
		eqn[8] = 1;
//0 & X^3 & ( 3, 0, 0 )
//1 & Y^3 & ( 0, 3, 0 )
//2 & Z^3 & ( 0, 0, 3 )
//3 & X^2Y & ( 2, 1, 0 )
//4 & X^2Z & ( 2, 0, 1 )
//5 & XY^2 & ( 1, 2, 0 )
//6 & Y^2Z & ( 0, 2, 1 )
//7 & XZ^2 & ( 1, 0, 2 )
//8 & YZ^2 & ( 0, 1, 2 )
//9 & XYZ & ( 1, 1, 1 )
#endif
		if (recognize(eqn,
				Elt, iso_type, verbose_level)) {
			Iso_type[d] = iso_type;
		}
		else {
			Iso_type[d] = -1;
		}

	}

	FREE_int(Elt);
}

void classify_cubic_curves::familyH_recognize(
		int *Iso_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	int eqn[10];
	int e, iso_type;

	if (f_v) {
		cout << "classify_cubic_curves::familyH_recognize" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);

	for (e = 0; e < F->q; e++) {

#if 1
		Int_vec_zero(eqn, 10);
		// 0 = x2^2x1 + x0^3 + ex0x1^2
		// 0 = Z^2Y + X^3 + eXY^2

		eqn[0] = 1;
		eqn[5] = e;
		eqn[8] = 1;
//0 & X^3 & ( 3, 0, 0 )
//1 & Y^3 & ( 0, 3, 0 )
//2 & Z^3 & ( 0, 0, 3 )
//3 & X^2Y & ( 2, 1, 0 )
//4 & X^2Z & ( 2, 0, 1 )
//5 & XY^2 & ( 1, 2, 0 )
//6 & Y^2Z & ( 0, 2, 1 )
//7 & XZ^2 & ( 1, 0, 2 )
//8 & YZ^2 & ( 0, 1, 2 )
//9 & XYZ & ( 1, 1, 1 )
#endif
		if (recognize(eqn,
				Elt, iso_type, verbose_level)) {
			Iso_type[e] = iso_type;
		}
		else {
			Iso_type[e] = -1;
		}

	}

	FREE_int(Elt);
}


void classify_cubic_curves::familyG_recognize(
		int *Iso_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	int eqn[10];
	int c, d, iso_type;

	if (f_v) {
		cout << "classify_cubic_curves::familyG_recognize" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);

	for (c = 0; c < F->q; c++) {
		for (d = 0; d < F->q; d++) {

#if 1
			Int_vec_zero(eqn, 10);
			// 0 = x2^2x1 + x0^3 + cx0x1^2 + dx1^3
			// 0 = Z^2Y + X^3 + cXY^2 + dY^3

			eqn[0] = 1;
			eqn[2] = d;
			eqn[5] = c;
			eqn[8] = 1;
//0 & X^3 & ( 3, 0, 0 )
//1 & Y^3 & ( 0, 3, 0 )
//2 & Z^3 & ( 0, 0, 3 )
//3 & X^2Y & ( 2, 1, 0 )
//4 & X^2Z & ( 2, 0, 1 )
//5 & XY^2 & ( 1, 2, 0 )
//6 & Y^2Z & ( 0, 2, 1 )
//7 & XZ^2 & ( 1, 0, 2 )
//8 & YZ^2 & ( 0, 1, 2 )
//9 & XYZ & ( 1, 1, 1 )
#endif
			if (recognize(eqn,
					Elt, iso_type, verbose_level)) {
				Iso_type[c * q + d] = iso_type;
			}
			else {
				Iso_type[c * q + d] = -1;
			}
		}

	}

	FREE_int(Elt);
}


void classify_cubic_curves::report(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_with_stabilizers = true;


	if (f_v) {
		cout << "classify_cubic_curves::report writing cheat sheet "
				"on cubic curves" << endl;
	}
	long int *Pts_on_curve;
	long int *inflexion_Pts;
	long int *singular_Pts;
	int *type;

	Pts_on_curve = NEW_lint(CCA->CC->P->Subspaces->N_points);
	inflexion_Pts = NEW_lint(CCA->CC->P->Subspaces->N_points);
	singular_Pts = NEW_lint(CCA->CC->P->Subspaces->N_points);
	type = NEW_int(CCA->CC->P->Subspaces->N_lines);



	ost << "The order of the group is ";
	Curves->go.print_not_scientific(ost);
	ost << "\\\\" << endl;

	ost << "\\bigskip" << endl;

	ost << "The group has " << Curves->nb_orbits
			<< " orbits: \\\\" << endl;

	int i;
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object go1, ol, Ol;
	Ol.create(0);

	vector<string> References;
	int *Ago;
	int *Nb_points;
	int *Nb_singular_points;
	int *Nb_inflexions;
	Ago = NEW_int(Curves->nb_orbits);
	Nb_points = NEW_int(Curves->nb_orbits);
	Nb_singular_points = NEW_int(Curves->nb_orbits);
	Nb_inflexions = NEW_int(Curves->nb_orbits);



	for (i = 0; i < Curves->nb_orbits; i++) {

		if (f_v) {
			cout << "Curve " << i << " / "
					<< Curves->nb_orbits << ": "
					"verbose_level=" << verbose_level << endl;
		}

		Curves->Orbit[i].gens->group_order(go1);

		if (f_v) {
			cout << "stab order " << go1 << endl;
		}

		Ago[i] = go1.as_int();

		D.integral_division_exact(Curves->go, go1, ol);

		if (f_v) {
			cout << "orbit length " << ol << endl;
		}

		long int *data;
		long int *eqn1;
		int eqn[10];
		int nb_pts_on_curve;
		int nb_singular_pts;
		int nb_inflection_pts;
		other::l1_interfaces::latex_interface L;

		data = Curves->Rep + i * Curves->representation_sz;
		eqn1 = data + 9;
		Lint_vec_copy_to_int(eqn1, eqn, 10);

		ost << "\\subsection*{Curve " << i << " / "
				<< Curves->nb_orbits << "}" << endl;
		//ost << "$" << i << " / " << Curves->nb_orbits << "$ $" << endl;

		ost << "$";
		L.lint_set_print_tex_for_inline_text(ost,
				data,
				9 /*CCC->Curves->representation_sz*/);
		ost << "_{";
		go1.print_not_scientific(ost);
		ost << "}$ orbit length $";
		ol.print_not_scientific(ost);
		ost << "$\\\\" << endl;


#if 0
		int_vec_zero(eqn, 10);
		// y = x^3 or X^3 - YZ^2
		eqn[0] = 1;
		eqn[8] = F->minus_one();
		eqn[2] = 0;
//0 & X^3 & ( 3, 0, 0 )
//1 & Y^3 & ( 0, 3, 0 )
//2 & Z^3 & ( 0, 0, 3 )
//3 & X^2Y & ( 2, 1, 0 )
//4 & X^2Z & ( 2, 0, 1 )
//5 & XY^2 & ( 1, 2, 0 )
//6 & Y^2Z & ( 0, 2, 1 )
//7 & XZ^2 & ( 1, 0, 2 )
//8 & YZ^2 & ( 0, 1, 2 )
//9 & XYZ & ( 1, 1, 1 )
#endif
#if 0
		int_vec_zero(eqn, 10);
		// y = x^3 + x + 3
		eqn[0] = 1;
		eqn[2] = 3;
		eqn[6] = 10;
		eqn[7] = 1;
#endif


		ost << "\\begin{eqnarray*}" << endl;
		ost << "&&";


		{
			vector<long int> Points;
			int h;

			CCA->CC->Poly->enumerate_points(eqn,
					Points,
					verbose_level - 4);

			nb_pts_on_curve = Points.size();
			for (h = 0; h < nb_pts_on_curve; h++) {
				Pts_on_curve[h] = Points[h];
			}
		}

		Nb_points[i] = nb_pts_on_curve;


		CC->Poly->print_equation_with_line_breaks_tex(ost,
				eqn,
				5 /* nb_terms_per_line */,
				"\\\\\n&&");
		ost << "\\end{eqnarray*}" << endl;

		ost << "The curve has " << nb_pts_on_curve
				<< " points.\\\\" << endl;


		CC->compute_singular_points(
				eqn,
				Pts_on_curve, nb_pts_on_curve,
				singular_Pts, nb_singular_pts,
				verbose_level - 2);

		ost << "The curve has " << nb_singular_pts
				<< " singular points.\\\\" << endl;
		Nb_singular_points[i] = nb_singular_pts;


		CC->compute_inflexion_points(
				eqn,
				Pts_on_curve, nb_pts_on_curve,
				inflexion_Pts, nb_inflection_pts,
				verbose_level - 2);


		Nb_inflexions[i] = nb_inflection_pts;

		ost << "The curve has " << nb_inflection_pts << " inflexion points: $";
		Lint_vec_print(ost, inflexion_Pts, nb_inflection_pts);
		ost << "$\\\\" << endl;


		CCA->CC->P->Subspaces->line_intersection_type(
				Pts_on_curve, nb_pts_on_curve /* set_size */,
				type, 0 /*verbose_level*/);
		// type[N_lines]

		ost << "The line type is $";
		other::data_structures::tally C;
		C.init(type, CCA->CC->P->Subspaces->N_lines, false, 0);
		C.print_bare_tex(ost, true /* f_backwards*/);
		ost << ".$ \\\\" << endl;


		if (f_with_stabilizers) {
			//ost << "Strong generators are:" << endl;
			Curves->Orbit[i].gens->print_generators_tex(ost);
			D.add_in_place(Ol, ol);
		}

#if 1
		if (nb_inflection_pts == 3) {
			int Basis[9];
			int Basis_t[9];
			int Basis_inv[9];
			int transformed_eqn[10];

			CC->P->unrank_point(Basis, inflexion_Pts[0]);
			CC->P->unrank_point(Basis + 3, inflexion_Pts[1]);

			CC->P->Subspaces->F->Linear_algebra->extend_basis(2, 3, Basis,
				verbose_level);

			//CC->P->unrank_point(Basis + 6, inflexion_Pts[2]);
			CC->F->Linear_algebra->transpose_matrix(Basis, Basis_t, 3, 3);
			CC->F->Linear_algebra->invert_matrix(Basis, Basis_inv, 3, 0 /* verbose_level */);
			CC->Poly->substitute_linear(eqn, transformed_eqn,
					Basis /* int *Mtx_inv */, 0 /* verbose_level */);


			ost << "The transformed equation is:\\\\" << endl;
			ost << "\\begin{eqnarray*}" << endl;
			ost << "&&";


			{
				vector<long int> Points;
				int h;


				CCA->CC->Poly->enumerate_points(transformed_eqn,
						Points,
						verbose_level - 4);

				nb_pts_on_curve = Points.size();
				for (h = 0; h < nb_pts_on_curve; h++) {
					Pts_on_curve[h] = Points[h];
				}
			}

			CC->Poly->print_equation_with_line_breaks_tex(ost,
					transformed_eqn,
					5 /* nb_terms_per_line */,
					"\\\\\n&&");
			ost << "\\end{eqnarray*}" << endl;

			ost << "The transformed curve has " << nb_pts_on_curve
					<< " points.\\\\" << endl;

			CC->compute_singular_points(
					transformed_eqn,
					Pts_on_curve, nb_pts_on_curve,
					singular_Pts, nb_singular_pts,
					verbose_level - 2);

			ost << "The curve has " << nb_singular_pts
					<< " singular points.\\\\" << endl;


			CC->compute_inflexion_points(
					transformed_eqn,
					Pts_on_curve, nb_pts_on_curve,
					inflexion_Pts, nb_inflection_pts,
					verbose_level - 2);

			ost << "The transformed curve has " << nb_inflection_pts
					<< " inflexion points: $";
			Lint_vec_print(ost, inflexion_Pts, nb_inflection_pts);
			ost << "$\\\\" << endl;



		}
#endif


	} // next i
	ost << "The overall number of objects is: " << Ol << "\\\\" << endl;




	ost << "summary of the stabilizer orders:\\\\" << endl;


	for (i = 0; i < Curves->nb_orbits; i++) {
		string ref;

		ref = "";
		References.push_back(ref);
	}


	int *Iso_type1;
	int *Iso_type2;
	int *Iso_type3;
	int *Iso_typeE;
	int *Iso_typeH;
	int *Iso_typeG;
	int e, c, d;

	Iso_type1 = NEW_int(F->q);
	Iso_type2 = NEW_int(F->q);
	Iso_type3 = NEW_int(F->q);
	Iso_typeE = NEW_int(F->q);
	Iso_typeH = NEW_int(F->q);
	Iso_typeG = NEW_int(F->q * F->q);
	family1_recognize(Iso_type1, verbose_level - 1);
	family2_recognize(Iso_type2, verbose_level - 1);
	family3_recognize(Iso_type3, verbose_level - 1);
	familyE_recognize(Iso_typeE, verbose_level - 1);
	familyH_recognize(Iso_typeH, verbose_level - 1);
	familyG_recognize(Iso_typeG, verbose_level - 1);

	ost << "Families 1, 2, 3, E, H: \\\\" << endl;
	for (e = 0; e < F->q; e++) {
		ost << "e=" << e
				<< " iso1=" << Iso_type1[e]
				<< " iso2=" << Iso_type2[e]
				<< " iso3=" << Iso_type2[e]
				<< " isoE=" << Iso_typeE[e]
				<< " isoH=" << Iso_typeH[e]
				<< " \\\\" << endl;
	}
	for (c = 1; c < F->q; c++) {
		for (d = 1; d < F->q; d++) {
			ost << "c=" << c << " d=" << d
					<< " isoG=" << Iso_typeG[c * F->q + d]
					<< " \\\\" << endl;
		}
	}
	for (e = 0; e < F->q; e++) {
		if (Iso_type1[e] != -1) {
			string ref;
			ref = References[Iso_type1[e]];
			if (ref.length()) {
				ref += ",";
			}
			ref += "F1_{" + std::to_string(e) + "}";
			References[Iso_type1[e]] = ref;
		}
	}
	for (e = 0; e < F->q; e++) {
		if (Iso_type2[e] != -1) {
			string ref;
			ref = References[Iso_type2[e]];
			if (ref.length()) {
				ref += ",";
			}
			ref += "F2_{" + std::to_string(e) + "}";
			References[Iso_type2[e]] = ref;
		}
	}
	for (e = 0; e < F->q; e++) {
		if (Iso_type3[e] != -1) {
			string ref;
			ref = References[Iso_type3[e]];
			if (ref.length()) {
				ref += ",";
			}
			ref += "F3_{" + std::to_string(e) + "}";
			References[Iso_type3[e]] = ref;
		}
	}
	for (e = 0; e < F->q; e++) {
		if (Iso_typeE[e] != -1) {
			string ref;
			ref = References[Iso_typeE[e]];
			if (ref.length()) {
				ref += ",";
			}
			ref += "E_{" + std::to_string(e) + "}";
			References[Iso_typeE[e]] = ref;
		}
	}
	for (e = 0; e < F->q; e++) {
		if (Iso_typeH[e] != -1) {
			string ref;
			ref = References[Iso_typeH[e]];
			if (ref.length()) {
				ref += ",";
			}
			ref += "H_{" + std::to_string(e) + "}";
			References[Iso_typeH[e]] = ref;
		}
	}
	for (c = 1; c < F->q; c++) {
		for (d = 1; d < F->q; d++) {
			int iso;

			iso = Iso_typeG[c * F->q + d];
			if (iso == -1) {
				continue;
			}
			string ref;
			ref = References[iso];
			if (ref.length()) {
				ref += ",";
			}
			ref += "G_{" + std::to_string(c) + "," + std::to_string(d) + "}";
			References[iso] = ref;
		}
	}



	other::data_structures::tally C;

	C.init(Ago, Curves->nb_orbits, false, 0);
	ost << "Distribution: $(";
	C.print_bare_tex(ost, true /* f_backwards */);
	ost << ")$\\\\" << endl;


	ost << "$$" << endl;
	ost << "\\begin{array}{|c||c|c|c|c|c|}";
	ost << "\\hline";
	ost << "\\mbox{Curve} & ";
	ost << "\\mbox{Ago} & ";
	ost << "\\mbox{Pts} & ";
	ost << "\\mbox{s. Pts} & ";
	ost << "\\mbox{Infl} & ";
	ost << "\\mbox{References} \\\\";
	ost << "\\hline";
	for (i = 0; i < Curves->nb_orbits; i++) {
		ost << i;
		ost << " & " << Ago[i];
		ost << " & " << Nb_points[i];
		ost << " & " << Nb_singular_points[i];
		ost << " & " << Nb_inflexions[i];
		ost << " & " << References[i];
		ost << "\\\\";
	}
	ost << "\\hline";
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

	ost << "with canonical forms " << endl;
	ost << "\\begin{eqnarray*}" << endl;
	ost << "F1_e &=& X^2Y + XY^2 + eZ^3 \\\\" << endl;
	ost << "F2_e &=& X^2Y + XY^2 + XYZ + eZ^3 \\\\" << endl;
	ost << "F3_e &=& XYZ + e(X + Y + Z)^3 \\\\" << endl;
	ost << "E_d &=& Z^2Y + X^3 - dY^3 \\\\" << endl;
	ost << "H_e &=& Z^2Y + X^3 + eXY^2 \\\\" << endl;
	ost << "G_{c,d} &=&  Z^2Y + X^3 + cXY^2 + dY^3 \\\\" << endl;
	ost << "\\end{eqnarray*}" << endl;
	ost << "for $c,d,e \\in {\\mathbb F}_{" << F->q << "}$ \\\\" << endl;

	FREE_int(Iso_type1);
	FREE_int(Iso_type2);
	FREE_int(Iso_type3);
	FREE_int(Iso_typeE);
	FREE_int(Iso_typeH);
	FREE_int(Iso_typeG);
	FREE_int(Ago);
	FREE_int(Nb_points);
	FREE_int(Nb_singular_points);
	FREE_int(Nb_inflexions);


	FREE_lint(Pts_on_curve);
	FREE_lint(inflexion_Pts);
	FREE_lint(singular_Pts);
	FREE_int(type);

}


}}}


