/*
 * classify_cubic_curves.cpp
 *
 *  Created on: Mar 7, 2019
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;
using namespace orbiter::foundations;

namespace orbiter {
namespace top_level {


classify_cubic_curves::classify_cubic_curves()
{
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
	//null();
}

classify_cubic_curves::~classify_cubic_curves()
{
	freeself();
}

void classify_cubic_curves::null()
{
}

void classify_cubic_curves::freeself()
{
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
	null();
}

void classify_cubic_curves::init(cubic_curve_with_action *CCA,
		const char *starter_directory_name,
		const char *base_fname,
		int argc, const char **argv,
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

	Arc_gen->read_arguments(argc, argv);


	Arc_gen->init(F,
			starter_directory_name,
			base_fname,
			9 /* starter_size */,
			argc, argv,
			verbose_level);


	if (f_v) {
		cout << "classify_cubic_curves::init after Arc_gen->init" << endl;
		}


	if (f_v) {
		cout << "classify_cubic_curves::init done" << endl;
		}
}

void classify_cubic_curves::compute_starter(int verbose_level)
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


void classify_cubic_curves::test_orbits(int verbose_level)
{
	//verbose_level += 2;
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; // (verbose_level >= 2);
	int i, r;
	int S[9];
	int *Pts;
	int *type;
	int nb_pts;

	if (f_v) {
		cout << "classify_cubic_curves::test_orbits" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	nb_orbits_on_sets = Arc_gen->gen->nb_orbits_at_level(9);

	Pts = NEW_int(CC->P->N_points);
	type = NEW_int(CC->P->N_lines);

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
			int_vec_print(cout, S, 5);
			cout << endl;
			}




#if 1
		if (f_vv) {
			CC->P->print_set(S, 9);
			}
#endif

		r = CC->compute_system_in_RREF(9,
				S, 0 /*verbose_level*/);
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

			CC->P->determine_cubic_in_plane(
					CC->Poly,
					9 /* nb_pts */, S /* int *Pts */, eqn,
					verbose_level - 5);

			CC->Poly->enumerate_points(eqn, Pts, nb_pts,
					verbose_level - 2);

			CC->P->line_intersection_type(
					Pts, nb_pts /* set_size */,
					type, 0 /*verbose_level*/);

			classify Cl;

			Cl.init(type, CC->P->N_lines, FALSE, 0);
			idx = Cl.determine_class_by_value(q + 1);

			if (idx == -1) {

				Idx[nb++] = i;
			}
		}
	}

	if (f_v) {
		cout << "classify_cubic_curves::test_orbits we found "
				<< nb << " / " << nb_orbits_on_sets
				<< " orbits where the rank is 9" << endl;
		cout << "Idx=";
		int_vec_print(cout, Idx, nb);
		cout << endl;
		}

	FREE_int(Pts);
	FREE_int(type);

	if (f_v) {
		cout << "classify_cubic_curves::test_orbits done" << endl;
		}
}


void classify_cubic_curves::downstep(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
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
		int_vec_print(cout, Idx, nb);
		cout << endl;
		}



	Flag_orbits = NEW_OBJECT(flag_orbits);
	Flag_orbits->init(A, A,
		nb_orbits_on_sets /* nb_primary_orbits_lower */,
		9 + 10 /* pt_representation_sz */,
		nb,
		verbose_level);

	if (f_v) {
		cout << "classify_cubic_curves::downstep "
				"initializing flag orbits" << endl;
		}

	nb_flag_orbits = 0;
	for (f = 0; f < nb; f++) {

		i = Idx[f];
		if (f_v) {
			cout << "classify_cubic_curves::downstep "
					"orbit " << f << " / " << nb
					<< " with rank = 9 is orbit "
					<< i << " / " << nb_orbits_on_sets << endl;
			}

		set_and_stabilizer *R;
		longinteger_object ol;
		longinteger_object go;
		int dataset[19];

		R = Arc_gen->gen->get_set_and_stabilizer(
				9 /* level */,
				i /* orbit_at_level */,
				0 /* verbose_level */);

		Arc_gen->gen->orbit_length(
				i /* node */, 9 /* level */, ol);

		R->Strong_gens->group_order(go);

		int_vec_copy(R->data, dataset, 9);

		int eqn[10];
		if (f_vv) {
			cout << "9 points = ";
			int_vec_print(cout, dataset, 9);
			cout << endl;
			}

		if (f_vv) {
			cout << "classify_cubic_curves::downstep before "
					"create_double_six_from_five_lines_with_"
					"a_common_transversal" << endl;
			}

		CC->P->determine_cubic_in_plane(
				CC->Poly,
				9 /* nb_pts */, dataset /* int *Pts */, eqn,
				verbose_level - 5);
		//c = Surf_A->create_double_six_from_five_lines_with_a_common_transversal(
		//		dataset + 5, pt0_line, double_six,
		//		0 /*verbose_level*/);

		if (f_vv) {
			cout << "The starter configuration is good, "
					"a cubic has been computed:" << endl;
			int_vec_print(cout, eqn, 10);
			}

		int_vec_copy(eqn, dataset + 9, 10);


		Flag_orbits->Flag_orbit_node[nb_flag_orbits].init(
			Flag_orbits,
			nb_flag_orbits /* flag_orbit_index */,
			i /* downstep_primary_orbit */,
			0 /* downstep_secondary_orbit */,
			ol.as_int() /* downstep_orbit_len */,
			FALSE /* f_long_orbit */,
			dataset /* int *pt_representation */,
			R->Strong_gens,
			verbose_level - 2);
		R->Strong_gens = NULL;

		if (f_vv) {
			cout << "orbit " << f << " / " << nb
				<< " with rank = 9 is orbit " << i
				<< " / " << nb_orbits_on_sets << ", stab order "
				<< go << endl;
			}
		nb_flag_orbits++;

		FREE_OBJECT(R);
		}

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


void classify_cubic_curves::upstep(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, r;
	int f, po, so;
	int *f_processed;
	int nb_processed;
	int *Elt;
	int idx_set[9];
	int set[9];
	int canonical_set[9];
	int *Pts;
	int *type;

	if (f_v) {
		cout << "classify_cubic_curves::upstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}


	Elt = NEW_int(A->elt_size_in_int);
	Pts = NEW_int(CCA->CC->P->N_points);
	type = NEW_int(CCA->CC->P->N_lines);

	f_processed = NEW_int(Flag_orbits->nb_flag_orbits);
	int_vec_zero(f_processed, Flag_orbits->nb_flag_orbits);
	nb_processed = 0;

	Curves = NEW_OBJECT(classification_step);

	longinteger_object go;
	A->group_order(go);

	Curves->init(A, A,
			Flag_orbits->nb_flag_orbits, 19, go,
			verbose_level);


	if (f_v) {
		cout << "flag orbit : downstep_primary_orbit" << endl;
		if (Flag_orbits->nb_flag_orbits < 1000) {
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
		int dataset[19];

		if (f_processed[f]) {
			continue;
			}

		progress = ((double)nb_processed * 100. ) /
				(double) Flag_orbits->nb_flag_orbits;

		if (f_v) {
			cout << "Defining n e w orbit "
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
		int_vec_copy(Flag_orbits->Pt + f * 19, dataset, 19);




		vector_ge *coset_reps;
		int nb_coset_reps;


		strong_generators *S;
		longinteger_object go;
		int eqn[10];

		int_vec_copy(dataset + 9, eqn, 10);

		if (f_v) {
			cout << "equation:";
			int_vec_print(cout, eqn, 10);
			cout << endl;
			}
		S = Flag_orbits->Flag_orbit_node[f].gens->create_copy();
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

		CCA->CC->Poly->enumerate_points(eqn, Pts, nb_pts,
				verbose_level - 2);
		if (f_v) {
			cout << "po=" << po << " so=" << so
					<< " we found a curve with " << nb_pts
					<< " points" << endl;
			}

		N = int_n_choose_k(nb_pts, 9);

		coset_reps = NEW_OBJECT(vector_ge);
		coset_reps->init(CCA->A);
		coset_reps->allocate(N);


		for (i = 0; i < N; i++) {
			if (FALSE) {
				cout << "po=" << po << " so=" << so
						<< " i=" << i << " / " << N << endl;
				}

			unrank_k_subset(i, idx_set, nb_pts, 9);
			for (j = 0; j < 9; j++) {
				set[j] = Pts[idx_set[j]];
			}

			r = CC->compute_system_in_RREF(9,
					set, 0 /*verbose_level*/);

			if (r < 9) {
				continue;
			}

			CCA->CC->P->line_intersection_type(
				set, 9 /* set_size */, type, 0 /*verbose_level*/);
			// type[N_lines]

			for (j = 0; j < CCA->CC->P->N_lines; j++) {
				if (type[j] > 3) {
					break;
				}
			}
			if (j < CCA->CC->P->N_lines) {
				continue;
			}


			orbit_index = Arc_gen->gen->trace_set(
					set, 9, 9,
					canonical_set,
					Elt,
					verbose_level - 2);

			if (!int_vec_search(Po, Flag_orbits->nb_flag_orbits,
					orbit_index, f2)) {
				cout << "cannot find orbit " << orbit_index
						<< " in Po" << endl;
				cout << "Po=";
				int_vec_print(cout, Po, Flag_orbits->nb_flag_orbits);
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
					A->element_print_quick(Elt, cout);
					cout << endl;
					}
				A->element_move(Elt, coset_reps->ith(nb_coset_reps), 0);
				nb_coset_reps++;
				//S->add_single_generator(Elt3,
				//2 /* group_index */, verbose_level - 2);
				}
			else {
				if (FALSE) {
					cout << "We are identifying flag orbit "
							<< f2 << " with flag orbit " << f << endl;
					}
				if (!f_processed[f2]) {
					Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit
						= Flag_orbits->nb_primary_orbits_upper;
					Flag_orbits->Flag_orbit_node[f2].f_fusion_node
						= TRUE;
					Flag_orbits->Flag_orbit_node[f2].fusion_with
						= f;
					Flag_orbits->Flag_orbit_node[f2].fusion_elt
						= NEW_int(A->elt_size_in_int);
					A->element_invert(Elt,
							Flag_orbits->Flag_orbit_node[f2].fusion_elt,
							0);
					f_processed[f2] = TRUE;
					nb_processed++;
					}
				else {
					if (FALSE) {
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

		coset_reps->reallocate(nb_coset_reps);

		strong_generators *Aut_gens;

		{
		longinteger_object ago;

		if (f_v) {
			cout << "classify_cubic_curves::upstep "
					"Extending the group by a factor of "
					<< nb_coset_reps << endl;
			}
		Aut_gens = NEW_OBJECT(strong_generators);
		Aut_gens->init_group_extension(S,
				coset_reps, nb_coset_reps,
				verbose_level - 2);
		if (f_v) {
			cout << "classify_double_sixes::upstep "
					"Aut_gens tl = ";
			int_vec_print(cout,
					Aut_gens->tl, Aut_gens->A->base_len);
			cout << endl;
			}

		Aut_gens->group_order(ago);


		if (f_v) {
			cout << "the double six has a stabilizer of order "
					<< ago << endl;
			cout << "The double six stabilizer is:" << endl;
			Aut_gens->print_generators_tex(cout);
			}
		}



		Curves->Orbit[Flag_orbits->nb_primary_orbits_upper].init(
				Curves,
			Flag_orbits->nb_primary_orbits_upper,
			Aut_gens, dataset, verbose_level);

		FREE_OBJECT(coset_reps);
		FREE_OBJECT(S);

		f_processed[f] = TRUE;
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
	FREE_int(Pts);
	FREE_int(type);


	if (f_v) {
		cout << "classify_cubic_curves::upstep done" << endl;
		}
}


void classify_cubic_curves::do_classify(int verbose_level)
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
				<< " double sixes out of "
				<< Flag_orbits->nb_flag_orbits
				<< " flag orbits" << endl;
		}

	if (f_v) {
		cout << "classify_cubic_curves::do_classify done" << endl;
		}
}




}}


