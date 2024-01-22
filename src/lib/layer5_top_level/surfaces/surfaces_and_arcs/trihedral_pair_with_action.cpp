/*
 * trihedral_pair_with_action.cpp
 *
 *  Created on: Jul 19, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_arcs {


trihedral_pair_with_action::trihedral_pair_with_action()
{
	AL = NULL;

	//int The_six_plane_equations[6 * 4];
	The_surface_equations = NULL;
	//long int plane6_by_dual_ranks[6];
	lambda = lambda_rk = 0;
	t_idx = 0;

	stab_gens_trihedral_pair = NULL;
	gens_subgroup = NULL;
	A_on_equations = NULL;
	Orb = NULL;
	trihedral_pair_orbit_index = 0;
	cosets = NULL;
	coset_reps = NULL;
	aut_T_index = NULL;
	aut_coset_index = NULL;
	Aut_gens = NULL;

	//int F_plane[3 * 4];
	//int G_plane[3 * 4];
	System = NULL;

	//int Iso_type_as_double_triplet[120];
	Double_triplet_type_distribution = NULL;
	Double_triplet_types = NULL;
	Double_triplet_type_values = NULL;
	nb_double_triplet_types = 0;

	transporter0 = NULL;
	transporter = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
	Elt5 = NULL;
}





trihedral_pair_with_action::~trihedral_pair_with_action()
{
	if (The_surface_equations) {
		FREE_int(The_surface_equations);
	}


	if (stab_gens_trihedral_pair) {
		FREE_OBJECT(stab_gens_trihedral_pair);
	}
	if (gens_subgroup) {
		FREE_OBJECT(gens_subgroup);
	}
	if (A_on_equations) {
		FREE_OBJECT(A_on_equations);
	}
	if (Orb) {
		FREE_OBJECT(Orb);
	}
	if (cosets) {
		FREE_OBJECT(cosets);
	}
	if (coset_reps) {
		FREE_OBJECT(coset_reps);
	}
	if (aut_T_index) {
		FREE_int(aut_T_index);
	}
	if (aut_coset_index) {
		FREE_int(aut_coset_index);
	}
	if (Aut_gens) {
		FREE_OBJECT(Aut_gens);
	}



	if (System) {
		FREE_int(System);
	}

	if (Double_triplet_type_distribution) {
		FREE_OBJECT(Double_triplet_type_distribution);
	}
	if (Double_triplet_types) {
		FREE_OBJECT(Double_triplet_types);
	}
	if (Double_triplet_type_values) {
		FREE_int(Double_triplet_type_values);
	}

	if (transporter0) {
		FREE_int(transporter0);
	}
	if (transporter) {
		FREE_int(transporter);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Elt2) {
		FREE_int(Elt2);
	}
	if (Elt3) {
		FREE_int(Elt3);
	}
	if (Elt4) {
		FREE_int(Elt4);
	}
	if (Elt5) {
		FREE_int(Elt5);
	}
}

void trihedral_pair_with_action::init(
		arc_lifting *AL, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "trihedral_pair_with_action::init" << endl;
	}

	trihedral_pair_with_action::AL = AL;


	transporter0 = NEW_int(AL->Surf_A->A->elt_size_in_int);
	transporter = NEW_int(AL->Surf_A->A->elt_size_in_int);
	Elt1 = NEW_int(AL->Surf_A->A->elt_size_in_int);
	Elt2 = NEW_int(AL->Surf_A->A->elt_size_in_int);
	Elt3 = NEW_int(AL->Surf_A->A->elt_size_in_int);
	Elt4 = NEW_int(AL->Surf_A->A->elt_size_in_int);
	Elt5 = NEW_int(AL->Surf_A->A->elt_size_in_int);



	if (f_v) {
		cout << "trihedral_pair_with_action::init before "
				"create_surface_from_trihedral_pair_and_arc"
				<< endl;
	}
	create_surface_from_trihedral_pair_and_arc(
		AL->Web->t_idx0,
		verbose_level - 2);
	if (f_v) {
		cout << "trihedral_pair_with_action::init after "
				"create_surface_from_trihedral_pair_and_arc"
				<< endl;
	}

	if (f_v) {
		print_equations();
	}


	if (f_v) {
		cout << "trihedral_pair_with_action::init "
				"before create_clebsch_system" << endl;
	}
	create_clebsch_system(
		//The_six_plane_equations,
		//lambda,
		0 /* verbose_level */);
	if (f_v) {
		cout << "trihedral_pair_with_action::init "
				"after create_clebsch_system" << endl;
	}



	if (f_v) {
		cout << "trihedral_pair_with_action::init before "
				"create_stabilizer_of_trihedral_pair" << endl;
	}
	stab_gens_trihedral_pair = create_stabilizer_of_trihedral_pair(
			//plane6_by_dual_ranks,
			trihedral_pair_orbit_index,
			verbose_level - 2);
	if (f_v) {
		cout << "trihedral_pair_with_action::init after "
				"create_stabilizer_of_trihedral_pair" << endl;
	}

	stab_gens_trihedral_pair->group_order(stabilizer_of_trihedral_pair_go);
	if (f_v) {
		cout << "trihedral_pair_with_action::init the stabilizer of "
				"the trihedral pair has order "
				<< stabilizer_of_trihedral_pair_go << endl;
	}



	if (f_v) {
		cout << "trihedral_pair_with_action::init before "
				"create_action_on_equations_and_compute_orbits" << endl;
		}
	create_action_on_equations_and_compute_orbits(
		The_surface_equations,
		stab_gens_trihedral_pair
		/* strong_generators *gens_for_stabilizer_of_trihedral_pair */,
		A_on_equations, Orb,
		verbose_level - 2);
	if (f_v) {
		cout << "trihedral_pair_with_action::init after "
				"create_action_on_equations_and_compute_orbits" << endl;
	}


	if (f_v) {
		cout << "trihedral_pair_with_action::init the orbits "
				"on the pencil of surfaces are:" << endl;
	}
	Orb->print_and_list_orbits(cout);



	//Surf_A->A->group_order(go_PGL);


	if (f_v) {
		cout << "trihedral_pair_with_action::init before "
				"Orb->stabilizer_any_point_plus_cosets" << endl;
	}
	gens_subgroup =
		Orb->stabilizer_any_point_plus_cosets(
			AL->Surf_A->A,
			stabilizer_of_trihedral_pair_go,
			lambda_rk /* pt */,
			cosets,
			verbose_level - 2);

	if (f_v) {
		cout << "trihedral_pair_with_action::init after "
				"Orb->stabilizer_any_point_plus_cosets" << endl;
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::init we found the "
				"following coset representatives:" << endl;
		cosets->print(cout);
	}




	if (f_v) {
		cout << "trihedral_pair_with_action::init after "
				"Orb->stabilizer_any_point" << endl;
	}
	gens_subgroup->group_order(stab_order);
	if (f_v) {
		cout << "trihedral_pair_with_action::init "
				"The stabilizer of the trihedral pair inside "
				"the group of the surface has order "
				<< stab_order << endl;
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::init elements "
				"in the stabilizer:" << endl;
		gens_subgroup->print_elements_ost(cout);
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::init The stabilizer of "
				"the trihedral pair inside the stabilizer of the "
				"surface is generated by:" << endl;
		gens_subgroup->print_generators_tex(cout);
	}






	AL->Surf->compute_nine_lines_by_dual_point_ranks(
			AL->Web->Dual_point_ranks + 0 * 6,
			AL->Web->Dual_point_ranks + 0 * 6 + 3,
			nine_lines,
			0 /* verbose_level */);


	if (f_v) {
		cout << "trihedral_pair_with_action::init before compute_iso_types_as_double_triplets" << endl;
	}
	compute_iso_types_as_double_triplets(verbose_level);
	if (f_v) {
		cout << "trihedral_pair_with_action::init after compute_iso_types_as_double_triplets" << endl;
		report_iso_type_as_double_triplets(cout);
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::init before "
				"loop_over_trihedral_pairs" << endl;
	}
	loop_over_trihedral_pairs(cosets,
		coset_reps,
		aut_T_index,
		aut_coset_index,
		verbose_level - 2);
	if (f_v) {
		cout << "trihedral_pair_with_action::init after "
				"loop_over_trihedral_pairs" << endl;
		cout << "arc_lifting::create_surface we found an "
				"orbit of length " << coset_reps->len << endl;
	}



	// the problem is here:
	{
		ring_theory::longinteger_object ago;

		if (f_v) {
			cout << "trihedral_pair_with_action::init "
					"Extending the group:" << endl;
		}
		Aut_gens = NEW_OBJECT(groups::strong_generators);
		Aut_gens->init_group_extension(gens_subgroup,
			coset_reps, coset_reps->len, verbose_level - 3);

		Aut_gens->group_order(ago);
		if (f_v) {
			cout << "trihedral_pair_with_action::init "
					"The automorphism group has order " << ago << endl;
			cout << "trihedral_pair_with_action::init "
					"The automorphism group is:" << endl;
			Aut_gens->print_generators_tex(cout);
		}
	}
	if (f_v) {
		cout << "trihedral_pair_with_action::init done" << endl;
	}
}


void trihedral_pair_with_action::loop_over_trihedral_pairs(
		data_structures_groups::vector_ge *cosets,
	data_structures_groups::vector_ge *&coset_reps,
	int *&aut_T_index, int *&aut_coset_index,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 3);
	//int f_vvv = (verbose_level >= 5);
	int i, j;
	long int planes6[6];
	int orbit_index0;
	int orbit_index;
	int orbit_length;
	//long int Nine_lines0[9];
	//long int Nine_lines[9];
	//long int *v;
	//int sz;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "trihedral_pair_with_action::loop_over_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
				"we are considering " << cosets->len
				<< " cosets from the downstep" << endl;
	}


	orbit_length = 0;


	AL->Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
			AL->Web->Dual_point_ranks + AL->Web->t_idx0 * 6,
			transporter0,
			orbit_index0,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
				"Trihedral pair " << AL->Web->t_idx0
				<< " lies in orbit " << orbit_index0 << endl;
	}

#if 0
	Surf->compute_nine_lines_by_dual_point_ranks(
			Web->Dual_point_ranks + Web->t_idx0 * 6,
			Web->Dual_point_ranks + Web->t_idx0 * 6 + 3,
		Nine_lines0,
		0 /* verbose_level */);

	if (false) {
		cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
				"The first trihedral pair gives "
				"the following nine lines: ";
		lint_vec_print(cout, Nine_lines0, 9);
		cout << endl;
	}
#endif

	coset_reps = NEW_OBJECT(data_structures_groups::vector_ge);
	coset_reps->init(AL->Surf_A->A, verbose_level - 2);
	coset_reps->allocate(AL->Web->nb_T * cosets->len, verbose_level - 2);

	aut_T_index = NEW_int(AL->Web->nb_T * cosets->len);
	aut_coset_index = NEW_int(AL->Web->nb_T * cosets->len);

	for (i = 0; i < AL->Web->nb_T; i++) {

		if (f_vv) {
			cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
					"testing if trihedral pair "
					<< i << " / " << AL->Web->nb_T << " = " << AL->Web->T_idx[i];
			cout << " lies in the orbit:" << endl;
		}

		Lint_vec_copy(AL->Web->Dual_point_ranks + i * 6, planes6, 6);

#if 0
		Surf->compute_nine_lines_by_dual_point_ranks(
			planes6,
			planes6 + 3,
			Nine_lines,
			0 /* verbose_level */);

		if (false) {
			cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
					"The " << i << "-th trihedral "
					"pair gives the following nine lines: ";
			lint_vec_print(cout, Nine_lines, 9);
			cout << endl;
		}

		Sorting.vec_intersect(Nine_lines0, 9, Nine_lines, 9, v, sz);

		if (false) {
			cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
					"The nine lines of the " << i
					<< "-th trihedral pair intersect the "
						"nine lines of the first in " << sz
						<< " lines, which are: ";
			lint_vec_print(cout, v, sz);
			cout << endl;
		}

		if (false) {
			Surf->print_trihedral_pair_in_dual_coordinates_in_GAP(
				planes6, planes6 + 3);
			cout << endl;
		}

		FREE_lint(v);
#endif



		AL->Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
			planes6,
			transporter,
			orbit_index,
			0 /*verbose_level */);


		if (orbit_index != orbit_index0) {
			if (f_vv) {
				cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
						"trihedral pair " << i << " / " << AL->Web->nb_T
						<< " lies in orbit " << orbit_index
						<< " and so $T_{" << AL->Web->t_idx0
						<< "}$ and T_i are not isomorphic" << endl;
			}
			continue;
		}
		if (f_v) {
			cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
					"trihedral pair " << i << " / " << AL->Web->nb_T
					<< " lies in orbit " << orbit_index
					<< " and so $T_{" << AL->Web->t_idx0
					<< "}$ and T_i are isomorphic" << endl;
		}


		AL->Surf_A->A->Group_element->element_invert(transporter, Elt1, 0);
		AL->Surf_A->A->Group_element->element_mult(transporter0, Elt1, Elt2, 0);
		if (f_vv) {
			cout << "Elt2:" << endl;
			AL->Surf_A->A->Group_element->element_print_quick(Elt2, cout);
		}

		for (j = 0; j < cosets->len; j++) {

			if (f_v) {
				cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
						"testing coset j=" << j << " / "
						<< cosets->len << " orbit_length = " << orbit_length << endl;
			}

			// contragredient action:

			AL->Surf_A->A->Group_element->element_transpose(Elt2, Elt3, 0 /* verbose_level*/);



			AL->Surf_A->A->Group_element->element_move(cosets->ith(j), Elt5, 0);
			//AL->Surf_A->A->element_invert(cosets->ith(j), Elt5, 0);
			AL->Surf_A->A->Group_element->element_mult(Elt5, Elt3, Elt4, 0);
			//AL->Surf_A->A->element_mult(Elt3, Elt5, Elt4, 0);

			//cout << "transporter transposed:" << endl;
			//A->print_quick(cout, Elt2);

			int coeff_out[20];


			//Surf_A->A->element_invert(Elt4, Elt5, 0);

			if (false) {
				cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
						"Elt4:" << endl;
				AL->Surf_A->A->Group_element->element_print_quick(Elt4, cout);
			}



			algebra::matrix_group *M;

			if (f_v) {
				cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
						"before M->Element->substitute_surface_equation" << endl;
			}

			M = AL->Surf_A->A->G.matrix_grp;

			M->Element->substitute_surface_equation(Elt4,
					AL->the_equation, coeff_out, AL->Surf,
					verbose_level - 6);


			AL->F->Projective_space_basic->PG_element_normalize(
					coeff_out, 1, 20);

			if (f_v) {
				cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
						"The transformed equation is:" << endl;
				Int_vec_print(cout, coeff_out, 20);
				cout << endl;
			}


			if (Sorting.int_vec_compare(coeff_out, AL->the_equation, 20) == 0) {
				if (f_v) {
					cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
							"trihedral pair " << i << " / " << AL->Web->nb_T
							<< ", coset " << j << " / " << cosets->len
							<< " gives automorphism, increased "
							"orbit length is " << orbit_length + 1 << endl;
					cout << "coset rep = " << endl;
					AL->Surf_A->A->Group_element->element_print_quick(Elt3, cout);
				}
#if 0
				Surf->compute_nine_lines_by_dual_point_ranks(
					planes6, planes6 + 3,
					Nine_lines,
					0 /* verbose_level */);


				if (false) {
					cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
							"The " << orbit_length + 1 << "-th "
							"trihedral pair in the orbit gives "
							"the following nine lines: ";
					lint_vec_print(cout, Nine_lines, 9);
					cout << endl;
				}
#endif


				AL->Surf_A->A->Group_element->element_move(Elt4, coset_reps->ith(orbit_length), 0);

				aut_T_index[orbit_length] = i;
				aut_coset_index[orbit_length] = j;
				orbit_length++;
			}
			else {
				if (f_v) {
					cout << "trihedral_pair_with_action::loop_over_trihedral_pairs "
							"trihedral pair " << i << " / " << AL->Web->nb_T
							<< " coset " << j << " / " << cosets->len
							<< " does not lie in the orbit" << endl;
				}
				//exit(1);
			}
		} // next j

	} // next i

	if (f_v) {
		cout << "trihedral_pair_with_action::loop_over_trihedral_pairs orbit_length = " << orbit_length << endl;
		cout << "i : aut_T_index[i] : aut_coset_index[i]" << endl;
		for (i = 0; i < orbit_length; i++) {
			cout << i << " : " << aut_T_index[i] << " : " << aut_coset_index[i] << endl;
		}
	}

	coset_reps->reallocate(orbit_length, verbose_level - 2);

	if (f_v) {
		cout << "trihedral_pair_with_action::loop_over_trihedral_pairs we found an "
				"orbit of trihedral pairs of length "
				<< orbit_length << endl;
		//cout << "coset reps:" << endl;
		//coset_reps->print_tex(cout);
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::loop_over_trihedral_pairs done" << endl;
	}
}



void trihedral_pair_with_action::create_the_six_plane_equations(
		int t_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "trihedral_pair_with_action::create_the_six_plane_equations "
				"t_idx=" << t_idx << endl;
	}


	Lint_vec_copy(AL->Surf->Schlaefli->Schlaefli_trihedral_pairs->Axes + t_idx * 6,
			AL->Web->row_col_Eckardt_points, 6);

	for (i = 0; i < 6; i++) {
		Int_vec_copy(AL->Web->Tritangent_plane_equations + AL->Web->row_col_Eckardt_points[i] * 4,
				The_six_plane_equations + i * 4, 4);
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::create_the_six_plane_equations" << endl;
		cout << "The_six_plane_equations=" << endl;
		Int_matrix_print(The_six_plane_equations, 6, 4);
	}

	for (i = 0; i < 6; i++) {
		plane6_by_dual_ranks[i] = AL->Surf->P->rank_point(The_six_plane_equations + i * 4);
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::create_the_six_plane_equations done" << endl;
	}
}

void trihedral_pair_with_action::create_surface_from_trihedral_pair_and_arc(
	int t_idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "trihedral_pair_with_action::create_surface_from_trihedral_pair_and_arc t_idx=" << t_idx << endl;
	}

	The_surface_equations = NEW_int((AL->q + 1) * 20);

	create_the_six_plane_equations(t_idx, verbose_level);


	if (f_v) {
		cout << "trihedral_pair_with_action::create_surface_from_trihedral_pair_and_arc "
				"before create_equations_for_pencil_of_surfaces_from_trihedral_pair" << endl;
	}
	AL->Surf->create_equations_for_pencil_of_surfaces_from_trihedral_pair(
		The_six_plane_equations, The_surface_equations,
		verbose_level);

	if (f_v) {
		cout << "trihedral_pair_with_action::create_surface_from_trihedral_pair_and_arc "
				"before create_lambda_from_trihedral_pair_and_arc" << endl;
	}
	AL->Web->create_lambda_from_trihedral_pair_and_arc(AL->arc,
		t_idx, lambda, lambda_rk,
		verbose_level);


	Int_vec_copy(The_surface_equations + lambda_rk * 20, AL->the_equation, 20);

	if (f_v) {
		cout << "trihedral_pair_with_action::create_surface_from_trihedral_pair_and_arc done" << endl;
	}
}

groups::strong_generators *trihedral_pair_with_action::create_stabilizer_of_trihedral_pair(
	int &trihedral_pair_orbit_index,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	groups::strong_generators *gens_dual;
	groups::strong_generators *gens;
	ring_theory::longinteger_object go;

	gens = NEW_OBJECT(groups::strong_generators);


	if (f_v) {
		cout << "trihedral_pair_with_action::create_stabilizer_of_trihedral_pair" << endl;
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::create_stabilizer_of_trihedral_pair "
				"before Surf_A->identify_trihedral_pair_and_get_stabilizer" << endl;
	}

	gens_dual =
			AL->Surf_A->Classify_trihedral_pairs->identify_trihedral_pair_and_get_stabilizer(
					plane6_by_dual_ranks, transporter, trihedral_pair_orbit_index,
					verbose_level);

	if (f_v) {
		cout << "trihedral_pair_with_action::create_stabilizer_of_trihedral_pair "
				"after Surf_A->identify_trihedral_pair_and_get_stabilizer" << endl;
	}
	gens_dual->group_order(go);


	if (f_v) {
		cout << "trihedral_pair_with_action::create_stabilizer_of_trihedral_pair "
				"trihedral_pair_orbit_index="
				<< trihedral_pair_orbit_index
				<< " group order = " << go << endl;
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::create_stabilizer_of_trihedral_pair "
				"group generators:" << endl;
		gens_dual->print_generators_tex(cout);
		//gens_dual->print_elements_ost(cout);
	}


	gens->init(AL->Surf_A->A);

	if (f_v) {
		cout << "trihedral_pair_with_action::create_stabilizer_of_trihedral_pair "
				"before gens->init_transposed_group" << endl;
	}
	gens->init_transposed_group(gens_dual, verbose_level);

	if (f_v) {
		cout << "trihedral_pair_with_action::create_stabilizer_of_trihedral_pair "
				"The transposed stabilizer is generated by:" << endl;
		gens->print_generators_tex(cout);
		//gens->print_elements_ost(cout);
	}

	FREE_OBJECT(gens_dual);

	if (f_v) {
		cout << "trihedral_pair_with_action::create_stabilizer_of_trihedral_pair done" << endl;
	}
	return gens;
}

void trihedral_pair_with_action::create_action_on_equations_and_compute_orbits(
	int *The_surface_equations,
	groups::strong_generators *gens_for_stabilizer_of_trihedral_pair,
	actions::action *&A_on_equations, groups::schreier *&Orb,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "trihedral_pair_with_action::create_action_on_equations_and_compute_orbits "
				"verbose_level = " << verbose_level << endl;
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::create_action_on_equations_and_compute_orbits "
				"before AG.orbits_on_equations" << endl;
	}

	actions::action_global AG;

	AG.orbits_on_equations(
			AL->Surf_A->A,
			AL->Surf->PolynomialDomains->Poly3_4,
			The_surface_equations,
			AL->q + 1 /* nb_equations */,
			gens_for_stabilizer_of_trihedral_pair,
			A_on_equations,
			Orb,
			verbose_level);

	if (f_v) {
		cout << "trihedral_pair_with_action::create_action_on_equations_and_compute_orbits "
				"after AG.orbits_on_equations" << endl;
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::create_action_on_equations_and_compute_orbits done" << endl;
	}
}

void trihedral_pair_with_action::create_clebsch_system(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "trihedral_pair_with_action::create_clebsch_system" << endl;
	}

	Int_vec_copy(The_six_plane_equations, F_plane, 12);
	Int_vec_copy(The_six_plane_equations + 12, G_plane, 12);
	if (f_v) {
		cout << "F_planes:" << endl;
		Int_matrix_print(F_plane, 3, 4);
		cout << "G_planes:" << endl;
		Int_matrix_print(G_plane, 3, 4);
	}

	AL->Surf->compute_nine_lines(F_plane, G_plane, nine_lines, 0 /* verbose_level */);

	if (f_v) {
		cout << "trihedral_pair_with_action::create_clebsch_system" << endl;
		cout << "The nine lines are: ";
		Lint_vec_print(cout, nine_lines, 9);
		cout << endl;
	}

	AL->Surf->prepare_system_from_FG(F_plane, G_plane,
		lambda, System, verbose_level);

	if (f_v) {
		cout << "trihedral_pair_with_action::create_clebsch_system "
				"The System:" << endl;
		for (i = 0; i < 3; i++) {
			for (j = 0; j < 4; j++) {
				int *p = System + (i * 4 + j) * 3;
				AL->Surf->PolynomialDomains->Poly1->print_equation(cout, p);
				cout << endl;
			}
		}
	}

	if (f_v) {
		cout << "trihedral_pair_with_action::create_clebsch_system done" << endl;
	}
}

void trihedral_pair_with_action::compute_iso_types_as_double_triplets(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int planes6[6];
	int i, orbit_index;

	if (f_v) {
		cout << "trihedral_pair_with_action::compute_iso_types_as_double_triplets" << endl;
	}

	for (i = 0; i < AL->Web->nb_T; i++) {

		if (f_v) {
			cout << "computing iso type of trihedral pair " << i << " / "
					<< AL->Web->nb_T << " = " << AL->Web->T_idx[i];
			cout << ":" << endl;
		}

		Lint_vec_copy(AL->Web->Dual_point_ranks + i * 6, planes6, 6);
		AL->Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
			planes6,
			transporter,
			orbit_index,
			0 /*verbose_level */);

		if (f_v) {
			cout << "Trihedral pair " << i << " lies in orbit "
					<< orbit_index << "\\\\" << endl;
			cout << "An isomorphism is given by" << endl;
			cout << "$$" << endl;
			AL->Surf_A->A->Group_element->element_print_latex(transporter, cout);
			cout << "$$" << endl;
		}


		Iso_type_as_double_triplet[i] = orbit_index;
	}


	Double_triplet_type_distribution = NEW_OBJECT(data_structures::tally);

	Double_triplet_type_distribution->init(Iso_type_as_double_triplet, 120, false, 0);
	data_structures::sorting Sorting;

	Double_triplet_types = Double_triplet_type_distribution->get_set_partition_and_types(
			Double_triplet_type_values,
			nb_double_triplet_types, 0 /*verbose_level*/);

	for (i = 0; i < nb_double_triplet_types; i++) {
		Sorting.lint_vec_heapsort(Double_triplet_types->Sets[i], Double_triplet_types->Set_size[i]);
	}



	if (f_v) {
		cout << "trihedral_pair_with_action::compute_iso_types_as_double_triplets done" << endl;
	}
}

void trihedral_pair_with_action::print_FG(
		std::ostream &ost)
{
	l1_interfaces::latex_interface L;

	ost << "$F$-planes:\\\\";
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
			F_plane, 3, 4, true /* f_tex*/);
	ost << "$$" << endl;
	ost << "$G$-planes:\\\\";
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
			G_plane, 3, 4, true /* f_tex*/);
	ost << "$$" << endl;
}




void trihedral_pair_with_action::print_equations()
{

	cout << "lambda = " << lambda << endl;
	cout << "lambda_rk = " << lambda_rk << endl;
	cout << "The six plane equations:" << endl;
	Int_matrix_print(The_six_plane_equations, 6, 4);
	cout << endl;
	cout << "The q+1 surface equations in the pencil:" << endl;
	Int_matrix_print(The_surface_equations, AL->q + 1, 20);
	cout << endl;

	cout << "The surface equation corresponding to "
			"lambda = " << lambda << " which is equation "
			"number " << lambda_rk << ":" << endl;
	Int_vec_print(cout, The_surface_equations + lambda_rk * 20, 20);
	cout << endl;
	cout << "the_equation:" << endl;
	Int_vec_print(cout, AL->the_equation, 20);
	cout << endl;
}


#if 0
void trihedral_pair_with_action::print_isomorphism_types_of_trihedral_pairs(
	ostream &ost,
	vector_ge *cosets)
{
	int i, j;
	long int planes6[6];
	int orbit_index0;
	int orbit_index;
	int list[120];
	int list_sz = 0;
	int Tt[17];
	int Iso[120];
	latex_interface L;

	cout << "trihedral_pair_with_action::print_isomorphism_types_of_trihedral_pairs" << endl;

	ost << "\\bigskip" << endl;
	ost << "" << endl;
	ost << "\\section*{Computing the Automorphism Group}" << endl;
	ost << "" << endl;

	ost << "The equation of the surface is: $" << endl;
	int_vec_print(ost, AL->the_equation, 20);
	ost << "$\\\\" << endl;



	ost << "\\bigskip" << endl;
	ost << "" << endl;
	ost << "\\subsection*{Computing the Automorphism "
			"Group, Step 1}" << endl;
	ost << "" << endl;




	AL->Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
			AL->Web->Dual_point_ranks + AL->Web->t_idx0 * 6,
		transporter0,
		orbit_index0,
		0 /*verbose_level*/);
	ost << "Trihedral pair $T_{" << AL->Web->t_idx0 << "}$ lies in orbit "
			<< orbit_index0 << "\\\\" << endl;
	ost << "An isomorphism is given by" << endl;
	ost << "$$" << endl;
	AL->Surf_A->A->element_print_latex(transporter0, ost);
	ost << "$$" << endl;




	for (i = 0; i < AL->Web->nb_T; i++) {

			cout << "testing if trihedral pair " << i << " / "
					<< AL->Web->nb_T << " = " << AL->Web->T_idx[i];
			cout << " lies in the orbit:" << endl;

		lint_vec_copy(AL->Web->Dual_point_ranks + i * 6, planes6, 6);
		AL->Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
			planes6,
			transporter,
			orbit_index,
			0 /*verbose_level */);

		ost << "Trihedral pair " << i << " lies in orbit "
				<< orbit_index << "\\\\" << endl;
		ost << "An isomorphism is given by" << endl;
		ost << "$$" << endl;
		AL->Surf_A->A->element_print_latex(transporter, ost);
		ost << "$$" << endl;


		Iso[i] = orbit_index;

		if (orbit_index != orbit_index0) {
			continue;
		}

		list[list_sz++] = i;

		AL->Surf_A->A->element_invert(transporter, Elt1, 0);
		AL->Surf_A->A->element_mult(transporter0, Elt1, Elt2, 0);

		ost << "An isomorphism between $T_{" << i << "}$ and $T_{"
				<< AL->Web->t_idx0 << "}$ is given by" << endl;
		ost << "$$" << endl;
		AL->Surf_A->A->element_print_latex(Elt2, ost);
		ost << "$$" << endl;


	} // next i

	ost << "The isomorphism types of the trihedral pairs "
			"in the list of double triplets are:" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
			Iso + 0 * 1, 40, 1, 0, 0, true /* f_tex */);
	ost << "\\quad" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
			Iso + 40 * 1, 40, 1, 40, 0, true /* f_tex */);
	ost << "\\quad" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
			Iso + 80 * 1, 40, 1, 80, 0, true /* f_tex */);
	ost << "$$" << endl;

	int I, h, iso;
	ost << "The isomorphism types of the trihedral pairs in the "
			"list of double triplets are:" << endl;
	for (I = 0; I < 12; I++) {
		ost << "$$" << endl;
		ost << "\\begin{array}{c|c|c|c|c|c|}" << endl;
		ost << "i & T_i & \\mbox{trihedral pair} & \\mbox{double "
				"triplet} & \\mbox{iso} & \\mbox{map}\\\\" << endl;
		ost << "\\hline" << endl;
		for (h = 0; h < 10; h++) {
			i = I * 10 + h;
			ost << i << " & T_{" << AL->Surf_A->Surf->Trihedral_pair_labels[i]
				<< "} & ";

			lint_vec_copy(AL->Web->Dual_point_ranks + i * 6, planes6, 6);
			AL->Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
				planes6,
				transporter,
				orbit_index,
				0 /*verbose_level */);


			ost << "\\{";
			for (j = 0; j < 3; j++) {
				ost << planes6[j];
				if (j < 3 - 1) {
					ost << ", ";
				}
			}
			ost << "; ";
			for (j = 0; j < 3; j++) {
				ost << planes6[3 + j];
				if (j < 3 - 1) {
					ost << ", ";
				}
			}
			ost << "\\}";

			iso = Iso[i];
			lint_vec_copy(
					AL->Surf_A->Classify_trihedral_pairs->Trihedral_pairs->Rep +
					iso * AL->Surf_A->Classify_trihedral_pairs->Trihedral_pairs->representation_sz,
				planes6, 6);

			ost << " & ";
			ost << "\\{";
			for (j = 0; j < 3; j++) {
				ost << planes6[j];
				if (j < 3 - 1) {
					ost << ", ";
					}
				}
			ost << "; ";
			for (j = 0; j < 3; j++) {
				ost << planes6[3 + j];
				if (j < 3 - 1) {
					ost << ", ";
				}
			}
			ost << "\\}";
			ost << " & " << iso << " & " << endl;
			AL->Surf_A->A->element_print_latex(transporter, ost);
			ost << "\\\\[4pt]" << endl;
			ost << "\\hline" << endl;
		}
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
	}

	ost << "There are " << list_sz << " trihedral pairs which "
			"are isomorphic to the double triplet of $T_0$:\\\\" << endl;
	L.int_set_print_tex(ost, list, list_sz);
	ost << "$$" << endl;
	ost << "\\{ ";
	for (i = 0; i < list_sz; i++) {
		ost << "T_{" << list[i] << "}";
		if (i < list_sz - 1) {
			ost << ", ";
		}
	}
	ost << " \\}";
	ost << "$$" << endl;


	ost << "\\bigskip" << endl;
	ost << "" << endl;
	ost << "\\subsection*{Computing the Automorphism Group, Step 2}" << endl;
	ost << endl;




	ost << "We are now looping over the " << list_sz
			<< " trihedral pairs which are isomorphic to the "
			"double triplet of $T_0$ and over the "
			<< cosets->len << " cosets:\\\\" << endl;
	for (i = 0; i < list_sz; i++) {
		ost << "i=" << i << " / " << list_sz
				<< " considering $T_{" << list[i] << "}$:\\\\";

		lint_vec_copy(AL->Web->Dual_point_ranks + list[i] * 6, planes6, 6);


		AL->Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
			planes6,
			transporter,
			orbit_index,
			0 /*verbose_level */);

		AL->Surf_A->A->element_invert(transporter, Elt1, 0);
		AL->Surf_A->A->element_mult(transporter0, Elt1, Elt2, 0);

		ost << "The isomorphism from $T_0$ to $T_{"
				<<  list[i] << "}$ is :" << endl;
		ost << "$$" << endl;
		AL->Surf_A->A->element_print_latex(Elt2, ost);
		ost << " = " << endl;
		AL->Surf_A->A->element_print_latex(transporter0, ost);
		ost << " \\cdot " << endl;
		ost << "\\left(" << endl;
		AL->Surf_A->A->element_print_latex(transporter, ost);
		ost << "\\right)^{-1}" << endl;
		ost << "$$" << endl;

		for (j = 0; j < cosets->len; j++) {
			ost << "i=" << i << " / " << list_sz << " j=" << j
					<< " / " << cosets->len << " considering "
					"coset given by:" << endl;
			ost << "$$" << endl;
			AL->Surf_A->A->element_print_latex(cosets->ith(j), ost);
			ost << "$$" << endl;



			matrix_group *mtx;

			mtx = AL->Surf_A->A->G.matrix_grp;

			// ToDo:
			AL->F->transpose_matrix(Elt2, Tt, 4, 4);
			if (mtx->f_semilinear) {
				// if we are doing semilinear:
				Tt[4 * 4] = Elt2[4 * 4];
			}


			AL->Surf_A->A->make_element(Elt3, Tt, 0);
			AL->Surf_A->A->element_invert(cosets->ith(j), Elt5, 0);
			AL->Surf_A->A->element_mult(Elt3, Elt5, Elt4, 0);

			//cout << "transporter transposed:" << endl;
			//A->print_quick(cout, Elt2);

			int coeff_out[20];


			//Surf_A->A->element_invert(Elt4, Elt5, 0);

			ost << "i=" << i << " / " << list_sz << " j=" << j
				<< " / " << cosets->len << " testing element:" << endl;
			ost << "$$" << endl;
			AL->Surf_A->A->element_print_latex(Elt4, ost);
			ost << " = " << endl;
			AL->Surf_A->A->element_print_latex(Elt3, ost);
			ost << " \\cdot " << endl;
			AL->Surf_A->A->element_print_latex(Elt5, ost);
			ost << "$$" << endl;


			//matrix_group *M;

			//M = A->G.matrix_grp;
			mtx->substitute_surface_equation(Elt4,
					AL->the_equation, coeff_out, AL->Surf,
					0 /*verbose_level - 1*/);


			AL->F->PG_element_normalize(coeff_out, 1, 20);

			ost << "The transformed equation is: $" << endl;
			int_vec_print(ost, coeff_out, 20);
			ost << "$\\\\" << endl;


			if (int_vec_compare(coeff_out, AL->the_equation, 20) == 0) {
				ost << "trihedral pair " << i << " / " << AL->Web->nb_T
					<< ", coset " << j << " / " << cosets->len
					<< " gives an automorphism\\\\" << endl;
				ost << "automorphism = " << endl;
				ost << "$$" << endl;
				AL->Surf_A->A->element_print_latex(Elt4, ost);
				ost << "$$" << endl;

			}
			else {
				ost << "The equation is different, the group "
						"element is not an automorphism\\\\" << endl;
			}

		} // next j
	} // next i


}
#endif

void trihedral_pair_with_action::report(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "trihedral_pair_with_action::report" << endl;
	}



	cout << "trihedral_pair_with_action::print before "
			"print_the_six_plane_equations" << endl;
	AL->Web->print_the_six_plane_equations(
			The_six_plane_equations, plane6_by_dual_ranks, ost);

	cout << "trihedral_pair_with_action::print before "
			"print_surface_equations_on_line" << endl;
	AL->Web->print_surface_equations_on_line(The_surface_equations,
		lambda, lambda_rk, ost);

	int *coeffs;
	int coeffs2[20];

	coeffs = The_surface_equations + lambda_rk * 20;
	Int_vec_copy(coeffs, coeffs2, 20);
	AL->F->Projective_space_basic->PG_element_normalize_from_front(
			coeffs2, 1, 20);

	ost << "\\bigskip" << endl;
	ost << "The normalized equation of the surface is:" << endl;
	ost << "$$" << endl;
	AL->Surf->print_equation_tex(ost, coeffs2);
	ost << "$$" << endl;
	ost << "The equation in coded form: $";
	for (i = 0; i < 20; i++) {
		if (coeffs2[i]) {
			ost << coeffs2[i] << ", " << i << ", ";
		}
	}
	ost << "$\\\\" << endl;

	//cout << "do_arc_lifting before arc_lifting->
	//print_trihedral_pairs" << endl;
	//AL->print_trihedral_pairs(fp);



	ost << "\\bigskip" << endl;

	report_iso_type_as_double_triplets(ost);


	ost << "\\bigskip" << endl;
	ost << "The trihedral pair is isomorphic to double triplet no "
			<< trihedral_pair_orbit_index << " in the classification."
			<< endl;
	ost << endl;
	ost << "\\bigskip" << endl;
	ost << endl;
	ost << "The stabilizer of the trihedral pair is a group of order "
			<< stabilizer_of_trihedral_pair_go << endl;
	ost << endl;

	ost << "The stabilizer of the trihedral pair is the following group:\\\\" << endl;
	stab_gens_trihedral_pair->print_generators_tex(ost);

	ost << "The orbits of the stabilizer of the trihedral pair on the $q+1$ "
			"surfaces on the line are:\\\\" << endl;
#if 0
	Orb->print_fancy(
			ost, true, AL->Surf_A->A, stab_gens_trihedral_pair);
#else
	ost << "The stabilizer of the trihedral pair has " << Orb->nb_orbits << " orbits on the pencil of surfaces.\\\\" << endl;
#endif


	ost << "The subgroup which stabilizes "
			"the equation has index " << cosets->len
			<< " in the stabilizer of "
			"the trihedral pair. Coset representatives are:\\\\" << endl;
	for (i = 0; i < cosets->len; i++) {
		ost << "Coset " << i << " / " << cosets->len
			<< ", coset rep:" << endl;
		ost << "$$" << endl;
		AL->Surf_A->A->Group_element->element_print_latex(cosets->ith(i), ost);
		ost << "$$" << endl;
		}
	ost << "The stabilizer of the trihedral pair and the equation is "
			"the following group\\\\" << endl;
	gens_subgroup->print_generators_tex(ost);

	ost << "The automorphism group consists of "
			<< coset_reps->len << " cosets of the subgroup.\\\\" << endl;
#if 0
	for (i = 0; i < coset_reps->len; i++) {
		ost << "Aut coset " << i << " / " << coset_reps->len
			<< ", trihedral pair " << aut_T_index[i]
			<< ", subgroup coset " <<  aut_coset_index[i]
			<< ", coset rep:" << endl;
		ost << "$$" << endl;
		AL->Surf_A->A->element_print_latex(coset_reps->ith(i), ost);
		ost << "$$" << endl;
	}
#endif


	ring_theory::longinteger_object go;

	Aut_gens->group_order(go);
	ost << "The automorphism group of the surface has order "
			<< go << "\\\\" << endl;
	Aut_gens->print_generators_tex(ost);


	//print_isomorphism_types_of_trihedral_pairs(ost, cosets);

	if (f_v) {
		cout << "trihedral_pair_with_action::report done" << endl;
	}
}

void trihedral_pair_with_action::report_iso_type_as_double_triplets(
		std::ostream &ost)
{
	l1_interfaces::latex_interface L;
	int i;

	ost << "The isomorphism types of the trihedral pairs "
			"in the list of double triplets are:" << endl;
	ost << "$$" << endl;
	for (i = 0; i < 6; i++) {
		L.print_integer_matrix_with_standard_labels_and_offset(ost,
				Iso_type_as_double_triplet + i * 20, 20, 1, i * 20, 0, true /* f_tex */);
		if (i < 6 - 1) {
			ost << "\\quad" << endl;
		}
	}
	ost << "$$" << endl;


	ost << "Distribution of isomorphism types:\\\\" << endl;
	ost << "$$" << endl;
	Double_triplet_type_distribution->print_bare_tex(ost, false/* f_backwards*/);
	ost << "$$" << endl;


	for (i = 0; i < nb_double_triplet_types; i++) {
		ost << "type value " << Double_triplet_type_values[i]
				<< " appears " << Double_triplet_types->Set_size[i]
				<< " times for these trihedral pairs: ";
		Lint_vec_print(ost, Double_triplet_types->Sets[i], Double_triplet_types->Set_size[i]);
		ost << "\\\\" << endl;
	}

}

}}}}

