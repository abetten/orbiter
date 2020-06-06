// arc_lifting.cpp
// 
// Anton Betten, Fatma Karaoglu
//
// January 24, 2017
// moved here from clebsch.cpp: March 22, 2017
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


arc_lifting::arc_lifting()
{
	q = 0;
	F = NULL;
	Surf = NULL;
	Surf_A = NULL;


	the_equation = NULL;

	Web = NULL;


	The_surface_equations = NULL;

	stab_gens_trihedral_pair = NULL;
	gens_subgroup = NULL;
	A_on_equations = NULL;
	Orb = NULL;
	cosets = NULL;
	coset_reps = NULL;
	aut_T_index = NULL;
	aut_coset_index = NULL;
	Aut_gens =NULL;
	
	System = NULL;
	transporter0 = NULL;
	transporter = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
	Elt5 = NULL;
	null();
}

arc_lifting::~arc_lifting()
{
	freeself();
}

void arc_lifting::null()
{
}

void arc_lifting::freeself()
{
	if (the_equation) {
		FREE_int(the_equation);
	}
	if (Web) {
		FREE_OBJECT(Web);
	}

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
	
	null();
}


void arc_lifting::create_surface_and_group(surface_with_action *Surf_A,
	long int *Arc6, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q;
	surface_domain *Surf;

	if (f_v) {
		cout << "arc_lifting::create_surface_and_group" << endl;
	}



	arc_lifting::arc = Arc6;
	arc_lifting::arc_size = 6;
	arc_lifting::Surf_A = Surf_A;
	Surf = Surf_A->Surf;
	F = Surf->F;
	q = F->q;



	if (f_v) {
		cout << "arc_lifting::create_surface_and_group "
				"before create_web_of_cubic_curves" << endl;
	}
	create_web_of_cubic_curves(verbose_level - 2);
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group "
				"after create_web_of_cubic_curves" << endl;
	}




	The_surface_equations = NEW_int((q + 1) * 20);
	
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group before "
				"create_surface_from_trihedral_pair_and_arc"
				<< endl;
	}
	create_surface_from_trihedral_pair_and_arc(
		Web->t_idx0, planes6,
		The_six_plane_equations,
		The_surface_equations,
		lambda, lambda_rk,
		verbose_level);
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group after "
				"create_surface_from_trihedral_pair_and_arc"
				<< endl;
	}

	if (f_v) {
		print_equations();
	}

	
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group "
				"before create_clebsch_system" << endl;
	}
	create_clebsch_system(
		The_six_plane_equations, 
		lambda, 
		0 /* verbose_level */);
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group "
				"after create_clebsch_system" << endl;
	}



	if (f_v) {
		cout << "arc_lifting::create_surface_and_group before "
				"create_stabilizer_of_trihedral_pair" << endl;
	}
	stab_gens_trihedral_pair = create_stabilizer_of_trihedral_pair(
		planes6, 
		trihedral_pair_orbit_index, 
		verbose_level - 2);
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group after "
				"create_stabilizer_of_trihedral_pair" << endl;
	}

	stab_gens_trihedral_pair->group_order(stabilizer_of_trihedral_pair_go);
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group the stabilizer of "
				"the trihedral pair has order "
				<< stabilizer_of_trihedral_pair_go << endl;
	}



	if (f_v) {
		cout << "arc_lifting::create_surface_and_group before "
				"create_action_on_equations_and_compute_orbits" << endl;
		}
	create_action_on_equations_and_compute_orbits(
		The_surface_equations, 
		stab_gens_trihedral_pair
		/* strong_generators *gens_for_stabilizer_of_trihedral_pair */,
		A_on_equations, Orb, 
		verbose_level - 2);
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group after "
				"create_action_on_equations_and_compute_orbits" << endl;
	}

	
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group the orbits "
				"on the pencil of surfaces are:" << endl;
	}
	Orb->print_and_list_orbits(cout);



	//Surf_A->A->group_order(go_PGL);


	if (f_v) {
		cout << "arc_lifting::create_surface_and_group before "
				"Orb->stabilizer_any_point_plus_cosets" << endl;
	}
	gens_subgroup = 
		Orb->stabilizer_any_point_plus_cosets(
			Surf_A->A, 
			stabilizer_of_trihedral_pair_go, 
			lambda_rk /* pt */, 
			cosets, 
			verbose_level - 2);

	if (f_v) {
		cout << "arc_lifting::create_surface_and_group after "
				"Orb->stabilizer_any_point_plus_cosets" << endl;
	}

	if (f_v) {
		cout << "arc_lifting::create_surface_and_group we found the "
				"following coset representatives:" << endl;
		cosets->print(cout);
	}




	if (f_v) {
		cout << "arc_lifting::create_surface_and_group after "
				"Orb->stabilizer_any_point" << endl;
	}
	gens_subgroup->group_order(stab_order);
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group "
				"The stabilizer of the trihedral pair inside "
				"the group of the surface has order "
				<< stab_order << endl;
	}

	if (f_v) {
		cout << "arc_lifting::create_surface_and_group elements "
				"in the stabilizer:" << endl;
		gens_subgroup->print_elements_ost(cout);
	}

	if (f_v) {
		cout << "arc_lifting::create_surface_and_group The stabilizer of "
				"the trihedral pair inside the stabilizer of the "
				"surface is generated by:" << endl;
		gens_subgroup->print_generators_tex(cout);
	}






	Surf->compute_nine_lines_by_dual_point_ranks(
		Web->Dual_point_ranks + 0 * 6,
		Web->Dual_point_ranks + 0 * 6 + 3,
		nine_lines, 
		0 /* verbose_level */);

	if (f_v) {
		cout << "arc_lifting::create_surface_and_group before "
				"loop_over_trihedral_pairs" << endl;
	}
	loop_over_trihedral_pairs(cosets, 
		coset_reps, 
		aut_T_index, 
		aut_coset_index, 
		verbose_level);
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group after "
				"loop_over_trihedral_pairs" << endl;
		cout << "arc_lifting::create_surface we found an "
				"orbit of length " << coset_reps->len << endl;
	}
	

	

	{
		longinteger_object ago;

		if (f_v) {
			cout << "arc_lifting::create_surface_and_group "
					"Extending the group:" << endl;
		}
		Aut_gens = NEW_OBJECT(strong_generators);
		Aut_gens->init_group_extension(gens_subgroup,
			coset_reps, coset_reps->len, verbose_level - 3);
	
		Aut_gens->group_order(ago);
		if (f_v) {
			cout << "arc_lifting::create_surface_and_group "
					"The automorphism group has order " << ago << endl;
			cout << "arc_lifting::create_surface_and_group "
					"The automorphism group is:" << endl;
			Aut_gens->print_generators_tex(cout);
		}
	}
	
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group done" << endl;
	}
}


void arc_lifting::create_web_of_cubic_curves(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "arc_lifting::create_web_of_cubic_curves" << endl;
	}


	the_equation = NEW_int(20);

	transporter0 = NEW_int(Surf_A->A->elt_size_in_int);
	transporter = NEW_int(Surf_A->A->elt_size_in_int);
	Elt1 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt2 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt3 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt4 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt5 = NEW_int(Surf_A->A->elt_size_in_int);

	

	Web = NEW_OBJECT(web_of_cubic_curves);

	Web->init(Surf, arc, verbose_level);





	if (f_v) {
		cout << "arc_lifting::create_web_of_cubic_curves done" << endl;
	}
}


void arc_lifting::loop_over_trihedral_pairs(
	vector_ge *cosets, vector_ge *&coset_reps,
	int *&aut_T_index, int *&aut_coset_index,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int planes6[6];
	int orbit_index0;
	int orbit_index;
	int orbit_length;
	long int Nine_lines0[9];
	long int Nine_lines[9];
	long int *v;
	int sz;
	sorting Sorting;

	if (f_v) {
		cout << "arc_lifting::loop_over_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "arc_lifting::loop_over_trihedral_pairs "
				"we are considering " << cosets->len
				<< " cosets from the downstep" << endl;
	}


	orbit_length = 0;


	Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
			Web->Dual_point_ranks + Web->t_idx0 * 6,
		transporter0, 
		orbit_index0, 
		0 /*verbose_level*/);
	if (f_v) {
		cout << "arc_lifting::loop_over_trihedral_pairs "
				"Trihedral pair " << Web->t_idx0
				<< " lies in orbit " << orbit_index0 << endl;
	}
	
	Surf->compute_nine_lines_by_dual_point_ranks(
			Web->Dual_point_ranks + Web->t_idx0 * 6,
			Web->Dual_point_ranks + Web->t_idx0 * 6 + 3,
		Nine_lines0, 
		0 /* verbose_level */);

	if (FALSE) {
		cout << "arc_lifting::loop_over_trihedral_pairs "
				"The first trihedral pair gives "
				"the following nine lines: ";
		lint_vec_print(cout, Nine_lines0, 9);
		cout << endl;
	}

	coset_reps = NEW_OBJECT(vector_ge);
	coset_reps->init(Surf_A->A, verbose_level - 2);
	coset_reps->allocate(Web->nb_T * cosets->len, verbose_level - 2);

	aut_T_index = NEW_int(Web->nb_T * cosets->len);
	aut_coset_index = NEW_int(Web->nb_T * cosets->len);

	for (i = 0; i < Web->nb_T; i++) {

		if (f_v) {
			cout << "arc_lifting::loop_over_trihedral_pairs "
					"testing if trihedral pair "
					<< i << " / " << Web->nb_T << " = " << Web->T_idx[i];
			cout << " lies in the orbit:" << endl;
		}

		lint_vec_copy(Web->Dual_point_ranks + i * 6, planes6, 6);

		Surf->compute_nine_lines_by_dual_point_ranks(
			planes6, 
			planes6 + 3, 
			Nine_lines, 
			0 /* verbose_level */);

		if (FALSE) {
			cout << "arc_lifting::loop_over_trihedral_pairs "
					"The " << i << "-th trihedral "
					"pair gives the following nine lines: ";
			lint_vec_print(cout, Nine_lines, 9);
			cout << endl;
		}

		Sorting.vec_intersect(Nine_lines0, 9, Nine_lines, 9, v, sz);

		if (FALSE) {
			cout << "arc_lifting::loop_over_trihedral_pairs "
					"The nine lines of the " << i
					<< "-th trihedral pair intersect the "
						"nine lines of the first in " << sz
						<< " lines, which are: ";
			lint_vec_print(cout, v, sz);
			cout << endl;
		}

		if (FALSE) {
			Surf->print_trihedral_pair_in_dual_coordinates_in_GAP(
				planes6, planes6 + 3);
			cout << endl;
		}

		FREE_lint(v);
		


		Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
			planes6, 
			transporter, 
			orbit_index, 
			0 /*verbose_level */);


		if (orbit_index != orbit_index0) {
			if (f_v) {
				cout << "arc_lifting::loop_over_trihedral_pairs "
						"trihedral pair " << i << " / " << Web->nb_T
						<< " lies in orbit " << orbit_index
						<< " and so $T_{" << Web->t_idx0
						<< "}$ and T_i are not isomorphic" << endl;
			}
			continue;
		}
		if (f_v) {
			cout << "arc_lifting::loop_over_trihedral_pairs "
					"trihedral pair " << i << " / " << Web->nb_T
					<< " lies in orbit " << orbit_index
					<< " and so $T_{" << Web->t_idx0
					<< "}$ and T_i are isomorphic" << endl;
		}
		

		Surf_A->A->element_invert(transporter, Elt1, 0);
		Surf_A->A->element_mult(transporter0, Elt1, Elt2, 0);
		if (f_v) {
			cout << "Elt2:" << endl;
			Surf_A->A->element_print_quick(Elt2, cout);
		}

		for (j = 0; j < cosets->len; j++) {

			if (f_v) {
				cout << "arc_lifting::loop_over_trihedral_pairs "
						"testing coset j=" << j << " / "
						<< cosets->len << endl;
			}

			// contragredient action:

			Surf_A->A->element_transpose(Elt2, Elt3, 0 /* verbose_level*/);



			Surf_A->A->element_invert(cosets->ith(j), Elt5, 0);
			Surf_A->A->element_mult(Elt3, Elt5, Elt4, 0);
	
			//cout << "transporter transposed:" << endl;
			//A->print_quick(cout, Elt2);

			int coeff_out[20];


			//Surf_A->A->element_invert(Elt4, Elt5, 0);

			if (f_v) {
				cout << "arc_lifting::loop_over_trihedral_pairs "
						"Elt4:" << endl;
				Surf_A->A->element_print_quick(Elt4, cout);
			}



			matrix_group *M;

			if (f_v) {
				cout << "arc_lifting::loop_over_trihedral_pairs "
						"before M->substitute_surface_equation" << endl;
			}

			M = Surf_A->A->G.matrix_grp;

			M->substitute_surface_equation(Elt4,
					the_equation, coeff_out, Surf,
					verbose_level - 1);




			F->PG_element_normalize(coeff_out, 1, 20);

			if (f_v) {
				cout << "arc_lifting::loop_over_trihedral_pairs "
						"The transformed equation is:" << endl;
				int_vec_print(cout, coeff_out, 20);
				cout << endl;
			}


			if (int_vec_compare(coeff_out, the_equation, 20) == 0) {
				if (f_v) {
					cout << "arc_lifting::loop_over_trihedral_pairs "
							"trihedral pair " << i << " / " << Web->nb_T
							<< ", coset " << j << " / " << cosets->len
							<< " gives automorphism, n e w "
							"orbit length is " << orbit_length + 1 << endl;
					cout << "coset rep = " << endl;
					Surf_A->A->element_print_quick(Elt3, cout);
				}
				Surf->compute_nine_lines_by_dual_point_ranks(
					planes6, planes6 + 3, 
					Nine_lines, 
					0 /* verbose_level */);


				if (FALSE) {
					cout << "arc_lifting::loop_over_trihedral_pairs "
							"The " << orbit_length + 1 << "-th "
							"trihedral pair in the orbit gives "
							"the following nine lines: ";
					lint_vec_print(cout, Nine_lines, 9);
					cout << endl;
				}


				Surf_A->A->element_move(Elt4, 
					coset_reps->ith(orbit_length), 0);

				aut_T_index[orbit_length] = i;
				aut_coset_index[orbit_length] = j;
				orbit_length++;
			}
			else {
				if (f_v) {
					cout << "arc_lifting::loop_over_trihedral_pairs "
							"trihedral pair " << i << " / " << Web->nb_T
							<< " coset " << j << " / " << cosets->len
							<< " does not lie in the orbit" << endl;
				}
				//exit(1);
			}
		} // next j

	} // next i

	coset_reps->reallocate(orbit_length, verbose_level - 2);

	if (f_v) {
		cout << "arc_lifting::loop_over_trihedral_pairs we found an "
				"orbit of trihedral pairs of length "
				<< orbit_length << endl;
		//cout << "coset reps:" << endl;
		//coset_reps->print_tex(cout);
	}

	if (f_v) {
		cout << "arc_lifting::loop_over_trihedral_pairs done" << endl;
	}
}



void arc_lifting::create_the_six_plane_equations(
		int t_idx, int *The_six_plane_equations, long int *plane6,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "arc_lifting::create_the_six_plane_equations "
				"t_idx=" << t_idx << endl;
	}


	int_vec_copy(Surf->Trihedral_to_Eckardt + t_idx * 6,
			Web->row_col_Eckardt_points, 6);

	for (i = 0; i < 6; i++) {
		int_vec_copy(Web->Tritangent_plane_equations + Web->row_col_Eckardt_points[i] * 4,
				The_six_plane_equations + i * 4, 4);
	}

	if (f_v) {
		cout << "arc_lifting::create_the_six_plane_equations" << endl;
		cout << "The_six_plane_equations=" << endl;
		int_matrix_print(The_six_plane_equations, 6, 4);
	}

	for (i = 0; i < 6; i++) {
		plane6[i] = Surf->P->rank_point(The_six_plane_equations + i * 4);
	}

	if (f_v) {
		cout << "arc_lifting::create_the_six_plane_equations done" << endl;
	}
}

void arc_lifting::create_surface_from_trihedral_pair_and_arc(
	int t_idx, long int *planes6,
	int *The_six_plane_equations, 
	int *The_surface_equations, 
	int &lambda, int &lambda_rk, 
	int verbose_level)
// plane6[6]
// The_six_plane_equations[6 * 4]
// The_surface_equations[(q + 1) * 20]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_lifting::create_surface_from_trihedral_pair_and_arc t_idx=" << t_idx << endl;
	}

	create_the_six_plane_equations(t_idx, 
		The_six_plane_equations, planes6, 
		verbose_level);


	if (f_v) {
		cout << "arc_lifting::create_surface_from_trihedral_pair_and_arc "
				"before create_equations_for_pencil_of_surfaces_from_trihedral_pair" << endl;
	}
	Surf->create_equations_for_pencil_of_surfaces_from_trihedral_pair(
		The_six_plane_equations, The_surface_equations, 
		verbose_level);

	if (f_v) {
		cout << "arc_lifting::create_surface_from_trihedral_pair_and_arc "
				"before create_lambda_from_trihedral_pair_and_arc" << endl;
	}
	Surf->create_lambda_from_trihedral_pair_and_arc(arc, 
		Web->Web_of_cubic_curves,
		Web->Tritangent_plane_equations, t_idx, lambda, lambda_rk,
		verbose_level);


	int_vec_copy(The_surface_equations + lambda_rk * 20,
			the_equation, 20);

	if (f_v) {
		cout << "arc_lifting::create_surface_from_trihedral_pair_and_arc done" << endl;
	}
}

strong_generators *arc_lifting::create_stabilizer_of_trihedral_pair(
	long int *planes6,
	int &trihedral_pair_orbit_index, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	strong_generators *gens_dual;
	strong_generators *gens;
	longinteger_object go;

	gens = NEW_OBJECT(strong_generators);


	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair" << endl;
	}

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair "
				"before Surf_A->identify_trihedral_pair_and_get_stabilizer" << endl;
	}

	gens_dual =
		Surf_A->Classify_trihedral_pairs->
			identify_trihedral_pair_and_get_stabilizer(
		planes6, transporter, trihedral_pair_orbit_index, 
		verbose_level);

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair "
				"after Surf_A->identify_trihedral_pair_and_get_stabilizer" << endl;
	}
	gens_dual->group_order(go);

	
	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair "
				"trihedral_pair_orbit_index="
				<< trihedral_pair_orbit_index
				<< " group order = " << go << endl;
	}

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair "
				"group generators:" << endl;
		gens_dual->print_generators_tex(cout);
		//gens_dual->print_elements_ost(cout);
	}


	gens->init(Surf_A->A);

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair "
				"before gens->init_transposed_group" << endl;
	}
	gens->init_transposed_group(gens_dual, verbose_level);

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair "
				"The transposed stabilizer is generated by:" << endl;
		gens->print_generators_tex(cout);
		//gens->print_generators_tex(cout);
	}

	FREE_OBJECT(gens_dual);

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair "
				"done" << endl;
	}
	return gens;
}

void arc_lifting::create_action_on_equations_and_compute_orbits(
	int *The_surface_equations, 
	strong_generators *gens_for_stabilizer_of_trihedral_pair, 
	action *&A_on_equations, schreier *&Orb, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_lifting::create_action_on_equations_and_compute_orbits" << endl;
	}
	
	if (f_v) {
		cout << "arc_lifting::create_action_on_equations_and_compute_orbits "
				"before orbits_on_equations" << endl;
	}

	Surf_A->A->orbits_on_equations(
		Surf->Poly3_4, 
		The_surface_equations, 
		q + 1 /* nb_equations */, 
		gens_for_stabilizer_of_trihedral_pair, 
		A_on_equations, 
		Orb, 
		verbose_level);

	if (f_v) {
		cout << "arc_lifting::create_action_on_equations_and_compute_orbits done" << endl;
	}
}

void arc_lifting::create_clebsch_system(
	int *The_six_plane_equations,
	int lambda, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "arc_lifting::create_clebsch_system" << endl;
		}
	
	int_vec_copy(The_six_plane_equations, F_plane, 12);
	int_vec_copy(The_six_plane_equations + 12, G_plane, 12);
	cout << "F_planes:" << endl;
	int_matrix_print(F_plane, 3, 4);
	cout << "G_planes:" << endl;
	int_matrix_print(G_plane, 3, 4);

	Surf->compute_nine_lines(F_plane, G_plane,
			nine_lines, 0 /* verbose_level */);

	if (f_v) {
		cout << "arc_lifting::create_clebsch_system" << endl;
		cout << "The nine lines are: ";
		lint_vec_print(cout, nine_lines, 9);
		cout << endl;
	}

	Surf->prepare_system_from_FG(F_plane, G_plane, 
		lambda, System, verbose_level);

	cout << "arc_lifting::create_clebsch_system "
			"The System:" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			int *p = System + (i * 4 + j) * 3;
			Surf->Poly1->print_equation(cout, p);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "arc_lifting::create_clebsch_system done" << endl;
	}
}


void arc_lifting::report(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "arc_lifting::report" << endl;
	}


	Web->report(ost, verbose_level);

	cout << "arc_lifting::print before "
			"print_the_six_plane_equations" << endl;
	Web->print_the_six_plane_equations(
			The_six_plane_equations, planes6, ost);

	cout << "arc_lifting::print before "
			"print_surface_equations_on_line" << endl;
	Web->print_surface_equations_on_line(The_surface_equations,
		lambda, lambda_rk, ost);

	int *coeffs;
	int coeffs2[20];

	coeffs = The_surface_equations + lambda_rk * 20;
	int_vec_copy(coeffs, coeffs2, 20);
	F->PG_element_normalize_from_front(coeffs2, 1, 20);
	
	ost << "\\bigskip" << endl;
	ost << "The normalized equation of the surface is:" << endl;
	ost << "$$" << endl;
	Surf->print_equation_tex(ost, coeffs2);
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
	ost << "The trihedral pair is isomorphic to trihedral pair no "
			<< trihedral_pair_orbit_index << " in the classification."
			<< endl;
	ost << endl;
	ost << "\\bigskip" << endl;
	ost << endl;
	ost << "The stabilizer of the trihedral pair is a group of order "
			<< stabilizer_of_trihedral_pair_go << endl;
	ost << endl;

	ost << "The stabilizer of the trihedral pair "
		"is the following group\\\\" << endl;
	stab_gens_trihedral_pair->print_generators_tex(ost);

	ost << "The orbits of the trihedral pair stabilizer on the $q+1$ "
			"surfaces on the line are:\\\\" << endl;
	Orb->print_fancy(
			ost, TRUE, Surf_A->A, stab_gens_trihedral_pair);


	ost << "The subgroup which stabilizes "
			"the equation has " << cosets->len
			<< " cosets in the stabilizer of "
			"the trihedral pair:\\\\" << endl;
	for (i = 0; i < cosets->len; i++) {
		ost << "Coset " << i << " / " << cosets->len
			<< ", coset rep:" << endl;
		ost << "$$" << endl;
		Surf_A->A->element_print_latex(cosets->ith(i), ost);
		ost << "$$" << endl;
		}
	ost << "The stabilizer of the trihedral pair and the equation is "
			"the following group\\\\" << endl;
	gens_subgroup->print_generators_tex(ost);

	ost << "The automorphism group consists of the follwing "
			<< coset_reps->len << " cosets\\\\" << endl;
	for (i = 0; i < coset_reps->len; i++) {
		ost << "Aut coset " << i << " / " << coset_reps->len 
			<< ", trihedral pair " << aut_T_index[i] 
			<< ", subgroup coset " <<  aut_coset_index[i] 
			<< ", coset rep:" << endl;
		ost << "$$" << endl;
		Surf_A->A->element_print_latex(coset_reps->ith(i), ost);
		ost << "$$" << endl;
	}


	longinteger_object go;
	
	Aut_gens->group_order(go);
	ost << "The automorphism group of the surface has order "
			<< go << "\\\\" << endl;
	Aut_gens->print_generators_tex(ost);


	print_isomorphism_types_of_trihedral_pairs(ost, cosets);

	if (f_v) {
		cout << "arc_lifting::report done" << endl;
	}
}








void arc_lifting::print_FG(ostream &ost)
{
	latex_interface L;

	ost << "$F$-planes:\\\\";
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
			F_plane, 3, 4, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$G$-planes:\\\\";
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
			G_plane, 3, 4, TRUE /* f_tex*/);
	ost << "$$" << endl;
}




void arc_lifting::print_equations()
{

	cout << "lambda = " << lambda << endl;
	cout << "lambda_rk = " << lambda_rk << endl;
	cout << "The six plane equations:" << endl;
	int_matrix_print(The_six_plane_equations, 6, 4);
	cout << endl;
	cout << "The q+1 surface equations in the pencil:" << endl;
	int_matrix_print(The_surface_equations, q + 1, 20);
	cout << endl;

	cout << "The surface equation corresponding to "
			"lambda = " << lambda << " which is equation "
			"number " << lambda_rk << ":" << endl;
	int_vec_print(cout, The_surface_equations + lambda_rk * 20, 20);
	cout << endl;
	cout << "the_equation:" << endl;
	int_vec_print(cout, the_equation, 20);
	cout << endl;
}



void arc_lifting::print_isomorphism_types_of_trihedral_pairs(
	ostream &ost,
	vector_ge *cosets)
{
	int i, j;
	long int planes6[6];
	int orbit_index0;
	int orbit_index;
	int *transporter0;
	int *transporter;
	int list[120];
	int list_sz = 0;
	int Tt[17];
	int Iso[120];
	latex_interface L;

	cout << "arc_lifting::print_isomorphism_types_of_trihedral_pairs" << endl;

	ost << "\\bigskip" << endl;
	ost << "" << endl;
	ost << "\\section*{Computing the Automorphism Group}" << endl;
	ost << "" << endl;

	ost << "The equation of the surface is: $" << endl;
	int_vec_print(ost, the_equation, 20);
	ost << "$\\\\" << endl;



	ost << "\\bigskip" << endl;
	ost << "" << endl;
	ost << "\\subsection*{Computing the Automorphism "
			"Group, Step 1}" << endl;
	ost << "" << endl;



	transporter0 = NEW_int(Surf_A->A->elt_size_in_int);
	transporter = NEW_int(Surf_A->A->elt_size_in_int);

	Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
			Web->Dual_point_ranks + Web->t_idx0 * 6,
		transporter0, 
		orbit_index0, 
		0 /*verbose_level*/);
	ost << "Trihedral pair $T_{" << Web->t_idx0 << "}$ lies in orbit "
			<< orbit_index0 << "\\\\" << endl;
	ost << "An isomorphism is given by" << endl;
	ost << "$$" << endl;
	Surf_A->A->element_print_latex(transporter0, ost);
	ost << "$$" << endl;
	



	for (i = 0; i < Web->nb_T; i++) {

			cout << "testing if trihedral pair " << i << " / "
					<< Web->nb_T << " = " << Web->T_idx[i];
			cout << " lies in the orbit:" << endl;

		lint_vec_copy(Web->Dual_point_ranks + i * 6, planes6, 6);
		Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
			planes6, 
			transporter, 
			orbit_index, 
			0 /*verbose_level */);
		
		ost << "Trihedral pair " << i << " lies in orbit "
				<< orbit_index << "\\\\" << endl;
		ost << "An isomorphism is given by" << endl;
		ost << "$$" << endl;
		Surf_A->A->element_print_latex(transporter, ost);
		ost << "$$" << endl;


		Iso[i] = orbit_index;
		
		if (orbit_index != orbit_index0) {
			continue;
		}
		
		list[list_sz++] = i;

		Surf_A->A->element_invert(transporter, Elt1, 0);
		Surf_A->A->element_mult(transporter0, Elt1, Elt2, 0);
		
		ost << "An isomorphism between $T_{" << i << "}$ and $T_{"
				<< Web->t_idx0 << "}$ is given by" << endl;
		ost << "$$" << endl;
		Surf_A->A->element_print_latex(Elt2, ost);
		ost << "$$" << endl;


	} // next i

	ost << "The isomorphism types of the trihedral pairs "
			"in the list of double triplets are:" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
			Iso + 0 * 1, 40, 1, 0, 0, TRUE /* f_tex */);
	ost << "\\quad" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
			Iso + 40 * 1, 40, 1, 40, 0, TRUE /* f_tex */);
	ost << "\\quad" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
			Iso + 80 * 1, 40, 1, 80, 0, TRUE /* f_tex */);
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
			ost << i << " & T_{" << Surf_A->Surf->Trihedral_pair_labels[i]
				<< "} & ";

			lint_vec_copy(Web->Dual_point_ranks + i * 6, planes6, 6);
			Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
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
				Surf_A->Classify_trihedral_pairs->Trihedral_pairs->Rep +
				iso * Surf_A->Classify_trihedral_pairs->
					Trihedral_pairs->representation_sz,
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
			Surf_A->A->element_print_latex(transporter, ost);
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

		lint_vec_copy(Web->Dual_point_ranks + list[i] * 6, planes6, 6);


		Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
			planes6, 
			transporter, 
			orbit_index, 
			0 /*verbose_level */);

		Surf_A->A->element_invert(transporter, Elt1, 0);
		Surf_A->A->element_mult(transporter0, Elt1, Elt2, 0);

		ost << "The isomorphism from $T_0$ to $T_{"
				<<  list[i] << "}$ is :" << endl;
		ost << "$$" << endl;
		Surf_A->A->element_print_latex(Elt2, ost);
		ost << " = " << endl;
		Surf_A->A->element_print_latex(transporter0, ost);
		ost << " \\cdot " << endl;
		ost << "\\left(" << endl;
		Surf_A->A->element_print_latex(transporter, ost);
		ost << "\\right)^{-1}" << endl;
		ost << "$$" << endl;

		for (j = 0; j < cosets->len; j++) {
			ost << "i=" << i << " / " << list_sz << " j=" << j
					<< " / " << cosets->len << " considering "
					"coset given by:" << endl;
			ost << "$$" << endl;
			Surf_A->A->element_print_latex(cosets->ith(j), ost);
			ost << "$$" << endl;



			matrix_group *mtx;

			mtx = Surf_A->A->G.matrix_grp;

			F->transpose_matrix(Elt2, Tt, 4, 4);
			if (mtx->f_semilinear) {
				// if we are doing semilinear:
				Tt[4 * 4] = Elt2[4 * 4]; 
			}


			Surf_A->A->make_element(Elt3, Tt, 0);
			Surf_A->A->element_invert(cosets->ith(j), Elt5, 0);
			Surf_A->A->element_mult(Elt3, Elt5, Elt4, 0);
	
			//cout << "transporter transposed:" << endl;
			//A->print_quick(cout, Elt2);

			int coeff_out[20];


			//Surf_A->A->element_invert(Elt4, Elt5, 0);

			ost << "i=" << i << " / " << list_sz << " j=" << j
				<< " / " << cosets->len << " testing element:" << endl;
			ost << "$$" << endl;
			Surf_A->A->element_print_latex(Elt4, ost);
			ost << " = " << endl;
			Surf_A->A->element_print_latex(Elt3, ost);
			ost << " \\cdot " << endl;
			Surf_A->A->element_print_latex(Elt5, ost);
			ost << "$$" << endl;


			//matrix_group *M;

			//M = A->G.matrix_grp;
			mtx->substitute_surface_equation(Elt4,
					the_equation, coeff_out, Surf,
					0 /*verbose_level - 1*/);


			F->PG_element_normalize(coeff_out, 1, 20);

			ost << "The transformed equation is: $" << endl;
			int_vec_print(ost, coeff_out, 20);
			ost << "$\\\\" << endl;


			if (int_vec_compare(coeff_out, the_equation, 20) == 0) {
				ost << "trihedral pair " << i << " / " << Web->nb_T
					<< ", coset " << j << " / " << cosets->len
					<< " gives an automorphism\\\\" << endl;
				ost << "automorphism = " << endl;
				ost << "$$" << endl;
				Surf_A->A->element_print_latex(Elt4, ost);
				ost << "$$" << endl;

			}
			else {
				ost << "The equation is different, the group "
						"element is not an automorphism\\\\" << endl;
			}
			
		} // next j
	} // next i


	FREE_int(transporter);
	FREE_int(transporter0);
}





}}



