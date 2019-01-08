// arc_lifting.C
// 
// Anton Betten, Fatma Karaoglu
//
// January 24, 2017
// moved here from clebsch.C: March 22, 2017
//
// 
//
//

#include "orbiter.h"


static void intersection_matrix_entry_print(int *p, 
	int m, int n, int i, int j, int val, char *output, void *data);
static void Web_of_cubic_curves_entry_print(int *p, 
	int m, int n, int i, int j, int val, char *output, void *data);

arc_lifting::arc_lifting()
{
	null();
}

arc_lifting::~arc_lifting()
{
	freeself();
}

void arc_lifting::null()
{
	q = 0;
	F = NULL;
	Surf = NULL;
	Surf_A = NULL;

	E = NULL;
	E_idx = NULL;

	T_idx = NULL;
	nb_T = 0;

	the_equation = NULL;
	Web_of_cubic_curves = NULL;
	The_plane_equations = NULL;
	The_plane_rank = NULL;
	The_plane_duals = NULL;
	Dual_point_ranks = NULL;
	base_curves = NULL;

	The_surface_equations = NULL;

	stab_gens = NULL;
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
}

void arc_lifting::freeself()
{
	if (E) {
		FREE_OBJECT(E);
	}
	if (E_idx) {
		FREE_int(E_idx);
		}
	if (T_idx) {
		FREE_int(T_idx);
		}
	if (the_equation) {
		FREE_int(the_equation);
		}
	if (Web_of_cubic_curves) {
		FREE_int(Web_of_cubic_curves);
		}
	if (The_plane_equations) {
		FREE_int(The_plane_equations);
		}
	if (The_plane_rank) {
		FREE_int(The_plane_rank);
		}
	if (The_plane_duals) {
		FREE_int(The_plane_duals);
		}
	if (Dual_point_ranks) {
		FREE_int(Dual_point_ranks);
		}
	if (base_curves) {
		FREE_int(base_curves);
		}

	if (The_surface_equations) {
		FREE_int(The_surface_equations);
		}


	if (stab_gens) {
		FREE_OBJECT(stab_gens);
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


void arc_lifting::create_surface(surface_with_action *Surf_A, 
	int *Arc6, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q;
	surface *Surf;

	if (f_v) {
		cout << "arc_lifting::create_surface" << endl;
		}

	q = Surf_A->F->q;
	Surf = Surf_A->Surf;


	if (f_v) {
		cout << "arc_lifting::create_surface "
				"before init" << endl;
		}
	init(Surf_A, Arc6, 6, verbose_level - 2);








	if (f_v) {
		cout << "arc_lifting::create_surface "
				"before lift_prepare" << endl;
		}
	lift_prepare(verbose_level - 2);



	The_surface_equations = NEW_int((q + 1) * 20);
	
	if (f_v) {
		cout << "arc_lifting::create_surface before "
				"create_surface_from_trihedral_pair_and_arc"
				<< endl;
		}
	create_surface_from_trihedral_pair_and_arc(
		t_idx0, planes6,
		The_six_plane_equations,
		The_surface_equations,
		lambda, lambda_rk,
		verbose_level);

	if (f_v) {
		print_equations();
		}

	
	if (f_v) {
		cout << "arc_lifting::create_surface "
				"before create_clebsch_system" << endl;
		}
	create_clebsch_system(
		The_six_plane_equations, 
		lambda, 
		0 /* verbose_level */);



	if (f_v) {
		cout << "arc_lifting::create_surface before "
				"create_stabilizer_of_trihedral_pair" << endl;
		}
	stab_gens = create_stabilizer_of_trihedral_pair(
		planes6, 
		trihedral_pair_orbit_index, 
		0 /*verbose_level*/);

	stab_gens->group_order(stabilizer_of_trihedral_pair_go);
	if (f_v) {
		cout << "arc_lifting::create_surface the stabilizer of "
				"the trihedral pair has order "
				<< stabilizer_of_trihedral_pair_go << endl;
		}



	if (f_v) {
		cout << "arc_lifting::create_surface before "
				"create_action_on_equations_and_compute_orbits" << endl;
		}
	create_action_on_equations_and_compute_orbits(
		The_surface_equations, 
		stab_gens
		/* strong_generators *gens_for_stabilizer_of_trihedral_pair */,
		A_on_equations, Orb, 
		0 /* verbose_level */);

	
	if (f_v) {
		cout << "arc_lifting::create_surface the orbits "
				"on the pencil of surfaces are:" << endl;
		}
	Orb->print_and_list_orbits(cout);



	//Surf_A->A->group_order(go_PGL);


	if (f_v) {
		cout << "arc_lifting::create_surface before "
				"Orb->stabilizer_any_point_plus_cosets" << endl;
		}
	gens_subgroup = 
		Orb->stabilizer_any_point_plus_cosets(
			Surf_A->A, 
			stabilizer_of_trihedral_pair_go, 
			lambda_rk /* pt */, 
			cosets, 
			0 /* verbose_level */);

	if (f_v) {
		cout << "arc_lifting::create_surface after "
				"Orb->stabilizer_any_point_plus_cosets" << endl;
		}

	if (f_v) {
		cout << "arc_lifting::create_surface we found the "
				"following coset representatives:" << endl;
		cosets->print(cout);
		}




	if (f_v) {
		cout << "arc_lifting::create_surface after "
				"Orb->stabilizer_any_point" << endl;
		}
	gens_subgroup->group_order(stab_order);
	if (f_v) {
		cout << "arc_lifting::create_surface "
				"The stabilizer of the trihedral pair inside "
				"the group of the surface has order "
				<< stab_order << endl;
		}

	if (f_v) {
		cout << "arc_lifting::create_surface elements "
				"in the stabilizer:" << endl;
		gens_subgroup->print_elements_ost(cout);
		}

	if (f_v) {
		cout << "arc_lifting::create_surface The stabilizer of "
				"the trihedral pair inside the stabilizer of the "
				"surface is generated by:" << endl;
		gens_subgroup->print_generators_tex(cout);
		}






	Surf->compute_nine_lines_by_dual_point_ranks(
		Dual_point_ranks + 0 * 6, 
		Dual_point_ranks + 0 * 6 + 3, 
		nine_lines, 
		0 /* verbose_level */);

	if (f_v) {
		cout << "arc_lifting::create_surface before "
				"loop_over_trihedral_pairs" << endl;
		}
	loop_over_trihedral_pairs(cosets, 
		coset_reps, 
		aut_T_index, 
		aut_coset_index, 
		verbose_level);
	if (f_v) {
		cout << "arc_lifting::create_surface after "
				"loop_over_trihedral_pairs" << endl;
		cout << "arc_lifting::create_surface we found an "
				"orbit of length " << coset_reps->len << endl;
		}
	

	

	{
	longinteger_object ago;
	
	if (f_v) {
		cout << "arc_lifting::create_surface "
				"Extending the group:" << endl;
		}
	Aut_gens = NEW_OBJECT(strong_generators);
	Aut_gens->init_group_extension(gens_subgroup, 
		coset_reps, coset_reps->len, verbose_level - 3);

	Aut_gens->group_order(ago);
	if (f_v) {
		cout << "arc_lifting::create_surface "
				"The automorphism group has order " << ago << endl;
		cout << "arc_lifting::create_surface "
				"The automorphism group is:" << endl;
		Aut_gens->print_generators_tex(cout);
		}
	}
	
	if (f_v) {
		cout << "arc_lifting::create_surface done" << endl;
		}
}


void arc_lifting::lift_prepare(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, c;
	
	if (f_v) {
		cout << "arc_lifting::lift_prepare nb_T=" << nb_T << endl;
		}

	the_equation = NEW_int(20);
	The_plane_rank = NEW_int(45);
	The_plane_duals = NEW_int(45);
	Dual_point_ranks = NEW_int(nb_T * 6);

	transporter0 = NEW_int(Surf_A->A->elt_size_in_int);
	transporter = NEW_int(Surf_A->A->elt_size_in_int);
	Elt1 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt2 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt3 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt4 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt5 = NEW_int(Surf_A->A->elt_size_in_int);

	
	//t_idx0 = T_idx[0];
	t_idx0 = T_idx[115];
	//t_idx = T_idx[0];
	t_idx = T_idx[115];
	if (f_v) {
		cout << "arc_lifting::lift_prepare "
				"We choose trihedral pair t_idx0=" << t_idx0 << endl;
		}
	int_vec_copy(Surf->Trihedral_to_Eckardt +
			t_idx0 * 6, row_col_Eckardt_points, 6);

#if 1
	base_curves4[0] = row_col_Eckardt_points[0];
	base_curves4[1] = row_col_Eckardt_points[1];
	base_curves4[2] = row_col_Eckardt_points[3];
	base_curves4[3] = row_col_Eckardt_points[4];
#else
	base_curves4[3] = row_col_Eckardt_points[0];
	base_curves4[0] = row_col_Eckardt_points[1];
	base_curves4[1] = row_col_Eckardt_points[3];
	base_curves4[2] = row_col_Eckardt_points[4];
#endif
	
	if (f_v) {
		cout << "arc_lifting::lift_prepare base_curves4=";
		int_vec_print(cout, base_curves4, 4);
		cout << endl;
		}


	if (f_v) {
		cout << "arc_lifting::lift_prepare "
				"Creating the web of cubic "
				"curves through the arc:" << endl;
		}
	Surf->create_web_of_cubic_curves_and_equations_based_on_four_tritangent_planes(
		arc, base_curves4, 
		Web_of_cubic_curves, The_plane_equations,
		0 /*verbose_level*/);


	if (f_v) {
		cout << "arc_lifting::lift_prepare "
				"Testing the web of cubic curves:" << endl;
		}

	int pt_vec[3];

	for (i = 0; i < 45; i++) {
		//cout << i << " / " << 45 << ":" << endl;
		for (j = 0; j < 6; j++) {
			Surf->P2->unrank_point(pt_vec, arc[j]);
			c = Surf->Poly3->evaluate_at_a_point(
					Web_of_cubic_curves + i * 10, pt_vec);
			if (c) {
				cout << "arc_lifting::lift_prepare "
						"the cubic curve does not "
						"pass through the arc" << endl;
				exit(1);
				}
			}
		}
	if (f_v) {
		cout << "arc_lifting::lift_prepare The cubic curves all pass "
				"through the arc" << endl;
		}


	if (f_v) {
		cout << "arc_lifting::lift_prepare "
				"Computing the ranks of 4-subsets:" << endl;
		}

	int *Rk;
	int N;
	
	Surf->web_of_cubic_curves_rank_of_foursubsets(
		Web_of_cubic_curves,
		Rk, N, 0 /*verbose_level*/);
	{
	classify C;
	C.init(Rk, N, FALSE, 0 /* verbose_level */);
	cout << "arc_lifting::lift_prepare "
			"classification of ranks of 4-subsets:" << endl;
	C.print_naked_tex(cout, TRUE /* f_backwards */);
	cout << endl;
	}

	FREE_int(Rk);

	if (f_vv) {
		cout << "arc_lifting::lift_prepare "
				"Web_of_cubic_curves:" << endl;
		int_matrix_print(Web_of_cubic_curves, 45, 10);
		}

	if (f_vv) {
		cout << "arc_lifting::lift_prepare "
				"base_curves4=";
		int_vec_print(cout, base_curves4, 4);
		cout << endl;
		}


	base_curves = NEW_int(4 * 10);
	for (i = 0; i < 4; i++) {
		int_vec_copy(Web_of_cubic_curves + base_curves4[i] * 10,
				base_curves + i * 10, 10);
		}
	if (f_vv) {
		cout << "arc_lifting::lift_prepare "
				"base_curves:" << endl;
		int_matrix_print(base_curves, 4, 10);
		}

	
	
	if (f_vv) {
		cout << "arc_lifting::lift_prepare "
				"The_plane_equations:" << endl;
		int_matrix_print(The_plane_equations, 45, 4);
		}


	int Basis[16];
	for (i = 0; i < 45; i++) {
		int_vec_copy(The_plane_equations + i * 4, Basis, 4);
		F->RREF_and_kernel(4, 1, Basis, 0 /* verbose_level */);
		The_plane_rank[i] = Surf->rank_plane(Basis + 4);
		}
	if (f_vv) {
		cout << "arc_lifting::lift_prepare "
				"The_plane_ranks:" << endl;
		print_integer_matrix_with_standard_labels(cout,
				The_plane_rank, 45, 1, TRUE /* f_tex */);
		}

	for (i = 0; i < 45; i++) {
		The_plane_duals[i] = Surf->rank_point(
				The_plane_equations + i * 4);
		}

	cout << "arc_lifting::lift_prepare "
			"computing Dual_point_ranks:" << endl;
	for (i = 0; i < nb_T; i++) {
		//cout << "trihedral pair " << i << " / "
		//<< Surf->nb_trihedral_pairs << endl;

		int e[6];
		
		int_vec_copy(Surf->Trihedral_to_Eckardt + T_idx[i] * 6, e, 6);
		for (j = 0; j < 6; j++) {
			Dual_point_ranks[i * 6 + j] = The_plane_duals[e[j]];
			}

		}

	if (f_vv) {
		cout << "arc_lifting::lift_prepare "
				"Dual_point_ranks:" << endl;
		int_matrix_print(Dual_point_ranks, nb_T, 6);
		}


	if (f_v) {
		cout << "arc_lifting::lift_prepare before "
				"Surf->create_lines_from_plane_equations" << endl;
		}
	Surf->create_lines_from_plane_equations(
			The_plane_equations, Lines27, verbose_level);
	if (f_v) {
		cout << "arc_lifting::lift_prepare after "
				"Surf->create_lines_from_plane_equations" << endl;
		}


	if (f_v) {
		cout << "arc_lifting::lift_prepare done" << endl;
		}
}


void arc_lifting::loop_over_trihedral_pairs(
	vector_ge *cosets, vector_ge *&coset_reps,
	int *&aut_T_index, int *&aut_coset_index,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int planes6[6];
	int orbit_index0;
	int orbit_index;
	int orbit_length;
	int Tt[4 * 4 + 1];
	int Nine_lines0[9];
	int Nine_lines[9];
	int *v;
	int sz;

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
		Dual_point_ranks + t_idx0 * 6, 
		transporter0, 
		orbit_index0, 
		0 /*verbose_level*/);
	if (f_v) {
		cout << "arc_lifting::loop_over_trihedral_pairs "
				"Trihedral pair " << t_idx0
				<< " lies in orbit " << orbit_index0 << endl;
		}
	
	Surf->compute_nine_lines_by_dual_point_ranks(
		Dual_point_ranks + t_idx0 * 6, 
		Dual_point_ranks + t_idx0 * 6 + 3, 
		Nine_lines0, 
		0 /* verbose_level */);

	if (FALSE) {
		cout << "arc_lifting::loop_over_trihedral_pairs "
				"The first trihedral pair gives "
				"the following nine lines: ";
		int_vec_print(cout, Nine_lines0, 9);
		cout << endl;
		}

	coset_reps = NEW_OBJECT(vector_ge);
	coset_reps->init(Surf_A->A);
	coset_reps->allocate(nb_T * cosets->len);

	aut_T_index = NEW_int(nb_T * cosets->len);
	aut_coset_index = NEW_int(nb_T * cosets->len);

	for (i = 0; i < nb_T; i++) {

		if (f_v) {
			cout << "arc_lifting::loop_over_trihedral_pairs "
					"testing if trihedral pair "
					<< i << " / " << nb_T << " = " << T_idx[i];
			cout << " lies in the orbit:" << endl;
			}

		int_vec_copy(Dual_point_ranks + i * 6, planes6, 6);

		Surf->compute_nine_lines_by_dual_point_ranks(
			planes6, 
			planes6 + 3, 
			Nine_lines, 
			0 /* verbose_level */);

		if (FALSE) {
			cout << "arc_lifting::loop_over_trihedral_pairs "
					"The " << i << "-th trihedral "
					"pair gives the following nine lines: ";
			int_vec_print(cout, Nine_lines, 9);
			cout << endl;
			}

		int_vec_intersect(Nine_lines0, 9, Nine_lines, 9, v, sz);

		if (FALSE) {
			cout << "arc_lifting::loop_over_trihedral_pairs "
					"The nine lines of the " << i
					<< "-th trihedral pair intersect the "
						"nine lines of the first in " << sz
						<< " lines, which are: ";
			int_vec_print(cout, v, sz);
			cout << endl;
			}

		if (FALSE) {
			Surf->print_trihedral_pair_in_dual_coordinates_in_GAP(
				planes6, planes6 + 3);
			cout << endl;
			}

		FREE_int(v);
		


		Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(
			planes6, 
			transporter, 
			orbit_index, 
			0 /*verbose_level */);


		if (orbit_index != orbit_index0) {
			if (f_v) {
				cout << "arc_lifting::loop_over_trihedral_pairs "
						"trihedral pair " << i << " / " << nb_T
						<< " lies in orbit " << orbit_index
						<< " and so $T_{" << t_idx0
						<< "}$ and T_i are not isomorphic" << endl;
				}
			continue;
			}
		if (f_v) {
			cout << "arc_lifting::loop_over_trihedral_pairs "
					"trihedral pair " << i << " / " << nb_T
					<< " lies in orbit " << orbit_index
					<< " and so $T_{" << t_idx0
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
			//Surf_A->A->element_invert(cosets->ith(j), Elt5, 0);
			//Surf_A->A->element_mult(Elt5, Elt2, Elt3, 0);

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

			if (f_v) {
				cout << "arc_lifting::loop_over_trihedral_pairs "
						"Elt4:" << endl;
				Surf_A->A->element_print_quick(Elt4, cout);
				}

			if (f_v) {
				cout << "arc_lifting::loop_over_trihedral_pairs "
						"mtx->f_semilinear=" << mtx->f_semilinear << endl;
				}


#if 1
			matrix_group *M;

			M = Surf_A->A->G.matrix_grp;
			M->substitute_surface_eqation(Elt4,
					the_equation, coeff_out, Surf,
					verbose_level - 1);
#else

			if (mtx->f_semilinear) {
				int n, frob; //, e;
				
				n = mtx->n;
				frob = Elt4[n * n];
				Surf->substitute_semilinear(the_equation, 
					coeff_out, 
					mtx->f_semilinear, 
					frob, 
					Elt4, 
					0 /* verbose_level */);
				}
			else {
				Surf->substitute_semilinear(the_equation, 
					coeff_out, 
					FALSE, 0, 
					Elt4, 
					0 /* verbose_level */);
				}
#endif

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
							"trihedral pair " << i << " / " << nb_T
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
					int_vec_print(cout, Nine_lines, 9);
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
							"trihedral pair " << i << " / " << nb_T
							<< " coset " << j << " / " << cosets->len
							<< " does not lie in the orbit" << endl;
					}
				//exit(1);
				}
			} // next j

		} // next i

	coset_reps->reallocate(orbit_length);

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



void arc_lifting::init(surface_with_action *Surf_A, 
	int *arc, int arc_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_lifting::init" << endl;
		}
	
	arc_lifting::arc = arc;
	arc_lifting::arc_size = arc_size;
	arc_lifting::Surf_A = Surf_A;
	Surf = Surf_A->Surf;
	F = Surf->F;
	q = F->q;


	if (arc_size != 6) {
		cout << "arc_lifting::init arc_size = 6" << endl;
		exit(1);
		}
	


	find_Eckardt_points(verbose_level);
	find_trihedral_pairs(verbose_level);


	if (f_v) {
		cout << "arc_lifting::init done" << endl;
		}
}

void arc_lifting::find_Eckardt_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "arc_lifting::find_Eckardt_points" << endl;
		}
	int s;
	
	E = Surf->P2->compute_eckardt_point_info(arc, verbose_level);
	if (f_v) {
		cout << "arc_lifting::init We found " << E->nb_E
				<< " Eckardt points" << endl;
		for (s = 0; s < E->nb_E; s++) {
			cout << s << " / " << E->nb_E << " : ";
			E->E[s].print();
			cout << " = E_{" << s << "}";
			cout << endl;
			}
		}


	E_idx = NEW_int(E->nb_E);
	for (s = 0; s < E->nb_E; s++) {
		E_idx[s] = E->E[s].rank();
		}
	if (f_v) {
		cout << "by rank: ";
		int_vec_print(cout, E_idx, E->nb_E);
		cout << endl;
		}
	if (f_v) {
		cout << "arc_lifting::find_Eckardt_points done" << endl;
		}
}

void arc_lifting::find_trihedral_pairs(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "arc_lifting::find_trihedral_pairs" << endl;
		}
#if 0
	Surf->find_trihedral_pairs_from_collinear_triples_of_Eckardt_points(
		E_idx, nb_E,
		T_idx, nb_T, verbose_level);
#else
	T_idx = NEW_int(120);
	nb_T = 120;
	for (i = 0; i < 120; i++) {
		T_idx[i] = i;
		}
#endif

	int t_idx;

	if (nb_T == 0) {
		cout << "nb_T == 0" << endl;	
		exit(1);
		}


	if (f_v) {
		cout << "List of special trihedral pairs:" << endl;
		for (i = 0; i < nb_T; i++) {
			t_idx = T_idx[i];
			cout << i << " / " << nb_T << ": T_{" << t_idx << "} =  T_{"
					<< Surf->Trihedral_pair_labels[t_idx] << "}" << endl;
			}
		}

	if (f_v) {
		cout << "arc_lifting::find_trihedral_pairs done" << endl;
		}
}

void arc_lifting::create_the_six_plane_equations(
		int t_idx, int *The_six_plane_equations, int *plane6,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "arc_lifting::create_the_six_plane_equations "
				"t_idx=" << t_idx << endl;
		}


	int_vec_copy(Surf->Trihedral_to_Eckardt + t_idx * 6,
			row_col_Eckardt_points, 6);

	for (i = 0; i < 6; i++) {
		int_vec_copy(The_plane_equations + row_col_Eckardt_points[i] * 4,
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
	int t_idx, int *planes6, 
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
		cout << "arc_lifting::create_surface_from_trihedral_"
				"pair_and_arc t_idx=" << t_idx << endl;
		}

	create_the_six_plane_equations(t_idx, 
		The_six_plane_equations, planes6, 
		verbose_level);


	if (f_v) {
		cout << "arc_lifting::create_surface_from_trihedral_"
				"pair_and_arc before create_equations_for_pencil_"
				"of_surfaces_from_trihedral_pair" << endl;
		}
	Surf->create_equations_for_pencil_of_surfaces_from_trihedral_pair(
		The_six_plane_equations, The_surface_equations, 
		verbose_level);

	if (f_v) {
		cout << "arc_lifting::create_surface_from_trihedral_"
				"pair_and_arc before create_lambda_from_trihedral_"
				"pair_and_arc" << endl;
		}
	Surf->create_lambda_from_trihedral_pair_and_arc(arc, 
		Web_of_cubic_curves, 
		The_plane_equations, t_idx, lambda, lambda_rk, 
		verbose_level);


	int_vec_copy(The_surface_equations + lambda_rk * 20,
			the_equation, 20);

	if (f_v) {
		cout << "arc_lifting::create_surface_from_trihedral_"
				"pair_and_arc done" << endl;
		}
}

strong_generators *arc_lifting::create_stabilizer_of_trihedral_pair(
	int *planes6, 
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
				"before Surf_A->identify_trihedral_pair_and_"
				"get_stabilizer" << endl;
		}

	gens_dual =
		Surf_A->Classify_trihedral_pairs->
			identify_trihedral_pair_and_get_stabilizer(
		planes6, transporter, trihedral_pair_orbit_index, 
		verbose_level);

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair "
				"after Surf_A->identify_trihedral_pair_and_get_"
				"stabilizer" << endl;
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
				"group elements:" << endl;
		gens_dual->print_elements_ost(cout);
		}


	gens->init(Surf_A->A);

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair "
				"before gens->init_transposed_group" << endl;
		}
	gens->init_transposed_group(gens_dual, verbose_level);

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair "
				"The transposed stabilizer is:" << endl;
		gens->print_generators_tex(cout);
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
		cout << "arc_lifting::create_action_on_equations_and_"
				"compute_orbits" << endl;
		}
	
	if (f_v) {
		cout << "arc_lifting::create_action_on_equations_and_"
				"compute_orbits before create_action_and_compute_"
				"orbits_on_equations" << endl;
		}

	create_action_and_compute_orbits_on_equations(
		Surf_A->A,
		Surf->Poly3_4, 
		The_surface_equations, 
		q + 1 /* nb_equations */, 
		gens_for_stabilizer_of_trihedral_pair, 
		A_on_equations, 
		Orb, 
		verbose_level);
		// in ACTION/action_global.C

	if (f_v) {
		cout << "arc_lifting::create_action_on_equations_and_"
				"compute_orbits done" << endl;
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
		int_vec_print(cout, nine_lines, 9);
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


void arc_lifting::print(ostream &ost)
{
	int i;

#if 0
	Surf->print_polynomial_domains(ost);
	Surf->print_line_labelling(ost);
	
	cout << "arc_lifting::print before print_Steiner_and_Eckardt" << endl;
	Surf->print_Steiner_and_Eckardt(ost);
	cout << "arc_lifting::print after print_Steiner_and_Eckardt" << endl;
#endif

	cout << "arc_lifting::print before print_Eckardt_point_data" << endl;
	print_Eckardt_point_data(ost);
	cout << "arc_lifting::print after print_Eckardt_point_data" << endl;

	cout << "arc_lifting::print before print_Eckardt_points" << endl;
	print_Eckardt_points(ost);
	cout << "arc_lifting::print before print_web_of_cubic_curves" << endl;
	print_web_of_cubic_curves(ost);


	cout << "arc_lifting::print before print_plane_equations" << endl;
	print_trihedral_plane_equations(ost);


	//cout << "arc_lifting::print before print_dual_point_ranks" << endl;
	//print_dual_point_ranks(ost);


	cout << "arc_lifting::print before "
			"print_the_six_plane_equations" << endl;
	print_the_six_plane_equations(
			The_six_plane_equations, planes6, ost);

	cout << "arc_lifting::print before "
			"print_surface_equations_on_line" << endl;
	print_surface_equations_on_line(The_surface_equations, 
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
	stab_gens->print_generators_tex(ost);

	ost << "The orbits of the trihedral pair stabilizer on the $q+1$ "
			"surfaces on the line are:\\\\" << endl;
	Orb->print_fancy(
			ost, TRUE, Surf_A->A, stab_gens);


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
}

void arc_lifting::print_Eckardt_point_data(ostream &ost)
{
	print_bisecants(ost);
	print_intersections(ost);
	print_conics(ost);
}

void arc_lifting::print_bisecants(ostream &ost)
{
	int i, j, h, a;
	int Mtx[9];
	
	ost << "The 15 bisecants are:\\\\" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "h & P_iP_j & \\mbox{rank} & \\mbox{line} "
			"& \\mbox{equation}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (h = 0; h < 15; h++) {
		a = E->bisecants[h];
		k2ij(h, i, j, 6);
		ost << h << " & P_{" << i + 1 << "}P_{" << j + 1
				<< "} & " << a << " & " << endl;
		ost << "\\left[ " << endl;
		Surf->P2->Grass_lines->print_single_generator_matrix_tex(ost, a);
		ost << "\\right] ";

		Surf->P2->Grass_lines->unrank_int_here_and_compute_perp(Mtx, a, 
			0 /*verbose_level */);
		F->PG_element_normalize(Mtx + 6, 1, 3);
		
		ost << " & ";
		Surf->Poly1->print_equation(ost, Mtx + 6);
		ost << "\\\\" << endl; 
		}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
}

void arc_lifting::print_intersections(ostream &ost)
{
	int labels[15];
	int fst[1];
	int len[1];
	fst[0] = 0;
	len[0] = 15;
	int i;
	
	for (i = 0; i < 15; i++) {
		labels[i] = i;
		}
	ost << "{\\small \\arraycolsep=1pt" << endl;
	ost << "$$" << endl;
	int_matrix_print_with_labels_and_partition(ost,
		E->Intersections, 15, 15,
		labels, labels, 
		fst, len, 1,  
		fst, len, 1,  
		intersection_matrix_entry_print, (void *) this, 
		TRUE /* f_tex */);
	ost << "$$}" << endl;
}

void arc_lifting::print_conics(ostream &ost)
{
	int h;
	
	ost << "The 6 conics are:\\\\" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & C_i & \\mbox{equation}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (h = 0; h < 6; h++) {
		ost << h + 1 << " & C_" << h + 1 << " & " << endl;
		Surf->Poly2->print_equation(ost,
				E->conic_coefficients + h * 6);
		ost << "\\\\" << endl; 
		}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
}

void arc_lifting::print_Eckardt_points(ostream &ost)
{
	int s;
	
	ost << "We found " << E->nb_E << " Eckardt points:\\\\" << endl;
	for (s = 0; s < E->nb_E; s++) {
		ost << s << " / " << E->nb_E << " : $";
		E->E[s].latex(ost);
		ost << "= E_{" << E_idx[s] << "}$\\\\" << endl;
		}
	//ost << "by rank: ";
	//int_vec_print(ost, E_idx, nb_E);
	//ost << "\\\\" << endl;
}

void arc_lifting::print_web_of_cubic_curves(ostream &ost)
{
	ost << "The web of cubic curves is:\\\\" << endl;

#if 0
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost, 
		Web_of_cubic_curves, 15, 10, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, 
		Web_of_cubic_curves + 15 * 10, 15, 10, 15, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, 
		Web_of_cubic_curves + 30 * 10, 15, 10, 30, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;
#endif

	int *bisecants;
	int *conics;

	int labels[15];
	int row_fst[1];
	int row_len[1];
	int col_fst[1];
	int col_len[1];
	row_fst[0] = 0;
	row_len[0] = 15;
	col_fst[0] = 0;
	col_len[0] = 10;
	char str[1000];
	int i, j, k, l, m, n, h, ij, kl, mn;

	Surf->P2->compute_bisecants_and_conics(arc,
			bisecants, conics, 0 /*verbose_level*/);

	for (h = 0; h < 45; h++) {
		ost << "$";
		sprintf(str, "W_{%s}=\\Phi\\big(\\pi_{%d}\\big) "
				"= \\Phi\\big(\\pi_{%s}\\big)",
				Surf->Eckard_point_label[h], h,
				Surf->Eckard_point_label[h]);
		ost << str;
		ost << " = ";
		if (h < 30) {
			ordered_pair_unrank(h, i, j, 6);
			ij = ij2k(i, j, 6);
			ost << "C_" << j + 1
				<< "P_{" << i + 1 << "}P_{" << j + 1 << "} = ";
			ost << "\\big(";
			Surf->Poly2->print_equation(ost, conics + j * 6);
			ost << "\\big)";
			ost << "\\big(";
			Surf->Poly1->print_equation(ost, bisecants + ij * 3);
			ost << "\\big)";
			//multiply_conic_times_linear(conics + j * 6,
			//bisecants + ij * 3, ten_coeff, 0 /* verbose_level */);
			}
		else {
			unordered_triple_pair_unrank(h - 30, i, j, k, l, m, n);
			ij = ij2k(i, j, 6);
			kl = ij2k(k, l, 6);
			mn = ij2k(m, n, 6);
			ost << "P_{" << i + 1 << "}P_{" << j + 1 << "},P_{"
					<< k + 1 << "}P_{" << l + 1 << "},P_{"
					<< m + 1 << "}P_{" << n + 1 << "} = ";
			ost << "\\big(";
			Surf->Poly1->print_equation(ost, bisecants + ij * 3);
			ost << "\\big)";
			ost << "\\big(";
			Surf->Poly1->print_equation(ost, bisecants + kl * 3);
			ost << "\\big)";
			ost << "\\big(";
			Surf->Poly1->print_equation(ost, bisecants + mn * 3);
			ost << "\\big)";
			//multiply_linear_times_linear_times_linear(
			//bisecants + ij * 3, bisecants + kl * 3,
			//bisecants + mn * 3, ten_coeff, 0 /* verbose_level */);
			}
		ost << " = ";
		Surf->Poly3->print_equation(ost, Web_of_cubic_curves + h * 10);
		ost << "$\\\\";
		}

	ost << "The coeffcients are:" << endl;
	for (i = 0; i < 15; i++) {
		labels[i] = i;
		}
	ost << "$$" << endl;
	int_matrix_print_with_labels_and_partition(ost,
			Web_of_cubic_curves, 15, 10,
		labels, labels, 
		row_fst, row_len, 1,  
		col_fst, col_len, 1,  
		Web_of_cubic_curves_entry_print, (void *) this, 
		TRUE /* f_tex */);
	ost << "$$" << endl;

	for (i = 0; i < 15; i++) {
		labels[i] = 15 + i;
		}
	ost << "$$" << endl;
	int_matrix_print_with_labels_and_partition(ost,
			Web_of_cubic_curves, 15, 10,
		labels, labels, 
		row_fst, row_len, 1,  
		col_fst, col_len, 1,  
		Web_of_cubic_curves_entry_print, (void *) this, 
		TRUE /* f_tex */);
	ost << "$$" << endl;

	for (i = 0; i < 15; i++) {
		labels[i] = 30 + i;
		}
	ost << "$$" << endl;
	int_matrix_print_with_labels_and_partition(ost,
			Web_of_cubic_curves, 15, 10,
		labels, labels, 
		row_fst, row_len, 1,  
		col_fst, col_len, 1,  
		Web_of_cubic_curves_entry_print, (void *) this, 
		TRUE /* f_tex */);
	ost << "$$" << endl;

	FREE_int(bisecants);
	FREE_int(conics);

}

void arc_lifting::print_trihedral_plane_equations(
		ostream &ost)
{
	int i;
	
	ost << "The chosen abstract trihedral pair is no "
			<< t_idx0 << ":" << endl;
	ost << "$$" << endl;
	Surf->latex_abstract_trihedral_pair(ost, t_idx0);
	ost << "$$" << endl;
	ost << "The six planes in the trihedral pair are:" << endl;
	ost << "$$" << endl;
	int_vec_print(ost, row_col_Eckardt_points, 6);
	ost << "$$" << endl;
	ost << "We choose planes $0,1,3,4$ for the base curves:" << endl;
	ost << "$$" << endl;
	int_vec_print(ost, base_curves4, 4);
	ost << "$$" << endl;
	ost << "The four base curves are:\\\\";
	for (i = 0; i < 4; i++) {
		ost << "$$" << endl;
		ost << "W_{" << Surf->Eckard_point_label[base_curves4[i]];
		ost << "}=\\Phi\\big(\\pi_{" << base_curves4[i]
			<< "}\\big) = \\Phi\\big(\\pi_{"
			<< Surf->Eckard_point_label[base_curves4[i]]
			<< "}\\big)=V\\Big(" << endl;
		Surf->Poly3->print_equation(ost, base_curves + i * 10);
		ost << "\\Big)" << endl;
		ost << "$$" << endl;
		}

	ost << "The coefficients of the four base curves are:\\\\";
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			base_curves, 4, 10, TRUE /* f_tex*/);
	ost << "$$" << endl;

	ost << "The resulting plane equations are:\\\\";
	for (i = 0; i < 45; i++) {
		ost << "$\\pi_{" << i << "}=\\pi_{"
			<< Surf->Eckard_point_label[i] << "}=V\\Big(";
		Surf->Poly1_4->print_equation(ost,
				The_plane_equations + i * 4);
		ost << "\\Big)$\\\\";
		}

	ost << "The dual coordinates of the plane equations are:\\\\";
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost, 
		The_plane_equations, 15, 4, TRUE /* f_tex*/);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, 
		The_plane_equations + 15 * 4, 15, 4, 15, 0, TRUE /* f_tex*/);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, 
		The_plane_equations + 30 * 4, 15, 4, 30, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "The dual ranks are:\\\\";
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost, 
		The_plane_duals, 15, 1, TRUE /* f_tex*/);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, 
		The_plane_duals + 15 * 1, 15, 1, 15, 0, TRUE /* f_tex*/);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, 
		The_plane_duals + 30 * 1, 15, 1, 30, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;

	print_lines(ost);
}

void arc_lifting::print_lines(ostream &ost)
{
	int i, a;
	int v[8];
	
	ost << "The 27 lines:\\\\";
	for (i = 0; i < 27; i++) {
		a = Lines27[i];
		ost << "$$" << endl;
		ost << "\\ell_{" << i << "} = "
				<< Surf->Line_label_tex[i] << " = " << a << " = ";
		Surf->unrank_line(v, a);
		ost << "\\left[ " << endl;
		Surf->Gr->print_single_generator_matrix_tex(ost, a);
		ost << "\\right] ";
		ost << "$$" << endl;
		}
}


void arc_lifting::print_dual_point_ranks(ostream &ost)
{
	ost << "Dual point ranks:\\\\";
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Dual_point_ranks, nb_T, 6, TRUE /* f_tex*/);
	ost << "$$" << endl;
}

void arc_lifting::print_FG(ostream &ost)
{
	ost << "$F$-planes:\\\\";
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			F_plane, 3, 4, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$G$-planes:\\\\";
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			G_plane, 3, 4, TRUE /* f_tex*/);
	ost << "$$" << endl;
}

void arc_lifting::print_the_six_plane_equations(
	int *The_six_plane_equations,
	int *plane6, ostream &ost)
{
	int i, h;
	
	ost << "The six plane equations are:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			The_six_plane_equations, 6, 4, TRUE /* f_tex*/);
	ost << "$$" << endl;

	ost << "The six plane equations are:\\\\";
	for (i = 0; i < 6; i++) {
		h = row_col_Eckardt_points[i];
		ost << "$\\pi_{" << h << "}=\\pi_{"
				<< Surf->Eckard_point_label[h] << "}=V\\big(";
		Surf->Poly1_4->print_equation(ost, The_plane_equations + h * 4);
		ost << "\\big)$\\\\";
		}
}

void arc_lifting::print_surface_equations_on_line(
	int *The_surface_equations,
	int lambda, int lambda_rk, ostream &ost)
{
	int i;
	int v[2];
	
	ost << "The $q+1$ equations on the line are:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			The_surface_equations, q + 1, 20, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$$" << endl;
	ost << "\\lambda = " << lambda << ", \\; \\mbox{in row} \\; "
			<< lambda_rk << endl;
	ost << "$$" << endl;
	
	ost << "The $q+1$ equations on the line are:\\\\" << endl;
	for (i = 0; i < q + 1; i++) {
		ost << "Row " << i << " : ";

		F->PG_element_unrank_modified(v, 1, 2, i);
		F->PG_element_normalize_from_front(v, 1, 2);
		
		ost << "$";
		ost << v[0] << " \\cdot ";
		ost << "\\big(";
		Surf->Poly1_4->print_equation(ost,
				The_plane_equations + row_col_Eckardt_points[0] * 4);
		ost << "\\big)";
		ost << "\\big(";
		Surf->Poly1_4->print_equation(ost,
				The_plane_equations + row_col_Eckardt_points[1] * 4);
		ost << "\\big)";
		ost << "\\big(";
		Surf->Poly1_4->print_equation(ost,
				The_plane_equations + row_col_Eckardt_points[2] * 4);
		ost << "\\big)";
		ost << "+";
		ost << v[1] << " \\cdot ";
		ost << "\\big(";
		Surf->Poly1_4->print_equation(ost,
				The_plane_equations + row_col_Eckardt_points[3] * 4);
		ost << "\\big)";
		ost << "\\big(";
		Surf->Poly1_4->print_equation(ost,
				The_plane_equations + row_col_Eckardt_points[4] * 4);
		ost << "\\big)";
		ost << "\\big(";
		Surf->Poly1_4->print_equation(ost,
				The_plane_equations + row_col_Eckardt_points[5] * 4);
		ost << "\\big)";
		ost << " = ";
		Surf->Poly3_4->print_equation(ost,
				The_surface_equations + i * 20);
		ost << "$\\\\";
		}
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
	int planes6[6];
	int orbit_index0;
	int orbit_index;
	int *transporter0;
	int *transporter;
	int list[120];
	int list_sz = 0;
	int Tt[17];
	int Iso[120];

	cout << "arc_lifting::print_isomorphism_types_of_"
			"trihedral_pairs" << endl;

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
		Dual_point_ranks + t_idx0 * 6, 
		transporter0, 
		orbit_index0, 
		0 /*verbose_level*/);
	ost << "Trihedral pair $T_{" << t_idx0 << "}$ lies in orbit "
			<< orbit_index0 << "\\\\" << endl;
	ost << "An isomorphism is given by" << endl;
	ost << "$$" << endl;
	Surf_A->A->element_print_latex(transporter0, ost);
	ost << "$$" << endl;
	



	for (i = 0; i < nb_T; i++) {

			cout << "testing if trihedral pair " << i << " / "
					<< nb_T << " = " << T_idx[i];
			cout << " lies in the orbit:" << endl;

		int_vec_copy(Dual_point_ranks + i * 6, planes6, 6);
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
				<< t_idx0 << "}$ is given by" << endl;
		ost << "$$" << endl;
		Surf_A->A->element_print_latex(Elt2, ost);
		ost << "$$" << endl;


		} // next i

	ost << "The isomorphism types of the trihedral pairs "
			"in the list of double triplets are:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Iso + 0 * 1, 40, 1, 0, 0, TRUE /* f_tex */);
	ost << "\\quad" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Iso + 40 * 1, 40, 1, 40, 0, TRUE /* f_tex */);
	ost << "\\quad" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
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

			int_vec_copy(Dual_point_ranks + i * 6, planes6, 6);
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
			int_vec_copy(
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
	int_set_print_tex(ost, list, list_sz);
	ost << "$$" << endl;
	ost << "\\{ ";
	for (i = 0; i < list_sz; i++) {
		ost << "T_{" << list[i] << "}";
		if (i < list_sz - 1)
			ost << ", ";
		}
	ost << " \\}";
	ost << "$$" << endl;


	ost << "\\bigskip" << endl;
	ost << "" << endl;
	ost << "\\subsection*{Computing the Automorphism "
			"Group, Step 2}" << endl;
	ost << "" << endl;




	ost << "We are now looping over the " << list_sz
			<< " trihedral pairs which are isomorphic to the "
			"double triplet of $T_0$ and over the "
			<< cosets->len << " cosets:\\\\" << endl;
	for (i = 0; i < list_sz; i++) {
		ost << "i=" << i << " / " << list_sz
				<< " considering $T_{" << list[i] << "}$:\\\\";

		int_vec_copy(Dual_point_ranks + list[i] * 6, planes6, 6);


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


#if 1
			//matrix_group *M;

			//M = A->G.matrix_grp;
			mtx->substitute_surface_eqation(Elt4,
					the_equation, coeff_out, Surf,
					0 /*verbose_level - 1*/);
#else
			if (mtx->f_semilinear) {
				int n, frob; //, e;
				
				n = mtx->n;
				frob = Elt4[n * n];
				Surf->substitute_semilinear(the_equation, 
					coeff_out, 
					mtx->f_semilinear, 
					frob, 
					Elt4, 
					0 /* verbose_level */);
				}
			else {
				Surf->substitute_semilinear(the_equation, 
					coeff_out, 
					FALSE, 0, 
					Elt4, 
					0 /* verbose_level */);
				}
#endif


			F->PG_element_normalize(coeff_out, 1, 20);

			ost << "The transformed equation is: $" << endl;
			int_vec_print(ost, coeff_out, 20);
			ost << "$\\\\" << endl;


			if (int_vec_compare(coeff_out, the_equation, 20) == 0) {
				ost << "trihedral pair " << i << " / " << nb_T
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





static void intersection_matrix_entry_print(int *p, 
	int m, int n, int i, int j, int val,
	char *output, void *data)
{
	//arc_lifting *AL;
	//AL = (arc_lifting *) data;
	int a, b;
	
	if (i == -1) {
		k2ij(j, a, b, 6);
		sprintf(output, "P_%dP_%d", a + 1, b + 1);
		}
	else if (j == -1) {
		k2ij(i, a, b, 6);
		sprintf(output, "P_%dP_%d", a + 1, b + 1);
		}
	else {
		if (val == -1) {
			strcpy(output, ".");
			}
		else {
			sprintf(output, "%d", val);
			}
		}
}

static void Web_of_cubic_curves_entry_print(int *p, 
	int m, int n, int i, int j, int val,
	char *output, void *data)
{
	arc_lifting *AL;
	AL = (arc_lifting *) data;
	
	if (i == -1) {
		AL->Surf->Poly3->print_monomial(output, j);
		}
	else if (j == -1) {
		sprintf(output, "\\pi_{%d} = \\pi_{%s}", i,
				AL->Surf->Eckard_point_label[i]);
		}
	else {
		sprintf(output, "%d", val);
		}
}




