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


static void intersection_matrix_entry_print(INT *p, INT m, INT n, INT i, INT j, INT val, BYTE *output, void *data);

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

	bisecants = NULL;
	Intersections = NULL;
	B_pts = NULL;
	B_pts_label = NULL;
	nb_B_pts = 0;
	E2 = NULL;
	nb_E2 = 0;
	conic_coefficients = NULL;


	E = NULL;
	E_idx = NULL;
	nb_E = 0;

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
	if (bisecants) {
		FREE_INT(bisecants);
		}
	if (Intersections) {
		FREE_INT(Intersections);
		}
	if (B_pts) {
		FREE_INT(B_pts);
		}
	if (B_pts_label) {
		FREE_INT(B_pts_label);
		}
	if (E2) {
		FREE_INT(E2);
		}
	if (conic_coefficients) {
		FREE_INT(conic_coefficients);
		}

	if (E) {
		delete [] E;
		}
	if (E_idx) {
		FREE_INT(E_idx);
		}
	if (T_idx) {
		FREE_INT(T_idx);
		}
	if (the_equation) {
		FREE_INT(the_equation);
		}
	if (Web_of_cubic_curves) {
		FREE_INT(Web_of_cubic_curves);
		}
	if (The_plane_equations) {
		FREE_INT(The_plane_equations);
		}
	if (The_plane_rank) {
		FREE_INT(The_plane_rank);
		}
	if (The_plane_duals) {
		FREE_INT(The_plane_duals);
		}
	if (Dual_point_ranks) {
		FREE_INT(Dual_point_ranks);
		}
	if (base_curves) {
		FREE_INT(base_curves);
		}

	if (The_surface_equations) {
		FREE_INT(The_surface_equations);
		}


	if (stab_gens) {
		delete stab_gens;
		}
	if (gens_subgroup) {
		delete gens_subgroup;
		}
	if (A_on_equations) {
		delete A_on_equations;
		}
	if (Orb) {
		delete Orb;
		}
	if (cosets) {
		delete cosets;
		}
	if (coset_reps) {
		delete coset_reps;
		}
	if (aut_T_index) {
		FREE_INT(aut_T_index);
		}
	if (aut_coset_index) {
		FREE_INT(aut_coset_index);
		}
	if (Aut_gens) {
		delete Aut_gens;
		}



	if (System) {
		FREE_INT(System);
		}
	if (transporter0) {
		FREE_INT(transporter0);
		}
	if (transporter) {
		FREE_INT(transporter);
		}
	if (Elt1) {
		FREE_INT(Elt1);
		}
	if (Elt2) {
		FREE_INT(Elt2);
		}
	if (Elt3) {
		FREE_INT(Elt3);
		}
	if (Elt4) {
		FREE_INT(Elt4);
		}
	if (Elt5) {
		FREE_INT(Elt5);
		}
	
	null();
}

void arc_lifting::init(surface_with_action *Surf_A, INT *arc, INT arc_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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

void arc_lifting::find_Eckardt_points(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "arc_lifting::find_Eckardt_points" << endl;
		}
	INT s;
	
	Surf->P2->find_Eckardt_points_from_arc_not_on_conic_prepare_data(arc, 
		bisecants, // [15]
		Intersections, // [15 * 15]
		B_pts, // [nb_B_pts]
		B_pts_label, // [nb_B_pts * 3]
		nb_B_pts, // at most 15
		E2, // [6 * 5 * 2] Eckardt points of the second type 
		nb_E2, // at most 30
		conic_coefficients, // [6 * 6]
		E, nb_E, 
		verbose_level);
	if (f_v) {
		cout << "arc_lifting::init We found " << nb_E << " Eckardt points" << endl;
		for (s = 0; s < nb_E; s++) {
			cout << s << " / " << nb_E << " : ";
			E[s].print();
			cout << " = E_{" << s << "}";
			cout << endl;
			}
		}


	E_idx = NEW_INT(nb_E);
	for (s = 0; s < nb_E; s++) {
		E_idx[s] = E[s].rank();
		}
	if (f_v) {
		cout << "by rank: ";
		INT_vec_print(cout, E_idx, nb_E);
		cout << endl;
		}
	if (f_v) {
		cout << "arc_lifting::find_Eckardt_points done" << endl;
		}
}

void arc_lifting::find_trihedral_pairs(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "arc_lifting::find_trihedral_pairs" << endl;
		}
#if 0
	Surf->find_trihedral_pairs_from_collinear_triples_of_Eckardt_points(E_idx, nb_E, 
		T_idx, nb_T, verbose_level);
#else
	T_idx = NEW_INT(120);
	nb_T = 120;
	for (i = 0; i < 120; i++) {
		T_idx[i] = i;
		}
#endif

	INT t_idx;

	if (nb_T == 0) {
		cout << "nb_T == 0" << endl;	
		exit(1);
		}


	if (f_v) {
		cout << "List of special trihedral pairs:" << endl;
		for (i = 0; i < nb_T; i++) {
			t_idx = T_idx[i];
			cout << i << " / " << nb_T << ": T_{" << t_idx << "} =  T_{" << Surf->Trihedral_pair_labels[t_idx] << "}" << endl;
			}
		}

	if (f_v) {
		cout << "arc_lifting::find_trihedral_pairs done" << endl;
		}
}

void arc_lifting::lift_prepare(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, c;
	
	if (f_v) {
		cout << "arc_lifting::lift_prepare nb_T=" << nb_T << endl;
		}

	the_equation = NEW_INT(20);
	The_plane_rank = NEW_INT(45);
	The_plane_duals = NEW_INT(45);
	Dual_point_ranks = NEW_INT(nb_T * 6);

	transporter0 = NEW_INT(Surf_A->A->elt_size_in_INT);
	transporter = NEW_INT(Surf_A->A->elt_size_in_INT);
	Elt1 = NEW_INT(Surf_A->A->elt_size_in_INT);
	Elt2 = NEW_INT(Surf_A->A->elt_size_in_INT);
	Elt3 = NEW_INT(Surf_A->A->elt_size_in_INT);
	Elt4 = NEW_INT(Surf_A->A->elt_size_in_INT);
	Elt5 = NEW_INT(Surf_A->A->elt_size_in_INT);

	
	t_idx0 = T_idx[0];
	if (f_v) {
		cout << "We choose thrihedral pair t_idx0=" << t_idx0 << endl;
		}
	INT_vec_copy(Surf->Trihedral_to_Eckardt + t_idx0 * 6, row_col_Eckardt_points, 6);

	base_curves4[0] = row_col_Eckardt_points[0];
	base_curves4[1] = row_col_Eckardt_points[1];
	base_curves4[2] = row_col_Eckardt_points[3];
	base_curves4[3] = row_col_Eckardt_points[4];
	
	if (f_v) {
		cout << "base_curves4=";
		INT_vec_print(cout, base_curves4, 4);
		cout << endl;
		}


	if (f_v) {
		cout << "Creating the web of cubic curves through the arc:" << endl;
		}
	Surf->create_web_of_cubic_curves_and_equations_based_on_four_tritangent_planes(arc, base_curves4, 
		Web_of_cubic_curves, The_plane_equations, 0 /*verbose_level*/);


	if (f_v) {
		cout << "Testing the web of cubic curves:" << endl;
		}

	INT pt_vec[3];

	for (i = 0; i < 45; i++) {
		//cout << i << " / " << 45 << ":" << endl;
		for (j = 0; j < 6; j++) {
			Surf->P2->unrank_point(pt_vec, arc[j]);
			c = Surf->Poly3->evaluate_at_a_point(Web_of_cubic_curves + i * 10, pt_vec);
			if (c) {
				cout << "the cubic curve does not pass through the arc" << endl;
				exit(1);
				}
			}
		}
	if (f_v) {
		cout << "The cubic curves all pass through the arc" << endl;
		}

	if (f_vv) {
		cout << "Web_of_cubic_curves:" << endl;
		INT_matrix_print(Web_of_cubic_curves, 45, 10);
		}

	if (f_vv) {
		cout << "base_curves4=";
		INT_vec_print(cout, base_curves4, 4);
		cout << endl;
		}


	base_curves = NEW_INT(4 * 10);
	for (i = 0; i < 4; i++) {
		INT_vec_copy(Web_of_cubic_curves + base_curves4[i] * 10, base_curves + i * 10, 10);
		}
	if (f_vv) {
		cout << "base_curves:" << endl;
		INT_matrix_print(base_curves, 4, 10);
		}

	
	
	if (f_vv) {
		cout << "The_plane_equations:" << endl;
		INT_matrix_print(The_plane_equations, 45, 4);
		}


	INT Basis[16];
	for (i = 0; i < 45; i++) {
		INT_vec_copy(The_plane_equations + i * 4, Basis, 4);
		F->RREF_and_kernel(4, 1, Basis, 0 /* verbose_level */);
		The_plane_rank[i] = Surf->rank_plane(Basis + 4);
		}
	if (f_vv) {
		cout << "The_plane_ranks:" << endl;
		print_integer_matrix_with_standard_labels(cout, The_plane_rank, 45, 1, TRUE /* f_tex */);
		}

	for (i = 0; i < 45; i++) {
		The_plane_duals[i] = Surf->rank_point(The_plane_equations + i * 4);
		}

	cout << "computing Dual_point_ranks:" << endl;
	for (i = 0; i < nb_T; i++) {
		//cout << "trihedral pair " << i << " / " << Surf->nb_trihedral_pairs << endl;

		INT e[6];
		
		INT_vec_copy(Surf->Trihedral_to_Eckardt + T_idx[i] * 6, e, 6);
		for (j = 0; j < 6; j++) {
			Dual_point_ranks[i * 6 + j] = The_plane_duals[e[j]];
			}

		}

	if (f_vv) {
		cout << "Dual_point_ranks:" << endl;
		INT_matrix_print(Dual_point_ranks, nb_T, 6);
		}


	if (f_v) {
		cout << "arc_lifting::lift_prepare before Surf->create_lines_from_plane_equations" << endl;
		}
	Surf->create_lines_from_plane_equations(The_plane_equations, Lines27, verbose_level);
	if (f_v) {
		cout << "arc_lifting::lift_prepare after Surf->create_lines_from_plane_equations" << endl;
		}


	if (f_v) {
		cout << "arc_lifting::lift_prepare done" << endl;
		}
}

void arc_lifting::print(ostream &ost)
{
	INT i;

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


	//cout << "arc_lifting_main before print_dual_point_ranks" << endl;
	//print_dual_point_ranks(ost);


	cout << "arc_lifting::print before print_the_six_plane_equations" << endl;
	print_the_six_plane_equations(The_six_plane_equations, planes6, ost);

	cout << "arc_lifting::print before print_surface_equations_on_line" << endl;
	print_surface_equations_on_line(The_surface_equations, 
		lambda, lambda_rk, ost);

	INT *coeffs;
	INT coeffs2[20];

	coeffs = The_surface_equations + lambda_rk * 20;
	INT_vec_copy(coeffs, coeffs2, 20);
	PG_element_normalize_from_front(*F, coeffs2, 1, 20);
	
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

	//cout << "do_arc_lifting before arc_lifting->print_trihedral_pairs" << endl;
	//AL->print_trihedral_pairs(fp);


	ost << "\\bigskip" << endl;
	ost << "The trihedral pair is isomorphic to trihedral pair no " << trihedral_pair_orbit_index << " in the classification." << endl;
	ost << endl;
	ost << "\\bigskip" << endl;
	ost << endl;
	ost << "The stabilizer of the trihedral pair is a group of order " << stabilizer_of_trihedral_pair_go << endl;
	ost << endl;

	ost << "The stabilizer of the trihedral pair is the following group\\\\" << endl;
	stab_gens->print_generators_tex(ost);

	ost << "The orbits of the trihedral pair stabilizer on the $q+1$ surfaces on the line are:\\\\" << endl;
	Orb->print_and_list_orbits_and_stabilizer_sorted_by_length_and_list_stabilizer_elements(ost, TRUE, Surf_A->A, stab_gens);


	ost << "The subgroup which stabilizes the equation has " << cosets->len << " cosets in the stabilizer of the trihedral pair:\\\\" << endl;
	for (i = 0; i < cosets->len; i++) {
		ost << "Coset " << i << " / " << cosets->len << ", coset rep:" << endl;
		ost << "$$" << endl;
		Surf_A->A->element_print_latex(cosets->ith(i), ost);
		ost << "$$" << endl;
		}
	ost << "The stabilizer of the trihedral pair and the equation is the following group\\\\" << endl;
	gens_subgroup->print_generators_tex(ost);

	ost << "The automorphism group consists of the follwing " << coset_reps->len << " cosets\\\\" << endl;
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
	ost << "The automorphism group of the surface has order " << go << "\\\\" << endl;
	Aut_gens->print_generators_tex(ost);

}

void arc_lifting::print_Eckardt_point_data(ostream &ost)
{
#if 0
		INT *bisecants; // [15]
		INT *Intersections; // [15 * 15]
		INT *B_pts; // [nb_B_pts]
		INT *B_pts_label; // [nb_B_pts * 3]
		INT nb_B_pts; // at most 15
		INT *E2; // [6 * 5 * 2] Eckardt points of the second type 
		INT nb_E2; // at most 30
		INT *conic_coefficients; // [6 * 6]
#endif
	print_bisecants(ost);
	print_intersections(ost);
	print_conics(ost);
}

void arc_lifting::print_bisecants(ostream &ost)
{
	INT i, j, h, a;
	
	ost << "The 15 bisecants are:\\\\" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "h & P_iP_j & \\mbox{rank} & \\mbox{line}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (h = 0; h < 15; h++) {
		a = bisecants[h];
		k2ij(h, i, j, 6);
		ost << h << " & P_" << i + 1 << "P_" << j + 1 << " & " << a << " & " << endl;
		ost << "\\left[ " << endl;
		Surf->P2->Grass_lines->print_single_generator_matrix_tex(ost, a);
		ost << "\\right] ";
		ost << "\\\\" << endl; 
		}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
}

void arc_lifting::print_intersections(ostream &ost)
{
	INT labels[15];
	INT fst[1];
	INT len[1];
	fst[0] = 0;
	len[0] = 15;
	INT i;
	
	for (i = 0; i < 15; i++) {
		labels[i] = i;
		}
	ost << "{\\small \\arraycolsep=1pt" << endl;
	ost << "$$" << endl;
	INT_matrix_print_with_labels_and_partition(ost, Intersections, 15, 15, 
		labels, labels, 
		fst, len, 1,  
		fst, len, 1,  
		intersection_matrix_entry_print, (void *) this, 
		TRUE /* f_tex */);
	ost << "$$}" << endl;
}

void arc_lifting::print_conics(ostream &ost)
{
	INT h;
	
	ost << "The 6 conics are:\\\\" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & C_i & \\mbox{equation}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (h = 0; h < 6; h++) {
		ost << h + 1 << " & C_" << h + 1 << " & " << endl;
		Surf->Poly2_x123->print_equation(ost, conic_coefficients + h * 6);
		ost << "\\\\" << endl; 
		}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
}

void arc_lifting::print_Eckardt_points(ostream &ost)
{
	INT s;
	
	ost << "We found " << nb_E << " Eckardt points:\\\\" << endl;
	for (s = 0; s < nb_E; s++) {
		ost << s << " / " << nb_E << " : $";
		E[s].latex(ost);
		ost << "= E_{" << E_idx[s] << "}$\\\\" << endl;
		}
	//ost << "by rank: ";
	//INT_vec_print(ost, E_idx, nb_E);
	//ost << "\\\\" << endl;
}

void arc_lifting::print_web_of_cubic_curves(ostream &ost)
{
	ost << "The web of cubic curves is:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost, Web_of_cubic_curves, 15, 10, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, Web_of_cubic_curves + 15 * 10, 15, 10, 15, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, Web_of_cubic_curves + 30 * 10, 15, 10, 30, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;
}

void arc_lifting::print_trihedral_plane_equations(ostream &ost)
{
	INT *M;
	INT i, j;

	M = NEW_INT(45 * 5);
	for (i = 0; i < 45; i++) {
		for (j = 0; j < 4; j++) {
			M[i * 5 + j] = The_plane_equations[i * 4 + j];
			}
		M[i * 5 + 4] = The_plane_duals[i];
		}
	
	ost << "The chosen abstract trihedral pair is no " << t_idx0 << ":" << endl;
	ost << "$$" << endl;
	Surf->latex_abstract_trihedral_pair(ost, t_idx0);
	ost << "$$" << endl;
	ost << "The six planes in the trihedral pair are:" << endl;
	ost << "$$" << endl;
	INT_vec_print(ost, row_col_Eckardt_points, 6);
	ost << "$$" << endl;
	ost << "We choose planes $0,1,3,4$ for the base curves:" << endl;
	ost << "$$" << endl;
	INT_vec_print(ost, base_curves4, 4);
	ost << "$$" << endl;
	ost << "The four base curves are:\\\\";
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost, base_curves, 4, 10, TRUE /* f_tex*/);
	ost << "$$" << endl;

	ost << "The resulting plane equations and dual point ranks are:\\\\";
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost, M, 15, 5, TRUE /* f_tex*/);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, M + 15 * 5, 15, 5, 15, 0, TRUE /* f_tex*/);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, M + 30 * 5, 15, 5, 30, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;


	print_lines(ost);

	FREE_INT(M);
}

void arc_lifting::print_lines(ostream &ost)
{
	INT i, a;
	INT v[8];
	
	ost << "The 27 lines:\\\\";
	for (i = 0; i < 27; i++) {
		a = Lines27[i];
		ost << "$$" << endl;
		ost << "\\ell_{" << i << "} = " << Surf->Line_label_tex[i] << " = " << a << " = ";
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
	print_integer_matrix_with_standard_labels(ost, Dual_point_ranks, nb_T, 6, TRUE /* f_tex*/);
	ost << "$$" << endl;
}

void arc_lifting::print_FG(ostream &ost)
{
	ost << "$F$-planes:\\\\";
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost, F_plane, 3, 4, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$G$-planes:\\\\";
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost, G_plane, 3, 4, TRUE /* f_tex*/);
	ost << "$$" << endl;
}

void arc_lifting::print_the_six_plane_equations(INT *The_six_plane_equations, INT *plane6, ostream &ost)
{
	INT *M;
	INT i, j;

	M = NEW_INT(6 * 5);
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 4; j++) {
			M[i * 5 + j] = The_six_plane_equations[i * 4 + j];
			}
		M[i * 5 + 4] = plane6[i];
		}
	
	ost << "The six plane equations are (with dual point rank):" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost, M, 6, 5, TRUE /* f_tex*/);
	ost << "$$" << endl;
	FREE_INT(M);
}

void arc_lifting::print_surface_equations_on_line(INT *The_surface_equations, 
	INT lambda, INT lambda_rk, ostream &ost)
{
	ost << "The $q+1$ equations on the line are:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost, The_surface_equations, q + 1, 20, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$$" << endl;
	ost << "\\lambda = " << lambda << ", \\; \\mbox{in row} \\; " << lambda_rk << endl;
	ost << "$$" << endl;
}

void arc_lifting::create_the_six_plane_equations(INT t_idx, INT *The_six_plane_equations, INT *plane6, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "arc_lifting::create_the_six_plane_equations" << endl;
		}


	INT_vec_copy(Surf->Trihedral_to_Eckardt + t_idx * 6, row_col_Eckardt_points, 6);

	INT_vec_copy(The_plane_equations + row_col_Eckardt_points[0] * 4, The_six_plane_equations, 4);
	INT_vec_copy(The_plane_equations + row_col_Eckardt_points[1] * 4, The_six_plane_equations + 4, 4);
	INT_vec_copy(The_plane_equations + row_col_Eckardt_points[2] * 4, The_six_plane_equations + 8, 4);
	INT_vec_copy(The_plane_equations + row_col_Eckardt_points[3] * 4, The_six_plane_equations + 12, 4);
	INT_vec_copy(The_plane_equations + row_col_Eckardt_points[4] * 4, The_six_plane_equations + 16, 4);
	INT_vec_copy(The_plane_equations + row_col_Eckardt_points[5] * 4, The_six_plane_equations + 20, 4);

	if (f_v) {
		cout << "arc_lifting::create_the_six_plane_equations" << endl;
		cout << "The_six_plane_equations=" << endl;
		INT_matrix_print(The_six_plane_equations, 6, 4);
		}

	for (i = 0; i < 6; i++) {
		plane6[i] = Surf->P->rank_point(The_six_plane_equations + i * 4);
		}

	if (f_v) {
		cout << "arc_lifting::create_the_six_plane_equations done" << endl;
		}
}

void arc_lifting::create_surface_from_trihedral_pair_and_arc(INT t_idx, INT *planes6, 
	INT *The_six_plane_equations, INT *The_surface_equations, 
	INT &lambda, INT &lambda_rk, INT verbose_level)
// plane6[6]
// The_six_plane_equations[6 * 4]
// The_surface_equations[(q + 1) * 20]
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_lifting::create_surface_from_trihedral_pair_and_arc" << endl;
		}

	create_the_six_plane_equations(t_idx, The_six_plane_equations, planes6, verbose_level);


	if (f_v) {
		cout << "arc_lifting::create_surface_from_trihedral_pair_and_arc before create_equations_for_pencil_of_surfaces_from_trihedral_pair" << endl;
		}
	Surf->create_equations_for_pencil_of_surfaces_from_trihedral_pair(
		The_six_plane_equations, The_surface_equations, verbose_level);

	if (f_v) {
		cout << "arc_lifting::create_surface_from_trihedral_pair_and_arc before create_lambda_from_trihedral_pair_and_arc" << endl;
		}
	Surf->create_lambda_from_trihedral_pair_and_arc(arc, Web_of_cubic_curves, 
		The_plane_equations, t_idx, lambda, lambda_rk, verbose_level);


	INT_vec_copy(The_surface_equations + lambda_rk * 20, the_equation, 20);

	if (f_v) {
		cout << "arc_lifting::create_surface_from_trihedral_pair_and_arc done" << endl;
		}
}

strong_generators *arc_lifting::create_stabilizer_of_trihedral_pair(INT *planes6, 
	INT &trihedral_pair_orbit_index, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	strong_generators *gens_dual;
	strong_generators *gens;
	longinteger_object go;

	gens = new strong_generators;


	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair" << endl;
		}

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair before Surf_A->identify_trihedral_pair_and_get_stabilizer" << endl;
		}

	gens_dual = Surf_A->Classify_trihedral_pairs->identify_trihedral_pair_and_get_stabilizer(
		planes6, transporter, trihedral_pair_orbit_index, verbose_level);

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair after Surf_A->identify_trihedral_pair_and_get_stabilizer" << endl;
		}
	gens_dual->group_order(go);

	
	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair trihedral_pair_orbit_index=" << trihedral_pair_orbit_index << " group order = " << go << endl;
		}

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair group elements:" << endl;
		gens_dual->print_elements_ost(cout);
		}


	gens->init(Surf_A->A);

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair before gens->init_transposed_group" << endl;
		}
	gens->init_transposed_group(gens_dual, verbose_level);

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair The transposed stabilizer is:" << endl;
		gens->print_generators_tex(cout);
		}

	delete gens_dual;

	if (f_v) {
		cout << "arc_lifting::create_stabilizer_of_trihedral_pair done" << endl;
		}
	return gens;
}

void arc_lifting::create_action_on_equations_and_compute_orbits(INT *The_surface_equations, 
	strong_generators *gens_for_stabilizer_of_trihedral_pair, 
	action *&A_on_equations, schreier *&Orb, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_lifting::create_action_on_equations_and_compute_orbits" << endl;
		}
	
	if (f_v) {
		cout << "arc_lifting::create_action_on_equations_and_compute_orbits before create_action_and_compute_orbits_on_equations" << endl;
		}

	create_action_and_compute_orbits_on_equations(Surf_A->A, Surf->Poly3_4, 
		The_surface_equations, q + 1 /* nb_equations */, gens_for_stabilizer_of_trihedral_pair, 
		A_on_equations, Orb, verbose_level);
		// in ACTION/action_global.C

	if (f_v) {
		cout << "arc_lifting::create_action_on_equations_and_compute_orbits done" << endl;
		}
}

void arc_lifting::create_clebsch_system(INT *The_six_plane_equations, INT lambda, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;

	if (f_v) {
		cout << "arc_lifting::create_clebsch_system" << endl;
		}
	
	INT_vec_copy(The_six_plane_equations, F_plane, 12);
	INT_vec_copy(The_six_plane_equations + 12, G_plane, 12);
	cout << "F_planes:" << endl;
	INT_matrix_print(F_plane, 3, 4);
	cout << "G_planes:" << endl;
	INT_matrix_print(G_plane, 3, 4);

	Surf->compute_nine_lines(F_plane, G_plane, nine_lines, 0 /* verbose_level */);

	if (f_v) {
		cout << "arc_lifting::create_clebsch_system" << endl;
		cout << "The nine lines are: ";
		INT_vec_print(cout, nine_lines, 9);
		cout << endl;
		}

	Surf->prepare_system_from_FG(F_plane, G_plane, lambda, System, verbose_level);

	cout << "The System:" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			INT *p = System + (i * 4 + j) * 3;
			Surf->Poly1->print_equation(cout, p);
			cout << endl;
			}
		}
	if (f_v) {
		cout << "arc_lifting::create_clebsch_system done" << endl;
		}
}

void arc_lifting::loop_over_trihedral_pairs(vector_ge *cosets, vector_ge *&coset_reps, 
	INT *&aut_T_index, INT *&aut_coset_index, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	INT planes6[6];
	INT orbit_index0;
	INT orbit_index;
	INT orbit_length;
	INT Tt[4 * 4 + 1];
	INT Nine_lines0[9];
	INT Nine_lines[9];
	INT *v;
	INT sz;

	if (f_v) {
		cout << "arc_lifting::loop_over_trihedral_pairs" << endl;
		}

	if (f_v) {
		cout << "arc_lifting::loop_over_trihedral_pairs we are considering " << cosets->len << " cosets from the downstep" << endl;
		}


	orbit_length = 0;


	Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(Dual_point_ranks + 0 * 6, 
		transporter0, orbit_index0, verbose_level);

	Surf->compute_nine_lines_by_dual_point_ranks(Dual_point_ranks + 0 * 6, Dual_point_ranks + 0 * 6 + 3, Nine_lines0, 0 /* verbose_level */);
	cout << "The first trihedral pair gives the following nine lines: ";
	INT_vec_print(cout, Nine_lines0, 9);
	cout << endl;

	coset_reps = new vector_ge;
	coset_reps->init(Surf_A->A);
	coset_reps->allocate(nb_T * 2);

	aut_T_index = NEW_INT(nb_T * cosets->len);
	aut_coset_index = NEW_INT(nb_T * cosets->len);

	for (i = 0; i < nb_T; i++) {

		if (f_v) {
			cout << "testing if trihedral pair " << i << " / " << nb_T << " = " << T_idx[i];
			cout << " lies in the orbit:" << endl;
			}

		INT_vec_copy(Dual_point_ranks + i * 6, planes6, 6);

		Surf->compute_nine_lines_by_dual_point_ranks(planes6, planes6 + 3, Nine_lines, 0 /* verbose_level */);

		cout << "The " << i << "-th trihedral pair gives the following nine lines: ";
		INT_vec_print(cout, Nine_lines, 9);
		cout << endl;

		INT_vec_intersect(Nine_lines0, 9, Nine_lines, 9, v, sz);
		cout << "The nine lines of the " << i << "-th trihedral pair intersect the nine lines of the first in " << sz << " lines, which are: ";
		INT_vec_print(cout, v, sz);
		cout << endl;

		Surf->print_trihedral_pair_in_dual_coordinates_in_GAP(planes6, planes6 + 3);
		cout << endl;

		FREE_INT(v);
		


		Surf_A->Classify_trihedral_pairs->identify_trihedral_pair(planes6, 
			transporter, orbit_index, 0 /*verbose_level */);


		if (orbit_index != orbit_index0) {
			if (f_v) {
				cout << "trihedral pair " << i << " / " << nb_T << " and 0 are not isomorphic" << endl;
				}
			continue;
			}
		if (f_v) {
			cout << "trihedral pair " << i << " / " << nb_T << " and 0 are in fact isomorphic" << endl;
			}
		

		Surf_A->A->element_invert(transporter, Elt1, 0);
		Surf_A->A->element_mult(transporter0, Elt1, Elt2, 0);

		for (j = 0; j < cosets->len; j++) {

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

			INT coeff_out[20];


			//Surf_A->A->element_invert(Elt4, Elt5, 0);


			if (mtx->f_semilinear) {
				INT n, frob; //, e;
				
				n = mtx->n;
				frob = Elt4[n * n];
#if 0
				e = mtx->GFq->e;
				if (frob) {
					frob = e - frob;
					}
#endif
				Surf->substitute_semilinear(the_equation, coeff_out, mtx->f_semilinear, frob, Elt4, 0 /* verbose_level */);
				}
			else {
				Surf->substitute_semilinear(the_equation, coeff_out, FALSE, 0, Elt4, 0 /* verbose_level */);
				}

			PG_element_normalize(*F, coeff_out, 1, 20);

			if (f_v) {
				cout << "The transformed equation is:" << endl;
				INT_vec_print(cout, coeff_out, 20);
				cout << endl;
				}


			if (INT_vec_compare(coeff_out, the_equation, 20) == 0) {
				if (f_v) {
					cout << "trihedral pair " << i << " / " << nb_T << " coset " << j << " lies in the orbit, new orbit length is " << orbit_length + 1 << endl;
					cout << "coset rep = " << endl;
					Surf_A->A->element_print_quick(Elt3, cout);
					}
				Surf->compute_nine_lines_by_dual_point_ranks(planes6, planes6 + 3, Nine_lines, 0 /* verbose_level */);
				cout << "The " << orbit_length + 1 << "-th trihedral pair in the orbit gives the following nine lines: ";
				INT_vec_print(cout, Nine_lines, 9);
				cout << endl;


				Surf_A->A->element_move(Elt4, coset_reps->ith(orbit_length), 0);

				aut_T_index[orbit_length] = i;
				aut_coset_index[orbit_length] = j;
				orbit_length++;
				}
			else {
				if (f_v) {
					cout << "trihedral pair " << i << " / " << nb_T << " coset " << j << " does not lie in the orbit" << endl;
					}
				//exit(1);
				}
			} // next j

		} // next i

	coset_reps->reallocate(orbit_length);

	if (f_v) {
		cout << "arc_lifting::loop_over_trihedral_pairs we found an orbit of trihedral pairs of length " << orbit_length << endl;
		//cout << "coset reps:" << endl;
		//coset_reps->print_tex(cout);
		}

	if (f_v) {
		cout << "arc_lifting::loop_over_trihedral_pairs done" << endl;
		}
}


void arc_lifting::create_surface(surface_with_action *Surf_A, INT *Arc6, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT q;
	surface *Surf;

	if (f_v) {
		cout << "arc_lifting::create_surface" << endl;
		}

	q = Surf_A->F->q;
	Surf = Surf_A->Surf;


	cout << "arc_lifting::create_surface before arc_lifting->init" << endl;
	init(Surf_A, Arc6, 6, verbose_level - 2);








	cout << "arc_lifting::create_surface before arc_lifting->lift_prepare" << endl;
	lift_prepare(verbose_level - 2);



	t_idx = T_idx[0];
	The_surface_equations = NEW_INT((q + 1) * 20);
	
	cout << "arc_lifting::create_surface before arc_lifting->create_surface_from_trihedral_pair_and_arc" << endl;
	create_surface_from_trihedral_pair_and_arc(t_idx, planes6, 
		The_six_plane_equations, The_surface_equations, 
		lambda, lambda_rk, verbose_level);

	if (f_v) {
		cout << "lambda = " << lambda << endl;
		cout << "lambda_rk = " << lambda_rk << endl;
		cout << "The six plane equations:" << endl;
		INT_matrix_print(The_six_plane_equations, 6, 4);
		cout << endl;
		cout << "The q+1 surface equations in the pencil:" << endl;
		INT_matrix_print(The_surface_equations, q + 1, 20);
		cout << endl;

		cout << "The surface equation corresponding to lambda = " << lambda << " which is equation number " << lambda_rk << ":" << endl;
		INT_vec_print(cout, The_surface_equations + lambda_rk * 20, 20);
		cout << endl;
		cout << "the_equation:" << endl;
		INT_vec_print(cout, the_equation, 20);
		cout << endl;
		}

	
	cout << "do_arc_lifting before create_clebsch_system" << endl;
	create_clebsch_system(The_six_plane_equations, lambda, 0 /* verbose_level */);



	cout << "arc_lifting::create_surface before arc_lifting->create_stabilizer_of_trihedral_pair" << endl;
	stab_gens = create_stabilizer_of_trihedral_pair(planes6, trihedral_pair_orbit_index, verbose_level);

	stab_gens->group_order(stabilizer_of_trihedral_pair_go);
	cout << "arc_lifting::create_surface the stabilizer of the trihedral pair has order " << stabilizer_of_trihedral_pair_go << endl;



	cout << "arc_lifting::create_surface before AL->create_action_on_equations_and_compute_orbits" << endl;
	create_action_on_equations_and_compute_orbits(The_surface_equations, 
		stab_gens /* strong_generators *gens_for_stabilizer_of_trihedral_pair */, 
		A_on_equations, Orb, 
		verbose_level);

	
	cout << "arc_lifting::create_surface the orbits on the pencil of surfaces are:" << endl;
	Orb->print_and_list_orbits(cout);



	//Surf_A->A->group_order(go_PGL);


	gens_subgroup = Orb->generators_for_stabilizer_of_arbitrary_point_and_transversal(Surf_A->A, 
		stabilizer_of_trihedral_pair_go, lambda_rk /* pt */, cosets, 0 /* verbose_level */);

	cout << "arc_lifting::create_surface we found the following coset representatives:" << endl;
	cosets->print(cout);




	cout << "arc_lifting::create_surface after Orb->generators_for_stabilizer_of_arbitrary_point" << endl;
	gens_subgroup->group_order(stab_order);
	cout << "The stabilizer of the trihedral pair inside the group of the surface has order " << stab_order << endl;

	cout << "arc_lifting::create_surface elements in the stabilizer:" << endl;
	gens_subgroup->print_elements_ost(cout);

	cout << "arc_lifting::create_surface The stabilizer of the trihedral pair inside the stabilizer of the surface is generated by:" << endl;
	gens_subgroup->print_generators_tex(cout);






	Surf->compute_nine_lines_by_dual_point_ranks(Dual_point_ranks + 0 * 6, Dual_point_ranks + 0 * 6 + 3, nine_lines, 0 /* verbose_level */);

	cout << "arc_lifting::create_surface before loop_over_trihedral_pairs" << endl;
	loop_over_trihedral_pairs(cosets, coset_reps, aut_T_index, aut_coset_index, verbose_level);
	cout << "arc_lifting::create_surface after loop_over_trihedral_pairs" << endl;
	cout << "arc_lifting::create_surface we found an orbit of length " << coset_reps->len << endl;
	

	

	{
	longinteger_object ago;
	
	cout << "arc_lifting::create_surface Extending the group:" << endl;
	Aut_gens = new strong_generators;
	Aut_gens->init_group_extension(gens_subgroup, coset_reps, coset_reps->len, verbose_level);

	Aut_gens->group_order(ago);
	cout << "arc_lifting::create_surface The automorphism group has order " << ago << endl;
	cout << "arc_lifting::create_surface The automorphism group is:" << endl;
	Aut_gens->print_generators_tex(cout);
	}
	
	if (f_v) {
		cout << "arc_lifting::create_surface done" << endl;
		}
}

static void intersection_matrix_entry_print(INT *p, INT m, INT n, INT i, INT j, INT val, BYTE *output, void *data)
{
	//arc_lifting *AL;
	//AL = (arc_lifting *) data;
	INT a, b;
	
	if (i == -1) {
		k2ij(j, a, b, 6);
		sprintf(output, "P_%ldP_%ld", a + 1, b + 1);
		}
	else if (j == -1) {
		k2ij(i, a, b, 6);
		sprintf(output, "P_%ldP_%ld", a + 1, b + 1);
		}
	else {
		if (val == -1) {
			strcpy(output, ".");
			}
		else {
			sprintf(output, "%ld", val);
			}
		}
}




