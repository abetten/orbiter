/*
 * semifield_level_two.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace semifields {



semifield_level_two::semifield_level_two()
{
	SC = NULL;
	n = k = k2 = q = 0;

	A = NULL;
	A_PGLk = NULL;
	M = NULL;
	F = NULL;
	C = NULL;
	desired_pivots = NULL;


	R = NULL;
	nb_classes = 0;

	Basis = NULL;
	Mtx = NULL;
	Mtx_Id = NULL;
	Mtx_2 = NULL;
	Elt = NULL;
	Elt2 = NULL;

	class_rep_rank = NULL;
	class_rep_plus_I_rank = NULL;
	class_rep_plus_I_Basis = NULL;
	class_rep_plus_I_Basis_inv = NULL;
	R_i_plus_I_class_idx = NULL;
	Flag_orbit_stabilizer = NULL;
	class_to_flag_orbit = NULL;

	nb_flag_orbits = 0;
	flag_orbit_classes = NULL;
	flag_orbit_number_of_matrices = NULL;
	flag_orbit_length = NULL;

	f_Fusion = NULL;
	Fusion_idx = NULL;
	Fusion_elt = NULL;

	nb_orbits = 0;
	defining_flag_orbit = NULL;
	So = NULL;
	Fo = NULL;
	Pt = NULL;
	Go = NULL;
	Stabilizer_gens = NULL;

	E1 = E2 = E3 = E4 = NULL;
	//Mnn = NULL;

	Mtx1 = Mtx2 = Mtx3 = Mtx4 = Mtx5 = Mtx6 = NULL;
	ELT1 = ELT2 = ELT3 = NULL;
	M1 = NULL;
	Basis1 = Basis2 = NULL;

	R1 = R2 = NULL;

	Candidates = NULL;
	Nb_candidates = NULL;


	//null();
}

semifield_level_two::~semifield_level_two()
{
	int verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two" << endl;
	}
	if (R) {
		FREE_OBJECTS(R);
	}

	if (desired_pivots) {
		FREE_int(desired_pivots);
	}

	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two before Basis" << endl;
	}




	if (Basis) {
		FREE_int(Basis);
	}
	if (Mtx) {
		FREE_int(Mtx);
	}
	if (Mtx_Id) {
		FREE_int(Mtx_Id);
	}
	if (Mtx_2) {
		FREE_int(Mtx_2);
	}
	if (Elt) {
		FREE_int(Elt);
	}
	if (Elt2) {
		FREE_int(Elt2);
	}



	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two before class_rep_rank" << endl;
	}



	if (class_rep_rank) {
		FREE_lint(class_rep_rank);
	}
	if (class_rep_plus_I_rank) {
		FREE_lint(class_rep_plus_I_rank);
	}
	if (class_rep_plus_I_Basis) {
		for (i = 0; i < nb_classes; i++) {
			FREE_int(class_rep_plus_I_Basis[i]);
		}
		FREE_pint(class_rep_plus_I_Basis);
	}
	if (class_rep_plus_I_Basis_inv) {
		for (i = 0; i < nb_classes; i++) {
			FREE_int(class_rep_plus_I_Basis_inv[i]);
		}
		FREE_pint(class_rep_plus_I_Basis_inv);
	}
	if (R_i_plus_I_class_idx) {
		FREE_int(R_i_plus_I_class_idx);
	}
	if (Flag_orbit_stabilizer) {
		FREE_OBJECTS(Flag_orbit_stabilizer);
	}
	if (class_to_flag_orbit) {
		FREE_int(class_to_flag_orbit);
	}


	if (flag_orbit_classes) {
		FREE_int(flag_orbit_classes);
	}
	if (flag_orbit_number_of_matrices) {
		FREE_int(flag_orbit_number_of_matrices);
	}
	if (flag_orbit_length) {
		FREE_int(flag_orbit_length);
	}

	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two before f_Fusion" << endl;
	}

	if (f_Fusion) {
		FREE_int(f_Fusion);
	}

	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two before Fusion_idx" << endl;
	}


	if (Fusion_idx) {
		FREE_int(Fusion_idx);
	}

	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two before Fusion_elt" << endl;
	}

	if (Fusion_elt) {
		for (i = 0; i < nb_flag_orbits; i++) {
			if (Fusion_elt[i]) {
				FREE_int(Fusion_elt[i]);
			}
		}
		FREE_pint(Fusion_elt);
	}

	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two before defining_flag_orbit" << endl;
	}

	if (defining_flag_orbit) {
		FREE_int(defining_flag_orbit);
	}

	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two before So" << endl;
	}

	if (So) {
		FREE_int(So);
	}

	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two before Fo" << endl;
	}

	if (Fo) {
		FREE_int(Fo);
	}

	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two before Pt" << endl;
	}


	if (Pt) {
		FREE_lint(Pt);
	}

	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two before Go" << endl;
	}


	if (Go) {
		FREE_lint(Go);
	}

	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two before Stabilizer_gens" << endl;
	}

	if (Stabilizer_gens) {
		FREE_OBJECTS(Stabilizer_gens);
	}
	if (E1) {
		FREE_int(E1);
	}
	if (E2) {
		FREE_int(E2);
	}
	if (E3) {
		FREE_int(E3);
	}
	if (E4) {
		FREE_int(E4);
	}
#if 0
	if (Mnn) {
		FREE_int(Mnn);
	}
#endif
	if (Mtx1) {
		FREE_int(Mtx1);
	}
	if (Mtx2) {
		FREE_int(Mtx2);
	}
	if (Mtx3) {
		FREE_int(Mtx3);
	}
	if (Mtx4) {
		FREE_int(Mtx4);
	}
	if (Mtx5) {
		FREE_int(Mtx5);
	}
	if (Mtx6) {
		FREE_int(Mtx6);
	}
	if (ELT1) {
		FREE_int(ELT1);
	}
	if (ELT2) {
		FREE_int(ELT2);
	}
	if (ELT3) {
		FREE_int(ELT3);
	}
	if (M1) {
		FREE_int(M1);
	}
	if (Basis1) {
		FREE_int(Basis1);
	}
	if (Basis2) {
		FREE_int(Basis2);
	}
	if (R1) {
		FREE_OBJECT(R1);
	}
	if (R2) {
		FREE_OBJECT(R2);
	}
	if (Candidates) {
		for (i = 0; i < nb_orbits; i++) {
			FREE_lint(Candidates[i]);
		}
		FREE_plint(Candidates);
	}
	if (Nb_candidates) {
		FREE_int(Nb_candidates);
	}

	if (f_v) {
		cout << "semifield_level_two::~semifield_level_two done" << endl;
	}

	//freeself();
}

void semifield_level_two::init(
		semifield_classify *SC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_object Go;

	if (f_v) {
		cout << "semifield_level_two::init" << endl;
	}
	semifield_level_two::SC = SC;
	k = SC->k;
	n = 2 * k;
	k2 = k * k;
	F = SC->Mtx->GFq;
	q = SC->q;

	A = SC->A;

	A_PGLk = SC->A0_linear;

	A_PGLk->print_base();
	A_PGLk->group_order(Go);

	if (f_v) {
		cout << "semifield_level_two::init "
				"the order of GL(" << k << "," << q << ") "
						"is " << Go << endl;
	}

	M = A_PGLk->G.matrix_grp;

	if (f_v) {
		cout << "semifield_level_two::init "
				"before M->init_gl_classes M->n=" << M->n << endl;
	}
	M->init_gl_classes(0 /*verbose_level*/);
	if (f_v) {
		cout << "semifield_level_two::init "
				"after M->init_gl_classes" << endl;
	}
	C = M->C;

	Basis = NEW_int(k * k);
	Mtx = NEW_int(k * k);
	Mtx_Id = NEW_int(k * k);
	Mtx_2 = NEW_int(k * k);
	Elt = NEW_int(A_PGLk->elt_size_in_int);
	Elt2 = NEW_int(A_PGLk->elt_size_in_int);

	F->Linear_algebra->identity_matrix(Mtx_Id, k);


	E1 = NEW_int(A_PGLk->elt_size_in_int);
	E2 = NEW_int(A_PGLk->elt_size_in_int);
	E3 = NEW_int(A_PGLk->elt_size_in_int);
	E4 = NEW_int(A_PGLk->elt_size_in_int);

	//Mnn = NEW_int(n * n);

	Mtx1 = NEW_int(k2);
	Mtx2 = NEW_int(k2);
	Mtx3 = NEW_int(k2);
	Mtx4 = NEW_int(k2);
	Mtx5 = NEW_int(k2);
	Mtx6 = NEW_int(k2);

	ELT1 = NEW_int(A->elt_size_in_int);
	ELT2 = NEW_int(A->elt_size_in_int);
	ELT3 = NEW_int(A->elt_size_in_int);

	M1 = NEW_int(n * n);
	Basis1 = NEW_int(k * k);
	Basis2 = NEW_int(k * k);

	R1 = NEW_OBJECT(linear_algebra::gl_class_rep);
	R2 = NEW_OBJECT(linear_algebra::gl_class_rep);

	init_desired_pivots(verbose_level - 1);

	if (f_v) {
		cout << "semifield_starter::init done" << endl;
	}
}


void semifield_level_two::init_desired_pivots(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "semifield_level_two::init_desired_pivots" << endl;
	}
	desired_pivots = NEW_int(k);

	for (i = 0; i < k; i++) {
		if (i < 2) {
			desired_pivots[i] = i * k;
		}
		else {
			desired_pivots[i] = (k - 1 - (i - 2)) * k;
		}
	}
	if (f_vv) {
		cout << "semifield_level_two::init_desired_pivots "
				"desired_pivots: ";
		Int_vec_print(cout, desired_pivots, k);
		cout << endl;
	}
	if (f_v) {
		cout << "semifield_level_two::init_desired_pivots done" << endl;
	}
}

void semifield_level_two::list_all_elements_in_conjugacy_class(
		int c, int verbose_level)
// This function lists all elements in a conjugacy class
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	groups::strong_generators *Centralizer_gens;
	groups::sims *G;
	groups::sims *U;
	ring_theory::longinteger_object Go, Co, Cl;
	ring_theory::longinteger_domain D;
	int rk, cl, r;

	if (f_v) {
		cout << "semifield_starter::list_all_elements_in_conjugacy_class "
				"c=" << c << endl;
	}

	Centralizer_gens = NEW_OBJECT(groups::strong_generators);

	C->make_matrix_from_class_rep(Mtx, R + c, 0 /*verbose_level - 1 */);

	A_PGLk->Group_element->make_element(Elt, Mtx, 0);

	Centralizer_gens->init_centralizer_of_matrix(
			A_PGLk, Mtx, verbose_level - 2);

	if (f_vv) {
		cout << "creating sims:" << endl;
	}
	U = Centralizer_gens->create_sims(0 /* verbose_level */);
	U->group_order(Co);

	if (f_vv) {
		cout << "Sims object for centralizer of order "
				<< Co << " has been created, transversal lengths are ";
		U->print_transversal_lengths();
	}

	G = A_PGLk->Sims;
	G->group_order(Go);
	D.integral_division_exact(Go, Co, Cl);

	if (f_vv) {
		cout << "Class " << c << " has length " << Cl << ":" << endl;
	}
	cl = Cl.as_int();

	int *Ranks;

	Ranks = NEW_int(cl);
	for (rk = 0; rk < cl; rk++) {

		if (f_v) {
			cout << "element " << rk << " / " << cl << ":" << endl;
		}

		A_PGLk->coset_unrank(G, U, rk, E1, 0 /* verbose_level */);

		A_PGLk->Group_element->element_invert(E1, E2, 0);


		A_PGLk->Group_element->element_mult(E2, Elt, E3, 0);
		A_PGLk->Group_element->element_mult(E3, E1, E4, 0);
		A_PGLk->Group_element->element_print(E4, cout);

		r = SC->matrix_rank(E4); // G->element_rank_int(E4);
		if (f_v) {
			cout << "Has rank " << r << endl;
		}
		Ranks[rk] = r;

		if (f_v) {
			cout << "Coset representative:" << endl;
			A_PGLk->Group_element->element_print(E1, cout);
			cout << endl;
		}
	}

	cout << "The elements of class " << c << " are: ";
	Int_vec_print(cout, Ranks, cl);
	cout << endl;

	FREE_int(Ranks);
	FREE_OBJECT(U);
	FREE_OBJECT(Centralizer_gens);
	if (f_v) {
		cout << "semifield_level_two::list_all_elements_in_conjugacy_class done" << endl;
	}
}

void semifield_level_two::compute_level_two(
		int nb_stages, int verbose_level)
// nb_stages = 4 for everything
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_level_two::compute_level_two, nb_stages=" << nb_stages << endl;
	}

	if (nb_stages >= 1) {
		if (f_v) {
			cout << "semifield_level_two::compute_level_two "
					"before downstep" << endl;
		}
		downstep(verbose_level - 1);
		if (f_v) {
			cout << "semifield_level_two::compute_level_two "
					"after downstep, nb_flag_orbits=" << nb_flag_orbits << endl;
		}
	}


	if (nb_stages >= 2) {
		if (f_v) {
			cout << "semifield_level_two::compute_level_two compute the stabilizers." << endl;
		}



		if (f_v) {
			cout << "semifield_level_two::compute_level_two "
					"before compute_stabilizers_downstep" << endl;
		}
		compute_stabilizers_downstep(verbose_level - 2);
		if (f_v) {
			cout << "semifield_level_two::compute_level_two "
					"after compute_stabilizers_downstep" << endl;
		}
	}

	if (nb_stages >= 3) {
		if (f_v) {
			cout << "semifield_level_two::compute_level_two "
					"before upstep" << endl;
		}
		upstep(verbose_level - 1);
		if (f_v) {
			cout << "semifield_level_two::compute_level_two "
					"after upstep, nb_orbits=" << nb_orbits << endl;
		}
	}

	if (nb_stages >= 4) {
		if (f_v) {
			cout << "semifield_level_two::compute_level_two "
					"before write_level_info_file" << endl;
		}
		write_level_info_file(verbose_level);
		if (f_v) {
			cout << "semifield_level_two::compute_level_two "
					"after write_level_info_file" << endl;
		}
	}

	if (f_v) {
		cout << "semifield_level_two::compute_level_two done" << endl;
	}
}



void semifield_level_two::downstep(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, idx;

	if (f_v) {
		cout << "semifield_level_two::downstep" << endl;
	}

	if (f_v) {
		cout << "semifield_level_two::downstep "
				"before make_classes" << endl;
	}


	if (C == NULL) {
		cout << "semifield_level_two::downstep "
				"C == NULL" << endl;
		exit(1);
	}
	C->make_classes(R, nb_classes,
			true /* f_no_eigenvalue_one */, 0 /*verbose_level - 1*/);

	if (f_v) {
		cout << "semifield_level_two::downstep after "
				"C->make_classes nb_classes = " << nb_classes << endl;
	}

#if 0
	if (f_write_class_reps) {

		const char *fname = "Class_reps.tex";
		C->report(fname, verbose_level);

#if 0
		const char *fname = "Class_reps.tex";
		ofstream fp(fname);
		int i;

		latex_head_easy(fp);
		fp << "\\section{Conjugacy Classes}" << endl;
		for (i = 0; i < nb_classes; i++) {
			fp << "Representative " << i << " / "
					<< nb_classes << endl;
			C->print_matrix_and_centralizer_order_latex(fp, R + i);
		}
		latex_foot(fp);
#endif
	}
#endif

	if (f_v) {
		cout << "semifield_level_two::downstep "
				"after make_classes" << endl;
	}


	class_rep_rank = NEW_lint(nb_classes);
	class_rep_plus_I_rank = NEW_lint(nb_classes);
	R_i_plus_I_class_idx = NEW_int(nb_classes);
	class_rep_plus_I_Basis = NEW_pint(nb_classes);
	class_rep_plus_I_Basis_inv = NEW_pint(nb_classes);

	for (i = 0; i < nb_classes; i++) {

		if (f_vv) {
			cout << "semifield_level_two::downstep "
					"class " << i << " / " << nb_classes << " :" << endl;
		}

		C->make_matrix_from_class_rep(Mtx, R + i, 0 /*verbose_level - 1 */);

		if (f_vv) {
			cout << "representative:" << endl;
			Int_matrix_print(Mtx, k, k);
		}

		class_rep_rank[i] = SC->matrix_rank(Mtx);
		F->Linear_algebra->add_vector(Mtx, Mtx_Id, Mtx_2, k * k);


		class_rep_plus_I_rank[i] = SC->matrix_rank(Mtx_2);

		linear_algebra::gl_class_rep *R2;

		R2 = NEW_OBJECT(linear_algebra::gl_class_rep);

		class_rep_plus_I_Basis[i] = NEW_int(k * k);
		class_rep_plus_I_Basis_inv[i] = NEW_int(k * k);

		if (false) {
			cout << "semifield_level_two::downstep "
					"class " << i << " before identify_matrix" << endl;
		}
		C->identify_matrix(Mtx_2, R2,
				class_rep_plus_I_Basis[i],
				verbose_level - 1);
		if (f_vv) {
			cout << "class_rep_plus_I_Basis[i]" << endl;
			Int_matrix_print(class_rep_plus_I_Basis[i], k, k);
		}

		R_i_plus_I_class_idx[i] = C->find_class_rep(
				R, nb_classes, R2, 0 /* verbose_level */);
		if (f_vv) {
			cout << "R_i_plus_I_class_idx[i]="
					<< R_i_plus_I_class_idx[i] << endl;
		}

		F->Linear_algebra->matrix_inverse(class_rep_plus_I_Basis[i],
				class_rep_plus_I_Basis_inv[i], k, 0 /*verbose_level */);
		if (f_vv) {
			cout << "class_rep_plus_I_Basis_inv[i]" << endl;
			Int_matrix_print(class_rep_plus_I_Basis_inv[i], k, k);
		}

		if (f_vv) {

			int f_elements_exponential = false;
			string symbol_for_print;


			symbol_for_print.assign("\\alpha");

			cout << "Representative R_i of class " << i
					<< " / " << nb_classes << " has rank "
					<< class_rep_rank[i] << " R_i + I has rank "
					<< class_rep_plus_I_rank[i] << " and belongs "
					"to conjugacy class "
					<< R_i_plus_I_class_idx[i] << endl;
#if 0
			cout << "R_i:" << endl;
			int_matrix_print(Elt, k, k);
			cout << "R_i + I:" << endl;
			int_matrix_print(Elt2, k, k);
#endif
			cout << "class_rep_plus_I_Basis_inv[i]:" << endl;
			Int_matrix_print(class_rep_plus_I_Basis_inv[i], k, k);
			cout << "$$" << endl;
			cout << "\\left[" << endl;
			F->Io->latex_matrix(cout, f_elements_exponential,
					symbol_for_print,
					class_rep_plus_I_Basis_inv[i], k, k);
			cout << "\\right]" << endl;
			cout << "$$" << endl;

		}

		FREE_OBJECT(R2);

	}



	class_to_flag_orbit = NEW_int(nb_classes);
	flag_orbit_number_of_matrices = NEW_int(nb_classes);
	flag_orbit_length = NEW_int(nb_classes);
	flag_orbit_classes = NEW_int(nb_classes * 2);
	nb_flag_orbits = 0;
	for (i = 0; i < nb_classes; i++) {
		if (R_i_plus_I_class_idx[i] < i) {
			continue;
		}
		flag_orbit_classes[nb_flag_orbits * 2 + 0] = i;
		flag_orbit_classes[nb_flag_orbits * 2 + 1] = R_i_plus_I_class_idx[i];
		class_to_flag_orbit[i] = nb_flag_orbits;
		class_to_flag_orbit[R_i_plus_I_class_idx[i]] = nb_flag_orbits;
		if (flag_orbit_classes[nb_flag_orbits * 2 + 0] ==
			flag_orbit_classes[nb_flag_orbits * 2 + 1]) {
			// R_i+I belongs to the conjugacy class of R_i:
			idx = flag_orbit_classes[nb_flag_orbits * 2 + 0];
			flag_orbit_number_of_matrices[nb_flag_orbits] =
					R[idx].class_length->as_int();
			flag_orbit_length[nb_flag_orbits] =
					R[idx].class_length->as_int() >> 1;
		}
		else {
			// R_i+I belongs to a different conjugacy class than R_i:
			flag_orbit_number_of_matrices[nb_flag_orbits] = 0;
			for (j = 0; j < 2; j++) {
				idx = flag_orbit_classes[nb_flag_orbits * 2 + j];
				flag_orbit_number_of_matrices[nb_flag_orbits] +=
						R[idx].class_length->as_int();
			}
			idx = flag_orbit_classes[nb_flag_orbits * 2 + 0];
			flag_orbit_length[nb_flag_orbits] =
					R[idx].class_length->as_int();
		}
		nb_flag_orbits++;
	}



	if (f_v) {
		cout << "semifield_level_two::downstep done" << endl;
	}
}


void semifield_level_two::compute_stabilizers_downstep(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int i, j, a, b;

	if (f_v) {
		cout << "semifield_level_two::compute_stabilizers_downstep, "
				"verbose_level=" << verbose_level << endl;
	}

	Flag_orbit_stabilizer = NEW_OBJECTS(groups::strong_generators, nb_flag_orbits);
	for (i = 0; i < nb_flag_orbits; i++) {

		if (f_vv) {
			cout << "semifield_level_two::compute_stabilizers_downstep "
					"down orbit " << i << " / " << nb_flag_orbits << endl;
		}

		a = flag_orbit_classes[i * 2 + 0];
		b = flag_orbit_classes[i * 2 + 1];

		C->make_matrix_from_class_rep(Mtx, R + a, 0 /*verbose_level - 1 */);

		//A_PGLk->make_element(Elt, Mtx, 0);

		if (f_vv) {
			cout << "semifield_level_two::compute_stabilizers_downstep "
					"down orbit " << i << " / " << nb_flag_orbits
					<< " before Flag_orbit_stabilizer[i].init_centralizer_of_matrix" << endl;
		}
		Flag_orbit_stabilizer[i].init_centralizer_of_matrix(
				A_PGLk, Mtx, verbose_level - 3);

		if (f_vv) {
			cout << "semifield_level_two::compute_stabilizers_downstep "
					"down orbit " << i << " / " << nb_flag_orbits
					<< " after Flag_orbit_stabilizer[i].init_centralizer_of_matrix" << endl;
		}

		if (false) {
			cout << "centralizer:" << endl;
			Flag_orbit_stabilizer[i].print_generators(cout, verbose_level - 1);
		}

		if (a == b) {
			if (f_vv) {
				cout << "semifield_level_two::compute_stabilizers_downstep "
						"down orbit " << i << " / " << nb_flag_orbits
						<< " a == b, extending Flag_orbit_stabilizer[i]" << endl;
			}

			linear_algebra::gl_class_rep *R2;

			R2 = NEW_OBJECT(linear_algebra::gl_class_rep);

			F->Linear_algebra->add_vector(Mtx, Mtx_Id, Mtx_2, k * k);

			C->identify_matrix(Mtx_2, R2, Basis, verbose_level - 3);

			A_PGLk->Group_element->make_element(Elt, Basis, 0);

			Flag_orbit_stabilizer[i].add_single_generator(Elt,
					2 /* group_index */, verbose_level - 3);

			FREE_OBJECT(R2);

			if (f_vv) {
				cout << "semifield_level_two::compute_stabilizers_downstep "
						"down orbit " << i << " / " << nb_flag_orbits
						<< " a == b, extending Flag_orbit_stabilizer[i] done" << endl;
			}

		}
	}

	if (f_v) {
		cout << "semifield_level_two::compute_stabilizers_downstep "
				"We found " << nb_flag_orbits << " down orbits:" << endl;
		for (i = 0; i < nb_flag_orbits; i++) {
			ring_theory::longinteger_object go;

			Flag_orbit_stabilizer[i].group_order(go);
			for (j = 0; j < 2; j++) {
				cout << flag_orbit_classes[i * 2 + j] << " ";
			}
			cout << " : nb of matrices = "
					<< flag_orbit_number_of_matrices[i]
					<< " : length = " << flag_orbit_length[i]
					<< " stab order = " << go;
			cout << endl;
		}
		cout << "i : class_to_flag_orbit" << endl;
		for (i = 0; i < nb_classes; i++) {
			cout << i << " : " << class_to_flag_orbit[i] << endl;
		}
	}
	if (f_vv) {
		cout << "Stabilizers of middle object:" << endl;
		for (i = 0; i < nb_flag_orbits; i++) {
			ring_theory::longinteger_object go;

			Flag_orbit_stabilizer[i].group_order(go);
			cout << "flag orbit " << i << " / " << nb_flag_orbits
				<< " has stabilizer order " << go << endl;
			if (false) {
				Flag_orbit_stabilizer[i].print_generators(cout, verbose_level - 1);
			}
		}
	}
	if (f_v) {
		cout << "semifield_level_two::compute_stabilizers_downstep "
				"done" << endl;
	}
}


void semifield_level_two::upstep(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int f, coset, idx, i;
	long int a, b;

	if (f_v) {
		cout << "semifield_level_two::upstep "
				"nb_flag_orbits = " << nb_flag_orbits
				<< " verbose_level = " << verbose_level << endl;
	}

	f_Fusion = NEW_int(nb_flag_orbits);
	Fusion_idx = NEW_int(nb_flag_orbits);
	Fusion_elt = NEW_pint(nb_flag_orbits);
	Stabilizer_gens = NEW_OBJECTS(groups::strong_generators, nb_flag_orbits);
	for (i = 0; i < nb_flag_orbits; i++) {
		f_Fusion[i] = false;
		Fusion_idx[i] = -1;
		Fusion_elt[i] = NULL;
	}

	defining_flag_orbit = NEW_int(nb_flag_orbits);
	nb_orbits = 0;

	if (f_vv) {
		cout << "semifield_level_two::upstep "
				" nb_flag_orbits = " << nb_flag_orbits << endl;
	}

	for (f = 0; f < nb_flag_orbits; f++) {

		if (f_vv) {
			cout << "semifield_level_two::upstep "
					"working on flag orbit " << f << " / " << nb_flag_orbits << endl;
		}

		if (Fusion_idx[f] >= 0) {
			continue;
		}

		Fusion_idx[f] = nb_orbits;

		idx = flag_orbit_classes[f * 2 + 0];
		a = class_rep_rank[idx];
		b = class_rep_plus_I_rank[idx];

		defining_flag_orbit[nb_orbits] = f; // !!!
		if (f_vv) {
			cout << "working on new up orbit " << nb_orbits
					<< " copying stabilizer over idx=" << idx << endl;
		}

		if (f_vv) {
			cout << "semifield_level_two::upstep "
					"working on flag orbit " << f << " / " << nb_flag_orbits
					<< " before setup_stabilizer" << endl;
		}

		Flag_orbit_stabilizer[f].diagonally_repeat(
				A,
				&Stabilizer_gens[nb_orbits],
				verbose_level - 3);

#if 0
		setup_stabilizer(&Flag_orbit_stabilizer[f],
				&Stabilizer_gens[nb_orbits],
				verbose_level - 3);
#endif

		// turns k x k matrices into n x n matrices
		// by repeating each matrix twice on the diagonal

		if (f_vv) {
			cout << "semifield_level_two::upstep "
					"working on flag orbit " << f << " / " << nb_flag_orbits
					<< " after setup_stabilizer" << endl;
		}


		if (f_vv) {
			cout << "semifield_level_two::upstep f=" << f << endl;
		}

		int **Aut_gens;
		int nb_aut_gens;
		Aut_gens = NEW_pint(2);
		nb_aut_gens = 0;

		for (coset = 0; coset < 2; coset++) {

			int f_automorphism;
			int *Aut;

			if (f_vv) {
				cout << "semifield_level_two::upstep "
						"f=" << f << " / " << nb_flag_orbits
						<< " coset=" << coset << " / " << 2 << endl;
			}
			if (coset == 0) {
				if (f_vv) {
					cout << "semifield_level_two::upstep "
							"f=" << f << " / " << nb_flag_orbits
							<< " coset=" << coset << " / " << 2 << " before trace(a, b)" << endl;
				}
				trace(f, coset, a, b,
						f_automorphism, Aut, verbose_level - 3);
				if (f_vv) {
					cout << "semifield_level_two::upstep "
							"f=" << f << " / " << nb_flag_orbits
							<< " coset=" << coset << " / " << 2 << " after trace" << endl;
				}
			}
			else {
				if (f_vv) {
					cout << "semifield_level_two::upstep "
							"f=" << f << " / " << nb_flag_orbits
							<< " coset=" << coset << " / " << 2 << " before trace(b, a)" << endl;
				}
				trace(f, coset, b, a,
						f_automorphism, Aut, verbose_level - 3);
				if (f_vv) {
					cout << "semifield_level_two::upstep "
							"f=" << f << " / " << nb_flag_orbits
							<< " coset=" << coset << " / " << 2 << " after trace" << endl;
				}
			}
			if (f_automorphism) {
				if (f_vv) {
					cout << "semifield_level_two::upstep "
							"f=" << f << " / " << nb_flag_orbits
							<< " coset=" << coset << " / " << 2 << " found automorphism" << endl;
				}
				Aut_gens[nb_aut_gens++] = Aut;
			}
		}

		if (f_vv) {
			cout << "semifield_level_two::upstep After tracing, we found " << nb_aut_gens
					<< " automorphisms" << endl;
		}

		data_structures_groups::vector_ge *coset_reps;

		if (f_vv) {
			cout << "semifield_level_two::upstep After tracing, "
					"creating coset reps" << endl;
		}
		coset_reps = NEW_OBJECT(data_structures_groups::vector_ge);
		coset_reps->init(A, verbose_level - 2);
		coset_reps->allocate(nb_aut_gens + 1, verbose_level - 2);
		A->Group_element->element_one(coset_reps->ith(0), 0);
		for (i = 0; i < nb_aut_gens; i++) {
			A->Group_element->element_move(Aut_gens[i], coset_reps->ith(i + 1), 0);
			FREE_int(Aut_gens[i]);
		}
		FREE_pint(Aut_gens);

		if (f_vv) {
			cout << "semifield_level_two::upstep We are now extending the group by a factor "
					"of " << nb_aut_gens + 1 << ":" << endl;
		}

		Stabilizer_gens[nb_orbits].add_generators(
				coset_reps, nb_aut_gens + 1, verbose_level - 3);

		FREE_OBJECT(coset_reps);


		ring_theory::longinteger_object go;

		Stabilizer_gens[nb_orbits].group_order(go);
		if (f_vv) {
			cout << "semifield_level_two::upstep The new group order is " << go << endl;
		}
		if (false) {
			cout << "semifield_level_two::upstep generators:" << endl;
			Stabilizer_gens[nb_orbits].print_generators(cout, verbose_level - 1);
		}



		nb_orbits++;
	}

	//Po = NEW_int(nb_orbits); // level two does not have Po[]
	So = NEW_int(nb_orbits);
	Fo = NEW_int(nb_orbits);
	Go = NEW_lint(nb_orbits);
	Pt = NEW_lint(nb_orbits);


	for (i = 0; i < nb_orbits; i++) {
		ring_theory::longinteger_object go, go1;

		Stabilizer_gens[i].group_order(go);

		f = defining_flag_orbit[i];
		idx = flag_orbit_classes[f * 2 + 0];
		a = class_rep_rank[idx];
		b = class_rep_plus_I_rank[idx];
		Fo[i] = f;
		So[i] = idx;
		Go[i] = go.as_lint();
		Pt[i] = a;
	}



	if (f_v) {
		cout << "semifield_level_two::upstep We found "
				<< nb_orbits << " orbits at level two" << endl;
	}
	if (f_vv) {
		cout << "i : defining_flag_orbit[i] : go" << endl;
		for (i = 0; i < nb_orbits; i++) {
			ring_theory::longinteger_object go, go1;
			int *Mtx;

			Stabilizer_gens[i].group_order(go);
			cout << i << " : " << defining_flag_orbit[i] << " : " << go << endl;

			f = defining_flag_orbit[i];

			idx = flag_orbit_classes[f * 2 + 0];
			a = class_rep_rank[idx];
			b = class_rep_plus_I_rank[idx];

			Mtx = NEW_int(k2);
			SC->matrix_unrank(a, Mtx);
			cout << "semifield_level_two::upstep The representative of class " << idx
					<< " is the following matrix of rank " << a << endl;
			Int_matrix_print(Mtx, k, k);
			cout << "semifield_level_two::upstep The stabilizer has order " << go << endl;
			if (false) {
				cout << "The stabilizer is generated by the "
						"following matrices:" << endl;
				Stabilizer_gens[i].print_generators(cout, verbose_level - 1);
			}


#if 0
			cout << "creating sims:" << endl;
			S = Stabilizer_gens[i].create_sims(0 /* verbose_level */);
			S->group_order(go1);

			cout << "Sims object for group of order " << go1
					<< " has been created, transversal lengths are ";
			S->print_transversal_lengths();
			cout << endl;

			FREE_OBJECT(S);
#endif

			FREE_int(Mtx);
		}
	}


	if (f_v) {
		cout << "semifield_level_two::upstep summary of "
				"fusion:" << endl;
		for (f = 0; f < nb_flag_orbits; f++) {
			if (f_Fusion[f]) {
				cout << "down orbit " << f << " is fused to down "
						"orbit " << Fusion_idx[f] << " under "
						"the element" << endl;
				if (false) {
					cout << "Fusion element:" << endl;
					A->Group_element->element_print(Fusion_elt[f], cout);
				}
			}
			else {
				cout << "down orbit " << f << " is associated "
					"to new orbit " << Fusion_idx[f] << endl;
			}
		}
	}
	if (f_v) {
		cout << "semifield_level_two::upstep done" << endl;
	}
}

void semifield_level_two::trace(
		int f, int coset,
		long int a, long int b, int &f_automorphism, int *&Aut,
		int verbose_level)
// a and b are the ranks of two matrices whose span we consider.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	int idx1, idx2;
	int c1, c2;
	long int rk1, rk2;
	int d1, d2, d, cc1, cc2;

	if (f_v) {
		cout << "semifield_level_two::trace a=" << a << " b=" << b << " f=" << f << endl;
	}

	f_automorphism = false;

	c1 = flag_orbit_classes[f * 2 + 0];
	c2 = flag_orbit_classes[f * 2 + 1];
	rk1 = class_rep_rank[c1];
	rk2 = class_rep_plus_I_rank[c1];
	if (f_vv) {
		cout << "semifield_level_two::trace c1=" << c1 << " c2=" << c2
			<< " rk1=" << rk1 << " rk2=" << rk2
			<< " a=" << a << " b=" << b << endl;
	}



	SC->matrix_unrank(a, Mtx1);
	SC->matrix_unrank(b, Mtx2);
	F->Linear_algebra->identity_matrix(Mtx3, k);

	Int_vec_zero(M1, n * n);
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			M1[i * n + j] = Mtx1[i * k + j];
		}
	}
	for (i = k; i < n; i++) {
		M1[i * n + i] = 1;
	}
	if (f_vv) {
		cout << "semifield_level_two::trace transformation matrix Mtx1=" << endl;
		Int_matrix_print(Mtx1, k, k);
		cout << "semifield_level_two::trace transformation matrix M1=" << endl;
		Int_matrix_print(M1, n, n);
	}

	A->Group_element->make_element(ELT1, M1, 0);
	if (f_vv) {
		cout << "semifield_level_two::trace ELT1=" << endl;
		A->Group_element->print_quick(cout, ELT1);
	}

	SC->A_on_S->compute_image_low_level(ELT1, Mtx1, Mtx4, 0 /* verbose_level */);
	SC->A_on_S->compute_image_low_level(ELT1, Mtx2, Mtx5, 0 /* verbose_level */);
	SC->A_on_S->compute_image_low_level(ELT1, Mtx3, Mtx6, 0 /* verbose_level */);
	//cout << "transformation matrix Mtx4=" << endl;
	//int_matrix_print(Mtx4, k, k);
	if (!F->Linear_algebra->is_identity_matrix(Mtx4, k)) {
		cout << "semifield_level_two::trace Mtx4 "
				"is not the identity matrix" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "semifield_level_two::trace after transform Mtx4=" << endl;
		Int_matrix_print(Mtx4, k, k);
		cout << "semifield_level_two::trace after transform Mtx5=" << endl;
		Int_matrix_print(Mtx5, k, k);
		cout << "semifield_level_two::trace after transform Mtx6=" << endl;
		Int_matrix_print(Mtx6, k, k);
	}

	if (f_v) {
		cout << "semifield_level_two::trace before identify_matrix Mtx5" << endl;
	}
	C->identify_matrix(Mtx5, R1, Basis1, verbose_level);
	if (f_v) {
		cout << "semifield_level_two::trace before identify_matrix Mtx6" << endl;
	}
	C->identify_matrix(Mtx6, R2, Basis2, verbose_level);

	if (f_v) {
		cout << "semifield_level_two::trace before find_class_rep R1" << endl;
	}
	idx1 = C->find_class_rep(R, nb_classes, R1, 0 /* verbose_level */);
	if (f_v) {
		cout << "semifield_level_two::trace before find_class_rep R2" << endl;
	}
	idx2 = C->find_class_rep(R, nb_classes, R2, 0 /* verbose_level */);

	if (f_vv) {
		cout << "semifield_level_two::trace f=" << f
			<< " c1=" << c1 << " c2=" << c2 << " coset=" << coset
			<< " rk1=" << rk1 << " rk2=" << rk2
			<< " idx1 = " << idx1 << " idx2 = " << idx2 << endl;
	}


	if (idx1 == c1 || idx1 == c2) {
		if (f_vv) {
			cout << "semifield_level_two::trace automorphism" << endl;
		}

		multiply_to_the_right(ELT1,
				Basis1, ELT2, ELT3, 0 /* verbose_level */);

		// check
		SC->A_on_S->compute_image_low_level(ELT3, Mtx1, Mtx4, 0 /* verbose_level */);
		SC->A_on_S->compute_image_low_level(ELT3, Mtx2, Mtx5, 0 /* verbose_level */);
		SC->A_on_S->compute_image_low_level(ELT3, Mtx3, Mtx6, 0 /* verbose_level */);

		if (f_vv) {
			cout << "semifield_level_two::trace after transform (2) Mtx4=" << endl;
			Int_matrix_print(Mtx4, k, k);
			cout << "semifield_level_two::trace after transform (2) Mtx5=" << endl;
			Int_matrix_print(Mtx5, k, k);
			cout << "semifield_level_two::trace after transform (2) Mtx6=" << endl;
			Int_matrix_print(Mtx6, k, k);
		}


		if (c1 != c2 && idx1 == c2) {
			if(f_vv) {
				cout << "semifield_level_two::trace multiplying Basis_inv to the right" << endl;
			}
			multiply_to_the_right(ELT3,
					class_rep_plus_I_Basis_inv[c1],
					ELT2, ELT1, 0 /* verbose_level */);
		}
		else {
			A->Group_element->element_move(ELT3, ELT1, 0);
		}


		// check
		SC->A_on_S->compute_image_low_level(ELT1, Mtx1, Mtx4, 0 /* verbose_level */);
		SC->A_on_S->compute_image_low_level(ELT1, Mtx2, Mtx5, 0 /* verbose_level */);
		SC->A_on_S->compute_image_low_level(ELT1, Mtx3, Mtx6, 0 /* verbose_level */);

		if (f_vv) {
			cout << "semifield_level_two::trace after transform (3) Mtx4=" << endl;
			Int_matrix_print(Mtx4, k, k);
			cout << "semifield_level_two::trace after transform (3) Mtx5=" << endl;
			Int_matrix_print(Mtx5, k, k);
			cout << "semifield_level_two::trace after transform (3) Mtx6=" << endl;
			Int_matrix_print(Mtx6, k, k);
		}




		if (f_vv) {
			A->Group_element->element_print_quick(ELT1, cout);
		}




#if 0
		image1 = SF->AS->element_image_of(0, ELT1, 0 /* verbose_level */);
		image2 = SF->AS->element_image_of(a, ELT1, 0 /* verbose_level */);
		image3 = SF->AS->element_image_of(b, ELT1, 0 /* verbose_level */);
		if (f_v) {
			cout << "images: " << endl;
			cout << "0 -> " << image1 << endl;
			cout << a << " -> " << image2 << endl;
			cout << b << " -> " << image3 << endl;
			}
		int set1[3];
		int set2[3];
		set1[0] = 0;
		set1[1] = a;
		set1[2] = b;
		set2[0] = image1;
		set2[1] = image2;
		set2[2] = image3;
		int_vec_heapsort(set1, 3);
		int_vec_heapsort(set2, 3);
		if (int_vec_compare(set1, set2, 3)) {
			cout << "Error: The automorphism does not stabilize "
					"the subspace." << endl;
			exit(1);
		}
#endif
		f_automorphism = true;
		Aut = NEW_int(A->elt_size_in_int);
		A->Group_element->element_move(ELT1, Aut, 0);

	}
	else {
		if (f_v) {
			cout << "semifield_level_two::trace Fusion" << endl;
		}
		multiply_to_the_right(ELT1, Basis1,
				ELT2, ELT3, 0 /* verbose_level */);
		A->Group_element->element_move(ELT3, ELT1, 0);

		d1 = class_to_flag_orbit[idx1];
		d2 = class_to_flag_orbit[idx2];
		if (d1 != d2) {
			cout << "d1 != d2" << endl;
			exit(1);
		}
		d = d1;
		cc1 = flag_orbit_classes[d * 2 + 0];
		cc2 = flag_orbit_classes[d * 2 + 1];
		if (cc1 != cc2 && idx1 == cc2) {
			if (f_vv) {
				cout << "semifield_level_two::trace "
						"multiplying Basis_inv to the right" << endl;
				}
			multiply_to_the_right(ELT1,
					class_rep_plus_I_Basis_inv[cc1],
					ELT2, ELT3,
					0 /* verbose_level */);
			A->Group_element->element_move(ELT3, ELT1, 0);
		}
		else {
		}
		A->Group_element->element_invert(ELT1, ELT3, 0);
		f_Fusion[d] = true;
		Fusion_idx[d] = f;
		Fusion_elt[d] = NEW_int(A->elt_size_in_int);
		A->Group_element->element_move(ELT3, Fusion_elt[d], 0);
	}

	if (f_v) {
		cout << "semifield_level_two::trace done" << endl;
	}
}

void semifield_level_two::multiply_to_the_right(
		int *ELT1, int *Mtx, int *ELT2, int *ELT3,
		int verbose_level)
// Creates the n x n matrix which is the 2 x 2 block matrix
// (A 0)
// (0 A)
// where A is Mtx.
// The resulting element is stored in ELT2.
// After this, ELT1 * ELT2 will be stored in ELT3
{
	int f_v = (verbose_level >= 1);
	int *M;
	int i, j, a;

	if (f_v) {
		cout << "semifield_level_two::multiply_to_the_right" << endl;
	}
	//M = Mnn;
	M = NEW_int(n * n);
	Int_vec_zero(M, n * n);
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			a = Mtx[i * k + j];
			M[i * n + j] = a;
			M[(k + i) * n + k + j] = a;
		}
	}
	A->Group_element->make_element(ELT2, M, 0);
	A->Group_element->element_mult(ELT1, ELT2, ELT3, 0);

	FREE_int(M);

	if (f_v) {
		cout << "semifield_level_two::multiply_to_the_right done" << endl;
	}
}

void semifield_level_two::compute_candidates_at_level_two_case(
	int orbit,
	long int *&Candidates, int &nb_candidates, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Mtx_A;
	int **Mtx_stack;
	int i;
	long int a;
	groups::sims *G;
	ring_theory::longinteger_object go_PGL;
	int alloc_length;


	if (f_v) {
		cout << "semifield_level_two::compute_candidates_at_level_two_case" << endl;
	}
	nb_candidates = 0;
	alloc_length = 1024;
	Candidates = NEW_lint(alloc_length);

	Mtx_A = NEW_int(k * k);
	Mtx_stack = NEW_pint(2);
	for (i = 0; i < 2; i++) {
		Mtx_stack[i] = NEW_int(k * k);
	}

	G = A_PGLk->Sims;
	G->group_order(go_PGL);

	F->Linear_algebra->identity_matrix(Mtx_stack[0], k);

	if (f_v) {
		cout << "semifield_level_two::compute_candidates_at_level_two_case "
				"Level 2, Looking at orbit " << orbit << " / "
				<< nb_orbits << ":" << endl;
	}


	a = Pt[orbit];

	if (false) {
		cout << "semifield_level_two::compute_candidates_at_level_two_case "
				"a=" << a << endl;
	}
	SC->matrix_unrank(a, Mtx_stack[1]);

	int nb_tested;
	int *Affine_k;
	int *Affine_2;
	int *Cnt;
	int *Mtx1, *Mtx2;
	int *B;
	int *base_cols;
	int N, N1, v[2], j, h, b1, b2;
	long int r;
	number_theory::number_theory_domain NT;
	geometry::other_geometry::geometry_global Gg;
	data_structures::sorting Sorting;

	N = NT.i_power_j(q, k);
	N1 = NT.i_power_j(q, 2);


	Affine_k = NEW_int(N * k);
	Affine_2 = NEW_int(N1 * 2);
	Cnt = NEW_int(k + 1);
	Mtx1 = NEW_int(k * k);
	Mtx2 = NEW_int(k * k);
	B = NEW_int(k2);
	base_cols = NEW_int(k);


	for (i = 0; i < N; i++) {
		Gg.AG_element_unrank(q, Affine_k + i * k, 1, k, i);
	}
	for (i = 0; i < N1; i++) {
		Gg.AG_element_unrank(q, Affine_2 + i * 2, 1, 2, i);
	}
	nb_tested = 0;
	Cnt[0] = 0;
	i = 0;
	while (i >= 0) {
		for (Cnt[i]++; Cnt[i] < N; Cnt[i]++) {

#if 0
			// for debug purposes:
			if (i == 0) {
				if (Cnt[i] > 2) {
					continue;
				}
			}
#endif


			Int_vec_copy(Affine_k + Cnt[i] * k, Mtx1 + i * k, k);
			//AG_element_unrank(q, Mtx1 + i * k, 1, k, Cnt[i]);
			if (i < 2 && Mtx1[i * k + 0]) {
				continue;
					// we need zeroes in the first
					// two entries in the first column
			}
			Int_vec_copy(Mtx1, Mtx2, (i + 1) * k);
			if (F->Linear_algebra->rank_of_rectangular_matrix_memory_given(
					Mtx2, i + 1, k, B, base_cols,
					false /* f_complete */,
					0 /* verbose_level */) < i + 1) {
				continue; // rank is bad
			}
			// now rank is OK
			for (h = 1; h < N1; h++) {
				Int_vec_copy(Affine_2 + h * 2, v, 2);
				//AG_element_unrank(q, v, 1, 2, h);

				// form the linear combination of
				// Mtx_stack and subtract from Mtx1:
				for (j = 0; j < (i + 1) * k; j++) {
					b1 = F->mult(Mtx_stack[0][j], v[0]);
					b2 = F->mult(Mtx_stack[1][j], v[1]);
					Mtx2[j] = F->add(Mtx1[j], F->negate(F->add(b1, b2)));
				}
#if 0
				cout << "semifield_level_two::compute_candidates_at_level_two_case "
						"testing linear combination ";
				int_vec_print(cout, v, 2);
				cout << endl;
				int_matrix_print(Mtx2, i + 1, k);
#endif
				if (F->Linear_algebra->rank_of_rectangular_matrix_memory_given(
						Mtx2, i + 1, k, B, base_cols,
						false /* f_complete */,
						0 /* verbose_level */) < i + 1) {
					break; // rank is bad
				}
			}
			if (h < N1) {
				// failed the test
				continue;
			}
			// we survived the tests:
			break;
		}
		if (Cnt[i] == N) {
			i--;
		}
		else {
			i++;
			Cnt[i] = 0;
		}
		nb_tested++;
		if ((nb_tested & ((1 << 17) - 1)) == 0) {
			cout << "semifield_level_two::compute_candidates_at_level_two_case "
					"orbit " << orbit << " / " << nb_orbits << " Cnt=";
			Int_vec_print(cout, Cnt, k);
			cout << " number tested = " << nb_tested
					<< " Number of candidates = " << nb_candidates << endl;
		}
		if (i == k) {
#if 0
			if (!test_candidate(Mtx_stack, 2, Mtx1, verbose_level - 2)) {
				cout << "we survived the tests but "
						"test_candidate fails" << endl;
				int_matrix_print(Mtx1, k, k);
				exit(1);
			}
#endif

			r = SC->matrix_rank(Mtx1);
			if (r < 0) {
				cout << "semifield_level_two::compute_candidates_at_level_two_case "
						"orbit r < 0" << endl;
				cout << "Mtx1:" << endl;
				Int_matrix_print(Mtx1, k, k);
				exit(1);
			}
			Sorting.lint_vec_append_and_reallocate_if_necessary(
					Candidates, nb_candidates, alloc_length, r,
					0 /*verbose_level*/);


			i--;
		}
	} // while


	if (f_v) {
		cout << "semifield_level_two::compute_candidates_at_level_two_case "
				"Level 2: orbit " << orbit << " / " << nb_orbits
				<< ": nb_tested = " << nb_tested << ", found "
				<< nb_candidates << " candidates, sorting them now." << endl;
	}
	Sorting.lint_vec_heapsort(Candidates, nb_candidates);



	FREE_int(Mtx_A);
	for (i = 0; i < 2; i++) {
		FREE_int(Mtx_stack[i]);
	}
	FREE_pint(Mtx_stack);

	FREE_int(Affine_k);
	FREE_int(Affine_2);
	FREE_int(Cnt);
	FREE_int(Mtx1);
	FREE_int(Mtx2);
	FREE_int(B);
	FREE_int(base_cols);
	if (false) {
		cout << "Level 2: orbit " << orbit << " / "
				<< nb_orbits << ": found "
				<< nb_candidates << " candidates:" << endl;
		//SC->print_set_of_matrices_numeric(Candidates, nb_candidates);
	}
	if (f_v) {
		cout << "semifield_level_two::compute_candidates_at_level_two_case "
				"done" << endl;
	}
}


void semifield_level_two::allocate_candidates_at_level_two(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "semifield_level_two::allocate_candidates_at_level_two" << endl;
	}
	Candidates = NEW_plint(nb_orbits);
	Nb_candidates = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		Candidates[i] = NULL;
	}
	if (f_v) {
		cout << "semifield_level_two::allocate_candidates_at_level_two done" << endl;
	}
}

int semifield_level_two::test_if_file_exists_candidates_at_level_two_case(
	int orbit, int verbose_level)
{
	string fname;
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	SC->make_fname_candidates_at_level_two_orbit(fname, orbit);
	if (Fio.file_size(fname) > 0) {
		if (f_v) {
			cout << "semifield_level_two::test_if_file_exists_candidates_at_level_two_case file "
					<< fname << " exists" << endl;
		}
		return true;
	}
	else {
		if (f_v) {
			cout << "semifield_level_two::test_if_file_exists_candidates_at_level_two_case file "
					<< fname << " does not exist" << endl;
		}
		return false;
	}
}

int semifield_level_two::test_if_txt_file_exists_candidates_at_level_two_case(
	int orbit, int verbose_level)
{
	string fname;
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	SC->make_fname_candidates_at_level_two_orbit_txt(fname, orbit);
	if (Fio.file_size(fname) > 0) {
		if (f_v) {
			cout << "semifield_level_two::test_if_txt_file_exists_candidates_at_level_two_case file "
					<< fname << " exists" << endl;
		}
		return true;
	}
	else {
		if (f_v) {
			cout << "semifield_level_two::test_if_txt_file_exists_candidates_at_level_two_case file "
					<< fname << " does not exist" << endl;
		}
		return false;
	}
}


void semifield_level_two::find_all_candidates_at_level_two(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_level_two::find_all_candidates_at_level_two" << endl;
	}
	allocate_candidates_at_level_two(verbose_level);
	for (orbit = 0; orbit < nb_orbits; orbit++) {

		if (f_v) {
			cout << "Level 2, looking at orbit " << orbit
					<< " / " << nb_orbits << ":" << endl;
		}


		if (test_if_file_exists_candidates_at_level_two_case(
				orbit, verbose_level)) {
			read_candidates_at_level_two_case(
				Candidates[orbit],
				Nb_candidates[orbit],
				orbit, verbose_level - 1);
		}
		else if (test_if_txt_file_exists_candidates_at_level_two_case(
				orbit, verbose_level)) {
			read_candidates_at_level_two_case_txt_file(
				Candidates[orbit],
				Nb_candidates[orbit],
				orbit, verbose_level - 1);

			write_candidates_at_level_two_case(
				Candidates[orbit], Nb_candidates[orbit],
				orbit, verbose_level - 1);
		}
		else {
			if (f_v) {
				cout << "semifield_level_two::find_all_candidates_"
						"at_level_two Cannot find candidate file. "
						"before compute_candidates_at_level_two_case" << endl;
			}
			compute_candidates_at_level_two_case(
				orbit,
				Candidates[orbit], Nb_candidates[orbit],
				verbose_level - 1);
			if (f_v) {
				cout << "semifield_level_two::find_all_candidates_"
						"at_two_three after compute_candidates_at_"
						"level_two_case" << endl;
			}
			if (f_v) {
				cout << "semifield_level_two::find_all_candidates_"
						"at_two_three before write_candidates_at_"
						"level_two_case" << endl;
			}
			write_candidates_at_level_two_case(
				Candidates[orbit], Nb_candidates[orbit],
				orbit, verbose_level - 1);
			if (f_v) {
				cout << "semifield_level_two::find_all_candidates_"
						"at_two_three after write_candidates_at_"
						"level_two_case" << endl;
			}
		}


		string fname_test;

		SC->make_fname_candidates_at_level_two_orbit_by_type(
				fname_test, orbit, 0);

		if (Fio.file_size(fname_test) >= 1) {
			cout << "Type files for orbit " << orbit
					<< " exist" << endl;
		}
		else {
			cout << "Type files for orbit " << orbit
					<< " do not exist" << endl;
			long int **Set;
			int *Set_sz;
			int Nb_sets;
			int window_bottom, window_size;
			int h;

			window_bottom = k - 1;
			window_size = k - 2;
			SC->candidates_classify_by_first_column(
					Candidates[orbit],
					Nb_candidates[orbit],
				window_bottom, window_size,
				Set, Set_sz, Nb_sets,
				verbose_level);

			for (h = 0; h < Nb_sets; h++) {
				string fname;

				SC->make_fname_candidates_at_level_two_orbit_by_type(
						fname, orbit, h);

				Fio.write_set_to_file_as_int8(fname,
					Set[h], Set_sz[h],
					verbose_level);
				cout << "Written file " << fname << " of size "
						<< Fio.file_size(fname) << endl;
			}

			for (h = 0; h < Nb_sets; h++) {
				FREE_lint(Set[h]);
			}
			FREE_plint(Set);
			FREE_int(Set_sz);
		}


	} // next up_orbit

	if (f_v) {
		cout << "semifield_level_two::find_all_candidates_at_level_two" << endl;
		cout << "orbit : Level 2 Nb_candidates" << endl;
		for (orbit = 0; orbit < nb_orbits; orbit++) {
			cout << orbit << " : "
					<< Nb_candidates[orbit] << endl;
		}
	}
	if (f_v) {
		cout << "semifield_level_two::find_all_candidates_at_level_two done" << endl;
	}
}

void semifield_level_two::read_candidates_at_level_two_case(
	long int *&Candidates, int &Nb_candidates, int orbit,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_level_two::read_candidates_at_level_two_case" << endl;
	}
	SC->make_fname_candidates_at_level_two_orbit(fname, orbit);

	if (Fio.file_size(fname) > 0) {
		if (f_v) {
			cout << "Reading candidates from file "
					<< fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
		//Fio.read_set_from_file_lint(fname,
		//		Candidates, Nb_candidates, verbose_level);
		Fio.read_set_from_file_int8(fname,
				Candidates, Nb_candidates, verbose_level);
		if (f_v) {
			cout << "Reading candidates from file "
					<< fname << " of size " << Fio.file_size(fname)
					<< " done" << endl;
			cout << "We found " << Nb_candidates << " candidates" << endl;
		}
	}
	else {
		cout << "semifield_level_two::read_candidates_at_level_two_case file " << fname
				<< " does not exist" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "semifield_level_two::read_candidates_at_level_two_case done" << endl;
	}
}

void semifield_level_two::read_candidates_at_level_two_case_txt_file(
	long int *&Candidates, int &Nb_candidates, int orbit,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_level_two::read_candidates_at_level_"
				"two_case_txt_file" << endl;
	}
	SC->make_fname_candidates_at_level_two_orbit_txt(fname, orbit);

	if (Fio.file_size(fname) > 0) {
		if (f_v) {
			cout << "Reading candidates from file "
					<< fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
		Fio.read_set_from_file_lint(fname,
				Candidates, Nb_candidates, verbose_level);
		//Fio.read_set_from_file_int8(fname,
		//		Candidates, Nb_candidates, verbose_level);
		if (f_v) {
			cout << "Reading candidates from file "
					<< fname << " of size " << Fio.file_size(fname)
					<< " done" << endl;
		}
	}
	else {
		cout << "semifield_level_two::read_candidates_at_level_"
				"two_case_txt_file file " << fname
				<< " does not exist" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "semifield_level_two::read_candidates_at_level_"
				"two_case_txt_file done" << endl;
	}
}


void semifield_level_two::write_candidates_at_level_two_case(
	long int *Candidates, int Nb_candidates, int orbit,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_level_two::write_candidates_"
				"at_level_two_case" << endl;
	}

	SC->make_fname_candidates_at_level_two_orbit(fname, orbit);

	//Fio.write_set_to_file_lint(fname,
	//		Candidates, Nb_candidates, 0 /*verbose_level*/);
	Fio.write_set_to_file_as_int8(fname,
			Candidates, Nb_candidates, 0 /*verbose_level*/);

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "semifield_level_two::write_candidates_"
				"at_level_two_case done" << endl;
	}
}

void semifield_level_two::read_candidates_at_level_two_by_type(
		data_structures::set_of_sets_lint *&Candidates_by_type,
		int orbit,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "semifield_level_two::read_candidates_at_level_"
				"two_by_type" << endl;
	}

	Candidates_by_type = NEW_OBJECT(data_structures::set_of_sets_lint);

	long int **Set;
	int *Set_sz;
	int Nb_sets;
	int /*window_bottom,*/ window_size;
	int h;
	number_theory::number_theory_domain NT;
	orbiter_kernel_system::file_io Fio;

	//window_bottom = k - 1;
	window_size = k - 2;
	Nb_sets = NT.i_power_j(q, window_size);

	Set = NEW_plint(Nb_sets);
	Set_sz = NEW_int(Nb_sets);
	for (h = 0; h < Nb_sets; h++) {
		string fname;

		SC->make_fname_candidates_at_level_two_orbit_by_type(
				fname, orbit, h);

		cout << "Reading file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		if (Fio.file_size(fname) <= 0) {
			cout << "semifield_level_two::read_candidates_at_"
					"level_two_by_type file " << fname
					<< " does not exist" << endl;
			exit(1);
		}
		Fio.read_set_from_file_int8(fname,
			Set[h], Set_sz[h],
			verbose_level);
	}
	int underlying_set_size = NT.i_power_j(q, k2);

	if (f_v) {
		cout << "semifield_level_two::read_candidates_at_level_"
				"two_by_type initializing set_of_sets" << endl;
	}

	Candidates_by_type->init(underlying_set_size,
			Nb_sets, Set, Set_sz, verbose_level);


	for (h = 0; h < Nb_sets; h++) {
		FREE_lint(Set[h]);
	}
	FREE_plint(Set);
	FREE_int(Set_sz);

	if (f_v) {
		cout << "semifield_level_two::read_candidates_at_level_"
				"two_by_type done" << endl;
	}
}

void semifield_level_two::get_basis_and_pivots(
		int po,
		int *basis, int *pivots, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	//int ext, idx;
	long int a;


	if (f_v) {
		cout << "semifield_level_two::get_basis_and_pivots"
				"pivots po=" << po << endl;
	}

	F->Linear_algebra->identity_matrix(basis, k);

#if 0
	ext = up_orbit_rep[po];
	idx = down_orbit_classes[ext * 2 + 0];
	a = class_rep_rank[idx];
#else
	a = Pt[po];
#endif


	SC->matrix_unrank(a, basis + k2);

	pivots[0] = 0;
	pivots[1] = k;


	if (f_v) {
		cout << "semifield_level_two::get_basis_and_pivots"
				"pivots po=" << po << " done" << endl;
	}
}

void semifield_level_two::report(
	std::ofstream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, ext, idx;
	long int a; //, b;


	if (f_v) {
		cout << "semifield_level_two::report" << endl;
	}

	{
		//const char *fname = "Reps_lvl_2.tex";
		//ofstream fp(fname);
		//latex_interface L;

		int *Mtx_Id;
		int *Mtx;

		Mtx_Id = NEW_int(k2);
		Mtx = NEW_int(k2);

		F->Linear_algebra->identity_matrix(Mtx_Id, k);

		if (f_v) {
			cout << "semifield_level_two::report "
					"before Conjugacy classes" << endl;
		}

		ost << "\\section{Conjugacy classes without eigenvalue one}" << endl;

		ost << "There are " << nb_classes << " conjugacy classes without eigenvalue one:\\\\" << endl;

		ost << "\\begin{enumerate}[(1)]" << endl;
		for (i = 0; i < nb_classes; i++) {
			ost << "\\item" << endl;

			int f_elements_exponential = false;
			string symbol_for_print;

			symbol_for_print.assign("\\alpha");


			a = class_rep_rank[i];
			//b = class_rep_plus_I_rank[i];
			if (f_v) {
				cout << "Representative of class " << i << " / " << nb_classes
						<< " is matrix " << a << ":\\\\" << endl;
			}
			ost << "Representative of class " << i << " / " << nb_classes
					<< " is matrix " << a << ":\\\\" << endl;
			SC->matrix_unrank(a, Mtx);
			ost << "$$" << endl;
			ost << "\\left[" << endl;
			F->Io->latex_matrix(ost, f_elements_exponential,
					symbol_for_print, Mtx, k, k);
			ost << "\\right]";
			ost << "$$";

#if 0
			int *class_rep_rank;
			int *class_rep_plus_I_rank;
			int **class_rep_plus_I_Basis;
			int **class_rep_plus_I_Basis_inv;
			int *R_i_plus_I_class_idx;
			strong_generators *Centralizer_gens;
			int *down_orbit_of_class;
#endif

			ost << "$A+I$ belongs to class " << R_i_plus_I_class_idx[i] << "\\\\" << endl;
			ost << "flag\\_orbit\\_of\\_class = " << class_to_flag_orbit[i] << "\\\\" << endl;



			//A_PGLk->make_element(Elt, Mtx, 0);

			groups::strong_generators *Centralizer_gens;
			ring_theory::longinteger_object go;


			C->make_matrix_from_class_rep(Mtx, R + i, 0 /*verbose_level - 1 */);

			Centralizer_gens = NEW_OBJECT(groups::strong_generators);
			Centralizer_gens->init_centralizer_of_matrix(
					A_PGLk, Mtx, verbose_level - 3);
			Centralizer_gens->group_order(go);
			ost << "Centralizer has order " << go << "\\\\" << endl;
			Centralizer_gens->print_generators_tex(ost);
			FREE_OBJECT(Centralizer_gens);
		}
		ost << "\\end{enumerate}" << endl;
		ost << endl;

		if (f_v) {
			cout << "semifield_level_two::report before flag orbits" << endl;
		}


		ost << "\\section{Flag orbits at level 2}" << endl;

		ost << "\\begin{enumerate}[(1)]" << endl;

		for (i = 0; i < nb_flag_orbits; i++) {


			ost << "\\item" << endl;
			ost << i << " / " << nb_flag_orbits << ":\\\\" << endl;
			ost << "Classes: ";
			Int_vec_print(ost, flag_orbit_classes + i * 2, 2);
			ost << "\\\\" << endl;
			ost << "number\\_of\\_matrices = " << flag_orbit_number_of_matrices[i] << "\\\\" << endl;
			ost << "orbit\\_length = " << flag_orbit_length[i] << "\\\\" << endl;
			ost << "f\\_Fusion = " << f_Fusion[i] << "\\\\" << endl;
			ost << "Fusion\\_idx = " << Fusion_idx[i] << "\\\\" << endl;
			if (f_Fusion[i]) {
				ost << "Fusion\\_elt = \\\\" << endl;
				ost << "$$" << endl;
				SC->A->Group_element->element_print_latex(Fusion_elt[i], ost);
				ost << "$$" << endl;
			}
			ring_theory::longinteger_object go;
			Flag_orbit_stabilizer[i].group_order(go);
			ost << "Flag orbit stabilizer has order " << go << "\\\\" << endl;
			Flag_orbit_stabilizer[i].print_generators_tex(ost);

		}

		ost << "\\end{enumerate}" << endl;

		if (f_v) {
			cout << "semifield_level_two::report before Orbits at level 2" << endl;
		}
		ost << "\\section{Orbits at level 2}" << endl;

		ost << "\\begin{enumerate}[(1)]" << endl;
		for (i = 0; i < nb_orbits; i++) {


			ost << "\\item" << endl;

			ring_theory::longinteger_object go, go1;
			//int *Elt1;

			Stabilizer_gens[i].group_order(go);
			//cout << i << " : " << up_orbit_rep[i] << " : " << go << endl;

#if 0
			ext = up_orbit_rep[i];

			idx = down_orbit_classes[ext * 2 + 0];
			a = class_rep_rank[idx];
			b = class_rep_plus_I_rank[idx];
#else
			a = Pt[i];
#endif

			//Elt1 = NEW_int(A_PGLk->elt_size_in_int);
			SC->matrix_unrank(a, Mtx);
				// A_PGLk->Sims->element_unrank_int(a, Elt1);
			//cout << "The representative of class " << idx
			//<< " is the following matrix of rank " << a << endl;
			//int_matrix_print(Mtx, k, k);
			//cout << "The stabilizer has order " << go << endl;

			int f_elements_exponential = false;
			string symbol_for_print;

			symbol_for_print.assign("\\alpha");


			ext = defining_flag_orbit[i];
			idx = flag_orbit_classes[ext * 2 + 0];
			ost << "Representative " << i << " / " << nb_orbits
				<< " classes " << idx << ","
				<< flag_orbit_classes[ext * 2 + 1] << endl;
			ost << "\\{" << endl;
			for (j = 0; j < nb_flag_orbits; j++) {
				if (Fusion_idx[j] == i) {
					ost << j << ", ";
				}
			}
			ost << "\\}" << endl;
			ost << "$$" << endl;
			ost << "\\left\\{" << endl;
			ost << "\\left[" << endl;
			F->Io->latex_matrix(ost, f_elements_exponential,
					symbol_for_print, Mtx_Id, k, k);
			ost << "\\right], \\;";
			ost << "\\left[" << endl;
			F->Io->latex_matrix(ost, f_elements_exponential,
					symbol_for_print, Mtx, k, k);
			ost << "\\right]";
			ost << "\\right\\}" << endl;
			ost << "_{";
			ost << go << "}" << endl;
			ost << "$$" << endl;

			Stabilizer_gens[i].print_generators_tex(ost);

		}
		ost << "\\end{enumerate}" << endl;
		ost << endl;
		if (f_v) {
			cout << "semifield_level_two::report "
					"after Orbits at level 2" << endl;
		}


		FREE_int(Mtx_Id);
		FREE_int(Mtx);

		//L.foot(fp);
	}
	if (f_v) {
		cout << "semifield_level_two::report done" << endl;
	}
}

void semifield_level_two::create_fname_level_info_file(
		std::string &fname)
{
	fname = SC->level_two_prefix + "Level_2_info.csv";
}



void semifield_level_two::write_level_info_file(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_level_two::write_level_info_file" << endl;
	}
	int i;
	int nb_vecs = 5;
	const char *column_label[] = {
		"Po",
		"So",
		"Fo",
		"Go",
		"Pt"
		};
	string fname;

	create_fname_level_info_file(fname);

	{
		ofstream f(fname);
		int j;

		f << "Row";
		for (j = 0; j < nb_vecs; j++) {
			f << "," << column_label[j];
			}
		f << endl;
		for (i = 0; i < nb_orbits; i++) {
			f << i;
			f << "," << 0 /* Po[i]*/
					<< "," << So[i]
					<< "," << Fo[i]
					<< "," << Go[i]
					<< "," << Pt[i] << endl;
			}
		f << "END" << endl;
	}

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "semifield_level_two::write_level_info_file done" << endl;
	}
}


void semifield_level_two::read_level_info_file(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	long int *M;
	int m, n, i;
	//int tmp;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "semifield_level_two::read_level_info_file" << endl;
	}
	create_fname_level_info_file(fname);

	cout << "semifield_level_two::read_level_info_file " << fname << endl;

	if (Fio.file_size(fname) <= 0) {
		cout << "semifield_lifting::read_level_info_file "
			"error trying to read the file " << fname << endl;
		exit(1);
	}

	Fio.Csv_file_support->lint_matrix_read_csv(
			fname, M, m, n, 0 /* verbose_level */);
		// Row,Po,So,Mo,Go,Pt

	nb_orbits = m;

	//Po = NEW_int(m);
	So = NEW_int(m);
	Fo = NEW_int(m);
	Go = NEW_lint(m);
	Pt = NEW_lint(m);

	for (i = 0; i < m; i++) {
		//Po[i] = M[i * n + 1];
		So[i] = M[i * n + 2];
		Fo[i] = M[i * n + 3];
		Go[i] = M[i * n + 4];
		Pt[i] = M[i * n + 5];
	}

	FREE_lint(M);

	if (f_v) {
		cout << "semifield_level_two::read_level_info_file done" << endl;
	}
}






}}}

