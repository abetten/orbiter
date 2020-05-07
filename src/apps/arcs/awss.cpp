/*
 * awss.cpp
 *
 *  Created on: Sep 10, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


using namespace orbiter;

// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);
void do_it(finite_field *F, linear_group *LG,
		int f_test, int *test3, int verbose_level);

int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;
	int f_linear = FALSE;
	linear_group_description *Descr = NULL;
	linear_group *LG = NULL;
	int f_test = FALSE;
	int test3[3];
	os_interface Os;

	t0 = Os.os_ticks();
	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-linear") == 0) {
			f_linear = TRUE;
			Descr = NEW_OBJECT(linear_group_description);
			i += Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-linear" << endl;
			}
		else if (strcmp(argv[i], "-test") == 0) {
			f_test = TRUE;
			test3[0] = atoi(argv[++i]);
			test3[1] = atoi(argv[++i]);
			test3[2] = atoi(argv[++i]);
			cout << "-test ";
			int_vec_print(cout, test3, 3);
			cout << endl;
		}
	}

	if (!f_linear) {
		cout << "please use option -linear ..." << endl;
		exit(1);
		}


	finite_field *F;
	int f_v = (verbose_level >= 1);
	file_io Fio;
	int q;


	F = NEW_OBJECT(finite_field);
	q = Descr->input_q;

	if (f_v) {
		cout << "awss q=" << q << endl;
		}

	F->init(q, 0);
	Descr->F = F;
	//q = Descr->input_q;



	LG = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "awss before LG->init, "
				"creating the group" << endl;
		}

	LG->init(Descr, verbose_level - 1);

	if (f_v) {
		cout << "awss after LG->init" << endl;
		}

	action *A;

	A = LG->A2;

	if (f_v) {
		cout << "awss created group " << LG->prefix << endl;
	}

	if (!A->is_matrix_group()) {
		cout << "arcs_main the group is not a matrix group " << endl;
		exit(1);
	}

	int f_semilinear;

	f_semilinear = A->is_semilinear_matrix_group();
	if (f_v) {
		cout << "awss f_semilinear=" << f_semilinear << endl;
	}

	matrix_group *M;

	M = A->get_matrix_group();
	int dim = M->n;

	if (f_v) {
		cout << "awss dim=" << dim << endl;
	}

	do_it(F, LG, f_test, test3, verbose_level);

	the_end(t0);
}

void do_it(finite_field *F, linear_group *LG,
		int f_test, int *test3, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "do_it" << endl;
	}


	homogeneous_polynomial_domain *HPD;

	HPD = NEW_OBJECT(homogeneous_polynomial_domain);

	if (f_v) {
		cout << "do_it before HPD->init" << endl;
	}
	HPD->init(F, 3 /* nb_vars */, 2 /* degree */,
		TRUE /* f_init_incidence_structure */,
		verbose_level - 2);


	int *Eqn;
	long int *Pts;
	int nb_pts;
	int lambda;
	long int *Conics;

	Eqn = NEW_int(HPD->nb_monomials);
	Pts = NEW_lint(HPD->P->N_points);
	Conics = NEW_lint((F->q - 1) * (F->q + 1));

	for (lambda = 1; lambda < F->q; lambda++) {

		// create the equation XY + lambda Z^2:

		int_vec_zero(Eqn, HPD->nb_monomials);
		Eqn[2] = lambda;
		Eqn[3] = 1;

		HPD->algebraic_set(Eqn, 1 /* nb_eqns */, Pts, nb_pts, verbose_level - 2);
		if (nb_pts != F->q + 1) {
			cout << "do_it nb_pts != F->q + 1" << endl;
			exit(1);
		}
		lint_vec_copy(Pts, Conics + (lambda - 1) * (F->q + 1), F->q + 1);
	}
	if (f_v) {
		cout << "do_it Conics:" << endl;
		lint_matrix_print(Conics, F->q - 1, F->q + 1);
	}

	action *A_on_conics;
	poset_classification_control *Control;
	poset *Poset;
	strong_generators *SG;
	poset_classification *PC;

	if (f_v) {
		cout << "do_it creating action on conics" << endl;
	}

	A_on_conics = LG->A2->create_induced_action_on_sets(F->q - 1 /* nb_sets */,
			F->q + 1 /* set_size */, Conics,
			verbose_level);
	if (f_v) {
		cout << "do_it creating action on conics is" << endl;
		A_on_conics->print_info();
	}

	SG = NEW_OBJECT(strong_generators);


	if (f_v) {
		cout << "do_it before SG->stabilizer_of_pencil_of_conics" << endl;
	}
	SG->stabilizer_of_pencil_of_conics(
			LG->A2,
			F,
			verbose_level);

	if (f_v) {
		cout << "do_it before generators" << endl;
	}
	SG->print_generators_tex(cout);



	Control = NEW_OBJECT(poset_classification_control);
	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(LG->A2, A_on_conics,
			SG,
			verbose_level);

	cout << "before Poset->orbits_on_k_sets_compute" << endl;

	PC = Poset->orbits_on_k_sets_compute(Control, 3 /* k */, verbose_level - 1);

	cout << "after Poset->orbits_on_k_sets_compute" << endl;
	cout << "nb_orbits at level 1 = " << PC->nb_orbits_at_level(1) << endl;
	cout << "nb_orbits at level 2 = " << PC->nb_orbits_at_level(2) << endl;
	cout << "nb_orbits at level 3 = " << PC->nb_orbits_at_level(3) << endl;

	long int *orbit_reps;
	int nb_orbits;
	int orbit_idx;
	long int *Set;
	int set_size;
	int i, a;
	int *line_type;

	PC->get_orbit_representatives(3 /*k*/, nb_orbits,
			orbit_reps, verbose_level);

	cout << "orbit reps:" << endl;
	lint_matrix_print(orbit_reps, nb_orbits, 3);

	set_size = 2 + 3 * (F->q - 1);
	Set = NEW_lint(set_size);
	line_type = NEW_int(HPD->P->N_lines);
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		Set[0] = 0;
		Set[1] = 1;
		for (i = 0; i < 3; i++) {
			a = orbit_reps[orbit_idx * 3 + i];
			lint_vec_copy(Conics + a * (F->q + 1) + 2, Set + 2 + i * (F->q - 1), F->q - 1);
		}
		cout << "orbit " << orbit_idx << " = ";
		lint_vec_print(cout, orbit_reps + orbit_idx * 3, 3);
		cout << " : ";
		//int_vec_print(cout, Set, set_size);
		//cout << " : ";
		//cout << endl;
		HPD->P->line_intersection_type_basic(
				Set, set_size, line_type, 0 /*verbose_level */);
		cout << "line_type = ";
		classify C;
		C.init(line_type, HPD->P->N_lines, FALSE, 0);
		C.print_naked(TRUE);
		cout << endl;
	}
	if (f_test) {
		cout << "testing ";
		int_vec_print(cout, test3, 3);
		cout << endl;

		Set[0] = 0;
		Set[1] = 1;
		for (i = 0; i < 3; i++) {
			a = test3[i];
			lint_vec_copy(Conics + a * (F->q + 1) + 2,
					Set + 2 + i * (F->q - 1), F->q - 1);
		}
		cout << "testing " << orbit_idx << " = ";
		int_vec_print(cout, test3, 3);
		cout << " : ";
		//int_vec_print(cout, Set, set_size);
		//cout << " : ";
		//cout << endl;
		HPD->P->line_intersection_type_basic(
				Set, set_size, line_type, 0 /*verbose_level */);
		cout << "line_type = ";
		classify C;
		C.init(line_type, HPD->P->N_lines, FALSE, 0);
		C.print_naked(TRUE);
		cout << endl;

	}


	if (f_v) {
		cout << "do_it done" << endl;
	}
}


