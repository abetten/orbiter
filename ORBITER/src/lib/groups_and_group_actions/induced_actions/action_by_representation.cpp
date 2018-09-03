// action_by_representation.C
//
// Anton Betten
// Mar18, 2010

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

action_by_representation::action_by_representation()
{
	null();
}

action_by_representation::~action_by_representation()
{
	free();
}

void action_by_representation::null()
{
	M = NULL;
	F = NULL;
	v1 = NULL;
	v2 = NULL;
	v3 = NULL;
	low_level_point_size = 0;
}

void action_by_representation::free()
{
	if (v1) {
		FREE_int(v1);
		}
	if (v2) {
		FREE_int(v2);
		}
	if (v3) {
		FREE_int(v3);
		}
	null();
}

void action_by_representation::init_action_on_conic(action &A, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_by_representation::init_action_on_conic" << endl;
		cout << "starting with action " << A.label << endl;
		}
	if (A.type_G != matrix_group_t) {
		cout << "action_by_representation::init fatal: A.type_G != matrix_group_t" << endl;
		exit(1);
		}
	M = A.G.matrix_grp;
	F = M->GFq;
	n = M->n;
	q = F->q;
	if (n != 2) {
		cout << "action_by_representation::init_action_on_conic needs n == 2" << endl;
		exit(1);
		}
	type = representation_type_PSL2_on_conic;
	dimension = 3;
	degree = nb_PG_elements(dimension - 1, q);
	low_level_point_size = 3;
	v1 = NEW_int(dimension);
	v2 = NEW_int(dimension);
	v3 = NEW_int(dimension);
}

int action_by_representation::compute_image_int(
	action &A, int *Elt, int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int b;
	
	if (f_v) {
		cout << "action_by_representation::compute_image_int" << endl;
		}
	PG_element_unrank_modified(*F, v1, 1, dimension, a);
	if (f_vv) {
		cout << "action_by_representation::compute_image_int a = " << a << " v1 = ";
		int_vec_print(cout, v1, dimension);
		cout << endl;
		}
	
	compute_image_int_low_level(A, Elt, v1, v2, verbose_level);
	if (f_vv) {
		cout << " v2=v1 * A=";
		int_vec_print(cout, v2, dimension);
		cout << endl;
		}

	PG_element_rank_modified(*F, v2, 1, dimension, b);
	if (f_v) {
		cout << "action_by_representation::compute_image_int done " << a << "->" << b << endl;
		}
	return b;
}

void action_by_representation::compute_image_int_low_level(
	action &A, int *Elt, int *input, int *output, int verbose_level)
{
	int *x = input;
	int *xA = output;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, f;
	
	if (f_v) {
		cout << "action_by_representation::compute_image_int_low_level" << endl;
		}
	if (f_vv) {
		cout << "action_by_representation::compute_image_int_low_level: x=";
		int_vec_print(cout, x, dimension);
		cout << endl;
		}
	int a, b, c, d;

	a = Elt[0];
	b = Elt[1];
	c = Elt[2];
	d = Elt[3];

	int AA[9];
	int two;

	two = F->add(1, 1);
	AA[0] = F->mult(a, a);
	AA[2] = F->mult(b, b);
	AA[6] = F->mult(c, c);
	AA[8] = F->mult(d, d);
	AA[1] = F->mult(a, b);
	AA[7] = F->mult(c, d);
	AA[3] = F->product3(two, a, c);
	AA[5] = F->product3(two, b, d);
	AA[4] = F->add(F->mult(a, d), F->mult(b, c));

	if (f_v) {
		cout << "A=" << endl;
		print_integer_matrix_width(cout, AA, 3, 3, 3, F->log10_of_q);
		}
	F->mult_matrix(x, AA, xA, 1, 3, 3);
	if (f_vv) {
		cout << "action_by_representation::compute_image_int_low_level: xA=";
		int_vec_print(cout, xA, dimension);
		cout << endl;
		}
	if (M->f_semilinear) {
		f = Elt[n * n];
		for (i = 0; i < dimension; i++) {
			xA[i] = F->frobenius_power(xA[i], f);
			}
		if (f_vv) {
			cout << "after " << f << " field automorphisms: xA=";
			int_vec_print(cout, xA, dimension);
			cout << endl;
			}
		}
	if (f_v) {
		cout << "action_by_representation::compute_image_int_low_level done" << endl;
		}
}


