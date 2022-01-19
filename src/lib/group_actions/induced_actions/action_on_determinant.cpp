// action_on_determinant.cpp
//
// Anton Betten
// January 16, 2009

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {

action_on_determinant::action_on_determinant()
{
	null();
}

action_on_determinant::~action_on_determinant()
{
	free();
}

void action_on_determinant::null()
{
	M = NULL;
}

void action_on_determinant::free()
{
	
	null();
}


void action_on_determinant::init(actions::action &A,
		int f_projective, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go;
	number_theory::number_theory_domain NT;
	
	if (f_v) {
		cout << "action_on_determinant::init" << endl;
		cout << "f_projective=" << f_projective << endl;
		cout << "m=" << m << endl;
		}
	action_on_determinant::f_projective = f_projective;
	action_on_determinant::m = m;
	if (A.type_G != matrix_group_t) {
		cout << "action_on_determinant::init action "
				"not of matrix group type" << endl;
		exit(1);
		}
	M = A.G.matrix_grp;
	q = M->GFq->q;
	if (f_projective) {
		degree = NT.gcd_lint(m, q - 1);
		}
	else {
		degree = q - 1;
		}
	if (f_v) {
		cout << "degree=" << degree << endl;
		}
	
	if (f_v) {
		cout << "action_on_determinant::init field order is " << q << endl;
		}
}

long int action_on_determinant::compute_image(actions::action *A,
		int *Elt, long int i, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	long int a, b, c, l = 0, j;
	
	if (f_v) {
		cout << "action_on_determinant::compute_image "
				"i = " << i << endl;
		}
	if (i < 0 || i >= degree) {
		cout << "action_on_determinant::compute_image "
				"i = " << i << " out of range" << endl;
		exit(1);
		}
	if (f_projective) {
		a = M->GFq->alpha_power(i);
		}
	else {
		a = i + 1;
		}
	b = M->GFq->Linear_algebra->matrix_determinant(Elt, M->n, 0);
	c = M->GFq->mult(a, b);
	if (f_projective) {
		l = M->GFq->log_alpha(c);
		j = l % degree;
		}
	else {
		j = c - 1;
		}
	if (f_v) {
		cout << "action_on_determinant::compute_image "
				"det = " << b << endl;
		cout << "action_on_determinant::compute_image "
				<< a << " * " << b << " = " << c << endl;
		if (f_projective) {
			cout << "f_projective, a = " << a
					<< " l = " << l << " c = " << c << endl;
			}
		cout << "image of " << i << " is " << j << endl;
		}
	return j;
}

}}

