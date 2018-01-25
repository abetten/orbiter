// action_on_homogeneous_polynomials.C
//
// Anton Betten
// September 10, 2016

#include "galois.h"
#include "action.h"

action_on_homogeneous_polynomials::action_on_homogeneous_polynomials()
{
	null();
}

action_on_homogeneous_polynomials::~action_on_homogeneous_polynomials()
{
	free();
}

void action_on_homogeneous_polynomials::null()
{
	A = NULL;
	HPD = NULL;
	M = NULL;
	F = NULL;
	v1 = NULL;
	v2 = NULL;
	v3 = NULL;
	Elt1 = NULL;
	low_level_point_size = 0;
	f_invariant_set = FALSE;
	Equations = NULL;
	nb_equations = 0;
}

void action_on_homogeneous_polynomials::free()
{
	if (v1) {
		FREE_INT(v1);
		}
	if (v2) {
		FREE_INT(v2);
		}
	if (v3) {
		FREE_INT(v3);
		}
	if (Elt1) {
		FREE_INT(Elt1);
		}
	if (Equations) {
		FREE_INT(Equations);
		}
	null();
}

void action_on_homogeneous_polynomials::init(action *A, homogeneous_polynomial_domain *HPD, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init" << endl;
		cout << "starting with action " << A->label << endl;
		}
	if (A->type_G != matrix_group_t) {
		cout << "action_on_homogeneous_polynomials::init fatal: A->type_G != matrix_group_t" << endl;
		exit(1);
		}
	action_on_homogeneous_polynomials::A = A;
	action_on_homogeneous_polynomials::HPD = HPD;
	M = A->G.matrix_grp;
	F = M->GFq;
	n = M->n;
	q = F->q;
	if (n != HPD->n) {
		cout << "action_on_homogeneous_polynomials::init fatal: n != HPD->n" << endl;
		exit(1);
		}
	dimension = HPD->nb_monomials;
	degree = nb_PG_elements(dimension - 1, q);
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init dimension = " << dimension << endl;
		cout << "action_on_homogeneous_polynomials::init degree = " << degree << endl;
		}
	low_level_point_size = dimension;
	v1 = NEW_INT(dimension);
	v2 = NEW_INT(dimension);
	v3 = NEW_INT(dimension);
	Elt1 = NEW_INT(A->elt_size_in_INT);
}

void action_on_homogeneous_polynomials::init_invariant_set_of_equations(INT *Equations, INT nb_equations, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init_invariant_set_of_equations" << endl;
		cout << "nb_equations = " << nb_equations << endl;
		}
	f_invariant_set = TRUE;
	action_on_homogeneous_polynomials::Equations = NEW_INT(nb_equations * dimension);
	action_on_homogeneous_polynomials::nb_equations = nb_equations;
	INT_vec_copy(Equations, action_on_homogeneous_polynomials::Equations, nb_equations * dimension);
	for (i = 0; i < nb_equations; i++) {
		PG_element_normalize(*F, Equations + i * dimension, 1, dimension);
		}
	degree = nb_equations;
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init_invariant_set_of_equations done" << endl;
		}
}
	
void action_on_homogeneous_polynomials::unrank_point(INT *v, INT rk)
{
	HPD->unrank_coeff_vector(v, rk);
	//PG_element_unrank_modified(*F, v, 1, dimension, rk);
}

INT action_on_homogeneous_polynomials::rank_point(INT *v)
{
#if 0
	INT rk;

	PG_element_rank_modified(*F, v, 1, dimension, rk);
	return rk;
#else
	return HPD->rank_coeff_vector(v);
#endif
}

INT action_on_homogeneous_polynomials::compute_image_INT(INT *Elt, INT a, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT b;
	
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_image_INT" << endl;
		}
	if (f_invariant_set) {
		INT_vec_copy(Equations + a * dimension, v1, dimension);
		}
	else {
		unrank_point(v1, a);
		}
	if (f_vv) {
		cout << "action_on_homogeneous_polynomials::compute_image_INT a = " << a << " v1 = ";
		INT_vec_print(cout, v1, dimension);
		cout << endl;
		}
	
	compute_image_INT_low_level(Elt, v1, v2, verbose_level);
	if (f_vv) {
		cout << " v2 = v1 * A = ";
		INT_vec_print(cout, v2, dimension);
		cout << endl;
		}

	if (f_invariant_set) {
		PG_element_normalize(*F, v2, 1, dimension);
		for (b = 0; b < nb_equations; b++) {
			if (INT_vec_compare(Equations + b * dimension, v2, dimension) == 0) {
				break;
				}
			}
		if (b == nb_equations) {
			cout << "action_on_homogeneous_polynomials::compute_image_INT could not find equation" << endl;
			exit(1);
			}
		}
	else {
		b = rank_point(v2);
		}
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_image_INT done " << a << "->" << b << endl;
		}
	return b;
}

void action_on_homogeneous_polynomials::compute_image_INT_low_level(
	INT *Elt, INT *input, INT *output, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_semilinear;
	matrix_group *mtx;
	INT n;
	
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_image_INT_low_level" << endl;
		}
	if (f_vv) {
		cout << "action_on_homogeneous_polynomials::compute_image_INT_low_level input = ";
		INT_vec_print(cout, input, dimension);
		cout << endl;
		}

	if (A->type_G != matrix_group_t) {
		cout << "action_on_homogeneous_polynomials::compute_image_INT_low_level A->type_G != matrix_group_t" << endl;
		exit(1);
		}
	
	mtx = A->G.matrix_grp;
	f_semilinear = mtx->f_semilinear;
	n = mtx->n;


	A->element_invert(Elt, Elt1, 0);


	if (f_semilinear) {
		HPD->substitute_semilinear(input, output, f_semilinear, Elt[n * n], Elt1, 0 /* verbose_level */);
		}
	else {
		HPD->substitute_linear(input, output, Elt1, 0 /* verbose_level */);
		}

	if (f_vv) {
		cout << "action_on_homogeneous_polynomials::compute_image_INT_low_level output = ";
		INT_vec_print(cout, output, dimension);
		cout << endl;
		}
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_image_INT_low_level done" << endl;
		}
}


