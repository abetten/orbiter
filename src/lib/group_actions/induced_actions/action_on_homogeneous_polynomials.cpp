// action_on_homogeneous_polynomials.cpp
//
// Anton Betten
// September 10, 2016

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {

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
		FREE_int(v1);
	}
	if (v2) {
		FREE_int(v2);
	}
	if (v3) {
		FREE_int(v3);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Equations) {
		FREE_int(Equations);
	}
	null();
}

void action_on_homogeneous_polynomials::init(action *A,
		homogeneous_polynomial_domain *HPD, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry_global Gg;

	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init" << endl;
		cout << "starting with action " << A->label << endl;
		}
	if (A->type_G != matrix_group_t) {
		cout << "action_on_homogeneous_polynomials::init "
				"fatal: A->type_G != matrix_group_t" << endl;
		exit(1);
		}
	action_on_homogeneous_polynomials::A = A;
	action_on_homogeneous_polynomials::HPD = HPD;
	M = A->G.matrix_grp;
	F = M->GFq;
	n = M->n;
	q = F->q;
	dimension = HPD->get_nb_monomials();
	degree = Gg.nb_PG_elements(dimension - 1, q);
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init "
				"dimension = " << dimension << endl;
		cout << "action_on_homogeneous_polynomials::init "
				"degree = " << degree << endl;
	}
	low_level_point_size = dimension;
	v1 = NEW_int(dimension);
	v2 = NEW_int(dimension);
	v3 = NEW_int(dimension);
	Elt1 = NEW_int(A->elt_size_in_int);
}

void action_on_homogeneous_polynomials::init_invariant_set_of_equations(
		int *Equations, int nb_equations, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init_invariant_set_of_equations" << endl;
		cout << "nb_equations = " << nb_equations << endl;
	}
	f_invariant_set = TRUE;
	action_on_homogeneous_polynomials::Equations =
			NEW_int(nb_equations * dimension);
	action_on_homogeneous_polynomials::nb_equations = nb_equations;
	Orbiter->Int_vec.copy(Equations,
			action_on_homogeneous_polynomials::Equations,
			nb_equations * dimension);
	for (i = 0; i < nb_equations; i++) {
		F->PG_element_normalize(Equations + i * dimension, 1, dimension);
	}
	degree = nb_equations;
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init_invariant_set_of_equations done" << endl;
	}
}
	
void action_on_homogeneous_polynomials::unrank_point(int *v, long int rk)
{
	HPD->unrank_coeff_vector(v, rk);
	//PG_element_unrank_modified(*F, v, 1, dimension, rk);
}

long int action_on_homogeneous_polynomials::rank_point(int *v)
{
#if 0
	int rk;

	PG_element_rank_modified(*F, v, 1, dimension, rk);
	return rk;
#else
	return HPD->rank_coeff_vector(v);
#endif
}

long int action_on_homogeneous_polynomials::compute_image_int(
		int *Elt, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int b;
	
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_image_int "
				"verbose_level=" << verbose_level << endl;
	}
	if (f_invariant_set) {
		Orbiter->Int_vec.copy(Equations + a * dimension, v1, dimension);
	}
	else {
		unrank_point(v1, a);
	}
	if (f_vv) {
		cout << "action_on_homogeneous_polynomials::compute_image_int "
				"a = " << a << " v1 = ";
		Orbiter->Int_vec.print(cout, v1, dimension);
		cout << endl;
	}
	
	compute_image_int_low_level(Elt, v1, v2, verbose_level);
	if (f_vv) {
		cout << " v2 = v1 * A = ";
		Orbiter->Int_vec.print(cout, v2, dimension);
		cout << endl;
	}

	if (f_invariant_set) {
		F->PG_element_normalize(v2, 1, dimension);
		for (b = 0; b < nb_equations; b++) {
			if (int_vec_compare(Equations + b * dimension, v2, dimension) == 0) {
				break;
			}
		}
		if (b == nb_equations) {
			cout << "action_on_homogeneous_polynomials::compute_image_int "
					"could not find equation" << endl;
			cout << "action_on_homogeneous_polynomials::compute_image_int "
					"a = " << a << " v1 = " << endl;
			Orbiter->Int_vec.print(cout, v1, dimension);
			cout << endl;
			cout << " v2 = v1 * A = " << endl;
			Orbiter->Int_vec.print(cout, v2, dimension);
			cout << endl;
			cout << "A=" << endl;
			A->element_print_quick(Elt, cout);
			cout << "equations:" << endl;
			for (b = 0; b < nb_equations; b++) {
				cout << setw(3) << b << " : ";
				Orbiter->Int_vec.print(cout, Equations + b * dimension, dimension);
				cout << endl;
			}
			exit(1);
		}
	}
	else {
		b = rank_point(v2);
	}
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_image_int "
				"done " << a << "->" << b << endl;
	}
	return b;
}

void action_on_homogeneous_polynomials::compute_image_int_low_level(
	int *Elt, int *input, int *output, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_semilinear;
	matrix_group *mtx;
	int n;
	
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_image_int_low_level" << endl;
	}
	if (f_vv) {
		cout << "action_on_homogeneous_polynomials::compute_image_int_low_level "
				"input = ";
		Orbiter->Int_vec.print(cout, input, dimension);
		cout << endl;
	}

	if (A->type_G != matrix_group_t) {
		cout << "action_on_homogeneous_polynomials::compute_image_int_low_level "
				"A->type_G != matrix_group_t" << endl;
		exit(1);
	}
	
	mtx = A->G.matrix_grp;
	f_semilinear = mtx->f_semilinear;
	n = mtx->n;


	A->element_invert(Elt, Elt1, 0);


	if (f_semilinear) {
		HPD->substitute_semilinear(input, output,
				f_semilinear, Elt[n * n], Elt1, 0 /* verbose_level */);
	}
	else {
		HPD->substitute_linear(input, output, Elt1, 0 /* verbose_level */);
	}

	if (f_vv) {
		cout << "action_on_homogeneous_polynomials::compute_image_int_low_level "
				"output = ";
		Orbiter->Int_vec.print(cout, output, dimension);
		cout << endl;
	}
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_image_int_low_level done" << endl;
	}
}

void action_on_homogeneous_polynomials::compute_representation(
	int *Elt, int *M, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_representation" << endl;
	}
	if (A->type_G != matrix_group_t) {
		cout << "action_on_homogeneous_polynomials::compute_representation "
				"A->type_G != matrix_group_t" << endl;
		exit(1);
	}


	for (i = 0; i < dimension; i++) {
		Orbiter->Int_vec.zero(v1, dimension);
		v1[i] = 1;
		compute_image_int_low_level(
			Elt, v1, M + i * dimension, 0 /* verbose_level */);
	}

	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_representation done" << endl;
	}
}


}}

