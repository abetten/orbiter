// action_on_homogeneous_polynomials.cpp
//
// Anton Betten
// September 10, 2016

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_homogeneous_polynomials::action_on_homogeneous_polynomials()
{
	n = 0;
	q = 0;
	A = NULL;
	HPD = NULL;
	M = NULL;
	F = NULL;
	low_level_point_size = 0;
	degree = 0;

	dimension = 0;
	v1 = NULL;
	v2 = NULL;
	v3 = NULL;
	Elt1 = NULL;

	f_invariant_set = false;
	Equations = NULL;
	nb_equations = 0;

	Table_of_equations = NULL;
}

action_on_homogeneous_polynomials::~action_on_homogeneous_polynomials()
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
}

void action_on_homogeneous_polynomials::init(
		actions::action *A,
		ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry::geometry_global Gg;

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
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init n = " << n << endl;
		cout << "action_on_homogeneous_polynomials::init q = " << q << endl;
	}
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
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init done" << endl;
	}
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
	f_invariant_set = true;
	action_on_homogeneous_polynomials::Equations = NEW_int(nb_equations * dimension);
	action_on_homogeneous_polynomials::nb_equations = nb_equations;
	Int_vec_copy(Equations,
			action_on_homogeneous_polynomials::Equations,
			nb_equations * dimension);
	for (i = 0; i < nb_equations; i++) {
		F->Projective_space_basic->PG_element_normalize(
				action_on_homogeneous_polynomials::Equations + i * dimension,
				1, dimension);
	}
	degree = nb_equations;


	Table_of_equations = NEW_OBJECT(data_structures::int_matrix);

	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init_invariant_set_of_equations "
				"before Table_of_equations->allocate_and_init" << endl;
	}
	Table_of_equations->allocate_and_init(
			nb_equations, dimension,
			action_on_homogeneous_polynomials::Equations);
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init_invariant_set_of_equations "
				"after Table_of_equations->allocate_and_init" << endl;
	}

	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init_invariant_set_of_equations "
				"before Table_of_equations->sort_rows" << endl;
	}
	Table_of_equations->sort_rows(verbose_level);
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init_invariant_set_of_equations "
				"after Table_of_equations->sort_rows" << endl;
	}
	if (f_v) {
		int a;

		for (i = 0; i < nb_equations; i++) {
			cout << i << " : ";
			Int_vec_print(cout, Table_of_equations->M + i * dimension, dimension);
			cout << " : ";
			HPD->print_equation(cout, Table_of_equations->M + i * dimension);
			cout << " : ";
			a = Table_of_equations->perm_inv[i];
			cout << a;
			cout << " : ";
			cout << endl;
		}
	}


	if (f_v) {
		cout << "action_on_homogeneous_polynomials::init_invariant_set_of_equations done" << endl;
	}
}
	
void action_on_homogeneous_polynomials::unrank_point(
		int *v, long int rk)
{
	HPD->unrank_coeff_vector(v, rk);
	//PG_element_unrank_modified(*F, v, 1, dimension, rk);
}

long int action_on_homogeneous_polynomials::rank_point(
		int *v)
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
	data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_image_int "
				"verbose_level=" << verbose_level << endl;
	}
	if (f_invariant_set) {
		Int_vec_copy(Table_of_equations->M + Table_of_equations->perm[a] * dimension, v1, dimension);
	}
	else {
		unrank_point(v1, a);
	}
	if (f_vv) {
		cout << "action_on_homogeneous_polynomials::compute_image_int "
				"a = " << a << " v1 = ";
		Int_vec_print(cout, v1, dimension);
		cout << endl;
	}
	
	compute_image_int_low_level(Elt, v1, v2, verbose_level);
	if (f_vv) {
		cout << " v2 = v1 * A = ";
		Int_vec_print(cout, v2, dimension);
		cout << endl;
	}

	if (f_invariant_set) {
		F->Projective_space_basic->PG_element_normalize(
				v2, 1, dimension);

#if 0
		// ToDo: get rid of linear search!
		for (b = 0; b < nb_equations; b++) {
			if (Sorting.int_vec_compare(Equations + b * dimension, v2, dimension) == 0) {
				break;
			}
		}
#endif

		if (!Table_of_equations->search(v2, b, 0 /* verbose_level */)) {
			cout << "action_on_homogeneous_polynomials::compute_image_int "
					"Table_of_equations->search is false, "
					"could not find equation" << endl;
			cout << "action_on_homogeneous_polynomials::compute_image_int "
					"a = " << a << " v1 = " << endl;
			Int_vec_print(cout, v1, dimension);
			cout << endl;
			cout << " v2 = v1 * A = " << endl;
			Int_vec_print(cout, v2, dimension);
			cout << endl;
			cout << "A=" << endl;
			A->Group_element->element_print_quick(Elt, cout);
#if 1
			{
				ofstream f("equations.txt");
				f << "equations:" << endl;
				int i;
				for (i = 0; i < nb_equations; i++) {
					f << setw(3) << i << " : ";
					Int_vec_print(f, Equations + i * dimension, dimension);
					f << endl;
				}
			}
			cout << "equations written to file equations.txt" << endl;
#endif
			exit(1);
		}

#if 0
		if (b == nb_equations) {
			cout << "action_on_homogeneous_polynomials::compute_image_int "
					"could not find equation" << endl;
			cout << "action_on_homogeneous_polynomials::compute_image_int "
					"a = " << a << " v1 = " << endl;
			Int_vec_print(cout, v1, dimension);
			cout << endl;
			cout << " v2 = v1 * A = " << endl;
			Int_vec_print(cout, v2, dimension);
			cout << endl;
			cout << "A=" << endl;
			A->element_print_quick(Elt, cout);
			cout << "equations:" << endl;
			int i;
			for (i = 0; i < nb_equations; i++) {
				cout << setw(3) << b << " : ";
				Int_vec_print(cout, Equations + i * dimension, dimension);
				cout << endl;
			}
			exit(1);
		}
#endif

		b = Table_of_equations->perm_inv[b];

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
	algebra::matrix_group *mtx;
	int n;
	
	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_image_int_low_level" << endl;
	}
	if (f_vv) {
		cout << "action_on_homogeneous_polynomials::compute_image_int_low_level "
				"input = ";
		Int_vec_print(cout, input, dimension);
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


	A->Group_element->element_invert(Elt, Elt1, 0);


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
		Int_vec_print(cout, output, dimension);
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
		Int_vec_zero(v1, dimension);
		v1[i] = 1;
		compute_image_int_low_level(
			Elt, v1, M + i * dimension,
			0 /* verbose_level */);
	}

	if (f_v) {
		cout << "action_on_homogeneous_polynomials::compute_representation done" << endl;
	}
}


}}}


