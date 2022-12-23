// action_on_wedge_product.cpp
//
// Anton Betten
// Jan 26, 2010

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_wedge_product::action_on_wedge_product()
{
	A = NULL;
	n = q = 0;
	M = NULL;
	F = NULL;
	low_level_point_size = 0;
	degree = 0;
	wedge_dimension = 0;
	wedge_v1 = NULL;
	wedge_v2 = NULL;
	wedge_v3 = NULL;
	Mtx_wedge = NULL;
}

action_on_wedge_product::~action_on_wedge_product()
{
	if (wedge_v1) {
		FREE_int(wedge_v1);
	}
	if (wedge_v2) {
		FREE_int(wedge_v2);
	}
	if (wedge_v3) {
		FREE_int(wedge_v3);
	}
	if (Mtx_wedge) {
		FREE_int(Mtx_wedge);
	}
}

void action_on_wedge_product::init(actions::action *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "action_on_wedge_product::init" << endl;
		cout << "starting with action " << A->label << endl;
	}
	if (A->type_G != matrix_group_t) {
		cout << "action_on_wedge_product::init "
				"fatal: A.type_G != matrix_group_t" << endl;
		exit(1);
	}
	action_on_wedge_product::A = A;
	M = A->G.matrix_grp;
	F = M->GFq;
	n = M->n;
	q = F->q;
	wedge_dimension = (n * (n - 1)) >> 1;
	//degree = i_power_j(q, wedge_dimension);
	degree = Gg.nb_PG_elements(wedge_dimension - 1, q);
	low_level_point_size = wedge_dimension;
	wedge_v1 = NEW_int(wedge_dimension);
	wedge_v2 = NEW_int(wedge_dimension);
	wedge_v3 = NEW_int(wedge_dimension);
	Mtx_wedge = NEW_int(wedge_dimension * wedge_dimension);
	if (f_v) {
		cout << "action_on_wedge_product::init done" << endl;
	}
}

void action_on_wedge_product::unrank_point(int *v, long int rk)
{
	F->PG_element_unrank_modified_lint(v, 1, wedge_dimension, rk);
}

long int action_on_wedge_product::rank_point(int *v)
{
	long int rk;

	F->PG_element_rank_modified_lint(v, 1, wedge_dimension, rk);
	return rk;
}

long int action_on_wedge_product::compute_image_int(
		int *Elt, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int b;
	
	if (f_v) {
		cout << "action_on_wedge_product::compute_image_int" << endl;
	}
	//AG_element_unrank(q, wedge_v1, 1, wedge_dimension, a);
	F->PG_element_unrank_modified_lint(wedge_v1, 1, wedge_dimension, a);
	if (f_vv) {
		cout << "action_on_wedge_product::compute_image_int "
				"a = " << a << " wedge_v1 = ";
		Int_vec_print(cout, wedge_v1, wedge_dimension);
		cout << endl;
	}
	
	compute_image_int_low_level(Elt, wedge_v1, wedge_v2, verbose_level);
	if (f_vv) {
		cout << " v2=v1 * A=";
		Int_vec_print(cout, wedge_v2, wedge_dimension);
		cout << endl;
	}

	//AG_element_rank(q, wedge_v2, 1, wedge_dimension, b);
	F->PG_element_rank_modified_lint(wedge_v2, 1, wedge_dimension, b);
	if (f_v) {
		cout << "action_on_wedge_product::compute_image_int "
				"done " << a << "->" << b << endl;
	}
	return b;
}

int action_on_wedge_product::element_entry_frobenius(
		int *Elt, int verbose_level)
{
	int f;

	f = A->element_linear_entry_frobenius(Elt, verbose_level);
	return f;
}

int action_on_wedge_product::element_entry_ij(
		int *Elt, int I, int J, int verbose_level)
{
	int i, j, k, l, w;
	combinatorics::combinatorics_domain Combi;

	Combi.k2ij(I, i, j, n);
	Combi.k2ij(J, k, l, n);
	w = element_entry_ijkl(Elt, i, j, k, l, verbose_level);
	return w;
}

int action_on_wedge_product::element_entry_ijkl(
		int *Elt, int i, int j, int k, int l, int verbose_level)
{
	int aki, alj, akj, ali, u, v, w;

	aki = A->element_linear_entry_ij(Elt, k, i, verbose_level); //Elt[k * n + i];
	alj = A->element_linear_entry_ij(Elt, l, j, verbose_level); //Elt[l * n + j];
	akj = A->element_linear_entry_ij(Elt, k, j, verbose_level); //Elt[k * n + j];
	ali = A->element_linear_entry_ij(Elt, l, i, verbose_level); //Elt[l * n + i];
	u = F->mult(aki, alj);
	v = F->mult(akj, ali);
	w = F->add(u, F->negate(v));
	return w;
}

void action_on_wedge_product::compute_image_int_low_level(
		int *Elt, int *input, int *output, int verbose_level)
// \sum_{i < j}  x_{i,j} e_i \wedge e_j * A =
// \sum_{k < l} \sum_{i < j} x_{i,j} (a_{i,k}a_{j,l} - a_{i,l}a_{j,k}) e_k \wedge e_l
// or (after a change of indices)
// \sum_{i<j} x_{i,j} e_i \wedge e_j * A = 
//   \sum_{i < j} \sum_{k < l} x_{k,l} (a_{k,i}a_{l,j} - a_{k,j}a_{l,i}) e_i \wedge e_j
//
// so, the image of e_i \wedge e_j is 
// \sum_{k < l} x_{k,l} (a_{k,i}a_{l,j} - a_{k,j}a_{l,i}) e_k \wedge e_l,
// =  \sum_{k < l} x_{k,l} w_{ij,kl} e_k \wedge e_l,
// The w_{ij,kl} are the entries in the row indexed by (i,j).
// w_{ij,kl} is the entry in row ij and column kl.
{
	int *x = input;
	int *xA = output;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "action_on_wedge_product::compute_image_int_low_level" << endl;
	}
	if (f_vv) {
		cout << "action_on_wedge_product::compute_image_int_low_level: x=";
		Int_vec_print(cout, x, wedge_dimension);
		cout << endl;
	}
#if 0
	int i, j, ij, k, l, kl, c, w, z, xkl;
	combinatorics::combinatorics_domain Combi;
	// (i,j) = row index
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			ij = Combi.ij2k(i, j, n);
			c = 0;

			// (k,l) = column index
			for (k = 0; k < n; k++) {
				for (l = k + 1; l < n; l++) {
					kl = Combi.ij2k(k, l, n);
					xkl = x[kl];


					// a_{k,i}a_{l,j} - a_{k,j}a_{l,i} = matrix entry
#if 0

					aki = Elt[k * n + i];
					alj = Elt[l * n + j];
					akj = Elt[k * n + j];
					ali = Elt[l * n + i];
					u = F->mult(aki, alj);
					v = F->mult(akj, ali);
					w = F->add(u, F->negate(v));
#endif

	
					w = element_entry_ijkl(Elt, i, j, k, l, verbose_level - 3);
					// now w is the matrix entry
					
					z = F->mult(xkl, w);
					c = F->add(c, z);

					if (z && f_v) {
						cout << "action_on_wedge_product::compute_image_int_low_level "
								"i=" << i << " j=" << j << " ij=" << ij << " k=" << k << " l=" << l << " kl=" << kl << " xkl=" << xkl << " w=" << w << " z=xkl*w=" << z << " c=" << c << endl;
					}
				} // next l
			} // next k
			if (c && f_v) {
				cout << "action_on_wedge_product::compute_image_int_low_level "
						"i=" << i << " j=" << j << " ij=" << ij << " xA[" << ij << "]=" << c << endl;
			}
			xA[ij] = c;
		} // next j
	} // next i
#else


	create_induced_matrix(Elt, Mtx_wedge, verbose_level - 2);

	F->Linear_algebra->mult_vector_from_the_right(Mtx_wedge, x,
			xA, wedge_dimension, wedge_dimension);

#endif

	if (f_vv) {
		cout << "action_on_wedge_product::compute_image_int_low_level xA=";
		Int_vec_print(cout, xA, wedge_dimension);
		cout << endl;
	}

	if (M->f_semilinear) {

		int f = A->linear_entry_frobenius(Elt); //Elt[n * n];

#if 0
		for (i = 0; i < wedge_dimension; i++) {
			xA[i] = F->frobenius_power(xA[i], f);
		}
#else
		F->Linear_algebra->vector_frobenius_power_in_place(xA, wedge_dimension, f);
#endif

		if (f_vv) {
			cout << "action_on_wedge_product::compute_image_int_low_level "
					"after " << f << " field automorphisms: xA=";
			Int_vec_print(cout, xA, wedge_dimension);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "action_on_wedge_product::compute_image_int_low_level done" << endl;
	}
}

void action_on_wedge_product::create_induced_matrix(
		int *Elt, int *Mtx2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_wedge_product::create_induced_matrix" << endl;
	}
	int i, j, ij, k, l, kl;
	int w;
	combinatorics::combinatorics_domain Combi;

	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {

			// (i,j) = row index
			ij = Combi.ij2k(i, j, n);

			for (k = 0; k < n; k++) {
				for (l = k + 1; l < n; l++) {

					// (k,l) = column index
					kl = Combi.ij2k(k, l, n);


					// a_{k,i}a_{l,j} - a_{k,j}a_{l,i} = matrix entry
#if 0

					aki = Elt[k * n + i];
					alj = Elt[l * n + j];
					akj = Elt[k * n + j];
					ali = Elt[l * n + i];
					u = F->mult(aki, alj);
					v = F->mult(akj, ali);
					w = F->add(u, F->negate(v));
#endif


					w = element_entry_ijkl(Elt, i, j, k, l, verbose_level - 3);
					// now w is the matrix entry
					Mtx2[ij * wedge_dimension + kl] = w;
				}
			}
		}
	}

	if (f_v) {
		cout << "action_on_wedge_product::create_induced_matrix done" << endl;
	}
}

void action_on_wedge_product::element_print_latex(int *Elt, std::ostream &ost)
{
	ost << "\\left(" << endl;
	F->print_matrix_latex(ost, Elt, n, n);
	if (M->f_semilinear) {
		ost << "_{" << Elt[n * n] << "}" << endl;
	}
	ost << "=";

	create_induced_matrix(Elt, Mtx_wedge, 0 /* verbose_level */);

	F->print_matrix_latex(ost, Mtx_wedge, wedge_dimension, wedge_dimension);
	if (M->f_semilinear) {
		ost << "_{" << Elt[n * n] << "}" << endl;
	}
	ost << "\\right)" << endl;

	int f;

	f = A->count_fixed_points(Elt, 0 /* verbose_level */);

	ost << "f=" << f << endl;
}



}}}



