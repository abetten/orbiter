// action_on_wedge_product.C
//
// Anton Betten
// Jan 26, 2010

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

namespace orbiter {

action_on_wedge_product::action_on_wedge_product()
{
	null();
}

action_on_wedge_product::~action_on_wedge_product()
{
	free();
}

void action_on_wedge_product::null()
{
	M = NULL;
	F = NULL;
	wedge_v1 = NULL;
	wedge_v2 = NULL;
	wedge_v3 = NULL;
	low_level_point_size = 0;
}

void action_on_wedge_product::free()
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
	null();
}

void action_on_wedge_product::init(action &A, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_wedge_product::init" << endl;
		cout << "starting with action " << A.label << endl;
		}
	if (A.type_G != matrix_group_t) {
		cout << "action_on_wedge_product::init "
				"fatal: A.type_G != matrix_group_t" << endl;
		exit(1);
		}
	M = A.G.matrix_grp;
	F = M->GFq;
	n = M->n;
	q = F->q;
	wedge_dimension = (n * (n - 1)) >> 1;
	//degree = i_power_j(q, wedge_dimension);
	degree = nb_PG_elements(wedge_dimension - 1, q);
	low_level_point_size = wedge_dimension;
	wedge_v1 = NEW_int(wedge_dimension);
	wedge_v2 = NEW_int(wedge_dimension);
	wedge_v3 = NEW_int(wedge_dimension);
}

void action_on_wedge_product::unrank_point(int *v, int rk)
{
	F->PG_element_unrank_modified(v, 1, wedge_dimension, rk);
}

int action_on_wedge_product::rank_point(int *v)
{
	int rk;

	F->PG_element_rank_modified(v, 1, wedge_dimension, rk);
	return rk;
}

int action_on_wedge_product::compute_image_int(
	action &A, int *Elt, int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int b;
	
	if (f_v) {
		cout << "action_on_wedge_product::compute_image_int" << endl;
		}
	//AG_element_unrank(q, wedge_v1, 1, wedge_dimension, a);
	F->PG_element_unrank_modified(wedge_v1, 1, wedge_dimension, a);
	if (f_vv) {
		cout << "action_on_wedge_product::compute_image_int "
				"a = " << a << " wedge_v1 = ";
		int_vec_print(cout, wedge_v1, wedge_dimension);
		cout << endl;
		}
	
	compute_image_int_low_level(A, Elt, wedge_v1, wedge_v2, verbose_level);
	if (f_vv) {
		cout << " v2=v1 * A=";
		int_vec_print(cout, wedge_v2, wedge_dimension);
		cout << endl;
		}

	//AG_element_rank(q, wedge_v2, 1, wedge_dimension, b);
	F->PG_element_rank_modified(wedge_v2, 1, wedge_dimension, b);
	if (f_v) {
		cout << "action_on_wedge_product::compute_image_int "
				"done " << a << "->" << b << endl;
		}
	return b;
}

int action_on_wedge_product::element_entry_frobenius(
		action &A, int *Elt, int verbose_level)
{
	int f;

	f = A.element_linear_entry_frobenius(Elt, verbose_level);
	return f;
}

int action_on_wedge_product::element_entry_ij(
		action &A, int *Elt, int I, int J, int verbose_level)
{
	int i, j, k, l, w;

	k2ij(I, i, j, n);
	k2ij(J, k, l, n);
	w = element_entry_ijkl(A, Elt, i, j, k, l, verbose_level);
	return w;
}

int action_on_wedge_product::element_entry_ijkl(action &A,
		int *Elt, int i, int j, int k, int l, int verbose_level)
{
	int aki, alj, akj, ali, u, v, w;

	aki = A.element_linear_entry_ij(Elt, k, i, verbose_level); //Elt[k * n + i];
	alj = A.element_linear_entry_ij(Elt, l, j, verbose_level); //Elt[l * n + j];
	akj = A.element_linear_entry_ij(Elt, k, j, verbose_level); //Elt[k * n + j];
	ali = A.element_linear_entry_ij(Elt, l, i, verbose_level); //Elt[l * n + i];
	u = F->mult(aki, alj);
	v = F->mult(akj, ali);
	w = F->add(u, F->negate(v));
	return w;
}

void action_on_wedge_product::compute_image_int_low_level(
	action &A, int *Elt, int *input, int *output, int verbose_level)
// x_{i,j} e_i \wedge e_j * A = 
// \sum_{k < l} \sum_{i < j} x_{i,j} (a_{i,k}a_{j,l} - a_{i,l}a_{j,k}) e_k \wedge e_l
// or (after a change of indices)
// \sum_{i<j} x_{i,j} e_i \wedge e_j * A = 
//   \sum_{i < j} \sum_{k < l} x_{k,l} (a_{k,i}a_{l,j} - a_{k,j}a_{l,i}) e_i \wedge e_j
//
// so, the image of e_i \wedge e_j is 
// \sum_{k < l} x_{k,l} (a_{k,i}a_{l,j} - a_{k,j}a_{l,i}),
// which are the entries in the row indexed by (i,j).
{
	int *x = input;
	int *xA = output;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, ij, k, l, kl, c, w, z, xkl;
	
	if (f_v) {
		cout << "action_on_wedge_product::compute_image_int_low_level" << endl;
		}
	if (f_vv) {
		cout << "wedge action: x=";
		int_vec_print(cout, x, wedge_dimension);
		cout << endl;
		}
	// (i,j) = row index
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			ij = ij2k(i, j, n);
			c = 0;

			// (k,l) = column index
			for (k = 0; k < n; k++) {
				for (l = k + 1; l < n; l++) {
					kl = ij2k(k, l, n);
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
	
					w = element_entry_ijkl(A, Elt, i, j, k, l, verbose_level - 3);
					// now w is the matrix entry
					
					z = F->mult(xkl, w);
					c = F->add(c, z);

					if (z && f_v) {
						cout << "i=" << i << " j=" << j << " ij=" << ij << " k=" << k << " l=" << l << " kl=" << kl << " xkl=" << xkl << " w=" << w << " z=xkl*w=" << z << " c=" << c << endl;
						}
					} // next l
				} // next k
			if (c && f_v) {
				cout << "i=" << i << " j=" << j << " ij=" << ij << " xA[" << ij << "]=" << c << endl;
				}
			xA[ij] = c;
			} // next j
		} // next i
	if (f_vv) {
		cout << "xA=";
		int_vec_print(cout, xA, wedge_dimension);
		cout << endl;
		}
	if (M->f_semilinear) {
		int f = A.linear_entry_frobenius(Elt); //Elt[n * n];
		for (i = 0; i < wedge_dimension; i++) {
			xA[i] = F->frobenius_power(xA[i], f);
			}
		if (f_vv) {
			cout << "after " << f << " field automorphisms: xA=";
			int_vec_print(cout, xA, wedge_dimension);
			cout << endl;
			}
		}
	if (f_v) {
		cout << "action_on_wedge_product::compute_image_int_low_level done" << endl;
		}
}

}


