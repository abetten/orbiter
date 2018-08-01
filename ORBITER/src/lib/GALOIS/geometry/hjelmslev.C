// hjelmslev.C
//
// Anton Betten
//
//
// June 22, 2010
//
//
// 
//
//

#include "galois.h"

hjelmslev::hjelmslev()
{
	null();
}

hjelmslev::~hjelmslev()
{
	freeself();
}

void hjelmslev::null()
{
	R = NULL;
	G = NULL;
	Mtx = NULL;
	base_cols = NULL;
	v = NULL;
}

void hjelmslev::freeself()
{
	if (G) {
		delete G;
		}
	if (Mtx) {
		FREE_INT(Mtx);
		}
	if (base_cols) {
		FREE_INT(base_cols);
		}
	if (v) {
		FREE_INT(v);
		}
	null();
}

void hjelmslev::init(finite_ring *R, INT n, INT k, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hjelmslev::init n=" << n << " k=" << k << " q=" << R->q << endl;
		}
	hjelmslev::R = R;
	hjelmslev::n = n;
	hjelmslev::k = k;
	n_choose_k_p = generalized_binomial(n, k, R->p);
	if (f_v) {
		cout << "hjelmslev::init n_choose_k_p = " << n_choose_k_p << endl;
		}
	G = new grassmann;
	G->init(n, k, R->Fp, verbose_level);
	Mtx = NEW_INT(k * n);
	base_cols = NEW_INT(n);
	v = NEW_INT(k * (n - k));
}

INT hjelmslev::number_of_submodules()
{
	return n_choose_k_p * i_power_j(R->p, (R->e - 1) * k * (n - k));
}

void hjelmslev::unrank_INT(INT *M, INT rk, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT a, b, c, i, j, h;
	
	if (f_v) {
		cout << "hjelmslev::unrank_INT " << rk << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	if (k == 0) {
		return;
		}
	a = rk % n_choose_k_p;
	b = (rk - a) / n_choose_k_p;
	if (f_vv) {
		cout << "rk=" << rk << " a=" << a << " b=" << b << endl;
		}
	G->unrank_INT(a, 0);
	AG_element_unrank(R->e, v, 1, k * (n - k), b);
	if (f_vv) {
		print_integer_matrix_width(cout, G->M, k, n, n, 5);
		INT_vec_print(cout, v, k * (n - k));
		cout << endl;
		}
	for (i = 0; i < k * n; i++) {
		Mtx[i] = G->M[i];
		}
	for (j = 0; j < n - k; j++) {
		h = G->base_cols[k + j];
		for (i = 0; i < k; i++) {
			c = v[i * (n - k) + j];
			Mtx[i * n + h] += c * R->p;
			}
		}
	for (i = 0; i < k * n; i++) {
		M[i] = Mtx[i];
		}
}

INT hjelmslev::rank_INT(INT *M, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT a, b, c, i, j, h, rk, rk_mtx;
	INT f_special = FALSE;
	INT f_complete = TRUE;
	
	if (f_v) {
		cout << "hjelmslev::rank_INT " << endl;
		print_integer_matrix_width(cout, M, k, n, n, 5);
		cout << "verbose_level=" << verbose_level << endl;
		}
	for (i = 0; i < k * n; i++) {
		Mtx[i] = M[i];
		}
	rk_mtx = R->Gauss_INT(Mtx, f_special, f_complete, base_cols, FALSE, NULL, k, n, n, 0);
	if (f_v) {
		cout << "hjelmslev::rank_INT after Gauss, rk_mtx=" << rk_mtx << endl;
		print_integer_matrix_width(cout, Mtx, k, n, n, 5);
		cout << "base_cols=";
		INT_vec_print(cout, base_cols, rk_mtx);
		cout << endl;
		}
	INT_vec_complement(base_cols, n, k);
	if (rk_mtx != k) {
		cout << "hjelmslev::rank_INT fatal: rk_mtx != k" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "complement:";
		INT_vec_print(cout, base_cols + k, n - k);
		cout << endl;
		}
	for (j = 0; j < n - k; j++) {
		h = G->base_cols[k + j];
		for (i = 0; i < k; i++) {
			c = Mtx[i * n + h] / R->p;
			v[i * (n - k) + j] = c;
			Mtx[i * n + h] -= c * R->p;
			}
		}
	
	for (i = 0; i < k * n; i++) {
		G->M[i] = Mtx[i];
		}
	if (f_vv) {
		INT_vec_print(cout, v, k * (n - k));
		cout << endl;
		}
	AG_element_rank(R->e, v, 1, k * (n - k), b);
	a = G->rank_INT(0);
	rk = b * n_choose_k_p + a;
	if (f_v) {
		cout << "hjelmslev::rank_INT rk=" << rk << " a=" << a << " b=" << b << endl;
		}
	return rk;
}

