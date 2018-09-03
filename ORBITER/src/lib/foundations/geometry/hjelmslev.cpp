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

#include "foundations.h"

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
		FREE_OBJECT(G);
		}
	if (Mtx) {
		FREE_int(Mtx);
		}
	if (base_cols) {
		FREE_int(base_cols);
		}
	if (v) {
		FREE_int(v);
		}
	null();
}

void hjelmslev::init(finite_ring *R, int n, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

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
	G = NEW_OBJECT(grassmann);
	G->init(n, k, R->Fp, verbose_level);
	Mtx = NEW_int(k * n);
	base_cols = NEW_int(n);
	v = NEW_int(k * (n - k));
}

int hjelmslev::number_of_submodules()
{
	return n_choose_k_p * i_power_j(R->p, (R->e - 1) * k * (n - k));
}

void hjelmslev::unrank_int(int *M, int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a, b, c, i, j, h;
	
	if (f_v) {
		cout << "hjelmslev::unrank_int " << rk << endl;
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
	G->unrank_int(a, 0);
	AG_element_unrank(R->e, v, 1, k * (n - k), b);
	if (f_vv) {
		print_integer_matrix_width(cout, G->M, k, n, n, 5);
		int_vec_print(cout, v, k * (n - k));
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

int hjelmslev::rank_int(int *M, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a, b, c, i, j, h, rk, rk_mtx;
	int f_special = FALSE;
	int f_complete = TRUE;
	
	if (f_v) {
		cout << "hjelmslev::rank_int " << endl;
		print_integer_matrix_width(cout, M, k, n, n, 5);
		cout << "verbose_level=" << verbose_level << endl;
		}
	for (i = 0; i < k * n; i++) {
		Mtx[i] = M[i];
		}
	rk_mtx = R->Gauss_int(Mtx, f_special, f_complete, base_cols, FALSE, NULL, k, n, n, 0);
	if (f_v) {
		cout << "hjelmslev::rank_int after Gauss, rk_mtx=" << rk_mtx << endl;
		print_integer_matrix_width(cout, Mtx, k, n, n, 5);
		cout << "base_cols=";
		int_vec_print(cout, base_cols, rk_mtx);
		cout << endl;
		}
	int_vec_complement(base_cols, n, k);
	if (rk_mtx != k) {
		cout << "hjelmslev::rank_int fatal: rk_mtx != k" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "complement:";
		int_vec_print(cout, base_cols + k, n - k);
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
		int_vec_print(cout, v, k * (n - k));
		cout << endl;
		}
	AG_element_rank(R->e, v, 1, k * (n - k), b);
	a = G->rank_int(0);
	rk = b * n_choose_k_p + a;
	if (f_v) {
		cout << "hjelmslev::rank_int rk=" << rk << " a=" << a << " b=" << b << endl;
		}
	return rk;
}

