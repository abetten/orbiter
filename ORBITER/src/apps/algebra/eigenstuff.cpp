// eigenstuff.C
// 
// Anton Betten
// 12/14/09
//
// Computes the characteristic polynomial of a matrix 
// using the Smith normal form.
// This part uses DISCRETA data structures.
// Then finds all eigenvalues as roots of the characteristic polynomial in GF(q).
// For each of the eigenvalues, we then find a basis 
// of the space of left-eigenvectors .
// This part uses only basic finite_field operations, but no DISCRETA parts.
//
// Usage:
// -M <size> <q> <i_1> .. <i_{size*size}>
// where <i_1>, ... <i_{size*size}> are the entries of the matrix 
// in a row-by-row fashion and <q> is the order of the field

#include "orbiter.h"

#include <fstream>

using namespace orbiter;

// global data:

int t0; // the system time when the program started


void do_eigenstuff(int q, int size, int *Data, int verbose_level);

int main(int argc, char **argv)
{
	t0 = os_ticks();
	discreta_init();
	int verbose_level = 0;
	int i;
	int f_M = FALSE;
	int size = 0;
	const char *data;
	int *Data;
	int Data_len = 0;
	int f_q = FALSE;
	int q = 0;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-M") == 0) {
			f_M = TRUE;
			size = atoi(argv[++i]);
			data = argv[++i];
			cout << "-M " << size << " " << data << endl;
			}
		}

	if (!f_q) {
		cout << "please use option -q <q> to specify "
				"the field size" << endl;
		exit(1);
		}
	cout << "q=" << q << endl;
	if (!f_M) {
		cout << "please use option -M <size> <list of entries> "
			"to specify the matrix" << endl;
		exit(1);
		}
	int_vec_scan(data, Data, Data_len);
	if (Data_len != size * size) {
		cout << "Data_len != size * size" << endl;
		exit(1);
		}
	

	do_eigenstuff(q, size, Data, verbose_level);
	
	the_end_quietly(t0);
}

void do_eigenstuff(int q, int size, int *Data, int verbose_level)
{
	matrix M;
	int i, j, k, a, p, h;
	finite_field Fq;
	//unipoly_domain U;
	//unipoly_object char_poly;

	M.m_mn(size, size);
	k = 0;
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a = Data[k++];
			M.m_iji(i, j, a);
			}
		}
	cout << "M=" << endl;
	cout << M << endl;

	if (!is_prime_power(q, p, h)) {
		cout << "q is not prime, we need a prime" << endl;
		exit(1);
		}
	Fq.init(q, verbose_level);

	domain d(q);
	with w(&d);

#if 0

	matrix M2;
	M2 = M;
	for (i = 0; i < size; i++) {
		unipoly mue;
		M2.KX_module_order_ideal(i, mue, verbose_level - 1);
		cout << "order ideal " << i << ":" << endl;
		cout << mue << endl;
		}
#endif
	
	// This part uses DISCRETA data structures:

	matrix M1, P, Pv, Q, Qv, S, T;
	
	M.elements_to_unipoly();
	M.minus_X_times_id();
	M1 = M;
	cout << "M - x * Id=" << endl << M << endl;
	M.smith_normal_form(P, Pv, Q, Qv, 0 /*verbose_level*/);

	cout << "the Smith normal form is:" << endl;
	cout << M << endl;

	S.mult(P, Pv);
	cout << "P * Pv=" << endl << S << endl;

	S.mult(Q, Qv);
	cout << "Q * Qv=" << endl << S << endl;

	S.mult(P, M1);
	cout << "T.mult(S, Q):" << endl;
	T.mult(S, Q);
	cout << "T=" << endl << T << endl;


	unipoly charpoly;
	int deg;
	int l, lv, b, c;

	charpoly = M.s_ij(size - 1, size - 1);
		
	cout << "characteristic polynomial:" << charpoly << endl;
	deg = charpoly.degree();
	cout << "has degree " << deg << endl;
	l = charpoly.s_ii(deg);
	cout << "leading coefficient " << l << endl;
	lv = Fq.inverse(l);
	cout << "leading coefficient inverse " << lv << endl;
	for (i = 0; i <= deg; i++) {
		b = charpoly.s_ii(i);
		c = Fq.mult(b, lv);
		charpoly.m_ii(i, c);
		}
	cout << "monic characteristic polynomial:" << charpoly << endl;
	
	integer x, y;
	int *roots;
	int nb_roots = 0;

	roots = new int[q];

	for (a = 0; a < q; a++) {
		x.m_i(a);
		charpoly.evaluate_at(x, y);
		if (y.s_i() == 0) {
			cout << "root " << a << endl;
			roots[nb_roots++] = a;
			}
		}
	cout << "we found the following eigenvalues: ";
	int_vec_print(cout, roots, nb_roots);
	cout << endl;
	
	int eigenvalue, eigenvalue_negative;

	for (h = 0; h < nb_roots; h++) {
		eigenvalue = roots[h];
		cout << "looking at eigenvalue " << eigenvalue << endl;
		int *A, *B, *Bt;
		eigenvalue_negative = Fq.negate(eigenvalue);
		A = new int[size * size];
		B = new int[size * size];
		Bt = new int[size * size];
		for (i = 0; i < size; i++) {
			for (j = 0; j < size; j++) {
				A[i * size + j] = Data[i * size + j];
				}
			}
		cout << "A:" << endl;
		print_integer_matrix_width(cout, A,
				size, size, size, Fq.log10_of_q);
		for (i = 0; i < size; i++) {
			for (j = 0; j < size; j++) {
				a = A[i * size + j];
				if (j == i) {
					a = Fq.add(a, eigenvalue_negative);
					}
				B[i * size + j] = a;
				}
			}
		cout << "B = A - eigenvalue * I:" << endl;
		print_integer_matrix_width(cout, B,
				size, size, size, Fq.log10_of_q);
		
		cout << "B transposed:" << endl;
		Fq.transpose_matrix(B, Bt, size, size);
		print_integer_matrix_width(cout, Bt,
				size, size, size, Fq.log10_of_q);

		int f_special = FALSE;
		int f_complete = TRUE;
		int *base_cols;
		int nb_base_cols;
		int f_P = FALSE;
		int kernel_m, kernel_n, *kernel;

		base_cols = new int[size];
		kernel = new int[size * size];
		
		nb_base_cols = Fq.Gauss_int(Bt,
			f_special, f_complete, base_cols,
			f_P, NULL, size, size, size,
			verbose_level - 1);
		cout << "rank = " << nb_base_cols << endl;
		
		Fq.matrix_get_kernel(Bt, size, size, base_cols, nb_base_cols, 
			kernel_m, kernel_n, kernel);
		cout << "kernel = left eigenvectors:" << endl;
		print_integer_matrix_width(cout, kernel,
				size, kernel_n, kernel_n, Fq.log10_of_q);
		
		int *vec1, *vec2;
		vec1 = new int[size];
		vec2 = new int[size];
		for (i = 0; i < size; i++) {
			vec1[i] = kernel[i * kernel_n + 0];
			}
		int_vec_print(cout, vec1, size);
		cout << endl;
		Fq.PG_element_normalize_from_front(vec1, 1, size);
		int_vec_print(cout, vec1, size);
		cout << endl;
		Fq.PG_element_rank_modified(vec1, 1, size, a);
		cout << "has rank " << a << endl;

		
		cout << "computing xA" << endl;

		Fq.mult_vector_from_the_left(vec1, A, vec2, size, size);
		int_vec_print(cout, vec2, size);
		cout << endl;
		Fq.PG_element_normalize_from_front(vec2, 1, size);
		int_vec_print(cout, vec2, size);
		cout << endl;
		Fq.PG_element_rank_modified(vec2, 1, size, a);
		cout << "has rank " << a << endl;

		delete [] vec1;
		delete [] vec2;
		
		delete [] A;
		delete [] B;
		delete [] Bt;
		}
}



