// galois_global.C
//
// Anton Betten
//
// started: Oct 16, 2013
//
// unipoly stuff:
// started:  November 16, 2002
// moved here: Oct 16, 2013




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


void test_unipoly()
{
	finite_field GFp;
	int p = 2;
	unipoly_object m, a, b, c;
	unipoly_object elts[4];
	int i, j;
	int verbose_level = 0;
	
	GFp.init(p, verbose_level);
	unipoly_domain FX(&GFp);
	
	FX.create_object_by_rank(m, 7);
	FX.create_object_by_rank(a, 5);
	FX.create_object_by_rank(b, 55);
	FX.print_object(a, cout); cout << endl;
	FX.print_object(b, cout); cout << endl;

	unipoly_domain Fq(&GFp, m);
	Fq.create_object_by_rank(c, 2);
	for (i = 0; i < 4; i++) {
		Fq.create_object_by_rank(elts[i], i);
		cout << "elt_" << i << " = ";
		Fq.print_object(elts[i], cout); cout << endl;
		}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			Fq.print_object(elts[i], cout);
			cout << " * ";
			Fq.print_object(elts[j], cout);
			cout << " = ";
			Fq.mult(elts[i], elts[j], c);
			Fq.print_object(c, cout); cout << endl;
			
			FX.mult(elts[i], elts[j], a);
			FX.print_object(a, cout); cout << endl;
			}
		}
	
}

void test_unipoly2()
{
	finite_field Fq;
	int q = 4, p = 2, i;
	int verbose_level = 0;
	
	Fq.init(q, verbose_level);
	unipoly_domain FX(&Fq);
	
	unipoly_object a;
	
	FX.create_object_by_rank(a, 0);
	for (i = 1; i < q; i++) {
		FX.minimum_polynomial(a, i, p, TRUE);
		//cout << "minpoly_" << i << " = ";
		//FX.print_object(a, cout); cout << endl;
		}
	
}

char *search_for_primitive_polynomial_of_given_degree(
		int p, int degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field Fp;
	
	Fp.init(p, 0 /*verbose_level*/);
	unipoly_domain FX(&Fp);
	
	unipoly_object m;
	longinteger_object rk;
	
	FX.create_object_by_rank(m, 0);
	
	if (f_v) {
		cout << "search_for_primitive_polynomial_of_given_degree "
				"p=" << p << " degree=" << degree << endl;
		}
	FX.get_a_primitive_polynomial(m, degree, verbose_level - 1);
	FX.rank_longinteger(m, rk);
	
	char *s;
	int i, j;
	if (f_v) {
		cout << "found a polynomial. It's rank is " << rk << endl;
		}
	
	s = NEW_char(rk.len() + 1);
	for (i = rk.len() - 1, j = 0; i >= 0; i--, j++) {
		s[j] = '0' + rk.rep()[i];
		}
	s[rk.len()] = 0;
	
	if (f_v) {
		cout << "created string " << s << endl;
		}
	
	return s;
}


void search_for_primitive_polynomials(
		int p_min, int p_max, int n_min, int n_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d, p;
	number_theory_domain NT;
	

	longinteger_f_print_scientific = FALSE;

	
	if (f_v) {
		cout << "search_for_primitive_polynomials "
				"p_min=" << p_min << " p_max=" << p_max
				<< " n_min=" << n_min << " n_max=" << n_max << endl;
		}
	for (p = p_min; p <= p_max; p++) {
		if (!NT.is_prime(p)) {
			continue;
			}
		if (f_v) {
			cout << "considering the prime " << p << endl;
			}

			{
			finite_field Fp;
			Fp.init(p, 0 /*verbose_level*/);
			unipoly_domain FX(&Fp);
	
			unipoly_object m;
			longinteger_object rk;
	
			FX.create_object_by_rank(m, 0);
	
			for (d = n_min; d <= n_max; d++) {
				if (f_v) {
					cout << "d=" << d << endl;
					}
				FX.get_a_primitive_polynomial(m, d, verbose_level - 1);
				FX.rank_longinteger(m, rk);
				//cout << d << " : " << rk << " : ";
				cout << "\"" << rk << "\", // ";
				FX.print_object(m, cout);
				cout << endl;
				}
			FX.delete_object(m);
			}
		}
}


void make_linear_irreducible_polynomials(int q, int &nb,
		int *&table, int verbose_level)
{
	int i;
	
	finite_field F;
	F.init(q, 0 /*verbose_level*/);
#if 0
	if (f_no_eigenvalue_one) {
		nb = q - 2;
		table = NEW_int(nb * 2);
		for (i = 0; i < nb; i++) {
			table[i * 2 + 0] = F.negate(i + 2);
			table[i * 2 + 1] = 1;
			}
		}
	else {
#endif
		nb = q - 1;
		table = NEW_int(nb * 2);
		for (i = 0; i < nb; i++) {
			table[i * 2 + 0] = F.negate(i + 1);
			table[i * 2 + 1] = 1;
			}
#if 0
		}
#endif
}



void gl_random_matrix(int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 1);
	int *M;
	int *M2;
	finite_field F;
	unipoly_object char_poly;

	if (f_v) {
		cout << "gl_random_matrix" << endl;
		}
	F.init(q, 0 /*verbose_level*/);
	M = NEW_int(k * k);
	M2 = NEW_int(k * k);
	
	F.random_invertible_matrix(M, k, verbose_level - 2);

	cout << "Random invertible matrix:" << endl;
	int_matrix_print(M, k, k);

	
	{
	unipoly_domain U(&F);



	U.create_object_by_rank(char_poly, 0);
		
	U.characteristic_polynomial(M, k, char_poly, verbose_level - 2);

	cout << "The characteristic polynomial is ";
	U.print_object(char_poly, cout);
	cout << endl;

	U.substitute_matrix_in_polynomial(char_poly, M, M2, k, verbose_level);
	cout << "After substitution, the matrix is " << endl;
	int_matrix_print(M2, k, k);

	U.delete_object(char_poly);

	}
	FREE_int(M);
	FREE_int(M2);

}

int is_diagonal_matrix(int *A, int n)
{
	int i, j;
	
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				continue;
				}
			else {
				if (A[i * n + j]) {
					return FALSE;
					}
				}
			}
		}
	return TRUE;
}

int is_association_scheme(int *color_graph, int n, int *&Pijk, 
	int *&colors, int &nb_colors, int verbose_level)
// color_graph[n * n]
// added Dec 22, 2010.
//Originally in BLT_ANALYZE/analyze_plane_invariant.C
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int N;
	int *M1;
	int k, i, j;
	int ret = FALSE;
	
	if (f_v) {
		cout << "is_association_scheme" << endl;
		}
	N = (n * (n - 1)) / 2;
	M1 = NEW_int(N);
	k = 0;
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			M1[k++] = color_graph[i * n + j];
			}
		}
	if (k != N) {
		cout << "N=" << N << endl;
		cout << "k=" << k << endl;
		exit(1);
		}

	classify Cl;

	Cl.init(M1, N, FALSE, 0);
	nb_colors = Cl.nb_types + 1;
	colors = NEW_int(nb_colors);
	colors[0] = color_graph[0];
	for (i = 0; i < Cl.nb_types; i++) {
		colors[i + 1] = Cl.data_sorted[Cl.type_first[i]];
		}

	if (f_vv) {
		cout << "colors (the 0-th color is the diagonal color): ";
		int_vec_print(cout, colors, nb_colors);
		cout << endl;
		}

	int C = nb_colors;
	int *M = color_graph;
	int pijk, pijk1, u, v, w, u0 = 0, v0 = 0;
	
	Pijk = NEW_int(C * C * C);
	int_vec_zero(Pijk, C * C * C);
	for (k = 0; k < C; k++) {
		for (i = 0; i < C; i++) {
			for (j = 0; j < C; j++) {
				pijk = -1;
				for (u = 0; u < n; u++) {
					for (v = 0; v < n; v++) {
						//if (v == u) continue;
						if (M[u * n + v] != colors[k])
							continue;
						// now: edge (u,v) is colored k
						pijk1 = 0;
						for (w = 0; w < n; w++) {
							//if (w == u)continue;
							//if (w == v)continue;
							if (M[u * n + w] != colors[i])
								continue;
							if (M[v * n + w] != colors[j])
								continue;
							//cout << "i=" << i << " j=" << j << " k=" << k
							//<< " u=" << u << " v=" << v << " w=" << w
							//<< " increasing pijk" << endl;
							pijk1++;
							} // next w
						//cout << "i=" << i << " j=" << j << " k=" << k
						//<< " u=" << u << " v=" << v
						//<< " pijk1=" << pijk1 << endl;
						if (pijk == -1) {
							pijk = pijk1;
							u0 = u;
							v0 = v;
							//cout << "u=" << u << " v=" << v
							//<< " p_{" << i << "," << j << ","
							//<< k << "}="
							//<< Pijk[i * C * C + j * C + k] << endl;
							}
						else {
							if (pijk1 != pijk) {
								//FREE_int(Pijk);
								//FREE_int(colors);

								cout << "not an association scheme" << endl;
								cout << "k=" << k << endl;
								cout << "i=" << i << endl;
								cout << "j=" << j << endl;
								cout << "u0=" << u0 << endl;
								cout << "v0=" << v0 << endl;
								cout << "pijk=" << pijk << endl;
								cout << "u=" << u << endl;
								cout << "v=" << v << endl;
								cout << "pijk1=" << pijk1 << endl;
								//exit(1);

								goto done;
								}
							}
						} // next v
					} // next u
				Pijk[i * C * C + j * C + k] = pijk;
				} // next j
			} // next i
		} // next k

	ret = TRUE;

	if (f_v) {
		cout << "it is an association scheme" << endl;


		if (f_v) {
			print_Pijk(Pijk, C);
			}

		if (C == 3 && colors[1] == 0 && colors[2] == 1) {
			int k, lambda, mu;

			k = Pijk[2 * C * C + 2 * C + 0]; // p220;
			lambda = Pijk[2 * C * C + 2 * C + 2]; // p222;
			mu = Pijk[2 * C * C + 2 * C + 1]; // p221;
			cout << "it is an srg(" << n << "," << k << ","
					<< lambda << "," << mu << ")" << endl;
			}


		}
	

done:
	FREE_int(M1);
	return ret;
}

void print_Pijk(int *Pijk, int nb_colors)
{
	int i, j, k;
	int C = nb_colors;
	
	for (k = 0; k < C; k++) {
		int *Mtx;

		Mtx = NEW_int(C * C);
		for (i = 0; i < C; i++) {
			for (j = 0; j < C; j++) {
				Mtx[i * C + j] = Pijk[i * C * C + j * C + k];
				}
			}
		cout << "P^{(" << k << ")}=(p_{i,j," << k << "})_{i,j}:" << endl;
		print_integer_matrix_width(cout, Mtx, C, C, C, 3);
		FREE_int(Mtx);
		}
}





int str2int(string &str)
{
	int i, res, l;
	
	l = (int) str.length();
	res = 0;
	for (i = 0; i < l; i++) {
		res = (res * 10) + (str[i] - 48);
		}
	return res;
}

void print_longinteger_after_multiplying(
		ostream &ost, int *factors, int len)
{
	longinteger_domain D;
	longinteger_object a;

	D.multiply_up(a, factors, len);
	ost << a;
}

void compute_decomposition_of_graph_wrt_partition(
	int *Adj, int N,
	int *first, int *len, int nb_parts, int *&R,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int I, J, i, j, f1, l1, f2, l2, r0 = 0, r;

	if (f_v) {
		cout << "compute_decomposition_of_graph_wrt_partition" << endl;
		cout << "The partition is:" << endl;
		cout << "first = ";
		int_vec_print(cout, first, nb_parts);
		cout << endl;
		cout << "len = ";
		int_vec_print(cout, len, nb_parts);
		cout << endl;
		}
	R = NEW_int(nb_parts * nb_parts);
	int_vec_zero(R, nb_parts * nb_parts);
	for (I = 0; I < nb_parts; I++) {
		f1 = first[I];
		l1 = len[I];
		for (J = 0; J < nb_parts; J++) {
			f2 = first[J];
			l2 = len[J];
			for (i = 0; i < l1; i++) {
				r = 0;
				for (j = 0; j < l2; j++) {
					if (Adj[(f1 + i) * N + f2 + j]) {
						r++;
						}
					}
				if (i == 0) {
					r0 = r;
					}
				else {
					if (r0 != r) {
						cout << "compute_decomposition_of_graph_"
							"wrt_partition not tactical" << endl;
						cout << "I=" << I << endl;
						cout << "J=" << J << endl;
						cout << "r0=" << r0 << endl;
						cout << "r=" << r << endl;
						exit(1); 
						}
					}
				}
			R[I * nb_parts + J] = r0;
			}
		}
	if (f_v) {
		cout << "compute_decomposition_of_graph_wrt_partition done" << endl;
		}
}

}
}


