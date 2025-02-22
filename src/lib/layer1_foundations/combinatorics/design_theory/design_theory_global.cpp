/*
 * design_theory_global.cpp
 *
 *  Created on: Feb 15, 2025
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace design_theory {




design_theory_global::design_theory_global()
{
	Record_birth();

}

design_theory_global::~design_theory_global()
{
	Record_death();

}


static const char *Baker_elliptic_semiplane_1978 =
		" IIIIIII       "
		"I AB C   AC B  "
		"I  AB C   AC B "
		"I   AB C   AC B"
		"IC   AB B   AC "
		"I C   AB B   AC"
		"IB C   AC B   A"
		"IAB C   AC B   "
		"  AC B  A  C CC"
		"   AC B CA  C C"
		"    AC BCCA  C "
		" B   AC  CCA  C"
		"  B   ACC CCA  "
		" C B   A C CCA "
		" AC B     C CCA";


void design_theory_global::make_Baker_elliptic_semiplane_1978_incma(
		int *&Inc, int &v, int &b,
		int verbose_level)
//AUTHOR = {Baker, Ronald D.},
// TITLE = {An elliptic semiplane},
//JOURNAL = {J. Combin. Theory Ser. A},
//FJOURNAL = {Journal of Combinatorial Theory. Series A},
//VOLUME = {25},
//  YEAR = {1978},
//NUMBER = {2},
// PAGES = {193--195},
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_theory_global::make_Baker_elliptic_semiplane_1978_incma" << endl;
	}

	other::data_structures::string_tools String;

	string input;

	input = Baker_elliptic_semiplane_1978;

	if (f_v) {
		cout << "design_theory_global::make_Baker_elliptic_semiplane_1978_incma input = " << input << endl;
		int i;
		for (i = 0; i < input.length(); i++) {
			cout << setw(3) << i << " : " << (int) input[i] << endl;
		}
	}



	int input_size, block_size;

	input_size = 15;
	block_size = 3;
	v = input_size * block_size;
	b = v;

	Inc = NEW_int(v * b);
	Int_vec_zero(Inc, v * b);

	int I, J;

	int block_zero[] = {0,0,0,0,0,0,0,0,0};
	int block_I[] = {1,0,0,0,1,0,0,0,1};
	int block_A[] = {0,1,0,1,0,0,0,0,1};
	int block_B[] = {1,0,0,0,0,1,0,1,0};
	int block_C[] = {0,0,1,0,1,0,1,0,0};

	int *block;

	for (I = 0; I < input_size; I++) {
		for (J = 0; J < input_size; J++) {

			char c;

			c = input[I * input_size + J];
			if (c == ' ') {
				block = block_zero;
			}
			else if (c == 'A') {
				block = block_A;
			}
			else if (c == 'B') {
				block = block_B;
			}
			else if (c == 'C') {
				block = block_C;
			}
			else if (c == 'I') {
				block = block_I;
			}
			else {
				cout << "character is unrecognized: c=" << (int) c << endl;
				cout << "I=" << I << endl;
				cout << "J=" << J << endl;
				exit(1);
			}

			int i, j;

			for (i = 0; i < block_size; i++) {
				for (j = 0; j < block_size; j++) {
					Inc[(I * block_size + i) * b + J * block_size + j]
						= block[i * block_size + j];
				}
			}
		}
	}

	if (f_v) {
		cout << "design_theory_global::make_Baker_elliptic_semiplane_1978_incma done" << endl;
	}

}


void design_theory_global::make_Mathon_elliptic_semiplane_1987_incma(
		int *&Inc, int &V, int &B,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_theory_global::make_Mathon_elliptic_semiplane_1987_incma" << endl;
	}

	int data[] = {
			0,0,0, 1,0,1, 2,0,1, 0,2,1, 3,2,2, 6,2,2, 0,3,0, 4,3,0, 8,3,1, 2,4,1, 4,4,1, 6,4,1,
			0,0,1, 1,0,0, 2,0,1, 2,2,0, 5,2,2, 8,2,0, 1,3,2, 5,3,1, 6,3,1, 0,4,1, 5,4,2, 7,4,0,
			0,0,1, 1,0,1, 2,0,0, 1,2,0, 4,2,0, 7,2,2, 2,3,2, 3,3,0, 7,3,2, 1,4,0, 3,4,2, 8,4,1,
			3,0,0, 4,0,1, 5,0,1, 0,2,2, 3,2,2, 6,2,1, 1,3,0, 5,3,0, 6,3,1, 1,4,2, 3,4,2, 8,4,2,
			3,0,1, 4,0,0, 5,0,1, 2,2,2, 5,2,0, 8,2,0, 2,3,1, 3,3,0, 7,3,0, 2,4,2, 4,4,0, 6,4,1,
			3,0,1, 4,0,1, 5,0,0, 1,2,0, 4,2,2, 7,2,0, 0,3,0, 4,3,1, 8,3,0, 0,4,1, 5,4,0, 7,4,2,
			6,0,0, 7,0,1, 8,0,1, 0,2,1, 3,2,0, 6,2,1, 2,3,0, 3,3,0, 7,3,1, 0,4,0, 5,4,0, 7,4,0,
			6,0,1, 7,0,0, 8,0,1, 2,2,2, 5,2,2, 8,2,1, 0,3,0, 4,3,2, 8,3,2, 1,4,0, 3,4,1, 8,4,2,
			6,0,1, 7,0,1, 8,0,0, 1,2,1, 4,2,2, 7,2,2, 1,3,1, 5,3,2, 6,3,1, 2,4,2, 4,4,1, 6,4,0 };

	V = 135;
	B = 135;
	Inc = NEW_int(V * B);
	Int_vec_zero(Inc, V * B);

	int i, j, h, u, v, x, y, z, pt;

	h = 0;
	for (i = 0; i < 5; i++) {
		for (j = 0; j < 3; j++) {
			for (u = 0; u < 9; u++, h++) {
				for (v = 0; v < 12; v++) {
					x = data[u * 36 + v * 3 + 0];
					y = (data[u * 36 + v * 3 + 1] + i) % 5;
					z = (data[u * 36 + v * 3 + 2] + j) % 3;
					pt = y * 27 + z * 9 + x;
					if (f_v) {
						cout << "h = " << h << " pt = " << pt << endl;
					}
					Inc[pt * B + h] = 1;
				}
			}
		}
	}
	if (h != 135) {
		cout << "design_theory_global::make_Mathon_elliptic_semiplane_1987_incma h != 135" << endl;
		exit(1);
	}

	if (false) {
		cout << "design_theory_global::make_Mathon_elliptic_semiplane_1987_incma incma:" << endl;
		Int_matrix_print(Inc, 135, 135);
	}

	if (f_v) {
		cout << "design_theory_global::make_Mathon_elliptic_semiplane_1987_incma done" << endl;
	}
}



void design_theory_global::make_design_from_incidence_matrix(
	int *&Inc, int &v, int &b, int &k,
	std::string &label,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_theory_global::make_design_from_incidence_matrix" << endl;
	}

	Get_matrix(label, Inc, v, b);

	int i, j, cnt;

	k = 0;
	for (j = 0; j < b; j++) {
		cnt = 0;
		for (i = 0; i < v; i++) {
			if (Inc[i * b + j]) {
				cnt++;
			}
		}
		if (f_v) {
			cout << "design_theory_global::make_design_from_incidence_matrix column " << j << " has k=" << k << endl;
		}

		if (j == 0) {
			k = cnt;
		}
		else {
			if (k != cnt) {
				cout << "design_theory_global::make_design_from_incidence_matrix the column sum is not constant" << endl;
				exit(1);
			}
		}
	}

	if (f_v) {
		cout << "design_theory_global::make_design_from_incidence_matrix done" << endl;
	}
}

void design_theory_global::compute_incidence_matrix(
		int v, int b, int k, long int *Blocks_coded,
		int *&M, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_theory_global::compute_incidence_matrix" << endl;
	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;

	int i, j, h;
	int *B;

	M = NEW_int(v * b);
	B = NEW_int(v);
	Int_vec_zero(M, v * b);
	for (j = 0; j < b; j++) {
		Combi.unrank_k_subset(Blocks_coded[j], B, v, k);
		for (h = 0; h < k; h++) {
			i = B[h];
			M[i * b + j] = 1;
		}
	}
	FREE_int(B);

	if (f_v) {
		cout << "design_theory_global::compute_incidence_matrix done" << endl;
	}
}

void design_theory_global::compute_incidence_matrix_from_blocks(
		int v, int b, int k, int *Blocks,
		int *&M, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_theory_global::compute_incidence_matrix_from_blocks" << endl;
	}
	int i, j, h;

	M = NEW_int(v * b);
	Int_vec_zero(M, v * b);
	for (j = 0; j < b; j++) {
		for (h = 0; h < k; h++) {
			i = Blocks[j * k + h];
			M[i * b + j] = 1;
		}
	}

	if (f_v) {
		cout << "design_theory_global::compute_incidence_matrix_from_blocks done" << endl;
	}
}

void design_theory_global::compute_incidence_matrix_from_blocks_lint(
		int v, int b, int k, long int *Blocks,
		int *&M, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_theory_global::compute_incidence_matrix_from_blocks_lint" << endl;
	}
	int i, j, h;

	M = NEW_int(v * b);
	Int_vec_zero(M, v * b);
	for (j = 0; j < b; j++) {
		for (h = 0; h < k; h++) {
			i = Blocks[j * k + h];
			M[i * b + j] = 1;
		}
	}

	if (f_v) {
		cout << "design_theory_global::compute_incidence_matrix_from_blocks_lint done" << endl;
	}
}



void design_theory_global::compute_incidence_matrix_from_sets(
		int v, int b, long int *Sets_coded,
		int *&M,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_theory_global::compute_incidence_matrix" << endl;
	}
	geometry::other_geometry::geometry_global Gg;

	int i, j;
	int *B;
	int *word;

	word = NEW_int(v);
	M = NEW_int(v * b);
	B = NEW_int(v);
	Int_vec_zero(M, v * b);
	for (j = 0; j < b; j++) {
		Gg.AG_element_unrank(2, word, 1, v, Sets_coded[j]);
		if (f_v) {
			cout << "design_theory_global::compute_incidence_matrix "
					"j=" << j << " coded set = " << Sets_coded[j];
			Int_vec_print(cout, word, v);
			cout << endl;
		}
		for (i = 0; i < v; i++) {

			if (word[i]) {

#if 0
				int ii;

				// we flip it:
				ii = v - 1 - i;
#endif

				M[i * b + j] = 1;
			}
		}
	}
	FREE_int(B);
	FREE_int(word);

	if (f_v) {
		cout << "design_theory_global::compute_incidence_matrix done" << endl;
	}
}


void design_theory_global::compute_blocks_from_coding(
		int v, int b, int k, long int *Blocks_coded,
		int *&Blocks, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_theory_global::compute_blocks_from_coding" << endl;
	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;

	int j;

	Blocks = NEW_int(b * k);
	Int_vec_zero(Blocks, b * k);
	for (j = 0; j < b; j++) {
		Combi.unrank_k_subset(Blocks_coded[j], Blocks + j * k, v, k);
		if (f_v) {
			cout << "block " << j << " : ";
			Int_vec_print(cout, Blocks + j * k, k);
			cout << endl;
		}

	}

	if (f_v) {
		cout << "design_theory_global::compute_blocks_from_coding done" << endl;
	}
}

void design_theory_global::compute_blocks_from_incma(
		int v, int b, int k, int *incma,
		int *&Blocks, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_theory_global::compute_blocks_from_incma" << endl;
	}
	int i, j, h;

	Blocks = NEW_int(b * k);
	Int_vec_zero(Blocks, b * k);
	for (j = 0; j < b; j++) {
		h = 0;
		for (i = 0; i < v; i++) {
			if (incma[i * b + j]) {
				Blocks[j * k + h] = i;
				h++;
			}
		}
		if (f_v) {
			cout << "design_theory_global::compute_blocks_from_incma "
					"column " << j << " has k = " << h << endl;
		}

		if (h != k) {
			cout << "design_theory_global::compute_blocks_from_incma "
					"block size is not equal to k" << endl;
			cout << "h=" << h << endl;
			cout << "k=" << k << endl;
			cout << "j=" << j << endl;
			cout << "b=" << b << endl;
			cout << "v=" << v << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "design_theory_global::compute_blocks_from_incma done" << endl;
	}
}






void design_theory_global::create_incidence_matrix_of_graph(
		int *Adj, int n,
		int *&M, int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, u;

	if (f_v) {
		cout << "design_theory_global::create_incidence_matrix_of_graph" << endl;
	}
	nb_rows = n;
	nb_cols = 0;
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (Adj[i * n + j]) {
				nb_cols++;
			}
		}
	}
	M = NEW_int(n * nb_cols);
	Int_vec_zero(M, n * nb_cols);
	u = 0;
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (Adj[i * n + j]) {
				M[i * nb_cols + u] = 1;
				M[j * nb_cols + u] = 1;
				u++;
			}
		}
	}
	if (f_v) {
		cout << "design_theory_global::create_incidence_matrix_of_graph done" << endl;
	}
}

void design_theory_global::create_wreath_product_design(
		int n, int k,
		long int *&Blocks, long int &nb_blocks,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_theory_global::create_wreath_product_design" << endl;
	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;

	long int n2, nk2;

	int v;

	v = 2 * n;
	n2 = Combi.binomial_lint(n, 2);
	nk2 = Combi.binomial_lint(n, k - 2);

	nb_blocks = n2 * nk2 * 2;

	Blocks = NEW_lint(nb_blocks);

	long int s, i, j, rk, cnt, u;
	int *B1;
	int *B2;
	int *B3;

	B1 = NEW_int(2);
	B2 = NEW_int(k - 2);
	B3 = NEW_int(k);

	cnt = 0;

	for (s = 0; s < 2; s++) {
		for (i = 0; i < n2; i++) {
			Combi.unrank_k_subset(i, B1, n, 2);
			for (j = 0; j < nk2; j++) {
				Combi.unrank_k_subset(j, B2, n, k - 2);
				if (s == 0) {
					Int_vec_copy(B1, B3, 2);
					Int_vec_copy(B2, B3 + 2, k - 2);
					for (u = 0; u < k - 2; u++) {
						B3[2 + u] += n;
					}
				}
				else {
					Int_vec_copy(B2, B3, k - 2);
					Int_vec_copy(B1, B3 + k - 2, 2);
					for (u = 0; u < 2; u++) {
						B3[k - 2 + u] += n;
					}
				}
				rk = Combi.rank_k_subset(B3, v, k);
				if (f_v) {
					cout << "block " << cnt << " : ";
					Int_vec_print(cout, B3, k);
					cout << " rk=" << rk;
					cout << endl;
				}

				Blocks[cnt++] = rk;
			}
		}
	}

	FREE_int(B1);
	FREE_int(B2);
	FREE_int(B3);

	if (f_v) {
		cout << "design_theory_global::create_wreath_product_design done" << endl;
	}
}

void design_theory_global::create_linear_space_from_latin_square(
		int *Mtx, int s,
		int &v, int &k,
		long int *&Blocks, long int &nb_blocks,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_theory_global::create_linear_space_from_latin_square" << endl;
	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;

	int i, j, a, rk, cnt;
	int block[3];

	v = 3 * s;
	k = 3;

	nb_blocks = s * s;

	Blocks = NEW_lint(nb_blocks);

	cnt = 0;
	for (i = 0; i < s; i++) {
		for (j = 0; j < s; j++) {
			a = Mtx[i * s + j];
			block[0] = i;
			block[1] = s + j;
			block[2] = 2 * s + a;
			rk = Combi.rank_k_subset(block, v, k);
			block[0] = i;
			block[1] = s + j;
			block[2] = 2 * s + a;
			if (f_v) {
				cout << "block " << cnt << " : ";
				Int_vec_print(cout, block, k);
				cout << " rk=" << rk;
				cout << endl;
			}

			Blocks[cnt++] = rk;
		}
	}
	if (cnt != nb_blocks) {
		cout << "design_theory_global::create_linear_space_from_latin_square cnt != nb_blocks" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "design_theory_global::create_linear_space_from_latin_square done" << endl;
	}
}

void design_theory_global::report_large_set(
		std::ostream &ost, long int *coding, int nb_designs,
		int design_v, int design_k, int design_sz, int verbose_level)
{

	combinatorics::other_combinatorics::combinatorics_domain Combi;

	int i, j, h;
	long int a;
	int *the_block;

	the_block = NEW_int(design_k);

	for (i = 0; i < nb_designs; i++) {
		for (j = 0; j < design_sz; j++) {
			a = coding[j];


			Combi.unrank_k_subset(a, the_block, design_v, design_k);
			for (h = 0; h < design_k; h++) {
				ost << the_block[h];
				if (h < design_k - 1) {
					ost << ", ";
				}
			}
			ost << "\\\\" << endl;

			}
		//Lint_vec_print(ost, set + i * design_sz, design_sz);
		ost << "\\\\" << endl;
	}
	FREE_int(the_block);

}

void design_theory_global::report_large_set_compact(
		std::ostream &ost, long int *coding, int nb_designs,
		int design_v, int design_k, int design_sz, int verbose_level)
{

	combinatorics::other_combinatorics::combinatorics_domain Combi;

	int i, j, h;
	long int a;
	char c;
	int *the_block;

	the_block = NEW_int(design_k);

	for (i = 0; i < nb_designs; i++) {
		for (j = 0; j < design_sz; j++) {
			a = coding[i * design_sz + j];


			Combi.unrank_k_subset(a, the_block, design_v, design_k);
			for (h = 0; h < design_k; h++) {
				c = '0' + the_block[h];
				ost << c;
			}
			ost << "," << endl;

			}
		ost << "\\\\" << endl;
	}
	FREE_int(the_block);

}







}}}}

