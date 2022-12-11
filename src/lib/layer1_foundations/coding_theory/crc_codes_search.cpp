/*
 * crc_codes_search.cpp
 *
 *  Created on: Dec 9, 2022
 *      Author: betten
 */







#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace coding_theory {



/*
 * twocoef.cpp
 *
 *  Created on: Oct 22, 2020
 *      Author: alissabrown
 *
 *	Received a lot of help from Anton and the recursive function in the possibleC function is modeled after code found at
 *	https://www.geeksforgeeks.org/print-all-combinations-of-given-length/
 *
 *
 */

void crc_codes::find_CRC_polynomials(
		field_theory::finite_field *F,
		int t, int da, int dc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "crc_codes::find_CRC_polynomials t=" << t
				<< " info=" << da << " check=" << dc << endl;
	}

	//int dc = 4; //dc is the number of parity bits & degree of g(x)
	//int da = 4; //da is the degree of the information polynomial
	int A[da + dc];
		// we have da information bits, which we can think of
		// as the coefficients of a polynomial.
		// After multiplying by x^dc,
		// A(x) has degree at most ad + dc - 1.
	long int nb_sol = 0;



	int C[dc + 1]; //Array C (what we divide by)
		// C(x) has the leading coefficient of one included,
		// hence we need one more array element

	int i = 0;

	for (i = 0; i <= dc; i++) {
		C[i] = 0;
	}


	std::vector<std::vector<int>> Solutions;

	if (F->q == 2) {
		search_for_CRC_polynomials_binary(t, da, A, dc, C, 0,
				nb_sol, Solutions, verbose_level - 1);
	}
	else {
		search_for_CRC_polynomials(t, da, A, dc, C, 0, F,
				nb_sol, Solutions, verbose_level - 1);
	}

	cout << "crc_codes::find_CRC_polynomials info=" << da
			<< " check=" << dc << " nb_sol=" << nb_sol << endl;

	for (i = 0; i < Solutions.size(); i++) {
		cout << i << " : ";
		for (int j = dc; j >= 0; j--) {
			cout << Solutions[i][j];
		}
		cout << endl;
	}
	cout << "crc_codes::find_CRC_polynomials info=" << da
			<< " check=" << dc << " nb_sol=" << nb_sol << endl;

}

void crc_codes::search_for_CRC_polynomials(int t,
		int da, int *A, int dc, int *C,
		int i, field_theory::finite_field *F,
		long int &nb_sol,
		std::vector<std::vector<int> > &Solutions,
		int verbose_level)
{

	if (i == dc + 1) {

		int ret;

		if (t >= 2) {
			ret = test_all_two_bit_patterns(da, A, dc, C, F, verbose_level);
			if (ret && t >= 3) {
				ret = test_all_three_bit_patterns(da, A, dc, C, F, verbose_level);
			}
		}
		else {
			cout << "illegal value for t, t=" << t << endl;
			exit(1);
		}
		if (ret) {
			cout << "solution " << nb_sol << " is ";
			for (int j = dc; j >= 0; j--) {
				cout << C[j];
			}
			cout << endl;

			vector<int> sol;

			for (int j = 0; j <= dc; j++) {
				sol.push_back(C[j]);
			}
			Solutions.push_back(sol);


			nb_sol++;
		}

		return;
	}

	if (i == dc) {

		// C(x) has a leading coefficient of one:
		C[i] = 1;
		search_for_CRC_polynomials(t, da, A, dc, C,
				i + 1, F, nb_sol, Solutions, verbose_level);

	}
	else {
		int c;

		for (c = 0; c < F->q; c++) {

			C[i] = c;

			search_for_CRC_polynomials(t, da, A, dc, C,
					i + 1, F, nb_sol, Solutions, verbose_level);
		}
	}
}

void crc_codes::search_for_CRC_polynomials_binary(int t,
		int da, int *A, int dc, int *C, int i,
		long int &nb_sol,
		std::vector<std::vector<int> > &Solutions,
		int verbose_level)
{

	if (i == dc + 1) {

		int ret;

		if (t >= 2) {
			ret = test_all_two_bit_patterns_binary(da, A, dc, C, verbose_level);
			if (ret && t >= 3) {
				ret = test_all_three_bit_patterns_binary(da, A, dc, C, verbose_level);
			}
		}
		else {
			cout << "illegal value for t, t=" << t << endl;
			exit(1);
		}
		if (ret) {
			cout << "solution " << nb_sol << " is ";
			for (int j = dc; j >= 0; j--) {
				cout << C[j];
			}
			cout << endl;

			vector<int> sol;

			for (int j = 0; j <= dc; j++) {
				sol.push_back(C[j]);
			}
			Solutions.push_back(sol);


			nb_sol++;
		}

		return;
	}

	if (i == dc) {

		C[i] = 1;
		search_for_CRC_polynomials_binary(t, da, A, dc, C,
				i + 1, nb_sol, Solutions, verbose_level);


	}
	else {
		int c;

		for (c = 0; c < 2; c++) {

			C[i] = c;

			search_for_CRC_polynomials_binary(t, da, A, dc, C,
					i + 1, nb_sol, Solutions, verbose_level);
		}
	}
}


int crc_codes::test_all_two_bit_patterns(int da, int *A,
		int dc, int *C,
		field_theory::finite_field *F,
		int verbose_level)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{

	//cout << "choosetwo" << endl;

	int f_v = (verbose_level >= 1);

	int i;
	int j;
	int k;
	int ai, aj;
	int ret;
	int B[da + dc];

	if (f_v) {
		cout << "testing polynomial: ";
		for (k = dc; k >= 0; k--) {
			cout << C[k];
		}
		cout << endl;
	}

	for (i = 0; i < da; i++) {
		A[i] = 0;
	}

	for (i = 0; i < da; i++) {

		for (ai = 1; ai < F->q; ai++) {

			A[i] = ai;

			for (j = i + 1; j < da; j++) {

				for (aj = 1; aj < F->q; aj++) {

					A[j] = aj;

					for (k = 0; k < dc; k++) {
						B[k] = 0;
					}
					for (k = 0; k < da; k++) {
						B[dc + k] = A[k];
					}

					if (f_v) {
						cout << "testing error pattern: ";
						for (k = dc + da - 1; k >= 0; k--) {
							cout << B[k];
						}
					}



					ret = remainder_is_nonzero (da, B, dc, C, F);

					if (f_v) {
						cout << " : ";
						for (k = dc - 1; k >= 0; k--) {
							cout << B[k];
						}
						cout << endl;
					}

					if (!ret) {
						return false;
					}

				}
				A[j] = 0;
			}

		}
		A[i] = 0;
	}
	return true;
}

int crc_codes::test_all_three_bit_patterns(int da, int *A,
		int dc, int *C,
		field_theory::finite_field *F,
		int verbose_level)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{

	//cout << "choosetwo" << endl;

	int f_v = (verbose_level >= 1);

	int i1, i2, i3;
	int k;
	int a1, a2, a3;
	int ret;
	int B[da + dc];

	if (f_v) {
		cout << "testing polynomial: ";
		for (k = dc; k >= 0; k--) {
			cout << C[k];
		}
		cout << endl;
	}

	for (int h = 0; h < da; h++) {
		A[h] = 0;
	}

	for (i1 = 0; i1 < da; i1++) {

		for (a1 = 1; a1 < F->q; a1++) {

			A[i1] = a1;

			for (i2 = i1 + 1; i2 < da; i2++) {

				for (a2 = 1; a2 < F->q; a2++) {

					A[i2] = a2;

					for (i3 = i2 + 1; i3 < da; i3++) {

						for (a3 = 1; a3 < F->q; a3++) {

							A[i3] = a3;

							for (int h = 0; h < dc; h++) {
								B[h] = 0;
							}
							for (int h = 0; h < da; h++) {
								B[dc + h] = A[h];
							}

							if (f_v) {
								cout << "testing error pattern: ";
								for (int h = dc + da - 1; h >= 0; h--) {
									cout << B[h];
								}
							}



							ret = remainder_is_nonzero (da, B, dc, C, F);

							if (f_v) {
								cout << " : ";
								for (int h = dc - 1; h >= 0; h--) {
									cout << B[h];
								}
								cout << endl;
							}

							if (!ret) {
								return false;
							}

						}
						A[i3] = 0;
					}
				}
				A[i2] = 0;
			}
		}
		A[i1] = 0;
	}
	return true;
}

int crc_codes::test_all_two_bit_patterns_binary(int da, int *A,
		int dc, int *C,
		int verbose_level)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{

	//cout << "choosetwo" << endl;

	int f_v = (verbose_level >= 1);

	int i;
	int j;
	int k;
	int ret;
	int B[da + dc];

	if (f_v) {
		cout << "testing polynomial: ";
		for (k = dc; k >= 0; k--) {
			cout << C[k];
		}
		cout << endl;
	}

	for (i = 0; i < da; i++) {
		A[i] = 0;
	}

	for (i = 0; i < da; i++) {


		A[i] = 1;

		for (j = i + 1; j < da; j++) {


			A[j] = 1;

			for (k = 0; k < dc; k++) {
				B[k] = 0;
			}
			for (k = 0; k < da; k++) {
				B[dc + k] = A[k];
			}

			if (f_v) {
				cout << "testing error pattern: ";
				for (k = dc + da - 1; k >= 0; k--) {
					cout << B[k];
				}
			}



			ret = remainder_is_nonzero_binary(da, B, dc, C);

			if (f_v) {
				cout << " : ";
				for (k = dc - 1; k >= 0; k--) {
					cout << B[k];
				}
				cout << endl;
			}

			if (!ret) {
				return false;
			}

			A[j] = 0;
		}

		A[i] = 0;
	}
	return true;
}

int crc_codes::test_all_three_bit_patterns_binary(int da, int *A,
		int dc, int *C,
		int verbose_level)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{

	//cout << "choosetwo" << endl;

	int f_v = (verbose_level >= 1);

	int i1, i2, i3;
	int k;
	int ret;
	int B[da + dc];

	if (f_v) {
		cout << "testing polynomial: ";
		for (k = dc; k >= 0; k--) {
			cout << C[k];
		}
		cout << endl;
	}

	for (int h = 0; h < da; h++) {
		A[h] = 0;
	}

	for (i1 = 0; i1 < da; i1++) {

		A[i1] = 1;

		for (i2 = i1 + 1; i2 < da; i2++) {


			A[i2] = 1;

			for (i3 = i2 + 1; i3 < da; i3++) {


				A[i3] = 1;

				for (int h = 0; h < dc; h++) {
					B[h] = 0;
				}
				for (int h = 0; h < da; h++) {
					B[dc + h] = A[h];
				}

				if (f_v) {
					cout << "testing error pattern: ";
					for (int h = dc + da - 1; h >= 0; h--) {
						cout << B[h];
					}
				}



				ret = remainder_is_nonzero_binary(da, B, dc, C);

				if (f_v) {
					cout << " : ";
					for (int h = dc - 1; h >= 0; h--) {
						cout << B[h];
					}
					cout << endl;
				}

				if (!ret) {
					return false;
				}

				A[i3] = 0;
			}
			A[i2] = 0;
		}

		A[i1] = 0;
	}
	return true;
}


int crc_codes::remainder_is_nonzero(int da, int *A,
		int db, int *B, field_theory::finite_field *F)
// returns true if the remainder of A after division by B is nonzero
{

	int i, j, k, a, mav;

	for (i = da + db - 1; i >= db; i--) {

		a = A[i];

		if (a) {

			mav = F->negate(F->inverse(a));

			for (j = db, k = i; j >= 0; j--, k--) {

				//A[k] = (A[k] + B[j]) % 2;
				A[k] = F->add(A[k], F->mult(mav, B[j]));
					//A[k] = subtraction(A[k], B[j], p);
					//A[k]=(A[k]+2-B[j])%2;
			}
		}
	}


	for (int k = db - 1; k >= 0; k--) {
		if (A[k]) {
			return true;
		}
	}
	return false;
}


int crc_codes::remainder_is_nonzero_binary(int da, int *A,
		int db, int *B)
// returns true if the remainder of A after division by B is nonzero
{

	int i, j, k, a;

	for (i = da + db - 1; i >= db; i--) {

		a = A[i];

		if (a) {

			//mav = F->negate(F->inverse(a));

			for (j = db, k = i; j >= 0; j--, k--) {

				A[k] = (A[k] + B[j]) % 2;
				//A[k] = F->add(A[k], F->mult(mav, B[j]));
					//A[k] = subtraction(A[k], B[j], p);
					//A[k]=(A[k]+2-B[j])%2;
			}
		}
	}


	for (int k = db - 1; k >= 0; k--) {
		if (A[k]) {
			return true;
		}
	}
	return false;
}


}}}

