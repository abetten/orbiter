/*
 * endrass.cpp
 *
 *  Created on: Jun 5, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


using namespace orbiter;


void do_it(int verbose_level);
int endrass_compare_func(void *a, void *b, void *data);

int main(int argc, const char **argv)
{
	int verbose_level = 0;


	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}
	do_it(verbose_level);
}


void do_it(int verbose_level)
{
	int data[] = {
			0,0,0,8, -33, 24,165,
			2,0,0,6, 168, 124,84,
			0,2,0,6, 168, 124,148,
			0,0,2,6, -752, 160,163,
			4,0,0,4, -288, -216,35,
			0,4,0,4, -288, -216,135,
			0,0,4,4, -608, -128,161,
			2,2,0,4, -576, -432,71,
			2,0,2,4, 1248, 304,82,
			0,2,2,4, 1248, 304,146,
			6,0,0,2, 192, 144,10,
			0,6,0,2, 192, 144,126,
			0,0,6,2, 256, -512,159,
			2,2,2,2, -1536, -1152,69,
			4,2,0,2, 576, 432,26,
			4,0,2,2, -768, -576,33,
			2,4,0,2, 576, 432,62,
			0,4,2,2, -768, -576,133,
			2,0,4,2, 384, 832,80,
			0,2,4,2, 384, 832,144,
			0,0,8,0, -256, 0,157,
			2,0,6,0, 512, 256,78,
			0,2,6,0, 512, 256,142,
			4,0,4,0, -512, -384,31,
			0,4,4,0, -512, -384,131,
			2,2,4,0, -1024, -768,67,
			6,0,2,0, 256, 192,8,
			0,6,2,0, 256, 192,124,
			4,2,2,0, 768, 576,24,
			2,4,2,0, 768, 576,60,
			0,8,0,0, -48, -32,121,
			2,6,0,0, -128, -128,57,
			4,4,0,0, -416, -192,21,
			6,2,0,0, -128, -128,5,
			8,0,0,0, -48, -32,1,
	};
	int sz = sizeof(data) / (7 * sizeof(int));

	int *data2;
	int **V;
	int i, j;
	sorting Sorting;
	int *perm;

	cout << "sz=" << sz << endl;
	data2 = NEW_int(sz * 5);
	V = NEW_pint(sz);
	perm = NEW_int(sz);
	for (i = 0; i < sz; i++) {
		V[i] = data2+ i * 5;
		perm[i] = i;
		data2[i * 5 + 0] = data[i * 7 + 0];
		data2[i * 5 + 1] = data[i * 7 + 1];
		data2[i * 5 + 2] = data[i * 7 + 2];
		data2[i * 5 + 3] = data[i * 7 + 3];
		data2[i * 5 + 4] = i;
	}

	Sorting.quicksort_array_with_perm(sz, (void **) V, perm,
			endrass_compare_func, NULL);

	for (i = sz - 1; i >= 0; i--) {
		j = perm[i];
		int_vec_print(cout, data + j * 7, 7);
		cout << endl;
	}

	double a, s2;

	s2 = sqrt(2.);

	for (i = sz - 1; i >= 0; i--) {
		j = perm[i];
		//int_vec_print(cout, data + j * 7, 7);
		//cout << endl;
		a = (double) data[j * 7 + 4] + (double) data[j * 7 + 5] * s2;
		cout << a << ",";
	}
	cout << endl;

	double Octic[165];
	int h;

	for (i = sz - 1; i >= 0; i--) {
		j = perm[i];
		//int_vec_print(cout, data + j * 7, 7);
		//cout << endl;
		a = (double) data[j * 7 + 4] + ((double) data[j * 7 + 5]) * s2;
		h = data[j * 7 + 6] - 1;
		Octic[h] = a;
	}
	for (i = 0; i < 165; i++) {
		if (ABS(Octic[i]) > 0.000000001) {
			cout << Octic[i] << ",";
		}
		else {
			cout << 0. << ",";

		}
	}
	cout << endl;

}

int endrass_compare_func(void *a, void *b, void *data)
{
	int *A = (int *) a;
	int *B = (int *) b;
	int c;

	c = int_vec_compare(A, B, 4);
	return c;
}

