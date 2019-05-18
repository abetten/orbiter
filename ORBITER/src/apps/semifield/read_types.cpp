/*
 * read_types.cpp
 *
 *  Created on: May 17, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


#define MY_BUFSIZE ONE_MILLION

int main(int argc, const char **argv)
{
	int i, j;
	int verbose_level = 0;
	int f_file = FALSE;
	const char *fname = NULL;

	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname = argv[++i];
			cout << "-file " << fname << endl;
		}
	}
	if (!f_file) {
		cout << "please use option -file <fname" << endl;
		exit(1);
	}

	file_io Fio;
	int *M;
	int m, n;
	int max_value;

	Fio.int_matrix_read_csv(fname, M,
			m, n, verbose_level);

	cout << "read a file with " << m << " rows and " << n << " columns" << endl;

	max_value = int_vec_maximum(M, m * n);

	cout << "max_value = " << max_value << endl;


	int *T;
	int len, a;

	len = max_value + 1;

	T = NEW_int(m * len);
	int_vec_zero(T, m * len);
	cout << "computing T:" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = M[i * n + j];
			T[i * len + a]++;
		}
	}
	cout << "T=" << endl;
	int_matrix_print(T, m, len);


	char fname2[1000];

	strcpy(fname2, fname);
	chop_off_extension(fname2);
	strcat(fname2, "_types.csv");
	Fio.int_matrix_write_csv(fname2, T, m, len);

	cout << "Writtem file " << fname2 << " of size " << Fio.file_size(fname2) << endl;


	print_integer_matrix_tex_block_by_block(cout,
			T,
			m, len, 40 /* block_width */);


	int *V;

	V = NEW_int(m);
	for (i = 0; i < m; i++) {
		a = T[i * len + len - 1];
		V[i] = a;
	}
	classify C;
	C.init(V, m, FALSE, 0);


	cout << "Classification by highest type:\\\\" << endl;
	cout << "$$" << endl;
	C.print_array_tex(cout, TRUE /*f_backwards */);
	cout << "$$" << endl;


	classify_vector_data CV;

	CV.init(T, m /* data_length */, len /* data_set_sz */,
			FALSE /* f_second */, verbose_level);

	CV.print();

	classify CM;

	CM.init(CV.Data_multiplicity, CV.data_unique_length, FALSE, 0);
	cout << "Data_multiplicity: ";
	CM.print_naked(TRUE /*f_backwards*/);
	cout << endl;
}
