/*
 * code_diagram.cpp
 *
 *  Created on: Dec 16, 2022
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace coding_theory {




code_diagram::code_diagram()
{
	Words = NULL;
	nb_words = 0;

	//std::string label;

	n = 0;
	N = 0;

	nb_rows = nb_cols = 0;
	v = NULL;
	Index_of_codeword = NULL;
	Place_values = NULL;
	Characteristic_function = NULL;
	Distance = NULL;
	Distance_H = NULL;

}

code_diagram::~code_diagram()
{

}

void code_diagram::init(std::string &label,
		long int *Words, int nb_words, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "code_diagram::init" << endl;
	}
	code_diagram::Words = Words;
	code_diagram::nb_words = nb_words;
	code_diagram::n = n;
	code_diagram::label.assign(label);

	N = 1 << n;

	dimensions(n, nb_rows, nb_cols);

	if (f_v) {
		cout << "code_diagram::init" << endl;
		cout << "nb_rows=" << nb_rows << endl;
		cout << "nb_cols=" << nb_cols << endl;
	}

	v = NEW_int(n);

	Index_of_codeword = NEW_int(nb_rows * nb_cols);
	Place_values = NEW_int(nb_rows * nb_cols);
	Characteristic_function = NEW_int(nb_rows * nb_cols);
	Distance = NEW_int(nb_rows * nb_cols);
	Distance_H = NEW_int(nb_rows * nb_cols);


	Int_vec_zero(Place_values, nb_rows * nb_cols);
	Int_vec_zero(Characteristic_function, nb_rows * nb_cols);

	int h;
	//int i, j;

	//Orbiter->Int_vec.zero(M3, nb_rows * nb_cols);
	for (h = 0; h < nb_rows * nb_cols; h++) {
		Distance_H[h] = n + 1;
	}


	if (f_v) {
		cout << "code_diagram::init before place_codewords" << endl;
	}
	place_codewords(verbose_level);
	if (f_v) {
		cout << "code_diagram::init after place_codewords" << endl;
	}

	if (f_v) {
		cout << "code_diagram::init before compute_distances" << endl;
	}
	compute_distances(verbose_level);
	if (f_v) {
		cout << "code_diagram::init after compute_distances" << endl;
	}



	if (f_v) {
		cout << "code_diagram::init done" << endl;
	}
}

void code_diagram::place_codewords(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "code_diagram::place_codewords" << endl;
	}

	if (f_v) {
		cout << "code_diagram::place_codewords placing codewords" << endl;
	}

	int h, i, j;

	for (h = 0; h < N; h++) {
		place_binary(h, i, j);
		Place_values[i * nb_cols + j] = h;
		//M2[i * nb_cols + j] = 1;
	}
	if (f_v) {
		cout << "code_diagram::place_codewords placing position values done" << endl;
	}


	if (f_v) {
		cout << "code_diagram::place_codewords placing codewords" << endl;
	}
	Int_vec_zero(Index_of_codeword, nb_rows * nb_cols);
	for (h = 0; h < nb_words; h++) {

		convert_to_binary(n, Words[h], v);

		if (nb_words < 10 && f_v) {
			cout << "code_diagram::place_codewords "
					"codeword " << h + 1 << " = " << setw(5) << Words[h];
			cout << " : ";
			print_binary(n, v);
			cout << endl;
		}

		place_binary(Words[h], i, j);

		Index_of_codeword[i * nb_cols + j] = h + 1;
		Characteristic_function[i * nb_cols + j] = 1;
		Distance_H[i * nb_cols + j] = 0; // distance is zero

#if 0
		if (f_enhance) {
			embellish(M, nb_rows, nb_cols, i, j, h + 1 /* value */, radius);
		}
		if (f_enhance) {
			embellish(M2, nb_rows, nb_cols, i, j, 1 /* value */, radius);
		}
#endif

	}
	if (f_v) {
		cout << "code_diagram::place_codewords placing codewords done" << endl;
		//Int_matrix_print(Index_of_codeword, nb_rows, nb_cols);
	}

	if (f_v) {
		cout << "code_diagram::place_codewords done" << endl;
	}
}

void code_diagram::place_metric_balls(int radius_of_metric_ball, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "code_diagram::place_metric_balls" << endl;
	}

	combinatorics::combinatorics_domain Combi;

	int h, i, j;
	int u, t, s, a;
	int nCk;

	int *set_of_errors;

	set_of_errors = NEW_int(radius_of_metric_ball);

	for (h = 0; h < nb_words; h++) {
		convert_to_binary(n, Words[h], v);

		if (f_v) {
			cout << "code_diagram::place_metric_balls "
					"codeword " << h + 1 << " = " << setw(5) << Words[h];
			cout << " : ";
			print_binary(n, v);
			cout << endl;
		}

		place_binary(Words[h], i, j);

		for (u = 1; u <= radius_of_metric_ball; u++) {


			nCk = Combi.int_n_choose_k(n, u);

			for (t = 0; t < nCk; t++) {

				Combi.unrank_k_subset(t, set_of_errors, n, u);

				convert_to_binary(n, Words[h], v);

				for (s = 0; s < u; s++) {
					a = set_of_errors[s];
					v[a] = (v[a] + 1) % 2;
				}

				place_binary(v, n, i, j);

				if (Characteristic_function[i * nb_cols + j]) {
					cout << "code_diagram::place_metric_balls "
							"the metric balls overlap!" << endl;
					cout << "h=" << h << endl;
					cout << "t=" << t << endl;
					cout << "i=" << i << endl;
					cout << "j=" << j << endl;
					exit(1);
				}
				Characteristic_function[i * nb_cols + j] = h + 1;
			}
		}
	}

	FREE_int(set_of_errors);
}

void code_diagram::compute_distances(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "code_diagram::compute_distances" << endl;
	}
	int *Dist_from_code_enumerator;
	int d;
	int s, original_value, a;
	int h, i, j;

	Dist_from_code_enumerator = NEW_int(n + 1);
	Int_vec_zero(Dist_from_code_enumerator, n + 1);

	for (d = 0; d < n; d++) {
		if (f_v) {
			cout << "code_diagram::compute_distances "
					"computing words of distance " << d + 1 << " from the code" << endl;
		}
		for (h = 0; h < nb_rows * nb_cols; h++) {
			if (Distance_H[h] == d) {
				Dist_from_code_enumerator[d]++;
				i = h / nb_cols;
				j = h % nb_cols;
				convert_to_binary(n, h, v);
				for (s = 0; s < n; s++) {
					original_value = v[s];
					v[s] = (v[s] + 1) % 2;
					place_binary(v, n, i, j);
					a = i * nb_cols + j;
					if (Distance_H[a] > d + 1) {
						Distance_H[a] = d + 1;
					}
					v[s] = original_value;
				}
			}
		}
		if (f_v) {
			cout << "We found " << Dist_from_code_enumerator[d]
				<< " words at distance " << d << " from the code" << endl;
		}

		if (Dist_from_code_enumerator[d] == 0) {
			break;
		}
	}
	for (h = 0; h < N; h++) {

		place_binary(h, i, j);
		d = Distance_H[i * nb_cols + j];
		Distance[h] = d;
	}

	if (f_v) {
		cout << "d : # words at distance d from code" << endl;
		for (d = 0; d < n; d++) {
			cout << d << " : " << Dist_from_code_enumerator[d] << endl;
		}
		cout << endl;
	}
	if (f_v) {
		cout << "code_diagram::compute_distances" << endl;
	}
}

void code_diagram::dimensions(int n, int &nb_rows, int &nb_cols)
{
	int i, j;

	place_binary((1 << n) - 1, i, j);
	nb_rows = i + 1;
	nb_cols = j + 1;
}

void code_diagram::place_binary(long int h, int &i, int &j)
{
	int o[2];
	int c;

	o[0] = 1;
	o[1] = 0;
	i = 0;
	j = 0;
	for (c = 0; h; c++) {
		if (h % 2) {
			i += o[0];
			j += o[1];
		}
		h >>= 1;
		if (c % 2) {
			o[0] = o[1] << 1;
			o[1] = 0;
		}
		else {
			o[1] = o[0];
			o[0] = 0;
		}
	}
}

void code_diagram::place_binary(int *v, int n, int &i, int &j)
{
	int o[2];
	int c;

	o[0] = 1;
	o[1] = 0;
	i = 0;
	j = 0;
	for (c = 0; c < n; c++) {
		if (v[c]) {
			i += o[0];
			j += o[1];
		}
		if (c % 2) {
			o[0] = o[1] << 1;
			o[1] = 0;
		}
		else {
			o[1] = o[0];
			o[0] = 0;
		}
	}
}

void code_diagram::convert_to_binary(int n, long int h, int *v)
{
	int c;

	for (c = 0; c < n; c++) {
		if (h % 2) {
			v[c] = 1;
		}
		else {
			v[c] = 0;
		}
		h >>= 1;
	}
}

void code_diagram::print_binary(int n, int *v)
{
	int c;

	for (c = n - 1; c >= 0; c--) {
		cout << v[c];
	}
}


void code_diagram::save_distance(int verbose_level)
{
	char str[1000];

	string fname;

	fname.assign(label);

	snprintf(str, sizeof(str), "_distance_%d_%d.csv", n, nb_words);
	fname.append(str);
	orbiter_kernel_system::file_io Fio;

	Fio.int_matrix_write_csv(fname, Distance, nb_rows, nb_cols);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
}


void code_diagram::save_distance_H(int verbose_level)
{
	char str[1000];

	string fname;

	fname.assign(label);

	snprintf(str, sizeof(str), "_distance_H_%d_%d.csv", n, nb_words);
	fname.append(str);
	orbiter_kernel_system::file_io Fio;

	Fio.int_matrix_write_csv(fname, Distance_H, nb_rows, nb_cols);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
}

void code_diagram::save_diagram(int verbose_level)
{
	char str[1000];

	string fname;

	fname.assign(label);

	snprintf(str, sizeof(str), "_idx_%d_%d.csv", n, nb_words);
	fname.append(str);
	orbiter_kernel_system::file_io Fio;

	Fio.int_matrix_write_csv(fname, Index_of_codeword, nb_rows, nb_cols);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
}

void code_diagram::save_char_func(int verbose_level)
{
	char str[1000];

	string fname;

	fname.assign(label);

	snprintf(str, sizeof(str), "_char_func_%d_%d.csv", n, nb_words);
	fname.append(str);
	orbiter_kernel_system::file_io Fio;

	Fio.int_matrix_write_csv(fname, Characteristic_function, nb_rows, nb_cols);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
}


void code_diagram::report(int verbose_level)
{
	orbiter_kernel_system::file_io Fio;
	char str[1000];

	string fname;

	fname.assign(label);

	snprintf(str, sizeof(str), "_%d_%d.tex", n, nb_words);
	fname.append(str);

	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;

		L.head_easy(fp);
		fp << "Place values:" << endl;
		fp << "$$" << endl;
		L.print_integer_matrix_tex(fp, Place_values, nb_rows, nb_cols);
		fp << "$$" << endl;



		fp << "Index of codeword:" << endl;
		fp << "$$" << endl;
		L.print_integer_matrix_tex(fp, Index_of_codeword, nb_rows, nb_cols);
		fp << "$$" << endl;


		L.foot(fp);
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
}


}}}


