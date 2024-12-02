/*
 * schlaefli_double_six.cpp
 *
 *  Created on: Nov 15, 2023
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace algebraic_geometry {


schlaefli_double_six::schlaefli_double_six()
{
	Record_birth();
	Schlaefli = NULL;

	Double_six = NULL;
	Double_six_label_tex = NULL;

	Half_double_six_characteristic_vector = NULL;

	Double_six_characteristic_vector = NULL;

	Half_double_sixes = NULL;
	Half_double_six_label_tex = NULL;
	Half_double_six_to_double_six = NULL;
	Half_double_six_to_double_six_row = NULL;

}


schlaefli_double_six::~schlaefli_double_six()
{
	Record_death();

	if (Double_six) {
		FREE_lint(Double_six);
	}
	if (Double_six_label_tex) {
		delete [] Double_six_label_tex;
	}

	if (Half_double_six_characteristic_vector) {
		FREE_int(Half_double_six_characteristic_vector);
	}

	if (Double_six_characteristic_vector) {
		FREE_int(Double_six_characteristic_vector);
	}

	if (Half_double_sixes) {
		FREE_lint(Half_double_sixes);
	}

	if (Half_double_six_label_tex) {
		delete [] Half_double_six_label_tex;
	}

	if (Half_double_six_to_double_six) {
		FREE_int(Half_double_six_to_double_six);
	}
	if (Half_double_six_to_double_six_row) {
		FREE_int(Half_double_six_to_double_six_row);
	}

}


void schlaefli_double_six::init(
		schlaefli *Schlaefli, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schlaefli_double_six::init" << endl;
	}

	schlaefli_double_six::Schlaefli = Schlaefli;


	if (f_v) {
		cout << "schlaefli::init "
				"before init_double_sixes" << endl;
	}
	init_double_sixes(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after init_double_sixes" << endl;
	}

	if (f_v) {
		cout << "schlaefli::init "
				"before create_half_double_sixes" << endl;
	}
	create_half_double_sixes(verbose_level);
	if (f_v) {
		cout << "schlaefli::init "
				"after create_half_double_sixes" << endl;
	}
	//print_half_double_sixes_in_GAP();


}


void schlaefli_double_six::init_double_sixes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, ij, u, v, l, m, n, h, a, b, c;
	int set[6];
	int size_complement;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "schlaefli_double_six::init_double_sixes" << endl;
	}
	Double_six = NEW_lint(36 * 12);
	h = 0;
	// first type: D : a_1,..., a_6; b_1, ..., b_6
	for (i = 0; i < 12; i++) {
		Double_six[h * 12 + i] = i;
	}
	h++;

	// second type:
	// D_{ij} :
	// a_1, b_1, c_23, c_24, c_25, c_26;
	// a_2, b_2, c_13, c_14, c_15, c_16
	for (ij = 0; ij < 15; ij++, h++) {
		//cout << "second type " << ij << " / " << 15 << endl;
		Combi.k2ij(ij, i, j, 6);
		set[0] = i;
		set[1] = j;
		Combi.set_complement(set, 2 /* subset_size */, set + 2,
			size_complement, 6 /* universal_set_size */);
		//cout << "set : ";
		//int_vec_print(cout, set, 6);
		//cout << endl;
		Double_six[h * 12 + 0] = Schlaefli->line_ai(i);
		Double_six[h * 12 + 1] = Schlaefli->line_bi(i);
		for (u = 0; u < 4; u++) {
			Double_six[h * 12 + 2 + u] = Schlaefli->line_cij(j, set[2 + u]);
		}
		Double_six[h * 12 + 6] = Schlaefli->line_ai(j);
		Double_six[h * 12 + 7] = Schlaefli->line_bi(j);
		for (u = 0; u < 4; u++) {
			Double_six[h * 12 + 8 + u] = Schlaefli->line_cij(i, set[2 + u]);
		}
	}

	// third type: D_{ijk} :
	// a_1, a_2, a_3, c_56, c_46, c_45;
	// c_23, c_13, c_12, b_4, b_5, b_6
	for (v = 0; v < 20; v++, h++) {
		//cout << "third type " << v << " / " << 20 << endl;
		Combi.unrank_k_subset(v, set, 6, 3);
		Combi.set_complement(set, 3 /* subset_size */, set + 3,
			size_complement, 6 /* universal_set_size */);
		i = set[0];
		j = set[1];
		k = set[2];
		l = set[3];
		m = set[4];
		n = set[5];
		Double_six[h * 12 + 0] = Schlaefli->line_ai(i);
		Double_six[h * 12 + 1] = Schlaefli->line_ai(j);
		Double_six[h * 12 + 2] = Schlaefli->line_ai(k);
		Double_six[h * 12 + 3] = Schlaefli->line_cij(m, n);
		Double_six[h * 12 + 4] = Schlaefli->line_cij(l, n);
		Double_six[h * 12 + 5] = Schlaefli->line_cij(l, m);
		Double_six[h * 12 + 6] = Schlaefli->line_cij(j, k);
		Double_six[h * 12 + 7] = Schlaefli->line_cij(i, k);
		Double_six[h * 12 + 8] = Schlaefli->line_cij(i, j);
		Double_six[h * 12 + 9] = Schlaefli->line_bi(l);
		Double_six[h * 12 + 10] = Schlaefli->line_bi(m);
		Double_six[h * 12 + 11] = Schlaefli->line_bi(n);
	}

	if (h != 36) {
		cout << "schlaefli_double_six::init_double_sixes h != 36" << endl;
		exit(1);
	}

	Double_six_label_tex = new string [36];

	for (i = 0; i < 36; i++) {
		if (i < 1) {
			Double_six_label_tex[i] = "{\\cal D}";
		}
		else if (i < 1 + 15) {
			ij = i - 1;
			Combi.k2ij(ij, a, b, 6);
			set[0] = a;
			set[1] = b;
			Combi.set_complement(set, 2 /* subset_size */, set + 2,
				size_complement, 6 /* universal_set_size */);
			Double_six_label_tex[i] =
					"{\\cal D}_{"
					+ std::to_string(a + 1)
					+ std::to_string(b + 1) + "}";
		}
		else {
			v = i - 16;
			Combi.unrank_k_subset(v, set, 6, 3);
			Combi.set_complement(set, 3 /* subset_size */, set + 3,
				size_complement, 6 /* universal_set_size */);
			a = set[0];
			b = set[1];
			c = set[2];
			Double_six_label_tex[i] =
					"{\\cal D}_{"
					+ std::to_string(a + 1)
					+ std::to_string(b + 1)
					+ std::to_string(c + 1) + "}";
		}
		if (f_v) {
			cout << "creating label " << Double_six_label_tex[i]
				<< " for Double six " << i << endl;
		}
	}

	if (f_v) {
		cout << "schlaefli_double_six::init_double_sixes done" << endl;
	}
}

void schlaefli_double_six::create_half_double_sixes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, ij, v, h;
	int set[6];
	int size_complement;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "schlaefli_double_six::create_half_double_sixes" << endl;
	}

	Half_double_six_characteristic_vector = NEW_int(72 * 27);
	Half_double_sixes = NEW_lint(72 * 6);
	Half_double_six_to_double_six = NEW_int(72);
	Half_double_six_to_double_six_row = NEW_int(72);

	Double_six_characteristic_vector = NEW_int(36 * 27);
	Int_vec_zero(Double_six_characteristic_vector, 36 * 27);
	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			for (h = 0; h < 6; h++) {
				a = Double_six[(2 * i + j) * 6 + h];
				Double_six_characteristic_vector[i * 27 + a] = 1;
			}
		}
	}


	Int_vec_zero(Half_double_six_characteristic_vector, 72 * 27);
	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			for (h = 0; h < 6; h++) {
				a = Double_six[(2 * i + j) * 6 + h];
				Half_double_six_characteristic_vector[(2 * i + j) * 27 + a] = 1;
			}
		}
	}


	Lint_vec_copy(Double_six, Half_double_sixes, 36 * 12);
	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			Sorting.lint_vec_heapsort(
				Half_double_sixes + (2 * i + j) * 6, 6);
			Half_double_six_to_double_six[2 * i + j] = i;
			Half_double_six_to_double_six_row[2 * i + j] = j;
		}
	}
	Half_double_six_label_tex = new string [72];

	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			string str;

			if (i < 1) {
				str = "D";
			}
			else if (i < 1 + 15) {
				ij = i - 1;
				Combi.k2ij(ij, a, b, 6);
				set[0] = a;
				set[1] = b;
				Combi.set_complement(set, 2 /* subset_size */,
					set + 2, size_complement,
					6 /* universal_set_size */);
				str = "D_{" + std::to_string(a + 1) + std::to_string(b + 1) + "}";
			}
			else {
				v = i - 16;
				Combi.unrank_k_subset(v, set, 6, 3);
				Combi.set_complement(set, 3 /* subset_size */,
					set + 3, size_complement,
					6 /* universal_set_size */);
				a = set[0];
				b = set[1];
				c = set[2];
				str = "D_{"
						+ std::to_string(a + 1)
						+ std::to_string(b + 1)
						+ std::to_string(c + 1) + "}";
			}


			if (j == 0) {
				str += "^\\top";
			}
			else {
				str += "^\\bot";
			}
			if (f_v) {
				cout << "creating label " << str
					<< " for half double six "
					<< 2 * i + j << endl;
			}
			Half_double_six_label_tex[2 * i + j] = str;
		}
	}

	if (f_v) {
		cout << "schlaefli_double_six::create_half_double_sixes done" << endl;
	}
}

int schlaefli_double_six::find_half_double_six(
		long int *half_double_six)
{
	int i;
	other::data_structures::sorting Sorting;

	Sorting.lint_vec_heapsort(half_double_six, 6);
	for (i = 0; i < 72; i++) {
		if (Sorting.lint_vec_compare(half_double_six,
			Half_double_sixes + i * 6, 6) == 0) {
			return i;
		}
	}
	cout << "schlaefli_double_six::find_half_double_six did not find "
			"half double six" << endl;
	exit(1);
}


void schlaefli_double_six::latex_table_of_double_sixes(
		std::ostream &ost)
{
	int h;

	//cout << "schlaefli::latex_table_of_double_sixes" << endl;



	//ost << "\\begin{multicols}{2}" << endl;
	for (h = 0; h < 36; h++) {

		ost << "$D_{" << h << "} = " << Double_six_label_tex[h] << endl;

		ost << " = " << endl;


		latex_double_six_symbolic(ost, h);

		ost << " = " << endl;


		latex_double_six_index_set(ost, h);

		ost << "$\\\\" << endl;
		}
	//ost << "\\end{multicols}" << endl;

	//cout << "schlaefli_double_six::latex_table_of_double_sixes done" << endl;

}


void schlaefli_double_six::latex_double_six_symbolic(
		std::ostream &ost, int idx)
{
	int i, j;
	long int D[12];

	Lint_vec_copy(Double_six + idx * 12, D, 12);


	ost << "\\left[";
	ost << "\\begin{array}{cccccc}" << endl;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 6; j++) {
			ost << Schlaefli->Labels->Line_label_tex[D[i * 6 + j]];
			if (j < 6 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
}

void schlaefli_double_six::latex_double_six_index_set(
		std::ostream &ost, int idx)
{
	int i, j;
	long int D[12];

	Lint_vec_copy(Double_six + idx * 12, D, 12);


	ost << "\\left[";
	ost << "\\begin{array}{cccccc}" << endl;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 6; j++) {
			ost << D[i * 6 + j];
			if (j < 6 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
}


void schlaefli_double_six::latex_table_of_half_double_sixes(
		std::ostream &ost)
{
	int i;

	//cout << "schlaefli_double_six::latex_table_of_half_double_sixes" << endl;



	//ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < 72; i++) {


		ost << "$" << endl;

		latex_half_double_six(ost, i);

		ost << "$\\\\" << endl;

	}
	//ost << "\\end{multicols}" << endl;



	//cout << "schlaefli_double_six::latex_table_of_double_sixes done" << endl;
}

void schlaefli_double_six::latex_half_double_six(
		std::ostream &ost, int idx)
{
	int j;
	long int H[6];

	//cout << "schlaefli_double_six::latex_table_of_half_double_sixes" << endl;




	Lint_vec_copy(Half_double_sixes + idx * 6, H, 6);

	ost << "H_{" << idx << "} = " << Half_double_six_label_tex[idx] << endl;

	ost << " = \\{";
	for (j = 0; j < 6; j++) {
		ost << Schlaefli->Labels->Line_label_tex[H[j]];
		if (j < 6 - 1) {
			ost << ", ";
		}
	}
	ost << "\\}";

	ost << "= \\{";

	for (j = 0; j < 6; j++) {
		ost << H[j];
		if (j < 6 - 1) {
			ost << ", ";
		}
	}
	ost << "\\}";




	//cout << "schlaefli_double_six::latex_table_of_double_sixes done" << endl;
}

void schlaefli_double_six::print_half_double_sixes_in_GAP()
{
	int i, j;

	cout << "[";
	for (i = 0; i < 72; i++) {
		cout << "[";
		for (j = 0; j < 6; j++) {
			cout << Half_double_sixes[i * 6 + j] + 1;
			if (j < 6 - 1) {
				cout << ", ";
			}
		}
		cout << "]";
		if (i < 72 - 1) {
			cout << "," << endl;
		}
	}
	cout << "];" << endl;
}

void schlaefli_double_six::write_double_sixes(
		std::string &prefix, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schlaefli_double_six::write_double_sixes" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	string fname;

	fname = prefix + "_single_sixes_char_vec.csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Half_double_six_characteristic_vector,
			72, 27);

	fname = prefix + "_double_sixes_char_vec.csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Double_six_characteristic_vector,
			36, 27);



	if (f_v) {
		cout << "schlaefli_double_six::write_double_sixes done" << endl;
	}
}

void schlaefli_double_six::print_half_double_sixes_numerically(
		std::ostream &ost)
{
	other::l1_interfaces::latex_interface L;

	ost << "The half double sixes are:\\\\" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
			Half_double_sixes, 36, 6, true /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
			Half_double_sixes + 36 * 6,
		36, 6, 36, 0, true /* f_tex */);
	ost << "$$" << endl;
}

void schlaefli_double_six::latex_double_six(
		std::ostream &ost, long int *Lines, int idx)
{
	int i, j;
	long int D[12];

	Lint_vec_copy(Double_six + idx * 12, D, 12);


	ost << "\\left[";
	ost << "\\begin{array}{cccccc}" << endl;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 6; j++) {
			ost << Lines[D[i * 6 + j]];
			if (j < 6 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
}

void schlaefli_double_six::latex_double_six_wedge(
		std::ostream &ost, long int *Lines, int idx)
{
	int i, j;
	long int D[12];
	long int l;

	Lint_vec_copy(Double_six + idx * 12, D, 12);


	ost << "\\left[";
	ost << "\\begin{array}{cccccc}" << endl;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 6; j++) {

			l = Schlaefli->Surf->line_to_wedge(Lines[D[i * 6 + j]]);

			ost << l;
			if (j < 6 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
}



void schlaefli_double_six::latex_double_six_Klein(
		std::ostream &ost, long int *Lines, int idx)
{
	int i, j;
	long int D[12];
	long int line_rk, a;

	Lint_vec_copy(Double_six + idx * 12, D, 12);


	ost << "\\left[";
	ost << "\\begin{array}{cccccc}" << endl;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 6; j++) {

			line_rk = Lines[D[i * 6 + j]];

			a = Schlaefli->Surf->Klein->line_to_point_on_quadric(line_rk, 0 /* verbose_level*/);


			ost << a;
			if (j < 6 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
}


void schlaefli_double_six::latex_double_six_Pluecker_coordinates_transposed(
		std::ostream &ost, long int *Lines, int idx)
{
	int i, j;
	long int D[12];
	long int line_rk;

	Lint_vec_copy(Double_six + idx * 12, D, 12);


	ost << "\\left[";
	ost << "\\begin{array}{cc}" << endl;
	for (j = 0; j < 6; j++) {
		for (i = 0; i < 2; i++) {

			line_rk = Lines[D[i * 6 + j]];

			//a = Surf->Klein->line_to_point_on_quadric(line_rk, 0 /* verbose_level*/);

			//Surf->Gr->unrank_lint(line_rk, 0 /*verbose_level*/);


			int v6[6];
			int vv[6];

			Schlaefli->Surf->Gr->Pluecker_coordinates(line_rk, v6, 0 /* verbose_level */);

			Int_vec_copy(v6, vv, 6); // mistake found by Alice Hui

			//klein_rk = F->Orthogonal_indexing->Qplus_rank(vv, 1, 5, 0 /* verbose_level*/);

			ost << "{\\rm\\bf Pl}(" << v6[0] << "," << v6[1] << ","
					<< v6[2] << "," << v6[3] << "," << v6[4]
					<< "," << v6[5] << " ";
			ost << ")";



			if (i < 2 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
}

void schlaefli_double_six::latex_double_six_Klein_transposed(
		std::ostream &ost, long int *Lines, int idx)
{
	int i, j;
	long int D[12];
	long int line_rk, a;

	Lint_vec_copy(Double_six + idx * 12, D, 12);


	ost << "\\left[";
	ost << "\\begin{array}{cc}" << endl;
	for (j = 0; j < 6; j++) {
		for (i = 0; i < 2; i++) {

			line_rk = Lines[D[i * 6 + j]];

			a = Schlaefli->Surf->Klein->line_to_point_on_quadric(
					line_rk, 0 /* verbose_level*/);

			ost << a;



			if (i < 2 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
}

void schlaefli_double_six::print_double_sixes(
		std::ostream &ost, long int *Lines)
{
	int idx;



	//SO->Surf->Schlaefli->latex_table_of_double_sixes(ost);

	for (idx = 0; idx < 36; idx++) {

		ost << "$D_{" << idx << "} = "
				<< Double_six_label_tex[idx] << endl;

		ost << " = " << endl;

		latex_double_six_symbolic(ost, idx);

		ost << " = " << endl;

		latex_double_six_index_set(ost, idx);

		ost << "$\\\\" << endl;



		ost << "$" << endl;

		ost << " = " << endl;

		latex_double_six(ost, Lines, idx);

		ost << "$\\\\" << endl;

		ost << "$" << endl;

		ost << " = " << endl;

		latex_double_six_wedge(ost, Lines, idx);

		ost << "$\\\\" << endl;

		ost << "$" << endl;

		ost << " = " << endl;

		latex_double_six_Klein(ost, Lines, idx);

		ost << "$\\\\" << endl;

		ost << "$" << endl;

		ost << " = " << endl;

		latex_double_six_Pluecker_coordinates_transposed(ost, Lines, idx);

		ost << "$\\\\" << endl;

		ost << "$" << endl;

		ost << " = " << endl;

		latex_double_six_Klein_transposed(ost, Lines, idx);

		ost << "$\\\\" << endl;

	}


}


}}}}

