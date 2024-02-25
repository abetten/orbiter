/*
 * decomposition_scheme.cpp
 *
 *  Created on: Nov 1, 2023
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {


decomposition_scheme::decomposition_scheme()
{
	Decomposition = NULL;

	RC = NULL;

#if 0
	row_classes = NULL;
	row_class_inv = NULL;
	nb_row_classes = 0;

	col_classes = NULL;
	col_class_inv = NULL;
	nb_col_classes = 0;
#endif

	f_has_row_scheme = false;
	row_scheme = NULL;

	f_has_col_scheme = false;
	col_scheme = NULL;

	SoS_points = NULL;
	SoS_lines = NULL;

}



decomposition_scheme::~decomposition_scheme()
{
	if (RC) {
		FREE_OBJECT(RC);
	}
#if 0
	if (row_classes) {
		FREE_int(row_classes);
	}
	if (row_class_inv) {
		FREE_int(row_class_inv);
	}
	if (col_classes) {
		FREE_int(col_classes);
	}
	if (col_class_inv) {
		FREE_int(col_class_inv);
	}
#endif

	if (f_has_row_scheme) {
		FREE_int(row_scheme);
	}
	if (f_has_col_scheme) {
		FREE_int(col_scheme);
	}

	if (SoS_points) {
		FREE_OBJECT(SoS_points);
	}
	if (SoS_lines) {
		FREE_OBJECT(SoS_lines);
	}


}

void decomposition_scheme::init_row_and_col_schemes(
		decomposition *Decomposition,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes" << endl;
	}

	decomposition_scheme::Decomposition = Decomposition;

	RC = NEW_OBJECT(row_and_col_partition);

	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"before RC->init_from_partitionstack" << endl;
	}
	RC->init_from_partitionstack(
			Decomposition->Stack,
			verbose_level);

	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"after RC->init_from_partitionstack" << endl;
	}

	f_has_col_scheme = true;
	col_scheme = NEW_int(RC->nb_row_classes * RC->nb_col_classes);

	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"before Decomposition->get_col_decomposition_scheme" << endl;
	}
	Decomposition->get_col_decomposition_scheme(
		RC,
		col_scheme, verbose_level);
	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"after Decomposition->get_col_decomposition_scheme" << endl;
	}

	f_has_row_scheme = true;
	row_scheme = NEW_int(RC->nb_row_classes * RC->nb_col_classes);

	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"before Decomposition->get_row_decomposition_scheme" << endl;
	}
	Decomposition->get_row_decomposition_scheme(
		RC,
		row_scheme, verbose_level);
	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"after Decomposition->get_row_decomposition_scheme" << endl;
	}


	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"before get_classes" << endl;
	}
	get_classes(verbose_level - 1);
	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"after get_classes" << endl;
	}


	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes done" << endl;
	}

}

void decomposition_scheme::get_classes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_scheme::get_classes" << endl;
	}

	if (f_v) {
		cout << "decomposition_scheme::get_classes "
				"before Decomposition->Stack->get_row_classes" << endl;
	}
	Decomposition->Stack->get_row_classes(
			SoS_points, 0 /*verbose_level*/);
	if (f_v) {
		cout << "decomposition_scheme::get_classes "
				"after Decomposition->Stack->get_row_classes" << endl;
	}
	if (f_v) {
		cout << "decomposition_scheme::get_classes "
				"before Decomposition->Stack->get_column_classes" << endl;
	}
	Decomposition->Stack->get_column_classes(
			SoS_lines, 0 /*verbose_level*/);
	if (f_v) {
		cout << "decomposition_scheme::get_classes "
				"after Decomposition->Stack->get_column_classes" << endl;
	}

	if (f_v) {
		cout << "decomposition_scheme::get_classes done" << endl;
	}

}

void decomposition_scheme::init_row_scheme(
		decomposition *Decomposition,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_scheme::init_row_scheme" << endl;
	}
	RC = NEW_OBJECT(row_and_col_partition);

	if (f_v) {
		cout << "decomposition_scheme::init_row_scheme "
				"before RC->init_from_partitionstack" << endl;
	}
	RC->init_from_partitionstack(
			Decomposition->Stack,
			verbose_level);

	if (f_v) {
		cout << "decomposition_scheme::init_row_scheme "
				"after RC->init_from_partitionstack" << endl;
	}


	f_has_row_scheme = true;
	row_scheme = NEW_int(RC->nb_row_classes * RC->nb_col_classes);
	//col_scheme = NEW_int(nb_row_classes * nb_col_classes);

	if (f_v) {
		cout << "decomposition_scheme::init_row_scheme "
				"before Decomposition->get_row_decomposition_scheme" << endl;
	}
	Decomposition->get_row_decomposition_scheme(
		RC,
		row_scheme, verbose_level);
	if (f_v) {
		cout << "decomposition_scheme::init_row_scheme "
				"after Decomposition->get_row_decomposition_scheme" << endl;
	}

	if (f_v) {
		cout << "decomposition_scheme::init_row_scheme "
				"before get_classes" << endl;
	}
	get_classes(verbose_level - 1);
	if (f_v) {
		cout << "decomposition_scheme::init_row_scheme "
				"after get_classes" << endl;
	}

	if (f_v) {
		cout << "decomposition_scheme::init_row_scheme done" << endl;
	}

}

void decomposition_scheme::init_col_scheme(
		decomposition *Decomposition,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_scheme::init_col_scheme" << endl;
	}
	RC = NEW_OBJECT(row_and_col_partition);

	if (f_v) {
		cout << "decomposition_scheme::init_col_scheme "
				"before RC->init_from_partitionstack" << endl;
	}
	RC->init_from_partitionstack(
			Decomposition->Stack,
			verbose_level);

	if (f_v) {
		cout << "decomposition_scheme::init_col_scheme "
				"after RC->init_from_partitionstack" << endl;
	}

	f_has_col_scheme = true;
	col_scheme = NEW_int(RC->nb_row_classes * RC->nb_col_classes);
	//col_scheme = NEW_int(nb_row_classes * nb_col_classes);

	if (f_v) {
		cout << "decomposition_scheme::init_col_scheme "
				"before Decomposition->get_col_decomposition_scheme" << endl;
	}
	Decomposition->get_col_decomposition_scheme(
		RC,
		row_scheme, verbose_level);
	if (f_v) {
		cout << "decomposition_scheme::init_col_scheme "
				"after Decomposition->get_col_decomposition_scheme" << endl;
	}

	if (f_v) {
		cout << "decomposition_scheme::init_col_scheme "
				"before get_classes" << endl;
	}
	get_classes(verbose_level - 1);
	if (f_v) {
		cout << "decomposition_scheme::init_col_scheme "
				"after get_classes" << endl;
	}

	if (f_v) {
		cout << "decomposition_scheme::init_col_scheme done" << endl;
	}

}


void decomposition_scheme::get_row_scheme(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_scheme::get_row_scheme" << endl;
	}

	f_has_row_scheme = true;
	row_scheme = NEW_int(RC->nb_row_classes * RC->nb_col_classes);
	Decomposition->get_row_decomposition_scheme(
		RC,
		row_scheme, 0);
	if (f_v) {
		cout << "decomposition_scheme::get_row_scheme done" << endl;
	}
}

void decomposition_scheme::get_col_scheme(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_scheme::get_col_scheme" << endl;
	}

	f_has_col_scheme = true;
	col_scheme = NEW_int(RC->nb_row_classes * RC->nb_col_classes);
	Decomposition->get_col_decomposition_scheme(
		RC,
		col_scheme, 0);
	if (f_v) {
		cout << "decomposition_scheme::get_col_scheme done" << endl;
	}
}

void decomposition_scheme::print_row_decomposition_tex(
	std::ostream &ost,
	int f_enter_math, int f_print_subscripts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_scheme::print_row_decomposition_tex" << endl;
	}
	if (!f_has_row_scheme) {
		cout << "decomposition_scheme::print_row_decomposition_tex "
				"!f_has_row_scheme" << endl;
		exit(1);
	}
	//I->get_and_print_row_tactical_decomposition_scheme_tex(
	//	file, false /* f_enter_math */, *Stack);

	print_row_tactical_decomposition_scheme_tex(
		ost, f_enter_math,
		f_print_subscripts);
}

void decomposition_scheme::print_column_decomposition_tex(
	std::ostream &ost,
	int f_enter_math, int f_print_subscripts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_scheme::print_column_decomposition_tex" << endl;
	}
	if (!f_has_col_scheme) {
		cout << "decomposition_scheme::print_column_decomposition_tex "
				"!f_has_col_scheme" << endl;
		exit(1);
	}
	//I->get_and_print_column_tactical_decomposition_scheme_tex(
	//	file, false /* f_enter_math */, *Stack);

	print_column_tactical_decomposition_scheme_tex(
		ost, f_enter_math,
		f_print_subscripts);
}


void decomposition_scheme::print_decomposition_scheme_tex(
		std::ostream &ost,
	int *scheme)
{
	int c, i, j;

	ost << "\\begin{align*}" << endl;
	ost << "\\begin{array}{r|*{" << RC->nb_col_classes << "}{r}}" << endl;
	ost << " ";
	for (j = 0; j < RC->nb_col_classes; j++) {
		ost << " & ";
		c = RC->col_classes[j];
		ost << setw(6) << Decomposition->Stack->cellSize[c] << "_{" << setw(3) << c << "}";
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < RC->nb_row_classes; i++) {
		c = RC->row_classes[i];
		ost << setw(6) << Decomposition->Stack->cellSize[c] << "_{" << setw(3) << c << "}";
		//f = P.startCell[c];
		for (j = 0; j < RC->nb_col_classes; j++) {
			ost << " & " << setw(12) << scheme[i * RC->nb_col_classes + j];
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\end{align*}" << endl;
}

void decomposition_scheme::print_tactical_decomposition_scheme_tex(
		std::ostream &ost,
	int f_print_subscripts)
{
	print_tactical_decomposition_scheme_tex_internal(ost, true,
		f_print_subscripts);
}

void decomposition_scheme::print_tactical_decomposition_scheme_tex_internal(
	std::ostream &ost, int f_enter_math_mode,
	int f_print_subscripts)
{
	int c, i, j;

	if (f_enter_math_mode) {
		ost << "\\begin{align*}" << endl;
	}
	ost << "\\begin{array}{r|*{" << RC->nb_col_classes << "}{r}}" << endl;
	ost << " ";
	for (j = 0; j < RC->nb_col_classes; j++) {
		ost << " & ";
		c = RC->col_classes[j];
		ost << setw(6) << Decomposition->Stack->cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
		}
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < RC->nb_row_classes; i++) {
		c = RC->row_classes[i];
		ost << setw(6) << Decomposition->Stack->cellSize[c];
			if (f_print_subscripts) {
				ost << "_{" << setw(3) << c << "}";
			}
		//f = P.startCell[c];
		for (j = 0; j < RC->nb_col_classes; j++) {
			ost << " & " << setw(12) << row_scheme[i * RC->nb_col_classes + j]
				<< "\\backslash " << col_scheme[i * RC->nb_col_classes + j];
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "\\end{align*}" << endl;
	}
}

void decomposition_scheme::print_row_tactical_decomposition_scheme_tex(
	std::ostream &ost, int f_enter_math_mode,
	int f_print_subscripts)
{
	int c, i, j;

	// prepare data:

	int m, n;
	string top_left_entry;
	string *cols_labels;
	string *row_labels;
	string *entries;

	m = RC->nb_row_classes;
	n = RC->nb_col_classes;
	top_left_entry = "\\rightarrow ";
	row_labels = new string[m];
	cols_labels = new string[n];
	entries = new string[m * n];
	for (j = 0; j < RC->nb_col_classes; j++) {
		c = RC->col_classes[j];
		cols_labels[j] = std::to_string(Decomposition->Stack->cellSize[c]);
		if (f_print_subscripts) {
			cols_labels[j] += "_{" + std::to_string(c) + "}";
		}
	}
	for (i = 0; i < RC->nb_row_classes; i++) {
		c = RC->row_classes[i];
		row_labels[i] = std::to_string(Decomposition->Stack->cellSize[c]);
		if (f_print_subscripts) {
			row_labels[i] += "_{" + std::to_string(c) + "}";
		}
	}
	for (i = 0; i < RC->nb_row_classes; i++) {
		for (j = 0; j < RC->nb_col_classes; j++) {
			c = row_scheme[i * n + j];
			entries[i * n + j] = std::to_string(c);
		}
	}

	l1_interfaces::latex_interface L;

	L.print_decomposition_matrix(
			ost,
			m, n,
			top_left_entry,
			cols_labels,
			row_labels,
			entries,
			f_enter_math_mode);


	delete [] cols_labels;
	delete [] row_labels;
	delete [] entries;
}

void decomposition_scheme::print_column_tactical_decomposition_scheme_tex(
	std::ostream &ost, int f_enter_math_mode,
	int f_print_subscripts)
{
	int c, i, j;


	// prepare data:

	int m, n;
	string top_left_entry;
	string *cols_labels;
	string *row_labels;
	string *entries;

	m = RC->nb_row_classes;
	n = RC->nb_col_classes;
	top_left_entry = "\\downarrow ";
	row_labels = new string[m];
	cols_labels = new string[n];
	entries = new string[m * n];
	for (j = 0; j < RC->nb_col_classes; j++) {
		c = RC->col_classes[j];
		cols_labels[j] = std::to_string(Decomposition->Stack->cellSize[c]);
		if (f_print_subscripts) {
			cols_labels[j] += "_{" + std::to_string(c) + "}";
		}
	}
	for (i = 0; i < RC->nb_row_classes; i++) {
		c = RC->row_classes[i];
		row_labels[i] = std::to_string(Decomposition->Stack->cellSize[c]);
		if (f_print_subscripts) {
			row_labels[i] += "_{" + std::to_string(c) + "}";
		}
	}
	for (i = 0; i < RC->nb_row_classes; i++) {
		for (j = 0; j < RC->nb_col_classes; j++) {
			c = col_scheme[i * n + j];
			entries[i * n + j] = std::to_string(c);
		}
	}

	l1_interfaces::latex_interface L;

	L.print_decomposition_matrix(
			ost,
			m, n,
			top_left_entry,
			cols_labels,
			row_labels,
			entries,
			f_enter_math_mode);

	delete [] cols_labels;
	delete [] row_labels;
	delete [] entries;

}

void decomposition_scheme::print_non_tactical_decomposition_scheme_tex(
	std::ostream &ost, int f_enter_math_mode,
	int f_print_subscripts)
{
	int c, i, j;

	if (f_enter_math_mode) {
		ost << "\\begin{align*}" << endl;
	}
	ost << "\\begin{array}{r|*{" << RC->nb_col_classes << "}{r}}" << endl;
	ost << " ";
	for (j = 0; j < RC->nb_col_classes; j++) {
		ost << " & ";
		c = RC->col_classes[j];
		ost << setw(6) << Decomposition->Stack->cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
		}
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < RC->nb_row_classes; i++) {
		c = RC->row_classes[i];
		ost << setw(6) << Decomposition->Stack->cellSize[c];
			if (f_print_subscripts) {
				ost << "_{" << setw(3) << c << "}";
			}
		//f = P.startCell[c];
		for (j = 0; j < RC->nb_col_classes; j++) {
			ost << " & ";
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "\\end{align*}" << endl;
	}
}

void decomposition_scheme::stringify_row_scheme(
		std::string *&Table, int f_print_subscripts)
// Table[(nb_row_classes + 1) * (nb_col_classes + 1)]
{
	Decomposition->stringify_decomposition(
			RC,
			Table,
			row_scheme,
			f_print_subscripts);

}

void decomposition_scheme::stringify_col_scheme(
		std::string *&Table, int f_print_subscripts)
// Table[(nb_row_classes + 1) * (nb_col_classes + 1)]
{
	Decomposition->stringify_decomposition(
			RC,
			Table,
			col_scheme,
			f_print_subscripts);

}


void decomposition_scheme::write_csv(
		std::string &fname_row, std::string &fname_col,
		std::string &fname_row_classes, std::string &fname_col_classes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_scheme::write_csv" << endl;
	}


	std::string *T_row;
	std::string *T_col;

	int f_print_subscripts = false;

	stringify_row_scheme(
			T_row, f_print_subscripts);

	stringify_col_scheme(
			T_col, f_print_subscripts);



	string *Headings;
	string headings;
	int nb_row, nb_col;

	nb_row = 1 + RC->nb_row_classes;
	nb_col = 1 + RC->nb_col_classes;

	Headings = new string[nb_col];


	int j;

	Headings[0] = "R";
	for (j = 0; j < RC->nb_col_classes; j++) {

		Headings[j + 1] = "C" + std::to_string(j);
	}

	for (j = 0; j < 1 + RC->nb_col_classes; j++) {
		headings += Headings[j];
		if (j < 1 + RC->nb_col_classes - 1) {
			headings += ",";
		}
	}

	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "decomposition_scheme::write_csv "
				"before Fio.Csv_file_support->write_table_of_strings" << endl;
	}
	Fio.Csv_file_support->write_table_of_strings(
			fname_row,
			nb_row, nb_col, T_row,
			headings,
			verbose_level);

	if (f_v) {
		cout << "decomposition_scheme::write_csv "
				"after Fio.Csv_file_support->write_table_of_strings" << endl;
	}

	if (f_v) {
		cout << "decomposition_scheme::write_csv "
				"before Fio.Csv_file_support->write_table_of_strings" << endl;
	}
	Fio.Csv_file_support->write_table_of_strings(
			fname_col,
			nb_row, nb_col, T_col,
			headings,
			verbose_level);

	if (f_v) {
		cout << "decomposition_scheme::write_csv "
				"after Fio.Csv_file_support->write_table_of_strings" << endl;
	}

	delete [] Headings;


	delete [] T_row;
	delete [] T_col;

	SoS_points->save_csv(
			fname_row_classes,
			verbose_level);

	SoS_lines->save_csv(
			fname_col_classes,
			verbose_level);

	if (f_v) {
		cout << "decomposition_scheme::write_csv done" << endl;
	}
}


}}}



