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
// called from
// combinatorics_domain::compute_TDO_decomposition_of_projective_space
// decomposition::compute_the_decomposition
// variety_with_TDO_and_TDA::init_and_compute_tactical_decompositions
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

void decomposition_scheme::report_latex_with_external_files(
		std::ostream &ost,
		std::string &label_scheme,
		std::string &label_txt,
		int upper_bound_on_size_for_printing,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_scheme::report_latex_with_external_files" << endl;
	}


	string fname1;
	string fname2;

	fname1 = "decomposition_" + label_scheme + "_" + label_txt + "_" + label_scheme + "_row";
	fname2 = "decomposition_" + label_scheme + "_" + label_txt + "_" + label_scheme + "_col";

	string fname1_tex;
	string fname2_tex;

	fname1_tex = fname1 + ".tex";
	fname2_tex = fname2 + ".tex";

	{
		std::ofstream ost(fname1_tex);

		int f_enter_math = false;
		int f_print_subscripts = true;

		print_row_tactical_decomposition_scheme_tex(
				ost, f_enter_math, f_print_subscripts);

	}
	{
		std::ofstream ost(fname2_tex);

		int f_enter_math = false;
		int f_print_subscripts = true;

		print_column_tactical_decomposition_scheme_tex(
				ost, f_enter_math, f_print_subscripts);

	}

	ost << endl << endl;

	int nb_row, nb_col;

	nb_row = RC->nb_row_classes;
	nb_col = RC->nb_col_classes;


	if (nb_row + nb_col < upper_bound_on_size_for_printing) {
		ost << label_scheme + " scheme of size " << nb_row << " x " << nb_col << ":" << endl;

		ost << "$$" << endl;
		ost << "\\input " << fname1_tex << endl;
		ost << "$$" << endl;
		ost << "$$" << endl;
		ost << "\\input " << fname2_tex << endl;
		ost << "$$" << endl;
	}
	else {
		ost << label_scheme + " scheme of size " << nb_row << " x " << nb_col
				<< " is too big for printing.\\" << endl;
		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;

	}

	if (f_v) {
		cout << "decomposition_scheme::report_latex_with_external_files done" << endl;
	}

}

void decomposition_scheme::report_classes_with_external_files(
		std::ostream &ost,
		std::string &label_scheme,
		std::string &label_txt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_scheme::report_classes_with_external_files" << endl;
	}


	string fname1;
	string fname2;

	fname1 = "decomposition_" + label_scheme + "_" + label_txt + "_" + label_scheme + "_row_classes";
	fname2 = "decomposition_" + label_scheme + "_" + label_txt + "_" + label_scheme + "_col_classes";


	string fname1_tex;
	string fname2_tex;

	fname1_tex = fname1 + ".tex";
	fname2_tex = fname2 + ".tex";

	{
		std::ofstream ost(fname1_tex);

		SoS_points->print_table_latex_simple(ost);

	}

	if (f_v) {
		cout << "decomposition_scheme::report_classes_with_external_files subtracting nb_points" << endl;
	}


	int nb_points;

	nb_points = Decomposition->nb_points;

	data_structures::set_of_sets *SoS_lines2;

	SoS_lines2 = SoS_lines->copy();

	SoS_lines2->add_constant_everywhere(
			0, //- nb_points,
			verbose_level - 2);


	{
		std::ofstream ost(fname2_tex);

		SoS_lines2->print_table_latex_simple(ost);

	}

	FREE_OBJECT(SoS_lines2);


	ost << label_scheme << " point classes:\\\\" << endl;
	//ost << "$$" << endl;
	ost << "\\input " << fname1_tex << endl;
	//ost << "$$" << endl;

	ost << label_scheme << " line classes:\\\\" << endl;
	//ost << "$$" << endl;
	ost << "\\input " << fname2_tex << endl;
	//ost << "$$" << endl;


}

void decomposition_scheme::export_csv(
		std::string &label_scheme,
		std::string &label_txt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_scheme::export_csv" << endl;
	}


	string fname1;
	string fname2;

	fname1 = "decomposition_" + label_scheme + "_" + label_txt + "_" + label_scheme + "_row";
	fname2 = "decomposition_" + label_scheme + "_" + label_txt + "_" + label_scheme + "_col";

	string fname1_csv;
	string fname2_csv;

	string fname1b_csv;
	string fname2b_csv;

	fname1_csv = fname1 + ".csv";
	fname2_csv = fname2 + ".csv";
	fname1b_csv = fname1 + "_sets.csv";
	fname2b_csv = fname2 + "_sets.csv";

	if (f_v) {
		cout << "decomposition_scheme::export_csv "
				"before write_csv" << endl;
	}
	write_csv(
			fname1_csv, fname2_csv,
			fname1b_csv, fname2b_csv,
			verbose_level);
	if (f_v) {
		cout << "decomposition_scheme::export_csv "
				"after write_csv" << endl;
	}

	if (f_v) {
		cout << "decomposition_scheme::export_csv done" << endl;
	}

}



}}}



