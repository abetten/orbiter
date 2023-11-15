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
namespace geometry {


decomposition_scheme::decomposition_scheme()
{
	Decomposition = NULL;


	row_classes = NULL;
	row_class_inv = NULL;
	nb_row_classes = 0;

	col_classes = NULL;
	col_class_inv = NULL;
	nb_col_classes = 0;


	f_has_row_scheme = false;
	row_scheme = NULL;

	f_has_col_scheme = false;
	col_scheme = NULL;

	SoS_points = NULL;
	SoS_lines = NULL;

}



decomposition_scheme::~decomposition_scheme()
{
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

	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"before Decomposition->Stack->allocate_and_get_decomposition" << endl;
	}
	Decomposition->Stack->allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		verbose_level - 2);
	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"after Decomposition->Stack->allocate_and_get_decomposition" << endl;
	}

	f_has_col_scheme = true;
	col_scheme = NEW_int(nb_row_classes * nb_col_classes);

	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"before Decomposition->get_col_decomposition_scheme" << endl;
	}
	Decomposition->get_col_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		col_scheme, verbose_level);
	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"after Decomposition->get_col_decomposition_scheme" << endl;
	}

	f_has_row_scheme = true;
	row_scheme = NEW_int(nb_row_classes * nb_col_classes);

	if (f_v) {
		cout << "decomposition_scheme::init_row_and_col_schemes "
				"before Decomposition->get_row_decomposition_scheme" << endl;
	}
	Decomposition->get_row_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
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

void decomposition_scheme::get_classes(int verbose_level)
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
	Decomposition->Stack->allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		verbose_level);

	f_has_row_scheme = true;
	row_scheme = NEW_int(nb_row_classes * nb_col_classes);
	//col_scheme = NEW_int(nb_row_classes * nb_col_classes);

	if (f_v) {
		cout << "decomposition_scheme::init_row_scheme "
				"before Decomposition->get_row_decomposition_scheme" << endl;
	}
	Decomposition->get_row_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
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
	Decomposition->Stack->allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		verbose_level);

	f_has_col_scheme = true;
	col_scheme = NEW_int(nb_row_classes * nb_col_classes);
	//col_scheme = NEW_int(nb_row_classes * nb_col_classes);

	if (f_v) {
		cout << "decomposition_scheme::init_col_scheme "
				"before Decomposition->get_col_decomposition_scheme" << endl;
	}
	Decomposition->get_col_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
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
	row_scheme = NEW_int(nb_row_classes * nb_col_classes);
	Decomposition->get_row_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
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
	col_scheme = NEW_int(nb_row_classes * nb_col_classes);
	Decomposition->get_col_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
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
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << " ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		c = col_classes[j];
		ost << setw(6) << Decomposition->Stack->cellSize[c] << "_{" << setw(3) << c << "}";
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << Decomposition->Stack->cellSize[c] << "_{" << setw(3) << c << "}";
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & " << setw(12) << scheme[i * nb_col_classes + j];
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
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << " ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		c = col_classes[j];
		ost << setw(6) << Decomposition->Stack->cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
		}
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << Decomposition->Stack->cellSize[c];
			if (f_print_subscripts) {
				ost << "_{" << setw(3) << c << "}";
			}
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & " << setw(12) << row_scheme[i * nb_col_classes + j]
				<< "\\backslash " << col_scheme[i * nb_col_classes + j];
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

	ost << "%{\\renewcommand{\\arraycolsep}{1pt}" << endl;
	if (f_enter_math_mode) {
		ost << "\\begin{align*}" << endl;
	}
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << "\\rightarrow ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		c = col_classes[j];
		ost << setw(6) << Decomposition->Stack->cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
		}
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << Decomposition->Stack->cellSize[c];
			if (f_print_subscripts) {
				ost << "_{" << setw(3) << c << "}";
			}
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & " << setw(12) << row_scheme[i * nb_col_classes + j];
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "\\end{align*}" << endl;
	}
	ost << "%}" << endl;
}

void decomposition_scheme::print_column_tactical_decomposition_scheme_tex(
	std::ostream &ost, int f_enter_math_mode,
	int f_print_subscripts)
{
	int c, i, j;

	ost << "%{\\renewcommand{\\arraycolsep}{1pt}" << endl;
	if (f_enter_math_mode) {
		ost << "\\begin{align*}" << endl;
	}
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << "\\downarrow ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		c = col_classes[j];
		ost << setw(6) << Decomposition->Stack->cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
		}
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << Decomposition->Stack->cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
		}
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & " << setw(12) << col_scheme[i * nb_col_classes + j];
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "\\end{align*}" << endl;
	}
	ost << "%}" << endl;
}

void decomposition_scheme::print_non_tactical_decomposition_scheme_tex(
	std::ostream &ost, int f_enter_math_mode,
	int f_print_subscripts)
{
	int c, i, j;

	if (f_enter_math_mode) {
		ost << "\\begin{align*}" << endl;
	}
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << " ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		c = col_classes[j];
		ost << setw(6) << Decomposition->Stack->cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
		}
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << Decomposition->Stack->cellSize[c];
			if (f_print_subscripts) {
				ost << "_{" << setw(3) << c << "}";
			}
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & ";
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "\\end{align*}" << endl;
	}
}




}}}



