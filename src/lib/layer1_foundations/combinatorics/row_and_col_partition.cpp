/*
 * row_and_col_partition.cpp
 *
 *  Created on: Nov 17, 2023
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {


row_and_col_partition::row_and_col_partition()
{

	Stack = NULL;

	row_classes = NULL;
	row_class_idx = NULL;
	nb_row_classes = 0;

	col_classes = NULL;
	col_class_idx = NULL;
	nb_col_classes = 0;

}



row_and_col_partition::~row_and_col_partition()
{
	if (row_classes) {
		FREE_int(row_classes);
	}
	if (row_class_idx) {
		FREE_int(row_class_idx);
	}

	if (col_classes) {
		FREE_int(col_classes);
	}
	if (col_class_idx) {
		FREE_int(col_class_idx);
	}

}

void row_and_col_partition::init_from_partitionstack(
		data_structures::partitionstack *Stack,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "row_and_col_partition::init_from_partitionstack" << endl;
	}

	int i, c, l;


	row_and_col_partition::Stack = Stack;


	row_classes = NEW_int(Stack->ht);
	col_classes = NEW_int(Stack->ht);
	row_class_idx = NEW_int(Stack->ht);
	col_class_idx = NEW_int(Stack->ht);

	nb_row_classes = 0;
	nb_col_classes = 0;
#if 0
	for (c = 0; c < ht; c++) {
		if (is_row_class(c)) {
			row_classes[nb_row_classes++] = c;
			}
		else {
			col_classes[nb_col_classes++] = c;
			}
		}
#endif
	i = 0;
	while (i < Stack->n) {
		c = Stack->cellNumber[i];
		if (Stack->is_row_class(c)) {
			row_classes[nb_row_classes++] = c;
		}
		else {
			col_classes[nb_col_classes++] = c;
		}
		l = Stack->cellSize[c];
		i += l;
	}

	for (i = 0; i < Stack->ht; i++) {
		row_class_idx[i] = col_class_idx[i] = -1;
	}
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		row_class_idx[c] = i;
	}
	for (i = 0; i < nb_col_classes; i++) {
		c = col_classes[i];
		col_class_idx[c] = i;
	}

	if (f_v) {
		cout << "row_and_col_partition::init_from_partitionstack done" << endl;
	}


}


void row_and_col_partition::print_classes_of_decomposition_tex(
		std::ostream &ost)
{
	int i, j, c, f, l, a;
	int first_column_element = Stack->startCell[1];

	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		f = Stack->startCell[c];
		l = Stack->cellSize[c];
		ost << "V" << i << " of size " << l << " contains ";
		for (j = 0; j < l; j++) {
			a = Stack->pointList[f + j];
			ost << a;
			if (j < l - 1) {
				ost << ", ";
			}
			if ((j + 1) % 25 == 0) {
				ost << "\\\\" << endl;
			}
		}
		ost << "\\\\" << endl;
	}
	for (i = 0; i < nb_col_classes; i++) {
		c = col_classes[i];
		f = Stack->startCell[c];
		l = Stack->cellSize[c];
		ost << "B" << i << " of size " << l << " contains ";
		for (j = 0; j < l; j++) {
			a = Stack->pointList[f + j] - first_column_element;
			ost << a;
			if (j < l - 1) {
				ost << ", ";
			}
			if ((j + 1) % 25 == 0) {
				ost << "\\\\" << endl;
			}
		}
		ost << "\\\\" << endl;
	}
}

void row_and_col_partition::print_decomposition_scheme(
		std::ostream &ost,
	int *scheme)
{
	int c, i, j;

	ost << "             | ";
	for (j = 0; j < nb_col_classes; j++) {
		c = col_classes[j];
		ost << setw(6) << Stack->cellSize[c] << "_{" << setw(3) << c << "}";
	}
	ost << endl;

	ost << "---------------";
	for (i = 0; i < nb_col_classes; i++) {
		ost << "------------";
	}
	ost << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << Stack->cellSize[c] << "_{" << setw(3) << c << "}";
		ost << " | ";
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << setw(12) << scheme[i * nb_col_classes + j];
		}
		ost << endl;
	}
	ost << endl;
	ost << endl;
}

void row_and_col_partition::print_row_tactical_decomposition_scheme_tex(
	std::ostream &ost, int f_enter_math_mode,
	int *row_scheme, int f_print_subscripts)
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
		ost << setw(6) << Stack->cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
		}
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << Stack->cellSize[c];
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

void row_and_col_partition::print_column_tactical_decomposition_scheme_tex(
	std::ostream &ost, int f_enter_math_mode,
	int *col_scheme, int f_print_subscripts)
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
		ost << setw(6) << Stack->cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
		}
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << Stack->cellSize[c];
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



}}}


