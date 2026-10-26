/*
 * incidence_structure_by_flags.cpp
 *
 *  Created on: Sep 20, 2025
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace design_theory {




incidence_structure_by_flags::incidence_structure_by_flags()
{
	Record_birth();

	flags = NULL;
	nb_flags = 0;
	nb_rows = 0;
	nb_cols = 0;
}



incidence_structure_by_flags::~incidence_structure_by_flags()
{
	Record_death();

	if (flags) {
		FREE_int(flags);
	}
}


void incidence_structure_by_flags::init(
		int *flags, int nb_flags, int nb_rows, int nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "incidence_structure_by_flags::init" << endl;
	}



	incidence_structure_by_flags::flags = NEW_int(nb_flags);
	Int_vec_copy(flags, incidence_structure_by_flags::flags, nb_flags);

	incidence_structure_by_flags::nb_flags = nb_flags;
	incidence_structure_by_flags::nb_rows = nb_rows;
	incidence_structure_by_flags::nb_cols = nb_cols;

	if (f_v) {
		cout << "incidence_structure_by_flags::init done" << endl;
	}


}

void incidence_structure_by_flags::print()
{
	cout << "flags=";
	Int_vec_print_fully(cout, flags, nb_flags);
	cout << endl;

	cout << "nb_rows=" << nb_rows << endl;
	cout << "nb_cols=" << nb_cols << endl;
	cout << "nb_flags=" << nb_flags << endl;

}

void incidence_structure_by_flags::print_latex(
		std::ostream &ost)
{
	ost << "Flags=";
	Int_vec_print_fully(ost, flags, nb_flags);
	ost << "\\\\" << endl;

	ost << "nb\\_rows=" << nb_rows << "\\\\" << endl;
	ost << "nb\\_cols=" << nb_cols << "\\\\" << endl;
	ost << "nb\\_flags=" << nb_flags << "\\\\" << endl;

}

void incidence_structure_by_flags::print_incma_latex(
		std::ostream &ost)
{
	int *Incma;
	int a, h, i, j;

	Incma = NEW_int(nb_rows * nb_cols);
	Int_vec_zero(Incma, nb_rows * nb_cols);
	for (h = 0; h < nb_flags; h++) {
		a = flags[h];
		i = a / nb_cols;
		j = a % nb_cols;
		Incma[i * nb_cols + j] = 1;
	}
	ost << "Incma=\\\\" << endl;
	for (i = 0; i < nb_rows; i++) {
		Int_vec_print_fully(ost, Incma + i * nb_cols, nb_cols);
		ost << "\\\\" << endl;
	}
	FREE_int(Incma);
}

void incidence_structure_by_flags::draw_incma_as_bitmap(
		std::string fname_base,
		int f_box_width, int box_width,
		int bit_depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "incidence_structure_by_flags::draw_incma_as_bitmap" << endl;
	}

	int *Incma;
	int a, h, i, j;

	Incma = NEW_int(nb_rows * nb_cols);
	Int_vec_zero(Incma, nb_rows * nb_cols);
	for (h = 0; h < nb_flags; h++) {
		a = flags[h];
		i = a / nb_cols;
		j = a % nb_cols;
		Incma[i * nb_cols + j] = 1;
	}

	other::l1_interfaces::easy_BMP_interface BMP;
	other::graphics::draw_bitmap_control Draw_bitmap_control;

	Draw_bitmap_control.M = Incma;
	Draw_bitmap_control.m = nb_rows;
	Draw_bitmap_control.n = nb_cols;
	Draw_bitmap_control.input_csv_file_name = fname_base + ".csv";
	Draw_bitmap_control.f_box_width = f_box_width;
	Draw_bitmap_control.box_width = box_width;
	Draw_bitmap_control.bit_depth = bit_depth;
	Draw_bitmap_control.f_invert_colors = false;

	BMP.draw_bitmap(&Draw_bitmap_control, verbose_level);

	FREE_int(Incma);

	if (f_v) {
		cout << "incidence_structure_by_flags::draw_incma_as_bitmap done" << endl;
	}
}

}}}}


