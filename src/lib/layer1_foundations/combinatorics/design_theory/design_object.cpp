/*
 * design_object.cpp
 *
 *  Created on: Feb 22, 2025
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace design_theory {




design_object::design_object()
{
	Record_birth();

	//std::string prefix;
	//std::string label_txt;
	//std::string label_tex;

	q = 0;
	F = NULL;
	k = 0;

	f_has_set = false;
	set = NULL;
	sz = 0;

	f_has_block_partition = false;
	block_partition_class_size = 0;

	block = NULL;

	v = b = nb_inc = 0;

	f_has_incma = false;
	incma = NULL;

	DC = NULL;

}



design_object::~design_object()
{
	Record_death();

	if (F) {
		FREE_OBJECT(F);
	}
	if (set) {
		FREE_lint(set);
	}
	if (block) {
		FREE_int(block);
	}
	if (incma) {
		FREE_int(incma);
	}

}

void design_object::compute_incidence_matrix_from_blocks(
		int *blocks, int nb_blocks, int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_object::compute_incidence_matrix_from_blocks" << endl;
	}

	if (f_v) {
		cout << "design_object::compute_incidence_matrix_from_blocks blocks = " << endl;
		Int_matrix_print(blocks, nb_blocks, k);
		cout << endl;
	}

	b = nb_blocks;
	int i, j, h;

	incma = NEW_int(v * b);
	Int_vec_zero(incma, v * b);

	for (j = 0; j < nb_blocks; j++) {
		for (h = 0; h < k; h++) {
			i = blocks[j * k + h];
			incma[i * b + j] = 1;
		}
	}

	nb_inc = nb_blocks * k;

	f_has_incma = true;

	if (f_v) {
		cout << "design_object::compute_incidence_matrix_from_blocks "
				"The incidence matrix is:" << endl;
		if (v + b > 50) {
			cout << "design_object::compute_incidence_matrix_from_blocks "
					"too large to print" << endl;
		}
		else {
			Int_matrix_print(incma, v, b);
		}
	}

	if (f_v) {
		cout << "design_object::compute_incidence_matrix_from_blocks done" << endl;
	}


}


void design_object::compute_blocks_from_incidence_matrix(
		long int *&blocks, int &nb_blocks, int &block_sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_object::compute_blocks_from_incidence_matrix" << endl;
	}

	nb_blocks = b;
	block_sz = k;
	int i, j, h;

	blocks = NEW_lint(b * block_sz);

	for (j = 0; j < b; j++) {
		h = 0;
		for (i = 0; i < v; i++) {
			if (incma[i * b + j]) {
				blocks[j * k + h++] = i;
			}
		}
		if (h != k) {
			cout << "the number of entries in the column "
					"which are one does not match" << endl;
			cout << "h = " << h << endl;
			cout << "k = " << k << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "design_object::compute_blocks_from_incidence_matrix "
				"blocks = " << endl;
		Lint_matrix_print(blocks, nb_blocks, k);
		cout << endl;
	}



	if (f_v) {
		cout << "design_object::compute_blocks_from_incidence_matrix done" << endl;
	}

}


void design_object::make_Baker_elliptic_semiplane_1978(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_object::make_Baker_elliptic_semiplane_1978" << endl;
	}

	design_theory_global Design_theory_global;

	if (f_v) {
		cout << "design_object::init "
				"before Design_theory_global.make_Baker_elliptic_semiplane_1978_incma" << endl;
	}
	Design_theory_global.make_Baker_elliptic_semiplane_1978_incma(
			incma, v, b, verbose_level - 1);
	if (f_v) {
		cout << "design_object::init "
				"after Design_theory_global.make_Baker_elliptic_semiplane_1978_incma" << endl;
	}


	k = 7;
	block = NEW_int(k);
	nb_inc = k * b;

	f_has_set = false;

	f_has_incma = true;

#if 0
	int *block; // [k]

	int v;
	int b;
	int nb_inc;
	int f_has_incma;
	int *incma; // [v * b]
#endif

	prefix = "Baker1978";
	label_txt = "Baker1978";
	label_tex = "Baker1978";

	if (f_v) {
		cout << "design_object::make_Baker_elliptic_semiplane_1978 done" << endl;
	}

}

void design_object::make_Mathon_elliptic_semiplane_1987(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_object::make_Mathon_elliptic_semiplane_1987" << endl;
	}

	combinatorics::design_theory::design_theory_global Design_theory_global;


	if (f_v) {
		cout << "design_object::make_Mathon_elliptic_semiplane_1987 "
				"before Design_theory_global.make_Mathon_elliptic_semiplane_1987_incma" << endl;
	}
	Design_theory_global.make_Mathon_elliptic_semiplane_1987_incma(
			incma, v, b, verbose_level - 1);
	if (f_v) {
		cout << "design_object::make_Mathon_elliptic_semiplane_1987 "
				"after Design_theory_global.make_Mathon_elliptic_semiplane_1987_incma" << endl;
	}


	k = 12;
	block = NEW_int(k);
	nb_inc = k * b;

	f_has_set = false;

	f_has_incma = true;

#if 0
	int *block; // [k]

	int v;
	int b;
	int nb_inc;
	int f_has_incma;
	int *incma; // [v * b]
#endif

	prefix = "Mathon1987";
	label_txt = "Mathon1987";
	label_tex = "Mathon1987";

	if (f_v) {
		cout << "design_object::make_Mathon_elliptic_semiplane_1987 done" << endl;
	}
}

void design_object::make_design_from_incidence_matrix(
		std::string &label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_object::make_design_from_incidence_matrix" << endl;
	}


	combinatorics::design_theory::design_theory_global Design_theory_global;


	if (f_v) {
		cout << "design_object::make_design_from_incidence_matrix "
				"before Design_theory_global.make_design_from_incidence_matrix" << endl;
	}
	Design_theory_global.make_design_from_incidence_matrix(
			incma, v, b, k,
			label,
			verbose_level - 1);
	if (f_v) {
		cout << "design_object::make_design_from_incidence_matrix "
				"after Design_theory_global.make_design_from_incidence_matrix" << endl;
	}

	block = NEW_int(k);
	nb_inc = k * b;

	f_has_set = false;

	f_has_incma = true;

#if 0
	int *block; // [k]

	int v;
	int b;
	int nb_inc;
	int f_has_incma;
	int *incma; // [v * b]
#endif

	prefix = label;
	label_txt = label;
	label_tex = label;

	if (f_v) {
		cout << "design_object::make_design_from_incidence_matrix done" << endl;
	}
}

void design_object::do_export_flags(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_object::do_export_flags" << endl;
	}

	string fname;

	fname = label_txt + "_flags.txt";

	if (f_v) {
		cout << "design_object::do_export_flags "
				"fname=" << fname << endl;
		cout << "design_object::do_export_flags "
				"v=" << v << endl;
		cout << "design_object::do_export_flags "
				"b=" << b << endl;
		cout << "design_object::do_export_flags "
				"nb_inc=" << nb_inc << endl;
	}


	{
		ofstream ost(fname);

		int h;
		int nb_inc1;

		nb_inc1 = 0;
		ost << v << " " << b << " " << nb_inc << endl;
		for (h = 0; h < v * b; h++) {
			if (incma[h]) {
				ost << h << " ";
				nb_inc1++;
			}
		}
		ost << endl;
		ost << "-1" << endl;

		if (nb_inc1 != nb_inc) {
			cout << "design_object::do_export_flags nb_inc1 != nb_inc" << endl;
			cout << "design_object::do_export_flags "
					"nb_inc=" << nb_inc << endl;
			cout << "design_object::do_export_flags "
					"nb_inc1=" << nb_inc1 << endl;
			exit(1);
		}
	}
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}

	//8 8 24
	//0 1 2 8 11 12 16 21 22 25 27 29 33 36 39 42 44 46 50 53 55 59 62 63
	//-1 1
	//48



	if (f_v) {
		cout << "design_object::do_export_flags done" << endl;
	}
}




void design_object::do_export_incidence_matrix_csv(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_object::do_export_incidence_matrix_csv" << endl;
	}

	string fname;

	fname = label_txt + "_incma.csv";

	if (f_v) {
		cout << "design_object::do_export_incidence_matrix_csv "
				"fname=" << fname << endl;
	}

	other::orbiter_kernel_system::file_io Fio;


	Fio.Csv_file_support->int_matrix_write_csv(
			fname, incma, v, b);
	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}



	if (f_v) {
		cout << "design_object::do_export_incidence_matrix_csv done" << endl;
	}
}

void design_object::do_export_incidence_matrix_latex(
		other::graphics::draw_incidence_structure_description *Draw_incidence_structure_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_object::do_export_incidence_matrix_latex" << endl;
	}

	string fname;

	fname = label_txt + "_incma.tex";

	if (f_v) {
		cout << "design_object::do_export_incidence_matrix_latex "
				"fname=" << fname << endl;
	}

	int nb_rows, nb_cols;

	nb_rows = v;
	nb_cols = b;

	combinatorics::canonical_form_classification::encoded_combinatorial_object *Enc;

	Enc = NEW_OBJECT(combinatorics::canonical_form_classification::encoded_combinatorial_object);
	Enc->init(nb_rows, nb_cols, verbose_level);

	int i, j, f;

	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < b; j++) {
			f = i * b + j;
			if (incma[f]) {
				Enc->set_incidence(f);
			}
		}
	}

	Enc->partition[nb_rows - 1] = 0;
	Enc->partition[nb_rows + nb_cols - 1] = 0;


	if (f_v) {
		cout << "design_object::do_export_incidence_matrix_latex "
				"partition:" << endl;
		Enc->print_partition();
	}

	other::orbiter_kernel_system::file_io Fio;


	{
		ofstream ost(fname);
		other::l1_interfaces::latex_interface L;

		L.head_easy(ost);


		Enc->latex_incma(
				ost,
				Draw_incidence_structure_description,
				verbose_level);

		ost << endl;

		ost << "\\bigskip" << endl;

		ost << endl;


		Enc->latex_incma_as_01_matrix(
				ost,
				verbose_level);

		ost << endl;

		ost << "\\bigskip" << endl;

		ost << endl;




		ost << "\\noindent Blocks: \\\\" << endl;

		Enc->latex_set_system_by_columns(
				ost,
				verbose_level);

		L.foot(ost);
	}



	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}



	if (f_v) {
		cout << "design_object::do_export_incidence_matrix_latex done" << endl;
	}
}



void design_object::do_intersection_matrix(
		int f_save,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_object::do_intersection_matrix" << endl;
	}


	if (!f_has_incma) {
		cout << "design_object::do_intersection_matrix "
				"the incidence matrix of the design is not available" << endl;
		exit(1);
	}

	int *AAt;
	int i, j, h, cnt;

	AAt = NEW_int(v * v);
	for (i = 0; i < v; i++) {
		for (j = 0; j < v; j++) {
			cnt = 0;
			for (h = 0; h < b; h++) {
				if (incma[i * b + h] && incma[j * b + h]) {
					cnt++;
				}
			}
			AAt[i * v + j] = cnt;
		}

	}

	algebra::basic_algebra::algebra_global Algebra;
	int coeff_I, coeff_J;

	if (Algebra.is_lc_of_I_and_J(
			AAt, v, coeff_I, coeff_J, 0 /* verbose_level*/)) {
		cout << "Is a linear combination of I and J with coefficients "
				"coeff(I)=" << coeff_I << " and coeff(J-I)=" << coeff_J << endl;
	}
	else {
		cout << "Is *not* a linear combination of I and J" << endl;

	}

	if (f_save) {

		other::orbiter_kernel_system::file_io Fio;
		string fname;

		fname = label_txt + "_AAt.csv";

		if (f_v) {
			cout << "design_object::do_intersection_matrix "
					"fname=" << fname << endl;
		}

		{
			ofstream ost(fname);

			Fio.Csv_file_support->int_matrix_write_csv(
					fname, AAt, v, v);

		}

		if (f_v) {
			cout << "Written file " << fname
					<< " of size " << Fio.file_size(fname) << endl;
		}
	}



	if (f_v) {
		cout << "design_object::do_intersection_matrix done" << endl;
	}
}


void design_object::do_export_blocks(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_object::do_export_blocks" << endl;
	}

	string fname;

	fname = label_txt + "_blocks.csv";

	//combinatorics::other_combinatorics::combinatorics_domain Combi;
	combinatorics::design_theory::design_theory_global Design;


	other::orbiter_kernel_system::file_io Fio;


	#if 0

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, DC->set, 1, b);

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
#endif


	fname = label_txt + "_blocks.csv";

	int *Blocks;


	if (f_has_set) {
		if (f_v) {
			cout << "design_object::do_export_blocks "
					"before Design.compute_blocks_from_coding" << endl;
		}
		Design.compute_blocks_from_coding(
				v, b, k, set, Blocks, verbose_level);
		if (f_v) {
			cout << "design_object::do_export_blocks "
					"after Design.compute_blocks_from_coding" << endl;
		}
	}
	else if (f_has_incma) {
		if (f_v) {
			cout << "design_object::do_export_blocks "
					"before Design.compute_blocks_from_incma" << endl;
		}
		Design.compute_blocks_from_incma(
				v, b, k, incma,
					Blocks, verbose_level);
		if (f_v) {
			cout << "design_object::do_export_blocks "
					"after Design.compute_blocks_from_incma" << endl;
		}
	}
	else {
		cout << "design_object::do_export_blocks "
				"we neither have a set nor an incma" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "design_object::do_export_blocks "
				"b = " << b << endl;
		cout << "design_object::do_export_blocks "
				"k = " << k << endl;
	}

#if 0
	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Blocks, b, k);
#endif

	string *Table;
	string headings;
	int nb_rows = b;
	int nb_cols = 2;
	int i;

	Table = new string [nb_rows * nb_cols];
	for (i = 0; i < nb_rows; i++) {
		Table[i * nb_cols + 0] = std::to_string(i);
		Table[i * nb_cols + 1] = "\"" + Int_vec_stringify(Blocks + i * k, k) + "\"";
	}
	headings = "row,block";



	Fio.Csv_file_support->write_table_of_strings(
			fname,
				nb_rows, nb_cols, Table,
				headings,
				verbose_level);

	if (f_v) {
		cout << "design_object::do_export_blocks Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	delete [] Table;

	FREE_int(Blocks);


	if (f_v) {
		cout << "design_object::do_export_blocks done" << endl;
	}
}

void design_object::do_row_sums(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_object::do_row_sums" << endl;
	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;


	int i, j;
	int *R;

	R = NEW_int(v);

	for (i = 0; i < v; i++) {
		R[i] = 0;
		for (j = 0; j < b; j++) {
			if (incma[i * b + j]) {
				R[i]++;
			}
		}
	}

	other::data_structures::tally T;

	T.init(R, v, false, 0);
	if (f_v) {
		cout << "distribution of row sums: ";
		T.print(true /* f_backwards */);
		cout << endl;
	}

	FREE_int(R);



	if (f_v) {
		cout << "design_object::do_row_sums done" << endl;
	}
}

void design_object::do_tactical_decomposition(
		int verbose_level)
// Computes the TDO and prints it in latex to cout
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_object::do_tactical_decomposition" << endl;
	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;



	{
		geometry::other_geometry::incidence_structure *Inc;


		Inc = NEW_OBJECT(geometry::other_geometry::incidence_structure);

		Inc->init_by_matrix(
				v, b, incma,
				0 /* verbose_level */);

		combinatorics::tactical_decompositions::decomposition *Decomposition;


		Decomposition = NEW_OBJECT(combinatorics::tactical_decompositions::decomposition);

		Decomposition->init_incidence_structure(
				Inc,
				verbose_level);

#if 0
		data_structures::partitionstack *Stack;

		Stack = NEW_OBJECT(data_structures::partitionstack);

		Stack->allocate_with_two_classes(
				DC->v + DC->b, DC->v, DC->b,
				0 /* verbose_level */);
#endif


		while (true) {

			int ht0, ht1;

			ht0 = Decomposition->Stack->ht;

			if (f_v) {
				cout << "design_object::do_tactical_decomposition "
						"before refine_column_partition_safe" << endl;
			}
			Decomposition->refine_column_partition_safe(verbose_level - 2);
			if (f_v) {
				cout << "design_object::do_tactical_decomposition "
						"after refine_column_partition_safe" << endl;
			}
			if (f_v) {
				cout << "design_object::do_tactical_decomposition "
						"before refine_row_partition_safe" << endl;
			}
			Decomposition->refine_row_partition_safe(verbose_level - 2);
			if (f_v) {
				cout << "design_object::do_tactical_decomposition "
						"after refine_row_partition_safe" << endl;
			}
			ht1 = Decomposition->Stack->ht;
			if (ht1 == ht0) {
				break;
			}
		}

		int f_labeled = true;

		Decomposition->print_partitioned(cout, f_labeled);
		Decomposition->get_and_print_decomposition_schemes();
		Decomposition->Stack->print_classes(cout);


		int f_print_subscripts = false;
		cout << "Decomposition:\\\\" << endl;
		cout << "Row scheme:\\\\" << endl;
		Decomposition->get_and_print_row_tactical_decomposition_scheme_tex(
				cout, true /* f_enter_math */,
			f_print_subscripts);
		cout << "Column scheme:\\\\" << endl;
		Decomposition->get_and_print_column_tactical_decomposition_scheme_tex(
				cout, true /* f_enter_math */,
			f_print_subscripts);

		other::data_structures::set_of_sets *Row_classes;
		other::data_structures::set_of_sets *Col_classes;

		Decomposition->Stack->get_row_classes(Row_classes, verbose_level);
		cout << "Row classes:\\\\" << endl;
		Row_classes->print_table_tex(cout);


		Decomposition->Stack->get_column_classes(Col_classes, verbose_level);
		cout << "Col classes:\\\\" << endl;
		Col_classes->print_table_tex(cout);

#if 0
		if (Row_classes->nb_sets > 1) {
			cout << "The row partition splits" << endl;
		}

		if (Col_classes->nb_sets > 1) {
			cout << "The col partition splits" << endl;
		}
#endif


		FREE_OBJECT(Inc);
		FREE_OBJECT(Decomposition);
		FREE_OBJECT(Row_classes);
		FREE_OBJECT(Col_classes);
	}

	if (f_v) {
		cout << "design_object::do_tactical_decomposition" << endl;
	}

}


}}}}

