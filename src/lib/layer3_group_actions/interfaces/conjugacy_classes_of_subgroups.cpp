/*
 * conjugacy_classes_of_subgroups.cpp
 *
 *  Created on: Sep 24, 2024
 *      Author: betten
 */







#include "layer1_foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace interfaces {


conjugacy_classes_of_subgroups::conjugacy_classes_of_subgroups()
{
	A = NULL;
	//std::string fname;
	nb_classes = 0;
	Subgroup_order = NULL;
	Length = NULL;
	Nb_gens = NULL;
	subgroup_gens = NULL;
	Conjugacy_class = NULL;

}

conjugacy_classes_of_subgroups::~conjugacy_classes_of_subgroups()
{

	if (Subgroup_order) {
		FREE_int(Subgroup_order);
	}
	if (Length) {
		FREE_int(Length);
	}
	if (Nb_gens) {
		FREE_int(Nb_gens);
	}
	if (subgroup_gens) {
		int i;
		for (i = 0; i < nb_classes; i++) {
			FREE_int(subgroup_gens[i]);
		}

		FREE_pint(subgroup_gens);
	}
	if (Conjugacy_class) {
		int i;

		for (i = 0; i < nb_classes; i++) {
			FREE_OBJECT(Conjugacy_class[i]);
		}

		FREE_pvoid((void **) Conjugacy_class);

	}

}

void conjugacy_classes_of_subgroups::read_magma_output_file(
		actions::action *A,
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::read_magma_output_file" << endl;
		cout << "conjugacy_classes_of_subgroups::read_magma_output_file "
				"fname=" << fname << endl;
		cout << "conjugacy_classes_of_subgroups::read_magma_output_file "
				"degree=" << A->degree << endl;
	}

	int i, j, h;

	conjugacy_classes_of_subgroups::A = A;
	conjugacy_classes_of_subgroups::fname.assign(fname);

	{
		ifstream fp(fname);

		fp >> nb_classes;
		if (f_v) {
			cout << "conjugacy_classes_of_subgroups::read_magma_output_file "
					"We found " << nb_classes
					<< " conjugacy classes" << endl;
		}

		Subgroup_order = NEW_int(nb_classes);
		Length = NEW_int(nb_classes);
		Nb_gens = NEW_int(nb_classes);
		subgroup_gens = NEW_pint(nb_classes);

		int a;
		long int total_length = 0;

		for (h = 0; h < nb_classes; h++) {
			fp >> a;
			if (a - 1 != h) {
				cout << "conjugacy_classes_of_subgroups::read_magma_output_file file is corrupt" << endl;
				exit(1);
			}
			fp >> Subgroup_order[h];
			fp >> Length[h];
			fp >> Nb_gens[h];

			subgroup_gens[h] = NEW_int(Nb_gens[h] * A->degree);
			for (i = 0; i < Nb_gens[h]; i++) {
				for (j = 0; j < A->degree; j++) {
					fp >> a;
					a--;
					subgroup_gens[h][i * A->degree + j] = a;
				}
			}
			total_length += Length[h];
		}
		fp >> a;
		if (a != -1) {
			cout << "conjugacy_classes_of_subgroups::read_magma_output_file "
					"file is corrupt (EOF marker is missing)" << endl;
			exit(1);
		}



		if (f_v) {
			cout << "conjugacy_classes_of_subgroups::read_magma_output_file "
					"we read all class representatives "
					"from file " << fname << endl;
		}
		if (f_v) {
			cout << "conjugacy_classes_of_subgroups::read_magma_output_file "
					"total_length = " << total_length << endl;
		}
	}
	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::read_magma_output_file done" << endl;
	}
}

void conjugacy_classes_of_subgroups::create_classes(
		groups::sims *group_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::create_classes" << endl;
	}

	Conjugacy_class = (groups::conjugacy_class_of_subgroups **) NEW_pvoid(nb_classes);

	int idx;

	for (idx = 0; idx < nb_classes; idx++) {
		Conjugacy_class[idx] = NEW_OBJECT(groups::conjugacy_class_of_subgroups);

		Conjugacy_class[idx]->init(
				this,
				idx,
				group_G,
				verbose_level - 1);

	}

	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::create_classes" << endl;
	}
}

void conjugacy_classes_of_subgroups::report(
		groups::sims *override_sims,
		std::string &label_latex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::report" << endl;
	}

	int i;
	orbiter_kernel_system::file_io Fio;

	cout << "i : class_order_of_element : class_normalizer_order" << endl;
	for (i = 0; i < nb_classes; i++) {
		cout << i << " : " << endl;
	}



	data_structures::string_tools ST;
	ring_theory::longinteger_object go;

	override_sims->group_order(go);
	cout << "The group has order " << go << endl;

	string fname_latex;

	fname_latex = fname;

	ST.replace_extension_with(fname_latex, ".tex");


	{
		ofstream ost(fname_latex);
		string title, author, extra_praeamble;
		l1_interfaces::latex_interface L;

		title = "Conjugacy classes subgroups of $" + label_latex + "$";

		author = "computed by Orbiter and MAGMA";

		L.head(ost,
			false /* f_book */, true /* f_title */,
			title, author /* const char *author */,
			false /* f_toc */, false /* f_landscape */, true /* f_12pt */,
			true /* f_enlarged_page */, true /* f_pagenumbers */,
			extra_praeamble /* extra_praeamble */);
		//latex_head_easy(fp);

		ost << "\\section{Conjugacy classes in $" << label_latex << "$}" << endl;


		ost << "The group order is " << endl;
		ost << "$$" << endl;
		go.print_not_scientific(ost);
		ost << endl;
		ost << "$$" << endl;

		cout << "second time" << endl;

		cout << "i : class_subgroup_order : class_length" << endl;
		for (i = 0; i < nb_classes; i++) {
			cout << i << " : " << Subgroup_order[i] << " : " << Length[i] << endl;
		}


		if (f_v) {
			cout << "conjugacy_classes_of_subgroups::report "
					"before report_classes" << endl;
		}
		report_classes(ost, verbose_level - 1);
		if (f_v) {
			cout << "conjugacy_classes_of_subgroups::report "
					"after report_classes" << endl;
		}



		L.foot(ost);
	}
	cout << "Written file " << fname_latex << " of size "
			<< Fio.file_size(fname_latex) << endl;



	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::report done" << endl;
	}
}

void conjugacy_classes_of_subgroups::export_csv(
		groups::sims *override_sims,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::export_csv" << endl;
	}

	data_structures::string_tools ST;
	string fname_csv;

	fname_csv.assign(fname);


	ST.replace_extension_with(fname_csv, ".csv");
	{
		ofstream fp(fname_csv);

		int idx;

		fp << "row,class_order_of_subgroup,class_size" << endl;
		for (idx = 0; idx < nb_classes; idx++) {
			fp << idx << "," << Subgroup_order[idx] << "," << Length[idx] << endl;
		}
		fp << "END" << endl;

	}

	string fname_data;

	fname_data = fname;

	ST.replace_extension_with(fname_data, "_data.csv");


	{
		ofstream fp(fname_data);

		int idx;
		int *data;

		data = NEW_int(A->make_element_size);

		fp << "subgroup_order,length" << endl;
		for (idx = 0; idx < nb_classes; idx++) {


			if (f_v) {
				cout << "conjugacy_classes_of_subgroups::export_csv idx = " << idx << " / " << nb_classes << endl;
			}


			fp << idx << "," << Subgroup_order[idx] << "," << Length[idx] << endl;


		}
		fp << "END" << endl;

		FREE_int(data);
	}

	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::export_csv done" << endl;
	}
}

void conjugacy_classes_of_subgroups::report_classes(
		std::ofstream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::report_classes" << endl;
	}
	int idx;

	cout << "The conjugacy classes are:" << endl;
	for (idx = 0; idx < nb_classes; idx++) {


		Conjugacy_class[idx]->report_single_class(ost, verbose_level - 1);

	}


	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::report_classes "
				"before export_csv" << endl;
	}
	export_csv(verbose_level);
	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::report_classes "
				"after export_csv" << endl;
	}

	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::report_classes done" << endl;
	}
}

void conjugacy_classes_of_subgroups::export_csv(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::export_csv" << endl;
	}

	int idx;
	int nb_cols = 4;
	string *Table;

	cout << "The conjugacy class representatives are:" << endl;

	Table = new string[nb_classes * nb_cols];

	for (idx = 0; idx < nb_classes; idx++) {

		Table[idx * nb_cols + 0] = std::to_string(Subgroup_order[idx]);
		Table[idx * nb_cols + 1] = std::to_string(Length[idx]);
		Table[idx * nb_cols + 2] = std::to_string(Conjugacy_class[idx]->gens->gens->len);
		Table[idx * nb_cols + 3] = "\"" + Conjugacy_class[idx]->gens->stringify_gens_data(0 /*verbose_level*/) + "\"";
	}



	orbiter_kernel_system::file_io Fio;


	string fname_csv;
	string headings;

	headings.assign("Go,Length,nb_gens,Generators");

	fname_csv = fname;

	data_structures::string_tools ST;
	ST.replace_extension_with(fname_csv, "_classes.csv");

	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::export_csv "
				"before Fio.Csv_file_support->write_table_of_strings" << endl;
	}
	Fio.Csv_file_support->write_table_of_strings(fname_csv,
			nb_classes, nb_cols, Table,
			headings,
			verbose_level);
	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::export_csv "
				"after Fio.Csv_file_support->write_table_of_strings" << endl;
	}

	delete [] Table;

	if (f_v) {
		cout << "conjugacy_classes_of_subgroups::export_csv" << endl;
	}
}


}}}

