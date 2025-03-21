/*
 * conjugacy_classes_and_normalizers.cpp
 *
 *  Created on: Nov 2, 2023
 *      Author: betten
 */






#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace interfaces {


conjugacy_classes_and_normalizers::conjugacy_classes_and_normalizers()
{
	Record_birth();
	A = NULL;
	//std::string fname;
	nb_classes = 0;
	perms = NULL;
	class_size = NULL;
	class_order_of_element = NULL;
	class_normalizer_order = NULL;
	class_normalizer_number_of_generators = NULL;
	normalizer_generators_perms = NULL;

	Conjugacy_class = NULL;

}

conjugacy_classes_and_normalizers::~conjugacy_classes_and_normalizers()
{
	Record_death();
	if (perms) {
		FREE_int(perms);
	}
	if (class_size) {
		FREE_lint(class_size);
	}
	if (class_order_of_element) {
		FREE_int(class_order_of_element);
	}
	if (class_normalizer_order) {
		FREE_lint(class_normalizer_order);
	}
	if (class_normalizer_number_of_generators) {
		FREE_int(class_normalizer_number_of_generators);
	}
	if (normalizer_generators_perms) {
		int i;

		for (i = 0; i < nb_classes; i++) {
			FREE_int(normalizer_generators_perms[i]);
		}

		FREE_pint(normalizer_generators_perms);
	}
	if (Conjugacy_class) {
		int i;

		for (i = 0; i < nb_classes; i++) {
			FREE_OBJECT(Conjugacy_class[i]);
		}

		FREE_pvoid((void **) Conjugacy_class);

	}

}

void conjugacy_classes_and_normalizers::read_magma_output_file(
		actions::action *A,
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::read_magma_output_file" << endl;
		cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
				"fname=" << fname << endl;
		cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
				"degree=" << A->degree << endl;
	}

	int i, j, h;

	conjugacy_classes_and_normalizers::A = A;
	conjugacy_classes_and_normalizers::fname.assign(fname);

	{
		ifstream fp(fname);

		fp >> nb_classes;
		if (f_v) {
			cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
					"We found " << nb_classes
					<< " conjugacy classes" << endl;
		}

		perms = NEW_int(nb_classes * A->degree);
		class_size = NEW_lint(nb_classes);
		class_order_of_element = NEW_int(nb_classes);

		for (i = 0; i < nb_classes; i++) {
			fp >> class_order_of_element[i];
			if (f_v) {
				cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
						"class " << i << " / " << nb_classes
						<< " order=" << class_order_of_element[i] << endl;
			}
			fp >> class_size[i];
			if (f_v) {
				cout << "class_size[i] = " << class_size[i] << endl;
			}
			for (j = 0; j < A->degree; j++) {
				fp >> perms[i * A->degree + j];
			}
		}
		if (false) {
			cout << "perms:" << endl;
			Int_matrix_print(perms, nb_classes, A->degree);
		}
		for (i = 0; i < nb_classes * A->degree; i++) {
			perms[i]--;
		}

		class_normalizer_order = NEW_lint(nb_classes);
		class_normalizer_number_of_generators = NEW_int(nb_classes);
		normalizer_generators_perms = NEW_pint(nb_classes);

		if (f_v) {
			cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
					"reading normalizer generators:" << endl;
		}
		for (i = 0; i < nb_classes; i++) {
			if (f_v) {
				cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
						"class " << i << " / " << nb_classes << endl;
			}
			fp >> class_normalizer_order[i];

			cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
					"class " << i << " class_normalizer_order[i]=" << class_normalizer_order[i] << endl;

			if (class_normalizer_order[i] <= 0) {
				cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
						"class_normalizer_order[i] <= 0" << endl;
				cout << "class_normalizer_order[i]=" << class_normalizer_order[i] << endl;
				exit(1);
			}
			if (f_v) {
				cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
						"class " << i << " / " << nb_classes
						<< " class_normalizer_order[i]=" << class_normalizer_order[i] << endl;
			}
			fp >> class_normalizer_number_of_generators[i];
			normalizer_generators_perms[i] =
					NEW_int(class_normalizer_number_of_generators[i] * A->degree);
			for (h = 0; h < class_normalizer_number_of_generators[i]; h++) {
				for (j = 0; j < A->degree; j++) {
					fp >> normalizer_generators_perms[i][h * A->degree + j];
				}
			}
			for (h = 0; h < class_normalizer_number_of_generators[i] * A->degree; h++) {
				normalizer_generators_perms[i][h]--;
			}
		}
		if (f_v) {
			cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
					"we read all class representatives "
					"from file " << fname << endl;
		}
	}
	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::read_magma_output_file done" << endl;
	}
}

void conjugacy_classes_and_normalizers::create_classes(
		groups::sims *group_G,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::create_classes" << endl;
	}

	Conjugacy_class = (groups::conjugacy_class_of_elements **) NEW_pvoid(nb_classes);

	int idx;

	for (idx = 0; idx < nb_classes; idx++) {
		Conjugacy_class[idx] = NEW_OBJECT(groups::conjugacy_class_of_elements);

		Conjugacy_class[idx]->init(
				this,
				idx,
				group_G,
				verbose_level - 1);

	}

	Conjugacy_class[0]->sub_idx = 0;

	for (idx = 1; idx < nb_classes; idx++) {
		if (idx == 1) {
			Conjugacy_class[idx]->sub_idx = 0;
		}
		else {
			if (Conjugacy_class[idx]->goi == Conjugacy_class[idx - 1]->goi) {
				Conjugacy_class[idx]->sub_idx = Conjugacy_class[idx - 1]->sub_idx + 1;
			}
			else {
				Conjugacy_class[idx]->sub_idx = 0;
			}
		}
	}
	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::create_classes" << endl;
	}
}

void conjugacy_classes_and_normalizers::report(
		groups::sims *override_sims,
		std::string &label,
		std::string &label_latex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::report" << endl;
	}

	int i;
	other::orbiter_kernel_system::file_io Fio;

	cout << "i : class_order_of_element : class_normalizer_order" << endl;
	for (i = 0; i < nb_classes; i++) {
		cout << i << " : " << class_order_of_element[i] << " : " << class_normalizer_order[i] << endl;
	}



	//other::data_structures::string_tools ST;
	algebra::ring_theory::longinteger_object go;

	override_sims->group_order(go);
	cout << "The group has order " << go << endl;

	string fname_report;

	fname_report = label + "_classes.tex";



	{
		ofstream ost(fname_report);
		string title, author, extra_praeamble;
		other::l1_interfaces::latex_interface L;

		title = "Conjugacy classes of $" + label_latex + "$";

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

		cout << "i : class_order_of_element : class_normalizer_order" << endl;
		for (i = 0; i < nb_classes; i++) {
			cout << i << " : " << class_order_of_element[i]
				<< " : " << class_normalizer_order[i] << endl;
		}


		if (f_v) {
			cout << "conjugacy_classes_and_normalizers::report "
					"before report_classes" << endl;
		}
		report_classes(ost, verbose_level - 1);
		if (f_v) {
			cout << "conjugacy_classes_and_normalizers::report "
					"after report_classes" << endl;
		}



		L.foot(ost);
	}
	cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;



	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::report done" << endl;
	}
}

void conjugacy_classes_and_normalizers::export_csv(
		groups::sims *override_sims,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::export_csv" << endl;
	}

	other::data_structures::string_tools ST;
	string fname_csv;

	fname_csv.assign(fname);


	ST.replace_extension_with(fname_csv, ".csv");
	{
		ofstream fp(fname_csv);

		int idx;

		fp << "ROW,class_order_of_element,class_size" << endl;
		for (idx = 0; idx < nb_classes; idx++) {
			fp << idx << "," << class_order_of_element[idx] << "," << class_size[idx] << endl;
		}
		fp << "END" << endl;

	}

	string fname_data;

	fname_data.assign(fname);

	ST.replace_extension_with(fname_data, "_data.csv");


	{
		ofstream fp(fname_data);

		int idx;
		int *data;

		data = NEW_int(A->make_element_size);

		fp << "ROW,class_order_of_element,class_size,classrep" << endl;
		for (idx = 0; idx < nb_classes; idx++) {


			if (f_v) {
				cout << "conjugacy_classes_and_normalizers::export_csv "
						"idx = " << idx << " / " << nb_classes << endl;
			}


			fp << idx << "," << class_order_of_element[idx] << "," << class_size[idx];


			long int goi, ngo;

			goi = class_order_of_element[idx];
			ngo = class_normalizer_order[idx];



			cout << "goi=" << goi << endl;
			cout << "ngo=" << ngo << endl;


			groups::strong_generators *gens;
			data_structures_groups::vector_ge *nice_gens;

			gens = NEW_OBJECT(groups::strong_generators);


			// create strong generators for the cyclic group generated by the i-th class rep
			// nice_gens will contain the single generator only.
			if (f_v) {
				cout << "conjugacy_classes_and_normalizers::export_csv computing H, "
					"before gens->init_from_permutation_representation" << endl;
			}
			gens->init_from_permutation_representation(
					A, override_sims,
					perms + idx * A->degree,
				1, goi, nice_gens,
				verbose_level - 5);

			if (f_v) {
				cout << "conjugacy_classes_and_normalizers::export_csv computing H, "
					"after gens->init_from_permutation_representation" << endl;
			}

			int *Elt;

			Elt = nice_gens->ith(0);
			A->Group_element->code_for_make_element(
						data, Elt);
			fp << ",\"";
			Int_vec_print_bare_fully(fp, data, A->make_element_size);
			fp << "\"";

			FREE_OBJECT(gens);

			fp << endl;
		}
		fp << "END" << endl;

		FREE_int(data);
	}

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::export_csv done" << endl;
	}
}


void conjugacy_classes_and_normalizers::get_representatives(
		groups::sims *override_sims,
		data_structures_groups::vector_ge *&Reps,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::get_representatives" << endl;
	}



	Reps = NEW_OBJECT(data_structures_groups::vector_ge);

	Reps->init(A, 0 /* verbose_level */);
	Reps->allocate(nb_classes, 0 /* verbose_level */);


	int idx;

	for (idx = 0; idx < nb_classes; idx++) {


		if (f_v) {
			cout << "conjugacy_classes_and_normalizers::get_representatives "
					"idx = " << idx << " / " << nb_classes << endl;
		}


		long int goi; //, ngo;

		goi = class_order_of_element[idx];
		//ngo = class_normalizer_order[idx];




		groups::strong_generators *gens;
		data_structures_groups::vector_ge *nice_gens;

		gens = NEW_OBJECT(groups::strong_generators);


		// create strong generators for the cyclic group generated by the i-th class rep
		// nice_gens will contain the single generator only.
		if (f_v) {
			cout << "conjugacy_classes_and_normalizers::get_representatives "
				"before gens->init_from_permutation_representation" << endl;
		}
		gens->init_from_permutation_representation(
				A, override_sims,
				perms + idx * A->degree,
			1, goi, nice_gens,
			verbose_level - 5);

		if (f_v) {
			cout << "conjugacy_classes_and_normalizers::get_representatives "
				"after gens->init_from_permutation_representation" << endl;
		}

		int *Elt;

		Elt = nice_gens->ith(0);
		A->Group_element->element_move(
				Elt, Reps->ith(idx), 0 /* verbose_level */);

		FREE_OBJECT(gens);
		FREE_OBJECT(nice_gens);

	}


	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::get_representatives done" << endl;
	}
}



void conjugacy_classes_and_normalizers::report_classes(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::report_classes" << endl;
	}


	report_summary_table(ost, verbose_level);

	int idx;

	for (idx = 0; idx < nb_classes; idx++) {


		Conjugacy_class[idx]->report_single_class(ost, verbose_level - 1);

	}


	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::report_classes "
				"before export_csv" << endl;
	}
	export_csv(verbose_level);
	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::report_classes "
				"after export_csv" << endl;
	}

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::report_classes done" << endl;
	}
}

void conjugacy_classes_and_normalizers::report_summary_table(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::report_summary_table" << endl;
	}


	cout << "The conjugacy classes are:" << endl;

	int idx;

	std::vector< std::vector<std::string > > Data;

	for (idx = 0; idx < nb_classes; idx++) {

		std::vector<std::string > data;

		Conjugacy_class[idx]->single_class_data(data, verbose_level - 1);

		Data.push_back(data);
	}

	std::string *Table;
	int nb_rows, nb_cols;
	int i, j;

	nb_cols = Data[0].size();
	nb_rows = nb_classes;
	Table = new string [nb_rows * nb_cols];
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Table[i * nb_cols + j] = Data[i][j];
		}
	}

	ost << endl;
	ost << "\\bigskip" << endl;
	ost << endl;

	ost << "The following table shows the conjugacy classes of the group. The columns display: "
			"the number of the class, "
			"the order of elements in the class, "
			"an ATLAS style label for the class, "
			"the size of the class, "
			"the order of the centralizer of an element of the class, "
			"the order of the normalizer of the group generated by an element of the class.\\\\" << endl;

	ost << endl;
	ost << "\\bigskip" << endl;
	ost << endl;


	other::l1_interfaces::latex_interface L;





	{
		int nb_pp = 20;
		int nb_p, nb_r;
		int I;

		nb_p = (nb_classes + nb_pp - 1)/ nb_pp;
		for (I = 0; I < nb_p; I++) {

			if (I == nb_p - 1) {
				nb_r = nb_rows - nb_pp * I;
			}
			else {
				nb_r = nb_pp;
			}
			ost << "\\begin{center}" << endl;
			L.print_tabular_of_strings(
					ost,
					Table + I * nb_pp * nb_cols, nb_r, nb_cols);
			ost << "\\end{center}" << endl;


		}

		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;
	}

}

void conjugacy_classes_and_normalizers::export_csv(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::export_csv" << endl;
	}

	int idx;
	int nb_cols = 1;
	string *Table;

	cout << "The conjugacy class representatives are:" << endl;

	Table = new string[nb_classes * nb_cols];

	for (idx = 0; idx < nb_classes; idx++) {

		Table[idx * nb_cols + 0] =
				Conjugacy_class[idx]->conjugacy_class_of_elements::stringify_representative_coded(
				verbose_level - 1);

	}



	other::orbiter_kernel_system::file_io Fio;


	string fname_csv;
	string headings;

	headings.assign("Rep");

	fname_csv = fname + "_classes.csv";

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::export_csv "
				"before Fio.Csv_file_support->write_table_of_strings" << endl;
	}
	Fio.Csv_file_support->write_table_of_strings(fname_csv,
			nb_classes, nb_cols, Table,
			headings,
			verbose_level);
	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::export_csv "
				"after Fio.Csv_file_support->write_table_of_strings" << endl;
	}

	delete [] Table;

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::export_csv" << endl;
	}
}


}}}


