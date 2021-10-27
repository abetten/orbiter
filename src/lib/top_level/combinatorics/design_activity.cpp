/*
 * design_activity.cpp
 *
 *  Created on: May 26, 2021
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


design_activity::design_activity()
{
	Descr = NULL;

}

design_activity::~design_activity()
{

}

void design_activity::perform_activity(design_activity_description *Descr,
		design_create *DC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::perform_activity" << endl;
	}

	design_activity::Descr = Descr;

	if (Descr->f_create_table) {
		do_create_table(
				DC,
				Descr->create_table_label,
				Descr->create_table_group_order,
				Descr->create_table_gens,
				verbose_level);
	}
	else if (Descr->f_load_table) {
		do_load_table(
				DC,
				Descr->create_table_label,
				Descr->create_table_group_order,
				Descr->create_table_gens,
				Descr->load_table_H_label,
				Descr->load_table_H_group_order,
				Descr->load_table_H_gens,
				Descr->load_table_selected_orbit_length,
				verbose_level);
	}
	else if (Descr->f_canonical_form) {
		do_canonical_form(Descr->Canonical_form_Descr,
				verbose_level);
	}
	else if (Descr->f_extract_solutions_by_index) {

		if (f_v) {
			cout << "design_activity::perform_activity f_extract_solutions_by_index" << endl;
		}

		do_extract_solutions_by_index(
				DC,
				Descr->create_table_label,
				Descr->create_table_group_order,
				Descr->create_table_gens,
				Descr->extract_solutions_by_index_fname_solutions_in,
				Descr->extract_solutions_by_index_fname_solutions_out,
				verbose_level);

		if (f_v) {
			cout << "design_activity::perform_activity f_extract_solutions_by_index done" << endl;
		}
	}
	else if (Descr->f_export_inc) {
		if (f_v) {
			cout << "design_activity::perform_activity export_inc" << endl;
		}
		do_export_inc(
				DC,
				verbose_level);
	}
	else if (Descr->f_export_blocks) {
		if (f_v) {
			cout << "design_activity::perform_activity export_blocks" << endl;
		}
		do_export_blocks(
				DC,
				verbose_level);
	}
	else if (Descr->f_row_sums) {
		if (f_v) {
			cout << "design_activity::perform_activity row_sums" << endl;
		}
		do_row_sums(
				DC,
				verbose_level);
	}
	else if (Descr->f_tactical_decomposition) {
		if (f_v) {
			cout << "design_activity::perform_activity f_tactical_decomposition" << endl;
		}
		do_tactical_decomposition(
				DC,
				verbose_level);
	}


	if (f_v) {
		cout << "design_activity::perform_activity done" << endl;
	}

}


void design_activity::do_extract_solutions_by_index(
		design_create *DC,
		std::string &label,
		std::string &go_text,
		std::string &generators_data,
		std::string &fname_in,
		std::string &fname_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index" << endl;
	}

	combinatorics_global Combi;
	design_tables *T;


	strong_generators *Gens;
	Gens = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index before Gens->init_from_data_with_go" << endl;
	}
	Gens->init_from_data_with_go(
			DC->A, generators_data,
			go_text,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index after Gens->init_from_data_with_go" << endl;
	}


	Combi.load_design_table(DC,
			label,
			T,
			Gens,
			verbose_level);

	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index after Combi.load_design_table" << endl;
	}

	file_io Fio;
	int *Sol_idx;
	int nb_sol;
	int sol_width;

	Fio.int_matrix_read_csv(fname_in, Sol_idx, nb_sol, sol_width, verbose_level);


	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index before T->extract_solutions_by_index" << endl;
	}

	T->extract_solutions_by_index(
			nb_sol, sol_width, Sol_idx,
			fname_out,
			verbose_level);

	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index after T->extract_solutions_by_index" << endl;
	}



	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index done" << endl;
	}
}



void design_activity::do_create_table(
		design_create *DC,
		std::string &label,
		std::string &go_text,
		std::string &generators_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_create_table" << endl;
	}

	combinatorics_global Combi;
	design_tables *T;


	strong_generators *Gens;
	Gens = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "design_activity::do_load_table before Gens->init_from_data_with_go" << endl;
	}
	Gens->init_from_data_with_go(
			DC->A, generators_data,
			go_text,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_load_table after Gens->init_from_data_with_go" << endl;
	}


	if (f_v) {
		cout << "design_activity::do_create_table before Combi.create_design_table" << endl;
	}

	Combi.create_design_table(DC,
			label,
			T,
			Gens,
			verbose_level);

	if (f_v) {
		cout << "design_activity::do_create_table after Combi.create_design_table" << endl;
	}

	if (f_v) {
		cout << "design_activity::do_create_table done" << endl;
	}
}


void design_activity::do_load_table(
		design_create *DC,
		std::string &label,
		std::string &go_text,
		std::string &generators_data,
		std::string &H_label,
		std::string &H_go_text,
		std::string &H_generators_data,
		int selected_orbit_length,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_load_table" << endl;
	}




	strong_generators *Gens;

	Gens = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "design_activity::do_load_table before Gens->init_from_data_with_go" << endl;
	}
	Gens->init_from_data_with_go(
			DC->A, generators_data,
			go_text,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_load_table after Gens->init_from_data_with_go" << endl;
	}


	if (f_v) {
		cout << "design_activity::do_create_table before Combi.load_design_table" << endl;
	}

	combinatorics_global Combi;
	design_tables *T;

	Combi.load_design_table(DC,
			label,
			T,
			Gens,
			verbose_level);

	if (f_v) {
		cout << "design_activity::do_create_table after Combi.load_design_table" << endl;
	}


	large_set_classify *LS;

	LS = NEW_OBJECT(large_set_classify);

	if (f_v) {
		cout << "design_activity::do_create_table before LS->init" << endl;
	}
	LS->init(DC,
			T,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_create_table after LS->init" << endl;
	}



	strong_generators *H_gens;
	H_gens = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "design_activity::do_load_table before H_gens->init_from_data_with_go" << endl;
	}
	H_gens->init_from_data_with_go(
			DC->A, H_generators_data,
			H_go_text,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_load_table after H_gens->init_from_data_with_go" << endl;
	}


#if 0
	large_set_was *LSW;

	LSW = NEW_OBJECT(large_set_was);


	if (f_v) {
		cout << "design_activity::do_load_table before LSW->init" << endl;
	}
	LSW->init(LS,
			H_gens, H_label,
			selected_orbit_length,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_load_table after LSW->init" << endl;
	}

#endif

	if (f_v) {
		cout << "design_activity::do_load_table done" << endl;
	}
}

void design_activity::do_canonical_form(projective_space_object_classifier_description *Canonical_form_Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_canonical_form" << endl;
	}

	projective_space_object_classifier *OC;

	if (f_v) {
		cout << "design_activity::do_canonical_form" << endl;
	}

	OC = NEW_OBJECT(projective_space_object_classifier);

	if (f_v) {
		cout << "design_activity::do_canonical_form before OC->do_the_work" << endl;
	}
	OC->do_the_work(
			Canonical_form_Descr,
			FALSE,
			NULL,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_canonical_form after OC->do_the_work" << endl;
	}

	FREE_OBJECT(OC);



	if (f_v) {
		cout << "design_activity::do_canonical_form done" << endl;
	}

}

void design_activity::do_export_inc(
		design_create *DC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_export_inc" << endl;
	}

	string fname;

	fname.assign(DC->label_txt);
	fname.append("_inc.txt");

	combinatorics_domain Combi;

	int v = DC->degree;
	int k = DC->k;
	int b = DC->sz;

	int N = k * b;
	int *M;
	int h;

	Combi.compute_incidence_matrix(v, b, k, DC->set,
			M, verbose_level);


	{
		ofstream ost(fname);

		ost << v << " " << b << " " << N << endl;
		for (h = 0; h < v * b; h++) {
			if (M[h]) {
				ost << h << " ";
			}
		}
		ost << endl;
		ost << "-1" << endl;
	}
	file_io Fio;

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	FREE_int(M);

	//8 8 24
	//0 1 2 8 11 12 16 21 22 25 27 29 33 36 39 42 44 46 50 53 55 59 62 63
	//-1 1
	//48



	if (f_v) {
		cout << "design_activity::do_export_inc done" << endl;
	}
}

void design_activity::do_export_blocks(
		design_create *DC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_export_blocks" << endl;
	}

	string fname;

	fname.assign(DC->label_txt);
	fname.append("_blocks_coded.csv");

	combinatorics_domain Combi;

	int v = DC->degree;
	int k = DC->k;
	int b = DC->sz;


	file_io Fio;
	Fio.lint_matrix_write_csv(fname, DC->set, 1, b);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;



	fname.assign(DC->label_txt);
	fname.append("_blocks.csv");

	int *Blocks;


	Combi.compute_blocks(v, b, k, DC->set, Blocks, verbose_level);

	Fio.int_matrix_write_csv(fname, Blocks, b, k);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	FREE_int(Blocks);


	if (f_v) {
		cout << "design_activity::do_export_blocks done" << endl;
	}
}

void design_activity::do_row_sums(
		design_create *DC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_row_sums" << endl;
	}

	string fname;

	fname.assign(DC->label_txt);
	fname.append("_inc.txt");

	combinatorics_domain Combi;

	int v = DC->degree;
	int k = DC->k;
	int b = DC->sz;

	//int N = k * b;
	int *M;
	int i, j;
	int *R;

	R = NEW_int(v);

	Combi.compute_incidence_matrix(v, b, k, DC->set,
			M, verbose_level);

	for (i = 0; i < v; i++) {
		R[i] = 0;
		for (j = 0; j < b; j++) {
			if (M[i * b + j]) {
				R[i]++;
			}
		}
	}

	tally T;

	T.init(R, v, FALSE, 0);
	cout << "distribution of row sums: ";
	T.print(TRUE /* f_backwards */);
	cout << endl;

	FREE_int(M);
	FREE_int(R);



	if (f_v) {
		cout << "design_activity::do_row_sums done" << endl;
	}
}

void design_activity::do_tactical_decomposition(
		design_create *DC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_tactical_decomposition" << endl;
	}

	string fname;

	fname.assign(DC->label_txt);
	fname.append("_inc.txt");

	combinatorics_domain Combi;

	int v = DC->degree;
	int k = DC->k;
	int b = DC->sz;

	//int N = k * b;
	int *M;
	//int i, j;
	int *R;

	R = NEW_int(v);

	Combi.compute_incidence_matrix(v, b, k, DC->set,
			M, verbose_level);

	{
		incidence_structure *Inc;
		partitionstack *Stack;


		Inc = NEW_OBJECT(incidence_structure);

		Inc->init_by_matrix(v, b, M, 0 /* verbose_level */);

		Stack = NEW_OBJECT(partitionstack);

		Stack->allocate_with_two_classes(v + b, v, b, 0 /* verbose_level */);



		while (TRUE) {

			int ht0, ht1;

			ht0 = Stack->ht;

			if (f_v) {
				cout << "process_single_case before refine_column_partition_safe" << endl;
			}
			Inc->refine_column_partition_safe(*Stack, verbose_level - 2);
			if (f_v) {
				cout << "process_single_case after refine_column_partition_safe" << endl;
			}
			if (f_v) {
				cout << "process_single_case before refine_row_partition_safe" << endl;
			}
			Inc->refine_row_partition_safe(*Stack, verbose_level - 2);
			if (f_v) {
				cout << "process_single_case after refine_row_partition_safe" << endl;
			}
			ht1 = Stack->ht;
			if (ht1 == ht0) {
				break;
			}
		}

		int f_labeled = TRUE;

		Inc->print_partitioned(cout, *Stack, f_labeled);
		Inc->get_and_print_decomposition_schemes(*Stack);
		Stack->print_classes(cout);


		int f_print_subscripts = FALSE;
		cout << "Decomposition:\\\\" << endl;
		cout << "Row scheme:\\\\" << endl;
		Inc->get_and_print_row_tactical_decomposition_scheme_tex(
				cout, TRUE /* f_enter_math */,
			f_print_subscripts, *Stack);
		cout << "Column scheme:\\\\" << endl;
		Inc->get_and_print_column_tactical_decomposition_scheme_tex(
				cout, TRUE /* f_enter_math */,
			f_print_subscripts, *Stack);

		set_of_sets *Row_classes;
		set_of_sets *Col_classes;

		Stack->get_row_classes(Row_classes, verbose_level);
		cout << "Row classes:\\\\" << endl;
		Row_classes->print_table_tex(cout);


		Stack->get_column_classes(Col_classes, verbose_level);
		cout << "Col classes:\\\\" << endl;
		Col_classes->print_table_tex(cout);

		if (Row_classes->nb_sets > 1) {
			cout << "The row partition splits" << endl;
		}

		if (Col_classes->nb_sets > 1) {
			cout << "The col partition splits" << endl;
		}


		FREE_OBJECT(Inc);
		FREE_OBJECT(Stack);
		FREE_OBJECT(Row_classes);
		FREE_OBJECT(Col_classes);
	}

	if (f_v) {
		cout << "design_activity::do_tactical_decomposition" << endl;
	}

}

}}



