/*
 * design_activity.cpp
 *
 *  Created on: May 26, 2021
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


design_activity::design_activity()
{
	Descr = NULL;

}

design_activity::~design_activity()
{

}

void design_activity::perform_activity(
		design_activity_description *Descr,
		design_create *DC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::perform_activity" << endl;
	}

	design_activity::Descr = Descr;

	if (Descr->f_load_table) {
		do_load_table(
				DC,
				Descr->load_table_label,
				Descr->load_table_group,
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
	else if (Descr->f_extract_solutions_by_index_csv) {

		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_extract_solutions_by_index_csv" << endl;
		}

		do_extract_solutions_by_index(
				DC,
				Descr->extract_solutions_by_index_label,
				Descr->extract_solutions_by_index_group,
				Descr->extract_solutions_by_index_fname_solutions_in,
				Descr->extract_solutions_by_index_fname_solutions_out,
				Descr->extract_solutions_by_index_prefix,
				true /* f_csv */,
				verbose_level);

		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_extract_solutions_by_index_csv done" << endl;
		}
	}
	else if (Descr->f_extract_solutions_by_index_txt) {

		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_extract_solutions_by_index_txt" << endl;
		}

		do_extract_solutions_by_index(
				DC,
				Descr->extract_solutions_by_index_label,
				Descr->extract_solutions_by_index_group,
				Descr->extract_solutions_by_index_fname_solutions_in,
				Descr->extract_solutions_by_index_fname_solutions_out,
				Descr->extract_solutions_by_index_prefix,
				false /* f_csv */,
				verbose_level);

		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_extract_solutions_by_index_txt done" << endl;
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
	else if (Descr->f_intersection_matrix) {
		if (f_v) {
			cout << "design_activity::perform_activity "
					"f_intersection_matrix" << endl;
		}
		do_intersection_matrix(
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
			cout << "design_activity::perform_activity "
					"f_tactical_decomposition" << endl;
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
		std::string &group_label,
		std::string &fname_in,
		std::string &fname_out,
		std::string &prefix_text,
		int f_csv_format,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index" << endl;
	}

	combinatorics_global Combi;
	design_tables *T;

	apps_algebra::any_group *AG;

	AG = Get_object_of_type_any_group(group_label);

	Combi.load_design_table(DC,
			label,
			T,
			AG->Subgroup_gens,
			verbose_level);

	if (f_v) {
		cout << "design_activity::do_extract_solutions_by_index after Combi.load_design_table" << endl;
	}

	int *prefix;
	int prefix_sz;

	Int_vec_scan(prefix_text, prefix, prefix_sz);

	orbiter_kernel_system::file_io Fio;
	int *Sol_idx;
	int nb_sol;
	int sol_width = 0;

	if (f_csv_format) {
		int *Sol_idx_1;
		int i, j;
		Fio.Csv_file_support->int_matrix_read_csv(
				fname_in, Sol_idx_1, nb_sol, sol_width, verbose_level);

		Sol_idx = NEW_int(nb_sol * (prefix_sz + sol_width));
		for (i = 0; i < nb_sol; i++) {
			for (j = 0; j < prefix_sz; j++) {
				Sol_idx[i * (prefix_sz + sol_width) + j] = prefix[j];
			}
			for (j = 0; j < sol_width; j++) {
				Sol_idx[i * (prefix_sz + sol_width) + prefix_sz + j] = Sol_idx_1[i * sol_width + j];
			}
		}
		FREE_int(Sol_idx_1);
		sol_width += prefix_sz;
	}
	else {
		data_structures::set_of_sets *SoS;
		int underlying_set_size = 0;
		int i, j;

		SoS = NEW_OBJECT(data_structures::set_of_sets);
		SoS->init_from_orbiter_file(underlying_set_size,
				fname_in, verbose_level);
		nb_sol = SoS->nb_sets;

		if (nb_sol) {
			if (!SoS->has_constant_size_property()) {
				cout << "design_activity::do_extract_solutions_by_index the sets have different sizes" << endl;
				exit(1);
			}
			sol_width = SoS->Set_size[0];

			Sol_idx = NEW_int(nb_sol * (prefix_sz + sol_width));
			for (i = 0; i < nb_sol; i++) {
				for (j = 0; j < prefix_sz; j++) {
					Sol_idx[i * (prefix_sz + sol_width) + j] = prefix[j];
				}
				for (j = 0; j < sol_width; j++) {
					Sol_idx[i * (prefix_sz + sol_width) + prefix_sz + j] = SoS->Sets[i][j];
				}
			}
			sol_width += prefix_sz;
		}
		else {
			Sol_idx = NEW_int(1);
		}
		FREE_OBJECT(SoS);
	}


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
		std::string &group_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_create_table" << endl;
	}

	combinatorics_global Combi;
	design_tables *T;


	apps_algebra::any_group *AG;

	AG = Get_object_of_type_any_group(group_label);

	if (f_v) {
		cout << "design_activity::do_create_table before Combi.create_design_table" << endl;
	}

	Combi.create_design_table(DC,
			label,
			T,
			AG->Subgroup_gens,
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
		std::string &group_label,
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

	apps_algebra::any_group *AG;

	AG = Get_object_of_type_any_group(group_label);

	if (f_v) {
		cout << "design_activity::do_create_table "
				"before Combi.load_design_table" << endl;
	}

	combinatorics_global Combi;
	design_tables *T;

	Combi.load_design_table(DC,
			label,
			T,
			AG->Subgroup_gens,
			verbose_level);

	if (f_v) {
		cout << "design_activity::do_create_table "
				"after Combi.load_design_table" << endl;
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



	groups::strong_generators *H_gens;
	H_gens = NEW_OBJECT(groups::strong_generators);

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

void design_activity::do_canonical_form(
		combinatorics::classification_of_objects_description
			*Canonical_form_Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_canonical_form" << endl;
	}


#if 0
	combinatorics::classification_of_objects *OC;

	if (f_v) {
		cout << "design_activity::do_canonical_form" << endl;
	}

	OC = NEW_OBJECT(classification_of_objects);

	if (f_v) {
		cout << "design_activity::do_canonical_form before OC->do_the_work" << endl;
	}
	OC->do_the_work(
			Canonical_form_Descr,
			false,
			NULL,
			verbose_level);
	if (f_v) {
		cout << "design_activity::do_canonical_form after OC->do_the_work" << endl;
	}

	FREE_OBJECT(OC);


#endif

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

	fname = DC->label_txt + "_inc.txt";

	if (f_v) {
		cout << "design_activity::do_export_inc fname=" << fname << endl;
	}


	{
		ofstream ost(fname);

		int h;
		ost << DC->v << " " << DC->b << " " << DC->nb_inc << endl;
		for (h = 0; h < DC->v * DC->b; h++) {
			if (DC->incma[h]) {
				ost << h << " ";
			}
		}
		ost << endl;
		ost << "-1" << endl;
	}
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	//8 8 24
	//0 1 2 8 11 12 16 21 22 25 27 29 33 36 39 42 44 46 50 53 55 59 62 63
	//-1 1
	//48



	if (f_v) {
		cout << "design_activity::do_export_inc done" << endl;
	}
}


void design_activity::do_intersection_matrix(
		design_create *DC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_activity::do_intersection_matrix" << endl;
	}



	int *AAt;
	int i, j, h, cnt;

	AAt = NEW_int(DC->v * DC->v);
	for (i = 0; i < DC->v; i++) {
		for (j = 0; j < DC->v; j++) {
			cnt = 0;
			for (h = 0; h < DC->b; h++) {
				if (DC->incma[i * DC->b + h] && DC->incma[j * DC->b + h]) {
					cnt++;
				}
			}
			AAt[i * DC->v + j] = cnt;
		}

	}

	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = DC->label_txt + "_AAt.csv";

	if (f_v) {
		cout << "design_activity::do_intersection_matrix fname=" << fname << endl;
	}

	{
		ofstream ost(fname);

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, AAt, DC->v, DC->v);

	}

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}



	if (f_v) {
		cout << "design_activity::do_intersection_matrix done" << endl;
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

	fname = DC->label_txt + "_blocks_coded.csv";

	combinatorics::combinatorics_domain Combi;

	int v = DC->degree;
	int k = DC->k;
	int b = DC->sz;


	orbiter_kernel_system::file_io Fio;
	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, DC->set, 1, b);

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	fname = DC->label_txt + "_blocks.csv";

	int *Blocks;


	if (DC->f_has_set) {
		if (f_v) {
			cout << "design_activity::do_export_blocks "
					"before Combi.compute_blocks_from_coding" << endl;
		}
		Combi.compute_blocks_from_coding(v, b, k, DC->set, Blocks, verbose_level);
		if (f_v) {
			cout << "design_activity::do_export_blocks "
					"after Combi.compute_blocks_from_coding" << endl;
		}
	}
	else if (DC->f_has_incma) {
		if (f_v) {
			cout << "design_activity::do_export_blocks "
					"before Combi.compute_blocks_from_incma" << endl;
		}
		Combi.compute_blocks_from_incma(v, b, k, DC->incma,
					Blocks, verbose_level);
		if (f_v) {
			cout << "design_activity::do_export_blocks "
					"after Combi.compute_blocks_from_incma" << endl;
		}
	}
	else {
		cout << "design_activity::do_export_blocks "
				"we neither have a set nor an incma" << endl;
		exit(1);
	}

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Blocks, b, k);

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


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

	fname = DC->label_txt + "_inc.txt";

	combinatorics::combinatorics_domain Combi;


	int i, j;
	int *R;

	R = NEW_int(DC->v);

	for (i = 0; i < DC->v; i++) {
		R[i] = 0;
		for (j = 0; j < DC->b; j++) {
			if (DC->incma[i * DC->b + j]) {
				R[i]++;
			}
		}
	}

	data_structures::tally T;

	T.init(R, DC->v, false, 0);
	if (f_v) {
		cout << "distribution of row sums: ";
		T.print(true /* f_backwards */);
		cout << endl;
	}

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

	fname = DC->label_txt + "_inc.txt";

	combinatorics::combinatorics_domain Combi;



	{
		geometry::incidence_structure *Inc;
		data_structures::partitionstack *Stack;


		Inc = NEW_OBJECT(geometry::incidence_structure);

		Inc->init_by_matrix(DC->v, DC->b, DC->incma, 0 /* verbose_level */);

		Stack = NEW_OBJECT(data_structures::partitionstack);

		Stack->allocate_with_two_classes(DC->v + DC->b, DC->v, DC->b, 0 /* verbose_level */);



		while (true) {

			int ht0, ht1;

			ht0 = Stack->ht;

			if (f_v) {
				cout << "design_activity::do_tactical_decomposition "
						"before refine_column_partition_safe" << endl;
			}
			Inc->refine_column_partition_safe(*Stack, verbose_level - 2);
			if (f_v) {
				cout << "design_activity::do_tactical_decomposition "
						"after refine_column_partition_safe" << endl;
			}
			if (f_v) {
				cout << "design_activity::do_tactical_decomposition "
						"before refine_row_partition_safe" << endl;
			}
			Inc->refine_row_partition_safe(*Stack, verbose_level - 2);
			if (f_v) {
				cout << "design_activity::do_tactical_decomposition "
						"after refine_row_partition_safe" << endl;
			}
			ht1 = Stack->ht;
			if (ht1 == ht0) {
				break;
			}
		}

		int f_labeled = true;

		Inc->print_partitioned(cout, *Stack, f_labeled);
		Inc->get_and_print_decomposition_schemes(*Stack);
		Stack->print_classes(cout);


		int f_print_subscripts = false;
		cout << "Decomposition:\\\\" << endl;
		cout << "Row scheme:\\\\" << endl;
		Inc->get_and_print_row_tactical_decomposition_scheme_tex(
				cout, true /* f_enter_math */,
			f_print_subscripts, *Stack);
		cout << "Column scheme:\\\\" << endl;
		Inc->get_and_print_column_tactical_decomposition_scheme_tex(
				cout, true /* f_enter_math */,
			f_print_subscripts, *Stack);

		data_structures::set_of_sets *Row_classes;
		data_structures::set_of_sets *Col_classes;

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

}}}




