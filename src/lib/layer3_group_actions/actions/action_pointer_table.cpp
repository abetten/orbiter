/*
 * action_pointer_table.cpp
 *
 *  Created on: Feb 5, 2019
 *      Author: betten
 */

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {


action_pointer_table::action_pointer_table()
{
	Record_birth();
	null_function_pointers();

	nb_times_image_of_called = 0;
	nb_times_image_of_low_level_called = 0;
	nb_times_unpack_called = 0;
	nb_times_pack_called = 0;
	nb_times_retrieve_called = 0;
	nb_times_store_called = 0;
	nb_times_mult_called = 0;
	nb_times_invert_called = 0;
}

action_pointer_table::~action_pointer_table()
{
	Record_death();

}

void action_pointer_table::reset_counters()
{
	nb_times_image_of_called = 0;
	nb_times_image_of_low_level_called = 0;
	nb_times_unpack_called = 0;
	nb_times_pack_called = 0;
	nb_times_retrieve_called = 0;
	nb_times_store_called = 0;
	nb_times_mult_called = 0;
	nb_times_invert_called = 0;

}

void action_pointer_table::save_stats(
		std::string &fname_base)
{
	long int stats[8];

	stats[0] = nb_times_image_of_called;
	stats[1] = nb_times_image_of_low_level_called;
	stats[2] = nb_times_unpack_called;
	stats[3] = nb_times_pack_called;
	stats[4] = nb_times_retrieve_called;
	stats[5] = nb_times_store_called;
	stats[6] = nb_times_mult_called;
	stats[7] = nb_times_invert_called;

	other::orbiter_kernel_system::file_io Fio;
	string fname;

	fname = fname_base + "_stats.csv";

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, stats, 1, 8);

}


void action_pointer_table::null_function_pointers()
{
	label.assign("null");
	// 26 function pointers:
	ptr_element_image_of = NULL;
	ptr_element_image_of_low_level = NULL;
	ptr_element_linear_entry_ij = NULL;
	ptr_element_linear_entry_frobenius = NULL;
	ptr_element_one = NULL;
	ptr_element_is_one = NULL;
	ptr_element_unpack = NULL;
	ptr_element_pack = NULL;
	ptr_element_retrieve = NULL;
	ptr_element_store = NULL;
	ptr_element_mult = NULL;
	ptr_element_invert = NULL;
	ptr_element_transpose = NULL;
	ptr_element_move = NULL;
	ptr_element_dispose = NULL;
	ptr_element_print = NULL;
	ptr_element_print_quick = NULL;
	ptr_element_print_latex = NULL;
	ptr_element_print_latex_with_point_labels = NULL;
	ptr_element_print_verbose = NULL;
	ptr_element_code_for_make_element = NULL;
	ptr_element_print_for_make_element = NULL;
	ptr_element_print_for_make_element_no_commas = NULL;
	ptr_print_point = NULL;
	ptr_unrank_point = NULL;
	ptr_rank_point = NULL;
}




void action_pointer_table::copy_from_but_reset_counters(
		action_pointer_table *T)
{
	// copy 26 function pointers:
	ptr_element_image_of = T->ptr_element_image_of;
	ptr_element_image_of_low_level = T->ptr_element_image_of_low_level;
	ptr_element_linear_entry_ij = T->ptr_element_linear_entry_ij;
	ptr_element_linear_entry_frobenius = T->ptr_element_linear_entry_frobenius;
	ptr_element_one = T->ptr_element_one;
	ptr_element_is_one = T->ptr_element_is_one;
	ptr_element_unpack = T->ptr_element_unpack;
	ptr_element_pack = T->ptr_element_pack;
	ptr_element_retrieve = T->ptr_element_retrieve;
	ptr_element_store = T->ptr_element_store;
	ptr_element_mult = T->ptr_element_mult;
	ptr_element_invert = T->ptr_element_invert;
	ptr_element_transpose = T->ptr_element_transpose;
	ptr_element_move = T->ptr_element_move;
	ptr_element_dispose = T->ptr_element_dispose;
	ptr_element_print = T->ptr_element_print;
	ptr_element_print_quick = T->ptr_element_print_quick;
	ptr_element_print_latex = T->ptr_element_print_latex;
	ptr_element_print_latex_with_point_labels = T->ptr_element_print_latex_with_point_labels;
	ptr_element_print_verbose = T->ptr_element_print_verbose;
	ptr_element_code_for_make_element = T->ptr_element_code_for_make_element;
	ptr_element_print_for_make_element = T->ptr_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas = T->ptr_element_print_for_make_element_no_commas;
	ptr_print_point = T->ptr_print_point;
	ptr_unrank_point = T->ptr_unrank_point;
	ptr_rank_point = T->ptr_rank_point;
	reset_counters();
}

}}}

