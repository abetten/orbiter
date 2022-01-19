/*
 * action_pointer_table.cpp
 *
 *  Created on: Feb 5, 2019
 *      Author: betten
 */

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {
namespace actions {


action_pointer_table::action_pointer_table()
{
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

}

void action_pointer_table::null_function_pointers()
{
	label.assign("null");
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
	ptr_element_print_verbose = NULL;
	ptr_element_code_for_make_element = NULL;
	ptr_element_print_for_make_element = NULL;
	ptr_element_print_for_make_element_no_commas = NULL;
	ptr_print_point = NULL;
	ptr_unrank_point = NULL;
	ptr_rank_point = NULL;
}



}}}

