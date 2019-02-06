/*
 * action_pointer_table.cpp
 *
 *  Created on: Feb 5, 2019
 *      Author: betten
 */

#include "foundations/foundations.h"
#include "group_actions.h"

namespace orbiter {
namespace group_actions {


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
}

void action_pointer_table::init_function_pointers_matrix_group()
{
	ptr_element_image_of = matrix_group_element_image_of;
	ptr_element_image_of_low_level = matrix_group_element_image_of_low_level;
	ptr_element_linear_entry_ij = matrix_group_element_linear_entry_ij;
	ptr_element_linear_entry_frobenius = matrix_group_element_linear_entry_frobenius;
	ptr_element_one = matrix_group_element_one;
	ptr_element_is_one = matrix_group_element_is_one;
	ptr_element_unpack = matrix_group_element_unpack;
	ptr_element_pack = matrix_group_element_pack;
	ptr_element_retrieve = matrix_group_element_retrieve;
	ptr_element_store = matrix_group_element_store;
	ptr_element_mult = matrix_group_element_mult;
	ptr_element_invert = matrix_group_element_invert;
	ptr_element_transpose = matrix_group_element_transpose;
	ptr_element_move = matrix_group_element_move;
	ptr_element_dispose = matrix_group_element_dispose;
	ptr_element_print = matrix_group_element_print;
	ptr_element_print_quick = matrix_group_element_print_quick;
	ptr_element_print_latex = matrix_group_element_print_latex;
	ptr_element_print_verbose = matrix_group_element_print_verbose;
	ptr_element_code_for_make_element =
			matrix_group_element_code_for_make_element;
	ptr_element_print_for_make_element =
			matrix_group_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas =
			matrix_group_element_print_for_make_element_no_commas;
	ptr_print_point = matrix_group_print_point;
}

void action_pointer_table::init_function_pointers_wreath_product_group()
{
	ptr_element_image_of = wreath_product_group_element_image_of;
	ptr_element_image_of_low_level =
			wreath_product_group_element_image_of_low_level;
	ptr_element_linear_entry_ij = wreath_product_group_element_linear_entry_ij;
	ptr_element_linear_entry_frobenius =
			wreath_product_group_element_linear_entry_frobenius;
	ptr_element_one = wreath_product_group_element_one;
	ptr_element_is_one = wreath_product_group_element_is_one;
	ptr_element_unpack = wreath_product_group_element_unpack;
	ptr_element_pack = wreath_product_group_element_pack;
	ptr_element_retrieve = wreath_product_group_element_retrieve;
	ptr_element_store = wreath_product_group_element_store;
	ptr_element_mult = wreath_product_group_element_mult;
	ptr_element_invert = wreath_product_group_element_invert;
	ptr_element_transpose = wreath_product_group_element_transpose;
	ptr_element_move = wreath_product_group_element_move;
	ptr_element_dispose = wreath_product_group_element_dispose;
	ptr_element_print = wreath_product_group_element_print;
	ptr_element_print_quick = wreath_product_group_element_print_quick;
	ptr_element_print_latex = wreath_product_group_element_print_latex;
	ptr_element_print_verbose = wreath_product_group_element_print_verbose;
	ptr_element_code_for_make_element =
			wreath_product_group_element_code_for_make_element;
	ptr_element_print_for_make_element =
			wreath_product_group_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas =
			wreath_product_group_element_print_for_make_element_no_commas;
	ptr_print_point = wreath_product_group_print_point;
}

void action_pointer_table::init_function_pointers_direct_product_group()
{
	ptr_element_image_of = direct_product_group_element_image_of;
	ptr_element_image_of_low_level =
			direct_product_group_element_image_of_low_level;
	ptr_element_linear_entry_ij =
			direct_product_group_element_linear_entry_ij;
	ptr_element_linear_entry_frobenius =
			direct_product_group_element_linear_entry_frobenius;
	ptr_element_one = direct_product_group_element_one;
	ptr_element_is_one = direct_product_group_element_is_one;
	ptr_element_unpack = direct_product_group_element_unpack;
	ptr_element_pack = direct_product_group_element_pack;
	ptr_element_retrieve = direct_product_group_element_retrieve;
	ptr_element_store = direct_product_group_element_store;
	ptr_element_mult = direct_product_group_element_mult;
	ptr_element_invert = direct_product_group_element_invert;
	ptr_element_transpose = direct_product_group_element_transpose;
	ptr_element_move = direct_product_group_element_move;
	ptr_element_dispose = direct_product_group_element_dispose;
	ptr_element_print = direct_product_group_element_print;
	ptr_element_print_quick = direct_product_group_element_print_quick;
	ptr_element_print_latex = direct_product_group_element_print_latex;
	ptr_element_print_verbose = direct_product_group_element_print_verbose;
	ptr_element_code_for_make_element =
			direct_product_group_element_code_for_make_element;
	ptr_element_print_for_make_element =
			direct_product_group_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas =
			direct_product_group_element_print_for_make_element_no_commas;
	ptr_print_point = direct_product_group_print_point;
}

void action_pointer_table::init_function_pointers_permutation_group()
{
	ptr_element_image_of = perm_group_element_image_of;
	ptr_element_image_of_low_level = NULL;
	ptr_element_linear_entry_ij = NULL;
	ptr_element_linear_entry_frobenius = NULL;
	ptr_element_one = perm_group_element_one;
	ptr_element_is_one = perm_group_element_is_one;
	ptr_element_unpack = perm_group_element_unpack;
	ptr_element_pack = perm_group_element_pack;
	ptr_element_retrieve = perm_group_element_retrieve;
	ptr_element_store = perm_group_element_store;
	ptr_element_mult = perm_group_element_mult;
	ptr_element_invert = perm_group_element_invert;
	ptr_element_transpose = NULL;
	ptr_element_move = perm_group_element_move;
	ptr_element_dispose = perm_group_element_dispose;
	ptr_element_print = perm_group_element_print;
	ptr_element_print_quick = perm_group_element_print; // no quick version here!
	ptr_element_print_latex = perm_group_element_print_latex;
	ptr_element_print_verbose = perm_group_element_print_verbose;
	ptr_element_code_for_make_element =
			perm_group_element_code_for_make_element;
	ptr_element_print_for_make_element =
			perm_group_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas =
			perm_group_element_print_for_make_element_no_commas;
	ptr_print_point = perm_group_print_point;
}

void action_pointer_table::init_function_pointers_induced_action()
{
	//ptr_get_transversal_rep = induced_action_get_transversal_rep;
	ptr_element_image_of = induced_action_element_image_of;
	ptr_element_image_of_low_level = induced_action_element_image_of_low_level;
	ptr_element_linear_entry_ij = NULL;
	ptr_element_linear_entry_frobenius = NULL;
	ptr_element_one = induced_action_element_one;
	ptr_element_is_one = induced_action_element_is_one;
	ptr_element_unpack = induced_action_element_unpack;
	ptr_element_pack = induced_action_element_pack;
	ptr_element_retrieve = induced_action_element_retrieve;
	ptr_element_store = induced_action_element_store;
	ptr_element_mult = induced_action_element_mult;
	ptr_element_invert = induced_action_element_invert;
	ptr_element_transpose = induced_action_element_transpose;
	ptr_element_move = induced_action_element_move;
	ptr_element_dispose = induced_action_element_dispose;
	ptr_element_print = induced_action_element_print;
	ptr_element_print_quick = induced_action_element_print_quick;
	ptr_element_print_latex = induced_action_element_print_latex;
	ptr_element_print_verbose = induced_action_element_print_verbose;
	ptr_element_code_for_make_element =
			induced_action_element_code_for_make_element;
	ptr_element_print_for_make_element =
			induced_action_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas =
			induced_action_element_print_for_make_element_no_commas;
	ptr_print_point = induced_action_print_point;
}

}}
