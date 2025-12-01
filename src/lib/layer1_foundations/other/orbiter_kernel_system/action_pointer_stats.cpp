/*
 * action_pointer_stats.cpp
 *
 *  Created on: Oct 24, 2025
 *      Author: betten
 */




#include "foundations.h"

using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace orbiter_kernel_system {


action_pointer_stats::action_pointer_stats()
{
	nb_times_image_of_called = 0;
	nb_times_image_of_low_level_called = 0;
	nb_times_unpack_called = 0;
	nb_times_pack_called = 0;
	nb_times_retrieve_called = 0;
	nb_times_store_called = 0;
	nb_times_mult_called = 0;
	nb_times_invert_called = 0;

	ff_nb_add = 0;
	ff_nb_add_by_table = 0;
	ff_nb_add_long = 0;
	ff_nb_mult = 0;
	ff_nb_mult_by_table = 0;
	ff_nb_mult_long = 0;
	ff_nb_negate = 0;
	ff_nb_inverse = 0;
	ff_nb_inverse_by_table = 0;
	ff_nb_inverse_long = 0;
	ff_nb_power = 0;
	ff_nb_frobenius_power = 0;
	ff_nb_absolute_trace = 0;
	ff_nb_absolute_norm = 0;
	ff_nb_alpha_power = 0;
	ff_nb_log_alpha = 0;

	mem_nb_new = 0;
	mem_nb_free = 0;

	reset_counters();

}


action_pointer_stats::~action_pointer_stats()
{
	nb_times_image_of_called = 0;
	nb_times_image_of_low_level_called = 0;
	nb_times_unpack_called = 0;
	nb_times_pack_called = 0;
	nb_times_retrieve_called = 0;
	nb_times_store_called = 0;
	nb_times_mult_called = 0;
	nb_times_invert_called = 0;

	ff_nb_add = 0;
	ff_nb_mult = 0;
}


void action_pointer_stats::reset_counters()
{
	nb_times_image_of_called = 0;
	nb_times_image_of_low_level_called = 0;
	nb_times_unpack_called = 0;
	nb_times_pack_called = 0;
	nb_times_retrieve_called = 0;
	nb_times_store_called = 0;
	nb_times_mult_called = 0;
	nb_times_invert_called = 0;

	ff_nb_add = 0;
	ff_nb_add_by_table = 0;
	ff_nb_add_long = 0;
	ff_nb_mult = 0;
	ff_nb_mult_by_table = 0;
	ff_nb_mult_long = 0;
	ff_nb_negate = 0;
	ff_nb_inverse = 0;
	ff_nb_inverse_by_table = 0;
	ff_nb_inverse_long = 0;
	ff_nb_power = 0;
	ff_nb_frobenius_power = 0;
	ff_nb_absolute_trace = 0;
	ff_nb_absolute_norm = 0;
	ff_nb_alpha_power = 0;
	ff_nb_log_alpha = 0;

	mem_nb_new = 0;
	mem_nb_free = 0;

}

void action_pointer_stats::save_stats(
		std::string &fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_pointer_stats::save_stats" << endl;
		cout << "action_pointer_stats::save_stats fname_base = " << fname_base << endl;
	}

	int nb_cols = 27;
	int nb_rows = 1;

	string *Table;

	Table = new string [nb_rows * nb_cols];
	Table[0] = fname_base;
	Table[1] = std::to_string(nb_times_image_of_called);
	Table[2] = std::to_string(nb_times_image_of_low_level_called);
	Table[3] = std::to_string(nb_times_unpack_called);
	Table[4] = std::to_string(nb_times_pack_called);
	Table[5] = std::to_string(nb_times_retrieve_called);
	Table[6] = std::to_string(nb_times_store_called);
	Table[7] = std::to_string(nb_times_mult_called);
	Table[8] = std::to_string(nb_times_invert_called);
	Table[9] = std::to_string(ff_nb_add);
	Table[10] = std::to_string(ff_nb_add_by_table);
	Table[11] = std::to_string(ff_nb_add_long);
	Table[12] = std::to_string(ff_nb_mult);
	Table[13] = std::to_string(ff_nb_mult_by_table);
	Table[14] = std::to_string(ff_nb_mult_long);
	Table[15] = std::to_string(ff_nb_negate);
	Table[16] = std::to_string(ff_nb_inverse);
	Table[17] = std::to_string(ff_nb_inverse_by_table);
	Table[18] = std::to_string(ff_nb_inverse_long);
	Table[19] = std::to_string(ff_nb_power);
	Table[20] = std::to_string(ff_nb_frobenius_power);
	Table[21] = std::to_string(ff_nb_absolute_trace);
	Table[22] = std::to_string(ff_nb_absolute_norm);
	Table[23] = std::to_string(ff_nb_alpha_power);
	Table[24] = std::to_string(ff_nb_log_alpha);
	Table[25] = std::to_string(mem_nb_new);
	Table[26] = std::to_string(mem_nb_free);


	other::orbiter_kernel_system::file_io Fio;

	std::string fname_stats;

	fname_stats = fname_base + "_stats.csv";

	std::string *Col_headings;

	Col_headings = new string [nb_cols];

	Col_headings[0] = "label";
	Col_headings[1] = "image_of";
	Col_headings[2] = "image_of_low_level";
	Col_headings[3] = "unpack";
	Col_headings[4] = "pack";
	Col_headings[5] = "retrieve";
	Col_headings[6] = "store";
	Col_headings[7] = "mult";
	Col_headings[8] = "invert";
	Col_headings[9] = "ff_add";
	Col_headings[10] = "ff_add_by_table";
	Col_headings[11] = "ff_add_long";
	Col_headings[12] = "ff_mult";
	Col_headings[13] = "ff_mult_by_table";
	Col_headings[14] = "ff_mult_long";
	Col_headings[15] = "ff_negate";
	Col_headings[16] = "ff_inverse";
	Col_headings[17] = "ff_inverse_by_table";
	Col_headings[18] = "ff_inverse_long";
	Col_headings[19] = "ff_power";
	Col_headings[20] = "ff_frobenius_power";
	Col_headings[21] = "ff_absolute_trace";
	Col_headings[22] = "ff_absolute_norm";
	Col_headings[23] = "ff_alpha_power";
	Col_headings[24] = "ff_log_alpha";
	Col_headings[25] = "men_nb_new";
	Col_headings[26] = "men_nb_free";





	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"nb_rows = " << nb_rows << endl;
		cout << "algebra_global_with_action::split_by_classes "
				"nb_cols = " << nb_cols << endl;
	}

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_stats,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"written file " << fname_stats << " of size "
				<< Fio.file_size(fname_stats) << endl;
	}






	delete [] Col_headings;
	delete [] Table;

	if (f_v) {
		cout << "action_pointer_stats::save_stats done" << endl;
	}

}




}}}}




