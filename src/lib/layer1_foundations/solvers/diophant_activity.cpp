/*
 * diophant_activity.cpp
 *
 *  Created on: May 29, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace solvers {



diophant_activity::diophant_activity()
{
	Descr = NULL;
}

diophant_activity::~diophant_activity()
{
}


void diophant_activity::init_from_file(
		diophant_activity_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant_activity::init_from_file" << endl;
	}

	diophant_activity::Descr = Descr;
	data_structures::string_tools ST;

	if (!Descr->f_input_file) {
		cout << "diophant_activity::init_from_file "
				"please use option -q <q>" << endl;
		exit(1);
	}

	diophant *Dio;

	Dio = NEW_OBJECT(diophant);
	Dio->read_general_format(Descr->input_file, verbose_level);

}

void diophant_activity::perform_activity(
		diophant_activity_description *Descr, diophant *Dio,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant_activity::perform_activity" << endl;
	}

	diophant_activity::Descr = Descr;
	data_structures::string_tools ST;


	if (Descr->f_solve_mckay) {

		long int nb_backtrack_nodes;

		cout << "solving with mckay" << endl;
		Dio->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level - 2);
		Dio->nb_steps_betten = nb_backtrack_nodes;

		cout << "Found " << Dio->_resultanz << " solutions with "
				<< Dio->nb_steps_betten << " backtrack steps" << endl;

		if (true) {
			string output_file1;

			output_file1.assign(Dio->label);
			ST.replace_extension_with(output_file1, "_sol.csv");


			Dio->write_solutions_full_length(output_file1, verbose_level);


			if (Dio->f_has_sum) {
				string output_file2;
				output_file2.assign(Dio->label);
				ST.replace_extension_with(output_file2, "_sol_index_set.txt");
				Dio->write_solutions_index_set(output_file2, verbose_level);
			}
		}
	}
	else if (Descr->f_solve_standard) {
		//long int nb_backtrack_nodes;

		cout << "solving with standard method" << endl;

		Dio->solve_all_betten(verbose_level - 2);

		cout << "Found " << Dio->_resultanz << " solutions with "
				<< Dio->nb_steps_betten << " backtrack steps" << endl;

		if (true) {
			string output_file;

			output_file.assign(Dio->label);
			ST.replace_extension_with(output_file, "_sol.csv");


			Dio->write_solutions_full_length(output_file, verbose_level);
		}
	}
	else if (Descr->f_solve_DLX) {
		//long int nb_backtrack_nodes;

		cout << "solving with DLX" << endl;

		Dio->solve_all_DLX(verbose_level - 2);

		cout << "Found " << Dio->_resultanz << " solutions with "
				<< Dio->nb_steps_betten << " backtrack steps" << endl;

		if (true) {
			string output_file;

			output_file.assign(Dio->label);
			ST.replace_extension_with(output_file, "_sol.csv");


			Dio->write_solutions_full_length(output_file, verbose_level);
		}
	}
	else if (Descr->f_draw_as_bitmap) {
		string fname_base;

		fname_base.assign(Descr->input_file);
		ST.replace_extension_with(fname_base, "_drawing");
		Dio->draw_as_bitmap(fname_base, true, Descr->box_width, Descr->bit_depth,
			verbose_level);

	}
	else if (Descr->f_test_single_equation) {
		Dio->project_to_single_equation_and_solve(
				Descr->max_number_of_coefficients,
				verbose_level);
	}
	else if (Descr->f_project_to_single_equation_and_solve) {
		Dio->split_by_equation(
				Descr->eqn_idx,
				true,
				Descr->solve_case_idx,
				verbose_level);
	}
	else if (Descr->f_project_to_two_equations_and_solve) {

		if (Descr->solve_case_idx_r == -1) {
			Dio->split_by_two_equations(
					Descr->eqn1_idx,
					Descr->eqn2_idx,
					false,
					0, 1,
					verbose_level);
		}
		else {
			Dio->split_by_two_equations(
					Descr->eqn1_idx,
					Descr->eqn2_idx,
					true,
					Descr->solve_case_idx_r,
					Descr->solve_case_idx_m,
					verbose_level);
		}
	}
	else if (Descr->f_draw) {
		string fname_base;

		fname_base.assign(Descr->input_file);
		ST.replace_extension_with(fname_base, "_drawing");

		graphics::layered_graph_draw_options *Draw_options;

		Draw_options = Get_draw_options(Descr->draw_options_label);

		Dio->draw_partitioned(
				fname_base,
				Draw_options,
			false, 0, 0,
			verbose_level);
	}
	else if (Descr->f_perform_column_reductions) {
		diophant *D2;

		D2 = Dio->trivial_column_reductions(verbose_level);

		string fname2;
		orbiter_kernel_system::file_io Fio;

		fname2.assign(Descr->input_file);
		ST.replace_extension_with(fname2, "_red.diophant");

		D2->save_in_general_format(fname2, verbose_level);
		cout << "Written file " << fname2 << " of size " << Fio.file_size(fname2) << endl;

	}
	else {
		cout << "diophant_activity::perform_activity no activity found" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "diophant_activity::perform_activity done" << endl;
	}
}



}}}

