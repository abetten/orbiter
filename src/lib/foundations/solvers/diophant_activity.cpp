/*
 * diophant_activity.cpp
 *
 *  Created on: May 29, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;

namespace orbiter {
namespace foundations {



diophant_activity::diophant_activity()
{
	Descr = NULL;
}

diophant_activity::~diophant_activity()
{
}


void diophant_activity::init(diophant_activity_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant_activity::init" << endl;
	}

	diophant_activity::Descr = Descr;

	if (!Descr->f_input_file) {
		cout << "diophant_activity::init please use option -q <q>" << endl;
		exit(1);
	}

	diophant *Dio;

	Dio = NEW_OBJECT(diophant);
	Dio->read_general_format(Descr->input_file, verbose_level);

	if (Descr->f_solve_mckay) {

		long int nb_backtrack_nodes;

		cout << "solving with mckay" << endl;
		Dio->solve_all_mckay(nb_backtrack_nodes, verbose_level - 2);
		Dio->nb_steps_betten = nb_backtrack_nodes;

		cout << "Found " << Dio->_resultanz << " solutions with "
				<< Dio->nb_steps_betten << " backtrack steps" << endl;

		if (TRUE) {
			char output_file[1000];

			strcpy(output_file, Descr->input_file);
			replace_extension_with(output_file, ".sol");


			Dio->write_solutions(output_file, verbose_level);
		}
	}
	else if (Descr->f_solve_standard) {
		long int nb_backtrack_nodes;

		cout << "solving with standard method" << endl;

		Dio->solve_all_betten(verbose_level - 2);

		cout << "Found " << Dio->_resultanz << " solutions with "
				<< Dio->nb_steps_betten << " backtrack steps" << endl;

		if (TRUE) {
			char output_file[1000];

			strcpy(output_file, Descr->input_file);
			replace_extension_with(output_file, ".sol");


			Dio->write_solutions(output_file, verbose_level);
		}
	}
	else if (Descr->f_draw) {
		char fname_base[1000];
		int xmax_in = ONE_MILLION;
		int ymax_in = ONE_MILLION;
		int xmax_out = ONE_MILLION;
		int ymax_out = ONE_MILLION;

		sprintf(fname_base, "%s", Descr->input_file);
		replace_extension_with(fname_base, "_drawing");
		//Dio->draw_it(fname_base, xmax_in, ymax_in, xmax_out, ymax_out);
		Dio->draw_partitioned(fname_base,
			xmax_in, ymax_in, xmax_out, ymax_out,
			FALSE, 0, 0,
			verbose_level);
	}
	else if (Descr->f_perform_column_reductions) {
		diophant *D2;

		D2 = Dio->trivial_column_reductions(verbose_level);

		char fname2[1000];
		file_io Fio;

		sprintf(fname2, "%s", Descr->input_file);
		replace_extension_with(fname2, "_red.diophant");

		D2->save_in_general_format(fname2, verbose_level);
		cout << "Written file " << fname2 << " of size " << Fio.file_size(fname2) << endl;

	}
	else {
		cout << "diophant_activity::init no activity found" << endl;
		exit(1);
	}

	FREE_OBJECT(Dio);


	if (f_v) {
		cout << "diophant_activity::init done" << endl;
	}
}



}}
