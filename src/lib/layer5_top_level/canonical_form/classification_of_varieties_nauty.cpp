/*
 * classification_of_varieties_nauty.cpp
 *
 *  Created on: Jul 15, 2024
 *      Author: betten
 */








#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {


classification_of_varieties_nauty::classification_of_varieties_nauty()
{
	Classifier = NULL;

	nb_objects_to_test = 0;
	Input_Vo = NULL;

	//std::string fname_base;

	// nauty:
	CB = NULL;
	canonical_labeling_len = 0;
	//alpha = NULL;
	//gamma = NULL;
	F_first_time = NULL;
	Iso_idx = NULL;
	Idx_canonical_form = NULL;
	Idx_equation = NULL;
	nb_iso_orbits = 0;
	Orbit_input_idx = NULL;

	Classification_table_nauty = NULL;


	// stuff common to both algorithms:

	Variety_table = NULL;

	Elt = NULL;
	eqn2 = NULL;
	Goi = NULL;



}

classification_of_varieties_nauty::~classification_of_varieties_nauty()
{
	if (F_first_time) {
		FREE_int(F_first_time);
	}
	if (Iso_idx) {
		FREE_int(Iso_idx);
	}
	if (Idx_canonical_form) {
		FREE_int(Idx_canonical_form);
	}
	if (Idx_equation) {
		FREE_int(Idx_equation);
	}
	if (Orbit_input_idx) {
		FREE_int(Orbit_input_idx);
	}

	if (Classification_table_nauty) {
		FREE_int(Classification_table_nauty);
	}

}

void classification_of_varieties_nauty::init(
		int nb_objects_to_test,
		variety_object_with_action *Input_Vo,
		std::string &fname_base,
		canonical_form_classifier *Classifier,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::init " << endl;
	}
	if (f_v) {
		cout << "classification_of_varieties_nauty::init" << endl;
		cout << "classification_of_varieties_nauty::init "
				"nb_objects_to_test = " << nb_objects_to_test << endl;
	}

	classification_of_varieties_nauty::Classifier = Classifier;

	classification_of_varieties_nauty::nb_objects_to_test = nb_objects_to_test;
	classification_of_varieties_nauty::Input_Vo = Input_Vo;
	classification_of_varieties_nauty::fname_base.assign(fname_base);

	if (f_v) {
		cout << "classification_of_varieties_nauty::init "
				"nb_objects_to_test=" << nb_objects_to_test << endl;
	}


	Variety_table = (canonical_form_of_variety **)
			NEW_pvoid(nb_objects_to_test);

	Elt = NEW_int(Classifier->PA->A->elt_size_in_int);
	eqn2 = NEW_int(Classifier->Poly_ring->get_nb_monomials());
	Goi = NEW_lint(nb_objects_to_test);





#if 0
	if (f_v) {
		cout << "classification_of_varieties_nauty::init "
				"before classify_nauty" << endl;
	}
	classify_nauty(verbose_level - 1);
	if (f_v) {
		cout << "classification_of_varieties_nauty::init "
				"after classify_nauty" << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::init "
				"before write_classification_by_nauty_csv" << endl;
	}
	write_classification_by_nauty_csv(
			fname_base,
			verbose_level);
	if (f_v) {
		cout << "classification_of_varieties_nauty::init "
				"after write_classification_by_nauty_csv" << endl;
	}

#endif


	if (f_v) {
		cout << "classification_of_varieties_nauty::init done" << endl;
	}

}



void classification_of_varieties_nauty::classify_nauty(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty" << endl;
	}
	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty" << endl;
		cout << "classification_of_varieties_nauty::classify_nauty "
				"nb_objects_to_test = " << nb_objects_to_test << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty "
				"before allocate_tables" << endl;
	}
	allocate_tables(verbose_level);
	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty "
				"after allocate_tables" << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty "
				"before main_loop" << endl;
	}
	main_loop(verbose_level);
	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty "
				"after main_loop" << endl;
	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::classify_nauty "
				"The number of graph theoretic canonical forms is "
				<< CB->nb_types << endl;
	}


}

void classification_of_varieties_nauty::allocate_tables(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "classification_of_varieties_nauty::allocate_tables" << endl;
	}

	CB = NEW_OBJECT(canonical_form_classification::classify_bitvectors);


	// classification by nauty:

	F_first_time = NEW_int(nb_objects_to_test);
	Iso_idx = NEW_int(nb_objects_to_test);
	Idx_canonical_form = NEW_int(nb_objects_to_test);
	Idx_equation = NEW_int(nb_objects_to_test);
	Orbit_input_idx = NEW_int(nb_objects_to_test);
	nb_iso_orbits = 0;


	if (f_v) {
		cout << "classification_of_varieties_nauty::allocate_tables done" << endl;
	}
}


void classification_of_varieties_nauty::main_loop(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);



	if (f_v) {
		cout << "classification_of_varieties_nauty::main_loop" << endl;
		cout << "classification_of_varieties_nauty::main_loop "
				"nb_objects_to_test = " << nb_objects_to_test << endl;
	}


	int input_counter;
	int nb_iso = 0;

	for (input_counter = 0; input_counter < nb_objects_to_test; input_counter++) {


		string fname_case_out;

		fname_case_out = fname_base + "_cnt" + std::to_string(input_counter);

		canonical_form_of_variety *Variety;

		Variety = NEW_OBJECT(canonical_form_of_variety);

		if (f_v) {
			cout << "classification_of_varieties_nauty::main_loop "
					"input_counter = " << input_counter << " / " << nb_objects_to_test
					<< " before Variety->init" << endl;
		}
		Variety->init(
				Classifier,
				fname_case_out,
				input_counter,
				&Input_Vo[input_counter],
				verbose_level - 2);



		if (f_v) {
			cout << "classification_of_varieties_nauty::main_loop "
					"input_counter = " << input_counter << " / " << nb_objects_to_test
					<< " after Variety->init" << endl;
		}

		Variety_table[input_counter] = Variety;



		if (f_v) {
			cout << "classification_of_varieties_nauty::main_loop "
					"input_counter = " << input_counter << " / " << nb_objects_to_test << endl;
		}

#if 0
		if (Classifier->Input->skip_this_one(input_counter)) {
			if (f_v) {
				cout << "classification_of_varieties_nauty::main_loop "
						"skipping case input_counter = " << input_counter << endl;
			}
			//Variety_table[input_counter] = NULL;
			Iso_idx[input_counter] = -1;
			F_first_time[input_counter] = false;
			Goi[input_counter] = -1;
			//continue;
		}
		else {
#endif

			if (f_v) {
				cout << "classification_of_varieties_nauty::main_loop "
						"input_counter = " << input_counter << " / " << nb_objects_to_test
						<< " before Variety->compute_canonical_form_nauty_new" << endl;
			}
			Variety->compute_canonical_form_nauty_new(
					verbose_level);


			if (f_v) {
				cout << "classification_of_varieties_nauty::main_loop "
						"input_counter = " << input_counter << " / " << nb_objects_to_test
						<< " after Variety->compute_canonical_form_nauty_new" << endl;
			}

			Goi[input_counter] = Variety->Stabilizer_of_set_of_rational_points->Stab_gens_variety->group_order_as_lint();

			if (Variety->Stabilizer_of_set_of_rational_points->f_found_canonical_form
					&& Variety->Stabilizer_of_set_of_rational_points->f_found_eqn) {

				F_first_time[input_counter] = false;

			}
			else if (Variety->Stabilizer_of_set_of_rational_points->f_found_canonical_form
					&& !Variety->Stabilizer_of_set_of_rational_points->f_found_eqn) {

				F_first_time[input_counter] = true;

			}
			else if (!Variety->Stabilizer_of_set_of_rational_points->f_found_canonical_form) {

				F_first_time[input_counter] = true;

			}
			else {
				cout << "classification_of_varieties_nauty::main_loop illegal combination" << endl;
				exit(1);
			}

			Idx_canonical_form[input_counter] = Variety->Stabilizer_of_set_of_rational_points->idx_canonical_form;
			Idx_equation[input_counter] = Variety->Stabilizer_of_set_of_rational_points->idx_equation;

			if (F_first_time[input_counter]) {


				Iso_idx[input_counter] = nb_iso;

				int idx, i;


				for (i = 0; i < input_counter; i++) {
					idx = Idx_canonical_form[i];
					if (idx >= Variety->Stabilizer_of_set_of_rational_points->idx_canonical_form) {
						Idx_canonical_form[i]++;
					}
				}

				Orbit_input_idx[nb_iso_orbits] = input_counter;
				nb_iso_orbits++;

				nb_iso++;

			}
			else {
				Iso_idx[input_counter] = Iso_idx[Idx_canonical_form[input_counter]];
			}
#if 0
		}
#endif



		// Don't free Qco, because it is now stored in Variety_table[]
		//FREE_OBJECT(Qco);

	}

	if (f_v) {
		cout << "classification_of_varieties_nauty::main_loop done" << endl;
	}
}


void classification_of_varieties_nauty::write_classification_by_nauty_csv(
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	std::string fname;


	if (f_v) {
		cout << "classification_of_varieties_nauty::write_classification_by_nauty_csv" << endl;
	}
	fname = fname_base + "_classification_by_nauty.csv";



	{
		ofstream ost(fname);


		string header;

		header = stringify_csv_header_line_nauty(verbose_level);
		ost << header << endl;

		int input_counter;

		for (input_counter = 0; input_counter < nb_objects_to_test; input_counter++) {

			if (f_v) {
				cout << "classification_of_varieties_nauty::write_classification_by_nauty_csv "
						"input_counter=" << input_counter << " / " << nb_objects_to_test << endl;
			}

			string line;

			line = Variety_table[input_counter]->stringify_csv_entry_one_line_nauty_new(
					input_counter, verbose_level);

			ost << input_counter << "," << line << endl;


		}
		ost << "END" << endl;
	}


	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "classification_of_varieties_nauty::write_classification_by_nauty_csv done" << endl;
	}
}

std::string classification_of_varieties_nauty::stringify_csv_header_line_nauty(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_varieties_nauty::stringify_csv_header_line_nauty" << endl;
	}

	std::string header;

	header = "ROW,CNT,PO,SO,PO_GO,PO_INDEX,Iso_idx,F_Fst,Idx_canonical,Idx_eqn,Eqn,Eqn2,NPts,Pts,Bitangents";

#if 0
	if (Classifier->Descr->carry_through.size()) {
		int i;

		for (i = 0; i < Classifier->Descr->carry_through.size(); i++) {
			header += "," + Classifier->Descr->carry_through[i];
		}
	}
#endif

	header += ",NO_N,NO_ago,NO_base_len,NO_aut_cnt,NO_base,NO_tl,NO_aut,NO_cl,NO_stats";
	header += ",nb_eqn,ago";

	return header;
}


}}}

