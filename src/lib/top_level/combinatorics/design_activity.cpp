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


}}



