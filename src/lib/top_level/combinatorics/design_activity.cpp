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



	if (f_v) {
		cout << "design_activity::perform_activity done" << endl;
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



}}



