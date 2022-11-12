/*
 * orbits_activity.cpp
 *
 *  Created on: Nov 8, 2022
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


orbits_activity::orbits_activity()
{
	Descr = NULL;

	OC = NULL;


}

orbits_activity::~orbits_activity()
{

}

void orbits_activity::init(orbits_activity_description *Descr,
		apps_algebra::orbits_create *OC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::init" << endl;
	}

	orbits_activity::Descr = Descr;
	orbits_activity::OC = OC;

	if (f_v) {
		cout << "orbits_activity::init done" << endl;
	}
}


void orbits_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::perform_activity" << endl;
	}

	if (Descr->f_report) {
		do_report(verbose_level);
	}
	if (Descr->f_export_something) {
		do_export(verbose_level);

	}
	else if (Descr->f_export_trees) {
		do_export_trees(verbose_level);
	}
	else if (Descr->f_draw_tree) {
		do_draw_tree(verbose_level);
	}
	else if (Descr->f_stabilizer) {
		do_stabilizer(verbose_level);
	}
	else if (Descr->f_stabilizer_of_orbit_rep) {
		do_stabilizer_of_orbit_rep(verbose_level);
	}


	if (f_v) {
		cout << "orbits_activity::perform_activity done" << endl;
	}
}

void orbits_activity::do_report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_report" << endl;
	}


	if (OC->f_has_Orb) {
		if (f_v) {
			cout << "orbits_activity::do_report "
					"before OC->Orb->create_latex_report" << endl;
		}
		OC->Orb->create_latex_report(verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_report "
					"after OC->Orb->create_latex_report" << endl;
		}
	}
	else if (OC->f_has_On_polynomials) {

		OC->On_polynomials->report(verbose_level);


	}
	else {
		cout << "orbits_activity::do_report no suitable data structure" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "orbits_activity::do_report done" << endl;
	}

}

void orbits_activity::do_export(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_export" << endl;
	}


	if (OC->f_has_Orb) {

		string fname;

		if (f_v) {
			cout << "orbits_activity::do_export "
					"before OC->Orb->export_something" << endl;
		}

		OC->Orb->export_something(Descr->export_something_what,
				Descr->export_something_data1, fname, verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_export "
					"after OC->Orb->export_something" << endl;
		}

		orbiter_kernel_system::file_io Fio;

		cout << "orbits_activity::do_export "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

	}

	else if (OC->f_has_On_polynomials) {

		string fname;

		if (f_v) {
			cout << "orbits_activity::do_export "
					"before OC->On_polynomials->export_something" << endl;
		}

		OC->On_polynomials->export_something(Descr->export_something_what,
				Descr->export_something_data1, fname, verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_export "
					"after OC->On_polynomials->export_something" << endl;
		}

		orbiter_kernel_system::file_io Fio;

		cout << "orbits_activity::do_export "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;


	}

	else {
		cout << "orbits_activity::do_export no suitable data structure" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "orbits_activity::do_export done" << endl;
	}

}

void orbits_activity::do_export_trees(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_export_trees" << endl;
	}

	if (OC->f_has_Orb) {
		string fname_tree_mask;
		int orbit_idx;

		fname_tree_mask.assign("orbit_");
		fname_tree_mask.append(OC->Group->A->label);
		fname_tree_mask.append("_%d.layered_graph");

		for (orbit_idx = 0; orbit_idx < OC->Orb->Sch->nb_orbits; orbit_idx++) {

			cout << "orbit " << orbit_idx << " / " <<  OC->Orb->Sch->nb_orbits
					<< " before Sch->export_tree_as_layered_graph" << endl;

			OC->Orb->Sch->export_tree_as_layered_graph(orbit_idx,
					fname_tree_mask,
					verbose_level - 1);
		}

	}
	else {
		cout << "orbits_activity::do_export_trees no suitable data structure" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_export_trees done" << endl;
	}

}

void orbits_activity::do_draw_tree(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_draw_tree" << endl;
	}

	if (f_v) {
		cout << "orbits_activity::do_draw_tree tree index = " << Descr->draw_tree_idx << endl;
	}


	if (OC->f_has_Orb) {
		string fname;
		char str[1000];

		snprintf(str, sizeof(str), "_orbit_%d_tree", Descr->draw_tree_idx);

		fname.assign(OC->Orb->prefix);
		fname.append(str);

		OC->Orb->Sch->draw_tree(fname,
				orbiter_kernel_system::Orbiter->draw_options,
				Descr->draw_tree_idx,
				FALSE /* f_has_point_labels */, NULL /* long int *point_labels*/,
				verbose_level);
	}
	else if (OC->f_has_On_polynomials) {

		string fname;
		char str[1000];


		snprintf(str, sizeof(str), "_orbit_%d_tree", Descr->draw_tree_idx);

		fname.assign(OC->prefix);
		fname.append(str);

		OC->On_polynomials->Sch->draw_tree(fname,
				orbiter_kernel_system::Orbiter->draw_options,
				Descr->draw_tree_idx,
				FALSE /* f_has_point_labels */, NULL /* long int *point_labels*/,
				verbose_level);


	}
	else {
		cout << "orbits_activity::do_draw_tree no suitable data structure" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_draw_tree done" << endl;
	}
}

void orbits_activity::do_stabilizer(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_stabilizer" << endl;
	}

	if (OC->f_has_Orb) {

		groups::strong_generators *Stab;

		if (f_v) {
			cout << "orbits_activity::do_stabilizer before Orb->stabilizer_of" << endl;
		}
		//OC->Orb->stabilizer_of(Descr->stabilizer_orbit_idx, verbose_level);

		OC->Orb->stabilizer_any_point(Descr->stabilizer_point,
				Stab, verbose_level);


		if (f_v) {
			cout << "orbits_activity::do_stabilizer after Orb->stabilizer_of" << endl;
		}


		std::string gens_str;
		ring_theory::longinteger_object stab_go;


		Stab->get_gens_data_as_string_with_quotes(gens_str, 0 /*verbose_level*/);
		Stab->group_order(stab_go);
		if (f_v) {
			cout << "orbits_activity::do_stabilizer The stabilizer has order " << stab_go << endl;
			cout << "orbits_activity::do_stabilizer Number of generators " << Stab->gens->len << endl;
			cout << "orbits_activity::do_stabilizer Generators for the stabilizer in coded form: " << endl;
			cout << gens_str << endl;
		}

		string fname_stab;
		string label_stab;
		char str[1000];

		snprintf(str, sizeof(str), "_stab_pt_%d", Descr->stabilizer_point);



		fname_stab.assign(OC->prefix);
		fname_stab.append(str);
		fname_stab.append(".makefile");

		label_stab.assign(OC->prefix);
		label_stab.append(str);

		Stab->report_group(label_stab, verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_stabilizer "
					"exporting stabilizer orbit representative "
					"of point " << Descr->stabilizer_point << " to " << fname_stab << endl;
		}
		Stab->export_to_orbiter_as_bsgs(
				Stab->A,
				fname_stab, label_stab, label_stab,
				verbose_level);

		FREE_OBJECT(Stab);

	}
	else {
		cout << "orbits_activity::do_stabilizer no suitable data structure" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_stabilizer done" << endl;
	}

}

void orbits_activity::do_stabilizer_of_orbit_rep(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_stabilizer_of_orbit_rep" << endl;
	}

	if (OC->f_has_Orb) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before OC->Orb->stabilizer_of" << endl;
		}
		OC->Orb->stabilizer_of(Descr->stabilizer_of_orbit_rep_orbit_idx, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after OC->Orb->stabilizer_of" << endl;
		}
	}
	else {
		cout << "orbits_activity::do_stabilizer_of_orbit_rep no suitable data structure" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_stabilizer_of_orbit_rep done" << endl;
	}

}

}}}




