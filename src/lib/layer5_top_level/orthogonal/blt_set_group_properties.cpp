 /*
 * blt_set_group_properties.cpp
 *
 *  Created on: Mar 27, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


blt_set_group_properties::blt_set_group_properties()
{
	Blt_set_with_action = NULL;

	A_on_points = NULL;
	Orbits_on_points = NULL;

	Flock = NULL;
	Point_idx = NULL;
}

blt_set_group_properties::~blt_set_group_properties()
{
	if (A_on_points) {
		FREE_OBJECT(A_on_points);
	}
	if (Orbits_on_points) {
		FREE_OBJECT(Orbits_on_points);
	}
	if (Flock) {
		FREE_OBJECTS(Flock);
	}
	if (Point_idx) {
		FREE_int(Point_idx);
	}
}

void blt_set_group_properties::init_blt_set_group_properties(
		blt_set_with_action *Blt_set_with_action,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_group_properties::init_blt_set_group_properties" << endl;
	}
	blt_set_group_properties::Blt_set_with_action = Blt_set_with_action;

	if (f_v) {
		cout << "blt_set_group_properties::init_blt_set_group_properties "
				"before init_orbits_on_points" << endl;
	}
	init_orbits_on_points(verbose_level - 1);
	if (f_v) {
		cout << "blt_set_group_properties::init_blt_set_group_properties "
				"after init_orbits_on_points" << endl;
	}

	if (f_v) {
		cout << "blt_set_group_properties::init_blt_set_group_properties "
				"before init_flocks" << endl;
	}
	init_flocks(verbose_level - 1);
	if (f_v) {
		cout << "blt_set_group_properties::init_blt_set_group_properties "
				"after init_flocks" << endl;
	}


	if (f_v) {
		cout << "blt_set_group_properties::init_blt_set_group_properties done" << endl;
	}
}

void blt_set_group_properties::init_orbits_on_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_with_action::init_orbits_on_points" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_on_points");
	label_of_set.assign("\\_on\\_points");


	if (f_v) {
		cout << "blt_set_with_action::init_orbits_on_points action "
				"on points:" << endl;
	}
	A_on_points = Blt_set_with_action->A->Induced_action->restricted_action(
			Blt_set_with_action->Inv->the_set_in_orthogonal,
			Blt_set_with_action->Blt_set_domain_with_action->Blt_set_domain->target_size,
			label_of_set, label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "blt_set_with_action::init_orbits_on_points action "
				"on points done" << endl;
	}


	Orbits_on_points = NEW_OBJECT(groups::orbits_on_something);

	int f_load_save = false;
	std::string prefix;

	if (f_v) {
		cout << "blt_set_group_properties::init_orbits_on_points "
				"before Orbits_on_points->init" << endl;
	}
	Orbits_on_points->init(
			A_on_points,
			Blt_set_with_action->Aut_gens,
			f_load_save,
			prefix,
			verbose_level);
	if (f_v) {
		cout << "blt_set_group_properties::init_orbits_on_points "
				"after Orbits_on_points->init" << endl;
	}

#if 0
	if (f_v) {
		cout << "blt_set_group_properties::init_orbits_on_points "
				"computing orbits on points:" << endl;
	}
	Orbits_on_points = Blt_set_with_action->Aut_gens->orbits_on_points_schreier(
			A_on_points, 0 /*verbose_level*/);
	if (f_v) {
		cout << "We found " << Orbits_on_points->nb_orbits
				<< " orbits on points" << endl;
	}
#endif

	if (f_v) {
		cout << "blt_set_group_properties::init_orbits_on_points "
				"Orbits on points:" << endl;
		Orbits_on_points->Sch->print_and_list_orbits(cout);
	}







	if (f_v) {
		cout << "blt_set_group_properties::init_orbits_on_points done" << endl;
	}
}

void blt_set_group_properties::init_flocks(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_with_action::init_flocks" << endl;
	}

	int orb_idx;
	int fst;
	int point_idx;

	if (f_v) {
		cout << "blt_set_with_action::init_flocks nb of orbits = " << Orbits_on_points->Sch->nb_orbits << endl;
	}


	Flock = NEW_OBJECTS(flock_from_blt_set, Orbits_on_points->Sch->nb_orbits);
	Point_idx = NEW_int(Orbits_on_points->Sch->nb_orbits);

	for (orb_idx = 0; orb_idx < Orbits_on_points->Sch->nb_orbits; orb_idx++) {

		if (f_v) {
			cout << "blt_set_with_action::init_flocks "
					"orbit " << orb_idx << " / " << Orbits_on_points->Sch->nb_orbits << endl;
		}

		fst = Orbits_on_points->Sch->orbit_first[orb_idx];
		point_idx = Orbits_on_points->Sch->orbit[fst];
		Point_idx[orb_idx] = point_idx;

		if (f_v) {
			cout << "blt_set_group_properties::init_flocks "
					"computing flock associated with orbit "
					<< orb_idx << " / " << Orbits_on_points->Sch->nb_orbits << endl;
		}

		if (f_v) {
			cout << "blt_set_group_properties::init_flocks "
					"before Flock->init" << endl;
		}
		Flock[orb_idx].init(Blt_set_with_action, point_idx, verbose_level);
		if (f_v) {
			cout << "blt_set_group_properties::init_flocks "
					"after Flock->init" << endl;
		}


	}

	if (f_v) {
		cout << "blt_set_with_action::init_flocks done" << endl;
	}
}


void blt_set_group_properties::print_automorphism_group(
	std::ostream &ost)
{
	if (Blt_set_with_action->Aut_gens) {
		ring_theory::longinteger_object go;

		Blt_set_with_action->Aut_gens->group_order(go);

		ost << "The automorphism group has order " << go << ".\\\\" << endl;
		ost << "\\bigskip" << endl;
		ost << "Orbits of the automorphism group on points "
				"of the BLT-set:\\\\" << endl;


		{
			string str;
			Orbits_on_points->Sch->print_orbit_length_distribution_to_string(str);
			ost << "Orbits on points: $" << str << "$\\\\" << endl;
		}


		Orbits_on_points->Sch->print_and_list_orbits_sorted_by_length_tex(ost);

		Orbits_on_points->report(ost, 0 /*verbose_level */);

	}
	else {
		ost << "The automorphism group is not available.\\\\" << endl;

	}
}

void blt_set_group_properties::report(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_group_properties::report" << endl;
	}


	print_summary(ost);


	print_automorphism_group(ost);

#if 0
	ost << "The stabilizer is generated by:\\\\" << endl;
	Blt_set_with_action->Aut_gens->print_generators_tex(ost);

	{
		string str;
		Orbits_on_points->Sch->print_orbit_length_distribution_to_string(str);
		ost << "Orbits on points: $" << str << "$\\\\" << endl;
	}
#endif



	Blt_set_with_action->report_basics(ost, verbose_level);

	ost << "\\subsection*{Flocks}" << endl;

	int orb_idx;

	for (orb_idx = 0; orb_idx < Orbits_on_points->Sch->nb_orbits; orb_idx++) {

		ost << "\\subsubsection*{Flock " << orb_idx << " / " << Orbits_on_points->Sch->nb_orbits << "}" << endl;

		Flock[orb_idx].report(ost, verbose_level);

	}

	if (f_v) {
		cout << "blt_set_group_properties::report done" << endl;
	}
}

void blt_set_group_properties::print_summary(
		std::ostream &ost)
{
	ost << "\\subsection*{Summary}" << endl;


	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|l|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Object} & \\mbox{Number}  & \\mbox{Orbit type} \\\\";
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Points} & " << Blt_set_with_action->Blt_set_domain_with_action->Blt_set_domain->target_size << " & ";
	{
		string str;
		Orbits_on_points->Sch->print_orbit_length_distribution_to_string(str);
		ost << str;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
}


}}}



