/*
 * spread_activity.cpp
 *
 *  Created on: Sep 19, 2022
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {


spread_activity::spread_activity()
{
	Descr = NULL;
	Spread_create = NULL;
	SD = NULL;
	A = NULL;
	A2 = NULL;
	AG = NULL;
	AGr = NULL;
}

spread_activity::~spread_activity()
{
}

void spread_activity::init(
		spread_activity_description *Descr,
		spread_create *Spread_create,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_activity::init" << endl;
	}

	spread_activity::Descr = Descr;
	spread_activity::Spread_create = Spread_create;

	SD = NEW_OBJECT(geometry::spread_domain);


	if (f_v) {
		cout << "spread_activity::init before SD->init" << endl;
	}

	SD->init(
			Spread_create->F,
			2 * Spread_create->k, Spread_create->k,
			verbose_level - 1);

	if (f_v) {
		cout << "spread_activity::init after SD->init" << endl;
	}



	A = Spread_create->A;

	A2 = NEW_OBJECT(actions::action);
	AG = NEW_OBJECT(induced_actions::action_on_grassmannian);

#if 0
	longinteger_object go;
	A->Sims->group_order(go);
	if (f_v) {
		cout << "spread_activity::init go = " << go <<  endl;
	}
#endif


	if (f_v) {
		cout << "action A: ";
		A->print_info();
	}





	if (f_v) {
		cout << "spread_activity::init before AG->init" <<  endl;
	}

	AG->init(*A, SD->Grass, 0 /*verbose_level - 2*/);

	if (f_v) {
		cout << "spread_activity::init after AG->init" <<  endl;
	}

	if (f_v) {
		cout << "spread_activity::init before "
				"A2->induced_action_on_grassmannian" <<  endl;
	}

	A2->induced_action_on_grassmannian(A, AG,
		FALSE /*f_induce_action*/, NULL /*sims *old_G */,
		0 /*verbose_level - 2*/);

	if (f_v) {
		cout << "spread_activity::init after "
				"A2->induced_action_on_grassmannian" <<  endl;
	}

	if (f_v) {
		cout << "action A2 created: ";
		A2->print_info();
	}

	AGr = A2->restricted_action(Spread_create->set, Spread_create->sz,
			verbose_level);

	if (f_v) {
		cout << "action AGr created: ";
		AGr->print_info();
	}


	if (f_v) {
		cout << "spread_activity::init done" << endl;
	}
}


void spread_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_activity::perform_activity" << endl;
	}


	if (Descr->f_report) {

		if (f_v) {
			cout << "spread_activity::perform_activity f_report" << endl;
		}


		if (f_v) {
			cout << "spread_classify_activity::perform_activity before Spread_classify->classify_partial_spreads" << endl;
		}
		report(verbose_level);
		if (f_v) {
			cout << "spread_classify_activity::perform_activity after Spread_classify->classify_partial_spreads" << endl;
		}

		if (f_v) {
			cout << "spread_classify_activity::perform_activity f_report done" << endl;
		}

	}


	if (f_v) {
		cout << "spread_activity::perform_activity done" << endl;
	}

}

void spread_activity::report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_activity::report" << endl;
	}

	string fname;

	fname.assign(Spread_create->label_txt);
	fname.append("_report.tex");

	char str[1000];
	string title, author, extra_praeamble;

	snprintf(str, 1000, "Translation plane %s", Spread_create->label_tex.c_str());
	title.assign(str);


	{
		ofstream ost(fname);

		orbiter_kernel_system::latex_interface L;

		L.head(ost,
				FALSE /* f_book*/,
				TRUE /* f_title */,
				title, author,
				FALSE /* f_toc */,
				FALSE /* f_landscape */,
				TRUE /* f_12pt */,
				TRUE /* f_enlarged_page */,
				TRUE /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);


		if (f_v) {
			cout << "spread_activity::report before report2" << endl;
		}
		report2(ost, verbose_level);
		if (f_v) {
			cout << "spread_activity::report after report2" << endl;
		}


		L.foot(ost);
	}


	if (f_v) {
		cout << "spread_activity::report done" << endl;
	}
}

void spread_activity::report2(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_activity::report2" << endl;
		cout << "spread_activity::report2 spread_size=" << SD->spread_size << endl;
	}


	ost << "The spread: \\\\" << endl;

	if (f_v) {
		cout << "spread_activity::report2 before Grass->print_set_tex" << endl;
	}
	SD->Grass->print_set_tex(ost, Spread_create->set, Spread_create->sz, verbose_level);
	if (f_v) {
		cout << "spread_activity::report2 after Grass->print_set_tex" << endl;
	}


	int *Spread_set;
	int sz, i, k, k2;
	k = Spread_create->k;
	k2 = k * k;
	orbiter_kernel_system::latex_interface Li;

	if (f_v) {
		cout << "spread_activity::report2 before SD->Grass->make_spread_set_from_spread" << endl;
	}

	SD->Grass->make_spread_set_from_spread(
			Spread_create->set, Spread_create->sz,
			Spread_set, sz,
			verbose_level);

	if (f_v) {
		cout << "spread_activity::report2 after SD->Grass->make_spread_set_from_spread" << endl;
	}

	ost << "The spread set: \\\\" << endl;
	for (i = 0; i < sz; i++) {
		ost << "$";
		ost << "\\left[";
		Li.print_integer_matrix_tex(ost, Spread_set + i * k2, k, k);
		ost << "\\right]";
		ost << "$";
		if (i < sz - 1) {
			ost << ", ";
		}
	}
	ost << "\\\\" << endl;


	if (Spread_create->f_has_group) {

		ring_theory::longinteger_object go;

		Spread_create->Sg->group_order(go);

		ost << "The spread stabilizer has order: " << go << " \\\\" << endl;

		Spread_create->Sg->print_generators_tex(ost);


		groups::orbits_on_something *O1;
		groups::orbits_on_something *O2;
		string prefix1, prefix2;

		prefix1.assign(Spread_create->label_txt);
		prefix1.append("_on_gr");
		prefix2.assign(Spread_create->label_txt);
		prefix2.append("_on_spread");

		O1 = NEW_OBJECT(groups::orbits_on_something);

		O1->init(A2,
				Spread_create->Sg,
				FALSE /* f_load_save */,
				prefix1,
				verbose_level);

		ost << "Orbits on grassmannian: ";
		O1->Sch->print_orbit_lengths_tex(ost);
		ost << "\\\\" << endl;

		O2 = NEW_OBJECT(groups::orbits_on_something);

		O2->init(AGr,
				Spread_create->Sg,
				FALSE /* f_load_save */,
				prefix2,
				verbose_level);


		ost << "Orbits on spread: ";
		O2->Sch->print_orbit_lengths_tex(ost);
		ost << "\\\\" << endl;

	}

	if (f_v) {
		cout << "spread_activity::report2 done" << endl;
	}
}




}}}




