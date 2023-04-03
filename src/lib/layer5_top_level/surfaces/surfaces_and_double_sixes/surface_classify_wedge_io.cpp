/*
 * surface_classify_wedge_io.cpp
 *
 *  Created on: Feb 17, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_double_sixes {




void surface_classify_wedge::write_file(
		std::ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::write_file" << endl;
	}
	fp.write((char *) &q, sizeof(int));

	Flag_orbits->write_file(fp, verbose_level);

	Surfaces->write_file(fp, verbose_level);

	if (f_v) {
		cout << "surface_classify_wedge::write_file finished" << endl;
	}
}

void surface_classify_wedge::read_file(
		std::ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q1;

	if (f_v) {
		cout << "surface_classify_wedge::read_file" << endl;
	}
	fp.read((char *) &q1, sizeof(int));
	if (q1 != q) {
		cout << "surface_classify_wedge::read_file q1 != q" << endl;
		exit(1);
	}

	Flag_orbits = NEW_OBJECT(invariant_relations::flag_orbits);
	Flag_orbits->read_file(fp, A, A2, verbose_level);

	Surfaces = NEW_OBJECT(invariant_relations::classification_step);

	ring_theory::longinteger_object go;

	A->group_order(go);

	Surfaces->read_file(fp, A, A2, go, verbose_level);

	if (f_v) {
		cout << "surface_classify_wedge::read_file finished" << endl;
	}
}







void surface_classify_wedge::generate_history(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::generate_history" << endl;
	}

	poset_classification::poset_classification_activity_description Activity_descr;
	poset_classification::poset_classification_activity Activity;


	if (f_v) {
		cout << "surface_classify_wedge::generate_history "
				"before Activity.init" << endl;
	}
	Activity.init(
			&Activity_descr,
			Five_p1->Five_plus_one,
			5 /* actual_size */,
			verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::generate_history "
				"after Activity.init" << endl;
	}


	if (f_v) {
		cout << "surface_classify_wedge::generate_history "
				"before Activity.generate_history" << endl;
	}
	Activity.generate_history(5, verbose_level - 2);
	if (f_v) {
		cout << "surface_classify_wedge::generate_history "
				"after Activity.generate_history" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::generate_history done" << endl;
	}

}

int surface_classify_wedge::test_if_surfaces_have_been_computed_already()
{
	char fname[1000];
	orbiter_kernel_system::file_io Fio;
	int ret;

	snprintf(fname, sizeof(fname), "Surfaces_q%d.data", q);
	if (Fio.file_size(fname) > 0) {
		//ret = true;
		ret = false; // !!! ToDo don't use data file
	}
	else {
		ret = false;
	}
	return ret;
}

void surface_classify_wedge::write_surfaces(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::write_surfaces" << endl;
	}
	char fname[1000];
	orbiter_kernel_system::file_io Fio;

	snprintf(fname, sizeof(fname), "Surfaces_q%d.data", q);
	{

		ofstream fp(fname);

		if (f_v) {
			cout << "surface_classify before write_file" << endl;
		}
		write_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after write_file" << endl;
		}
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "surface_classify_wedge::write_surfaces done" << endl;
	}
}

void surface_classify_wedge::read_surfaces(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::read_surfaces" << endl;
	}
	char fname[1000];
	orbiter_kernel_system::file_io Fio;

	snprintf(fname, sizeof(fname), "Surfaces_q%d.data", q);
	cout << "Reading file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	{
		ifstream fp(fname);

		if (f_v) {
			cout << "surface_classify_wedge::read_surfaces "
					"before read_file" << endl;
			}
		read_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify_wedge::read_surfaces "
					"after read_file" << endl;
		}
		if (f_v) {
			cout << "surface_classify_wedge::read_surfaces "
					"before post_process" << endl;
		}
		post_process(verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::read_surfaces "
					"after post_process" << endl;
		}

	}
	if (f_v) {
		cout << "surface_classify_wedge::read_surfaces done" << endl;
	}
}

int surface_classify_wedge::test_if_double_sixes_have_been_computed_already()
{
	char fname[1000];
	orbiter_kernel_system::file_io Fio;
	int ret;

	snprintf(fname, sizeof(fname), "Double_sixes_q%d.data", q);
	if (Fio.file_size(fname) > 0) {
		//ret = true;
		ret = false; // !!! ToDo
	}
	else {
		ret = false;
	}
	return ret;
}

void surface_classify_wedge::write_double_sixes(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::write_double_sixes" << endl;
	}
	char fname[1000];
	orbiter_kernel_system::file_io Fio;

	snprintf(fname, sizeof(fname), "Double_sixes_q%d.data", q);
	{

	ofstream fp(fname);

	if (f_v) {
		cout << "surface_classify before "
				"SCW->Classify_double_sixes->write_file" << endl;
		}
	Classify_double_sixes->write_file(fp, verbose_level - 1);
	if (f_v) {
		cout << "surface_classify after "
				"SCW->Classify_double_sixes->write_file" << endl;
		}
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "surface_classify_wedge::write_double_sixes done" << endl;
	}
}

void surface_classify_wedge::read_double_sixes(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::read_double_sixes" << endl;
	}
	char fname[1000];
	orbiter_kernel_system::file_io Fio;

	snprintf(fname, sizeof(fname), "Double_sixes_q%d.data", q);
	if (f_v) {
		cout << "Reading file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	{

		ifstream fp(fname);

		if (f_v) {
			cout << "surface_classify before "
					"SCW->Classify_double_sixes->read_file" << endl;
		}
		Classify_double_sixes->read_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after "
					"SCW->Classify_double_sixes->read_file" << endl;
		}
	}
	if (f_v) {
		cout << "surface_classify_wedge::read_double_sixes done" << endl;
	}
}


void surface_classify_wedge::create_report(
		int f_with_stabilizers,
		graphics::layered_graph_draw_options *draw_options,
		poset_classification::poset_classification_report_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::create_report" << endl;
	}
	char str[1000];
	string fname, title, author, extra_praeamble;
	orbiter_kernel_system::file_io Fio;

	snprintf(str, 1000, "Cubic Surfaces with 27 Lines over GF(%d) ", q);
	title.assign(str);

	strcpy(str, "Orbiter");
	author.assign(str);

	snprintf(str, 1000, "Surfaces_q%d.tex", q);
	fname.assign(str);


		{
		ofstream fp(fname);
		l1_interfaces::latex_interface L;

		//latex_head_easy(fp);
		L.head(fp,
			false /* f_book */,
			true /* f_title */,
			title, author,
			false /*f_toc */,
			false /* f_landscape */,
			false /* f_12pt */,
			true /*f_enlarged_page */,
			true /* f_pagenumbers*/,
			extra_praeamble /* extra_praeamble */);


		if (f_v) {
			cout << "surface_classify_wedge::create_report before report" << endl;
		}
		report(fp, f_with_stabilizers, draw_options, Opt, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify_wedge::create_report after report" << endl;
		}


		L.foot(fp);
		}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
}

void surface_classify_wedge::report(
		std::ostream &ost,
		int f_with_stabilizers,
		graphics::layered_graph_draw_options *draw_options,
		poset_classification::poset_classification_report_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::report" << endl;
	}
	l1_interfaces::latex_interface L;


#if 0
	ost << "\\section{The field of order " << LG->F->q << "}" << endl;
	ost << "\\noindent The field ${\\mathbb F}_{"
			<< LG->F->q
			<< "}$ :\\\\" << endl;
	LG->F->cheat_sheet(ost, verbose_level);
#endif

	if (f_v) {
		cout << "surface_classify_wedge::report "
				"before Five_p1->report" << endl;
	}
	Five_p1->report(ost, draw_options, Opt, verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::report "
				"after Five_p1->report" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::report "
				"before Classify_double_sixes->print_five_plus_ones" << endl;
	}
	Classify_double_sixes->print_five_plus_ones(ost);
	if (f_v) {
		cout << "surface_classify_wedge::report "
				"after Classify_double_sixes->print_five_plus_ones" << endl;
	}


	if (f_v) {
		cout << "surface_classify_wedge::report "
				"before Classify_double_sixes->Flag_orbits->print_latex" << endl;
	}

	{
		string title;

		title.assign("Flag orbits for double sixes");

		Classify_double_sixes->Flag_orbits->print_latex(ost, title, true);
	}
	if (f_v) {
		cout << "surface_classify_wedge::report "
				"after Classify_double_sixes->Flag_orbits->print_latex" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::report "
				"before Classify_double_sixes->Double_sixes->print_latex" << endl;
	}
	{
		string title;

		title.assign("Double Sixes");
		Classify_double_sixes->Double_sixes->print_latex(ost, title, true,
				false, NULL, NULL);
	}
	if (f_v) {
		cout << "surface_classify_wedge::report "
				"after Classify_double_sixes->Double_sixes->print_latex" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::report "
				"before Flag_orbits->print_latex" << endl;
	}
	{
		string title;

		title.assign("Flag orbits for cubic surfaces");

		Flag_orbits->print_latex(ost, title, true);
	}
	if (f_v) {
		cout << "surface_classify_wedge::report "
				"after Flag_orbits->print_latex" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::report "
				"before Surfaces->print_latex" << endl;
	}
	{
		string title;

		title.assign("Surfaces");
		Surfaces->print_latex(ost, title, true,
				false, NULL, NULL);
	}
	if (f_v) {
		cout << "surface_classify_wedge::report "
				"after Surfaces->print_latex" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::report "
				"before latex_surfaces" << endl;
	}
	latex_surfaces(ost, f_with_stabilizers, verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::report "
				"after latex_surfaces" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::report done" << endl;
	}
}

void surface_classify_wedge::latex_surfaces(
		std::ostream &ost,
		int f_with_stabilizers, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char str[1000];
	string title;

	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces" << endl;
	}
	snprintf(str, sizeof(str),
			"Cubic Surfaces with 27 Lines in $\\PG(3,%d)$", q);
	title.assign(str);


	ost << "\\subsection*{The Group $\\PGGL(4," << q << ")$}" << endl;

	{
		ring_theory::longinteger_object go;
		A->Strong_gens->group_order(go);

		ost << "The order of the group is ";
		go.print_not_scientific(ost);
		ost << "\\\\" << endl;

		ost << "\\bigskip" << endl;
	}

#if 0
	Classify_double_sixes->print_five_plus_ones(ost);


	Classify_double_sixes->Double_sixes->print_latex(ost, title_ds);
#endif

	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces "
				"before Surfaces->print_latex" << endl;
	}
	Surfaces->print_latex(
			ost, title, f_with_stabilizers,
			false, NULL, NULL);
	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces "
				"after Surfaces->print_latex" << endl;
	}


#if 1
	int orbit_index;

	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces "
				"before loop over all surfaces" << endl;
	}
	for (orbit_index = 0; orbit_index < Surface_repository->nb_surfaces; orbit_index++) {
		if (f_v) {
			cout << "surface_classify_wedge::latex_surfaces "
					"before report_surface, "
					"orbit_index = " << orbit_index << endl;
		}
		Surface_repository->report_surface(ost, orbit_index, verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::latex_surfaces "
					"after report_surface" << endl;
		}
	}
	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces "
				"after loop over all surfaces" << endl;
	}
#endif
	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces done" << endl;
	}
}


void surface_classify_wedge::create_report_double_sixes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::create_report_double_sixes" << endl;
	}


	char str[1000];
	string fname, title, author, extra_praeamble;

	snprintf(str, 1000, "Cheat Sheet on Double Sixes over GF(%d) ", q);
	title.assign(str);
	snprintf(str, 1000, "Double_sixes_q%d.tex", q);
	fname.assign(str);

	{
		ofstream fp(fname);
		l1_interfaces::latex_interface L;

		//latex_head_easy(fp);
		L.head(fp,
			false /* f_book */,
			true /* f_title */,
			title, author,
			false /*f_toc */,
			false /* f_landscape */,
			false /* f_12pt */,
			true /*f_enlarged_page */,
			true /* f_pagenumbers*/,
			extra_praeamble /* extra_praeamble */);


		if (f_v) {
			cout << "surface_classify_wedge::create_report_double_sixes "
					"before Classify_double_sixes->print_five_plus_ones" << endl;
		}
		Classify_double_sixes->print_five_plus_ones(fp);
		if (f_v) {
			cout << "surface_classify_wedge::create_report_double_sixes "
					"after Classify_double_sixes->print_five_plus_ones" << endl;
		}

		{
			string title;

			title.assign("Double Sixes");
			if (f_v) {
				cout << "surface_classify_wedge::create_report_double_sixes "
						"before Classify_double_sixes->Double_sixes->print_latex" << endl;
			}
			Classify_double_sixes->Double_sixes->print_latex(
					fp,
				title, false /* f_with_stabilizers*/,
				false, NULL, NULL);
			if (f_v) {
				cout << "surface_classify_wedge::create_report_double_sixes "
						"after Classify_double_sixes->Double_sixes->print_latex" << endl;
			}
		}

		L.foot(fp);
	}
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "surface_classify_wedge::create_report_double_sixes done" << endl;
	}
}





}}}}


