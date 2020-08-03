/*
 * surfaces_arc_lifting_definition_node.cpp
 *
 *  Created on: Aug 3, 2020
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

surfaces_arc_lifting_definition_node::surfaces_arc_lifting_definition_node()
{
	Lift = NULL;

	f = 0;
	orbit_idx = 0;

	SO = NULL;

	Flag_stab_gens = NULL;
	//longinteger_object Flag_stab_go;

	//int three_lines_idx[45 * 3];
	//long int three_lines[45 * 3];
	//seventytwo_cases Seventytwo[45 * 72];

	nb_coset_reps = 0;
	T = NULL;
	coset_reps = NULL;

	relative_order_table = NULL;
}


surfaces_arc_lifting_definition_node::~surfaces_arc_lifting_definition_node()
{
}

void surfaces_arc_lifting_definition_node::init(surfaces_arc_lifting *Lift,
		int f, int orbit_idx, surface_object *SO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::init" << endl;
	}

	surfaces_arc_lifting_definition_node::Lift = Lift;
	surfaces_arc_lifting_definition_node::f = f;
	surfaces_arc_lifting_definition_node::orbit_idx = orbit_idx;
	surfaces_arc_lifting_definition_node::SO = SO;


	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::init done" << endl;
	}
}

void surfaces_arc_lifting_definition_node::report(int verbose_level)
{
	char fname_base[1000];
	latex_interface L;

	sprintf(fname_base, "clebsch_maps_surface_%d", orbit_idx);

	char title[1000];
	char author[1000];
	string fname_report;

	fname_report.assign(fname_base);
	fname_report.append(".tex");
	snprintf(title, 1000, "Clebsch maps of surface %d", orbit_idx);
	strcpy(author, "");

	{
		ofstream fp(fname_report.c_str());
		L.head(fp,
			FALSE /* f_book */,
			TRUE /* f_title */,
			title, author,
			FALSE /*f_toc */,
			FALSE /* f_landscape */,
			FALSE /* f_12pt */,
			TRUE /*f_enlarged_page */,
			TRUE /* f_pagenumbers*/,
			NULL /* extra_praeamble */);


		report2(fp, verbose_level);

		L.foot(fp);

	}
	file_io Fio;

	cout << "Written file " << fname_report << " of size " << Fio.file_size(fname_report.c_str()) << endl;

}

void surfaces_arc_lifting_definition_node::report2(ostream &ost, int verbose_level)
{
	report_Clebsch_maps(ost, verbose_level);
}

void surfaces_arc_lifting_definition_node::report_Clebsch_maps(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, j;

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_Clebsch_maps" << endl;
	}
	for (t = 0; t < 45; t++) {
		ost << "Trihedral plane " << t << ": \\\\" << endl;
		for (i = 0; i < 3; i++) {
			report_seventytwo_maps_top(ost);
			for (j = 0; j < 24; j++) {
				report_seventytwo_maps_line(ost, Seventytwo + t * 72, i, j);
			}
			report_seventytwo_maps_bottom(ost);
		}
	}
	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_Clebsch_maps done" << endl;
	}
}

void surfaces_arc_lifting_definition_node::report_seventytwo_maps_top(ostream &ost)
{
	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|c|c|c|c|c|c|}" << endl;
	ost << "\\hline" << endl;

}

void surfaces_arc_lifting_definition_node::report_seventytwo_maps_bottom(ostream &ost)
{
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
	ost << "\\bigskip" << endl;
}

void surfaces_arc_lifting_definition_node::report_seventytwo_maps_line(ostream &ost, seventytwo_cases *S, int i, int j)
{
	int c;

	c = i * 24 + j;
	ost << c << " & ";
	ost << S[c].line_idx << " & ";
	ost << "(" << Lift->Surf_A->Surf->Line_label_tex[S[c].m1]
		<< ", " << Lift->Surf_A->Surf->Line_label_tex[S[c].m2]
		<< ", " << Lift->Surf_A->Surf->Line_label_tex[S[c].m3] << ") & ";
	ost << "(" << Lift->Surf_A->Surf->Line_label_tex[S[c].l1]
		<< ", " << Lift->Surf_A->Surf->Line_label_tex[S[c].l2] << ") & ";
	ost << "(" << Lift->Surf_A->Surf->Line_label_tex[S[c].transversals4[0]]
		<< ", " << Lift->Surf_A->Surf->Line_label_tex[S[c].transversals4[1]]
		<< ", " << Lift->Surf_A->Surf->Line_label_tex[S[c].transversals4[2]]
		<< ", " << Lift->Surf_A->Surf->Line_label_tex[S[c].transversals4[3]]
		<< ") & ";
	ost << S[c].orbit_not_on_conic_idx << " & ";
	ost << S[c].pair_orbit_idx << " & ";
	ost << S[c].partition_orbit_idx << " & ";
	ost << S[c].f2; // << " & ";
	ost << "\\\\";
}



}}


