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

	f_has_F2 = FALSE;
	F2 = NULL;
	tally_F2 = NULL;
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

void surfaces_arc_lifting_definition_node::tally_f2(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::tally_f2" << endl;
	}
	F2 = NEW_int(45 * 72);
	for (i = 0; i < 45 * 72; i++) {
		F2[i] = Seventytwo[i].f2;
	}
	tally_F2 = NEW_OBJECT(tally);
	tally_F2->init(F2, 45 * 72, FALSE, 0);

	f_has_F2 = TRUE;
	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::tally_f2 done" << endl;
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

void surfaces_arc_lifting_definition_node::report_cosets(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_cosets" << endl;
	}
	ost << "\\begin{enumerate}" << endl;
	for (i = 0; i < nb_coset_reps; i++) {

		ost << "\\item" << endl;
		ost << "Aut coset " << i << " / " << nb_coset_reps << ": relative order is "
				<< relative_order_table[i] << "\\\\" << endl;
		ost << "$$" << endl;
		Lift->A4->element_print_latex(coset_reps->ith(i), ost);
		ost << "$$" << endl;
	}
	ost << "\\end{enumerate}" << endl;


	//surfaces_arc_lifting_trace *T; // [nb_coset_reps]
	//vector_ge *coset_reps;

	//int *relative_order_table; // [nb_coset_reps]

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_cosets done" << endl;
	}
}

void surfaces_arc_lifting_definition_node::report_cosets_detailed(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_cosets_detailed" << endl;
	}

	ost << "We consider the Clebsch map " << endl;
	ost << "$$" << endl;
	ost << "\\Phi_{\\ell_1,\\ell_2,\\pi}" << endl;
	ost << "$$" << endl;
	ost << "with defining lines $\\ell_1,\\ell_2$ and image tritangent plane $\\pi$.\\\\" << endl;
	ost << "The line $m_1$ is in $\\pi$ and $\\ell_i \\cap \\pi =P_i \\in m_1.$\\\\" << endl;
	ost << "The four transversals besides $m_1$ to $\\ell_1,\\ell_2$ are called $t_3,t_4,t_5,t_6$.\\\\" << endl;
	//ost << "The points $P_1,P_2$ are the points of intersection of $(\\ell_1,\\ell_2)$ with the tritangent plane.\\\\" << endl;
	ost << "The points $P_3,P_4,P_5,P_6$ are the points of intersection of $t_3,t_4,t_5,t_6$ with the tritangent plane.\\\\" << endl;
	ost << "Alpha1 * Alpha2 * Beta1 * Beta2 * Beta3 = AutCoset\\\\" << endl;
	ost << "Alpha1 takes the chosen tritangent plane to the tritangent plane from the defining flag orbit.\\\\" << endl;
	ost << "Alpha2 maps the 6-arc to the canonical representative.\\\\" << endl;
	ost << "Beta1 maps the two points $P_1$ and $P_2$ defined by the lines $(\\ell_1,\\ell_2)$ to the canonical pair.\\\\" << endl;
	ost << "Beta2 maps the partition defined by $\\{\\{P_3,P_4\\}, \\{P_5,P_6\\}\\}$ to the canonical partition.\\\\" << endl;
	ost << "Beta3 maps the image of the lines $(\\ell_1,\\ell_2)$ under Alpha1*Alpha2*Beta1*Beta2 to the canonical pair.\\\\" << endl;
	ost << "\\begin{enumerate}" << endl;
	for (i = 0; i < nb_coset_reps; i++) {

		ost << "\\item" << endl;
		ost << "Aut coset " << i << " / " << nb_coset_reps << ": relative order is "
				<< relative_order_table[i] << "\\\\" << endl;
		ost << "$$" << endl;
		Lift->A4->element_print_latex(T[i]->Elt_Alpha1, ost);
		Lift->A4->element_print_latex(T[i]->Elt_Alpha2, ost);
		Lift->A4->element_print_latex(T[i]->Elt_Beta1, ost);
		Lift->A4->element_print_latex(T[i]->Elt_Beta2, ost);
		Lift->A4->element_print_latex(T[i]->Elt_Beta3, ost);
		ost << "=";
		Lift->A4->element_print_latex(coset_reps->ith(i), ost);
		ost << "$$" << endl;
	}
	ost << "\\end{enumerate}" << endl;


	//surfaces_arc_lifting_trace *T; // [nb_coset_reps]
	//vector_ge *coset_reps;

	//int *relative_order_table; // [nb_coset_reps]

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_cosets_detailed done" << endl;
	}
}


void surfaces_arc_lifting_definition_node::report_tally_F2(ostream &ost, int verbose_level)
{
	if (f_has_F2) {
		ost << "Tally of flag orbits: ";
		tally_F2->print_file_tex(ost, FALSE /* f_backwards */);
		ost << "\\\\" << endl;
	}
}

void surfaces_arc_lifting_definition_node::report_Clebsch_maps(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, j;

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_Clebsch_maps" << endl;
	}

	report_tally_F2(ost, verbose_level);

	for (t = 0; t < 45; t++) {
		ost << "\\clearpage" << endl;
		//ost << "Tritangent plane " << t << ": \\\\" << endl;
		for (i = 0; i < 3; i++) {
			report_seventytwo_maps_top(ost, t, i);
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

void surfaces_arc_lifting_definition_node::report_seventytwo_maps_top(ostream &ost, int t, int i)
{
	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|c|c|c|c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\multicolumn{9}{|c|}{\\mbox{Tritangent Plane}\\; \\pi_{" << t << "} = \\pi_{" << Lift->Surf->Eckard_point_label_tex[t] << "} \\; \\mbox{Part "<< i << "}}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Clebsch} & m_1-\\mbox{Case} & (m_1,m_2,m_3) & (\\ell_1,\\ell_2) & (t_3,t_4,t_5,t_6) & \\mbox{Arc} & \\mbox{Pair} & \\mbox{Part} &  \\mbox{Flag-Orb}  \\\\" << endl;
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


