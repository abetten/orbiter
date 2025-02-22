/*
 * surfaces_arc_lifting_definition_node.cpp
 *
 *  Created on: Aug 3, 2020
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_arcs {


surfaces_arc_lifting_definition_node::surfaces_arc_lifting_definition_node()
{
	Record_birth();
	Lift = NULL;

	f = 0;
	orbit_idx = 0;

	SO = NULL;
	SOA = NULL;

	Flag_stab_gens = NULL;
	//longinteger_object Flag_stab_go;

	//int three_lines_idx[45 * 3];
	//long int three_lines[45 * 3];
	//seventytwo_cases Seventytwo[45 * 72];

	nb_coset_reps = 0;
	T = NULL;
	coset_reps = NULL;

	relative_order_table = NULL;

	f_has_F2 = false;
	F2 = NULL;
	tally_F2 = NULL;
}


surfaces_arc_lifting_definition_node::~surfaces_arc_lifting_definition_node()
{
	Record_death();
}

void surfaces_arc_lifting_definition_node::init_with_27_lines(
		surfaces_arc_lifting *Lift,
		int f, int orbit_idx, long int *Lines27, int *eqn20,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::init_with_27_lines" << endl;
	}

	surfaces_arc_lifting_definition_node::Lift = Lift;
	surfaces_arc_lifting_definition_node::f = f;
	surfaces_arc_lifting_definition_node::orbit_idx = orbit_idx;

	string label_txt;
	string label_tex;

	label_txt = "arc_lifting";
	label_tex = "arc\\_lifting";

	SO = NEW_OBJECT(geometry::algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::init_with_27_lines before SO->init_with_27_lines" << endl;
	}
	SO->init_with_27_lines(Lift->Surf_A->Surf, Lines27, eqn20,
			label_txt, label_tex,
			false /* f_find_double_six_and_rearrange_lines */,
			verbose_level - 2);

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::init_with_27_lines after SO->init_with_27_lines" << endl;
	}


	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::init_with_27_lines done" << endl;
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
	tally_F2 = NEW_OBJECT(other::data_structures::tally);
	tally_F2->init(F2, 45 * 72, false, 0);

	f_has_F2 = true;
	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::tally_f2 done" << endl;
	}
}

void surfaces_arc_lifting_definition_node::report(
		int verbose_level)
{
	other::l1_interfaces::latex_interface L;

	string fname_base;

	fname_base = "clebsch_maps_surface_" + std::to_string(orbit_idx);

	string title, author, extra_praeamble;

	string fname_report;

	fname_report = fname_base + ".tex";

	title = "Clebsch maps of surface " + std::to_string(orbit_idx);


	{
		ofstream fp(fname_report);
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


		report2(fp, verbose_level);

		L.foot(fp);

	}
	other::orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;

}

void surfaces_arc_lifting_definition_node::report2(
		std::ostream &ost, int verbose_level)
{
	report_Clebsch_maps(ost, verbose_level);

	if (SOA) {
		SOA->cheat_sheet_basic(ost, verbose_level);
	}
}

void surfaces_arc_lifting_definition_node::report_cosets(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_cosets" << endl;
	}
	//ost << "\\begin{enumerate}" << endl;
	for (i = 0; i < nb_coset_reps; i++) {

		ost << endl << "\\bigskip" << endl << endl;
		//ost << "\\item" << endl;
		ost << "Aut coset " << i << " / " << nb_coset_reps
				<< ": relative order is "
				<< relative_order_table[i] << "\\\\" << endl;
		ost << "$$" << endl;
		Lift->A4->Group_element->element_print_latex(coset_reps->ith(i), ost);
		ost << "$$" << endl;
	}
	//ost << "\\end{enumerate}" << endl;


	//surfaces_arc_lifting_trace *T; // [nb_coset_reps]
	//vector_ge *coset_reps;

	//int *relative_order_table; // [nb_coset_reps]

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_cosets done" << endl;
	}
}

void surfaces_arc_lifting_definition_node::report_cosets_detailed(
		std::ostream &ost, int verbose_level)
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
	//ost << "\\begin{enumerate}" << endl;
	for (i = 0; i < nb_coset_reps; i++) {

		ost << endl << "\\bigskip" << endl << endl;
		//ost << "\\item" << endl;
		ost << "Aut coset " << i << " / " << nb_coset_reps << ": relative order is "
				<< relative_order_table[i] << "\\\\" << endl;

		T[i]->report_product(ost, coset_reps->ith(i), verbose_level);

		T[i]->The_case.report_single_Clebsch_map(ost, verbose_level);


		//SO->print_lines(ost);


		T[i]->The_case.report_Clebsch_map_details(ost, SO, verbose_level);

		//T[i]->report_product(ost, coset_reps->ith(i), verbose_level);

	}
	//ost << "\\end{enumerate}" << endl;


	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_cosets_detailed done" << endl;
	}
}

void surfaces_arc_lifting_definition_node::report_cosets_HDS(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int coset;

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_cosets_HDS" << endl;
	}

	report_HDS_top(ost);

	for (coset = 0; coset < nb_coset_reps; coset++) {

		//ost << "Aut coset " << i << " / " << nb_coset_reps << ": relative order is "
		//		<< relative_order_table[i] << "\\\\" << endl;

		//T[i]->The_case.report_single_Clebsch_map(ost, verbose_level);



		if ((coset % 24) == 0) {
			report_HDS_bottom(ost);
			report_HDS_top(ost);
		}


		T[coset]->The_case.report_Clebsch_map_aut_coset(ost, coset,
				relative_order_table[coset], verbose_level);


	}

	report_HDS_bottom(ost);


	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_cosets_HDS done" << endl;
	}
}


void surfaces_arc_lifting_definition_node::report_HDS_top(
		std::ostream &ost)
{
	int t = T[0]->The_case.tritangent_plane_idx;

	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|c|c|c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\multicolumn{8}{|c|}{\\mbox{Tritangent Plane}\\; \\pi_{" << t
			<< "} = \\pi_{" << Lift->Surf->Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[t] << "}}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Coset} & \\mbox{Clebsch} & (m_1,m_2,m_3) & "
			"(\\ell_1',\\ell_2') & (t_3',t_4',t_5',t_6') & DS & HDS & \\mbox{r.o.}\\\\" << endl;
	ost << "\\hline" << endl;
}

void surfaces_arc_lifting_definition_node::report_HDS_bottom(
		std::ostream &ost)
{
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
	ost << "\\bigskip" << endl;
}


void surfaces_arc_lifting_definition_node::report_cosets_T3(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int coset;

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_cosets_T3" << endl;
	}

	report_T3_top(ost);

	for (coset = 0; coset < nb_coset_reps; coset++) {

		//ost << "Aut coset " << i << " / " << nb_coset_reps << ": relative order is "
		//		<< relative_order_table[i] << "\\\\" << endl;

		//T[i]->The_case.report_single_Clebsch_map(ost, verbose_level);


		if ((coset % 24) == 0) {
			report_T3_bottom(ost);
			report_T3_top(ost);
		}

		ost << coset << " & ";
		//Lift->Surf->F->display_table_of_projective_points2(ost, T[coset]->The_case.P6, 6, 4);
		ost << " & \\\\";


	}

	report_T3_bottom(ost);


	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_cosets_T3 done" << endl;
	}
}

void surfaces_arc_lifting_definition_node::report_T3_top(
		std::ostream &ost)
{
	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Coset} & \\mbox{Arc} & T_3 \\\\" << endl;
	ost << "\\hline" << endl;
}

void surfaces_arc_lifting_definition_node::report_T3_bottom(
		std::ostream &ost)
{
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
	ost << "\\bigskip" << endl;
}



void surfaces_arc_lifting_definition_node::report_tally_F2(
		std::ostream &ost, int verbose_level)
{
	if (f_has_F2) {
		ost << "Tally of flag orbits: ";
		tally_F2->print_file_tex(ost, false /* f_backwards */);
		ost << "\\\\" << endl;
	}
}

void surfaces_arc_lifting_definition_node::report_Clebsch_maps(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_Clebsch_maps" << endl;
	}

	report_tally_F2(ost, verbose_level);

#if 1
	int plane_idx[] = {12, 30};
	int i;

	for (i = 0; i < sizeof(plane_idx) / sizeof(int); i++) {

		report_Clebsch_maps_for_one_tritangent_plane(ost,
				plane_idx[i], verbose_level);
	}

#else
	int t;
	for (t = 0; t < 45; t++) {
		ost << "\\clearpage" << endl;
		//ost << "Tritangent plane " << t << ": \\\\" << endl;
		report_Clebsch_maps_for_one_tritangent_plane(ost, t, verbose_level);
	}
#endif

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_Clebsch_maps done" << endl;
	}
}


void surfaces_arc_lifting_definition_node::report_Clebsch_maps_for_one_tritangent_plane(
		ostream &ost, int plane_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_Clebsch_maps_for_one_tritangent_plane" << endl;
	}

	for (i = 0; i < 3; i++) {
		Seventytwo[plane_idx * 72 + i * 24].report_seventytwo_maps_top(ost);
		for (j = 0; j < 24; j++) {
			Seventytwo[plane_idx * 72 + i * 24 + j].report_seventytwo_maps_line(ost);
		}
		Seventytwo[plane_idx * 72 + i * 24].report_seventytwo_maps_bottom(ost);
	}

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_Clebsch_maps_for_one_tritangent_plane done" << endl;
	}
}




}}}}



