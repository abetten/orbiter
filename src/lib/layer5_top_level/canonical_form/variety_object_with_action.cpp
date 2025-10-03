/*
 * variety_object_with_action.cpp
 *
 *  Created on: Dec 11, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



variety_object_with_action::variety_object_with_action()
{
	Record_birth();

	PA = NULL;

	cnt = 0;
	po_go = 0;
	po_index = 0;
	po = 0;
	so = 0;

	f_has_nauty_output = false;
	nauty_output_index_start = 0;

	Variety_object = NULL;

	f_has_automorphism_group = false;
	Stab_gens = NULL;

	f_has_set_stabilizer = false;
	Set_stab_gens = NULL;

	TD = NULL;

	TD_set_stabilizer = NULL;

}

variety_object_with_action::~variety_object_with_action()
{
	Record_death();
	if (Variety_object) {
		FREE_OBJECT(Variety_object);
	}
	if (f_has_automorphism_group && Stab_gens) {
		FREE_OBJECT(Stab_gens);
	}
	if (f_has_set_stabilizer && Set_stab_gens) {
		FREE_OBJECT(Set_stab_gens);
	}
	if (TD) {
		FREE_OBJECT(TD);
	}
	if (TD_set_stabilizer) {
		FREE_OBJECT(TD_set_stabilizer);
	}
}

void variety_object_with_action::create_variety(
		projective_geometry::projective_space_with_action *PA,
		int cnt, int po_go, int po_index, int po, int so,
		geometry::algebraic_geometry::variety_description *VD,
		int verbose_level)
// computes the tactical decomposition as well
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::create_variety" << endl;
	}

	other::data_structures::string_tools ST;


	variety_object_with_action::PA = PA;
	variety_object_with_action::cnt = cnt;
	variety_object_with_action::po_go = po_go;
	variety_object_with_action::po_index = po_index;
	variety_object_with_action::po = po;
	variety_object_with_action::so = so;


	if (VD->f_projective_space) {
		VD->f_projective_space_pointer = true;
		VD->Projective_space_pointer = Get_projective_space(VD->projective_space_label)->P;
	}


#if 0
	if (VD->f_bitangents == false) {
		VD->f_bitangents = true;
		VD->bitangents_txt = "";
	}
#endif


	Variety_object = NEW_OBJECT(geometry::algebraic_geometry::variety_object);


	if (f_v) {
		cout << "variety_object_with_action::create_variety "
				"before Variety_object->init" << endl;
	}
	Variety_object->init(
			VD,
			verbose_level);
	if (f_v) {
		cout << "variety_object_with_action::create_variety "
				"after Variety_object->init" << endl;
	}

#if 0
	int i;

	for (i = 0; i < VD->transformations.size(); i++) {
		if (f_v) {
			cout << "variety_object_with_action::create_variety "
					"transformation " << i << " / " << VD->transformations.size() << endl;
		}

		int *data;
		int *Elt;
		int sz;

		Elt = NEW_int(PA->A->elt_size_in_int);


		Int_vec_scan(VD->transformations[i], data, sz);
		PA->A->Group_element->make_element(Elt, data, 0 /* verbose_level */);



		if (VD->transformation_inverse[i]) {

			if (f_v) {
				cout << "variety_object_with_action::create_variety "
						"-transform_inverse " << VD->transformations[i] << endl;
			}
			int *Elt1;

			Elt1 = NEW_int(PA->A->elt_size_in_int);
			PA->A->Group_element->element_invert(Elt, Elt1, 0 /* verbose_level */);
			PA->A->Group_element->element_move(Elt1, Elt, 0 /* verbose_level */);
			FREE_int(Elt1);
		}
		else {
			if (f_v) {
				cout << "variety_object_with_action::create_variety "
						"-transform " << VD->transformations[i] << endl;
			}

		}


		if (f_v) {
			cout << "variety_object_with_action::create_variety "
					"before apply_transformation_to_self" << endl;
		}

		apply_transformation_to_self(
				Elt,
				PA->A,
				PA->A_on_lines,
				verbose_level - 2);

		if (f_v) {
			cout << "variety_object_with_action::create_variety "
					"after apply_transformation_to_self" << endl;
		}

		FREE_int(data);

	}
#else
	if (f_v) {
		cout << "variety_object_with_action::create_variety "
				"before apply_transformations_if_necessary" << endl;
	}
	apply_transformations_if_necessary(VD, verbose_level - 2);
	if (f_v) {
		cout << "variety_object_with_action::create_variety "
				"after apply_transformations_if_necessary" << endl;
	}
#endif

	if (f_v) {
		cout << "variety_object_with_action::create_variety "
				"before compute_tactical_decompositions" << endl;
	}
	compute_tactical_decompositions(
			verbose_level - 2);
	if (f_v) {
		cout << "variety_object_with_action::create_variety "
				"after compute_tactical_decompositions" << endl;
	}


	if (f_v) {
		cout << "variety_object_with_action::create_variety before print" << endl;
		print(cout);
		cout << "variety_object_with_action::create_variety after print" << endl;
	}

	if (f_v) {
		cout << "variety_object_with_action::create_variety done" << endl;
	}
}

void variety_object_with_action::apply_transformations_if_necessary(
		geometry::algebraic_geometry::variety_description *VD,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::apply_transformations_if_necessary" << endl;
	}

	int i;

	for (i = 0; i < VD->transformations.size(); i++) {
		if (f_v) {
			cout << "variety_object_with_action::create_variety "
					"transformation " << i << " / " << VD->transformations.size() << endl;
		}

		int *data;
		int *Elt;
		int sz;

		Elt = NEW_int(PA->A->elt_size_in_int);


		Int_vec_scan(VD->transformations[i], data, sz);
		PA->A->Group_element->make_element(Elt, data, 0 /* verbose_level */);



		if (VD->transformation_inverse[i]) {

			if (f_v) {
				cout << "variety_object_with_action::create_variety "
						"-transform_inverse " << VD->transformations[i] << endl;
			}
			int *Elt1;

			Elt1 = NEW_int(PA->A->elt_size_in_int);
			PA->A->Group_element->element_invert(Elt, Elt1, 0 /* verbose_level */);
			PA->A->Group_element->element_move(Elt1, Elt, 0 /* verbose_level */);
			FREE_int(Elt1);
		}
		else {
			if (f_v) {
				cout << "variety_object_with_action::create_variety "
						"-transform " << VD->transformations[i] << endl;
			}

		}


		if (f_v) {
			cout << "variety_object_with_action::create_variety "
					"before apply_transformation_to_self" << endl;
		}

		apply_transformation_to_self(
				Elt,
				PA->A,
				PA->A_on_lines,
				verbose_level - 2);

		if (f_v) {
			cout << "variety_object_with_action::create_variety "
					"after apply_transformation_to_self" << endl;
		}

		FREE_int(data);

	}


}
void variety_object_with_action::apply_transformation_to_self(
		int *Elt,
		actions::action *A,
		actions::action *A_on_lines,
		int verbose_level)
// Creates an action on the homogeneous polynomials on the fly
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::apply_transformation_to_self" << endl;
	}



	geometry::algebraic_geometry::variety_object *old_Variety_object;

	old_Variety_object = Variety_object;

	Variety_object = NEW_OBJECT(geometry::algebraic_geometry::variety_object);


	Variety_object->Descr = old_Variety_object->Descr;
	Variety_object->Projective_space = old_Variety_object->Projective_space;
	Variety_object->Ring = old_Variety_object->Ring;
	Variety_object->label_txt = old_Variety_object->label_txt;
	Variety_object->label_tex = old_Variety_object->label_tex;

	Variety_object->eqn = NEW_int(old_Variety_object->Ring->get_nb_monomials());


	actions::action *A_on_equations;
	induced_actions::action_on_homogeneous_polynomials *AonHPD;

	A_on_equations = A->Induced_action->induced_action_on_homogeneous_polynomials(
			Variety_object->Ring,
		false /* f_induce_action */, NULL,
		verbose_level - 2);

	AonHPD = A_on_equations->G.OnHP;

	if (f_v) {
		cout << "created action A_on_equations" << endl;
		A_on_equations->print_info();
	}

	AonHPD->compute_image_int_low_level(
		Elt,
		old_Variety_object->eqn,
		Variety_object->eqn,
		verbose_level - 2);


	FREE_OBJECT(A_on_equations);

	actions::action_global AG;


	if (f_v) {
		cout << "variety_object_with_action::apply_transformation_to_self "
				"before AG.set_of_sets_copy_and_apply, Point_sets" << endl;
	}
	Variety_object->Point_sets = AG.set_of_sets_copy_and_apply(
			A,
			Elt,
			old_Variety_object->Point_sets,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_object_with_action::apply_transformation_to_self "
				"after AG.set_of_sets_copy_and_apply, Point_sets" << endl;
	}

	// we are sorting the points:

	Variety_object->Point_sets->sort();

	if (f_v) {
		cout << "variety_object_with_action::apply_transformation_to_self "
				"before AG.set_of_sets_copy_and_apply, Line_sets" << endl;
	}
	Variety_object->Line_sets = AG.set_of_sets_copy_and_apply(
			A_on_lines,
			Elt,
			old_Variety_object->Line_sets,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_object_with_action::apply_transformation_to_self "
				"after AG.set_of_sets_copy_and_apply, Line_sets" << endl;
	}

	// We are not sorting the lines because the lines are often in the Schlaefli ordering

	FREE_OBJECT(old_Variety_object);



	if (f_v) {
		cout << "variety_object_with_action::apply_transformation_to_self "
				"after transforming:" << endl;
		print(cout);
	}

	if (f_v) {
		cout << "variety_object_with_action::apply_transformation_to_self done" << endl;
	}
}

void variety_object_with_action::compute_tactical_decompositions(
		int verbose_level)
// computes the tactical decomposition TD if f_has_automorphism_group is true
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::compute_tactical_decompositions" << endl;
	}


	if (f_has_automorphism_group && Stab_gens) {

		TD = NEW_OBJECT(apps_combinatorics::variety_with_TDO_and_TDA);


		if (f_v) {
			cout << "variety_object_with_action::compute_tactical_decompositions "
					"before TD->init_and_compute_tactical_decompositions" << endl;
		}
		TD->init_and_compute_tactical_decompositions(
				PA, Variety_object, Stab_gens,
				verbose_level);
		if (f_v) {
			cout << "variety_object_with_action::compute_tactical_decompositions "
					"after TD->init_and_compute_tactical_decompositions" << endl;
		}
	}
	else {
		cout << "variety_object_with_action::compute_tactical_decompositions "
				"the automorphism group is not available" << endl;
		TD = NULL;
	}




	if (f_v) {
		cout << "variety_object_with_action::compute_tactical_decompositions done" << endl;
	}


}


void variety_object_with_action::compute_tactical_decompositions_wrt_set_stabilizer(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::compute_tactical_decompositions_wrt_set_stabilizer" << endl;
	}


	if (f_has_set_stabilizer && Set_stab_gens) {

		TD_set_stabilizer = NEW_OBJECT(apps_combinatorics::variety_with_TDO_and_TDA);


		if (f_v) {
			cout << "variety_object_with_action::compute_tactical_decompositions_wrt_set_stabilizer "
					"before TD_set_stabilizer->init_and_compute_tactical_decompositions" << endl;
		}
		TD_set_stabilizer->init_and_compute_tactical_decompositions(
				PA, Variety_object, Set_stab_gens,
				verbose_level);
		if (f_v) {
			cout << "variety_object_with_action::compute_tactical_decompositions_wrt_set_stabilizer "
					"after TD_set_stabilizer->init_and_compute_tactical_decompositions" << endl;
		}
	}
	else {
		cout << "variety_object_with_action::compute_tactical_decompositions_wrt_set_stabilizer "
				"the set stabilizer is not available" << endl;
		TD_set_stabilizer = NULL;
	}


	if (f_v) {
		cout << "variety_object_with_action::compute_tactical_decompositions_wrt_set_stabilizer done" << endl;
	}


}





void variety_object_with_action::print(
		std::ostream &ost)
{
	ost << "cnt=" << cnt;
	ost << " po=" << po;
	ost << " so=" << so;

	Variety_object->print(ost);
}

std::string variety_object_with_action::stringify_Pts()
{
	std::string s;


	s = Lint_vec_stringify(
			Variety_object->Point_sets->Sets[0],
			Variety_object->Point_sets->Set_size[0]);

	return s;

}

std::string variety_object_with_action::stringify_bitangents()
{
	std::string s;

	s = Lint_vec_stringify(
			Variety_object->Line_sets->Sets[0],
			Variety_object->Line_sets->Set_size[0]);
	return s;

}

void variety_object_with_action::do_report(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::do_report" << endl;
	}

	int q;

	q = Variety_object->Projective_space->Subspaces->F->q;

	{
		string fname_report;

		fname_report = "variety_" + Variety_object->label_txt + "_report.tex";


		{
			ofstream ost(fname_report);


			string title, author, extra_praeamble;

			title = "Variety $" + Variety_object->label_tex + "$ over GF("
					+ std::to_string(q) + ")";


			other::l1_interfaces::latex_interface L;

			//latex_head_easy(fp);
			L.head(ost,
				false /* f_book */,
				true /* f_title */,
				title, author,
				false /*f_toc */,
				false /* f_landscape */,
				false /* f_12pt */,
				true /*f_enlarged_page */,
				true /* f_pagenumbers*/,
				extra_praeamble /* extra_praeamble */);




			//ost << "\\subsection*{The surface $" << SC->label_tex << "$}" << endl;
			if (f_v) {
				cout << "variety_object_with_action::do_report "
						"before do_report2" << endl;
			}
			do_report2(
					ost, verbose_level);
			if (f_v) {
				cout << "variety_object_with_action::do_report "
						"after do_report2" << endl;
			}


			L.foot(ost);
		}
		other::orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	}



	if (f_v) {
		cout << "variety_object_with_action::do_report done" << endl;
	}
}

void variety_object_with_action::do_report2(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::do_report2" << endl;
	}

	//int q;

	//q = Variety_object->Projective_space->Subspaces->F->q;


	Variety_object->print_equation(ost);


	print_summary(ost);


	if (f_has_automorphism_group) {
		ost << "\\subsection*{Automorphism group}" << endl;
		Stab_gens->print_generators_in_latex_individually(
				ost, verbose_level);
	}
	else {
	}


	if (f_has_set_stabilizer) {
		ost << "\\subsection*{Set stabilizer}" << endl;
		Set_stab_gens->print_generators_in_latex_individually(
				ost, verbose_level);
	}
	else {
	}


	int d;

	d = Variety_object->Projective_space->Subspaces->n + 1;



	long int nb_pts;
	nb_pts = Variety_object->get_nb_points();


	if (nb_pts >= 0) {
		long int *Points;
		int *v;

		Points = Variety_object->Point_sets->Sets[0];

		v = NEW_int(d);

		ost << "The variety has " << nb_pts << " points. They are: " << endl;
		Lint_vec_print_fully(ost, Points, nb_pts);
		ost << "\\\\" << endl;


		ost << "\\begin{multicols}{3}" << endl;
		ost << "\\noindent" << endl;
		int i;

		for (i = 0; i < nb_pts; i++) {
			Variety_object->Projective_space->unrank_point(v, Points[i]);
			ost << i << " : $P_{" << Points[i] << "}=";
			Int_vec_print_fully(ost, v, d);
			ost << "$\\\\" << endl;
		}
		ost << "\\end{multicols}" << endl;
		FREE_int(v);
	}





	//data_structures::set_of_sets *Point_sets;

	if (Variety_object->f_has_singular_points) {

		int *v;
		v = NEW_int(d);

		if (f_v) {
			cout << "variety_object_with_action::do_report2 "
					"number of singular points = "
					<< Variety_object->Singular_points.size() << endl;
		}

		ost << "The singular points are: " << endl;
		Lint_vec_stl_print_fully(ost, Variety_object->Singular_points);
		ost << "\\\\" << endl;
		ost << "\\begin{multicols}{3}" << endl;
		ost << "\\noindent" << endl;
		int i;

		nb_pts = Variety_object->Singular_points.size();

		for (i = 0; i < nb_pts; i++) {
			Variety_object->Projective_space->unrank_point(
					v, Variety_object->Singular_points[i]);
			ost << i << " : $P_{" << Variety_object->Singular_points[i] << "}=";
			Int_vec_print_fully(ost, v, d);
			ost << "$\\\\" << endl;
		}
		ost << "\\end{multicols}" << endl;

		FREE_int(v);

	}



	if (Variety_object->get_nb_lines() >= 0) {

		long int nb_lines;
		long int *Lines;
		int *w;
		nb_lines = Variety_object->Line_sets->Set_size[0];
		Lines = Variety_object->Line_sets->Sets[0];

		w = NEW_int(2 * d);

		ost << "The variety has " << nb_lines << " lines. They are: " << endl;
		Lint_vec_print_fully(ost, Lines, nb_lines);
		ost << "\\\\" << endl;



		int *Adj;

		if (f_v) {
			cout << "variety_object_with_action::do_report2 "
					"before line_intersection_graph_for_a_given_set" << endl;
		}
		Variety_object->Projective_space->Subspaces->line_intersection_graph_for_a_given_set(
				Lines, nb_lines,
				Adj,
				verbose_level);
		if (f_v) {
			cout << "variety_object_with_action::do_report2 "
					"after line_intersection_graph_for_a_given_set" << endl;
		}


		int i, j, a;

		ost << "Pairwise intersection of lines:" << endl;
		ost << "$$" << endl;
		ost << "\\begin{array}{|rr|*{" << nb_lines << "}{r}|}" << endl;
		ost << "\\hline" << endl;
		ost << "& ";
		for (j = 0; j < nb_lines; j++) {
			ost << "& " << j << endl;
		}
		ost << "\\\\" << endl;
		ost << "& ";
		for (j = 0; j < nb_lines; j++) {
			ost << "& " << Lines[j] << endl;
		}
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
		for (i = 0; i < nb_lines; i++) {
			ost << i;
			ost << " & " << Lines[i];
			for (j = 0; j < nb_lines; j++) {
				a = Adj[i * nb_lines + j];
				ost << " & ";
				if (i != j) {
					ost << a;
				}
			}
			ost << "\\\\" << endl;
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;


#if 0
		Variety_object->Projective_space->Reporting->cheat_sheet_line_intersection(
				ost, verbose_level);


		Variety_object->Projective_space->Reporting->cheat_sheet_lines_on_points(
			ost, verbose_level);
#endif

		FREE_int(w);

	}




	string TDO_label1;
	string TDO_label2;
	string TDA_label1;
	string TDA_label2;

	TDO_label1 = "TDO";
	TDO_label2 = "TDOsetstab";
	TDA_label1 = "TDA";
	TDA_label2 = "TDAsetstab";

	if (TD) {
		TD->report_decomposition_schemes(
				ost,
				TDO_label1,
				TDA_label1,
				verbose_level);
	}
	if (TD_set_stabilizer) {
		TD_set_stabilizer->report_decomposition_schemes(
				ost,
				TDO_label2,
				TDA_label2,
				verbose_level);
	}




	if (f_v) {
		cout << "variety_object_with_action::do_report2 done" << endl;
	}
}

void variety_object_with_action::print_summary(
		std::ostream &ost)
{
	ost << "\\subsection*{Summary}" << endl;


	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|l|r|}" << endl;

	algebra::ring_theory::longinteger_object ago;
	algebra::ring_theory::longinteger_object set_stab_go;

	if (f_has_automorphism_group) {
		Stab_gens->group_order(ago);
	}
	else {
		ago.create(-1);
	}

	if (f_has_set_stabilizer) {
		Set_stab_gens->group_order(set_stab_go);
	}
	else {
		set_stab_go.create(-1);
	}

	int nb_points;
	int nb_lines;
	int nb_singular_points = -1;

	nb_points = Variety_object->get_nb_points();
	nb_lines = Variety_object->get_nb_lines();

	if (Variety_object->f_has_singular_points) {
		nb_singular_points = Variety_object->Singular_points.size();
	}

	ost << "\\hline" << endl;
	ost << "\\mbox{Number of automorphisms} & " << ago << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Order of set stabilizer} & " << set_stab_go << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of points} & " << nb_points << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of lines} & " << nb_lines << " \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of singular points} & " << nb_singular_points << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
}


void variety_object_with_action::export_col_headings(
		std::string *&Col_headings, int &nb_cols,
		int verbose_level)
{
	nb_cols = 15;
	Col_headings = new std::string [nb_cols];

	Col_headings[0] = "n";
	Col_headings[1] = "q";
	Col_headings[2] = "d";
	Col_headings[3] = "label_txt";
	Col_headings[4] = "label_tex";
	Col_headings[5] = "equation_af";
	Col_headings[6] = "equation_vec";
	Col_headings[7] = "Ago";
	Col_headings[8] = "SetStab";
	Col_headings[9] = "NbPoints";
	Col_headings[10] = "NbLines";
	Col_headings[11] = "NbSingPoints";
	Col_headings[12] = "Points";
	Col_headings[13] = "Lines";
	Col_headings[14] = "LinesKlein";
}

void variety_object_with_action::export_data(
		std::vector<std::string> &Table, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::export_data" << endl;
	}

	algebra::ring_theory::longinteger_object ago;
	algebra::ring_theory::longinteger_object set_stab_go;




	if (f_has_automorphism_group) {
		Stab_gens->group_order(ago);
	}
	else {
		ago.create(-1);
	}

	if (f_has_set_stabilizer) {
		Set_stab_gens->group_order(set_stab_go);
	}
	else {
		set_stab_go.create(-1);
	}


	int nb_points;
	int nb_lines;
	int nb_singular_points = -1;

	nb_points = Variety_object->get_nb_points();
	nb_lines = Variety_object->get_nb_lines();

	if (Variety_object->f_has_singular_points) {
		nb_singular_points = Variety_object->Singular_points.size();
	}

	string s_Pts, s_Lines, s_Lines_Klein;

	s_Pts = "\"" + Variety_object->stringify_points() + "\"";
	s_Lines = "\"" + Variety_object->stringify_lines() + "\"";


	if (PA->P->Subspaces->n == 3 && Variety_object->Line_sets) {



		geometry::orthogonal_geometry::orthogonal *O;
		geometry::projective_geometry::klein_correspondence *Klein;

		O = PA->Surf_A->Surf->O;
		Klein = PA->Surf_A->Surf->Klein;

		int v[6];
		long int *Lines;
		long int *Points_on_Klein_quadric;
		long int line_rk;
		int i;

		Lines = Variety_object->Line_sets->Sets[0];

		Points_on_Klein_quadric = NEW_lint(nb_lines);

		for (i = 0; i < nb_lines; i++) {
			line_rk = Lines[i];

			Klein->line_to_Pluecker(
				line_rk, v, 0 /* verbose_level*/);

			Points_on_Klein_quadric[i] = O->Orthogonal_indexing->Qplus_rank(
					v,
					1, 5, 0 /*verbose_level */);

		}

		other::data_structures::sorting Sorting;
		Sorting.lint_vec_heapsort(Points_on_Klein_quadric, nb_lines);


		s_Lines_Klein = "\"" + Lint_vec_stringify(Points_on_Klein_quadric, nb_lines) + "\"";

		FREE_lint(Points_on_Klein_quadric);
	}
	else {
		s_Lines_Klein = "\"\"";
	}

	int n, q, d;
	string s_eqn;
	string s_eqn_vec;

	n = PA->P->Subspaces->n;
	q = PA->P->Subspaces->F->q;
	d = Variety_object->Ring->degree;

	s_eqn = "\"" + Variety_object->Ring->stringify_equation(Variety_object->eqn) + "\"";
	s_eqn_vec = "\"" + Int_vec_stringify(Variety_object->eqn, Variety_object->Ring->get_nb_monomials()) + "\"";

	Table.push_back(std::to_string(n));
	Table.push_back(std::to_string(q));
	Table.push_back(std::to_string(d));
	Table.push_back("\"" + Variety_object->label_txt + "\"");
	Table.push_back("\"" + Variety_object->label_tex + "\"");

	Table.push_back(s_eqn);
	Table.push_back(s_eqn_vec);



	Table.push_back(ago.stringify());
	Table.push_back(set_stab_go.stringify());
	Table.push_back(std::to_string(nb_points));
	Table.push_back(std::to_string(nb_lines));
	Table.push_back(std::to_string(nb_singular_points));
	Table.push_back(s_Pts);
	Table.push_back(s_Lines);
	Table.push_back(s_Lines_Klein);

	if (f_v) {
		cout << "variety_object_with_action::export_data done" << endl;
	}
}


void variety_object_with_action::do_export(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::do_export" << endl;
	}

	std::string *Col_headings;
	int nb_cols2;

	export_col_headings(
			Col_headings, nb_cols2,
			verbose_level);


	std::vector<std::string> table;


	export_data(
			table, verbose_level);


	string *Table;
	int nb_cols;
	int nb_rows;
	int i, j;

	nb_rows = 1;
	nb_cols = table.size();

	if (nb_cols2 != nb_cols) {
		cout << "variety_object_with_action::do_export "
				"nb_cols2 != nb_cols" << endl;
		exit(1);
	}


	Table = new string[nb_rows * nb_cols];
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Table[i * nb_cols + j] =
					table[j];
		}
	}


	string fname;

	fname = "variety_" + Variety_object->label_txt + "_data.csv";

	other::orbiter_kernel_system::file_io Fio;


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	delete [] Table;
	delete [] Col_headings;


	if (f_v) {
		cout << "variety_object_with_action::do_export done" << endl;
	}
}


}}}




