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

	TD = NULL;

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
	if (TD) {
		FREE_OBJECT(TD);
	}
}

void variety_object_with_action::create_variety(
		projective_geometry::projective_space_with_action *PA,
		int cnt, int po_go, int po_index, int po, int so,
		geometry::algebraic_geometry::variety_description *VD,
		int verbose_level)
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

	if (VD->f_bitangents == false) {
		VD->f_bitangents = true;
		VD->bitangents_txt = "";
	}



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

	int i;

	for (i = 0; i < VD->transformations.size(); i++) {
		if (f_v) {
			cout << "variety_object_with_action::create_variety "
					"-transform " << VD->transformations[i] << endl;
		}

		int *data;
		int *Elt;
		int sz;

		Elt = NEW_int(PA->A->elt_size_in_int);

		Int_vec_scan(VD->transformations[i], data, sz);
		PA->A->Group_element->make_element(Elt, data, 0 /* verbose_level */);

		if (f_v) {
			cout << "variety_object_with_action::create_variety "
					"before apply_transformation" << endl;
		}

		apply_transformation(
				Elt,
				PA->A,
				PA->A_on_lines,
				verbose_level - 2);

		if (f_v) {
			cout << "variety_object_with_action::create_variety "
					"after apply_transformation" << endl;
		}

		FREE_int(data);

	}


	if (f_v) {
		cout << "variety_object_with_action::create_variety "
				"before compute_tactical_decompositions" << endl;
	}
	compute_tactical_decompositions(
			verbose_level);
	if (f_v) {
		cout << "variety_object_with_action::create_variety "
				"after compute_tactical_decompositions" << endl;
	}


	if (f_v) {
		print(cout);
	}

	if (f_v) {
		cout << "variety_object_with_action::create_variety done" << endl;
	}
}

void variety_object_with_action::apply_transformation(
		int *Elt,
		actions::action *A,
		actions::action *A_on_lines,
		int verbose_level)
// Creates an action on the homogeneous polynomials on the fly
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "variety_object_with_action::apply_transformation" << endl;
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
		cout << "variety_object_with_action::apply_transformation "
				"before AG.set_of_sets_copy_and_apply, Point_sets" << endl;
	}
	Variety_object->Point_sets = AG.set_of_sets_copy_and_apply(
			A,
			Elt,
			old_Variety_object->Point_sets,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_object_with_action::apply_transformation "
				"after AG.set_of_sets_copy_and_apply, Point_sets" << endl;
	}

	// we are sorting the points:

	Variety_object->Point_sets->sort();

	if (f_v) {
		cout << "variety_object_with_action::apply_transformation "
				"before AG.set_of_sets_copy_and_apply, Line_sets" << endl;
	}
	Variety_object->Line_sets = AG.set_of_sets_copy_and_apply(
			A_on_lines,
			Elt,
			old_Variety_object->Line_sets,
			verbose_level - 2);
	if (f_v) {
		cout << "variety_object_with_action::apply_transformation "
				"after AG.set_of_sets_copy_and_apply, Line_sets" << endl;
	}

	// We are not sorting the lines because the lines are often in the Schlaefli ordering

	FREE_OBJECT(old_Variety_object);



	if (f_v) {
		cout << "variety_object_with_action::apply_transformation "
				"after transforming:" << endl;
		print(cout);
	}

	if (f_v) {
		cout << "variety_object_with_action::apply_transformation done" << endl;
	}
}

void variety_object_with_action::compute_tactical_decompositions(
		int verbose_level)
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
		Stab_gens->print_generators_in_latex_individually(
				ost, verbose_level);
	}
	else {
	}


	int d;
	long int nb_pts;
	long int *Points;
	int *v;

	d = Variety_object->Projective_space->Subspaces->n + 1;
	nb_pts = Variety_object->Point_sets->Set_size[0];
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



	//data_structures::set_of_sets *Point_sets;

	if (Variety_object->f_has_singular_points) {

		if (f_v) {
			cout << "variety_object_with_action::do_report2 "
					"number of singular points = " << Variety_object->Singular_points.size() << endl;
		}

		ost << "The singular points are: " << endl;
		Lint_vec_stl_print_fully(ost, Variety_object->Singular_points);
		ost << "\\\\" << endl;
		ost << "\\begin{multicols}{3}" << endl;
		ost << "\\noindent" << endl;
		int i;

		nb_pts = Variety_object->Singular_points.size();

		for (i = 0; i < nb_pts; i++) {
			Variety_object->Projective_space->unrank_point(v, Variety_object->Singular_points[i]);
			ost << i << " : $P_{" << Variety_object->Singular_points[i] << "}=";
			Int_vec_print_fully(ost, v, d);
			ost << "$\\\\" << endl;
		}
		ost << "\\end{multicols}" << endl;


	}
	if (TD) {
		TD->report_decomposition_schemes(ost, verbose_level);
	}
	FREE_int(v);




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

	if (f_has_automorphism_group) {
		Stab_gens->group_order(ago);
	}
	else {
		ago.create(-1);
	}
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of automorphisms} & " << ago << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of points} & " << Variety_object->Point_sets->Set_size[0] << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of lines} & " << Variety_object->Line_sets->Set_size[0] << " \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of singular points} & " << Variety_object->Singular_points.size() << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
}




}}}




