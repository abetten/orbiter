/*
 * group_theoretic_activity_for_surfaces.cpp
 *
 *  Created on: Oct 19, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




void group_theoretic_activity::do_create_surface(
		surface_create_description *Surface_Descr,
		poset_classification_control *Control_six_arcs,
		int f_sweep, std::string &sweep_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface" << endl;
		cout << "group_theoretic_activity::do_create_surface verbose_level=" << verbose_level << endl;
	}

	if (f_sweep) {
		cout << "group_theoretic_activity::do_create_surface f_sweep, fname = " << sweep_fname << endl;
	}

	int q;
	finite_field *F;
	surface_domain *Surf;
	surface_with_action *Surf_A;

	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface before Surface_Descr->get_q" << endl;
	}
	q = Surface_Descr->get_q();
	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface q = " << q << endl;
	}


	F = LG->F;
	if (F->q != q) {
		cout << "F->q != q" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, LG->A_linear, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface after Surf_A->init" << endl;
	}


	if (f_sweep) {
		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface before Surf_A->create_surface_sweep" << endl;
		}
		Surf_A->create_surface_sweep(
					Surface_Descr,
					Control_six_arcs,
					f_sweep, sweep_fname,
					verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface after Surf_A->create_surface_sweep" << endl;
		}

	}
	else {
		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface before Surf_A->create_surface_and_do_report" << endl;
			cout << "Descr->f_surface_quartic = " << Descr->f_surface_quartic << endl;
		}
		Surf_A->create_surface_and_do_report(
					Surface_Descr,
					TRUE, Control_six_arcs,
					Descr->f_surface_clebsch,
					Descr->f_surface_codes,
					Descr->f_surface_quartic,
					verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::do_create_surface after Surf_A->create_surface_and_do_report" << endl;
		}
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_create_surface done" << endl;
	}
}



void group_theoretic_activity::do_surface_classify(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;

	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		cout << "please use option -poset_classification_control" << endl;
		exit(1);
		//Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify "
				"before Algebra.classify_surfaces, control=" << endl;
		Control->print();
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify "
				"after Algebra.classify_surfaces" << endl;
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify "
				"before SCW->generate_source_code" << endl;
	}
	SCW->generate_source_code(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_classify "
				"after SCW->generate_source_code" << endl;
	}


	if (Descr->f_report) {

		layered_graph_draw_options *O;


		if (!Orbiter->f_draw_options) {
			cout << "please use option -draw_options .. -end" << endl;
			exit(1);
		}
		O = Orbiter->draw_options;


		SCW->create_report(TRUE /*f_with_stabilizers */,
				O,
				verbose_level);
	}

	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "interface_algebra::do_surface_classify done" << endl;
	}
}

void group_theoretic_activity::do_surface_report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;
	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report "
				"after Algebra.classify_surfaces" << endl;
	}

	int f_with_stabilizers = TRUE;

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report "
				"before SCW->create_report" << endl;
	}
	SCW->create_report(f_with_stabilizers, Control->draw_options, verbose_level - 1);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_report "
				"after SCW->create_report" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "interface_algebra::do_surface_report done" << endl;
	}
}

void group_theoretic_activity::do_surface_identify_HCV(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_HCV" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;
	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_HCV "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_HCV "
				"after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_HCV "
				"before SCW->identify_HCV_and_print_table" << endl;
	}
	SCW->identify_HCV_and_print_table(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_HCV "
				"after SCW->identify_HCV_and_print_table" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_HCV done" << endl;
	}
}

void group_theoretic_activity::do_surface_identify_F13(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_F13" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;
	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_F13 "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_F13 "
				"after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_F13 "
				"before SCW->identify_HCV_and_print_table" << endl;
	}
	SCW->identify_F13_and_print_table(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_F13 "
				"after SCW->identify_HCV_and_print_table" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_F13 done" << endl;
	}
}

void group_theoretic_activity::do_surface_identify_Bes(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Bes" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;
	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Bes "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Bes "
				"after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Bes "
				"before SCW->identify_Bes_and_print_table" << endl;
	}
	SCW->identify_Bes_and_print_table(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Bes "
				"after SCW->identify_Bes_and_print_table" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_Bes done" << endl;
	}
}

void group_theoretic_activity::do_surface_identify_general_abcd(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_general_abcd" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;
	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_general_abcd "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_general_abcd "
				"after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_general_abcd "
				"before SCW->identify_general_abcd_and_print_table" << endl;
	}
	SCW->identify_general_abcd_and_print_table(verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_general_abcd "
				"after SCW->identify_general_abcd_and_print_table" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_identify_general_abcd done" << endl;
	}
}

void group_theoretic_activity::do_surface_isomorphism_testing(
		surface_create_description *surface_descr_isomorph1,
		surface_create_description *surface_descr_isomorph2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;
	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing "
				"after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing "
				"before SCW->test_isomorphism" << endl;
	}
	SCW->test_isomorphism(
			surface_descr_isomorph1,
			surface_descr_isomorph2,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing "
				"after SCW->test_isomorphism" << endl;
	}


	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_isomorphism_testing done" << endl;
	}
}

void group_theoretic_activity::do_surface_recognize(
		surface_create_description *surface_descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize" << endl;
	}

	algebra_global_with_action Algebra;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	surface_classify_wedge *SCW;
	poset_classification_control *Control;

	if (Descr->f_poset_classification_control) {
		Control = Descr->Control;
	}
	else {
		Control = NEW_OBJECT(poset_classification_control);
	}


	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize "
				"before Algebra.classify_surfaces" << endl;
	}
	Algebra.classify_surfaces(
			F, LG,
			Control,
			Surf, Surf_A,
			SCW,
			verbose_level - 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize "
				"after Algebra.classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize "
				"before SCW->recognition" << endl;
	}
	SCW->recognition(
			surface_descr,
			verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize "
				"after SCW->recognition" << endl;
	}

	FREE_OBJECT(SCW);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);
	if (f_v) {
		cout << "group_theoretic_activity::do_surface_recognize done" << endl;
	}
}





void group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines(
		poset_classification_control *Control_six_arcs,
		int f_test_nb_Eckardt_points, int nb_E,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines" << endl;
	}

	surface_with_action *Surf_A;
	surface_domain *Surf;
	number_theory_domain NT;



	Surf = NEW_OBJECT(surface_domain);
	Surf_A = NEW_OBJECT(surface_with_action);


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
				"before Surf->init" << endl;
	}
	Surf->init(F, 0 /*verbose_level*/);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
				"after Surf->init" << endl;
	}


#if 0
	if (f_v) {
		cout << "before Surf->init_large_polynomial_domains" << endl;
	}
	Surf->init_large_polynomial_domains(0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf->init_large_polynomial_domains" << endl;
	}
#endif


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
				"before Surf_A->init_with_linear_group" << endl;
	}
	Surf_A->init_with_linear_group(Surf, LG, TRUE /* f_recoordinatize */, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
				"after Surf_A->init_with_linear_group" << endl;
	}


	surfaces_arc_lifting *SAL;

	SAL = NEW_OBJECT(surfaces_arc_lifting);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
				"before SAL->init" << endl;
	}
	SAL->init(
		LG->F, LG /* LG4 */,
		Surf_A,
		Control_six_arcs,
		f_test_nb_Eckardt_points, nb_E,
		verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
				"after SAL->init" << endl;
	}

	if (Descr->f_report) {
		if (f_v) {
			cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
					"before SAL->report" << endl;
		}
		SAL->report(Control_six_arcs->draw_options, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines "
					"after SAL->report" << endl;
		}

	}
	FREE_OBJECT(SAL);


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_two_lines done" << endl;
	}

}

void group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs(
		poset_classification_control *Control1,
		poset_classification_control *Control2,
		poset_classification_control *Control_six_arcs,
		int f_test_nb_Eckardt_points, int nb_E,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	surface_with_action *Surf_A;
	surface_domain *Surf;
	number_theory_domain NT;



	Surf = NEW_OBJECT(surface_domain);
	Surf_A = NEW_OBJECT(surface_with_action);


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf->init" << endl;
	}
	Surf->init(F, 0 /*verbose_level*/);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf->init" << endl;
	}


#if 0
	if (f_v) {
		cout << "before Surf->init_large_polynomial_domains" << endl;
	}
	Surf->init_large_polynomial_domains(0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf->init_large_polynomial_domains" << endl;
	}
#endif


	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_A->init_with_linear_group" << endl;
	}
	Surf_A->init_with_linear_group(Surf, LG, TRUE /* f_recoordinatize */, verbose_level - 1);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_A->init_with_linear_group" << endl;
	}



	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_A->Classify_trihedral_pairs->classify" << endl;
	}
	Surf_A->Classify_trihedral_pairs->classify(Control1, Control2, verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_A->Classify_trihedral_pairs->classify" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_arc->classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	surface_classify_using_arc *Surf_arc;

	Surf_arc = NEW_OBJECT(surface_classify_using_arc);


	Surf_arc->classify_surfaces_through_arcs_and_trihedral_pairs(
			Control_six_arcs,
			Surf_A,
			f_test_nb_Eckardt_points, nb_E,
			verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_arc->classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_arc->report" << endl;
	}


	if (Orbiter->f_draw_options) {
		Surf_arc->report(Orbiter->draw_options, verbose_level);
	}
	else {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"please use -draw_option for a report" << endl;
	}

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_arc->report" << endl;
	}

	FREE_OBJECT(Surf_arc);

	if (f_v) {
		cout << "group_theoretic_activity::do_classify_surfaces_through_arcs_and_trihedral_pairs done" << endl;
	}

}

void group_theoretic_activity::do_six_arcs(
		poset_classification_control *Control_six_arcs,
		int f_filter_by_nb_Eckardt_points, int nb_Eckardt_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs" << endl;
	}

	finite_field *F;

	F = LG->F;


	surface_domain *Surf;

	if (f_v) {
			cout << "group_theoretic_activity::do_six_arcs before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, 0/*verbose_level - 1*/);
	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs after Surf->init" << endl;
	}



	six_arcs_not_on_a_conic *Six_arcs;
	arc_generator_description *Six_arc_descr;

	int *transporter;

	Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);

	Six_arc_descr = NEW_OBJECT(arc_generator_description);
	Six_arc_descr->F = F;
	Six_arc_descr->f_q = TRUE;
	Six_arc_descr->q = F->q;
	Six_arc_descr->f_n = TRUE;
	Six_arc_descr->n = 3;
	Six_arc_descr->f_target_size = TRUE;
	Six_arc_descr->target_size = 6;
	Six_arc_descr->Control = Control_six_arcs;



	// classify six arcs not on a conic:

	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs "
				"Setting up the group of the plane:" << endl;
	}



	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs "
				"before Six_arcs->init:" << endl;
	}


	Six_arcs->init(
			Six_arc_descr,
			LG->A_linear,
			Surf->P2,
			FALSE, 0, NULL,
			verbose_level);

	transporter = NEW_int(Six_arcs->Gen->A->elt_size_in_int);

	int nb_orbits;
	int level = 6;

	nb_orbits = Six_arcs->Gen->gen->nb_orbits_at_level(level);

	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs "
				"We found " << nb_orbits << " isomorphism types "
				"of 6-arcs" << endl;
	}



	long int Arc6[6];
	int h, a, b, c, d;
	int v1[3];
	int v2[3];


	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs "
				"testing the arcs" << endl;
	}

	longinteger_object ago;
	int *Abcd;
	int *Nb_E;
	int *Ago;

	Abcd = NEW_int(nb_orbits * 4);
	Nb_E = NEW_int(nb_orbits);
	Ago = NEW_int(nb_orbits);

	for (h = 0; h < nb_orbits; h++) {

		if (f_v) {
			cout << "group_theoretic_activity::do_six_arcs "
					"testing arc " << h << " / " << nb_orbits << endl;
		}


		Six_arcs->Gen->gen->get_set_by_level(level, h, Arc6);

		Six_arcs->Gen->gen->get_stabilizer_order(level, h, ago);

		if (Arc6[0] != 0) {
			cout << "Arc6[0] != 0" << endl;
			exit(1);
		}
		if (Arc6[1] != 1) {
			cout << "Arc6[1] != 1" << endl;
			exit(1);
		}
		if (Arc6[2] != 2) {
			cout << "Arc6[2] != 2" << endl;
			exit(1);
		}
		if (Arc6[3] != 3) {
			cout << "Arc6[3] != 3" << endl;
			exit(1);
		}
		Surf->P2->unrank_point(v1, Arc6[4]);
		Surf->P2->unrank_point(v2, Arc6[5]);
		if (v1[2] != 1) {
			cout << "v1[2] != 1" << endl;
			exit(1);
		}
		if (v2[2] != 1) {
			cout << "v2[2] != 1" << endl;
			exit(1);
		}
		a = v1[0];
		b = v1[1];
		c = v2[0];
		d = v2[1];

		Abcd[h * 4 + 0] = a;
		Abcd[h * 4 + 1] = b;
		Abcd[h * 4 + 2] = c;
		Abcd[h * 4 + 3] = d;

		eckardt_point_info *E;

		E = Surf->P2->compute_eckardt_point_info(Surf, Arc6, 0/*verbose_level*/);


		Nb_E[h] = E->nb_E;
		Ago[h] = ago.as_int();

		//cout << h << " : " << a << "," << b << "," << c << "," << d << " : " << E->nb_E << " : " << ago << endl;

		FREE_OBJECT(E);
	}

#if 0
	cout << "Summary of " << nb_orbits << " arcs:" << endl;
	for (h = 0; h < nb_orbits; h++) {
		a = Abcd[h * 4 + 0];
		b = Abcd[h * 4 + 1];
		c = Abcd[h * 4 + 2];
		d = Abcd[h * 4 + 3];

		cout << h << " : " << a << "," << b << "," << c << "," << d << " : " << Nb_E[h] << " : " << Ago[h] << endl;
	}
#endif

	tally C;

	C.init(Nb_E, nb_orbits, FALSE, 0);

	cout << "nb_E distribution: ";
	C.print_naked_tex(cout, FALSE);
	cout << endl;


	if (f_filter_by_nb_Eckardt_points) {
		cout << "Nonconical six-arcs associated with surfaces with " << nb_Eckardt_points << " Eckardt points in PG(2," << F->q << "):" << endl;

	}
	else {
		cout << "Nonconical six-arcs associated in PG(2," << F->q << "):" << endl;

	}
	int nb_E;
	int cnt = 0;

	cout << "$$" << endl;
	cout << "\\begin{array}{|r|c|r|}" << endl;
	cout << "\\hline" << endl;
	cout << "\\mbox{Orbit} & a,b,c,d & \\mbox{Ago} \\\\" << endl;
	cout << "\\hline" << endl;

	for (h = 0; h < nb_orbits; h++) {
		a = Abcd[h * 4 + 0];
		b = Abcd[h * 4 + 1];
		c = Abcd[h * 4 + 2];
		d = Abcd[h * 4 + 3];

		nb_E = Nb_E[h];

		if (f_filter_by_nb_Eckardt_points) {
			if (nb_E != nb_Eckardt_points) {
				continue;
			}
		}
		cout << h << " & " << a << "," << b << "," << c << "," << d << " & ";
		//<< Nb_E[h] << " & "
		cout << Ago[h] << "\\\\" << endl;

		cnt++;
	}
	cout << "\\hline" << endl;
	cout << "\\end{array}" << endl;
	cout << "$$" << endl;
	cout << "There are " << cnt << " such arcs.\\\\" << endl;


	FREE_int(Abcd);
	FREE_int(Nb_E);
	FREE_int(Ago);


	FREE_OBJECT(Six_arcs);
	FREE_OBJECT(Six_arc_descr);
	FREE_int(transporter);


	if (f_v) {
		cout << "group_theoretic_activity::do_six_arcs done" << endl;
	}

}


}}

