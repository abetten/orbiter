/*
 * surface_domain_high_level.cpp
 *
 *  Created on: Mar 28, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_in_general {



surface_domain_high_level::surface_domain_high_level()
{
}

surface_domain_high_level::~surface_domain_high_level()
{
}

void surface_domain_high_level::do_sweep_4_15_lines(
		projective_geometry::projective_space_with_action *PA,
		surface_create_description *Surface_Descr,
		std::string &sweep_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_4_15_lines" << endl;
		cout << "surface_domain_high_level::do_sweep_4_15_lines "
				"verbose_level=" << verbose_level << endl;
	}


	surface_with_action *Surf_A;

	PA->setup_surface_with_action(
			Surf_A,
			verbose_level);

	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_4_15_lines "
				"before Surf_A->sweep_4_15_lines" << endl;
	}
	Surf_A->sweep_4_15_lines(
				Surface_Descr,
				sweep_fname,
				verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_4_15_lines "
				"after Surf_A->sweep_4_15_lines" << endl;
	}


	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_4_15_lines done" << endl;
	}
}

void surface_domain_high_level::do_sweep_F_beta_9_lines(
		projective_geometry::projective_space_with_action *PA,
		surface_create_description *Surface_Descr,
		std::string &sweep_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_F_beta_9_lines" << endl;
		cout << "surface_domain_high_level::do_sweep_F_beta_9_lines "
				"verbose_level=" << verbose_level << endl;
	}


	surface_with_action *Surf_A;

	PA->setup_surface_with_action(
			Surf_A,
			verbose_level);

	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_F_beta_9_lines "
				"before Surf_A->sweep_F_beta_9_lines" << endl;
	}
	Surf_A->sweep_F_beta_9_lines(
				Surface_Descr,
				sweep_fname,
				verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_F_beta_9_lines "
				"after Surf_A->sweep_F_beta_9_lines" << endl;
	}


	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_F_beta_9_lines done" << endl;
	}
}


void surface_domain_high_level::do_sweep_6_9_lines(
		projective_geometry::projective_space_with_action *PA,
		surface_create_description *Surface_Descr,
		std::string &sweep_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_6_9_lines" << endl;
		cout << "surface_domain_high_level::do_sweep_6_9_lines "
				"verbose_level=" << verbose_level << endl;
	}


	surface_with_action *Surf_A;

	PA->setup_surface_with_action(
			Surf_A,
			verbose_level);

	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_6_9_lines "
				"before Surf_A->sweep_6_9_lines" << endl;
	}
	Surf_A->sweep_6_9_lines(
				Surface_Descr,
				sweep_fname,
				verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_6_9_lines "
				"after Surf_A->sweep_6_9_lines" << endl;
	}


	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_6_9_lines done" << endl;
	}
}

void surface_domain_high_level::do_sweep_4_27(
		projective_geometry::projective_space_with_action *PA,
		surface_create_description *Surface_Descr,
		std::string &sweep_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_4_27" << endl;
		cout << "surface_domain_high_level::do_sweep_4_27 "
				"verbose_level=" << verbose_level << endl;
	}

	surface_with_action *Surf_A;

	PA->setup_surface_with_action(
			Surf_A,
			verbose_level);

	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_4_27 before Surf_A->sweep_4" << endl;
	}
	Surf_A->sweep_4_27(
				Surface_Descr,
				sweep_fname,
				verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_4_27 after Surf_A->sweep_4" << endl;
	}

	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_4_27 done" << endl;
	}
}



void surface_domain_high_level::do_sweep_4_L9_E4(
		projective_geometry::projective_space_with_action *PA,
		surface_create_description *Surface_Descr,
		std::string &sweep_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_4_L9_E4" << endl;
		cout << "surface_domain_high_level::do_sweep_4_L9_E4 "
				"verbose_level=" << verbose_level << endl;
	}

	surface_with_action *Surf_A;

	PA->setup_surface_with_action(
			Surf_A,
			verbose_level);

	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_4_L9_E4 before Surf_A->sweep_4" << endl;
	}
	Surf_A->sweep_4_L9_E4(
				Surface_Descr,
				sweep_fname,
				verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_4_L9_E4 after Surf_A->sweep_4" << endl;
	}

	if (f_v) {
		cout << "surface_domain_high_level::do_sweep_4_L9_E4 done" << endl;
	}
}




void surface_domain_high_level::classify_surfaces_with_double_sixes(
		projective_geometry::projective_space_with_action *PA,
		std::string &control_label,
		cubic_surfaces_and_double_sixes::surface_classify_wedge *&SCW,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes" << endl;
	}

	//algebraic_geometry::surface_domain *Surf;
	//surface_with_action *Surf_A;

	poset_classification::poset_classification_control *Control;

	Control =
			Get_object_of_type_poset_classification_control(control_label);


	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes "
				"before classify_surfaces, control=" << endl;
		Control->print();
	}
	prepare_surface_classify_wedge(
			PA->F, PA,
			Control,
			SCW,
			verbose_level);

	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes "
				"after classify_surfaces" << endl;
	}

#if 0
	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes "
				"before SCW->generate_source_code" << endl;
	}
	SCW->generate_source_code(verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes "
				"after SCW->generate_source_code" << endl;
	}
#endif

#if 0
	if (FALSE) {

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
#endif

	//FREE_OBJECT(Surf_A);
	//FREE_OBJECT(Surf);
	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes done" << endl;
	}
}


void surface_domain_high_level::prepare_surface_classify_wedge(
		field_theory::finite_field *F,
		projective_geometry::projective_space_with_action *PA,
		poset_classification::poset_classification_control *Control,
		//algebraic_geometry::surface_domain *&Surf, surface_with_action *&Surf_A,
		cubic_surfaces_and_double_sixes::surface_classify_wedge *&SCW,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "surface_domain_high_level::prepare_surface_classify_wedge" << endl;
	}


	surface_with_action *Surf_A;

	PA->setup_surface_with_action(
			Surf_A,
			verbose_level);

	//Surf = Surf_A->Surf;



	SCW = NEW_OBJECT(cubic_surfaces_and_double_sixes::surface_classify_wedge);

	if (f_v) {
		cout << "surface_domain_high_level::prepare_surface_classify_wedge before SCW->init" << endl;
	}

	SCW->init(Surf_A,
			Control,
			verbose_level - 1);

	if (f_v) {
		cout << "surface_domain_high_level::prepare_surface_classify_wedge after SCW->init" << endl;
	}


	if (f_v) {
		cout << "surface_domain_high_level::prepare_surface_classify_wedge before SCW->do_classify_double_sixes" << endl;
	}
	SCW->do_classify_double_sixes(verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::prepare_surface_classify_wedge after SCW->do_classify_double_sixes" << endl;
	}

	if (f_v) {
		cout << "surface_domain_high_level::prepare_surface_classify_wedge before SCW->do_classify_surfaces" << endl;
	}
	SCW->do_classify_surfaces(verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::prepare_surface_classify_wedge after SCW->do_classify_surfaces" << endl;
	}

	if (f_v) {
		cout << "surface_domain_high_level::prepare_surface_classify_wedge done" << endl;
	}

}

void surface_domain_high_level::do_study_surface(field_theory::finite_field *F, int nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_study_surface" << endl;
	}

	surface_study *study;

	study = NEW_OBJECT(surface_study);

	cout << "before study->init" << endl;
	study->init(F, nb, verbose_level);
	cout << "after study->init" << endl;

	cout << "before study->study_intersection_points" << endl;
	study->study_intersection_points(verbose_level);
	cout << "after study->study_intersection_points" << endl;

	cout << "before study->study_line_orbits" << endl;
	study->study_line_orbits(verbose_level);
	cout << "after study->study_line_orbits" << endl;

	cout << "before study->study_group" << endl;
	study->study_group(verbose_level);
	cout << "after study->study_group" << endl;

	cout << "before study->study_orbits_on_lines" << endl;
	study->study_orbits_on_lines(verbose_level);
	cout << "after study->study_orbits_on_lines" << endl;

	cout << "before study->study_find_eckardt_points" << endl;
	study->study_find_eckardt_points(verbose_level);
	cout << "after study->study_find_eckardt_points" << endl;

#if 0
	if (study->nb_Eckardt_pts == 6) {
		cout << "before study->study_surface_with_6_eckardt_points" << endl;
		study->study_surface_with_6_eckardt_points(verbose_level);
		cout << "after study->study_surface_with_6_eckardt_points" << endl;
		}
#endif

	if (f_v) {
		cout << "surface_domain_high_level::do_study_surface done" << endl;
	}
}




void surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines(
		projective_geometry::projective_space_with_action *PA,
		std::string &Control_six_arcs_label,
		int f_test_nb_Eckardt_points, int nb_E,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines" << endl;
	}

	surface_with_action *Surf_A;
	algebraic_geometry::surface_domain *Surf;
	number_theory::number_theory_domain NT;



	Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
	Surf_A = NEW_OBJECT(surface_with_action);


	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines "
				"before Surf->init" << endl;
	}
	Surf->init(PA->F, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines "
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
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines "
				"after Surf_A->init" << endl;
	}


	cubic_surfaces_and_arcs::surfaces_arc_lifting *SAL;

	SAL = NEW_OBJECT(cubic_surfaces_and_arcs::surfaces_arc_lifting);

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines "
				"before SAL->init" << endl;
	}
	SAL->init(
		Surf_A,
		Control_six_arcs_label,
		f_test_nb_Eckardt_points, nb_E,
		verbose_level - 2);
	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines "
				"after SAL->init" << endl;
	}

	if (TRUE) {
		if (f_v) {
			cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines "
					"before SAL->report" << endl;
		}
		SAL->report(Control_six_arcs_label, verbose_level - 2);
		if (f_v) {
			cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines "
					"after SAL->report" << endl;
		}

	}
	FREE_OBJECT(SAL);


	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines done" << endl;
	}

}

void surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs(
		projective_geometry::projective_space_with_action *PA,
		poset_classification::poset_classification_control *Control1,
		poset_classification::poset_classification_control *Control2,
		std::string &Control_six_arcs_label,
		int f_test_nb_Eckardt_points, int nb_E,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	surface_with_action *Surf_A;
	algebraic_geometry::surface_domain *Surf;
	number_theory::number_theory_domain NT;



	Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
	Surf_A = NEW_OBJECT(surface_with_action);


	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf->init" << endl;
	}
	Surf->init(PA->F, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
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
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, verbose_level - 1);
	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_A->init" << endl;
	}



	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_A->Classify_trihedral_pairs->classify" << endl;
	}
	Surf_A->Classify_trihedral_pairs->classify(Control1, Control2, verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_A->Classify_trihedral_pairs->classify" << endl;
	}

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_arc->classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	cubic_surfaces_and_arcs::surface_classify_using_arc *Surf_arc;

	Surf_arc = NEW_OBJECT(cubic_surfaces_and_arcs::surface_classify_using_arc);


	Surf_arc->classify_surfaces_through_arcs_and_trihedral_pairs(
			Control_six_arcs_label,
			Surf_A,
			f_test_nb_Eckardt_points, nb_E,
			verbose_level);

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_arc->classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_arc->report" << endl;
	}


	if (orbiter_kernel_system::Orbiter->f_draw_options) {
		Surf_arc->report(orbiter_kernel_system::Orbiter->draw_options, verbose_level);
	}
	else {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"please use -draw_option for a report" << endl;
	}

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_arc->report" << endl;
	}

	FREE_OBJECT(Surf_arc);

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs done" << endl;
	}

}

void surface_domain_high_level::do_six_arcs(
		projective_geometry::projective_space_with_action *PA,
		std::string &Control_six_arcs_label,
		int f_filter_by_nb_Eckardt_points, int nb_Eckardt_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_six_arcs" << endl;
	}

	field_theory::finite_field *F;

	F = PA->F;


	algebraic_geometry::surface_domain *Surf;

	if (f_v) {
			cout << "surface_domain_high_level::do_six_arcs before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
	Surf->init(F, 0/*verbose_level - 1*/);
	if (f_v) {
		cout << "surface_domain_high_level::do_six_arcs after Surf->init" << endl;
	}



	cubic_surfaces_and_arcs::six_arcs_not_on_a_conic *Six_arcs;
	apps_geometry::arc_generator_description *Six_arc_descr;

	int *transporter;

	Six_arcs = NEW_OBJECT(cubic_surfaces_and_arcs::six_arcs_not_on_a_conic);

	Six_arc_descr = NEW_OBJECT(apps_geometry::arc_generator_description);
	Six_arc_descr->f_target_size = TRUE;
	Six_arc_descr->target_size = 6;
	Six_arc_descr->f_control = TRUE;
	Six_arc_descr->control_label.assign(Control_six_arcs_label);



	// classify six arcs not on a conic:

	if (f_v) {
		cout << "surface_domain_high_level::do_six_arcs "
				"Setting up the group of the plane:" << endl;
	}



	if (f_v) {
		cout << "surface_domain_high_level::do_six_arcs "
				"before Six_arcs->init:" << endl;
	}


	Six_arcs->init(
			Six_arc_descr,
			PA,
			FALSE, 0, //NULL,
			verbose_level);

	transporter = NEW_int(Six_arcs->Gen->PA->A->elt_size_in_int);

	int nb_orbits;
	int level = 6;

	nb_orbits = Six_arcs->Gen->gen->nb_orbits_at_level(level);

	if (f_v) {
		cout << "surface_domain_high_level::do_six_arcs "
				"We found " << nb_orbits << " isomorphism types "
				"of 6-arcs" << endl;
	}



	long int Arc6[6];
	int h, a, b, c, d;
	int v1[3];
	int v2[3];


	if (f_v) {
		cout << "surface_domain_high_level::do_six_arcs "
				"testing the arcs" << endl;
	}

	ring_theory::longinteger_object ago;
	int *Abcd;
	int *Nb_E;
	int *Ago;

	Abcd = NEW_int(nb_orbits * 4);
	Nb_E = NEW_int(nb_orbits);
	Ago = NEW_int(nb_orbits);

	for (h = 0; h < nb_orbits; h++) {

		if (f_v && (h % 10000) == 0) {
			cout << "surface_domain_high_level::do_six_arcs "
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

		algebraic_geometry::eckardt_point_info *E;

		geometry::geometry_global Gg;

		E = Gg.compute_eckardt_point_info(Surf->P2, Arc6, 0/*verbose_level*/);


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

	data_structures::tally C;

	C.init(Nb_E, nb_orbits, FALSE, 0);

	cout << "nb_E distribution: ";
	C.print_naked_tex(cout, FALSE);
	cout << endl;


	if (f_filter_by_nb_Eckardt_points) {
		cout << "Nonconical six-arcs associated with surfaces with "
				<< nb_Eckardt_points << " Eckardt points in PG(2," << F->q << "):" << endl;

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
		cout << "surface_domain_high_level::do_six_arcs done" << endl;
	}

}



void surface_domain_high_level::do_cubic_surface_properties(
		projective_geometry::projective_space_with_action *PA,
		std::string &fname_csv, int defining_q,
		int column_offset,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_cubic_surface_properties" << endl;
	}

	int i;
	field_theory::finite_field *F0;
	field_theory::finite_field *F;
	algebraic_geometry::surface_domain *Surf;
	surface_with_action *Surf_A;
	number_theory::number_theory_domain NT;
	data_structures::sorting Sorting;
	orbiter_kernel_system::file_io Fio;




	F0 = NEW_OBJECT(field_theory::finite_field);
	F0->finite_field_init(defining_q, FALSE /* f_without_tables */, 0);

	F = PA->P->F;


	Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
	Surf->init(F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "surface_domain_high_level::do_cubic_surface_properties "
				"after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "surface_domain_high_level::do_cubic_surface_properties "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_domain_high_level::do_cubic_surface_properties "
				"after Surf_A->init" << endl;
	}






	long int *M;
	int nb_orbits, n;

	Fio.lint_matrix_read_csv(fname_csv, M, nb_orbits, n, verbose_level);

	if (n != 3 + column_offset) {
		cout << "surface_domain_high_level::do_cubic_surface_properties "
				"n != 3 + column_offset" << endl;
		exit(1);
	}

	int orbit_idx;

	long int *Orbit;
	long int *Rep;
	long int *Stab_order;
	long int *Orbit_length;
	long int *Nb_pts;
	long int *Nb_lines;
	long int *Nb_Eckardt_points;
	long int *Nb_singular_pts;
	long int *Nb_Double_points;
	long int *Ago;

	Orbit = NEW_lint(nb_orbits);
	Rep = NEW_lint(nb_orbits);
	Stab_order = NEW_lint(nb_orbits);
	Orbit_length = NEW_lint(nb_orbits);
	Nb_pts = NEW_lint(nb_orbits);
	Nb_lines = NEW_lint(nb_orbits);
	Nb_Eckardt_points = NEW_lint(nb_orbits);
	Nb_singular_pts = NEW_lint(nb_orbits);
	Nb_Double_points = NEW_lint(nb_orbits);
	Ago = NEW_lint(nb_orbits);

	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (f_v) {
			cout << "surface_domain_high_level::do_cubic_surface_properties "
					"orbit_idx = " << orbit_idx << " / " << nb_orbits << endl;
		}
		int coeff20[20];
		char str[1000];


		Orbit[orbit_idx] = M[orbit_idx * n + 0];
		Rep[orbit_idx] = M[orbit_idx * n + column_offset + 0];
		Stab_order[orbit_idx] = M[orbit_idx * n + column_offset + 1];
		Orbit_length[orbit_idx] = M[orbit_idx * n + column_offset + 2];

		cout << "Rep=" << Rep[orbit_idx] << endl;
		F0->PG_element_unrank_modified_lint(coeff20, 1, 20, Rep[orbit_idx]);
		cout << "coeff20=";
		Int_vec_print(cout, coeff20, 20);
		cout << endl;

		surface_create_description *Descr;

		Descr = NEW_OBJECT(surface_create_description);
		//Descr->f_q = TRUE;
		//Descr->q = F->q;
		Descr->f_by_coefficients = TRUE;
		snprintf(str, sizeof(str), "%d,0", coeff20[0]);
		Descr->coefficients_text.assign(str);
		for (i = 1; i < 20; i++) {
			snprintf(str, sizeof(str), ",%d,%d", coeff20[i], i);
			Descr->coefficients_text.append(str);
		}
		cout << "Descr->coefficients_text = " << Descr->coefficients_text << endl;


		surface_create *SC;
		SC = NEW_OBJECT(surface_create);

		if (f_v) {
			cout << "surface_domain_high_level::do_cubic_surface_properties "
					"before SC->init" << endl;
		}
		SC->init(Descr, 0 /*verbose_level*/);
		if (f_v) {
			cout << "surface_domain_high_level::do_cubic_surface_properties "
					"after SC->init" << endl;
		}


		if (SC->F->e == 1) {
			SC->F->f_print_as_exponentials = FALSE;
		}

		SC->F->PG_element_normalize(SC->SO->eqn, 1, 20);

		if (f_v) {
			cout << "surface_domain_high_level::do_cubic_surface_properties "
					"We have created the following surface:" << endl;
			cout << "$$" << endl;
			SC->Surf->print_equation_tex(cout, SC->SO->eqn);
			cout << endl;
			cout << "$$" << endl;

			cout << "$$" << endl;
			Int_vec_print(cout, SC->SO->eqn, 20);
			cout << endl;
			cout << "$$" << endl;
		}


		// compute the group of the surface if we are over a small field.
		// Otherwise we don't, because it would take too long.


		if (FALSE /* F->q <= 8*/) {

#if 0
			if (f_v) {
				cout << "surface_domain_high_level::do_cubic_surface_properties "
						"before SC->compute_group" << endl;
			}
			SC->compute_group(PA, verbose_level);
			if (f_v) {
				cout << "surface_domain_high_level::do_cubic_surface_properties "
						"after SC->compute_group" << endl;
			}
			Ago[orbit_idx] = SC->Sg->group_order_as_lint();
#endif
		}
		else {
			cout << "F->q = " << F->q << " we are not computing the automorphism group" << endl;
			Ago[orbit_idx] = 0;
		}


		Nb_pts[orbit_idx] = SC->SO->nb_pts;
		Nb_lines[orbit_idx] = SC->SO->nb_lines;
		Nb_Eckardt_points[orbit_idx] = SC->SO->SOP->nb_Eckardt_points;
		Nb_singular_pts[orbit_idx] = SC->SO->SOP->nb_singular_pts;
		Nb_Double_points[orbit_idx] = SC->SO->SOP->nb_Double_points;

		//SC->SO->SOP->print_everything(ost, verbose_level);






		FREE_OBJECT(SC);
		FREE_OBJECT(Descr);


	}


	string fname_data;
	data_structures::string_tools ST;

	fname_data.assign(fname_csv);
	ST.chop_off_extension(fname_data);

	char str[1000];
	snprintf(str, sizeof(str), "_F%d.csv", F->q);
	fname_data.append(str);

	long int *Vec[10];
	char str_A[1000];
	char str_P[1000];
	char str_L[1000];
	char str_E[1000];
	char str_S[1000];
	char str_D[1000];
	snprintf(str_A, sizeof(str_A), "Ago-%d", F->q);
	snprintf(str_P, sizeof(str_P), "Nb_P-%d", F->q);
	snprintf(str_L, sizeof(str_L), "Nb_L-%d", F->q);
	snprintf(str_E, sizeof(str_E), "Nb_E-%d", F->q);
	snprintf(str_S, sizeof(str_S), "Nb_S-%d", F->q);
	snprintf(str_D, sizeof(str_D), "Nb_D-%d", F->q);
	const char *column_label[] = {
			"Orbit_idx",
			"Rep",
			"StabOrder",
			"OrbitLength",
			str_A,
			str_P,
			str_L,
			str_E,
			str_S,
			str_D,
	};

	Vec[0] = Orbit;
	Vec[1] = Rep;
	Vec[2] = Stab_order;
	Vec[3] = Orbit_length;
	Vec[4] = Ago;
	Vec[5] = Nb_pts;
	Vec[6] = Nb_lines;
	Vec[7] = Nb_Eckardt_points;
	Vec[8] = Nb_singular_pts;
	Vec[9] = Nb_Double_points;

	Fio.lint_vec_array_write_csv(10 /* nb_vecs */, Vec, nb_orbits,
			fname_data, column_label);

	if (f_v) {
		cout << "Written file " << fname_data << " of size "
				<< Fio.file_size(fname_data) << endl;
	}



	FREE_lint(M);
	//FREE_OBJECT(PA);
	FREE_OBJECT(F0);
	FREE_OBJECT(Surf);
	FREE_OBJECT(Surf_A);

	if (f_v) {
		cout << "surface_domain_high_level::do_cubic_surface_properties done" << endl;
	}
}


//! numerical data for one cubic surface to be used in reports

struct cubic_surface_data_set {

	int orbit_idx;
	long int Orbit_idx;
	long int Rep;
	long int Stab_order;
	long int Orbit_length;
	long int Ago;
	long int Nb_pts;
	long int Nb_lines;
	long int Nb_Eckardt_points;
	long int Nb_singular_pts;
	long int Nb_Double_points;

};

void surface_domain_high_level::do_cubic_surface_properties_analyze(
		projective_geometry::projective_space_with_action *PA,
		std::string &fname_csv, int defining_q,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_cubic_surface_properties_analyze" << endl;
	}

	field_theory::finite_field *F0;
	field_theory::finite_field *F;
	algebraic_geometry::surface_domain *Surf;
	surface_with_action *Surf_A;
	number_theory::number_theory_domain NT;
	data_structures::sorting Sorting;
	orbiter_kernel_system::file_io Fio;



	F0 = NEW_OBJECT(field_theory::finite_field);
	F0->finite_field_init(defining_q, FALSE /* f_without_tables */, 0);

	F = PA->P->F;


	Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
	Surf->init(F, 0 /* verbose_level - 1 */);
	if (f_v) {
		cout << "surface_domain_high_level::do_cubic_surface_properties_analyze "
				"after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "surface_domain_high_level::do_cubic_surface_properties_analyze "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_domain_high_level::do_cubic_surface_properties_analyze "
				"after Surf_A->init" << endl;
	}


	int nb_orbits, n;
	int orbit_idx;
	struct cubic_surface_data_set *Data;

	{
		long int *M;

		Fio.lint_matrix_read_csv(fname_csv, M, nb_orbits, n, verbose_level);

		if (n != 10) {
			cout << "surface_domain_high_level::do_cubic_surface_properties_analyze n != 10" << endl;
			exit(1);
		}





		Data = new struct cubic_surface_data_set [nb_orbits];

		for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
			Data[orbit_idx].orbit_idx = orbit_idx;
			Data[orbit_idx].Orbit_idx = M[orbit_idx * n + 0];
			Data[orbit_idx].Rep = M[orbit_idx * n + 1];
			Data[orbit_idx].Stab_order = M[orbit_idx * n + 2];
			Data[orbit_idx].Orbit_length = M[orbit_idx * n + 3];
			Data[orbit_idx].Ago = M[orbit_idx * n + 4];
			Data[orbit_idx].Nb_pts = M[orbit_idx * n + 5];
			Data[orbit_idx].Nb_lines = M[orbit_idx * n + 6];
			Data[orbit_idx].Nb_Eckardt_points = M[orbit_idx * n + 7];
			Data[orbit_idx].Nb_singular_pts = M[orbit_idx * n + 8];
			Data[orbit_idx].Nb_Double_points = M[orbit_idx * n + 9];
		}
		FREE_lint(M);
	}
	long int *Nb_singular_pts;

	Nb_singular_pts = NEW_lint(nb_orbits);
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		Nb_singular_pts[orbit_idx] = Data[orbit_idx].Nb_singular_pts;
	}


	data_structures::tally T_S;

	T_S.init_lint(Nb_singular_pts, nb_orbits, FALSE, 0);

	cout << "Classification by the number of singular points:" << endl;
	T_S.print(TRUE /* f_backwards */);

	{
		string fname_report;
		data_structures::string_tools ST;

		fname_report.assign(fname_csv);
		ST.chop_off_extension(fname_report);
		fname_report.append("_report.tex");
		orbiter_kernel_system::latex_interface L;
		orbiter_kernel_system::file_io Fio;

		{
			ofstream ost(fname_report);
			L.head_easy(ost);

#if 0
			if (f_v) {
				cout << "surface_domain_high_level::do_cubic_surface_properties_analyze "
						"before get_A()->report" << endl;
			}

			if (!Descr->f_draw_options) {
				cout << "please use -draw_options" << endl;
				exit(1);
			}
			PA->A->report(ost,
					FALSE /* f_sims */,
					NULL, //A1/*LG->A_linear*/->Sims,
					FALSE /* f_strong_gens */,
					NULL,
					Descr->draw_options,
					verbose_level - 1);

			if (f_v) {
				cout << "surface_domain_high_level::do_cubic_surface_properties_analyze "
						"after LG->A_linear->report" << endl;
			}
#endif

			if (f_v) {
				cout << "surface_domain_high_level::do_cubic_surface_properties_analyze "
						"before report" << endl;
			}


			ost << "\\section{Surfaces over ${\\mathbb F}_{" << F->q << "}$}" << endl;


			ost << "Number of surfaces: " << nb_orbits << "\\\\" << endl;
			ost << "Classification by the number of singular points:" << endl;
			ost << "$$" << endl;
			T_S.print_file_tex_we_are_in_math_mode(ost, TRUE /* f_backwards */);
			ost << "$$" << endl;


			ost << "\\section{Singular Surfaces}" << endl;

			report_singular_surfaces(ost, Data, nb_orbits, verbose_level);

			ost << "\\section{Nonsingular Surfaces}" << endl;

			report_non_singular_surfaces(ost, Data, nb_orbits, verbose_level);



			if (f_v) {
				cout << "surface_domain_high_level::do_cubic_surface_properties_analyze "
						"after report" << endl;
			}

			L.foot(ost);
		}
		cout << "Written file " << fname_report << " of size "
				<< Fio.file_size(fname_report) << endl;
	}





	//FREE_OBJECT(PA);
	FREE_OBJECT(F0);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);

	if (f_v) {
		cout << "surface_domain_high_level::do_cubic_surface_properties_analyze done" << endl;
	}
}

void surface_domain_high_level::report_singular_surfaces(std::ostream &ost,
		struct cubic_surface_data_set *Data, int nb_orbits, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::report_singular_surfaces" << endl;
	}

	struct cubic_surface_data_set *Data_S;
	int nb_S, h, orbit_idx;


	nb_S = 0;
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (Data[orbit_idx].Nb_singular_pts) {
			nb_S++;
		}
	}


	Data_S = new struct cubic_surface_data_set [nb_S];

	h = 0;
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (Data[orbit_idx].Nb_singular_pts) {
			Data_S[h] = Data[orbit_idx];
			h++;
		}
	}
	if (h != nb_S) {
		cout << "h != nb_S" << endl;
		exit(1);
	}

	long int *Selected_Nb_lines;


	Selected_Nb_lines = NEW_lint(nb_S);


	for (h = 0; h < nb_S; h++) {
		Selected_Nb_lines[h] = Data_S[h].Nb_lines;
	}

	data_structures::tally T_L;

	T_L.init_lint(Selected_Nb_lines, nb_S, FALSE, 0);

	ost << "Number of surfaces: " << nb_S << "\\\\" << endl;
	ost << "Classification by the number of lines:" << endl;
	ost << "$$" << endl;
	T_L.print_file_tex_we_are_in_math_mode(ost, TRUE /* f_backwards */);
	ost << "$$" << endl;

	report_surfaces_by_lines(ost, Data_S, T_L, verbose_level);



	FREE_lint(Selected_Nb_lines);
	delete [] Data_S;

	if (f_v) {
		cout << "surface_domain_high_level::report_singular_surfaces done" << endl;
	}
}


void surface_domain_high_level::report_non_singular_surfaces(std::ostream &ost,
		struct cubic_surface_data_set *Data, int nb_orbits, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::report_non_singular_surfaces" << endl;
	}

	struct cubic_surface_data_set *Data_NS;
	int nb_NS, h, orbit_idx;


	nb_NS = 0;
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (Data[orbit_idx].Nb_singular_pts == 0) {
			nb_NS++;
		}
	}


	Data_NS = new struct cubic_surface_data_set [nb_NS];

	h = 0;
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (Data[orbit_idx].Nb_singular_pts == 0) {
			Data_NS[h] = Data[orbit_idx];
			h++;
		}
	}
	if (h != nb_NS) {
		cout << "h != nb_NS" << endl;
		exit(1);
	}

	long int *Selected_Nb_lines;


	Selected_Nb_lines = NEW_lint(nb_NS);


	for (h = 0; h < nb_NS; h++) {
		Selected_Nb_lines[h] = Data_NS[h].Nb_lines;
	}

	for (h = 0; h < nb_NS; h++) {
		cout << h << " : " << Data_NS[h].orbit_idx << " : " << Data_NS[h].Nb_lines << endl;
	}

	data_structures::tally T_L;

	T_L.init_lint(Selected_Nb_lines, nb_NS, FALSE, 0);

	ost << "Number of surfaces: " << nb_NS << "\\\\" << endl;
	ost << "Classification by the number of lines:" << endl;
	ost << "$$" << endl;
	T_L.print_file_tex_we_are_in_math_mode(ost, TRUE /* f_backwards */);
	ost << "$$" << endl;


	report_surfaces_by_lines(ost, Data_NS, T_L, verbose_level);


	FREE_lint(Selected_Nb_lines);
	delete [] Data_NS;

	if (f_v) {
		cout << "surface_domain_high_level::report_non_singular_surfaces done" << endl;
	}
}

void surface_domain_high_level::report_surfaces_by_lines(std::ostream &ost,
		struct cubic_surface_data_set *Data,
		data_structures::tally &T, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::report_surfaces_by_lines" << endl;
	}

	int i, j, f, l, a, idx;

	for (i = T.nb_types - 1; i >= 0; i--) {
		f = T.type_first[i];
		l = T.type_len[i];
		a = T.data_sorted[f];

		int nb_L;
		struct cubic_surface_data_set *Data_L;

		nb_L = l;

		Data_L = new struct cubic_surface_data_set [nb_L];

		ost << "The number of surfaces with exactly " << a << " lines is " << nb_L << ": \\\\" << endl;

		for (j = 0; j < l; j++) {
			idx = T.sorting_perm_inv[f + j];
			Data_L[j] = Data[idx];

		}


		for (j = 0; j < l; j++) {
			ost << j
					<< " : i=" << Data_L[j].orbit_idx
					<< " : id=" << Data_L[j].Orbit_idx
					<< " : P=" << Data_L[j].Nb_pts
					<< " : S=" << Data_L[j].Nb_singular_pts
					<< " : E=" << Data_L[j].Nb_Eckardt_points
					<< " : D=" << Data_L[j].Nb_Double_points
					<< " : ago=" << Data_L[j].Ago
					<< " : Rep=" << Data_L[j].Rep
				<< "\\\\" << endl;
		}

		delete [] Data_L;
	}
	if (f_v) {
		cout << "surface_domain_high_level::report_surfaces_by_lines done" << endl;
	}

}



//! data on a single cubic surface used to prepare the ATLAS of cubic surfaces

struct table_surfaces_field_order {
	int q;
	int p;
	int h;

	field_theory::finite_field *F;

	projective_geometry::projective_space_with_action *PA;

	algebraic_geometry::surface_domain *Surf;
	surface_with_action *Surf_A;

	int nb_total;
	int *nb_E;

	data_structures::tally *T_nb_E;



};

void surface_domain_high_level::do_create_surface_reports(std::string &field_orders_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_create_surface_reports" << endl;
		cout << "surface_domain_high_level::do_create_surface_reports verbose_level=" << verbose_level << endl;
	}

	knowledge_base K;
	number_theory::number_theory_domain NT;
	orbiter_kernel_system::file_io Fio;


	int *Q;
	int nb_q;

	Int_vec_scan(field_orders_text, Q, nb_q);

	int q;
	//int cur;
	int i;

	//cur = 0;
	for (i = 0; i < nb_q; i++) {

		q = Q[i];


		int nb_total;
		int ocn;

		nb_total = K.cubic_surface_nb_reps(q);

		if (f_v) {
			cout << "surface_domain_high_level::do_create_surface_reports considering q=" << q << " with " << nb_total << " surfaces" << endl;
		}


		for (ocn = 0; ocn < nb_total; ocn++) {

			string cmd;
			string fname;
			char str[1000];
			char str_ocn[1000];

			if (f_v) {
				cout << "surface_domain_high_level::do_create_surface_reports considering q=" << q << " ocn=" << ocn << " / " << nb_total << endl;
			}

			make_fname_surface_report_tex(fname, q, ocn);

#if 0
			$(ORBITER_PATH)orbiter.out -v 3 \
				-define F -finite_field -q 4 -end \
				-define P -projective_space 3 F -end \
				-with P -do \
				-projective_space_activity \
					-define_surface S -q 4 -catalogue 0 -end \
				-end \
				-with S -do \
				-cubic_surface_activity \
					-report \
					-report_with_group \
					-all_quartic_curves \
				-end
#endif


				snprintf(str, sizeof(str), "%d ", q);
				snprintf(str_ocn, sizeof(str_ocn), "%d ", ocn);

			cmd.assign(orbiter_kernel_system::Orbiter->orbiter_path);
			cmd.append("/orbiter.out -v 3 ");
			cmd.append("-define F -finite_field -q ");
			cmd.append(str);
			cmd.append("-end ");
			cmd.append("-define P -projective_space 3 F -end ");
			cmd.append("-with P -do ");
			cmd.append("-projective_space_activity ");
			cmd.append("-define_surface S -q ");
			cmd.append(str);
			cmd.append("-catalogue ");
			cmd.append(str_ocn);
			cmd.append("-end ");
			cmd.append("-end ");
			cmd.append("-with S -do ");
			cmd.append("-cubic_surface_activity ");
			cmd.append("-report ");
			cmd.append("-report_with_group ");
			//cmd.append("-all_quartic_curves ");
			cmd.append("-end >log_surface");

			if (f_v) {
				cout << "executing command: " << cmd << endl;
			}
			system(cmd.c_str());

			std::string fname_report_tex;

			make_fname_surface_report_tex(fname_report_tex, q, ocn);

			cmd.assign("pdflatex ");
			cmd.append(fname_report_tex);
			cmd.append(" >log_pdflatex");

			if (f_v) {
				cout << "executing command: " << cmd << endl;
			}
			system(cmd.c_str());


		}


	}

	if (f_v) {
		cout << "surface_domain_high_level::do_create_surface_reports done" << endl;
	}
}

void surface_domain_high_level::do_create_surface_atlas(int q_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_create_surface_atlas" << endl;
		cout << "surface_domain_high_level::do_create_surface_atlas verbose_level=" << verbose_level << endl;
	}

	knowledge_base K;


	number_theory::number_theory_domain NT;
	data_structures::sorting Sorting;
	orbiter_kernel_system::file_io Fio;


	struct table_surfaces_field_order *T;

	T = new struct table_surfaces_field_order[q_max];

	int q;
	int cur;
	int j;

	cur = 0;
	for (q = 2; q <= q_max; q++) {

		int p;
		int h;

		if (!NT.is_prime_power(q, p, h)) {
			continue;
		}

		cout << "considering q=" << q << endl;


		T[cur].q = q;
		T[cur].p = p;
		T[cur].h = h;

		int f_semilinear;

		if (h > 1) {
			f_semilinear = TRUE;
		}
		else {
			f_semilinear = FALSE;
		}

#if 0
		T[cur].Descr = NEW_OBJECT(linear_group_description);

		T[cur].Descr->n = 4;
		T[cur].Descr->input_q = q;
		T[cur].Descr->f_projective = TRUE;
		T[cur].Descr->f_general = FALSE;
		T[cur].Descr->f_affine = FALSE;
		T[cur].Descr->f_semilinear = FALSE;

		if (h > 1) {
			T[cur].Descr->f_semilinear = TRUE;
		}
		T[cur].Descr->f_special = FALSE;
#endif

		T[cur].F = NEW_OBJECT(field_theory::finite_field);
		T[cur].F->finite_field_init(q, FALSE /* f_without_tables */, 0);

		//T[cur].Descr->F = T[cur].F;


		T[cur].PA = NEW_OBJECT(projective_geometry::projective_space_with_action);
		T[cur].PA->init(T[cur].F, 3, f_semilinear,
				TRUE /* f_init_incidence_structure */,
				verbose_level);

		//T[cur].LG = NEW_OBJECT(linear_group);

		//cout << "before LG->linear_group_init" << endl;
		//T[cur].LG->linear_group_init(T[cur].Descr, verbose_level);








		if (f_v) {
			cout << "surface_domain_high_level::do_create_surface_atlas before Surf->init" << endl;
		}

		T[cur].Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
		T[cur].Surf->init(T[cur].F, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "do_create_surface_atlas after Surf->init" << endl;
		}

		T[cur].Surf_A = NEW_OBJECT(surface_with_action);

		if (f_v) {
			cout << "surface_domain_high_level::do_create_surface_atlas "
					"before Surf_A->init_with_linear_group" << endl;
		}
		T[cur].Surf_A->init(T[cur].Surf, T[cur].PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
		if (f_v) {
			cout << "surface_domain_high_level::do_create_surface_atlas "
					"after Surf_A->init_with_linear_group" << endl;
		}


		if (T[cur].q == 2) {
			cur++;
			continue;
		}
		if (T[cur].q == 3) {
			cur++;
			continue;
		}
		if (T[cur].q == 5) {
			cur++;
			continue;
		}


		T[cur].nb_total = K.cubic_surface_nb_reps(T[cur].q);


		T[cur].nb_E = NEW_int(T[cur].nb_total);

		for (j = 0; j < T[cur].nb_total; j++) {
			T[cur].nb_E[j] = K.cubic_surface_nb_Eckardt_points(T[cur].q, j);
		}

		T[cur].T_nb_E = NEW_OBJECT(data_structures::tally);

		T[cur].T_nb_E->init(T[cur].nb_E, T[cur].nb_total, FALSE, 0);


		cur++;
	}

	cout << "we found the following field orders:" << endl;

	int nb_fields;
	int c;


	nb_fields = cur;

	for (c = 0; c < nb_fields; c++) {
		cout << c << " : " << T[c].q << endl;
	}



	{
		string fname_report;

		fname_report.assign("surface");
		fname_report.append("_atlas.tex");

		{
			ofstream ost(fname_report);

			string title, author, extra_praeamble;

			title.assign("ATLAS of Cubic Surfaces");
			author.assign("Anton Betten and Fatma Karaoglu");

			orbiter_kernel_system::latex_interface L;

			L.head(ost,
				FALSE /* f_book */,
				TRUE /* f_title */,
				title, author,
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				TRUE /*f_enlarged_page */,
				TRUE /* f_pagenumbers*/,
				extra_praeamble /* extra_praeamble */);


			int E[] = {0,1,2,3,4,5,6,9,10,13,18,45};
			int nb_possible_E = sizeof(E) / sizeof(int);
			int j;

			ost << "$$" << endl;
			ost << "\\begin{array}{|c|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "\\ \\ q \\ \\ ";
			ost << "& \\ \\ \\mbox{Total} \\ \\ ";
			for (j = 0; j < nb_possible_E; j++) {
				ost << "&\\ \\ " << E[j] << "\\ \\ ";
			}
			ost << "\\\\" << endl;
			ost << "\\hline" << endl;
			for (c = 0; c < nb_fields; c++) {

				if (T[c].q == 2) {
					continue;
				}
				if (T[c].q == 3) {
					continue;
				}
				if (T[c].q == 5) {
					continue;
				}
				//ost << c << " & ";
				ost << T[c].q << " " << endl;

				ost << " & " << T[c].nb_total << " " << endl;

				for (j = 0; j < nb_possible_E; j++) {

					int *Idx;
					int nb;

					T[c].T_nb_E->get_class_by_value(Idx, nb, E[j], 0);

					if (nb) {

						int nb_e = E[j];
						string fname_report_tex;
						string fname_report_html;

						do_create_surface_atlas_q_e(q_max,
								T + c, nb_e, Idx, nb,
								fname_report_tex,
								verbose_level);

						data_structures::string_tools ST;

						fname_report_html.assign(fname_report_tex);
						ST.chop_off_extension(fname_report_html);
						fname_report_html.append(".html");


						ost << " & ";
						ost << "%%tth: \\begin{html} <a href=\"" << fname_report_html << "\"> " << nb << " </a> \\end{html}" << endl;


						string cmd;

						cmd.assign("~/bin/tth ");
						cmd.append(fname_report_tex);
						system(cmd.c_str());
					}
					else {
						ost << " & ";
					}

					FREE_int(Idx);
				}
				ost << "\\\\" << endl;
				ost << "\\hline" << endl;
			}

			//

			ost << "\\end{array}" << endl;

			ost << "$$" << endl;

#if 0
			ost << "\\subsection*{The surface $" << SC->label_tex << "$}" << endl;


			if (SC->SO->SOP == NULL) {
				cout << "group_theoretic_activity::do_create_surface SC->SO->SOP == NULL" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "group_theoretic_activity::do_create_surface "
						"before SC->SO->SOP->print_everything" << endl;
			}
			SC->SO->SOP->print_everything(ost, verbose_level);
			if (f_v) {
				cout << "group_theoretic_activity::do_create_surface "
						"after SC->SO->SOP->print_everything" << endl;
			}
#endif

			L.foot(ost);
		}
		orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	}





	if (f_v) {
		cout << "surface_domain_high_level::do_create_surface_atlas done" << endl;
	}
}



void surface_domain_high_level::do_create_surface_atlas_q_e(int q_max,
		struct table_surfaces_field_order *T, int nb_e, int *Idx, int nb,
		std::string &fname_report_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_create_surface_atlas_q_e" << endl;
		cout << "surface_domain_high_level::do_create_surface_atlas q=" << T->q << " " << nb_e << endl;
	}

	knowledge_base K;


	number_theory::number_theory_domain NT;
	data_structures::sorting Sorting;
	orbiter_kernel_system::file_io Fio;




	{
		char str[1000];

		snprintf(str, sizeof(str), "_q%d_e%d", T->q, nb_e);
		fname_report_tex.assign("surface_atlas");
		fname_report_tex.append(str);
		fname_report_tex.append(".tex");

		{
			ofstream ost(fname_report_tex);


			string title, author, extra_praeamble;

			title.assign("ATLAS of Cubic Surfaces");
			snprintf(str, sizeof(str), ", q=%d, \\#E=%d", T->q, nb_e);
			title.append(str);

			author.assign("Anton Betten and Fatma Karaoglu");

			orbiter_kernel_system::latex_interface L;

			//latex_head_easy(fp);
			L.head(ost,
				FALSE /* f_book */,
				TRUE /* f_title */,
				title, author,
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				TRUE /*f_enlarged_page */,
				TRUE /* f_pagenumbers*/,
				extra_praeamble /* extra_praeamble */);


			int i;

			ost << "$$" << endl;
			ost << "\\begin{array}{|c|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "\\ \\ i \\ \\ ";
			ost << "& \\ \\ \\mbox{Orbiter Number} \\ \\ ";
			ost << "& \\ \\ \\mbox{Report} \\ \\ ";
			ost << "\\\\" << endl;
			ost << "\\hline" << endl;
			for (i = 0; i < nb; i++) {

				//ost << c << " & ";
				ost << i << " " << endl;

				ost << " & " << Idx[i] << " " << endl;


				std::string fname;

				make_fname_surface_report_pdf(fname, T->q, Idx[i]);

				ost << " & " << endl;
				ost << "%%tth: \\begin{html} <a href=\"" << fname << "\"> report </a> \\end{html}" << endl;

				ost << "\\\\" << endl;
				ost << "\\hline" << endl;
			}

			//

			ost << "\\end{array}" << endl;

			ost << "$$" << endl;


			L.foot(ost);
		}
		orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_report_tex << " of size "
			<< Fio.file_size(fname_report_tex) << endl;


	}

}

void surface_domain_high_level::do_create_dickson_atlas(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_create_dickson_atlas" << endl;
		cout << "surface_domain_high_level::do_create_dickson_atlas verbose_level=" << verbose_level << endl;
	}

	orbiter_kernel_system::file_io Fio;





	{
		string fname_report;

		fname_report.assign("dickson_surfaces");
		fname_report.append(".tex");

		{
			ofstream ost(fname_report);

			string title, author, extra_praeamble;

			title.assign("ATLAS of Dickson Surfaces");
			author.assign("Fatma Karaoglu");

			orbiter_kernel_system::latex_interface L;

			//latex_head_easy(fp);
			L.head(ost,
				FALSE /* f_book */,
				TRUE /* f_title */,
				title, author,
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				TRUE /*f_enlarged_page */,
				TRUE /* f_pagenumbers*/,
				extra_praeamble /* extra_praeamble */);


			int field_orders[] = {2,4,8,16,32,64};
			int nb_of_fields = sizeof(field_orders) / sizeof(int);
			int i, j, c;
			int I, N;

			N = (141 + 24) / 25;
			for (I = 0; I < N; I++) {

				ost << "$$" << endl;
				ost << "\\begin{array}{|r|*{" << nb_of_fields << "}{r|}}" << endl;
				ost << "\\hline" << endl;
				ost << "\\ \\ D-i \\ \\ ";
				for (j = 0; j < nb_of_fields; j++) {
					ost << "&\\ \\ " << field_orders[j] << "\\ \\ ";
				}
				ost << "\\\\" << endl;
				ost << "\\hline" << endl;
				for (i = 0; i < 25; i++) {
					c = I * 25 + i;


					if (c >= 141) {
						continue;
					}

					cout << "creating line " << c << endl;

					ost << c << " " << endl;


					for (j = 0; j < nb_of_fields; j++) {

						string fname_base;
						string fname_tex;
						string fname_pdf;
						string fname_surface_report;


						char str[1000];


						snprintf(str, sizeof(str), "Orb%d_q%d", c, field_orders[j]);
						fname_base.assign(str);
						fname_tex.assign(fname_base);
						fname_tex.append(".tex");
						fname_pdf.assign(fname_base);
						fname_pdf.append(".pdf");
						fname_surface_report.assign(fname_base);
						fname_surface_report.append(".pdf");


						ost << " & " << endl;
						ost << "%%tth: \\begin{html} <a href=\"" << fname_surface_report << "\"> " << fname_surface_report << " </a> \\end{html}" << endl;


						if (Fio.file_size(fname_tex.c_str()) > 0) {

							if (Fio.file_size(fname_pdf.c_str()) <= 0) {
								string cmd;

								cmd.assign("pdflatex ");
								cmd.append(fname_tex);
								cmd.append(" ");
								system(cmd.c_str());
							}
						}

					}
					ost << "\\\\" << endl;
					ost << "\\hline" << endl;
				}

				//

				ost << "\\end{array}" << endl;

				ost << "$$" << endl;
			}

			L.foot(ost);
		}
		orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	}


	if (f_v) {
		cout << "surface_domain_high_level::do_create_dickson_atlas done" << endl;
	}
}




void surface_domain_high_level::make_fname_surface_report_tex(std::string &fname, int q, int ocn)
{
	char str[1000];

	snprintf(str, sizeof(str), "_q%d_iso%d_with_group", q, ocn);
	fname.assign("surface_catalogue");
	fname.append(str);
	fname.append(".tex");
}

void surface_domain_high_level::make_fname_surface_report_pdf(std::string &fname, int q, int ocn)
{
	char str[1000];

	snprintf(str, sizeof(str), "_q%d_iso%d_with_group", q, ocn);
	fname.assign("surface_catalogue");
	fname.append(str);
	fname.append(".pdf");
}






}}}}


