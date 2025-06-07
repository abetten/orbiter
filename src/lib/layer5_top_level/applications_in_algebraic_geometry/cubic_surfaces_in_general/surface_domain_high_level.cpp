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
	Record_birth();
}

surface_domain_high_level::~surface_domain_high_level()
{
	Record_death();
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

	poset_classification::poset_classification_control *Control;

	Control =
			Get_poset_classification_control(control_label);


	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes "
				"before classify_surfaces, control=" << endl;
		Control->print();
	}


	SCW = NEW_OBJECT(cubic_surfaces_and_double_sixes::surface_classify_wedge);

	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes "
				"before SCW->init" << endl;
	}

	SCW->init(PA,
			Control,
			verbose_level - 1);

	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes "
				"after SCW->init" << endl;
	}


	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes "
				"before SCW->do_classify_double_sixes" << endl;
	}
	SCW->do_classify_double_sixes(verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes "
				"after SCW->do_classify_double_sixes" << endl;
	}

	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes "
				"before SCW->do_classify_surfaces" << endl;
	}
	SCW->do_classify_surfaces(verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes "
				"after SCW->do_classify_surfaces" << endl;
	}

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
	if (false) {

		layered_graph_draw_options *O;


		if (!Orbiter->f_draw_options) {
			cout << "please use option -draw_options .. -end" << endl;
			exit(1);
		}
		O = Orbiter->draw_options;


		SCW->create_report(true /*f_with_stabilizers */,
				O,
				verbose_level);
	}
#endif

	if (f_v) {
		cout << "surface_domain_high_level::classify_surfaces_with_double_sixes done" << endl;
	}
}


void surface_domain_high_level::do_study_surface(
		algebra::field_theory::finite_field *F,
		int nb, int verbose_level)
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


void surface_domain_high_level::do_recognize_surfaces(
		projective_geometry::projective_space_with_action *PA,
		std::string &Control_six_arcs_label,
		int f_test_nb_Eckardt_points, int nb_E,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_recognize_surfaces" << endl;
	}


	cubic_surfaces_and_arcs::surfaces_arc_lifting *SAL;

	SAL = NEW_OBJECT(cubic_surfaces_and_arcs::surfaces_arc_lifting);

	if (f_v) {
		cout << "surface_domain_high_level::do_recognize_surfaces "
				"before SAL->init" << endl;
	}
	SAL->init(
		PA->Surf_A,
		Control_six_arcs_label,
		f_test_nb_Eckardt_points, nb_E,
		verbose_level - 2);
	if (f_v) {
		cout << "surface_domain_high_level::do_recognize_surfaces "
				"after SAL->init" << endl;
	}


	// create all surfaces from the knowledge base

	geometry::algebraic_geometry::surface_object **SO;
	int nb_iso;

	if (f_v) {
		cout << "surface_domain_high_level::do_recognize_surfaces "
				"before get_list_of_all_surfaces" << endl;
	}
	PA->Surf_A->Surf->get_list_of_all_surfaces(
			SO, nb_iso,
			verbose_level - 2);
	if (f_v) {
		cout << "surface_domain_high_level::do_recognize_surfaces "
				"nb_iso = " << nb_iso << endl;
	}



	if (f_v) {
		cout << "surface_domain_high_level::do_recognize_surfaces "
				"after get_list_of_all_surfaces" << endl;
	}

	PA->Surf_A->Surf->dispose_of_list_of_all_surfaces(
			SO,
			verbose_level);


	if (f_v) {
		cout << "surface_domain_high_level::do_recognize_surfaces done" << endl;
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


	cubic_surfaces_and_arcs::surfaces_arc_lifting *SAL;

	SAL = NEW_OBJECT(cubic_surfaces_and_arcs::surfaces_arc_lifting);

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines "
				"before SAL->init" << endl;
	}
	SAL->init(
		PA->Surf_A,
		Control_six_arcs_label,
		f_test_nb_Eckardt_points, nb_E,
		verbose_level - 2);
	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines "
				"after SAL->init" << endl;
	}

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines "
				"before SAL->upstep" << endl;
	}

	SAL->upstep(
		verbose_level);

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_two_lines "
				"after SAL->upstep" << endl;
	}


	if (true) {
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
		other::graphics::layered_graph_draw_options *Draw_options,
		std::string &Control_six_arcs_label,
		int f_test_nb_Eckardt_points, int nb_E,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}



	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_A->Classify_trihedral_pairs->classify" << endl;
	}
	PA->Surf_A->Classify_trihedral_pairs->classify(Control1, Control2, verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Surf_A->Classify_trihedral_pairs->classify" << endl;
	}


	cubic_surfaces_and_arcs::surface_classify_using_arc *Surf_arc;

	Surf_arc = NEW_OBJECT(cubic_surfaces_and_arcs::surface_classify_using_arc);

	if (f_v) {
		cout << "surface_domain_high_level::do_classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Surf_arc->classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	Surf_arc->classify_surfaces_through_arcs_and_trihedral_pairs(
			Control_six_arcs_label,
			PA,
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


	Surf_arc->report(Draw_options, verbose_level);

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

	algebra::field_theory::finite_field *F;

	F = PA->F;




	cubic_surfaces_and_arcs::six_arcs_not_on_a_conic *Six_arcs;
	apps_geometry::arc_generator_description *Six_arc_descr;

	int *transporter;

	Six_arcs = NEW_OBJECT(cubic_surfaces_and_arcs::six_arcs_not_on_a_conic);

	Six_arc_descr = NEW_OBJECT(apps_geometry::arc_generator_description);
	Six_arc_descr->f_target_size = true;
	Six_arc_descr->target_size = 6;
	Six_arc_descr->f_control = true;
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
			false, 0, //NULL,
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

	algebra::ring_theory::longinteger_object ago;
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
		PA->Surf_A->Surf->P2->unrank_point(v1, Arc6[4]);
		PA->Surf_A->Surf->P2->unrank_point(v2, Arc6[5]);
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

		geometry::algebraic_geometry::eckardt_point_info *E;

		geometry::algebraic_geometry::algebraic_geometry_global Gg;

		E = Gg.compute_eckardt_point_info(
				PA->Surf_A->Surf->P2, Arc6,
				0/*verbose_level*/);


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

	other::data_structures::tally C;

	C.init(Nb_E, nb_orbits, false, 0);

	cout << "nb_E distribution: ";
	C.print_bare_tex(cout, false);
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
	algebra::field_theory::finite_field *F0;
	algebra::field_theory::finite_field *F;
	algebra::number_theory::number_theory_domain NT;
	other::data_structures::sorting Sorting;
	other::orbiter_kernel_system::file_io Fio;




	F0 = NEW_OBJECT(algebra::field_theory::finite_field);
	F0->finite_field_init_small_order(
			defining_q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);

	F = PA->P->Subspaces->F;





	long int *M;
	int nb_orbits, n;

	Fio.Csv_file_support->lint_matrix_read_csv(
			fname_csv, M, nb_orbits, n, verbose_level);

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


		Orbit[orbit_idx] = M[orbit_idx * n + 0];
		Rep[orbit_idx] = M[orbit_idx * n + column_offset + 0];
		Stab_order[orbit_idx] = M[orbit_idx * n + column_offset + 1];
		Orbit_length[orbit_idx] = M[orbit_idx * n + column_offset + 2];

		cout << "Rep=" << Rep[orbit_idx] << endl;
		F0->Projective_space_basic->PG_element_unrank_modified_lint(
				coeff20, 1, 20, Rep[orbit_idx]);
		cout << "coeff20=";
		Int_vec_print(cout, coeff20, 20);
		cout << endl;

		surface_create_description *Descr;

		Descr = NEW_OBJECT(surface_create_description);
		//Descr->f_q = true;
		//Descr->q = F->q;
		Descr->f_by_coefficients = true;
		Descr->coefficients_text = std::to_string(coeff20[0]) + ",0";
		for (i = 1; i < 20; i++) {
			Descr->coefficients_text += "," + std::to_string(coeff20[i]) + "," + std::to_string(i);
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

#if 0
		if (SC->F->e == 1) {
			SC->F->f_print_as_exponentials = false;
		}
#endif

		SC->F->Projective_space_basic->PG_element_normalize(
				SC->SO->Variety_object->eqn, 1, 20);

		if (f_v) {
			cout << "surface_domain_high_level::do_cubic_surface_properties "
					"We have created the following surface:" << endl;
			cout << "$$" << endl;
			SC->Surf->print_equation_tex(cout, SC->SO->Variety_object->eqn);
			cout << endl;
			cout << "$$" << endl;

			cout << "$$" << endl;
			Int_vec_print(cout, SC->SO->Variety_object->eqn, 20);
			cout << endl;
			cout << "$$" << endl;
		}


		// compute the group of the surface if we are over a small field.
		// Otherwise we don't, because it would take too long.


		if (false /* F->q <= 8*/) {

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


		Nb_pts[orbit_idx] = SC->SO->Variety_object->Point_sets->Set_size[0];
		Nb_lines[orbit_idx] = SC->SO->Variety_object->Line_sets->Set_size[0];
		Nb_Eckardt_points[orbit_idx] = SC->SO->SOP->nb_Eckardt_points;
		Nb_singular_pts[orbit_idx] = SC->SO->SOP->nb_singular_pts;
		Nb_Double_points[orbit_idx] = SC->SO->SOP->nb_Double_points;

		//SC->SO->SOP->print_everything(ost, verbose_level);






		FREE_OBJECT(SC);
		FREE_OBJECT(Descr);


	}


	string fname_data;
	other::data_structures::string_tools ST;

	fname_data = fname_csv;
	ST.chop_off_extension(fname_data);

	fname_data += "_F" + std::to_string(F->q) + ".csv";


	long int *Vec[10];
	string *column_label;

	column_label = new string[10];
	column_label[0] = "Orbit_idx";
	column_label[1] = "Rep";
	column_label[2] = "StabOrder";
	column_label[3] = "OrbitLength";
	column_label[4] = "Ago-" + std::to_string(F->q);
	column_label[5] = "Nb_P-" + std::to_string(F->q);
	column_label[6] = "Nb_L-" + std::to_string(F->q);
	column_label[7] = "Nb_E-" + std::to_string(F->q);
	column_label[8] = "Nb_S-" + std::to_string(F->q);
	column_label[9] = "Nb_D-" + std::to_string(F->q);

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

	Fio.Csv_file_support->lint_vec_array_write_csv(
			10 /* nb_vecs */, Vec, nb_orbits,
			fname_data, column_label);

	if (f_v) {
		cout << "Written file " << fname_data << " of size "
				<< Fio.file_size(fname_data) << endl;
	}



	FREE_lint(M);
	//FREE_OBJECT(PA);
	FREE_OBJECT(F0);
	//FREE_OBJECT(Surf);
	//FREE_OBJECT(Surf_A);

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

	algebra::field_theory::finite_field *F0;
	algebra::field_theory::finite_field *F;
	algebra::number_theory::number_theory_domain NT;
	other::data_structures::sorting Sorting;
	other::orbiter_kernel_system::file_io Fio;



	F0 = NEW_OBJECT(algebra::field_theory::finite_field);
	F0->finite_field_init_small_order(defining_q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);

	F = PA->P->Subspaces->F;


	int nb_orbits, n;
	int orbit_idx;
	struct cubic_surface_data_set *Data;

	{
		long int *M;

		Fio.Csv_file_support->lint_matrix_read_csv(
				fname_csv, M, nb_orbits, n,
				verbose_level);

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


	other::data_structures::tally T_S;

	T_S.init_lint(Nb_singular_pts, nb_orbits, false, 0);

	cout << "Classification by the number of singular points:" << endl;
	T_S.print(true /* f_backwards */);

	{
		string fname_report;
		other::data_structures::string_tools ST;

		fname_report = fname_csv;
		ST.chop_off_extension(fname_report);
		fname_report += "_report.tex";

		other::l1_interfaces::latex_interface L;
		other::orbiter_kernel_system::file_io Fio;

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
					false /* f_sims */,
					NULL, //A1/*LG->A_linear*/->Sims,
					false /* f_strong_gens */,
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
			T_S.print_file_tex_we_are_in_math_mode(ost, true /* f_backwards */);
			ost << "$$" << endl;


			ost << "\\section{Singular Surfaces}" << endl;

			report_singular_surfaces(
					ost, Data, nb_orbits,
					verbose_level);

			ost << "\\section{Nonsingular Surfaces}" << endl;

			report_non_singular_surfaces(
					ost, Data, nb_orbits,
					verbose_level);



			if (f_v) {
				cout << "surface_domain_high_level::do_cubic_surface_properties_analyze "
						"after report" << endl;
			}

			L.foot(ost);
		}
		cout << "Written file " << fname_report << " of size "
				<< Fio.file_size(fname_report) << endl;
	}





	FREE_OBJECT(F0);

	if (f_v) {
		cout << "surface_domain_high_level::do_cubic_surface_properties_analyze done" << endl;
	}
}

void surface_domain_high_level::report_singular_surfaces(
		std::ostream &ost,
		struct cubic_surface_data_set *Data,
		int nb_orbits, int verbose_level)
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

	other::data_structures::tally T_L;

	T_L.init_lint(Selected_Nb_lines, nb_S, false, 0);

	ost << "Number of surfaces: " << nb_S << "\\\\" << endl;
	ost << "Classification by the number of lines:" << endl;
	ost << "$$" << endl;
	T_L.print_file_tex_we_are_in_math_mode(ost, true /* f_backwards */);
	ost << "$$" << endl;

	report_surfaces_by_lines(ost, Data_S, T_L, verbose_level);



	FREE_lint(Selected_Nb_lines);
	delete [] Data_S;

	if (f_v) {
		cout << "surface_domain_high_level::report_singular_surfaces done" << endl;
	}
}


void surface_domain_high_level::report_non_singular_surfaces(
		std::ostream &ost,
		struct cubic_surface_data_set *Data,
		int nb_orbits, int verbose_level)
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

	other::data_structures::tally T_L;

	T_L.init_lint(Selected_Nb_lines, nb_NS, false, 0);

	ost << "Number of surfaces: " << nb_NS << "\\\\" << endl;
	ost << "Classification by the number of lines:" << endl;
	ost << "$$" << endl;
	T_L.print_file_tex_we_are_in_math_mode(ost, true /* f_backwards */);
	ost << "$$" << endl;


	report_surfaces_by_lines(ost, Data_NS, T_L, verbose_level);


	FREE_lint(Selected_Nb_lines);
	delete [] Data_NS;

	if (f_v) {
		cout << "surface_domain_high_level::report_non_singular_surfaces done" << endl;
	}
}

void surface_domain_high_level::report_surfaces_by_lines(
		std::ostream &ost,
		struct cubic_surface_data_set *Data,
		other::data_structures::tally &T,
		int verbose_level)
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

		ost << "The number of surfaces with exactly " << a
				<< " lines is " << nb_L << ": \\\\" << endl;

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

	algebra::field_theory::finite_field *F;

	projective_geometry::projective_space_with_action *PA;

	geometry::algebraic_geometry::surface_domain *Surf;
	surface_with_action *Surf_A;

	int nb_total;
	int *nb_E;

	other::data_structures::tally *T_nb_E;



};

void surface_domain_high_level::do_create_surface_reports(
		std::string &field_orders_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_create_surface_reports" << endl;
		cout << "surface_domain_high_level::do_create_surface_reports "
				"verbose_level=" << verbose_level << endl;
	}

	combinatorics::knowledge_base::knowledge_base K;
	algebra::number_theory::number_theory_domain NT;
	other::orbiter_kernel_system::file_io Fio;


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
			cout << "surface_domain_high_level::do_create_surface_reports "
					"considering q=" << q << " with " << nb_total << " surfaces" << endl;
		}


		for (ocn = 0; ocn < nb_total; ocn++) {

			string cmd;
			string fname;

			if (f_v) {
				cout << "surface_domain_high_level::do_create_surface_reports "
						"considering q=" << q
						<< " ocn=" << ocn << " / " << nb_total << endl;
			}

			make_fname_surface_report_tex(fname, q, ocn);



			cmd = other::orbiter_kernel_system::Orbiter->orbiter_path
					+ "/orbiter.out -v 3 "
					+ "-define F -finite_field -q " + std::to_string(q) + " "
					+ "-end "
					+ "-define P -projective_space -n 3 -field F -end "
					+ "-define S -cubic_surface -space P "
					+ "-catalogue " + std::to_string(ocn) + " "
					+ "-end "
					+ "-with S -do "
					+ "-cubic_surface_activity "
					+ "-report "
					+ "-end >log_surface";

			if (f_v) {
				cout << "executing command: " << cmd << endl;
			}
			system(cmd.c_str());

			std::string fname_report_tex;

			make_fname_surface_report_tex(fname_report_tex, q, ocn);

			cmd = "pdflatex " + fname_report_tex + " >log_pdflatex";

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

void surface_domain_high_level::do_create_surface_atlas(
		int q_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_create_surface_atlas" << endl;
		cout << "surface_domain_high_level::do_create_surface_atlas "
				"verbose_level=" << verbose_level << endl;
	}

	combinatorics::knowledge_base::knowledge_base K;


	algebra::number_theory::number_theory_domain NT;
	other::data_structures::sorting Sorting;
	other::orbiter_kernel_system::file_io Fio;


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
			f_semilinear = true;
		}
		else {
			f_semilinear = false;
		}

#if 0
		T[cur].Descr = NEW_OBJECT(linear_group_description);

		T[cur].Descr->n = 4;
		T[cur].Descr->input_q = q;
		T[cur].Descr->f_projective = true;
		T[cur].Descr->f_general = false;
		T[cur].Descr->f_affine = false;
		T[cur].Descr->f_semilinear = false;

		if (h > 1) {
			T[cur].Descr->f_semilinear = true;
		}
		T[cur].Descr->f_special = false;
#endif

		T[cur].F = NEW_OBJECT(algebra::field_theory::finite_field);
		T[cur].F->finite_field_init_small_order(q,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);

		//T[cur].Descr->F = T[cur].F;


		T[cur].PA = NEW_OBJECT(projective_geometry::projective_space_with_action);

		T[cur].PA->init(
				T[cur].F, 3, f_semilinear,
				true /* f_init_incidence_structure */,
				verbose_level);

		//T[cur].LG = NEW_OBJECT(linear_group);

		//cout << "before LG->linear_group_init" << endl;
		//T[cur].LG->linear_group_init(T[cur].Descr, verbose_level);








		if (f_v) {
			cout << "surface_domain_high_level::do_create_surface_atlas "
					"before Surf->init" << endl;
		}

		T[cur].Surf = NEW_OBJECT(geometry::algebraic_geometry::surface_domain);
		T[cur].Surf->init_surface_domain(T[cur].F, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "do_create_surface_atlas after Surf->init_surface_domain" << endl;
		}

		T[cur].Surf_A = NEW_OBJECT(surface_with_action);

		if (f_v) {
			cout << "surface_domain_high_level::do_create_surface_atlas "
					"before Surf_A->init_with_linear_group" << endl;
		}
		T[cur].Surf_A->init(
				T[cur].Surf, T[cur].PA,
				true /* f_recoordinatize */,
				0 /*verbose_level*/);
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

		T[cur].T_nb_E = NEW_OBJECT(other::data_structures::tally);

		T[cur].T_nb_E->init(T[cur].nb_E, T[cur].nb_total, false, 0);


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

		fname_report = "surface_atlas.tex";

		{
			ofstream ost(fname_report);

			string title, author, extra_praeamble;

			title = "ATLAS of Cubic Surfaces";
			author = "Anton Betten and Fatma Karaoglu";

			other::l1_interfaces::latex_interface L;

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

						other::data_structures::string_tools ST;

						fname_report_html = fname_report_tex;
						ST.chop_off_extension(fname_report_html);
						fname_report_html += ".html";


						ost << " & ";
						ost << "%%tth: \\begin{html} <a href=\"" << fname_report_html << "\"> " << nb << " </a> \\end{html}" << endl;


						string cmd;

						cmd = "~/bin/tth " + fname_report_tex;
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
		other::orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	}





	if (f_v) {
		cout << "surface_domain_high_level::do_create_surface_atlas done" << endl;
	}
}



void surface_domain_high_level::do_create_surface_atlas_q_e(
		int q_max,
		struct table_surfaces_field_order *T,
		int nb_e, int *Idx, int nb,
		std::string &fname_report_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_create_surface_atlas_q_e" << endl;
		cout << "surface_domain_high_level::do_create_surface_atlas_q_e "
				"q=" << T->q << " " << nb_e << endl;
	}

	combinatorics::knowledge_base::knowledge_base K;


	algebra::number_theory::number_theory_domain NT;
	other::data_structures::sorting Sorting;
	other::orbiter_kernel_system::file_io Fio;




	{
		fname_report_tex = "surface_atlas_q" + std::to_string(T->q)
				+ "_e" + std::to_string(nb_e) + ".tex";

		{
			ofstream ost(fname_report_tex);


			string title, author, extra_praeamble;

			title = "ATLAS of Cubic Surfaces, q=" + std::to_string(T->q)
					+ ", \\#E=" + std::to_string(nb_e);

			author.assign("Anton Betten and Fatma Karaoglu");

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
		other::orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_report_tex << " of size "
			<< Fio.file_size(fname_report_tex) << endl;


	}

}

void surface_domain_high_level::do_create_dickson_atlas(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::do_create_dickson_atlas" << endl;
		cout << "surface_domain_high_level::do_create_dickson_atlas "
				"verbose_level=" << verbose_level << endl;
	}

	other::orbiter_kernel_system::file_io Fio;





	{
		string fname_report;

		fname_report = "dickson_surfaces.tex";

		{
			ofstream ost(fname_report);

			string title, author, extra_praeamble;

			title = "ATLAS of Dickson Surfaces";
			author = "Fatma Karaoglu";

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

						string fname_tex;
						string fname_pdf;
						string fname_surface_report;


						fname_tex = "Orb" + std::to_string(c)
								+ "_q" + std::to_string(field_orders[j]) + ".tex";
						fname_pdf = "Orb" + std::to_string(c)
								+ "_q" + std::to_string(field_orders[j]) + ".pdf";
						fname_surface_report = "Orb" + std::to_string(c)
								+ "_q" + std::to_string(field_orders[j]) + ".pdf";


						ost << " & " << endl;
						ost << "%%tth: \\begin{html} <a href=\"" << fname_surface_report << "\"> "
								<< fname_surface_report << " </a> \\end{html}" << endl;


						if (Fio.file_size(fname_tex) > 0) {

							if (Fio.file_size(fname_pdf) <= 0) {
								string cmd;

								cmd = "pdflatex " + fname_tex + " ";
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
		other::orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	}


	if (f_v) {
		cout << "surface_domain_high_level::do_create_dickson_atlas done" << endl;
	}
}




void surface_domain_high_level::make_fname_surface_report_tex(
		std::string &fname, int q, int ocn)
{
	fname = "surface_catalogue_q" + std::to_string(q)
			+ "_iso" + std::to_string(ocn) + "_with_group.tex";
}

void surface_domain_high_level::make_fname_surface_report_pdf(
		std::string &fname, int q, int ocn)
{
	fname = "surface_catalogue_q" + std::to_string(q)
			+ "_iso" + std::to_string(ocn) + "_with_group.pdf";
}


void surface_domain_high_level::table_of_cubic_surfaces(
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::table_of_cubic_surfaces" << endl;
	}

	if (PA->n != 3) {
		cout << "surface_domain_high_level::table_of_cubic_surfaces "
				"we need a three-dimensional projective space" << endl;
		exit(1);
	}

	applications_in_algebraic_geometry::cubic_surfaces_in_general::table_of_surfaces *T;

	T = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_in_general::table_of_surfaces);

	if (f_v) {
		cout << "surface_domain_high_level::table_of_cubic_surfaces "
				"before T->init" << endl;
	}
	T->init(PA, verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::table_of_cubic_surfaces "
				"after T->init" << endl;
	}

	if (f_v) {
		cout << "surface_domain_high_level::table_of_cubic_surfaces "
				"before T->do_export" << endl;
	}
	T->do_export(verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::table_of_cubic_surfaces "
				"after T->do_export" << endl;
	}

	FREE_OBJECT(T);


	if (f_v) {
		cout << "surface_domain_high_level::table_of_cubic_surfaces done" << endl;
	}

}


void surface_domain_high_level::make_table_of_surfaces(
		int verbose_level)
{


	//int f_v = (verbose_level >= 1);


	string fname;
	string author;
	string title;
	string extras_for_preamble;

	fname.assign("surfaces_report.tex");

	author.assign("Orbiter");

	title.assign("Cubic Surfaces with 27 Lines over Finite Fields");

	int f_quartic_curves = false;

	{
		ofstream fp(fname);
		other::l1_interfaces::latex_interface L;
		//latex_head_easy(fp);
		L.head(fp,
			false /* f_book */, true /* f_title */,
			title, author,
			false /*f_toc*/, false /* f_landscape*/, false /* f_12pt*/,
			true /*f_enlarged_page*/, true /* f_pagenumbers*/,
			extras_for_preamble);


		{
			int Q[] = {
					4,7,8,9,11,13,16,17,19,23,25,27,29,31,32,37,
					41,43,47,49,53,59,61,64,67,71,73,79,81,83, 89,
					97, 101, 103, 107, 109, 113, 121, 128
				};
			int nb_Q = sizeof(Q) / sizeof(int);

			fp << "\\subsection*{Cubic Surfaces by Field Order and Number of Eckardt Points}" << endl;

			string prefix;

			prefix.assign("even_and_odd");

			make_table_of_objects(fp, prefix, Q, nb_Q, f_quartic_curves, verbose_level);
		}

		fp << endl;

#if 1
		fp << "\\clearpage" << endl;

		fp << endl;


		{
			int Q_even[] = {
				4,8,16,32,64,128
				};
			int nb_Q_even = sizeof(Q_even) / sizeof(int);

			fp << "\\subsection*{Even Characteristic}" << endl;

			string prefix;

			prefix.assign("even");

			make_table_of_objects(fp, prefix, Q_even, nb_Q_even, f_quartic_curves, verbose_level);
		}

		fp << endl;

		fp << "\\clearpage" << endl;

		fp << endl;


		{
			int Q_odd[] = {
					7,9,11,13,17,19,23,25,27,29,31,37,
					41,43,47,49,53,59,61,67,71,73,79,81,83,
					89, 97, 101, 103, 107, 109, 113, 121
				};
			int nb_Q_odd = sizeof(Q_odd) / sizeof(int);


			fp << "\\subsection*{Odd Characteristic}" << endl;

			string prefix;

			prefix.assign("odd");

			make_table_of_objects(fp, prefix, Q_odd, nb_Q_odd, f_quartic_curves, verbose_level);
		}
#endif

		L.foot(fp);
	}

}


void surface_domain_high_level::make_table_of_quartic_curves(
		int verbose_level)
{


	//int f_v = (verbose_level >= 1);


	string fname;
	string author;
	string title;
	string extras_for_preamble;

	fname.assign("quartic_curves_report.tex");

	author.assign("Orbiter");

	title.assign("Quartic Curves with 28 Bitangents over Finite Fields");

	int f_quartic_curves = true;

	{
		ofstream fp(fname);
		other::l1_interfaces::latex_interface L;
		//latex_head_easy(fp);
		L.head(fp,
			false /* f_book */, true /* f_title */,
			title, author,
			false /*f_toc*/, false /* f_landscape*/, false /* f_12pt*/,
			true /*f_enlarged_page*/, true /* f_pagenumbers*/,
			extras_for_preamble);


		{
			int Q[] = {
					9,13,17,19,23,25,27,29,31
				};
			int nb_Q = sizeof(Q) / sizeof(int);

			fp << "\\subsection*{Quartic Curves by Field Order and Number of Kovalevski Points}" << endl;

			string prefix;

			prefix.assign("quartic_curves");

			make_table_of_objects(fp, prefix, Q, nb_Q, f_quartic_curves, verbose_level);
		}

		fp << endl;


		L.foot(fp);
	}

}





void surface_domain_high_level::make_table_of_objects(
		std::ostream &ost,
		std::string &prefix,
		int *Q_table, int Q_table_len,
		int f_quartic_curves,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain_high_level::make_table_of_objects" << endl;
	}
	//int q, nb_reps;
	//int i, j;
	//int *data;
	//int nb_gens;
	//int data_size;
	//knowledge_base::knowledge_base K;



#if 0
	const char *fname_ago = "ago.csv";
	{
	ofstream f(fname_ago);

	f << "q,j,nb_E,stab_order" << endl;
	for (i = 0; i < Q_table_len; i++) {
		q = Q_table[i];
		nb_reps = K.cubic_surface_nb_reps(q);
		for (j = 0; j < nb_reps; j++) {
			nb_E = K.cubic_surface_nb_Eckardt_points(q, j);
			K.cubic_surface_stab_gens(q, j,
					data, nb_gens, data_size, stab_order);
			f << q << "," << j << ", " << nb_E << ", "
					<< stab_order << endl;
			}
		}
	f << "END" << endl;
	}
	cout << "Written file " << fname_ago << " of size "
			<< Fio.file_size(fname_ago) << endl;

	const char *fname_dist = "ago_dist.csv";
	{
	ofstream f(fname_dist);
	int *Ago;

	f << "q,ago" << endl;
	for (i = 0; i < Q_table_len; i++) {
		q = Q_table[i];
		nb_reps = K.cubic_surface_nb_reps(q);
		Ago = NEW_int(nb_reps);
		for (j = 0; j < nb_reps; j++) {
			nb_E = K.cubic_surface_nb_Eckardt_points(q, j);
			K.cubic_surface_stab_gens(q, j, data,
					nb_gens, data_size, stab_order);
			sscanf(stab_order, "%d", &Ago[j]);
			//f << q << "," << j << ", " << nb_E << ", " << stab_order << endl;
		}
		tally C;

		C.init(Ago, nb_reps, false, 0);
		f << q << ", ";
		C.print_bare_tex(f, true /* f_backwards*/);
		f << endl;

		FREE_int(Ago);
	}
	f << "END" << endl;
	}
	cout << "Written file " << fname_dist << " of size "
			<< Fio.file_size(fname_dist) << endl;
#endif


	projective_geometry::summary_of_properties_of_objects *Summary;

	Summary = NEW_OBJECT(projective_geometry::summary_of_properties_of_objects);


	if (f_quartic_curves) {
		if (f_v) {
			cout << "surface_domain_high_level::make_table_of_objects "
					"before Summary->init_quartic_curves" << endl;
		}
		Summary->init_quartic_curves(
				Q_table, Q_table_len,
				verbose_level);
		if (f_v) {
			cout << "surface_domain_high_level::make_table_of_objects "
					"after Summary->init_quartic_curves" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "surface_domain_high_level::make_table_of_objects "
					"before Summary->init_surfaces" << endl;
		}
		Summary->init_surfaces(
				Q_table, Q_table_len,
				verbose_level);
		if (f_v) {
			cout << "surface_domain_high_level::make_table_of_objects "
					"after Summary->init_surfaces" << endl;
		}
	}


	{
		cout << "q : number of objects\\\\" << endl;
		int i, q, nb_reps;
		for (i = 0; i < Q_table_len; i++) {
			q = Q_table[i];
			nb_reps = Summary->Nb_objects[i];
			cout << q << " : " << nb_reps << "\\\\" << endl;
		}
	}


	if (f_v) {
		cout << "surface_domain_high_level::make_table_of_objects "
				"before Summary->export_table_csv" << endl;
	}
	Summary->export_table_csv(
			prefix,
			verbose_level);
	if (f_v) {
		cout << "surface_domain_high_level::make_table_of_objects "
				"after Summary->export_table_csv" << endl;
	}

	if (f_v) {
		cout << "surface_domain_high_level::make_table_of_objects "
				"before Summary->table_latex" << endl;
	}


	Summary->table_latex(ost, verbose_level);


	if (f_v) {
		cout << "surface_domain_high_level::make_table_of_objects "
				"after Summary->table_latex" << endl;
	}


	if (f_v) {
		cout << "surface_domain_high_level::make_table_of_objects "
				"before Summary->table_ago" << endl;
	}

	Summary->table_ago(ost, verbose_level);

	if (f_v) {
		cout << "surface_domain_high_level::make_table_of_objects "
				"after Summary->table_ago" << endl;
	}


	if (f_v) {
		cout << "surface_domain_high_level::make_table_of_objects "
				"before Summary->make_detailed_table_of_objects" << endl;
	}

	Summary->make_detailed_table_of_objects(verbose_level);

	if (f_v) {
		cout << "surface_domain_high_level::make_table_of_objects "
				"after Summary->make_detailed_table_of_objects" << endl;
	}

	FREE_OBJECT(Summary);


	if (f_v) {
		cout << "surface_domain_high_level::make_table_of_objects" << endl;
	}


}

#if 0
void surface_domain_high_level::table_top(
		std::ostream &ost)
{
	ost << "$" << endl;
	ost << "\\begin{array}{|c|c||c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "q & \\mbox{Iso} & \\mbox{Ago} & \\# E & "
			"\\mbox{Comment}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
}

void surface_domain_high_level::table_bottom(
		std::ostream &ost)
{
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	//ost << "\\quad" << endl;
	ost << "$" << endl;
}
#endif






}}}}


