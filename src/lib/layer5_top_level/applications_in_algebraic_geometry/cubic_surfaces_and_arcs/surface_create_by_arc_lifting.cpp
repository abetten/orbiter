/*
 * surface_create_by_arc_lifting.cpp
 *
 *  Created on: Jul 17, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_arcs {


surface_create_by_arc_lifting::surface_create_by_arc_lifting()
{
	Record_birth();
	arc_idx = 0;
	surface_idx = 0;
	SCA = NULL;
	arc_idx = 0;
	Arc6 = NULL;
	AL = NULL;
	SOA = NULL;
	Clebsch = NULL;
	Other_arc_idx = NULL;
}


surface_create_by_arc_lifting::~surface_create_by_arc_lifting()
{
	Record_death();
	if (Arc6) {
		FREE_lint(Arc6);
	}
	if (AL) {
		FREE_OBJECT(AL);
	}
	if (SOA) {
		FREE_OBJECT(SOA);
	}
	if (Clebsch) {
		FREE_OBJECTS(Clebsch);
	}
	if (Other_arc_idx) {
		FREE_int(Other_arc_idx);
	}
}


void surface_create_by_arc_lifting::init(
		int arc_idx,
		surface_classify_using_arc *SCA, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create_by_arc_lifting::init" << endl;
	}

	surface_create_by_arc_lifting::SCA = SCA;
	surface_create_by_arc_lifting::arc_idx = arc_idx;


	surface_idx = SCA->nb_surfaces;


	arc_label = std::to_string(arc_idx) + " / " + std::to_string(SCA->Six_arcs->nb_arcs_not_on_conic);


	arc_label_short = "Arc" + std::to_string(arc_idx);


	if (f_v) {
		cout << "surface_create_by_arc_lifting::init extending arc "
				<< arc_idx << " / "
				<< SCA->Six_arcs->nb_arcs_not_on_conic << ":" << endl;
	}


	Arc6 = NEW_lint(6);

	SCA->Six_arcs->Gen->gen->get_set_by_level(
			6 /* level */,
			SCA->Six_arcs->Not_on_conic_idx[arc_idx],
			Arc6);


	if (f_v) {
		cout << "surface_create_by_arc_lifting::init extending arc "
				<< arc_idx << " / "
				<< SCA->Six_arcs->nb_arcs_not_on_conic << " : Arc6 = ";
		Lint_vec_print(cout, Arc6, 6);
		cout << endl;
	}

	AL = NEW_OBJECT(arc_lifting);


	if (f_v) {
		cout << "surface_create_by_arc_lifting::init "
				"before AL->create_surface_and_group" << endl;
	}
	AL->create_surface_and_group(SCA->Surf_A, Arc6, verbose_level);
	if (f_v) {
		cout << "surface_create_by_arc_lifting::init "
				"after AL->create_surface_and_group" << endl;
	}

	string magma_fname;
	other::orbiter_kernel_system::file_io Fio;

	magma_fname = "surface_q" + std::to_string(SCA->Surf_A->PA->F->q)
			+ "_iso" + std::to_string(SCA->nb_surfaces)
			+ "_group.magma";


	interfaces::magma_interface M;

	if (f_v) {
		cout << "surface_create_by_arc_lifting::init "
				"before M.export_permutation_group_to_magma" << endl;
	}
	M.export_permutation_group_to_magma(
			magma_fname,
			AL->Trihedral_pair->Aut_gens->A,
			AL->Trihedral_pair->Aut_gens,
			verbose_level);
	if (f_v) {
		cout << "surface_create_by_arc_lifting::init "
				"after M.export_permutation_group_to_magma" << endl;
	}


#if 0
	AL->Trihedral_pair->Aut_gens->export_permutation_group_to_magma(
			magma_fname, AL->Trihedral_pair->Aut_gens->A, verbose_level - 2);
#endif

	if (f_v) {
		cout << "written file " << magma_fname << " of size "
				<< Fio.file_size(magma_fname) << endl;
	}

	algebra::ring_theory::longinteger_object go;

	AL->Trihedral_pair->Aut_gens->group_order(go);



	string label_txt;
	string label_tex;

	label_txt = "arc_lifting";
	label_tex = "arc\\_lifting";

	SOA = NEW_OBJECT(cubic_surfaces_in_general::surface_object_with_group);

	if (f_v) {
		cout << "surface_create_by_arc_lifting::init "
				"before SOA->init_with_27_lines" << endl;
	}

	SOA->init_with_group(SCA->Surf_A,
		AL->Web->Lines27, 27, AL->the_equation,
		AL->Trihedral_pair->Aut_gens,
		label_txt, label_tex,
		false /* f_find_double_six_and_rearrange_lines */,
		false, NULL,
		verbose_level);
	if (f_v) {
		cout << "surface_create_by_arc_lifting::init "
				"after SOA->init_with_27_lines" << endl;
	}




	algebra::ring_theory::longinteger_object ago;
	AL->Trihedral_pair->Aut_gens->group_order(ago);
	cout << "The automorphism group of the surface has order "
			<< ago << "\\\\" << endl;





	SOA->SO->identify_lines(
			AL->Trihedral_pair->nine_lines, 9,
			nine_lines_idx,
			false /* verbose_level */);





	if (f_v) {
		cout << "surface_create_by_arc_lifting::init "
				"arc " << arc_label << " yields a surface with "
			<< AL->Web->E->nb_E << " Eckardt points and a stabilizer "
				"of order " << go << " with "
			<< SOA->Orbits_on_single_sixes->nb_orbits
			<< " orbits on single sixes" << endl;
	}


	SCA->Arc_identify_nb[SCA->nb_surfaces] = SOA->Orbits_on_single_sixes->nb_orbits;


	int orbit_idx;

	if (f_v) {
		cout << "surface_create_by_arc_lifting::init "
				"performing isomorph rejection" << endl;
	}

	Clebsch = NEW_OBJECTS(cubic_surfaces_in_general::surface_clebsch_map, SOA->Orbits_on_single_sixes->nb_orbits);
	Other_arc_idx = NEW_int(SOA->Orbits_on_single_sixes->nb_orbits);

	for (orbit_idx = 0; orbit_idx < SOA->Orbits_on_single_sixes->nb_orbits; orbit_idx++) {

		if (f_v) {
			cout << "surface_create_by_arc_lifting::init "
					"orbit " << orbit_idx << " / " << SOA->Orbits_on_single_sixes->nb_orbits << endl;
		}
		Clebsch[orbit_idx].init(SOA, orbit_idx, verbose_level);


#if 0
		Six_arcs->Gen->gen->identify(Arc, 6, transporter,
				orbit_at_level, 0 /*verbose_level */);





		if (!Sorting.int_vec_search(Six_arcs->Not_on_conic_idx,
			Six_arcs->nb_arcs_not_on_conic, orbit_at_level, idx)) {
			cout << "could not find orbit" << endl;
			exit(1);
		}
#else
		SCA->Six_arcs->recognize(Clebsch[orbit_idx].Clebsch_map->Arc, SCA->transporter,
				Other_arc_idx[orbit_idx], verbose_level - 2);

#endif
		SCA->f_deleted[Other_arc_idx[orbit_idx]] = true;

		SCA->Arc_identify[SCA->nb_surfaces * SCA->Six_arcs->nb_arcs_not_on_conic + orbit_idx] = Other_arc_idx[orbit_idx];


		if (f_v) {
			cout << "arc " << arc_label << " yields a surface with "
				<< AL->Web->E->nb_E
				<< " Eckardt points and a stabilizer of order "
				<< go << " with "
				<< SOA->Orbits_on_single_sixes->nb_orbits
				<< " orbits on single sixes";
			cout << " orbit " << orbit_idx << " yields an arc which "
					"is isomorphic to arc " << Other_arc_idx[orbit_idx] << endl;
		}





	}

	if (f_v) {
		cout << "surface_create_by_arc_lifting::init done" << endl;
	}

}

void surface_create_by_arc_lifting::report_summary(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create_by_arc_lifting::report_summary" << endl;
	}



	algebra::ring_theory::longinteger_object ago;
	AL->Trihedral_pair->Aut_gens->group_order(ago);

	ost << "The equation of the surface is" << endl;

	if (f_v) {
		cout << "surface_create_by_arc_lifting::report_summary "
				"before AL->report_equation" << endl;
	}
	AL->report_equation(ost);
	if (f_v) {
		cout << "surface_create_by_arc_lifting::report_summary "
				"after AL->report_equation" << endl;
	}


	ost << "Extension of arc " << arc_idx
			<< " / " << SCA->Six_arcs->nb_arcs_not_on_conic << ":" << endl;


	SCA->Six_arcs->report_specific_arc_basic(ost, arc_idx);

	ost << "arc " << arc_idx << " yields a surface with "
		<< AL->Web->E->nb_E << " Eckardt points and a "
				"stabilizer of order " << ago << " with "
		<< SOA->Orbits_on_single_sixes->nb_orbits
		<< " orbits on single sixes\\\\" << endl;


	if (f_v) {
		cout << "surface_create_by_arc_lifting::report_summary "
				"before SOA->SO->print_Eckardt_points" << endl;
	}
	SOA->SO->SOP->print_Eckardt_points(ost);
	if (f_v) {
		cout << "surface_create_by_arc_lifting::report_summary "
				"after SOA->SO->print_Eckardt_points" << endl;
	}


}

void surface_create_by_arc_lifting::report(
		std::ostream &ost,
		other::graphics::layered_graph_draw_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create_by_arc_lifting::report" << endl;
	}


	algebra::ring_theory::longinteger_object ago;
	AL->Trihedral_pair->Aut_gens->group_order(ago);

	ost << "The equation of the surface is" << endl;

	if (f_v) {
		cout << "surface_create_by_arc_lifting::report "
				"before AL->report_equation" << endl;
	}
	AL->report_equation(ost);
	if (f_v) {
		cout << "surface_create_by_arc_lifting::report "
				"after AL->report_equation" << endl;
	}


	ost << "Extension of arc " << arc_idx
			<< " / " << SCA->Six_arcs->nb_arcs_not_on_conic << ":" << endl;


	SCA->Six_arcs->report_specific_arc(ost, arc_idx);

	ost << "arc " << arc_idx << " yields a surface with "
		<< AL->Web->E->nb_E << " Eckardt points and a "
				"stabilizer of order " << ago << " with "
		<< SOA->Orbits_on_single_sixes->nb_orbits
		<< " orbits on single sixes\\\\" << endl;

#if 0
	{
		set_and_stabilizer *The_arc;

		The_arc = SCA->Six_arcs->Gen->gen->get_set_and_stabilizer(
				6 /* level */,
				SCA->Six_arcs->Not_on_conic_idx[arc_idx],
				0 /* verbose_level */);


		ost << "Arc " << arc_idx << " / "
				<< SCA->Six_arcs->nb_arcs_not_on_conic << " is: ";
		ost << "$$" << endl;
		//int_vec_print(fp, Arc6, 6);
		The_arc->print_set_tex(ost);
		ost << "$$" << endl;

		SCA->Surf_A->F->display_table_of_projective_points(ost,
			The_arc->data, 6, 3);


		ost << "The arc-stabilizer is the following group:\\\\" << endl;
		The_arc->Strong_gens->print_generators_tex(ost);

		FREE_OBJECT(The_arc);
	}
#endif


	AL->report(ost, verbose_level);



	if (f_v) {
		cout << "surface_create_by_arc_lifting::report "
				"before report_properties_simple" << endl;
	}
	SOA->SO->SOP->report_properties_simple(ost, verbose_level);
	if (f_v) {
		cout << "surface_create_by_arc_lifting::report "
				"after report_properties_simple" << endl;
	}


	ost << "The nine lines in the selected trihedral pair are:" << endl;

	SOA->SO->print_nine_lines_latex(ost,
			AL->Trihedral_pair->nine_lines,
			nine_lines_idx);

	//SOA->SO->latex_table_of_trihedral_pairs_and_clebsch_system(
	//fp, AL->T_idx, AL->nb_T);

	if (f_v) {
		cout << "surface_create_by_arc_lifting::report "
			"before SOA->print_automorphism_group" << endl;
	}

	string fname_mask;

	fname_mask = "orbit_half_double_sixes_q" + std::to_string(SCA->Surf_A->PA->F->q)
			+ "_iso_" + std::to_string(SCA->nb_surfaces) + "_%d";

	SOA->print_automorphism_group(
			ost,
		true /* f_print_orbits */,
		fname_mask, Opt,
		verbose_level - 1);

	ost << "arc " << arc_label << " yields a surface with "
		<< AL->Web->E->nb_E << " Eckardt points and a "
				"stabilizer of order " << ago << " with "
		<< SOA->Orbits_on_single_sixes->nb_orbits
		<< " orbits on single sixes\\\\" << endl;


	int orbit_idx;

	for (orbit_idx = 0; orbit_idx < SOA->Orbits_on_single_sixes->nb_orbits; orbit_idx++) {

		cout << "arc " << arc_label << " yields a surface with "
			<< AL->Web->E->nb_E
			<< " Eckardt points and a stabilizer of order "
			<< ago << " with "
			<< SOA->Orbits_on_single_sixes->nb_orbits
			<< " orbits on single sixes \\\\" << endl;
		cout << " orbit " << orbit_idx << " yields an arc which is "
				"isomorphic to arc " << Other_arc_idx[orbit_idx] << "\\\\" << endl;

	}


	for (orbit_idx = 0; orbit_idx < SOA->Orbits_on_single_sixes->nb_orbits; orbit_idx++) {
		Clebsch[orbit_idx].report(ost, verbose_level);
	}

	if (f_v) {
		cout << "surface_create_by_arc_lifting::report done" << endl;
	}
}


}}}}

