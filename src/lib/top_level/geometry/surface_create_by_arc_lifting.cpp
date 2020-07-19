/*
 * surface_create_by_arc_lifting.cpp
 *
 *  Created on: Jul 17, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


surface_create_by_arc_lifting::surface_create_by_arc_lifting()
{
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


void surface_create_by_arc_lifting::init(int arc_idx,
		surface_classify_using_arc *SCA, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create_by_arc_lifting::init" << endl;
	}

	surface_create_by_arc_lifting::SCA = SCA;
	surface_create_by_arc_lifting::arc_idx = arc_idx;


	surface_idx = SCA->nb_surfaces;

	char str[1000];

	snprintf(str, 1000, "%d / %d", arc_idx, SCA->Six_arcs->nb_arcs_not_on_conic);

	arc_label.assign(str);

	snprintf(str, 1000, "Arc%d", arc_idx);

	arc_label_short.assign(str);


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
		lint_vec_print(cout, Arc6, 6);
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

	char magma_fname[1000];
	file_io Fio;

	sprintf(magma_fname, "surface_q%d_iso%d_group.magma", SCA->Surf_A->F->q, SCA->nb_surfaces);
	AL->Aut_gens->export_permutation_group_to_magma(
			magma_fname, verbose_level - 2);

	if (f_v) {
		cout << "written file " << magma_fname << " of size "
				<< Fio.file_size(magma_fname) << endl;
	}

	longinteger_object go;

	AL->Aut_gens->group_order(go);





	SOA = NEW_OBJECT(surface_object_with_action);

	if (f_v) {
		cout << "surface_create_by_arc_lifting::init "
				"before SOA->init" << endl;
	}

	SOA->init(SCA->Surf_A,
		AL->Web->Lines27, AL->the_equation,
		AL->Aut_gens,
		FALSE /* f_find_double_six_and_rearrange_lines */,
		FALSE, NULL,
		verbose_level);
	if (f_v) {
		cout << "surface_create_by_arc_lifting::init "
				"after SOA->init" << endl;
	}




	longinteger_object ago;
	AL->Aut_gens->group_order(ago);
	cout << "The automorphism group of the surface has order "
			<< ago << "\\\\" << endl;





	SOA->SO->identify_lines(
			AL->nine_lines, 9,
			nine_lines_idx,
			FALSE /* verbose_level */);





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

	Clebsch = NEW_OBJECTS(surface_clebsch_map, SOA->Orbits_on_single_sixes->nb_orbits);
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
		SCA->Six_arcs->recognize(Clebsch[orbit_idx].Arc, SCA->transporter,
				Other_arc_idx[orbit_idx], verbose_level - 2);

#endif
		SCA->f_deleted[Other_arc_idx[orbit_idx]] = TRUE;

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

void surface_create_by_arc_lifting::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);




	ost << "The equation of the surface is" << endl;

	if (f_v) {
		cout << "surface_create_by_arc_lifting::report "
				"before Surf->print_equation_in_trihedral_form" << endl;
	}
	SCA->Surf_A->Surf->print_equation_in_trihedral_form(ost,
			AL->The_six_plane_equations,
			AL->lambda,
			AL->the_equation);
	//Surf->print_equation_in_trihedral_form(fp,
	//AL->the_equation, AL->t_idx0, lambda);
	if (f_v) {
		cout << "surface_create_by_arc_lifting::report "
				"after Surf->print_equation_in_trihedral_form" << endl;
	}


	ost << "Extension of arc " << arc_idx
			<< " / " << SCA->Six_arcs->nb_arcs_not_on_conic << ":" << endl;

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


		ost << "The stabilizer is the following group:\\\\" << endl;
		The_arc->Strong_gens->print_generators_tex(ost);

		FREE_OBJECT(The_arc);
	}


	AL->report(ost, verbose_level);



	longinteger_object ago;
	AL->Aut_gens->group_order(ago);
	ost << "The automorphism group of the surface has order "
			<< ago << "\\\\" << endl;
	ost << "The automorphism group is the following group\\\\" << endl;
	if (f_v) {
		cout << "surface_with_action::arc_lifting_and_classify "
				"before Aut_gens->print_generators_tex" << endl;
	}
	AL->Aut_gens->print_generators_tex(ost);

	if (f_v) {
		cout << "surface_create_by_arc_lifting::report "
				"before report_properties" << endl;
	}
	SOA->SO->report_properties(ost, verbose_level);
	if (f_v) {
		cout << "surface_create_by_arc_lifting::report "
				"after report_properties" << endl;
	}


	ost << "The nine lines in the selected trihedral pair are:" << endl;

	SOA->SO->print_nine_lines_latex(ost,
			AL->nine_lines,
			nine_lines_idx);

	//SOA->SO->latex_table_of_trihedral_pairs_and_clebsch_system(
	//fp, AL->T_idx, AL->nb_T);

	if (f_v) {
		cout << "surface_create_by_arc_lifting::report "
			"before SOA->print_automorphism_group" << endl;
	}

	char fname_mask[1000];

	sprintf(fname_mask, "orbit_half_double_sixes_q%d_iso%d_%%d", SCA->Surf_A->F->q, SCA->nb_surfaces);
	SOA->print_automorphism_group(ost,
		TRUE /* f_print_orbits */,
		fname_mask);

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
}


}}
