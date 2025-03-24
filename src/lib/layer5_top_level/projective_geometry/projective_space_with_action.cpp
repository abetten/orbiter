// projective_space_with_action.cpp
// 
// Anton Betten
//
// December 22, 2017
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace projective_geometry {



projective_space_with_action::projective_space_with_action()
{
	Record_birth();
	Descr = NULL;

	n = 0;
	d = 0;
	q = 0;
	F = NULL;
	f_semilinear = false;
	f_init_incidence_structure = false;
	P = NULL;
	PA2 = NULL;
	Surf_A = NULL;
	Dom = NULL;
	QCDA = NULL;
	A = NULL;
	A_on_lines = NULL;

	f_has_action_on_planes = false;
	A_on_planes = NULL;

	Elt1 = NULL;
}





projective_space_with_action::~projective_space_with_action()
{
	Record_death();
	if (P) {
		FREE_OBJECT(P);
	}
	if (PA2) {
		FREE_OBJECT(PA2);
	}
	if (Surf_A) {
		FREE_OBJECT(Surf_A);
	}
	if (Dom) {
		FREE_OBJECT(Dom);
	}
	if (QCDA) {
		FREE_OBJECT(QCDA);
	}
	if (A) {
		FREE_OBJECT(A);
	}
	if (A_on_lines) {
		FREE_OBJECT(A_on_lines);
	}
	if (f_has_action_on_planes) {
		FREE_OBJECT(A_on_planes);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
}

void projective_space_with_action::init_from_description(
		projective_space_with_action_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::init_from_description, "
				"verbose_level=" << verbose_level << endl;
	}

	projective_space_with_action::Descr = Descr;

	algebra::field_theory::finite_field *F;

	if (Descr->f_field_label) {

		F = Get_finite_field(
				Descr->field_label);
	}
	else if (Descr->f_field_pointer) {
		F = Descr->F;
	}
	else if (Descr->f_q) {

		if (f_v) {
			cout << "projective_space_with_action::init_from_description "
					"creating the finite field of order " << q << endl;
		}
		F = NEW_OBJECT(algebra::field_theory::finite_field);
		F->finite_field_init_small_order(Descr->q,
				false /* f_without_tables */,
				true /* f_compute_related_fields */,
				verbose_level - 1);
		if (f_v) {
			cout << "projective_space_with_action::init_from_description "
					"the finite field of order " << Descr->q
					<< " has been created" << endl;
		}

	}
	else {
		cout << "projective_space_with_action::init_from_description, "
				"I need a field by label or by pointer or by its order" << endl;
		exit(1);
	}


	Descr->F = F;
	Descr->q = F->q;

	if (!Descr->f_n) {
		cout << "projective_space_with_action::init_from_description, "
				"I need a dimension n" << endl;
		exit(1);
	}



	int f_semilinear = true;

	algebra::number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "projective_space_with_action::init_from_description, "
				"q=" << Descr->q << endl;
	}

	if (NT.is_prime(Descr->q)) {
		f_semilinear = false;
	}
	else {
		f_semilinear = true;
	}

	if (Descr->f_use_projectivity_subgroup) {
		f_semilinear = false;
	}

	if (f_v) {
		cout << "projective_space_with_action::init_from_description, "
				"f_semilinear=" << f_semilinear << endl;
	}


	if (f_v) {
		cout << "projective_space_with_action::init_from_description, "
				"before init" << endl;
	}
	init(
			F,
			Descr->n, f_semilinear,
			true /* f_init_incidence_structure */,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::init_from_description, "
				"after init" << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::init_from_description done" << endl;
	}
}


void projective_space_with_action::init(
		algebra::field_theory::finite_field *F,
		int n, int f_semilinear,
	int f_init_incidence_structure,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::init, "
				"verbose_level=" << verbose_level << endl;
		cout << "projective_space_with_action::init, "
				"f_semilinear=" << f_semilinear << endl;
	}
	projective_space_with_action::f_init_incidence_structure
		= f_init_incidence_structure;
	projective_space_with_action::n = n;
	projective_space_with_action::F = F;
	projective_space_with_action::f_semilinear = f_semilinear;
	d = n + 1;
	q = F->q;
	
	P = NEW_OBJECT(geometry::projective_geometry::projective_space);

	if (f_v) {
		cout << "projective_space_with_action::init "
				"before P->projective_space_init" << endl;
	}

	P->projective_space_init(n, F,
		f_init_incidence_structure, 
		verbose_level);
	

	if (f_v) {
		cout << "projective_space_with_action::init "
				"after P->projective_space_init" << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::init "
				"before init_group" << endl;
	}

	init_group(f_semilinear, verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::init "
				"after init_group" << endl;
	}


	if (n == 2) {
		if (f_v) {
			cout << "projective_space_with_action::init "
					"n == 2" << endl;
		}
		Dom = NEW_OBJECT(geometry::algebraic_geometry::quartic_curve_domain);

		if (f_v) {
			cout << "projective_space_with_action::init "
					"before Dom->init" << endl;
		}
		Dom->init(F, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"after Dom->init" << endl;
		}
		QCDA = NEW_OBJECT(applications_in_algebraic_geometry::
				quartic_curves::quartic_curve_domain_with_action);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"before QCDA->init" << endl;
		}
		QCDA->init(Dom, this, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"after QCDA->init" << endl;
		}
	}

	if (n >= 3) {
		if (f_v) {
			cout << "projective_space_with_action::init "
					"n >= 3, so we initialize a plane" << endl;
		}
		PA2 = NEW_OBJECT(projective_space_with_action);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"before PA2->init" << endl;
		}
		PA2->init(F, 2, f_semilinear,
			f_init_incidence_structure,
			verbose_level - 2);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"after PA2->init" << endl;
		}
	}
	if (n == 3) {
		if (f_v) {
			cout << "projective_space_with_action::init "
					"n == 3, so we initialize a Surf_A object" << endl;
		}

		geometry::algebraic_geometry::surface_domain *Surf;

		Surf = NEW_OBJECT(geometry::algebraic_geometry::surface_domain);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"before Surf->init_surface_domain" << endl;
		}
		Surf->init_surface_domain(F, verbose_level - 1);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"after Surf->init_surface_domain" << endl;
		}

		Surf_A = NEW_OBJECT(applications_in_algebraic_geometry::
				cubic_surfaces_in_general::surface_with_action);

		if (f_v) {
			cout << "projective_space_with_action::init before Surf_A->init" << endl;
		}
		Surf_A->init(Surf, this, true /* f_recoordinatize */, verbose_level - 1);
		if (f_v) {
			cout << "projective_space_with_action::init after Surf_A->init" << endl;
		}

	}


	
	Elt1 = NEW_int(A->elt_size_in_int);


	if (f_v) {
		cout << "projective_space_with_action::init done" << endl;
	}
}

void projective_space_with_action::init_group(
		int f_semilinear, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "projective_space_with_action::init_group" << endl;
	}

	data_structures_groups::vector_ge *nice_gens;

	A = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"before A->Known_groups->init_linear_group" << endl;
	}
	A->Known_groups->init_linear_group(
		F, d, 
		true /*f_projective*/,
		false /* f_general*/,
		false /* f_affine */,
		f_semilinear,
		false /* f_special */,
		nice_gens,
		verbose_level - 2);
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"after A->Known_groups->init_linear_group" << endl;
	}
#if 0
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"before create_sims" << endl;
	}
	S = A->Strong_gens->create_sims(verbose_level - 2);

	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"after create_sims" << endl;
	}
#endif
	FREE_OBJECT(nice_gens);


	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating action on lines" << endl;
	}
	A_on_lines = A->Induced_action->induced_action_on_grassmannian(
			2, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating action on lines done" << endl;
	}

	if (d >= 4) {
		if (f_v) {
			cout << "projective_space_with_action::init_group "
					"creating action on planes" << endl;
		}
		f_has_action_on_planes = true;
		A_on_planes = A->Induced_action->induced_action_on_grassmannian(
				3, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::init_group "
					"creating action on lines planes" << endl;
		}
	}
	else {
		f_has_action_on_planes = false;
		A_on_planes = NULL;
	}

	if (f_v) {
		cout << "projective_space_with_action::init_group done" << endl;
	}
}



void projective_space_with_action::report_orbits_on_points_lines_and_planes(
	int *Elt, std::ostream &ost,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_orbits_on_points_lines_and_planes" << endl;
	}

	if (P->Subspaces->n < 3) {
		cout << "projective_space_with_action::report_orbits_on_points_lines_and_planes "
				"P->Subspaces->n < 3" << endl;
		exit(1);
	}
	//projective_space *P3;
	int order;

	algebra::ring_theory::longinteger_object full_group_order;
	order = A->Group_element->element_order(Elt);

	full_group_order.create(order);

	//P3 = P;

	ost << "Fixed Objects:\\\\" << endl;



	ost << "The group generated by the element" << endl;
	ost << "$$" << endl;
	A->Group_element->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following orbits:\\\\" << endl;

	ost << "Orbits on points:\\\\" << endl;


	groups::schreier *Sch;
	actions::action_global AcGl;

	Sch = NEW_OBJECT(groups::schreier);
	AcGl.all_point_orbits_from_single_generator(A,
			*Sch,
			Elt,
			0 /*verbose_level*/);
	Sch->Forest->print_orbit_lengths_tex(ost);


	FREE_OBJECT(Sch);

	ost << "Orbits on lines:\\\\" << endl;

	{
		actions::action *A2;
		groups::schreier *Sch;

		A2 = A->Induced_action->induced_action_on_grassmannian(
				2, 0 /* verbose_level*/);

		Sch = NEW_OBJECT(groups::schreier);
		AcGl.all_point_orbits_from_single_generator(A2,
				*Sch,
				Elt,
				0 /*verbose_level*/);
		Sch->Forest->print_orbit_lengths_tex(ost);


		FREE_OBJECT(Sch);
		FREE_OBJECT(A2);
	}

	ost << "Orbits on planes:\\\\" << endl;

	{
		actions::action *A2;
		groups::schreier *Sch;


		A2 = A->Induced_action->induced_action_on_grassmannian(
				3, 0 /* verbose_level*/);

		Sch = NEW_OBJECT(groups::schreier);
		AcGl.all_point_orbits_from_single_generator(A2,
				*Sch,
				Elt,
				0 /*verbose_level*/);
		Sch->Forest->print_orbit_lengths_tex(ost);


		FREE_OBJECT(Sch);
		FREE_OBJECT(A2);
	}

	if (f_v) {
		cout << "projective_space_with_action::report_orbits_on_points_lines_and_planes done" << endl;
	}
}






void projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG(
		int decomposition_by_element_power,
		std::string &decomposition_by_element_data,
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG "
				"verbose_level=" << verbose_level << endl;
	}


	algebra::field_theory::finite_field *F;

	F = P->Subspaces->F;


	{
		string title, author, extra_praeamble;

		title = "Decomposition of PG($" + std::to_string(n) + "," + std::to_string(F->q) + "$)";



		string fname_tex;

		fname_tex = fname_base + ".tex";

		{
			ofstream ost(fname_tex);
			other::l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG "
						"f_decomposition_by_element" << endl;
			}

			int *Elt;

			Elt = NEW_int(A->elt_size_in_int);


			A->Group_element->make_element_from_string(Elt,
					decomposition_by_element_data, verbose_level);


			A->Group_element->element_power_int_in_place(Elt,
					decomposition_by_element_power, verbose_level);

			apps_geometry::top_level_geometry_global Geo;


			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG "
						"before Geo.report_decomposition_by_single_automorphism" << endl;
			}

			Geo.report_decomposition_by_single_automorphism(
					this,
					Elt, ost, fname_base,
					verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG "
						"after Geo.report_decomposition_by_single_automorphism" << endl;
			}
			FREE_int(Elt);


			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname_tex << " of size "
					<< Fio.file_size(fname_tex) << endl;
		}
	}

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG done" << endl;
	}

}

void projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup(
		std::string &label,
		group_constructions::linear_group_description * subgroup_Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup" << endl;
	}


	group_constructions::linear_group *H_LG;

	H_LG = NEW_OBJECT(group_constructions::linear_group);

	subgroup_Descr->F = P->Subspaces->F;

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup "
				"H_LG->init, "
				"creating the group" << endl;
	}

	H_LG->linear_group_init(subgroup_Descr, verbose_level - 2);

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup "
				"after H_LG->linear_group_init" << endl;
	}


	string fname;

	fname = H_LG->label + "_decomp.tex";


	{
		string title, author, extra_praeamble;


		title = "Decomposition of PG($" + std::to_string(n) + "," + std::to_string(F->q) + "$)";



		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG "
						"f_decomposition_by_element" << endl;
			}


			report_decomposition_by_group(
					H_LG->Strong_gens, ost, H_LG->label,
					verbose_level);



			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup done" << endl;
	}
}




void projective_space_with_action::canonical_form_of_code(
		std::string &label_txt,
		int *genma, int m, int n,
		combinatorics::canonical_form_classification::classification_of_objects_description
			*Canonical_form_codes_Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code" << endl;
	}
	int i, j;
	int *v;
	long int *set;

	if (f_v) {
		cout << "Generator matrix: " << endl;
		Int_matrix_print(genma, m, n);
		cout << endl;
	}
	v = NEW_int(m);
	set = NEW_lint(n);
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			v[i] = genma[i * n + j];
		}
		if (f_v) {
			cout << "projective_space_with_action::canonical_form_of_code "
					"before PA->P->rank_point" << endl;
			Int_vec_print(cout, v, m);
			cout << endl;
		}
		if (P == NULL) {
			cout << "P == NULL" << endl;
			exit(1);
		}
		set[j] = P->rank_point(v);
	}
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code set=";
		Lint_vec_print(cout, set, n);
		cout << endl;
	}

	other::data_structures::string_tools ST;
	string points_as_string;

	ST.create_comma_separated_list(points_as_string, set, n);
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code "
				"points_as_string=" << points_as_string << endl;
	}

	combinatorics::canonical_form_classification::data_input_stream_description ISD;
	combinatorics::canonical_form_classification::data_input_stream_description_element E;

	E.init_set_of_points(points_as_string);
	ISD.Input.push_back(E);
	ISD.nb_inputs++;


	ISD.f_label = true;
	ISD.label_txt = label_txt;
	ISD.label_tex = label_txt;


	//data_structures::data_input_stream IS;

	//IS.init(&ISD, verbose_level);



	apps_combinatorics::combinatorial_object_activity_description COAD;


	COAD.f_canonical_form_PG = true;
	COAD.f_canonical_form_PG_has_PA = true;
	COAD.Canonical_form_PG_PA = this;
	COAD.Canonical_form_PG_Descr = Canonical_form_codes_Descr;

	COAD.f_report = true;
	COAD.Objects_report_options =
			NEW_OBJECT(combinatorics::canonical_form_classification::objects_report_options);

	COAD.Objects_report_options->f_export_flag_orbits = true;
	COAD.Objects_report_options->f_show_incidence_matrices = true;
	COAD.Objects_report_options->f_show_TDO = true;
	COAD.Objects_report_options->f_show_TDA = true;
	COAD.Objects_report_options->f_export_group_GAP = true;
	COAD.Objects_report_options->f_export_group_orbiter = true;


	apps_combinatorics::combinatorial_object_stream *Combo;

	Combo = NEW_OBJECT(apps_combinatorics::combinatorial_object_stream);
	Combo->init(
			&ISD,
			verbose_level);


	apps_combinatorics::combinatorial_object_activity COA;

	COA.init_combo(&COAD,
			Combo,
			verbose_level);


	other::orbiter_kernel_system::activity_output *AO;

	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code "
				"before COA.perform_activity" << endl;
	}
	COA.perform_activity(AO, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code "
				"after COA.perform_activity" << endl;
	}

#if 0
    if (AO) {
        FREE_OBJECT(AO);
    }
#endif

	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code "
				"before FREE_int(v);FREE_lint(set);" << endl;
	}


	FREE_int(v);
	FREE_lint(set);
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code "
				"after FREE_int(v);FREE_lint(set);" << endl;
	}
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code done" << endl;
	}

}



void projective_space_with_action::cheat_sheet(
		other::graphics::layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::cheat_sheet" << endl;
	}



	{

		string fname, title, author, extra_praeamble;

		fname = "PG_" + std::to_string(n) + "_" + std::to_string(F->q) + ".tex";
		title = "Cheat Sheet ${\\rm PG}(" + std::to_string(n) + "," + std::to_string(F->q) + ")$";





		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "projective_space_with_action::cheat_sheet "
						"before report" << endl;
			}



			report(ost, O, verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::cheat_sheet "
						"after report" << endl;
			}

			if (f_v) {
				cout << "projective_space_with_action::cheat_sheet "
						"before A->report" << endl;
			}

			A->report(
					ost, A->f_has_sims, A->Sims,
					A->f_has_strong_generators, A->Strong_gens,
					O,
					verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::cheat_sheet "
						"after A->report" << endl;
			}




			if (n == 3) {

				// ToDo PA now has a Surf_A

#if 0
				applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action *Surf_A;

				setup_surface_with_action(
						Surf_A,
						verbose_level);
#endif

				Surf_A->Surf->Schlaefli->print_Steiner_and_Eckardt(ost);

				//FREE_OBJECT(Surf_A);

			}



			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}

	}

	if (f_v) {
		cout << "projective_space_with_action::cheat_sheet done" << endl;
	}


}


void projective_space_with_action::print_points(
		long int *Pts, int nb_pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::print_points" << endl;
	}



	{

		string fname, title, author, extra_praeamble;

		fname = "PG_" + std::to_string(n) + "_" + std::to_string(F->q) + "_points.tex";
		title = "Cheat Sheet ${\\rm PG}(" + std::to_string(n) + "," + std::to_string(F->q) + ")$";





		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "projective_space_with_action::print_points "
						"before report" << endl;
			}


			ost << "\\subsection*{A set of points of ${\\rm \\PG}(" << P->Subspaces->n << "," << P->Subspaces->F->q << ")$}" << endl;
			//P->Reporting->cheat_sheet_points(ost, verbose_level);
			P->Reporting->cheat_sheet_given_set_of_points(
					ost,
					Pts, nb_pts,
					verbose_level);


			if (f_v) {
				cout << "projective_space_with_action::print_points "
						"after report" << endl;
			}



			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}

	}

	if (f_v) {
		cout << "projective_space_with_action::print_points done" << endl;
	}


}


void projective_space_with_action::report(
		std::ostream &ost,
		other::graphics::layered_graph_draw_options *Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report" << endl;
	}

	ost << "\\subsection*{The projective space ${\\rm \\PG}(" << P->Subspaces->n << "," << P->Subspaces->F->q << ")$}" << endl;
	ost << "\\noindent" << endl;
	ost << "\\arraycolsep=2pt" << endl;
	ost << "\\parindent=0pt" << endl;
	ost << "Field order $q = " << P->Subspaces->F->q << "$\\\\" << endl;
	ost << "Characteristic $p = " << P->Subspaces->F->p << "$\\\\" << endl;
	ost << "Extension degree $e = " << P->Subspaces->F->e << "$\\\\" << endl;
	ost << "Projective dimension $n = " << P->Subspaces->n << "$\\\\" << endl;

	ost << "Symmetry group = $" << A->label_tex << "$\\\\" << endl;

	if (A->f_has_strong_generators) {

		algebra::ring_theory::longinteger_object go;

		A->Strong_gens->group_order(go);
		ost << "Order of the group = " << go << "\\\\" << endl;

	}


	ost << "Number of points = " << P->Subspaces->N_points << "\\\\" << endl;
	ost << "Number of lines = " << P->Subspaces->N_lines << "\\\\" << endl;
	ost << "Number of lines on a point = " << P->Subspaces->r << "\\\\" << endl;
	ost << "Number of points on a line = " << P->Subspaces->k << "\\\\" << endl;



	if (A->Stabilizer_chain) {
		if (A->base_len()) {
			ost << "Base: $";
			Lint_vec_print(ost, A->get_base(), A->base_len());
			ost << "$\\\\" << endl;
		}
		if (A->f_has_strong_generators) {
			ost << "{\\small\\arraycolsep=2pt" << endl;
			A->Strong_gens->print_generators_tex(ost);
			ost << "}" << endl;
		}
		else {
			ost << "Does not have strong generators.\\\\" << endl;
		}
	}



	//ost<< "\\clearpage" << endl << endl;
	//ost << "\\section{The Finite Field with $" << q << "$ Elements}" << endl;
	//F->cheat_sheet(ost, verbose_level);

#if 0
	if (f_v) {
		cout << "projective_space_with_action::report before incidence_matrix_save_csv" << endl;
	}
	incidence_matrix_save_csv();
	if (f_v) {
		cout << "projective_space_with_action::report after incidence_matrix_save_csv" << endl;
	}
#endif

	if (P->Subspaces->n == 2) {
		//ost << "\\clearpage" << endl << endl;
		ost << "\\subsection*{The plane}" << endl;

		if (f_v) {
			cout << "projective_space_with_action::report "
					"before create_drawing_of_plane" << endl;
		}
		P->Reporting->create_drawing_of_plane(ost, Draw_options, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::report "
					"after create_drawing_of_plane" << endl;
		}
	}

	//ost << "\\clearpage" << endl << endl;
	ost << "\\subsection*{The points of ${\\rm \\PG}(" << P->Subspaces->n << "," << P->Subspaces->F->q << ")$}" << endl;
	P->Reporting->cheat_sheet_points(ost, verbose_level);

	//cheat_sheet_point_table(ost, verbose_level);


#if 0
	//ost << "\\clearpage" << endl << endl;
	cheat_sheet_points_on_lines(ost, verbose_level);

	//ost << "\\clearpage" << endl << endl;
	cheat_sheet_lines_on_points(ost, verbose_level);
#endif

	// report subspaces:
	int k;

	for (k = 1; k < P->Subspaces->n; k++) {
		//ost << "\\clearpage" << endl << endl;
		if (k == 1) {
			ost << "\\subsection*{The lines of ${\\rm \\PG}(" << P->Subspaces->n << "," << P->Subspaces->F->q << ")$}" << endl;
		}
		else if (k == 2) {
			ost << "\\subsection*{The planes of ${\\rm \\PG}(" << P->Subspaces->n << "," << P->Subspaces->F->q << ")$}" << endl;
		}
		else {
			ost << "\\subsection*{The subspaces of dimension " << k << " of ${\\rm \\PG}(" << P->Subspaces->n << "," << P->Subspaces->F->q << ")$}" << endl;
		}
		//ost << "\\section{Subspaces of dimension " << k << "}" << endl;


		if (f_v) {
			cout << "projective_space_with_action::report "
					"before report_subspaces_of_dimension" << endl;
		}
		P->Reporting->report_subspaces_of_dimension(ost, k + 1, verbose_level);
		//Grass_stack[k + 1]->cheat_sheet_subspaces(ost, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::report "
					"after report_subspaces_of_dimension" << endl;
		}
	}


#if 0
	if (n >= 2 && N_lines < 25) {
		//ost << "\\clearpage" << endl << endl;
		ost << "\\section*{Line intersections}" << endl;
		cheat_sheet_line_intersection(ost, verbose_level);
	}


	if (n >= 2 && N_points < 25) {
		//ost << "\\clearpage" << endl << endl;
		ost << "\\section*{Line through point-pairs}" << endl;
		cheat_sheet_line_through_pairs_of_points(ost, verbose_level);
	}
#endif

	if (f_v) {
		cout << "projective_space_reporting::report "
				"before report_polynomial_rings" << endl;
	}
	P->Reporting->report_polynomial_rings(
			ost,
			verbose_level);
	if (f_v) {
		cout << "projective_space_reporting::report "
				"after report_polynomial_rings" << endl;
	}


	if (f_v) {
		cout << "projective_space_with_action::report done" << endl;
	}

}




void projective_space_with_action::do_spread_classify(
		int k,
		poset_classification::poset_classification_control
			*Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify" << endl;
	}

	int n;

	n = A->matrix_group_dimension();

	geometry::finite_geometries::spread_domain *SD;

	SD = NEW_OBJECT(geometry::finite_geometries::spread_domain);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify "
				"before SD->init_spread_domain" << endl;
	}

	SD->init_spread_domain(
			F,
			n, k,
			verbose_level - 1);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify "
				"after SD->init_spread_domain" << endl;
	}

	spreads::spread_classify_description *Descr;

	Descr = NEW_OBJECT(spreads::spread_classify_description);


	Descr->f_recoordinatize = true;


	spreads::spread_classify *SC;

	SC = NEW_OBJECT(spreads::spread_classify);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify "
				"before SC->init" << endl;
	}

	SC->init(
			Descr,
			SD,
			this,
			verbose_level - 1);
	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify "
				"after SC->init" << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify "
				"before SC->classify_partial_spreads" << endl;
	}

	SC->classify_partial_spreads(verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify "
				"after SC->classify_partial_spreads" << endl;
	}


	FREE_OBJECT(SC);
	FREE_OBJECT(SD);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify done" << endl;
	}
}

void projective_space_with_action::report_decomposition_by_group(
		groups::strong_generators *SG,
		std::ostream &ost, std::string &fname_base,
	int verbose_level)
// used by projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_group" << endl;
	}


	apps_geometry::top_level_geometry_global Geo;


	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_group "
				"before Geo.report_decomposition_by_group" << endl;
	}
	Geo.report_decomposition_by_group(
			this,
			SG, ost, fname_base,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_group "
				"after Geo.report_decomposition_by_group" << endl;
	}



	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_group done" << endl;
	}
}




void projective_space_with_action::report_fixed_objects(
		std::string &Elt_text,
		std::string &fname_latex, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_fixed_objects" << endl;
	}

	int *data;
	int sz;
	int *Elt;

	Get_int_vector_from_label(Elt_text, data, sz, 0 /* verbose_level */);
	if (sz != A->make_element_size) {
		cout << "projective_space_with_action::report_fixed_objects "
				"the size of the input does not match" << endl;
		cout << "expected: " << A->make_element_size << endl;
		cout << "seeing: " << sz << endl;
		exit(1);
	}
	Elt = NEW_int(A->elt_size_in_int);
	A->Group_element->make_element(Elt, data, verbose_level);


	{
		ofstream fp(fname_latex);
		string title, author, extra_praeamble;

		other::l1_interfaces::latex_interface L;

		title = "Fixed Objects";
		author = "";

		L.head(fp,
			false /* f_book */, true /* f_title */,
			title, author /* const char *author */,
			false /* f_toc */, false /* f_landscape */, true /* f_12pt */,
			true /* f_enlarged_page */, true /* f_pagenumbers */,
			extra_praeamble /* extra_praeamble */);
		//latex_head_easy(fp);



		if (f_v) {
			cout << "projective_space_with_action::report_fixed_objects "
					"before A->Group_element->report_fixed_objects_in_PG" << endl;
		}
		A->Group_element->report_fixed_objects_in_PG(fp,
				P,
				Elt,
				verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::report_fixed_objects "
					"after A->Group_element->report_fixed_objects_in_PG" << endl;
		}


		L.foot(fp);
	}
	other::orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname_latex << " of size "
			<< Fio.file_size(fname_latex) << endl;

	FREE_int(Elt);

	if (f_v) {
		cout << "projective_space_with_action::report_fixed_objects done" << endl;
	}
}





}}}

