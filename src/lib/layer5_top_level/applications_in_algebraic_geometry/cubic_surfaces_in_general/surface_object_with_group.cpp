// surface_object_with_group.cpp
// 
// Anton Betten
//
// October 4, 2017
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_in_general {



surface_object_with_group::surface_object_with_group()
{
	Record_birth();
	Surf = NULL;
	Surf_A = NULL;
	SO = NULL;
	Aut_gens = NULL;

	f_has_nice_gens = false;
	nice_gens = NULL;


	projectivity_group_gens = NULL;
	Syl = NULL;

	TD = NULL;


	A_on_points = NULL;
	A_on_Eckardt_points = NULL;
	A_on_Double_points = NULL;
	A_on_Single_points = NULL;
	A_on_the_lines = NULL;
	A_single_sixes = NULL;
	A_double_sixes = NULL;
	A_on_tritangent_planes = NULL;
	A_on_Hesse_planes = NULL;
	A_on_axes = NULL;
	A_on_trihedral_pairs = NULL;
	A_on_pts_not_on_lines = NULL;


	Orbits_on_points = NULL;
	Orbits_on_Eckardt_points = NULL;
	Orbits_on_Double_points = NULL;
	Orbits_on_Single_points = NULL;
	Orbits_on_lines = NULL;
	Orbits_on_single_sixes = NULL;
	Orbits_on_double_sixes = NULL;
	Orbits_on_tritangent_planes = NULL;
	Orbits_on_Hesse_planes = NULL;
	Orbits_on_axes = NULL;
	Orbits_on_trihedral_pairs = NULL;
	Orbits_on_points_not_on_lines = NULL;

	print_interval = 10000;

}

surface_object_with_group::~surface_object_with_group()
{
	Record_death();
	int verbose_level = 0;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::~surface_object_with_group" << endl;
	}
	if (projectivity_group_gens) {
		if (f_v) {
			cout << "surface_object_with_group::~surface_object_with_group before FREE_OBJECT(projectivity_group_gens)" << endl;
		}
		FREE_OBJECT(projectivity_group_gens);
	}
	if (Syl) {
		if (f_v) {
			cout << "surface_object_with_group::~surface_object_with_group before FREE_OBJECT(Syl)" << endl;
		}
		FREE_OBJECT(Syl);
		if (f_v) {
			cout << "surface_object_with_group::~surface_object_with_group after FREE_OBJECT(Syl)" << endl;
		}
	}
	if (TD) {
		if (f_v) {
			cout << "surface_object_with_group::~surface_object_with_group before FREE_OBJECT(TD)" << endl;
		}
		FREE_OBJECT(TD);
	}

	if (A_on_points) {
		if (f_v) {
			cout << "surface_object_with_group::~surface_object_with_group before FREE_OBJECT(A_on_points)" << endl;
		}
		FREE_OBJECT(A_on_points);
	}
	if (A_on_Eckardt_points) {
		FREE_OBJECT(A_on_Eckardt_points);
	}
	if (A_on_Double_points) {
		FREE_OBJECT(A_on_Double_points);
	}
	if (A_on_Single_points) {
		FREE_OBJECT(A_on_Single_points);
	}
	if (A_on_the_lines) {
		FREE_OBJECT(A_on_the_lines);
	}
	if (A_single_sixes) {
		FREE_OBJECT(A_single_sixes);
	}
	if (A_double_sixes) {
		FREE_OBJECT(A_double_sixes);
	}
	if (A_on_tritangent_planes) {
		FREE_OBJECT(A_on_tritangent_planes);
	}
	if (A_on_Hesse_planes) {
		FREE_OBJECT(A_on_Hesse_planes);
	}
	if (A_on_axes) {
		if (f_v) {
			cout << "surface_object_with_group::~surface_object_with_group before FREE_OBJECT(A_on_axes)" << endl;
		}
		FREE_OBJECT(A_on_axes);
	}
	if (A_on_trihedral_pairs) {
		FREE_OBJECT(A_on_trihedral_pairs);
	}
	if (A_on_pts_not_on_lines) {
		FREE_OBJECT(A_on_pts_not_on_lines);
	}
	if (Orbits_on_points) {
		FREE_OBJECT(Orbits_on_points);
	}
	if (Orbits_on_Eckardt_points) {
		FREE_OBJECT(Orbits_on_Eckardt_points);
	}
	if (Orbits_on_Double_points) {
		FREE_OBJECT(Orbits_on_Double_points);
	}
	if (Orbits_on_Single_points) {
		FREE_OBJECT(Orbits_on_Single_points);
	}
	if (Orbits_on_lines) {
		FREE_OBJECT(Orbits_on_lines);
	}
	if (Orbits_on_single_sixes) {
		FREE_OBJECT(Orbits_on_single_sixes);
	}
	if (Orbits_on_double_sixes) {
		if (f_v) {
			cout << "surface_object_with_group::~surface_object_with_group before FREE_OBJECT(Orbits_on_double_sixes)" << endl;
		}
		FREE_OBJECT(Orbits_on_double_sixes);
	}
	if (Orbits_on_tritangent_planes) {
		FREE_OBJECT(Orbits_on_tritangent_planes);
	}
	if (Orbits_on_Hesse_planes) {
		FREE_OBJECT(Orbits_on_Hesse_planes);
	}
	if (Orbits_on_axes) {
		FREE_OBJECT(Orbits_on_axes);
	}
	if (Orbits_on_trihedral_pairs) {
		FREE_OBJECT(Orbits_on_trihedral_pairs);
	}
	if (Orbits_on_points_not_on_lines) {
		FREE_OBJECT(Orbits_on_points_not_on_lines);
	}
	if (f_v) {
		cout << "surface_object_with_group::~surface_object_with_group done" << endl;
	}
}

void surface_object_with_group::cheat_sheet(
		std::ostream &ost,
		int f_print_orbits, std::string &fname_mask,
		other::graphics::layered_graph_draw_options *Opt,
		int max_nb_elements_printed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet" << endl;
		cout << "surface_object_with_group::cheat_sheet "
				"verbose_level = " << verbose_level << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet "
				"before SO->print_equation" << endl;
	}
	SO->SOP->print_equation(ost);
	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet "
				"after SO->print_equation" << endl;
	}




	algebra::ring_theory::longinteger_object ago;
	Aut_gens->group_order(ago);
	ost << "The automorphism group has order "
			<< ago << "\\\\" << endl;




	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet "
				"before print_everything" << endl;
	}

	print_everything(ost, verbose_level - 1);



	ost << "\\section*{The Automorphism Group}" << endl;


	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet "
				"after print_everything" << endl;
	}

	print_automorphism_group_generators(ost, max_nb_elements_printed, verbose_level);



	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet "
				"before print_automorphism_group" << endl;
	}
	print_automorphism_group(
			ost, verbose_level - 1);


#if 0
	if (SO->nb_pts_not_on_lines) {

		if (f_v) {
			cout << "surface_object_with_group::cheat_sheet "
					"before cheat_sheet_quartic_curve" << endl;
		}
		cheat_sheet_quartic_curve(ost,
			label_txt, label_tex, verbose_level);
		if (f_v) {
			cout << "surface_object_with_group::cheat_sheet "
					"after cheat_sheet_quartic_curve" << endl;
		}

	}
#endif


	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet "
				"before print_orbits_of_automorphism_group" << endl;
	}
	print_orbits_of_automorphism_group(
			ost, f_print_orbits,
			fname_mask, Opt, max_nb_elements_printed, verbose_level - 1);
	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet "
				"after print_orbits_of_automorphism_group" << endl;
	}




	ost << "\\clearpage\\subsection*{The Elements of "
			"the Automorphism Group}" << endl;


	Aut_gens->print_elements_latex_ost(max_nb_elements_printed, ost);

	ost << "\\clearpage\\subsection*{The Group Table}" << endl;
	long int go;
	int block_width = 12;

	go = Aut_gens->group_order_as_lint();
	if (go < 50) {

		other::l1_interfaces::latex_interface L;

		int *Table;
		Aut_gens->create_group_table(
				Table, go, verbose_level - 1);
		L.print_integer_matrix_tex_block_by_block(
				ost,
				Table, go, go, block_width);
		FREE_int(Table);
	}
	else {
		ost << "Too big to print." << endl;
	}


	Aut_gens->export_group_and_copy_to_latex(
			SO->label_txt,
			ost,
			A_on_the_lines,
			verbose_level - 2);


	if (Aut_gens->A->degree < 500) {

		Aut_gens->export_group_and_copy_to_latex(
				SO->label_txt,
				ost,
				Aut_gens->A,
				verbose_level - 2);

	}
	else {
		cout << "permutation degree is too large, "
				"skipping export to magma and GAP" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet "
				"before SO->Surf->Schlaefli->print_Steiner_and_Eckardt" << endl;
	}

	SO->Surf->Schlaefli->print_Steiner_and_Eckardt(ost);

	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet "
				"after SO->Surf->Schlaefli->print_Steiner_and_Eckardt" << endl;
	}


	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet done" << endl;
	}


}


void surface_object_with_group::init_equation(
	surface_with_action *Surf_A, int *eqn,
	groups::strong_generators *Aut_gens,
	std::string &label_txt,
	std::string &label_tex,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::init_equation" << endl;
	}

	surface_object_with_group::Surf_A = Surf_A;
	surface_object_with_group::Aut_gens = Aut_gens;
	Surf = Surf_A->Surf;

	SO = NEW_OBJECT(geometry::algebraic_geometry::surface_object);
	if (f_v) {
		cout << "surface_object_with_group::init_equation "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(
			Surf_A->Surf, eqn,
			label_txt, label_tex,
			verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::init_equation "
				"after SO->init_equation" << endl;
	}

#if 0
	if (SO->nb_lines != 27) {
		cout << "surface_object_with_group::init_equation "
				"the surface does not have 27 lines" << endl;
		return false;
	}
#endif

	
	compute_projectivity_group(verbose_level);

	if (f_v) {
		cout << "surface_object_with_group::init_equation "
				"before compute_orbits_of_automorphism_group" << endl;
	}
	compute_orbits_of_automorphism_group(verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::init_equation "
				"after compute_orbits_of_automorphism_group" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::init_equation done" << endl;
	}
}



void surface_object_with_group::init_with_group(
		surface_with_action *Surf_A,
	long int *Lines, int nb_lines, int *eqn,
	groups::strong_generators *Aut_gens,
	std::string &label_txt,
	std::string &label_tex,
	int f_find_double_six_and_rearrange_lines,
	int f_has_nice_gens,
	data_structures_groups::vector_ge *nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::init_with_group" << endl;
	}

	geometry::algebraic_geometry::surface_object *SO;

	SO = NEW_OBJECT(geometry::algebraic_geometry::surface_object);

	if (nb_lines == 27) {
		if (f_v) {
			cout << "surface_object_with_group::init_with_group "
					"before SO->init_with_27_lines" << endl;
		}
		SO->init_with_27_lines(
				Surf_A->Surf, Lines, eqn,
				label_txt, label_tex,
				f_find_double_six_and_rearrange_lines, verbose_level);
		if (f_v) {
			cout << "surface_object_with_group::init_with_group "
					"after SO->init_with_27_lines" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "surface_object_with_group::init_with_group "
					"before SO->init_equation" << endl;
		}
		SO->init_equation(
				Surf_A->Surf, eqn,
				label_txt, label_tex,
				verbose_level);
		if (f_v) {
			cout << "surface_object_with_group::init_with_group "
					"after SO->init_equation" << endl;
		}

	}

	if (f_v) {
		cout << "surface_object_with_group::init_with_group "
				"before SO->init_with_surface_object" << endl;
	}

	init_with_surface_object(
			Surf_A,
			SO,
			Aut_gens,
			f_has_nice_gens, nice_gens,
			verbose_level);

	if (f_v) {
		cout << "surface_object_with_group::init_with_group "
				"after SO->init_with_surface_object" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::init_with_group done" << endl;
	}
}


void surface_object_with_group::init_with_surface_object(
		surface_with_action *Surf_A,
		geometry::algebraic_geometry::surface_object *SO,
		groups::strong_generators *Aut_gens,
		int f_has_nice_gens,
		data_structures_groups::vector_ge *nice_gens,
		int verbose_level)
// ToDo: Aut_gens and nice_gens will become part of the object. This is not good.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object" << endl;
	}

	surface_object_with_group::Surf_A = Surf_A;


	if (f_has_nice_gens) {
		if (f_v) {
			cout << "surface_object_with_group::init_with_surface_object "
					"before nice_gens->copy" << endl;
		}
		nice_gens->copy(
				surface_object_with_group::nice_gens, verbose_level);
		if (f_v) {
			cout << "surface_object_with_group::init_with_surface_object "
					"after nice_gens->copy" << endl;
		}

	}
	else {
		nice_gens = NULL;
	}
	surface_object_with_group::f_has_nice_gens = f_has_nice_gens;

	//surface_object_with_group::nice_gens = nice_gens;

	surface_object_with_group::SO = SO;

	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object "
				"before Aut_gens->create_copy" << endl;
	}
	surface_object_with_group::Aut_gens = Aut_gens->create_copy(verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object "
				"after Aut_gens->create_copy" << endl;
	}

	Surf = Surf_A->Surf;

	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object "
				"testing Aut_gens" << endl;
	}
	Aut_gens->test_if_set_is_invariant_under_given_action(
			Surf_A->A2,
			SO->Variety_object->Line_sets->Sets[0],
			SO->Variety_object->Line_sets->Set_size[0],
			verbose_level - 2);
	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object "
				"testing Aut_gens done" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object "
				"group order = ";
		algebra::ring_theory::longinteger_object go;
		Aut_gens->group_order(go);
		cout << go << endl;
	}


	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object "
				"before compute_projectivity_group" << endl;
	}
	compute_projectivity_group(verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object "
				"after compute_projectivity_group" << endl;
	}


#if 0
	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object "
				"before compute_tactical_decompositions" << endl;
	}
	compute_tactical_decompositions(verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object "
				"after compute_tactical_decompositions" << endl;
	}
#endif

	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object "
				"before compute_orbits_of_automorphism_group" << endl;
	}
	compute_orbits_of_automorphism_group(verbose_level - 2);
	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object "
				"after compute_orbits_of_automorphism_group" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::init_with_surface_object done" << endl;
	}
}


void surface_object_with_group::init_surface_object(
	surface_with_action *Surf_A,
	geometry::algebraic_geometry::surface_object *SO,
	groups::strong_generators *Aut_gens,
	int verbose_level)
// Does not compute tactical decompositions for now
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::init_surface_object" << endl;
	}
	surface_object_with_group::Surf_A = Surf_A;
	surface_object_with_group::SO = SO;
	surface_object_with_group::Aut_gens = Aut_gens;


	Surf = Surf_A->Surf;


	
	if (f_v) {
		cout << "surface_object_with_group::init_surface_object "
				"before compute_projectivity_group" << endl;
	}
	compute_projectivity_group(verbose_level - 5);
	if (f_v) {
		cout << "surface_object_with_group::init_surface_object "
				"after compute_projectivity_group" << endl;
	}

#if 1
	if (f_v) {
		cout << "surface_object_with_group::init_surface_object "
				"before compute_tactical_decompositions" << endl;
	}
	compute_tactical_decompositions(verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::init_surface_object "
				"after compute_tactical_decompositions" << endl;
	}
#endif

	if (f_v) {
		cout << "surface_object_with_group::init_surface_object "
				"before compute_orbits_of_automorphism_group" << endl;
	}
	compute_orbits_of_automorphism_group(verbose_level - 2);
	if (f_v) {
		cout << "surface_object_with_group::init_surface_object "
				"after compute_orbits_of_automorphism_group" << endl;
	}
	
	if (f_v) {
		cout << "surface_object_with_group::init_surface_object "
				"done" << endl;
	}
}

void surface_object_with_group::compute_projectivity_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::compute_projectivity_group" << endl;
		cout << "surface_object_with_group::compute_projectivity_group "
				"verbose_level=" << verbose_level << endl;
	}

	actions::action_global Action_global;


	if (f_v) {
		cout << "surface_object_with_group::compute_projectivity_group "
				"before Action_global.compute_projectivity_subgroup" << endl;
	}
	Action_global.compute_projectivity_subgroup(
			Surf_A->A,
			projectivity_group_gens,
			Aut_gens, verbose_level - 2);
	if (f_v) {
		cout << "surface_object_with_group::compute_projectivity_group "
				"after Action_global.compute_projectivity_subgroup" << endl;
	}
	if (f_v) {
		cout << "surface_object_with_group::compute_projectivity_group "
				"computing order of projectivity_group " << endl;

		algebra::ring_theory::longinteger_object go;

		projectivity_group_gens->group_order(go);

		cout << "surface_object_with_group::compute_projectivity_group "
				"projectivity_group has order " << go << endl;

	}




	if (f_v) {
		cout << "surface_object_with_group::compute_projectivity_group "
				"computing Sylow structure" << endl;
	}
	// compute the Sylow structure:
	groups::sims *S = NULL;

	if (projectivity_group_gens) {
		S = projectivity_group_gens->create_sims(0 /*verbose_level */);
	}
	else {
		if (Aut_gens) {
			S = Aut_gens->create_sims(0 /*verbose_level */);
		}
	}

	if (S) {
		if (f_v) {
			cout << "surface_object_with_group::compute_projectivity_group "
					"before Syl->init" << endl;
		}
		Syl = NEW_OBJECT(groups::sylow_structure);
		Syl->init(
				S,
				SO->label_txt,
				SO->label_tex,
				verbose_level - 2);
		if (f_v) {
			cout << "surface_object_with_group::compute_projectivity_group "
					"after Syl->init" << endl;
		}
	}


	if (f_v) {
		cout << "surface_object_with_group::compute_projectivity_group done" << endl;
	}
}

void surface_object_with_group::recognize_Fabcd(
		int f_extra_point, int point_rk,
		orbits::orbits_create *Classification_of_arcs,
		int &a, int &b, int &c, int &d, int &e, int &f,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd" << endl;
	}
	if (f_extra_point) {
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd extra_point = " << point_rk << endl;
		}
	}
	else {
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd no extra point" << endl;
		}
	}


	if (!Classification_of_arcs->f_has_arcs) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"the orbits object has the wrong kind of orbit objects. "
				"It should have orbits on arcs." << endl;
		exit(1);
	}

	// Find a transformation which maps plane1 to plane2:

	long int plane1, plane2;

	if (SO->SOP->SmoothProperties == NULL) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"SmoothProperties is NULL" << endl;
		exit(1);
	}

	plane1 = SO->SOP->SmoothProperties->Tritangent_plane_rk[30];
	// \pi_{12,34,56}

	plane2 = 0; // the plane X_3 = 0 has orbit number 0

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
			"plane1 = " << plane1 << " plane2=" << plane2 << endl;
	}

	//int *transporter;
	int *Transporter;
	// the transformation that maps
	// the plane \pi_{12,34,56} to X_3 = 0

	int *equation_nice;

	//transporter = NEW_int(17);
	equation_nice = NEW_int(20);

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
			"before make_element_which_moves_a_point_from_A_to_B" << endl;
	}

	Transporter = NEW_int(Surf_A->A->elt_size_in_int);

	Surf_A->A->Strong_gens->make_element_which_moves_a_point_from_A_to_B(
			Surf_A->A_on_planes,
			plane1, plane2, Transporter,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
			"after make_element_which_moves_a_point_from_A_to_B" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"transporter element=" << endl;
		Surf_A->A->Group_element->element_print_quick(
				Transporter, cout);
	}

	//SOA->Surf_A->A->Group_element->make_element(Transporter, transporter, 0 /* verbose_level */);

	long int extra_point_image = -1;


	if (f_extra_point) {
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd "
				"before Surf_A->map_a_point" << endl;
		}
		extra_point_image = Surf_A->map_a_point(
				Transporter,
				point_rk, 0 /* verbose_level */);
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd "
				"after Surf_A->map_a_point" << endl;
		}
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd "
					"extra point " << point_rk << " -> " << extra_point_image << endl;
		}
	}

	// Transform the equation:

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"before Surf_A->AonHPD_3_4->compute_image_int_low_level" << endl;
	}
	Surf_A->AonHPD_3_4->compute_image_int_low_level(
			Transporter,
			SO->Variety_object->eqn /*int *input*/,
			equation_nice /* int *output */,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
			"equation_nice=" << endl;
		Surf->PolynomialDomains->Poly3_4->print_equation(
				cout, equation_nice);
		cout << endl;
		cout << "surface_object_with_group::recognize_Fabcd "
			"equation_nice=" << endl;
		Int_vec_print(cout, equation_nice, 20);
		cout << endl;
		if (f_extra_point) {
			cout << "surface_object_with_group::recognize_Fabcd "
					"extra point " << point_rk << " -> " << extra_point_image << endl;
		}
	}

	int f_inverse = false;
	int *transformation_coeffs;
	int f_has_group = false;

	transformation_coeffs = Transporter;

	geometry::algebraic_geometry::variety_object *Variety_object_transformed;

	groups::group_theory_global Group_theory_global;

	groups::strong_generators *Strong_gens_out = NULL;

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd before "
				"Group_theory_global.variety_apply_single_transformation" << endl;
	}
	Variety_object_transformed =
			Group_theory_global.variety_apply_single_transformation(
					SO->Variety_object,
					Surf_A->A,
					Surf_A->A2 /* A_on_lines */,
					f_inverse,
					transformation_coeffs,
					f_has_group, NULL /* groups::strong_generators *Strong_gens_in */,
					Strong_gens_out,
					verbose_level - 1);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd after "
				"Group_theory_global.variety_apply_single_transformation" << endl;
	}

	//long int *Lines;
	int nb_lines;

	//Lines = Variety_object_transformed->Line_sets->Sets[0];
	nb_lines = Variety_object_transformed->Line_sets->Set_size[0];

	if (nb_lines != 27) {
		cout << "surface_object_with_group::recognize_Fabcd nb_lines != 27" << endl;
		exit(1);
	}

	int c12, c34, c56;
	int b1, b2, a3, a4, a5, a6;

	int p1, p2, p3, p4, p5, p6;


	c12 = Surf->Schlaefli->line_cij(0, 1);

	c34 = Surf->Schlaefli->line_cij(2, 3);

	c56 = Surf->Schlaefli->line_cij(4, 5);


	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd c12,c34,c56="
				<< c12 << ","
				<< c34 << ","
				<< c56 << endl;
	}


	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd finding b1" << endl;
	}
	b1 = Surf->Schlaefli->line_bi(0);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd finding b2" << endl;
	}
	b2 = Surf->Schlaefli->line_bi(1);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd finding b3" << endl;
	}
	a3 = Surf->Schlaefli->line_ai(2);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd finding b4" << endl;
	}
	a4 = Surf->Schlaefli->line_ai(3);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd finding b5" << endl;
	}
	a5 = Surf->Schlaefli->line_ai(4);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd finding b6" << endl;
	}
	a6 = Surf->Schlaefli->line_ai(5);

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd b1,b2,a3,a4,a5,a6="
				<< b1 << ","
				<< b2 << ","
				<< a3 << ","
				<< a4 << ","
				<< a5 << ","
				<< a6 << endl;
	}

	//long int extra_point_image = -1;


	long int p7 = -1;

	if (f_extra_point) {
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd "
					"applying Clebsch map to extra_point_image" << endl;
			cout << "surface_object_with_group::recognize_Fabcd "
					"extra_point_image = " << extra_point_image << endl;
		}

		long int line_rk;
		long int *Lines;

		Lines = Variety_object_transformed->Line_sets->Sets[0];

		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd "
					"before transversal_to_two_skew_lines_through_a_point" << endl;
		}
		line_rk = Surf_A->PA->P->Solid->transversal_to_two_skew_lines_through_a_point(
				Lines[b1], Lines[b2], extra_point_image, 0 /*verbose_level*/);
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd "
					"after transversal_to_two_skew_lines_through_a_point" << endl;
			cout << "surface_object_with_group::recognize_Fabcd "
					"line_rk = " << line_rk << endl;
			Surf->print_a_line_tex(
					cout,
					line_rk);
		}

		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd "
					"before point_of_intersection_of_a_line_and_a_plane_in_three_space" << endl;
		}
		p7 = Surf_A->PA->P->Solid->point_of_intersection_of_a_line_and_a_plane_in_three_space(
				line_rk, plane2, 0 /* verbose_level */);
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd "
					"after point_of_intersection_of_a_line_and_a_plane_in_three_space" << endl;
			cout << "surface_object_with_group::recognize_Fabcd "
					"p7 = " << p7 << endl;
		}

	}



	// need a new surface object with the transformed lines

	geometry::algebraic_geometry::surface_object *SO_trans;

	SO_trans = NEW_OBJECT(geometry::algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"before SO_trans->init_variety_object" << endl;
	}
	SO_trans->init_variety_object(
			Surf,
			Variety_object_transformed,
			verbose_level - 1);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"after SO_trans->init_variety_object" << endl;
	}

	if (f_v) {
		cout << "c12=" << endl;
		SO_trans->print_one_line_tex(
				cout, c12);
	}
	if (f_v) {
		cout << "c34=" << endl;
		SO_trans->print_one_line_tex(
				cout, c34);
	}
	if (f_v) {
		cout << "c56=" << endl;
		SO_trans->print_one_line_tex(
				cout, c56);
	}


	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd finding p1" << endl;
	}
	p1 = SO_trans->find_double_point(
			c12, b1, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd finding p2" << endl;
	}
	p2 = SO_trans->find_double_point(
			c12, b2, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd finding p3" << endl;
	}
	p3 = SO_trans->find_double_point(
			c34, a3, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd finding p4" << endl;
	}
	p4 = SO_trans->find_double_point(
			c34, a4, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd finding p5" << endl;
	}
	p5 = SO_trans->find_double_point(
			c56, a5, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd finding p6" << endl;
	}
	p6 = SO_trans->find_double_point(
			c56, a6, 0 /* verbose_level */);

	if (f_v) {
		if (f_extra_point) {
			cout << "surface_object_with_group::recognize_Fabcd p1,p2,p3,p4,p5,p6,p7="
					<< p1 << ","
					<< p2 << ","
					<< p3 << ","
					<< p4 << ","
					<< p5 << ","
					<< p6 << ","
					<< p7 << endl;

		}
		else {
			cout << "surface_object_with_group::recognize_Fabcd p1,p2,p3,p4,p5,p6="
					<< p1 << ","
					<< p2 << ","
					<< p3 << ","
					<< p4 << ","
					<< p5 << ","
					<< p6 << endl;
		}
	}

	long int *Points;
	geometry::projective_geometry::projective_space *P2;

	P2 = Surf_A->PA->PA2->P;
	Points = SO_trans->Variety_object->Point_sets->Sets[0];
	int v[4];
	long int q1, q2, q3, q4, q5, q6, q7;
	long int arc6[7];

	Surf->unrank_point(v, Points[p1]);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd p1=";
		Int_vec_print(cout, v, 4);
		cout << endl;
	}
	q1 = P2->rank_point(v);
	Surf->unrank_point(v, Points[p2]);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd p2=";
		Int_vec_print(cout, v, 4);
		cout << endl;
	}
	q2 = P2->rank_point(v);
	Surf->unrank_point(v, Points[p3]);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd p3=";
		Int_vec_print(cout, v, 4);
		cout << endl;
	}
	q3 = P2->rank_point(v);
	Surf->unrank_point(v, Points[p4]);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd p4=";
		Int_vec_print(cout, v, 4);
		cout << endl;
	}
	q4 = P2->rank_point(v);
	Surf->unrank_point(v, Points[p5]);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd p5=";
		Int_vec_print(cout, v, 4);
		cout << endl;
	}
	q5 = P2->rank_point(v);
	Surf->unrank_point(v, Points[p6]);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd p6=";
		Int_vec_print(cout, v, 4);
		cout << endl;
	}
	q6 = P2->rank_point(v);


	if (f_extra_point) {
		Surf->unrank_point(v, p7);
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd p7=";
			Int_vec_print(cout, v, 4);
			cout << endl;
		}
		q7 = P2->rank_point(v);
	}
	else {
		q7 = -1;
	}

	arc6[0] = q1;
	arc6[1] = q2;
	arc6[2] = q3;
	arc6[3] = q4;
	arc6[4] = q5;
	arc6[5] = q6;

	int arc_sz;

	if (f_extra_point) {
		arc6[6] = q7;
		arc_sz = 7;
	}
	else {
		arc_sz = 6;
	}
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd arc6[]=";
		Lint_vec_print(cout, arc6, arc_sz);
		cout << endl;
	}

	int v1[3];
	int v2[3];
	int v3[3];
	int v4[3];
	int v5[3];
	int v6[3];
	//int v7[3];

	P2->unrank_point(v1, arc6[0]);
	P2->unrank_point(v2, arc6[1]);
	P2->unrank_point(v3, arc6[2]);
	P2->unrank_point(v4, arc6[3]);
	P2->unrank_point(v5, arc6[4]);
	P2->unrank_point(v6, arc6[5]);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd v1[]=";
		Int_vec_print(cout, v1, 3);
		cout << endl;
		cout << "surface_object_with_group::recognize_Fabcd v2[]=";
		Int_vec_print(cout, v2, 3);
		cout << endl;
		cout << "surface_object_with_group::recognize_Fabcd v3[]=";
		Int_vec_print(cout, v3, 3);
		cout << endl;
		cout << "surface_object_with_group::recognize_Fabcd v4[]=";
		Int_vec_print(cout, v4, 3);
		cout << endl;
		cout << "surface_object_with_group::recognize_Fabcd v5[]=";
		Int_vec_print(cout, v5, 3);
		cout << endl;
		cout << "surface_object_with_group::recognize_Fabcd v6[]=";
		Int_vec_print(cout, v6, 3);
		cout << endl;
	}




	algebra::linear_algebra::linear_algebra_global LA;

	int Mtx[10];
	int image_of_all_one[3];
	//int orbit_at_level;

	Int_vec_copy(v1, Mtx, 3);
	Int_vec_copy(v2, Mtx + 3, 3);
	Int_vec_copy(v3, Mtx + 6, 3);

	Int_vec_copy(v4, image_of_all_one, 3);


	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"before LA.adjust_scalars_in_frame" << endl;
	}
	LA.adjust_scalars_in_frame(
			Surf->F,
			3 /*d*/, Mtx /* Image_of_basis_in_rows */,
			image_of_all_one,
			verbose_level - 1);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"after LA.adjust_scalars_in_frame" << endl;
	}

	int *Transporter2;
	int *Transporter2_inv;
	Transporter2 = NEW_int(Surf_A->PA->PA2->A->elt_size_in_int);
	Transporter2_inv = NEW_int(Surf_A->PA->PA2->A->elt_size_in_int);
	Mtx[9] = 0;
	Surf_A->PA->PA2->A->Group_element->make_element(
			Transporter2_inv, Mtx, 0 /*verbose_level - 2*/);
	Surf_A->PA->PA2->A->Group_element->element_invert(
			Transporter2_inv, Transporter2, 0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"Transporter2:" << endl;
		Surf_A->PA->PA2->A->Group_element->element_print(Transporter2, cout);
	}

	int i;
	long int canonical_arc[7];

	for (i = 0; i < arc_sz; i++) {
		canonical_arc[i] = Surf_A->PA->PA2->A->Group_element->image_of(Transporter2, arc6[i]);
	}


	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd canonical_arc[]=";
		Lint_vec_print(cout, canonical_arc, arc_sz);
		cout << endl;
	}



#if 0
	apps_geometry::arc_generator *Arc_generator;

	Arc_generator = Classification_of_arcs->Arc_generator;

	data_structures_groups::set_and_stabilizer
					*Set_and_stab_original;
	data_structures_groups::set_and_stabilizer
					*Set_and_stab_canonical;

	Set_and_stab_original = NEW_OBJECT(data_structures_groups::set_and_stabilizer);
	Set_and_stab_canonical = NEW_OBJECT(data_structures_groups::set_and_stabilizer);

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"before Arc_generator->gen->identify_and_get_stabilizer" << endl;
	}
	Arc_generator->gen->identify_and_get_stabilizer(
			arc6, 6, Transporter2,
			orbit_at_level,
			Set_and_stab_original,
			Set_and_stab_canonical,
			verbose_level - 3);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"after Arc_generator->gen->identify_and_get_stabilizer" << endl;
	}


	long int canonical_arc[7];

	if (Set_and_stab_canonical->sz != 6) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"Set_and_stab_canonical->sz != 6" << endl;
		exit(1);
	}

	Lint_vec_copy(Set_and_stab_canonical->data, canonical_arc, 6);
	P2->unrank_point(v5, canonical_arc[4]);
	P2->unrank_point(v6, canonical_arc[5]);

	if (f_extra_point) {
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd "
					"before Surf_A->PA->PA2->A->Group_element->image_of" << endl;
		}
		canonical_arc[6] = Surf_A->PA->PA2->A->Group_element->image_of(Transporter2, q7);
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd "
					"after Surf_A->PA->PA2->A->Group_element->image_of" << endl;
			cout << "surface_object_with_group::recognize_Fabcd "
					"canonical_arc[6] = " << canonical_arc[6] << endl;
		}
		P2->unrank_point(v7, canonical_arc[6]);

	}
#endif

	int w1[3];
	int w2[3];
	int w3[3];
	int w4[3];
	int w5[3];
	int w6[3];
	int w7[3];

	P2->unrank_point(w1, canonical_arc[0]);
	P2->unrank_point(w2, canonical_arc[1]);
	P2->unrank_point(w3, canonical_arc[2]);
	P2->unrank_point(w4, canonical_arc[3]);
	P2->unrank_point(w5, canonical_arc[4]);
	P2->unrank_point(w6, canonical_arc[5]);
	if (f_extra_point) {
		P2->unrank_point(w7, canonical_arc[6]);
	}

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd w1[]=";
		Int_vec_print(cout, w1, 3);
		cout << endl;
		cout << "surface_object_with_group::recognize_Fabcd w2[]=";
		Int_vec_print(cout, w2, 3);
		cout << endl;
		cout << "surface_object_with_group::recognize_Fabcd w3[]=";
		Int_vec_print(cout, w3, 3);
		cout << endl;
		cout << "surface_object_with_group::recognize_Fabcd w4[]=";
		Int_vec_print(cout, w4, 3);
		cout << endl;
		cout << "surface_object_with_group::recognize_Fabcd w5[]=";
		Int_vec_print(cout, w5, 3);
		cout << endl;
		cout << "surface_object_with_group::recognize_Fabcd w6[]=";
		Int_vec_print(cout, w6, 3);
		cout << endl;
		if (f_extra_point) {
			cout << "surface_object_with_group::recognize_Fabcd w7[]=";
			Int_vec_print(cout, w7, 3);
			cout << endl;
		}
	}


	if (w5[2] != 1) {
		cout << "surface_object_with_group::recognize_Fabcd w5[2] != 1" << endl;
		exit(1);
	}
	a = w5[0];
	b = w5[1];

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"a,b=" << a << "," << b << endl;
	}

	if (w6[2] != 1) {
		cout << "surface_object_with_group::recognize_Fabcd w6[2] != 1" << endl;
		exit(1);
	}
	c = w6[0];
	d = w6[1];

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"c,d=" << c << "," << d << endl;
	}

	if (f_extra_point) {
		if (w7[2] != 1) {
			cout << "surface_object_with_group::recognize_Fabcd w7[2] != 1" << endl;
			exit(1);
		}
		e = w7[0];
		f = w7[1];
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd "
					"e,f=" << e << "," << f << endl;
		}

	}



	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"before FREE_int(Transporter)" << endl;
	}
	FREE_int(Transporter);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"before FREE_int(equation_nice)" << endl;
	}
	FREE_int(equation_nice);

	if (Strong_gens_out) {
		if (f_v) {
			cout << "surface_object_with_group::recognize_Fabcd "
					"before FREE_OBJECT(Strong_gens_out)" << endl;
		}
		FREE_OBJECT(Strong_gens_out);
	}

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"before FREE_OBJECT(SO_trans)" << endl;
	}
	FREE_OBJECT(SO_trans);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"before FREE_int(Transporter2)" << endl;
	}
	FREE_int(Transporter2);
	FREE_int(Transporter2_inv);
#if 0
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"before FREE_OBJECT(Set_and_stab_original)" << endl;
	}
	FREE_OBJECT(Set_and_stab_original);
	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd "
				"before FREE_OBJECT(Set_and_stab_canonical)" << endl;
	}
	FREE_OBJECT(Set_and_stab_canonical);
#endif

	if (f_v) {
		cout << "surface_object_with_group::recognize_Fabcd done" << endl;
	}
}


void surface_object_with_group::compute_orbits_of_automorphism_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_group::compute_orbits_of_automorphism_group" << endl;
	}

	// orbits on points:
	
	if (f_v) {
		cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
				"orbits on points" << endl;
	}
	init_orbits_on_points(verbose_level - 1);


	// orbits on Eckardt points:
	
	if (f_v) {
		cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
				"orbits on Eckardt points" << endl;
	}
	init_orbits_on_Eckardt_points(verbose_level - 1);


	// orbits on Double points:
	
	if (f_v) {
		cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
				"orbits on double points" << endl;
	}
	init_orbits_on_Double_points(verbose_level - 1);

	// orbits on Single points:

	if (f_v) {
		cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
				"orbits on single points" << endl;
	}
	init_orbits_on_Single_points(verbose_level - 1);


	// orbits on lines:

	if (f_v) {
		cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
				"orbits on lines" << endl;
	}
	init_orbits_on_lines(verbose_level);



	if (SO->Variety_object->Line_sets->Set_size[0] == 27) {

		// orbits on half double sixes:

		if (f_v) {
			cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
					"before init_orbits_on_half_double_sixes" << endl;
		}
		init_orbits_on_half_double_sixes(verbose_level - 1);

		if (f_v) {
			cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
					"after init_orbits_on_half_double_sixes" << endl;
		}

		if (f_v) {
			cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
					"before init_orbits_on_double_sixes" << endl;
		}
		init_orbits_on_double_sixes(verbose_level - 1);

		if (f_v) {
			cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
					"after init_orbits_on_double_sixes" << endl;
		}

		// orbits on tritangent planes:

		if (f_v) {
			cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
					"before init_orbits_on_tritangent_planes" << endl;
		}
		init_orbits_on_tritangent_planes(verbose_level - 1);
		if (f_v) {
			cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
					"after init_orbits_on_tritangent_planes" << endl;
		}


		// orbits on Hesse planes:

		if (f_v) {
			cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
					"before init_orbits_on_Hesse_planes" << endl;
		}
		init_orbits_on_Hesse_planes(verbose_level - 1);
		if (f_v) {
			cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
					"after init_orbits_on_Hesse_planes" << endl;
		}

		// orbits on axes:

		if (f_v) {
			cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
					"before init_orbits_on_axes" << endl;
		}
		init_orbits_on_axes(verbose_level - 1);
		if (f_v) {
			cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
					"after init_orbits_on_axes" << endl;
		}


		// orbits on trihedral pairs:

		if (f_v) {
			cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
					"before init_orbits_on_trihedral_pairs" << endl;
		}
		init_orbits_on_trihedral_pairs(verbose_level - 1);
		if (f_v) {
			cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
					"after init_orbits_on_trihedral_pairs" << endl;
		}

	}



	// orbits on points not on lines:

	if (f_v) {
		cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
				"before init_orbits_on_points_not_on_lines" << endl;
	}
	init_orbits_on_points_not_on_lines(verbose_level - 1);
	if (f_v) {
		cout << "surface_object_with_group::compute_orbits_of_automorphism_group "
				"after init_orbits_on_points_not_on_lines" << endl;
	}


	if (f_v) {
		cout << "surface_object_with_group::compute_orbits_of_automorphism_group done" << endl;
	}
}

void surface_object_with_group::init_orbits_on_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_points" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_points action: ";
		Surf_A->A->print_info();
	}


	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_Pts");
	label_of_set_tex.assign("\\_Pts");

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_points "
				"creating action on points:" << endl;
	}

	A_on_points = Surf_A->A->Induced_action->restricted_action(
			SO->Variety_object->Point_sets->Sets[0],
			SO->Variety_object->Point_sets->Set_size[0],
			label_of_set,
			label_of_set_tex,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_points "
				"creating action on points done" << endl;
	}


	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_points "
				"computing orbits on points:" << endl;
	}
	if (f_has_nice_gens) {
		if (f_v) {
			cout << "surface_object_with_group::init_orbits_on_points "
					"computing orbits on points using nice gens:" << endl;
		}
		Orbits_on_points = nice_gens->compute_all_point_orbits_schreier(
				A_on_points, print_interval, 0 /*verbose_level*/);

	}
	else {
		if (f_v) {
			cout << "surface_object_with_group::init_orbits_on_points "
					"computing orbits on points using Aut_gens:" << endl;
			Aut_gens->print_generators(cout, verbose_level - 1);
		}
		if (f_v) {
			cout << "surface_object_with_group::init_orbits_on_points "
					"before Aut_gens->compute_all_point_orbits_schreier" << endl;
		}

		Orbits_on_points = Aut_gens->compute_all_point_orbits_schreier(
				A_on_points, print_interval, verbose_level - 2);
		if (f_v) {
			cout << "surface_object_with_group::init_orbits_on_points "
					"after Aut_gens->compute_all_point_orbits_schreier" << endl;
		}
	}
	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_points "
				"We found " << Orbits_on_points->Forest->nb_orbits
				<< " orbits on points" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_points done" << endl;
	}
}

void surface_object_with_group::init_orbits_on_Eckardt_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_Eckardt_points" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_Eckardt");
	label_of_set_tex.assign("\\_Eckardt");

	if (f_v) {
		cout << "creating action on Eckardt points:" << endl;
	}
	A_on_Eckardt_points = Surf_A->A->Induced_action->restricted_action(
			SO->SOP->Eckardt_points,
			SO->SOP->nb_Eckardt_points,
			label_of_set,
			label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on Eckardt points done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on Eckardt points:" << endl;
	}
	if (f_has_nice_gens) {
		if (f_v) {
			cout << "surface_object_with_group::init_orbits_on_Eckardt_points "
					"computing orbits on points using nice gens:" << endl;
		}
		Orbits_on_Eckardt_points = nice_gens->compute_all_point_orbits_schreier(
				A_on_Eckardt_points, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_Eckardt_points = Aut_gens->compute_all_point_orbits_schreier(
				A_on_Eckardt_points, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_Eckardt_points->Forest->nb_orbits
				<< " orbits on Eckardt points" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_Eckardt_points done" << endl;
	}
}

void surface_object_with_group::init_orbits_on_Double_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_Double_points" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_Double_pts");
	label_of_set_tex.assign("\\_Double\\_pts");

	if (f_v) {
		cout << "creating action on Double points:" << endl;
	}
	A_on_Double_points = Surf_A->A->Induced_action->restricted_action(
			SO->SOP->Double_points,
			SO->SOP->nb_Double_points,
			label_of_set,
			label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on Double points done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on Double points:" << endl;
	}
	if (f_has_nice_gens) {
		Orbits_on_Double_points = nice_gens->compute_all_point_orbits_schreier(
				A_on_Double_points, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_Double_points = Aut_gens->compute_all_point_orbits_schreier(
				A_on_Double_points, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_Double_points->Forest->nb_orbits
				<< " orbits on Double points" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_Double_points done" << endl;
	}
}

void surface_object_with_group::init_orbits_on_Single_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_Single_points" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_Single_pts");
	label_of_set_tex.assign("\\_Single\\_pts");

	if (f_v) {
		cout << "creating action on Single points:" << endl;
	}
	A_on_Single_points = Surf_A->A->Induced_action->restricted_action(
			SO->SOP->Single_points,
			SO->SOP->nb_Single_points,
			label_of_set,
			label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on Single points done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on Single points:" << endl;
	}
	if (f_has_nice_gens) {
		Orbits_on_Single_points = nice_gens->compute_all_point_orbits_schreier(
				A_on_Single_points, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_Single_points = Aut_gens->compute_all_point_orbits_schreier(
				A_on_Single_points, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_Single_points->Forest->nb_orbits
				<< " orbits on Single points" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_Single_points done" << endl;
	}
}

void surface_object_with_group::init_orbits_on_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_lines" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_Lines");
	label_of_set_tex.assign("\\_Lines");

	if (f_v) {
		cout << "creating restricted action "
				"on the lines:" << endl;
	}
	A_on_the_lines = Surf_A->A2->Induced_action->restricted_action(
			SO->Variety_object->Line_sets->Sets[0],
			SO->Variety_object->Line_sets->Set_size[0],
			label_of_set,
			label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating restricted action "
				"on the lines done" << endl;
	}

	if (f_v) {
		cout << "computing orbits on lines:" << endl;
	}
	if (f_has_nice_gens) {
		Orbits_on_lines = nice_gens->compute_all_point_orbits_schreier(
				A_on_the_lines, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_lines = Aut_gens->compute_all_point_orbits_schreier(
				A_on_the_lines, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_lines->Forest->nb_orbits
				<< " orbits on lines" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_lines done" << endl;
	}
}

void surface_object_with_group::init_orbits_on_half_double_sixes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_half_double_sixes" << endl;
	}

	if (f_v) {
		cout << "creating action on half double sixes:" << endl;
	}
	A_single_sixes = A_on_the_lines->Induced_action->create_induced_action_on_sets(
			72, 6,
			Surf->Schlaefli->Schlaefli_double_six->Double_six,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on half double sixes done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on single sixes:" << endl;
	}
	if (f_has_nice_gens) {
		Orbits_on_single_sixes = nice_gens->compute_all_point_orbits_schreier(
				A_single_sixes, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_single_sixes = Aut_gens->compute_all_point_orbits_schreier(
				A_single_sixes, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "computing orbits on single sixes done" << endl;
	}
	if (f_v) {
		cout << "We found " << Orbits_on_single_sixes->Forest->nb_orbits
				<< " orbits on single sixes" << endl;
	}

	//nb_orbits_on_single_sixes = Orbits_on_single_sixes->nb_orbits;

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_half_double_sixes done" << endl;
	}
}

void surface_object_with_group::init_orbits_on_double_sixes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_double_sixes" << endl;
	}

	long int double_six_sets[72];
	int i, j;

	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			double_six_sets[i * 2 + j] = i * 2 + j;
		}
	}

	if (f_v) {
		cout << "creating action on half double sixes:" << endl;
	}
	A_double_sixes = A_single_sixes->Induced_action->create_induced_action_on_sets(
			36, 2,
			double_six_sets,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on half double sixes done" << endl;
	}


	if (f_v) {
		cout << "computing orbits on double sixes:" << endl;
	}
	if (f_has_nice_gens) {
		Orbits_on_double_sixes = nice_gens->compute_all_point_orbits_schreier(
				A_double_sixes, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_double_sixes = Aut_gens->compute_all_point_orbits_schreier(
				A_double_sixes, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "computing orbits on double sixes done" << endl;
	}
	if (f_v) {
		cout << "We found " << Orbits_on_double_sixes->Forest->nb_orbits
				<< " orbits on double sixes" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_double_sixes done" << endl;
	}
}

void surface_object_with_group::init_orbits_on_tritangent_planes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_tritangent_planes" << endl;
	}

	if (f_v) {
		cout << "creating action on tritangent planes:" << endl;
		cout << "SO->SOP->nb_tritangent_planes = "
				<< SO->SOP->SmoothProperties->nb_tritangent_planes << endl;
	}
	A_on_tritangent_planes =
			A_on_the_lines->Induced_action->create_induced_action_on_sets(
			SO->SOP->SmoothProperties->nb_tritangent_planes, 3,
			Surf->Schlaefli->Schlaefli_tritangent_planes->Lines_in_tritangent_planes,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "action on tritangent planes done" << endl;
	}

	if (f_has_nice_gens) {
		Orbits_on_tritangent_planes = nice_gens->compute_all_point_orbits_schreier(
				A_on_tritangent_planes, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_tritangent_planes = Aut_gens->compute_all_point_orbits_schreier(
				A_on_tritangent_planes, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_tritangent_planes->Forest->nb_orbits
				<< " orbits on the set of " << SO->SOP->SmoothProperties->nb_tritangent_planes
				<< " tritangent planes" << endl;
	}

	if (f_vv) {
		Orbits_on_tritangent_planes->Forest->print_and_list_orbits(cout);
	}

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_tritangent_planes done" << endl;
	}
}

void surface_object_with_group::init_orbits_on_Hesse_planes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_Hesse_planes" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_Hesse_planes");
	label_of_set_tex.assign("\\_Hesse\\_planes");

	if (f_v) {
		cout << "creating action on Hesse planes:" << endl;
		cout << "SO->SOP->nb_Hesse_planes = "
				<< SO->SOP->nb_Hesse_planes << endl;
	}
	A_on_Hesse_planes = Surf_A->A_on_planes->Induced_action->restricted_action(
			SO->SOP->Hesse_planes, SO->SOP->nb_Hesse_planes,
			label_of_set, label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "action on Hesse planes done" << endl;
	}

	if (f_has_nice_gens) {
		Orbits_on_Hesse_planes = nice_gens->compute_all_point_orbits_schreier(
				A_on_Hesse_planes, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_Hesse_planes = Aut_gens->compute_all_point_orbits_schreier(
				A_on_Hesse_planes, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_Hesse_planes->Forest->nb_orbits
				<< " orbits on the set of " << SO->SOP->nb_Hesse_planes
				<< " Hesse planes" << endl;
	}

	if (f_vv) {
		Orbits_on_Hesse_planes->Forest->print_and_list_orbits(cout);
	}

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_Hesse_planes done" << endl;
	}
}

void surface_object_with_group::init_orbits_on_axes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_axes" << endl;
	}

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set = "_axes";
	label_of_set_tex = "\\_axes";

	if (f_v) {
		cout << "creating action on axes:" << endl;
		cout << "SO->SOP->nb_axes = "
				<< SO->SOP->nb_axes << endl;
		cout << "Axes_line_rank:";
		Lint_vec_print(cout, SO->SOP->Axes_line_rank, SO->SOP->nb_axes);
		cout << endl;
	}
	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_axes "
				"before Surf_A->A2->restricted_action" << endl;
	}
	A_on_axes = Surf_A->A2->Induced_action->restricted_action(
			SO->SOP->Axes_line_rank, SO->SOP->nb_axes,
			label_of_set, label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_axes "
				"after Surf_A->A2->restricted_action" << endl;
	}

	if (f_has_nice_gens) {
		Orbits_on_axes = nice_gens->compute_all_point_orbits_schreier(
				A_on_axes, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_axes = Aut_gens->compute_all_point_orbits_schreier(
				A_on_axes, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_axes->Forest->nb_orbits
				<< " orbits on the set of " << SO->SOP->nb_axes
				<< " axes" << endl;
	}

	if (f_vv) {
		Orbits_on_axes->Forest->print_and_list_orbits(cout);
	}

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_axes done" << endl;
	}
}


void surface_object_with_group::init_orbits_on_trihedral_pairs(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "creating action on trihedral pairs:" << endl;
	}
	A_on_trihedral_pairs =
			A_on_tritangent_planes->Induced_action->create_induced_action_on_sets(
					120, 6,
					Surf->Schlaefli->Schlaefli_trihedral_pairs->Axes,
					0 /*verbose_level*/);
	if (f_v) {
		cout << "action on trihedral pairs created" << endl;
	}

	if (f_has_nice_gens) {
		Orbits_on_trihedral_pairs = nice_gens->compute_all_point_orbits_schreier(
				A_on_trihedral_pairs, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_trihedral_pairs = Aut_gens->compute_all_point_orbits_schreier(
				A_on_trihedral_pairs, print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_trihedral_pairs->Forest->nb_orbits
				<< " orbits on trihedral pairs" << endl;
	}

	if (f_vv) {
		Orbits_on_trihedral_pairs->Forest->print_and_list_orbits(cout);
	}

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_trihedral_pairs done" << endl;
	}
}

void surface_object_with_group::init_orbits_on_points_not_on_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_points_not_on_lines" << endl;
	}


	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set = "_pts_not_on_lines";
	label_of_set_tex = "\\_pts\\_not\\_on\\_lines";

	if (f_v) {
		cout << "creating action on points not on lines:" << endl;
	}
	A_on_pts_not_on_lines = Surf_A->A->Induced_action->restricted_action(
			SO->SOP->Pts_not_on_lines, SO->SOP->nb_pts_not_on_lines,
			label_of_set, label_of_set_tex,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on points not on lines done" << endl;
	}

	if (f_has_nice_gens) {
		Orbits_on_points_not_on_lines =
				nice_gens->compute_all_point_orbits_schreier(
						A_on_pts_not_on_lines, print_interval, 0 /*verbose_level*/);
	}
	else {
		Orbits_on_points_not_on_lines =
				Aut_gens->compute_all_point_orbits_schreier(
						A_on_pts_not_on_lines,  print_interval, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "We found " << Orbits_on_points_not_on_lines->Forest->nb_orbits
				<< " orbits on points not on lines" << endl;
	}

	if (f_vv) {
		Orbits_on_points_not_on_lines->Forest->print_and_list_orbits(cout);
	}

	if (f_v) {
		cout << "surface_object_with_group::init_orbits_on_points_not_on_lines done" << endl;
	}
}


void surface_object_with_group::print_generators_on_lines(
		ostream &ost,
		groups::strong_generators *Aut_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::print_generators_on_lines" << endl;
	}
	//Aut_gens->print_generators_tex(ost);
	Aut_gens->print_generators_tex_with_point_labels(
			A_on_the_lines,
			ost,
			Surf->Schlaefli->Labels->Line_label_tex,
			//algebraic_geometry::callback_surface_domain_sstr_line_label,
			Surf);

}

void surface_object_with_group::print_elements_on_lines(
		std::ostream &ost,
		groups::strong_generators *Aut_gens,
		int max_nb_elements_printed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::print_elements_on_lines" << endl;
	}
	//Aut_gens->print_generators_tex(ost);
	Aut_gens->print_elements_latex_ost_with_point_labels(
			A_on_the_lines,
			ost,
			Surf->Schlaefli->Labels->Line_label_tex,
			max_nb_elements_printed,
			Surf);

}

void surface_object_with_group::print_elements_on_tritangent_planes(
		std::ostream &ost,
		groups::strong_generators *Aut_gens,
		int max_nb_elements_printed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::print_elements_on_tritangent_planes" << endl;
	}
	//Aut_gens->print_generators_tex(ost);
	Aut_gens->print_elements_latex_ost_with_point_labels(
			A_on_tritangent_planes,
			ost,
			Surf->Schlaefli->Labels->Tritangent_plane_label_tex,
			max_nb_elements_printed,
			Surf);

}



void surface_object_with_group::print_automorphism_group(
	std::ostream &ost,
	int verbose_level)
// called from surface_object_with_group::cheat_sheet
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::print_automorphism_group" << endl;
	}


	algebra::ring_theory::longinteger_object go;
	other::l1_interfaces::latex_interface L;

	Aut_gens->group_order(go);
	
	ost << "\\section*{Orbits of the automorphism group}" << endl;
	ost << "surface\\_object\\_with\\_group::print\\_automorphism\\_group" << endl;
	ost << "The automorphism group has order " << go << endl;
	ost << "\\bigskip" << endl;
	ost << "\\subsection*{Orbits on points}" << endl;
	//Orbits_on_points->print_and_list_orbits_and_
	//stabilizer_sorted_by_length(ost, true, Surf_A->A, go);
	Orbits_on_points->print_and_list_orbits_with_original_labels_tex(ost);
	Orbits_on_points->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost, Surf_A->A, Aut_gens,
			verbose_level);


	if (f_v) {
		cout << "surface_object_with_group::print_automorphism_group done" << endl;
	}
}



void surface_object_with_group::print_orbits_of_automorphism_group(
	std::ostream &ost,
	int f_print_orbits, std::string &fname_mask,
	other::graphics::layered_graph_draw_options *Opt,
	int max_nb_elements_printed,
	int verbose_level)
// called from surface_object_with_group::cheat_sheet
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::print_orbits_of_automorphism_group" << endl;
	}


	algebra::ring_theory::longinteger_object go;
	other::l1_interfaces::latex_interface L;

	ost << "\\subsection*{Orbits on Eckardt points}" << endl;
	Orbits_on_Eckardt_points->print_and_list_orbits_with_original_labels_tex(ost);
	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask = fname_mask + "_Eckardt_points";

		Orbits_on_Eckardt_points->Forest->make_orbit_trees(ost,
				my_fname_mask, Opt,
				verbose_level);
	}

	Orbits_on_Eckardt_points->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost, Surf_A->A, Aut_gens,
			verbose_level);


	ost << "\\subsection*{Orbits on Double points}" << endl;
	Orbits_on_Double_points->print_and_list_orbits_with_original_labels_tex(ost);

	ost << "\\subsection*{Orbits on Single points}" << endl;
	Orbits_on_Single_points->print_and_list_orbits_with_original_labels_tex(ost);

	ost << "\\subsection*{Orbits on points not on lines}" << endl;
	//Orbits_on_points_not_on_lines->print_and_list_orbits_sorted_by_length_tex(ost);
	Orbits_on_points_not_on_lines->print_and_list_orbits_with_original_labels_tex(ost);

	print_full_del_Pezzo(ost, verbose_level);


	ost << "\\subsection*{Orbits on lines}" << endl;
	Orbits_on_lines->Forest->print_and_list_orbits_tex(ost);
	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask = fname_mask + "_on_lines";

		Orbits_on_lines->Forest->make_orbit_trees(ost,
				my_fname_mask, Opt,
				verbose_level);
	}


	ost << "\\bigskip" << endl;

	Surf->Schlaefli->latex_table_of_Schlaefli_labeling_of_lines(ost);

	ost << "\\bigskip" << endl;

#if 0
	Orbits_on_lines->print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
		int i, action *default_action,
		strong_generators *gens, std::ostream &ost);
#endif

	Orbits_on_lines->print_and_list_orbits_with_original_labels_tex(ost);
	Orbits_on_lines->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost, Surf_A->A, Aut_gens,
			verbose_level);
	int orbit_idx;
	for (orbit_idx = 0; orbit_idx < Orbits_on_lines->Forest->nb_orbits; orbit_idx++) {

		groups::strong_generators *gens_stab;

		Orbits_on_lines->get_stabilizer_orbit_rep(
			orbit_idx, Surf_A->A,
			Aut_gens,
			gens_stab);

		ost << "The stabilizer of orbit " << orbit_idx << " in the action on the lines:\\\\" << endl;
		print_elements_on_lines(
				ost,
				gens_stab,
				max_nb_elements_printed,
				verbose_level);

		print_elements_on_tritangent_planes(
				ost,
				gens_stab,
				max_nb_elements_printed,
				verbose_level);

		FREE_OBJECT(gens_stab);
	}

	int *Decomp_scheme;
	int nb;
	int block_width = 10;
	nb = Orbits_on_lines->Forest->nb_orbits;

	Orbits_on_lines->Forest->get_orbit_decomposition_scheme_of_graph(
			SO->SOP->Adj_line_intersection_graph,
			SO->Variety_object->Line_sets->Set_size[0],
			Decomp_scheme,
			0 /*verbose_level*/);

	ost << "\\subsection*{Decomposition scheme of line intersection graph}" << endl;
	ost << "Decomposition scheme of line intersection graph:" << endl;
	L.print_integer_matrix_tex_block_by_block(ost,
			Decomp_scheme, nb, nb, block_width);
	FREE_int(Decomp_scheme);
	

	if (SO->Variety_object->Line_sets->Set_size[0] == 27) {

		if (f_v) {
			cout << "surface_object_with_group::print_orbits_of_automorphism_group "
					"before print_orbits_on_schlaefli_related_things" << endl;
		}
		print_orbits_on_schlaefli_related_things(ost,
				f_print_orbits, fname_mask,
				Opt,
				max_nb_elements_printed,
				verbose_level);
		if (f_v) {
			cout << "surface_object_with_group::print_orbits_of_automorphism_group "
					"after print_orbits_on_schlaefli_related_things" << endl;
		}
	}


	ost << "\\clearpage" << endl;

	if (f_v) {
		cout << "surface_object_with_group::print_orbits_of_automorphism_group done" << endl;
	}
}

void surface_object_with_group::print_orbits_on_schlaefli_related_things(
	std::ostream &ost,
	int f_print_orbits, std::string &fname_mask,
	other::graphics::layered_graph_draw_options *Opt,
	int max_nb_elements_printed,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::print_orbits_on_schlaefli_related_things" << endl;
	}

	ost << "\\subsection*{Orbits on single sixes}" << endl;
	Orbits_on_single_sixes->Forest->print_and_list_orbits_tex(ost);

	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask = fname_mask + "_single_sixes";

		Orbits_on_single_sixes->Forest->make_orbit_trees(
				ost,
				my_fname_mask, Opt,
				verbose_level);
	}

	ost << "\\subsection*{Orbits on double sixes}" << endl;
	Orbits_on_double_sixes->Forest->print_and_list_orbits_tex(ost);
	Orbits_on_double_sixes->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost, Surf_A->A, Aut_gens,
			verbose_level);

	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask = fname_mask + "_double_sixes";

		Orbits_on_double_sixes->Forest->make_orbit_trees(
				ost,
				my_fname_mask, Opt,
				verbose_level);
	}

	int orbit_idx;

	for (orbit_idx = 0; orbit_idx < Orbits_on_lines->Forest->nb_orbits; orbit_idx++) {

		groups::strong_generators *gens_stab;

		Orbits_on_double_sixes->get_stabilizer_orbit_rep(
			orbit_idx, Surf_A->A,
			Aut_gens,
			gens_stab);

		ost << "The stabilizer of orbit " << orbit_idx << " in the action on double sixes:\\\\" << endl;
		print_elements_on_lines(
				ost,
				gens_stab,
				max_nb_elements_printed,
				verbose_level);

		print_elements_on_tritangent_planes(
				ost,
				gens_stab,
				max_nb_elements_printed,
				verbose_level);

		FREE_OBJECT(gens_stab);
	}

	ost << "\\subsection*{Orbits on tritangent planes}" << endl;
	Orbits_on_tritangent_planes->Forest->print_and_list_orbits_tex(ost);
	Orbits_on_tritangent_planes->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost, Surf_A->A, Aut_gens,
			verbose_level);
	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask = fname_mask + "_tritangent_planes";

		Orbits_on_tritangent_planes->Forest->make_orbit_trees(
				ost,
				my_fname_mask, Opt,
				verbose_level);
	}

	ost << "\\subsection*{Orbits on Hesse planes}" << endl;
	Orbits_on_Hesse_planes->Forest->print_and_list_orbits_tex(ost);
	Orbits_on_Hesse_planes->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost, Surf_A->A, Aut_gens,
			verbose_level);
	if (f_print_orbits) {

		string my_fname_mask;

		my_fname_mask = fname_mask + "_Hesse_planes";

		Orbits_on_Hesse_planes->Forest->make_orbit_trees(
				ost,
				my_fname_mask, Opt,
				verbose_level);
	}
	Orbits_on_Hesse_planes->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			ost, Surf_A->A, Aut_gens,
			verbose_level);

	ost << "\\subsection*{Orbits on trihedral pairs}" << endl;
	Orbits_on_trihedral_pairs->Forest->print_and_list_orbits_tex(ost);

	if (f_v) {
		cout << "surface_object_with_group::print_orbits_on_schlaefli_related_things done" << endl;
	}

}

void surface_object_with_group::cheat_sheet_basic(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;
	other::l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet_basic" << endl;
	}


	algebra::ring_theory::longinteger_object ago;
	Aut_gens->group_order(ago);
	ost << "The automorphism group has order "
			<< ago << "\\\\" << endl;
	ost << "The automorphism group is generated by:\\\\" << endl;
	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet_basic "
				"before Aut_gens->"
				"print_generators_tex" << endl;
	}
	Aut_gens->print_generators_tex(ost);


	if (f_has_nice_gens) {
		ost << "The stabilizer is generated by the following nice generators:\\\\" << endl;
		nice_gens->print_tex(ost);

	}

	ost << "Orbits on Eckardt points:\\\\" << endl;
	Orbits_on_Eckardt_points->print_and_list_orbits_sorted_by_length_tex(ost);

	ost << "\\bigskip" << endl;

	if (SO->Variety_object->Line_sets->Set_size[0] == 27) {
		ost << "Orbits on half double-sixes:\\\\" << endl;
		int i, idx;

		for (i = 0; i < Orbits_on_single_sixes->Forest->nb_orbits; i++) {

			//ost << "\\bigskip" << endl;
			//ost << "" << endl;
			ost << "Orbit " << i << " / " << Orbits_on_single_sixes->Forest->nb_orbits
					<< " of length " << Orbits_on_single_sixes->Forest->orbit_len[i]
					<< " consists of the following half double sixes:" << endl;


			ost << "$$" << endl;
			L.int_set_print_tex(ost,
				Orbits_on_single_sixes->Forest->orbit +
					Orbits_on_single_sixes->Forest->orbit_first[i],
				Orbits_on_single_sixes->Forest->orbit_len[i]);
			ost << "$$" << endl;

			idx = Orbits_on_single_sixes->Forest->orbit[Orbits_on_single_sixes->Forest->orbit_first[i]];

			ost << "orbit rep:" << endl;
			ost << "$$" << endl;
			Surf->Schlaefli->Schlaefli_double_six->latex_half_double_six(ost, idx);
			ost << "$$" << endl;

		}
	}

	ost << "\\bigskip" << endl;

	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet_basic done" << endl;
	}
}


void surface_object_with_group::cheat_sheet_group_elements(
		std::ostream &ost,
		std::string &fname_csv,
		std::string &col_heading,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet_group_elements" << endl;
		cout << "surface_object_with_group::cheat_sheet_group_elements "
				"verbose_level = " << verbose_level << endl;
	}

	other::orbiter_kernel_system::file_io Fio;

	other::data_structures::string_tools ST;

	other::data_structures::spreadsheet S;

	S.read_spreadsheet(fname_csv, 0 /*verbose_level*/);

	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet_group_elements "
				"S.nb_rows = " << S.nb_rows << endl;
		cout << "surface_object_with_group::cheat_sheet_group_elements "
				"S.nb_cols = " << S.nb_cols << endl;
	}


	int idx_input;
	int row;

	int *Data;
	int *data;
	int sz;
	int N;


	N = S.nb_rows - 1;
	Data = NEW_int(N * Surf_A->A->make_element_size);

	idx_input = S.find_column(col_heading);

	for (row = 0; row < N; row++) {

		string s_data;
		string s_data2;


		s_data = S.get_entry_ij(row + 1, idx_input);

		ST.drop_quotes(s_data, s_data2);

		Int_vec_scan(s_data2, data, sz);
		cout << "read:" << s_data2 << endl;
		if (sz != Surf_A->A->make_element_size) {
			cout << "data size mismatch" << endl;
			cout << "sz = " << sz << endl;
			cout << "Surf_A->A->make_element_size = " << Surf_A->A->make_element_size << endl;
			exit(1);
		}
		Int_vec_copy(data,
				Data + row * Surf_A->A->make_element_size,
				Surf_A->A->make_element_size);
	}

	int *Elt;
	int i;

	Elt = NEW_int(Surf_A->A->elt_size_in_int);

	int nb_col = 13;
	int nb;
	string *Table;

	nb = S.nb_cols + nb_col;
	Table = new string[N * nb];

	for (i = 0; i < N; i++) {

		int j;


		for (j = 0; j < S.nb_cols; j++) {

			Table[i * nb + j] = S.get_entry_ij(i + 1, j);
		}
		cout << "Element " << i << " / " << N << " is:" << endl;
		ost << "$" << endl;

		Surf_A->A->Group_element->make_element(
				Elt, Data + i * Surf_A->A->make_element_size,
				0 /*verbose_level*/);

		Surf_A->A->Group_element->element_print_latex(Elt, ost);

#if 0
		actions::action *A_on_points;
		actions::action *A_on_Eckardt_points;
		actions::action *A_on_Double_points;
		actions::action *A_on_Single_points;
		actions::action *A_on_the_lines;
		actions::action *A_single_sixes;
		actions::action *A_double_sixes;
		actions::action *A_on_tritangent_planes;
		actions::action *A_on_Hesse_planes;
		actions::action *A_on_axes;
		actions::action *A_on_trihedral_pairs;
		actions::action *A_on_pts_not_on_lines;
#endif

#if 0
		A_on_points->Group_element->element_order_and_cycle_type_verbose(
				void *elt, int *cycle_type, int verbose_level)
		// cycle_type[i - 1] is the number of cycles of length i for 1 le i le n
#endif

		ost << "$" << endl;
		ost << "\\\\" << endl;
		ost << "\\bigskip" << endl;

		data_structures_groups::vector_ge *gens;

		gens = NEW_OBJECT(data_structures_groups::vector_ge);

		gens->init_single(Surf_A->A, Elt, verbose_level);


		groups::schreier **Orbits;

		Orbits = (groups::schreier **) NEW_pvoid(nb_col);

		// ToDo:
		if (f_v) {
			cout << "surface_object_with_group::cheat_sheet_group_elements "
					"before compute_all_point_orbits_schreier" << endl;
		}
		Orbits[0] = gens->compute_all_point_orbits_schreier(Surf_A->A, print_interval, verbose_level - 2);
		Orbits[1] = gens->compute_all_point_orbits_schreier(A_on_points, print_interval, verbose_level - 2);
		Orbits[2] = gens->compute_all_point_orbits_schreier(A_on_Eckardt_points, print_interval, verbose_level - 2);
		Orbits[3] = gens->compute_all_point_orbits_schreier(A_on_Double_points, print_interval, verbose_level - 2);
		Orbits[4] = gens->compute_all_point_orbits_schreier(A_on_Single_points, print_interval, verbose_level - 2);
		Orbits[5] = gens->compute_all_point_orbits_schreier(A_on_the_lines, print_interval, verbose_level - 2);
		Orbits[6] = gens->compute_all_point_orbits_schreier(A_single_sixes, print_interval, verbose_level - 2);
		Orbits[7] = gens->compute_all_point_orbits_schreier(A_double_sixes, print_interval, verbose_level - 2);
		Orbits[8] = gens->compute_all_point_orbits_schreier(A_on_tritangent_planes, print_interval, verbose_level - 2);
		Orbits[9] = gens->compute_all_point_orbits_schreier(A_on_Hesse_planes, print_interval, verbose_level - 2);
		Orbits[10] = gens->compute_all_point_orbits_schreier(A_on_axes, print_interval, verbose_level - 2);
		Orbits[11] = gens->compute_all_point_orbits_schreier(A_on_trihedral_pairs, print_interval, verbose_level - 2);
		Orbits[12] = gens->compute_all_point_orbits_schreier(A_on_pts_not_on_lines, print_interval, verbose_level - 2);

		for (j = 0; j < nb_col; j++) {

			string s;
			other::data_structures::tally *Classify_orbits_by_length;

			Classify_orbits_by_length = NEW_OBJECT(other::data_structures::tally);

			Classify_orbits_by_length->init(Orbits[j]->Forest->orbit_len, Orbits[j]->Forest->nb_orbits, false, 0);

			cout << "j=" << j << " : ";
			s = Classify_orbits_by_length->stringify_bare_tex(false);
			cout << s << endl;

			Table[i * nb + S.nb_cols + j] = "\"" + s + "\"";


			FREE_OBJECT(Classify_orbits_by_length);
		}

		for (j = 0; j < nb_col; j++) {
			FREE_OBJECT(Orbits[j]);
		}

		FREE_pvoid((void **) Orbits);

		if (f_v) {
			cout << "surface_object_with_group::cheat_sheet_group_elements "
					"after compute_all_point_orbits_schreier" << endl;
		}



	}

	string fname;
	string *Headings;
	string headings;

	Headings = new string[nb];


	int j;

	for (j = 0; j < S.nb_cols; j++) {

		Headings[j] = S.get_entry_ij(0, j);
	}
	for (j = 0; j < nb_col; j++) {
		Headings[S.nb_cols + j] = "ORB" + std::to_string(j);
	}

	for (j = 0; j < nb; j++) {
		headings += Headings[j];
		if (j < nb - 1) {
			headings += ",";
		}
	}

	fname = fname_csv;
	ST.chop_off_extension(fname);
	fname += "_properties.csv";

	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet_group_elements "
				"before Fio.Csv_file_support->write_table_of_strings" << endl;
	}
	Fio.Csv_file_support->write_table_of_strings(
			fname,
			N, nb, Table,
			headings,
			verbose_level);

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::export_csv "
				"after Fio.Csv_file_support->write_table_of_strings" << endl;
	}

	delete [] Headings;
	delete [] Table;

	FREE_int(Elt);
	FREE_int(Data);

	if (f_v) {
		cout << "surface_object_with_group::cheat_sheet_group_elements "
				"done" << verbose_level << endl;
	}
}



void surface_object_with_group::print_automorphism_group_generators(
		std::ostream &ost,
		int max_nb_elements_printed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "surface_object_with_group::print_automorphism_group_generators" << endl;
	}

	if (Aut_gens == NULL) {
		if (f_v) {
			cout << "surface_object_with_action::print_automorphism_group_generators "
					"the automorphism group is not available" << endl;
		}
		return;
	}
	ost << "The automorphism group is generated by:\\\\" << endl;
	if (f_v) {
		cout << "surface_object_with_group::print_automorphism_group_generators "
				"before Aut_gens->print_generators_tex" << endl;
	}
	Aut_gens->print_generators_tex(ost);

	if (f_v) {
		cout << "surface_object_with_group::print_automorphism_group_generators "
				"before Aut_gens->print_generators_in_different_action_tex" << endl;
		A_on_the_lines->print_info();
	}
	Aut_gens->print_generators_in_different_action_tex(
			ost, A_on_the_lines);


	if (f_has_nice_gens) {
		ost << "The stabilizer is generated by the following nice generators:\\\\" << endl;
		nice_gens->print_tex(ost);

	}


	if (projectivity_group_gens) {

		if (f_v) {
			cout << "surface_object_with_group::print_automorphism_group_generators "
					"before print_projectivity_group" << endl;
		}

		print_projectivity_group(ost, max_nb_elements_printed, verbose_level - 2);

		if (f_v) {
			cout << "surface_object_with_group::print_automorphism_group_generators "
					"after print_projectivity_group" << endl;
		}

		if (Syl) {

			if (f_v) {
				cout << "surface_object_with_group::print_automorphism_group_generators "
						"before print_sylow_groups_of_projectivity_group" << endl;
			}

			print_sylow_groups_of_projectivity_group(ost, verbose_level - 2);

			if (f_v) {
				cout << "surface_object_with_group::print_automorphism_group_generators "
						"after print_sylow_groups_of_projectivity_group" << endl;
			}
		}

	}



}

void surface_object_with_group::print_projectivity_group(
		std::ostream &ost,
		int max_nb_elements_printed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_object_with_group::print_projectivity_group" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_projectivity_group "
				"projectivity stabilizer" << endl;
	}
	algebra::ring_theory::longinteger_object go;
	projectivity_group_gens->group_order(go);
	ost << "The projectivity group has order "
			<< go << "\\\\" << endl;
	ost << "The projectivity group is generated by:\\\\" << endl;
	if (f_v) {
		cout << "surface_object_with_group::print_projectivity_group "
				"before projectivity_group_gens->"
				"print_generators_tex" << endl;
	}
	projectivity_group_gens->print_generators_tex(ost);
	projectivity_group_gens->print_generators_in_different_action_tex(ost, A_on_the_lines);


	ost << "The projectivity group in the action on the lines:\\\\" << endl;
	print_generators_on_lines(
			ost,
			projectivity_group_gens,
			verbose_level);

#if 1
	ost << "The elements of the projectivity group "
			"in the action on the lines:\\\\" << endl;
	print_elements_on_lines(
			ost,
			projectivity_group_gens,
			max_nb_elements_printed,
			verbose_level);
#endif



	string label_group;

	label_group = SO->label_txt + "_proj_grp";
	projectivity_group_gens->export_group_and_copy_to_latex(label_group,
			ost,
			projectivity_group_gens->A,
			verbose_level - 2);

	label_group = SO->label_txt + "_proj_grp_on_lines";
	projectivity_group_gens->export_group_and_copy_to_latex(label_group,
			ost,
			A_on_the_lines,
			verbose_level - 2);

	label_group = SO->label_txt + "_proj_grp_on_tritangent_planes";
	projectivity_group_gens->export_group_and_copy_to_latex(label_group,
			ost,
			A_on_tritangent_planes,
			verbose_level - 2);



	if (f_v) {
		cout << "surface_object_with_group::print_projectivity_group done" << endl;
	}

}


void surface_object_with_group::print_sylow_groups_of_projectivity_group(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_object_with_group::print_sylow_groups_of_projectivity_group" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_sylow_groups_of_projectivity_group "
				"Sylow subgroups" << endl;
	}
	int idx;

	for (idx = 0; idx < Syl->nb_primes; idx++) {
		if (f_v) {
			cout << "surface_object_with_group::print_sylow_groups_of_projectivity_group "
					"idx=" << idx << " / " << Syl->nb_primes << endl;
		}
		ost << "The " << Syl->primes[idx]
			<< "-Sylow subgroup is generated by:\\\\" << endl;
		Syl->Sub[idx].SG->print_generators_tex(ost);


		if (f_v) {
			cout << "surface_object_with_group::print_sylow_groups_of_projectivity_group "
					"idx=" << idx << " / " << Syl->nb_primes
					<< " making label_group" << endl;
		}

		string label_group;


		label_group = SO->label_txt + "_proj_grp_syl_" + std::to_string(Syl->primes[idx]);

		if (f_v) {
			cout << "surface_object_with_group::print_sylow_groups_of_projectivity_group "
					"idx=" << idx << " / " << Syl->nb_primes
					<< " label_group=" << label_group << endl;
		}

		if (f_v) {
			cout << "surface_object_with_group::print_sylow_groups_of_projectivity_group "
					"idx=" << idx << " / " << Syl->nb_primes
					<< " before export_group_and_copy_to_latex" << endl;
		}

		Syl->Sub[idx].SG->export_group_and_copy_to_latex(
				label_group,
				ost,
				projectivity_group_gens->A,
				verbose_level - 2);

		label_group = SO->label_txt + "_proj_grp_syl_" + std::to_string(Syl->primes[idx]) + "_on_lines";

		if (f_v) {
			cout << "surface_object_with_group::print_sylow_groups_of_projectivity_group "
					"idx=" << idx << " / " << Syl->nb_primes
					<< " label_group=" << label_group << endl;
		}

		if (f_v) {
			cout << "surface_object_with_group::print_sylow_groups_of_projectivity_group "
					"idx=" << idx << " / " << Syl->nb_primes
					<< " before export_group_and_copy_to_latex" << endl;
		}
		Syl->Sub[idx].SG->export_group_and_copy_to_latex(
				label_group,
				ost,
				A_on_the_lines,
				verbose_level - 2);

	}

	if (f_v) {
		cout << "surface_object_with_group::print_sylow_groups_of_projectivity_group done" << endl;
	}
}


void surface_object_with_group::investigate_surface_and_write_report(
		other::graphics::layered_graph_draw_options *Opt,
		actions::action *A,
		surface_create *SC,
		cubic_surfaces_and_arcs::six_arcs_not_on_a_conic *Six_arcs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_object_with_group::investigate_surface_and_write_report" << endl;
	}

	string fname;
	string fname_mask;


	fname = "surface_" + SC->SO->label_txt + "_with_group.tex";


	fname_mask = "surface_" + SC->SO->label_txt + "_orbit_%d";

	{
		ofstream fp(fname);
		other::l1_interfaces::latex_interface L;

		L.head_easy(fp);

		investigate_surface_and_write_report2(
					fp,
					Opt,
					A,
					SC,
					Six_arcs,
					fname_mask,
					verbose_level);


		L.foot(fp);
	}
	other::orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


}




void surface_object_with_group::investigate_surface_and_write_report2(
		std::ostream &ost,
		other::graphics::layered_graph_draw_options *Opt,
		actions::action *A,
		surface_create *SC,
		cubic_surfaces_and_arcs::six_arcs_not_on_a_conic *Six_arcs,
		std::string &fname_mask,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::investigate_surface_and_write_report2" << endl;
	}


	if (f_v) {
		cout << "surface_object_with_group::investigate_surface_and_write_report2 "
				"before cheat_sheet" << endl;
	}
	ost << "\\section{The Cubic Surface " << SC->SO->label_tex
			<< " over $\\mathbb F_{" << SC->F->q << "}$}" << endl;


	int max_nb_elements_printed = 15;

	cheat_sheet(
			ost,
		true /* f_print_orbits */,
		fname_mask,
		Opt,
		max_nb_elements_printed,
		verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::investigate_surface_and_write_report2 "
				"after cheat_sheet" << endl;
	}



	ost << "\\bigskip" << endl;

	ost << "\\section{The Finite Field $\\mathbb F_{" << SC->F->q << "}$}" << endl;
	SC->F->Io->cheat_sheet(ost, verbose_level);





	ost << "\\setlength{\\parindent}{0pt}" << endl;

#if 0
	if (f_surface_clebsch) {

		if (f_v) {
			cout << "surface_object_with_group::investigate_surface_and_write_report2 f_surface_clebsch" << endl;
		}

		//surface_object *SO;
		//SO = SoA->SO;

		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;
		ost << "\\section{Points on the surface}" << endl;
		ost << endl;

		SO->SOP->print_affine_points_in_source_code(ost);


		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;

		ost << "\\section{Clebsch maps}" << endl;

		SC->Surf->Schlaefli->latex_table_of_clebsch_maps(ost);


		ost << endl;
		ost << "\\clearpage" << endl;
		ost << endl;



		ost << "\\section{Six-arcs not on a conic}" << endl;
		ost << endl;


		//ost << "The six-arcs not on a conic are:\\\\" << endl;
		Six_arcs->report_latex(ost);




	}
	else {
		if (f_v) {
			cout << "surface_object_with_group::investigate_surface_and_write_report2 !f_surface_clebsch" << endl;
		}

	}



	if (f_surface_quartic) {

		if (f_v) {
			cout << "surface_object_with_group::investigate_surface_and_write_report2 f_surface_quartic" << endl;
		}

		{
			ofstream ost_quartics("quartics.txt");



			all_quartic_curves(ost, ost_quartics, verbose_level);
		}

	}
	else {
		if (f_v) {
			cout << "surface_object_with_group::investigate_surface_and_write_report2 !f_surface_quartic" << endl;
		}


	}




	if (f_surface_codes) {

		if (f_v) {
			cout << "surface_object_with_group::investigate_surface_and_write_report2 f_surface_codes" << endl;
		}

		homogeneous_polynomial_domain *HPD;

		HPD = NEW_OBJECT(homogeneous_polynomial_domain);

		HPD->init(SC->F, 3, 2 /* degree */,
				true /* f_init_incidence_structure */,
				t_PART,
				verbose_level);

		action *A_on_poly;

		A_on_poly = NEW_OBJECT(action);
		A_on_poly->induced_action_on_homogeneous_polynomials(A,
			HPD,
			false /* f_induce_action */, NULL,
			verbose_level);

		cout << "created action A_on_poly" << endl;
		A_on_poly->print_info();

		schreier *Sch;
		longinteger_object full_go;

		//Sch = new schreier;
		//A2->all_point_orbits(*Sch, verbose_level);

		cout << "computing orbits:" << endl;

		Sch = A->Strong_gens->orbits_on_points_schreier(A_on_poly, verbose_level);

		//SC->Sg->
		//Sch = SC->Sg->orbits_on_points_schreier(A_on_poly, verbose_level);

		orbit_transversal *T;

		A->group_order(full_go);
		T = NEW_OBJECT(orbit_transversal);

		cout << "before T->init_from_schreier" << endl;

		T->init_from_schreier(
				Sch,
				A,
				full_go,
				verbose_level);

		cout << "after T->init_from_schreier" << endl;

		Sch->print_orbit_reps(cout);

		cout << "orbit reps:" << endl;

		ost << "\\section{Orbits on conics}" << endl;
		ost << endl;

		T->print_table_latex(
				ost,
				true /* f_has_callback */,
				HPD_callback_print_function2,
				HPD /* callback_data */,
				true /* f_has_callback */,
				HPD_callback_print_function,
				HPD /* callback_data */,
				verbose_level);


	}
	else {
		if (f_v) {
			cout << "surface_object_with_group::investigate_surface_and_write_report2 !f_surface_codes" << endl;
		}


	}
#endif



	if (f_v) {
		cout << "surface_object_with_group::investigate_surface_and_write_report2 done" << endl;
	}
}

void surface_object_with_group::report_all_flag_orbits(
		std::string &surface_label_txt,
		std::string &surface_label_tex,
		std::ostream &ost,
		orbits::orbits_create *Classification_of_arcs,
		int verbose_level)
// reports on all quartic curves associated
// with the orbits on points not on lines
{
	int f_v = (verbose_level >= 1);

	int f_TDO = false;

	if (f_v) {
		cout << "surface_object_with_group::report_all_flag_orbits "
				"surface_label_txt=" << surface_label_txt << endl;
	}
	int pt_orbit;


	ost << "Orbits on points not on lines nb orbits = "
			<< Orbits_on_points_not_on_lines->Forest->nb_orbits << "\\\\" << endl;

	for (pt_orbit = 0; pt_orbit < Orbits_on_points_not_on_lines->Forest->nb_orbits; pt_orbit++) {

		ost << "\\section{Quartic curve associated with orbit " << pt_orbit
				<< " / " << Orbits_on_points_not_on_lines->Forest->nb_orbits << "}" << endl;


		quartic_curves::quartic_curve_from_surface *QC;

		QC = NEW_OBJECT(quartic_curves::quartic_curve_from_surface);

		QC->init(this, verbose_level);

		QC->init_labels(surface_label_txt, surface_label_tex, verbose_level);


		if (f_v) {
			cout << "surface_object_with_group::report_all_flag_orbits "
					"before QC->create_quartic_curve" << endl;
		}
		QC->create_quartic_curve(Classification_of_arcs, pt_orbit, verbose_level);
		if (f_v) {
			cout << "surface_object_with_group::report_all_flag_orbits "
					"after QC->create_quartic_curve" << endl;
		}

		// the quartic curve is now in QC->curve
		// as a Surf->Poly4_x123

		if (f_v) {
			cout << "surface_object_with_group::report_all_flag_orbits "
					"before QC->compute_stabilizer_with_nauty" << endl;
		}
		QC->compute_stabilizer_with_nauty(verbose_level);
		if (f_v) {
			cout << "surface_object_with_group::report_all_flag_orbits "
					"after QC->compute_stabilizer_with_nauty" << endl;
		}


		if (f_v) {
			cout << "surface_object_with_group::report_all_flag_orbits "
					"before QC->cheat_sheet_quartic_curve" << endl;
		}
		QC->cheat_sheet_quartic_curve(ost, f_TDO, verbose_level);
		if (f_v) {
			cout << "surface_object_with_group::report_all_flag_orbits "
					"after QC->cheat_sheet_quartic_curve" << endl;
		}

		FREE_OBJECT(QC);
	}
	if (f_v) {
		cout << "surface_object_with_group::report_all_flag_orbits done" << endl;
	}
}

void surface_object_with_group::export_all_quartic_curves(
		std::string &fname_curves,
		orbits::orbits_create *Classification_of_arcs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::export_all_quartic_curves" << endl;
	}

	std::string headings;
	std::string *Table;
	int nb_rows, nb_cols;

	if (f_v) {
		cout << "surface_create::export_all_quartic_curves "
				"before prepare_data" << endl;
	}

	prepare_data(
			Classification_of_arcs,
			headings,
			Table,
			nb_rows, nb_cols,
			verbose_level - 2);

	if (f_v) {
		cout << "surface_create::export_all_quartic_curves "
				"after prepare_data" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;


	Fio.Csv_file_support->write_table_of_strings(
			fname_curves,
			nb_rows, nb_cols, Table,
			headings,
			verbose_level - 2);

	delete [] Table;



	if (f_v) {
		cout << "Written file " << fname_curves << " of size "
			<< Fio.file_size(fname_curves) << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::export_all_quartic_curves done" << endl;
	}
}

void surface_object_with_group::prepare_data(
		orbits::orbits_create *Classification_of_arcs,
		std::string &headings,
		std::string *&Table,
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "surface_object_with_group::prepare_data" << endl;
		cout << "surface_object_with_group::prepare_data verbose_level=" << verbose_level << endl;
		cout << "surface_object_with_group::prepare_data "
				"nb_orbits = " << Orbits_on_points_not_on_lines->Forest->nb_orbits << endl;
	}
	int pt_orbit;

	nb_rows = Orbits_on_points_not_on_lines->Forest->nb_orbits;

	create_heading(headings, nb_cols);

	Table = new string[nb_rows * nb_cols];


	for (pt_orbit = 0; pt_orbit < Orbits_on_points_not_on_lines->Forest->nb_orbits; pt_orbit++) {

		if (f_vv) {
			cout << "Quartic curve associated with surface and with orbit counter = " << pt_orbit
					<< " / " << Orbits_on_points_not_on_lines->Forest->nb_orbits << endl;
		}


		std::vector<std::string> v;

		export_one_quartic_curve(
				Classification_of_arcs,
				pt_orbit,
				v,
				verbose_level - 1);


		int j;

		for (j = 0; j < nb_cols; j++) {
			Table[pt_orbit * nb_cols + j] = v[j];
		}

	}
	if (f_v) {
		cout << "surface_object_with_group::prepare_data done" << endl;
	}
}


void surface_object_with_group::export_one_quartic_curve(
		orbits::orbits_create *Classification_of_arcs,
		int pt_orbit,
		std::vector<std::string> &v,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::export_one_quartic_curve "
				"pt_orbit = " << pt_orbit << endl;
	}

	quartic_curves::quartic_curve_from_surface *QC;

	QC = NEW_OBJECT(quartic_curves::quartic_curve_from_surface);

	if (f_v) {
		cout << "surface_object_with_group::export_one_quartic_curve "
				"pt_orbit = " << pt_orbit << " before QC->init" << endl;
	}
	QC->init(this, verbose_level - 1);
	if (f_v) {
		cout << "surface_object_with_group::export_one_quartic_curve "
				"pt_orbit = " << pt_orbit << " after QC->init" << endl;
	}


	if (f_v) {
		cout << "surface_object_with_group::export_all_quartic_curves "
				"pt_orbit = " << pt_orbit << " before QC->create_quartic_curve" << endl;
	}
	QC->create_quartic_curve(Classification_of_arcs, pt_orbit, verbose_level - 1);
	if (f_v) {
		cout << "surface_object_with_group::export_all_quartic_curves "
				"pt_orbit = " << pt_orbit << " after QC->create_quartic_curve" << endl;
	}


#if 0
	// the quartic curve is now in QC->curve
	// as a Surf->Poly4_x123

	if (f_v) {
		cout << "surface_object_with_group::export_all_quartic_curves "
				"before QC->compute_stabilizer" << endl;
	}
	QC->compute_stabilizer(verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::export_all_quartic_curves "
				"after QC->compute_stabilizer" << endl;
	}

#endif


	if (f_v) {
		cout << "surface_object_with_group::export_all_quartic_curves "
				"before creating quartic_curve_object" << endl;
	}


	{
		geometry::algebraic_geometry::quartic_curve_object *QO;

		QO = NEW_OBJECT(geometry::algebraic_geometry::quartic_curve_object);

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
					"before QO->init_equation_and_bitangents_and_compute_properties" << endl;
		}
		QO->init_equation_and_bitangents_and_compute_properties(
				Surf_A->PA->PA2->QCDA->Dom,
				QC->curve /* eqn15 */,
				QC->Bitangents,
				verbose_level - 1);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
					"after QO->init_equation_and_bitangents_and_compute_properties" << endl;
		}

		//int nb_Kovalevski = -1;
		//nb_Kovalevski = QO->QP->Kovalevski->nb_Kovalevski;

#if 0
		applications_in_algebraic_geometry::quartic_curves::quartic_curve_object_with_group *QOG;

		QOG = NEW_OBJECT(applications_in_algebraic_geometry::quartic_curves::quartic_curve_object_with_group);

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
					"before QOG->init" << endl;
		}
		QOG->init(
				Surf_A->PA->PA2->QCDA,
				QO,
				QC->Aut_of_variety->Stab_gens_quartic,
				verbose_level);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
					"after QOG->init" << endl;
		}

		string label;

		label = "surface_" + SO->label_txt + "_pt_orb_" + std::to_string(pt_orbit);

		//label_txt = prefix;

		FREE_OBJECT(QOG);
#endif




		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
					"before create_vector_of_strings" << endl;
		}
		create_vector_of_strings(
				pt_orbit,
				QC,
				QO,
				v,
				verbose_level - 1);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
					"after create_vector_of_strings" << endl;
		}


		FREE_OBJECT(QO);

	}
	if (f_v) {
		cout << "surface_object_with_group::export_all_quartic_curves "
				"after creating quartic_curve_object" << endl;
	}


	FREE_OBJECT(QC);
	if (f_v) {
		cout << "surface_object_with_group::export_one_quartic_curve "
				"pt_orbit = " << pt_orbit << " done" << endl;
	}

}

void surface_object_with_group::create_heading(
		std::string &heading, int &nb_cols)
{
	heading = "Q,SO,SEQN,SEQNALG,orbit,PT,PTCOEFF,abcdef,PO_GO,PO_INDEX,curve,CURVEALG,"
			"nb_pts_on_curve,pts_on_curve,bitangents,NB_E,NB_DOUBLE,NB_SINGLE,NB_ZERO,NB_K,Kovalevski_pts,go";
	nb_cols = 22;

}

void surface_object_with_group::create_vector_of_strings(
		int pt_orbit,
		quartic_curves::quartic_curve_from_surface *QC,
		geometry::algebraic_geometry::quartic_curve_object *QO,
		std::vector<std::string> &v,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::create_vector_of_strings" << endl;
	}

	int nb_cols = 22;

	std::string s_eqn_surface;
	std::string s_eqn_surface_algebraic;
	std::string s_pt_coeff;
	std::string s_abcdef;

	s_eqn_surface = Surf->PolynomialDomains->Poly3_4->stringify(SO->Variety_object->eqn);

	s_eqn_surface_algebraic = Surf->PolynomialDomains->Poly3_4->stringify_algebraic_notation(
			SO->Variety_object->eqn);


	s_pt_coeff = Int_vec_stringify(QC->pt_A_coeff, 4);
	s_abcdef = Int_vec_stringify(QC->abcdef, 6);

	std::string s_eqn, s_eqn_algebraic, s_nb_Pts, s_Pts, s_Lines, s_Kovalevski;

	s_eqn = Surf->PolynomialDomains->Poly4_x123->stringify(QC->curve);

	s_eqn_algebraic = Surf->PolynomialDomains->Poly4_x123->stringify_algebraic_notation(QC->curve);

	s_nb_Pts = std::to_string(QC->sz_curve);

	s_Pts = Lint_vec_stringify(QC->Pts_on_curve, QC->sz_curve);

	s_Lines = Lint_vec_stringify(QC->Bitangents, QC->nb_bitangents);

	s_Kovalevski = Lint_vec_stringify(
			QO->QP->Kovalevski->Kovalevski_points,
			QO->QP->Kovalevski->nb_Kovalevski);

	algebra::ring_theory::longinteger_object ago;

	Aut_gens->group_order(ago);

	v.resize(nb_cols);

	v[0] = std::to_string(Surf->F->q);
	v[1] = std::to_string(pt_orbit);
	v[2] = "\"" + s_eqn_surface + "\"";
	v[3] = "\"" + s_eqn_surface_algebraic + "\"";
	v[4] = std::to_string(QC->pt_orbit);
	v[5] = std::to_string(QC->pt_A);
	v[6] = "\"" + s_pt_coeff + "\"";
	v[7] = "\"" + s_abcdef + "\"";
	v[8] = ago.stringify();
	v[9] = std::to_string(QC->po_index);
	v[10] = "\"" + s_eqn + "\"";
	v[11] = "\"" + s_eqn_algebraic + "\"";
	v[12] = "\"" + s_nb_Pts + "\"";
	v[13] = "\"" + s_Pts + "\"";
	v[14] = "\"" + s_Lines + "\"";
	v[15] = std::to_string(SO->SOP->nb_Eckardt_points);
	v[16] = std::to_string(SO->SOP->nb_Double_points);
	v[17] = std::to_string(SO->SOP->nb_Single_points);
	v[18] = std::to_string(SO->SOP->nb_pts_not_on_lines);
	v[19] = std::to_string(QO->QP->Kovalevski->nb_Kovalevski);
	v[20] =  "\"" + s_Kovalevski + "\"";
	v[21] = std::to_string(-1);

	if (f_v) {
		cout << "surface_object_with_group::create_vector_of_strings done" << endl;
	}
}

void surface_object_with_group::print_full_del_Pezzo(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, f, P_idx, P_idx_local;
	long int P;

	if (f_v) {
		cout << "surface_object_with_group::print_full_del_Pezzo" << endl;
	}


	//schreier *Orbits_on_points_not_on_lines;

	ost << "Full del Pezzo surfaces:\\\\" << endl;
	ost << "testing all " << Orbits_on_points_not_on_lines->Forest->nb_orbits << " orbits:\\\\" << endl;

	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|c|c|}" << endl;
	for (i = 0; i < Orbits_on_points_not_on_lines->Forest->nb_orbits; i++) {
		f = Orbits_on_points_not_on_lines->Forest->orbit_first[i];
		P_idx_local = Orbits_on_points_not_on_lines->Forest->orbit[f];
		P = SO->SOP->Pts_not_on_lines[P_idx_local];
		if (!SO->find_point(P, P_idx)) {
			cout << "surface_object_with_group::print_full_del_Pezzo "
					"could not find point" << endl;
			exit(1);
		}
		ost << i << " & " << P_idx << " & "  << P << " & ";

		int *f_deleted;
		int j, f_first;

		SO->SOP->compute_reduced_set_of_points_not_on_lines_wrt_P(
				P_idx, f_deleted, verbose_level);
		// P_idx = index into SO->Pts[]

		ost << "\\{";
		f_first = true;
		for (j = 0; j < SO->SOP->nb_pts_not_on_lines; j++) {
			if (!f_deleted[j]) {
				if (f_first) {
					f_first = false;
				}
				else {
					ost << ",";
				}
				ost << j;
			}
		}
		ost << "\\}";
		ost << " & ";
		if (SO->SOP->test_full_del_pezzo(
				P_idx, f_deleted, verbose_level)) {
			ost << " \\mbox{is full}\\\\" << endl;
		}
		else {
			ost << " \\mbox{is not full}\\\\" << endl;
		}


		//, SO->SOP->nb_pts_not_on_lines,

		FREE_int(f_deleted);
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

	if (f_v) {
		cout << "surface_object_with_group::print_full_del_Pezzo done" << endl;
	}
}

void surface_object_with_group::print_everything(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::print_everything" << endl;
	}

#if 0
	// this has already been printed:
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_equation" << endl;
	}
	SO->SOP->print_equation(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after print_equation" << endl;
	}
#endif

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_summary" << endl;
	}
	print_summary(ost, verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after print_summary" << endl;
	}



	if (TD) {



		string label_TDO;
		string label_TDA;


		label_TDO = "TDO";
		label_TDA = "TDO";

		if (f_v) {
			cout << "surface_object_with_group::print_everything "
					"before TD->report_decomposition_schemes" << endl;
		}
		TD->report_decomposition_schemes(
				ost,
				label_TDO,
				label_TDA,
				verbose_level);
		if (f_v) {
			cout << "surface_object_with_group::print_everything "
					"after TD->report_decomposition_schemes" << endl;
		}

	}
	else {
		if (f_v) {
			cout << "surface_object_with_group::print_everything "
					"TD is not available." << endl;
		}

	}


	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_lines" << endl;
	}
	SO->SOP->print_lines(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after print_lines" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before report_orbits_on_lines" << endl;
	}
	report_orbits_on_lines(
			ost,
			verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after report_orbits_on_lines" << endl;
	}


	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_points" << endl;
	}
	SO->SOP->print_points(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after print_points" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before report_orbits_on_points" << endl;
	}
	report_orbits_on_points(
			ost,
			verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after report_orbits_on_points" << endl;
	}

	report_orbits_on_Eckardt_points(
			ost,
			verbose_level);
	report_orbits_on_Double_points(
			ost,
			verbose_level);
	report_orbits_on_Single_points(
			ost,
			verbose_level);
	report_orbits_on_Zero_points(
			ost,
			verbose_level);


	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_lines_with_points_on_them" << endl;
	}
	SO->SOP->print_lines_with_points_on_them(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after print_lines_with_points_on_them" << endl;
	}



	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before SO->print_line_intersection_graph" << endl;
	}
	SO->SOP->print_line_intersection_graph(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after SO->print_line_intersection_graph" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_adjacency_matrix_with_intersection_points" << endl;
	}
	SO->SOP->print_adjacency_matrix_with_intersection_points(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after print_adjacency_matrix_with_intersection_points" << endl;
	}


	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_neighbor_sets" << endl;
	}
	SO->SOP->print_neighbor_sets(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after print_neighbor_sets" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_tritangent_planes" << endl;
	}
	SO->SOP->SmoothProperties->print_tritangent_planes(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after print_tritangent_planes" << endl;
	}

	report_orbits_on_tritangent_planes(
				ost,
				verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before SO->SOP->print_Hesse_planes" << endl;
	}
	SO->SOP->print_Hesse_planes(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after SO->SOP->print_Hesse_planes" << endl;
	}

	report_orbits_on_Hesse_planes(
				ost,
				verbose_level);

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before SO->SOP->print_axes" << endl;
	}
	SO->SOP->print_axes(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after SO->SOP->print_axes" << endl;
	}

	report_orbits_on_axes(
				ost,
				verbose_level);

#if 0
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before report_orbits" << endl;
	}
	report_orbits(
			ost,
			verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after report_orbits" << endl;
	}
#endif




	//SO->print_planes_in_trihedral_pairs(ost);

#if 0
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_generalized_quadrangle" << endl;
	}
	SO->SOP->print_generalized_quadrangle(ost);
#endif

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before SO->print_double sixes" << endl;
	}
	SO->print_double_sixes(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after SO->print_double sixes" << endl;
	}

	report_orbits_on_Double_Sixes(
				ost,
				verbose_level);

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_half_double_sixes" << endl;
	}
	SO->SOP->print_half_double_sixes(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after print_half_double_sixes" << endl;
	}

	report_orbits_on_Single_Sixes(
				ost,
				verbose_level);

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_half_double_sixes_numerically" << endl;
	}
	Surf->Schlaefli->Schlaefli_double_six->print_half_double_sixes_numerically(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after print_half_double_sixes_numerically" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_trihedral_pairs" << endl;
	}
	SO->SOP->print_trihedral_pairs(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after print_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before print_trihedral_pairs_numerically" << endl;
	}
	SO->SOP->print_trihedral_pairs_numerically(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after print_trihedral_pairs_numerically" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before report_orbits_on_trihedral_pairs" << endl;
	}
	report_orbits_on_trihedral_pairs(
			ost,
			verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after report_orbits_on_trihedral_pairs" << endl;
	}


	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before latex_table_of_trihedral_pairs" << endl;
	}
	Surf->Schlaefli->Schlaefli_trihedral_pairs->latex_table_of_trihedral_pairs(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after latex_table_of_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything done" << endl;
	}
}


void surface_object_with_group::print_summary(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::print_summary" << endl;
	}

	ost << "\\subsection*{Summary}" << endl;

	string s_orbits_lines;
	string s_orbits_points;
	string s_orbits_Eckardt_points;
	string s_orbits_Double_points;
	string s_orbits_Single_points;
	string s_orbits_Zero_points;
	string s_orbits_Hesse_planes;
	string s_orbits_Axes;
	string s_orbits_single_sixes;
	string s_orbits_double_sixes;
	string s_orbits_tritangent_planes;
	string s_orbits_trihedral_pairs;

	if (f_v) {
		cout << "surface_object_with_group::print_summary 1" << endl;
	}
	if (Orbits_on_lines) {
		Orbits_on_lines->Forest->print_orbit_length_distribution_to_string(s_orbits_lines);
	}
	if (f_v) {
		cout << "surface_object_with_group::print_summary 2" << endl;
	}
	if (Orbits_on_points) {
		Orbits_on_points->Forest->print_orbit_length_distribution_to_string(s_orbits_points);
	}
	if (f_v) {
		cout << "surface_object_with_group::print_summary 3" << endl;
	}
	if (Orbits_on_Eckardt_points) {
		Orbits_on_Eckardt_points->Forest->print_orbit_length_distribution_to_string(s_orbits_Eckardt_points);
	}
	if (f_v) {
		cout << "surface_object_with_group::print_summary 4" << endl;
	}
	if (Orbits_on_Double_points) {
		Orbits_on_Double_points->Forest->print_orbit_length_distribution_to_string(s_orbits_Double_points);
	}
	if (f_v) {
		cout << "surface_object_with_group::print_summary 5" << endl;
	}
	if (Orbits_on_Single_points) {
		Orbits_on_Single_points->Forest->print_orbit_length_distribution_to_string(s_orbits_Single_points);
	}
	if (f_v) {
		cout << "surface_object_with_group::print_summary 6" << endl;
	}
	if (Orbits_on_points_not_on_lines) {
		Orbits_on_points_not_on_lines->Forest->print_orbit_length_distribution_to_string(s_orbits_Zero_points);
	}
	if (f_v) {
		cout << "surface_object_with_group::print_summary 7" << endl;
	}
	if (Orbits_on_Hesse_planes) {
		Orbits_on_Hesse_planes->Forest->print_orbit_length_distribution_to_string(s_orbits_Hesse_planes);
	}
	if (f_v) {
		cout << "surface_object_with_group::print_summary 8" << endl;
	}
	if (Orbits_on_axes) {
		Orbits_on_axes->Forest->print_orbit_length_distribution_to_string(s_orbits_Axes);
	}
	if (f_v) {
		cout << "surface_object_with_group::print_summary 9" << endl;
	}
	if (Orbits_on_single_sixes) {
		Orbits_on_single_sixes->Forest->print_orbit_length_distribution_to_string(s_orbits_single_sixes);
	}
	if (f_v) {
		cout << "surface_object_with_group::print_summary 10" << endl;
	}
	if (Orbits_on_double_sixes) {
		Orbits_on_double_sixes->Forest->print_orbit_length_distribution_to_string(s_orbits_double_sixes);
	}
	if (f_v) {
		cout << "surface_object_with_group::print_summary 11" << endl;
	}
	if (Orbits_on_tritangent_planes) {
		Orbits_on_tritangent_planes->Forest->print_orbit_length_distribution_to_string(s_orbits_tritangent_planes);
	}
	if (f_v) {
		cout << "surface_object_with_group::print_summary 12" << endl;
	}
	if (Orbits_on_trihedral_pairs) {
		Orbits_on_trihedral_pairs->Forest->print_orbit_length_distribution_to_string(s_orbits_trihedral_pairs);
	}
	if (f_v) {
		cout << "surface_object_with_group::print_summary 13" << endl;
	}


	int nb_lines;
	int nb_points;

	if (SO->Variety_object->Line_sets) {
		nb_lines = SO->Variety_object->Line_sets->Set_size[0];
	}
	else {
		nb_lines = -1;
	}
	if (SO->Variety_object->Point_sets) {
		nb_points = SO->Variety_object->Point_sets->Set_size[0];
	}
	else {
		nb_points = -1;
	}

	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|l|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Object} & \\mbox{Number}  & \\mbox{Orbit type} \\\\";
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Lines} & " << nb_lines << " & " << s_orbits_lines;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Points on surface} & " << nb_points << " & " << s_orbits_points;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Singular points} & " << SO->SOP->nb_singular_pts << " & \\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Eckardt points} & " << SO->SOP->nb_Eckardt_points << " & " << s_orbits_Eckardt_points;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Double points} & " << SO->SOP->nb_Double_points << " & " << s_orbits_Double_points;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Single points} & " << SO->SOP->nb_Single_points << " & " << s_orbits_Single_points;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Points off lines} & " << SO->SOP->nb_pts_not_on_lines << " & " << s_orbits_Zero_points;
	ost << "\\\\" << endl;

	ost << "\\hline" << endl;
	ost << "\\mbox{Hesse planes} & " << SO->SOP->nb_Hesse_planes << " & " << s_orbits_Hesse_planes;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Axes} & " << SO->SOP->nb_axes << " & " << s_orbits_Axes;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;


	ost << "\\mbox{Single sixes} & " << 72 << " & " << s_orbits_single_sixes;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Double sixes} & " << 36 << " & " << s_orbits_double_sixes;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;

	ost << "\\mbox{Tritangent planes} & " << 45 << " & " << s_orbits_tritangent_planes;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;


	ost << "\\mbox{Trihedral pairs} & " << 120 << " & " << s_orbits_trihedral_pairs;
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;




	ost << "\\mbox{Type of points on lines} & ";
	SO->SOP->Type_pts_on_lines->print_bare_tex(ost, true);
	ost << " & \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Type of lines on points} & ";
	SO->SOP->Type_lines_on_point->print_bare_tex(ost, true);
	ost << " & \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
#if 0
	ost << "Points on lines:" << endl;
	ost << "$$" << endl;
	Type_pts_on_lines->print_bare_tex(ost, true);
	ost << "$$" << endl;
	ost << "Lines on points:" << endl;
	ost << "$$" << endl;
	Type_lines_on_point->print_bare_tex(ost, true);
	ost << "$$" << endl;
#endif

	if (f_v) {
		cout << "surface_object_with_group::print_summary done" << endl;
	}


}


void surface_object_with_group::print_action_on_surface(
		std::string &label_of_elements,
		data_structures_groups::vector_ge *Elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::print_action_on_surface" << endl;
	}


	actions::action *A;

	A = Surf_A->A;

	other::orbiter_kernel_system::file_io Fio;



	int *Elt;
	algebra::ring_theory::longinteger_object go;

	//Elt = NEW_int(A->elt_size_in_int);


	string fname;

	fname = label_of_elements + "_action_on_surface.tex";

	int nb_elements;

	nb_elements = Elements->len;

	{
		ofstream ost(fname);
		other::l1_interfaces::latex_interface L;
		int i, ord;

		L.head_easy(ost);

		//H->print_all_group_elements_tex(fp, f_with_permutation, f_override_action, A_special);
		//H->print_all_group_elements_tree(fp);
		//H->print_all_group_elements_with_permutations_tex(fp);

		//Schreier.print_and_list_orbits_tex(fp);

		ost << "Action $" << A->label_tex << "$:\\\\" << endl;
		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;

		for (i = 0; i < nb_elements; i++) {

#if 0
			A->Group_element->make_element(
					Elt,
					element_data + i * A->make_element_size,
					verbose_level);
#endif

			Elt = Elements->ith(i);

			ord = A->Group_element->element_order(Elt);

			ost << "Element " << setw(5) << i << " / "
					<< nb_elements << " of order " << ord << ":" << endl;

			//A->print_one_element_tex(
			//		ost, Elt, false /*f_with_permutation*/);


			ost << "$$" << endl;
			A->Group_element->element_print_latex(Elt, ost);
			ost << "$$" << endl;



			if (true /* f_with_fix_structure*/) {
				int f;

				f = A->Group_element->count_fixed_points(Elt, 0 /* verbose_level */);

				ost << "$f=" << f << "$\\\\" << endl;
			}

			ost << "\\bigskip" << endl;
			ost << endl;

			ost << "Action on points: \\\\" << endl;
			A_on_points->Group_element->element_print_as_permutation(Elt, ost);
			ost << "\\bigskip" << endl;
			ost << endl;


			ost << "Action on Eckardt points: \\\\" << endl;
			A_on_Eckardt_points->Group_element->element_print_as_permutation(Elt, ost);
			ost << "\\bigskip" << endl;
			ost << endl;

			ost << "Action on Double points: \\\\" << endl;
			A_on_Double_points->Group_element->element_print_as_permutation(Elt, ost);
			ost << "\\bigskip" << endl;
			ost << endl;

			ost << "Action on Single points: \\\\" << endl;
			A_on_Single_points->Group_element->element_print_as_permutation(Elt, ost);
			ost << "\\bigskip" << endl;
			ost << endl;

			ost << "Action on lines: \\\\" << endl;
			A_on_the_lines->Group_element->element_print_as_permutation(Elt, ost);
			ost << "\\bigskip" << endl;
			ost << endl;

			ost << "Action on single sixes: \\\\" << endl;
			A_single_sixes->Group_element->element_print_as_permutation(Elt, ost);
			ost << "\\bigskip" << endl;
			ost << endl;

			ost << "Action on tritangent planes: \\\\" << endl;
			A_on_tritangent_planes->Group_element->element_print_as_permutation(Elt, ost);
			ost << "\\bigskip" << endl;
			ost << endl;

			ost << "Action on Hesse planes: \\\\" << endl;
			A_on_Hesse_planes->Group_element->element_print_as_permutation(Elt, ost);
			ost << "\\bigskip" << endl;
			ost << endl;


			ost << "Action on trihedral pairs: \\\\" << endl;
			A_on_trihedral_pairs->Group_element->element_print_as_permutation(Elt, ost);
			ost << "\\bigskip" << endl;
			ost << endl;

			ost << "Action on points not on lines: \\\\" << endl;
			A_on_pts_not_on_lines->Group_element->element_print_as_permutation(Elt, ost);
			ost << "\\bigskip" << endl;
			ost << endl;


		}


		L.foot(ost);
	}
	cout << "Written file " << fname
			<< " of size " << Fio.file_size(fname) << endl;


	//FREE_int(Elt);


	if (f_v) {
		cout << "surface_object_with_group::print_action_on_surface done" << endl;
	}
}

#if 0
void surface_object_with_group::print_double_sixes(
		std::ostream &ost)
{
	//int idx;
	ost << "\\bigskip" << endl;

	ost << "\\subsection*{Double sixes}" << endl;

	SO->Surf->Schlaefli->Schlaefli_double_six->print_double_sixes(
			ost, SO->Variety_object->Line_sets->Sets[0]);


}
#endif

void surface_object_with_group::compute_tactical_decompositions(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::compute_tactical_decompositions" << endl;
	}


	if (Surf->F->q > 10) {

		cout << "surface_object_with_group::compute_tactical_decompositions, Surf->F->q > 10, skipping." << endl;
		return;
	}

	TD = NEW_OBJECT(apps_combinatorics::variety_with_TDO_and_TDA);


	if (f_v) {
		cout << "surface_object_with_group::compute_tactical_decompositions "
				"before TD->init_and_compute_tactical_decompositions" << endl;
	}
	TD->init_and_compute_tactical_decompositions(
			Surf_A->PA, SO->Variety_object, Aut_gens,
			verbose_level);
	if (f_v) {
		cout << "surface_object_with_group::compute_tactical_decompositions "
				"after TD->init_and_compute_tactical_decompositions" << endl;
	}



	if (f_v) {
		cout << "surface_object_with_group::compute_tactical_decompositions done" << endl;
	}
}

void surface_object_with_group::report_orbits_on_lines(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_lines" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_lines;

	SoS_Orbits_on_lines = Orbits_on_lines->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Lines}" << endl;

	//SoS_Orbits_on_lines->print_table_tex(ost);

	{
		other::l1_interfaces::latex_interface L;
		int i, j, idx, len;

		//cout << "set of sets with " << nb_sets << " sets :" << endl;
		for (i = 0; i < SoS_Orbits_on_lines->nb_sets; i++) {

			len = SoS_Orbits_on_lines->Set_size[i];

			ost << "Set " << i << " has size " << len << " : ";

			ost << "$";

			L.lint_set_print_tex(ost, SoS_Orbits_on_lines->Sets[i], len);
			//SO->Surf->print_lines_tex(ost, SoS_Orbits_on_lines->Sets[i], len);

			ost << " = ";
			ost << "\\{ ";
			for (j = 0; j < len; j++) {
				idx = SoS_Orbits_on_lines->Sets[i][j];
				ost << SO->Surf->Schlaefli->Labels->Line_label_tex[idx];
				if (j < len - 1) {
					ost << ", ";
				}
			}
			ost << " \\}";


			ost << "$";


			ost << "\\\\" << endl;
		}
	}


	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_lines done" << endl;
	}
}

void surface_object_with_group::report_orbits_on_points(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_points" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_points;

	SoS_Orbits_on_points = Orbits_on_points->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Points}" << endl;

	SoS_Orbits_on_points->print_table_tex(ost);


	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_points done" << endl;
	}

}

void surface_object_with_group::report_orbits_on_Eckardt_points(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Eckardt_points" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_Eckardt_points;

	SoS_Orbits_on_Eckardt_points = Orbits_on_Eckardt_points->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Eckardt Points}" << endl;

	if (SoS_Orbits_on_Eckardt_points->nb_sets == 0) {
		ost << "There are no orbits on Eckardt points.\\\\" << endl;
	}
	else {
		SoS_Orbits_on_Eckardt_points->print_table_tex(ost);
	}

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Eckardt_points done" << endl;
	}

}

void surface_object_with_group::report_orbits_on_Double_points(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Double_points" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_Double_points;

	SoS_Orbits_on_Double_points = Orbits_on_Double_points->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Double Points}" << endl;

	if (SoS_Orbits_on_Double_points->nb_sets == 0) {
		ost << "There are no orbits on double points.\\\\" << endl;
	}
	else {
		SoS_Orbits_on_Double_points->print_table_tex(ost);
	}

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Double_points done" << endl;
	}

}

void surface_object_with_group::report_orbits_on_Single_points(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Single_points" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_Single_points;

	SoS_Orbits_on_Single_points = Orbits_on_Single_points->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Single Points}" << endl;

	if (SoS_Orbits_on_Single_points->nb_sets == 0) {
		ost << "There are no orbits on single points.\\\\" << endl;
	}
	else {
		SoS_Orbits_on_Single_points->print_table_tex(ost);
	}

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Single_points done" << endl;
	}

}

void surface_object_with_group::report_orbits_on_Zero_points(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Zero_points" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_points_not_on_lines;

	SoS_Orbits_on_points_not_on_lines = Orbits_on_points_not_on_lines->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Zero Points}" << endl;

	if (SoS_Orbits_on_points_not_on_lines->nb_sets == 0) {
		ost << "There are no orbits on points not on lines.\\\\" << endl;
	}
	else {
		SoS_Orbits_on_points_not_on_lines->print_table_tex(ost);
	}

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Zero_points done" << endl;
	}

}


void surface_object_with_group::report_orbits_on_Hesse_planes(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Hesse_planes" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_Hesse_planes;

	SoS_Orbits_on_Hesse_planes = Orbits_on_Hesse_planes->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on Hesse Planes}" << endl;

	SoS_Orbits_on_Hesse_planes->print_table_tex(ost);


	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Hesse_planes done" << endl;
	}

}

void surface_object_with_group::report_orbits_on_axes(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_axes" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_axes;

	SoS_Orbits_on_axes = Orbits_on_axes->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on axes}" << endl;

	SoS_Orbits_on_axes->print_table_tex(ost);


	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_axes done" << endl;
	}

}

void surface_object_with_group::report_orbits_on_Single_Sixes(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Single_Sixes" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_single_sixes;

	SoS_Orbits_on_single_sixes = Orbits_on_single_sixes->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on single sixes}" << endl;

	SoS_Orbits_on_single_sixes->print_table_tex(ost);


	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Single_Sixes done" << endl;
	}

}

void surface_object_with_group::report_orbits_on_Double_Sixes(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Double_Sixes" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_double_sixes;

	SoS_Orbits_on_double_sixes = Orbits_on_double_sixes->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on double sixes}" << endl;

	SoS_Orbits_on_double_sixes->print_table_tex(ost);


	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_Double_Sixes done" << endl;
	}

}

void surface_object_with_group::report_orbits_on_tritangent_planes(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_tritangent_planes" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_tritangent_planes;

	SoS_Orbits_on_tritangent_planes = Orbits_on_tritangent_planes->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on tritangent planes}" << endl;

	//SoS_Orbits_on_tritangent_planes->print_table_tex(ost);

	{
		other::l1_interfaces::latex_interface L;
		int i, j, idx, len;

		//cout << "set of sets with " << nb_sets << " sets :" << endl;
		for (i = 0; i < SoS_Orbits_on_tritangent_planes->nb_sets; i++) {

			len = SoS_Orbits_on_tritangent_planes->Set_size[i];

			ost << "Set " << i << " has size " << len << " : ";

			ost << "$";

			L.lint_set_print_tex(ost, SoS_Orbits_on_tritangent_planes->Sets[i], len);
			//SO->Surf->print_lines_tex(ost, SoS_Orbits_on_lines->Sets[i], len);

			ost << " = ";
			ost << "\\{ ";
			for (j = 0; j < len; j++) {
				idx = SoS_Orbits_on_tritangent_planes->Sets[i][j];
				//ost << SO->Surf->Schlaefli->Labels->Tritangent_plane_label_tex[idx];

				ost << "\\pi_{" << SO->Surf->Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[idx] << "}";
				//ost << SO->Surf->Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[idx];
				//ost << SO->Surf->Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[idx];


				if (j < len - 1) {
					ost << ", ";
				}
			}
			ost << " \\}";


			ost << "$";


			ost << "\\\\" << endl;
		}
	}



	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_tritangent_planes done" << endl;
	}

}


void surface_object_with_group::report_orbits_on_trihedral_pairs(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_trihedral_pairs" << endl;
	}

	other::data_structures::set_of_sets *SoS_Orbits_on_trihedral_pairs;

	SoS_Orbits_on_trihedral_pairs = Orbits_on_trihedral_pairs->Forest->get_set_of_sets(verbose_level);

	ost << "\\subsection*{Orbits on trihedral pairs}" << endl;

	SoS_Orbits_on_trihedral_pairs->print_table_tex(ost);


	if (f_v) {
		cout << "surface_object_with_group::report_orbits_on_trihedral_pairs done" << endl;
	}

}


#if 0
void surface_object_with_group::report_orbits(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_group::report_orbits" << endl;
	}

	ost << "\\subsection*{Orbits}" << endl;


	//other::data_structures::set_of_sets *SoS_Orbits_on_lines;
	//other::data_structures::set_of_sets *SoS_Orbits_on_points;
	//other::data_structures::set_of_sets *SoS_Orbits_on_Eckardt_points;
	//other::data_structures::set_of_sets *SoS_Orbits_on_Double_points;
	//other::data_structures::set_of_sets *SoS_Orbits_on_Single_points;
	//other::data_structures::set_of_sets *SoS_Orbits_on_points_not_on_lines;
	//other::data_structures::set_of_sets *SoS_Orbits_on_Hesse_planes;
	//other::data_structures::set_of_sets *SoS_Orbits_on_axes;
	//other::data_structures::set_of_sets *SoS_Orbits_on_single_sixes;
	//other::data_structures::set_of_sets *SoS_Orbits_on_double_sixes;
	//other::data_structures::set_of_sets *SoS_Orbits_on_tritangent_planes;
	other::data_structures::set_of_sets *SoS_Orbits_on_trihedral_pairs;

	//SoS_Orbits_on_lines = Orbits_on_lines->Forest->get_set_of_sets(verbose_level);
	//SoS_Orbits_on_points = Orbits_on_points->Forest->get_set_of_sets(verbose_level);
	//SoS_Orbits_on_Eckardt_points = Orbits_on_Eckardt_points->Forest->get_set_of_sets(verbose_level);
	//SoS_Orbits_on_Double_points = Orbits_on_Double_points->Forest->get_set_of_sets(verbose_level);
	//SoS_Orbits_on_Single_points = Orbits_on_Single_points->Forest->get_set_of_sets(verbose_level);
	//SoS_Orbits_on_points_not_on_lines = Orbits_on_points_not_on_lines->Forest->get_set_of_sets(verbose_level);
	//SoS_Orbits_on_Hesse_planes = Orbits_on_Hesse_planes->Forest->get_set_of_sets(verbose_level);
	//SoS_Orbits_on_axes = Orbits_on_axes->Forest->get_set_of_sets(verbose_level);
	//SoS_Orbits_on_single_sixes = Orbits_on_single_sixes->Forest->get_set_of_sets(verbose_level);
	//SoS_Orbits_on_double_sixes = Orbits_on_double_sixes->Forest->get_set_of_sets(verbose_level);
	//SoS_Orbits_on_tritangent_planes = Orbits_on_tritangent_planes->Forest->get_set_of_sets(verbose_level);
	SoS_Orbits_on_trihedral_pairs = Orbits_on_trihedral_pairs->Forest->get_set_of_sets(verbose_level);

#if 0
	ost << "\\subsection*{Orbits on Lines}" << endl;

	//SoS_Orbits_on_lines->print_table_tex(ost);

	{
		other::l1_interfaces::latex_interface L;
		int i, j, idx, len;

		//cout << "set of sets with " << nb_sets << " sets :" << endl;
		for (i = 0; i < SoS_Orbits_on_lines->nb_sets; i++) {

			len = SoS_Orbits_on_lines->Set_size[i];

			ost << "Set " << i << " has size " << len << " : ";

			ost << "$";

			L.lint_set_print_tex(ost, SoS_Orbits_on_lines->Sets[i], len);
			//SO->Surf->print_lines_tex(ost, SoS_Orbits_on_lines->Sets[i], len);

			ost << " = ";
			ost << "\\{ ";
			for (j = 0; j < len; j++) {
				idx = SoS_Orbits_on_lines->Sets[i][j];
				ost << SO->Surf->Schlaefli->Labels->Line_label_tex[idx];
				if (j < len - 1) {
					ost << ", ";
				}
			}
			ost << " \\}";


			ost << "$";


			ost << "\\\\" << endl;
		}
	}


	ost << "\\subsection*{Orbits on Points}" << endl;

	SoS_Orbits_on_points->print_table_tex(ost);


	ost << "\\subsection*{Orbits on Eckardt Points}" << endl;

	SoS_Orbits_on_Eckardt_points->print_table_tex(ost);

	ost << "\\subsection*{Orbits on Double Points}" << endl;

	SoS_Orbits_on_Double_points->print_table_tex(ost);

	ost << "\\subsection*{Orbits on Single Points}" << endl;

	SoS_Orbits_on_Single_points->print_table_tex(ost);

	ost << "\\subsection*{Orbits on Points not on Lines}" << endl;

	SoS_Orbits_on_points_not_on_lines->print_table_tex(ost);

	ost << "\\subsection*{Orbits on Hesse Planes}" << endl;

	SoS_Orbits_on_Hesse_planes->print_table_tex(ost);

	ost << "\\subsection*{Orbits on Axes}" << endl;

	SoS_Orbits_on_axes->print_table_tex(ost);

	ost << "\\subsection*{Orbits on Single Sixes}" << endl;

	SoS_Orbits_on_single_sixes->print_table_tex(ost);

	ost << "\\subsection*{Orbits on Double Sixes}" << endl;

	SoS_Orbits_on_double_sixes->print_table_tex(ost);

	ost << "\\subsection*{Orbits on Tritangent Planes}" << endl;

	//SoS_Orbits_on_tritangent_planes->print_table_tex(ost);


	{
		other::l1_interfaces::latex_interface L;
		int i, j, idx, len;

		//cout << "set of sets with " << nb_sets << " sets :" << endl;
		for (i = 0; i < SoS_Orbits_on_tritangent_planes->nb_sets; i++) {

			len = SoS_Orbits_on_tritangent_planes->Set_size[i];

			ost << "Set " << i << " has size " << len << " : ";

			ost << "$";

			L.lint_set_print_tex(ost, SoS_Orbits_on_tritangent_planes->Sets[i], len);
			//SO->Surf->print_lines_tex(ost, SoS_Orbits_on_lines->Sets[i], len);

			ost << " = ";
			ost << "\\{ ";
			for (j = 0; j < len; j++) {
				idx = SoS_Orbits_on_tritangent_planes->Sets[i][j];
				//ost << SO->Surf->Schlaefli->Labels->Tritangent_plane_label_tex[idx];

				ost << "\\pi_{" << SO->Surf->Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[idx] << "}";
				//ost << SO->Surf->Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[idx];
				//ost << SO->Surf->Schlaefli->Schlaefli_tritangent_planes->Eckard_point_label_tex[idx];


				if (j < len - 1) {
					ost << ", ";
				}
			}
			ost << " \\}";


			ost << "$";


			ost << "\\\\" << endl;
		}
	}

#endif


	ost << "\\subsection*{Orbits on Trihedral Pairs}" << endl;

	SoS_Orbits_on_trihedral_pairs->print_table_tex(ost);



	if (f_v) {
		cout << "surface_object_with_group::report_orbits done" << endl;
	}

}
#endif





}}}}

