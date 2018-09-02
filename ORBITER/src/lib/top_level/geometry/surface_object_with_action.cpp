// surface_object_with_action.C
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


surface_object_with_action::surface_object_with_action()
{
	null();
}

surface_object_with_action::~surface_object_with_action()
{
	freeself();
}

void surface_object_with_action::null()
{
	q = 0;
	F = NULL;
	Surf = NULL;
	Surf_A = NULL;
	SO = NULL;
	Aut_gens = NULL;

	A_on_points = NULL;
	A_on_Eckardt_points = NULL;
	A_on_Double_points = NULL;
	A_on_the_lines = NULL;
	A_single_sixes = NULL;
	A_on_tritangent_planes = NULL;
	A_on_trihedral_pairs = NULL;
	A_on_pts_not_on_lines = NULL;


	Orbits_on_points = NULL;
	Orbits_on_Eckardt_points = NULL;
	Orbits_on_Double_points = NULL;
	Orbits_on_lines = NULL;
	Orbits_on_single_sixes = NULL;
	Orbits_on_tritangent_planes = NULL;
	Orbits_on_trihedral_pairs = NULL;
	Orbits_on_points_not_on_lines = NULL;
}

void surface_object_with_action::freeself()
{
	if (A_on_points) {
		FREE_OBJECT(A_on_points);
		}
	if (A_on_Eckardt_points) {
		FREE_OBJECT(A_on_Eckardt_points);
		}
	if (A_on_Double_points) {
		FREE_OBJECT(A_on_Double_points);
		}
	if (A_on_the_lines) {
		FREE_OBJECT(A_on_the_lines);
		}
	if (A_single_sixes) {
		FREE_OBJECT(A_single_sixes);
		}
	if (A_on_tritangent_planes) {
		FREE_OBJECT(A_on_tritangent_planes);
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
	if (Orbits_on_lines) {
		FREE_OBJECT(Orbits_on_lines);
		}
	if (Orbits_on_single_sixes) {
		FREE_OBJECT(Orbits_on_single_sixes);
		}
	if (Orbits_on_tritangent_planes) {
		FREE_OBJECT(Orbits_on_tritangent_planes);
		}
	if (Orbits_on_trihedral_pairs) {
		FREE_OBJECT(Orbits_on_trihedral_pairs);
		}
	if (Orbits_on_points_not_on_lines) {
		FREE_OBJECT(Orbits_on_points_not_on_lines);
		}
	null();
}

INT surface_object_with_action::init_equation(
	surface_with_action *Surf_A, INT *eqn,
	strong_generators *Aut_gens, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::init_equation" << endl;
		}

	surface_object_with_action::Surf_A = Surf_A;
	Surf = Surf_A->Surf;
	F = Surf->F;
	q = F->q;

	SO = NEW_OBJECT(surface_object);
	if (f_v) {
		cout << "surface_object_with_action::init_equation "
				"before SO->init_equation" << endl;
		}
	if (!SO->init_equation(Surf_A->Surf, eqn, verbose_level)) {
		cout << "surface_object_with_action::init_equation "
				"the surface does not have 27 lines" << endl;
		return FALSE;
		}
	if (f_v) {
		cout << "surface_object_with_action::init_equation "
				"after SO->init_equation" << endl;
		}

	surface_object_with_action::Aut_gens = Aut_gens;
	
	if (f_v) {
		cout << "surface_object_with_action::init_equation "
				"before compute_orbits_of_automorphism_group" << endl;
		}
	compute_orbits_of_automorphism_group(verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init_equation "
				"after compute_orbits_of_automorphism_group" << endl;
		}

	if (f_v) {
		cout << "surface_object_with_action::init_equation done" << endl;
		}
	return TRUE;
}



void surface_object_with_action::init(surface_with_action *Surf_A, 
	INT *Lines, INT *eqn, 
	strong_generators *Aut_gens,
	INT f_find_double_six_and_rearrange_lines,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::init" << endl;
		}

	surface_object_with_action::Surf_A = Surf_A;
	Surf = Surf_A->Surf;
	F = Surf->F;
	q = F->q;

	SO = NEW_OBJECT(surface_object);
	if (f_v) {
		cout << "surface_object_with_action::init "
				"before SO->init" << endl;
		}
	SO->init(Surf_A->Surf, Lines, eqn,
			f_find_double_six_and_rearrange_lines, verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init "
				"after SO->init" << endl;
		}


	surface_object_with_action::Aut_gens = Aut_gens;

	if (f_v) {
		cout << "surface_object_with_action::init_surface_object "
				"testing Aut_gens" << endl;
		}
	Aut_gens->test_if_set_is_invariant_under_given_action(
			Surf_A->A2, Lines, 27, verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init_surface_object "
				"testing Aut_gens done" << endl;
		}
	
	if (f_v) {
		cout << "surface_object_with_action::init_surface_object "
				"before compute_orbits_of_automorphism_group" << endl;
		}
	compute_orbits_of_automorphism_group(verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init_surface_object "
				"after compute_orbits_of_automorphism_group" << endl;
		}

	if (f_v) {
		cout << "surface_object_with_action::init done" << endl;
		}
}


void surface_object_with_action::init_surface_object(
	surface_with_action *Surf_A, surface_object *SO,
	strong_generators *Aut_gens, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::init_surface_object" << endl;
		}
	surface_object_with_action::Surf_A = Surf_A;
	Surf = Surf_A->Surf;
	F = Surf->F;
	q = F->q;


	surface_object_with_action::SO = SO;
	surface_object_with_action::Aut_gens = Aut_gens;
	

	if (f_v) {
		cout << "surface_object_with_action::init_surface_object "
				"before compute_orbits_of_automorphism_group" << endl;
		}
	compute_orbits_of_automorphism_group(verbose_level);
	if (f_v) {
		cout << "surface_object_with_action::init_surface_object "
				"after compute_orbits_of_automorphism_group" << endl;
		}
	
	if (f_v) {
		cout << "surface_object_with_action::init_surface_object "
				"done" << endl;
		}
}

void surface_object_with_action::compute_orbits_of_automorphism_group(
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_"
				"automorphism_group" << endl;
		}

	// orbits on points:
	
	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_"
				"automorphism_group orbits on points" << endl;
		}
	init_orbits_on_points(verbose_level - 1);


	// orbits on Eckardt points:
	
	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_"
				"automorphism_group orbits on Eckardt points" << endl;
		}
	init_orbits_on_Eckardt_points(verbose_level - 1);


	// orbits on Double points:
	
	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_"
				"automorphism_group orbits on double points" << endl;
		}
	init_orbits_on_Double_points(verbose_level - 1);


	// orbits on lines:

	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_"
				"automorphism_group orbits on lines" << endl;
		}
	init_orbits_on_lines(verbose_level);


	// orbits on half double sixes:

	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_"
				"automorphism_group orbits on half double sixes" << endl;
		}
	init_orbits_on_half_double_sixes(verbose_level);



	// orbits on tritangent planes:

	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_"
				"automorphism_group orbits on tritangent planes" << endl;
		}
	init_orbits_on_tritangent_planes(verbose_level);


	// orbits on trihedral pairs:

	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_"
				"automorphism_group orbits on trihedral pairs" << endl;
		}
	init_orbits_on_trihedral_pairs(verbose_level);



	// orbits on points not on lines:

	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_"
				"automorphism_group orbits on points not on lines" << endl;
		}
	init_orbits_on_points_not_on_lines(verbose_level);


	if (f_v) {
		cout << "surface_object_with_action::compute_orbits_of_"
				"automorphism_group done" << endl;
		}
}

void surface_object_with_action::init_orbits_on_points(
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_points" << endl;
		}

	if (f_v) {
		cout << "surface_object_with_action action "
				"on points:" << endl;
		}
	A_on_points = Surf_A->A->restricted_action(
			SO->Pts, SO->nb_pts, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_object_with_action action "
				"on points done" << endl;
		}


	if (f_v) {
		cout << "computing orbits on points:" << endl;
		}
	Orbits_on_points = Aut_gens->orbits_on_points_schreier(
			A_on_points, 0 /*verbose_level*/);
	if (f_v) {
		cout << "We found " << Orbits_on_points->nb_orbits
				<< " orbits on points" << endl;
		}

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_points done" << endl;
		}
}

void surface_object_with_action::init_orbits_on_Eckardt_points(
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_Eckardt_points" << endl;
		}

	if (f_v) {
		cout << "creating action on Eckardt points:" << endl;
		}
	A_on_Eckardt_points = Surf_A->A->restricted_action(
			SO->Eckardt_points, SO->nb_Eckardt_points, 0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on Eckardt points done" << endl;
		}


	if (f_v) {
		cout << "computing orbits on Eckardt points:" << endl;
		}
	Orbits_on_Eckardt_points = Aut_gens->orbits_on_points_schreier(
			A_on_Eckardt_points, 0 /*verbose_level*/);
	if (f_v) {
		cout << "We found " << Orbits_on_Eckardt_points->nb_orbits
				<< " orbits on Eckardt points" << endl;
		}

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_Eckardt_points done" << endl;
		}
}

void surface_object_with_action::init_orbits_on_Double_points(
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_Double_points" << endl;
		}

	if (f_v) {
		cout << "creating action on Double points:" << endl;
		}
	A_on_Double_points = Surf_A->A->restricted_action(
			SO->Double_points, SO->nb_Double_points,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on Double points done" << endl;
		}


	if (f_v) {
		cout << "computing orbits on Double points:" << endl;
		}
	Orbits_on_Double_points = Aut_gens->orbits_on_points_schreier(
			A_on_Double_points, 0 /*verbose_level*/);
	if (f_v) {
		cout << "We found " << Orbits_on_Double_points->nb_orbits
				<< " orbits on Double points" << endl;
		}

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_on_"
				"Double_points done" << endl;
		}
}

void surface_object_with_action::init_orbits_on_lines(
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "clebsch::init_orbits_on_lines" << endl;
		}

	if (f_v) {
		cout << "creating restricted action "
				"on the lines:" << endl;
		}
	A_on_the_lines = Surf_A->A2->restricted_action(
			SO->Lines, 27, 0 /*verbose_level*/);
	if (f_v) {
		cout << "creating restricted action "
				"on the lines done" << endl;
		}

	if (f_v) {
		cout << "computing orbits on lines:" << endl;
		}
	Orbits_on_lines = Aut_gens->orbits_on_points_schreier(
			A_on_the_lines, 0 /*verbose_level*/);
	if (f_v) {
		cout << "We found " << Orbits_on_lines->nb_orbits
				<< " orbits on lines" << endl;
		}

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_lines done" << endl;
		}
}

void surface_object_with_action::init_orbits_on_half_double_sixes(
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_half_double_sixes" << endl;
		}

	if (f_v) {
		cout << "creating action on half double sixes:" << endl;
		}
	A_single_sixes = A_on_the_lines->create_induced_action_on_sets(
			72, 6, Surf->Double_six, 0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on half double sixes done" << endl;
		}


	if (f_v) {
		cout << "computing orbits on single sixes:" << endl;
		}
	Orbits_on_single_sixes = Aut_gens->orbits_on_points_schreier(
			A_single_sixes, 0 /*verbose_level*/);
	if (f_v) {
		cout << "computing orbits on single sixes done" << endl;
		}
	if (f_v) {
		cout << "We found " << Orbits_on_single_sixes->nb_orbits
				<< " orbits on single sixes" << endl;
		}

	//nb_orbits_on_single_sixes = Orbits_on_single_sixes->nb_orbits;

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_half_double_sixes done" << endl;
		}
}

void surface_object_with_action::init_orbits_on_tritangent_planes(
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_tritangent_planes" << endl;
		}

	if (f_v) {
		cout << "creating action on tritangent planes:" << endl;
		cout << "SO->nb_tritangent_planes = "
				<< SO->nb_tritangent_planes << endl;
		}
	A_on_tritangent_planes = A_on_the_lines->create_induced_action_on_sets(
			SO->nb_tritangent_planes, 3,
			SO->Lines_in_tritangent_plane, 0 /*verbose_level*/);
	if (f_v) {
		cout << "action on tritangent planes done" << endl;
		}

	Orbits_on_tritangent_planes = Aut_gens->orbits_on_points_schreier(
			A_on_tritangent_planes, 0 /*verbose_level*/);
	if (f_v) {
		cout << "We found " << Orbits_on_tritangent_planes->nb_orbits
				<< " orbits on the set of " << SO->nb_tritangent_planes
				<< " tritangent planes" << endl;
		}

	Orbits_on_tritangent_planes->print_and_list_orbits(cout);

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_tritangent_planes done" << endl;
		}
}

void surface_object_with_action::init_orbits_on_trihedral_pairs(
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_trihedral_pairs" << endl;
		}

	if (f_v) {
		cout << "creating action on trihedral pairs:" << endl;
		}
	A_on_trihedral_pairs =
			A_on_tritangent_planes->create_induced_action_on_sets(
					120, 6, SO->Trihedral_pairs_as_tritangent_planes,
					0 /*verbose_level*/);
	if (f_v) {
		cout << "action on trihedral pairs created" << endl;
		}

	Orbits_on_trihedral_pairs = Aut_gens->orbits_on_points_schreier(
			A_on_trihedral_pairs, 0 /*verbose_level*/);
	if (f_v) {
		cout << "We found " << Orbits_on_trihedral_pairs->nb_orbits
				<< " orbits on trihedral pairs" << endl;
		}

	Orbits_on_trihedral_pairs->print_and_list_orbits(cout);

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_trihedral_pairs done" << endl;
		}
}

void surface_object_with_action::init_orbits_on_points_not_on_lines(
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_points_not_on_lines" << endl;
		}

	if (f_v) {
		cout << "creating action on points not on lines:" << endl;
		}
	A_on_pts_not_on_lines = Surf_A->A->restricted_action(
			SO->Pts_not_on_lines, SO->nb_pts_not_on_lines,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "creating action on points not on lines done" << endl;
		}

	Orbits_on_points_not_on_lines =
			Aut_gens->orbits_on_points_schreier(
					A_on_pts_not_on_lines,  0 /*verbose_level*/);
	if (f_v) {
		cout << "We found " << Orbits_on_points_not_on_lines->nb_orbits
				<< " orbits on points not on lines" << endl;
		}

	Orbits_on_points_not_on_lines->print_and_list_orbits(cout);

	if (f_v) {
		cout << "surface_object_with_action::init_orbits_"
				"on_points_not_on_lines done" << endl;
		}
}


void surface_object_with_action::print_automorphism_group(
	ostream &ost,
	INT f_print_orbits, const char *fname_mask)
{
	longinteger_object go;

	Aut_gens->group_order(go);
	
	ost << "\\clearpage" << endl;
	ost << "\\section*{Orbits of the automorphism group}" << endl;
	ost << "The automorphism group has order " << go << endl;
	ost << "\\bigskip" << endl;
	ost << "\\subsection*{Orbits on points}" << endl;
	//Orbits_on_points->print_and_list_orbits_and_
	//stabilizer_sorted_by_length(ost, TRUE, Surf_A->A, go);
	Orbits_on_points->print_and_list_orbits_sorted_by_length_tex(ost);

	ost << "\\subsection*{Orbits on Eckardt points}" << endl;
	Orbits_on_Eckardt_points->print_and_list_orbits_sorted_by_length_tex(ost);

	ost << "\\subsection*{Orbits on Double points}" << endl;
	Orbits_on_Double_points->print_and_list_orbits_sorted_by_length_tex(ost);

	ost << "\\subsection*{Orbits on points not on lines}" << endl;
	//Orbits_on_points_not_on_lines->print_and_list_orbits_sorted_by_length_tex(ost);
	Orbits_on_points_not_on_lines->print_and_list_orbits_sorted_by_length_tex(ost);


	ost << "\\subsection*{Orbits on lines}" << endl;
	Orbits_on_lines->print_and_list_orbits_sorted_by_length_tex(ost);

	INT *Decomp_scheme;
	INT nb;
	INT block_width = 10;
	nb = Orbits_on_lines->nb_orbits;
	Orbits_on_lines->get_orbit_decomposition_scheme_of_graph(
			SO->Adj_line_intersection_graph, 27, Decomp_scheme,
			0 /*verbose_level*/);
	ost << "\\subsection*{Decomposition scheme of line intersection graph}" << endl;
	ost << "Decomposition scheme of line intersection graph:" << endl;
	print_integer_matrix_tex_block_by_block(ost,
			Decomp_scheme, nb, nb, block_width);
	

	ost << "\\subsection*{Orbits on single sixes}" << endl;
	Orbits_on_single_sixes->print_and_list_orbits_sorted_by_length_tex(ost);

	if (f_print_orbits) {

		INT xmax = 1000000;
		INT ymax = 1000000;
		INT f_circletext = TRUE;
		INT rad = 22000;
		INT f_embedded = FALSE;
		INT f_sideways = FALSE;
		double scale = 0.33;
		double line_width = 0.5;
		INT f_has_point_labels = FALSE;
		INT *point_labels = NULL;
	
		Orbits_on_single_sixes->draw_forest(fname_mask, 
			xmax, ymax, 
			f_circletext, rad, 
			f_embedded, f_sideways, 
			scale, line_width, 
			f_has_point_labels, point_labels, 
			0 /*verbose_level*/);


		INT i;
		for (i = 0; i < Orbits_on_single_sixes->nb_orbits; i++) {
			char fname[1000];

			sprintf(fname, fname_mask, i);
			ost << "" << endl; 
			ost << "\\bigskip" << endl; 
			ost << "" << endl; 
			ost << "Orbit " << i << " consisting of the following "
					<< Orbits_on_single_sixes->orbit_len[i]
					<< " half double sixes:" << endl;
			ost << "$$" << endl;
			INT_set_print_tex(ost, 
				Orbits_on_single_sixes->orbit + 
					Orbits_on_single_sixes->orbit_first[i], 
				Orbits_on_single_sixes->orbit_len[i]);
			ost << "$$" << endl;
			ost << "" << endl; 
			ost << "\\begin{center}" << endl;
			ost << "\\input " << fname << endl; 
			ost << "\\end{center}" << endl;
			ost << "" << endl; 
			}


		}

	
	ost << "\\subsection*{Orbits on tritangent planes}" << endl;
	Orbits_on_tritangent_planes->print_and_list_orbits_sorted_by_length_tex(ost);

	ost << "\\subsection*{Orbits on trihedral pairs}" << endl;
	Orbits_on_trihedral_pairs->print_and_list_orbits_sorted_by_length_tex(ost);

}

void surface_object_with_action::compute_quartic(INT pt_orbit, 
	INT &pt_A, INT &pt_B, INT *transporter, 
	INT *equation, INT *equation_nice, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT *Elt;

	if (f_v) {
		cout << "surface_object_with_action::compute_quartic" << endl;
		cout << "pt_orbit=" << pt_orbit << endl;
		}
	
	if (Orbits_on_points_not_on_lines == NULL) {
		cout << "surface_object_with_action::compute_quartic "
				"Orbits_on_points_not_on_lines has not been computed" << endl;
		exit(1);
		}
	if (pt_orbit >= Orbits_on_points_not_on_lines->nb_orbits) {
		cout << "surface_object_with_action::compute_quartic "
				"pt_orbit >= Orbits_on_points_not_on_lines->nb_orbits" << endl;
		exit(1);
		}
	INT v[4];
	INT i;

	//Elt = NEW_INT(Surf_A->A->elt_size_in_INT);
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	v[3] = 0;
	pt_B = Surf->rank_point(v);
	i = Orbits_on_points_not_on_lines->orbit[0];
	pt_A = SO->Pts_not_on_lines[i];

	cout << "surface_object_with_action::compute_quartic "
			"pt_A = " << pt_A << " pt_B=" << pt_B << endl;
	
	Surf_A->A->Strong_gens->make_element_which_moves_a_point_from_A_to_B(
		Surf_A->A,
		pt_A, pt_B, transporter, verbose_level);

	cout << "surface_object_with_action::compute_quartic "
			"transporter element=" << endl;
	Surf_A->A->element_print_quick(transporter, cout);
	
	Surf_A->AonHPD_3_4->compute_image_INT_low_level(
			transporter, equation /*INT *input*/,
			equation_nice /* INT *output */, verbose_level);
	cout << "surface_object_with_action::compute_quartic "
			"equation_nice=" << endl;
	Surf->Poly3_4->print_equation(cout, equation_nice);
	cout << endl;

	
	//FREE_INT(Elt);
	if (f_v) {
		cout << "surface_object_with_action::compute_quartic" << endl;
		}
}


void surface_object_with_action::quartic(
		ostream &ost, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT *Elt;

	if (f_v) {
		cout << "surface_object_with_action::quartic" << endl;
		}
	

	if (Orbits_on_points_not_on_lines->nb_orbits == 0) {
		return;
		}

	INT equation_nice[20];
	INT *transporter;
	INT i, a;
	INT v[4];
	INT pt_A, pt_B;

	transporter = NEW_INT(Surf_A->A->elt_size_in_INT);

	cout << "surface_object_with_action::quartic "
			"The surface has points not on lines, "
			"we are computing the quartic" << endl;
	compute_quartic(0 /* pt_orbit */, pt_A, pt_B,
			transporter, SO->eqn, equation_nice, verbose_level);

	cout << "surface_object_with_action::quartic "
			"equation_nice=" << endl;
	Surf->Poly3_4->print_equation(cout, equation_nice);
	cout << endl;
	
	ost << "An equivalent surface containing the point (1,0,0,0) "
			"on no line of the surface is obtained by applying "
			"the transformation" << endl;
	ost << "$$" << endl;
	Surf_A->A->element_print_latex(transporter, ost);
	ost << "$$" << endl;
	ost << "Which moves $P_{" << pt_A << "}$ to $P_{" << pt_B << "}$." << endl;
	ost << endl;
	ost << "\\bigskip" << endl;
	ost << endl;
	ost << "The transformed surface is" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal F}^3 &={\\bf \\rm v}(" << endl;
	Surf->Poly3_4->print_equation_with_line_breaks_tex(ost,
			equation_nice, 9 /* nb_terms_per_line */, "\\\\\n&");
	ost << ")" << endl;
	ost << "\\end{align*}" << endl;

	INT *f1;
	INT *f2;
	INT *f3;
	
	cout << "surface_object_with_action::quartic "
			"before Surf->split_nice_equation" << endl;
	Surf->split_nice_equation(equation_nice, f1, f2, f3,
			0 /* verbose_level */);
	cout << "surface_object_with_action::quartic "
			"after Surf->split_nice_equation" << endl;


	cout << "The equation is of the form $x_0^2f_1(x_1,x_2,x_3) "
			"+ x_0f_2(x_1,x_2,x_3) + f_3(x_1,x_2,x_3)$, where" << endl;
	cout << "f1=" << endl;
	Surf->Poly1_x123->print_equation(cout, f1);
	cout << endl;
	cout << "f2=" << endl;
	Surf->Poly2_x123->print_equation(cout, f2);
	cout << endl;
	cout << "f3=" << endl;
	Surf->Poly3_x123->print_equation(cout, f3);
	cout << endl;

	ost << "\\begin{align*}" << endl;
	ost << "f_1 = & ";
	Surf->Poly1_x123->print_equation_with_line_breaks_tex(ost,
			f1, 8 /* nb_terms_per_line */, "\\\\\n");
	ost << "\\\\" << endl;
	ost << "f_2 = & ";
	Surf->Poly2_x123->print_equation_with_line_breaks_tex(ost,
			f2, 8 /* nb_terms_per_line */, "\\\\\n&");
	ost << "\\\\" << endl;
	ost << "f_3 = & ";
	Surf->Poly3_x123->print_equation_with_line_breaks_tex(ost,
			f3, 8 /* nb_terms_per_line */, "\\\\\n");
	ost << "\\\\" << endl;
	ost << "\\end{align*}" << endl;

	INT *Pts_on_surface;
	INT nb_pts_on_surface;
	
	nb_pts_on_surface = SO->nb_pts;
	Pts_on_surface = NEW_INT(nb_pts_on_surface);

	
	cout << "surface_object_with_action::quartic "
			"before Surf_A->A->map_a_set_and_reorder" << endl;
	Surf_A->A->map_a_set_and_reorder(SO->Pts, Pts_on_surface,
			nb_pts_on_surface, transporter, 0 /* verbose_level */);
	for (i = 0; i < nb_pts_on_surface; i++) {
		Surf->unrank_point(v, Pts_on_surface[i]);
		if (Surf->Poly3_4->evaluate_at_a_point(equation_nice, v)) {
			cout << "the transformed point does not satisfy "
					"the transformed equation" << endl;
			exit(1);
			}
		}
	ost << "The points on the moved surface are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_pts_on_surface; i++) {
		Surf->unrank_point(v, Pts_on_surface[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_on_surface[i] << "}=";
		INT_vec_print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;

	for (i = 0; i < nb_pts_on_surface; i++) {
	
		a = Surf->Poly3_4->evaluate_at_a_point_by_rank(
				equation_nice, Pts_on_surface[i]);
		if (a) {
			cout << "error, the transformed point " << i
					<< " does not lie on the transformed surface" << endl;
			exit(1);
			}
		}
	ost << "The points satisfy the equation of the moved surface.\\\\" << endl;

	


	INT *curve;
	INT *poly1;
	INT *poly2;
	INT two, four, mfour;

	curve = NEW_INT(Surf->Poly4_x123->nb_monomials);
	poly1 = NEW_INT(Surf->Poly4_x123->nb_monomials);
	poly2 = NEW_INT(Surf->Poly4_x123->nb_monomials);
	Surf->multiply_Poly2_3_times_Poly2_3(f2, f2, poly1,
			0 /* verbose_level */);
	Surf->multiply_Poly1_3_times_Poly3_3(f1, f3, poly2,
			0 /* verbose_level */);
	two = F->add(1, 1);
	four = F->add(two, two);
	mfour = F->negate(four);
	F->scalar_multiply_vector_in_place(mfour, poly2,
			Surf->Poly4_x123->nb_monomials);
	F->add_vector(poly1, poly2, curve, Surf->Poly4_x123->nb_monomials);
	
	INT *tangent_quadric;

	cout << "surface_object_with_action::quartic before "
			"Surf->assemble_tangent_quadric" << endl;
	Surf->assemble_tangent_quadric(f1, f2, f3,
			tangent_quadric, verbose_level);

	ost << "The tangent quadric is given as" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal C}_2 = & {\\rm \\bf v}(2x_0 \\cdot f_1 + f_2) "
			"= {\\rm \\bf v}(";
	Surf->Poly2_4->print_equation_with_line_breaks_tex(
			ost, tangent_quadric, 8 /* nb_terms_per_line */, "\\\\\n&");
	ost << ")\\\\" << endl;
	ost << "\\end{align*}" << endl;

	INT *Pts_on_tangent_quadric;
	INT nb_pts_on_tangent_quadric;
	
	Pts_on_tangent_quadric = NEW_INT(Surf->P->N_points);
	
	cout << "surface_object_with_action::quartic "
			"before Surf->Poly2_4->enumerate_points" << endl;
	Surf->Poly2_4->enumerate_points(tangent_quadric,
			Pts_on_tangent_quadric, nb_pts_on_tangent_quadric,
			0 /* verbose_level */);
	cout << "We found " << nb_pts_on_tangent_quadric
			<< " points on the tangent quadric." << endl;

	ost << "The tangent quadric has " << nb_pts_on_tangent_quadric
			<< " points.\\\\" << endl;

	INT_vec_heapsort(Pts_on_tangent_quadric, nb_pts_on_tangent_quadric);
	ost << "The points on the tangent quadric are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_pts_on_tangent_quadric; i++) {
		Surf->unrank_point(v, Pts_on_tangent_quadric[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_on_tangent_quadric[i] << "}=";
		INT_vec_print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;


	INT *line_type;
	
	line_type = NEW_INT(Surf->P->N_lines);

	Surf->P->line_intersection_type(Pts_on_tangent_quadric,
			nb_pts_on_tangent_quadric, line_type, verbose_level);


	INT *type_collected;

	type_collected = NEW_INT(nb_pts_on_tangent_quadric + 1);

	INT_vec_zero(type_collected, nb_pts_on_tangent_quadric + 1);
	for (i = 0; i < Surf->P->N_lines; i++) {
		type_collected[line_type[i]]++;
		}




	ost << "The line type of the tangent quadric is:" << endl;
	ost << "$$" << endl;
	for (i = 0; i <= nb_pts_on_tangent_quadric; i++) {
		if (type_collected[i] == 0) {
			continue;
			}
		
		ost << i << "^{" << type_collected[i] <<"}";
		
		ost << ", \\;" << endl;
		}
	ost << "$$" << endl;

	classify C;
	INT *Class_pts;
	INT nb_class_pts;

	C.init(line_type, Surf->P->N_lines, FALSE, 0);
	C.get_class_by_value(Class_pts, nb_class_pts,
			q + 1 /* value */, 0 /* verbose_level */);
	


	INT *Pts_intersection;
	INT nb_pts_intersection;

	INT_vec_intersect(Pts_on_surface, nb_pts_on_surface, 
		Pts_on_tangent_quadric, nb_pts_on_tangent_quadric, 
		Pts_intersection, nb_pts_intersection);


	ost << "The tangent quadric intersects the cubic surface in "
			<< nb_pts_intersection << " points." << endl;

	
	ost << "The intersection points are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_pts_intersection; i++) {
		Surf->unrank_point(v, Pts_intersection[i]);
		ost << i << " : $P_{" << i << "} = P_{" << Pts_intersection[i] << "}=";
		INT_vec_print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;



	ost << "The quartic curve is given as" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal C}_4 = & {\\rm \\bf v}(";
	Surf->Poly4_x123->print_equation_with_line_breaks_tex(ost,
			curve, 10 /* nb_terms_per_line */, "\\\\\n&");
	ost << ")\\\\" << endl;
	ost << "\\end{align*}" << endl;


#if 1
	INT *Pts_on_curve;
	INT sz_curve;


	Pts_on_curve = NEW_INT(Surf->P2->N_points);

	cout << "surface_object_with_action::quartic before "
			"Surf->Poly4_x123->enumerate_points" << endl;
	Surf->Poly4_x123->enumerate_points(curve, Pts_on_curve,
			sz_curve, 0 /* verbose_level */);
	cout << "We found " << sz_curve
			<< " points on the quartic quadric." << endl;

	ost << "The " << sz_curve
			<< " points on the quartic curve are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < sz_curve; i++) {
		Surf->P2->unrank_point(v, Pts_on_curve[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_on_curve[i] << "}=";
		INT_vec_print_fully(ost, v, 3);
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;

#else
	INT *Pts_on_curve;
	INT sz_curve;

	sz_curve = nb_pts_intersection - 1;
	Pts_on_curve = NEW_INT(sz_curve);

	
	// skip the first point (1,0,0,0):
	for (i = 1; i < nb_pts_intersection; i++) {
		Surf->unrank_point(v, Pts_intersection[i]);
		Pts_on_curve[i - 1] = Surf->P2->rank_point(v + 1);
		}

	ost << "The " << sz_curve << " projected points are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < sz_curve; i++) {
		Surf->P2->unrank_point(v, Pts_on_curve[i]);
		ost << i << " : $P_{" << i << "} = P_{" << Pts_on_curve[i] << "}=";
		INT_vec_print_fully(ost, v, 3);
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;

	
	INT r;
	INT *Kernel;
	
	Kernel = NEW_INT(Surf->Poly4_x123->nb_monomials
				* Surf->Poly4_x123->nb_monomials);
	Surf->Poly4_x123->vanishing_ideal(Pts_on_curve,
			sz_curve, r, Kernel, verbose_level);
	cout << "r=" << r << endl;
	ost << "The quartics have "
			<< Surf->Poly4_x123->nb_monomials << " terms.\\\\" << endl;
	ost << "The kernel has dimension "
			<< Surf->Poly4_x123->nb_monomials - r << " .\\\\" << endl;
#endif


	strong_generators *gens_copy;
	set_and_stabilizer *moved_surface;
	//strong_generators *stab_gens_moved_surface;
	strong_generators *stab_gens_P0;


	gens_copy = Aut_gens->create_copy();

	moved_surface = NEW_OBJECT(set_and_stabilizer);

	cout << "creating moved_surface" << endl;
	moved_surface->init_everything(Surf_A->A,
		Surf_A->A, SO->Pts, SO->nb_pts,
		gens_copy, 0 /*verbose_level */);

	//stab_gens_moved_surface = SaS->Strong_gens->create_copy();

	cout << "before apply_to_self" << endl;
	moved_surface->apply_to_self(transporter,
			0 /* verbose_level */);

	cout << "before moved_surface->Strong_gens->point_stabilizer" << endl;
	stab_gens_P0 = moved_surface->Strong_gens->point_stabilizer(
			0 /*INT pt */, verbose_level);
	
	ost << "The stabilizer of $P0$ and the moved surface "
			"is the following group:\\\\" << endl;
	stab_gens_P0->print_generators_tex(ost);
			
}


void surface_object_with_action::cheat_sheet(ostream &ost, 
	const char *label_txt, const char *label_tex, 
	INT f_print_orbits, const char *fname_mask, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet" << endl;
		}
	
	cout << "surface_object_with_action::cheat_sheet "
			"before New_clebsch->init_surface_equation_given" << endl;





	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before SO->print_equation" << endl;
		}
	SO->print_equation(ost);
	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"after SO->print_equation" << endl;
		}


	longinteger_object ago;
	Aut_gens->group_order(ago);
	ost << "The automorphism group has order "
			<< ago << "\\\\" << endl;
	ost << "The automorphism group is:\\\\" << endl;
	if (f_v) {
		cout << "cheat_sheet before Aut_gens->"
				"print_generators_tex" << endl;
		}
	Aut_gens->print_generators_tex(ost);

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before SO->print_general" << endl;
		}
	SO->print_general(ost);


	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before SO->print_lines" << endl;
		}
	SO->print_lines(ost);


	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before SO->print_points" << endl;
		}
	SO->print_points(ost);


	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before SO->print_lines_with_points_on_them" << endl;
		}
	SO->print_lines_with_points_on_them(ost);



	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before SO->print_line_intersection_graph" << endl;
		}
	SO->print_line_intersection_graph(ost);



	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before SO->print_tritangent_planes" << endl;
		}
	SO->print_tritangent_planes(ost);


	//SO->print_planes_in_trihedral_pairs(ost);

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before SO->print_generalized_quadrangle" << endl;
		}
	SO->print_generalized_quadrangle(ost);

	
	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before SO->print_double sixes" << endl;
		}
	SO->print_double_sixes(ost);

	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before SO->print_trihedral_pairs" << endl;
		}
	SO->print_trihedral_pairs(ost);

	//SO->latex_table_of_trihedral_pairs_and_clebsch_system(
	//*Clebsch->ost, AL->T_idx, AL->nb_T);





	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet "
				"before print_automorphism_group" << endl;
		}
	print_automorphism_group(ost, 
		f_print_orbits, fname_mask);
	

#if 0
	if (SO->nb_pts_not_on_lines) {
		
		if (f_v) {
			cout << "surface_object_with_action::cheat_sheet "
					"before cheat_sheet_quartic_curve" << endl;
			}
		cheat_sheet_quartic_curve(ost, 
			label_txt, label_tex, verbose_level);
		if (f_v) {
			cout << "surface_object_with_action::cheat_sheet "
					"after cheat_sheet_quartic_curve" << endl;
			}

		}
#endif

	ost << "\\clearpage\\subsection*{The Elements of "
			"the Automorphism Group}" << endl;
	Aut_gens->print_elements_latex_ost(ost);

	ost << "\\clearpage\\subsection*{The Group Table}" << endl;
	INT go;
	INT block_width = 24;

	go = Aut_gens->group_order_as_INT();
	if (go < 50) {
		INT *Table;
		Aut_gens->create_group_table(Table, go, verbose_level - 1);
		print_integer_matrix_tex_block_by_block(ost,
				Table, go, go, block_width);
		FREE_INT(Table);
		}
	else {
		ost << "Too big to print." << endl;
		}


	char magma_fname[1000];

	sprintf(magma_fname, "%s_group.magma", label_txt);
	Aut_gens->export_permutation_group_to_magma(
			magma_fname, verbose_level - 2);
	if (f_v) {
		cout << "written file " << magma_fname << " of size "
				<< file_size(magma_fname) << endl;
		}

	ost << "\\clearpage\\subsection*{Magma Export}" << endl;
	ost << "To export the group to Magma, use the following file\\\\" << endl;
	ost << "\\begin{verbatim}" << endl;
	
	{
	ifstream fp1(magma_fname);
	char line[100000];

	while (TRUE) {
		if (fp1.eof()) {
			break;
			}
	
		//cout << "count_number_of_orbits_in_file reading
		//line, nb_sol = " << nb_sol << endl;
		fp1.getline(line, 100000, '\n');
		ost << line << endl;
		}
	
	}
	ost << "\\end{verbatim}" << endl;
	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet done" << endl;
		}
	

}


void surface_object_with_action::cheat_sheet_quartic_curve(
	ostream &ost,
	const char *label_txt, const char *label_tex,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_with_action::"
				"cheat_sheet_quartic_curve" << endl;
		}
	

	INT equation_nice[20];
	INT *transporter;
	INT *f1;
	INT *f2;
	INT *f3;
	INT *Pts_on_surface;
	INT nb_pts_on_surface;
	INT *curve;
	INT *poly1;
	INT *poly2;
	INT *tangent_quadric;
	INT *Pts_on_tangent_quadric;
	INT *Pts_intersection;
	INT *Pts_on_curve;
	INT sz_curve;
	INT nb_pts_intersection;
	INT nb_pts_on_tangent_quadric;
	strong_generators *gens_copy;
	set_and_stabilizer *moved_surface;
	strong_generators *stab_gens_P0;

	INT two, four, mfour;
	INT i;
	INT v[4];
	INT pt_A, pt_B;

	transporter = NEW_INT(Surf_A->A->elt_size_in_INT);

	cout << "surface_object_with_action::cheat_sheet_quartic_curve "
			"The surface has points not on lines, we are computing "
			"the quartic" << endl;
	compute_quartic(0 /* pt_orbit */, pt_A, pt_B, transporter,
			SO->eqn, equation_nice, verbose_level);

	cout << "surface_object_with_action::cheat_sheet_quartic_curve "
			"equation_nice=" << endl;
	Surf->Poly3_4->print_equation(cout, equation_nice);
	cout << endl;
	
	ost << "An equivalent surface containing the point (1,0,0,0) "
			"on no line of the surface is obtained by applying "
			"the transformation" << endl;
	ost << "$$" << endl;
	Surf_A->A->element_print_latex(transporter, ost);
	ost << "$$" << endl;
	ost << "Which moves $P_{" << pt_A << "}$ to $P_{"
			<< pt_B << "}$." << endl;
	ost << endl;
	ost << "\\bigskip" << endl;
	ost << endl;
	ost << "The transformed surface is" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal F}^3 &={\\bf \\rm v}(" << endl;
	Surf->Poly3_4->print_equation_with_line_breaks_tex(ost,
			equation_nice, 9 /* nb_terms_per_line */, "\\\\\n&");
	ost << ")" << endl;
	ost << "\\end{align*}" << endl;

	
	cout << "surface_object_with_action::cheat_sheet_quartic_curve "
			"before Surf->split_nice_equation" << endl;
	Surf->split_nice_equation(equation_nice, f1, f2, f3,
			0 /* verbose_level */);
	cout << "surface_object_with_action::cheat_sheet_quartic_curve "
			"after Surf->split_nice_equation" << endl;


	ost << "The equation is of the form $x_0^2f_1(x_1,x_2,x_3) "
			"+ x_0f_2(x_1,x_2,x_3) + f_3(x_1,x_2,x_3)$, where" << endl;
	cout << "f1=" << endl;
	Surf->Poly1_x123->print_equation(cout, f1);
	cout << endl;
	cout << "f2=" << endl;
	Surf->Poly2_x123->print_equation(cout, f2);
	cout << endl;
	cout << "f3=" << endl;
	Surf->Poly3_x123->print_equation(cout, f3);
	cout << endl;

	ost << "\\begin{align*}" << endl;
	ost << "f_1 = & ";
	Surf->Poly1_x123->print_equation_with_line_breaks_tex(ost,
			f1, 8 /* nb_terms_per_line */, "\\\\\n");
	ost << "\\\\" << endl;
	ost << "f_2 = & ";
	Surf->Poly2_x123->print_equation_with_line_breaks_tex(ost,
			f2, 8 /* nb_terms_per_line */, "\\\\\n&");
	ost << "\\\\" << endl;
	ost << "f_3 = & ";
	Surf->Poly3_x123->print_equation_with_line_breaks_tex(ost,
			f3, 8 /* nb_terms_per_line */, "\\\\\n");
	ost << "\\\\" << endl;
	ost << "\\end{align*}" << endl;

	
	nb_pts_on_surface = SO->nb_pts;
	Pts_on_surface = NEW_INT(nb_pts_on_surface);

	
	cout << "surface_object_with_action::cheat_sheet_quartic_curve "
			"before Surf_A->A->map_a_set_and_reorder" << endl;
	Surf_A->A->map_a_set_and_reorder(SO->Pts, Pts_on_surface,
			nb_pts_on_surface, transporter, 0 /* verbose_level */);
	for (i = 0; i < nb_pts_on_surface; i++) {
		Surf->unrank_point(v, Pts_on_surface[i]);
		if (Surf->Poly3_4->evaluate_at_a_point(equation_nice, v)) {
			cout << "the transformed point does not satisfy "
					"the transformed equation" << endl;
			exit(1);
			}
		}
	ost << "The points on the moved surface are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_pts_on_surface; i++) {
		Surf->unrank_point(v, Pts_on_surface[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_on_surface[i] << "}=";
		INT_vec_print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;




	curve = NEW_INT(Surf->Poly4_x123->nb_monomials);
	poly1 = NEW_INT(Surf->Poly4_x123->nb_monomials);
	poly2 = NEW_INT(Surf->Poly4_x123->nb_monomials);
	Surf->multiply_Poly2_3_times_Poly2_3(f2, f2, poly1, 0 /* verbose_level */);
	Surf->multiply_Poly1_3_times_Poly3_3(f1, f3, poly2, 0 /* verbose_level */);
	two = F->add(1, 1);
	four = F->add(two, two);
	mfour = F->negate(four);
	F->scalar_multiply_vector_in_place(mfour, poly2,
			Surf->Poly4_x123->nb_monomials);
	F->add_vector(poly1, poly2, curve,
			Surf->Poly4_x123->nb_monomials);
	

	cout << "surface_object_with_action::cheat_sheet_quartic_curve "
			"before Surf->assemble_tangent_quadric" << endl;
	Surf->assemble_tangent_quadric(f1, f2, f3, tangent_quadric, verbose_level);

	ost << "The tangent quadric is given as" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal C}_2 = & {\\rm \\bf v}(2x_0 \\cdot f_1 + f_2) = {\\rm \\bf v}(";
	Surf->Poly2_x123->print_equation_with_line_breaks_tex(ost,
			tangent_quadric, 8 /* nb_terms_per_line */, "\\\\\n&");
	ost << ")\\\\" << endl;
	ost << "\\end{align*}" << endl;
	
	Pts_on_tangent_quadric = NEW_INT(Surf->P->N_points);
	
	cout << "surface_object_with_action::cheat_sheet_quartic_curve "
			"before Surf->Poly2_4->enumerate_points" << endl;
	Surf->Poly2_4->enumerate_points(tangent_quadric,
			Pts_on_tangent_quadric, nb_pts_on_tangent_quadric,
			0 /* verbose_level */);
	cout << "We found " << nb_pts_on_tangent_quadric
			<< " points on the tangent quadric." << endl;

	ost << "The tangent quadric has " << nb_pts_on_tangent_quadric
			<< " points.\\\\" << endl;

	INT_vec_heapsort(Pts_on_tangent_quadric, nb_pts_on_tangent_quadric);
	ost << "The points on the tangent quadric are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_pts_on_tangent_quadric; i++) {
		Surf->unrank_point(v, Pts_on_tangent_quadric[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_on_tangent_quadric[i] << "}=";
		INT_vec_print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;





	INT_vec_intersect(Pts_on_surface, nb_pts_on_surface, 
		Pts_on_tangent_quadric, nb_pts_on_tangent_quadric, 
		Pts_intersection, nb_pts_intersection);


	ost << "The tangent quadric intersects the cubic surface in "
			<< nb_pts_intersection << " points." << endl;

	
	ost << "The intersection points are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_pts_intersection; i++) {
		Surf->unrank_point(v, Pts_intersection[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_intersection[i] << "}=";
		INT_vec_print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;



	ost << "The quartic curve is given as" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal C}_4 = & {\\rm \\bf v}(";
	Surf->Poly4_x123->print_equation_with_line_breaks_tex(
			ost, curve, 10 /* nb_terms_per_line */, "\\\\\n&");
	ost << ")\\\\" << endl;
	ost << "\\end{align*}" << endl;


#if 1


	Pts_on_curve = NEW_INT(Surf->P2->N_points);

	cout << "surface_object_with_action::cheat_sheet_quartic_curve "
			"before Surf->Poly4_x123->enumerate_points" << endl;
	Surf->Poly4_x123->enumerate_points(curve,
			Pts_on_curve, sz_curve, 0 /* verbose_level */);
	cout << "We found " << sz_curve << " points on "
			"the quartic quadric." << endl;

	ost << "The " << sz_curve << " points on the "
			"quartic curve are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < sz_curve; i++) {
		Surf->P2->unrank_point(v, Pts_on_curve[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_on_curve[i] << "}=";
		INT_vec_print_fully(ost, v, 3);
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;

#else

	sz_curve = nb_pts_intersection - 1;
	Pts_on_curve = NEW_INT(sz_curve);

	
	// skip the first point (1,0,0,0):
	for (i = 1; i < nb_pts_intersection; i++) {
		Surf->unrank_point(v, Pts_intersection[i]);
		Pts_on_curve[i - 1] = Surf->P2->rank_point(v + 1);
		}

	ost << "The " << sz_curve << " projected points are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < sz_curve; i++) {
		Surf->P2->unrank_point(v, Pts_on_curve[i]);
		ost << i << " : $P_{" << i << "} = P_{" << Pts_on_curve[i] << "}=";
		INT_vec_print_fully(ost, v, 3);
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;

	
	INT r;
	INT *Kernel;
	
	Kernel = NEW_INT(Surf->Poly4_x123->nb_monomials *
			Surf->Poly4_x123->nb_monomials);
	Surf->Poly4_x123->vanishing_ideal(Pts_on_curve,
			sz_curve, r, Kernel, verbose_level);
	cout << "r=" << r << endl;
	ost << "The quartics have " << Surf->Poly4_x123->nb_monomials
			<< " terms.\\\\" << endl;
	ost << "The kernel has dimension "
			<< Surf->Poly4_x123->nb_monomials - r << " .\\\\" << endl;
	FREE_INT(Kernel);
#endif




	gens_copy = Aut_gens->create_copy();
	//gens_copy = New_clebsch->SaS->Strong_gens->create_copy();

	moved_surface = NEW_OBJECT(set_and_stabilizer);

	cout << "creating moved_surface" << endl;
	moved_surface->init_everything(Surf_A->A,
		Surf_A->A, SO->Pts, SO->nb_pts,
		gens_copy, 0 /*verbose_level */);

	//stab_gens_moved_surface = SaS->Strong_gens->create_copy();

	cout << "before apply_to_self" << endl;
	moved_surface->apply_to_self(transporter,
			0 /* verbose_level */);

	cout << "before moved_surface->Strong_gens->point_stabilizer"
			<< endl;
	stab_gens_P0 = moved_surface->Strong_gens->point_stabilizer(
			0 /*INT pt */, verbose_level);
	
	ost << "The stabilizer of $P0$ and the moved surface is "
			"the following group:\\\\" << endl;
	stab_gens_P0->print_generators_tex(ost);

	FREE_INT(transporter);
	FREE_INT(f1);
	FREE_INT(f2);
	FREE_INT(f3);
	FREE_INT(Pts_on_surface);
	FREE_INT(curve);
	FREE_INT(poly1);
	FREE_INT(poly2);
	FREE_INT(tangent_quadric);
	FREE_INT(Pts_on_tangent_quadric);
	FREE_INT(Pts_intersection);
	FREE_INT(Pts_on_curve);
	FREE_OBJECT(gens_copy);
	FREE_OBJECT(moved_surface);
	FREE_OBJECT(stab_gens_P0);


	if (f_v) {
		cout << "surface_object_with_action::cheat_sheet_"
				"quartic_curve" << endl;
		}
}

