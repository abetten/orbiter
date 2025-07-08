// surface_object.cpp
// 
// Anton Betten
// March 18, 2017
//
// 
//
//

#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace algebraic_geometry {





surface_object::surface_object()
{
	Record_birth();
	q = 0;
	F = NULL;
	Surf = NULL;

	//std::string label_txt;
	//std::string label_tex;

	Variety_object = NULL;
	SOP = NULL;

	//null();


}

surface_object::~surface_object()
{
	Record_death();
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::~surface_object" << endl;
	}
	if (Variety_object) {
		FREE_OBJECT(Variety_object);
	}
	if (SOP) {
		FREE_OBJECT(SOP);
	}

	if (f_v) {
		cout << "surface_object::~surface_object done" << endl;
	}
}

void surface_object::init_variety_object(
		surface_domain *Surf,
		variety_object *Variety_object,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::init_variety_object" << endl;
	}

	surface_object::Surf = Surf;
	F = Surf->F;
	q = F->q;

	surface_object::Variety_object = Variety_object;
	surface_object::label_txt = Variety_object->label_txt;
	surface_object::label_tex = Variety_object->label_tex;

	if (f_v) {
		cout << "surface_object::init_variety_object done" << endl;
	}
}

void surface_object::init_equation_points_and_lines_only(
		surface_domain *Surf, int *eqn,
		std::string &label_txt,
		std::string &label_tex,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::init_equation_points_and_lines_only" << endl;
		Int_vec_print(cout, eqn, 20);
		cout << endl;
	}

	surface_object::Surf = Surf;
	F = Surf->F;
	q = F->q;

	surface_object::label_txt = label_txt;
	surface_object::label_tex = label_tex;

	Variety_object = NEW_OBJECT(variety_object);

	if (f_v) {
		cout << "surface_object::init_equation_points_and_lines_only "
				"before Variety_object->init_equation_only" << endl;
	}
	Variety_object->init_equation_only(
			Surf->P,
			Surf->PolynomialDomains->Poly3_4,
			eqn,
			verbose_level);
	if (f_v) {
		cout << "surface_object::init_equation_points_and_lines_only "
				"after Variety_object->init_equation_only" << endl;
	}

	// does not enumerate the points and lines




	if (f_v) {
		cout << "surface_object::init_equation_points_and_lines_only "
				"before enumerate_points_and_lines" << endl;
	}
	enumerate_points_and_lines(verbose_level - 1);
	if (f_v) {
		cout << "surface_object::init_equation_points_and_lines_only "
				"after enumerate_points_and_lines" << endl;
	}


#if 0
	if (nb_points != Surf->nb_pts_on_surface) {
		cout << "surface_object::init_equation nb_points != "
				"Surf->nb_pts_on_surface" << endl;
		cout << "Surf->nb_pts_on_surface=" << Surf->nb_pts_on_surface << endl;
		//FREE_lint(Points);
		Points = NULL;
		return false;
	}
#endif
	if (f_v) {
		cout << "surface_object::init_equation_points_and_lines_only done" << endl;
	}
}


void surface_object::init_equation(
		surface_domain *Surf, int *eqn,
		std::string &label_txt,
		std::string &label_tex,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::init_equation" << endl;
	}

	if (f_v) {
		cout << "surface_object::init_equation "
				"before init_equation_points_and_lines_only" << endl;
	}
	init_equation_points_and_lines_only(
			Surf, eqn, label_txt, label_tex,
			verbose_level);
	if (f_v) {
		cout << "surface_object::init_equation "
				"after init_equation_points_and_lines_only" << endl;
	}

	
	if (Variety_object->Line_sets->Set_size[0] == 27) {
		if (f_v) {
			cout << "surface_object::init_equation before "
					"find_double_six_and_rearrange_lines" << endl;
		}
		find_double_six_and_rearrange_lines(
				Variety_object->Line_sets->Sets[0],
				verbose_level - 2);

		if (f_v) {
			cout << "surface_object::init_equation after "
					"find_double_six_and_rearrange_lines" << endl;
			cout << "surface_object::init_equation Lines:";
			Lint_vec_print(cout, Variety_object->Line_sets->Sets[0], 27);
			cout << endl;
		}
	}
	else {
		cout << "The surface does not have 27 lines. "
				"nb_lines=" << Variety_object->Line_sets->Set_size[0]
				<< " A double six has not been computed" << endl;
	}


	if (f_v) {
		cout << "surface_object::init_equation before "
				"compute_properties" << endl;
	}
	compute_properties(0/*verbose_level - 2*/);
	if (f_v) {
		cout << "surface_object::init_equation after "
				"compute_properties" << endl;
	}



	if (f_v) {
		cout << "surface_object::init_equation after "
				"enumerate_points" << endl;
	}
}



void surface_object::enumerate_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::enumerate_points" << endl;
	}

	vector<long int> Points;

	if (f_v) {
		cout << "surface_object::enumerate_points before "
				"Surf->enumerate_points" << endl;
	}
	Surf->enumerate_points(
			Variety_object->eqn,
		Points,
		0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "surface_object::enumerate_points after "
				"Surf->enumerate_points" << endl;
	}
	if (f_v) {
		cout << "surface_object::enumerate_points The surface "
				"has " << Points.size() << " points" << endl;
	}
	int i;

	int nb_pts;
	long int *Pts;

	nb_pts = Points.size();
	Pts = NEW_lint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Points[i];
	}
	if (Variety_object->Point_sets) {
		FREE_OBJECT(Variety_object->Point_sets);
	}
	Variety_object->Point_sets = NEW_OBJECT(other::data_structures::set_of_sets);

	Variety_object->Point_sets->init_single(
			Variety_object->Projective_space->Subspaces->N_points /* underlying_set_size */,
			Pts, nb_pts, 0 /* verbose_level */);

	FREE_lint(Pts);


	if (f_v) {
		cout << "surface_object::enumerate_points done" << endl;
	}
}



void surface_object::enumerate_points_and_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::enumerate_points_and_lines" << endl;
	}

	geometry::other_geometry::geometry_global Geo;
	vector<long int> Points;
	vector<long int> The_Lines;

	if (f_v) {
		cout << "surface_object::enumerate_points_and_lines "
				"before Surf->enumerate_points" << endl;
	}
	Surf->enumerate_points(
			Variety_object->eqn,
		Points,
		0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "surface_object::enumerate_points_and_lines "
				"after Surf->enumerate_points" << endl;
	}
	if (f_v) {
		cout << "surface_object::enumerate_points_and_lines "
				"The surface has " << Points.size() << " points" << endl;
	}


	if (f_v) {
		cout << "surface_object::enumerate_points_and_lines before "
				"Geo.find_lines_which_are_contained" << endl;
	}
	Geo.find_lines_which_are_contained(
			Surf->P,
			Points, The_Lines, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "surface_object::enumerate_points_and_lines after "
				"Geo.find_lines_which_are_contained" << endl;
	}

	if (f_v) {
		cout << "surface_object::enumerate_points_and_lines The surface "
				"has " << The_Lines.size() << " lines" << endl;
	}
#if 0
	if (nb_lines != 27) {
		cout << "surface_object::enumerate_points_and_lines the surface "
				"does not have 27 lines" << endl;
		//FREE_lint(Points);
		//Points = NULL;
		return false;
	}
#endif


#if 1
	if (F->q == 2) {
		if (f_v) {
			cout << "surface_object::enumerate_points_and_lines "
					"before find_real_lines" << endl;
		}

		find_real_lines(The_Lines, verbose_level);

		if (f_v) {
			cout << "surface_object::enumerate_points_and_lines "
					"after find_real_lines" << endl;
		}
	}
#endif

	int i;
	long int *Pts;
	int nb_pts;
	long int *Lines;
	int nb_lines;

	nb_pts = Points.size();
	Pts = NEW_lint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Points[i];
	}

	nb_lines = The_Lines.size();
	Lines = NEW_lint(nb_lines);
	for (i = 0; i < nb_lines; i++) {
		Lines[i] = The_Lines[i];
	}

	if (f_v) {
		cout << "surface_object::enumerate_points_and_lines "
				"nb_pts=" << nb_pts << " nb_lines=" << nb_lines << endl;
		cout << "Lines:";
		Lint_vec_print(cout, Lines, nb_lines);
		cout << endl;
	}


	if (Variety_object->Point_sets) {
		FREE_OBJECT(Variety_object->Point_sets);
	}
	Variety_object->Point_sets = NEW_OBJECT(other::data_structures::set_of_sets);

	Variety_object->Point_sets->init_single(
			Variety_object->Projective_space->Subspaces->N_points /* underlying_set_size */,
			Pts, nb_pts, 0 /* verbose_level */);

	FREE_lint(Pts);

	if (Variety_object->Line_sets) {
		FREE_OBJECT(Variety_object->Line_sets);
	}
	Variety_object->Line_sets = NEW_OBJECT(other::data_structures::set_of_sets);

	Variety_object->Line_sets->init_single(
			Variety_object->Projective_space->Subspaces->N_lines /* underlying_set_size */,
			Lines, nb_lines, 0 /* verbose_level */);

	FREE_lint(Lines);


	if (f_v) {
		cout << "surface_object::enumerate_points_and_lines done" << endl;
	}
}

void surface_object::find_real_lines(
		std::vector<long int> &The_Lines,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int rk;
	int M[8];
	int coeff_out[4];

	if (f_v) {
		cout << "surface_object::find_real_lines" << endl;
	}
	for (i = 0, j = 0; i < The_Lines.size(); i++) {
		rk = The_Lines[i];
		Surf->P->Subspaces->Grass_lines->unrank_lint_here(
				M, rk, 0 /* verbose_level */);
		if (f_v) {
			cout << "surface_object::find_real_lines testing line" << endl;
			Int_matrix_print(M, 2, 4);
		}

		Surf->PolynomialDomains->Poly3_4->substitute_line(
			Variety_object->eqn /* coeff_in */,
			coeff_out,
			M /* Pt1_coeff */, M + 4 /* Pt2_coeff */,
			verbose_level - 3);
		// coeff_in[nb_monomials], coeff_out[degree + 1]

		if (f_v) {
			cout << "surface_object::find_real_lines coeff_out=";
			Int_vec_print(cout, coeff_out, 4);
			cout << endl;
		}
		if (!Int_vec_is_zero(coeff_out, 4)) {
			if (f_v) {
				cout << "surface_object::find_real_lines not a real line" << endl;
			}
		}
		else {
			The_Lines[j] = rk;
			j++;
		}
	}
	The_Lines.resize(j);
	if (f_v) {
		cout << "surface_object::find_real_lines done" << endl;
	}
}

void surface_object::init_with_27_lines(
		surface_domain *Surf,
	long int *Lines27, int *eqn,
	std::string &label_txt,
	std::string &label_tex,
	int f_find_double_six_and_rearrange_lines, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::init_with_27_lines" << endl;
	}
	surface_object::Surf = Surf;
	F = Surf->F;
	q = F->q;

	surface_object::label_txt = label_txt;
	surface_object::label_tex = label_tex;


	//Int_vec_copy(eqn, surface_object::eqn, 20);
	Variety_object = NEW_OBJECT(variety_object);

	Variety_object->init_equation_only(
			Surf->P,
			Surf->PolynomialDomains->Poly3_4,
			eqn,
			verbose_level);



	if (f_v) {
		cout << "surface_object::init_with_27_lines "
				"before enumerate_points" << endl;
	}
	enumerate_points(verbose_level - 2);
	if (f_v) {
		cout << "surface_object::init_with_27_lines "
				"after enumerate_points" << endl;
	}


	//nb_lines = 27;
	//surface_object::Lines = NEW_lint(27);
	//Lint_vec_copy(Lines27, surface_object::Lines, 27);

	if (Variety_object->Line_sets) {
		FREE_OBJECT(Variety_object->Line_sets);
	}
	Variety_object->Line_sets = NEW_OBJECT(other::data_structures::set_of_sets);

	Variety_object->Line_sets->init_single(
			Variety_object->Projective_space->Subspaces->N_lines /* underlying_set_size */,
			Lines27, 27, 0 /* verbose_level */);


	if (f_v) {
		cout << "surface_object::init_with_27_lines Lines:";
		Lint_vec_print(cout, Variety_object->Line_sets->Sets[0], 27);
		cout << endl;
	}

	if (f_find_double_six_and_rearrange_lines) {
		if (f_v) {
			cout << "surface_object::init_with_27_lines before "
					"find_double_six_and_rearrange_lines" << endl;
		}
		find_double_six_and_rearrange_lines(
				Variety_object->Line_sets->Sets[0],
				verbose_level);
		if (f_v) {
			cout << "surface_object::init_with_27_lines after "
					"find_double_six_and_rearrange_lines" << endl;
		}
	}

	if (f_v) {
		cout << "surface_object::init_with_27_lines Lines:";
		Lint_vec_print(cout, Variety_object->Line_sets->Sets[0], 27);
		cout << endl;
	}



	if (f_v) {
		cout << "surface_object::init_with_27_lines "
				"before compute_properties" << endl;
	}
	compute_properties(verbose_level);
	if (f_v) {
		cout << "surface_object::init_with_27_lines "
				"after compute_properties" << endl;
	}

	if (f_v) {
		cout << "surface_object::init_with_27_lines done" << endl;
	}
}

void surface_object::compute_properties(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::compute_properties" << endl;
	}

	SOP = NEW_OBJECT(surface_object_properties);

	SOP->init(this, verbose_level);

	if (f_v) {
		cout << "surface_object::compute_properties done" << endl;
	}
}

void surface_object::recompute_properties(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::recompute_properties" << endl;
	}

	if (SOP) {
		FREE_OBJECT(SOP);
		SOP = NULL;
	}

	SOP = NEW_OBJECT(surface_object_properties);

	SOP->init(this, verbose_level);

	if (f_v) {
		cout << "surface_object::recompute_properties done" << endl;
	}
}



void surface_object::find_double_six_and_rearrange_lines(
	long int *Lines,
	int verbose_level)
// Lines are given as line ranks in PG(3,q)
// there must be exactly 27 lines.
{
	int f_v = (verbose_level >= 1);
	long int Lines0[27];
	long int Lines1[27];
	long int double_six[12];
	int *Adj;
	other::data_structures::set_of_sets *line_intersections;
	int *Starter_Table;
	int nb_starter;
	int l, line_idx, subset_idx;
	long int S3[6];
	other::data_structures::sorting Sorting;



	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines, "
				"verbose_level = " << verbose_level << endl;
	}
	Lint_vec_copy(Lines, Lines0, 27);
	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines, "
				"Lines = ";
		Lint_vec_print(cout, Lines, 27);
		cout << endl;
	}

	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"before Surf->compute_adjacency_matrix_of_line_intersection_graph" << endl;
	}
	Surf->compute_adjacency_matrix_of_line_intersection_graph(
			Adj, Lines0, 27,
			0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"after Surf->compute_adjacency_matrix_of_line_intersection_graph" << endl;
	}

	line_intersections = NEW_OBJECT(other::data_structures::set_of_sets);

	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"before line_intersections->init_from_adjacency_matrix" << endl;
	}
	line_intersections->init_from_adjacency_matrix(
			27,
			Adj,
			0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"after line_intersections->init_from_adjacency_matrix" << endl;
	}

	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"before Surf->list_starter_configurations" << endl;
	}
	Surf->list_starter_configurations(
			Lines0, 27,
		line_intersections, Starter_Table, nb_starter,
		verbose_level);
	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"after Surf->list_starter_configurations" << endl;
	}

		// 432 = 36 * 12
		// is the number of double sixes with a distinguished line.

	if (nb_starter != 432) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"nb_starter != 432" << endl;
		exit(1);
	}
	l = 0;
	line_idx = Starter_Table[l * 2 + 0];
	subset_idx = Starter_Table[l * 2 + 1];

	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"line_idx = " << line_idx << endl;
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"subset_idx = " << subset_idx << endl;
	}



	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"before Surf->create_starter_configuration" << endl;
	}
	Surf->create_starter_configuration(
			line_idx,
		subset_idx, line_intersections, 
		Lines0, S3, 
		0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"after Surf->create_starter_configuration" << endl;
	}

	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines, "
				"Starter configuration of 6 lines = ";
		Lint_vec_print(cout, S3, 6);
		cout << endl;
	}


	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"before Surf->five_plus_one_to_double_six" << endl;
	}
	if (!Surf->five_plus_one_to_double_six(
		S3, double_six, verbose_level)) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"The starter configuration is bad, there "
				"is no double six" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines after "
				"Surf->five_plus_one_to_double_six" << endl;
	}

	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines, "
				"double_six = ";
		Lint_vec_print(cout, double_six, 12);
		cout << endl;
	}

	Lint_vec_copy(double_six, Lines1, 12);
	
	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"before Surf->create_remaining_fifteen_lines" << endl;
	}
	Surf->create_remaining_fifteen_lines(
			double_six,
			Lines1 + 12,
			verbose_level - 1);
	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"after Surf->create_remaining_fifteen_lines" << endl;
	}

	Lint_vec_copy(Lines1, Lines, 27);

	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines, "
				"double_six and remaining 15 lines = ";
		Lint_vec_print(cout, Lines, 27);
		cout << endl;
	}


	// check that Lines0 and Lines1 are the same set of lines:

	Sorting.lint_vec_heapsort(Lines0, 27);
	Sorting.lint_vec_heapsort(Lines1, 27);

	int i;
	for (i = 0; i < 27; i++) {
		if (Lines0[i] != Lines1[i]) {
			cout << "surface_object::find_double_six_and_rearrange_lines "
					"Lines0[i] != Lines1[i]" << endl;
			exit(1);
		}
	}

	FREE_int(Adj);
	FREE_int(Starter_Table);
	FREE_OBJECT(line_intersections);
	
	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines done" << endl;
	}
}







#if 0
void surface_object::print_generalized_quadrangle(std::ostream &ost)
{
	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{The Generalized Quadrangle}" << endl;

	ost << "The lines in the tritangent planes are:\\\\" << endl;
	int i, j, a, h;

	//ost << "\\begin{multicols}{1}" << endl;
	ost << "\\noindent" << endl;
	for (i = 0; i < nb_tritangent_planes; i++) {
		j = Eckardt_to_Tritangent_plane[i];
		a = Tritangent_planes[j];
		ost << "$\\pi_{" << i << "}";
		ost << "=\\pi_{" << Surf->Eckard_point_label_tex[i] << "}";
		ost << " = \\{ \\ell_i \\mid i =";
		for (h = 0; h < 3; h++) {
			ost << Lines_in_tritangent_plane[j * 3 + h];
			if (h < 3 - 1) {
				ost << ", ";
			}
		}
		ost << "\\}";
		ost << " = \\{ ";
		for (h = 0; h < 3; h++) {
			ost << Surf->Line_label_tex[Lines_in_tritangent_plane[j * 3 + h]];
			if (h < 3 - 1) {
				ost << ", ";
			}
		}
		ost << "\\}$" << endl;
		ost << "\\\\" << endl;
	}
	//ost << "\\end{multicols}" << endl;

#if 0
	ost << "The lines in the tritangent planes in "
			"Schlaefli's notation are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	ost << "\\noindent" << endl;
	for (i = 0; i < nb_tritangent_planes; i++) {
		j = Eckardt_to_Tritangent_plane[i];
		a = Tritangent_planes[j];
		ost << "$\\pi_{" << i << "} = \\{ ";
		for (h = 0; h < 3; h++) {
			ost << Surf->Line_label_tex[
						Lines_in_tritangent_plane[j * 3 + h]];
			if (h < 3 - 1) {
				ost << ", ";
				}
			}
		ost << "\\}$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;
#endif


#if 0
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Lines_in_tritangent_plane, 15, 3,
			true /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Lines_in_tritangent_plane + 15 * 3, 15, 3, 15, 0,
			true /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Lines_in_tritangent_plane + 30 * 3, 15, 3, 30, 0,
			true /* f_tex */);
	ost << "$$" << endl;
#endif

	ost << "The tritangent planes through the 27 lines are:\\\\" << endl;

	//ost << "\\begin{multicols}{1}" << endl;
	ost << "\\noindent" << endl;
	for (i = 0; i < 27; i++) {
		ost << "$";
		ost << Surf->Line_label_tex[i];
		ost << "=\\ell_{" << i << "} \\in \\{ \\pi_i \\mid i = ";
		for (h = 0; h < 5; h++) {
			a = Tritangent_planes_on_lines[i * 5 + h];
			j = Tritangent_plane_to_Eckardt[a];
			ost << j;
			if (h < 5 - 1) {
				ost << ", ";
			}
		}
		ost << "\\}";
		ost << "=\\{";
		for (h = 0; h < 5; h++) {
			a = Tritangent_planes_on_lines[i * 5 + h];
			j = Tritangent_plane_to_Eckardt[a];
			ost << "\\pi_{" << Surf->Eckard_point_label_tex[j] << "}";
			if (h < 5 - 1) {
				ost << ", ";
			}
		}
		ost << "\\}";
		ost << "$\\\\" << endl;
	}
	//ost << "\\end{multicols}" << endl;


#if 0
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
		Tritangent_planes_on_lines, 9, 5, true /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
		Tritangent_planes_on_lines + 9 * 5, 9, 5, 9, 0,
		true /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
		Tritangent_planes_on_lines + 18 * 5, 9, 5, 18, 0,
		true /* f_tex */);
	ost << "$$" << endl;
#endif

#if 0
	ost << "The unitangent planes through the 27 lines are:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
		Unitangent_planes_on_lines, 27, q + 1 - 5,
		true /* f_tex */);
	ost << "$$" << endl;
#endif

}
#endif




#if 0
void surface_object::identify_double_six_from_trihedral_pair(
	int *Lines, int t_idx, int *nine_lines, int *double_sixes, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nine_line_idx[9];
	int i, idx;
	sorting Sorting;

	if (f_v) {
		cout << "surface_object::identify_double_six_from_trihedral_pair" << endl;
	}
	if (f_v) {
		cout << "surface_object::identify_double_six_from_trihedral_pair t_idx = " << t_idx << endl;
	}

	for (i = 0; i < 9; i++) {
		if (!Sorting.int_vec_search_linear(Lines, 27, nine_lines[i], idx)) {
			cout << "surface_object::identify_double_six_from_"
					"trihedral_pair cannot find line" << endl;
			exit(1);
		}
		nine_line_idx[i] = idx;
	}
	if (t_idx < 20) {
		identify_double_six_from_trihedral_pair_type_one(Lines,
				t_idx, nine_line_idx, double_sixes, verbose_level);
	}
	else if (t_idx < 110) {
		identify_double_six_from_trihedral_pair_type_two(Lines,
				t_idx, nine_line_idx, double_sixes, verbose_level);
	}
	else if (t_idx < 120) {
		identify_double_six_from_trihedral_pair_type_three(Lines,
				t_idx, nine_line_idx, double_sixes, verbose_level);
	}
	else {
		cout << "surface_object::identify_double_six_from_"
				"trihedral_pair t_idx is out of range" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "surface_object::identify_double_six_from_trihedral_pair done" << endl;
	}
}


void surface_object::identify_double_six_from_trihedral_pair_type_one(
		int *Lines, int t_idx, int *nine_line_idx, int *double_sixes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int subset[6];
	int size_complement;
	int i, j, k, l, m, n;
	int T[9];
	int h;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "surface_object::identify_double_six_from_trihedral_pair_type_one" << endl;
	}
	if (f_v) {
		cout << "t_idx=" << t_idx << endl;
		cout << "Lines:" << endl;
		for (h = 0; h < 27; h++) {
			cout << h << " : " << Lines[h] << endl;
		}
		cout << "nine_line_idx:" << endl;
		for (h = 0; h < 9; h++) {
			cout << h << " : " << nine_line_idx[h] << endl;
		}
	}
	Combi.unrank_k_subset(t_idx, subset, 6, 3);
	Combi.set_complement(subset, 3, subset + 3, size_complement, 6);
	i = subset[0];
	j = subset[1];
	k = subset[2];
	l = subset[3];
	m = subset[4];
	n = subset[5];

	if (f_v) {
		cout << "i=" << i << " j=" << j << " k=" << k
				<< " l=" << l << " m=" << m << " n=" << n << endl;
	}
	T[0] = Surf->line_cij(j, k);
	T[1] = Surf->line_bi(k);
	T[2] = Surf->line_ai(j);
	T[3] = Surf->line_ai(k);
	T[4] = Surf->line_cij(i, k);
	T[5] = Surf->line_bi(i);
	T[6] = Surf->line_bi(j);
	T[7] = Surf->line_ai(i);
	T[8] = Surf->line_cij(i, j);

	int new_lines[27];

	int_vec_mone(new_lines, 27);

	
	for (h = 0; h < 9; h++) {
		new_lines[T[h]] = nine_line_idx[h];
	}

	int X1[5], X1_len;
	int X2[5], X2_len;

	find_common_transversals_to_three_disjoint_lines(
			new_lines[Surf->line_ai(i)],
			new_lines[Surf->line_ai(j)],
			new_lines[Surf->line_ai(k)],
			X1);
	X1_len = 3;
	
	if (f_v) {
		cout << "X1=";
		int_vec_print(cout, X1, X1_len);
		cout << endl;
	}

	int c1, c2, c2b;

	int nb_double_sixes;
	nb_double_sixes = 0;


	for (c1 = 0; c1 < X1_len; c1++) {

		if (f_v) {
			cout << "c1=" << c1 << " / " << X1_len << endl;
		}

		// pick b_l according to c1:
		new_lines[Surf->line_bi(l)] = X1[c1];
		if (f_v) {
			cout << "b_l=" << X1[c1] << endl;
		}


		int X4[2];
		if (c1 == 0) {
			X4[0] = X1[1];
			X4[1] = X1[2];
		}
		else if (c1 == 1) {
			X4[0] = X1[0];
			X4[1] = X1[2];
		}
		else if (c1 == 2) {
			X4[0] = X1[0];
			X4[1] = X1[1];
		}
		else {
			cout << "c1 is illegal" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "X4=";
			int_vec_print(cout, X4, 2);
			cout << endl;
		}



		find_common_transversals_to_four_disjoint_lines(
				new_lines[Surf->line_bi(i)],
				new_lines[Surf->line_bi(j)],
				new_lines[Surf->line_bi(k)],
				new_lines[Surf->line_bi(l)],
				X2);
		X2_len = 2;
		if (f_v) {
			cout << "X2=";
			int_vec_print(cout, X2, 2);
			cout << endl;
		}

		for (c2 = 0; c2 < X2_len; c2++) {

			if (f_v) {
				cout << "c2=" << c2 << " / " << X2_len << endl;
			}

			// pick a_m according to c2:
			new_lines[Surf->line_ai(m)] = X2[c2];
			if (f_v) {
				cout << "a_m=" << X2[c2] << endl;
			}
			if (c2 == 0) {
				c2b = 1;
			}
			else {
				c2b = 0;
			}
			new_lines[Surf->line_ai(n)] = X2[c2b];
			if (f_v) {
				cout << "a_n=" << X2[c2b] << endl;
			}
			
			int p_ml, p_il, p_jl, p_kl;
			int c_ml, c_il, c_jl, c_kl;

			// determine c_ml:
			p_ml = find_tritangent_plane_through_two_lines(
					new_lines[Surf->line_ai(m)],
					new_lines[Surf->line_bi(l)]);
			c_ml = find_unique_line_in_plane(p_ml,
					new_lines[Surf->line_ai(m)],
					new_lines[Surf->line_bi(l)]);
			new_lines[Surf->line_cij(m, l)] = c_ml;
			if (f_v) {
				cout << "c_ml=" << c_ml << endl;
			}
			
			// determine c_il:
			p_il = find_tritangent_plane_through_two_lines(
					new_lines[Surf->line_ai(i)],
					new_lines[Surf->line_bi(l)]);
			c_il = find_unique_line_in_plane(p_il,
					new_lines[Surf->line_ai(i)],
					new_lines[Surf->line_bi(l)]);
			new_lines[Surf->line_cij(i, l)] = c_il;
			
			// determine c_jl:
			p_jl = find_tritangent_plane_through_two_lines(
					new_lines[Surf->line_ai(j)],
					new_lines[Surf->line_bi(l)]);
			c_jl = find_unique_line_in_plane(p_jl,
					new_lines[Surf->line_ai(j)],
					new_lines[Surf->line_bi(l)]);
			new_lines[Surf->line_cij(j, l)] = c_jl;
			
			// determine c_kl:
			p_kl = find_tritangent_plane_through_two_lines(
					new_lines[Surf->line_ai(k)],
					new_lines[Surf->line_bi(l)]);
			c_kl = find_unique_line_in_plane(p_kl,
					new_lines[Surf->line_ai(k)],
					new_lines[Surf->line_bi(l)]);
			new_lines[Surf->line_cij(k, l)] = c_kl;
			
			int planes[5];

			int_vec_copy(Tritangent_planes_on_lines + c_ml * 5,
					planes, 5);
			for (h = 0; h < 5; h++) {
				if (planes[h] == p_ml) {
					continue;
				}
				if (planes[h] == p_il) {
					continue;
				}
				if (planes[h] == p_jl) {
					continue;
				}
				if (planes[h] == p_kl) {
					continue;
				}
				break;
			}
			if (h == 5) {
				cout << "could not find the plane" << endl;
				exit(1);
			}

			int plane_idx, b_m, b_n, a_l;
			int X3[2];
			
			plane_idx = planes[h];
			if (f_v) {
				cout << "plane_idx=" << plane_idx << endl;
			}
			find_two_lines_in_plane(plane_idx, c_ml, X3[0], X3[1]);
			if (f_v) {
				cout << "X3=";
				int_vec_print(cout, X3, 2);
				cout << endl;
			}


			if (X4[0] == X3[0]) {
				b_m = X4[0];
				b_n = X4[1];
				a_l = X3[1];
			}
			else if (X4[0] == X3[1]) {
				b_m = X4[0];
				b_n = X4[1];
				a_l = X3[0];
			}
			else if (X4[1] == X3[0]) {
				b_m = X4[1];
				b_n = X4[0];
				a_l = X3[1];
			}
			else if (X4[1] == X3[1]) {
				b_m = X4[1];
				b_n = X4[0];
				a_l = X3[0];
			}
			else {
				cout << "surface_object::identify_double_six_from_"
						"trihedral_pair_type_one something is wrong "
						"with this choice of c2" << endl;
				continue;
				//exit(1);
			}
			new_lines[Surf->line_ai(l)] = a_l;
			new_lines[Surf->line_bi(m)] = b_m;
			new_lines[Surf->line_bi(n)] = b_n;
			if (f_v) {
				cout << "a_l=" << a_l << " b_m=" << b_m
						<< " b_n=" << b_n << endl;
			}

			for (h = 0; h < 6; h++) {
				double_sixes[nb_double_sixes * 12 + h] =
						new_lines[Surf->line_ai(h)];
			}
			for (h = 0; h < 6; h++) {
				double_sixes[nb_double_sixes * 12 + 6 + h] =
						new_lines[Surf->line_bi(h)];
			}

			cout << "We found the following double six, "
					"nb_double_sixes=" << nb_double_sixes << endl;
			for (h = 0; h < 6; h++) {
				cout << setw(2) << new_lines[Surf->line_ai(h)];
				if (h < 6 - 1) {
					cout << ", ";
				}
			}
			cout << endl;
			for (h = 0; h < 6; h++) {
				cout << setw(2) << new_lines[Surf->line_bi(h)];
				if (h < 6 - 1) {
					cout << ", ";
				}
			}
			cout << endl;

			nb_double_sixes++;
		} // next c2

	} // next c1

	if (f_v) {
		cout << "surface_object::identify_double_six_from_trihedral_pair_type_one done" << endl;
	}
}

void surface_object::identify_double_six_from_trihedral_pair_type_two(
	int *Lines, int t_idx, int *nine_line_idx,
	int *double_sixes, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx, rk_h, rk_s;
	int subset[6];
	int second_subset[6];
	int complement[6];
	int size_complement;
	int l, m, n, p;
	int c1, c2;
	combinatorics_domain Combi;
	sorting Sorting;
	
	if (f_v) {
		cout << "surface_object::identify_double_six_from_trihedral_pair_type_two" << endl;
	}

	idx = t_idx - 20;
	rk_h = idx / 6;
	rk_s = idx % 6;

	Combi.unrank_k_subset(rk_h, subset, 6, 4);
	Combi.unrank_k_subset(rk_s, second_subset, 4, 2);
	Combi.set_complement(second_subset, 2, complement, size_complement, 4);
	l = subset[second_subset[0]];
	m = subset[second_subset[1]];
	n = subset[complement[0]];
	p = subset[complement[1]];


	int subset2[4];
	int complement2[2];
	int r, s;
	int T[9];

	subset2[0] = l;
	subset2[1] = m;
	subset2[2] = n;
	subset2[3] = p;
	Sorting.int_vec_heapsort(subset2, 4);
	Combi.set_complement(subset2, 4, complement2, size_complement, 6);
	r = complement2[0];
	s = complement2[1];
	if (f_v) {
		cout << "l=" << l << " m=" << m << " n=" << n
				<< " p=" << p << " r=" << r << " s=" << s << endl;
	}

	T[0] = Surf->line_ai(l);
	T[1] = Surf->line_bi(p);
	T[2] = Surf->line_cij(l, p);
	T[3] = Surf->line_bi(n);
	T[4] = Surf->line_ai(m);
	T[5] = Surf->line_cij(m, n);
	T[6] = Surf->line_cij(l, n);
	T[7] = Surf->line_cij(m, p);
	T[8] = Surf->line_cij(r, s);
	
	int new_lines[27];

	int_vec_mone(new_lines, 27);

	int i, pi, a, line;
	
	for (i = 0; i < 9; i++) {
		new_lines[T[i]] = nine_line_idx[i];
	}

	
	int X1[5], X1_len;
	int X2[6], X2_len;
	int X3[5], X3_len;
	int X4[6], X4_len;
	int X5[6], X5_len;
	int X6[27]; //, X6_len;

	get_planes_through_line(new_lines, Surf->line_cij(l, n), X1);
	X1_len = 5;
	if (f_v) {
		cout << "X1=";
		int_vec_print(cout, X1, X1_len);
		cout << endl;
	}
	int_vec_delete_element_assume_sorted(X1, X1_len,
			find_tritangent_plane_through_two_lines(
					new_lines[Surf->line_ai(l)],
					new_lines[Surf->line_bi(n)]));
	int_vec_delete_element_assume_sorted(X1, X1_len,
			find_tritangent_plane_through_two_lines(
					new_lines[Surf->line_cij(n, l)],
					new_lines[Surf->line_cij(p, m)]));
	if (f_v) {
		cout << "X1=";
		int_vec_print(cout, X1, X1_len);
		cout << endl;
	}
	for (i = 0; i < 3; i++) {
		pi = X1[i];
		find_two_lines_in_plane(pi,
				new_lines[Surf->line_cij(l, n)],
				X2[2 * i + 0],
				X2[2 * i + 1]);
	}
	X2_len = 6;
	if (f_v) {
		cout << "X2=";
		int_vec_print(cout, X2, X2_len);
		cout << endl;
	}

	get_planes_through_line(new_lines, Surf->line_cij(m, n), X3);
	X3_len = 5;
	if (f_v) {
		cout << "X3=";
		int_vec_print(cout, X3, X3_len);
		cout << endl;
	}
	int_vec_delete_element_assume_sorted(X3, X3_len,
			find_tritangent_plane_through_two_lines(
					new_lines[Surf->line_ai(m)],
					new_lines[Surf->line_bi(n)]));
	int_vec_delete_element_assume_sorted(X3, X3_len,
			find_tritangent_plane_through_two_lines(
					new_lines[Surf->line_cij(p, l)],
					new_lines[Surf->line_cij(n, m)]));
	if (f_v) {
		cout << "X3=";
		int_vec_print(cout, X3, X3_len);
		cout << endl;
	}
	for (i = 0; i < 3; i++) {
		pi = X3[i];
		find_two_lines_in_plane(pi,
				new_lines[Surf->line_cij(m, n)],
				X4[2 * i + 0],
				X4[2 * i + 1]);
	}
	X4_len = 6;
	if (f_v) {
		cout << "X4=";
		int_vec_print(cout, X4, X4_len);
		cout << endl;
	}
	X5_len = 0;
	Sorting.int_vec_heapsort(X2, X2_len);
	Sorting.int_vec_heapsort(X4, X4_len);
	for (i = 0; i < X2_len; i++) {
		a = X2[i];
		if (Sorting.int_vec_search(X4, X4_len, a, idx)) {
			X5[X5_len++] = a;
		}
	}
	if (f_v) {
		cout << "found a set X5 of size " << X5_len << " : ";
		int_vec_print(cout, X5, X5_len);
		cout << endl;
	}
	if (X5_len != 3) {
		cout << "X5_len != 3" << endl;
		exit(1);
	}

	int nb_double_sixes;
	nb_double_sixes = 0;


	for (c1 = 0; c1 < X5_len; c1++) {

		if (f_v) {
			cout << "c1=" << c1 << " / " << X5_len << endl;
		}

		// pick a_n according to c1:
		new_lines[Surf->line_ai(n)] = X5[c1];

		// determine b_l:
		pi = find_tritangent_plane_through_two_lines(
				new_lines[Surf->line_ai(n)],
				new_lines[Surf->line_cij(l, n)]);
		line = find_unique_line_in_plane(pi,
				new_lines[Surf->line_ai(n)],
				new_lines[Surf->line_cij(l, n)]);
		new_lines[Surf->line_bi(l)] = line;

		// determine b_m:
		pi = find_tritangent_plane_through_two_lines(
				new_lines[Surf->line_ai(n)],
				new_lines[Surf->line_cij(m, n)]);
		line = find_unique_line_in_plane(pi,
				new_lines[Surf->line_ai(n)],
				new_lines[Surf->line_cij(m, n)]);
		new_lines[Surf->line_bi(m)] = line;

		// determine a_p:
		pi = find_tritangent_plane_through_two_lines(
				new_lines[Surf->line_bi(m)],
				new_lines[Surf->line_cij(m, p)]);
		line = find_unique_line_in_plane(pi,
				new_lines[Surf->line_bi(m)],
				new_lines[Surf->line_cij(m, p)]);
		new_lines[Surf->line_ai(p)] = line;

		find_common_transversals_to_four_disjoint_lines(
				new_lines[Surf->line_ai(l)],
				new_lines[Surf->line_ai(m)],
				new_lines[Surf->line_ai(n)],
				new_lines[Surf->line_ai(p)], X6);
		//X6_len = 2;

		for (c2 = 0; c2 < 2; c2++) {
			
			if (f_v) {
				cout << "c2=" << c2 << " / " << 2 << endl;
			}

			// pick b_r according to c2:

			new_lines[Surf->line_bi(r)] = X6[c2];
			if (c2 == 0) {
				new_lines[Surf->line_bi(s)] = X6[1];
			}
			else {
				new_lines[Surf->line_bi(s)] = X6[0];
			}

			// determine c_nr:
			pi = find_tritangent_plane_through_two_lines(
					new_lines[Surf->line_ai(n)],
					new_lines[Surf->line_bi(r)]);
			line = find_unique_line_in_plane(pi,
					new_lines[Surf->line_ai(n)],
					new_lines[Surf->line_bi(r)]);
			new_lines[Surf->line_cij(n, r)] = line;

			// determine a_r:
			pi = find_tritangent_plane_through_two_lines(
					new_lines[Surf->line_bi(n)],
					new_lines[Surf->line_cij(n, r)]);
			line = find_unique_line_in_plane(pi,
					new_lines[Surf->line_bi(n)],
					new_lines[Surf->line_cij(n, r)]);
			new_lines[Surf->line_ai(r)] = line;

			// determine c_ns:
			pi = find_tritangent_plane_through_two_lines(
					new_lines[Surf->line_ai(n)],
					new_lines[Surf->line_bi(s)]);
			line = find_unique_line_in_plane(pi,
					new_lines[Surf->line_ai(n)],
					new_lines[Surf->line_bi(s)]);
			new_lines[Surf->line_cij(n, s)] = line;

			// determine a_s:
			pi = find_tritangent_plane_through_two_lines(
					new_lines[Surf->line_bi(n)],
					new_lines[Surf->line_cij(n, s)]);
			line = find_unique_line_in_plane(pi,
					new_lines[Surf->line_bi(n)],
					new_lines[Surf->line_cij(n, s)]);
			new_lines[Surf->line_ai(s)] = line;

			for (i = 0; i < 6; i++) {
				double_sixes[nb_double_sixes * 12 + i] =
						new_lines[Surf->line_ai(i)];
			}
			for (i = 0; i < 6; i++) {
				double_sixes[nb_double_sixes * 12 + 6 + i] =
						new_lines[Surf->line_bi(i)];
			}

			cout << "We found the following double six, "
					"nb_double_sixes=" << nb_double_sixes << endl;
			for (i = 0; i < 6; i++) {
				cout << setw(2) << new_lines[Surf->line_ai(i)];
				if (i < 6 - 1) {
					cout << ", ";
				}
			}
			cout << endl;
			for (i = 0; i < 6; i++) {
				cout << setw(2) << new_lines[Surf->line_bi(i)];
				if (i < 6 - 1) {
					cout << ", ";
				}
			}
			cout << endl;

			nb_double_sixes++;

		} // next c2

	} // next c1

	if (nb_double_sixes != 6) {
		cout << "surface_object::identify_double_six_from_"
				"trihedral_pair_type_two nb_double_sixes != 6" << endl;
		exit(1);
	}

	

	if (f_v) {
		cout << "surface_object::identify_double_six_from_trihedral_pair_type_two done" << endl;
	}
}

void surface_object::identify_double_six_from_trihedral_pair_type_three(
	int *Lines, int t_idx, int *nine_line_idx, int *double_sixes,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::identify_double_six_from_"
				"trihedral_pair_type_three" << endl;
	}
	if (f_v) {
		cout << "surface_object::identify_double_six_from_"
				"trihedral_pair_type_three done" << endl;
	}
}


void surface_object::find_common_transversals_to_two_disjoint_lines(
	int a, int b, int *transversals5)
{
	int i, c;

	c = 0;
	for (i = 0; i < 27; i++) {
		if (i == a || i == b) {
			continue;
		}
		if (Adj_line_intersection_graph[i * 27 + a] 
			&& Adj_line_intersection_graph[i * 27 + b]) {
			transversals5[c++] = i;
		}
	}
	if (c != 5) {
		cout << "surface_object::find_common_transversals_"
				"to_two_disjoint_lines c != 5" << endl;
		exit(1);
	}
}

void surface_object::find_common_transversals_to_three_disjoint_lines(
	int a1, int a2, int a3, int *transversals3)
{
	int i, c;

	c = 0;
	for (i = 0; i < 27; i++) {
		if (i == a1 || i == a2 || i == a3) {
			continue;
		}
		if (Adj_line_intersection_graph[i * 27 + a1] 
			&& Adj_line_intersection_graph[i * 27 + a2] 
			&& Adj_line_intersection_graph[i * 27 + a3]) {
			transversals3[c++] = i;
		}
	}
	if (c != 3) {
		cout << "surface_object::find_common_transversals_"
				"to_three_disjoint_lines c != 3" << endl;
		cout << "c=" << c << endl;
		exit(1);
	}
}

void surface_object::find_common_transversals_to_four_disjoint_lines(
	int a1, int a2, int a3, int a4, int *transversals2)
{
	int i, c;

	c = 0;
	for (i = 0; i < 27; i++) {
		if (i == a1 || i == a2 || i == a3 || i == a4) {
			continue;
		}
		if (Adj_line_intersection_graph[i * 27 + a1] 
			&& Adj_line_intersection_graph[i * 27 + a2] 
			&& Adj_line_intersection_graph[i * 27 + a3] 
			&& Adj_line_intersection_graph[i * 27 + a4]) {
			transversals2[c++] = i;
		}
	}
	if (c != 2) {
		cout << "surface_object::find_common_transversals_"
				"to_four_disjoint_lines c != 2" << endl;
		exit(1);
	}
}

int surface_object::find_tritangent_plane_through_two_lines(
	int line_a, int line_b)
{
	int i, idx, pi;
	sorting Sorting;

	for (i = 0; i < 5; i++) {
		pi = Tritangent_planes_on_lines[line_a * 5 + i];
		if (Sorting.lint_vec_search_linear(
				Lines_in_tritangent_plane + pi * 3, 3,
				line_b, idx)) {
			return pi;
		}
	}
	cout << "surface_object::find_tritangent_plane_through_"
			"two_lines we could not find the tritangent "
			"plane through these two lines" << endl;
	exit(1);
}

void surface_object::get_planes_through_line(int *new_lines, 
	int line_idx, int *planes5)
{
	int f_v = false;

	if (f_v) {
		cout << "surface_object::get_planes_through_line " << endl;
		cout << "line=" << Surf->Line_label[line_idx] << endl;
	}
	int_vec_copy(Tritangent_planes_on_lines + new_lines[line_idx] * 5,
			planes5, 5);
}

void surface_object::find_two_lines_in_plane(int plane_idx, 
	int forbidden_line, int &line1, int &line2)
{
	int i;
	
	for (i = 0; i < 3; i++) {
		if (Lines_in_tritangent_plane[plane_idx * 3 + i] == forbidden_line) {
			if (i == 0) {
				line1 = Lines_in_tritangent_plane[plane_idx * 3 + 1];
				line2 = Lines_in_tritangent_plane[plane_idx * 3 + 2];
			}
			else if (i == 1) {
				line1 = Lines_in_tritangent_plane[plane_idx * 3 + 0];
				line2 = Lines_in_tritangent_plane[plane_idx * 3 + 2];
			}
			else if (i == 2) {
				line1 = Lines_in_tritangent_plane[plane_idx * 3 + 0];
				line2 = Lines_in_tritangent_plane[plane_idx * 3 + 1];
			}
			return;
		}
	}
	cout << "surface_object::find_two_lines_in_plane we "
			"could not find the forbidden line" << endl;
}

int surface_object::find_unique_line_in_plane(int plane_idx, 
	int forbidden_line1, int forbidden_line2)
{
	int i, a;

	for (i = 0; i < 3; i++) {
		a = Lines_in_tritangent_plane[plane_idx * 3 + i];
		if (a == forbidden_line1) {
			continue;
			}
		if (a == forbidden_line2) {
			continue;
			}
		return a;
		}
	cout << "surface_object::find_unique_line_in_plane we "
			"could not find the unique line" << endl;
	exit(1);
}

int surface_object::choose_tritangent_plane(
	int line_a, int line_b, int transversal_line, int verbose_level)
{
	int f_v = true; // (verbose_level >= 1);
	int i, plane, idx, a;
	sorting Sorting;

	if (f_v) {
		cout << "surface_object::choose_tritangent_plane" << endl;
		cout << "line_a=" << line_a << endl;
		cout << "line_b=" << line_b << endl;
		cout << "transversal_line=" << transversal_line << endl;
		//cout << "Tritangent_planes_on_lines:" << endl;
		//int_matrix_print(Tritangent_planes_on_lines, 27, 5);
		}
	if (false) {
		cout << "Testing the following planes: ";
		int_vec_print(cout,
			Tritangent_planes_on_lines + transversal_line * 5, 5);
		cout << endl;
		}
	for (i = 4; i >= 0; i--) {
		a = Tritangent_planes_on_lines[transversal_line * 5 + i];
		plane = Tritangent_plane_to_Eckardt[a];
		if (f_v) {
			cout << "testing plane " << a << " = " << plane << endl;
			}
		if (Sorting.lint_vec_search_linear(
				Lines_in_tritangent_plane + a * 3, 3, line_a, idx)) {
			if (f_v) {
				cout << "The plane is bad, it contains line_a" << endl;
				}
			continue;
			}
		if (Sorting.lint_vec_search_linear(
				Lines_in_tritangent_plane + a * 3, 3, line_b, idx)) {
			if (f_v) {
				cout << "The plane is bad, it contains line_b" << endl;
				}
			continue;
			}
		break;
		}
	if (i == 5) {
		cout << "surface_object::choose_tritangent_plane "
				"could not find a tritangent plane" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface_object::choose_tritangent_plane done" << endl;
		}
	return plane;
}

void surface_object::find_all_tritangent_planes(
	int line_a, int line_b, int transversal_line, 
	int *tritangent_planes3, 
	int verbose_level)
{
	int f_v = true; // (verbose_level >= 1);
	int i, plane, idx, a, nb;
	sorting Sorting;
	
	if (f_v) {
		cout << "surface_object::find_all_tritangent_planes" << endl;
		cout << "line_a=" << line_a << endl;
		cout << "line_b=" << line_b << endl;
		cout << "transversal_line=" << transversal_line << endl;
		//cout << "Tritangent_planes_on_lines:" << endl;
		//int_matrix_print(Tritangent_planes_on_lines, 27, 5);
		}
	if (false) {
		cout << "Testing the following planes: ";
		int_vec_print(cout,
				Tritangent_planes_on_lines + transversal_line * 5, 5);
		cout << endl;
		}
	nb = 0;
	for (i = 4; i >= 0; i--) {
		a = Tritangent_planes_on_lines[transversal_line * 5 + i];
		plane = Tritangent_plane_to_Eckardt[a];
		if (f_v) {
			cout << "testing plane " << a << " = " << plane << endl;
			}
		if (Sorting.lint_vec_search_linear(
				Lines_in_tritangent_plane + a * 3, 3, line_a, idx)) {
			if (f_v) {
				cout << "The plane is bad, it contains line_a" << endl;
				}
			continue;
			}
		if (Sorting.lint_vec_search_linear(
				Lines_in_tritangent_plane + a * 3, 3, line_b, idx)) {
			if (f_v) {
				cout << "The plane is bad, it contains line_b" << endl;
				}
			continue;
			}
		tritangent_planes3[nb++] = plane;
		}
	if (nb != 3) {
		cout << "surface_object::find_all_tritangent_planes "
				"nb != 3" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface_object::choose_tritangent_plane "
				"done" << endl;
		}
}
#endif

void surface_object::identify_lines(
		long int *lines, int nb_lines,
	int *line_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::identify_lines" << endl;
	}
#if 0
	int i, idx;
	data_structures::sorting Sorting;
	for (i = 0; i < nb_lines; i++) {
		if (!Sorting.lint_vec_search_linear(
				Variety_object->Line_sets->Sets[0], 27, lines[i], idx)) {
			cout << "surface_object::identify_lines could "
					"not find lines[" << i << "]=" << lines[i]
					<< " in Lines[]" << endl;
			exit(1);
		}
		line_idx[i] = idx;
	}
#else
	Variety_object->identify_lines(lines, nb_lines,
			line_idx, verbose_level);
#endif
	if (f_v) {
		cout << "surface_object::identify_lines done" << endl;
	}
}

void surface_object::print_nine_lines_latex(
		std::ostream &ost,
	long int *nine_lines, int *nine_lines_idx)
{
	other::l1_interfaces::latex_interface L;
	int i, j, idx;

	ost << "$$";
	L.print_lint_matrix_with_standard_labels(ost,
			nine_lines, 3, 3, true /* f_tex*/);


	ost << "\\qquad" << endl;
	ost << "\\begin{array}{c|ccc}" << endl;
	for (j = 0; j < 3; j++) {
		ost << " & " << j;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < 3; i++) {
		ost << i << " & ";
		for (j = 0; j < 3; j++) {

			idx = nine_lines_idx[i * 3 + j];
			ost << "\\ell_{" << idx << "}";
			if (j < 3 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\qquad" << endl;
	ost << "\\begin{array}{c|ccc}" << endl;
	for (j = 0; j < 3; j++) {
		ost << " & " << j;
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < 3; i++) {
		ost << i << " & ";
		for (j = 0; j < 3; j++) {

			idx = nine_lines_idx[i * 3 + j];
			ost << Surf->Schlaefli->Labels->Line_label_tex[idx];
			if (j < 3 - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
}

#if 0
void surface_object::compute_clebsch_maps(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	
	if (f_v) {
		cout << "surface_object::compute_clebsch_maps" << endl;
		}

	int line_idx[2];
	int *Clebsch_map;
	int *Clebsch_coeff;
	int cnt, h;
	int Arc[6];
	int Blown_up_lines[6];
	int Blown_up2[6];
	int transversal_line, plane_rk, plane_rk_global;

	cnt = 0;
	Clebsch_map = NEW_int(nb_pts);
	Clebsch_coeff = NEW_int(nb_pts * 4);
	// loop over all pairs of disjoint lines:
	for (i = 0; i < 27; i++) {
		for (j = i + 1; j < 27; j++) {
			if (Adj_line_intersection_graph[i * 27 + j] == 1) {
				continue;
				}
			line_idx[0] = i;
			line_idx[1] = j;

			cout << "#######################" << endl;
			cout << "clebsch map for lines " << i << ", "
					<< j << ":" << endl;

			transversal_line = compute_transversal_line(i, j,
					0 /* verbose_level */);
			
			plane_rk = choose_tritangent_plane(i, j,
					transversal_line, 0 /* verbose_level */);

			plane_rk_global = Tritangent_planes[plane_rk];

			cout << "transversal\\_line = " << transversal_line
					<< "\\\\" << endl;
			cout << "plane\\_rank = " << plane_rk << " = "
					<< plane_rk_global << "\\\\" << endl;


			if (!Surf->clebsch_map(Lines, Pts, nb_pts,
					line_idx, plane_rk_global,
				Clebsch_map, Clebsch_coeff, 0 /*verbose_level*/)) {
				cout << "The plane contains one of the lines, "
						"this should not happen" << endl;
				exit(1);
				}

			cout << "clebsch map for lines " << i << ", " << j
					<< " clebsch_map_print_fibers:" << endl;
			clebsch_map_print_fibers(Clebsch_map);


			cout << "clebsch map for lines " << i << ", " << j
					<< " clebsch_map_find_arc_and_lines:" << endl;
			clebsch_map_find_arc_and_lines(Clebsch_map, Arc,
					Blown_up_lines, 1 /* verbose_level */);

			cout << "after clebsch_map_find_arc_and_lines" << endl;
			//clebsch_map_find_arc(Clebsch_map, Pts, nb_pts,
			//Arc, 0 /* verbose_level */);
			cout << "Clebsch map for lines " << i << ", " << j
					<< " cnt=" << cnt << " : arc = ";
			int_vec_print(cout, Arc, 6);
			cout << " : blown up lines = ";
			int_vec_print(cout, Blown_up_lines, 6);




			cout << " : ";
			
			int_vec_copy(Blown_up_lines, Blown_up2, 6);
			int_vec_heapsort(Blown_up2, 6);
			for (h = 0; h < 6; h++) {
				if (Blown_up2[h] >= 0 && Blown_up2[h] < 27) {
					cout << Surf->Line_label[Blown_up2[h]];
					if (h < 6 - 1) {
						cout << ", ";
						}
					}
				}
			cout << endl;
			
			cnt++;
			}
		}

	FREE_int(Clebsch_map);
	FREE_int(Clebsch_coeff);

	if (f_v) {
		cout << "surface_object::compute_clebsch_maps done" << endl;
		}
}
#endif



int surface_object::find_point(
		long int P, int &idx)
{

	return Variety_object->find_point(P, idx);
#if 0
	data_structures::sorting Sorting;

	if (Sorting.lint_vec_search(Variety_object->Point_sets->Sets[0], Variety_object->Point_sets->Set_size[0], P,
			idx, 0 /* verbose_level */)) {
		return true;
	}
	else {
		return false;
	}
#endif
}

void surface_object::export_something(
		std::string &what,
		std::string &fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::export_something" << endl;
	}

	other::data_structures::string_tools ST;
	string fname;
	other::orbiter_kernel_system::file_io Fio;


	if (ST.stringcmp(what, "points") == 0) {

		fname = fname_base + "_points.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname,
				Variety_object->Point_sets->Sets[0],
				1, Variety_object->Point_sets->Set_size[0]);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "points_off") == 0) {

		fname = fname_base + "_points_off.csv";

		long int *Pts_off;
		int nb_pts_off;

		nb_pts_off = Surf->P->Subspaces->N_points - Variety_object->Point_sets->Set_size[0];

		Pts_off = NEW_lint(Surf->P->Subspaces->N_points);

		Lint_vec_complement_to(
				Variety_object->Point_sets->Sets[0],
				Pts_off,
				Surf->P->Subspaces->N_points,
				Variety_object->Point_sets->Set_size[0]);

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, Pts_off, 1, nb_pts_off);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

		FREE_lint(Pts_off);
	}
	else if (ST.stringcmp(what, "Eckardt_points") == 0) {

		fname = fname_base + "_Eckardt_points.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, SOP->Eckardt_points, 1, SOP->nb_Eckardt_points);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "double_points") == 0) {

		fname = fname_base + "_double_points.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, SOP->Double_points, 1, SOP->nb_Double_points);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "single_points") == 0) {

		fname = fname_base + "_single_points.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, SOP->Single_points, 1, SOP->nb_Single_points);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "zero_points") == 0) {

		fname = fname_base + "_zero_points.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, SOP->Pts_not_on_lines, 1, SOP->nb_pts_not_on_lines);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "singular_points") == 0) {

		fname = fname_base + "_singular_points.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, SOP->singular_pts, 1, SOP->nb_singular_pts);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	else if (ST.stringcmp(what, "lines") == 0) {

		fname = fname_base + "_lines.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, Variety_object->Line_sets->Sets[0],
				Variety_object->Line_sets->Set_size[0], 1);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	else if (ST.stringcmp(what, "lines_in_Pluecker_coordinates") == 0) {

		fname = fname_base + "_lines_Pluecker.csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, SOP->Pluecker_coordinates,
				Variety_object->Line_sets->Set_size[0], 6);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	else if (ST.stringcmp(what, "axes") == 0) {

		fname = fname_base + "_axes.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, SOP->Axes_line_rank, 1, SOP->nb_axes);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	else if (ST.stringcmp(what, "tritangent_planes") == 0) {

		fname = fname_base + "_tritangent_planes.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname,
				SOP->SmoothProperties->Tritangent_plane_rk,
				1,
				SOP->SmoothProperties->nb_tritangent_planes);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "trihedral_pairs") == 0) {

		fname = fname_base + "_trihedral_pairs.csv";


		Fio.Csv_file_support->lint_matrix_write_csv(
				fname,
				Surf->Schlaefli->Schlaefli_trihedral_pairs->Axes,
				120,
				6);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "trihedral_pairs_last10") == 0) {

		fname = fname_base + "_trihedral_pairs_last10.csv";


		int i, j, k, a;

		int *blocks;

		blocks = NEW_int(20 * 3);
		for (i = 0; i < 10; i++) {
			for (j = 0; j < 2; j++) {
				for (k = 0; k < 3; k++) {
					a = Surf->Schlaefli->Schlaefli_trihedral_pairs->Axes[(110 + i) * 6 + j * 3 + k];
					blocks[(i * 2 + j) * 3 + k] = a - 30;
				}
			}
		}

		cout << "surface_object::export_something "
				"Blocks: " << endl;
		Int_matrix_print(blocks, 20, 3);

		other::data_structures::sorting Sorting;

		for (i = 0; i < 20; i++) {
			Sorting.int_vec_heapsort(
					blocks + i * 3, 3);
		}

		cout << "surface_object::export_something "
				"Blocks after sorting: " << endl;
		Int_matrix_print(blocks, 20, 3);

		combinatorics::other_combinatorics::combinatorics_domain Combi;

		int *Rk;

		Rk = NEW_int(20);
		for (i = 0; i < 20; i++) {
			Rk[i] = Combi.rank_k_subset(blocks + i * 3, 15, 3);
		}

		cout << "surface_object::export_something "
				"Ranks: ";
		Int_vec_print(cout, Rk, 20);
		cout << endl;

#if 0
		Fio.lint_matrix_write_csv(fname,
				Surf->Schlaefli->Trihedral_to_Eckardt,
				120,
				6);
#endif
		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "Hesse_planes") == 0) {

		fname = fname_base + "_Hesse_planes.csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, SOP->Hesse_planes, 1, SOP->nb_Hesse_planes);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else if (ST.stringcmp(what, "roots") == 0) {

		fname = fname_base + "_roots.csv";

		if (Variety_object->Line_sets->Set_size[0] != 27) {
			cout << "surface must have 27 lines to be able to export roots" << endl;
			exit(1);
		}
		Fio.Csv_file_support->int_matrix_write_csv(
				fname, SOP->SmoothProperties->Roots, 72, 6);

		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	else {
		cout << "surface_object::export_something "
				"unrecognized export target: " << what << endl;
	}

	if (f_v) {
		cout << "surface_object::export_something done" << endl;
	}

}






void surface_object::print_lines_tex(
		std::ostream &ost)
{

	Surf->print_lines_tex(
			ost,
			Variety_object->Line_sets->Sets[0],
			Variety_object->Line_sets->Set_size[0]);

}

void surface_object::print_one_line_tex(
		std::ostream &ost, int idx)
{

	Surf->print_one_line_tex(
			ost,
			Variety_object->Line_sets->Sets[0],
			Variety_object->Line_sets->Set_size[0], idx);

}

void surface_object::print_double_sixes(
		std::ostream &ost)
{
	//int idx;
	ost << "\\bigskip" << endl;

	ost << "\\subsection*{Double sixes}" << endl;

	Surf->Schlaefli->Schlaefli_double_six->print_double_sixes(
			ost, Variety_object->Line_sets->Sets[0]);


}

void surface_object::Clebsch_map_up(
		std::string &fname_base,
		int line_1_idx, int line_2_idx, int verbose_level)
// Computes the Clebsch map going up onto the surface.
// The image point indices are written to file.
// The domain of the map is PG(3,q).
// So, each point of PG(3,q) is mapped onto the surface.
// There is a closed set of points for which the map is undefined.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::Clebsch_map_up" << endl;
	}

	long int *Image_pts;
	long int N_points;

	if (f_v) {
		cout << "surface_object::Clebsch_map_up "
				"before computing the map" << endl;
	}



	int *v;
	int *w;
	int h;
	long int i;
	int f_vv = false;
	int len = 4;

	int Line_a[8];
	int Line_b[8];
	int M[16];
	int Dual_planes[16];
	int Transversal_line[8];
	long int *point_list;
	long int *Line_a_point_list;
	long int *Line_b_point_list;


	point_list = NEW_lint(Surf->P->Subspaces->k);
	Line_a_point_list = NEW_lint(Surf->P->Subspaces->k);
	Line_b_point_list = NEW_lint(Surf->P->Subspaces->k);

	// get generator matrices for the two skew lines:
	Surf->Gr->unrank_lint_here(
			Line_a,
			Variety_object->Line_sets->Sets[0][line_1_idx],
			0 /*verbose_level*/);
	Surf->Gr->unrank_lint_here(
			Line_b,
			Variety_object->Line_sets->Sets[0][line_2_idx],
			0 /*verbose_level*/);


	Surf->P->Subspaces->create_points_on_line_with_line_given(
			Line_a, Line_a_point_list, verbose_level - 2);
	Surf->P->Subspaces->create_points_on_line_with_line_given(
			Line_b, Line_b_point_list, verbose_level - 2);

	other::data_structures::sorting Sorting;

	Sorting.lint_vec_heapsort(Line_a_point_list, Surf->P->Subspaces->k);
	Sorting.lint_vec_heapsort(Line_b_point_list, Surf->P->Subspaces->k);



	N_points = Surf->P->Subspaces->N_points;



	Image_pts = NEW_lint(N_points);

	v = NEW_int(Surf->P->Subspaces->n + 1);
	w = NEW_int(len);


	for (i = 0; i < N_points; i++) {

		Surf->P->unrank_point(v, i);

		if (f_vv) {
			cout << "surface_object::Clebsch_map_up "
					"point " << i << " is ";
			Int_vec_print(cout, v, Surf->P->Subspaces->n + 1);
			cout << endl;
		}

#if 0
		for (h = 0; h < len; h++) {

			w[h] = Object->Formula_vector->V[h].tree->evaluate(
					symbol_table,
					verbose_level - 2);

		}
#else
		//Int_vec_zero(Image_coeff + h * 4, 4);
		if (f_v) {
			cout << "surface_object::Clebsch_map_up "
					"pt " << i << " / " << N_points << " is ";
			Int_vec_print(cout, v, 4);
			cout << ":" << endl;
		}

		// make sure the points do not lie on either line_a or line_b
		// because the map is undefined there:
		Int_vec_copy(Line_a, M, 2 * 4);
		Int_vec_copy(v, M + 2 * 4, 4);
		if (F->Linear_algebra->Gauss_easy(M, 3, 4) == 2) {
			if (f_vv) {
				cout << "The point is on line_a" << endl;
			}
			Image_pts[i] = -1;
			continue;
		}
		Int_vec_copy(Line_b, M, 2 * 4);
		Int_vec_copy(v, M + 2 * 4, 4);
		if (F->Linear_algebra->Gauss_easy(M, 3, 4) == 2) {
			if (f_vv) {
				cout << "The point is on line_b" << endl;
			}
			Image_pts[i] = -1;
			continue;
		}

		// The point is good:

		// Compute the first plane in dual coordinates:

		Int_vec_copy(Line_a, M, 2 * 4);
		Int_vec_copy(v, M + 2 * 4, 4);
		F->Linear_algebra->RREF_and_kernel(
				4, 3, M, 0 /* verbose_level */);
		Int_vec_copy(M + 3 * 4, Dual_planes, 4);
		if (f_vv) {
			cout << "surface_object::Clebsch_map_up "
					"First plane in dual coordinates: ";
			Int_vec_print(cout, M + 3 * 4, 4);
			cout << endl;
		}

		// Compute the second plane in dual coordinates:

		Int_vec_copy(Line_b, M, 2 * 4);
		Int_vec_copy(v, M + 2 * 4, 4);
		F->Linear_algebra->RREF_and_kernel(
				4, 3, M, 0 /* verbose_level */);
		Int_vec_copy(M + 3 * 4, Dual_planes + 4, 4);
		if (f_vv) {
			cout << "surface_object::Clebsch_map_up "
					"Second plane in dual coordinates: ";
			Int_vec_print(cout, M + 3 * 4, 4);
			cout << endl;
		}

		// Compute the transversal line:

		if (F->Linear_algebra->RREF_and_kernel(
				4, 2, Dual_planes, 0 /* verbose_level */) != 2) {
			Image_pts[i] = -1;
			continue;
		}
		Int_vec_copy(Dual_planes + 2 * 4, Transversal_line, 8);

		// Compute all points on the transversal line:

		Surf->P->Subspaces->create_points_on_line_with_line_given(
				Transversal_line, point_list, verbose_level - 2);

		Image_pts[i] = -1;

		// find the points on the transversal line which lie on the surface:
		// There should be three of these points.



		// at first, we count the number of such points (it should be three)
		int cnt;


		cnt = 0;

		for (h = 0; h < Surf->P->Subspaces->k; h++) {

			long int pt;
			int idx;

			pt = point_list[h];

			if (!find_point(pt, idx)) {
				continue;
			}
			cnt++;

		}

		if (cnt != 3) {
			cout << "cnt = " << cnt << ", skipping" << endl;
			continue;
		}

		for (h = 0; h < Surf->P->Subspaces->k; h++) {

			long int pt;
			int idx, idx_ab;
			int f_lies_on_line_a = false;
			int f_lies_on_line_b = false;

			pt = point_list[h];

			if (!find_point(pt, idx)) {
				continue;
			}

			// test if the point lies on line a:

			if (Sorting.lint_vec_search(
					Line_a_point_list, Surf->P->Subspaces->k,
					pt, idx_ab, 0 /* verbose_level */)) {
				f_lies_on_line_a = true;
			}

			// test if the point lies on line b:

			if (Sorting.lint_vec_search(
					Line_b_point_list, Surf->P->Subspaces->k,
					pt, idx_ab, 0 /* verbose_level */)) {
				f_lies_on_line_b = true;
			}

			// If the point does not lie one either line a or line b,
			// then this must be the image point:

			if (!f_lies_on_line_a && !f_lies_on_line_b) {
				Image_pts[i] = pt;
				break;
			}

		}

		if (Image_pts[i] == -1) {
			cout << "surface_object::Clebsch_map_up "
					"could not find image point" << endl;
			exit(1);
		}

#endif

		if (f_vv) {
			cout << "surface_object::Clebsch_map_up maps to ";
			Int_vec_print(cout, w, len);
			cout << " = " << Image_pts[i] << endl;
		}

	}

	FREE_int(v);
	FREE_int(w);

	FREE_lint(point_list);


	if (f_v) {
		cout << "surface_object::Clebsch_map_up "
				"after computing the map" << endl;
	}

	if (f_v) {
		cout << "surface_object::Clebsch_map_up Image_pts:" << endl;
		Lint_vec_print(cout, Image_pts, N_points);
		cout << endl;
	}

	string fname_map;
	other::orbiter_kernel_system::file_io Fio;

	fname_map = fname_base + "_Clebsch_map_up.csv";


	Fio.Csv_file_support->lint_matrix_write_csv(
			fname_map, Image_pts, N_points, 1);
	if (f_v) {
		cout << "Written file " << fname_map
				<< " of size " << Fio.file_size(fname_map) << endl;
	}

	FREE_lint(Image_pts);


	if (f_v) {
		cout << "surface_object::Clebsch_map_up done" << endl;
	}

}



long int surface_object::Clebsch_map_up_single_point(
		long int input_point,
		int line_1_idx, int line_2_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::Clebsch_map_up_single_point" << endl;
	}

	long int image;
	long int N_points;

	if (f_v) {
		cout << "surface_object::Clebsch_map_up_single_point "
				"before computing the map" << endl;
	}



	int *v;
	int *w;
	int *z;
	int h;
	int f_vv = (verbose_level >= 2);
	int len = 4;

	int Line_a[8];
	int Line_b[8];
	int M[16];
	int Dual_planes[16];
	int Transversal_line[8];
	long int *point_list;
	long int *Line_a_point_list;
	long int *Line_b_point_list;


	point_list = NEW_lint(Surf->P->Subspaces->k);
	Line_a_point_list = NEW_lint(Surf->P->Subspaces->k);
	Line_b_point_list = NEW_lint(Surf->P->Subspaces->k);

	// get generator matrices for the two skew lines:
	Surf->Gr->unrank_lint_here(Line_a, Variety_object->Line_sets->Sets[0][line_1_idx], 0 /*verbose_level*/);
	Surf->Gr->unrank_lint_here(Line_b, Variety_object->Line_sets->Sets[0][line_2_idx], 0 /*verbose_level*/);


	if (f_vv) {
		cout << "Line1:" << endl;
		Int_matrix_print(Line_a, 2, 4);
		cout << "Line2:" << endl;
		Int_matrix_print(Line_b, 2, 4);
	}

	Surf->P->Subspaces->create_points_on_line_with_line_given(
			Line_a, Line_a_point_list, verbose_level - 2);
	Surf->P->Subspaces->create_points_on_line_with_line_given(
			Line_b, Line_b_point_list, verbose_level - 2);

	other::data_structures::sorting Sorting;

	Sorting.lint_vec_heapsort(Line_a_point_list, Surf->P->Subspaces->k);
	Sorting.lint_vec_heapsort(Line_b_point_list, Surf->P->Subspaces->k);



	N_points = Surf->P->Subspaces->N_points;




	v = NEW_int(Surf->P->Subspaces->n + 1);
	w = NEW_int(len);
	z = NEW_int(len);



		Surf->P->unrank_point(v, input_point);

		if (f_vv) {
			cout << "surface_object::Clebsch_map_up_single_point "
					"point " << input_point << " is ";
			Int_vec_print(cout, v, Surf->P->Subspaces->n + 1);
			cout << endl;
		}

#if 0
		for (h = 0; h < len; h++) {

			w[h] = Object->Formula_vector->V[h].tree->evaluate(
					symbol_table,
					verbose_level - 2);

		}
#else
		//Int_vec_zero(Image_coeff + h * 4, 4);
		if (f_v) {
			cout << "surface_object::Clebsch_map_up_single_point "
					"pt " << input_point << " / " << N_points << " is ";
			Int_vec_print(cout, v, 4);
			cout << ":" << endl;
		}

		// make sure the points do not lie on either line_a or line_b
		// because the map is undefined there:
		Int_vec_copy(Line_a, M, 2 * 4);
		Int_vec_copy(v, M + 2 * 4, 4);
		if (f_vv) {
			cout << "Testing point and line1:" << endl;
			Int_matrix_print(M, 3, 4);
		}
		if (F->Linear_algebra->Gauss_easy(M, 3, 4) == 2) {
			if (f_vv) {
				cout << "The point is on line_a" << endl;
			}
			image = -1;
			goto the_end;
		}
		Int_vec_copy(Line_b, M, 2 * 4);
		Int_vec_copy(v, M + 2 * 4, 4);
		if (f_vv) {
			cout << "Testing point and line2:" << endl;
			Int_matrix_print(M, 3, 4);
		}
		if (F->Linear_algebra->Gauss_easy(M, 3, 4) == 2) {
			if (f_vv) {
				cout << "The point is on line_b" << endl;
			}
			image = -1;
			goto the_end;
		}

		// The point is good:

		// Compute the first plane in dual coordinates:
		Int_vec_copy(Line_a, M, 2 * 4);
		Int_vec_copy(v, M + 2 * 4, 4);
		if (f_vv) {
			cout << "First system:" << endl;
			Int_matrix_print(M, 3, 4);
		}
		F->Linear_algebra->RREF_and_kernel(
				4, 3, M, 0 /* verbose_level */);
		Int_vec_copy(M + 3 * 4, Dual_planes, 4);
		if (f_vv) {
			cout << "surface_object::Clebsch_map_up_single_point "
					"First plane in dual coordinates: ";
			Int_vec_print(cout, M + 3 * 4, 4);
			cout << endl;
		}

		// Compute the second plane in dual coordinates:
		Int_vec_copy(Line_b, M, 2 * 4);
		Int_vec_copy(v, M + 2 * 4, 4);
		if (f_vv) {
			cout << "Second system:" << endl;
			Int_matrix_print(M, 3, 4);
		}
		F->Linear_algebra->RREF_and_kernel(
				4, 3, M, 0 /* verbose_level */);
		Int_vec_copy(M + 3 * 4, Dual_planes + 4, 4);
		if (f_vv) {
			cout << "surface_object::Clebsch_map_up_single_point "
					"Second plane in dual coordinates: ";
			Int_vec_print(cout, M + 3 * 4, 4);
			cout << endl;
		}
		if (F->Linear_algebra->RREF_and_kernel(
				4, 2, Dual_planes, 0 /* verbose_level */) != 2) {
			image = -1;
			goto the_end;
		}
		Int_vec_copy(Dual_planes + 2 * 4, Transversal_line, 8);

		if (f_vv) {
			cout << "Transversal_line:" << endl;
			Int_matrix_print(Transversal_line, 2, 4);
		}

		Surf->P->Subspaces->create_points_on_line_with_line_given(
				Transversal_line, point_list, verbose_level - 2);


		if (f_vv) {
			cout << "Points on Transversal_line:" << endl;
			Lint_vec_print(cout, point_list, Surf->P->Subspaces->k);
			cout << endl;
		}


		image = -1;

		int cnt;


		cnt = 0;

		for (h = 0; h < Surf->P->Subspaces->k; h++) {

			long int pt;
			int idx; //, idx_ab;
			//int f_lies_on_line_a = false;
			//int f_lies_on_line_b = false;

			pt = point_list[h];

			if (!find_point(pt, idx)) {
				continue;
			}
			if (f_vv) {
				Surf->P->unrank_point(z, pt);
				cout << "point " << h << " = " << pt << " = ";
				Int_vec_print(cout, z, 4);
				cout << " lies on the surface" << endl;
			}

			cnt++;

		}

		if (cnt != 3) {
			if (f_vv) {
				cout << "cnt = " << cnt << ", skipping" << endl;
			}
			image = -1;
			goto the_end;
		}

		for (h = 0; h < Surf->P->Subspaces->k; h++) {

			long int pt;
			int idx, idx_ab;
			int f_lies_on_line_a = false;
			int f_lies_on_line_b = false;

			pt = point_list[h];

			if (!find_point(pt, idx)) {
				continue;
			}

			Surf->P->unrank_point(z, pt);


			if (Sorting.lint_vec_search(
					Line_a_point_list, Surf->P->Subspaces->k,
					pt, idx_ab, 0 /* verbose_level */)) {
				if (f_vv) {
					cout << "point " << h << " = " << pt << " = ";
					Int_vec_print(cout, z, 4);
					cout << " lies on line1" << endl;
				}
				f_lies_on_line_a = true;
			}
			if (Sorting.lint_vec_search(
					Line_b_point_list, Surf->P->Subspaces->k,
					pt, idx_ab, 0 /* verbose_level */)) {
				if (f_vv) {
					cout << "point " << h << " = " << pt << " = ";
					Int_vec_print(cout, z, 4);
					cout << " lies on line2" << endl;
				}
				f_lies_on_line_b = true;
			}
			if (!f_lies_on_line_a && !f_lies_on_line_b) {
				if (f_vv) {
					cout << "point " << h << " = " << pt << " = ";
					Int_vec_print(cout, z, 4);
					cout << " is the image" << endl;
				}
				image = pt;
				//goto the_end;
			}

		}


#endif


the_end:
	FREE_int(v);
	FREE_int(w);
	FREE_int(z);

	FREE_lint(point_list);
	FREE_lint(Line_a_point_list);
	FREE_lint(Line_b_point_list);


	if (f_v) {
		cout << "surface_object::Clebsch_map_up_single_point "
				"after computing the map" << endl;
	}


	if (f_v) {
		cout << "surface_object::Clebsch_map_up_single_point done" << endl;
	}
	return image;

}

std::string surface_object::stringify_eqn()
{
	return Variety_object->stringify_eqn();
#if 0
	string s;

	s = Int_vec_stringify(Variety_object->eqn, 20);
	return s;
#endif
}



std::string surface_object::stringify_Pts()
{
	return Variety_object->stringify_Pts();
#if 0
	string s;

	s = Lint_vec_stringify(
			Variety_object->Point_sets->Sets[0],
			Variety_object->Point_sets->Set_size[0]);
	return s;
#endif
}

std::string surface_object::stringify_Lines()
{
	return Variety_object->stringify_Lines();
#if 0
	string s;

	s = Lint_vec_stringify(
			Variety_object->Line_sets->Sets[0],
			Variety_object->Line_sets->Set_size[0]);
	return s;
#endif
}

int surface_object::find_double_point(
		int line1, int line2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::find_double_point" << endl;
	}
	long int pt;
	int idx;

	pt = Surf->compute_double_point(
			Variety_object->Line_sets->Sets[0],
			Variety_object->Line_sets->Set_size[0],
			line1, line2,
			verbose_level - 2);
	if (!find_point(
			pt, idx)) {
		cout << "surface_object::find_double_point cannot find point" << endl;
		exit(1);
	}
	return idx;

}


}}}}


