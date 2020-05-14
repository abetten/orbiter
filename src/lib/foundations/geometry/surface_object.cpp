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
namespace foundations {


#define MAX_NUMBER_OF_PLANES_FOR_PLANE_TYPE 100000



surface_object::surface_object()
{
	q = 0;
	F = NULL;
	Surf = NULL;

	//int Lines[27];
	//int eqn[20];

	Pts = NULL;
	nb_pts = 0;

	nb_planes = 0;

	pts_on_lines = NULL;
	lines_on_point = NULL;

	Eckardt_points = NULL;
	Eckardt_points_index = NULL;
	nb_Eckardt_points = 0;
	Double_points = NULL;
	Double_points_index = NULL;
	nb_Double_points = 0;

	Pts_not_on_lines = NULL;
	nb_pts_not_on_lines = 0;

	plane_type_by_points = NULL;
	plane_type_by_lines = NULL;
	C_plane_type_by_points = NULL;
	Type_pts_on_lines = NULL;
	Type_lines_on_point = NULL;
	Tritangent_plane_rk = NULL;
	Tritangent_planes = NULL;
	nb_tritangent_planes = 0;
	Lines_in_tritangent_plane = NULL;
	Tritangent_plane_dual = NULL;

	iso_type_of_tritangent_plane = NULL;
	Type_iso_tritangent_planes = NULL;

	Unitangent_planes = NULL;
	nb_unitangent_planes = 0;
	Line_in_unitangent_plane = NULL;

	Tritangent_planes_on_lines = NULL;
	Tritangent_plane_to_Eckardt = NULL;
	Eckardt_to_Tritangent_plane = NULL;
	Trihedral_pairs_as_tritangent_planes = NULL;
	Unitangent_planes_on_lines = NULL;

	All_Planes = NULL;
	Dual_point_ranks = NULL;

	Adj_line_intersection_graph = NULL;
	Line_neighbors = NULL;
	Line_intersection_pt = NULL;
	Line_intersection_pt_idx = NULL;
	//null();





}

surface_object::~surface_object()
{
	freeself();
}

void surface_object::freeself()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::freeself" << endl;
		}
	if (Pts) {
		FREE_lint(Pts);
		}
	if (Eckardt_points) {
		FREE_lint(Eckardt_points);
		}
	if (Eckardt_points_index) {
		FREE_int(Eckardt_points_index);
		}
	if (Double_points) {
		FREE_lint(Double_points);
		}
	if (Double_points_index) {
		FREE_int(Double_points_index);
		}
	if (Pts_not_on_lines) {
		FREE_lint(Pts_not_on_lines);
		}
	if (pts_on_lines) {
		FREE_OBJECT(pts_on_lines);
		}
	if (lines_on_point) {
		FREE_OBJECT(lines_on_point);
		}
	if (f_v) {
		cout << "surface_object::freeself 2" << endl;
		}
	if (plane_type_by_points) {
		FREE_int(plane_type_by_points);
		}
	if (plane_type_by_lines) {
		FREE_int(plane_type_by_lines);
		}
	if (C_plane_type_by_points) {
		FREE_OBJECT(C_plane_type_by_points);
		}
	if (Type_pts_on_lines) {
		FREE_OBJECT(Type_pts_on_lines);
		}
	if (Type_lines_on_point) {
		FREE_OBJECT(Type_lines_on_point);
		}
	if (Tritangent_plane_rk) {
		FREE_lint(Tritangent_plane_rk);
	}
	if (Tritangent_planes) {
		FREE_lint(Tritangent_planes);
		}
	if (Lines_in_tritangent_plane) {
		FREE_lint(Lines_in_tritangent_plane);
		}
	if (Tritangent_plane_dual) {
		FREE_int(Tritangent_plane_dual);
		}
	if (Unitangent_planes) {
		FREE_lint(Unitangent_planes);
		}
	if (f_v) {
		cout << "surface_object::freeself 3" << endl;
		}
	if (Line_in_unitangent_plane) {
		FREE_lint(Line_in_unitangent_plane);
		}
	if (f_v) {
		cout << "surface_object::freeself 4" << endl;
		}
	if (iso_type_of_tritangent_plane) {
		FREE_int(iso_type_of_tritangent_plane);
		}
	if (f_v) {
		cout << "surface_object::freeself 5" << endl;
		}
	if (Type_iso_tritangent_planes) {
		FREE_OBJECT(Type_iso_tritangent_planes);
		}
	if (f_v) {
		cout << "surface_object::freeself 6" << endl;
		}
	if (Tritangent_planes_on_lines) {
		FREE_int(Tritangent_planes_on_lines);
		}
	if (f_v) {
		cout << "surface_object::freeself 7" << endl;
		}
	if (Tritangent_plane_to_Eckardt) {
		FREE_int(Tritangent_plane_to_Eckardt);
		}
	if (f_v) {
		cout << "surface_object::freeself 8" << endl;
		}
	if (Eckardt_to_Tritangent_plane) {
		FREE_int(Eckardt_to_Tritangent_plane);
		}
	if (Trihedral_pairs_as_tritangent_planes) {
		FREE_lint(Trihedral_pairs_as_tritangent_planes);
		}
	if (Unitangent_planes_on_lines) {
		FREE_int(Unitangent_planes_on_lines);
		}
	if (All_Planes) {
		FREE_lint(All_Planes);
		}
	if (Dual_point_ranks) {
		FREE_int(Dual_point_ranks);
		}
	if (Adj_line_intersection_graph) {
		FREE_int(Adj_line_intersection_graph);
		}
	if (Line_neighbors) {
		FREE_OBJECT(Line_neighbors);
		}
	if (Line_intersection_pt) {
		FREE_int(Line_intersection_pt);
		}
	if (Line_intersection_pt_idx) {
		FREE_int(Line_intersection_pt_idx);
		}
	if (f_v) {
		cout << "surface_object::freeself done" << endl;
		}
}

void surface_object::null()
{
}

int surface_object::init_equation(surface_domain *Surf, int *eqn,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::init_equation" << endl;
		}

	surface_object::Surf = Surf;
	F = Surf->F;
	q = F->q;

	//int_vec_copy(Lines, surface_object::Lines, 27);
	int_vec_copy(eqn, surface_object::eqn, 20);
	
	nb_planes = Surf->P->Nb_subspaces[2];
	plane_type_by_points = NEW_int(nb_planes);
	plane_type_by_lines = NEW_int(nb_planes);


	long int *Points;
	int nb_points;
	int nb_lines;


	Points = NEW_lint(Surf->nb_pts_on_surface);
	if (f_v) {
		cout << "surface_object::init_equation before "
				"Surf->enumerate_points" << endl;
		}
	Surf->enumerate_points(surface_object::eqn, 
		Points, nb_points, 
		0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object::init_equation The surface "
				"has " << nb_points << " points" << endl;
		}
	if (nb_points != Surf->nb_pts_on_surface) {
		cout << "surface_object::init_equation nb_points != "
				"Surf->nb_pts_on_surface" << endl;
		exit(1);
		}
	
	Surf->P->find_lines_which_are_contained(Points, nb_points, 
		Lines, nb_lines, 27 /* max_lines */, 
		0 /* verbose_level */);

	if (f_v) {
		cout << "surface_object::init_equation The surface "
				"has " << nb_lines << " lines" << endl;
		}
	if (nb_lines != 27) {
		cout << "surface_object::init_equation the surface "
				"does not have 27 lines" << endl;
		exit(1);
		}

	
	FREE_lint(Points);

	if (f_v) {
		cout << "surface_object::init_equation Lines:";
		lint_vec_print(cout, Lines, 27);
		cout << endl;
		}

	if (f_v) {
		cout << "surface_object::init_equation before "
				"find_double_six_and_rearrange_lines" << endl;
		}
	find_double_six_and_rearrange_lines(Lines,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "surface_object::init_equation after "
				"find_double_six_and_rearrange_lines" << endl;
		cout << "surface_object::init_equation Lines:";
		lint_vec_print(cout, Lines, 27);
		cout << endl;
		}



	if (f_v) {
		cout << "surface_object::init_equation before "
				"compute_properties" << endl;
		}
	compute_properties(verbose_level);
	if (f_v) {
		cout << "surface_object::init_equation after "
				"compute_properties" << endl;
		}



	if (f_v) {
		cout << "surface_object::init_equation after "
				"enumerate_points" << endl;
		}
	return TRUE;

}

void surface_object::init(surface_domain *Surf,
	long int *Lines, int *eqn,
	int f_find_double_six_and_rearrange_lines, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::init" << endl;
		}
	surface_object::Surf = Surf;
	F = Surf->F;
	q = F->q;




	lint_vec_copy(Lines, surface_object::Lines, 27);

	if (f_v) {
		cout << "surface_object::init Lines:";
		lint_vec_print(cout, surface_object::Lines, 27);
		cout << endl;
		}

	if (f_find_double_six_and_rearrange_lines) {
		if (f_v) {
			cout << "surface_object::init before "
					"find_double_six_and_rearrange_lines" << endl;
			}
		find_double_six_and_rearrange_lines(surface_object::Lines,
				verbose_level);
		if (f_v) {
			cout << "surface_object::init after "
					"find_double_six_and_rearrange_lines" << endl;
			}
		}

	if (f_v) {
		cout << "surface_object::init Lines:";
		lint_vec_print(cout, surface_object::Lines, 27);
		cout << endl;
		}


	int_vec_copy(eqn, surface_object::eqn, 20);
	
	nb_planes = Surf->P->Nb_subspaces[2];
	plane_type_by_points = NEW_int(nb_planes);
	plane_type_by_lines = NEW_int(nb_planes);


	if (f_v) {
		cout << "surface_object::init before compute_properties" << endl;
		}
	compute_properties(verbose_level);
	if (f_v) {
		cout << "surface_object::init after compute_properties" << endl;
		}

	if (f_v) {
		cout << "surface_object::init done" << endl;
		}
}

void surface_object::compute_properties(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::compute_properties" << endl;
		}
	if (f_v) {
		cout << "surface_object::compute_properties before "
				"enumerate_points" << endl;
		}
	enumerate_points(verbose_level);
	if (f_v) {
		cout << "surface_object::compute_properties after "
				"enumerate_points" << endl;
		}

	if (f_v) {
		cout << "surface_object::compute_properties before "
				"compute_adjacency_matrix_of_line_intersection_"
				"graph" << endl;
		}
	compute_adjacency_matrix_of_line_intersection_graph(
			verbose_level);
	//Surf->compute_adjacency_matrix_of_line_intersection_graph(
	//Adj, Lines, 27, 0 /*verbose_level*/);
	//cout << "The adjacency matrix is:" << endl;
	//int_matrix_print(Adj, 27, 27);
	if (f_v) {
		cout << "surface_object::compute_properties after "
				"compute_adjacency_matrix_of_line_intersection_"
				"graph" << endl;
		}


	if (f_v) {
		cout << "surface_object::compute_properties before "
				"compute_tritangent_planes_by_rank" << endl;
		}
	compute_tritangent_planes_by_rank(verbose_level - 1);
	if (f_v) {
		cout << "surface_object::compute_properties after "
				"compute_tritangent_planes_by_rank" << endl;
		}


	if (f_v) {
		cout << "surface_object::compute_properties before "
				"compute_plane_type_by_points" << endl;
		}
	compute_plane_type_by_points(verbose_level - 1);
	if (f_v) {
		cout << "surface_object::compute_properties after "
				"compute_plane_type_by_points" << endl;
		}

	if (f_v) {
		cout << "surface_object::compute_properties before "
				"compute_tritangent_planes" << endl;
		}
	compute_tritangent_planes(verbose_level - 1);
	if (f_v) {
		cout << "surface_object::compute_properties after "
				"compute_tritangent_planes" << endl;
		}

	if (f_v) {
		cout << "surface_object::compute_properties before "
				"compute_planes_and_dual_point_ranks" << endl;
		}
	compute_planes_and_dual_point_ranks(verbose_level - 1);
	if (f_v) {
		cout << "surface_object::compute_properties after "
				"compute_planes_and_dual_point_ranks" << endl;
		}
	if (f_v) {
		cout << "surface_object::compute_properties done" << endl;
		}
}


void surface_object::find_double_six_and_rearrange_lines(
	long int *Lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int Lines0[27];
	long int Lines1[27];
	long int double_six[12];
	int *Adj;
	set_of_sets *line_intersections;
	int *Starter_Table;
	int nb_starter;
	int l, line_idx, subset_idx;
	long int S3[6];
	sorting Sorting;



	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines" << endl;
		}
	lint_vec_copy(Lines, Lines0, 27);

	Surf->compute_adjacency_matrix_of_line_intersection_graph(
			Adj, Lines0, 27, 0 /* verbose_level */);

	line_intersections = NEW_OBJECT(set_of_sets);

	line_intersections->init_from_adjacency_matrix(27,
			Adj, 0 /* verbose_level */);

	Surf->list_starter_configurations(Lines0, 27, 
		line_intersections, Starter_Table, nb_starter, verbose_level);


	if (nb_starter != 432) {
		cout << "surface_object::find_double_six_and_rearrange_lines nb_starter != 432" << endl;
		exit(1);
		}
	l = 0;
	line_idx = Starter_Table[l * 2 + 0];
	subset_idx = Starter_Table[l * 2 + 1];

	Surf->create_starter_configuration(line_idx, 
		subset_idx, line_intersections, 
		Lines0, S3, 
		0 /* verbose_level */);



	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"before Surf->create_double_six_from_five_lines_with_a_common_transversal" << endl;
		}
	if (!Surf->create_double_six_from_five_lines_with_a_common_transversal(
		S3, double_six, verbose_level)) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"The starter configuration is bad, there "
				"is no double six" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines after "
				"Surf->create_double_six_from_five_lines_with_a_common_transversal" << endl;
		}


	lint_vec_copy(double_six, Lines1, 12);
	
	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"before Surf->create_remaining_fifteen_lines" << endl;
		}
	Surf->create_remaining_fifteen_lines(double_six, 
		Lines1 + 12, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object::find_double_six_and_rearrange_lines "
				"after Surf->create_remaining_fifteen_lines" << endl;
		}

	lint_vec_copy(Lines1, Lines, 27);
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




void surface_object::enumerate_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vvv = (verbose_level >= 3);
	sorting Sorting;
	latex_interface L;
	
	if (f_v) {
		cout << "surface_object::enumerate_points" << endl;
		}

	
	Pts = NEW_lint(Surf->nb_pts_on_surface);
	Surf->enumerate_points(eqn, Pts, nb_pts, 0 /* verbose_level */);

	if (nb_pts != Surf->nb_pts_on_surface) {
		cout << "surface_object::enumerate_points nb_pts != "
				"Surf->nb_pts_on_surface" << endl;
		exit(1);
		}
	Sorting.lint_vec_heapsort(Pts, nb_pts);
	if (f_v) {
		cout << "surface_object::enumerate_points we found "
				<< nb_pts << " points on the surface" << endl;
		}
	if (f_vvv) {
		cout << "surface_object::enumerate_points The points "
				"on the surface are:" << endl;
		L.print_lint_matrix_with_standard_labels(cout,
			Pts, nb_pts, 1, FALSE /* f_tex */);
		}

	if (f_v) {
		cout << "surface_object::enumerate_points before "
				"Surf->compute_points_on_lines" << endl;
		}
	Surf->compute_points_on_lines(Pts, nb_pts, 
		Lines, 27, 
		pts_on_lines, 
		0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object::enumerate_points after "
				"Surf->compute_points_on_lines" << endl;
		}

	pts_on_lines->sort();
	
	if (f_vvv) {
		cout << "pts_on_lines:" << endl;
		pts_on_lines->print_table();
		}

	Type_pts_on_lines = NEW_OBJECT(classify);
	Type_pts_on_lines->init(pts_on_lines->Set_size, 
		pts_on_lines->nb_sets, FALSE, 0);
	if (f_v) {
		cout << "type of pts_on_lines:" << endl;
		Type_pts_on_lines->print_naked_tex(cout, TRUE);
		cout << endl;
		}

	pts_on_lines->dualize(lines_on_point, 0 /* verbose_level */);
	if (f_vvv) {
		cout << "lines_on_point:" << endl;
		lines_on_point->print_table();
		}

	Type_lines_on_point = NEW_OBJECT(classify);
	Type_lines_on_point->init(lines_on_point->Set_size, 
		lines_on_point->nb_sets, FALSE, 0);
	if (f_v) {
		cout << "type of lines_on_point:" << endl;
		Type_lines_on_point->print_naked_tex(cout, TRUE);
		cout << endl;
		}

	if (f_v) {
		cout << "computing Eckardt points:" << endl;
		}
	Type_lines_on_point->get_class_by_value(Eckardt_points_index, 
		nb_Eckardt_points, 3 /* value */, 0 /* verbose_level */);
	Sorting.int_vec_heapsort(Eckardt_points_index, nb_Eckardt_points);
	if (f_v) {
		cout << "computing Eckardt points done, we found "
				<< nb_Eckardt_points << " Eckardt points" << endl;
		}
	if (f_vvv) {
		cout << "Eckardt_points_index="; 
		int_vec_print(cout, Eckardt_points_index, nb_Eckardt_points);
		cout << endl;
		}
	Eckardt_points = NEW_lint(nb_Eckardt_points);
	int_vec_apply_lint(Eckardt_points_index, Pts,
		Eckardt_points, nb_Eckardt_points);
	if (f_v) {
		cout << "computing Eckardt points done, we found "
				<< nb_Eckardt_points << " Eckardt points" << endl;
		}
	if (f_vvv) {
		cout << "Eckardt_points="; 
		lint_vec_print(cout, Eckardt_points, nb_Eckardt_points);
		cout << endl;
		}

	if (f_v) {
		cout << "computing Double points:" << endl;
		}
	Type_lines_on_point->get_class_by_value(Double_points_index, 
		nb_Double_points, 2 /* value */, 0 /* verbose_level */);
	Sorting.int_vec_heapsort(Double_points_index, nb_Double_points);
	if (f_v) {
		cout << "computing Double points done, we found "
				<< nb_Double_points << " Double points" << endl;
		}
	if (f_vvv) {
		cout << "Double_points_index="; 
		int_vec_print(cout, Double_points_index, nb_Double_points);
		cout << endl;
		}
	Double_points = NEW_lint(nb_Double_points);
	int_vec_apply_lint(Double_points_index, Pts,
		Double_points, nb_Double_points);
	if (f_v) {
		cout << "computing Double points done, we found "
				<< nb_Double_points << " Double points" << endl;
		}
	if (f_vvv) {
		cout << "Double_points="; 
		lint_vec_print(cout, Double_points, nb_Double_points);
		cout << endl;
		}

	Pts_not_on_lines = NEW_lint(nb_pts);
	lint_vec_copy(Pts, Pts_not_on_lines, nb_pts);
	nb_pts_not_on_lines = nb_pts;

	int i, j, a, b, idx, h;

	for (i = 0; i < pts_on_lines->nb_sets; i++) {

		for (j = 0; j < pts_on_lines->Set_size[i]; j++) {
			a = pts_on_lines->Sets[i][j];
			b = Pts[a];

			if (Sorting.lint_vec_search(Pts_not_on_lines,
				nb_pts_not_on_lines, b, idx, 0)) {
				for (h = idx + 1; h < nb_pts_not_on_lines; h++) {
					Pts_not_on_lines[h - 1] = Pts_not_on_lines[h];
					}
				nb_pts_not_on_lines--;
				}
			}
		}
	if (f_v) {
		cout << "nb_pts_not_on_lines=" << nb_pts_not_on_lines << endl;
		}



	if (f_v) {
		cout << "surface_object::enumerate_points done" << endl;
		}
}

void surface_object::compute_adjacency_matrix_of_line_intersection_graph(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_object::compute_adjacency_matrix_of_"
				"line_intersection_graph" << endl;
		}

	if (f_v) {
		cout << "surface_object::compute_adjacency_matrix_of_"
				"line_intersection_graph before Surf->compute_adjacency_"
				"matrix_of_line_intersection_graph" << endl;
		}
	Surf->compute_adjacency_matrix_of_line_intersection_graph(
		Adj_line_intersection_graph, Lines, 27, verbose_level - 2);
	if (f_v) {
		cout << "surface_object::compute_adjacency_matrix_of_"
				"line_intersection_graph after Surf->compute_adjacency_"
				"matrix_of_line_intersection_graph" << endl;
		}

	Line_neighbors = NEW_OBJECT(set_of_sets);
	Line_neighbors->init_from_adjacency_matrix(27, 
		Adj_line_intersection_graph, 0 /* verbose_level*/);
	
	if (f_v) {
		cout << "surface_object::compute_adjacency_matrix_of_"
				"line_intersection_graph before Surf->compute_"
				"intersection_points_and_indices" << endl;
		}
	Surf->compute_intersection_points_and_indices(
		Adj_line_intersection_graph, 
		Pts, nb_pts, 
		Lines, 27 /* nb_lines */, 
		Line_intersection_pt, Line_intersection_pt_idx, 
		verbose_level);
	if (f_v) {
		cout << "surface_object::compute_adjacency_matrix_of_"
				"line_intersection_graph after Surf->compute_"
				"intersection_points_and_indices" << endl;
		}
#if 0
	Surf->compute_intersection_points(Adj_line_intersection_graph,
			Line_intersection_pt, Line_intersection_pt_idx,
		Lines, 27, verbose_level - 2);
#endif

	if (f_v) {
		cout << "surface_object::compute_adjacency_matrix_of_"
				"line_intersection_graph done" << endl;
		}

}


void surface_object::compute_plane_type_by_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N_planes;
	
	if (f_v) {
		cout << "surface_object::compute_plane_type_by_points" << endl;
		}

	N_planes = Surf->P->nb_rk_k_subspaces_as_lint(3);

	if (N_planes < MAX_NUMBER_OF_PLANES_FOR_PLANE_TYPE) {
		Surf->P->plane_intersection_type_basic(Pts, nb_pts,
			plane_type_by_points, 0 /* verbose_level */);


		C_plane_type_by_points = NEW_OBJECT(classify);
	
		C_plane_type_by_points->init(plane_type_by_points, nb_planes, FALSE, 0);
		if (f_v) {
			cout << "plane types by points: ";
			C_plane_type_by_points->print_naked(TRUE);
			cout << endl;
			}
	}
	else {
		cout << "surface_object::compute_plane_type_by_points "
				"too many planes, skipping plane type " << endl;
	}


	if (f_v) {
		cout << "surface_object::compute_plane_type_by_points done" << endl;
		}
}

void surface_object::compute_tritangent_planes_by_rank(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::compute_tritangent_planes_by_rank" << endl;
		}

	Tritangent_plane_rk = NEW_lint(45);

	int tritangent_plane_idx;
	int three_lines_idx[3];
	long int three_lines[3];
	int i, r;
	int Basis[6 * 4];
	int base_cols[4];

	for (tritangent_plane_idx = 0;
			tritangent_plane_idx < 45;
			tritangent_plane_idx++) {
		Surf->Eckardt_points[tritangent_plane_idx].three_lines(
				Surf, three_lines_idx);

		for (i = 0; i < 3; i++) {
			three_lines[i] = Lines[three_lines_idx[i]];
			Surf->Gr->unrank_lint_here(Basis + i * 8,
					three_lines[i], 0 /* verbose_level */);
		}
		r = F->Gauss_simple(Basis, 6, 4,
			base_cols, 0 /* verbose_level */);
		if (r != 3) {
			cout << "surface_object::compute_tritangent_planes_by_rank r != 3" << endl;
			exit(1);
		}
		Tritangent_plane_rk[tritangent_plane_idx] =
				Surf->Gr3->rank_lint_here(Basis, 0 /* verbose_level */);
	}
	if (TRUE) {
		cout << "surface_object::compute_tritangent_planes_by_rank" << endl;
		for (tritangent_plane_idx = 0;
				tritangent_plane_idx < 45;
				tritangent_plane_idx++) {
			cout << tritangent_plane_idx << " : " << Tritangent_plane_rk[tritangent_plane_idx] << endl;
		}
	}



	if (f_v) {
		cout << "surface_object::compute_tritangent_planes_by_rank done" << endl;
		}
}

void surface_object::compute_tritangent_planes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	latex_interface L;

	if (f_v) {
		cout << "surface_object::compute_tritangent_planes" << endl;
		}

	
	if (f_v) {
		cout << "surface_object::compute_tritangent_planes "
				"computing tritangent planes:" << endl;
		}
	Surf->compute_tritangent_planes(Lines, 
		Tritangent_planes, nb_tritangent_planes, 
		Unitangent_planes, nb_unitangent_planes, 
		Lines_in_tritangent_plane, 
		Line_in_unitangent_plane, 
		verbose_level);


	if (f_v) {
		cout << "surface_object::compute_tritangent_planes "
				"Lines_in_tritangent_plane: " << endl;
		L.print_lint_matrix_with_standard_labels(cout,
				Lines_in_tritangent_plane, nb_tritangent_planes, 3,
				FALSE);
		}


	int *Cnt;
	int i, j, a;

	Cnt = NEW_int(27);
	int_vec_zero(Cnt, 27);
	Tritangent_planes_on_lines = NEW_int(27 * 5);
	for (i = 0; i < nb_tritangent_planes; i++) {
		for (j = 0; j < 3; j++) {
			a = Lines_in_tritangent_plane[i * 3 + j];
			Tritangent_planes_on_lines[a * 5 + Cnt[a]++] = i;
			}
		}
	for (i = 0; i < 27; i++) {
		if (Cnt[i] != 5) {
			cout << "surface_object::compute_tritangent_planes "
					"Cnt[i] != 5" << endl;
			exit(1);
			}
		}
	FREE_int(Cnt);


	Unitangent_planes_on_lines = NEW_int(27 * (q + 1 - 5));
	Cnt = NEW_int(27);
	int_vec_zero(Cnt, 27);
	for (i = 0; i < nb_unitangent_planes; i++) {
		a = Line_in_unitangent_plane[i];
		Unitangent_planes_on_lines[a * (q + 1 - 5) + Cnt[a]++] = i;
		}
	for (i = 0; i < 27; i++) {
		if (Cnt[i] != (q + 1 - 5)) {
			cout << "surface_object::compute_tritangent_planes "
					"Cnt[i] != (q + 1 - 5)" << endl;
			exit(1);
			}
		}
	FREE_int(Cnt);


	iso_type_of_tritangent_plane = NEW_int(nb_tritangent_planes);
	for (i = 0; i < nb_tritangent_planes; i++) {
		long int three_lines[3];
		for (j = 0; j < 3; j++) {
			three_lines[j] = Lines[Lines_in_tritangent_plane[i * 3 + j]];
			}
		iso_type_of_tritangent_plane[i] = 
			Surf->identify_three_lines(
			three_lines, 0 /* verbose_level */);
		}

	Type_iso_tritangent_planes = NEW_OBJECT(classify);
	Type_iso_tritangent_planes->init(iso_type_of_tritangent_plane, 
		nb_tritangent_planes, FALSE, 0);
	if (f_v) {
		cout << "Type iso of tritangent planes: ";
		Type_iso_tritangent_planes->print_naked(TRUE);
		cout << endl;
		}
	
	Tritangent_plane_to_Eckardt = NEW_int(nb_tritangent_planes);
	Eckardt_to_Tritangent_plane = NEW_int(nb_tritangent_planes);
	Tritangent_plane_dual = NEW_int(nb_tritangent_planes);
	for (i = 0; i < nb_tritangent_planes; i++) {
		int three_lines[3];
		for (j = 0; j < 3; j++) {
			three_lines[j] = Lines_in_tritangent_plane[i * 3 + j];
			}
		a = Surf->Eckardt_point_from_tritangent_plane(three_lines);
		Tritangent_plane_to_Eckardt[i] = a;
		Eckardt_to_Tritangent_plane[a] = i;
		}
	for (i = 0; i < nb_tritangent_planes; i++) {
		Tritangent_plane_dual[i] = 
			Surf->P->dual_rank_of_plane_in_three_space(
			Tritangent_planes[i], 0 /* verbose_level */);
		}


	Trihedral_pairs_as_tritangent_planes = NEW_lint(Surf->nb_trihedral_pairs * 6);
	for (i = 0; i < Surf->nb_trihedral_pairs; i++) {
		for (j = 0; j < 6; j++) {
			a = Surf->Trihedral_to_Eckardt[i * 6 + j];
			Trihedral_pairs_as_tritangent_planes[i * 6 + j] = Eckardt_to_Tritangent_plane[a];
			}
		}


	if (f_v) {
		cout << "surface_object::compute_tritangent_planes done" << endl;
		}
}

void surface_object::compute_planes_and_dual_point_ranks(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	
	if (f_v) {
		cout << "surface_object::compute_planes_and_dual_point_ranks" << endl;
		}
	
	All_Planes = NEW_lint(Surf->nb_trihedral_pairs * 6);
	Dual_point_ranks = NEW_int(Surf->nb_trihedral_pairs * 6);
	//Iso_trihedral_pair = NEW_int(Surf->nb_trihedral_pairs);


	Surf->Trihedral_pairs_to_planes(Lines, All_Planes,
			0 /*verbose_level*/);
	
	
	for (i = 0; i < Surf->nb_trihedral_pairs; i++) {
		//cout << "trihedral pair " << i << " / "
		// << Surf->nb_trihedral_pairs << endl;
		for (j = 0; j < 6; j++) {
			Dual_point_ranks[i * 6 + j] = 
				Surf->P->dual_rank_of_plane_in_three_space(
				All_Planes[i * 6 + j], 0 /* verbose_level */);
			}

		}
	if (f_v) {
		cout << "surface_object::compute_planes_and_dual_point_ranks done" << endl;
		}
}

void surface_object::report_properties(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object::report_properties" << endl;
	}


	if (f_v) {
		cout << "surface_object::report_properties before print_general" << endl;
	}
	print_general(ost);


	if (f_v) {
		cout << "surface_object::report_properties before print_lines" << endl;
	}
	print_lines(ost);

	if (f_v) {
		cout << "surface_object::report_properties before print_points" << endl;
	}
	print_points(ost);


	if (f_v) {
		cout << "surface_object::report_properties print_tritangent_planes" << endl;
	}
	print_tritangent_planes(ost);


	if (f_v) {
		cout << "surface_object::report_properties "
				"before print_Steiner_and_Eckardt" << endl;
	}
	print_Steiner_and_Eckardt(ost);

	//SOA->SO->print_planes_in_trihedral_pairs(fp);

	if (f_v) {
		cout << "surface_object::report_properties "
				"before print_generalized_quadrangle" << endl;
	}
	print_generalized_quadrangle(ost);

	if (f_v) {
		cout << "surface_object::report_properties done" << endl;
	}
}

void surface_object::print_line_intersection_graph(ostream &ost)
{
	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Line Intersection Graph}" << endl;

	print_adjacency_matrix(ost);

	print_adjacency_matrix_with_intersection_points(ost);

	print_neighbor_sets(ost);
}

void surface_object::print_adjacency_matrix(ostream &ost)
{
	int i, j, m, n;
	int *p;
	
	m = 27;
	n = 27;
	p = Adj_line_intersection_graph;
	ost << "{\\arraycolsep=1pt" << endl;
	//int block_width = 27;
	//print_integer_matrix_tex_block_by_block(ost,
	//Adj_line_intersection_graph, 27, 27, block_width);
	ost << "$$" << endl;
	ost << "\\begin{array}{rr|*{" << n << "}r}" << endl;
	ost << " & ";
	for (j = 0; j < n; j++) {
		ost << " & " << j;
		}
	ost << "\\\\" << endl;
	ost << " & ";
	for (j = 0; j < n; j++) {
		ost << " & " << Surf->Line_label_tex[j];
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << i << " & " << Surf->Line_label_tex[i];
		for (j = 0; j < n; j++) {
			ost << " & " << p[i * n + j];
			}
		ost << "\\\\";
		ost << endl;
		}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
	ost << "}%%" << endl;
}

void surface_object::print_adjacency_matrix_with_intersection_points(
		ostream &ost)
{
	int i, j, m, n, idx;
	int *p;
	
	m = 27;
	n = 27;
	p = Adj_line_intersection_graph;
	ost << "{\\arraycolsep=1pt" << endl;
	//int block_width = 27;
	//print_integer_matrix_tex_block_by_block(ost,
	//Adj_line_intersection_graph, 27, 27, block_width);
	ost << "$$" << endl;
	ost << "\\begin{array}{rr|*{" << n << "}r}" << endl;
	ost << " & ";
	for (j = 0; j < n; j++) {
		ost << " & " << j;
		}
	ost << "\\\\" << endl;
	ost << " & ";
	for (j = 0; j < n; j++) {
		ost << " & " << Surf->Line_label_tex[j];
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << i << " & " << Surf->Line_label_tex[i];
		for (j = 0; j < n; j++) {
			ost << " & ";
			if (p[i * n + j]) {
				//a = Line_intersection_pt[i * n + j];
				idx = Line_intersection_pt_idx[i * n + j];
				ost << "P_{" << idx << "}";
				}
			else {
				ost << ".";
				}
			}
		ost << "\\\\";
		ost << endl;
		}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
	ost << "}%%" << endl;
}

void surface_object::print_neighbor_sets(ostream &ost)
{
	int i, j, h, p, idx;
	sorting Sorting;

	//ost << "\\clearpage" << endl;
	ost << "Neighbor sets in the line intersection graph:\\\\" << endl;
	//Line_neighbors->print_table_tex(ost);
	for (i = 0; i < Line_neighbors->nb_sets; i++) {
		ost << "Line " << i << " intersects " << endl;
		ost << "$$" << endl;
		ost << "\\begin{array}{|r*{"
				<< Line_neighbors->Set_size[i] << "}{|c}|}" << endl;
		//int_set_print_tex(ost, Line_neighbors->Sets[i],
		//Line_neighbors->Set_size[i]);
		ost << "\\hline" << endl;
		ost << "\\mbox{Line} ";
		for (h = 0; h < Line_neighbors->Set_size[i]; h++) {
			j = Line_neighbors->Sets[i][h];
			ost << " & " << j;
			}
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\mbox{in point} ";
		for (h = 0; h < Line_neighbors->Set_size[i]; h++) {
			j = Line_neighbors->Sets[i][h];
			p = Line_intersection_pt[i * 27 + j];
			if (!Sorting.lint_vec_search_linear(Pts, nb_pts, p, idx)) {
				cout << "surface_object::print_line_intersection_graph "
						"did not find intersection point" << endl;
				exit(1);
				}
			ost << " & " << idx;
			}
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
		}
}

void surface_object::print_planes_in_trihedral_pairs(ostream &ost)
{
	latex_interface L;

	ost << "\\clearpage\n\\subsection*{All planes "
			"in trihedral pairs}" << endl;

	ost << "All planes by plane rank:" << endl;

	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
		All_Planes, 30, 6, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		All_Planes + 30 * 6, 30, 6, 30, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		All_Planes + 60 * 6, 30, 6, 60, 0, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		All_Planes + 90 * 6, 30, 6, 90, 0, TRUE /* f_tex */);
	ost << "$$" << endl;



	ost << "All planes by dual point rank:" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
		Dual_point_ranks, 30, 6, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
		Dual_point_ranks + 30 * 6, 30, 6, 30, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
		Dual_point_ranks + 60 * 6, 30, 6, 60, 0, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
		Dual_point_ranks + 90 * 6, 30, 6, 90, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
}

void surface_object::print_tritangent_planes(ostream &ost)
{
	int i, j, plane_rk, b, v4[4];
	int Mtx[16];

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Tritangent planes}" << endl;
	ost << "The " << nb_tritangent_planes << " tritangent "
			"planes are:\\\\" << endl;
	for (i = 0; i < nb_tritangent_planes; i++) {
		j = Eckardt_to_Tritangent_plane[i];
		plane_rk = Tritangent_planes[j];
		b = Tritangent_plane_dual[j];
		ost << "$$" << endl;
		ost << "\\pi_{" << Surf->Eckard_point_label_tex[i] << "} = ";
		ost << "\\pi_{" << i << "} = " << plane_rk << " = ";
		//ost << "\\left[" << endl;
		Surf->Gr3->print_single_generator_matrix_tex(ost, plane_rk);
		//ost << "\\right]" << endl;
		ost << " = ";
		Surf->Gr3->print_single_generator_matrix_tex_numerical(ost, plane_rk);

		Surf->Gr3->unrank_lint_here_and_compute_perp(Mtx, plane_rk,
			0 /*verbose_level */);
		F->PG_element_normalize(Mtx + 12, 1, 4);
		ost << "$$" << endl;
		ost << "$$" << endl;
		ost << "=V\\big(" << endl;
		Surf->Poly1_4->print_equation(ost, Mtx + 12);
		ost << "\\big)" << endl;
		ost << "=V\\big(" << endl;
		Surf->Poly1_4->print_equation_numerical(ost, Mtx + 12);
		ost << "\\big)" << endl;
		ost << "$$" << endl;
		ost << "dual pt rank = $" << b << "$ ";
		F->PG_element_unrank_modified(v4, 1, 4, b);
		ost << "$=";
		int_vec_print(ost, v4, 4);
		ost << "$.\\\\" << endl;
		}

#if 0
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Tritangent_planes, 9, 5, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "Their dual point ranks are:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Tritangent_plane_dual, 9, 5, TRUE /* f_tex */);
	ost << "$$" << endl;

	for (i = 0; i < nb_tritangent_planes; i++) {
		a = Tritangent_planes[i];
		b = Tritangent_plane_dual[i];
		//b = Surf->P->dual_rank_of_plane_in_three_space(a, 0);
		ost << "plane " << i << " / " << nb_tritangent_planes
				<< " : rank " << a << " is $";
		ost << "\\left[" << endl;
		Surf->Gr3->print_single_generator_matrix_tex(ost, a);
		ost << "\\right]" << endl;
		ost << "$, dual pt rank = $" << b << "$ ";
		PG_element_unrank_modified(*F, v4, 1, 4, b);
		ost << "$=";
		int_vec_print(ost, v4, 4);
		ost << "$\\\\" << endl;
		}

	ost << "The iso types of the tritangent planes are:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			iso_type_of_tritangent_plane, nb_tritangent_planes / 5, 5,
			TRUE /* f_tex */);
	ost << "$$" << endl;

	ost << "Type iso of tritangent planes: ";
	ost << "$$" << endl;
	Type_iso_tritangent_planes->print_naked_tex(ost, TRUE);
	ost << endl;
	ost << "$$" << endl;
	ost << endl;

	ost << "Tritangent\\_plane\\_to\\_Eckardt:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Tritangent_plane_to_Eckardt, nb_tritangent_planes / 5, 5,
			TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "Eckardt\\_to\\_Tritangent\\_plane:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Eckardt_to_Tritangent_plane, nb_tritangent_planes / 5, 5,
			TRUE /* f_tex */);
	ost << "$$" << endl;

	ost << "Trihedral\\_pairs\\_as\\_tritangent\\_planes:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Trihedral_pairs_as_tritangent_planes, 30, 6, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Trihedral_pairs_as_tritangent_planes + 30 * 6, 30, 6, 30, 0,
			TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Trihedral_pairs_as_tritangent_planes + 60 * 6, 30, 6, 60, 0,
			TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Trihedral_pairs_as_tritangent_planes + 90 * 6, 30, 6, 90, 0,
			TRUE /* f_tex */);
	ost << "$$" << endl;
#endif

}

void surface_object::print_generalized_quadrangle(ostream &ost)
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
			TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Lines_in_tritangent_plane + 15 * 3, 15, 3, 15, 0,
			TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Lines_in_tritangent_plane + 30 * 3, 15, 3, 30, 0,
			TRUE /* f_tex */);
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
		Tritangent_planes_on_lines, 9, 5, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
		Tritangent_planes_on_lines + 9 * 5, 9, 5, 9, 0,
		TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
		Tritangent_planes_on_lines + 18 * 5, 9, 5, 18, 0,
		TRUE /* f_tex */);
	ost << "$$" << endl;
#endif

#if 0
	ost << "The unitangent planes through the 27 lines are:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
		Unitangent_planes_on_lines, 27, q + 1 - 5,
		TRUE /* f_tex */);
	ost << "$$" << endl;
#endif

}

void surface_object::print_plane_type_by_points(ostream &ost)
{
	ost << "\\subsection*{Plane types by points}" << endl;
		//*fp << "$$" << endl;
		//*fp << "\\Big(" << endl;
	C_plane_type_by_points->print_naked_tex(ost, TRUE);
		//*fp << "\\Big)" << endl;
	ost << "\\\\" << endl;
}

void surface_object::print_lines(ostream &ost)
{
	ost << "\\subsection*{The 27 lines}" << endl;
	Surf->print_lines_tex(ost, Lines);
}

void surface_object::print_lines_with_points_on_them(ostream &ost)
{
	latex_interface L;

	ost << "\\subsection*{The 27 lines with points on them}" << endl;
	int i;
	
	for (i = 0; i < 27; i++) {
		//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
		Surf->Gr->unrank_lint(Lines[i], 0 /*verbose_level*/);
		ost << "$$" << endl;
		ost << "\\ell_{" << i << "} = " << Surf->Line_label_tex[i]
			<< " = \\left[" << endl;
		//print_integer_matrix_width(cout, Gr->M,
		// k, n, n, F->log10_of_q + 1);
		L.print_integer_matrix_tex(ost, Surf->Gr->M, 2, 4);
		ost << "\\right]_{" << Lines[i] << "}" << endl;
		ost << "$$" << endl;
		ost << "which contains the point set " << endl;
		ost << "$$" << endl;
		ost << "\\{ P_{i} \\mid i \\in ";
		L.lint_set_print_tex(ost, pts_on_lines->Sets[i],
				pts_on_lines->Set_size[i]);
		ost << "\\}." << endl;
		ost << "$$" << endl; 
		}
}

void surface_object::print_equation(ostream &ost)
{
	ost << "\\subsection*{The equation}" << endl;
	ost << "The equation of the surface ";
	ost << " is :" << endl;
	ost << "$$" << endl;
	Surf->print_equation_tex(ost, eqn);
	ost << endl << "=0\n$$" << endl;
	int_vec_print(ost, eqn, 20);
	ost << "\\\\" << endl;
	ost << "Number of points on the surface " << nb_pts << "\\\\" << endl;
}

void surface_object::print_general(ostream &ost)
{
	ost << "\\subsection*{General information}" << endl;

	if (C_plane_type_by_points) {
		ost << "Plane types by points: ";
		ost << "$$" << endl;
		C_plane_type_by_points->print_naked_tex(ost, TRUE);
		ost << "$$" << endl;
	}
	ost << "Type of pts on lines:" << endl;
	ost << "$$" << endl;
	Type_pts_on_lines->print_naked_tex(ost, TRUE);
	ost << "$$" << endl;
	ost << endl;
	ost << "Type of lines on point:" << endl;
	ost << "$$" << endl;
	Type_lines_on_point->print_naked_tex(ost, TRUE);
	ost << "$$" << endl;
	ost << endl;
	ost << "Type iso of tritangent planes: ";
	ost << "$$" << endl;
	Type_iso_tritangent_planes->print_naked_tex(ost, TRUE);
	ost << "$$" << endl;
	ost << endl;
}

void surface_object::print_affine_points_in_source_code(ostream &ost)
{
	int i, j, cnt;
	int v[4];

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Affine points on surface}" << endl;
	ost << "\\begin{verbatim}" << endl;
	ost << "int Pts[] = {" << endl;
	cnt = 0;
	for (i = 0; i < nb_pts; i++) {
		Surf->unrank_point(v, Pts[i]);
		Surf->F->PG_element_normalize(v, 1, 4);
		if (v[3]) {
			ost << "\t";
			for (j = 0; j < 4; j++) {
				ost << v[j] << ", ";
			}
			ost << endl;
			cnt++;
		}
	}
	ost << "};" << endl;
	ost << "nb_affine_pts = " << cnt << ";" << endl;
	ost << "\\end{verbatim}" << endl;
}

void surface_object::print_points(ostream &ost)
{
	latex_interface L;
	int i, j, p, a, b, c;
	int v[4];

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Points on surface}" << endl;
	ost << "\\subsubsection*{All Points}" << endl;
	ost << "The surface has " << nb_pts << " points:\\\\" << endl;

	if (nb_pts < 1000) {
		ost << "$$" << endl;
		L.lint_vec_print_as_matrix(ost, Pts, nb_pts, 10, TRUE /* f_tex */);
		ost << "$$" << endl;
		//ost << "\\clearpage" << endl;
		ost << "The points on the surface are:\\\\" << endl;
		ost << "\\begin{multicols}{2}" << endl;
		ost << "\\noindent" << endl;
		for (i = 0; i < nb_pts; i++) {
			Surf->unrank_point(v, Pts[i]);
			ost << i << " : $P_{" << i << "} = P_{" << Pts[i] << "}=";
			int_vec_print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}

	//ost << "\\clearpage" << endl;
	ost << "\\subsubsection*{Eckardt Points}" << endl;
	ost << "The surface has " << nb_Eckardt_points
			<< " Eckardt points:\\\\" << endl;
	ost << "$$" << endl;
	L.lint_vec_print_as_matrix(ost,
			Eckardt_points, nb_Eckardt_points, 10,
			TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.int_vec_print_as_matrix(ost,
			Eckardt_points_index, nb_Eckardt_points, 10,
			TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "%%\\clearpage" << endl;
	ost << "The Eckardt points are:\\\\" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	ost << "\\begin{align*}" << endl;
	for (i = 0; i < nb_Eckardt_points; i++) {
		p = Eckardt_points_index[i];
		Surf->unrank_point(v, Eckardt_points[i]);
		ost << "E_{" << i << "} &= P_{" << p
				<< "} = P_{" << Eckardt_points[i] << "}=";
		int_vec_print_fully(ost, v, 4);
		if (lines_on_point->Set_size[p] != 3) {
			cout << "Eckardt point is not on three lines" << endl;
			exit(1);
			}
		a = lines_on_point->Sets[p][0];
		b = lines_on_point->Sets[p][1];
		c = lines_on_point->Sets[p][2];
		ost << " = ";
		ost << "\\ell_{" << a << "} \\cap ";
		ost << "\\ell_{" << b << "} \\cap ";
		ost << "\\ell_{" << c << "}";
		ost << " = ";
		ost << Surf->Line_label_tex[a] << " \\cap ";
		ost << Surf->Line_label_tex[b] << " \\cap ";
		ost << Surf->Line_label_tex[c];
		ost << "\\\\" << endl;
		}
	ost << "\\end{align*}" << endl;
	//ost << "\\end{multicols}" << endl;



	ost << "%%\\clearpage" << endl;
	ost << "The Eckardt points are:\\\\" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	ost << "\\begin{align*}" << endl;
	for (i = 0; i < nb_Eckardt_points; i++) {
		p = Eckardt_points_index[i];
		Surf->unrank_point(v, Eckardt_points[i]);
		ost << "E_{" << i << "} &= P_{" << Eckardt_points[i] << "}=\\bP(";
		//int_vec_print_fully(ost, v, 4);
		for (j = 0; j < 4; j++) {
			F->print_element(ost, v[j]);
			if (j < 4 - 1) {
				ost << ", ";
			}
		}
		ost << ")";
		ost << " = \\bP(";
		for (j = 0; j < 4; j++) {
			ost << v[j];
			if (j < 4 - 1) {
				ost << ", ";
			}
		}
		ost << ")";

		if (lines_on_point->Set_size[p] != 3) {
			cout << "Eckardt point is not on three lines" << endl;
			exit(1);
			}
		a = lines_on_point->Sets[p][0];
		b = lines_on_point->Sets[p][1];
		c = lines_on_point->Sets[p][2];
		ost << " = ";
		//ost << "\\ell_{" << a << "} \\cap ";
		//ost << "\\ell_{" << b << "} \\cap ";
		//ost << "\\ell_{" << c << "}";
		//ost << " = ";
		ost << Surf->Line_label_tex[a] << " \\cap ";
		ost << Surf->Line_label_tex[b] << " \\cap ";
		ost << Surf->Line_label_tex[c];
		if (i < nb_Eckardt_points - 1) {
			ost << ",";
		}
		else {
			ost << ".";
		}
		ost << "\\\\" << endl;
		}
	ost << "\\end{align*}" << endl;



	//ost << "\\clearpage" << endl;
	ost << "\\subsubsection*{Double Points}" << endl;
	ost << "The surface has " << nb_Double_points
			<< " Double points:\\\\" << endl;
	if (nb_Double_points < 1000) {
		ost << "$$" << endl;
		L.lint_vec_print_as_matrix(ost,
				Double_points, nb_Double_points, 10,
				TRUE /* f_tex */);
		ost << "$$" << endl;
		ost << "$$" << endl;
		L.int_vec_print_as_matrix(ost,
				Double_points_index, nb_Double_points, 10,
				TRUE /* f_tex */);
		ost << "$$" << endl;
		//ost << "\\clearpage" << endl;
		ost << "The Double points on the surface are:\\\\" << endl;
		ost << "\\begin{multicols}{2}" << endl;
		ost << "\\noindent" << endl;
		for (i = 0; i < nb_Double_points; i++) {
			Surf->unrank_point(v, Double_points[i]);
			ost << i << " : $P_{" << Double_points_index[i]
					<< "} = P_{" << Double_points[i] << "}=";
			int_vec_print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
		ost << "The double points on the surface are:\\\\" << endl;
		//ost << "\\begin{multicols}{2}" << endl;

		int *pt_idx;

		pt_idx = NEW_int(27 * 27);
		for (i = 0; i < 27 * 27; i++) {
			pt_idx[i] = -1;
		}
		for (p = 0; p < nb_pts; p++) {
			if (lines_on_point->Set_size[p] != 2) {
				continue;
				}
			a = lines_on_point->Sets[p][0];
			b = lines_on_point->Sets[p][1];
			if (a > b) {
				a = lines_on_point->Sets[p][1];
				b = lines_on_point->Sets[p][0];
			}
			pt_idx[a * 27 + b] = p;
		}
		ost << "\\begin{align*}" << endl;
		for (a = 0; a < 27; a++) {
			for (b = a + 1; b < 27; b++) {
				p = pt_idx[a * 27 + b];
				if (p == -1) {
					continue;
				}
				Surf->unrank_point(v, Pts[p]);
				ost << "P_{" << p
						<< "} = P_{" << Pts[p] << "}=";
				int_vec_print_fully(ost, v, 4);
				ost << " = ";
				ost << "\\ell_{" << a << "} \\cap ";
				ost << "\\ell_{" << b << "} ";
				ost << " = ";
				ost << Surf->Line_label_tex[a] << " \\cap ";
				ost << Surf->Line_label_tex[b] << "\\\\";
			}
		}
		ost << "\\end{align*}" << endl;

		FREE_int(pt_idx);
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}

	

	//ost << "\\clearpage" << endl;
	ost << "\\subsubsection*{Points on lines}" << endl;
	//pts_on_lines->print_table_tex(ost);
	ost << "\\noindent" << endl;
	for (i = 0; i < pts_on_lines->nb_sets; i++) {
		ost << "Line " << i << " = $" << Surf->Line_label_tex[i]
			<< "$ has " << pts_on_lines->Set_size[i]
			<< " points: $\\{ P_{i} \\mid i \\in ";
		L.lint_set_print_tex(ost, pts_on_lines->Sets[i],
				pts_on_lines->Set_size[i]);
		ost << "\\}$\\\\" << endl;
		}

	//ost << "\\clearpage" << endl;
	ost << "\\subsubsection*{Points on surface but on no line}" << endl;
	ost << "The surface has " << nb_pts_not_on_lines
			<< " points not on any line:\\\\" << endl;
	if (nb_pts_not_on_lines < 1000) {
		ost << "$$" << endl;
		L.lint_vec_print_as_matrix(ost,
				Pts_not_on_lines, nb_pts_not_on_lines, 10,
				TRUE /* f_tex */);
		//print_integer_matrix_with_standard_labels(ost, Pts3,
		//(nb_pts_not_on_lines + 9) / 10, 10, TRUE /* f_tex */);
		ost << "$$" << endl;
		//ost << "%%\\clearpage" << endl;
		ost << "The points on the surface but not "
				"on lines are:\\\\" << endl;
		ost << "\\begin{multicols}{2}" << endl;
		ost << "\\noindent" << endl;
		for (i = 0; i < nb_pts_not_on_lines; i++) {
			Surf->unrank_point(v, Pts_not_on_lines[i]);
			ost << i << " : $P_{" << Pts_not_on_lines[i] << "}=";
			int_vec_print_fully(ost, v, 4);
			ost << "$\\\\" << endl;
			}
		ost << "\\end{multicols}" << endl;
	}
	else {
		ost << "Too many to print.\\\\" << endl;
	}

#if 0
	ost << "\\clearpage" << endl;
	ost << "\\section*{Lines through points}" << endl;
	lines_on_point->print_table_tex(ost);
#endif
}

void surface_object::print_double_sixes(ostream &ost)
{
	int i, j, a;
	latex_interface L;
	
	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Double sixes}" << endl;
	ost << "The double sixes are:\\\\" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
		Surf->Double_six, 36, 12, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r||*{6}{c|}|*{6}{c|}}" << endl;
	ost << "\\hline" << endl;
	for (j = 0; j < 12; j++) {
		ost << " & " << j;
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < 36; i++) {
		ost << i;
		for (j = 0; j < 12; j++) {
			a = Surf->Double_six[i * 12 + j];
			ost << " & " << Surf->Line_label_tex[a];
			}
		ost << "\\\\" << endl;
		}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
	//ost << "\\clearpage" << endl;


}

void surface_object::print_half_double_sixes(ostream &ost)
{
	int h, i, j, a;
	latex_interface L;


	ost << "\\subsection*{Half Double sixes}" << endl;


	ost << "The half double sixes are:\\\\" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
		Surf->Double_six, 36, 6, TRUE /* f_tex */);
	ost << "$$" << endl;

	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
		Surf->Double_six + 36 * 6, 36, 6, TRUE /* f_tex */);
	ost << "$$" << endl;


	ost << "$$" << endl;
	ost << "\\begin{array}{|r||*{6}{c|}}" << endl;
	ost << "\\hline" << endl;
	for (j = 0; j < 6; j++) {
		ost << " & " << j;
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (h = 0; h < 18; h++) {
		for (i = 0; i < 2; i++) {
			ost << 2 * h + i;
			for (j = 0; j < 6; j++) {
				a = Surf->Double_six[h * 12 + i * 6 + j];
				ost << " & " << Surf->Line_label_tex[a];
				}
			ost << "\\\\" << endl;
		}
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;


	ost << "$$" << endl;
	ost << "\\begin{array}{|r||*{6}{c|}}" << endl;
	ost << "\\hline" << endl;
	for (j = 0; j < 6; j++) {
		ost << " & " << j;
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (h = 18; h < 36; h++) {
		for (i = 0; i < 2; i++) {
			ost << 2 * h + i;
			for (j = 0; j < 6; j++) {
				a = Surf->Double_six[h * 12 + i * 6 + j];
				ost << " & " << Surf->Line_label_tex[a];
				}
			ost << "\\\\" << endl;
		}
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

	//ost << "\\clearpage" << endl;

}

void surface_object::print_half_double_sixes_numerically(ostream &ost)
{
	latex_interface L;

	ost << "The half double sixes are:\\\\" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
		Surf->Half_double_sixes, 36, 6, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		Surf->Half_double_sixes + 36 * 6,
		36, 6, 36, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
}

void surface_object::print_trihedral_pairs(ostream &ost)
{
	latex_interface L;
	int i, j, a;

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Trihedral pairs}" << endl;
	ost << "The 120 trihedral pairs are:\\\\" << endl;
	ost << "{\\renewcommand{\\arraystretch}{1.3}" << endl;
	ost << "$$" << endl;

	int n = 6;
	int n_offset = 0;
	int m = 40;
	int m_offset = 0;
	int *p = Surf->Trihedral_to_Eckardt;

	ost << "\\begin{array}{|r|r|*{" << n << "}r|}" << endl;
	ost << "\\hline" << endl;
	ost << " & ";
	for (j = 0; j < n; j++) {
		ost << " & " << n_offset + j;
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << m_offset + i << " & S_{";
		ost << Surf->Trihedral_pair_labels[m_offset + i] << "}";
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			ost << " & \\pi_{" << Surf->Eckard_point_label_tex[a] << "}";
			}
		ost << "\\\\";
		ost << endl;
		}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;

	//L.print_integer_matrix_with_standard_labels(ost,
	//	Surf->Trihedral_to_Eckardt, 40, 6, TRUE /* f_tex */);
	ost << "$$" << endl;


	ost << "$$" << endl;

	m_offset = 40;
	p = Surf->Trihedral_to_Eckardt + 40 * 6;

	ost << "\\begin{array}{|r|r|*{" << n << "}r|}" << endl;
	ost << "\\hline" << endl;
	ost << " & ";
	for (j = 0; j < n; j++) {
		ost << " & " << n_offset + j;
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << m_offset + i << " & S_{";
		ost << Surf->Trihedral_pair_labels[m_offset + i] << "}";
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			ost << " & \\pi_{" << Surf->Eckard_point_label_tex[a] << "}";
			}
		ost << "\\\\";
		ost << endl;
		}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;


	//L.print_integer_matrix_with_standard_labels_and_offset(ost,
	//	Surf->Trihedral_to_Eckardt + 40 * 6, 40, 6, 40, 0,
	//	TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;

	m_offset = 80;
	p = Surf->Trihedral_to_Eckardt + 80 * 6;

	ost << "\\begin{array}{|r|r|*{" << n << "}r|}" << endl;
	ost << "\\hline" << endl;
	ost << " & ";
	for (j = 0; j < n; j++) {
		ost << " & " << n_offset + j;
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << m_offset + i << " & S_{";
		ost << Surf->Trihedral_pair_labels[m_offset + i] << "}";
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			ost << " & \\pi_{" << Surf->Eckard_point_label_tex[a] << "}";
			}
		ost << "\\\\";
		ost << endl;
		}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;


	//L.print_integer_matrix_with_standard_labels_and_offset(ost,
	//	Surf->Trihedral_to_Eckardt + 80 * 6, 40, 6, 80, 0,
	//	TRUE /* f_tex */);
	ost << "$$}" << endl;
}

void surface_object::print_trihedral_pairs_numerically(ostream &ost)
{
	latex_interface L;

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Trihedral pairs}" << endl;
	ost << "The planes in the trihedral pairs in Eckardt "
			"point labeling are:\\\\" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
		Surf->Trihedral_to_Eckardt, 40, 6, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
		Surf->Trihedral_to_Eckardt + 40 * 6, 40, 6, 40, 0,
		TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
		Surf->Trihedral_to_Eckardt + 80 * 6, 40, 6, 80, 0,
		TRUE /* f_tex */);
	ost << "$$" << endl;
}



#if 1
void surface_object::latex_table_of_trihedral_pairs_and_clebsch_system(
	ostream &ost, int *T, int nb_T)
{
	int t_idx, t;
		
	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Trihedral Pairs and the Clebsch System}" << endl;

	for (t = 0; t < nb_T; t++) {

		t_idx = T[t];
		

		int F_planes[12];
		int G_planes[12];
		int lambda;
		int equation[20];
		int *system;

		make_equation_in_trihedral_form(t_idx,
			F_planes, G_planes, lambda, equation,
			0 /* verbose_level */);

#if 0
		if (t_idx == 71) {
			int_vec_swap(F_planes, F_planes + 8, 4);
			}
#endif

		Surf->prepare_system_from_FG(F_planes, G_planes,
				lambda, system, 0 /*verbose_level*/);


		ost << "$" << t << " / " << nb_T << "$ ";
		ost << "$T_{" << t_idx << "} = T_{"
			<< Surf->Trihedral_pair_labels[t_idx] << "} = \\\\" << endl;
		latex_trihedral_pair(ost, t_idx);
		ost << "$\\\\" << endl;
		ost << "$";
		print_equation_in_trihedral_form_equation_only(ost,
				F_planes, G_planes, lambda);
		ost << "$\\\\" << endl;
		//ost << "$";
		Surf->print_system(ost, system);
		//ost << "$\\\\" << endl;
		FREE_int(system);


		}
}
#endif

void surface_object::latex_table_of_trihedral_pairs(ostream &ost,
		int *T, int nb_T)
{
	int h, i, j, t_idx;
	
	cout << "surface_object::latex_table_of_trihedral_pairs" << endl;
	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Trihedral Pairs}" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	ost << "\\noindent" << endl;
	for (h = 0; h < nb_T; h++) {
		ost << "$" << h << " / " << nb_T << "$ ";
		t_idx = T[h];
		ost << "$T_{" << t_idx << "} = T_{"
				<< Surf->Trihedral_pair_labels[t_idx]
				<< "} = \\\\" << endl;
		latex_trihedral_pair(ost, t_idx);
		ost << "$\\\\" << endl;
		ost << "$";
		make_and_print_equation_in_trihedral_form(ost, t_idx);
		ost << "$\\\\" << endl;
		}
	ost << "Dual point ranks: \\\\" << endl;
	for (i = 0; i < Surf->nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{"
			<< Surf->Trihedral_pair_labels[i]
			<< "}: \\quad " << endl;
		for (j = 0; j < 6; j++) {
			ost << Dual_point_ranks[i * 6 + j];
			if (j < 6 - 1) {
				ost << ", ";
				}
			}
		ost << "$\\\\" << endl;
		}


#if 0
	ost << "Planes by generator matrix: \\\\" << endl;;
	for (i = 0; i < Surf->nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{"
			<< Surf->Trihedral_pair_labels[i] << "}$" << endl;
		for (j = 0; j < 6; j++) {
			int d;
			
			d = All_Planes[i * 6 + j];
			ost << "Plane " << j << " has rank " << d << "\\\\" << endl;
			Surf->Gr3->unrank_int(d, 0 /* verbose_level */);
			ost << "$";
			ost << "\\left[";
			print_integer_matrix_tex(ost, Surf->Gr3->M, 3, 4);
			ost << "\\right]";
			ost << "$\\\\" << endl;
			}
		}
#endif
	//ost << "\\end{multicols}" << endl;
	cout << "surface_object::latex_table_of_trihedral_pairs done" << endl;
}

void surface_object::latex_trihedral_pair(ostream &ost, int t_idx)
{
	int i, j, e, t, a;
	
	//ost << "\\left[" << endl;
	ost << "\\begin{array}{c||ccc|cc}" << endl;
	ost << " & G_0 & G_1 & G_2 & \\mbox{plane} & "
			"\\mbox{dual rank} \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < 3; i++) {
		ost << "F_" << i;
		for (j = 0; j < 3; j++) {
			a = Surf->Trihedral_pairs[t_idx * 9 + i * 3 + j];
			ost << " & {" << Surf->Line_label_tex[a] << "=\\atop";
			ost << "\\left[" << endl;
			Surf->Gr->print_single_generator_matrix_tex(ost, Lines[a]);
			ost << "\\right]}" << endl;
			}
		e = Surf->Trihedral_to_Eckardt[t_idx * 6 + i];
		ost << " & {\\pi_{" << e << "} =\\atop";
		t = Eckardt_to_Tritangent_plane[e];
		a = Tritangent_planes[t];
		ost << "\\left[" << endl;
		Surf->Gr3->print_single_generator_matrix_tex(ost, a);
		ost << "\\right]}" << endl;
		ost << " & ";
		a = Dual_point_ranks[t_idx * 6 + i];
		ost << a << "\\\\" << endl;
		}
	ost << "\\hline" << endl;
	for (j = 0; j < 3; j++) {
		e = Surf->Trihedral_to_Eckardt[t_idx * 6 + 3 + j];
		ost << " & {\\pi_{" << e << "} =\\atop";
		t = Eckardt_to_Tritangent_plane[e];
		a = Tritangent_planes[t];
		ost << "\\left[" << endl;
		Surf->Gr3->print_single_generator_matrix_tex(ost, a);
		ost << "\\right]}" << endl;
		}
	ost << " & & \\\\" << endl;
	for (j = 0; j < 3; j++) {
		a = Dual_point_ranks[t_idx * 6 + 3 + j];
		ost << " & " << a;
		}
	ost << " & & \\\\" << endl;
	//Surf->latex_trihedral_pair(ost, Surf->Trihedral_pairs + h * 9);
	ost << "\\end{array}" << endl;
	//ost << "\\right]" << endl;
}

void surface_object::make_equation_in_trihedral_form(int t_idx, 
	int *F_planes, int *G_planes, int &lambda, int *equation,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, c, h;
	int row_col_Eckardt_points[6];
	//int plane_rk[6];
	int plane_idx[6];

	if (f_v) {
		cout << "surface_object::make_equation_in_trihedral_"
				"form t_idx=" << t_idx << endl;
		}
	
	if (f_v) {
		cout << "Trihedral pair T_{"
			<< Surf->Trihedral_pair_labels[t_idx] << "}"
			<< endl;
		}

	int_vec_copy(Surf->Trihedral_to_Eckardt + t_idx * 6,
			row_col_Eckardt_points, 6);
	for (i = 0; i < 6; i++) {
		plane_idx[i] = Eckardt_to_Tritangent_plane[
						row_col_Eckardt_points[i]];
		//plane_rk[i] = Tritangent_planes[plane_idx[i]];
		}
	for (i = 0; i < 3; i++) {
		c = Tritangent_plane_dual[plane_idx[i]];
		F->PG_element_unrank_modified(F_planes + i * 4, 1, 4, c);
		}
	for (i = 0; i < 3; i++) {
		c = Tritangent_plane_dual[plane_idx[3 + i]];
		F->PG_element_unrank_modified(G_planes + i * 4, 1, 4, c);
		}
	int evals[6];
	int pt_on_surface[4];
	int a, b, ma, bv, pt;
	int eqn_F[20];
	int eqn_G[20];
	int eqn_G2[20];

	for (h = 0; h < nb_pts; h++) {
		pt = Pts[h];
		F->PG_element_unrank_modified(pt_on_surface, 1, 4, pt);
		for (i = 0; i < 3; i++) {
			evals[i] = Surf->Poly1_4->evaluate_at_a_point(
					F_planes + i * 4, pt_on_surface);
			}
		for (i = 0; i < 3; i++) {
			evals[3 + i] = Surf->Poly1_4->evaluate_at_a_point(
					G_planes + i * 4, pt_on_surface);
			}
		a = F->mult3(evals[0], evals[1], evals[2]);
		b = F->mult3(evals[3], evals[4], evals[5]);
		if (b) {
			ma = F->negate(a);
			bv = F->inverse(b);
			lambda = F->mult(ma, bv);
			break;
			}
		}
	if (h == nb_pts) {
		cout << "surface_object::make_equation_in_trihedral_form could "
				"not determine lambda" << endl;
		exit(1);
		}

	Surf->multiply_linear_times_linear_times_linear_in_space(F_planes, 
		F_planes + 4, F_planes + 8, 
		eqn_F, FALSE /* verbose_level */);
	Surf->multiply_linear_times_linear_times_linear_in_space(G_planes, 
		G_planes + 4, G_planes + 8, 
		eqn_G, FALSE /* verbose_level */);

	int_vec_copy(eqn_G, eqn_G2, 20);
	F->scalar_multiply_vector_in_place(lambda, eqn_G2, 20);
	F->add_vector(eqn_F, eqn_G2, equation, 20);
	F->PG_element_normalize(equation, 1, 20);

	

	if (f_v) {
		cout << "surface_object::make_equation_in_trihedral_"
				"form done" << endl;
		}
	
}

void surface_object::print_equation_in_trihedral_form(ostream &ost, 
	int *F_planes, int *G_planes, int lambda)
{

	ost << "\\begin{align*}" << endl;
	ost << "0 & = F_0F_1F_2 + \\lambda G_0G_1G_2\\\\" << endl;
	ost << "& = " << endl;

	print_equation_in_trihedral_form_equation_only(ost, 
		F_planes, G_planes, lambda);
}

void surface_object::print_equation_in_trihedral_form_equation_only(
	ostream &ost, 
	int *F_planes, int *G_planes, int lambda)
{

	ost << "\\Big(";
	Surf->Poly1_4->print_equation(ost, F_planes);
	ost << "\\Big)";
	ost << "\\Big(";
	Surf->Poly1_4->print_equation(ost, F_planes + 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Surf->Poly1_4->print_equation(ost, F_planes + 8);
	ost << "\\Big)";
	ost << "+ " << lambda;
	ost << "\\Big(";
	Surf->Poly1_4->print_equation(ost, G_planes);
	ost << "\\Big)";
	ost << "\\Big(";
	Surf->Poly1_4->print_equation(ost, G_planes + 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Surf->Poly1_4->print_equation(ost, G_planes + 8);
	ost << "\\Big)";
}

void surface_object::make_and_print_equation_in_trihedral_form(
	ostream &ost, int t_idx)
{
	int F_planes[12];
	int G_planes[12];
	int lambda;
	int equation[20];
	//int *system;

	make_equation_in_trihedral_form(t_idx, F_planes, G_planes,
		lambda, equation, 0 /* verbose_level */);
	print_equation_in_trihedral_form_equation_only(ost,
		F_planes, G_planes, lambda);
	//FREE_int(system);
}

void surface_object::identify_double_six_from_trihedral_pair(
	int *Lines, int t_idx, int *nine_lines, int *double_sixes, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nine_line_idx[9];
	int i, idx;
	sorting Sorting;

	if (f_v) {
		cout << "surface_object::identify_double_six_from_"
				"trihedral_pair" << endl;
		}
	if (f_v) {
		cout << "surface_object::identify_double_six_from_"
				"trihedral_pair t_idx = " << t_idx << endl;
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
		cout << "surface_object::identify_double_six_from_"
				"trihedral_pair done" << endl;
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
		cout << "surface_object::identify_double_six_from_"
				"trihedral_pair_type_one" << endl;
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
		cout << "surface_object::identify_double_six_from_"
				"trihedral_pair_type_one done" << endl;
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
		cout << "surface_object::identify_double_six_from_"
				"trihedral_pair_type_two" << endl;
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
		cout << "surface_object::identify_double_six_from_"
				"trihedral_pair_type_two done" << endl;
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
	int f_v = FALSE;

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

void surface_object::identify_lines(long int *lines, int nb_lines,
	int *line_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, idx;
	sorting Sorting;

	if (f_v) {
		cout << "surface_object::identify_lines" << endl;
		}
	for (i = 0; i < nb_lines; i++) {
		if (!Sorting.lint_vec_search_linear(Lines, 27, lines[i], idx)) {
			cout << "surface_object::identify_lines could "
					"not find lines[" << i << "]=" << lines[i]
					<< " in Lines[]" << endl;
			exit(1);
			}
		line_idx[i] = idx;
		}
	if (f_v) {
		cout << "surface_object::identify_lines done" << endl;
		}
}

void surface_object::print_nine_lines_latex(ostream &ost, 
	long int *nine_lines, int *nine_lines_idx)
{
	latex_interface L;
	int i, j, idx;

	ost << "$$";
	L.print_lint_matrix_with_standard_labels(ost,
			nine_lines, 3, 3, TRUE /* f_tex*/);


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
			ost << Surf->Line_label_tex[idx];
			if (j < 3 - 1) {
				ost << " & ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
}

int surface_object::choose_tritangent_plane(
	int line_a, int line_b, int transversal_line, int verbose_level)
{
	int f_v = TRUE; // (verbose_level >= 1);
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
	if (FALSE) {
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
	int f_v = TRUE; // (verbose_level >= 1);
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
	if (FALSE) {
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

int surface_object::compute_transversal_line(
	int line_a, int line_b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "surface_object::compute_transversal_line" << endl;
		}
	for (i = 0; i < 27; i++) {
		if (i == line_a) {
			continue;
			}
		if (i == line_b) {
			continue;
			}
		if (Adj_line_intersection_graph[i * 27 + line_a] &&
				Adj_line_intersection_graph[i * 27 + line_b]) {
			break;
			}
		}
	if (i == 27) {
		cout << "surface_object::compute_transversal_line "
				"did not find transversal line" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface_object::compute_transversal_line "
				"done" << endl;
		}
	return i;
}

void surface_object::compute_transversal_lines(
	int line_a, int line_b, int *transversals5, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int nb_trans = 0;
	
	if (f_v) {
		cout << "surface_object::compute_transversal_lines" << endl;
		}
	for (i = 0; i < 27; i++) {
		if (i == line_a) {
			continue;
			}
		if (i == line_b) {
			continue;
			}
		if (Adj_line_intersection_graph[i * 27 + line_a] &&
				Adj_line_intersection_graph[i * 27 + line_b]) {
			transversals5[nb_trans++] = i;
			}
		}
	if (nb_trans != 5) {
		cout << "surface_object::compute_transversal_lines "
				"nb_trans != 5" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface_object::compute_transversal_lines "
				"done" << endl;
		}
}

void surface_object::clebsch_map_find_arc_and_lines(
	long int *Clebsch_map,
	long int *Arc, long int *Blown_up_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, pt, nb_blow_up_lines;
	
	if (f_v) {
		cout << "surface_object::clebsch_map_find_arc_and_lines" << endl;
		}


	if (f_v) {
		cout << "lines_on_point:" << endl;
		lines_on_point->print_table();
		}
	
	{
	classify C2;

	C2.init_lint(Clebsch_map, nb_pts, TRUE, 0);
	if (f_v) {
		cout << "surface_object::clebsch_map_find_arc_and_lines "
				"The fibers have the following sizes: ";
		C2.print_naked(TRUE);
		cout << endl;
		}

	int t2, f2, l2, sz;
	int t1, f1, l1;
	int fiber[2];
	int common_elt;
	int u; //, v, w;
	



	nb_blow_up_lines = 0;
	for (t2 = 0; t2 < C2.second_nb_types; t2++) {
		f2 = C2.second_type_first[t2];
		l2 = C2.second_type_len[t2];
		sz = C2.second_data_sorted[f2];
		if (f_v) {
			cout << "surface_object::clebsch_map_find_arc_and_lines "
					"fibers of size " << sz << ":" << endl;
			}
		if (sz == 1) {
			continue;
			}

		if (f_v) {
			for (i = 0; i < l2; i++) {
				t1 = C2.second_sorting_perm_inv[f2 + i];
				f1 = C2.type_first[t1];
				l1 = C2.type_len[t1];
				pt = C2.data_sorted[f1];
				cout << "arc point " << pt << " belongs to the " << l1
						<< " surface points in the list of Pts "
						"(point indices): ";
				for (j = 0; j < l1; j++) {
					u = C2.sorting_perm_inv[f1 + j];
					cout << u;
					//cout << Pts[u];
					if (j < l1 - 1) {
						cout << ", ";
						}
					}
				cout << endl;
				}
			}


		for (i = 0; i < l2; i++) {
			t1 = C2.second_sorting_perm_inv[f2 + i];
			f1 = C2.type_first[t1];
			l1 = C2.type_len[t1];
			pt = C2.data_sorted[f1];

			if (pt == -1) {
				continue;
				}
			fiber[0] = C2.sorting_perm_inv[f1 + 0];
			fiber[1] = C2.sorting_perm_inv[f1 + 1];

			if (f_v) {
				cout << "lines through point fiber[0]="
						<< fiber[0] << " : ";
				Surf->print_set_of_lines_tex(cout,
						lines_on_point->Sets[fiber[0]],
						lines_on_point->Set_size[fiber[0]]);
				cout << endl;
				cout << "lines through point fiber[1]="
						<< fiber[1] << " : ";
				Surf->print_set_of_lines_tex(cout,
						lines_on_point->Sets[fiber[1]],
						lines_on_point->Set_size[fiber[1]]);
				cout << endl;
				}
			
			// find the unique line which passes through
			// the surface points fiber[0] and fiber[1]:
			if (!lines_on_point->find_common_element_in_two_sets(
					fiber[0], fiber[1], common_elt)) {
				cout << "The fiber does not seem to come "
						"from a line, i=" << i << endl;
			

#if 1
				cout << "The fiber does not seem to come "
						"from a line" << endl;
				cout << "i=" << i << " / " << l2 << endl;
				cout << "pt=" << pt << endl;
				cout << "fiber[0]=" << fiber[0] << endl;
				cout << "fiber[1]=" << fiber[1] << endl;
				cout << "lines through point fiber[0]=" << fiber[0] << " : ";
				Surf->print_set_of_lines_tex(cout,
						lines_on_point->Sets[fiber[0]],
						lines_on_point->Set_size[fiber[0]]);
				cout << endl;
				cout << "lines through point fiber[1]=" << fiber[1] << " : ";
				Surf->print_set_of_lines_tex(cout,
						lines_on_point->Sets[fiber[1]],
						lines_on_point->Set_size[fiber[1]]);
				cout << endl;
				//exit(1);
#endif
				}
			else {
				if (nb_blow_up_lines == 6) {
					cout << "too many long fibers" << endl;
					exit(1);
					}
				cout << "i=" << i << " fiber[0]=" << fiber[0]
					<< " fiber[1]=" << fiber[1]
					<< " common_elt=" << common_elt << endl;
				Arc[nb_blow_up_lines] = pt;
				Blown_up_lines[nb_blow_up_lines] = common_elt;
				nb_blow_up_lines++;
				}

#if 0
			w = 0;
			for (u = 0; u < l2; u++) {
				fiber[0] = C2.sorting_perm_inv[f1 + u];
				for (v = u + 1; v < l2; v++, w++) {
					fiber[1] = C2.sorting_perm_inv[f1 + v];
					Fiber_recognize[w] = -1;
					if (!lines_on_point->find_common_element_in_two_sets(
							fiber[0], fiber[1], common_elt)) {
#if 0
						cout << "The fiber does not seem to "
								"come from a line" << endl;
						cout << "i=" << i << " / " << l2 << endl;
						cout << "pt=" << pt << endl;
						cout << "fiber[0]=" << fiber[0] << endl;
						cout << "fiber[1]=" << fiber[1] << endl;
						cout << "lines_on_point:" << endl;
						lines_on_point->print_table();
						exit(1);
#endif
						}
					else {
						Fiber_recognize[w] = common_elt;
						}
					}
				}
			{
				classify C_fiber;

				C_fiber.init(Fiber_recognize, w, FALSE, 0);
				cout << "The fiber type is : ";
				C_fiber.print_naked(TRUE);
				cout << endl;
			}
			Blown_up_lines[i] = -1;
#endif

			} // next i
		}

	if (nb_blow_up_lines != 6) {
		cout << "nb_blow_up_lines != 6" << endl;
		cout << "nb_blow_up_lines = " << nb_blow_up_lines << endl;
		exit(1);
		}
	} // end of classify C2
	if (f_v) {
		cout << "surface_object::clebsch_map_find_arc_and_lines "
				"done" << endl;
		}
}

void surface_object::clebsch_map_print_fibers(long int *Clebsch_map)
{
	int i, j, u, pt;
	
	cout << "surface_object::clebsch_map_print_fibers" << endl;
	{
	classify C2;

	C2.init_lint(Clebsch_map, nb_pts, TRUE, 0);
	cout << "surface_object::clebsch_map_print_fibers The fibers "
			"have the following sizes: ";
	C2.print_naked(TRUE);
	cout << endl;

	int t2, f2, l2, sz;
	int t1, f1, l1;
	
	for (t2 = 0; t2 < C2.second_nb_types; t2++) {
		f2 = C2.second_type_first[t2];
		l2 = C2.second_type_len[t2];
		sz = C2.second_data_sorted[f2];
		cout << "surface_object::clebsch_map_print_fibers fibers "
				"of size " << sz << ":" << endl;
		if (sz == 1) {
			continue;
			}
		for (i = 0; i < l2; i++) {
			t1 = C2.second_sorting_perm_inv[f2 + i];
			f1 = C2.type_first[t1];
			l1 = C2.type_len[t1];
			pt = C2.data_sorted[f1];
			cout << "arc point " << pt << " belongs to the " << l1
				<< " surface points in the list of Pts "
				"(local numbering): ";
			for (j = 0; j < l1; j++) {
				u = C2.sorting_perm_inv[f1 + j];
				cout << u;
				//cout << Pts[u];
				if (j < l1 - 1) {
					cout << ", ";
					}
				}
			cout << endl;
			}
		}
	cout << endl;
	}
}

#if 0
void surface_object::compute_clebsch_maps(int verbose_level)
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

void surface_object::compute_clebsch_map(int line_a, int line_b, 
	int transversal_line, 
	long int &tritangent_plane_rk,
	long int *Clebsch_map, int *Clebsch_coeff,
	int verbose_level)
// Clebsch_map[nb_pts]
// Clebsch_coeff[nb_pts * 4]
{
	int f_v = (verbose_level >= 1);
	int line_idx[2];
	long int plane_rk_global;
	
	if (f_v) {
		cout << "surface_object::compute_clebsch_map" << endl;
		}

	if (Adj_line_intersection_graph[line_a * 27 + line_b] == 1) {
		cout << "surface_object::compute_clebsch_map the lines "
				"are adjacent" << endl;
		exit(1);
		}

	line_idx[0] = line_a;
	line_idx[1] = line_b;

	if (f_v) {
		cout << "#######################" << endl;
		cout << "clebsch map for lines " << line_a << ", " << line_b
				<< " with transversal " << transversal_line << ":" << endl;
		}

	//transversal_line = compute_transversal_line(line_a, line_b,
	//0 /* verbose_level */);
	
	tritangent_plane_rk = choose_tritangent_plane(line_a, line_b,
			transversal_line, 0 /* verbose_level */);

	plane_rk_global = Tritangent_planes[
			Eckardt_to_Tritangent_plane[tritangent_plane_rk]];

	if (f_v) {
		cout << "transversal\\_line = " << transversal_line
				<< "\\\\" << endl;
		cout << "tritangent\\_plane\\_rank = " << tritangent_plane_rk
				<< " = " << plane_rk_global << "\\\\" << endl;
		}


	if (!Surf->clebsch_map(Lines, Pts, nb_pts,
			line_idx, plane_rk_global,
			Clebsch_map, Clebsch_coeff,
			0 /*verbose_level*/)) {
		cout << "surface_object::compute_clebsch_map The plane "
				"contains one of the lines, this should "
				"not happen" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "surface_object::compute_clebsch_map done" << endl;
		}
}

void surface_object::clebsch_map_latex(ostream &ost, 
	long int *Clebsch_map, int *Clebsch_coeff)
{
	long int i, j, a;
	int v[4];
	int w[3];

	ost << "$$";
	ost << "\\begin{array}{|c|c|c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & P_i & \\mbox{lines} & \\Phi(P_i) & \\Phi(P_i)\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_pts; i++) {
		ost << i;
		ost << " & ";
		a = Pts[i];
		Surf->unrank_point(v, a);
		int_vec_print(ost, v, 4);
		ost << " & ";

		for (j = 0; j < lines_on_point->Set_size[i]; j++) {
			a = lines_on_point->Sets[i][j];
			ost << Surf->Line_label_tex[a];
			if (j < lines_on_point->Set_size[i] - 1) {
				ost << ", ";
				}
			}
		ost << " & ";


		if (Clebsch_map[i] >= 0) {
			int_vec_print(ost, Clebsch_coeff + i * 4, 4);
			}
		else {
			ost << "\\mbox{undef}";
			}
		ost << " & ";
		if (Clebsch_map[i] >= 0) {
			Surf->P2->unrank_point(w, Clebsch_map[i]);
			int_vec_print(ost, w, 3);
			}
		else {
			ost << "\\mbox{undef}";
			}
		ost << "\\\\" << endl;
		if (((i + 1) % 30) == 0) {
			ost << "\\hline" << endl;
			ost << "\\end{array}" << endl;
			ost << "$$" << endl;
			ost << "$$";
			ost << "\\begin{array}{|c|c|c|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "i & P_i & \\mbox{lines} & \\Phi(P_i) & "
					"\\Phi(P_i)\\\\" << endl;
			ost << "\\hline" << endl;
			}
		}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
}


void surface_object::print_Steiner_and_Eckardt(ostream &ost)
{
#if 0
	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Eckardt Points}" << endl;
	latex_table_of_Eckardt_points(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Tritangent Planes}" << endl;
	latex_table_of_tritangent_planes(ost);
#endif

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Steiner Trihedral Pairs}" << endl;
	latex_table_of_trihedral_pairs(ost);

}

void surface_object::latex_table_of_trihedral_pairs(ostream &ost)
{
	int i;
	
	cout << "surface_object::latex_table_of_trihedral_pairs" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < Surf->nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{"
				<< Surf->Trihedral_pair_labels[i]
				<< "} = $\\\\" << endl;
		ost << "$" << endl;
		//ost << "\\left[" << endl;
		//ost << "\\begin{array}" << endl;
		latex_trihedral_pair(ost, 
			Surf->Trihedral_pairs + i * 9, 
			Surf->Trihedral_to_Eckardt + i * 6);
		//ost << "\\end{array}" << endl;
		//ost << "\\right]" << endl;
		ost << "$\\\\" << endl;
#if 0
		ost << "planes: $";
		int_vec_print(ost, Trihedral_to_Eckardt + i * 6, 6);
		ost << "$\\\\" << endl;
#endif
		}
	//ost << "\\end{multicols}" << endl;

	//print_trihedral_pairs(ost);
	
	cout << "surface_object::latex_table_of_trihedral_pairs done" << endl;
}

void surface_object::latex_trihedral_pair(ostream &ost,
		int *T, int *TE)
{
	int i, j, t, plane_rk;
	int Mtx[16];
	
	ost << "\\begin{array}{*{" << 3 << "}{c}|c}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Surf->print_line(ost, T[i * 3 + j]);
			ost << " & ";
			}
		ost << "\\pi_{";
		Surf->Eckardt_points[TE[i]].latex_index_only(ost);
		ost << "}=" << endl;
		t = Eckardt_to_Tritangent_plane[TE[i]];
		plane_rk = Tritangent_planes[t];
		Surf->Gr3->unrank_lint_here_and_compute_perp(Mtx, plane_rk,
			0 /*verbose_level */);
		F->PG_element_normalize(Mtx + 12, 1, 4);
		ost << "V\\big(";
		Surf->Poly1_4->print_equation(ost, Mtx + 12);
		ost << "\\big)=" << plane_rk;
		ost << "\\\\" << endl;
		}
	ost << "\\hline" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\pi_{";
		Surf->Eckardt_points[TE[3 + j]].latex_index_only(ost);
		ost << "} & ";
		}
	ost << "\\\\" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\multicolumn{4}{l}{" << endl;
		t = Eckardt_to_Tritangent_plane[TE[3 + j]];
		plane_rk = Tritangent_planes[t];
		ost << "\\pi_{";
		Surf->Eckardt_points[TE[3 + j]].latex_index_only(ost);
		ost << "}=" << endl;
		ost << "V\\big(" << endl;
		Surf->Gr3->unrank_lint_here_and_compute_perp(Mtx, plane_rk,
			0 /*verbose_level */);
		F->PG_element_normalize(Mtx + 12, 1, 4);
		Surf->Poly1_4->print_equation(ost, Mtx + 12);
		ost << "\\big)=" << plane_rk << "}\\\\" << endl;
		}
	ost << "\\\\" << endl;
	ost << "\\end{array}" << endl;
}

}
}


