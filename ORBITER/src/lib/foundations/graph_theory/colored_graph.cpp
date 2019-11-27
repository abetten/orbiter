// colored_graph.cpp
//
// Anton Betten
//
// started:  October 28, 2012




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


colored_graph::colored_graph()
{
	fname_base[0] = 0;
	nb_points = nb_colors = nb_colors_per_vertex = 0;
	bitvector_length = 0;
	L = 0;

	points = NULL;
	point_color = NULL;

	user_data_size = 0;
	user_data = NULL;

	f_ownership_of_bitvec = FALSE;
	bitvector_adjacency = NULL;
	f_has_list_of_edges = FALSE;
	nb_edges = 0;
	list_of_edges = NULL;
	null();
}

colored_graph::~colored_graph()
{
	freeself();
}

void colored_graph::null()
{
}

void colored_graph::freeself()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::freeself" << endl;
		}
	if (user_data) {
		if (f_v) {
			cout << "colored_graph::freeself user_data" << endl;
			}
		FREE_lint(user_data);
		}
	if (points) {
		if (f_v) {
			cout << "colored_graph::freeself points" << endl;
			}
		FREE_lint(points);
		}
	if (point_color) {
		if (f_v) {
			cout << "colored_graph::freeself point_color" << endl;
			}
		FREE_int(point_color);
		}
	if (f_ownership_of_bitvec) {
		if (bitvector_adjacency) {
			if (f_v) {
				cout << "colored_graph::freeself "
						"bitvector_adjacency" << endl;
				}
			FREE_uchar(bitvector_adjacency);
			}
		}
	if (list_of_edges) {
		if (f_v) {
			cout << "colored_graph::freeself list_of_edges" << endl;
			}
		FREE_int(list_of_edges);
		}
	null();
	if (f_v) {
		cout << "colored_graph::freeself" << endl;
		}
}

void colored_graph::compute_edges(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, nb, a;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "colored_graph::compute_edges" << endl;
		}
	if (f_has_list_of_edges) {
		cout << "colored_graph::compute_edges "
				"f_has_list_of_edges" << endl;
		exit(1);
		}
	nb = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			if (is_adjacent(i, j)) {
				nb++;
				}
			}
		}
	list_of_edges = NEW_int(nb);
	nb_edges = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			if (is_adjacent(i, j)) {
				a = Combi.ij2k_lint(i, j, nb_points);
				list_of_edges[nb_edges++] = a;
				}
			}
		}
	if (nb_edges != nb) {
		cout << "colored_graph::compute_edges "
				"nb_edges != nb" << endl;
		exit(1);
		}

	f_has_list_of_edges = TRUE;
	if (f_v) {
		cout << "colored_graph::compute_edges "
				"done" << endl;
		}
}


int colored_graph::is_adjacent(int i, int j)
{
	combinatorics_domain Combi;

	if (i == j) {
		return FALSE;
		}
	if (i > j) {
		return is_adjacent(j, i);
		}
	long int k;
	
	k = Combi.ij2k_lint(i, j, nb_points);
	return bitvector_s_i(bitvector_adjacency, k);
}

void colored_graph::set_adjacency(int i, int j, int a)
{
	combinatorics_domain Combi;
	long int k;

	k = Combi.ij2k_lint(i, j, nb_points);
	bitvector_m_ii(bitvector_adjacency, k, a);
}

void colored_graph::partition_by_color_classes(
	int *&partition, int *&partition_first, 
	int &partition_length, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l, c;

	if (f_v) {
		cout << "colored_graph::partition_by_color_classes" << endl;
	}
	if (nb_colors_per_vertex != 1) {
		cout << "colored_graph::partition_by_color_classes nb_colors_per_vertex != 1" << endl;
		exit(1);
	}
	partition = NEW_int(nb_colors);
	partition_first = NEW_int(nb_colors + 1);
	partition_length = nb_colors;
	partition_first[0] = 0;
	i = 0;
	for (c = 0; c < nb_colors; c++) {
		l = 0;
		while (i < nb_points) {
			if (point_color[i] != c) {
				break;
			}
			l++;
			i++;
		}
		partition[c] = l;
		partition_first[c + 1] = i;
	}
	if (f_v) {
		cout << "colored_graph::partition_by_color_classes done" << endl;
	}
}

colored_graph *colored_graph::sort_by_color_classes(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::sort_by_color_classes" << endl;
	}
	if (nb_colors_per_vertex != 1) {
		cout << "colored_graph::sort_by_color_classes nb_colors_per_vertex != 1" << endl;
		exit(1);
	}

	classify C;

	C.init(point_color, nb_points, FALSE, 0);
	if (f_v) {
		cout << "point color distribution: ";
		C.print_naked(TRUE);
		cout << endl;
	}

	int *A;
	long int *Pts;
	int *Color;
	int i, j, I, J, f1, l1, f2, l2, ii, jj, idx1, idx2, aij;

	A = NEW_int((long int) nb_points * (long int) nb_points);
	Pts = NEW_lint(nb_points);
	Color = NEW_int(nb_points);
	for (I = 0; I < C.nb_types; I++) {
		f1 = C.type_first[I];
		l1 = C.type_len[I];
		for (i = 0; i < l1; i++) {
			ii = f1 + i;
			idx1 = C.sorting_perm_inv[ii];
			Color[ii] = point_color[idx1];
			Pts[ii] = points[idx1];
		}
	}
	
	for (I = 0; I < C.nb_types; I++) {
		f1 = C.type_first[I];
		l1 = C.type_len[I];
		for (J = 0; J < C.nb_types; J++) {
			f2 = C.type_first[J];
			l2 = C.type_len[J];
			for (i = 0; i < l1; i++) {
				ii = f1 + i;
				idx1 = C.sorting_perm_inv[ii];
				for (j = 0; j < l2; j++) {
					jj = f2 + j;
					idx2 = C.sorting_perm_inv[jj];
					aij = is_adjacent(idx1, idx2);
					A[ii * nb_points + jj] = aij;
				}
			}
		}
	}

	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency(nb_points, nb_colors, 1 /* nb_colors_per_vertex */,
		Color, A, 0 /* verbose_level */);
	CG->init_user_data(user_data, user_data_size,
			0 /* verbose_level */);
	lint_vec_copy(Pts, CG->points, nb_points);
	FREE_int(A);	
	FREE_int(Color);	
	FREE_lint(Pts);

	cout << "-partition \"";
	for (I = 0; I < C.nb_types; I++) {
		l1 = C.type_len[I];
		cout << l1;
		if (I < C.nb_types - 1) {
			cout << ", ";
		}
	}
	cout << "\"" << endl;
	
	if (f_v) {
		cout << "colored_graph::sort_by_color_classes done" << endl;
	}
	return CG;
}

colored_graph *colored_graph::subgraph_by_color_classes(
		int c, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::subgraph_by_color_classes c=" << c << endl;
	}
	if (nb_colors_per_vertex != 1) {
		cout << "colored_graph::subgraph_by_color_classes nb_colors_per_vertex != 1" << endl;
		exit(1);
	}

	classify C;

	C.init(point_color, nb_points, FALSE, 0);
	if (f_v) {
		cout << "point color distribution: ";
		C.print_naked(TRUE);
		cout << endl;
	}

	int *A;
	long int *Pts;
	int *Color;
	int i, j, I, f, l, ii, jj, idx1, idx2;

	I = C.determine_class_by_value(c);
	f = C.type_first[I];
	l = C.type_len[I];

	A = NEW_int(l * l);
	Pts = NEW_lint(l);
	Color = NEW_int(l);

	int_vec_zero(A, l * l);

	for (i = 0; i < l; i++) {
		ii = f + i;
		idx1 = C.sorting_perm_inv[ii];
		Color[i] = point_color[idx1];
		Pts[i] = points[idx1];
	}

	for (i = 0; i < l; i++) {
		ii = f + i;
		idx1 = C.sorting_perm_inv[ii];
		for (j = i + 1; j < l; j++) {
			jj = f + j;
			idx2 = C.sorting_perm_inv[jj];
			if (is_adjacent(idx1, idx2)) {
				A[i * l + j] = 1;
				A[j * l + i] = 1;
			}
		}
	}

	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency(l /* nb_points */, 1 /*nb_colors */, 1 /* nb_colors_per_vertex*/,
		Color, A, 0 /* verbose_level */);
	CG->init_user_data(user_data, user_data_size,
			0 /* verbose_level */);
	lint_vec_copy(Pts, CG->points, l);
	FREE_int(A);
	FREE_int(Color);
	FREE_lint(Pts);

	if (f_v) {
		cout << "colored_graph::subgraph_by_color_classes done" << endl;
	}
	return CG;
}

colored_graph *colored_graph::subgraph_by_color_classes_with_condition(
		int *seed_pts, int nb_seed_pts,
		int c, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::subgraph_by_color_classes_with_condition "
				"c=" << c << " nb_seed_pts=" << nb_seed_pts << endl;
	}
	if (nb_colors_per_vertex != 1) {
		cout << "colored_graph::subgraph_by_color_classes_with_condition "
				"nb_colors_per_vertex != 1" << endl;
		exit(1);
	}

	classify C;

	C.init(point_color, nb_points, FALSE, 0);
	if (f_v) {
		cout << "point color distribution: ";
		C.print_naked(TRUE);
		cout << endl;
	}

	int *A;
	long int *Pts;
	int nb_pts;
	int *Color;
	int i, j, I, f, l, ii, jj, idx1, idx2;

	I = C.determine_class_by_value(c);
	f = C.type_first[I];
	l = C.type_len[I];

	Pts = NEW_lint(l);
	Color = NEW_int(l);


	nb_pts = 0;
	for (i = 0; i < l; i++) {
		ii = f + i;
		idx1 = C.sorting_perm_inv[ii];
		if (point_color[idx1] != c) {
			cout << "colored_graph::subgraph_by_color_classes_with_condition "
					"point_color[idx1] != c" << endl;
			exit(1);
		}
		for (j = 0; j < nb_seed_pts; j++) {
			if (!is_adjacent(idx1, seed_pts[j])) {
				break;
			}
		}
		if (j < nb_seed_pts) {
			continue;
		}
		Color[nb_pts] = c;
		Pts[nb_pts] = points[idx1];
		nb_pts++;
		}

	A = NEW_int(nb_pts * nb_pts);
	int_vec_zero(A, nb_pts * nb_pts);

	for (i = 0; i < nb_pts; i++) {
		ii = f + i;
		idx1 = C.sorting_perm_inv[ii];
		for (j = i + 1; j < nb_pts; j++) {
			jj = f + j;
			idx2 = C.sorting_perm_inv[jj];
			if (is_adjacent(idx1, idx2)) {
				A[i * nb_pts + j] = 1;
				A[j * nb_pts + i] = 1;
			}
		}
	}
	if (f_v) {
		cout << "colored_graph::subgraph_by_color_classes_with_condition "
				"subgraph has " << nb_pts << " vertices" << endl;
		}

	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency(nb_pts /* nb_points */, nb_colors, 1 /* nb_colors_per_vertex */,
		Color, A, 0 /* verbose_level */);
	CG->init_user_data(user_data, user_data_size,
			0 /* verbose_level */);
	lint_vec_copy(Pts, CG->points, nb_pts);
	FREE_int(A);
	FREE_int(Color);
	FREE_lint(Pts);

	if (f_v) {
		cout << "colored_graph::subgraph_by_color_classes_with_condition "
				"done" << endl;
		}
	return CG;
}

void colored_graph::print()
{
	int i;
	
	cout << "colored graph with " << nb_points << " points and "
			<< nb_colors << " colors" << endl;

#if 0
	cout << "i : point_label[i] : point_color[i]" << endl;
	for (i = 0; i < nb_points; i++) {
		cout << i << " : " << points[i] << " : " << point_color[i] << endl;
		}
	cout << endl;
#endif
	
	classify C;

	C.init(point_color, nb_points, TRUE, 0);

	int I, f1, l1, ii, idx1;

	cout << "color : size  of color class: color class" << endl;
	for (I = 0; I < C.nb_types; I++) {
		f1 = C.type_first[I];
		l1 = C.type_len[I];
		cout << I << " : " << l1 << " : ";
		for (i = 0; i < l1; i++) {
			ii = f1 + i;
			idx1 = C.sorting_perm_inv[ii];
			cout << idx1;
			if (i < l1 - 1) {
				cout << ", ";
				}
			}
		cout << endl;
		}
	cout << endl;
	

	cout << "point colors: ";
	C.print_first(TRUE);
	cout << endl;

	cout << "color class sizes: ";
	C.print_second(TRUE);
	cout << endl;

#if 0
	int *A;
	int j, J, f2, l2, jj, idx2, aij;
	cout << "Adjacency (blocked off by color classes):" << endl;
	for (i = 0; i < nb_points; i++) {
		for (j = 0; j < nb_points; j++) {
			aij = is_adjacent(i, j);
			cout << aij;
			}
		cout << endl;
		}


	A = NEW_int(nb_points * nb_points);
	for (I = 0; I < C.nb_types; I++) {
		f1 = C.type_first[I];
		l1 = C.type_len[I];
		for (J = 0; J < C.nb_types; J++) {
			f2 = C.type_first[J];
			l2 = C.type_len[J];
			cout << "block (" << I << "," << J << ")" << endl;
			for (i = 0; i < l1; i++) {
				ii = f1 + i;
				idx1 = C.sorting_perm_inv[ii];
				for (j = 0; j < l2; j++) {
					jj = f2 + j;
					idx2 = C.sorting_perm_inv[jj];
					aij = is_adjacent(idx1, idx2);
					cout << aij;
					}
				cout << endl;
				}
			cout << endl;
			}
		}
	FREE_int(A);
#endif
}

void colored_graph::print_points_and_colors()
{
	int i;
	
	cout << "colored graph with " << nb_points << " points and "
			<< nb_colors << " colors" << endl;

	cout << "i : points[i] : point_color[i]" << endl;
	for (i = 0; i < nb_points; i++) {
		cout << i << " : " << points[i] << " : " << point_color[i] << endl;
		}
}

void colored_graph::print_adjacency_list()
{
	int i, j;
	int f_first = TRUE;
	
	cout << "Adjacency list:" << endl;
	for (i = 0; i < nb_points; i++) {
		cout << i << " : ";
		f_first = TRUE;
		for (j = 0; j < nb_points; j++) {
			if (i == j) {
				continue;
			}
			if (is_adjacent(i, j)) {
				if (f_first) {
				}
				else {
					cout << ", ";
				}
				cout << j;
				f_first = FALSE;
			}
		}
		cout << endl;
	}
	cout << "Adjacency list using point labels:" << endl;
	for (i = 0; i < nb_points; i++) {
		cout << points[i] << " : ";
		f_first = TRUE;
		for (j = 0; j < nb_points; j++) {
			if (i == j) {
				continue;
			}
			if (is_adjacent(i, j)) {
				if (f_first) {
				}
				else {
					cout << ", ";
				}
				cout << points[j];
				f_first = FALSE;
			}
		}
		cout << endl;
	}
	
}

void colored_graph::init_with_point_labels(int nb_points, int nb_colors, int nb_colors_per_vertex,
	int *colors, uchar *bitvec, int f_ownership_of_bitvec, 
	long int *point_labels,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "colored_graph::init_with_point_labels" << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "nb_colors=" << nb_colors << endl;
		cout << "nb_colors_per_vertex=" << nb_colors_per_vertex << endl;
	}
	init(nb_points, nb_colors, nb_colors_per_vertex,
		colors, bitvec, f_ownership_of_bitvec, 
		verbose_level);
	lint_vec_copy(point_labels, points, nb_points);
	if (f_v) {
		cout << "colored_graph::init_with_point_labels done" << endl;
	}
}

void colored_graph::init(int nb_points, int nb_colors, int nb_colors_per_vertex,
	int *colors, uchar *bitvec, int f_ownership_of_bitvec, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "colored_graph::init" << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "nb_colors=" << nb_colors << endl;
		cout << "nb_colors_per_vertex=" << nb_colors_per_vertex << endl;
		}
	colored_graph::nb_points = nb_points;
	colored_graph::nb_colors = nb_colors;
	colored_graph::nb_colors_per_vertex = nb_colors_per_vertex;
	
	L = ((long int) nb_points * (long int) (nb_points - 1)) >> 1;

	bitvector_length = (L + 7) >> 3;

	user_data_size = 0;
	
	points = NEW_lint(nb_points);
	for (i = 0; i < nb_points; i++) {
		points[i] = i;
		}
	point_color = NEW_int(nb_points * nb_colors_per_vertex);

	if (colors) {
		int_vec_copy(colors, point_color, nb_points * nb_colors_per_vertex);
		}
	else {
		int_vec_zero(point_color, nb_points * nb_colors_per_vertex);
		}
	
	colored_graph::f_ownership_of_bitvec = f_ownership_of_bitvec;
	bitvector_adjacency = bitvec;

	if (f_v) {
		cout << "colored_graph::init" << endl;
		}

}

void colored_graph::init_no_colors(int nb_points,
	uchar *bitvec, int f_ownership_of_bitvec,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *vertex_colors;

	if (f_v) {
		cout << "colored_graph::init_no_colors" << endl;
		cout << "nb_points=" << nb_points << endl;
		}
	vertex_colors = NEW_int(nb_points);
	int_vec_zero(vertex_colors, nb_points);

	init(nb_points, 1 /* nb_colors */, 1 /* nb_colors_per_vertex */,
		vertex_colors, bitvec, f_ownership_of_bitvec, verbose_level);

	FREE_int(vertex_colors);
	if (f_v) {
		cout << "colored_graph::init_no_colors done" << endl;
		}
}

void colored_graph::init_adjacency(int nb_points, int nb_colors, int nb_colors_per_vertex,
	int *colors, int *Adj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k;
	long int bitvector_length;
	uchar *bitvec;
	combinatorics_domain Combi;


	if (f_v) {
		cout << "colored_graph::init_adjacency" << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "nb_colors=" << nb_colors << endl;
		}
	L = ((long int) nb_points * (long int) (nb_points - 1)) >> 1;

	bitvector_length = (L + 7) >> 3;
	bitvec = NEW_uchar(bitvector_length);
	for (i = 0; i < bitvector_length; i++) {
		bitvec[i] = 0;
		}
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			if (Adj[i * nb_points + j]) {
				k = Combi.ij2k_lint(i, j, nb_points);
				bitvector_m_ii(bitvec, k, 1);
				}
			}
		}
	init(nb_points, nb_colors, nb_colors_per_vertex,
		colors, bitvec, TRUE /* f_ownership_of_bitvec */, 
		verbose_level);

	// do not free bitvec here

	if (f_v) {
		cout << "colored_graph::init_adjacency" << endl;
		}

}

void colored_graph::init_adjacency_upper_triangle(
	int nb_points, int nb_colors, int nb_colors_per_vertex,
	int *colors, int *Adj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k;
	long int bitvector_length;
	uchar *bitvec;
	combinatorics_domain Combi;


	if (f_v) {
		cout << "colored_graph::init_adjacency_upper_triangle" << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "nb_colors=" << nb_colors << endl;
		cout << "nb_colors_per_vertex=" << nb_colors_per_vertex << endl;
		}
	L = ((long int) nb_points * (long int) (nb_points - 1)) >> 1;

	bitvector_length = (L + 7) >> 3;
	bitvec = NEW_uchar(bitvector_length);
	for (i = 0; i < bitvector_length; i++) {
		bitvec[i] = 0;
	}
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			k = Combi.ij2k_lint(i, j, nb_points);
			if (Adj[k]) {
				bitvector_m_ii(bitvec, k, 1);
			}
		}
	}
	init(nb_points, nb_colors, nb_colors_per_vertex,
		colors, bitvec, TRUE /* f_ownership_of_bitvec */, 
		verbose_level);

	// do not free bitvec here

	if (f_v) {
		cout << "colored_graph::init_adjacency_upper_triangle done" << endl;
	}

}

void colored_graph::init_adjacency_no_colors(int nb_points,
	int *Adj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *vertex_colors;

	if (f_v) {
		cout << "colored_graph::init_adjacency_no_colors" << endl;
		cout << "nb_points=" << nb_points << endl;
	}
	vertex_colors = NEW_int(nb_points);
	int_vec_zero(vertex_colors, nb_points);

	init_adjacency(nb_points, 1 /* nb_colors */, 1 /* nb_colors_per_vertex */,
		vertex_colors, Adj, verbose_level);

	FREE_int(vertex_colors);
	if (f_v) {
		cout << "colored_graph::init_adjacency_no_colors done" << endl;
	}
}

void colored_graph::init_user_data(long int *data,
	int data_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i;
	
	if (f_v) {
		cout << "colored_graph::init_user_data" << endl;
	}
	user_data_size = data_size;
	user_data = NEW_lint(data_size);
	lint_vec_copy(data, user_data, data_size);
	if (f_v) {
		cout << "colored_graph::init_user_data done" << endl;
	}
}

void colored_graph::save(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	graph_theory_domain Graph;

	if (f_v) {
		cout << "colored_graph::save" << endl;
	}

	Graph.save_colored_graph(fname, nb_points, nb_colors, nb_colors_per_vertex,
		points, point_color, 
		user_data, user_data_size, 
		bitvector_adjacency, bitvector_length,
		verbose_level - 1);
	
	if (f_v) {
		cout << "colored_graph::save done" << endl;
	}
}

void colored_graph::load(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	graph_theory_domain Graph;
	file_io Fio;

	if (f_v) {
		cout << "colored_graph::load" << endl;
	}
	if (f_v) {
		cout << "colored_graph::load before Graph.load_colored_graph" << endl;
	}
	Graph.load_colored_graph(fname,
		nb_points, nb_colors, nb_colors_per_vertex,
		points /*vertex_labels*/, point_color /*vertex_colors*/, 
		user_data, user_data_size, 
		bitvector_adjacency, bitvector_length,
		verbose_level);
	if (f_v) {
		cout << "colored_graph::load after Graph.load_colored_graph" << endl;
	}

	f_ownership_of_bitvec = TRUE;

	strcpy(fname_base, fname);
	replace_extension_with(fname_base, "");


	if (f_v) {
		cout << "colored_graph::load Read file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}
}

void colored_graph::all_cliques_of_size_k_ignore_colors(
	int target_depth,
	int *&Sol, int &nb_solutions,
	unsigned long int &decision_step_counter,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	clique_finder *CF;
	int print_interval = 10000000;

	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_"
				"ignore_colors" << endl;
	}
	CF = NEW_OBJECT(clique_finder);

	CF->init("", nb_points, 
		target_depth, 
		FALSE /* f_has_adj_list */, NULL /* int *adj_list_coded */, 
		TRUE /* f_has_bitvector */, bitvector_adjacency, 
		print_interval, 
		FALSE /* f_maxdepth */, 0 /* maxdepth */, 
		TRUE /* f_store_solutions */, 
		verbose_level - 1);

	CF->backtrack_search(0 /* depth */, 0 /* verbose_level */);

	nb_solutions = CF->nb_sol;
	decision_step_counter = CF->decision_step_counter;

	CF->get_solutions(Sol,
			nb_solutions, target_depth, verbose_level);



	FREE_OBJECT(CF);
	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_"
				"ignore_colors done" << endl;
	}
}

void
colored_graph::all_cliques_of_size_k_ignore_colors_and_write_solutions_to_file(
	int target_depth,
	const char *fname, 
	int f_restrictions, int *restrictions, 
	int &nb_sol, unsigned long int &decision_step_counter,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	clique_finder *CF;
	int print_interval = 1000000;

	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors_"
				"and_write_solutions_to_file " << fname << endl;
		if (f_restrictions) {
			cout << "with restrictions: ";
			int_vec_print(cout, restrictions, 3);
			cout << endl;
			}
		}
	CF = NEW_OBJECT(clique_finder);


	file_output *FO;
	FO = NEW_OBJECT(file_output);
	FO->open(fname, CF, verbose_level);

	CF->call_back_clique_found =
			call_back_clique_found_using_file_output;
	CF->call_back_clique_found_data1 = FO;
	CF->call_back_clique_found_data2 = this;

	CF->init("", nb_points, 
		target_depth, 
		FALSE /* f_has_adj_list */, NULL /* int *adj_list_coded */, 
		TRUE /* f_has_bitvector */, bitvector_adjacency, 
		print_interval, 
		FALSE /* f_maxdepth */, 0 /* maxdepth */, 
		TRUE /* f_store_solutions */, 
		verbose_level - 1);

	if (f_restrictions) {
		if (f_v) {
			cout << "colored_graph::all_cliques_of_size_k_ignore_"
					"colors_and_write_solutions_to_file "
					"before init_restrictions" << endl;
			}
		CF->init_restrictions(restrictions, verbose_level - 2);
		}



	CF->backtrack_search(0 /* depth */, 0 /* verbose_level */);

	nb_sol = CF->nb_sol;
	decision_step_counter = CF->decision_step_counter;

	FO->write_EOF(nb_sol, 0 /* verbose_level*/);
	
	FREE_OBJECT(FO);
	FREE_OBJECT(CF);
	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_"
				"colors_and_write_solutions_to_file done" << endl;
		}
}

void colored_graph::all_rainbow_cliques(ofstream *fp,
	int f_output_solution_raw,
	int f_maxdepth, int maxdepth, 
	int f_restrictions, int *restrictions, 
	int f_tree, int f_decision_nodes_only, const char *fname_tree,  
	int print_interval, 
	unsigned long int &search_steps, unsigned long int &decision_steps,
	int &nb_sol, int &dt,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	rainbow_cliques *R;

	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques" << endl;
		}
	R = NEW_OBJECT(rainbow_cliques);
	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques "
				"before R->search" << endl;
		}
	R->search(this, fp, f_output_solution_raw, 
		f_maxdepth, maxdepth, 
		f_restrictions, restrictions, 
		f_tree, f_decision_nodes_only, fname_tree,  
		print_interval, 
		search_steps, decision_steps, nb_sol, dt, 
		verbose_level - 1);
	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques "
				"after R->search" << endl;
		}
	FREE_OBJECT(R);
	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques "
				"done" << endl;
		}
}

void colored_graph::all_rainbow_cliques_with_additional_test_function(
	ofstream *fp, int f_output_solution_raw,
	int f_maxdepth, int maxdepth, 
	int f_restrictions, int *restrictions, 
	int f_tree, int f_decision_nodes_only, const char *fname_tree,  
	int print_interval, 
	int f_has_additional_test_function,
	void (*call_back_additional_test_function)(
		rainbow_cliques *R, void *user_data,
		int current_clique_size, int *current_clique, 
		int nb_pts, int &reduced_nb_pts, 
		int *pt_list, int *pt_list_inv, 
		int verbose_level), 
	int f_has_print_current_choice_function,
	void (*call_back_print_current_choice)(clique_finder *CF, 
		int depth, void *user_data, int verbose_level), 
	void *user_data, 
	unsigned long int &search_steps, unsigned long int &decision_steps,
	int &nb_sol, int &dt,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	rainbow_cliques *R;

	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques_with_additional_"
				"test_function" << endl;
		}
	R = NEW_OBJECT(rainbow_cliques);
	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques_with_additional_"
				"test_function before R->search_with_additional_"
				"test_function" << endl;
		}
	R->search_with_additional_test_function(this, fp, f_output_solution_raw, 
		f_maxdepth, maxdepth, 
		f_restrictions, restrictions, 
		f_tree, f_decision_nodes_only, fname_tree,  
		print_interval, 
		f_has_additional_test_function,
		call_back_additional_test_function, 
		f_has_print_current_choice_function,
		call_back_print_current_choice, 
		user_data, 
		search_steps, decision_steps, nb_sol, dt, 
		verbose_level - 1);
	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques_with_additional_"
				"test_function after R->search_with_additional_"
				"test_function" << endl;
		}
	FREE_OBJECT(R);
	if (f_v) {
		cout << "colored_graph::all_rainbow_cliques_with_additional_"
				"test_function done" << endl;
		}
}

void colored_graph::draw_on_circle(char *fname, 
	int xmax_in, int ymax_in, int xmax_out, int ymax_out,
	int f_radius, double radius, 
	int f_labels, int f_embedded, int f_sideways, 
	double tikz_global_scale, double tikz_global_line_width,
	int verbose_level)
{
	char fname_full[1000];
	file_io Fio;
	
	sprintf(fname_full, "%s.mp", fname);
	{
	mp_graphics G;
	G.setup(fname, 0, 0, 
		xmax_in /* ONE_MILLION */, ymax_in /* ONE_MILLION */, 
		xmax_out, ymax_out, 
		f_embedded, 
		f_sideways, 
		tikz_global_scale, tikz_global_line_width, verbose_level - 1);
	

	//G.header();
	//G.begin_figure(1000 /* factor_1000 */);
	
	draw_on_circle_2(G, f_labels, f_radius, radius);


	G.finish(cout, TRUE);
	}
	cout << "written file " << fname_full << " of size "
			<< Fio.file_size(fname_full) << endl;
	
}

void colored_graph::draw_on_circle_2(
	mp_graphics &G, int f_labels,
	int f_radius, double radius)
{
	int n = nb_points;
	int i, j;
	int *Px, *Py;
	int *Px1, *Py1;
	double phi = 360. / (double) n;
	double rad1 = 500000;
	double rad2 = 5000;
	//char str[1000];
	numerics Num;
	
	Px = NEW_int(n);
	Py = NEW_int(n);
	Px1 = NEW_int(n);
	Py1 = NEW_int(n);
	
	if (f_radius) {
		rad2 = radius;
		}
	for (i = 0; i < n; i++) {
		Num.on_circle_int(Px, Py, i,
				((int)(90. + (double)i * phi)) % 360, rad1);
		//cout << "i=" << i << " Px=" << Px[i]
		// << " Py=" << Py[i] << endl;
		}

	if (f_labels) {
		int rad_big;

		rad_big = (int)((double)rad1 * 1.1);
		cout << "rad_big=" << rad_big << endl;
		for (i = 0; i < n; i++) {
			Num.on_circle_int(Px1, Py1, i,
					((int)(90. + (double)i * phi)) % 360, rad_big);
			//cout << "i=" << i << " Px=" << Px[i]
			// << " Py=" << Py[i] << endl;
			}
		}
	for (i = 0; i < n; i++) {

		if (i) {
			//continue;
			}
		
		for (j = i + 1; j < n; j++) {
			if (is_adjacent(i, j)) {
				G.polygon2(Px, Py, i, j);
				}
			}
		}
	for (i = 0; i < n; i++) {

#if 0
		G.sf_interior(100);
		G.sf_color(0);
		G.circle(Px[i], Py[i], rad2);
#endif

		// draw solid:
		G.sf_interior(1);
		G.sf_color(2 + point_color[i]);
		//G.sf_color(0);
		G.circle(Px[i], Py[i], rad2);

		// draw outline:
		G.sf_interior(0);
		G.sf_color(1);
		//G.sf_color(0);
		G.circle(Px[i], Py[i], rad2);
		}
	if (f_labels) {
		char str[1000];
		for (i = 0; i < n; i++) {
			sprintf(str, "%d", i);
			G.aligned_text(Px1[i], Py1[i], "", str);
			}
		}
	
	FREE_int(Px);
	FREE_int(Py);
	FREE_int(Px1);
	FREE_int(Py1);
}



void colored_graph::draw(const char *fname, 
	int xmax_in, int ymax_in, int xmax_out, int ymax_out,
	double scale, double line_width, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_dots = FALSE;
	uchar *D = NULL;
	long int len, i, j, k;
	int nb_vertices;
	combinatorics_domain Combi;
	graph_theory_domain Graph;
	
	if (f_v) {
		cout << "colored_graph::draw" << endl;
		}


	nb_vertices = nb_points;

	len = ((long int) nb_vertices * (long int) nb_vertices + 7) >> 3;
	if (f_v) {
		cout << "colored_graph::draw len = " << len << endl;
		}
	D = NEW_uchar(len);
	for (i = 0; i < len; i++) {
		D[i] = 0;
		}
	for (i = 0; i < nb_vertices; i++) {
		for (j = i + 1; j < nb_vertices; j++) {
			k = Combi.ij2k(i, j, nb_vertices);
			if (bitvector_s_i(bitvector_adjacency, k)) {
				bitvector_m_ii(D, i * nb_vertices + j, 1);
				bitvector_m_ii(D, j * nb_vertices + i, 1);
				}
			}
		}

	int f_row_grid = FALSE;
	int f_col_grid = FALSE;
	
	Graph.draw_bitmatrix(fname, f_dots,
		FALSE, 0, NULL, 0, NULL, 
		f_row_grid, f_col_grid, 
		TRUE /* f_bitmatrix */, D, NULL, 
		nb_vertices, nb_vertices, 
		xmax_in, ymax_in, xmax_out, ymax_out, 
		scale, line_width, 
		FALSE, NULL, verbose_level - 1);
	

	FREE_uchar(D);
	
	if (f_v) {
		cout << "colored_graph::draw done" << endl;
		}
}

void colored_graph::draw_Levi(const char *fname, 
	int xmax_in, int ymax_in, int xmax_out, int ymax_out,
	int f_partition, int nb_row_parts, int *row_part_first, 
	int nb_col_parts, int *col_part_first, 
	int m, int n, int f_draw_labels, 
	double scale, double line_width, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_dots = FALSE;
	uchar *D = NULL;
	long int len, i, j, k;
	combinatorics_domain Combi;
	graph_theory_domain Graph;
	
	if (f_v) {
		cout << "colored_graph::draw_Levi" << endl;
		}


	if (m + n != nb_points) {
		cout << "colored_graph::draw_Levi "
				"m + n != nb_points" << endl;
		cout << "m = " << m << endl;
		cout << "n = " << n << endl;
		cout << "nb_points = " << nb_points << endl;
		exit(1);
		}

	len = ((long int) m * (long int) n + 7) >> 3;
	if (f_v) {
		cout << "colored_graph::draw_Levi len = " << len << endl;
		}
	D = NEW_uchar(len);
	for (i = 0; i < len; i++) {
		D[i] = 0;
		}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			k = Combi.ij2k_lint(i, m + j, nb_points);
			if (bitvector_s_i(bitvector_adjacency, k)) {
				bitvector_m_ii(D, i * n + j, 1);
				}
			}
		}

	int f_row_grid = FALSE;
	int f_col_grid = FALSE;
	int *labels = NULL;

	if (f_draw_labels) {
		labels = NEW_int(m + n);
		for (i = 0; i < m + n; i++) {
			labels[i] = points[i];
			}
		cout << "colored_graph::draw_Levi label=";
		int_vec_print(cout, labels, m + n);
		cout << endl;
		}
	
	Graph.draw_bitmatrix(fname, f_dots,
		//FALSE, 0, NULL, 0, NULL, 
		f_partition, nb_row_parts, row_part_first,
			nb_col_parts, col_part_first,
		f_row_grid, f_col_grid, 
		TRUE /* f_bitmatrix */, D, NULL, 
		m, n, 
		xmax_in, ymax_in, xmax_out, ymax_out, 
		scale, line_width, 
		f_draw_labels, labels, verbose_level - 1);
	

	FREE_uchar(D);
	if (f_draw_labels) {
		FREE_int(labels);
		}
	
	if (f_v) {
		cout << "colored_graph::draw_Levi done" << endl;
		}
}

void colored_graph::draw_with_a_given_partition(
	const char *fname,
	int xmax_in, int ymax_in, int xmax_out, int ymax_out,
	int *parts, int nb_parts, 
	double scale, double line_width, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_dots = FALSE;
	int f_row_grid = FALSE;
	int f_col_grid = FALSE;
	int nb_vertices;
	uchar *D = NULL;
	long int i, j, k, len;
	int *P;
	combinatorics_domain Combi;
	graph_theory_domain Graph;


	if (f_v) {
		cout << "colored_graph::draw_with_a_given_partition" << endl;
		}

	P = NEW_int(nb_parts + 1);
	P[0] = 0;
	for (i = 0; i < nb_parts; i++) {
		P[i + 1] = P[i] + parts[i];
		}
	
	nb_vertices = nb_points;

	len = ((long int) nb_vertices * (long int) nb_vertices + 7) >> 3;
	if (f_v) {
		cout << "colored_graph::draw_with_a_given_partition "
				"len = " << len << endl;
		}
	D = NEW_uchar(len);
	for (i = 0; i < len; i++) {
		D[i] = 0;
		}

	for (i = 0; i < nb_vertices; i++) {
		for (j = i + 1; j < nb_vertices; j++) {
			k = Combi.ij2k_lint(i, j, nb_vertices);
			if (bitvector_s_i(bitvector_adjacency, k)) {
				bitvector_m_ii(D, i * nb_vertices + j, 1);
				bitvector_m_ii(D, j * nb_vertices + i, 1);
				}
			}
		}

	Graph.draw_bitmatrix(fname, f_dots,
		TRUE, nb_parts, P, nb_parts, P, 
		f_row_grid, f_col_grid, 
		TRUE /* f_bitmatrix */, D, NULL, 
		nb_points, nb_points, 
		xmax_in, ymax_in, xmax_out, ymax_out, 
		scale, line_width, 
		FALSE /*f_has_labels*/, NULL /*labels*/, verbose_level - 1);


	FREE_uchar(D);
	FREE_int(P);
	
	if (f_v) {
		cout << "colored_graph::draw_with_a_given_partition done" << endl;
		}

}

void colored_graph::draw_partitioned(const char *fname, 
	int xmax_in, int ymax_in, int xmax_out, int ymax_out,
	int f_labels, 
	double scale, double line_width, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_dots = FALSE;
	uchar *D = NULL;
	//int xmax_out = 1000000;
	//int ymax_out = 1000000;
	long int len, i, j, k, ii, jj;
	int nb_vertices;
	combinatorics_domain Combi;
	graph_theory_domain Graph;
	
	if (f_v) {
		cout << "colored_graph::draw_partitioned" << endl;
		}


	nb_vertices = nb_points;

	len = ((long int) nb_vertices * (long int) nb_vertices + 7) >> 3;
	if (f_v) {
		cout << "colored_graph::draw_partitioned len = " << len << endl;
		}
	D = NEW_uchar(len);
	for (i = 0; i < len; i++) {
		D[i] = 0;
		}

	classify C;

	C.init(point_color, nb_vertices, FALSE, 0);
	if (f_v) {
		cout << "colored_graph::draw_partitioned we found "
				<< C.nb_types << " classes" << endl;
		}
	
	
	for (i = 0; i < nb_vertices; i++) {
		ii = C.sorting_perm_inv[i];
		for (j = i + 1; j < nb_vertices; j++) {
			jj = C.sorting_perm_inv[j];
			k = Combi.ij2k(ii, jj, nb_vertices);
			if (bitvector_s_i(bitvector_adjacency, k)) {
				bitvector_m_ii(D, i * nb_vertices + j, 1);
				bitvector_m_ii(D, j * nb_vertices + i, 1);
				}
			}
		}
	
	int *part;

	part = NEW_int(C.nb_types + 1);
	for (i = 0; i < C.nb_types; i++) {
		part[i] = C.type_first[i];
		}
	part[C.nb_types] = nb_vertices;

	int f_row_grid = FALSE;
	int f_col_grid = FALSE;

	Graph.draw_bitmatrix(fname, f_dots,
		TRUE, C.nb_types, part, C.nb_types, part, 
		f_row_grid, f_col_grid, 
		TRUE /* f_bitmatrix */, D, NULL, 
		nb_vertices, nb_vertices, 
		xmax_in, ymax_in, xmax_out, ymax_out, 
		scale, line_width, 
		f_labels /*f_has_labels*/, C.sorting_perm_inv /*labels*/, verbose_level - 1);


	FREE_uchar(D);
	FREE_int(part);
	
	if (f_v) {
		cout << "colored_graph::draw_partitioned done" << endl;
		}
}

colored_graph *colored_graph::compute_neighborhood_subgraph(
	int pt,
	fancy_set *&vertex_subset, fancy_set *&color_subset,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	colored_graph *S;
	int *color_in_graph;
	int *color_in_subgraph;
	long int i, j, l, len, ii, jj;
	int c, idx;
	int nb_points_subgraph;
	uchar *bitvec;
	sorting Sorting;

	if (f_v) {
		cout << "colored_graph::compute_neighborhood_subgraph "
				"of point " << pt << endl;
		}
	if (f_v) {
		cout << "The graph has " << nb_points << " vertices and "
				<< nb_colors << " colors" << endl;
		}
	S = NEW_OBJECT(colored_graph);
	vertex_subset = NEW_OBJECT(fancy_set);
	color_subset = NEW_OBJECT(fancy_set);
	color_in_graph = NEW_int(nb_points);
	color_in_subgraph = NEW_int(nb_points * nb_colors_per_vertex);

	vertex_subset->init(nb_points, 0 /* verbose_level */);
	color_subset->init(nb_colors, 0 /* verbose_level */);
	
	for (i = 0; i < nb_points; i++) {
		if (i == pt) {
			continue;
			}
		if (is_adjacent(i, pt)) {
			for (j = 0; j < nb_colors_per_vertex; j++) {
				c = point_color[i * nb_colors_per_vertex + j];
				color_in_graph[vertex_subset->k * nb_colors_per_vertex + j] = c;
				color_subset->add_element(c);
			}
			vertex_subset->add_element(i);
			}
		}


	nb_points_subgraph = vertex_subset->k;

	color_subset->sort();

	if (f_v) {
		cout << "The subgraph has " << nb_points_subgraph
				<< " vertices and " << color_subset->k
				<< " colors" << endl;
		}

	for (i = 0; i < nb_points_subgraph; i++) {
		c = color_in_graph[i];
		if (!Sorting.lint_vec_search(
			color_subset->set, color_subset->k, c, idx, 0)) {
			cout << "error, did not find color" << endl;
			exit(1);
			}
		color_in_subgraph[i] = idx;
		}
	
	l = ((long int) nb_points_subgraph * (long int) (nb_points_subgraph - 1)) >> 1;
	len = (l + 7) >> 3;
	bitvec = NEW_uchar(len);
	for (i = 0; i < len; i++) {
		bitvec[i] = 0;
		}
	S->init(nb_points_subgraph, color_subset->k, nb_colors_per_vertex,
			color_in_subgraph, bitvec, TRUE, verbose_level);
	for (i = 0; i < nb_points_subgraph; i++) {
		ii = vertex_subset->set[i];
		for (j = i + 1; j < nb_points_subgraph; j++) {
			jj = vertex_subset->set[j];
			if (is_adjacent(ii, jj)) {
				S->set_adjacency(i, j, 1);
				S->set_adjacency(j, i, 1);
				}
			}
		}
	FREE_int(color_in_graph);
	FREE_int(color_in_subgraph);
	if (f_v) {
		cout << "colored_graph::compute_neighborhood_subgraph done" << endl;
		}
	return S;
}

colored_graph
*colored_graph::compute_neighborhood_subgraph_with_additional_test_function(
	int pt,
	fancy_set *&vertex_subset, fancy_set *&color_subset, 
	int (*test_function)(colored_graph *CG, int test_point,
			int pt, void *test_function_data, int verbose_level),
	void *test_function_data, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	colored_graph *S;
	int *color_in_graph;
	int *color_in_subgraph;
	long int i, j, l, len, ii, jj;
	int c, idx;
	int nb_points_subgraph;
	uchar *bitvec;
	sorting Sorting;

	if (f_v) {
		cout << "colored_graph::compute_neighborhood_subgraph_with_"
				"additional_test_function of point " << pt << endl;
		}
	if (f_v) {
		cout << "The graph has " << nb_points << " vertices and "
				<< nb_colors << " colors" << endl;
		}
	S = NEW_OBJECT(colored_graph);
	vertex_subset = NEW_OBJECT(fancy_set);
	color_subset = NEW_OBJECT(fancy_set);
	color_in_graph = NEW_int(nb_points);
	color_in_subgraph = NEW_int(nb_points * nb_colors_per_vertex);

	vertex_subset->init(nb_points, 0 /* verbose_level */);
	color_subset->init(nb_colors, 0 /* verbose_level */);
	
	for (i = 0; i < nb_points; i++) {
		if (i == pt) {
			continue;
			}
		if (is_adjacent(i, pt)) {

			if ((*test_function)(this, i, pt, test_function_data,
					0 /*verbose_level*/)) {
				for (j = 0; j < nb_colors_per_vertex; j++) {
					c = point_color[i * nb_colors_per_vertex + j];
					color_in_graph[vertex_subset->k * nb_colors_per_vertex + j] = c;
					color_subset->add_element(c);
				}
				vertex_subset->add_element(i);
				}
			}
		}


	nb_points_subgraph = vertex_subset->k;

	color_subset->sort();

	if (f_v) {
		cout << "The subgraph has " << nb_points_subgraph
				<< " vertices and " << color_subset->k
				<< " colors" << endl;
		}

	for (i = 0; i < nb_points_subgraph; i++) {
		c = color_in_graph[i];
		if (!Sorting.lint_vec_search(color_subset->set,
				color_subset->k, c, idx, 0)) {
			cout << "error, did not find color" << endl;
			exit(1);
			}
		color_in_subgraph[i] = idx;
		}
	
	l = ((long int) nb_points_subgraph * (long int) (nb_points_subgraph - 1)) >> 1;
	len = (l + 7) >> 3;
	bitvec = NEW_uchar(len);
	for (i = 0; i < len; i++) {
		bitvec[i] = 0;
		}
	S->init(nb_points_subgraph, color_subset->k, nb_colors_per_vertex,
			color_in_subgraph, bitvec, TRUE,
			verbose_level);
	for (i = 0; i < nb_points_subgraph; i++) {
		ii = vertex_subset->set[i];
		for (j = i + 1; j < nb_points_subgraph; j++) {
			jj = vertex_subset->set[j];
			if (is_adjacent(ii, jj)) {
				S->set_adjacency(i, j, 1);
				S->set_adjacency(j, i, 1);
				}
			}
		}
	FREE_int(color_in_graph);
	FREE_int(color_in_subgraph);
	if (f_v) {
		cout << "colored_graph::compute_neighborhood_subgraph_"
				"with_additional_test_function done" << endl;
		}
	return S;
}


void colored_graph::export_to_magma(
		const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int *neighbors;
	int nb_neighbors;
	file_io Fio;

	if (f_v) {
		cout << "colored_graph::export_to_magma" << endl;
		}
	{
		ofstream fp(fname);

		neighbors = NEW_int(nb_points);
		fp << "G := Graph< " << nb_points << " | [" << endl;
		for (i = 0; i < nb_points; i++) {


			nb_neighbors = 0;
			for (j = 0; j < nb_points; j++) {
				if (j == i) {
					continue;
					}
				if (is_adjacent(i, j)) {
					neighbors[nb_neighbors++] = j;
					}
				}

			fp << "{";
			for (j = 0; j < nb_neighbors; j++) {
				fp << neighbors[j] + 1;
				if (j < nb_neighbors - 1) {
					fp << ",";
					}
				}
			fp << "}";
			if (i < nb_points - 1) {
				fp << ", " << endl;
				}
			}

		FREE_int(neighbors);
		
		fp << "]>;" << endl;

//> G := Graph< 9 | [ {4,5,6,7,8,9}, {4,5,6,7,8,9}, {4,5,6,7,8,9},
//>                   {1,2,3,7,8,9}, {1,2,3,7,8,9}, {1,2,3,7,8,9},
//>                   {1,2,3,4,5,6}, {1,2,3,4,5,6}, {1,2,3,4,5,6} ]>;


	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "colored_graph::export_to_magma" << endl;
		}
}

void colored_graph::export_to_maple(
		const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;
	int nb_edges;
	file_io Fio;

	if (f_v) {
		cout << "colored_graph::export_to_maple" << endl;
		}
	{
		ofstream fp(fname);




		//4
		//6
		//1 2   1 3   1 4   2 3   2 4   3 4
		//3 4
		//The first integer 4 is the number of vertices.  
		// The second integer 6 is the number of edges. 
		// This is followed by the edges on multiple lines with more 
		// than one edge per line. 
		// To read the graph back into Maple you would do
		//G := ImportGraph(K4, edges);


		fp << nb_points << endl;

		nb_edges = 0;
		for (i = 0; i < nb_points; i++) {
			for (j = i + 1; j < nb_points; j++) {
				if (is_adjacent(i, j)) {
					nb_edges++;
					}
				}

			}
		fp << nb_edges << endl;
		h = 0;
		for (i = 0; i < nb_points; i++) {
			for (j = i + 1; j < nb_points; j++) {
				if (is_adjacent(i, j)) {
					h++;
					fp << i + 1 << " " << j + 1 << " ";
					if ((h % 10) == 0) {
						fp << endl;
						}
					}
				}
			}
		fp << endl;
		fp << endl;



	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "colored_graph::export_to_maple" << endl;
		}
}

void colored_graph::export_to_file(
		const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	file_io Fio;

	if (f_v) {
		cout << "colored_graph::export_to_file" << endl;
		}
	{
		ofstream fp(fname);

		fp << "[" << endl;
		for (i = 0; i < nb_points; i++) {



			fp << "[";
			for (j = 0; j < nb_points; j++) {
				if (is_adjacent(i, j)) {
					fp << "1";
					}
				else {
					fp << "0";
					}
				if (j < nb_points - 1) {
					fp << ",";
					}
				}
			fp << "]";
			if (i < nb_points - 1) {
				fp << ", " << endl;
				}
			}
		fp << "];" << endl;

		


	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "colored_graph::export_to_file" << endl;
		}
}

void colored_graph::export_to_text(
		const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	file_io Fio;

	if (f_v) {
		cout << "colored_graph::export_to_text" << endl;
		}
	{
		ofstream fp(fname);

		fp << "" << endl;
		for (i = 0; i < nb_points; i++) {



			fp << "";
			for (j = 0; j < nb_points; j++) {
				if (is_adjacent(i, j)) {
					fp << "1";
					}
				else {
					fp << "0";
					}
				if (j < nb_points - 1) {
					fp << ", ";
					}
				}
			fp << "";
			if (i < nb_points - 1) {
				fp << " " << endl;
				}
			}
		fp << "" << endl;

		


	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "colored_graph::export_to_text" << endl;
		}
}

void colored_graph::export_laplacian_to_file(
		const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, d;
	file_io Fio;

	if (f_v) {
		cout << "colored_graph::export_laplacian_to_file" << endl;
		}
	{
		ofstream fp(fname);

		fp << "[" << endl;
		for (i = 0; i < nb_points; i++) {


			d = 0;
			for (j = 0; j < nb_points; j++) {
				if (j == i) {
					continue;
					}
				if (is_adjacent(i, j)) {
					d++;
					}
				}

			fp << "[";
			for (j = 0; j < nb_points; j++) {
				if (j == i) {
					fp << d;
					}
				else {
					if (is_adjacent(i, j)) {
						fp << "-1";
						}
					else {
						fp << "0";
						}
					}
				if (j < nb_points - 1) {
					fp << ",";
					}
				}
			fp << "]";
			if (i < nb_points - 1) {
				fp << ", " << endl;
				}
			}
		fp << "];" << endl;

		


	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "colored_graph::export_laplacian_to_file" << endl;
		}
}

void colored_graph::export_to_file_matlab(
		const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	file_io Fio;

	if (f_v) {
		cout << "colored_graph::export_to_file_matlab" << endl;
		}
	{
		ofstream fp(fname);

		fp << "A = [" << endl;
		for (i = 0; i < nb_points; i++) {



			//fp << "[";
			for (j = 0; j < nb_points; j++) {
				if (is_adjacent(i, j)) {
					fp << "1";
					}
				else {
					fp << "0";
					}
				if (j < nb_points - 1) {
					fp << ",";
					}
				}
			//fp << "]";
			if (i < nb_points - 1) {
				fp << "; " << endl;
				}
			}
		fp << "]" << endl;

		


	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "colored_graph::export_to_file" << endl;
		}
}


void colored_graph::early_test_func_for_clique_search(
	long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	long int j, a, pt;

	if (f_v) {
		cout << "colored_graph::early_test_func_for_clique_"
				"search checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}
	if (len == 0) {
		nb_good_candidates = nb_candidates;
		lint_vec_copy(candidates, good_candidates, nb_candidates);
		return;
		}

	pt = S[len - 1];

	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {
		a = candidates[j];
		
		if (is_adjacent(pt, a)) {
			good_candidates[nb_good_candidates++] = a;
			}
		} // next j
	
}

void colored_graph::early_test_func_for_coclique_search(
	long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	long int j, a, pt;

	if (f_v) {
		cout << "colored_graph::early_test_func_for_"
				"coclique_search checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}
	if (len == 0) {
		nb_good_candidates = nb_candidates;
		lint_vec_copy(candidates, good_candidates, nb_candidates);
		return;
		}

	pt = S[len - 1];

	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {
		a = candidates[j];
		
		if (!is_adjacent(pt, a)) {
			good_candidates[nb_good_candidates++] = a;
			}
		} // next j
	
}

void colored_graph::early_test_func_for_path_and_cycle_search(
		long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	long int i, j, a, b, /*pt,*/ x, y;
	int *v;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "colored_graph::early_test_func_for_path_and_"
				"cycle_search checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}
	if (len == 0) {
		nb_good_candidates = nb_candidates;
		lint_vec_copy(candidates, good_candidates, nb_candidates);
		return;
		}

	v = NEW_int(nb_points);
	int_vec_zero(v, nb_points);

	//pt = S[len - 1];

	for (i = 0; i < len; i++) {
		a = S[i];
		b = list_of_edges[a];
		Combi.k2ij_lint(b, x, y, nb_points);
		v[x]++;
		v[y]++;
		}

	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {
		a = candidates[j];
		b = list_of_edges[a];
		Combi.k2ij_lint(b, x, y, nb_points);
		
		if (v[x] < 2 && v[y] < 2) {
			good_candidates[nb_good_candidates++] = a;
			}
		} // next j
	
	FREE_int(v);
}

int colored_graph::is_cycle(int nb_e, long int *edges,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, b, x, y;
	int *v;
	//int ret = TRUE;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "colored_graph::is_cycle" << endl;
		}
	v = NEW_int(nb_points);
	int_vec_zero(v, nb_points);
	
	for (i = 0; i < nb_e; i++) {
		a = edges[i];
		b = list_of_edges[a];
		Combi.k2ij(b, x, y, nb_points);
		v[x]++;
		v[y]++;
		}

	//ret = TRUE;
	for (i = 0; i < nb_points; i++) {
		if (v[i] != 0 && v[i] != 2) {
			//ret = FALSE;
			break;
			}
		}
	

	FREE_int(v);	
	if (f_v) {
		cout << "colored_graph::is_cycle done" << endl;
		}
	return TRUE;
}


void colored_graph::draw_it(const char *fname_base, 
	int xmax_in, int ymax_in, int xmax_out, int ymax_out, 
	double scale, double line_width, int verbose_level)
{
	int f_dots = FALSE;
	int f_partition = FALSE;
	int f_bitmatrix = TRUE;
	int f_row_grid = FALSE;
	int f_col_grid = FALSE;

	int L, length, i, j, k, a;
	uchar *bitvec;
	combinatorics_domain Combi;
	graph_theory_domain Graph;

	L = nb_points * nb_points;
	length = (L + 7) >> 3;
	bitvec = NEW_uchar(length);
	for (i = 0; i < length; i++) {
		bitvec[i] = 0;
		}
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			k = Combi.ij2k(i, j, nb_points);
			a = bitvector_s_i(bitvector_adjacency, k);
			if (a) {
				k = i * nb_points + j;
				bitvector_m_ii(bitvec, k, 1);
				k = j * nb_points + i;
				bitvector_m_ii(bitvec, k, 1);
				}
			}
		}

	Graph.draw_bitmatrix(fname_base, f_dots,
		f_partition, 0, NULL, 0, NULL, 
		f_row_grid, f_col_grid, 
		f_bitmatrix, bitvec, NULL, 
		nb_points, nb_points,
		xmax_in, ymax_in, xmax_out, ymax_out,
		scale, line_width, 
		FALSE, NULL, verbose_level - 1);

	FREE_uchar(bitvec);
	
}




#if 0
int colored_graph::rainbow_cliques_nonrecursive(
	int &nb_backtrack_nodes,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "colored_graph::rainbow_cliques_nonrecursive" << endl;
		}

	int *live_pts;
	int *start;
	int *end;
	int *min_color;
	int *choice;
	int *cc; // [nb_colors]
	int *cf; // [nb_colors]
	int *cs; // [nb_colors]
	int target_depth = nb_colors;
	int nb_sol = 0;
	int depth;
	int c0;
	int nb_min_color;
	int i, a, c, s;


	live_pts = NEW_int(nb_points);
	start = NEW_int(nb_colors + 1);
	end = NEW_int(nb_colors + 1);
	min_color = NEW_int(nb_colors + 1);
	choice = NEW_int(nb_colors + 1);
	cc = NEW_int(nb_colors);
	cf = NEW_int(nb_colors);
	cs = NEW_int(nb_colors);

	for (i = 0; i < nb_points; i++) {
		live_pts[i] = i;
		}
	
	start[0] = 0;
	end[0] = nb_points;
	for (i = 0; i < nb_colors; i++) {
		min_color[i] = -1;
		choice[i] = 0;
		cs[i] = FALSE;
		}
	
	depth = 0;
	nb_backtrack_nodes = 0;

	while (TRUE) {

		nb_backtrack_nodes++;


#if 0
		if (depth == 2 && cc[1] != 9963) {
			depth--;
			}

		if (depth == 3 && cc[2] != 10462) {
			depth--;
			}
		if (depth == 4 && cc[3] != 1) {
			depth--;
			}
#endif

		if (f_vv) {
			cout << "nb_backtrack=" << nb_backtrack_nodes
					<< " depth=" << depth
			<< " : cc=";
			int_vec_print(cout, cc, depth);
			cout << " : start=";
			int_vec_print(cout, start, depth + 1);
			cout << " : end=";
			int_vec_print(cout, end, depth + 1);
			cout << " : min_color=";
			int_vec_print(cout, min_color, depth + 1);
			cout << " : choice=";
			int_vec_print(cout, choice, depth + 1);
			cout << endl;
			cout << "live_pts=";
			int_vec_print_fully(cout, live_pts + start[depth],
					end[depth] - start[depth]);
			cout << endl;
			cout << "live points (full) = ";
			int_vec_print_fully(cout, live_pts, nb_points);
			cout << endl;
			}

		
		if (depth == target_depth) {
			cout << "solution " << nb_sol << " : ";
			int_vec_print_fully(cout, cc, depth);
			cout << endl;
			nb_sol++;
			depth--;
			}

		if (min_color[depth] == -1) {

			if (f_vv) {
				cout << "nb_backtrack=" << nb_backtrack_nodes
						<< " depth=" << depth
						<< ", initializing n e w level" << endl;
				}
			// clump by adjacency:
			if (depth) {

				a = cc[depth - 1];
				s = start[depth];
				for (i = s; i < end[depth]; i++) {
					if (is_adjacent(a, live_pts[i])) {
						if (i != s) {
							int tmp;

							tmp = live_pts[s];
							live_pts[s] = live_pts[i];
							live_pts[i] = tmp;
							}
						s++;
						}
					}
				end[depth + 1] = s;
				}
			else {
				end[depth + 1] = end[depth];
				}

			if (f_vv) {
				cout << "nb_backtrack=" << nb_backtrack_nodes
						<< " depth=" << depth << ", after clump "
						"by adjacency end[" << depth + 1
						<< "]=" << end[depth + 1] << endl;
				cout << "nb_backtrack=" << nb_backtrack_nodes
						<< " depth=" << depth;
				cout << endl;
				cout << "live points = ";
				int_vec_print_fully(cout, live_pts + start[depth],
						end[depth + 1] - start[depth]);
				cout << endl;
				cout << "live points (full) = ";
				int_vec_print_fully(cout, live_pts, nb_points);
				cout << endl;
				}

			// compute color frequency:
			for (i = 0; i < nb_colors; i++) {
				cf[i] = 0;
				}
			for (i = start[depth]; i < end[depth + 1]; i++) {
				a = live_pts[i];
				c = point_color[a];
				cf[c]++;
				}
			nb_min_color = INT_MAX;
			c0 = -1;
			for (c = 0; c < nb_colors; c++) {
				if (cf[c] < nb_min_color && !cs[c]) {
					c0 = c;
					nb_min_color = cf[c];
					}
				}
			if (f_vv) {
				cout << "nb_backtrack=" << nb_backtrack_nodes
						<< " depth=" << depth << ", cf = ";
				int_vec_print(cout, cf, nb_colors);
				cout << endl;
				cout << "nb_backtrack=" << nb_backtrack_nodes
						<< " depth=" << depth << ", cs = ";
				int_vec_print(cout, cs, nb_colors);
				cout << endl;
				}


			min_color[depth] = c0;

			// clamp by color class:
			s = start[depth];
			for (i = s; i < end[depth + 1]; i++) {
				if (point_color[live_pts[i]] == c0) {
					if (i != s) {
						int tmp;

						tmp = live_pts[s];
						live_pts[s] = live_pts[i];
						live_pts[i] = tmp;
						}
					s++;
					}
				}
			start[depth + 1] = s;
			choice[depth] = 0;

			if (f_vv) {
				cout << "nb_backtrack=" << nb_backtrack_nodes
						<< " depth=" << depth
						<< ", after clump by color class start["
						<< depth + 1 << "]=" << start[depth + 1]
													  << endl;
				cout << "nb_backtrack=" << nb_backtrack_nodes
						<< " depth=" << depth << ", min color = "
						<< c0 << endl;
				cout << "min color class " << min_color[depth]
						<< " of size "
						<< start[depth + 1] - start[depth] << endl;
				//cout << "starts at start[depth]= " << start[depth] << endl;
				//cout << "ends at start[depth+1]= " << start[depth+1] << endl;
				int_vec_print_fully(cout, live_pts + start[depth],
						start[depth + 1] - start[depth]);
				cout << endl;
				}

#if 0
			if (depth == 5) {
				exit(1);
				}
#endif
			} // if mincolor

		int j;

		j = choice[depth];


		if (start[depth + 1] - start[depth]) {
			if (j < start[depth + 1] - start[depth]) {
				if (f_vv) {
					cout << "nb_backtrack=" << nb_backtrack_nodes
							<< " depth=" << depth
							<< ", j < start[depth + 1] - start[depth]"
							<< endl;
					}
				if (j) {
					a = cc[depth];
					c = point_color[a];
					cs[c] = FALSE;
						// this can be dropped since
						// all points have the same color
					if (f_vv) {
						cout << "nb_backtrack=" << nb_backtrack_nodes
								<< " depth=" << depth << ", dropping point "
								<< a << " of color " << c << endl;
						}
					}
				a = live_pts[start[depth] + j];
				c = point_color[a];
				if (f_vv) {
					cout << "nb_backtrack=" << nb_backtrack_nodes
						<< " depth=" << depth << ", adding point "
						<< a << " of color " << c << endl;
					}

				cs[c] = TRUE;
				cc[depth] = a;
				choice[depth]++;


#if 0
				if (depth == 4 
					&& cc[0] == 28 
					&& cc[1] == 9963
					&& cc[2] == 10462 
					&& cc[3] == 1 
					&& cc[4] == 948
					//&& cc[5] == 7816
					) {
					cout << "nb_backtrack=" << nb_backtrack_nodes
							<< " depth=" << depth << " at checkpoint" << endl;
					cout << "live points = ";
					int_vec_print_fully(cout, live_pts + start[depth],
							end[depth + 1] - start[depth]);
					cout << endl;
					cout << "live points (full) = ";
					int_vec_print_fully(cout, live_pts, nb_points);
					cout << endl;
					cout << "cc=";
					int_vec_print(cout, cc, depth + 1);
					cout << "start=";
					int_vec_print(cout, start, depth + 1);
					cout << "end=";
					int_vec_print(cout, end, depth + 1);
					cout << "min_color=";
					int_vec_print(cout, min_color, depth + 1);
					cout << "choice=";
					int_vec_print(cout, choice, depth + 1);
					cout << endl;
					cout << "cs=";
					int_vec_print(cout, cs, nb_colors);
					cout << endl;
					cout << "min color class " << min_color[depth]
						<< " of size " << start[depth + 1] - start[depth]
						<< endl;
					//cout << "starts at start[depth]= "
					// << start[depth] << endl;
					//cout << "ends at start[depth+1]= "
					// << start[depth+1] << endl;
					int_vec_print_fully(cout, live_pts + start[depth],
						start[depth + 1] - start[depth]);
					cout << endl;

					//exit(1);
					f_vv = TRUE;
					}
#endif


				depth++;
				}
			else {
				if (f_vv) {
					cout << "nb_backtrack=" << nb_backtrack_nodes
						<< " depth=" << depth
						<< ", j not < start[depth + 1] - start[depth]" << endl;
					}
				a = cc[depth];
				c = point_color[a];
				if (f_vv) {
					cout << "nb_backtrack=" << nb_backtrack_nodes
						<< " depth=" << depth << ", dropping point "
						<< a << " of color " << c << endl;
					}
				cs[c] = FALSE;
				min_color[depth] = -1;
				choice[depth] = 0;
				depth--;
				}
			}
		else {
			if (f_vv) {
				cout << "nb_backtrack=" << nb_backtrack_nodes << " depth="
					<< depth << ", minimum color class is empty, "
					"backtracking" << endl;
				}
			// we could not go in, so we don't have to clean up anything
			min_color[depth] = -1;
			choice[depth] = 0;
			depth--;
			}

		if (depth < 0) break;
		
		} // while


	FREE_int(live_pts);
	FREE_int(start);
	FREE_int(end);
	FREE_int(min_color);
	FREE_int(choice);
	FREE_int(cc);
	FREE_int(cf);
	FREE_int(cs);
	if (f_v) {
		cout << "colored_graph::rainbow_cliques_nonrecursive done" << endl;
		}
	return nb_sol;
}
#endif

void colored_graph::create_Levi_graph_from_incidence_matrix(
	int *M, int nb_rows, int nb_cols,
	int f_point_labels, long int *point_labels,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	uchar *bitvector_adjacency;
	int L, /*bitvector_length_in_bits,*/ bitvector_length;
	int i, j, k, r, c;
	int N;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "colored_graph::create_Levi_graph_from_incidence_matrix" << endl;
		}
	N = nb_rows + nb_cols;
	L = (N * (N - 1)) >> 1;

	//bitvector_length_in_bits = L;
	bitvector_length = (L + 7) >> 3;
	bitvector_adjacency = NEW_uchar(bitvector_length);
	for (i = 0; i < bitvector_length; i++) {
		bitvector_adjacency[i] = 0;
		}


	for (r = 0; r < nb_rows; r++) {
		i = r;
		for (c = 0; c < nb_cols; c++) {
			if (M[r * nb_cols + c]) {
				j = nb_rows + c;
				k = Combi.ij2k(i, j, N);
				bitvector_m_ii(bitvector_adjacency, k, 1);
				}
			}
		}

	if (f_point_labels) {
		init_with_point_labels(N, 1 /* nb_colors */, 1 /* nb_colors_per_vertex */,
			NULL /*point_color*/, bitvector_adjacency,
			TRUE, point_labels, verbose_level - 2);
			// the adjacency becomes part of the colored_graph object
		}
	else {
		init(N, 1 /* nb_colors */, 1 /* nb_colors_per_vertex */,
			NULL /*point_color*/, bitvector_adjacency,
			TRUE, verbose_level - 2);
			// the adjacency becomes part of the colored_graph object
		}

	if (f_v) {
		cout << "colored_graph::create_Levi_graph_from_incidence_matrix "
				"done" << endl;
		}
}



// #############################################################################
// global functions:
// #############################################################################

void call_back_clique_found_using_file_output(
	clique_finder *CF, int verbose_level)
{
	//int f_v = (verbose_level >= 1);

	//cout << "call_back_clique_found_using_file_output" << endl;
	
	file_output *FO = (file_output *) CF->call_back_clique_found_data1;
	colored_graph *CG = (colored_graph *) CF->call_back_clique_found_data2;
	//clique_finder *CF = (clique_finder *) FO->user_data;

	if (CG->user_data_size && CG->points) {
		int i, a;
		*FO->fp << CG->user_data_size + CF->target_depth;
		for (i = 0; i < CG->user_data_size; i++) {
			*FO->fp << " " << CG->user_data[i];
			}
		for (i = 0; i < CF->target_depth; i++) {
			a = CF->current_clique[i];
			*FO->fp << " " << CG->points[a];
			}
		*FO->fp << endl;
		}
	else {
		FO->write_line(CF->target_depth,
				CF->current_clique, verbose_level);
		}
}




}
}

