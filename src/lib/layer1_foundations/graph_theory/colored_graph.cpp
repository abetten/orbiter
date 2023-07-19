// colored_graph.cpp
//
// Anton Betten
//
// started:  October 28, 2012




#include "foundations.h"
#include "Clique/RainbowClique.h"
#include "KClique.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace graph_theory {


colored_graph::colored_graph()
{

	//fname_base[0] = 0;
	nb_points = nb_colors = nb_colors_per_vertex = 0;
	//bitvector_length = 0;
	L = 0;

	points = NULL;
	point_color = NULL;

	user_data_size = 0;
	user_data = NULL;

	f_ownership_of_bitvec = false;
	Bitvec = NULL;
	//bitvector_adjacency = NULL;
	f_has_list_of_edges = false;
	nb_edges = 0;
	list_of_edges = NULL;
}

colored_graph::~colored_graph()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::~colored_graph" << endl;
	}
	if (user_data) {
		if (f_v) {
			cout << "colored_graph::~colored_graph user_data" << endl;
		}
		FREE_lint(user_data);
	}
	if (points) {
		if (f_v) {
			cout << "colored_graph::~colored_graph points" << endl;
		}
		FREE_lint(points);
	}
	if (point_color) {
		if (f_v) {
			cout << "colored_graph::~colored_graph point_color" << endl;
		}
		FREE_int(point_color);
	}
	if (f_ownership_of_bitvec) {
#if 0
		if (bitvector_adjacency) {
			if (f_v) {
				cout << "colored_graph::~colored_graph "
						"bitvector_adjacency" << endl;
			}
			FREE_uchar(bitvector_adjacency);
		}
#else
		if (Bitvec) {
			FREE_OBJECT(Bitvec);
			Bitvec = NULL;
		}
#endif
	}
	if (list_of_edges) {
		if (f_v) {
			cout << "colored_graph::~colored_graph list_of_edges" << endl;
		}
		FREE_int(list_of_edges);
	}
	if (f_v) {
		cout << "colored_graph::~colored_graph" << endl;
	}
}

void colored_graph::compute_edges(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, nb, a;
	combinatorics::combinatorics_domain Combi;

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
		cout << "colored_graph::compute_edges nb_edges != nb" << endl;
		exit(1);
	}

	f_has_list_of_edges = true;
	if (f_v) {
		cout << "colored_graph::compute_edges done" << endl;
	}
}


int colored_graph::is_adjacent(int i, int j)
{
	combinatorics::combinatorics_domain Combi;

	if (i == j) {
		return false;
	}
	if (i > j) {
		return is_adjacent(j, i);
	}
	long int k;
	
	k = Combi.ij2k_lint(i, j, nb_points);
	return Bitvec->s_i(k);
}

void colored_graph::set_adjacency(int i, int j, int a)
{
	combinatorics::combinatorics_domain Combi;
	long int k;

	k = Combi.ij2k_lint(i, j, nb_points);
	Bitvec->m_i(k, a);
}

void colored_graph::set_adjacency_k(long int k, int a)
{
	combinatorics::combinatorics_domain Combi;

	Bitvec->m_i(k, a);
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
		cout << "colored_graph::partition_by_color_classes "
				"nb_colors_per_vertex != 1" << endl;
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

	data_structures::tally C;

	C.init(point_color, nb_points, false, 0);
	if (f_v) {
		cout << "point color distribution: ";
		C.print_naked(true);
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
		Color, A,
		label, label_tex,
		0 /* verbose_level */);
	CG->init_user_data(user_data, user_data_size,
			0 /* verbose_level */);
	Lint_vec_copy(Pts, CG->points, nb_points);
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
		cout << "colored_graph::subgraph_by_color_classes "
				"nb_colors_per_vertex != 1" << endl;
		exit(1);
	}

	data_structures::tally C;

	C.init(point_color, nb_points, false, 0);
	if (f_v) {
		cout << "point color distribution: ";
		C.print_naked(true);
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

	Int_vec_zero(A, l * l);

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
	CG->init_adjacency(
			l /* nb_points */, 1 /*nb_colors */, 1 /* nb_colors_per_vertex*/,
		Color, A,
		label, label_tex,
		0 /* verbose_level */);
	CG->init_user_data(user_data, user_data_size,
			0 /* verbose_level */);
	Lint_vec_copy(Pts, CG->points, l);
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

	data_structures::tally C;

	C.init(point_color, nb_points, false, 0);
	if (f_v) {
		cout << "point color distribution: ";
		C.print_naked(true);
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
	Int_vec_zero(A, nb_pts * nb_pts);

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
	CG->init_adjacency(nb_pts /* nb_points */,
			nb_colors, 1 /* nb_colors_per_vertex */,
			Color, A,
			label, label_tex,
			0 /* verbose_level */);
	CG->init_user_data(user_data, user_data_size,
			0 /* verbose_level */);
	Lint_vec_copy(Pts, CG->points, nb_pts);
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
	
	data_structures::tally C;

	C.init(point_color, nb_points, true, 0);

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
	C.print_first(true);
	cout << endl;

	cout << "color class sizes: ";
	C.print_second(true);
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
	int f_first = true;
	
	cout << "Adjacency list:" << endl;
	for (i = 0; i < nb_points; i++) {
		cout << i << " : ";
		f_first = true;
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
				f_first = false;
			}
		}
		cout << endl;
	}
	cout << "Adjacency list using point labels:" << endl;
	for (i = 0; i < nb_points; i++) {
		cout << points[i] << " : ";
		f_first = true;
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
				f_first = false;
			}
		}
		cout << endl;
	}
	
}

void colored_graph::init_with_point_labels(int nb_points,
		int nb_colors, int nb_colors_per_vertex,
		int *colors,
		data_structures::bitvector *Bitvec,
		int f_ownership_of_bitvec,
		long int *point_labels,
		std::string &label,
		std::string &label_tex,
		int verbose_level)
// point_labels is copied over
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "colored_graph::init_with_point_labels" << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "nb_colors=" << nb_colors << endl;
		cout << "nb_colors_per_vertex=" << nb_colors_per_vertex << endl;
	}
	init(
			nb_points, nb_colors, nb_colors_per_vertex,
		colors,
		Bitvec, f_ownership_of_bitvec,
		label, label_tex,
		verbose_level);
	Lint_vec_copy(point_labels, points, nb_points);
	if (f_v) {
		cout << "colored_graph::init_with_point_labels done" << endl;
	}
}

void colored_graph::init(
		int nb_points, int nb_colors, int nb_colors_per_vertex,
	int *colors, data_structures::bitvector *Bitvec,
	int f_ownership_of_bitvec,
	std::string &label, std::string &label_tex,
	int verbose_level)
// colors is copied over
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
	colored_graph::label.assign(label);
	colored_graph::label_tex.assign(label_tex);
	if (f_v) {
		cout << "colored_graph::init colored_graph::label = " << label << endl;
		cout << "colored_graph::init colored_graph::label_tex = " << label_tex << endl;
	}


	L = ((long int) nb_points * (long int) (nb_points - 1)) >> 1;

	user_data_size = 0;
	
	points = NEW_lint(nb_points);
	for (i = 0; i < nb_points; i++) {
		points[i] = i;
	}
	point_color = NEW_int(nb_points * nb_colors_per_vertex);

	if (colors) {
		Int_vec_copy(colors, point_color, nb_points * nb_colors_per_vertex);
	}
	else {
		Int_vec_zero(point_color, nb_points * nb_colors_per_vertex);
	}
	
	colored_graph::f_ownership_of_bitvec = f_ownership_of_bitvec;
	colored_graph::Bitvec = Bitvec;

	if (f_v) {
		cout << "colored_graph::init" << endl;
	}

}

void colored_graph::init_no_colors(
		int nb_points,
		data_structures::bitvector *Bitvec,
		int f_ownership_of_bitvec,
		std::string &label, std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *vertex_colors;

	if (f_v) {
		cout << "colored_graph::init_no_colors" << endl;
		cout << "nb_points=" << nb_points << endl;
	}
	vertex_colors = NEW_int(nb_points);
	Int_vec_zero(vertex_colors, nb_points);

	init(
			nb_points, 1 /* nb_colors */, 1 /* nb_colors_per_vertex */,
		vertex_colors, Bitvec, f_ownership_of_bitvec,
		label, label_tex,
		verbose_level);

	FREE_int(vertex_colors);
	if (f_v) {
		cout << "colored_graph::init_no_colors done" << endl;
	}
}

void colored_graph::init_adjacency(
		int nb_points,
		int nb_colors, int nb_colors_per_vertex,
		int *colors, int *Adj,
		std::string &label, std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k;
	combinatorics::combinatorics_domain Combi;


	if (f_v) {
		cout << "colored_graph::init_adjacency" << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "nb_colors=" << nb_colors << endl;
	}
	L = ((long int) nb_points * (long int) (nb_points - 1)) >> 1;

	Bitvec = NEW_OBJECT(data_structures::bitvector);
	Bitvec->allocate(L);
	k = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++, k++) {
			if (Adj[i * nb_points + j]) {
				//k = Combi.ij2k_lint(i, j, nb_points);
				Bitvec->m_i(k, 1);
			}
		}
	}
	init(
			nb_points, nb_colors, nb_colors_per_vertex,
		colors, Bitvec, true /* f_ownership_of_bitvec */,
		label, label_tex,
		verbose_level);

	// do not free bitvec here

	if (f_v) {
		cout << "colored_graph::init_adjacency" << endl;
	}
}

void colored_graph::init_adjacency_upper_triangle(
	int nb_points, int nb_colors, int nb_colors_per_vertex,
	int *colors, int *Adj,
	std::string &label, std::string &label_tex,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k;
	combinatorics::combinatorics_domain Combi;


	if (f_v) {
		cout << "colored_graph::init_adjacency_upper_triangle" << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "nb_colors=" << nb_colors << endl;
		cout << "nb_colors_per_vertex=" << nb_colors_per_vertex << endl;
		}
	L = ((long int) nb_points * (long int) (nb_points - 1)) >> 1;


	Bitvec = NEW_OBJECT(data_structures::bitvector);
	Bitvec->allocate(L);

	k = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++, k++) {
			//k = Combi.ij2k_lint(i, j, nb_points);
			if (Adj[k]) {
				Bitvec->m_i(k, 1);
			}
		}
	}
	init(
			nb_points, nb_colors, nb_colors_per_vertex,
		colors, Bitvec, true /* f_ownership_of_bitvec */,
		label, label_tex,
		verbose_level);

	// do not free Bitvec here

	if (f_v) {
		cout << "colored_graph::init_adjacency_upper_triangle done" << endl;
	}

}

void colored_graph::init_adjacency_no_colors(
		int nb_points,
	int *Adj,
	std::string &label, std::string &label_tex,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *vertex_colors;

	if (f_v) {
		cout << "colored_graph::init_adjacency_no_colors" << endl;
		cout << "nb_points=" << nb_points << endl;
	}
	vertex_colors = NEW_int(nb_points);
	Int_vec_zero(vertex_colors, nb_points);

	init_adjacency(
			nb_points,
			1 /* nb_colors */, 1 /* nb_colors_per_vertex */,
			vertex_colors, Adj,
			label, label_tex,
			verbose_level);

	FREE_int(vertex_colors);
	if (f_v) {
		cout << "colored_graph::init_adjacency_no_colors done" << endl;
	}
}

void colored_graph::init_adjacency_two_colors(
		int nb_points,
	int *Adj, int *subset, int sz,
	std::string &label, std::string &label_tex,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *vertex_colors;
	int i, a;

	if (f_v) {
		cout << "colored_graph::init_adjacency_two_colors" << endl;
		cout << "nb_points=" << nb_points << endl;
		cout << "sz=" << sz << endl;
	}

	vertex_colors = NEW_int(nb_points);
	Int_vec_zero(vertex_colors, nb_points);
	for (i = 0; i < sz; i++) {
		a = subset[i];
		vertex_colors[a] = 1;
	}

	init_adjacency(
			nb_points,
			2 /* nb_colors */, 1 /* nb_colors_per_vertex */,
			vertex_colors, Adj,
			label, label_tex,
			verbose_level);

	FREE_int(vertex_colors);
	if (f_v) {
		cout << "colored_graph::init_adjacency_two_colors done" << endl;
	}
}

void colored_graph::init_user_data(
		long int *data,
	int data_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "colored_graph::init_user_data" << endl;
	}
	user_data_size = data_size;
	user_data = NEW_lint(data_size);
	Lint_vec_copy(data, user_data, data_size);
	if (f_v) {
		cout << "colored_graph::init_user_data done" << endl;
	}
}

void colored_graph::save(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	graph_theory_domain Graph;

	if (f_v) {
		cout << "colored_graph::save" << endl;
	}

	Graph.save_colored_graph(
			fname,
			nb_points, nb_colors, nb_colors_per_vertex,
			points, point_color,
			user_data, user_data_size,
			Bitvec,
			verbose_level - 1);
	
	if (f_v) {
		cout << "colored_graph::save done" << endl;
	}
}

void colored_graph::load(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	graph_theory_domain Graph;
	orbiter_kernel_system::file_io Fio;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "colored_graph::load" << endl;
	}
	if (f_v) {
		cout << "colored_graph::load before Graph.load_colored_graph" << endl;
	}
	Graph.load_colored_graph(
			fname,
		nb_points, nb_colors, nb_colors_per_vertex,
		points /*vertex_labels*/, point_color /*vertex_colors*/, 
		user_data, user_data_size, 
		Bitvec,
		verbose_level);
	if (f_v) {
		cout << "colored_graph::load after Graph.load_colored_graph" << endl;
	}

	f_ownership_of_bitvec = true;

	fname_base.assign(fname);
	ST.replace_extension_with(fname_base, "");


	if (f_v) {
		cout << "colored_graph::load Read file " << fname
				<< " of size " << Fio.file_size(fname.c_str()) << endl;
	}
}

void colored_graph::draw_on_circle(
		std::string &fname,
		graphics::layered_graph_draw_options *Draw_options,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::draw_on_circle" << endl;
	}
	string fname_full;
	orbiter_kernel_system::file_io Fio;
	
	fname_full = fname + ".mp";

	{
		graphics::mp_graphics G;
		int factor_1000 = 1000;

		G.init(fname, Draw_options, verbose_level - 1);

		G.header();

		G.begin_figure(factor_1000);

		draw_on_circle_2(G, Draw_options);

		G.end_figure();
		G.footer();
	
		//G.finish(cout, true);
	}
	if (f_v) {
		cout << "colored_graph::draw_on_circle done" << endl;
	}
}

void colored_graph::draw_on_circle_2(
		graphics::mp_graphics &G,
	graphics::layered_graph_draw_options *Draw_options)
{
	int n = nb_points;
	int i, j;
	int *Px, *Py;
	int *Px1, *Py1;
	double phi = 360. / (double) n;
	double rad1 = 5000; // a big circle for the vertices
	double rad2 = 6000; // a bigger circle for the labels
	orbiter_kernel_system::numerics Num;
	
	Px = NEW_int(n);
	Py = NEW_int(n);
	Px1 = NEW_int(n);
	Py1 = NEW_int(n);
	
	if (true) {
		rad2 = Draw_options->rad;
	}
	for (i = 0; i < n; i++) {
		Num.on_circle_int(Px, Py, i,
				((int)(90. + (double)i * phi)) % 360, rad1);
		//cout << "i=" << i << " Px=" << Px[i]
		// << " Py=" << Py[i] << endl;
	}

	if (!Draw_options->f_nodes_empty) {
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
	if (!Draw_options->f_nodes_empty) {
		string s;
		for (i = 0; i < n; i++) {




			if (nb_colors_per_vertex == 1) {

				s = "$" + std::to_string(i) + "_{" + std::to_string(point_color[i]) + "}$";

			}
			else {

				s = std::to_string(i);

			}


			G.aligned_text(Px1[i], Py1[i], "", s);
		}
	}
	
	FREE_int(Px);
	FREE_int(Py);
	FREE_int(Px1);
	FREE_int(Py1);
}

void colored_graph::create_bitmatrix(
		data_structures::bitmatrix *&Bitmatrix,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::create_bitmatrix" << endl;
	}
	long int i, j, k;

	Bitmatrix = NEW_OBJECT(data_structures::bitmatrix);
	Bitmatrix->init(nb_points, nb_points, verbose_level);
	k = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++, k++) {
			//k = Combi.ij2k(i, j, nb_points);
			if (Bitvec->s_i(k)) {
				Bitmatrix->m_ij(i, j, 1);
				Bitmatrix->m_ij(j, i, 1);
			}
		}
	}

	if (f_v) {
		cout << "colored_graph::create_bitmatrix done" << endl;
	}
}


void colored_graph::draw(
		std::string &fname,
		graphics::layered_graph_draw_options *Draw_options,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::draw" << endl;
	}
	int f_dots = false;
	data_structures::bitmatrix *Bitmatrix;
	combinatorics::combinatorics_domain Combi;
	graph_theory_domain Graph;
	


	create_bitmatrix(Bitmatrix, verbose_level);

	int f_row_grid = false;
	int f_col_grid = false;
	
	Graph.draw_bitmatrix(
			fname,
			Draw_options,
			f_dots,
		false, 0, NULL, 0, NULL, 
		f_row_grid, f_col_grid, 
		true /* f_bitmatrix */, Bitmatrix, NULL,
		nb_points, nb_points,
		false, NULL, verbose_level - 1);
	

	FREE_OBJECT(Bitmatrix);
	
	if (f_v) {
		cout << "colored_graph::draw done" << endl;
	}
}

void colored_graph::draw_Levi(
		std::string &fname,
		graphics::layered_graph_draw_options *Draw_options,
	int f_partition, int nb_row_parts, int *row_part_first, 
	int nb_col_parts, int *col_part_first, 
	int m, int n, int f_draw_labels, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_dots = false;
	data_structures::bitmatrix *Bitmatrix;
	long int len, i, j, k;
	combinatorics::combinatorics_domain Combi;
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
	Bitmatrix = NEW_OBJECT(data_structures::bitmatrix);
	Bitmatrix->init(m + n, m + n, verbose_level);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			k = Combi.ij2k_lint(i, m + j, nb_points);
			if (Bitvec->s_i(k)) {
				Bitmatrix->m_ij(i, m + j, 1);
				Bitmatrix->m_ij(m + j, i, 1);
			}
		}
	}

	int f_row_grid = false;
	int f_col_grid = false;
	int *labels = NULL;

	if (f_draw_labels) {
		labels = NEW_int(m + n);
		for (i = 0; i < m + n; i++) {
			labels[i] = points[i];
		}
		cout << "colored_graph::draw_Levi label=";
		Int_vec_print(cout, labels, m + n);
		cout << endl;
	}
	
	Graph.draw_bitmatrix(
			fname,
			Draw_options,
			f_dots,
		f_partition, nb_row_parts, row_part_first,
			nb_col_parts, col_part_first,
		f_row_grid, f_col_grid, 
		true /* f_bitmatrix */, Bitmatrix, NULL,
		m, n, 
		f_draw_labels, labels,
		verbose_level - 1);
	

	FREE_OBJECT(Bitmatrix);

	if (f_draw_labels) {
		FREE_int(labels);
	}
	
	if (f_v) {
		cout << "colored_graph::draw_Levi done" << endl;
	}
}

void colored_graph::draw_with_a_given_partition(
		std::string &fname,
		graphics::layered_graph_draw_options *Draw_options,
		int *parts, int nb_parts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_dots = false;
	int f_row_grid = false;
	int f_col_grid = false;
	data_structures::bitmatrix *Bitmatrix;
	int i;
	int *P;
	combinatorics::combinatorics_domain Combi;
	graph_theory_domain Graph;


	if (f_v) {
		cout << "colored_graph::draw_with_a_given_partition" << endl;
	}

	P = NEW_int(nb_parts + 1);
	P[0] = 0;
	for (i = 0; i < nb_parts; i++) {
		P[i + 1] = P[i] + parts[i];
	}
	
	create_bitmatrix(Bitmatrix, verbose_level);

	Graph.draw_bitmatrix(
			fname,
			Draw_options,
			f_dots,
		true, nb_parts, P, nb_parts, P, 
		f_row_grid, f_col_grid, 
		true /* f_bitmatrix */, Bitmatrix, NULL,
		nb_points, nb_points, 
		false /*f_has_labels*/, NULL /*labels*/,
		verbose_level - 1);


	FREE_OBJECT(Bitmatrix);
	FREE_int(P);
	
	if (f_v) {
		cout << "colored_graph::draw_with_a_given_partition done" << endl;
	}

}

void colored_graph::draw_partitioned(
		std::string &fname,
		graphics::layered_graph_draw_options *Draw_options,
	int f_labels,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_dots = false;
	data_structures::bitmatrix *Bitmatrix;
	long int len, i, j, k;
	combinatorics::combinatorics_domain Combi;
	graph_theory_domain Graph;
	
	if (f_v) {
		cout << "colored_graph::draw_partitioned" << endl;
	}



	len = ((long int) nb_points * (long int) nb_points + 7) >> 3;
	if (f_v) {
		cout << "colored_graph::draw_partitioned len = " << len << endl;
	}


	data_structures::tally C;

	C.init(point_color, nb_points, false, 0);
	if (f_v) {
		cout << "colored_graph::draw_partitioned we found "
				<< C.nb_types << " classes" << endl;
	}


	Bitmatrix = NEW_OBJECT(data_structures::bitmatrix);
	Bitmatrix->init(nb_points, nb_points, verbose_level);
	k = 0;
	for (i = 0; i < nb_points; i++) {
		//ii = C.sorting_perm_inv[i];
		for (j = i + 1; j < nb_points; j++, k++) {
			//jj = C.sorting_perm_inv[j];
			//k = Combi.ij2k_lint(ii, jj, nb_vertices);
			if (Bitvec->s_i(k)) {
				Bitmatrix->m_ij(i, j, 1);
				Bitmatrix->m_ij(j, i, 1);
			}
		}
	}

	
	int *part;

	part = NEW_int(C.nb_types + 1);
	for (i = 0; i < C.nb_types; i++) {
		part[i] = C.type_first[i];
	}
	part[C.nb_types] = nb_points;

	int f_row_grid = false;
	int f_col_grid = false;

	Graph.draw_bitmatrix(
			fname,
			Draw_options,
			f_dots,
		true, C.nb_types, part, C.nb_types, part, 
		f_row_grid, f_col_grid, 
		true /* f_bitmatrix */, Bitmatrix, NULL,
		nb_points, nb_points,
		f_labels /*f_has_labels*/,
		C.sorting_perm_inv /*labels*/,
		verbose_level - 1);


	FREE_OBJECT(Bitmatrix);
	FREE_int(part);
	
	if (f_v) {
		cout << "colored_graph::draw_partitioned done" << endl;
	}
}


colored_graph *colored_graph::compute_neighborhood_subgraph(
	int pt,
	data_structures::fancy_set *&vertex_subset,
	data_structures::fancy_set *&color_subset,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	colored_graph *S;
	int *color_in_graph;
	int *color_in_subgraph;
	long int i, j, l, ii, jj;
	long int *point_labels;
	int c, idx;
	int nb_points_subgraph;
	data_structures::bitvector *Bitvec;
	data_structures::sorting Sorting;
	long int *subgraph_user_data;

	if (f_v) {
		cout << "colored_graph::compute_neighborhood_subgraph "
				"of point " << pt << endl;
	}
	if (f_v) {
		cout << "The graph has " << nb_points << " vertices and "
				<< nb_colors << " colors" << endl;
	}
	S = NEW_OBJECT(colored_graph);
	vertex_subset = NEW_OBJECT(data_structures::fancy_set);
	color_subset = NEW_OBJECT(data_structures::fancy_set);
	point_labels = NEW_lint(nb_points);

	// new user data = old user data plus the label of the point pt:
	subgraph_user_data = NEW_lint(user_data_size + 1);
	Lint_vec_copy(user_data, subgraph_user_data, user_data_size);
	subgraph_user_data[user_data_size] = points[pt];

	color_in_graph = NEW_int(nb_points * nb_colors_per_vertex);
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
			point_labels[vertex_subset->k] = points[i];
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
		for (j = 0; j < nb_colors_per_vertex; j++) {
			c = color_in_graph[i * nb_colors_per_vertex + j];
			if (!Sorting.lint_vec_search(
				color_subset->set, color_subset->k, c, idx, 0)) {
				cout << "error, did not find color" << endl;
				exit(1);
			}
			color_in_subgraph[i * nb_colors_per_vertex + j] = idx;
		}
	}
	
	Bitvec = NEW_OBJECT(data_structures::bitvector);

	l = ((long int) nb_points_subgraph * (long int) (nb_points_subgraph - 1)) >> 1;


	Bitvec->allocate(l);

	S->init(nb_points_subgraph,
			color_subset->k, nb_colors_per_vertex,
			color_in_subgraph,
			Bitvec, true,
			label, label_tex,
			verbose_level);




	// set the vertex labels:
	Lint_vec_copy(point_labels, S->points, nb_points_subgraph);

	S->init_user_data(subgraph_user_data, user_data_size + 1, verbose_level);

	if (f_v) {
		cout << "colored_graph::compute_neighborhood_subgraph "
				"computing adjacency matrix of subgraph" << endl;
	}
	long int k;

	k = 0;
	for (i = 0; i < nb_points_subgraph; i++) {
		ii = vertex_subset->set[i];
		for (j = i + 1; j < nb_points_subgraph; j++, k++) {
			jj = vertex_subset->set[j];
			if (is_adjacent(ii, jj)) {
				S->set_adjacency_k(k, 1);
				//S->set_adjacency(j, i, 1);
			}
		}
	}
	FREE_lint(subgraph_user_data);
	FREE_lint(point_labels);
	FREE_int(color_in_graph);
	FREE_int(color_in_subgraph);
	if (f_v) {
		cout << "colored_graph::compute_neighborhood_subgraph done" << endl;
	}
	return S;
}


colored_graph *colored_graph::compute_neighborhood_subgraph_based_on_subset(
	long int *subset, int subset_sz,
	data_structures::fancy_set *&vertex_subset,
	data_structures::fancy_set *&color_subset,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	colored_graph *S;
	int *color_in_graph;
	int *color_in_subgraph;
	long int i, j, l, ii, jj;
	long int *point_labels;
	int c, idx;
	int nb_points_subgraph;
	data_structures::bitvector *Bitvec;
	data_structures::sorting Sorting;
	long int *subgraph_user_data;

	if (f_v) {
		cout << "colored_graph::compute_neighborhood_subgraph "
				"of set ";
		Lint_vec_print(cout, subset, subset_sz);
		cout << endl;
	}
	if (f_v) {
		cout << "The graph has " << nb_points << " vertices and "
				<< nb_colors << " colors" << endl;
	}
	S = NEW_OBJECT(colored_graph);
	vertex_subset = NEW_OBJECT(data_structures::fancy_set);
	color_subset = NEW_OBJECT(data_structures::fancy_set);
	point_labels = NEW_lint(nb_points);

	// new user data = old user data plus the label of the point pt:
	subgraph_user_data = NEW_lint(user_data_size + subset_sz);
	Lint_vec_copy(user_data, subgraph_user_data, user_data_size);
	Lint_vec_copy(subset, subgraph_user_data + user_data_size, subset_sz);

	color_in_graph = NEW_int(nb_points * nb_colors_per_vertex);
	color_in_subgraph = NEW_int(nb_points * nb_colors_per_vertex);

	vertex_subset->init(nb_points, 0 /* verbose_level */);
	color_subset->init(nb_colors, 0 /* verbose_level */);

	for (i = 0; i < nb_points; i++) {
		for (j = 0; j < subset_sz; j++) {
			if (i == subset[j]) {
				break;
			}
		}
		if (j < subset_sz) {
			continue;
		}
		for (j = 0; j < subset_sz; j++) {
			if (!is_adjacent(i, subset[j])) {
				break;
			}
		}
		if (j < subset_sz) {
			continue;
		}
		for (j = 0; j < nb_colors_per_vertex; j++) {
			c = point_color[i * nb_colors_per_vertex + j];
			color_in_graph[vertex_subset->k * nb_colors_per_vertex + j] = c;
			color_subset->add_element(c);
		}
		point_labels[vertex_subset->k] = points[i];
		vertex_subset->add_element(i);
	}


	nb_points_subgraph = vertex_subset->k;

	color_subset->sort();

	if (f_v) {
		cout << "The subgraph has " << nb_points_subgraph
				<< " vertices and " << color_subset->k
				<< " colors" << endl;
	}

	for (i = 0; i < nb_points_subgraph; i++) {
		for (j = 0; j < nb_colors_per_vertex; j++) {
			c = color_in_graph[i * nb_colors_per_vertex + j];
			if (!Sorting.lint_vec_search(
				color_subset->set, color_subset->k, c, idx, 0)) {
				cout << "error, did not find color" << endl;
				exit(1);
			}
			color_in_subgraph[i * nb_colors_per_vertex + j] = idx;
		}
	}

	Bitvec = NEW_OBJECT(data_structures::bitvector);

	l = ((long int) nb_points_subgraph * (long int) (nb_points_subgraph - 1)) >> 1;


	Bitvec->allocate(l);

	S->init(nb_points_subgraph,
			color_subset->k, nb_colors_per_vertex,
			color_in_subgraph,
			Bitvec, true,
			label, label_tex,
			0 /*verbose_level*/);




	// set the vertex labels:
	Lint_vec_copy(point_labels, S->points, nb_points_subgraph);

	S->init_user_data(subgraph_user_data,
			user_data_size + subset_sz, verbose_level);

	if (f_v) {
		cout << "colored_graph::compute_neighborhood_subgraph "
				"computing adjacency matrix of subgraph" << endl;
	}
	long int k;

	k = 0;
	for (i = 0; i < nb_points_subgraph; i++) {
		ii = vertex_subset->set[i];
		for (j = i + 1; j < nb_points_subgraph; j++, k++) {
			jj = vertex_subset->set[j];
			if (is_adjacent(ii, jj)) {
				S->set_adjacency_k(k, 1);
				//S->set_adjacency(j, i, 1);
			}
		}
	}
	FREE_lint(subgraph_user_data);
	FREE_lint(point_labels);
	FREE_int(color_in_graph);
	FREE_int(color_in_subgraph);
	if (f_v) {
		cout << "colored_graph::compute_neighborhood_subgraph done" << endl;
	}
	return S;
}



#if 0
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
			color_in_subgraph, bitvec, true,
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
#endif


void colored_graph::export_to_magma(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::export_to_magma" << endl;
	}

	l1_interfaces::interface_magma_low Interface;

	if (f_v) {
		cout << "colored_graph::export_to_magma "
				"before Interface.export_colored_graph_to_magma" << endl;
	}
	Interface.export_colored_graph_to_magma(
			this,
			fname, verbose_level);
	if (f_v) {
		cout << "colored_graph::export_to_magma "
				"after Interface.export_colored_graph_to_magma" << endl;
	}


	if (f_v) {
		cout << "colored_graph::export_to_magma" << endl;
	}
}

void colored_graph::export_to_maple(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;
	int nb_edges;
	orbiter_kernel_system::file_io Fio;

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
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	orbiter_kernel_system::file_io Fio;

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
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "colored_graph::export_to_text" << endl;
	}
	{
		ofstream fp(fname.c_str());

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
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, d;
	orbiter_kernel_system::file_io Fio;

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
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	orbiter_kernel_system::file_io Fio;

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

void colored_graph::export_to_csv(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int *M;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "colored_graph::export_to_csv" << endl;
	}

	M = NEW_int(nb_points * nb_points);
	Int_vec_zero(M, nb_points * nb_points);

	for (i = 0; i < nb_points; i++) {
		for (j = 0; j < nb_points; j++) {
			if (is_adjacent(i, j)) {
				M[i * nb_points + j] = 1;
			}
		}
	}


	Fio.Csv_file_support->int_matrix_write_csv(
			fname, M, nb_points, nb_points);

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "colored_graph::export_to_csv done" << endl;
		}
}


void colored_graph::export_to_graphviz(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int *M;
	orbiter_kernel_system::file_io Fio;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "colored_graph::export_to_graphviz" << endl;
	}

	M = NEW_int(nb_points * nb_points);
	Int_vec_zero(M, nb_points * nb_points);

	for (i = 0; i < nb_points; i++) {
		for (j = 0; j < nb_points; j++) {
			if (is_adjacent(i, j)) {
				M[i * nb_points + j] = 1;
			}
		}
	}

	{
		ofstream ost(fname);

		string label;

		label.assign(fname);
		ST.chop_off_extension(label);

		ost << "graph " << label << " {" << std::endl;

		//export_graphviz_recursion(ost);
		for (i = 0; i < nb_points; i++) {
			ost << i << " [label=\"" << i << "\" ]" << endl;
			for (j = i + 1; j < nb_points; j++) {
				if (is_adjacent(i, j)) {

					ost << i << " -- " << j << endl;
				}
			}
		}

		ost << "}" << std::endl;

	}
	//Fio.int_matrix_write_csv(fname, M, nb_points, nb_points);

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "colored_graph::export_to_graphviz done" << endl;
		}
}

void colored_graph::early_test_func_for_clique_search(
	long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int j, a, pt;

	if (f_v) {
		cout << "colored_graph::early_test_func_for_clique_search "
				"checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}
	if (len == 0) {
		nb_good_candidates = nb_candidates;
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
		return;
	}

	pt = S[len - 1];

	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {
		a = candidates[j];
		
		if (is_adjacent(pt, a)) {
			good_candidates[nb_good_candidates++] = a;
		}
	}
	
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
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}
	if (len == 0) {
		nb_good_candidates = nb_candidates;
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
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
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "colored_graph::early_test_func_for_path_and_"
				"cycle_search checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}
	if (len == 0) {
		nb_good_candidates = nb_candidates;
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
		return;
	}

	v = NEW_int(nb_points);
	Int_vec_zero(v, nb_points);

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

int colored_graph::is_cycle(
		int nb_e, long int *edges,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, b, x, y;
	int *v;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "colored_graph::is_cycle" << endl;
	}
	v = NEW_int(nb_points);
	Int_vec_zero(v, nb_points);
	
	for (i = 0; i < nb_e; i++) {
		a = edges[i];
		b = list_of_edges[a];
		Combi.k2ij(b, x, y, nb_points);
		v[x]++;
		v[y]++;
	}

	//ret = true;
	for (i = 0; i < nb_points; i++) {
		if (v[i] != 0 && v[i] != 2) {
			//ret = false;
			break;
		}
	}
	

	FREE_int(v);	
	if (f_v) {
		cout << "colored_graph::is_cycle done" << endl;
	}
	return true;
}





#if 0
void colored_graph::create_Levi_graph_from_incidence_matrix(
	int *M, int nb_rows, int nb_cols,
	int f_point_labels, long int *point_labels,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//uchar *bitvector_adjacency;
	long int L, bitvector_length;
	long int k;
	int i, j, r, c;
	int N;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "colored_graph::create_Levi_graph_from_incidence_matrix" << endl;
	}
	N = nb_rows + nb_cols;
	L = ((long int) N * ((long int) N - 1)) >> 1;

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
				k = Combi.ij2k_lint(i, j, N);
				bitvector_m_ii(bitvector_adjacency, k, 1);
			}
		}
	}

	if (f_point_labels) {
		init_with_point_labels(N, 1 /* nb_colors */, 1 /* nb_colors_per_vertex */,
			NULL /*point_color*/, Bitvec_adjacency,
			true, point_labels, verbose_level - 2);
			// the adjacency becomes part of the colored_graph object
	}
	else {
		init(N, 1 /* nb_colors */, 1 /* nb_colors_per_vertex */,
			NULL /*point_color*/, bitvector_adjacency,
			true, verbose_level - 2);
			// the adjacency becomes part of the colored_graph object
	}

	if (f_v) {
		cout << "colored_graph::create_Levi_graph_from_incidence_matrix "
				"done" << endl;
	}
}
#endif



void colored_graph::all_cliques(
		clique_finder_control *Control,
		std::string &graph_label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	string fname_sol;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "colored_graph::all_cliques" << endl;
	}
	if (Control->f_output_file) {
		fname_sol.assign(Control->output_file);
	}
	else {

		fname_sol = graph_label;
		ST.chop_off_extension(fname_sol);
		fname_sol += "_sol";
		//ST.replace_extension_with(fname_sol, "_sol.txt");
	}
	if (f_v) {
		cout << "colored_graph::all_cliques "
				"graph_label=" << graph_label << endl;
		cout << "colored_graph::all_cliques "
				"fname_sol=" << fname_sol << endl;
	}



	{
		string fname_sol_csv;
		string fname_sol_txt;


		fname_sol_csv = fname_sol + ".csv";
		fname_sol_txt = fname_sol + ".txt";

		ofstream fp(fname_sol_txt);
		ofstream fp_csv(fname_sol_csv);


		if (Control->f_rainbow) {

			if (f_v) {
				cout << "colored_graph::all_cliques f_rainbow" << endl;
			}
			if (f_v) {
				cout << "colored_graph::all_cliques "
						"before all_cliques_rainbow" << endl;
			}
			all_cliques_rainbow(
					Control,
					fp,
					fp_csv,
					verbose_level);
			if (f_v) {
				cout << "colored_graph::all_cliques "
						"after all_cliques_rainbow" << endl;
			}
			if (f_v) {
				cout << "colored_graph::all_cliques "
						"f_rainbow done" << endl;
			}

		}
		else if (Control->f_weighted) {

			if (f_v) {
				cout << "colored_graph::all_cliques weighted cliques" << endl;
			}



			if (f_v) {
				cout << "colored_graph::all_cliques "
						"before all_cliques_weighted_with_two_colors" << endl;
			}
			all_cliques_weighted_with_two_colors(Control, verbose_level);
			if (f_v) {
				cout << "colored_graph::all_cliques "
						"after all_cliques_weighted_with_two_colors" << endl;
			}



			if (f_v) {
				cout << "colored_graph::all_cliques "
						"weighted cliques done" << endl;
			}

		}
		else {

			if (f_v) {
				cout << "colored_graph::all_cliques "
						"before all_cliques_black_and_white" << endl;
			}
			all_cliques_black_and_white(
					Control,
					fp,
					fp_csv,
					verbose_level);
			if (f_v) {
				cout << "colored_graph::all_cliques "
						"after all_cliques_black_and_white" << endl;
			}

		}
		fp << -1 << " " << Control->nb_sol << " " << Control->nb_search_steps
			<< " " << Control->nb_decision_steps << " " << Control->dt << endl;
		fp_csv << "END" << endl;
	}


	if (f_v) {
		cout << "colored_graph::all_cliques done" << endl;
	}
}




void colored_graph::all_cliques_rainbow(
		clique_finder_control *Control,
		std::ostream &ost_txt,
		std::ostream &ost_csv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::all_cliques_rainbow" << endl;
		cout << "nb_points = " << nb_points << endl;
	}

	if (Control->f_Sajeeb) {
		if (f_v) {
			cout << "colored_graph::all_cliques_rainbow "
					"before do_Sajeeb" << endl;
		}
		std::vector<std::vector<unsigned int> > solutions;

		do_Sajeeb(Control, solutions, verbose_level);
		if (f_v) {
			cout << "colored_graph::all_cliques_rainbow "
					"after do_Sajeeb" << endl;
		}
		if (Control->f_store_solutions) {




			int sz;

			sz = nb_colors;

			ost_csv << "ROW";
			for (int j = 0; j < sz; ++j) {
				ost_csv << ",C" << j;
			}
			ost_csv << endl;

			for (int i = 0; i < solutions.size(); ++i) {
				ost_csv << i << ",";

				if (sz != solutions[i].size()) {
					cout << "colored_graph::all_cliques_rainbow "
							"sz != solutions[i].size()" << endl;
					cout << "sz = " << sz << endl;
					cout << "solutions[i].size() = " << solutions[i].size() << endl;
							exit(1);
				}
				for (int j = 0; j < sz; ++j) {

					if (points) {
						ost_csv << points[solutions[i][j]];
					}
					else {
						ost_csv << solutions[i][j];
					}
					//fp_csv << Control->Sol[i * Control->target_size + j];
					if (j < sz - 1) {
						ost_csv << ",";
					}
				}
				ost_csv << endl;
			}

			Control->Sol = NEW_int(solutions.size() * sz);
			Control->nb_sol = solutions.size();
			for (int i = 0; i < solutions.size(); ++i) {
				for (int j = 0; j < sz; ++j) {
					long int a;
					if (points) {
						a = points[solutions[i][j]];
					}
					else {
						a = solutions[i][j];
					}
					Control->Sol[i * Control->target_size + j] = a;
				}
			}

		}
	}
	else {
		if (f_v) {
			cout << "colored_graph::all_cliques_rainbow "
					"before CG.all_rainbow_cliques" << endl;
		}
		all_rainbow_cliques(Control,
				ost_txt,
				verbose_level - 1);
		if (f_v) {
			cout << "colored_graph::all_cliques_rainbow "
					"after CG.all_rainbow_cliques" << endl;
		}


		if (Control->f_store_solutions) {
			if (f_v) {
				cout << "colored_graph::all_cliques_rainbow "
						"before write_solutions_to_csv_file" << endl;
			}
			write_solutions_to_csv_file(Control, ost_csv, verbose_level);
			if (f_v) {
				cout << "colored_graph::all_cliques_rainbow "
						"after write_solutions_to_csv_file" << endl;
			}
		}

	}
	if (f_v) {
		cout << "colored_graph::all_cliques_rainbow done" << endl;
	}
}

void colored_graph::find_subgraph(
		std::string &subgraph_label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	data_structures::string_tools ST;

	if (f_v) {
		cout << "colored_graph::find_subgraph" << endl;
	}

	if (ST.stringcmp(subgraph_label, "E6") == 0) {
		if (f_v) {
			cout << "colored_graph::find_subgraph "
					"before find_subgraph_E6" << endl;
		}
		find_subgraph_E6(verbose_level);
		if (f_v) {
			cout << "colored_graph::find_subgraph "
					"after find_subgraph_E6" << endl;
		}
	}
	else {
		cout << "colored_graph::find_subgraph "
				"subgraph label not recognized" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "colored_graph::find_subgraph done" << endl;
	}
}

void colored_graph::find_subgraph_E6(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	data_structures::string_tools ST;

	if (f_v) {
		cout << "colored_graph::find_subgraph_E6" << endl;
	}
	if (f_v) {
		cout << "colored_graph::find_subgraph_E6 "
				"nb_points = " << nb_points << endl;
	}

	int E6[6];
	int *N0;
	int *M0;
	int sz_N0;
	int sz_M0;
	int i0, i1, i2, i3, i4, i5, i, j, a, b, cnt;
	int S[36];
	int T[] = {
			0,1,1,1,0,0,
			1,0,0,0,1,0,
			1,0,0,0,0,1,
			1,0,0,0,0,0,
			0,1,0,0,0,0,
			0,0,1,0,0,0 };

	N0 = NEW_int(nb_points);
	M0 = NEW_int(nb_points);

	cnt = 0;

	for (i0 = 0; i0 < nb_points; i0++) {
		E6[0] = i0;

		// compute neighbors and non-neighbors of the first chosen vertex:
		sz_N0 = 0;
		sz_M0 = 0;
		for (j = 0; j < nb_points; j++) {
			if (j == i0) {
				continue;
			}
			if (is_adjacent(i0, j)) {
				N0[sz_N0++] = j;
			}
			else {
				M0[sz_M0++] = j;
			}
		}

		// choose the next three, adjacent to the first:

		for (i1 = 0; i1 < sz_N0; i1++) {
			E6[1] = N0[i1];
			for (i2 = 0; i2 < sz_N0; i2++) {
				if (i2 == i1) {
					continue;
				}
				if (is_adjacent(N0[i2], N0[i1])) {
					continue;
				}
				E6[2] = N0[i2];
				for (i3 = 0; i3 < sz_N0; i3++) {
					if (i3 == i1) {
						continue;
					}
					if (i3 == i2) {
						continue;
					}
					if (is_adjacent(N0[i3], N0[i1])) {
						continue;
					}
					if (is_adjacent(N0[i3], N0[i2])) {
						continue;
					}
					E6[3] = N0[i3];


					for (i4 = 0; i4 < sz_M0; i4++) {
						if (!is_adjacent(M0[i4], E6[1])) {
							continue;
						}
						if (is_adjacent(M0[i4], E6[2])) {
							continue;
						}
						if (is_adjacent(M0[i4], E6[3])) {
							continue;
						}
						E6[4] = M0[i4];

						for (i5 = 0; i5 < sz_M0; i5++) {
							if (i5 == i4) {
								continue;
							}
							if (!is_adjacent(M0[i5], E6[2])) {
								continue;
							}
							if (is_adjacent(M0[i5], E6[1])) {
								continue;
							}
							if (is_adjacent(M0[i5], E6[3])) {
								continue;
							}
							if (is_adjacent(M0[i5], M0[i4])) {
								continue;
							}
							E6[5] = M0[i5];

							Int_vec_zero(S, 36);
							for (i = 0; i < 6; i++) {
								a = E6[i];
								for (j = 0; j < 6; j++) {
									b = E6[j];
									if (is_adjacent(a, b)) {
										S[i * 6 + j] = 1;
									}
								}
							}

							cnt++;
							cout << "solution " << cnt << " : ";
							Int_vec_print(cout, E6, 6);
							cout << endl;
							//Int_matrix_print(S, 6, 6);
							//cout << endl;

							for (i = 0; i < 36; i++) {
								if (S[i] != T[i]) {
									cout << "solution is not correct" << endl;
									Int_matrix_print(S, 6, 6);
									cout << endl;
									exit(1);
								}
							}

							//exit(1);
						}
					}

				}
			}
		}

	}

	if (f_v) {
		cout << "colored_graph::find_subgraph_E6 done" << endl;
	}
}




void colored_graph::all_cliques_black_and_white(
		clique_finder_control *Control,
		std::ostream &ost_txt,
		std::ostream &ost_csv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::all_cliques_black_and_white" << endl;
	}

	if (!Control->f_target_size) {
		cout << "colored_graph::all_cliques_black_and_white "
				"please use -target_size <int : target_size>" << endl;
		exit(1);
	}

	if (Control->f_Sajeeb) {
		if (f_v) {
			cout << "colored_graph::all_cliques_black_and_white "
					"before do_Sajeeb_black_and_white" << endl;
		}
		std::vector<std::vector<long int> > solutions;

		do_Sajeeb_black_and_white(Control, solutions, verbose_level - 2);

		// Print the solutions
		if (f_v) {
			cout << "colored_graph::all_cliques_black_and_white "
					"after do_Sajeeb_black_and_white "
					"Found " << solutions.size() << " solution(s)." << endl;
		}


		if (f_v) {
			cout << "colored_graph::all_cliques_black_and_white "
					"before writing solutions to file" << endl;
		}

		#if 1
		for (size_t i = 0; i < solutions.size(); ++i) {
			ost_txt << solutions[i].size() << " ";
			for (size_t j = 0; j < solutions[i].size(); ++j) {
				ost_txt << points[solutions[i][j]] << " ";
			}
			ost_txt << endl;
		}

		ost_csv << "ROW";
		for (int j = 0; j < Control->target_size; ++j) {
			ost_csv << ",C" << j;
		}
		ost_csv << endl;

		for (size_t i = 0; i < solutions.size(); ++i) {
			for (size_t j = 0; j < solutions[i].size(); ++j) {
				ost_csv << points[solutions[i][j]];
				if (j < solutions[i].size() - 1) {
					ost_csv << " ";
				}
			}
			ost_csv << endl;
		}
		#endif
		if (f_v) {
			cout << "colored_graph::all_cliques_black_and_white "
					"after writing solutions to file" << endl;
		}
	}
	else {


		if (f_v) {
			cout << "colored_graph::all_cliques_black_and_white "
					"before CG->all_cliques_of_size_k_ignore_colors" << endl;
		}
		all_cliques_of_size_k_ignore_colors(
				Control,
				verbose_level - 2);
		if (f_v) {
			cout << "colored_graph::all_cliques_black_and_white "
					"after CG->all_cliques_of_size_k_ignore_colors, "
					"nb_cliques = " << Control->nb_sol << endl;
		}

		if (f_v) {
			cout << "colored_graph::all_cliques_black_and_white "
					"before writing solutions to file" << endl;
		}
		for (int i = 0; i < Control->nb_sol; ++i) {
			ost_txt << Control->target_size << " ";
			for (int j = 0; j < Control->target_size; ++j) {
				ost_txt << points[Control->Sol[i * Control->target_size + j]];
				if (j < Control->target_size - 1) {
					ost_txt << " ";
				}
			}
			ost_txt << endl;
		}

		write_solutions_to_csv_file(Control, ost_csv, verbose_level);

		if (f_v) {
			cout << "colored_graph::all_cliques_black_and_white "
					"after writing solutions to file" << endl;
		}

	}

	if (f_v) {
		cout << "colored_graph::all_cliques_black_and_white done" << endl;
	}
}


void colored_graph::write_solutions_to_csv_file(
		clique_finder_control *Control,
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::write_solutions_to_csv_file" << endl;
	}

	ost << "ROW";
	for (int j = 0; j < Control->target_size; ++j) {
		ost << ",C" << j;
	}
	ost << endl;

	for (int i = 0; i < Control->nb_sol; ++i) {
		ost << i << ",";
		for (int j = 0; j < Control->target_size; ++j) {

			if (points) {
				ost << points[Control->Sol[i * Control->target_size + j]];
			}
			else {
				ost << Control->Sol[i * Control->target_size + j];
			}
			//fp_csv << Control->Sol[i * Control->target_size + j];
			if (j < Control->target_size - 1) {
				ost << ",";
			}
		}
		ost << endl;
	}

	if (f_v) {
		cout << "colored_graph::write_solutions_to_csv_file done" << endl;
	}

}

void colored_graph::do_Sajeeb(
		clique_finder_control *Control,
		std::vector<std::vector<unsigned int> > &solutions,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::do_Sajeeb" << endl;
	}

#if 1
	Graph<> G (nb_points, nb_colors, nb_colors_per_vertex);

	for (size_t i=0; i<G.nb_vertices; ++i) G.vertex_label[i] = points[i];
	for (size_t i=0; i<G.nb_vertices * G.nb_colors_per_vertex; ++i) G.vertex_color[i] = point_color[i]; // Anton: error corrected, was nb_colors should be nb_vertices

	G.set_edge_from_bitvector_adjacency(Bitvec);

	// Create the solution storage. The base type of the solution
	// storage must be the same as data type of the vertex label
	// in the graph
	//std::vector<std::vector<unsigned int> > solutions;
	if (f_v) {
		cout << __FILE__ << ":" << __LINE__ << endl;
	}

	if (f_v) {
		cout << "colored_graph::do_Sajeeb "
				"before RainbowClique::find_cliques" << endl;
	}
    // Call the Rainbow Clique finding algorithm
	RainbowClique::find_cliques(G, solutions, 0 /* nb_threads */);
		// nb_threads = 0 automatically detects the number of threads
	if (f_v) {
		cout << "colored_graph::do_Sajeeb "
				"after RainbowClique::find_cliques" << endl;
	}
	if (f_v) {
		cout << __FILE__ << ":" << __LINE__ << endl;
	}

	// Print the solutions
	if (f_v) {
		cout << "colored_graph::do_Sajeeb Found " << solutions.size()
				<< " solution(s)." << endl;
		for (size_t i=0; i<solutions.size(); ++i) {
			for (size_t j=0; j<solutions[i].size(); ++j) {
				cout << solutions[i][j] << " ";
			}
			cout << endl;
		}
	}

	//this->nb_sol = solutions.size();
#endif



	if (f_v) {
		cout << "colored_graph::do_Sajeeb done" << endl;
	}
}

void colored_graph::do_Sajeeb_black_and_white(
		clique_finder_control *Control,
		std::vector<std::vector<long int> >& solutions,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::do_Sajeeb" << endl;
	}

#if 1
	Graph<long int, int> G (nb_points, nb_colors, nb_colors_per_vertex);
	G.set_vertex_labels(points);
	G.set_vertex_colors(point_color);
	G.set_edge_from_bitvector_adjacency(Bitvec);

    // Call the Rainbow Clique finding algorithm
	KClique::find_cliques(G, solutions, Control->target_size);
	//RainbowClique::find_cliques(G, solutions, 0 /* nb_threads */);

	//this->nb_sol = solutions.size();
#endif



	if (f_v) {
		cout << "colored_graph::do_Sajeeb done" << endl;
	}
}


void colored_graph::all_cliques_weighted_with_two_colors(
		clique_finder_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::all_cliques_weighted_with_two_colors" << endl;
	}

	int *weights;
	int nb_weights;
	int *bounds;
	int nb_bounds;
	int target_value;
	int i;


	Int_vec_scan(Control->weights_string, weights, nb_weights);
	Int_vec_scan(Control->weights_bounds, bounds, nb_bounds);

	if (nb_bounds != nb_weights) {
		cout << "colored_graph::all_cliques_weighted_with_two_colors "
				"nb_bounds != nb_weights" << endl;
		exit(1);
	}

	if (nb_weights != 2) {
		cout << "colored_graph::all_cliques_weighted_with_two_colors "
				"nb_weights != 2" << endl;
		exit(1);
	}
	if (nb_colors < nb_weights + Control->weights_offset) {
		cout << "colored_graph::all_cliques_weighted_with_two_colors "
				"CG->nb_colors < nb_weights + weights_offset" << endl;
		exit(1);
	}

	target_value = Control->weights_total;

	if (f_v) {
		cout << "colored_graph::all_cliques_weighted_with_two_colors "
				"target_value = " << target_value << endl;
		cout << "colored_graph::all_cliques_weighted_with_two_colors "
				"the weights are ";
		Int_vec_print(cout, weights, nb_weights);
		cout << endl;
	}


	solvers::diophant D;
	long int nb_backtrack_nodes;
	int nb_sol;
	int *Sol_weights;
	int j;
	vector<int> res;
	string label;

	label = "weights";

	D.init_partition_problem_with_bounds(
			weights, bounds, nb_weights, target_value,
			verbose_level);


	if (f_v) {
		cout << "colored_graph::all_cliques_weighted_with_two_colors "
				"before D.solve_mckay" << endl;
	}
	D.solve_mckay(label, INT_MAX /* maxresults */,
			nb_backtrack_nodes, nb_sol, 0 /*verbose_level*/);
	if (f_v) {
		cout << "colored_graph::all_cliques_weighted_with_two_colors "
				"after D.solve_mckay" << endl;
	}
	if (f_v) {
		cout << "colored_graph::all_cliques_weighted_with_two_colors "
				"we found " << nb_sol << " solutions for the "
			"weight distribution" << endl;
	}

	Sol_weights = NEW_int(nb_sol * nb_weights);

	for (i = 0; i < D._resultanz; i++) {
		res = D._results.front();
		for (j = 0; j < nb_weights; j++) {
			Sol_weights[i * nb_weights + j] = res[j];
			}
		D._results.pop_front();
		}

	if (f_v) {
		cout << "colored_graph::all_cliques_weighted_with_two_colors "
				"The solutions are:" << endl;
		for (i = 0; i < nb_sol; i++) {
			cout << i << " : ";
			Int_vec_print(cout, Sol_weights + i * nb_weights, nb_weights);
			cout << endl;
		}
	}

	int c1 = Control->weights_offset + 0;
	int c2 = Control->weights_offset + 1;

	if (f_v) {
		cout << "creating subgraph of color " << c1 << ":" << endl;
	}

	colored_graph *subgraph;

	subgraph = subgraph_by_color_classes(c1, verbose_level);

	if (f_v) {
		cout << "The subgraph of color " << c1 << " has size "
				<< subgraph->nb_points << endl;
	}

	int target_depth1;
	int target_depth2;
	int nb_solutions_total;

	//clique_finder_control *Control1;

	//Control1 = NEW_OBJECT(clique_finder_control);

	nb_solutions_total = 0;

	for (i = 0; i < nb_sol; i++) {

		target_depth1 = Sol_weights[i * nb_weights + c1];
		target_depth2 = Sol_weights[i * nb_weights + c2];


		clique_finder_control *Control1;

		Control1 = NEW_OBJECT(clique_finder_control);
		Control1->target_size = target_depth1;

		subgraph->all_cliques_of_size_k_ignore_colors(Control1,
				verbose_level);

		if (f_v) {
			cout << "solution " << i << " / " << nb_sol
					<< " with target_depth = " << target_depth1
					<< " Control1->nb_sol=" << Control1->nb_sol << endl;
		}

		for (j = 0; j < Control1->nb_sol; j++) {




			colored_graph *subgraph2;
			clique_finder_control *Control2;

			Control2 = NEW_OBJECT(clique_finder_control);
			Control2->target_size = target_depth2;

			if (f_v) {
				cout <<  "solution " << i << " / " << nb_sol
						<< ", clique1 " << j << " / " << Control1->nb_sol << ":" << endl;
			}

			subgraph2 = subgraph_by_color_classes_with_condition(
					Control1->Sol + j * target_depth1, target_depth1,
					c2, verbose_level);

			if (f_v) {
				cout << "solution " << i << " / " << nb_sol
						<< ", clique1 " << j << " / " << Control1->nb_sol
						<< ", subgraph2 has " << subgraph2->nb_points << " vertices" << endl;
			}

			subgraph2->all_cliques_of_size_k_ignore_colors(
					Control2,
					verbose_level);

			nb_solutions_total += Control2->nb_sol;

			if (f_v) {
				cout << "solution " << i << " / " << nb_sol << ", "
						"clique1 " << j << " / " << Control1->nb_sol
						<< ", Control2->nb_sol=" << Control2->nb_sol
					<< " nb_solutions_total=" << nb_solutions_total << endl;
			}

			FREE_OBJECT(subgraph2);
			FREE_OBJECT(Control2);
		}
		FREE_OBJECT(Control1);
	}

	if (f_v) {
		cout << "nb_solutions_total=" << nb_solutions_total << endl;
	}

	FREE_int(Sol_weights);

	if (f_v) {
		cout << "colored_graph::all_cliques_weighted_with_two_colors done" << endl;
	}
}


void colored_graph::all_cliques_of_size_k_ignore_colors(
	clique_finder_control *Control,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	clique_finder *CF;

	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors "
				"target_size = " << Control->target_size << endl;
	}
	CF = NEW_OBJECT(clique_finder);

	string dummy;

	dummy.assign("");
	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors "
				"before CF->init" << endl;
	}
	CF->init(
			Control,
			dummy, nb_points,
			false /* f_has_adj_list */, NULL /* int *adj_list_coded */,
			true /* f_has_bitvector */, Bitvec,
			verbose_level - 1);
	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors "
				"after CF->init" << endl;
	}

	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors "
				"before CF->backtrack_search" << endl;
	}
	CF->backtrack_search(0 /* depth */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors "
				"after CF->backtrack_search, nb_sol = " << CF->solutions.size() << endl;
	}

	Control->nb_sol = CF->solutions.size();
	Control->nb_decision_steps = CF->decision_step_counter;

	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors "
				"before CF->get_solutions" << endl;
	}
	long int nb_sol;
	CF->get_solutions(Control->Sol,
			nb_sol, Control->target_size, verbose_level);
	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors "
				"after CF->get_solutions" << endl;
	}
	if (nb_sol != Control->nb_sol) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors "
				"nb_sol != Control->nb_sol" << endl;
		exit(1);
	}

	FREE_OBJECT(CF);
	if (f_v) {
		cout << "colored_graph::all_cliques_of_size_k_ignore_colors done" << endl;
	}
}


void colored_graph::all_rainbow_cliques(
		clique_finder_control *Control,
		std::ostream &ost,
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
	R->search(Control, this, ost, verbose_level - 1);
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


void colored_graph::complement(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;


	if (f_v) {
		cout << "colored_graph::complement" << endl;
		//cout << "nb_points=" << nb_points << endl;
		//print_adjacency_list();
		//Bitvec->print();
	}


	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			if (is_adjacent(i, j)) {
				if (false) {
					cout << "set_adjacency(" << i << "," << j << ",0)" << endl;
				}
				set_adjacency(i, j, 0);
			}
			else {
				if (false) {
					cout << "set_adjacency(" << i << "," << j << ",1)" << endl;
				}
				set_adjacency(i, j, 1);
			}
		}
	}
	if (f_v) {
		//Bitvec->print();
		//print_adjacency_list();
		cout << "colored_graph::complement done" << endl;
	}
}

void colored_graph::distance_2(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k;
	int *M;


	if (f_v) {
		cout << "colored_graph::distance_2" << endl;
		//cout << "nb_points=" << nb_points << endl;
		//print_adjacency_list();
		//Bitvec->print();
	}


	M = NEW_int(nb_points * nb_points);
	Int_vec_zero(M, nb_points * nb_points);

	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			if (is_adjacent(i, j)) {
				continue;
			}
			for (k = 0; k < nb_points; k++) {
				if (k == i) {
					continue;
				}
				if (k == j) {
					continue;
				}
				if (is_adjacent(i, k) && is_adjacent(k, j)) {
					if (false) {
						cout << "set_adjacency(" << i << "," << j << ",0)" << endl;
					}
					M[i * nb_points + j] = 1;
					M[j * nb_points + i] = 1;
					break;
				}
			}
		}
	}
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			if (M[i * nb_points + j]) {
				if (false) {
					cout << "set_adjacency(" << i << "," << j << ",0)" << endl;
				}
				set_adjacency(i, j, 1);
			}
			else {
				if (false) {
					cout << "set_adjacency(" << i << "," << j << ",1)" << endl;
				}
				set_adjacency(i, j, 0);
			}
		}
	}

	FREE_int(M);

	if (f_v) {
		//Bitvec->print();
		//print_adjacency_list();
		cout << "colored_graph::distance_2 done" << endl;
	}
}

void colored_graph::properties(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "colored_graph::properties" << endl;
	}

	int *Degree;
	int i, j;

	Degree = NEW_int(nb_points);
	Int_vec_zero(Degree, nb_points);

	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			if (is_adjacent(i, j)) {
				Degree[i]++;
				Degree[j]++;
			}
		}
	}
	data_structures::tally T;

	T.init(Degree, nb_points, false, 0);
	cout << "Degree type: ";
	T.print_first_tex(true /* f_backwards */);
	cout << endl;

	if (f_v) {
		cout << "colored_graph::properties done" << endl;
	}
}

int colored_graph::test_distinguishing_property(
		long int *set, int sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_distinguishing = false;
	int *neighbor_set;

	if (f_v) {
		cout << "colored_graph::test_distinguishing_property" << endl;
	}

	int N;
	int i, j, n, h;

	N = 1 << sz;

	neighbor_set = NEW_int(N);
	Int_vec_zero(neighbor_set, N);
	for (i = 0; i < nb_points; i++) {
		n = 0;
		for (h = 0; h < sz; h++) {
			n <<= 1;
			j = set[h];
			if (is_adjacent(i, j)) {
				n++;
			}
		}
		if (n == 0) {
			FREE_int(neighbor_set);
			return false;
		}
		if (neighbor_set[n]) {
			FREE_int(neighbor_set);
			return false;
		}
		neighbor_set[n] = true;
	}
	f_distinguishing = true;

	if (f_v) {
		cout << "colored_graph::test_distinguishing_property done" << endl;
	}
	return f_distinguishing;
}

void colored_graph::eigenvalues(
		double *&E, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::eigenvalues" << endl;
	}

	int *Adj;
	int i, j;

	Adj = NEW_int(nb_points * nb_points);
	Int_vec_zero(Adj, nb_points * nb_points);
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			if (is_adjacent(i, j)) {
				Adj[i * nb_points + j] = 1;
				Adj[j * nb_points + i] = 1;
			}
		}
	}


	E = new double[nb_points];

	if (f_v) {
		cout << "colored_graph::eigenvalue Adj=" << endl;
		Int_matrix_print(Adj, nb_points, nb_points);
	}

	l1_interfaces::orbiter_eigenvalues(Adj, nb_points, E, verbose_level - 2);

	FREE_int(Adj);

	//delete [] E;

	if (f_v) {
		cout << "colored_graph::eigenvalues done" << endl;
	}
}

void colored_graph::Laplace_eigenvalues(
		double *&E, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "colored_graph::Laplace_eigenvalues" << endl;
	}

	int *Adj;
	int *D;
	int i, j;

	Adj = NEW_int(nb_points * nb_points);
	D = NEW_int(nb_points);
	Int_vec_zero(Adj, nb_points * nb_points);
	Int_vec_zero(D, nb_points);
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			if (is_adjacent(i, j)) {
				D[i]++;
				D[j]++;
				Adj[i * nb_points + j] = -1;
				Adj[j * nb_points + i] = -1;
			}
		}
	}
	for (i = 0; i < nb_points; i++) {
		Adj[i * nb_points + i] = D[i];
	}

	if (f_v) {
		cout << "colored_graph::Laplace_eigenvalue Adj=" << endl;
		Int_matrix_print(Adj, nb_points, nb_points);
	}

	E = new double[nb_points];

	l1_interfaces::orbiter_eigenvalues(Adj, nb_points, E, verbose_level - 2);

	FREE_int(Adj);
	FREE_int(D);

	//delete [] E;

	if (f_v) {
		cout << "colored_graph::Laplace_eigenvalues done" << endl;
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
	
	orbiter_kernel_system::file_output *FO =
			(orbiter_kernel_system::file_output *)
			CF->call_back_clique_found_data1;
	colored_graph *CG = (colored_graph *) CF->call_back_clique_found_data2;
	//clique_finder *CF = (clique_finder *) FO->user_data;

	if (CG->user_data_size && CG->points) {
		int i, a;
		*FO->fp << CG->user_data_size + CF->Control->target_size;
		for (i = 0; i < CG->user_data_size; i++) {
			*FO->fp << " " << CG->user_data[i];
		}
		for (i = 0; i < CF->Control->target_size; i++) {
			a = CF->current_clique[i];
			*FO->fp << " " << CG->points[a];
		}
		*FO->fp << endl;
	}
	else {
		FO->write_line(CF->Control->target_size,
				CF->current_clique, verbose_level);
	}
}




}}}

