/*
 * graph_theory_domain.cpp
 *
 *  Created on: Apr 21, 2019
 *      Author: betten
 */

#include "foundations.h"

using namespace std;

namespace orbiter {
namespace foundations {

graph_theory_domain::graph_theory_domain() {

}

graph_theory_domain::~graph_theory_domain() {

}

void graph_theory_domain::colored_graph_draw(std::string &fname, int xmax_in,
		int ymax_in, int xmax_out, int ymax_out, double scale,
		double line_width, int verbose_level) {
	int f_v = (verbose_level >= 1);
	std::string fname_draw;
	colored_graph CG;

	if (f_v) {
		cout << "colored_graph_draw" << endl;
	}
	CG.load(fname, verbose_level - 1);
	fname_draw.assign(CG.fname_base);
	fname_draw.append("_graph");
	if (f_v) {
		cout << "colored_graph_draw before CG.draw_partitioned" << endl;
	}
	CG.draw_partitioned(fname_draw, xmax_in, ymax_in, xmax_out, ymax_out,
			FALSE /* f_labels */, scale, line_width, verbose_level);
	if (f_v) {
		cout << "colored_graph_draw after CG.draw_partitioned" << endl;
	}
	if (f_v) {
		cout << "colored_graph_draw done" << endl;
	}
}

void graph_theory_domain::colored_graph_all_cliques(std::string &fname,
		int f_output_solution_raw, int f_output_fname, std::string &output_fname,
		int f_maxdepth, int maxdepth, int f_restrictions, int *restrictions,
		int f_tree, int f_decision_nodes_only, std::string &fname_tree,
		int print_interval, unsigned long int &search_steps,
		unsigned long int &decision_steps, int &nb_sol, int &dt,
		int verbose_level) {
	int f_v = (verbose_level >= 1);
	colored_graph CG;
	std::string fname_sol;
	std::string fname_success;

	if (f_v) {
		cout << "colored_graph_all_cliques" << endl;
	}
	CG.load(fname, verbose_level - 1);
	if (f_output_fname) {
		fname_sol.assign(output_fname);
		fname_success.assign(output_fname);
		fname_success.append(".success");
	}
	else {
		fname_sol.assign(CG.fname_base);
		fname_sol.append("_sol.txt");
		fname_success.assign(CG.fname_base);
		fname_success.append("_sol.success");
	}

	//CG.print();

	{
		ofstream fp(fname_sol.c_str());

		if (f_v) {
			cout << "colored_graph_all_cliques "
					"before CG.all_rainbow_cliques" << endl;
		}
		CG.all_rainbow_cliques(&fp, f_output_solution_raw, f_maxdepth, maxdepth,
				f_restrictions, restrictions, f_tree, f_decision_nodes_only,
				fname_tree, print_interval, search_steps, decision_steps,
				nb_sol, dt, verbose_level - 1);
		if (f_v) {
			cout << "colored_graph_all_cliques "
					"after CG.all_rainbow_cliques" << endl;
		}
		fp << -1 << " " << nb_sol << " " << search_steps << " "
				<< decision_steps << " " << dt << endl;
	}
	{
		ofstream fp(fname_success);
		fp << "success" << endl;
	}
	if (f_v) {
		cout << "colored_graph_all_cliques done" << endl;
	}
}

void graph_theory_domain::colored_graph_all_cliques_list_of_cases(
		long int *list_of_cases, int nb_cases, int f_output_solution_raw,
		std::string &fname_template, std::string &fname_sol,
		std::string &fname_stats,
		int f_split, int split_r, int split_m,
		int f_maxdepth, int maxdepth,
		int f_prefix, std::string &prefix,
		int print_interval, int verbose_level) {
	int f_v = (verbose_level >= 1);
	int i, c;
	int Search_steps = 0, Decision_steps = 0, Nb_sol = 0, Dt = 0;
	unsigned long int search_steps, decision_steps;
	int nb_sol, dt;
	std::string fname;
	char fname_tmp[2000];

	if (f_v) {
		cout << "colored_graph_all_cliques_list_of_cases" << endl;
	}
	{
		ofstream fp(fname_sol.c_str());
		ofstream fp_stats(fname_stats.c_str());

		fp_stats << "i,Case,Nb_sol,Nb_vertices,search_steps,"
				"decision_steps,dt" << endl;
		for (i = 0; i < nb_cases; i++) {

			if (f_split && ((i % split_m) == split_r)) {
				continue;
			}
			colored_graph *CG;

			CG = NEW_OBJECT(colored_graph);

			c = list_of_cases[i];
			if (f_v) {
				cout << "colored_graph_all_cliques_list_of_cases case " << i
						<< " / " << nb_cases << " which is " << c << endl;
			}
			snprintf(fname_tmp, 2000, fname_template.c_str(), c);
			if (f_prefix) {
				fname.assign(prefix);
				fname.append(fname_tmp);
			}
			else {
				fname.assign(fname_tmp);
			}
			CG->load(fname, verbose_level - 2);

			//CG->print();

			fp << "# start case " << c << endl;

			string dummy;

			CG->all_rainbow_cliques(&fp, f_output_solution_raw, f_maxdepth,
					maxdepth,
					FALSE /* f_restrictions */, NULL /* restrictions */,
					FALSE /* f_tree */, FALSE /* f_decision_nodes_only */,
					dummy /* fname_tree */, print_interval, search_steps,
					decision_steps, nb_sol, dt, verbose_level - 1);
			fp << "# end case " << c << " " << nb_sol << " " << search_steps
					<< " " << decision_steps << " " << dt << endl;
			fp_stats << i << "," << c << "," << nb_sol << "," << CG->nb_points
					<< "," << search_steps << "," << decision_steps << "," << dt
					<< endl;
			Search_steps += search_steps;
			Decision_steps += decision_steps;
			Nb_sol += nb_sol;
			Dt += dt;

			FREE_OBJECT(CG);
		}
		fp << -1 << " " << Nb_sol << " " << Search_steps << " "
				<< Decision_steps << " " << Dt << endl;
		fp_stats << "END" << endl;
	}
	if (f_v) {
		cout << "colored_graph_all_cliques_list_of_cases "
				"done Nb_sol=" << Nb_sol << endl;
	}
}

void graph_theory_domain::colored_graph_all_cliques_list_of_files(int nb_cases,
		int *Case_number, const char **Case_fname, int f_output_solution_raw,
		const char *fname_sol, const char *fname_stats, int f_maxdepth,
		int maxdepth, int f_prefix, const char *prefix, int print_interval,
		int verbose_level) {
	int f_v = (verbose_level >= 1);
	int i, c;
	int Search_steps = 0, Decision_steps = 0, Nb_sol = 0, Dt = 0;
	unsigned long int search_steps, decision_steps;
	int nb_sol, dt;
	file_io Fio;

	if (f_v) {
		cout << "colored_graph_all_cliques_list_of_files" << endl;
	}
	{
		ofstream fp(fname_sol);
		ofstream fp_stats(fname_stats);

		fp_stats << "i,Case,Nb_sol,Nb_vertices,search_steps,"
				"decision_steps,dt" << endl;
		for (i = 0; i < nb_cases; i++) {

			colored_graph *CG;
			std::string fname;

			CG = NEW_OBJECT(colored_graph);

			c = Case_number[i];
			fname.assign(Case_fname[i]);
			//fname = Case_fname[i];

			if (f_v) {
				cout << "colored_graph_all_cliques_list_of_files case " << i
						<< " / " << nb_cases << " which is " << c << " in file "
						<< fname << endl;
			}

			if (Fio.file_size(fname) <= 0) {
				cout << "colored_graph_all_cliques_list_of_files file " << fname
						<< " does not exist" << endl;
				exit(1);
			}
			CG->load(fname, verbose_level - 2);

			//CG->print();

			fp << "# start case " << c << endl;

			string dummy;

			CG->all_rainbow_cliques(&fp, f_output_solution_raw, f_maxdepth,
					maxdepth,
					FALSE /* f_restrictions */, NULL /* restrictions */,
					FALSE /* f_tree */, FALSE /* f_decision_nodes_only */,
					dummy /* fname_tree */, print_interval, search_steps,
					decision_steps, nb_sol, dt, verbose_level - 1);
			fp << "# end case " << c << " " << nb_sol << " " << search_steps
					<< " " << decision_steps << " " << dt << endl;
			fp_stats << i << "," << c << "," << nb_sol << "," << CG->nb_points
					<< "," << search_steps << "," << decision_steps << "," << dt
					<< endl;
			Search_steps += search_steps;
			Decision_steps += decision_steps;
			Nb_sol += nb_sol;
			Dt += dt;

			FREE_OBJECT(CG);
		}
		fp << -1 << " " << Nb_sol << " " << Search_steps << " "
				<< Decision_steps << " " << Dt << endl;
		fp_stats << "END" << endl;
	}
	if (f_v) {
		cout << "colored_graph_all_cliques_list_of_files "
				"done Nb_sol=" << Nb_sol << endl;
	}
}

#if 0
int graph_theory_domain::colored_graph_all_rainbow_cliques_nonrecursive(
	const char *fname,
	int &nb_backtrack_nodes,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	colored_graph CG;
	int nb_sol;

	if (f_v) {
		cout << "colored_graph_all_rainbow_cliques_"
				"nonrecursive" << endl;
		}
	CG.load(fname, verbose_level - 1);
	//CG.print();

	{
	if (f_v) {
		cout << "colored_graph_all_cliques "
				"before CG.all_rainbow_cliques" << endl;
		}
	nb_sol = CG.rainbow_cliques_nonrecursive(
			nb_backtrack_nodes, verbose_level - 1);
	}
	if (f_v) {
		cout << "colored_graph_all_rainbow_cliques_"
				"nonrecursive done" << endl;
		}
	return nb_sol;
}
#endif

void graph_theory_domain::save_as_colored_graph_easy(std::string &fname_base,
		int n, int *Adj, int verbose_level)
{
	std::string fname;
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "save_as_colored_graph_easy" << endl;
	}
	fname.assign(fname_base);
	fname.append(".colored_graph");

	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency_no_colors(n, Adj, 0 /*verbose_level*/);

	CG->save(fname, verbose_level);

	FREE_OBJECT(CG);

	if (f_v) {
		cout << "save_as_colored_graph_easy Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}
}

void graph_theory_domain::save_colored_graph(std::string &fname,
		int nb_vertices, int nb_colors, int nb_colors_per_vertex,
		long int *points, int *point_color,
		long int *data, int data_sz,
		bitvector *Bitvec,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b;

	if (f_v) {
		cout << "save_colored_graph" << endl;
		cout << "save_colored_graph fname=" << fname << endl;
		cout << "save_colored_graph nb_vertices=" << nb_vertices << endl;
		cout << "save_colored_graph nb_colors=" << nb_colors << endl;
		cout << "save_colored_graph nb_colors_per_vertex="
				<< nb_colors_per_vertex << endl;
		//cout << "save_colored_graph bitvector_length=" << bitvector_length << endl;
	}

	{
		ofstream fp(fname, ios::binary);

		a = -1; // marker for new file format
		b = 1; // file format version number

		fp.write((char*) &a, sizeof(int));
		fp.write((char*) &b, sizeof(int));
		fp.write((char*) &nb_vertices, sizeof(int));
		fp.write((char*) &nb_colors, sizeof(int));
		fp.write((char*) &nb_colors_per_vertex, sizeof(int));
		fp.write((char*) &data_sz, sizeof(int));
		for (i = 0; i < data_sz; i++) {
			fp.write((char*) &data[i], sizeof(long int));
		}
		for (i = 0; i < nb_vertices; i++) {
			if (points) {
				fp.write((char*) &points[i], sizeof(long int));
			}
			else {
				a = 0;
				fp.write((char*) &a, sizeof(int));
			}
			for (j = 0; j < nb_colors_per_vertex; j++) {
				fp.write((char*) &point_color[i * nb_colors_per_vertex + j],
						sizeof(int));
			}
		}
		//Bitvec->save(fp);
		fp.write((char*) Bitvec->get_data(), Bitvec->get_allocated_length());
	}

	if (f_v) {
		cout << "save_colored_graph done" << endl;
	}
}

void graph_theory_domain::load_colored_graph(std::string &fname,
		int &nb_vertices, int &nb_colors, int &nb_colors_per_vertex,
		long int *&vertex_labels, int *&vertex_colors, long int *&user_data,
		int &user_data_size,
		bitvector *&Bitvec,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int L;
	int i, j, a, b;
	file_io Fio;

	if (f_v) {
		cout << "graph_theory_domain::load_colored_graph" << endl;
	}

	if (Fio.file_size(fname) <= 0) {
		cout << "graph_theory_domain::load_colored_graph the file " << fname << " does not exist or is empty" << endl;
		exit(1);
	}

	{
		ifstream fp(fname, ios::binary);

		fp.read((char *) &a, sizeof(int));
		if (a == -1) {


			if (f_v) {
				cout << "graph_theory_domain::load_colored_graph detected new file format" << endl;
			}

			// new file format
			// the new format allows for multiple colors per vertex
			// (must be constant across all vertices)
			// The nb_colors_per_vertex tells how many colors each vertex has.
			// So, vertex_colors[] is now a two-dimensional array:
			// vertex_colors[nb_vertices * nb_colors_per_vertex]
			// Also, vertex_labels[] is now long int.

			// read the version number:

			fp.read((char *) &b, sizeof(int));
			if (f_v) {
				cout << "load_colored_graph version=" << b << endl;
			}

			fp.read((char *) &nb_vertices, sizeof(int));
			fp.read((char *) &nb_colors, sizeof(int));
			fp.read((char *) &nb_colors_per_vertex, sizeof(int));
			if (f_v) {
				cout << "load_colored_graph nb_vertices=" << nb_vertices
						<< " nb_colors=" << nb_colors
						<< " nb_colors_per_vertex=" << nb_colors_per_vertex
					<< endl;
				}


			L = ((long int) nb_vertices * (long int) (nb_vertices - 1)) >> 1;

#if 0
			bitvector_length = (L + 7) >> 3;
			if (f_v) {
				cout << "load_colored_graph bitvector_length="
						<< bitvector_length << endl;
				}
#endif

			fp.read((char *) &user_data_size, sizeof(int));
			if (f_v) {
				cout << "load_colored_graph user_data_size="
						<< user_data_size << endl;
				}
			user_data = NEW_lint(user_data_size);

			for (i = 0; i < user_data_size; i++) {
				fp.read((char *) &user_data[i], sizeof(long int));
				}

			vertex_labels = NEW_lint(nb_vertices);
			vertex_colors = NEW_int(nb_vertices * nb_colors_per_vertex);

			for (i = 0; i < nb_vertices; i++) {
				fp.read((char *) &vertex_labels[i], sizeof(long int));
				for (j = 0; j < nb_colors_per_vertex; j++) {
					fp.read((char *) &vertex_colors[i * nb_colors_per_vertex + j], sizeof(int));
					if (vertex_colors[i * nb_colors_per_vertex + j] >= nb_colors) {
						cout << "load_colored_graph" << endl;
						cout << "vertex_colors[i * nb_colors_per_vertex + j] >= nb_colors" << endl;
						cout << "vertex_colors[i * nb_colors_per_vertex + j]=" << vertex_colors[i * nb_colors_per_vertex + j] << endl;
						cout << "i=" << i << endl;
						cout << "j=" << j << endl;
						cout << "nb_colors=" << nb_colors << endl;
						exit(1);
						}
				}
			}
		}
		else {

			if (f_v) {
				cout << "graph_theory_domain::load_colored_graph detected old file format in file " << fname << endl;
			}
			// old file format is still supported:

			//cout << "graph_theory_domain::load_colored_graph old file format no longer supported" << endl;
			//exit(1);
			cout << "graph_theory_domain::load_colored_graph old file format detected, using compatibility mode" << endl;
			nb_vertices = a;
			fp.read((char *) &nb_colors, sizeof(int));
			nb_colors_per_vertex = 1;
			if (f_v) {
				cout << "load_colored_graph nb_vertices=" << nb_vertices
						<< " nb_colors=" << nb_colors
						<< " nb_colors_per_vertex=" << nb_colors_per_vertex
					<< endl;
				}


			L = ((long int) nb_vertices * (long int) (nb_vertices - 1)) >> 1;

#if 0
			bitvector_length = (L + 7) >> 3;
			if (f_v) {
				cout << "load_colored_graph bitvector_length="
						<< bitvector_length << endl;
				}
#endif

			fp.read((char *) &user_data_size, sizeof(int));
			if (f_v) {
				cout << "load_colored_graph user_data_size="
						<< user_data_size << endl;
				}
			user_data = NEW_lint(user_data_size);

			for (i = 0; i < user_data_size; i++) {
				fp.read((char *) &a, sizeof(int));
				user_data[i] = a;
				}

			vertex_labels = NEW_lint(nb_vertices);
			vertex_colors = NEW_int(nb_vertices * nb_colors_per_vertex);

			for (i = 0; i < nb_vertices; i++) {
				fp.read((char *) &a, sizeof(int));
				vertex_labels[i] = a;
				for (j = 0; j < nb_colors_per_vertex; j++) {
					fp.read((char *) &vertex_colors[i * nb_colors_per_vertex + j], sizeof(int));
					if (vertex_colors[i * nb_colors_per_vertex + j] >= nb_colors) {
						cout << "load_colored_graph" << endl;
						cout << "vertex_colors[i * nb_colors_per_vertex + j] >= nb_colors" << endl;
						cout << "vertex_colors[i * nb_colors_per_vertex + j]=" << vertex_colors[i * nb_colors_per_vertex + j] << endl;
						cout << "i=" << i << endl;
						cout << "j=" << j << endl;
						cout << "nb_colors=" << nb_colors << endl;
						exit(1);
						}
				}
			}
		}

		if (f_v) {
			cout << "graph_theory_domain::load_colored_graph before allocating bitvector_adjacency" << endl;
			}
		Bitvec = NEW_OBJECT(bitvector);
		Bitvec->allocate(L);
		//bitvector_adjacency = NEW_uchar(bitvector_length);
		fp.read((char *) Bitvec->get_data(), Bitvec->get_allocated_length());
	}


	if (f_v) {
		cout << "graph_theory_domain::load_colored_graph done" << endl;
	}
}

#if 0
void graph_theory_domain::write_colored_graph(ofstream &ost, char *label,
		int point_offset, int nb_points,
		int f_has_adjacency_matrix, int *Adj,
		int f_has_adjacency_list, int *adj_list, int f_has_bitvector,
		uchar *bitvector_adjacency, int f_has_is_adjacent_callback,
		int (*is_adjacent_callback)(int i, int j, void *data),
		void *is_adjacent_callback_data, int f_colors, int nb_colors,
		int *point_color, int f_point_labels, long int *point_label) {
	long int i, j, h, d;
	int aij = 0;
	int w;
	number_theory_domain NT;
	combinatorics_domain Combi;

	cout << "write_graph " << label << " with " << nb_points
			<< " points, point_offset=" << point_offset << endl;
	w = NT.int_log10(nb_points);
	cout << "w=" << w << endl;
	ost << "<GRAPH label=\"" << label << "\" num_pts=\"" << nb_points
			<< "\" f_has_colors=\"" << f_colors << "\" num_colors=\""
			<< nb_colors << "\" point_offset=\"" << point_offset
			<< "\" f_point_labels=\"" << f_point_labels << "\">" << endl;
	for (i = 0; i < nb_points; i++) {
		d = 0;
		for (j = 0; j < nb_points; j++) {
			if (j == i) {
				continue;
			}
			if (f_has_adjacency_matrix) {
				aij = Adj[i * nb_points + j];
			}
			else if (f_has_adjacency_list) {
				if (i < j) {
					h = Combi.ij2k_lint(i, j, nb_points);
				}
				else {
					h = Combi.ij2k_lint(j, i, nb_points);
				}
				aij = adj_list[h];
			}
			else if (f_has_bitvector) {
				if (i < j) {
					h = Combi.ij2k_lint(i, j, nb_points);
				}
				else {
					h = Combi.ij2k_lint(j, i, nb_points);
				}
				aij = bitvector_s_i(bitvector_adjacency, h);
			}
			else if (f_has_is_adjacent_callback) {
				aij = (*is_adjacent_callback)(i, j, is_adjacent_callback_data);
			}
			else {
				cout << "write_colored_graph cannot "
						"determine adjacency" << endl;
			}

			if (aij) {
				d++;
			}
		}
		ost << setw(w) << i + point_offset << " " << setw(w) << d << " ";
		for (j = 0; j < nb_points; j++) {
			if (j == i) {
				continue;
			}
			if (f_has_adjacency_matrix) {
				aij = Adj[i * nb_points + j];
			}
			else if (f_has_adjacency_list) {
				if (i < j) {
					h = Combi.ij2k_lint(i, j, nb_points);
				}
				else {
					h = Combi.ij2k_lint(j, i, nb_points);
				}
				aij = adj_list[h];
			}
			else if (f_has_bitvector) {
				if (i < j) {
					h = Combi.ij2k_lint(i, j, nb_points);
				}
				else {
					h = Combi.ij2k_lint(j, i, nb_points);
				}
				aij = bitvector_s_i(bitvector_adjacency, h);
			}
			else if (f_has_is_adjacent_callback) {
				aij = (*is_adjacent_callback)(i, j, is_adjacent_callback_data);
			}
			else {
				cout << "write_colored_graph cannot "
						"determine adjacency" << endl;
			}
			if (aij) {
				ost << setw(w) << j + point_offset << " ";
			}
		}
		ost << endl;

	}

	if (f_colors) {
		ost << endl;
		for (j = 0; j < nb_colors; j++) {
			d = 0;
			for (i = 0; i < nb_points; i++) {
				if (point_color[i] == j)
					d++;
			}
			ost << setw(w) << j + point_offset << " " << setw(w) << d << " ";
			for (i = 0; i < nb_points; i++) {
				if (point_color[i] == j)
					ost << setw(w) << i + point_offset << " ";
			}
			ost << endl;
		}
	}

	if (f_point_labels) {
		ost << endl;
		for (i = 0; i < nb_points; i++) {
			ost << setw(w) << i + point_offset << " " << setw(6)
					<< point_label[i] << endl;
		}
	}

	ost << "</GRAPH>" << endl;

}
#endif

int graph_theory_domain::is_association_scheme(int *color_graph, int n,
		int *&Pijk, int *&colors, int &nb_colors, int verbose_level)
// color_graph[n * n]
// added Dec 22, 2010.
		{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int N;
	int *M1;
	int k, i, j;
	int ret = FALSE;

	if (f_v) {
		cout << "is_association_scheme" << endl;
	}
	N = (n * (n - 1)) / 2;
	M1 = NEW_int(N);
	k = 0;
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			M1[k++] = color_graph[i * n + j];
		}
	}
	if (k != N) {
		cout << "N=" << N << endl;
		cout << "k=" << k << endl;
		exit(1);
	}

	tally Cl;

	Cl.init(M1, N, FALSE, 0);
	nb_colors = Cl.nb_types + 1;
	colors = NEW_int(nb_colors);
	colors[0] = color_graph[0];
	for (i = 0; i < Cl.nb_types; i++) {
		colors[i + 1] = Cl.data_sorted[Cl.type_first[i]];
	}

	if (f_vv) {
		cout << "colors (the 0-th color is the diagonal color): ";
		Orbiter->Int_vec.print(cout, colors, nb_colors);
		cout << endl;
	}

	int C = nb_colors;
	int *M = color_graph;
	int pijk, pijk1, u, v, w, u0 = 0, v0 = 0;

	Pijk = NEW_int(C * C * C);
	Orbiter->Int_vec.zero(Pijk, C * C * C);
	for (k = 0; k < C; k++) {
		for (i = 0; i < C; i++) {
			for (j = 0; j < C; j++) {
				pijk = -1;
				for (u = 0; u < n; u++) {
					for (v = 0; v < n; v++) {
						//if (v == u) continue;
						if (M[u * n + v] != colors[k])
							continue;
						// now: edge (u,v) is colored k
						pijk1 = 0;
						for (w = 0; w < n; w++) {
							//if (w == u)continue;
							//if (w == v)continue;
							if (M[u * n + w] != colors[i])
								continue;
							if (M[v * n + w] != colors[j])
								continue;
							//cout << "i=" << i << " j=" << j << " k=" << k
							//<< " u=" << u << " v=" << v << " w=" << w
							//<< " increasing pijk" << endl;
							pijk1++;
						} // next w
						//cout << "i=" << i << " j=" << j << " k=" << k
						//<< " u=" << u << " v=" << v
						//<< " pijk1=" << pijk1 << endl;
						if (pijk == -1) {
							pijk = pijk1;
							u0 = u;
							v0 = v;
							//cout << "u=" << u << " v=" << v
							//<< " p_{" << i << "," << j << ","
							//<< k << "}="
							//<< Pijk[i * C * C + j * C + k] << endl;
						}
						else {
							if (pijk1 != pijk) {
								//FREE_int(Pijk);
								//FREE_int(colors);

								cout << "not an association scheme" << endl;
								cout << "k=" << k << endl;
								cout << "i=" << i << endl;
								cout << "j=" << j << endl;
								cout << "u0=" << u0 << endl;
								cout << "v0=" << v0 << endl;
								cout << "pijk=" << pijk << endl;
								cout << "u=" << u << endl;
								cout << "v=" << v << endl;
								cout << "pijk1=" << pijk1 << endl;
								//exit(1);

								goto done;
							}
						}
					} // next v
				} // next u
				Pijk[i * C * C + j * C + k] = pijk;
			} // next j
		} // next i
	} // next k

	ret = TRUE;

	if (f_v) {
		cout << "it is an association scheme" << endl;

		if (f_v) {
			print_Pijk(Pijk, C);
		}

		if (C == 3 && colors[1] == 0 && colors[2] == 1) {
			int k, lambda, mu;

			k = Pijk[2 * C * C + 2 * C + 0]; // p220;
			lambda = Pijk[2 * C * C + 2 * C + 2]; // p222;
			mu = Pijk[2 * C * C + 2 * C + 1]; // p221;
			cout << "it is an SRG(" << n << "," << k << "," << lambda << ","
					<< mu << ")" << endl;
		}

	}

	done:
	FREE_int(M1);
	return ret;
}

void graph_theory_domain::print_Pijk(int *Pijk, int nb_colors) {
	int i, j, k;
	int C = nb_colors;

	for (k = 0; k < C; k++) {
		int *Mtx;

		Mtx = NEW_int(C * C);
		for (i = 0; i < C; i++) {
			for (j = 0; j < C; j++) {
				Mtx[i * C + j] = Pijk[i * C * C + j * C + k];
			}
		}
		cout << "P^{(" << k << ")}=(p_{i,j," << k << "})_{i,j}:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, Mtx, C, C, C, 3);
		FREE_int(Mtx);
	}
}

void graph_theory_domain::compute_decomposition_of_graph_wrt_partition(
		int *Adj,
		int N, int *first, int *len, int nb_parts, int *&R,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int I, J, i, j, f1, l1, f2, l2, r0 = 0, r;

	if (f_v) {
		cout << "compute_decomposition_of_graph_wrt_partition" << endl;
		cout << "The partition is:" << endl;
		cout << "first = ";
		Orbiter->Int_vec.print(cout, first, nb_parts);
		cout << endl;
		cout << "len = ";
		Orbiter->Int_vec.print(cout, len, nb_parts);
		cout << endl;
	}
	R = NEW_int(nb_parts * nb_parts);
	Orbiter->Int_vec.zero(R, nb_parts * nb_parts);
	for (I = 0; I < nb_parts; I++) {
		f1 = first[I];
		l1 = len[I];
		for (J = 0; J < nb_parts; J++) {
			f2 = first[J];
			l2 = len[J];
			for (i = 0; i < l1; i++) {
				r = 0;
				for (j = 0; j < l2; j++) {
					if (Adj[(f1 + i) * N + f2 + j]) {
						r++;
					}
				}
				if (i == 0) {
					r0 = r;
				}
				else {
					if (r0 != r) {
						cout << "compute_decomposition_of_graph_"
								"wrt_partition not tactical" << endl;
						cout << "I=" << I << endl;
						cout << "J=" << J << endl;
						cout << "r0=" << r0 << endl;
						cout << "r=" << r << endl;
						exit(1);
					}
				}
			}
			R[I * nb_parts + J] = r0;
		}
	}
	if (f_v) {
		cout << "compute_decomposition_of_graph_wrt_partition done" << endl;
	}
}

void graph_theory_domain::draw_bitmatrix(
		std::string &fname_base,
		int f_dots,
		int f_partition, int nb_row_parts, int *row_part_first,
		int nb_col_parts, int *col_part_first, int f_row_grid, int f_col_grid,
		int f_bitmatrix, bitmatrix *Bitmatrix,
		int *M, int m, int n, int xmax_in,
		int ymax_in, int xmax_out, int ymax_out, double scale,
		double line_width, int f_has_labels, int *labels,
		int verbose_level)
{
	mp_graphics G;
	std::string fname_base2;
	std::string fname;
	int f_embedded = TRUE;
	int f_sideways = FALSE;
	file_io Fio;

	fname_base2.assign(fname_base);
	fname.assign(fname_base2);
	fname.append(".mp");
	{
		G.setup(fname_base2, 0, 0, xmax_in /* ONE_MILLION */,
				ymax_in /* ONE_MILLION */, xmax_out, ymax_out, f_embedded,
				f_sideways, scale, line_width, verbose_level - 1);

		//G.frame(0.05);

		G.draw_bitmatrix2(f_dots, f_partition, nb_row_parts, row_part_first,
				nb_col_parts, col_part_first, f_row_grid, f_col_grid,
				f_bitmatrix, Bitmatrix, M, m, n, xmax_in, ymax_in, f_has_labels,
				labels);

		G.finish(cout, TRUE);
	}
	cout << "draw_it written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
}


void graph_theory_domain::list_parameters_of_SRG(int v_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::list_parameters_of_SRG" << endl;
	}

	int v, v2, k, lambda, mu, cnt = 0;
	int top, top2, bottom, b, tb;
	int i, f, g, r, s;
	number_theory_domain NT;

	for (v = 2; v <= v_max; v++) {
		v2 = v >> 1;
		for (k = 1; k <= v2; k++) {
			for (lambda = 0; lambda <= k; lambda++) {
				for (mu = 1; mu <= k; mu++) {
					if (k * (k - lambda - 1) != mu * (v - k - 1)) {
						continue;
						}
					top = (v - 1) * (mu - lambda) - 2 * k;
					top2 = top * top;
					bottom = (mu - lambda) * (mu - lambda) + 4 * (k - mu);
					cnt++;
					cout << "cnt=" << cnt << " v=" << v << " k=" << k
							<< " lambda=" << lambda << " mu=" << mu
							<< " top=" << top << " bottom=" << bottom << endl;
					if (top2 % bottom) {
						cout << "is ruled out by integrality condition" << endl;
						continue;
						}

					int nb;
					int *primes, *exponents;
					nb = NT.factor_int(bottom, primes, exponents);
					for (i = 0; i < nb; i++) {
						if (ODD(exponents[i])) {
							break;
							}
						}
					if (i < nb) {
						cout << "bottom is not a square" << endl;
						continue;
						}
					for (i = 0; i < nb; i++) {
						exponents[i] >>= 1;
						}
					b = 1;
					for (i = 0; i < nb; i++) {
						b *= NT.i_power_j(primes[i], exponents[i]);
						}
					cout << "b=" << b << endl;
					tb = top / b;
					cout << "tb=" << tb << endl;
					if (ODD(v - 1 + tb)) {
						cout << "is ruled out by integrality condition (2)" << endl;
						continue;
						}
					if (ODD(v - 1 - tb)) {
						cout << "is ruled out by integrality condition (3)" << endl;
						continue;
						}
					f = (v - 1 + tb) >> 1;
					g = (v - 1 - tb) >> 1;
					if (ODD(lambda - mu + b)) {
						cout << "r is not integral, skip" << endl;
						continue;
						}
					if (ODD(lambda - mu - b)) {
						cout << "r is not integral, skip" << endl;
						continue;
						}
					r = (lambda - mu + b) >> 1;
					s = (lambda - mu - b) >> 1;
					cout << "f=" << f << " g=" << g << " r=" << r << " s=" << s << endl;

					int L1, R1, L2, R2;

					L1 = (r + 1) * (k + r + 2 * r * s);
					R1 = (k + r) * (s + 1) * (s + 1);
					L2 = (s + 1) * (k + s + 2 * r * s);
					R2 = (k + s) * (r + 1) * (r + 1);

					if (L1 > R1) {
						cout << "is ruled out by Krein condition (1)" << endl;
						continue;
						}
					if (L2 > R2) {
						cout << "is ruled out by Krein condition (2)" << endl;
						continue;
						}
					}
				}
			}
		}

	if (f_v) {
		cout << "graph_theory_domain::list_parameters_of_SRG done" << endl;
	}
}

void graph_theory_domain::make_cycle_graph(int *&Adj, int &N,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_cycle_graph" << endl;
	}
	int i, j;

	N = n;


	Adj = NEW_int(N * N);
	Orbiter->Int_vec.zero(Adj, N * N);

	for (i = 0; i < N; i++) {
		j = (i + 1) % N;
		Adj[i * N + j] = 1;
		Adj[j * N + i] = 1;
	}

	if (f_v) {
		cout << "graph_theory_domain::make_cycle_graph done" << endl;
	}

}




void graph_theory_domain::make_Hamming_graph(int *&Adj, int &N,
		int n, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Hamming_graph" << endl;
	}
	geometry_global GG;
	number_theory_domain NT;
	coding_theory_domain Coding;
	int *v1;
	int *v2;
	int *v3;
	int i, j, d;

	N = NT.i_power_j(q, n);


	Adj = NEW_int(N * N);
	Orbiter->Int_vec.zero(Adj, N * N);

	v1 = NEW_int(n);
	v2 = NEW_int(n);
	v3 = NEW_int(n);

	for (i = 0; i < N; i++) {
		GG.AG_element_unrank(q, v1, 1, n, i);
		for (j = i + 1; j < N; j++) {
			GG.AG_element_unrank(q, v2, 1, n, j);

			d = Coding.Hamming_distance(v1, v2, n);
			if (d == 1) {
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
			}
		}
	}

	FREE_int(v1);
	FREE_int(v2);
	FREE_int(v3);

	if (f_v) {
		cout << "graph_theory_domain::make_Hamming_graph done" << endl;
	}

}


void graph_theory_domain::make_Johnson_graph(int *&Adj, int &N,
		int n, int k, int s, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Johnson_graph" << endl;
	}
	combinatorics_domain Combi;
	sorting Sorting;
	int *set1;
	int *set2;
	int *set3;
	int i, j, sz;

	N = Combi.int_n_choose_k(n, k);


	Adj = NEW_int(N * N);
	Orbiter->Int_vec.zero(Adj, N * N);

	set1 = NEW_int(k);
	set2 = NEW_int(k);
	set3 = NEW_int(k);

	for (i = 0; i < N; i++) {
		Combi.unrank_k_subset(i, set1, n, k);
		for (j = i + 1; j < N; j++) {
			Combi.unrank_k_subset(j, set2, n, k);

			Sorting.int_vec_intersect_sorted_vectors(set1, k, set2, k, set3, sz);
			if (sz == s) {
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
			}
		}
	}

	FREE_int(set1);
	FREE_int(set2);
	FREE_int(set3);

	if (f_v) {
		cout << "graph_theory_domain::make_Johnson_graph done" << endl;
	}

}

void graph_theory_domain::make_Paley_graph(int *&Adj, int &N,
		int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Paley_graph" << endl;
	}

	if (EVEN(q)) {
		cout << "graph_theory_domain::make_Paley_graph q must be odd" << endl;
		exit(1);
	}
	if (!DOUBLYEVEN(q - 1)) {
		cout << "graph_theory_domain::make_Paley_graph q must be congruent to 1 modulo 4" << endl;
	}

	finite_field *F;
	int *f_is_square;
	int i, j, a;

	F = NEW_OBJECT(finite_field);
	F->finite_field_init(q, verbose_level);

	f_is_square = NEW_int(q);
	Orbiter->Int_vec.zero(f_is_square, q);

	for (i = 0; i < q; i++) {
		j = F->mult(i, i);
		f_is_square[j] = TRUE;
	}

	Adj = NEW_int(q * q);
	Orbiter->Int_vec.zero(Adj, q * q);

	for (i = 0; i < q; i++) {
		for (j = i + 1; j < q; j++) {
			a = F->add(i, F->negate(j));
			if (f_is_square[a]) {
				Adj[i * q + j] = 1;
				Adj[j * q + i] = 1;
			}
		}
	}
	N = q;

	FREE_OBJECT(F);
	FREE_int(f_is_square);

	if (f_v) {
		cout << "graph_theory_domain::make_Paley_graph done" << endl;
	}
}

void graph_theory_domain::make_Schlaefli_graph(int *&Adj, int &N,
		int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Schlaefli_graph" << endl;
	}

	finite_field *F;
	grassmann *Gr;
	int n = 4;
	int k = 2;


	F = NEW_OBJECT(finite_field);
	F->finite_field_init(q, verbose_level);

	Gr = NEW_OBJECT(grassmann);
	Gr->init(n, k, F, verbose_level);

	Gr->create_Schlaefli_graph(Adj, N, verbose_level);

	FREE_OBJECT(Gr);
	FREE_OBJECT(F);

	if (f_v) {
		cout << "graph_theory_domain::make_Schlaefli_graph done" << endl;
	}
}

void graph_theory_domain::make_Winnie_Li_graph(int *&Adj, int &N,
		int q, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Winnie_Li_graph" << endl;
	}

	finite_field *F;
	int i, j, h, u, p, k, co_index, q1, relative_norm;
	int *N1;
	number_theory_domain NT;


	F = NEW_OBJECT(finite_field);
	F->finite_field_init(q, verbose_level - 1);
	p = F->p;

#if 0
	if (!f_index) {
		index = F->e;
		}
#endif

	co_index = F->e / index;

	if (co_index * index != F->e) {
		cout << "graph_theory_domain::make_Winnie_Li_graph "
				"the index has to divide the field degree" << endl;
		exit(1);
	}
	q1 = NT.i_power_j(p, co_index);

	k = (q - 1) / (q1 - 1);

	if (f_v) {
		cout << "q=" << q << endl;
		cout << "index=" << index << endl;
		cout << "co_index=" << co_index << endl;
		cout << "q1=" << q1 << endl;
		cout << "k=" << k << endl;
	}

	relative_norm = 0;
	j = 1;
	for (i = 0; i < index; i++) {
		relative_norm += j;
		j *= q1;
	}
	if (f_v) {
		cout << "graph_theory_domain::make_Winnie_Li_graph "
				"relative_norm=" << relative_norm << endl;
	}

	N1 = NEW_int(k);
	j = 0;
	for (i = 0; i < q; i++) {
		if (F->power(i, relative_norm) == 1) {
			N1[j++] = i;
		}
	}
	if (j != k) {
		cout << "graph_theory_domain::make_Winnie_Li_graph "
				"j != k" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "graph_theory_domain::make_Winnie_Li_graph "
				"found " << k << " norm-one elements:" << endl;
		Orbiter->Int_vec.print(cout, N1, k);
		cout << endl;
	}

	Adj = NEW_int(q * q);
	for (i = 0; i < q; i++) {
		for (h = 0; h < k; h++) {
			j = N1[h];
			u = F->add(i, j);
			Adj[i * q + u] = 1;
			Adj[u * q + i] = 1;
		}
	}

	N = q;


	FREE_int(N1);
	FREE_OBJECT(F);


	if (f_v) {
		cout << "graph_theory_domain::make_Winnie_Li_graph done" << endl;
	}
}

void graph_theory_domain::make_Grassmann_graph(int *&Adj, int &N,
		int n, int k, int q, int r, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Grassmann_graph" << endl;
	}


	finite_field *F;
	grassmann *Gr;
	int i, j, rr;
	int *M1; // [k * n]
	int *M2; // [k * n]
	int *M; // [2 * k * n]
	combinatorics_domain Combi;

	F = NEW_OBJECT(finite_field);
	F->finite_field_init(q, verbose_level);


	Gr = NEW_OBJECT(grassmann);
	Gr->init(n, k, F, verbose_level);

	N = Combi.generalized_binomial(n, k, q);

	M1 = NEW_int(k * n);
	M2 = NEW_int(k * n);
	M = NEW_int(2 * k * n);

	Adj = NEW_int(N * N);
	Orbiter->Int_vec.zero(Adj, N * N);

	for (i = 0; i < N; i++) {

		Gr->unrank_lint_here(M1, i, 0 /* verbose_level */);

		for (j = i + 1; j < N; j++) {

			Gr->unrank_lint_here(M2, j, 0 /* verbose_level */);

			Orbiter->Int_vec.copy(M1, M, k * n);
			Orbiter->Int_vec.copy(M2, M + k * n, k * n);

			rr = F->rank_of_rectangular_matrix(M, 2 * k, n, 0 /* verbose_level */);
			if (rr == r) {
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
			}
		}
	}



	FREE_int(M1);
	FREE_int(M2);
	FREE_int(M);
	FREE_OBJECT(Gr);
	FREE_OBJECT(F);

	if (f_v) {
		cout << "graph_theory_domain::make_Grassmann_graph done" << endl;
	}
}


void graph_theory_domain::make_orthogonal_collinearity_graph(int *&Adj, int &N,
		int epsilon, int d, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_orthogonal_collinearity_graph" << endl;
	}

	finite_field *F;
	int i, j;
	int n, a, nb_e, nb_inc;
	int c1 = 0, c2 = 0, c3 = 0;
	int *v, *v2;
	int *Gram; // Gram matrix
	geometry_global Gg;


	n = d - 1; // projective dimension

	v = NEW_int(d);
	v2 = NEW_int(d);
	Gram = NEW_int(d * d);

	if (f_v) {
		cout << "graph_theory_domain::make_orthogonal_collinearity_graph "
				"epsilon=" << epsilon << " n=" << n << " q=" << q << endl;
	}

	N = Gg.nb_pts_Qepsilon(epsilon, n, q);

	if (f_v) {
		cout << "graph_theory_domain::make_orthogonal_collinearity_graph "
				"number of points = " << N << endl;
	}

	F = NEW_OBJECT(finite_field);

	F->finite_field_init(q, verbose_level - 1);
	F->print();

	if (epsilon == 0) {
		c1 = 1;
	}
	else if (epsilon == -1) {
		F->choose_anisotropic_form(c1, c2, c3, verbose_level - 2);
		//cout << "incma.cpp: epsilon == -1, need irreducible polynomial" << endl;
		//exit(1);
	}
	F->Gram_matrix(epsilon, n, c1, c2, c3, Gram, verbose_level - 1);
	if (f_v) {
		cout << "graph_theory_domain::make_orthogonal_collinearity_graph "
				"Gram matrix" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout, Gram, d, d, d, 2);
	}

#if 0
	if (f_list_points) {
		for (i = 0; i < N; i++) {
			F->Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i, 0 /* verbose_level */);
			cout << i << " : ";
			int_vec_print(cout, v, n + 1);
			j = F->Q_epsilon_rank(v, 1, epsilon, n, c1, c2, c3, 0 /* verbose_level */);
			cout << " : " << j << endl;

			}
		}
#endif


	if (f_v) {
		cout << "graph_theory_domain::make_orthogonal_collinearity_graph "
				"allocating adjacency matrix" << endl;
	}
	Adj = NEW_int(N * N);
	if (f_v) {
		cout << "graph_theory_domain::make_orthogonal_collinearity_graph "
				"allocating adjacency matrix was successful" << endl;
	}
	nb_e = 0;
	nb_inc = 0;
	for (i = 0; i < N; i++) {
		F->Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i, 0 /* verbose_level */);
		for (j = i + 1; j < N; j++) {
			F->Q_epsilon_unrank(v2, 1, epsilon, n, c1, c2, c3, j, 0 /* verbose_level */);
			a = F->evaluate_bilinear_form(v, v2, n + 1, Gram);
			if (a == 0) {
				nb_e++;
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
			}
			else {
				Adj[i * N + j] = 0;
				Adj[j * N + i] = 0;
				nb_inc++;
			}
		}
		Adj[i * N + i] = 0;
	}
	if (f_v) {
		cout << "graph_theory_domain::make_orthogonal_collinearity_graph "
				"The adjacency matrix of the collinearity graph has been computed" << endl;
	}


	FREE_int(v);
	FREE_int(v2);
	FREE_int(Gram);
	FREE_OBJECT(F);

	if (f_v) {
		cout << "graph_theory_domain::make_orthogonal_collinearity_graph done" << endl;
	}
}

void graph_theory_domain::make_non_attacking_queens_graph(int *&Adj, int &N,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_non_attacking_queens_graph" << endl;
	}
	int i1, j1, n1;
	int i2, j2, n2;

	N = n * n;


	Adj = NEW_int(N * N);
	Orbiter->Int_vec.zero(Adj, N * N);


	for (n1 = 0; n1 < N; n1++) {
		i1 = n1 / n;
		j1 = n1 % n;
		for (n2 = n1 + 1; n2 < N; n2++) {
			i2 = n2 / n;
			j2 = n2 % n;
			if (i2 == i1) {
				continue;
			}
			if (j2 == j1) {
				continue;
			}
			if (j2 - j1 == i2 - i1) {
				continue;
			}
			if (j2 - j1 == i1 - i2) {
				continue;
			}
			Adj[n1 * N + n2] = 1;
			Adj[n2 * N + n1] = 1;
		}
	}

	if (f_v) {
		cout << "graph_theory_domain::make_non_attacking_queens_graph done" << endl;
	}

}



void graph_theory_domain::compute_adjacency_matrix(
		int *Table, int nb_sets, int set_size,
		std::string &prefix_for_graph,
		bitvector *&B,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k, cnt, N2, N2_100;

	if (f_v) {
		cout << "graph_theory_domain::compute_adjacency_matrix" << endl;
	}

	N2 = (nb_sets * (nb_sets - 1)) >> 1;
	if (f_v) {
		cout << "graph_theory_domain::compute_adjacency_matrix N2=" << N2 << endl;
	}
	N2_100 = (N2 / 100) + 1;

	B = NEW_OBJECT(bitvector);

	B->allocate(N2);

	if (f_v) {
		cout << "graph_theory_domain::compute_adjacency_matrix after allocating adjacency bitvector" << endl;
		cout << "computing adjacency matrix:" << endl;
	}
	k = 0;
	cnt = 0;
	for (i = 0; i < nb_sets; i++) {
		for (j = i + 1; j < nb_sets; j++) {

			int *p, *q;
			int u, v;

			p = Table + i * set_size;
			q = Table + j * set_size;
			u = v = 0;
			while (u + v < 2 * set_size) {
				if (p[u] == q[v]) {
					break;
				}
				if (u == set_size) {
					v++;
				}
				else if (v == set_size) {
					u++;
				}
				else if (p[u] < q[v]) {
					u++;
				}
				else {
					v++;
				}
			}
			if (u + v < 2 * set_size) {
				B->m_i(k, 0);

			}
			else {
				B->m_i(k, 1);
				cnt++;
			}

			k++;
			if ((k % N2_100) == 0) {
				cout << "i=" << i << " j=" << j << " " << k / N2_100 << "% done, k=" << k << endl;
			}
#if 0
			if ((k & ((1 << 21) - 1)) == 0) {
				cout << "i=" << i << " j=" << j << " k=" << k
						<< " / " << N2 << endl;
				}
#endif
		}
	}


	if (f_v) {
		cout << "graph_theory_domain::compute_adjacency_matrix making a graph" << endl;
	}

	{
	colored_graph *CG;
	std::string fname;
	file_io Fio;

	CG = NEW_OBJECT(colored_graph);
	int *color;

	color = NEW_int(nb_sets);
	Orbiter->Int_vec.zero(color, nb_sets);

	CG->init(nb_sets, 1 /* nb_colors */, 1 /* nb_colors_per_vertex */,
			color, B,
			FALSE, verbose_level);

	fname.assign(prefix_for_graph);
	fname.append("_disjointness.colored_graph");

	CG->save(fname, verbose_level);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	FREE_int(color);
	FREE_OBJECT(CG);
	}


	if (f_v) {
		cout << "graph_theory_domain::compute_adjacency_matrix done" << endl;
		}
}

void graph_theory_domain::make_graph_of_disjoint_sets_from_rows_of_matrix(
	int *M, int m, int n,
	int *&Adj, int verbose_level)
// assumes that the rows are sorted
{
	int f_v = (verbose_level >= 1);
	int i, j, a;

	if (f_v) {
		cout << "graph_theory_domain::make_graph_of_disjoint_sets_from_rows_of_matrix" << endl;
	}
	Adj = NEW_int(m * m);
	for (i = 0; i < m * m; i++) {
		Adj[i] = 0;
	}

	for (i = 0; i < m; i++) {
		for (j = i + 1; j < m; j++) {
			if (test_if_sets_are_disjoint_assuming_sorted(
				M + i * n, M + j * n, n, n)) {
				a = 1;
			}
			else {
				a = 0;
			}
			Adj[i * m + j] = a;
			Adj[j * m + i] = a;
		}
	}
	if (f_v) {
		cout << "graph_theory_domain::make_graph_of_disjoint_sets_from_rows_of_matrix done" << endl;
	}
}





}}
