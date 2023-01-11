/*
 * graph_theory_domain.cpp
 *
 *  Created on: Apr 21, 2019
 *      Author: betten
 */

#include "foundations.h"

using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace graph_theory {


graph_theory_domain::graph_theory_domain() {

}

graph_theory_domain::~graph_theory_domain() {

}

void graph_theory_domain::colored_graph_draw(
		std::string &fname_graph,
		graphics::layered_graph_draw_options *Draw_options,
		int f_labels,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	std::string fname_draw;
	colored_graph CG;

	if (f_v) {
		cout << "graph_theory_domain::colored_graph_draw" << endl;
	}

	CG.load(fname_graph, verbose_level - 1);

	fname_draw.assign(CG.fname_base);
	fname_draw.append("_graph");

	if (f_v) {
		cout << "graph_theory_domain::colored_graph_draw before CG.draw_partitioned" << endl;
	}
	CG.draw_partitioned(
			fname_draw,
			Draw_options,
			//fname_draw, xmax_in, ymax_in, xmax_out, ymax_out,
			//FALSE /* f_labels */, scale, line_width,
			f_labels,
			verbose_level);
	if (f_v) {
		cout << "graph_theory_domain::colored_graph_draw after CG.draw_partitioned" << endl;
	}
	if (f_v) {
		cout << "graph_theory_domain::colored_graph_draw done" << endl;
	}
}

void graph_theory_domain::colored_graph_all_cliques(
		clique_finder_control *Control,
		std::string &fname,
		int f_output_solution_raw,
		int f_output_fname, std::string &output_fname,
		int verbose_level)
{
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
		CG.all_rainbow_cliques(Control,
				fp,
				verbose_level - 1);
		if (f_v) {
			cout << "colored_graph_all_cliques "
					"after CG.all_rainbow_cliques" << endl;
		}
		fp << -1 << " " << Control->nb_sol << " " << Control->nb_search_steps << " "
				<< Control->nb_decision_steps << " " << Control->dt << endl;
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
		clique_finder_control *Control,
		long int *list_of_cases, int nb_cases,
		std::string &fname_template, std::string &fname_sol,
		std::string &fname_stats,
		int f_split, int split_r, int split_m,
		int f_prefix, std::string &prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, c;
	int Search_steps = 0, Decision_steps = 0, Nb_sol = 0, Dt = 0;
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

			CG->all_rainbow_cliques(Control,
					fp,
					verbose_level - 1);

			fp << "# end case " << c << " " << Control->nb_sol << " " << Control->nb_search_steps
					<< " " << Control->nb_decision_steps << " " << Control->dt << endl;
			fp_stats << i << "," << c << "," << Control->nb_sol << "," << CG->nb_points
					<< "," << Control->nb_search_steps << "," << Control->nb_decision_steps << "," << Control->dt
					<< endl;
			Search_steps += Control->nb_search_steps;
			Decision_steps += Control->nb_decision_steps;
			Nb_sol += Control->nb_sol;
			Dt += Control->dt;

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


void graph_theory_domain::save_as_colored_graph_easy(std::string &fname_base,
		int n, int *Adj, int verbose_level)
{
	std::string fname;
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "save_as_colored_graph_easy" << endl;
	}
	fname.assign(fname_base);
	fname.append(".colored_graph");

	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);
	CG->init_adjacency_no_colors(n, Adj, fname_base, fname_base, 0 /*verbose_level*/);

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
		data_structures::bitvector *Bitvec,
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
		if (FALSE) {
			cout << "save_colored_graph before writing data" << endl;
		}
		for (i = 0; i < data_sz; i++) {
			fp.write((char*) &data[i], sizeof(long int));
		}
		for (i = 0; i < nb_vertices; i++) {
			if (FALSE) {
				cout << "save_colored_graph before writing vertex " << i << " / " << nb_vertices << endl;
			}
			if (points) {
				fp.write((char*) &points[i], sizeof(long int));
			}
			else {
				a = 0;
				fp.write((char*) &a, sizeof(int));
			}
			for (j = 0; j < nb_colors_per_vertex; j++) {
				fp.write((char*) &point_color[i * nb_colors_per_vertex + j], sizeof(int));
			}
		}
		if (FALSE) {
			cout << "save_colored_graph before writing bitvec" << endl;
		}
		//Bitvec->save(fp);
		fp.write((char*) Bitvec->get_data(), Bitvec->get_allocated_length());
		if (FALSE) {
			cout << "save_colored_graph after writing bitvec" << endl;
		}
	}

	if (f_v) {
		cout << "save_colored_graph done" << endl;
	}
}

void graph_theory_domain::load_colored_graph(std::string &fname,
		int &nb_vertices, int &nb_colors, int &nb_colors_per_vertex,
		long int *&vertex_labels, int *&vertex_colors, long int *&user_data,
		int &user_data_size,
		data_structures::bitvector *&Bitvec,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int L;
	int i, j, a, b;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "graph_theory_domain::load_colored_graph" << endl;
	}

	if (Fio.file_size(fname) <= 0) {
		cout << "graph_theory_domain::load_colored_graph the file " << fname << " does not exist or is empty" << endl;
		exit(1);
	}

	{
		ifstream fp(fname, ios::binary);
		data_structures::sorting Sorting;

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
				cout << "graph_theory_domain::load_colored_graph version=" << b << endl;
			}

			fp.read((char *) &nb_vertices, sizeof(int));
			fp.read((char *) &nb_colors, sizeof(int));
			fp.read((char *) &nb_colors_per_vertex, sizeof(int));
			if (f_v) {
				cout << "graph_theory_domain::load_colored_graph nb_vertices=" << nb_vertices
						<< " nb_colors=" << nb_colors
						<< " nb_colors_per_vertex=" << nb_colors_per_vertex
					<< endl;
				}


			L = ((long int) nb_vertices * (long int) (nb_vertices - 1)) >> 1;

#if 0
			bitvector_length = (L + 7) >> 3;
			if (f_v) {
				cout << "graph_theory_domain::load_colored_graph bitvector_length="
						<< bitvector_length << endl;
				}
#endif

			fp.read((char *) &user_data_size, sizeof(int));
			if (f_v) {
				cout << "graph_theory_domain::load_colored_graph user_data_size="
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
						cout << "graph_theory_domain::load_colored_graph" << endl;
						cout << "vertex_colors[i * nb_colors_per_vertex + j] >= nb_colors" << endl;
						cout << "vertex_colors[i * nb_colors_per_vertex + j]=" << vertex_colors[i * nb_colors_per_vertex + j] << endl;
						cout << "i=" << i << endl;
						cout << "j=" << j << endl;
						cout << "nb_colors=" << nb_colors << endl;
						exit(1);
						}
				}
				Sorting.int_vec_heapsort(vertex_colors + i * nb_colors_per_vertex, nb_colors_per_vertex);
				for (j = 1; j < nb_colors_per_vertex; j++) {
					if (vertex_colors[i * nb_colors_per_vertex + j - 1] == vertex_colors[i * nb_colors_per_vertex + j]) {
						cout << "graph_theory_domain::load_colored_graph repeated color for vertex " << i << " : " << endl;
						Int_vec_print(cout, vertex_colors + i * nb_colors_per_vertex, nb_colors_per_vertex);
						cout << endl;
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
				cout << "graph_theory_domain::load_colored_graph nb_vertices=" << nb_vertices
						<< " nb_colors=" << nb_colors
						<< " nb_colors_per_vertex=" << nb_colors_per_vertex
					<< endl;
				}


			L = ((long int) nb_vertices * (long int) (nb_vertices - 1)) >> 1;

#if 0
			bitvector_length = (L + 7) >> 3;
			if (f_v) {
				cout << "graph_theory_domain::load_colored_graph bitvector_length="
						<< bitvector_length << endl;
				}
#endif

			fp.read((char *) &user_data_size, sizeof(int));
			if (f_v) {
				cout << "graph_theory_domain::load_colored_graph user_data_size="
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
						cout << "graph_theory_domain::load_colored_graph" << endl;
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
		Bitvec = NEW_OBJECT(data_structures::bitvector);
		Bitvec->allocate(L);
		//bitvector_adjacency = NEW_uchar(bitvector_length);
		fp.read((char *) Bitvec->get_data(), Bitvec->get_allocated_length());
	}


	if (f_v) {
		cout << "graph_theory_domain::load_colored_graph done" << endl;
	}
}


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
		cout << "graph_theory_domain::is_association_scheme" << endl;
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

	data_structures::tally Cl;

	Cl.init(M1, N, FALSE, 0);
	nb_colors = Cl.nb_types + 1;
	colors = NEW_int(nb_colors);
	colors[0] = color_graph[0];
	for (i = 0; i < Cl.nb_types; i++) {
		colors[i + 1] = Cl.data_sorted[Cl.type_first[i]];
	}

	if (f_vv) {
		cout << "graph_theory_domain::is_association_scheme "
				"colors (the 0-th color is the diagonal color): ";
		Int_vec_print(cout, colors, nb_colors);
		cout << endl;
	}

	int C = nb_colors;
	int *M = color_graph;
	int pijk, pijk1, u, v, w, u0 = 0, v0 = 0;

	Pijk = NEW_int(C * C * C);
	Int_vec_zero(Pijk, C * C * C);
	for (k = 0; k < C; k++) {
		for (i = 0; i < C; i++) {
			for (j = 0; j < C; j++) {
				pijk = -1;
				for (u = 0; u < n; u++) {
					for (v = 0; v < n; v++) {
						//if (v == u) continue;
						if (M[u * n + v] != colors[k]) {
							continue;
						}
						// now: edge (u,v) is colored k
						pijk1 = 0;
						for (w = 0; w < n; w++) {
							//if (w == u)continue;
							//if (w == v)continue;
							if (M[u * n + w] != colors[i]) {
								continue;
							}
							if (M[v * n + w] != colors[j]) {
								continue;
							}
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

								cout << "graph_theory_domain::is_association_scheme "
										"it is not an association scheme" << endl;
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
		cout << "graph_theory_domain::is_association_scheme "
				"it is an association scheme" << endl;

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
	if (f_v) {
		cout << "graph_theory_domain::is_association_scheme done" << endl;
	}
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
		Int_vec_print_integer_matrix_width(cout, Mtx, C, C, C, 3);
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
		cout << "graph_theory_domain::compute_decomposition_of_graph_wrt_partition" << endl;
		cout << "The partition is:" << endl;
		cout << "first = ";
		Int_vec_print(cout, first, nb_parts);
		cout << endl;
		cout << "len = ";
		Int_vec_print(cout, len, nb_parts);
		cout << endl;
	}
	R = NEW_int(nb_parts * nb_parts);
	Int_vec_zero(R, nb_parts * nb_parts);
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
						cout << "graph_theory_domain::compute_decomposition_of_graph_wrt_partition "
								"not tactical" << endl;
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
		cout << "graph_theory_domain::compute_decomposition_of_graph_wrt_partition done" << endl;
	}
}

void graph_theory_domain::draw_bitmatrix(
		std::string &fname_base,
		graphics::layered_graph_draw_options *Draw_options,
		int f_dots,
		int f_partition,
		int nb_row_parts, int *row_part_first,
		int nb_col_parts, int *col_part_first,
		int f_row_grid, int f_col_grid,
		int f_bitmatrix, data_structures::bitmatrix *Bitmatrix,
		int *M, int m, int n,
		int f_has_labels, int *labels,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::draw_bitmatrix" << endl;
	}
	{
		graphics::mp_graphics G;

		G.init(fname_base, Draw_options, verbose_level - 1);

		G.draw_bitmatrix2(f_dots,
				f_partition,
				nb_row_parts, row_part_first,
				nb_col_parts, col_part_first,
				f_row_grid, f_col_grid,
				f_bitmatrix, Bitmatrix, M, m, n,
				f_has_labels, labels);

		G.finish(cout, TRUE);
	}
	if (f_v) {
		cout << "graph_theory_domain::draw_bitmatrix done" << endl;
	}
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
	number_theory::number_theory_domain NT;

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
	Int_vec_zero(Adj, N * N);

	for (i = 0; i < N; i++) {
		j = (i + 1) % N;
		Adj[i * N + j] = 1;
		Adj[j * N + i] = 1;
	}

	if (f_v) {
		cout << "graph_theory_domain::make_cycle_graph done" << endl;
	}

}

void graph_theory_domain::make_inversion_graph(int *&Adj, int &N,
		int *perm, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_inversion_graph" << endl;
	}
	int i, j, pi, pj;

	N = n;


	Adj = NEW_int(N * N);
	Int_vec_zero(Adj, N * N);

	for (i = 0; i < N; i++) {
		pi = perm[i];
		for (j = i + 1; j < N; j++) {
			pj = perm[j];
			if (pj < pi) {
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
			}
		}
	}

	if (f_v) {
		cout << "graph_theory_domain::make_inversion_graph done" << endl;
	}

}




void graph_theory_domain::make_Hamming_graph(int *&Adj, int &N,
		int n, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Hamming_graph" << endl;
	}
	geometry::geometry_global GG;
	number_theory::number_theory_domain NT;
	coding_theory::coding_theory_domain Coding;
	int *v1;
	int *v2;
	int *v3;
	int i, j, d;

	N = NT.i_power_j(q, n);


	Adj = NEW_int(N * N);
	Int_vec_zero(Adj, N * N);

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
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;
	int *set1;
	int *set2;
	int *set3;
	int i, j, sz;

	N = Combi.int_n_choose_k(n, k);


	Adj = NEW_int(N * N);
	Int_vec_zero(Adj, N * N);

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
		field_theory::finite_field *Fq, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Paley_graph" << endl;
	}

	if (EVEN(Fq->q)) {
		cout << "graph_theory_domain::make_Paley_graph "
				"q must be odd" << endl;
		exit(1);
	}
	if (!DOUBLYEVEN(Fq->q - 1)) {
		cout << "graph_theory_domain::make_Paley_graph "
				"q must be congruent to 1 modulo 4" << endl;
	}

	int i, j, a;

	Adj = NEW_int(Fq->q * Fq->q);
	Int_vec_zero(Adj, Fq->q * Fq->q);

	for (i = 0; i < Fq->q; i++) {
		for (j = i + 1; j < Fq->q; j++) {
			a = Fq->add(i, Fq->negate(j));
			if (Fq->is_square(a)) {
				Adj[i * Fq->q + j] = 1;
				Adj[j * Fq->q + i] = 1;
			}
		}
	}
	N = Fq->q;

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

	field_theory::finite_field *F;
	geometry::grassmann *Gr;
	int n = 4;
	int k = 2;


	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init(q, FALSE /* f_without_tables */, verbose_level);

	Gr = NEW_OBJECT(geometry::grassmann);
	Gr->init(n, k, F, verbose_level);

	Gr->create_Schlaefli_graph(Adj, N, verbose_level);

	FREE_OBJECT(Gr);
	FREE_OBJECT(F);

	if (f_v) {
		cout << "graph_theory_domain::make_Schlaefli_graph done" << endl;
	}
}

void graph_theory_domain::make_Winnie_Li_graph(int *&Adj, int &N,
		field_theory::finite_field *Fq, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Winnie_Li_graph" << endl;
	}

	int i, j, h, u, p, k, co_index, q1, relative_norm;
	int *N1;
	number_theory::number_theory_domain NT;


	p = Fq->p;


	co_index = Fq->e / index;

	if (co_index * index != Fq->e) {
		cout << "graph_theory_domain::make_Winnie_Li_graph "
				"the index has to divide the field degree" << endl;
		exit(1);
	}
	q1 = NT.i_power_j(p, co_index);

	k = (Fq->q - 1) / (q1 - 1);

	if (f_v) {
		cout << "q=" << Fq->q << endl;
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
	for (i = 0; i < Fq->q; i++) {
		if (Fq->power(i, relative_norm) == 1) {
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
		Int_vec_print(cout, N1, k);
		cout << endl;
	}

	Adj = NEW_int(Fq->q * Fq->q);
	for (i = 0; i < Fq->q; i++) {
		for (h = 0; h < k; h++) {
			j = N1[h];
			u = Fq->add(i, j);
			Adj[i * Fq->q + u] = 1;
			Adj[u * Fq->q + i] = 1;
		}
	}

	N = Fq->q;


	FREE_int(N1);


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


	field_theory::finite_field *F;
	geometry::grassmann *Gr;
	int i, j, rr;
	int *M1; // [k * n]
	int *M2; // [k * n]
	int *M; // [2 * k * n]
	combinatorics::combinatorics_domain Combi;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init(q, FALSE /* f_without_tables */, verbose_level);


	Gr = NEW_OBJECT(geometry::grassmann);
	Gr->init(n, k, F, verbose_level);

	N = Combi.generalized_binomial(n, k, q);

	M1 = NEW_int(k * n);
	M2 = NEW_int(k * n);
	M = NEW_int(2 * k * n);

	Adj = NEW_int(N * N);
	Int_vec_zero(Adj, N * N);

	for (i = 0; i < N; i++) {

		Gr->unrank_lint_here(M1, i, 0 /* verbose_level */);

		for (j = i + 1; j < N; j++) {

			Gr->unrank_lint_here(M2, j, 0 /* verbose_level */);

			Int_vec_copy(M1, M, k * n);
			Int_vec_copy(M2, M + k * n, k * n);

			rr = F->Linear_algebra->rank_of_rectangular_matrix(
					M, 2 * k, n, 0 /* verbose_level */);
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

	field_theory::finite_field *F;
	int i, j;
	int n, a, nb_e, nb_inc;
	int c1 = 0, c2 = 0, c3 = 0;
	int *v, *v2;
	int *Gram; // Gram matrix
	geometry::geometry_global Gg;


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

	F = NEW_OBJECT(field_theory::finite_field);

	F->finite_field_init(q, FALSE /* f_without_tables */, verbose_level - 1);
	if (f_v) {
		cout << "graph_theory_domain::make_orthogonal_collinearity_graph field:" << endl;
		F->print();
	}

	if (epsilon == 0) {
		c1 = 1;
	}
	else if (epsilon == -1) {
		F->Linear_algebra->choose_anisotropic_form(c1, c2, c3, verbose_level - 2);
		//cout << "incma.cpp: epsilon == -1, need irreducible polynomial" << endl;
		//exit(1);
	}
	F->Linear_algebra->Gram_matrix(epsilon, n, c1, c2, c3, Gram, verbose_level - 1);
	if (f_v) {
		cout << "graph_theory_domain::make_orthogonal_collinearity_graph "
				"Gram matrix" << endl;
		Int_vec_print_integer_matrix_width(cout, Gram, d, d, d, 2);
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
		F->Orthogonal_indexing->Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i, 0 /* verbose_level */);
		for (j = i + 1; j < N; j++) {
			F->Orthogonal_indexing->Q_epsilon_unrank(v2, 1, epsilon, n, c1, c2, c3, j, 0 /* verbose_level */);
			a = F->Linear_algebra->evaluate_bilinear_form(v, v2, n + 1, Gram);
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
	Int_vec_zero(Adj, N * N);


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

void graph_theory_domain::make_disjoint_sets_graph(int *&Adj, int &N,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_disjoint_sets_graph" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	long int *M;
	int m, n;

	Fio.lint_matrix_read_csv(fname, M, m, n, verbose_level);


	int i, j;
	data_structures::sorting Sorting;

	N = m;

	if (f_v) {
		cout << "graph_theory_domain::make_disjoint_sets_graph N=" << N << endl;
	}

	Adj = NEW_int(N * N);
	Int_vec_zero(Adj, N * N);


	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++) {

			if (Sorting.test_if_sets_are_disjoint(M + i * n, n, M + j * n, n)) {
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
			}
		}
	}

	if (f_v) {
		cout << "graph_theory_domain::make_disjoint_sets_graph done" << endl;
	}

}




void graph_theory_domain::compute_adjacency_matrix(
		int *Table, int nb_sets, int set_size,
		std::string &prefix_for_graph,
		data_structures::bitvector *&B,
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

	B = NEW_OBJECT(data_structures::bitvector);

	B->allocate(N2);

	if (f_v) {
		cout << "graph_theory_domain::compute_adjacency_matrix "
				"after allocating adjacency bitvector" << endl;
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
		orbiter_kernel_system::file_io Fio;

		CG = NEW_OBJECT(colored_graph);
		int *color;

		color = NEW_int(nb_sets);
		Int_vec_zero(color, nb_sets);

		CG->init(nb_sets, 1 /* nb_colors */, 1 /* nb_colors_per_vertex */,
				color, B,
				FALSE,
				prefix_for_graph, prefix_for_graph,
				verbose_level);

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
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "graph_theory_domain::make_graph_of_disjoint_sets_from_rows_of_matrix" << endl;
	}
	Adj = NEW_int(m * m);
	for (i = 0; i < m * m; i++) {
		Adj[i] = 0;
	}

	for (i = 0; i < m; i++) {
		for (j = i + 1; j < m; j++) {
			if (Sorting.test_if_sets_are_disjoint_assuming_sorted(
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

void graph_theory_domain::all_cliques_of_given_size(int *Adj,
		int nb_pts, int clique_sz, int *&Sol, long int &nb_sol,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::all_cliques_of_given_size" << endl;
	}

	int *adj_list_coded;
	int n2;
	int i, j, h;
	clique_finder *C;
	std::string label;
	int f_maxdepth = FALSE;
	int maxdepth = 0;


	label.assign("all_cliques_of_given_size");

	n2 = (nb_pts * (nb_pts - 1)) >> 1;
	adj_list_coded = NEW_int(n2);
	h = 0;
	cout << "graph_theory_domain::all_cliques_of_given_size: "
			"computing adj_list_coded" << endl;
	for (i = 0; i < nb_pts; i++) {
		for (j = i + 1; j < nb_pts; j++) {
			adj_list_coded[h++] = Adj[i * nb_pts + j];
		}
	}

	clique_finder_control *Control;

	Control = NEW_OBJECT(clique_finder_control);
	Control->target_size = clique_sz;
	Control->f_maxdepth = f_maxdepth;
	Control->maxdepth = maxdepth;
	Control->f_store_solutions = TRUE;

	C = NEW_OBJECT(clique_finder);

	if (f_v) {
		cout << "graph_theory_domain::all_cliques_of_given_size: "
				"before C->init" << endl;
	}
	C->init(Control,
			label, nb_pts,
			TRUE, adj_list_coded,
			FALSE, NULL,
			verbose_level);

	C->backtrack_search(0 /* depth */, 0 /* verbose_level */);

	if (f_v) {
		cout << "graph_theory_domain::all_cliques_of_given_size "
				"done with search, "
				"we found " << C->solutions.size() << " solutions" << endl;
	}

	int sz;
	if (f_v) {
		cout << "graph_theory_domain::all_cliques_of_given_size "
				"before C->get_solutions" << endl;
	}
	C->get_solutions(Sol, nb_sol, sz, verbose_level);
	if (f_v) {
		cout << "graph_theory_domain::all_cliques_of_given_size "
				"after C->get_solutions" << endl;
	}
	if (sz != clique_sz) {
		cout << "graph_theory_domain::all_cliques_of_given_size "
				"sz != clique_sz" << endl;
		exit(1);
	}
	FREE_OBJECT(C);
	FREE_OBJECT(Control);
	FREE_int(adj_list_coded);
	if (f_v) {
		cout << "graph_theory_domain::all_cliques_of_given_size done" << endl;
	}
}

void graph_theory_domain::eigenvalues(graph_theory::colored_graph *CG, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::eigenvalues" << endl;
	}
	double *E;
	double *L;
	int i;

	CG->eigenvalues(E, verbose_level - 2);
	CG->Laplace_eigenvalues(L, verbose_level - 2);

	cout << "The eigenvalues are:" << endl;
	for (i = 0; i < CG->nb_points; i++) {
		cout << i << " : " << E[i] << endl;
	}

	double energy = 0;
	for (i = 0; i < CG->nb_points; i++) {
		energy += ABS(E[i]);
	}
	cout << "The energy is " << energy << endl;

	cout << "The Laplace eigenvalues are:" << endl;
	for (i = 0; i < CG->nb_points; i++) {
		cout << i << " : " << L[i] << endl;
	}


	{
		string fname;

		string title, author, extra_praeamble;
		char str[1000];

		snprintf(str, 1000, "Eigenvalues of %s", CG->label_tex.c_str());
		title.assign(str);


		fname.assign(CG->label);
		fname.append("_eigenvalues.tex");

		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface Li;

			Li.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "graph_theory_domain::perform_activity before report" << endl;
			}
			//report(ost, verbose_level);

			ost << "$$" << endl;
			ost << "\\begin{array}{|r|r|r|}" << endl;
			ost << "\\hline" << endl;
			ost << " i  & \\lambda_i & \\theta_i \\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\hline" << endl;
			for (i = 0; i < CG->nb_points; i++) {
				ost << i;
				ost << " & ";
				ost << E[CG->nb_points - 1 - i];
				ost << " & ";
				ost << L[CG->nb_points - 1 - i];
				ost << "\\\\" << endl;
				ost << "\\hline" << endl;
			}
			ost << "\\end{array}" << endl;
			ost << "$$" << endl;

			ost << "The energy is " << energy << "\\\\" << endl;
			ost << "Eigenvalues: $\\lambda_i$\\\\" << endl;
			ost << "Laplace eigenvalues: $\\theta_i$\\\\" << endl;

			if (f_v) {
				cout << "graph_theory_domain::perform_activity after report" << endl;
			}


			Li.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "graph_theory_domain::perform_activity "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	delete [] E;

	if (f_v) {
		cout << "graph_theory_domain::eigenvalues done" << endl;
	}

}






}}}

