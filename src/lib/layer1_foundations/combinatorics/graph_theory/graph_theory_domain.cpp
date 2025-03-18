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
namespace combinatorics {
namespace graph_theory {


graph_theory_domain::graph_theory_domain()
{
	Record_birth();

}

graph_theory_domain::~graph_theory_domain()
{
	Record_death();

}

void graph_theory_domain::colored_graph_draw(
		std::string &fname_graph,
		other::graphics::layered_graph_draw_options *Draw_options,
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

	fname_draw = CG.fname_base + "_graph";

	if (f_v) {
		cout << "graph_theory_domain::colored_graph_draw "
				"before CG.draw_partitioned" << endl;
	}
	CG.draw_partitioned(
			fname_draw,
			Draw_options,
			f_labels,
			verbose_level);
	if (f_v) {
		cout << "graph_theory_domain::colored_graph_draw "
				"after CG.draw_partitioned" << endl;
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
		fname_sol = output_fname;
		fname_success = output_fname + ".success";
	}
	else {
		fname_sol = CG.fname_base + "_sol.txt";
		fname_success = CG.fname_base + "_sol.success";
	}

	//CG.print();

	{

		if (f_v) {
			cout << "colored_graph_all_cliques "
					"before CG.Colored_graph_cliques->all_rainbow_cliques" << endl;
		}
		CG.Colored_graph_cliques->all_rainbow_cliques(
				Control,
				//fp,
				verbose_level - 1);
		if (f_v) {
			cout << "colored_graph_all_cliques "
					"after CG.Colored_graph_cliques->all_rainbow_cliques" << endl;
		}
	}
	{
		ofstream fp(fname_sol);

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

#if 0
void graph_theory_domain::colored_graph_all_cliques_list_of_cases(
		clique_finder_control *Control,
		long int *list_of_cases, int nb_cases,
		std::string &fname_template,
		std::string &fname_sol,
		std::string &fname_stats,
		int f_split, int split_r, int split_m,
		int f_prefix, std::string &prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, c;
	int Search_steps = 0, Decision_steps = 0, Nb_sol = 0, Dt = 0;
	std::string fname;
	data_structures::string_tools ST;



	if (f_v) {
		cout << "colored_graph_all_cliques_list_of_cases" << endl;
	}
	{
		ofstream fp(fname_sol);
		ofstream fp_stats(fname_stats);

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

			string fname_tmp;

			fname_tmp = ST.printf_d(fname_template, c);

			if (f_prefix) {
				fname = prefix + fname_tmp;
			}
			else {
				fname = fname_tmp;
			}
			CG->load(fname, verbose_level - 2);

			//CG->print();

			fp << "# start case " << c << endl;

			string dummy;

			CG->all_rainbow_cliques(Control,
					//fp,
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
#endif

int graph_theory_domain::is_association_scheme(
		int *color_graph, int n,
		int *&Pijk, int *&colors, int &nb_colors,
		int verbose_level)
// color_graph[n * n]
// added Dec 22, 2010.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int N;
	int *M1;
	int k, i, j;
	int ret = false;

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

	other::data_structures::tally Cl;

	Cl.init(M1, N, false, 0);
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

	ret = true;

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

void graph_theory_domain::print_Pijk(
		int *Pijk, int nb_colors) {
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
		int *Adj, int N,
		int *first, int *len, int nb_parts, int *&R,
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
		other::graphics::layered_graph_draw_options *Draw_options,
		int f_dots,
		int f_partition,
		int nb_row_parts, int *row_part_first,
		int nb_col_parts, int *col_part_first,
		int f_row_grid, int f_col_grid,
		int f_bitmatrix, other::data_structures::bitmatrix *Bitmatrix,
		int *M, int m, int n,
		int f_has_labels, int *labels,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::draw_bitmatrix" << endl;
	}
	{
		other::graphics::mp_graphics G;

		G.init(fname_base, Draw_options, verbose_level - 1);

		G.draw_bitmatrix2(f_dots,
				f_partition,
				nb_row_parts, row_part_first,
				nb_col_parts, col_part_first,
				f_row_grid, f_col_grid,
				f_bitmatrix, Bitmatrix, M, m, n,
				f_has_labels, labels);

		G.finish(cout, true);
	}
	if (f_v) {
		cout << "graph_theory_domain::draw_bitmatrix done" << endl;
	}
}


void graph_theory_domain::list_parameters_of_SRG(
		int v_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::list_parameters_of_SRG" << endl;
	}

	int v, v2, k, lambda, mu, cnt = 0;
	int top, top2, bottom, b, tb;
	int i, f, g, r, s;
	algebra::number_theory::number_theory_domain NT;

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
						cout << "is ruled out by the integrality condition" << endl;
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
						cout << "is ruled out by the integrality condition (2)" << endl;
						continue;
					}
					if (ODD(v - 1 - tb)) {
						cout << "is ruled out by the integrality condition (3)" << endl;
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
						cout << "is ruled out by the Krein condition (1)" << endl;
						continue;
					}
					if (L2 > R2) {
						cout << "is ruled out by the Krein condition (2)" << endl;
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

void graph_theory_domain::load_dimacs(
		int *&Adj, int &N,
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::load_dimacs" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	int nb_V;
	int i, j, h;
	std::vector<std::vector<int>> Edges;

	if (f_v) {
		cout << "graph_theory_domain::load_dimacs "
				"before Fio.read_graph_dimacs_format" << endl;
	}
	Fio.read_graph_dimacs_format(
			fname,
			nb_V, Edges,
			verbose_level);
	if (f_v) {
		cout << "graph_theory_domain::load_dimacs "
				"after Fio.read_graph_dimacs_format" << endl;
	}

	N = nb_V;
	if (f_v) {
		cout << "graph_theory_domain::load_dimacs "
				"N=" << N << endl;
	}
	if (f_v) {
		cout << "graph_theory_domain::load_dimacs "
				"nb_E=" << Edges.size() << endl;
	}
	Adj = NEW_int(nb_V * nb_V);
	Int_vec_zero(Adj, nb_V * nb_V);

	for (h = 0; h < Edges.size(); h++) {
		i = Edges[h][0];
		j = Edges[h][1];
		if (false) {
			cout << "graph_theory_domain::load_dimacs "
					"edge " << h << " is " << i << " to " << j << endl;
		}
		Adj[i * nb_V + j] = 1;
		Adj[j * nb_V + i] = 1;
	}

	if (f_v) {
		cout << "graph_theory_domain::load_dimacs done" << endl;
	}
}

void graph_theory_domain::make_cycle_graph(
		int *&Adj, int &N,
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

void graph_theory_domain::make_inversion_graph(
		int *&Adj, int &N,
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




void graph_theory_domain::make_Hamming_graph(
		int *&Adj, int &N,
		int n, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Hamming_graph" << endl;
	}
	geometry::other_geometry::geometry_global GG;
	algebra::number_theory::number_theory_domain NT;
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


void graph_theory_domain::make_Johnson_graph(
		int *&Adj, int &N,
		int n, int k, int s, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Johnson_graph" << endl;
	}
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;
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

void graph_theory_domain::make_Paley_graph(
		int *&Adj, int &N,
		algebra::field_theory::finite_field *Fq, int verbose_level)
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

void graph_theory_domain::make_Schlaefli_graph(
		int *&Adj, int &N,
		algebra::field_theory::finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Schlaefli_graph" << endl;
	}

	geometry::projective_geometry::grassmann *Gr;
	int n = 4;
	int k = 2;

	Gr = NEW_OBJECT(geometry::projective_geometry::grassmann);
	Gr->init(n, k, F, verbose_level);

	Gr->create_Schlaefli_graph(Adj, N, verbose_level);

	FREE_OBJECT(Gr);

	if (f_v) {
		cout << "graph_theory_domain::make_Schlaefli_graph done" << endl;
	}
}

void graph_theory_domain::make_Winnie_Li_graph(
		int *&Adj, int &N,
		algebra::field_theory::finite_field *Fq,
		int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Winnie_Li_graph" << endl;
	}

	int i, j, h, u, p, k, co_index, q1, relative_norm;
	int *N1;
	algebra::number_theory::number_theory_domain NT;


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

void graph_theory_domain::make_Grassmann_graph(
		int *&Adj, int &N,
		int n, int k,
		algebra::field_theory::finite_field *F,
		int r, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Grassmann_graph" << endl;
	}


	geometry::projective_geometry::grassmann *Gr;
	int i, j, rr;
	int *M1; // [k * n]
	int *M2; // [k * n]
	int *M; // [2 * k * n]
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	Gr = NEW_OBJECT(geometry::projective_geometry::grassmann);
	Gr->init(n, k, F, verbose_level);

	N = Combi.generalized_binomial(n, k, F->q);

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

	if (f_v) {
		cout << "graph_theory_domain::make_Grassmann_graph done" << endl;
	}
}


void graph_theory_domain::make_tritangent_plane_disjointness_graph(
		int *&Adj, int &N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_tritangent_plane_disjointness_graph" << endl;
	}

	geometry::algebraic_geometry::surface_domain *Surf;
	algebra::field_theory::finite_field *F;

	F = NEW_OBJECT(algebra::field_theory::finite_field);
	Surf = NEW_OBJECT(geometry::algebraic_geometry::surface_domain);

	if (f_v) {
		cout << "graph_theory_domain::make_tritangent_plane_disjointness_graph "
				"before F->finite_field_init_small_order" << endl;
	}
	F->finite_field_init_small_order(
			5,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);
	if (f_v) {
		cout << "graph_theory_domain::make_tritangent_plane_disjointness_graph "
				"after F->finite_field_init_small_order" << endl;
	}
	Surf->init_surface_domain(F, verbose_level);

	if (f_v) {
		cout << "graph_theory_domain::make_tritangent_plane_disjointness_graph "
				"before Surf->Schlaefli->Schlaefli_tritangent_planes->make_tritangent_plane_disjointness_graph" << endl;
	}
	Surf->Schlaefli->Schlaefli_tritangent_planes->make_tritangent_plane_disjointness_graph(
			Adj, N, verbose_level);
	if (f_v) {
		cout << "graph_theory_domain::make_tritangent_plane_disjointness_graph "
				"after Surf->Schlaefli->Schlaefli_tritangent_planes->make_tritangent_plane_disjointness_graph" << endl;
	}

	FREE_OBJECT(Surf);
	FREE_OBJECT(F);

	if (f_v) {
		cout << "graph_theory_domain::make_tritangent_plane_disjointness_graph done" << endl;
	}
}

void graph_theory_domain::make_trihedral_pair_disjointness_graph(
		int *&Adj, int &N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_trihedral_pair_disjointness_graph" << endl;
	}

	geometry::algebraic_geometry::surface_domain *Surf;
	algebra::field_theory::finite_field *F;

	F = NEW_OBJECT(algebra::field_theory::finite_field);
	Surf = NEW_OBJECT(geometry::algebraic_geometry::surface_domain);

	F->finite_field_init_small_order(
			5,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);
	Surf->init_surface_domain(F, verbose_level);

	Surf->Schlaefli->Schlaefli_trihedral_pairs->make_trihedral_pair_disjointness_graph(
			Adj, verbose_level);
	N = 120;

	FREE_OBJECT(Surf);
	FREE_OBJECT(F);

	if (f_v) {
		cout << "graph_theory_domain::make_trihedral_pair_disjointness_graph done" << endl;
	}
}

void graph_theory_domain::make_non_attacking_queens_graph(
		int *&Adj, int &N,
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

void graph_theory_domain::make_disjoint_sets_graph(
		int *&Adj, int &N,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_disjoint_sets_graph" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	long int *M;
	int m, n;

	Fio.Csv_file_support->lint_matrix_read_csv(
			fname, M, m, n, verbose_level);

	N = m;

	int i, j;
	other::data_structures::sorting Sorting;

#if 0
	for (i = 0; i < N; i++) {
		if (!Sorting.lint_vec_is_sorted(M + i * n, N)) {
			cout << "graph_theory_domain::make_disjoint_sets_graph the set is not sorted" << endl;
			exit(1);
		}
	}
#endif

	if (f_v) {
		cout << "graph_theory_domain::make_disjoint_sets_graph "
				"N=" << N << " n=" << n << endl;
	}

	if (f_v) {
		cout << "graph_theory_domain::make_disjoint_sets_graph "
				"before sorting" << endl;
	}
	for (i = 0; i < N; i++) {
		if ((i % 1000) == 0) {
			cout << i << endl;
		}
		Sorting.lint_vec_heapsort(M + i * n, n);
	}
	if (f_v) {
		cout << "graph_theory_domain::make_disjoint_sets_graph after sorting" << endl;
	}



	Adj = NEW_int(N * N);
	Int_vec_zero(Adj, N * N);


	if (f_v) {
		cout << "graph_theory_domain::make_disjoint_sets_graph "
				"computing adjacency matrix" << endl;
	}
	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++) {

			if (Sorting.test_if_sets_are_disjoint_assuming_sorted_lint(
					M + i * n, M + j * n, n, n)) {
				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;
			}
		}
	}

	FREE_lint(M);

	if (f_v) {
		cout << "graph_theory_domain::make_disjoint_sets_graph done" << endl;
	}

}

#if 0
static const char *Neumaier_graph_25_blocks =
		"0,j,j,j,j,z,z,z,z, "
		"jt,B,E1,E2,E3,E1b,E2b,E3b,Z, "
		"jt,E1t,B,I,I,IA,A,I,B, "
		"jt,E2t,I,B,I,I,IA,A,IAt, "
		"jt,E3t,I,I,B,A,I,IA,IA, "
		"zt,E1bt,IAt,I,At,B,I,I,IAt, "
		"zt,E2bt,At,IAt,I,I,B,I,IA, "
		"zt,E3bt,I,At,IAt,I,I,B,B, "
		"zt,Z,B,IA,IAt,IA,IAt,B,Z";
#endif

static const char *Neumaier_graph_25_blocks_reduced =
		"B,E1,E2,E3,E1b,E2b,E3b,Z,"
		"E1t,B,I,I,IA,A,I,B,"
		"E2t,I,B,I,I,IA,A,IAt,"
		"E3t,I,I,B,A,I,IA,IA,"
		"E1bt,IAt,I,At,B,I,I,IAt,"
		"E2bt,At,IAt,I,I,B,I,IA,"
		"E3bt,I,At,IAt,I,I,B,B,"
		"Z,B,IA,IAt,IA,IAt,B,Z";

static int example_graph_VOplus_4_2[] = {
		1,1,1,0,0,0,1,0,1,0,1,1,1,1,0,
		1,1,0,0,1,0,1,0,1,0,1,1,0,1,
		1,1,1,0,1,1,0,0,1,0,0,0,1,
		1,1,1,0,0,1,1,0,0,0,1,0,
		1,0,1,0,1,0,1,1,1,1,0,
		1,0,1,0,1,0,1,1,0,1,
		1,1,1,0,0,1,0,1,1,
		1,1,0,0,0,1,1,1,
		1,1,1,0,1,0,0,
		1,1,1,0,0,0,
		1,1,0,1,1,
		0,1,1,1,
		1,1,0,
		0,1,
		1

};


void graph_theory_domain::make_Neumaier_graph_16(
		int *&Adj, int &N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Neumaier_graph_16" << endl;
	}


	int i, j, h, c;

	N = 16;
	Adj = NEW_int(N * N);
	Int_vec_zero(Adj, N * N);
	h = 0;
	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++) {
			c = example_graph_VOplus_4_2[h];
			h++;
			Adj[i * N + j] = Adj[j * N + i] = c;
		}
	}

	if (f_v) {
		cout << "graph_theory_domain::make_Neumaier_graph_16 done" << endl;
	}

}


void graph_theory_domain::make_Neumaier_graph_25(
		int *&Adj, int &N,
		int verbose_level)
// Abiad, DeBoeck, Zijlemaker: On the existence of small strictly Neumaier graphs
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_Neumaier_graph_25" << endl;
	}

	other::data_structures::string_tools String;

	string input;

	input = Neumaier_graph_25_blocks_reduced;

	std::vector<std::string> output;

	String.parse_comma_separated_list(
			input,
			output,
			verbose_level);

	int h;

	if (f_v) {
		for (h = 0; h < output.size(); h++) {
			cout << output[h];
			if (h < output.size() - 1) {
				cout << " ";
			}
		}
		cout << endl;
	}

	string I, A, IA, At, IAt, B, Z, E1, E2, E3, E1t, E2t, E3t, E1b, E2b, E3b, E1bt, E2bt, E3bt;

	I = "1,0,0,0,1,0,0,0,1";
	A = "0,1,0,0,0,1,1,0,0";
	IA = "1,1,0,0,1,1,1,0,1";
	At = "0,0,1,1,0,0,0,1,0";
	IAt = "1,0,1,1,1,0,0,1,1";
	B = "0,1,1,1,0,1,1,1,0";
	Z = "0,0,0,0,0,0,0,0,0";
	E1 = "1,1,1,0,0,0,0,0,0";
	E2 = "0,0,0,1,1,1,0,0,0";
	E3 = "0,0,0,0,0,0,1,1,1";
	E1t = "1,0,0,1,0,0,1,0,0";
	E2t = "0,1,0,0,1,0,0,1,0";
	E3t = "0,0,1,0,0,1,0,0,1";
	E1b = "0,0,0,1,1,1,1,1,1";
	E2b = "1,1,1,0,0,0,1,1,1";
	E3b = "1,1,1,1,1,1,0,0,0";
	E1bt = "0,1,1,0,1,1,0,1,1";
	E2bt = "1,0,1,1,0,1,1,0,1";
	E3bt = "1,1,0,1,1,0,1,1,0";


	int i, j, c, block_size;

	block_size = 8;
	if (output.size() != block_size * block_size) {
		cout << "output.size() != block_size * block_size" << endl;
		exit(1);
	}

	int N0;
	int *Adj0;

	N0 = 24;
	Adj0 = NEW_int(N0 * N0);
	Int_vec_zero(Adj0, N0 * N0);

	h = 0;

	int u, v;

	for (u = 0; u < block_size; u++) {
		for (v = 0; v < block_size; v++) {
			string str;
			if (output[h] == "I") {
				str = I;
			}
			else if (output[h] == "A") {
				str = A;
			}
			else if (output[h] == "IA") {
				str = IA;
			}
			else if (output[h] == "At") {
				str = At;
			}
			else if (output[h] == "IAt") {
				str = IAt;
			}
			else if (output[h] == "B") {
				str = B;
			}
			else if (output[h] == "Z") {
				str = Z;
			}
			else if (output[h] == "E1") {
				str = E1;
			}
			else if (output[h] == "E2") {
				str = E2;
			}
			else if (output[h] == "E3") {
				str = E3;
			}
			else if (output[h] == "E1t") {
				str = E1t;
			}
			else if (output[h] == "E2t") {
				str = E2t;
			}
			else if (output[h] == "E3t") {
				str = E3t;
			}
			else if (output[h] == "E1b") {
				str = E1b;
			}
			else if (output[h] == "E2b") {
				str = E2b;
			}
			else if (output[h] == "E3b") {
				str = E3b;
			}
			else if (output[h] == "E1bt") {
				str = E1bt;
			}
			else if (output[h] == "E2bt") {
				str = E2bt;
			}
			else if (output[h] == "E3bt") {
				str = E3bt;
			}
			else {
				cout << "symbol is unrecognized: " << output[h] << endl;
				exit(1);
			}


			if (f_v) {
				cout << "entry " << output[h] << " parsing " << str << endl;
			}

			h++;

			std::vector<std::string> block;

			String.parse_comma_separated_list(
					str,
					block,
					0 /*verbose_level*/);

			int s, t;

			for (s = 0; s < 3; s++) {
				for (t = 0; t < 3; t++) {
					if (block[s * 3 + t] == "0") {
						c = 0;
					}
					else if (block[s * 3 + t] == "1") {
						c = 1;
					}
					else {
						cout << "entry in block is unrecognized" << endl;
						exit(1);
					}
					i = u * 3 + s;
					j = v * 3 + t;
					Adj0[i * N0 + j] = Adj0[j * N0 + i] = c;
				}
			}
		}
	}

	N = 25;
	Adj = NEW_int(N * N);
	Int_vec_zero(Adj, N * N);


	for (u = 0; u < N0; u++) {
		for (v = 0; v < N0; v++) {
			c = Adj0[u * N0 + v];
			i = u + 1;
			j = v + 1;
			Adj[i * N + j] = c;
		}
	}
	for (u = 0; u < 12; u++) {
		i = 0;
		j = 1 + u;
		c = 1;
		Adj[i * N + j] = c;
		Adj[j * N + i] = c;
	}

	if (f_v) {
		cout << "graph_theory_domain::make_Neumaier_graph_25 done" << endl;
	}

}

void graph_theory_domain::make_chain_graph(
		int *&Adj, int &N,
		int *part1, int sz1,
		int *part2, int sz2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_chain_graph" << endl;
	}
	if (sz1 != sz2) {
		cout << "graph_theory_domain::make_chain_graph sz1 != sz2" << endl;
	}

	int i, j;
	int N1, N2;
	int *first1;
	int *first2;

	first1 = NEW_int(sz1 + 1);
	first2 = NEW_int(sz1 + 1);

	N1 = 0;
	first1[0] = 0;
	for (i = 0; i < sz1; i++) {
		N1 += part1[i];
		first1[i + 1] = first1[i] + part1[i];
	}
	N2 = 0;
	first2[0] = N1;
	for (i = 0; i < sz2; i++) {
		N2 += part2[i];
		first2[i + 1] = first2[i] + part2[i];
	}
	N = N1 + N2;

	Adj = NEW_int(N * N);
	Int_vec_zero(Adj, N * N);

	int I, J, ii, jj;

	for (I = 0; I < sz1; I++) {
		for (i = 0; i < part1[I]; i++) {
			ii = first1[I] + i;
			for (J = 0; J < sz2 - I; J++) {
				for (j = 0; j < part2[J]; j++) {
					jj = first2[J] + j;
					Adj[ii * N + jj] = 1;
					Adj[jj * N + ii] = 1;
				}
			}
		}
	}

	if (f_v) {
		cout << "graph_theory_domain::make_chain_graph done" << endl;
	}
}

void graph_theory_domain::make_collinearity_graph(
		int *&Adj, int &N,
		int *Inc, int nb_rows, int nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_collinearity_graph" << endl;
	}

	N = nb_rows;
	Adj = NEW_int(N * N);
	Int_vec_zero(Adj, N * N);

	int j, i1, i2;

	for (j = 0; j < nb_cols; j++) {
		for (i1 = 0; i1 < nb_rows; i1++) {
			if (Inc[i1 * nb_cols + j] == 0) {
				continue;
			}
			for (i2 = i1 + 1; i2 < nb_rows; i2++) {
				if (Inc[i2 * nb_cols + j] == 0) {
					continue;
				}
				Adj[i1 * N + i2] = 1;
				Adj[i2 * N + i1] = 1;
			}
		}
	}

	if (f_v) {
		cout << "graph_theory_domain::make_collinearity_graph done" << endl;
	}
}

void graph_theory_domain::make_adjacency_bitvector(
		int *&Adj, int *v, int N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_domain::make_adjacency_bitvector" << endl;
	}


	int i, j, h, c;

	Adj = NEW_int(N * N);
	Int_vec_zero(Adj, N * N);
	h = 0;
	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++) {
			c = v[h];
			h++;
			Adj[i * N + j] = Adj[j * N + i] = c;
		}
	}

	if (f_v) {
		cout << "graph_theory_domain::make_adjacency_bitvector done" << endl;
	}

}



#if 0
void graph_theory_domain::compute_adjacency_matrix_for_disjoint_sets_graph(
		int *Table, int nb_sets, int set_size,
		std::string &prefix_for_graph,
		data_structures::bitvector *&B,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k, N2, N2_100;

	if (f_v) {
		cout << "graph_theory_domain::compute_adjacency_matrix_for_disjoint_sets_graph" << endl;
	}

	N2 = (nb_sets * (nb_sets - 1)) >> 1;
	if (f_v) {
		cout << "graph_theory_domain::compute_adjacency_matrix_for_disjoint_sets_graph N2=" << N2 << endl;
	}
	N2_100 = (N2 / 100) + 1;

	B = NEW_OBJECT(data_structures::bitvector);

	B->allocate(N2);

	if (f_v) {
		cout << "graph_theory_domain::compute_adjacency_matrix_for_disjoint_sets_graph "
				"after allocating adjacency bitvector" << endl;
		cout << "computing adjacency matrix:" << endl;
	}
	k = 0;
	//cnt = 0;
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
				//cnt++;
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
		cout << "graph_theory_domain::compute_adjacency_matrix_for_disjoint_sets_graph "
				"making a graph" << endl;
	}

	{
		colored_graph *CG;
		std::string fname;
		orbiter_kernel_system::file_io Fio;

		CG = NEW_OBJECT(colored_graph);
		int *color;

		color = NEW_int(nb_sets);
		Int_vec_zero(color, nb_sets);

		CG->init(
				nb_sets, 1 /* nb_colors */, 1 /* nb_colors_per_vertex */,
				color, B,
				false,
				prefix_for_graph, prefix_for_graph,
				verbose_level);

		fname = prefix_for_graph + "_disjointness.colored_graph";

		CG->save(fname, verbose_level);

		cout << "Written file " << fname
				<< " of size "
				<< Fio.file_size(fname) << endl;

		FREE_int(color);
		FREE_OBJECT(CG);
	}


	if (f_v) {
		cout << "graph_theory_domain::compute_adjacency_matrix_for_disjoint_sets_graph done" << endl;
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
	Int_vec_zero(Adj, m * m);
#if 0
	for (i = 0; i < m * m; i++) {
		Adj[i] = 0;
	}
#endif

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
#endif

#if 0
void graph_theory_domain::all_cliques_of_given_size(
		int *Adj,
		int nb_pts, int clique_sz, int *&Sol, long int &nb_sol,
		int f_write_cliques, std::string &fname_cliques,
		int verbose_level)
// this functions is nowhere used
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
	int f_maxdepth = false;
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
	Control->f_store_solutions = true;

	C = NEW_OBJECT(clique_finder);

	if (f_v) {
		cout << "graph_theory_domain::all_cliques_of_given_size: "
				"before C->init" << endl;
	}
	C->init(Control,
			label, nb_pts,
			true, adj_list_coded,
			false, NULL,
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


	if (f_write_cliques) {

		if (f_v) {
			cout << "graph_theory_domain::all_cliques_of_given_size "
					"writing cliques" << endl;
		}
		string *Table;
		int nb_cols = 1;
		std::string headings;

		Table = new string[nb_sol * nb_cols];

		headings = "clique";

		for (i = 0; i < nb_sol; i++) {
			Table[i * nb_cols + 0] = Int_vec_stringify(Sol + i * sz, sz);
		}

		orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->write_table_of_strings(
				fname_cliques,
				nb_sol, nb_cols, Table,
				headings,
				verbose_level);

		delete [] Table;
		if (f_v) {
			cout << "graph_theory_domain::all_cliques_of_given_size "
					"Written file " << fname_cliques
					<< " of size " << Fio.file_size(fname_cliques) << endl;
		}

	}

	FREE_OBJECT(C);
	FREE_OBJECT(Control);
	FREE_int(adj_list_coded);
	if (f_v) {
		cout << "graph_theory_domain::all_cliques_of_given_size done" << endl;
	}
}
#endif

void graph_theory_domain::eigenvalues(
		combinatorics::graph_theory::colored_graph *CG,
		int verbose_level)
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

	std::string *Table;
	std::string *Col_headings;
	int nb_rows, nb_cols;

	nb_rows = CG->nb_points;
	nb_cols = 3;

	Table = new string[nb_rows * nb_cols];
	Col_headings = new string[nb_cols];

	Col_headings[0] = "i";
	Col_headings[1] = "Ei";
	Col_headings[2] = "Li";
	for (i = 0; i < CG->nb_points; i++) {
		Table[3 * i + 0] = std::to_string(i);
		Table[3 * i + 1] = std::to_string(E[CG->nb_points - 1 - i]);
		Table[3 * i + 2] = std::to_string(L[CG->nb_points - 1 - i]);
	}

	string fname;

	fname = CG->label + "_eigenvalues.csv";

	other::orbiter_kernel_system::file_io Fio;

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	delete [] Table;
	delete [] Col_headings;

	cout << "graph_theory_domain::eigenvalues "
			"written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


#if 0

	{
		string fname;

		string title, author, extra_praeamble;

		title = "Eigenvalues of graph"; //\\verb'" + CG->label_tex + "'";

		fname = CG->label + "_eigenvalues.tex";

		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface Li;

			Li.head(ost,
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
		other::orbiter_kernel_system::file_io Fio;

		cout << "graph_theory_domain::perform_activity "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
#endif




	delete [] E;

	if (f_v) {
		cout << "graph_theory_domain::eigenvalues done" << endl;
	}

}



}}}}


