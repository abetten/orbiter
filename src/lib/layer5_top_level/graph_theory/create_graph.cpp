/*
 * create_graph.cpp
 *
 *  Created on: Nov 28, 2019
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_graph_theory {


create_graph::create_graph()
{
	description = NULL;

	f_has_CG = false;
	CG = NULL;

	N = 0;
	Adj = NULL;


}

create_graph::~create_graph()
{
	if (f_has_CG) {
		FREE_OBJECT(CG);
	}
}

void create_graph::init(
		create_graph_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::init" << endl;
	}
	create_graph::description = description;

	f_has_CG = false;

	if (description->f_load) {
		if (f_v) {
			cout << "create_graph::init f_load" << endl;
		}

		f_has_CG = true;
		CG = NEW_OBJECT(graph_theory::colored_graph);
		if (f_v) {
			cout << "create_graph::init "
					"before CG->load, "
					"fname=" << description->fname << endl;
		}
		CG->load(description->fname, verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after CG->load, "
					"fname=" << description->fname << endl;
		}
		f_has_CG = true;
		N = CG->nb_points;
		if (f_v) {
			cout << "create_graph::init "
					"number of vertices = " << N << endl;
		}
		label.assign(description->fname);
		if (f_v) {
			cout << "create_graph::init "
					"label = " << label << endl;
		}

		data_structures::string_tools String;
		String.chop_off_extension(label);

		label_tex = "File\\_" + label;

	}

	else if (description->f_Cayley_graph) {
		if (f_v) {
			cout << "create_graph::init f_Cayley_graph" << endl;
		}

		if (f_v) {
			cout << "create_graph::init "
					"group=" << description->Cayley_graph_group << endl;
			cout << "create_graph::init "
					"generators=" << description->Cayley_graph_gens << endl;
		}

		apps_algebra::any_group *G;

		G = Get_object_of_type_any_group(description->Cayley_graph_group);


		groups::strong_generators *SG;

		SG = G->get_strong_generators();


		groups::sims *Sims;

		Sims = SG->create_sims(verbose_level);

		if (f_v) {
			cout << "create_graph::init group order "
					"G = " << Sims->group_order_lint() << endl;
			cout << "create_graph::init group order "
					"coded element size = "
					<< G->A_base->elt_size_in_int << endl;
		}

		int *v;
		int sz;
		int nb_gens;

		Get_int_vector_from_label(description->Cayley_graph_gens,
				v, sz, verbose_level);

		nb_gens = sz / G->A_base->elt_size_in_int;

		if (f_v) {
			cout << "create_graph::init "
					"number of generators = " << nb_gens << endl;

			cout << "create_graph::init generators: ";
			Int_vec_print(cout, v, sz);
			cout << endl;
		}

		data_structures_groups::vector_ge *gens;

		gens = NEW_OBJECT(data_structures_groups::vector_ge);

		gens->init_from_data(G->A, v, nb_gens,
				G->A_base->elt_size_in_int, verbose_level - 1);

		if (f_v) {
			cout << "create_graph::init generators:" << endl;
			gens->print(cout);
		}



		int *Elt1;
		int *Elt2;
		ring_theory::longinteger_object go;
		int i, h, j;

		Elt1 = NEW_int(G->A_base->elt_size_in_int);
		Elt2 = NEW_int(G->A_base->elt_size_in_int);
		Sims->group_order(go);


		N = go.as_lint();

		Adj = NEW_int(N * N);
		Int_vec_zero(Adj, N * N);

		for (i = 0; i < go.as_lint(); i++) {


			Sims->element_unrank_lint(i, Elt1);

			if (f_v) {
				cout << "create_graph::init "
						"Element " << setw(5) << i << " / "
						<< go.as_int() << ":" << endl;
				G->A->Group_element->element_print(Elt1, cout);
				cout << endl;
				G->A->Group_element->element_print_as_permutation(Elt1, cout);
				cout << endl;
			}
			for (h = 0; h < nb_gens; h++) {

				G->A->Group_element->element_mult(
						Elt1, gens->ith(h), Elt2, 0 /*verbose_level*/);

				j = Sims->element_rank_lint(Elt2);

				Adj[i * N + j] = 1;
				Adj[j * N + i] = 1;

			}


		}
		FREE_int(Elt1);
		FREE_int(Elt2);


		f_has_CG = false;
		if (f_v) {
			cout << "create_graph::init "
					"number of vertices = " << N << endl;
		}
		label = "Cayley_graph_" + G->A->label;
		if (f_v) {
			cout << "create_graph::init label = " << label << endl;
		}

		data_structures::string_tools String;
		String.chop_off_extension(label);

		label_tex = "Cayley\\_graph\\_" + G->A->label_tex;

	}

	else if (description->f_load_csv_no_border) {
		if (f_v) {
			cout << "create_graph::init "
					"f_load_from_file_csv_no_border" << endl;
		}

		orbiter_kernel_system::file_io Fio;
		int *M;
		int m, n;

		Fio.Csv_file_support->int_matrix_read_csv_no_border(
				description->fname, M, m, n, verbose_level);
		N = n;
		Adj = M;

		label = description->fname;

		data_structures::string_tools String;
		String.chop_off_extension(label);


		label_tex = "File\\_" + label;
	}

	else if (description->f_load_adjacency_matrix_from_csv_and_select_value) {
		if (f_v) {
			cout << "create_graph::init "
					"f_load_adjacency_matrix_from_csv_and_select_value" << endl;
		}

		orbiter_kernel_system::file_io Fio;
		int *M;
		int m, n;
		int i;

		Fio.Csv_file_support->int_matrix_read_csv(
				description->load_adjacency_matrix_from_csv_and_select_value_fname,
				M, m, n, verbose_level);

		if (m != n) {
			cout << "create_graph::init "
					"the matrix is not square" << endl;
			exit(1);
		}
		N = n;
		for (i = 0; i < N * N; i++) {
			if (M[i] == description->load_adjacency_matrix_from_csv_and_select_value_value) {
				M[i] = 1;
			}
			else {
				M[i] = 0;
			}
		}
		Adj = M;

		label = description->fname;

		data_structures::string_tools String;
		String.chop_off_extension(label);


		label_tex = "File\\_" + label;
	}

	else if (description->f_load_dimacs) {
		if (f_v) {
			cout << "create_graph::init f_load_from_file_dimacs" << endl;
		}

		orbiter_kernel_system::file_io Fio;
		int nb_V;
		int i, j, h;
		std::vector<std::vector<int>> Edges;

		if (f_v) {
			cout << "create_graph::init "
					"before Fio.read_dimacs_graph_format" << endl;
		}
		Fio.read_dimacs_graph_format(description->fname,
				nb_V, Edges, verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after Fio.read_dimacs_graph_format" << endl;
		}

		N = nb_V;
		if (f_v) {
			cout << "create_graph::init "
					"N=" << N << endl;
		}
		if (f_v) {
			cout << "create_graph::init "
					"nb_E=" << Edges.size() << endl;
		}
		Adj = NEW_int(nb_V * nb_V);
		Int_vec_zero(Adj, nb_V * nb_V);

		for (h = 0; h < Edges.size(); h++) {
			i = Edges[h][0];
			j = Edges[h][1];
			if (false) {
				cout << "create_graph::init "
						"edge " << h << " is " << i << " to " << j << endl;
			}
			Adj[i * nb_V + j] = 1;
			Adj[j * nb_V + i] = 1;
		}

		label = description->fname;

		data_structures::string_tools String;
		String.chop_off_extension_and_path(label);


		label_tex = "File\\_" + label;
	}
	else if (description->f_edge_list) {

		combinatorics::combinatorics_domain Combi;
		int h, i, j, a;

		int *Idx;
		int sz;

		Int_vec_scan(description->edge_list_text, Idx, sz);

		N = description->n;


		Adj = NEW_int(N * N);
		Int_vec_zero(Adj, N * N);
		for (h = 0; h < sz; h++) {
			a = Idx[h];
			Combi.k2ij(a, i, j, N);
			Adj[i * N + j] = 1;
			Adj[j * N + i] = 1;
		}
		FREE_int(Idx);

		label = "graph_v" + std::to_string(description->n) + "_e" + std::to_string(sz);
		label_tex = "Graph\\_v" + std::to_string(description->n) + "\\_e" + std::to_string(sz);
	}
	else if (description->f_edges_as_pairs) {
		int h, i, j;
		int *Idx;
		int sz, sz2;

		Int_vec_scan(description->edges_as_pairs_text, Idx, sz);

		N = description->n;


		Adj = NEW_int(N * N);
		Int_vec_zero(Adj, N * N);
		sz2 = sz >> 1;
		for (h = 0; h < sz2; h++) {
			i = Idx[2 * h + 0];
			j = Idx[2 * h + 1];
			Adj[i * N + j] = 1;
			Adj[j * N + i] = 1;
		}
		FREE_int(Idx);

		label = "graph_v" + std::to_string(description->n) + "_e" + std::to_string(sz2);
		label_tex = "Graph\\_v" + std::to_string(description->n) + "\\_e" + std::to_string(sz2);

	}
	else if (description->f_cycle) {

		if (f_v) {
			cout << "create_graph::init "
					"before create_cycle" << endl;
		}
		create_cycle(description->cycle_n,
				verbose_level);


		if (f_v) {
			cout << "create_graph::init "
					"after create_cycle" << endl;
		}
	}
	else if (description->f_inversion_graph) {

		if (f_v) {
			cout << "create_graph::init "
					"before create_inversion_graph" << endl;
		}
		create_inversion_graph(description->inversion_graph_text,
				verbose_level);


		if (f_v) {
			cout << "create_graph::init "
					"after create_inversion_graph" << endl;
		}
	}
	else if (description->f_Hamming) {

		if (f_v) {
			cout << "create_graph::init "
					"before create_Hamming" << endl;
		}
		create_Hamming(
				description->Hamming_n,
				description->Hamming_q,
				verbose_level);


		if (f_v) {
			cout << "create_graph::init "
					"after create_Hamming" << endl;
		}
	}
	else if (description->f_Johnson) {

		if (f_v) {
			cout << "create_graph::init "
					"before create_Johnson" << endl;
		}
		create_Johnson(
				description->Johnson_n,
				description->Johnson_k, description->Johnson_s,
				verbose_level);


		if (f_v) {
			cout << "create_graph::init "
					"after create_Johnson" << endl;
		}
	}
	else if (description->f_Paley) {

		if (f_v) {
			cout << "create_graph::init "
					"before create_Paley" << endl;
		}
		create_Paley(
				description->Paley_label_Fq,
				verbose_level);


		if (f_v) {
			cout << "create_graph::init "
					"after create_Paley" << endl;
		}
	}
	else if (description->f_Sarnak) {

		if (f_v) {
			cout << "create_graph::init "
					"before create_Sarnak" << endl;
		}
		create_Sarnak(
				description->Sarnak_p,
				description->Sarnak_q,
				verbose_level);


		if (f_v) {
			cout << "create_graph::init "
					"after create_Sarnak" << endl;
		}
	}
	else if (description->f_Schlaefli) {

		if (f_v) {
			cout << "create_graph::init "
					"before create_Schlaefli" << endl;
		}
		create_Schlaefli(
				description->Schlaefli_label_Fq,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after create_Schlaefli" << endl;
		}
	}
	else if (description->f_Shrikhande) {

		if (f_v) {
			cout << "create_graph::init "
					"before create_Shrikhande" << endl;
		}
		create_Shrikhande(verbose_level);

		if (f_v) {
			cout << "create_graph::init "
					"after create_Shrikhande" << endl;
		}
	}
	else if (description->f_Winnie_Li) {

		if (f_v) {
			cout << "create_graph::init "
					"before create_Winnie_Li" << endl;
		}
		create_Winnie_Li(
				description->Winnie_Li_label_Fq,
				description->Winnie_Li_index,
				verbose_level);

		if (f_v) {
			cout << "create_graph::init "
					"after create_Winnie_Li" << endl;
		}
	}
	else if (description->f_Grassmann) {

		if (f_v) {
			cout << "create_graph::init "
					"before create_Grassmann" << endl;
		}
		create_Grassmann(
				description->Grassmann_n,
				description->Grassmann_k,
				description->Grassmann_label_Fq,
				description->Grassmann_r,
				verbose_level);

		if (f_v) {
			cout << "create_graph::init "
					"after create_Grassmann" << endl;
		}
	}
	else if (description->f_coll_orthogonal) {

		if (f_v) {
			cout << "create_graph::init "
					"before create_coll_orthogonal" << endl;
		}
		create_coll_orthogonal(
				description->coll_orthogonal_space_label,
				description->coll_orthogonal_set_of_points_label,
				verbose_level);

		if (f_v) {
			cout << "create_graph::init "
					"after create_coll_orthogonal" << endl;
		}
	}
	else if (description->f_trihedral_pair_disjointness_graph) {

		algebraic_geometry::surface_domain *Surf;
		field_theory::finite_field *F;

		F = NEW_OBJECT(field_theory::finite_field);
		Surf = NEW_OBJECT(algebraic_geometry::surface_domain);

		F->finite_field_init_small_order(5,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);
		Surf->init_surface_domain(F, verbose_level);

		Surf->Schlaefli->make_trihedral_pair_disjointness_graph(Adj, verbose_level);
		N = 120;
		label.assign("trihedral_pair_disjointness");
		label_tex.assign("trihedral\\_pair\\_disjointness");

		FREE_OBJECT(Surf);
		FREE_OBJECT(F);
	}
	else if (description->f_non_attacking_queens_graph) {


		graph_theory::graph_theory_domain GT;

		int n;

		n = description->non_attacking_queens_graph_n;


		if (f_v) {
			cout << "create_graph::init "
					"before GT.make_non_attacking_queens_graph" << endl;
		}
		GT.make_non_attacking_queens_graph(Adj, N, n, verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after GT.make_non_attacking_queens_graph" << endl;
		}


		label = "non_attacking_queens_graph_" + std::to_string(n);
		label_tex = "non\\_attacking\\_queens\\_graph\\_" + std::to_string(n);

	}
	else if (description->f_disjoint_sets_graph) {

		graph_theory::graph_theory_domain GT;


		if (f_v) {
			cout << "create_graph::init "
					"before GT.make_disjoint_sets_graph" << endl;
		}
		GT.make_disjoint_sets_graph(Adj, N,
				description->disjoint_sets_graph_fname,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after GT.make_disjoint_sets_graph" << endl;
		}

		string L;
		data_structures::string_tools String;

		L = description->disjoint_sets_graph_fname;
		String.chop_off_extension(L);
		L += "_disjoint_sets";

		label = L;

		L = description->disjoint_sets_graph_fname;
		String.chop_off_extension(L);
		L += "\\_disjoint\\_sets";

		label_tex = L;
	}
	else if (description->f_orbital_graph) {

		graph_theory::graph_theory_domain GT;

		apps_algebra::any_group *AG;

		AG = Get_object_of_type_any_group(description->orbital_graph_group);


		if (f_v) {
			cout << "create_graph::init "
					"before GT.make_orbital_graph" << endl;
		}
		make_orbital_graph(
				AG, description->orbital_graph_orbit_idx,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after GT.make_orbital_graph" << endl;
		}
		if (f_v) {
			cout << "create_graph::init label = " << label << endl;
			cout << "create_graph::init label_tex = " << label_tex << endl;
			cout << "create_graph::init done" << endl;
		}

	}
	else if (description->f_collinearity_graph) {



		int *v;
		int m, n;

		Get_matrix(description->collinearity_graph_matrix, v, m, n);


		if (f_v) {
			cout << "create_graph::init "
					"before make_collinearity_graph" << endl;
		}
		make_collinearity_graph(
				v, m, n,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after make_collinearity_graph" << endl;
		}



	}
	else if (description->f_chain_graph) {



		int *v1;
		int sz1;
		int *v2;
		int sz2;

		Get_int_vector_from_label(
				description->chain_graph_partition_1,
				v1, sz1, 0 /* verbose_level*/);
		Get_int_vector_from_label(
				description->chain_graph_partition_2,
				v2, sz2, 0 /* verbose_level*/);


		if (f_v) {
			cout << "create_graph::init "
					"before make_chain_graph" << endl;
		}
		make_chain_graph(
				v1, sz1,
				v2, sz2,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after make_chain_graph" << endl;
		}



	}



	if (description->f_subset) {
		if (f_v) {
			cout << "create_graph::init the graph has a subset" << endl;
		}
		CG = NEW_OBJECT(graph_theory::colored_graph);
		CG->init_adjacency_no_colors(N, Adj,
				description->subset_label,
				description->subset_label_tex,
				verbose_level);

		int *subset;
		int sz;

		Int_vec_scan(description->subset_text, subset, sz);

		CG->init_adjacency_two_colors(N,
				Adj, subset, sz,
				description->subset_label,
				description->subset_label_tex,
				verbose_level);

		f_has_CG = true;

		label += description->subset_label;
		label_tex += description->subset_label_tex;

		FREE_int(subset);
		if (f_v) {
			cout << "create_graph::init created colored graph with two colors" << endl;
		}

	}
	else {

		if (!f_has_CG) {

			CG = NEW_OBJECT(graph_theory::colored_graph);
			if (f_v) {
				cout << "create_graph::init "
						"before CG->init_adjacency_no_colors" << endl;
			}
			CG->init_adjacency_no_colors(N, Adj, label, label_tex,
					verbose_level);
			if (f_v) {
				cout << "create_graph::init "
						"after CG->init_adjacency_no_colors" << endl;
			}

			f_has_CG = true;

			if (f_v) {
				cout << "create_graph::init "
						"created colored graph with one color" << endl;
			}
		}

	}

	int i;

	for (i = 0; i < description->Modifications.size(); i++) {
		description->Modifications[i].apply(CG, verbose_level);
	}

	CG->label = label;
	CG->label_tex = label_tex;


	if (f_v) {
		cout << "create_graph::init label = " << label << endl;
		cout << "create_graph::init done" << endl;
	}
}


void create_graph::create_cycle(
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_cycle" << endl;
	}

	graph_theory::graph_theory_domain GT;


	if (f_v) {
		cout << "create_graph::create_cycle "
				"before GT.make_cycle_graph" << endl;
	}
	GT.make_cycle_graph(Adj, N, n, verbose_level);
	if (f_v) {
		cout << "create_graph::create_cycle "
				"after GT.make_cycle_graph" << endl;
	}

	label = "Cycle_" + std::to_string(n);
	label_tex = "Cycle\\_" + std::to_string(n);


	if (f_v) {
		cout << "create_graph::create_cycle done" << endl;
	}
}

void create_graph::create_inversion_graph(
		std::string &perm_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_inversion_graph" << endl;
	}

	int *perm;
	int n;

	Int_vec_scan(perm_text, perm, n);


	graph_theory::graph_theory_domain GT;


	if (f_v) {
		cout << "create_graph::create_inversion_graph "
				"before GT.make_inversion_graph" << endl;
	}
	GT.make_inversion_graph(Adj, N, perm, n, verbose_level);
	if (f_v) {
		cout << "create_graph::create_inversion_graph "
				"after GT.make_inversion_graph" << endl;
	}

	label = "Inversion_" + std::to_string(n);
	label_tex = "Inversion\\_" + std::to_string(n);


	FREE_int(perm);

	if (f_v) {
		cout << "create_graph::create_inversion_graph done" << endl;
	}
}




void create_graph::create_Hamming(
		int n, int q,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Hamming" << endl;
	}

	graph_theory::graph_theory_domain GT;


	if (f_v) {
		cout << "create_graph::create_Hamming "
				"before GT.make_Hamming_graph" << endl;
	}
	GT.make_Hamming_graph(Adj, N, n, q, verbose_level);
	if (f_v) {
		cout << "create_graph::create_Hamming "
				"after GT.make_Hamming_graph" << endl;
	}

	label = "Hamming_" + std::to_string(n) + "_" + std::to_string(q);
	label_tex = "Hamming\\_" + std::to_string(n) + "\\_" + std::to_string(q);


	if (f_v) {
		cout << "create_graph::create_Hamming done" << endl;
	}
}


void create_graph::create_Johnson(
		int n, int k, int s,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Johnson" << endl;
	}

	graph_theory::graph_theory_domain GT;


	if (f_v) {
		cout << "create_graph::create_Johnson "
				"before GT.make_Johnson_graph" << endl;
	}
	GT.make_Johnson_graph(Adj, N, n, k, s, verbose_level);
	if (f_v) {
		cout << "create_graph::create_Johnson "
				"after GT.make_Johnson_graph" << endl;
	}

	label = "Johnson_" + std::to_string(n) + "_" + std::to_string(k) + "_" + std::to_string(s);
	label_tex = "Johnson\\_" + std::to_string(n) + "\\_" + std::to_string(k) + "\\_" + std::to_string(s);


	if (f_v) {
		cout << "create_graph::create_Johnson done" << endl;
	}
}

void create_graph::create_Paley(
		std::string &label_Fq, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Paley" << endl;
	}


	graph_theory::graph_theory_domain GT;
	field_theory::finite_field *Fq;

	Fq = Get_finite_field(label_Fq);


	if (f_v) {
		cout << "create_graph::create_Paley "
				"before GT.make_Paley_graph" << endl;
	}
	GT.make_Paley_graph(Adj, N, Fq, verbose_level);
	if (f_v) {
		cout << "create_graph::create_Paley "
				"after GT.make_Paley_graph" << endl;
	}

	label = "Paley_" + std::to_string(Fq->q);
	label_tex = "Paley\\_" + std::to_string(Fq->q);


	if (f_v) {
		cout << "create_graph::create_Paley done" << endl;
	}
}

void create_graph::create_Sarnak(
		int p, int q,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Sarnak" << endl;
	}


	int l, f_special = false;
	number_theory::number_theory_domain NT;



	l = NT.Legendre(p, q, 0);
	if (f_v) {
		cout << "create_graph::create_Sarnak "
				"Legendre(" << p << ", " << q << ")=" << l << endl;
	}


	field_theory::finite_field *F;
	actions::action *A;
	int f_semilinear = false;
	int f_basis = true;

	F = NEW_OBJECT(field_theory::finite_field);
	F->finite_field_init_small_order(q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);
	//F->init_override_polynomial(q, override_poly, verbose_level);

	A = NEW_OBJECT(actions::action);


	// create PSL(2,q) or PGL(2,q) depending on Legendre(p, q):

	if (l == 1) {
		f_special = true;

		if (f_v) {
			cout << "create_graph::create_Sarnak "
					"Creating projective special linear group:" << endl;
		}
		A->Known_groups->init_projective_special_group(2, F,
			f_semilinear,
			f_basis,
			verbose_level - 2);
	}
	else {
		data_structures_groups::vector_ge *nice_gens;

		if (f_v) {
			cout << "create_graph::create_Sarnak "
					"Creating projective linear group:" << endl;
		}
		A->Known_groups->init_projective_group(2, F,
			f_semilinear,
			f_basis, true /* f_init_sims */,
			nice_gens,
			verbose_level - 2);
		FREE_OBJECT(nice_gens);
	}





	graph_theory_apps GTA;


	if (f_v) {
		cout << "create_graph::create_Sarnak before GTA.expander_graph" << endl;
	}
	GTA.expander_graph(
			p, q, f_special,
			F, A,
			Adj, N,
			verbose_level);
	if (f_v) {
		cout << "create_graph::create_Sarnak before GTA.expander_graph" << endl;
	}


	label = "Sarnak_" + std::to_string(p) + "_" + std::to_string(q);
	label_tex = "Sarnak\\_" + std::to_string(p) + "\\_" + std::to_string(q);

	FREE_OBJECT(A);
	FREE_OBJECT(F);


	if (f_v) {
		cout << "create_graph::create_Sarnak done" << endl;
	}
}


void create_graph::create_Schlaefli(
		std::string &label_Fq, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Schlaefli" << endl;
	}

	graph_theory::graph_theory_domain GT;
	field_theory::finite_field *F;

	F = Get_finite_field(label_Fq);

	if (f_v) {
		cout << "create_graph::create_Schlaefli "
				"before GT.make_Schlaefli_graph" << endl;
	}
	GT.make_Schlaefli_graph(Adj, N, F, verbose_level);
	if (f_v) {
		cout << "create_graph::create_Schlaefli "
				"after GT.make_Schlaefli_graph" << endl;
	}

	label = "Schlaefli_" + std::to_string(F->q);
	label_tex = "Schlaefli\\_" + std::to_string(F->q);


	if (f_v) {
		cout << "create_graph::create_Schlaefli done" << endl;
	}
}

void create_graph::create_Shrikhande(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Shrikhande" << endl;
	}

	actions::action *A;
	data_structures_groups::vector_ge *gens_G;
	data_structures_groups::vector_ge *gens_S;
	int *v;
	int n = 8;
	int i, j;
	int nb_G, nb_S;
	long int goi;
	int f_no_base = false;

	A = NEW_OBJECT(actions::action);
	A->Known_groups->init_symmetric_group(n, f_no_base, verbose_level);
	goi = A->group_order_lint();

	if (f_v) {
		cout << "create_graph::create_Shrikhande "
				"Created group Sym(" << n << ") of size " << goi << endl;
	}

	nb_G = 2;
	nb_S = 6;

	gens_G = NEW_OBJECT(data_structures_groups::vector_ge);
	gens_G->init(A, verbose_level - 2);
	gens_G->allocate(nb_G, verbose_level - 2);


	gens_S = NEW_OBJECT(data_structures_groups::vector_ge);
	gens_S->init(A, verbose_level - 2);
	gens_S->allocate(nb_S, verbose_level - 2);

	v = NEW_int(n);


	for (i = 0; i < nb_G; i++) {
		if (i == 0) {
			for (j = 0; j < 4; j++) {
				v[j] = (j + 1) % 4;
			}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + j;
			}
		}
		else {
			for (j = 0; j < 4; j++) {
				v[j] = j;
			}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + ((j + 1) % 4);
			}
		}
		A->Group_element->make_element(gens_G->ith(i), v, 0 /* verbose_level */);
	}

	if (f_v) {
		cout << "create_graph::create_Shrikhande "
				"generators for G:" << endl;
		for (i = 0; i < nb_G; i++) {
			cout << "generator " << i << ":" << endl;
			A->Group_element->element_print(gens_G->ith(i), cout);
		}
	}

	for (i = 0; i < nb_S; i++) {
		if (i == 0) {
			for (j = 0; j < 4; j++) {
				v[j] = (j + 1) % 4;
			}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + j;
			}
		}
		else if (i == 1) {
			for (j = 0; j < 4; j++) {
				v[j] = (4 + j - 1) % 4;
			}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + j;
			}
		}
		else if (i == 2) {
			for (j = 0; j < 4; j++) {
				v[j] = j;
			}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + ((j + 1) % 4);
			}
		}
		else if (i == 3) {
			for (j = 0; j < 4; j++) {
				v[j] = j;
			}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + ((4 + j - 1) % 4);
			}
		}
		else if (i == 4) {
			for (j = 0; j < 4; j++) {
				v[j] = (j + 1) % 4;
			}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + ((j + 1) % 4);
			}
		}
		else if (i == 5) {
			for (j = 0; j < 4; j++) {
				v[j] = (4 + j - 1) % 4;
			}
			for (j = 0; j < 4; j++) {
				v[4 + j] = 4 + ((4 + j - 1) % 4);
			}
		}
		A->Group_element->make_element(gens_S->ith(i), v, 0 /* verbose_level */);
	}

	if (f_v) {
		cout << "create_graph::create_Shrikhande "
				"generators for S:" << endl;
		for (i = 0; i < nb_S; i++) {
			cout << "generator " << i << ":" << endl;
			A->Group_element->element_print(gens_S->ith(i), cout);
		}
	}

	groups::sims *G;


	G = A->create_sims_from_generators_with_target_group_order_lint(
		gens_G, 16, verbose_level);



	if (f_v) {
		cout << "create_graph::create_Shrikhande "
				"before G->Cayley_graph" << endl;
	}
	G->Cayley_graph(Adj, N, gens_S, verbose_level);
	if (f_v) {
		cout << "create_graph::create_Shrikhande "
				"after G->Cayley_graph" << endl;
	}



	if (f_v) {
		cout << "create_graph::create_Shrikhande "
				"The adjacency matrix of a graph with " <<
				goi << " vertices has been computed" << endl;
		//int_matrix_print(Adj, goi, goi);
	}

	//N = goi;

	label = "Shrikhande";
	label_tex = "Shrikhande";


	FREE_int(v);
	FREE_OBJECT(gens_G);
	FREE_OBJECT(gens_S);

	if (f_v) {
		cout << "create_graph::create_Shrikhande done" << endl;
	}
}

void create_graph::create_Winnie_Li(
		std::string &label_Fq, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Winnie_Li" << endl;
	}

	graph_theory::graph_theory_domain GT;
	field_theory::finite_field *Fq;

	Fq = Get_finite_field(label_Fq);


	if (f_v) {
		cout << "create_graph::create_Winnie_Li "
				"before Combi.make_Winnie_Li_graph" << endl;
	}
	GT.make_Winnie_Li_graph(Adj, N, Fq, index, verbose_level);
	if (f_v) {
		cout << "create_graph::create_Winnie_Li "
				"after Combi.make_Winnie_Li_graph" << endl;
	}


	label = "Winnie_Li_" + std::to_string(Fq->q) + "_" + std::to_string(index);
	label_tex = "Winnie\\_Li\\_" + std::to_string(Fq->q) + "\\_" + std::to_string(index);



	if (f_v) {
		cout << "create_graph::create_Winnie_Li done" << endl;
	}
}

void create_graph::create_Grassmann(
		int n, int k, std::string &label_Fq,
		int r, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Grassmann" << endl;
	}


	graph_theory::graph_theory_domain GT;
	field_theory::finite_field *F;

	F = Get_finite_field(label_Fq);



	if (f_v) {
		cout << "create_graph::create_Grassmann "
				"before GT.make_Grassmann_graph" << endl;
	}
	GT.make_Grassmann_graph(Adj, N, n, k, F, r, verbose_level);
	if (f_v) {
		cout << "create_graph::create_Grassmann "
				"after GT.make_Grassmann_graph" << endl;
	}


	label = "Grassmann_" + std::to_string(n) + "_" + std::to_string(k) + " " + std::to_string(F->q) + "_" + std::to_string(r);
	label_tex = "Grassmann\\_" + std::to_string(n) + "\\_" + std::to_string(k) + "\\_" + std::to_string(F->q) + "\\_" + std::to_string(r);


	if (f_v) {
		cout << "create_graph::create_Grassmann done" << endl;
	}
}

void create_graph::create_coll_orthogonal(
		std::string &orthogonal_space_label,
		std::string &set_of_points_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_coll_orthogonal" << endl;
	}

	graph_theory::graph_theory_domain GT;

	orthogonal_geometry_applications::orthogonal_space_with_action *OA;

	OA = Get_orthogonal_space(orthogonal_space_label);

	long int *Set;
	int sz;

	Get_lint_vector_from_label(
			set_of_points_label, Set, sz, 0 /* verbose_level */);


	if (f_v) {
		cout << "create_graph::create_coll_orthogonal before "
				"OA->O->Quadratic_form->make_collinearity_graph" << endl;
	}
	OA->O->Quadratic_form->make_collinearity_graph(
			Adj, N,
			Set, sz,
			verbose_level);
	if (f_v) {
		cout << "create_graph::create_coll_orthogonal after "
				"OA->O->Quadratic_form->make_collinearity_graph" << endl;
	}


	label = OA->O->label_txt + "_coll_" + set_of_points_label;

	label_tex = OA->O->label_tex + "\\_coll\\_" + set_of_points_label;

	if (f_v) {
		cout << "create_graph::create_coll_orthogonal done" << endl;
	}
}

void create_graph::make_orbital_graph(
		apps_algebra::any_group *AG, int orbit_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_orbital_graph" << endl;
	}

	poset_classification::poset_classification_control *Control;

	Control = NEW_OBJECT(poset_classification::poset_classification_control);

	poset_classification::poset_classification *PC;

	if (f_v) {
		cout << "create_graph::make_orbital_graph "
				"before AG->orbits_on_subsets" << endl;
	}
	AG->orbits_on_subsets(Control, PC, 2, verbose_level);
	if (f_v) {
		cout << "create_graph::make_orbital_graph "
				"after AG->orbits_on_subsets" << endl;
	}

	long int set[2];
	int size;

	PC->get_Poo()->get_set(2 /* level */, orbit_idx, set, size);

	if (f_v) {
		cout << "create_graph::make_orbital_graph set: ";
		Lint_vec_print(cout, set, 2);
		cout << endl;
	}

	orbits_schreier::orbit_of_sets *Orb;

	Orb = NEW_OBJECT(orbits_schreier::orbit_of_sets);

	if (f_v) {
		cout << "create_graph::make_orbital_graph "
				"before Orb->init" << endl;
	}
	Orb->init(AG->A_base, AG->A,
			set, 2, AG->Subgroup_gens->gens, verbose_level);
	if (f_v) {
		cout << "create_graph::make_orbital_graph "
				"after Orb->init" << endl;
	}

	int *M;
	int nb_points;
	int i, j, h;

	nb_points = AG->A->degree;

	M = NEW_int(nb_points * nb_points);
	Int_vec_zero(M, nb_points * nb_points);
	for (h = 0; h < Orb->used_length; h++) {
		i = Orb->Sets[h][0];
		j = Orb->Sets[h][1];
		M[i * nb_points + j] = 1;
		M[j * nb_points + i] = 1;
	}

	FREE_OBJECT(Orb);
	N = nb_points;
	Adj = M;

	if (f_v) {
		cout << "create_graph::make_orbital_graph "
				"AG->A->label = " << AG->A->label << endl;
		cout << "create_graph::make_orbital_graph "
				"AG->A->label_tex = " << AG->A->label_tex << endl;
	}

	label = "Group_" + AG->A->label + "Orbital_" + std::to_string(orbit_idx);
	label_tex = "Group\\_" + AG->A->label_tex + "Orbital\\_" + std::to_string(orbit_idx);

	if (f_v) {
		cout << "create_graph::make_orbital_graph done" << endl;
	}
}

void create_graph::make_collinearity_graph(
		int *Inc, int nb_rows, int nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_collinearity_graph" << endl;
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

	label = "collinearity_graph";
	label_tex = "collinearity\\_graph";

	if (f_v) {
		cout << "create_graph::make_collinearity_graph done" << endl;
	}
}

void create_graph::make_chain_graph(
		int *part1, int sz1,
		int *part2, int sz2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_chain_graph" << endl;
	}
	if (sz1 != sz2) {
		cout << "create_graph::make_chain_graph sz1 != sz2" << endl;
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

	label = "chain_graph";
	label_tex = "chain\\_graph";

	if (f_v) {
		cout << "create_graph::make_chain_graph done" << endl;
	}
}


}}}
