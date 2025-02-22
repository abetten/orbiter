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
	Record_birth();
	description = NULL;

	f_has_CG = false;
	CG = NULL;

	N = 0;
	Adj = NULL;


}

create_graph::~create_graph()
{
	Record_death();
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

		if (f_v) {
			cout << "create_graph::init "
					"before load" << endl;
		}
		load(
				description->fname,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after load" << endl;
		}

	}

	else if (description->f_Cayley_graph) {
		if (f_v) {
			cout << "create_graph::init f_Cayley_graph" << endl;
		}


		if (f_v) {
			cout << "create_graph::init "
					"before make_Cayley_graph" << endl;
		}
		make_Cayley_graph(
				description->Cayley_graph_group,
				description->Cayley_graph_gens,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after make_Cayley_graph" << endl;
		}


	}

	else if (description->f_load_csv_no_border) {
		if (f_v) {
			cout << "create_graph::init "
					"f_load_from_file_csv_no_border" << endl;
		}



		if (f_v) {
			cout << "create_graph::init "
					"before load_csv_without_border" << endl;
		}
		load_csv_without_border(
				description->load_csv_no_border_fname,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after load_csv_without_border" << endl;
		}

		other::data_structures::string_tools String;
		string fname_base;

		fname_base = description->load_csv_no_border_fname;
		String.chop_off_extension_and_path(
				fname_base);

		label = "File_" + fname_base;
		label_tex = "{\\rm File_" + fname_base + "}";

	}

	else if (description->f_load_adjacency_matrix_from_csv_and_select_value) {
		if (f_v) {
			cout << "create_graph::init "
					"f_load_adjacency_matrix_from_csv_and_select_value" << endl;
		}

		other::orbiter_kernel_system::file_io Fio;
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

		std::string fname;

		fname = description->load_adjacency_matrix_from_csv_and_select_value_fname;

		other::data_structures::string_tools String;
		String.chop_off_extension(fname);


		label = "File_" + fname;
		label_tex = "{\\rm File_" + fname + "}";
	}

	else if (description->f_load_dimacs) {
		if (f_v) {
			cout << "create_graph::init f_load_dimacs" << endl;
		}


		if (f_v) {
			cout << "create_graph::init "
					"before load_dimacs" << endl;
		}
		load_dimacs(
				description->load_dimacs_fname,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after load_dimacs" << endl;
		}

	}

	else if (description->f_load_Brouwer) {
		if (f_v) {
			cout << "create_graph::init f_load_Brouwer" << endl;
		}


		if (f_v) {
			cout << "create_graph::init "
					"before load_Brouwer" << endl;
		}
		load_Brouwer(
				description->load_Brouwer_fname,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after load_Brouwer" << endl;
		}

	}

	else if (description->f_edge_list) {

		combinatorics::other_combinatorics::combinatorics_domain Combi;
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
		label_tex = "{\\rm Graph\\_v" + std::to_string(description->n) + "\\_e" + std::to_string(sz) + "}";
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
		label_tex = "{\\rm Graph\\_v" + std::to_string(description->n) + "\\_e" + std::to_string(sz2) + "}";

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
	else if (description->f_affine_polar) {

		if (f_v) {
			cout << "create_graph::init f_affine_polar "
					"before create_affine_polar" << endl;
		}
		create_affine_polar(
				description->affine_polar_space_label,
				verbose_level);

		if (f_v) {
			cout << "create_graph::init "
					"after create_affine_polar" << endl;
		}
	}



	else if (description->f_tritangent_planes_disjointness_graph) {

		if (f_v) {
			cout << "create_graph::init f_affine_polar "
					"before make_tritangent_plane_disjointness_graph" << endl;
		}

		make_tritangent_plane_disjointness_graph(
				verbose_level);

		if (f_v) {
			cout << "create_graph::init "
					"after make_tritangent_plane_disjointness_graph" << endl;
		}
	}

	else if (description->f_trihedral_pair_disjointness_graph) {

		if (f_v) {
			cout << "create_graph::init f_affine_polar "
					"before make_trihedral_pair_disjointness_graph" << endl;
		}

		make_trihedral_pair_disjointness_graph(
				verbose_level);

		if (f_v) {
			cout << "create_graph::init "
					"after make_trihedral_pair_disjointness_graph" << endl;
		}
	}
	else if (description->f_non_attacking_queens_graph) {


		combinatorics::graph_theory::graph_theory_domain GT;

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
		label_tex = "{\\rm non\\_attacking\\_queens\\_graph\\_" + std::to_string(n) + "}";

	}
	else if (description->f_disjoint_sets_graph) {



		if (f_v) {
			cout << "create_graph::init "
					"before make_disjoint_sets_graph" << endl;
		}
		make_disjoint_sets_graph(
				description->disjoint_sets_graph_fname,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after make_disjoint_sets_graph" << endl;
		}

	}
	else if (description->f_orbital_graph) {

		combinatorics::graph_theory::graph_theory_domain GT;


		if (f_v) {
			cout << "create_graph::init "
					"before GT.make_orbital_graph" << endl;
		}
		make_orbital_graph(
				description->orbital_graph_group,
				description->orbital_graph_orbit_idx,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after GT.make_orbital_graph" << endl;
		}
	}
	else if (description->f_collinearity_graph) {





		if (f_v) {
			cout << "create_graph::init "
					"before make_collinearity_graph" << endl;
		}
		make_collinearity_graph(
				description->collinearity_graph_matrix,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after make_collinearity_graph" << endl;
		}



	}



	else if (description->f_chain_graph) {



		if (f_v) {
			cout << "create_graph::init "
					"before make_chain_graph" << endl;
		}
		make_chain_graph(
				description->chain_graph_partition_1,
				description->chain_graph_partition_2,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after make_chain_graph" << endl;
		}



	}

	else if (description->f_Neumaier_graph_16) {




		if (f_v) {
			cout << "create_graph::init "
					"before make_Neumaier_graph_16" << endl;
		}
		make_Neumaier_graph_16(
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after make_Neumaier_graph_16" << endl;
		}



	}

	else if (description->f_Neumaier_graph_25) {



		if (f_v) {
			cout << "create_graph::init "
					"before make_Neumaier_graph_25" << endl;
		}
		make_Neumaier_graph_25(
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after make_Neumaier_graph_25" << endl;
		}



	}

	else if (description->f_adjacency_bitvector) {



		if (f_v) {
			cout << "create_graph::init "
					"f_adjacency_bitvector" << endl;
		}


		if (f_v) {
			cout << "create_graph::init "
					"before make_adjacency_bitvector" << endl;
		}
		make_adjacency_bitvector(
				description->adjacency_bitvector_data_text,
				description->adjacency_bitvector_N,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after make_adjacency_bitvector" << endl;
		}



	}


	if (description->f_subset) {
		if (f_v) {
			cout << "create_graph::init the graph has a subset" << endl;
		}
		CG = NEW_OBJECT(combinatorics::graph_theory::colored_graph);
		if (f_v) {
			cout << "create_graph::init "
					"before CG->init_from_adjacency_no_colors" << endl;
		}
		CG->init_from_adjacency_no_colors(
				N, Adj,
				description->subset_label,
				description->subset_label_tex,
				verbose_level);
		if (f_v) {
			cout << "create_graph::init "
					"after CG->init_from_adjacency_no_colors" << endl;
		}

		int *subset;
		int sz;

		Int_vec_scan(description->subset_text, subset, sz);

		CG->init_adjacency_two_colors(
				N,
				Adj, subset, sz,
				description->subset_label,
				description->subset_label_tex,
				verbose_level);

		f_has_CG = true;

		label = label + description->subset_label;
		label_tex = label_tex + "{\\rm " + description->subset_label_tex + "}";

		FREE_int(subset);
		if (f_v) {
			cout << "create_graph::init created colored graph with two colors" << endl;
		}

	}
	else {

		if (!f_has_CG) {

			CG = NEW_OBJECT(combinatorics::graph_theory::colored_graph);
			if (f_v) {
				cout << "create_graph::init "
						"before CG->init_from_adjacency_no_colors" << endl;
			}
			CG->init_from_adjacency_no_colors(
					N, Adj, label, label_tex,
					verbose_level);
			if (f_v) {
				cout << "create_graph::init "
						"after CG->init_from_adjacency_no_colors" << endl;
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


void create_graph::load(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::load" << endl;
	}

	f_has_CG = true;
	CG = NEW_OBJECT(combinatorics::graph_theory::colored_graph);
	if (f_v) {
		cout << "create_graph::load "
				"before CG->load, "
				"fname=" << description->fname << endl;
	}
	CG->load(fname, verbose_level);
	if (f_v) {
		cout << "create_graph::load "
				"after CG->load, "
				"fname=" << fname << endl;
	}
	f_has_CG = true;
	N = CG->nb_points;
	if (f_v) {
		cout << "create_graph::load "
				"number of vertices = " << N << endl;
	}
	label.assign(fname);
	if (f_v) {
		cout << "create_graph::load "
				"label = " << label << endl;
	}

	other::data_structures::string_tools String;
	String.chop_off_extension(label);

	label_tex = "File\\_" + label;

}

void create_graph::make_Cayley_graph(
		std::string &group_label,
		std::string &generators_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_Cayley_graph" << endl;
	}

	if (f_v) {
		cout << "create_graph::init "
				"group=" << group_label << endl;
		cout << "create_graph::init "
				"generators=" << generators_label << endl;
	}

	groups::any_group *G;

	G = Get_any_group(group_label);


	groups::strong_generators *SG;

	SG = G->get_strong_generators();


	groups::sims *Sims;

	Sims = SG->create_sims(verbose_level);

	if (f_v) {
		cout << "create_graph::make_Cayley_graph group order "
				"G = " << Sims->group_order_lint() << endl;
		cout << "create_graph::make_Cayley_graph group order "
				"coded element size = G->A_base->make_element_size = "
				<< G->A_base->make_element_size << endl;
	}

	int *v;
	int sz;
	int nb_gens;

	Get_int_vector_from_label(generators_label,
			v, sz, verbose_level);

	nb_gens = sz / G->A_base->make_element_size;

	if (f_v) {
		cout << "create_graph::make_Cayley_graph "
				"number of generators = " << nb_gens << endl;

		cout << "create_graph::make_Cayley_graph generators: ";
		Int_vec_print(cout, v, sz);
		cout << endl;
	}

	data_structures_groups::vector_ge *gens;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	gens->init_from_data(G->A, v, nb_gens,
			G->A_base->make_element_size, verbose_level - 1);

	if (f_v) {
		cout << "create_graph::make_Cayley_graph generators:" << endl;
		gens->print(cout);
	}



	int *Elt1;
	int *Elt2;
	algebra::ring_theory::longinteger_object go;
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
			cout << "create_graph::make_Cayley_graph "
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
		cout << "create_graph::make_Cayley_graph "
				"number of vertices = " << N << endl;
	}
	label = "Cayley_graph_" + G->A->label;
	if (f_v) {
		cout << "create_graph::init label = " << label << endl;
	}

	other::data_structures::string_tools String;
	String.chop_off_extension(label);

	label_tex = "{\\rm Cayley\\_graph\\_}" + G->A->label_tex;

}

void create_graph::load_csv_without_border(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::load_csv_without_border" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	int *M;
	int m, n;

	Fio.Csv_file_support->int_matrix_read_csv_no_border(
			fname, M, m, n, verbose_level);
	N = n;
	Adj = M;

	label = description->fname;

	other::data_structures::string_tools String;
	String.chop_off_extension(label);


	label_tex = "{\\rm File\\_}" + label;

}

void create_graph::load_dimacs(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::load_dimaccs" << endl;
	}


	combinatorics::graph_theory::graph_theory_domain GT;


	if (f_v) {
		cout << "create_graph::create_cycle "
				"before GT.load_dimacs" << endl;
	}
	GT.load_dimacs(Adj, N, fname, verbose_level);
	if (f_v) {
		cout << "create_graph::create_cycle "
				"after GT.load_dimacs" << endl;
	}



	label = fname;

	other::data_structures::string_tools String;
	String.chop_off_extension_and_path(label);


	label_tex = "{\\rm File\\_}" + label;

}

void create_graph::load_Brouwer(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::load_Brouwer" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	int nb_V;

	if (f_v) {
		cout << "create_graph::load_Brouwer "
				"before Fio.read_graph_Brouwer_format" << endl;
	}
	Fio.read_graph_Brouwer_format(
			fname,
			nb_V, Adj,
			verbose_level);
	if (f_v) {
		cout << "create_graph::load_Brouwer "
				"after Fio.read_graph_Brouwer_format" << endl;
	}

	N = nb_V;
	if (f_v) {
		cout << "create_graph::load_Brouwer "
				"N=" << N << endl;
	}

	label = description->load_Brouwer_fname;

	other::data_structures::string_tools String;
	String.chop_off_extension_and_path(label);


	label_tex = "{\\rm File\\_}" + label;

}

void create_graph::create_cycle(
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_cycle" << endl;
	}

	combinatorics::graph_theory::graph_theory_domain GT;


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


	combinatorics::graph_theory::graph_theory_domain GT;


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
	label_tex = "{\\rm Inversion\\_}" + std::to_string(n);


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

	combinatorics::graph_theory::graph_theory_domain GT;


	if (f_v) {
		cout << "create_graph::create_Hamming "
				"before GT.make_Hamming_graph" << endl;
	}
	GT.make_Hamming_graph(Adj, N, n, q, verbose_level);
	if (f_v) {
		cout << "create_graph::create_Hamming "
				"after GT.make_Hamming_graph" << endl;
	}

	label = "Hamming_" + std::to_string(n)
			+ "_" + std::to_string(q);
	label_tex = "{\\rm Hamming\\_" + std::to_string(n)
			+ "\\_" + std::to_string(q) + "}";


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

	combinatorics::graph_theory::graph_theory_domain GT;


	if (f_v) {
		cout << "create_graph::create_Johnson "
				"before GT.make_Johnson_graph" << endl;
	}
	GT.make_Johnson_graph(Adj, N, n, k, s, verbose_level);
	if (f_v) {
		cout << "create_graph::create_Johnson "
				"after GT.make_Johnson_graph" << endl;
	}

	label = "Johnson_" + std::to_string(n)
			+ "_" + std::to_string(k)
			+ "_" + std::to_string(s);
	label_tex = "{\\rm Johnson\\_" + std::to_string(n)
			+ "\\_" + std::to_string(k)
			+ "\\_" + std::to_string(s) + "}";


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


	combinatorics::graph_theory::graph_theory_domain GT;
	algebra::field_theory::finite_field *Fq;

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
	label_tex = "{\\rm Paley\\_" + std::to_string(Fq->q) + "}";


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
	algebra::number_theory::number_theory_domain NT;



	l = NT.Legendre(p, q, 0);
	if (f_v) {
		cout << "create_graph::create_Sarnak "
				"Legendre(" << p << ", " << q << ")=" << l << endl;
	}


	algebra::field_theory::finite_field *F;
	actions::action *A;
	int f_semilinear = false;
	int f_basis = true;

	F = NEW_OBJECT(algebra::field_theory::finite_field);

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
		A->Known_groups->init_projective_special_group(
				2, F,
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
		A->Known_groups->init_projective_group(
				2, F,
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
	label_tex = "{\\rm Sarnak\\_" + std::to_string(p) + "\\_" + std::to_string(q) + "}";

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

	combinatorics::graph_theory::graph_theory_domain GT;
	algebra::field_theory::finite_field *F;

	F = Get_finite_field(label_Fq);

	if (f_v) {
		cout << "create_graph::create_Schlaefli "
				"before GT.make_Schlaefli_graph" << endl;
	}
	GT.make_Schlaefli_graph(
			Adj, N, F, verbose_level);
	if (f_v) {
		cout << "create_graph::create_Schlaefli "
				"after GT.make_Schlaefli_graph" << endl;
	}

	label = "Schlaefli_" + std::to_string(F->q);
	label_tex = "{\\rm Schlaefli\\_" + std::to_string(F->q) + "}";


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

	A = NEW_OBJECT(actions::action);
	A->Known_groups->init_symmetric_group(n, verbose_level);
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
		A->Group_element->make_element(
				gens_G->ith(i), v, 0 /* verbose_level */);
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
		A->Group_element->make_element(
				gens_S->ith(i), v, 0 /* verbose_level */);
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
	label_tex = "{\\rm Shrikhande}";


	FREE_int(v);
	FREE_OBJECT(gens_G);
	FREE_OBJECT(gens_S);

	if (f_v) {
		cout << "create_graph::create_Shrikhande done" << endl;
	}
}

void create_graph::create_Winnie_Li(
		std::string &label_Fq, int index,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_Winnie_Li" << endl;
	}

	combinatorics::graph_theory::graph_theory_domain GT;
	algebra::field_theory::finite_field *Fq;

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


	label = "Winnie_Li_" + std::to_string(Fq->q)
			+ "_" + std::to_string(index);
	label_tex = "{\\rm Winnie\\_Li\\_" + std::to_string(Fq->q)
			+ "\\_" + std::to_string(index) + "}";



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


	combinatorics::graph_theory::graph_theory_domain GT;
	algebra::field_theory::finite_field *F;

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


	label = "Grassmann_" + std::to_string(n)
			+ "_" + std::to_string(k)
			+ " " + std::to_string(F->q)
			+ "_" + std::to_string(r);
	label_tex = "{\\rm Grassmann\\_" + std::to_string(n)
			+ "\\_" + std::to_string(k)
			+ "\\_" + std::to_string(F->q)
			+ "\\_" + std::to_string(r) + "}";


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

	combinatorics::graph_theory::graph_theory_domain GT;

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

	label_tex = "{\\rm " + OA->O->label_tex + "\\_coll\\_" + set_of_points_label + "}";

	if (f_v) {
		cout << "create_graph::create_coll_orthogonal done" << endl;
	}
}

void create_graph::create_affine_polar(
		std::string &orthogonal_space_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::create_affine_polar" << endl;
	}

	combinatorics::graph_theory::graph_theory_domain GT;

	orthogonal_geometry_applications::orthogonal_space_with_action *OA;

	OA = Get_orthogonal_space(orthogonal_space_label);



	if (f_v) {
		cout << "create_graph::create_affine_polar before "
				"OA->O->Quadratic_form->make_affine_polar_graph" << endl;
	}
	OA->O->Quadratic_form->make_affine_polar_graph(
			Adj, N,
			verbose_level);
	if (f_v) {
		cout << "create_graph::create_affine_polar after "
				"OA->O->Quadratic_form->make_affine_polar_graph" << endl;
	}

	label = "affine_polar_" + OA->O->label_txt;

	label_tex = "{\\rm affine\\_polar\\_" + OA->O->label_tex + "}";


	if (f_v) {
		cout << "create_graph::create_affine_polar done" << endl;
	}
}

void create_graph::make_tritangent_plane_disjointness_graph(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_tritangent_plane_disjointness_graph" << endl;
	}

	combinatorics::graph_theory::graph_theory_domain GT;

	GT.make_tritangent_plane_disjointness_graph(
			Adj, N,
			verbose_level);

	label = "tritangent_planes_disjointness";
	label_tex = "{\\rm tritangent\\_planes\\_disjointness}";

}

void create_graph::make_trihedral_pair_disjointness_graph(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_trihedral_pair_disjointness_graph" << endl;
	}


	combinatorics::graph_theory::graph_theory_domain GT;

	GT.make_trihedral_pair_disjointness_graph(
			Adj, N,
			verbose_level);


	label = "trihedral_pair_disjointness";
	label_tex= "{\\rm trihedral\\_pair\\_disjointness}";


}

void create_graph::make_orbital_graph(
		std::string &group_label, int orbit_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_orbital_graph" << endl;
	}

	groups::any_group *AG;

	AG = Get_any_group(group_label);


	poset_classification::poset_classification_control *Control;

	Control = NEW_OBJECT(poset_classification::poset_classification_control);


	poset_classification::poset_classification *PC;

	orbits::orbits_global Orbits;

	if (f_v) {
		cout << "create_graph::make_orbital_graph "
				"before Orbits.orbits_on_subsets" << endl;
	}
	Orbits.orbits_on_subsets(
			AG, Control, PC, 2, verbose_level);
	if (f_v) {
		cout << "create_graph::make_orbital_graph "
				"after Orbits.orbits_on_subsets" << endl;
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

	label = "Group_" + AG->A->label
			+ "_Orbital_" + std::to_string(orbit_idx);
	label_tex = "{\\rm Group\\_" + AG->A->label_tex
			+ "\\_Orbital\\_" + std::to_string(orbit_idx) + "}";

	if (f_v) {
		cout << "create_graph::make_orbital_graph done" << endl;
	}
}

void create_graph::make_disjoint_sets_graph(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_disjoint_sets_graph" << endl;
	}

	combinatorics::graph_theory::graph_theory_domain GT;


	if (f_v) {
		cout << "create_graph::make_disjoint_sets_graph "
				"before GT.make_disjoint_sets_graph" << endl;
	}
	GT.make_disjoint_sets_graph(
			Adj, N,
			fname,
			verbose_level);
	if (f_v) {
		cout << "create_graph::make_disjoint_sets_graph "
				"after GT.make_disjoint_sets_graph" << endl;
	}

	string L;
	other::data_structures::string_tools String;

	L = fname;
	String.chop_off_extension(L);
	L += "_disjoint_sets";

	label = L;

	L = fname;
	String.chop_off_extension(L);
	L += "\\_disjoint\\_sets";

	label_tex = "{\\rm " + L + "}";



	if (f_v) {
		cout << "create_graph::make_disjoint_sets_graph done" << endl;
	}

}



void create_graph::make_collinearity_graph(
		std::string &matrix_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_collinearity_graph" << endl;
	}

	int *Inc;
	int nb_rows, nb_cols;

	Get_matrix(matrix_label, Inc, nb_rows, nb_cols);

	combinatorics::graph_theory::graph_theory_domain Graph;


	Graph.make_collinearity_graph(
			Adj, N,
			Inc, nb_rows, nb_cols,
			verbose_level);

	FREE_int(Inc);

	label = "collinearity_graph";
	label_tex = "{\\rm collinearity\\_graph}";

	if (f_v) {
		cout << "create_graph::make_collinearity_graph done" << endl;
	}
}

void create_graph::make_chain_graph(
		std::string &partition1,
		std::string &partition2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_chain_graph" << endl;
	}

	int *part1;
	int sz1;
	int *part2;
	int sz2;

	Get_int_vector_from_label(
			description->chain_graph_partition_1,
			part1, sz1, 0 /* verbose_level*/);
	Get_int_vector_from_label(
			description->chain_graph_partition_2,
			part2, sz2, 0 /* verbose_level*/);


	combinatorics::graph_theory::graph_theory_domain Graph;


	Graph.make_chain_graph(
			Adj, N,
			part1, sz1,
			part2, sz2,
			verbose_level);

	FREE_int(part1);
	FREE_int(part2);

	label = "chain_graph";
	label_tex = "{\\rm chain\\_graph}";

	if (f_v) {
		cout << "create_graph::make_chain_graph done" << endl;
	}
}

void create_graph::make_Neumaier_graph_16(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_Neumaier_graph_16" << endl;
	}

	combinatorics::graph_theory::graph_theory_domain Graph;


	Graph.make_Neumaier_graph_16(
			Adj, N,
			verbose_level);

	label = "Neumaier_graph_16";
	label_tex = "{\\rm Neumaier\\_graph\\_16}";

}

void create_graph::make_Neumaier_graph_25(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_Neumaier_graph_25" << endl;
	}

	combinatorics::graph_theory::graph_theory_domain Graph;


	Graph.make_Neumaier_graph_25(
			Adj, N,
			verbose_level);

	label = "Neumaier_graph_25";
	label_tex = "{\\rm Neumaier\\_graph\\_25}";

}



void create_graph::make_adjacency_bitvector(
		std::string &data_text, int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_graph::make_adjacency_bitvector" << endl;
		cout << "create_graph::make_adjacency_bitvector n=" << n << endl;
	}


	combinatorics::graph_theory::graph_theory_domain Graph;
	int *v;
	int sz;

	Int_vec_scan(data_text, v, sz);

	N = n;

	if (sz != (N * (N - 1)) >> 1) {
		cout << "the data length is incorrect" << endl;
		exit(1);
	}


	Graph.make_adjacency_bitvector(
			Adj, v, N,
			verbose_level);

	label = "bitvector_" + std::to_string(N);
	label_tex = "{\\rm bitvector\\_" + std::to_string(N) + "}";

	FREE_int(v);

	if (f_v) {
		cout << "create_graph::make_adjacency_bitvector done" << endl;
	}
}


}}}
