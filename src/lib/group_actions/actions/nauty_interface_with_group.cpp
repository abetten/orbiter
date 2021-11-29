/*
 * nauty_interface_with_group.cpp
 *
 *  Created on: Feb 18, 2019
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {


nauty_interface_with_group::nauty_interface_with_group()
{

}

nauty_interface_with_group::~nauty_interface_with_group()
{

}

action *nauty_interface_with_group::create_automorphism_group_of_colored_graph_object(
		colored_graph *CG, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	int *labeling;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_colored_graph" << endl;
		}

	labeling = NEW_int(CG->nb_points);


	A = create_automorphism_group_and_canonical_labeling_of_colored_graph(
		CG->nb_points,
		TRUE /* f_bitvec */, CG->Bitvec, NULL /* int  *Adj */,
		CG->point_color,
		labeling,
		verbose_level);

	FREE_int(labeling);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_colored_graph done" << endl;
		}
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_colored_graph_object(
		colored_graph *CG, int *labeling, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_"
				"canonical_labeling_of_colored_graph_object" << endl;
		}

	A = create_automorphism_group_and_canonical_labeling_of_colored_graph(
		CG->nb_points,
		TRUE /* f_bitvec */, CG->Bitvec, NULL /* int  *Adj */,
		CG->point_color,
		labeling,
		verbose_level);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_"
				"canonical_labeling_of_colored_graph_object done" << endl;
		}
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_colored_graph(
	int n, int f_bitvec, bitvector *Bitvec, int *Adj,
	int *vertex_colors,
	int *labeling,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	bitvector *Adj1;
	int *parts;
	int nb_parts;
	int i, j, k, n1, N, f_on = 0, c, nb_edges;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_"
				"and_canonical_labeling_of_colored_graph" << endl;
	}

	tally C;

	C.init(vertex_colors, n, FALSE, 0);


	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_"
				"and_canonical_labeling_of_colored_graph "
				"nb_types = " << C.nb_types << endl;
	}


	n1 = n + C.nb_types;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_"
				"and_canonical_labeling_of_colored_graph "
				"n1 = " << n1 << endl;
	}

	N = (n1 * (n1 - 1)) >> 1;
	Adj1 = NEW_OBJECT(bitvector);
	Adj1->allocate(N);

	nb_edges = 0;
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			f_on = FALSE;
			k = Combi.ij2k(i, j, n);
			if (f_bitvec) {
				if (Bitvec->s_i(k)) {
					f_on = TRUE;
				}
			}
			else {
				f_on = Adj[i * n + j];
			}
			if (f_on) {
				k = Combi.ij2k(i, j, n1);
				Adj1->m_i(k, 1);
				nb_edges++;
			}
		}
	}
	for (i = 0; i < n; i++) {
		c = C.class_of(i);
		j = n + c;
		k = Combi.ij2k(i, j, n1);
		Adj1->m_i(k, 1);
	}


	nb_parts = 1 + C.nb_types;
	parts = NEW_int(nb_parts);
	parts[0] = n;
	for (i = 0; i < C.nb_types; i++) {
		parts[1 + i] = 1;
	}


#if 0
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_"
				"and_canonical_labeling_of_colored_graph "
				"before create_automorphism_group_of_graph_with_"
				"partition_and_labeling" << endl;
		cout << "nb_edges=" << nb_edges << endl;
		cout << "extended adjacency matrix:" << endl;
		int a;
		for (i = 0; i < n1; i++) {
			for (j = 0; j < n1; j++) {
				if (i == j) {
					a = 0;
				}
				else if (i < j) {
					k = Combi.ij2k(i, j, n1);
					a = bitvector_s_i(Adj1, k);
				}
				else {
					k = Combi.ij2k(j, i, n1);
					a = bitvector_s_i(Adj1, k);
				}
				cout << a << " ";
			}
			cout << endl;
		}
	}
#endif

	A = create_automorphism_group_of_graph_with_partition_and_labeling(
			n1, TRUE, Adj1, NULL,
			nb_parts, parts, labeling, verbose_level);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_"
				"and_canonical_labeling_of_colored_graph done" << endl;
	}

	FREE_int(parts);
	FREE_OBJECT(Adj1);
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_of_graph_bitvec(
	int n, bitvector *Bitvec,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int parts[1];
	action *A;
	int *labeling;

	parts[0] = n;
	labeling = NEW_int(n);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph_bitvec" << endl;
	}
	A = create_automorphism_group_of_graph_with_partition_and_labeling(
			n, TRUE, Bitvec, NULL,
			1, parts, labeling, verbose_level);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph_bitvec done" << endl;
	}
	FREE_int(labeling);
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_of_graph_with_partition_and_labeling(
	int n,
	int f_bitvector, bitvector *Bitvec, int *Adj,
	int nb_parts, int *parts,
	int *labeling,
	int verbose_level)
// labeling[n]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *partitions;
	int i, u, a;
	nauty_interface Nau;
	nauty_output *NO;


	NO = NEW_OBJECT(nauty_output);

	NO->N = n;

	if (f_vv) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph "
				"before NO->allocate" << endl;
	}

	NO->allocate(n, verbose_level - 2);


	partitions = NEW_int(n);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph_with_partition_and_labeling "
				"creating partition" << endl;
	}
	for (i = 0; i < n; i++) {
		partitions[i] = 1;
	}
	u = 0;
	for (i = 0; i < nb_parts; i++) {
		a = parts[i];
		u += a;
		partitions[u - 1] = 0;
	}
	if (u != n) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph_with_partition_and_labeling "
				"partition does not add up" << endl;
		exit(1);
	}

	if (f_bitvector) {
		if (f_v) {
			cout << "nauty_interface_with_group::create_automorphism_group_of_graph_with_"
					"partition_and_labeling "
					"before nauty_interface_graph_bitvec" << endl;
		}
		Nau.nauty_interface_graph_bitvec(n, Bitvec,
			//labeling,
			partitions,
			NO,
			verbose_level);
	}
	else {
		Nau.nauty_interface_graph_int(n, Adj,
			//labeling,
			partitions,
			NO,
			verbose_level);
	}

	for (i = 0; i < n; i++) {
		labeling[i] = NO->canonical_labeling[i];
	}


	if (f_v) {
		if (TRUE /*(input_no % 500) == 0*/) {
			cout << "nauty_interface_with_group::create_automorphism_group_of_graph_with_"
					"partition_and_labeling: "
					"The group order is = " << *NO->Ago << " = ";
			//cout << "transversal length: ";
			Orbiter->Int_vec.print(cout, NO->Transversal_length, NO->Base_length);
			cout << endl;
			NO->print_stats();
		}
	}


#if 0
	for (i = 0; i < n; i++) {
		j = labeling[i];
		labeling_inv[j] = i;
	}
#endif

	if (f_vv) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph_with_partition_and_labeling: "
				"labeling:" << endl;
		cout << "skipped" << endl;
		//Orbiter->Int_vec.print(cout, labeling, n);
		cout << endl;
		//cout << "labeling_inv:" << endl;
		//int_vec_print(cout, labeling_inv, n);
		//cout << endl;

		cout << "nauty_interface_with_group::create_automorphism_group_of_graph_with_partition_and_labeling: "
				"Base:" << endl;
		Orbiter->Int_vec.print(cout, NO->Base, NO->Base_length);
		cout << endl;

		cout << "nauty_interface_with_group::create_automorphism_group_of_graph_with_partition_and_labeling: "
				"generators:" << endl;
		cout << "skipped" << endl;
		//Orbiter->Int_vec.print_integer_matrix_width(cout, NO->Aut, NO->Aut_counter, n, n, 2);
	}



	action *A;


	A = NEW_OBJECT(action);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph_with_partition_and_labeling "
				"before A->init_permutation_group_from_nauty_output" << endl;
	}
	A->init_permutation_group_from_nauty_output(NO,
		verbose_level - 2);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph_with_partition_and_labeling "
				"after A->init_permutation_group_from_nauty_output" << endl;
	}

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph_with_partition_and_labeling: "
				"created action ";
		A->print_info();
		cout << endl;
	}

	FREE_OBJECT(NO);
	FREE_int(partitions);

	return A;
}



action *nauty_interface_with_group::create_automorphism_group_of_graph(
		int *Adj, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph n=" << n << endl;
	}

	int *partition;
	int i;
	nauty_interface Nau;
	nauty_output *NO;


	NO = NEW_OBJECT(nauty_output);

	NO->N = n;

	if (f_vv) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph "
				"before NO->allocate" << endl;
	}

	NO->allocate(n, verbose_level - 2);


	partition = NEW_int(n);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph" << endl;
	}
	for (i = 0; i < n; i++) {
		partition[i] = 1;
	}
	partition[n - 1] = 0;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph "
				"before Nau.nauty_interface_graph_int" << endl;
	}
	Nau.nauty_interface_graph_int(n, Adj,
		partition,
		NO,
		verbose_level);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph "
				"after Nau.nauty_interface_graph_int Ago=" << *NO->Ago << endl;
	}

	action *A;

	A = NEW_OBJECT(action);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph "
				"before A->init_permutation_group_from_nauty_output" << endl;
	}
	A->init_permutation_group_from_nauty_output(NO,
		verbose_level - 2);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph "
				"after A->init_permutation_group_from_nauty_output" << endl;
	}

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph: "
				"created action ";
		A->print_info();
		cout << endl;
	}

	FREE_OBJECT(NO);

	FREE_int(partition);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_graph done" << endl;
	}
	return A;
}


action *nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph(
		int *Adj, int n, int *labeling,
		int verbose_level)
// labeling[n]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph "
				"n=" << n << endl;
		}

	int *partition;
	//longinteger_object Ago;
	int i;
	nauty_interface Nau;
	nauty_output *NO;


	NO = NEW_OBJECT(nauty_output);

	NO->N = n;

	if (f_vv) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph "
				"before NO->allocate" << endl;
	}

	NO->allocate(n, verbose_level - 2);

	if (f_vv) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph "
				"after NO->allocate" << endl;
	}

	partition = NEW_int(n);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph "
				"initializing partition" << endl;
	}
	for (i = 0; i < n; i++) {
		partition[i] = 1;
	}
	partition[n - 1] = 0;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph "
				"before Nau.nauty_interface_graph_int" << endl;
	}
	Nau.nauty_interface_graph_int(n, Adj,
		partition,
		NO,
		verbose_level);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph "
				"after Nau.nauty_interface_graph_int" << endl;
	}


	for (i = 0; i < n; i++) {
		labeling[i] = NO->canonical_labeling[i];
	}


	action *A;

	A = NEW_OBJECT(action);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph "
				"before A->init_permutation_group_from_nauty_output" << endl;
	}
	A->init_permutation_group_from_nauty_output(NO,
		verbose_level - 2);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph "
				"after A->init_permutation_group_from_nauty_output" << endl;
	}

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph: "
				"created action ";
		A->print_info();
		cout << endl;
	}

	FREE_OBJECT(NO);

	FREE_int(partition);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph "
				"done" << endl;
	}
	return A;
}

#if 0
action *nauty_interface_with_group::create_automorphism_group_of_block_system(
	int nb_points, int nb_blocks, int block_size, long int *Blocks,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	action *A;
	int i, j, h;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_block_system" << endl;
		}
	M = NEW_int(nb_points * nb_blocks);
	Orbiter->Int_vec.zero(M, nb_points * nb_blocks);
	for (j = 0; j < nb_blocks; j++) {
		for (h = 0; h < block_size; h++) {
			i = Blocks[j * block_size + h];
			M[i * nb_blocks + j] = 1;
			}
		}
	A = create_automorphism_group_of_incidence_matrix(
		nb_points, nb_blocks, M, verbose_level);

	FREE_int(M);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_block_system done" << endl;
		}
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_of_collection_of_two_block_systems(
	int nb_points,
	int nb_blocks1, int block_size1, long int *Blocks1,
	int nb_blocks2, int block_size2, long int *Blocks2,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	action *A;
	int i, j, h;
	int nb_cols;
	int nb_rows;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_collection_"
				"of_two_block_systems" << endl;
		}
	nb_cols = nb_blocks1 + nb_blocks2 + 1;
	nb_rows = nb_points + 2;

	M = NEW_int(nb_rows * nb_cols);
	Orbiter->Int_vec.zero(M, nb_rows * nb_cols);

	// first system:
	for (j = 0; j < nb_blocks1; j++) {
		for (h = 0; h < block_size1; h++) {
			i = Blocks1[j * block_size1 + h];
			M[i * nb_cols + j] = 1;
			}
		i = nb_points + 0;
		M[i * nb_cols + j] = 1;
		}
	// second system:
	for (j = 0; j < nb_blocks2; j++) {
		for (h = 0; h < block_size2; h++) {
			i = Blocks2[j * block_size2 + h];
			M[i * nb_cols + nb_blocks1 + j] = 1;
			}
		i = nb_points + 1;
		M[i * nb_cols + nb_blocks1 + j] = 1;
		}
	// the extra column:
	for (i = 0; i < 2; i++) {
		M[(nb_points + i) * nb_cols + nb_blocks1 + nb_blocks2] = 1;
	}

	A = create_automorphism_group_of_incidence_matrix(
		nb_rows, nb_cols, M, verbose_level);

	FREE_int(M);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_collection_"
				"of_two_block_systems done" << endl;
		}
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_of_incidence_matrix(
	int m, int n, int *Mtx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_inc;
	int *X;
	action *A;
	int i, j, h;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_matrix" << endl;
		}
	nb_inc = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (Mtx[i * n + j]) {
				nb_inc++;
				}
			}
		}
	X = NEW_int(nb_inc);
	h = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (Mtx[i * n + j]) {
				X[h++] = i * n + j;
				}
			}
		}
	A = create_automorphism_group_of_incidence_structure_low_level(
		m, n, nb_inc, X, verbose_level);

	FREE_int(X);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_matrix done" << endl;
		}
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_of_incidence_structure(
	incidence_structure *Inc,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	int m, n, nb_inc;
	int *X;
	int *data;
	int nb;
	int i, j, h, a;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_structure" << endl;
		}
	m = Inc->nb_points();
	n = Inc->nb_lines();
	nb_inc = Inc->get_nb_inc();
	X = NEW_int(nb_inc);
	data = NEW_int(n);
	h = 0;
	for (i = 0; i < m; i++) {
		nb = Inc->get_lines_on_point(data, i, 0 /* verbose_level */);
		for (j = 0; j < nb; j++) {
			a = data[j];
			X[h++] = i * m + a;
			}
		}
	if (h != nb_inc) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_structure "
				"h != nb_inc" << endl;
		exit(1);
		}

	A = create_automorphism_group_of_incidence_structure_low_level(
		m, n, nb_inc, X,
		verbose_level - 1);

	FREE_int(X);
	FREE_int(data);
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_of_incidence_structure_low_level(
	int m, int n, int nb_inc, int *X,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *partition;
	int i;
	action *A;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_"
				"of_incidence_structure_low_level" << endl;
		}

	partition = NEW_int(m + n);
	for (i = 0; i < m + n; i++) {
		partition[i] = 1;
		}

	partition[m - 1] = 0;

	A = create_automorphism_group_of_incidence_structure_with_partition(
			m, n, nb_inc, X, partition,
			verbose_level);

	FREE_int(partition);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_"
				"of_incidence_structure_low_level done" << endl;
		}
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_of_incidence_structure_with_partition(
	int m, int n, int nb_inc, int *X, int *partition,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_v10 = (verbose_level >= 10);
	int *labeling; //, *labeling_inv;
	int *Aut;
	int *Base, *Transversal_length;
	long int *Base_lint;
	int Aut_counter = 0, Base_length = 0;
	longinteger_object Ago;
	nauty_interface Nau;


	//m = # rows
	//n = # cols

	Aut = NEW_int((m+n) * (m+n));
	Base = NEW_int(m+n);
	Base_lint = NEW_lint(m+n);
	Transversal_length = NEW_int(m + n);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_"
				"incidence_structure_with_partition" << endl;
		}

	labeling = NEW_int(m + n);
	//labeling_inv = NEW_int(m + n);

	Nau.nauty_interface_int(m, n, X, nb_inc,
		labeling, partition,
		Aut, Aut_counter,
		Base, Base_length,
		Transversal_length, Ago);

	if (f_v) {
		if (TRUE /*(input_no % 500) == 0*/) {
			cout << "nauty_interface_with_group::create_automorphism_group_of_"
					"incidence_structure_with_partition: "
					"The group order is = " << Ago << endl;
			}
		}

	Orbiter->Int_vec.copy_to_lint(Base, Base_lint, Base_length);

#if 0
	for (i = 0; i < m + n; i++) {
		j = labeling[i];
		labeling_inv[j] = i;
		}
#endif

	if (f_v10) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_"
				"incidence_structure_with_partition: "
				"labeling:" << endl;
		Orbiter->Int_vec.print(cout, labeling, m + n);
		cout << endl;
		//cout << "labeling_inv:" << endl;
		//int_vec_print(cout, labeling_inv, m + n);
		//cout << endl;

		cout << "nauty_interface_with_group::create_automorphism_group_of_"
				"incidence_structure_with_partition: "
				"Base:" << endl;
		Orbiter->Lint_vec.print(cout, Base_lint, Base_length);
		cout << endl;

		cout << "nauty_interface_with_group::create_automorphism_group_of_"
				"incidence_structure_with_partition: "
				"generators:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Aut, Aut_counter, m + n, m + n, 2);
		}



	action *A;
	longinteger_object ago;


	A = NEW_OBJECT(action);

	Ago.assign_to(ago);
	//ago.create(Ago, __FILE__, __LINE__);
	A->init_permutation_group_from_generators(m + n,
		TRUE, ago,
		Aut_counter, Aut,
		Base_length, Base_lint,
		verbose_level - 2);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_"
				"incidence_structure_with_partition: "
				"created action ";
		A->print_info();
		cout << endl;
		}

	FREE_int(Aut);
	FREE_int(Base);
	FREE_lint(Base_lint);
	FREE_int(Transversal_length);
	FREE_int(labeling);

	return A;
}

void nauty_interface_with_group::test_self_dual_self_polar(int input_no,
	int m, int n, int nb_inc, int *X,
	int &f_self_dual, int &f_self_polar,
	int verbose_level)
{
	int M, N, i, j, h, Nb_inc, a;
	int *Mtx, *Y;

	if (m != n) {
		f_self_dual = FALSE;
		f_self_polar = FALSE;
		return;
		}
	M = 2 * m;
	N = 2 + nb_inc;
	Mtx = NEW_int(M * N);
	Y = NEW_int(M * N);
	for (i = 0; i < M * N; i++) {
		Mtx[i] = 0;
		}
	for (i = 0; i < m; i++) {
		Mtx[i * N + 0] = 1;
		}
	for (i = 0; i < m; i++) {
		Mtx[(m + i) * N + 1] = 1;
		}
	for (h = 0; h < nb_inc; h++) {
		a = X[h];
		i = a / n;
		j = a % n;
		Mtx[i * N + 2 + h] = 1;
		Mtx[(m + j) * N + 2 + h] = 1;
		}
	Nb_inc = 0;
	for (i = 0; i < M * N; i++) {
		if (Mtx[i]) {
			Y[Nb_inc++] = i;
			}
		}

	do_self_dual_self_polar(input_no,
			M, N, Nb_inc, Y, f_self_dual, f_self_polar,
			verbose_level - 1);

	FREE_int(Mtx);
	FREE_int(Y);
}


void nauty_interface_with_group::do_self_dual_self_polar(int input_no,
	int m, int n, int nb_inc, int *X,
	int &f_self_dual, int &f_self_polar,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *labeling; //, *labeling_inv;
	int *Aut;
	int *Base, *Transversal_length, *partitions;
	long int *Base_lint;
	int Aut_counter = 0, Base_length = 0;
	longinteger_object Ago;
	int i; //, j;
	nauty_interface Nau;

	//m = # rows
	//n = # cols

	if (ODD(m)) {
		f_self_dual = f_self_polar = FALSE;
		return;
		}
	Aut = NEW_int((m+n) * (m+n));
	Base = NEW_int(m+n);
	Base_lint = NEW_lint(m+n);
	Transversal_length = NEW_int(m + n);
	partitions = NEW_int(m + n);

	if (f_v) {
		if ((input_no % 500) == 0) {
			cout << "nauty_interface_with_group::do_self_dual_self_polar input_no=" << input_no << endl;
			}
		}
	for (i = 0; i < m + n; i++) {
		partitions[i] = 1;
		}

#if 0
	for (i = 0; i < PB.P.ht; i++) {
		j = PB.P.startCell[i] + PB.P.cellSize[i] - 1;
		partitions[j] = 0;
		}
#endif

#if 0
	j = 0;
	for (i = 0; i < nb_row_parts; i++) {
		l = row_parts[i];
		partitions[j + l - 1] = 0;
		j +=l;
		}
	for (i = 0; i < nb_col_parts; i++) {
		l = col_parts[i];
		partitions[j + l - 1] = 0;
		j +=l;
		}
#endif

	labeling = NEW_int(m + n);
	//labeling_inv = NEW_int(m + n);

	Nau.nauty_interface_int(m, n, X, nb_inc,
			labeling, partitions, Aut, Aut_counter,
			Base, Base_length, Transversal_length, Ago);

	if (f_vv) {
		if ((input_no % 500) == 0) {
			cout << "The group order is = " << Ago << endl;
			}
		}

	Orbiter->Int_vec.copy_to_lint(Base, Base_lint, Base_length);

#if 0
	for (i = 0; i < m + n; i++) {
		j = labeling[i];
		labeling_inv[j] = i;
		}
#endif

	int *aut;
	int *p_aut;
	int h, a, b, c, m_half;

	m_half = m >> 1;
	aut = NEW_int(Aut_counter * m);
	for (h = 0; h < Aut_counter; h++) {
		for (i = 0; i < m; i++) {
			aut[h * m + i] = Aut[h * (m + n) + i];
			}
		}
	f_self_dual = FALSE;
	f_self_polar = FALSE;
	for (h = 0; h < Aut_counter; h++) {
		p_aut = aut + h * m;

		a = p_aut[0];
		if (a >= m_half ) {
			f_self_dual = TRUE;
			if (f_v) {
				cout << "no " << input_no << " is self dual" << endl;
				}
			break;
			}
		}

#if 0

	int *AUT;
	int *BASE;

	AUT = NEW_int(Aut_counter * (m + n));
	BASE = NEW_int(Base_length);
	for (h = 0; h < Aut_counter; h++) {
		for (i = 0; i < m + n; i++) {
			j = labeling_inv[i];
			j = Aut[h * (m + n) + j];
			j = labeling[j];
			AUT[h * (m + 1) + i] = j;
			}
		}
	for (i = 0; i < Base_length; i++) {
		j = Base[i];
		j = labeling[j];
		BASE[i] = j;
		}
#endif

	action A;
	longinteger_object ago;



	Ago.assign_to(ago);

	//ago.create(Ago, __FILE__, __LINE__);
	A.init_permutation_group_from_generators(m + n,
		TRUE, ago,
		Aut_counter, Aut,
		Base_length, Base_lint,
		verbose_level);

	cout << "created action ";
	A.print_info();
	cout << endl;


	if (f_self_dual) {


		sims *S;
		longinteger_object go;
		int goi;
		int *Elt;

		S = A.Sims;
		S->group_order(go);
		goi = go.as_int();
		Elt = NEW_int(A.elt_size_in_int);

		cout << "the group order is: " << goi << endl;
		for (i = 0; i < goi; i++) {
			S->element_unrank_lint(i, Elt);
			if (Elt[0] < m_half) {
				continue; // not a duality
				}

			for (a = 0; a < m_half; a++) {
				b = Elt[a];
				c = Elt[b];
				if (c != a)
					break;
				}
			if (a == m_half) {
				cout << "found a polarity:" << endl;
				A.element_print(Elt, cout);
				cout << endl;
				f_self_polar = TRUE;
				break;
				}
			}


		FREE_int(Elt);
		}




	FREE_int(aut);
	FREE_int(Aut);
	FREE_int(Base);
	FREE_lint(Base_lint);
	FREE_int(Transversal_length);
	FREE_int(partitions);
	FREE_int(labeling);
	//FREE_int(labeling_inv);
	//FREE_int(AUT);
	//FREE_int(BASE);
}

void nauty_interface_with_group::add_configuration_graph(ofstream &g,
		int m, int n, int nb_inc, int *X, int f_first,
		int verbose_level)
{
	incidence_structure Inc;
	int *joining_table;
	int *M1;
	int i, j, h, nb_joined_pairs, nb_missing_pairs;
	int n1, nb_inc1;
	action *A;
	longinteger_object ago;
	combinatorics_domain Combi;

	A = create_automorphism_group_of_incidence_structure_low_level(
			m, n, nb_inc, X,
			verbose_level - 2);
	A->group_order(ago);

	Inc.init_by_incidences(m, n, nb_inc, X, verbose_level);
	joining_table = NEW_int(m * m);
	for (i = 0; i < m * m; i++) {
		joining_table[i] = FALSE;
		}
	nb_joined_pairs = 0;
	for (i = 0; i < m; i++) {
		for (j = i + 1; j < m; j++) {
			for (h = 0; h < n; h++) {
				if (Inc.get_ij(i, h) && Inc.get_ij(j, h)) {
					joining_table[i * m + j] = TRUE;
					joining_table[j * m + i] = TRUE;
					nb_joined_pairs++;
					}
				}
			}
		}
	nb_missing_pairs = Combi.int_n_choose_k(m, 2) - nb_joined_pairs;
	n1 = n + nb_missing_pairs;
	M1 = NEW_int(m * n1);
	for (i = 0; i < m * n1; i++) {
		M1[i] = 0;
		}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			M1[i * n1 + j] = Inc.get_ij(i, j);
			}
		}
	h = 0;
	for (i = 0; i < m; i++) {
		for (j = i + 1; j < m; j++) {
			if (joining_table[i * m + j] == FALSE) {
				M1[i * n1 + n + h] = 1;
				M1[j * n1 + n + h] = 1;
				h++;
				}
			}
		}
	if (f_first) {
		nb_inc1 = 0;
		for (i = 0; i < m; i++) {
			for (j = 0; j < n1; j++) {
				if (M1[i * n1 + j]) {
					nb_inc1++;
					}
				}
			}
		g << m << " " << n1 << " " << nb_inc1 << endl;
		}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n1; j++) {
			if (M1[i * n1 + j]) {
				g << i * n1 + j << " ";
				}
			}
		}
	g << ago << endl;

	FREE_int(joining_table);
	FREE_int(M1);
	FREE_OBJECT(A);
}
#endif


void nauty_interface_with_group::automorphism_group_as_permutation_group(
		strong_generators *&SG,
		nauty_output *NO,
		action *&A_perm,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);


	if (f_v) {
		cout << "nauty_interface_with_group::automorphism_group_as_permutation_group" << endl;
	}
	A_perm = NEW_OBJECT(action);


	if (f_v) {
		cout << "nauty_interface_with_group::automorphism_group_as_permutation_group "
				"before init_permutation_group_from_generators" << endl;
	}
	A_perm->init_permutation_group_from_nauty_output(NO,
		verbose_level - 2);
	if (f_v) {
		cout << "nauty_interface_with_group::automorphism_group_as_permutation_group "
				"after init_permutation_group_from_generators" << endl;
	}

	if (f_vv) {
		cout << "nauty_interface_with_group::automorphism_group_as_permutation_group "
				"create_automorphism_group_of_incidence_structure: created action ";
		A_perm->print_info();
		cout << endl;
	}


	if (f_v) {
		cout << "nauty_interface_with_group::automorphism_group_as_permutation_group done" << endl;
	}
}

void nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group(
		action *A_linear,
		projective_space *P,
		strong_generators *&SG,
		action *&A_perm,
		nauty_output *NO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);


	if (f_v) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group" << endl;
	}

	//action *A_perm;

	int d;

	d = A_linear->matrix_group_dimension();


	if (f_v) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group before automorphism_group_as_permutation_group" << endl;
	}

	automorphism_group_as_permutation_group(
				SG,
				NO,
				A_perm,
				verbose_level);

	if (f_v) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group after automorphism_group_as_permutation_group" << endl;
	}

	//action *A_linear;

	//A_linear = A;

	if (A_linear == NULL) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
				"A_linear == NULL" << endl;
		exit(1);
	}

	vector_ge *gens; // permutations from nauty
	vector_ge *gens1; // matrices
	int g, frobenius, pos;
	int *Mtx;
	int *Elt1;

	gens = A_perm->Strong_gens->gens;

	gens1 = NEW_OBJECT(vector_ge);
	gens1->init(A_linear, verbose_level - 2);
	gens1->allocate(gens->len, verbose_level - 2);
	Elt1 = NEW_int(A_linear->elt_size_in_int);

	Mtx = NEW_int(d * d + 1); // leave space for frobenius

	pos = 0;
	for (g = 0; g < gens->len; g++) {
		if (f_vv) {
			cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
					"strong generator " << g << ":" << endl;
			//A_perm->element_print(gens->ith(g), cout);
			cout << endl;
		}

		if (A_perm->reverse_engineer_semilinear_map(P,
			gens->ith(g), Mtx, frobenius,
			0 /*verbose_level - 2*/)) {

			Mtx[d * d] = frobenius;
			A_linear->make_element(Elt1, Mtx, 0 /*verbose_level - 2*/);
			if (f_vv) {
				cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
						"semi-linear group element:" << endl;
				A_linear->element_print(Elt1, cout);
			}
			A_linear->element_move(Elt1, gens1->ith(pos), 0);


			pos++;
		}
		else {
			if (f_vv) {
				cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
						"generator " << g << " does not "
						"correspond to a semilinear mapping" << endl;
			}
		}
	}
	gens1->reallocate(pos, verbose_level - 2);
	if (f_vv) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
				"we found " << gens1->len << " generators" << endl;
	}

	if (f_vvv) {
		gens1->print(cout);
	}


	if (f_vv) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
				"we are now testing the generators:" << endl;
	}
	int i, j1, j2;

	for (g = 0; g < gens1->len; g++) {
		if (f_vv) {
			cout << "generator " << g << ":" << endl;
		}
		//A_linear->element_print(gens1->ith(g), cout);
		for (i = 0; i < P->N_points; i++) {
			j1 = A_linear->element_image_of(i, gens1->ith(g), 0);
			j2 = A_perm->element_image_of(i, gens->ith(g), 0);
			if (j1 != j2) {
				cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
						"problem with generator: "
						"j1 != j2" << endl;
				cout << "i=" << i << endl;
				cout << "j1=" << j1 << endl;
				cout << "j2=" << j2 << endl;
				cout << endl;
				exit(1);
			}
		}
	}
	if (f_vv) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
				"the generators are OK" << endl;
	}



	sims *S;
	longinteger_object go;

	if (f_vv) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
				"before A_linear->create_sims_from_generators_with_target_group_order" << endl;
	}

	S = A_linear->create_sims_from_generators_with_target_group_order(
		gens1, *NO->Ago, 0 /*verbose_level*/);

	if (f_vv) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
				"after A_linear->create_sims_from_generators_with_target_group_order" << endl;
	}


	S->group_order(go);


	if (f_vv) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
				"Found a group of order " << go << endl;
	}
	if (f_vvv) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
				"strong generators are:" << endl;
		S->print_generators();
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
				"strong generators are (in tex):" << endl;
		S->print_generators_tex(cout);
	}


	longinteger_domain D;

	if (D.compare_unsigned(*NO->Ago, go)) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
				"the group order does not match" << endl;
		cout << "ago = " << NO->Ago << endl;
		cout << "go = " << go << endl;
		exit(1);
	}

#if 0
	if (f_v) {
		cout << "before freeing labeling" << endl;
	}
	FREE_int(labeling);
#endif

#if 0
	if (f_v) {
		cout << "before freeing A_perm" << endl;
	}
	FREE_OBJECT(A_perm);
#endif

	if (f_v) {
		cout << "not freeing gens" << endl;
	}
	//FREE_OBJECT(gens);
	if (f_v) {
		cout << "before freeing Mtx" << endl;
	}
	FREE_int(Mtx);
	FREE_int(Elt1);


	// ToDo what about gens1, should it be freed ?


	if (f_v) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
				"before initializing strong generators" << endl;
	}

	SG = NEW_OBJECT(strong_generators);
	SG->init_from_sims(S, 0 /* verbose_level*/);
	FREE_OBJECT(S);

	if (f_v) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group "
				"after initializing strong generators" << endl;
	}


	if (f_v) {
		cout << "nauty_interface_with_group::reverse_engineer_linear_group_from_permutation_group done" << endl;
	}

}


strong_generators *nauty_interface_with_group::set_stabilizer_of_object(
	object_with_canonical_form *OwCF,
	action *A_linear,
	int f_compute_canonical_form, bitvector *&Canonical_form,
	nauty_output *&NO,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);



	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}




	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object before OiP->run_nauty" << endl;

	}

	OwCF->run_nauty(
			f_compute_canonical_form, Canonical_form,
			NO,
			verbose_level);

	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object after OiP->run_nauty" << endl;

	}

	long int ago;


	ago = NO->Ago->as_lint();

	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object "
				"after OiP->run_nauty" << endl;

		cout << "nauty_interface_with_group::set_stabilizer_of_object "
				"go = " << ago << endl;

		NO->print_stats();
	}


	strong_generators *SG;
	action *A_perm;

	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object "
				"before Nauty.reverse_engineer_linear_group_from_permutation_group" << endl;
		}
	reverse_engineer_linear_group_from_permutation_group(
			A_linear,
			OwCF->P,
			SG,
			A_perm,
			NO,
			verbose_level);
	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object "
				"after Nauty.reverse_engineer_linear_group_from_permutation_group" << endl;
	}


	FREE_OBJECT(A_perm);

	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object done" << endl;
	}
	return SG;
}



}}

