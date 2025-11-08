/*
 * nauty_interface_for_graphs.cpp
 *
 *  Created on: Aug 25, 2024
 *      Author: betten
 */





#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace interfaces {


nauty_interface_for_graphs::nauty_interface_for_graphs()
{
	Record_birth();

}

nauty_interface_for_graphs::~nauty_interface_for_graphs()
{
	Record_death();

}

actions::action *nauty_interface_for_graphs::create_automorphism_group_of_colored_graph_object(
		combinatorics::graph_theory::colored_graph *CG,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int verbose_level)
// called from
// graph_theory_apps::automorphism_group
{
	int f_v = (verbose_level >= 1);
	actions::action *A;
	int *labeling;

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_colored_graph" << endl;
	}

	labeling = NEW_int(CG->nb_points);


	A = create_automorphism_group_and_canonical_labeling_of_colored_graph(
		CG->nb_points,
		true /* f_bitvec */, CG->Bitvec, NULL /* int  *Adj */,
		CG->point_color,
		labeling,
		Nauty_control,
		verbose_level);

	FREE_int(labeling);

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_colored_graph done" << endl;
	}
	return A;
}

actions::action *nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_colored_graph_object(
		combinatorics::graph_theory::colored_graph *CG,
		int *labeling,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	actions::action *A;

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_"
				"canonical_labeling_of_colored_graph_object" << endl;
	}

	A = create_automorphism_group_and_canonical_labeling_of_colored_graph(
		CG->nb_points,
		true /* f_bitvec */, CG->Bitvec, NULL /* int  *Adj */,
		CG->point_color,
		labeling,
		Nauty_control,
		verbose_level);

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_"
				"canonical_labeling_of_colored_graph_object done" << endl;
	}
	return A;
}

actions::action *nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_colored_graph(
	int n,
	int f_bitvec,
	other::data_structures::bitvector *Bitvec,
	int *Adj,
	int *vertex_colors,
	int *labeling,
	other::l1_interfaces::nauty_interface_control *Nauty_control,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	actions::action *A;
	other::data_structures::bitvector *Adj1;
	int *parts;
	int nb_parts;
	int i, j, k, n1, N, f_on = 0, c;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_colored_graph" << endl;
	}

	other::data_structures::tally C;

	C.init(vertex_colors, n, false, 0);


	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_colored_graph "
				"nb_types = " << C.nb_types << endl;
	}

	// create a new graph with more vertices
	// to be able to represent the color classes:

	n1 = n + C.nb_types;

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_colored_graph "
				"n1 = " << n1 << endl;
	}

	N = (n1 * (n1 - 1)) >> 1;
	Adj1 = NEW_OBJECT(other::data_structures::bitvector);
	Adj1->allocate(N);

	//nb_edges = 0;
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			f_on = false;
			k = Combi.ij2k(i, j, n);
			if (f_bitvec) {
				if (Bitvec->s_i(k)) {
					f_on = true;
				}
			}
			else {
				f_on = Adj[i * n + j];
			}
			if (f_on) {
				k = Combi.ij2k(i, j, n1);
				Adj1->m_i(k, 1);
				//nb_edges++;
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
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_colored_graph "
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

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_colored_graph "
				"before create_automorphism_group_of_graph_with_partition_and_labeling" << endl;
	}
	A = create_automorphism_group_of_graph_with_partition_and_labeling(
			n1, true, Adj1, NULL,
			nb_parts, parts, labeling,
			Nauty_control,
			verbose_level);
	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_colored_graph "
				"after create_automorphism_group_of_graph_with_partition_and_labeling" << endl;
	}

	FREE_int(parts);
	FREE_OBJECT(Adj1);
	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_colored_graph done" << endl;
	}
	return A;
}


actions::action *nauty_interface_for_graphs::create_automorphism_group_of_colored_graph_ignoring_colors(
		combinatorics::graph_theory::colored_graph *CG,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int verbose_level)
// called from
// graph_theory_apps::automorphism_group
{
	int f_v = (verbose_level >= 1);
	actions::action *A;

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_colored_graph_ignoring_colors" << endl;
	}


	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_colored_graph_ignoring_colors "
				"before create_automorphism_group_of_graph_bitvec" << endl;
	}

	A = create_automorphism_group_of_graph_bitvec(
		CG->nb_points,
		CG->Bitvec,
		Nauty_control,
		verbose_level);

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_colored_graph_ignoring_colors "
				"after create_automorphism_group_of_graph_bitvec" << endl;
	}


	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_colored_graph_ignoring_colors done" << endl;
	}
	return A;
}


actions::action *nauty_interface_for_graphs::create_automorphism_group_of_graph_bitvec(
	int n,
	other::data_structures::bitvector *Bitvec,
	other::l1_interfaces::nauty_interface_control *Nauty_control,
	int verbose_level)
// called from
// hadamard_classify::init
{
	int f_v = (verbose_level >= 1);
	int parts[1];
	actions::action *A;
	int *labeling;

	parts[0] = n;
	labeling = NEW_int(n);

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph_bitvec" << endl;
	}
	A = create_automorphism_group_of_graph_with_partition_and_labeling(
			n, true, Bitvec, NULL,
			1, parts, labeling,
			Nauty_control,
			verbose_level);
	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph_bitvec done" << endl;
	}
	FREE_int(labeling);
	return A;
}

actions::action *nauty_interface_for_graphs::create_automorphism_group_of_graph_with_partition_and_labeling(
	int n,
	int f_bitvector,
	other::data_structures::bitvector *Bitvec,
	int *Adj,
	int nb_parts, int *parts,
	int *labeling,
	other::l1_interfaces::nauty_interface_control *Nauty_control,
	int verbose_level)
// labeling[n]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *partitions;
	int i, u, a;
	layer1_foundations::other::l1_interfaces::nauty_interface Nau;
	other::l1_interfaces::nauty_output *NO;


	NO = NEW_OBJECT(other::l1_interfaces::nauty_output);

	NO->N = n;

	if (f_vv) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph "
				"before NO->nauty_output_allocate" << endl;
	}

	NO->nauty_output_allocate(
			n,
			0,
			n,
			verbose_level - 2);


	partitions = NEW_int(n);

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph_with_partition_and_labeling "
				"creating partition" << endl;
	}
	Int_vec_one(partitions, n);

	u = 0;
	for (i = 0; i < nb_parts; i++) {
		a = parts[i];
		u += a;
		partitions[u - 1] = 0;
	}
	if (u != n) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph_with_partition_and_labeling "
				"partition does not add up" << endl;
		exit(1);
	}

	if (f_bitvector) {
		if (f_v) {
			cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph_with_"
					"partition_and_labeling "
					"before nauty_interface_graph_bitvec" << endl;
		}
		Nau.nauty_interface_graph_bitvec(
				n, Bitvec,
			partitions,
			Nauty_control->f_nauty_log, Nauty_control->nauty_log_fname,
			NO,
			verbose_level);
	}
	else {
		Nau.nauty_interface_graph_int(
				n, Adj,
			partitions,
			Nauty_control->f_nauty_log, Nauty_control->nauty_log_fname,
			NO,
			verbose_level);
	}

	Int_vec_copy(NO->canonical_labeling, labeling, n);


	if (f_v) {
		if (true /*(input_no % 500) == 0*/) {
			cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph_with_"
					"partition_and_labeling: "
					"The group order is = " << *NO->Ago << " = ";
			//cout << "transversal length: ";
			Int_vec_print(cout, NO->Transversal_length, NO->Base_length);
			cout << endl;
			NO->print_stats();
		}
	}


	if (f_vv) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph_with_partition_and_labeling: "
				"labeling:" << endl;
		cout << "skipped" << endl;
		//Orbiter->Int_vec.print(cout, labeling, n);
		cout << endl;
		//cout << "labeling_inv:" << endl;
		//int_vec_print(cout, labeling_inv, n);
		//cout << endl;

		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph_with_partition_and_labeling: "
				"Base:" << endl;
		Int_vec_print(cout, NO->Base, NO->Base_length);
		cout << endl;

		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph_with_partition_and_labeling: "
				"generators:" << endl;
		cout << "skipped" << endl;
		//Orbiter->Int_vec.print_integer_matrix_width(cout, NO->Aut, NO->Aut_counter, n, n, 2);
	}



	actions::action *A;


	A = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph_with_partition_and_labeling "
				"before A->Known_groups->init_permutation_group_from_nauty_output" << endl;
	}
	A->Known_groups->init_permutation_group_from_nauty_output(NO,
		verbose_level - 2);
	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph_with_partition_and_labeling "
				"after A->Known_groups->init_permutation_group_from_nauty_output" << endl;
	}

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph_with_partition_and_labeling: "
				"created action ";
		A->print_info();
		cout << endl;
	}

	FREE_OBJECT(NO);
	FREE_int(partitions);

	return A;
}



actions::action *nauty_interface_for_graphs::create_automorphism_group_of_graph(
		int *Adj, int n,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph n=" << n << endl;
	}

	int *partition;
	layer1_foundations::other::l1_interfaces::nauty_interface Nau;
	other::l1_interfaces::nauty_output *NO;


	NO = NEW_OBJECT(other::l1_interfaces::nauty_output);

	NO->N = n;

	if (f_vv) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph "
				"before NO->nauty_output_allocate" << endl;
	}

	NO->nauty_output_allocate(
			n,
			0,
			n,
			verbose_level - 2);


	partition = NEW_int(n);

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph" << endl;
	}
	Int_vec_one(partition, n);
	partition[n - 1] = 0;

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph "
				"before Nau.nauty_interface_graph_int" << endl;
	}
	Nau.nauty_interface_graph_int(
			n, Adj,
		partition,
		Nauty_control->f_nauty_log, Nauty_control->nauty_log_fname,
		NO,
		verbose_level);
	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph "
				"after Nau.nauty_interface_graph_int Ago=" << *NO->Ago << endl;
	}

	actions::action *A;

	A = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph "
				"before A->Known_groups->init_permutation_group_from_nauty_output" << endl;
	}
	A->Known_groups->init_permutation_group_from_nauty_output(NO,
		verbose_level - 2);
	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph "
				"after A->Known_groups->init_permutation_group_from_nauty_output" << endl;
	}

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph: "
				"created action ";
		A->print_info();
		cout << endl;
	}

	FREE_OBJECT(NO);

	FREE_int(partition);
	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_of_graph done" << endl;
	}
	return A;
}


actions::action *nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph(
		int *Adj, int n, int *labeling,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int verbose_level)
// labeling[n]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph "
				"n=" << n << endl;
	}

	int *partition;
	//longinteger_object Ago;
	//int i;
	layer1_foundations::other::l1_interfaces::nauty_interface Nau;
	other::l1_interfaces::nauty_output *NO;


	NO = NEW_OBJECT(other::l1_interfaces::nauty_output);

	NO->N = n;

	if (f_vv) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph "
				"before NO->nauty_output_allocate" << endl;
	}

	NO->nauty_output_allocate(
			n,
			0,
			n,
			verbose_level - 2);

	if (f_vv) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph "
				"after NO->nauty_output_allocate" << endl;
	}

	partition = NEW_int(n);

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph "
				"initializing partition" << endl;
	}
	Int_vec_one(partition, n);
	partition[n - 1] = 0;

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph "
				"before Nau.nauty_interface_graph_int" << endl;
	}
	Nau.nauty_interface_graph_int(
			n, Adj,
		partition,
		Nauty_control->f_nauty_log, Nauty_control->nauty_log_fname,
		NO,
		verbose_level);
	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph "
				"after Nau.nauty_interface_graph_int" << endl;
	}


	Int_vec_copy(NO->canonical_labeling, labeling, n);


	actions::action *A;

	A = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph "
				"before A->Known_groups->init_permutation_group_from_nauty_output" << endl;
	}
	A->Known_groups->init_permutation_group_from_nauty_output(NO,
		verbose_level - 2);
	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph "
				"after A->Known_groups->init_permutation_group_from_nauty_output" << endl;
	}

	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph: "
				"created action ";
		A->print_info();
		cout << endl;
	}

	FREE_OBJECT(NO);

	FREE_int(partition);
	if (f_v) {
		cout << "nauty_interface_for_graphs::create_automorphism_group_and_canonical_labeling_of_graph "
				"done" << endl;
	}
	return A;
}



}}}


