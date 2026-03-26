/*
 * poset_of_orbits.cpp
 *
 *  Created on: Aug 1, 2021
 *      Author: betten
 */



#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


poset_of_orbits::poset_of_orbits()
{
	Record_birth();
	PC = NULL;

	sz = 0;
	max_set_size = 0;
	t0 = 0;


	nb_poset_orbit_nodes_used = 0;
	nb_poset_orbit_nodes_allocated = 0;
	poset_orbit_nodes_increment = 0;
	poset_orbit_nodes_increment_last = 0;

	root = NULL;
	first_poset_orbit_node_at_level = NULL;
	nb_extension_nodes_at_level_total = NULL;
	nb_extension_nodes_at_level = NULL;
	nb_fusion_nodes_at_level = NULL;
	nb_unprocessed_nodes_at_level = NULL;

	set0 = NULL;
	set1 = NULL;
	set3 = NULL;

}

poset_of_orbits::~poset_of_orbits()
{
	Record_death();
	int f_v = false; //(verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::exit_poset_orbit_node" << endl;
	}

	if (root) {
		if (f_v) {
			cout << "poset_of_orbits::exit_poset_orbit_node deleting root" << endl;
		}
		FREE_OBJECTS(root);
		if (f_v) {
			cout << "poset_of_orbits::exit_poset_orbit_node after deleting root" << endl;
		}
		root = NULL;
	}
	if (first_poset_orbit_node_at_level) {
		FREE_lint(first_poset_orbit_node_at_level);
		first_poset_orbit_node_at_level = NULL;
	}

	if (nb_extension_nodes_at_level_total) {
		FREE_lint(nb_extension_nodes_at_level_total);
		nb_extension_nodes_at_level_total = NULL;
	}
	if (nb_extension_nodes_at_level) {
		FREE_lint(nb_extension_nodes_at_level);
		nb_extension_nodes_at_level = NULL;
	}
	if (nb_fusion_nodes_at_level) {
		FREE_lint(nb_fusion_nodes_at_level);
		nb_fusion_nodes_at_level = NULL;
	}
	if (nb_unprocessed_nodes_at_level) {
		FREE_lint(nb_unprocessed_nodes_at_level);
		nb_unprocessed_nodes_at_level = NULL;
	}
	if (set0) {
		FREE_lint(set0);
		set0 = NULL;
	}
	if (set1) {
		FREE_lint(set1);
		set1 = NULL;
	}
	if (set3) {
		FREE_lint(set3);
		set3 = NULL;
	}
}


void poset_of_orbits::init(
		poset_classification *PC,
		int nb_poset_orbit_nodes,
		int sz, int max_set_size, long int t0,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::init" << endl;
	}

	poset_of_orbits::PC = PC;
	poset_of_orbits::sz = sz;
	poset_of_orbits::max_set_size = max_set_size;
	poset_of_orbits::t0 = t0;

	nb_poset_orbit_nodes_used = 0;
	nb_poset_orbit_nodes_allocated = 0;

	if (f_v) {
		cout << "poset_of_orbits::init before init_poset_orbit_node" << endl;
	}
	init_poset_orbit_node(
			nb_poset_orbit_nodes,
			verbose_level);
	if (f_v) {
		cout << "poset_of_orbits::init after init_poset_orbit_node" << endl;
	}

	if (f_v) {
		cout << "poset_of_orbits::init done" << endl;
	}
}


void poset_of_orbits::init_poset_orbit_node(
		int nb_poset_orbit_nodes, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "poset_of_orbits::init_poset_orbit_node" << endl;
	}
	root = NEW_OBJECTS(poset_orbit_node, nb_poset_orbit_nodes);
	for (i = 0; i < nb_poset_orbit_nodes; i++) {
		root[i].set_node(i);
	}
	nb_poset_orbit_nodes_allocated = nb_poset_orbit_nodes;
	nb_poset_orbit_nodes_used = 0;
	poset_orbit_nodes_increment = nb_poset_orbit_nodes;
	poset_orbit_nodes_increment_last = nb_poset_orbit_nodes;
	first_poset_orbit_node_at_level = NEW_lint(sz + 2);
	first_poset_orbit_node_at_level[0] = 0;
	first_poset_orbit_node_at_level[1] = 1;
	set0 = NEW_lint(max_set_size);
	set1 = NEW_lint(max_set_size);
	set3 = NEW_lint(max_set_size);
	nb_extension_nodes_at_level_total = NEW_lint(sz + 1);
	nb_extension_nodes_at_level = NEW_lint(sz + 1);
	nb_fusion_nodes_at_level = NEW_lint(sz + 1);
	nb_unprocessed_nodes_at_level = NEW_lint(sz + 1);
	for (i = 0; i < sz + 1; i++) {
		nb_extension_nodes_at_level_total[i] = 0;
		nb_extension_nodes_at_level[i] = 0;
		nb_fusion_nodes_at_level[i] = 0;
		nb_unprocessed_nodes_at_level[i] = 0;
	}
	if (f_v) {
		cout << "poset_of_orbits::init_poset_orbit_node done" << endl;
	}
}



void poset_of_orbits::reallocate()
{
	long int increment_new;
	long int length;
	int verbose_level = 0;

	increment_new = poset_orbit_nodes_increment + poset_orbit_nodes_increment_last;


	length = nb_poset_orbit_nodes_allocated +
			poset_orbit_nodes_increment;

	if (length > (1L << 31) - 1) {
		long int length_wanted;

		length_wanted = length;
		length = (1L << 31) - 1;
		cout << "poset_of_orbits::reallocate "
				"reducing length from " << length_wanted << " to " << length << endl;
		poset_orbit_nodes_increment = length_wanted - nb_poset_orbit_nodes_allocated;
	}
	cout << "poset_of_orbits::reallocate "
			"from " << nb_poset_orbit_nodes_allocated << " to " << length << endl;
	reallocate_to(length, verbose_level - 1);
	poset_orbit_nodes_increment_last = poset_orbit_nodes_increment;
	poset_orbit_nodes_increment = increment_new;

}

void poset_of_orbits::reallocate_to(
		long int new_number_of_nodes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	poset_orbit_node *new_root;

	if (f_v) {
		cout << "poset_of_orbits::reallocate_to" << endl;
	}
	if (new_number_of_nodes <= nb_poset_orbit_nodes_allocated) {
		cout << "poset_of_orbits::reallocate_to "
				"new_number_of_nodes <= "
				"nb_poset_orbit_nodes_allocated" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "poset_of_orbits::reallocate_to from "
				<< nb_poset_orbit_nodes_allocated
				<< " to " << new_number_of_nodes << endl;
	}
	new_root = NEW_OBJECTS(poset_orbit_node, new_number_of_nodes);
	for (i = 0; i < nb_poset_orbit_nodes_allocated; i++) {
		new_root[i] = root[i];
		root[i].null();
	}
	FREE_OBJECTS(root);
	root = new_root;
	nb_poset_orbit_nodes_allocated = new_number_of_nodes;
	if (f_v) {
		cout << "poset_of_orbits::reallocate_to done" << endl;
	}
}

int poset_of_orbits::get_max_set_size()
{
	return max_set_size;
}


long int poset_of_orbits::get_nb_poset_orbit_nodes_allocated()
{
	return nb_poset_orbit_nodes_allocated;
}

long int poset_of_orbits::get_nb_extension_nodes_at_level_total(
		int level)
{
	return nb_extension_nodes_at_level_total[level];
}

void poset_of_orbits::set_nb_poset_orbit_nodes_used(
		int value)
{
	nb_poset_orbit_nodes_used = value;
}

int poset_of_orbits::first_node_at_level(
		int i)
{
	return first_poset_orbit_node_at_level[i];
}

void poset_of_orbits::set_first_node_at_level(
		int i, int value)
{
	first_poset_orbit_node_at_level[i] = value;
}

poset_orbit_node *poset_of_orbits::get_node(
		int node_idx)
{
	return root + node_idx;
}

long int *poset_of_orbits::get_set0()
{
	return set0;
}

long int *poset_of_orbits::get_set1()
{
	return set1;
}

long int *poset_of_orbits::get_set3()
{
	return set3;
}

int poset_of_orbits::nb_orbits_at_level(
		int level)
{
	int f, l;

	f = first_poset_orbit_node_at_level[level];
	l = first_poset_orbit_node_at_level[level + 1] - f;
	return l;
}

long int poset_of_orbits::nb_flag_orbits_up_at_level(
		int level)
{
	int f, l, i;
	long int F;

	f = first_poset_orbit_node_at_level[level];
	l = nb_orbits_at_level(level);
	F = 0;
	for (i = 0; i < l; i++) {
		F += root[f + i].get_nb_of_extensions();
	}
	return F;
}

void poset_of_orbits::node_to_lvl_po(
		int node_idx, int &level, int &po)
{

	poset_orbit_node * Node;

	Node = root + node_idx;
	level = Node->depth_of_node(PC);
	po = node_idx - first_poset_orbit_node_at_level[level];
}

poset_orbit_node *poset_of_orbits::get_node_ij(
		int level, int node)
{
	int f;

	f = first_poset_orbit_node_at_level[level];
	return root + f + node;
}

int poset_of_orbits::node_get_nb_of_extensions(
		int node)
{
	return root[node].get_nb_of_extensions();
}

void poset_of_orbits::get_set(
		int node, long int *set, int &size)
{
	size = root[node].depth_of_node(PC);
	root[node].store_set_to(PC, size - 1, set);
}

void poset_of_orbits::get_set(
		int level, int orbit, long int *set, int &size)
{
	int node;

	node = first_poset_orbit_node_at_level[level] + orbit;
	size = root[node].depth_of_node(PC);
	root[node].store_set_to(PC, size - 1, set);
}

int poset_of_orbits::find_extension_from_point(
		int node_idx,
		long int pt, int verbose_level)
// a -1 means not found
{
	int i;

	for (i = 0; i < root[node_idx].get_nb_of_extensions(); i++) {
		if (root[node_idx].get_E(i)->get_pt() == pt) {
			break;
		}
	}
	if (i == root[node_idx].get_nb_of_extensions()) {
		return -1;
	}
	return i;
}

long int poset_of_orbits::count_extension_nodes_at_level(
		int lvl)
{
	int prev;

	nb_extension_nodes_at_level_total[lvl] = 0;
	for (prev = first_poset_orbit_node_at_level[lvl];
			prev < first_poset_orbit_node_at_level[lvl + 1];
			prev++) {

		nb_extension_nodes_at_level_total[lvl] +=
				root[prev].get_nb_of_extensions();

	}
	nb_unprocessed_nodes_at_level[lvl] =
			nb_extension_nodes_at_level_total[lvl];
	nb_fusion_nodes_at_level[lvl] = 0;
	nb_extension_nodes_at_level[lvl] = 0;
	return nb_extension_nodes_at_level_total[lvl];
}

double poset_of_orbits::level_progress(
		int lvl)
{
	return
		((double)(nb_fusion_nodes_at_level[lvl] +
				nb_extension_nodes_at_level[lvl])) /
			(double) nb_extension_nodes_at_level_total[lvl];
}

void poset_of_orbits::change_extension_type(
		int level,
		int node, int cur_ext, int type, int verbose_level)
{
	if (type == EXTENSION_TYPE_EXTENSION) {
		// extension node
		if (root[node].get_E(cur_ext)->get_type() != EXTENSION_TYPE_UNPROCESSED &&
			root[node].get_E(cur_ext)->get_type() != EXTENSION_TYPE_PROCESSING) {
			cout << "poset_of_orbits::change_extension_type trying to install "
					"extension node, fatal: root[node].get_E(cur_ext)->type != "
					"EXTENSION_TYPE_UNPROCESSED && root[node].get_E(cur_ext)->type "
					"!= EXTENSION_TYPE_PROCESSING" << endl;
			cout << "root[node].get_E(cur_ext)->get_type()="
					<< root[node].get_E(cur_ext)->get_type() << endl;
			exit(1);
		}
		nb_extension_nodes_at_level[level]++;
		nb_unprocessed_nodes_at_level[level]--;
		root[node].get_E(cur_ext)->set_type(EXTENSION_TYPE_EXTENSION);
	}
	else if (type == EXTENSION_TYPE_FUSION) {
		// fusion
		if (root[node].get_E(cur_ext)->get_type() != EXTENSION_TYPE_UNPROCESSED) {
			cout << "poset_of_orbits::change_extension_type trying to install "
					"fusion node, fatal: root[node].E[cur_ext].get_type() != "
					"EXTENSION_TYPE_UNPROCESSED" << endl;
			cout << "root[node].get_E(cur_ext)->get_type()="
					<< root[node].get_E(cur_ext)->get_type() << endl;
			exit(1);
		}
		nb_fusion_nodes_at_level[level]++;
		nb_unprocessed_nodes_at_level[level]--;
		root[node].get_E(cur_ext)->set_type(EXTENSION_TYPE_FUSION);
	}
}

void poset_of_orbits::get_table_of_nodes(
		long int *&Table,
		int &nb_rows, int &nb_cols, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "poset_of_orbits::get_table_of_nodes "
				"nb_poset_orbit_nodes_used="
				<< nb_poset_orbit_nodes_used << endl;
	}
	nb_rows = nb_poset_orbit_nodes_used;
	nb_cols = 6;
	Table = NEW_lint(nb_poset_orbit_nodes_used * nb_cols);

	for (i = 0; i < nb_poset_orbit_nodes_used; i++) {

		if (f_v) {
			cout << "poset_of_orbits::get_table_of_nodes "
					"node " << i
					<< " / " << nb_poset_orbit_nodes_used << endl;
		}

		Table[i * nb_cols + 0] = root[i].get_level(PC);
		Table[i * nb_cols + 1] = root[i].get_node_in_level(PC);
		Table[i * nb_cols + 2] = root[i].get_pt();

		algebra::ring_theory::longinteger_object go;

		root[i].get_stabilizer_order(PC, go);
		Table[i * nb_cols + 3] = go.as_int();
		Table[i * nb_cols + 4] = root[i].get_nb_of_live_points();
		Table[i * nb_cols + 5] = root[i].get_nb_of_orbits_under_stabilizer();
	}
	if (f_v) {
		cout << "poset_of_orbits::get_table_of_nodes done" << endl;
	}
}

int poset_of_orbits::count_live_points(
		int level,
		int node_local, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int node;
	int nb_points;

	if (f_v) {
		cout << "poset_of_orbits::count_live_points" << endl;
	}
	node = first_node_at_level(level) + node_local;
	if (!root[node].has_Schreier_vector()) {
		root[node].compute_schreier_vector(PC, level, verbose_level - 2);
	}
	nb_points = root[node].get_nb_of_live_points();

	return nb_points;
}

void poset_of_orbits::print_progress_by_level(
		int lvl)
{
	int i;

	for (i = 0; i < lvl; i++) {
		//remaining = nb_extension_nodes_at_level_total[i]
		//	- nb_extension_nodes_at_level[i] - nb_fusion_nodes_at_level[i];
		cout << setw(5) << i << " : " << setw(10)
			<< nb_extension_nodes_at_level[i] << " : "
			<< setw(10) << nb_fusion_nodes_at_level[i] << " : "
			<< setw(10) << nb_extension_nodes_at_level_total[i] << " : "
			<< setw(10) << nb_unprocessed_nodes_at_level[i];
		cout << endl;
		}
	//print_statistic_on_callbacks();
}

void poset_of_orbits::print_tree()
{
	int i;

	cout << "poset_of_orbits::print_tree "
			"nb_poset_orbit_nodes_used="
			<< nb_poset_orbit_nodes_used << endl;
	for (i = 0; i < nb_poset_orbit_nodes_used; i++) {
		PC->print_node(i);
	}
}

void poset_of_orbits::init_root_node_from_base_case(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "poset_of_orbits::init_root_node_from_base_case" << endl;
	}

	if (PC->get_Base_case() == NULL) {
		cout << "poset_of_orbits::init_root_node_from_base_case Base_case == NULL" << endl;
		exit(1);
	}

	first_poset_orbit_node_at_level[0] = 0;

	root[0].init_node(0,
			-1 /* the root node does not have an ancestor */,
			-1 /* the root node does not have a pt */,
			verbose_level);

	for (i = 0; i < PC->get_Base_case()->size; i++) {

		nb_extension_nodes_at_level_total[i] = 0;
		nb_extension_nodes_at_level[i] = 0;
		nb_fusion_nodes_at_level[i] = 0;
		nb_unprocessed_nodes_at_level[i] = 0;

		if (f_v) {
			cout << "poset_of_orbits::init_root_node_from_base_case "
					"initializing node at level " << i << endl;
		}
		first_poset_orbit_node_at_level[i + 1] =
				first_poset_orbit_node_at_level[i] + 1;
		root[i].allocate_E(1 /* nb_extensions */, verbose_level);

		root[i].get_E(0)->set_type(EXTENSION_TYPE_EXTENSION);
		root[i].get_E(0)->set_data(i + 1);

		root[i + 1].init_node(i + 1,
					i,
					PC->get_Base_case()->orbit_rep[i],
					verbose_level);
	}
	if (f_v) {
		cout << "poset_of_orbits::init_root_node_from_base_case "
				"before store_strong_generators" << endl;
	}
	root[PC->get_Base_case()->size].store_strong_generators(
			PC, PC->get_Base_case()->Stab_gens,
			verbose_level);
	if (f_v) {
		cout << "poset_of_orbits::init_root_node_from_base_case "
				"after store_strong_generators" << endl;
	}

	first_poset_orbit_node_at_level[PC->get_Base_case()->size + 1] =
			PC->get_Base_case()->size + 1;

	if (f_v) {
		cout << "i : first_poset_orbit_node_at_level[i]" << endl;
		for (i = 0; i <= PC->get_Base_case()->size + 1; i++) {
			cout << i << " : "
					<< first_node_at_level(i) << endl;
		}
	}

	if (f_v) {
		cout << "poset_of_orbits::init_root_node_from_base_case done" << endl;
	}
}

void poset_of_orbits::init_root_node(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::init_root_node" << endl;
	}
	if (PC->has_base_case()) {

		if (f_v) {
			cout << "poset_of_orbits::init_root_node "
					"before init_root_node_from_base_case" << endl;
		}
		init_root_node_from_base_case(verbose_level);
		if (f_v) {
			cout << "poset_of_orbits::init_root_node "
					"after init_root_node_from_base_case" << endl;
		}

	}
	else {
		if (f_v) {
			cout << "poset_of_orbits::init_root_node "
					"before root[0].init_root_node" << endl;
		}
		root[0].init_root_node(PC, verbose_level);
		if (f_v) {
			cout << "poset_of_orbits::init_root_node "
					"after root[0].init_root_node" << endl;
		}
	}
	if (f_v) {
		cout << "poset_of_orbits::init_root_node done" << endl;
	}
}

void poset_of_orbits::make_table_of_nodes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::make_table_of_nodes" << endl;
	}
	long int *Table;
	int nb_rows, nb_cols;
	string fname;
	other::orbiter_kernel_system::file_io Fio;

	get_table_of_nodes(
			Table,
		nb_rows, nb_cols,
		0 /*verbose_level*/);

	fname = PC->get_problem_label_with_path() + "_table_of_orbits.csv";

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, Table, nb_rows, nb_cols);

	if (f_v) {
		cout << "poset_of_orbits::make_table_of_nodes "
				"written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}


	FREE_lint(Table);

	if (f_v) {
		cout << "poset_of_orbits::make_table_of_nodes done" << endl;
	}
}

void poset_of_orbits::poset_orbit_node_depth_breadth_perm_and_inverse(
	int max_depth,
	int *&perm, int *&perm_inv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx = 0;
	int N;

	if (f_v) {
		cout << "poset_of_orbits::poset_orbit_node_depth_breadth_perm_and_inverse" << endl;
		cout << "max_depth = " << max_depth << endl;
	}

	N = first_node_at_level(max_depth + 1);
	if (f_v) {
		cout << "N = first_poset_orbit_node_at_level[max_depth + 1] = "
				<< N << endl;
	}

	perm = NEW_int(N);
	perm_inv = NEW_int(N);

	if (f_v) {
		cout << "calling root->poset_orbit_node_"
				"depth_breadth_perm_and_inverse" << endl;
	}
	root->poset_orbit_node_depth_breadth_perm_and_inverse(
			PC,
			max_depth, idx, 0, 0, perm, perm_inv);

	if (f_v) {
		cout << "poset_of_orbits::poset_orbit_node_depth_breadth_perm_and_inverse done" << endl;
	}
}


void poset_of_orbits::read_memory_object(
		int &depth_completed,
		other::orbiter_kernel_system::memory_object *m,
		int &nb_group_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	long int i;
	long int nb_nodes;
	int version, magic_sync;
	int *Elt_tmp;

	if (f_v) {
		cout << "poset_of_orbits::read_memory_object, "
				"data size (in chars) = " << m->used_length << endl;
		}

	Elt_tmp = NEW_int(PC->get_poset()->A->elt_size_in_int);

	nb_group_elements = 0;
	m->read_int(&version);
	if (version != 1) {
		cout << "poset_of_orbits::read_memory_object "
				"version = " << version << " unknown" << endl;
		exit(1);
		}
	m->read_int(&depth_completed);
	if (f_v) {
		cout << "poset_of_orbits::read_memory_object "
				"depth_completed = " << depth_completed << endl;
		}

	if (depth_completed > sz) {
		cout << "poset_of_orbits::read_memory_object "
				"depth_completed > sz" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "poset_of_orbits::read_memory_object "
				"before m->read_int" << endl;
	}
	m->read_lint(&nb_nodes);
	if (f_v) {
		cout << "poset_of_orbits::read_memory_object "
				"nb_nodes = " << nb_nodes << endl;
	}


#if 1
	if (nb_nodes > nb_poset_orbit_nodes_allocated) {
		reallocate_to(nb_nodes, verbose_level - 1);
	}
#endif
	for (i = 0; i <= depth_completed + 1; i++) {
		m->read_lint(&first_poset_orbit_node_at_level[i]);
	}



	int one_percent;
	//int verbose_level_down = 0;


	one_percent = (int)((double) nb_nodes * 0.01);
	if (f_v) {
		cout << "poset_of_orbits::read_memory_object "
				" one_percent = " << one_percent << " nodes" << endl;
	}

	for (i = 0; i < nb_nodes; i++) {
		if (nb_nodes > 1000) {
			if ((i % one_percent) == 0) {
				int t1, dt;
				other::orbiter_kernel_system::os_interface Os;

				t1 = Os.os_ticks();
				dt = t1 - t0;

				cout << "Time ";
				Os.time_check_delta(cout, dt);
				PC->print_problem_label();
				cout << " : " << i / one_percent << " percent done, "
						" node=" << i << " / " << nb_nodes << " "
						"nb_group_elements=" << nb_group_elements << endl;
			}
		}

		root[i].read_memory_object(PC, PC->get_poset()->A, m,
				nb_group_elements,
				Elt_tmp,
				0 /*verbose_level_down*/ /*verbose_level - 1*/);
	}
	if (f_v) {
		cout << "poset_of_orbits::read_memory_object "
				"reading nodes completed" << endl;
	}
	m->read_int(&magic_sync);
	if (magic_sync != MAGIC_SYNC) {
		cout << "poset_of_orbits::read_memory_object "
				"could not read MAGIC_SYNC, file is corrupt" << endl;
		exit(1);
	}
	nb_poset_orbit_nodes_used = nb_nodes;

	FREE_int(Elt_tmp);

	if (f_v) {
		cout << "poset_of_orbits::read_memory_object finished ";
		cout << "depth_completed=" << depth_completed
			<< ", with " << nb_nodes << " nodes"
			<< " and " << nb_group_elements << " group elements"
			<< endl;
	}
}

void poset_of_orbits::write_memory_object(
		int depth_completed,
		other::orbiter_kernel_system::memory_object *m,
		int &nb_group_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i;
	long int nb_nodes;
	int *Elt_tmp;


	nb_nodes = first_node_at_level(depth_completed + 1);
	if (f_v) {
		cout << "poset_of_orbits::write_memory_object "
				<< nb_nodes << " nodes" << endl;
	}

	Elt_tmp = NEW_int(PC->get_poset()->A->elt_size_in_int);

	nb_group_elements = 0;
	m->write_int(1); // version number of this file format
	m->write_int(depth_completed);
	m->write_lint(nb_nodes);
	for (i = 0; i <= depth_completed + 1; i++) {
		m->write_lint(first_node_at_level(i));
	}
	if (f_v) {
		cout << "poset_of_orbits::write_memory_object "
				" writing " << nb_nodes << " node" << endl;
	}

	int one_percent;
	int verbose_level_down = 0;


	one_percent = (int)((double) nb_nodes * 0.01);
	if (f_v) {
		cout << "poset_of_orbits::write_memory_object "
				" one_percent = " << one_percent << " nodes" << endl;
	}

	for (i = 0; i < nb_nodes; i++) {
		if (nb_nodes > 1000) {
			if ((i % one_percent) == 0) {
				int t1, dt;
				other::orbiter_kernel_system::os_interface Os;

				t1 = Os.os_ticks();
				dt = t1 - t0;

				cout << "Time ";
				Os.time_check_delta(cout, dt);
				PC->print_problem_label();
				cout << " : " << i / one_percent << " percent done, "
						" node=" << i << " / " << nb_nodes << " "
						"nb_group_elements=" << nb_group_elements << endl;
			}
		}
		get_node(i)->write_memory_object(PC, PC->get_poset()->A, m,
				nb_group_elements,
				Elt_tmp,
				verbose_level_down /*verbose_level - 2*/);
	}
	m->write_int(MAGIC_SYNC); // a check to see if the file is not corrupt
	if (f_v) {
		cout << "poset_of_orbits::write_memory_object "
				" done, written " << nb_group_elements
				<< " group elements" << endl;
	}

	FREE_int(Elt_tmp);

	if (f_v) {
		cout << "poset_of_orbits::write_memory_object "
				"finished, data size (in chars) = "
				<< m->used_length << endl;
	}
}

long int poset_of_orbits::calc_size_on_file(
		int depth_completed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	long int s = 0;
	int nb_nodes;

	if (f_v) {
		cout << "poset_of_orbits::calc_size_on_file "
				"depth_completed=" << depth_completed << endl;
	}
	nb_nodes = first_node_at_level(depth_completed + 1);
	s += sizeof(int);
	s += sizeof(int);
	s += sizeof(long int);
	//m->write_int(1); // version number of this file format
	//m->write_int(depth_completed);
	//m->write_int(nb_nodes);
	for (i = 0; i <= depth_completed + 1; i++) {
		s += sizeof(long int);
	}
	for (i = 0; i < nb_nodes; i++) {
		s += root[i].calc_size_on_file(PC->get_poset()->A, verbose_level);
	}
	s += sizeof(int); // MAGIC_SYNC
	if (f_v) {
		cout << "poset_of_orbits::calc_size_on_file "
				"depth_completed=" << depth_completed
				<< " s=" << s << endl;
	}
	return s;
}

void poset_of_orbits::read_sv_level_file_binary2(
	int level, std::ifstream &fp,
	int f_split, int split_mod, int split_case,
	int f_recreate_extensions, int f_dont_keep_sv,
	int verbose_level)
{
	int f, i, nb_nodes;
	int f_v = (verbose_level >= 1);
	int I;
	other::orbiter_kernel_system::file_io Fio;

	f = first_node_at_level(level);
	nb_nodes = nb_orbits_at_level(level);
	if (f_v) {
		cout << "poset_of_orbits::read_sv_level_file_binary2 "
				<< nb_nodes << " nodes" << endl;
		cout << "f_recreate_extensions="
				<< f_recreate_extensions << endl;
		cout << "f_dont_keep_sv=" << f_dont_keep_sv << endl;
		if (f_split) {
			cout << "f_split is true, split_mod=" << split_mod
					<< " split_case=" << split_case << endl;
		}
	}

	// version number of this file format
	fp.read((char *) &I, sizeof(int));
	//I = Fio.fread_int4(fp);
	if (I != 1) {
		cout << "poset_of_orbits::read_sv_level_file_binary2: "
				"unknown file version" << endl;
		exit(1);
	}
	fp.read((char *) &I, sizeof(int));
	//I = Fio.fread_int4(fp);
	if (I != level) {
		cout << "poset_of_orbits::read_sv_level_file_binary2: "
				"level does not match" << endl;
		exit(1);
	}
	fp.read((char *) &I, sizeof(int));
	//I = Fio.fread_int4(fp);
	if (I != nb_nodes) {
		cout << "poset_of_orbits::read_sv_level_file_binary2: "
				"nb_nodes does not match" << endl;
		exit(1);
	}
	for (i = 0; i < nb_nodes; i++) {
		if (f_split) {
			if ((i % split_mod) != split_case)
				continue;
		}
		root[f + i].sv_read_file(PC, fp, 0 /*verbose_level - 2*/);
		if (f_recreate_extensions) {
			root[f + i].reconstruct_extensions_from_sv(
					PC, 0 /*verbose_level - 1*/);
		}
		if (f_dont_keep_sv) {
			root[f + i].delete_Schreier_vector();
			//FREE_OBJECT(root[f + i].Schreier_vector);
			//root[f + i].Schreier_vector = NULL;
		}
	}
	fp.read((char *) &I, sizeof(int));
	//I = Fio.fread_int4(fp);
	if (I != MAGIC_SYNC) {
		cout << "poset_of_orbits::read_sv_level_file_binary2: "
				"MAGIC_SYNC does not match" << endl;
		exit(1);
	}
	// a check to see if the file is not corrupt
	if (f_v) {
		cout << "poset_of_orbits::read_sv_level_file_binary2 "
				"finished" << endl;
	}
}

void poset_of_orbits::write_sv_level_file_binary2(
	int level, std::ofstream &fp,
	int f_split, int split_mod, int split_case,
	int verbose_level)
{
	int f, i, nb_nodes, tmp;
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	f = first_node_at_level(level);
	nb_nodes = nb_orbits_at_level(level);
	if (f_v) {
		cout << "poset_of_orbits::write_sv_level_file_binary2 "
				<< nb_nodes << " nodes" << endl;
	}
	// version number of this file format
	tmp = 1;
	fp.write((char *) &tmp, sizeof(int));
	//Fio.fwrite_int4(fp, 1);
	fp.write((char *) &level, sizeof(int));
	//Fio.fwrite_int4(fp, level);
	fp.write((char *) &nb_nodes, sizeof(int));
	//Fio.fwrite_int4(fp, nb_nodes);
	for (i = 0; i < nb_nodes; i++) {
		if (f_split) {
			if ((i % split_mod) != split_case)
				continue;
		}
		root[f + i].sv_write_file(PC, fp, verbose_level - 2);
	}
	tmp = MAGIC_SYNC;
	fp.write((char *) &tmp, sizeof(int));
	//Fio.fwrite_int4(fp, MAGIC_SYNC);
	// a check to see if the file is not corrupt
	if (f_v) {
		cout << "poset_of_orbits::write_sv_level_file_binary2 "
				"finished" << endl;
	}
}


void poset_of_orbits::read_level_file_binary2(
	int level, std::ifstream &fp,
	int &nb_group_elements, int verbose_level)
{
	int f, i, nb_nodes, magic_sync;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int_4 I;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "poset_of_orbits::read_level_file_binary2" << endl;
	}
	f = first_node_at_level(level);
	nb_group_elements = 0;
	fp.read((char *) &I, sizeof(int));
	if (I != 1) {
		cout << "poset_of_orbits::read_level_file_binary2 "
				"version = " << I << " unknown" << endl;
		exit(1);
	}

	fp.read((char *) &I, sizeof(int));
	if (I != level) {
		cout << "poset_of_orbits::read_level_file_binary2 "
				"level = " << I << " should be " << level << endl;
		exit(1);
	}

	fp.read((char *) &nb_nodes, sizeof(int));
	if (f_v) {
		cout << "poset_of_orbits::read_level_file_binary, "
				"nb_nodes = " << nb_nodes << endl;
		}
	first_poset_orbit_node_at_level[level + 1] = f + nb_nodes;

	if (f_v) {
		cout << "poset_of_orbits::read_level_file_binary2 "
				"f + nb_nodes = " << f + nb_nodes << endl;
		cout << "poset_of_orbits::read_level_file_binary2 "
				"nb_poset_orbit_nodes_allocated = "
			<< nb_poset_orbit_nodes_allocated << endl;
	}
	if (f + nb_nodes > nb_poset_orbit_nodes_allocated) {
		reallocate_to(f + nb_nodes, verbose_level - 2);
	}
	for (i = 0; i < nb_nodes; i++) {
		if (f_vv && nb_nodes > 1000 && ((i % 1000) == 0)) {
			cout << "reading node " << i << endl;
			}
		root[f + i].read_file(PC->get_poset()->A, fp, nb_group_elements,
				verbose_level - 2);
	}
	if (f_v) {
		cout << "reading nodes completed" << endl;
	}
	fp.read((char *) &magic_sync, sizeof(int));
	if (magic_sync != MAGIC_SYNC) {
		cout << "poset_of_orbits::read_level_file_binary2 "
				"could not read MAGIC_SYNC, file is corrupt" << endl;
		cout << "MAGIC_SYNC=" << MAGIC_SYNC << endl;
		cout << "we read   =" << magic_sync << endl;
		exit(1);
	}
	if (f_v) {
		cout << "poset_of_orbits::read_level_file_binary2 "
				"finished ";
		cout << "level=" << level
			<< ", with " << nb_nodes << " nodes"
			<< " and " << nb_group_elements << " group elements"
			<< endl;
	}
}

void poset_of_orbits::write_level_file_binary2(
	int level, std::ofstream &fp,
	int &nb_group_elements, int verbose_level)
{
	int f, i, nb_nodes, tmp;
	int f_v = false;//(verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	f = first_node_at_level(level);
	nb_nodes = nb_orbits_at_level(level);
	if (f_v) {
		cout << "poset_of_orbits::write_level_file_binary2 "
				<< nb_nodes << " nodes" << endl;
	}
	nb_group_elements = 0;
	// version number of this file format
	tmp = 1;
	fp.write((char *) &tmp, sizeof(int));
	fp.write((char *) &level, sizeof(int));
	fp.write((char *) &nb_nodes, sizeof(int));
	for (i = 0; i < nb_nodes; i++) {
		root[f + i].write_file(PC->get_poset()->A, fp,
				nb_group_elements, verbose_level - 2);
	}
	tmp = MAGIC_SYNC;
	fp.write((char *) &tmp, sizeof(int));
	// a check to see if the file is not corrupt
	if (f_v) {
		cout << "poset_of_orbits::write_level_file_binary2 "
				"finished" << endl;
	}
}

void poset_of_orbits::write_candidates_binary_using_sv(
		std::string &fname_base,
		int lvl, int t0, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false;
	string fname;

	if (f_v) {
		cout << "poset_of_orbits::write_candidates_binary_using_sv "
				"lvl=" << lvl << " fname_base=" << fname_base << endl;
	}
	PC->make_fname_candidates_file_default(fname, lvl);

	{
	int fst, len;
	int *nb_cand;
	int *cand_first;
	int total_nb_cand = 0;
	int *subset;
	//int *Cand;
	int i, j, node, nb, pos;
	other::orbiter_kernel_system::file_io Fio;

	fst = first_node_at_level(lvl);
	len = nb_orbits_at_level(lvl);
	if (f_v) {
		cout << "poset_of_orbits::write_candidates_binary_using_sv "
				"first node at level " << lvl << " is " << fst << endl;
		cout << "poset_of_orbits::write_candidates_binary_using_sv "
				"number of nodes at level " << lvl << " is " << len << endl;
	}
	nb_cand = NEW_int(len);
	cand_first = NEW_int(len);
	for (i = 0; i < len; i++) {
		node = fst + i;
		if (!root[node].has_Schreier_vector()) {
			cout << "poset_of_orbits::write_candidates_binary_using_sv "
					"node " << i << " / " << len
					<< " no schreier vector" << endl;
		}
		nb = root[node].get_nb_of_live_points();

		if (f_vv) {
			cout << "poset_of_orbits::write_candidates_binary_using_sv "
					"node " << i << " / " << len << endl;
		}

		nb_cand[i] = nb;
		total_nb_cand += nb;
	}
	if (f_v) {
		cout << "poset_of_orbits::write_candidates_binary_using_sv "
				"total_nb_cand=" << total_nb_cand << endl;
	}
	//Cand = NEW_int(total_nb_cand);
	pos = 0;
	for (i = 0; i < len; i++) {
		node = fst + i;
		nb = root[node].get_nb_of_live_points();
		subset = root[node].live_points();
		cand_first[i] = pos;
#if 0
		for (j = 0; j < nb; j++) {
			Cand[pos + j] = subset[j];
		}
#endif
		pos += nb;
	}

	if (f_v) {
		cout << "poset_of_orbits::write_candidates_binary_using_sv "
				"writing file" << fname << endl;
	}
	{
		ofstream fp(fname, ios::binary);

		fp.write((char *) &len, sizeof(int));
		for (i = 0; i < len; i++) {
			fp.write((char *) &nb_cand[i], sizeof(int));
			fp.write((char *) &cand_first[i], sizeof(int));
		}
		for (i = 0; i < len; i++) {
			node = fst + i;
			nb = root[node].get_nb_of_live_points();
			subset = root[node].live_points();
			for (j = 0; j < nb; j++) {
				fp.write((char *) &subset[j], sizeof(int));
			}
		}
	}

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "poset_of_orbits::write_candidates_binary_using_sv "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	FREE_int(nb_cand);
	FREE_int(cand_first);
	//FREE_int(Cand);
	}
	if (f_v) {
		cout << "poset_of_orbits::write_candidates_binary_using_sv "
				"done" << endl;
	}
}

void poset_of_orbits::read_level_file(
		int level,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *set_sizes;
	long int **sets;
	char **data;
	int nb_cases;
	int nb_nodes, first_at_level;
	int i, I, J;
	poset_orbit_node *O;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "poset_of_orbits::read_level_file "
				"fname=" << fname << endl;
	}

	Fio.read_and_parse_data_file(fname, nb_cases,
			data, sets, set_sizes, verbose_level - 1);


	first_at_level = first_node_at_level(level);
	nb_nodes = first_at_level + nb_cases;

	if (nb_nodes > nb_poset_orbit_nodes_allocated) {
		if (f_vv) {
			cout << "poset_of_orbits::read_level_file "
					"reallocating to " << nb_nodes << " nodes" << endl;
			}
		reallocate_to(nb_nodes, verbose_level - 1);
	}
	first_poset_orbit_node_at_level[level + 1] = nb_nodes;
	for (i = 0; i < nb_cases; i++) {
		I = first_at_level + i;
		O = &root[I];

		cout << setw(10) << i << " : ";
		Lint_vec_print(cout, sets[i], level);
		cout << endl;

		J = PC->get_Poo()->find_poset_orbit_node_for_set(level - 1,
				sets[i], false /* f_tolerant */,
				0/*verbose_level*/);
		cout << "J=" << J << endl;

#if 0
		O->node = I;
		O->prev = J;
		O->pt = sets[i][level - 1];
		O->nb_strong_generators = 0;
		O->hdl_strong_generators = NULL;
		O->tl = NULL;
		O->nb_extensions = 0;
		O->E = NULL;
		O->Schreier_vector = NULL;
#else
		O->init_node(I /* node*/, J /* prev*/, sets[i][level - 1] /*pt*/, verbose_level);
#endif

		{
			data_structures_groups::group_container Aut;

			Aut.init(PC->get_poset()->A, verbose_level - 2);

			if (strlen(data[i])) {

				string s;

				s.assign(data[i]);
				Aut.init_ascii_coding(s, verbose_level - 2);

				Aut.decode_ascii(false);

				// now strong poset_classifications are available

				Aut.schreier_sims(0);

				cout << "the automorphism group has order ";
				Aut.print_group_order(cout);
				cout << endl;

				groups::strong_generators *Strong_gens;

				Strong_gens = NEW_OBJECT(groups::strong_generators);
				Strong_gens->init_from_sims(Aut.S, 0);

	#if 0
				cout << "and is strongly generated by the "
						"following " << Aut.SG->len << " elements:" << endl;

				Aut.SG->print(cout);
				cout << endl;
	#endif
				O->store_strong_generators(PC, Strong_gens, verbose_level - 2);
				cout << "strong poset_classifications stored" << endl;

				FREE_OBJECT(Strong_gens);
			}
			else {
				//cout << "trivial group" << endl;
				//Aut.init_strong_generators_empty_set();

			}
		}

	}
	FREE_int(set_sizes);
	if (f_v) {
		cout << "poset_of_orbits::read_level_file "
				"fname=" << fname << " done" << endl;
	}
}

void poset_of_orbits::write_lvl_file_with_candidates(
		std::string &fname_base, int lvl, int t0,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname1;
	other::orbiter_kernel_system::file_io Fio;
	other::orbiter_kernel_system::os_interface Os;

	fname1 = fname_base + "_lvl_" + std::to_string(lvl) + "_candidates.txt";
	{
		ofstream f(fname1);
		int cur;

		//f << "# " << lvl << endl;
		for (cur = first_node_at_level(lvl);
			cur < first_node_at_level(lvl + 1); cur++) {
			root[cur].log_current_node_with_candidates(
					PC, lvl, f, verbose_level - 2);
		}
		f << "-1 " << first_node_at_level(lvl + 1)
					- first_node_at_level(lvl)
			<< " " << first_node_at_level(lvl) << " in ";
		Os.time_check(f, t0);
		f << endl;
		f << "# in action " << PC->get_poset()->A->label << endl;
	}
	if (f_v) {
		cout << "written file " << fname1
				<< " of size " << Fio.file_size(fname1) << endl;
	}
}

void poset_of_orbits::get_orbit_reps_at_level(
		int lvl, long int *&Data,
		int &nb_reps, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::get_orbit_reps_at_level" << endl;
	}
	int i, fst;

	fst = first_node_at_level(lvl);
	nb_reps = nb_orbits_at_level(lvl);

	Data = NEW_lint(nb_reps * lvl);

	for (i = 0; i < nb_reps; i++) {
		root[fst + i].store_set_to(PC, Data + i * lvl);
	}

	if (f_v) {
		cout << "poset_of_orbits::get_orbit_reps_at_level done" << endl;
	}
}

void poset_of_orbits::write_orbit_reps_at_level(
		std::string &fname_base,
		int lvl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname1;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "poset_of_orbits::write_orbit_reps_at_level" << endl;
	}
	PC->make_fname_lvl_reps_file(fname1, fname_base, lvl);

	long int *Data;
	int nb_reps;

	get_orbit_reps_at_level(
			lvl, Data, nb_reps, verbose_level);


	{
		ofstream f(fname1);
		int i;


		f << "Row,REP" << endl;
		for (i = 0; i < nb_reps; i++) {
			f << i;

			string S;

			S = Lint_vec_stringify(Data + i * lvl, lvl);
			f << ",\"" << S << "\"" << endl;
		}
		f << "END" << endl;

	}
	if (f_v) {
		cout << "poset_of_orbits::write_orbit_reps_at_level Written file "
				<< fname1 << " of size " << Fio.file_size(fname1) << endl;

	}
	FREE_lint(Data);
	if (f_v) {
		cout << "poset_of_orbits::write_orbit_reps_at_level done" << endl;
	}
}


void poset_of_orbits::write_lvl_file(
		std::string &fname_base,
		int lvl, int t0, int f_with_stabilizer_generators,
		int f_long_version,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname1;
	other::orbiter_kernel_system::file_io Fio;
	other::orbiter_kernel_system::os_interface Os;

	PC->make_fname_lvl_file(fname1, fname_base, lvl);
	{
		ofstream f(fname1);
		int i, fst, len;


		fst = first_node_at_level(lvl);
		len = nb_orbits_at_level(lvl);

		f << "# " << lvl << endl;
		for (i = 0; i < len; i++) {
			root[fst + i].log_current_node(PC,
					lvl, f, f_with_stabilizer_generators,
					f_long_version);
		}
		f << "-1 " << len << " "
				<< first_node_at_level(lvl) << " in ";
		Os.time_check(f, t0);

		poset_classification_global PCG;

		PCG.init(
				PC,
				verbose_level);

		string s_ago;

		s_ago = PCG.compute_and_stringify_automorphism_group_orders(lvl, verbose_level - 2);

		f << s_ago << endl;
		f << "# in action " << PC->get_poset()->A->label << endl;
	}
	if (f_v) {
		cout << "written file " << fname1
				<< " of size " << Fio.file_size(fname1) << endl;
	}
}

void poset_of_orbits::write_lvl(
		std::ostream &f, int lvl, int t0,
		int f_with_stabilizer_generators, int f_long_version,
		int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int i;
	int fst, len;
	other::orbiter_kernel_system::os_interface Os;


	fst = first_node_at_level(lvl);
	len = nb_orbits_at_level(lvl);

	f << "# " << lvl << endl;
	for (i = 0; i < len; i++) {
		root[fst + i].log_current_node(PC, lvl, f,
				f_with_stabilizer_generators, f_long_version);
	}
	f << "-1 " << len << " " << first_node_at_level(lvl)
		<< " in ";
	Os.time_check(f, t0);
	f << endl;

	poset_classification_global PCG;

	PCG.init(
			PC,
			verbose_level);


	string s_ago;
	s_ago = PCG.compute_and_stringify_automorphism_group_orders(lvl, verbose_level - 2);
	//PCG.compute_and_print_automorphism_group_orders(lvl, f, verbose_level - 2);

	f << s_ago << endl;
	f << "# in action " << PC->get_poset()->A->label << endl;
}

void poset_of_orbits::log_nodes_for_treefile(
		int cur, int depth,
		std::ostream &f, int f_recurse, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, next;
	poset_orbit_node *node = &root[cur];


	if (f_v) {
		cout << "poset_of_orbits::log_nodes_for_treefile "
				"cur=" << cur << endl;
	}
	if (PC->has_base_case() && cur < PC->get_Base_case()->size) {
		return; // !!!
	}

	node->log_current_node(PC, depth, f,
			false /* f_with_strong_generators */, 0);

	if (f_recurse) {
		//cout << "recursing into dependent nodes" << endl;
		for (i = 0; i < node->get_nb_of_extensions(); i++) {
			if (node->get_E(i)->get_type() == EXTENSION_TYPE_EXTENSION) {
				if (node->get_E(i)->get_data() >= 0) {
					next = node->get_E(i)->get_data();
					log_nodes_for_treefile(next,
							depth + 1, f, true, verbose_level);
				}
			}
		}
	}
}

void poset_of_orbits::make_table_of_orbit_reps(
		std::string *&Headings,
		std::string *&Table,
		int &nb_rows, int &nb_cols,
		int level_min, int level_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::make_table_of_orbit_reps" << endl;
	}
	int Nb_orbits, nb_orbits, i, level, first, cur;
	long int *rep;

	rep = NEW_lint(level_max);

	Nb_orbits = 0;
	for (level = level_min; level <= level_max; level++) {
		Nb_orbits += nb_orbits_at_level(level);
	}

	nb_rows = Nb_orbits;
	nb_cols = 8;

	Table = new string [nb_rows * nb_cols];
	Headings = new string [nb_cols];

	Headings[0] = "Line";
	Headings[1] = "Node";
	Headings[2] = "Level";
	Headings[3] = "OrbitIdx";
	Headings[4] = "OrbitRep";
	Headings[5] = "StabOrder";
	Headings[6] = "OrbitLength";
	Headings[7] = "SV_length";

	cur = 0;
	for (level = level_min; level <= level_max; level++) {

		first = first_node_at_level(level);

		nb_orbits = nb_orbits_at_level(level);

		for (i = 0; i < nb_orbits; i++, cur++) {

			get_set_by_level(level, i, rep);

			algebra::ring_theory::longinteger_object stab_order, orbit_length;

			get_orbit_length_and_stabilizer_order(
					i, level,
				stab_order, orbit_length);

			poset_orbit_node *O;
			int schreier_vector_length;

			O = get_node_ij(level, i);
			if (O->has_Schreier_vector()) {
				schreier_vector_length = O->get_nb_of_live_points();
			}
			else {
				schreier_vector_length = 0;
			}


			Table[cur * nb_cols + 0] = std::to_string(cur);
			Table[cur * nb_cols + 1] = std::to_string(first + i);
			Table[cur * nb_cols + 2] = std::to_string(level);
			Table[cur * nb_cols + 3] = std::to_string(i);
			Table[cur * nb_cols + 4] = "\"" + Lint_vec_stringify(rep, level) + "\"";
			Table[cur * nb_cols + 5] = stab_order.stringify();
			Table[cur * nb_cols + 6] = orbit_length.stringify();
			Table[cur * nb_cols + 7] = std::to_string(schreier_vector_length);

			//Text_level[first + i] = std::to_string(level);

			//Text_node[first + i] = std::to_string(i);

			//get_set_by_level(level, i, rep);
			//Lint_vec_print_to_str(Text_orbit_reps[first + i], rep, level);

			//stab_order.print_to_string(Text_stab_order[first + i]);

			//orbit_length.print_to_string(Text_orbit_length[first + i]);

			//Text_schreier_vector_length[first + i] = std::to_string(schreier_vector_length);
		}
	}


	FREE_lint(rep);
	if (f_v) {
		cout << "poset_of_orbits::make_table_of_orbit_reps done" << endl;
	}
}


void poset_of_orbits::save_representatives_up_to_a_given_level_to_csv(
		int lvl, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::save_representatives_up_to_a_given_level_to_csv" << endl;
	}

	//other::data_structures::spreadsheet *Sp;
	std::string *Headings;
	std::string *Table;
	int nb_rows, nb_cols;

	if (f_v) {
		cout << "poset_of_orbits::save_representatives_up_to_a_given_level_to_csv "
				"before make_table_of_orbit_reps" << endl;
	}
	make_table_of_orbit_reps(
			Headings,
			Table,
			nb_rows, nb_cols,
			0 /* level_min */, lvl /* level_max */,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "poset_of_orbits::save_representatives_up_to_a_given_level_to_csv "
				"after make_table_of_orbit_reps" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	string fname_csv;

	fname_csv = PC->get_problem_label_with_path()
			+ "_orbits_up_to_level_"
			+ std::to_string(lvl)
			+ ".csv";

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_csv,
			nb_rows, nb_cols, Table,
			Headings,
			verbose_level);

	delete [] Table;
	delete [] Headings;

#if 0
	PC->make_spreadsheet_of_orbit_reps(
			Sp, actual_size);
#endif
	//Sp->save(fname_csv, verbose_level);
	//FREE_OBJECT(Sp);

	if (f_v) {
		cout << "poset_of_orbits::save_representatives_up_to_a_given_level_to_csv done" << endl;
	}
}


void poset_of_orbits::save_representatives_at_level_to_csv(
		std::string &fname,
		int lvl, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::save_representatives_at_level_to_csv" << endl;
	}

	{
		std::string *Headings;
		std::string *Table;
		int nb_rows, nb_cols;

		PC->get_Poo()->make_table_of_orbit_reps(
				Headings,
				Table,
				nb_rows, nb_cols,
				lvl /* level_min */, lvl /* level_max */,
				0 /*verbose_level*/);

		other::orbiter_kernel_system::file_io Fio;
		string fname_csv;


		Fio.Csv_file_support->write_table_of_strings_with_col_headings(
				fname,
				nb_rows, nb_cols, Table,
				Headings,
				verbose_level);

		delete [] Table;
		delete [] Headings;

		if (f_v) {
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}

	}

	if (f_v) {
		cout << "poset_of_orbits::save_representatives_at_level_to_csv done" << endl;
	}
}

void poset_of_orbits::get_set_orbits_at_level(
		int lvl, other::data_structures::set_of_sets *&SoS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::get_set_orbits_at_level" << endl;
	}

	int i, nb_orbits;
	long int *set;
	groups::strong_generators *Strong_gens;
	long int *Length;
	long int *Length_expanded;

	set = NEW_lint(lvl);

	nb_orbits = nb_orbits_at_level(lvl);
	Length = NEW_lint(nb_orbits);
	Length_expanded = NEW_lint(nb_orbits);


	Strong_gens = PC->get_poset()->Strong_gens;

	orbits_schreier::orbit_of_sets **Orb;

	Orb = (orbits_schreier::orbit_of_sets **) NEW_pvoid(nb_orbits);

	for (i = 0; i < nb_orbits; i++) {

		get_node_ij(lvl, i)->store_set_to(PC, lvl - 1, set /*gen->S0*/);

		other::orbiter_kernel_system::file_io Fio;


		Orb[i] = NEW_OBJECT(orbits_schreier::orbit_of_sets);

		Orb[i]->init(PC->get_poset()->A, PC->get_poset()->A2,
				set, lvl /* sz */,
				Strong_gens->gens,
				verbose_level);

		if (f_v) {
			cout << "poset_of_orbits::get_set_orbits_at_level "
					"orbit " << i << " / " << nb_orbits
					<< " Found an orbit of size " << Orb[i]->used_length << endl;
		}

		Length[i] = Orb[i]->used_length;
		Length_expanded[i] = Length[i] * lvl;

	}

	SoS = NEW_OBJECT(other::data_structures::set_of_sets);

	SoS->init_basic(
			PC->get_poset()->A2->degree,
			nb_orbits /* nb_sets */,
			Length_expanded,
			verbose_level);
	for (i = 0; i < nb_orbits; i++) {

		long int *Table;
		int orbit_length;

		Orb[i]->get_table_of_orbits(
				Table,
				orbit_length, lvl,
				verbose_level);

		if (orbit_length * lvl != Length_expanded[i]) {
			cout << "poset_of_orbits::get_set_orbits_at_level "
					"orbit length is wrong" << endl;
			exit(1);
		}

		Lint_vec_copy(Table, SoS->Sets[i], Length_expanded[i]);

		FREE_lint(Table);
	}


	FREE_lint(set);
	FREE_lint(Length);
	FREE_lint(Length_expanded);

	for (i = 0; i < nb_orbits; i++) {
		FREE_OBJECT(Orb[i]);
	}
	FREE_pvoid((void **) Orb);

	if (f_v) {
		cout << "poset_of_orbits::get_set_orbits_at_level done" << endl;
	}
}

int poset_of_orbits::find_poset_orbit_node_for_set_basic(
		int from,
		int node, int len, long int *set, int f_tolerant,
		int verbose_level)
{
	int i, j;
	long int pt;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);

	if (f_vv) {
		cout << "poset_of_orbits::"
				"find_poset_orbit_node_for_set_basic "
				"looking for set ";
		Lint_vec_print(cout, set, len);
		cout << endl;
		cout << "node=" << node << endl;
		cout << "from=" << from << endl;
		cout << "len=" << len << endl;
		cout << "f_tolerant=" << f_tolerant << endl;
	}
	for (i = from; i < len; i++) {
		pt = set[i];
		if (f_vv) {
			cout << "pt=" << pt << endl;
			cout << "calling root[node].find_extension_from_point" << endl;
		}
		j = find_extension_from_point(
				node, pt, 0 /* verbose_level */);

		//j = root[node].find_extension_from_point(this, pt, false);

		if (j == -1) {
			if (f_v) {
				cout << "poset_of_orbits::"
						"find_poset_orbit_node_for_set_basic "
						"depth " << i << " no extension for point "
						<< pt << " found" << endl;
			}
			if (f_tolerant) {
				if (f_v) {
					cout << "poset_of_orbits::"
							"find_poset_orbit_node_for_set_basic "
							"since we are tolerant, we return -1" << endl;
				}
				return -1;
			}
			else {
				cout << "poset_of_orbits::"
						"find_poset_orbit_node_for_set_basic "
						"failure in find_extension_from_point" << endl;
				Lint_vec_print(cout, set, len);
				cout << endl;
				cout << "node=" << node << endl;
				cout << "from=" << from << endl;
				cout << "i=" << i << endl;
				cout << "pt=" << pt << endl;
				get_node(node)->print_extensions(PC);
				exit(1);
			}
		}
		if (get_node(node)->get_E(j)->get_pt() != pt) {
			cout << "poset_of_orbits::"
					"find_poset_orbit_node_for_set_basic "
					"root[node].E[j].pt != pt" << endl;
			exit(1);
		}
		if (get_node(node)->get_E(j)->get_type() != EXTENSION_TYPE_EXTENSION &&
				get_node(node)->get_E(j)->get_type() != EXTENSION_TYPE_PROCESSING) {
			cout << "poset_of_orbits::"
					"find_poset_orbit_node_for_set_basic "
					"root[node].get_E(j)->type != "
					"EXTENSION_TYPE_EXTENSION" << endl;
			cout << "root[node].get_E(j)->type="
					<< get_node(node)->get_E(j)->get_type() << " = ";
			PC->print_extension_type(cout, get_node(node)->get_E(j)->get_type());
			cout << endl;
			cout << "poset_of_orbits::"
					"find_poset_orbit_node_for_set_basic "
					"looking for set ";
			Lint_vec_print(cout, set, len);
			cout << endl;
			cout << "node=" << node << endl;
			cout << "from=" << from << endl;
			cout << "i=" << i << endl;
			cout << "node=" << node << endl;
			cout << "f_tolerant=" << f_tolerant << endl;
			cout << "node=" << node << endl;
			cout << "pt=" << pt << endl;
			cout << "j=" << j << endl;
			exit(1);
		}
		node = get_node(node)->get_E(j)->get_data();
		if (f_v) {
			cout << "depth " << i << " extension " << j
					<< " new node " << node << endl;
		}
	}
	return node;
}

int poset_of_orbits::find_poset_orbit_node_for_set(
		int len,
		long int *set, int f_tolerant, int verbose_level)
// finds the node that represents s_0,...,s_{len - 1}
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "poset_of_orbits::find_poset_orbit_node_for_set ";
		Lint_vec_print(cout, set, len);
		cout << endl;
	}
	if (PC->has_base_case()) {
		int i, j, h;
		if (len < PC->get_Base_case()->size) {
			cout << "poset_of_orbits::find_poset_orbit_node_for_set "
					"len < starter_size" << endl;
			cout << "len=" << len << endl;
			exit(1);
		}
		for (i = 0; i < PC->get_Base_case()->size; i++) {
			for (j = i; j < len; j++) {
				if (set[j] == PC->get_Base_case()->orbit_rep[i]) {
					if (f_v) {
						cout << "found " << i << "-th element "
								"of the starter which is " << PC->get_Base_case()->orbit_rep[i]
							<< " at position " << j << endl;
					}
					break;
				}
			}
			if (j == len) {
				cout << "poset_of_orbits::find_poset_orbit_node_for_set "
						"did not find " << i << "-th element "
						"of the starter" << endl;
			}
			for (h = j; h > i; h--) {
				set[h] = set[h - 1];
			}
			set[i] = PC->get_Base_case()->orbit_rep[i];
		}
		int from = PC->get_Base_case()->size;
		int node = PC->get_Base_case()->size;
		ret = find_poset_orbit_node_for_set_basic(from,
				node, len, set, f_tolerant, verbose_level);
	}
	else {
		int from = 0;
		int node = 0;
		ret = find_poset_orbit_node_for_set_basic(from,
				node, len, set, f_tolerant, verbose_level);
	}
	if (ret == -1) {
		if (f_tolerant) {
			if (f_v) {
				cout << "poset_of_orbits::find_poset_orbit_node_for_set ";
				Lint_vec_print(cout, set, len);
				cout << " extension not found, "
						"we are tolerant, returnning -1" << endl;
			}
			return -1;
		}
		else {
			cout << "poset_of_orbits::find_poset_orbit_node_for_set "
					"we should not be here" << endl;
			exit(1);
		}
	}
	return ret;

}



data_structures_groups::orbit_transversal *poset_of_orbits::get_orbit_transversal(
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::orbit_transversal *T;
	int orbit_at_level;

	if (f_v) {
		cout << "poset_of_orbits::get_orbit_transversal" << endl;
	}
	T = NEW_OBJECT(data_structures_groups::orbit_transversal);
	T->A = PC->get_poset()->A;
	T->A2 = PC->get_poset()->A2;


	T->nb_orbits = nb_orbits_at_level(level);


	if (f_v) {
		cout << "poset_of_orbits::get_orbit_transversal "
				"processing " << T->nb_orbits
				<< " orbit representatives" << endl;
	}


	T->Reps = NEW_OBJECTS(data_structures_groups::set_and_stabilizer, T->nb_orbits);

	for (orbit_at_level = 0;
			orbit_at_level < T->nb_orbits;
			orbit_at_level++) {

		data_structures_groups::set_and_stabilizer *SaS;

		SaS = get_set_and_stabilizer(level,
				orbit_at_level, verbose_level);



		T->Reps[orbit_at_level].init_everything(
				PC->get_poset()->A, PC->get_poset()->A2, SaS->data, level,
				SaS->Strong_gens, 0 /* verbose_level */);

		SaS->data = NULL;
		SaS->Strong_gens = NULL;

		FREE_OBJECT(SaS);

	}



	if (f_v) {
		cout << "poset_of_orbits::get_orbit_transversal done" << endl;
	}
	return T;
}

int poset_of_orbits::test_if_stabilizer_is_trivial(
		int level, int orbit_at_level, int verbose_level)
{
	poset_orbit_node *O;

	O = get_node_ij(level, orbit_at_level);
	return O->test_if_stabilizer_is_trivial();
}

data_structures_groups::set_and_stabilizer *poset_of_orbits::get_set_and_stabilizer(
		int level, int orbit_at_level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::set_and_stabilizer *SaS;

	if (f_v) {
		cout << "poset_of_orbits::get_set_and_stabilizer" << endl;
	}

	SaS = NEW_OBJECT(data_structures_groups::set_and_stabilizer);

	SaS->init(PC->get_poset()->A, PC->get_poset()->A2, 0 /*verbose_level */);

	SaS->allocate_data(level, 0 /* verbose_level */);

	get_set_by_level(level, orbit_at_level, SaS->data);

	get_stabilizer_generators(SaS->Strong_gens,
		level, orbit_at_level, 0 /* verbose_level */);

	SaS->Strong_gens->group_order(SaS->target_go);

	SaS->Stab = SaS->Strong_gens->create_sims(0 /*verbose_level*/);

	if (f_v) {
		cout << "poset_of_orbits::get_set_and_stabilizer done" << endl;
	}
	return SaS;
}

void poset_of_orbits::get_set_by_level(
		int level, int node, long int *set)
{
	int size;
	poset_orbit_node *O;

	O = get_node_ij(level, node);
	size = O->depth_of_node(PC);
	if (size != level) {
		cout << "poset_of_orbits::get_set_by_level "
				"size != level" << endl;
		exit(1);
	}
	//root[n].store_set_to(this, size - 1, set);
	O->store_set_to(PC, size - 1, set);
}


void poset_of_orbits::stabilizer_order(
		int node, algebra::ring_theory::longinteger_object &go)
{
#if 0
	if (root[node].get_nb_strong_generators()) {
		go.create_product(Poset->A->base_len(), root[node].tl);
	}
	else {
		go.create(1, __FILE__, __LINE__);
	}
#else
	get_node(node)->get_stabilizer_order(PC, go);
#endif
}


void poset_of_orbits::orbit_length(
		int orbit_at_level,
		int level, algebra::ring_theory::longinteger_object &len)
// uses poset_classification::go for the group order
{
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object stab_order, quo, rem;

	get_stabilizer_order(level, orbit_at_level, stab_order);
	D.integral_division(PC->get_poset()->go, stab_order, len, rem, 0);
	if (!rem.is_zero()) {
		cout << "poset_of_orbits::orbit_length stabilizer order does "
				"not divide group order" << endl;
		exit(1);
	}
}

void poset_of_orbits::get_orbit_length_and_stabilizer_order(
		int node,
		int level, algebra::ring_theory::longinteger_object &stab_order,
		algebra::ring_theory::longinteger_object &len)
// uses poset_classification::go for the group order
{
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object quo, rem;

	get_stabilizer_order(level, node, stab_order);
	D.integral_division(PC->get_poset()->go, stab_order, len, rem, 0);
	if (!rem.is_zero()) {
		cout << "poset_of_orbits::orbit_length "
				"stabilizer order "
				"does not divide group order" << endl;
		exit(1);
	}
}

int poset_of_orbits::orbit_length_as_int(
		int orbit_at_level, int level)
{
	algebra::ring_theory::longinteger_object len;

	orbit_length(orbit_at_level, level, len);
	return len.as_int();

}


void poset_of_orbits::recreate_schreier_vectors_up_to_level(
		int lvl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "poset_of_orbits::recreate_schreier_vectors_up_to_level "
				"creating Schreier vectors up to "
				"level " << lvl << endl;
	}
	for (i = 0; i <= lvl; i++) {
		if (f_v) {
			cout << "poset_of_orbits::recreate_schreier_vectors_up_to_level "
					"creating Schreier vectors at "
					"level " << i << endl;
		}
		recreate_schreier_vectors_at_level(i, verbose_level - 1);
	}
	if (f_v) {
		cout << "poset_of_orbits::recreate_schreier_vectors_up_to_level done" << endl;
	}
}

void poset_of_orbits::recreate_schreier_vectors_at_level(
		int level,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = false;//(verbose_level >= 3);
	int f, l, prev, u;
	int f_recreate_extensions = false;
	int f_dont_keep_sv = false;

	if (f_v) {
		cout << "poset_of_orbits::recreate_schreier_vectors_at_level "
				"level = " << level << endl;
	}
	f = first_node_at_level(level);
	if (f_v) {
		cout << "poset_of_orbits::recreate_schreier_vectors_at_level "
				"f = " << f << endl;
	}
	//cur = Poo->first_node_at_level(level + 1);
	//l = cur - f;
	l = nb_orbits_at_level(level);

	if (f_vv) {
		cout << "creating Schreier vectors at depth " << level
				<< " for " << l << " orbits" << endl;
	}
	if (f_vv) {
		cout << "poset_of_orbits::recreate_schreier_vectors_at_level "
				"Testing if a schreier vector file exists" << endl;
	}
	if (PC->test_sv_level_file_binary(level, PC->get_problem_label_with_path())) {

		if (f_vv) {
			cout << "poset_of_orbits::recreate_schreier_vectors_at_level "
					"Yes, a schreier vector file exists. "
					"We will read this file" << endl;
		}

		PC->read_sv_level_file_binary(level, PC->get_problem_label_with_path(), false, 0, 0,
			f_recreate_extensions, f_dont_keep_sv,
			verbose_level - 2);
		if (f_vv) {
			cout << "read Schreier vectors at depth " << level
					<< " from file" << endl;
		}
		return;
	}


	if (f_vv) {
		cout << "poset_of_orbits::recreate_schreier_vectors_at_level "
				"No, a schreier vector file does not exist. "
				"We will create such a file now" << endl;
	}



	for (u = 0; u < l; u++) {

		prev = f + u;

		if (f_vv && !f_vvv) {
			cout << ".";
			if (((u + 1) % 50) == 0) {
				cout << "; " << u + 1 << " / " << l << endl;
			}
			if (((u + 1) % 1000) == 0) {
				cout << " " << u + 1 << endl;
			}
		}
		if (f_vv) {
			cout << "poset_of_orbits::recreate_schreier_vectors_at_level "
				<< level << " node " << u << " / " << l
				<< " before compute_schreier_vector" << endl;
		}

		get_node(prev)->compute_schreier_vector(
				PC, level,
				verbose_level - 1);
	}
	PC->write_sv_level_file_binary(
			level, PC->get_problem_label_with_path(), false, 0, 0,
			verbose_level);
	if (f_vv) {
		cout << "poset_of_orbits::recreate_schreier_vectors_at_level "
				"Written a file with Schreier "
				"vectors at depth " << level << endl;
	}
	if (f_vv) {
		cout << endl;
	}
	if (f_v) {
		cout << "poset_of_orbits::recreate_schreier_vectors_at_level done" << endl;
	}
}


void poset_of_orbits::find_node_by_stabilizer_order(
		int level, int order, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_nodes, node, i, j, elt_order;
	algebra::ring_theory::longinteger_object ago;
	long int set[300];

	if (f_v) {
		cout << "poset_of_orbits::find_node_by_stabilizer_order" << endl;
	}
	nb_nodes = nb_orbits_at_level(level);
	for (i = 0; i < nb_nodes; i++) {
		node = first_node_at_level(level) + i;

		get_node(node)->get_stabilizer_order(PC, ago);

		if (ago.as_int() == order) {
			cout << "found a node whose automorphism group is order "
					<< order << endl;
			cout << "the node is # " << i << " at level "
					<< level << endl;
			get_set(first_node_at_level(level) + i,
					set, level);
			Lint_vec_print(cout, set, level);
			cout << endl;

			groups::strong_generators *Strong_gens;

			get_stabilizer_generators(Strong_gens,
				level, i, 0  /* verbose_level */);

			for (j = 0; j < Strong_gens->gens->len; j++) {
				elt_order = PC->get_poset()->A->Group_element->element_order(
						Strong_gens->gens->ith(j));
				cout << "poset_classification " << j << " of order "
						<< elt_order << ":" << endl;
				if (order == elt_order) {
					cout << "CYCLIC" << endl;
					}
				PC->get_poset()->A->Group_element->element_print(
						Strong_gens->gens->ith(j), cout);
				PC->get_poset()->A->Group_element->element_print_as_permutation(
						Strong_gens->gens->ith(j), cout);
			}
			FREE_OBJECT(Strong_gens);
		}
	}
}

void poset_of_orbits::get_all_stabilizer_orders_at_level(
		int level,
		long int *&Ago, int &nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::get_all_stabilizer_orders_at_level, "
				"level = " << level << endl;
	}
	int i;

	nb = nb_orbits_at_level(level);
	if (f_v) {
		cout << "poset_of_orbits::get_all_stabilizer_orders_at_level "
				"nb = " << nb << endl;
	}
	Ago = NEW_lint(nb);
	for (i = 0; i < nb; i++) {
		Ago[i] = get_stabilizer_order_lint(level, i);
	}
	if (f_v) {
		cout << "poset_of_orbits::get_all_stabilizer_orders_at_level done" << endl;
	}
}

void poset_of_orbits::get_stabilizer_order(
		int level,
		int orbit_at_level, algebra::ring_theory::longinteger_object &go)
{
	poset_orbit_node *O;

	O = get_node_ij(level, orbit_at_level);


#if 0
	if (O->nb_strong_generators == 0) {
		go.create(1, __FILE__, __LINE__);
	}
	else {
		longinteger_domain D;

		D.multiply_up(go, O->tl, Poset->A->base_len(), 0 /* verbose_level */);
	}
#else
	O->get_stabilizer_order(PC, go);
#endif
}

long int poset_of_orbits::get_stabilizer_order_lint(
		int level,
		int orbit_at_level)
{
	poset_orbit_node *O;

	O = get_node_ij(level, orbit_at_level);
	return O->get_stabilizer_order_lint(PC);
}

void poset_of_orbits::get_stabilizer_group(
		data_structures_groups::group_container *&G,
	int level, int orbit_at_level,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	poset_orbit_node *O;
	//int node;

	if (f_v) {
		cout << "poset_of_orbits::get_stabilizer_group "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level << endl;
	}

	O = get_node_ij(level, orbit_at_level);


#if 0
	G = NEW_OBJECT(group);
	//node = first_poset_orbit_node_at_level[level] + orbit_at_level;
	//O = root + node;

	G->init(Poset->A, verbose_level - 2);
	if (f_vv) {
		cout << "poset_of_orbits::get_stabilizer_group before "
				"G->init_strong_generators_by_hdl" << endl;
	}
	G->init_strong_generators_by_hdl(O->nb_strong_generators,
			O->hdl_strong_generators, O->tl, false);
	G->schreier_sims(0);
#else
	algebra::ring_theory::longinteger_object go;

	G = NEW_OBJECT(data_structures_groups::group_container);
	O->get_stabilizer(
		PC,
		*G, go,
		verbose_level - 2);
#endif

	if (f_v) {
		cout << "poset_of_orbits::get_stabilizer_group "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level
				<< " done" << endl;
	}
}

void poset_of_orbits::get_stabilizer_generators_cleaned_up(
		groups::strong_generators *&gens,
	int level, int orbit_at_level, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::get_stabilizer_generators_cleaned_up "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level << endl;
	}
	data_structures_groups::group_container *G;

	get_stabilizer_group(G,
			level, orbit_at_level, verbose_level - 1);

	gens = NEW_OBJECT(groups::strong_generators);

	gens->init_from_sims(G->S, 0 /* verbose_level */);
	FREE_OBJECT(G);
	if (f_v) {
		cout << "poset_of_orbits::get_stabilizer_generators_cleaned_up "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level
				<< " done" << endl;
	}

}

void poset_of_orbits::get_stabilizer_generators(
		groups::strong_generators *&gens,
	int level, int orbit_at_level, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::get_stabilizer_generators "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level << endl;
	}

	poset_orbit_node *O;
	//int node;

	//node = first_poset_orbit_node_at_level[level] + orbit_at_level;
	//O = root + node;
	O = get_node_ij(
			level, orbit_at_level);

	if (f_v) {
		cout << "poset_of_orbits::get_stabilizer_generators "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level
				<< " before O->get_stabilizer_generators" << endl;
	}

	O->get_stabilizer_generators(
			PC, gens, verbose_level);

	if (f_v) {
		cout << "poset_of_orbits::get_stabilizer_generators "
				"level=" << level
				<< " orbit_at_level=" << orbit_at_level
				<< " done" << endl;
	}
}



void poset_of_orbits::orbit_element_unrank(
		int depth,
		int orbit_idx, long int rank, long int *set,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1;
	int *Elt2;
	long int *the_set;
	poset_orbit_node *O;


	if (f_v) {
		cout << "poset_of_orbits::orbit_element_unrank "
				"depth=" << depth
				<< " orbit_idx=" << orbit_idx
				<< " rank=" << rank << endl;
	}

	Elt1 = NEW_int(PC->get_poset()->A->elt_size_in_int);
	Elt2 = NEW_int(PC->get_poset()->A->elt_size_in_int);
	the_set = NEW_lint(depth);

	//O = &root[first_poset_orbit_node_at_level[depth] + orbit_idx];
	O = get_node_ij(
			depth, orbit_idx);
	coset_unrank(
			depth, orbit_idx, rank, Elt1, 0 /*verbose_level*/);

	PC->get_poset()->A->Group_element->element_invert(
			Elt1, Elt2, 0);
	O->store_set_to(PC, depth - 1, the_set);
	PC->get_poset()->A2->Group_element->map_a_set(
			the_set, set, depth, Elt2,
			0 /*verbose_level*/);

	FREE_lint(the_set);
	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "poset_of_orbits::orbit_element_unrank ";
		Lint_vec_print(cout, set, depth);
		cout << endl;
	}
}

void poset_of_orbits::orbit_element_rank(
	int depth,
	int &orbit_idx, long int &rank, long int *set,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt1;
	long int *the_set;
	long int *canonical_set;
	int i;


	if (f_v) {
		cout << "poset_of_orbits::orbit_element_rank "
				"depth=" << depth << " ";
		Lint_vec_print(cout, set, depth);
		cout << endl;
	}

	Elt1 = NEW_int(PC->get_poset()->A->elt_size_in_int);
	the_set = NEW_lint(depth);
	canonical_set = NEW_lint(depth);
	for (i = 0; i < depth; i++) {
		the_set[i] = set[i];
	}

	orbit_idx = PC->trace_set(the_set, depth, depth,
		canonical_set, Elt1,
		verbose_level - 3);

	// now Elt1 is the transporter element that moves
	// the given set to the orbit representative

	if (f_vv) {
		cout << "poset_of_orbits::orbit_element_rank "
				"after trace_set, "
				"orbit_idx = " << orbit_idx << endl;
		cout << "transporter:" << endl;
		PC->get_poset()->A->Group_element->element_print_quick(Elt1, cout);
		cout << "as permutation:" << endl;
		PC->get_poset()->A2->Group_element->element_print_as_permutation(Elt1, cout);
	}
	if (f_v) {
		cout << "calling coset_rank" << endl;
	}
	rank = coset_rank(depth, orbit_idx, Elt1, verbose_level);
	if (f_v) {
		cout << "after coset_rank, rank=" << rank << endl;
	}

	FREE_int(Elt1);
	FREE_lint(the_set);
	FREE_lint(canonical_set);
	if (f_v) {
		cout << "poset_of_orbits::orbit_element_rank "
				"orbit_idx="
				<< orbit_idx << " rank=" << rank << endl;
	}
}

void poset_of_orbits::coset_unrank(
		int depth, int orbit_idx,
		long int rank, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *the_set;
	data_structures_groups::group_container *G1, *G2;
	int *Elt_gk;
	algebra::ring_theory::longinteger_object G_order, U_order;
	poset_orbit_node *O1, *O2;

	if (f_v) {
		cout << "poset_of_orbits::coset_unrank "
				"depth=" << depth
				<< " orbit_idx=" << orbit_idx << endl;
		cout << "action A:" << endl;
		PC->get_poset()->A->print_info();
		cout << "action A2:" << endl;
		PC->get_poset()->A2->print_info();
	}

	//O1 = &root[0];
	//O2 = &root[first_poset_orbit_node_at_level[depth] + orbit_idx];
	O1 = get_node_ij(0, 0);
	O2 = get_node_ij(depth, orbit_idx);



	G1 = NEW_OBJECT(data_structures_groups::group_container);
	G2 = NEW_OBJECT(data_structures_groups::group_container);
	the_set = NEW_lint(depth);
	Elt_gk = NEW_int(PC->get_poset()->A->elt_size_in_int);

	O2->store_set_to(PC, depth - 1, the_set);

	if (f_v) {
		cout << "the set representing orbit " << orbit_idx
			<< " at level " << depth << " is ";
		Lint_vec_print(cout, the_set, depth);
		cout << endl;
	}

	O1->get_stabilizer(PC, *G1, G_order, verbose_level - 2);
	O2->get_stabilizer(PC, *G2, U_order, verbose_level - 2);


	PC->get_poset()->A->coset_unrank(G1->S, G2->S, rank, Elt, verbose_level);

	FREE_OBJECT(G1);
	FREE_OBJECT(G2);
	FREE_lint(the_set);
	FREE_int(Elt_gk);

}

long int poset_of_orbits::coset_rank(
		int depth, int orbit_idx,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int rank;
	long int *the_set;
	data_structures_groups::group_container *G1, *G2;
	int *Elt_gk;
	algebra::ring_theory::longinteger_object G_order, U_order;
	poset_orbit_node *O1, *O2;

	if (f_v) {
		cout << "poset_of_orbits::coset_rank "
				"depth=" << depth
				<< " orbit_idx=" << orbit_idx << endl;
		cout << "action A:" << endl;
		PC->get_poset()->A->print_info();
		cout << "action A2:" << endl;
		PC->get_poset()->A2->print_info();
	}

	//O1 = &root[0];
	//O2 = &root[first_poset_orbit_node_at_level[depth] + orbit_idx];
	O1 = get_node_ij(0, 0);
	O2 = get_node_ij(depth, orbit_idx);



	G1 = NEW_OBJECT(data_structures_groups::group_container);
	G2 = NEW_OBJECT(data_structures_groups::group_container);
	the_set = NEW_lint(depth);
	Elt_gk = NEW_int(PC->get_poset()->A->elt_size_in_int);

	O2->store_set_to(PC, depth - 1, the_set);

	if (f_v) {
		cout << "the set representing orbit " << orbit_idx
			<< " at level " << depth << " is ";
		Lint_vec_print(cout, the_set, depth);
		cout << endl;
	}

	O1->get_stabilizer(PC, *G1, G_order, verbose_level - 2);
	O2->get_stabilizer(PC, *G2, U_order, verbose_level - 2);


	rank = PC->get_poset()->A->coset_rank(
			G1->S, G2->S, Elt, verbose_level);

	FREE_OBJECT(G1);
	FREE_OBJECT(G2);
	FREE_lint(the_set);
	FREE_int(Elt_gk);

	return rank;
}


void poset_of_orbits::map_to_canonical_k_subset(
	long int *the_set, int set_size,
	int subset_size, int subset_rk,
	long int *reduced_set, int *transporter, int &local_idx,
	int verbose_level)
// fills reduced_set[set_size - subset_size], transporter and local_idx
// local_idx is the index of the orbit that the subset belongs to
// (in the list of orbit of subsets of size subset_size)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "poset_of_orbits::map_to_canonical_k_subset" << endl;
	}
	int *our_set;
	long int *subset;
	long int *canonical_subset;
	int *Elt1;
	int i; //, j, k;
	int reduced_set_size;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	our_set = NEW_int(set_size);
	subset = NEW_lint(set_size);
	canonical_subset = NEW_lint(set_size);
	Elt1 = NEW_int(PC->get_poset()->A->elt_size_in_int);
	reduced_set_size = set_size - subset_size;

	// unrank the k-subset and its complement to our_set[set_size]:
	Combi.unrank_k_subset_and_complement(subset_rk,
			our_set, set_size, subset_size);

	if (f_v) {
		cout << "poset_of_orbits::map_to_canonical_k_subset "
				"our_set=";
		Int_vec_print(cout, our_set, set_size);
		cout << endl;
	}

	for (i = 0; i < set_size; i++) {
		subset[i] = the_set[our_set[i]];
		//set[0][i] = subset[i]; // ToDo
	}
	for (i = 0; i < sz; i++) {
		//set[0][i] = subset[i]; // ToDo
	}
	if (f_v) {
		cout << "poset_of_orbits::map_to_canonical_k_subset "
				"subset=";
		Lint_vec_print(cout, subset, set_size);
		cout << endl;
	}

	// ToDo
	//Poset->A->element_one(poset_classification::transporter->ith(0), false);


	// trace the subset:

	if (set_size > max_set_size) {
		cout << "poset_of_orbits::map_to_canonical_k_subset "
				"set_size > max_set_size" << endl;
		cout << "poset_of_orbits::map_to_canonical_k_subset "
				"set_size = " << set_size << endl;
		cout << "poset_of_orbits::map_to_canonical_k_subset "
				"max_set_size = " << max_set_size << endl;
		exit(1);
	}

	if (f_v) {
		cout << "poset_of_orbits::map_to_canonical_k_subset "
				"before trace_set" << endl;
	}

	local_idx = PC->trace_set(
		subset, set_size, subset_size,
		canonical_subset, Elt1,
		verbose_level - 2);

	if (f_v) {
		cout << "poset_of_orbits::map_to_canonical_k_subset "
				"after trace_set local_idx=" << local_idx << endl;
		cout << "poset_of_orbits::map_to_canonical_k_subset "
				"canonical_subset=";
		Lint_vec_print(cout, canonical_subset, set_size);
		cout << endl;
	}


	if (f_v) {
		cout << "poset_of_orbits::map_to_canonical_k_subset "
				"the transporter is" << endl;
		PC->get_poset()->A->Group_element->element_print(Elt1, cout);
		cout << endl;
	}
	PC->get_poset()->A->Group_element->element_move(Elt1, transporter, false);

	for (i = 0; i < reduced_set_size; i++) {
		reduced_set[i] = canonical_subset[subset_size + i];
	}
	if (f_v) {
		cout << "poset_of_orbits::map_to_canonical_k_subset "
				"reduced set = ";
		Lint_vec_print(cout, reduced_set, reduced_set_size);
		cout << endl;
	}
	FREE_int(Elt1);
	FREE_int(our_set);
	FREE_lint(subset);
	FREE_lint(canonical_subset);

	if (f_v) {
		cout << "poset_of_orbits::"
				"map_to_canonical_k_subset done" << endl;
	}
}

void poset_of_orbits::get_representative_of_subset_orbit(
	long int *set, int size, int local_orbit_no,
	groups::strong_generators *&Strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int fst, node, sz;
	poset_orbit_node *O;

	if (f_v) {
		cout << "poset_of_orbits::get_representative_of_subset_orbit "
				"verbose_level=" << verbose_level << endl;
	}
	fst = first_node_at_level(size);
	node = fst + local_orbit_no;
	if (f_vv) {
		cout << "poset_of_orbits::get_representative_of_subset_orbit "
				"before get_set" << endl;
	}
	get_set(node, set, sz);
	if (sz != size) {
		cout << "poset_of_orbits::get_representative_of_subset_orbit "
				"sz != size" << endl;
		exit(1);
	}
	//O = root + node;
	O = get_node_ij(size, local_orbit_no);
	if (f_vv) {
		cout << "poset_of_orbits::get_representative_of_subset_orbit "
				"before get_stabilizer_poset_classifications" << endl;
	}
	O->get_stabilizer_generators(PC, Strong_gens, 0);
	if (f_v) {
		cout << "poset_of_orbits::get_representative_of_subset_orbit done" << endl;
	}
}


void poset_of_orbits::get_orbit_representatives(
		int level,
		int &nb_orbits, long int *&Orbit_reps, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "poset_of_orbits::get_orbit_representatives" << endl;
	}
	nb_orbits = nb_orbits_at_level(level);
	if (f_v) {
		cout << "poset_of_orbits::get_orbit_representatives we found " << nb_orbits
				<< " orbits on " << level << "-sets" << endl;
	}
	Orbit_reps = NEW_lint(nb_orbits * level);
	for (i = 0; i < nb_orbits; i++) {
		get_set_by_level(level, i, Orbit_reps + i * level);
	}

	if (f_v) {
		cout << "poset_of_orbits::get_orbit_representatives done" << endl;
	}
}


void poset_of_orbits::get_all_orbits(
		other::data_structures::set_of_sets *&All_orbits,
		int *&Nb_orbits,
		int &nb_orbits_total,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "poset_of_orbits::get_all_orbits" << endl;
	}

	int lvl, po, ol, el, node, i;
	//int *Nb_orbits;
	//int nb_orbits_total;

	Nb_orbits = NEW_int(PC->get_depth() + 1);

	nb_orbits_total = 0;
	for (i = 0; i <= PC->get_depth(); i++) {
		Nb_orbits[i] = nb_orbits_at_level(i);
		nb_orbits_total += Nb_orbits[i];
	}

	int underlying_set_size;

	underlying_set_size = PC->get_A2()->degree;

	long int *Sz;

	Sz = NEW_lint(nb_orbits_total);

	for (lvl = 0; lvl <= PC->get_depth(); lvl++) {

		for (po = 0; po < nb_orbits_at_level(lvl); po++) {

			node = first_node_at_level(lvl) + po;
			ol = orbit_length_as_int(po, lvl);
			Sz[node] = ol * lvl;
		}
	}


	All_orbits = NEW_OBJECT(other::data_structures::set_of_sets);
	All_orbits->init_basic(
			underlying_set_size, nb_orbits_total,
			Sz, verbose_level);


	for (lvl = 0; lvl <= PC->get_depth(); lvl++) {

		if (f_vv) {
			cout << "poset_of_orbits::get_all_orbits "
					"lvl=" << lvl << " / " << PC->get_depth() << endl;
		}

		for (po = 0; po < nb_orbits_at_level(lvl); po++) {

			node = first_node_at_level(lvl) + po;


			ol = orbit_length_as_int(po, lvl);

			long int *Orbit;

			Orbit = All_orbits->Sets[node];


			for (el = 0; el < ol; el++) {
				if (false) {
					cout << "unrank " << lvl << ", " << po
							<< ", " << el << endl;
				}
				orbit_element_unrank(lvl, po, el, Orbit + el * lvl,
						0 /* verbose_level */);

				if (false) {
					cout << "set=";
					Lint_vec_print(cout, Orbit + el * lvl, lvl);
					cout << endl;
				}
			}

		}
	}

	//FREE_int(Nb_orbits);
	FREE_lint(Sz);

	if (f_v) {
		cout << "poset_of_orbits::get_all_orbits done" << endl;
	}
}


void poset_of_orbits::get_all_orbits_expanded(
		other::data_structures::set_of_sets *&All_orbits,
		int *&Nb_orbits,
		int *&Orbit_first,
		int &nb_orbits_total,
		int &nb_sets_total,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "poset_of_orbits::get_all_orbits_expanded" << endl;
	}

	int lvl, po, ol, el, node;

	Nb_orbits = NEW_int(PC->get_depth() + 1);

	nb_orbits_total = 0;
	nb_sets_total = 0;
	for (lvl = 0; lvl <= PC->get_depth(); lvl++) {
		Nb_orbits[lvl] = nb_orbits_at_level(lvl);
		nb_orbits_total += Nb_orbits[lvl];
		for (po = 0; po < Nb_orbits[lvl]; po++) {
			ol = orbit_length_as_int(po, lvl);
			nb_sets_total += ol;
		}
	}

	Orbit_first = NEW_int(nb_sets_total);

	int underlying_set_size;

	underlying_set_size = PC->get_A2()->degree;

	long int *Sz;
	int cur, j;

	Sz = NEW_lint(nb_sets_total);

	cur = 0;
	for (lvl = 0; lvl <= PC->get_depth(); lvl++) {

		for (po = 0; po < nb_orbits_at_level(lvl); po++) {

			node = first_node_at_level(lvl) + po;
			Orbit_first[node] = cur;
			ol = orbit_length_as_int(po, lvl);
			for (j = 0; j < ol; j++) {
				Sz[cur] = lvl;
				cur++;
			}
		}
	}


	All_orbits = NEW_OBJECT(other::data_structures::set_of_sets);
	All_orbits->init_basic(
			underlying_set_size, nb_sets_total,
			Sz, verbose_level);


	cur = 0;
	for (lvl = 0; lvl <= PC->get_depth(); lvl++) {

		if (f_vv) {
			cout << "poset_of_orbits::get_all_orbits_expanded "
					"lvl=" << lvl << " / " << PC->get_depth() << endl;
		}

		for (po = 0; po < nb_orbits_at_level(lvl); po++) {

			//node = Poo->first_node_at_level(lvl) + po;


			ol = orbit_length_as_int(po, lvl);

			long int *Set;



			for (el = 0; el < ol; el++) {
				if (false) {
					cout << "poset_of_orbits::get_all_orbits_expanded "
							"unrank " << lvl << ", " << po
							<< ", " << el << endl;
				}

				Set = All_orbits->Sets[cur];

				orbit_element_unrank(lvl, po, el, Set,
						0 /* verbose_level */);

				if (false) {
					cout << "poset_of_orbits::get_all_orbits_expanded "
							"set=";
					Lint_vec_print(cout, Set, lvl);
					cout << endl;
				}

				cur++;
			}

		}
	}

	//FREE_int(Nb_orbits);
	FREE_lint(Sz);

	if (f_v) {
		cout << "poset_of_orbits::get_all_orbits_expanded done" << endl;
	}
}

other::data_structures::lint_matrix *poset_of_orbits::get_all_orbit_elements(
		int lvl, int po,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::get_all_orbit_elements" << endl;
	}


	int ol, el;

	ol = orbit_length_as_int(po, lvl);

	other::data_structures::lint_matrix *M;


	M = NEW_OBJECT(other::data_structures::lint_matrix);
	M->allocate(
		ol, lvl);

	long int *Set;


	Set = NEW_lint(lvl);


	for (el = 0; el < ol; el++) {
		if (false) {
			cout << "poset_of_orbits::get_all_orbit_elements "
					"unrank " << lvl << ", " << po
					<< ", " << el << endl;
		}


		orbit_element_unrank(lvl, po, el, Set,
				0 /* verbose_level */);

		if (false) {
			cout << "poset_of_orbits::get_all_orbit_elements set=";
			Lint_vec_print(cout, Set, lvl);
			cout << endl;
		}

		Lint_vec_copy(Set, M->Data + el * lvl, lvl);

	}

	FREE_lint(Set);

	if (f_v) {
		cout << "poset_of_orbits::get_all_orbit_elements done" << endl;
	}
	return M;

}


void poset_of_orbits::get_all_orbits_expanded_table(
		other::data_structures::set_of_sets *&All_orbits,
		int *&Nb_orbits,
		int *&Orbit_first,
		int &nb_orbits_total,
		int &nb_sets_total,
		std::string *&Table,
		std::string *&Col_headings,
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "poset_of_orbits::get_all_orbits_expanded_table" << endl;
	}


	//other::data_structures::set_of_sets *All_orbits;


	if (f_v) {
		cout << "poset_of_orbits::get_all_orbits_expanded_table "
				"before get_all_orbits_expanded" << endl;
	}
	get_all_orbits_expanded(
			All_orbits,
			Nb_orbits,
			Orbit_first,
			nb_orbits_total,
			nb_sets_total,
			verbose_level);
	if (f_v) {
		cout << "poset_of_orbits::get_all_orbits_expanded_table "
				"after get_all_orbits_expanded" << endl;
	}


	nb_cols = 5;
	nb_rows = nb_sets_total;
	Table = new std::string [nb_rows * nb_cols];
	Col_headings = new std::string [nb_cols];

	Col_headings[0] = "Index";
	Col_headings[1] = "Layer";
	Col_headings[2] = "Orbit";
	Col_headings[3] = "Element";
	Col_headings[4] = "Set";

	int lvl, po, ol, el, node, cur;

	for (lvl = 0; lvl <= PC->get_depth(); lvl++) {

		if (f_vv) {
			cout << "poset_of_orbits::get_all_orbits_expanded_table "
					"lvl=" << lvl << " / " << PC->get_depth() << endl;
		}

		for (po = 0; po < nb_orbits_at_level(lvl); po++) {

			node = first_node_at_level(lvl) + po;


			ol = orbit_length_as_int(po, lvl);

			long int *Set;



			for (el = 0; el < ol; el++) {
				if (false) {
					cout << "unrank " << lvl << ", " << po
							<< ", " << el << endl;
				}

				cur = Orbit_first[node] + el;

				Set = All_orbits->Sets[cur];

				Table[cur * nb_cols + 0] = std::to_string(cur);
				Table[cur * nb_cols + 1] = std::to_string(lvl);
				Table[cur * nb_cols + 2] = std::to_string(po);
				Table[cur * nb_cols + 3] = std::to_string(el);
				Table[cur * nb_cols + 4] = "\"" + Lint_vec_stringify(Set, lvl) + "\"";
			}
		}
	}

	if (f_v) {
		cout << "poset_of_orbits::get_all_orbits_expanded_table done" << endl;
	}
}



}}}


