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
	int f_v = FALSE; //(verbose_level >= 1);

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


void poset_of_orbits::init(poset_classification *PC,
		int nb_poset_orbit_nodes, int sz, int max_set_size, long int t0,
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

	init_poset_orbit_node(
			nb_poset_orbit_nodes,
			verbose_level);

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
		cout << "poset_of_orbits::reallocate reducing length from " << length_wanted << " to " << length << endl;
		poset_orbit_nodes_increment = length_wanted - nb_poset_orbit_nodes_allocated;
	}
	cout << "poset_of_orbits::reallocate from " << nb_poset_orbit_nodes_allocated << " to " << length << endl;
	reallocate_to(length, verbose_level - 1);
	poset_orbit_nodes_increment_last = poset_orbit_nodes_increment;
	poset_orbit_nodes_increment = increment_new;

}

void poset_of_orbits::reallocate_to(long int new_number_of_nodes,
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

long int poset_of_orbits::get_nb_poset_orbit_nodes_allocated()
{
	return nb_poset_orbit_nodes_allocated;
}

long int poset_of_orbits::get_nb_extension_nodes_at_level_total(int level)
{
	return nb_extension_nodes_at_level_total[level];
}

void poset_of_orbits::set_nb_poset_orbit_nodes_used(int value)
{
	nb_poset_orbit_nodes_used = value;
}

int poset_of_orbits::first_node_at_level(int i)
{
	return first_poset_orbit_node_at_level[i];
}

void poset_of_orbits::set_first_node_at_level(int i, int value)
{
	first_poset_orbit_node_at_level[i] = value;
}

poset_orbit_node *poset_of_orbits::get_node(int node_idx)
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

int poset_of_orbits::nb_orbits_at_level(int level)
{
	int f, l;

	f = first_poset_orbit_node_at_level[level];
	l = first_poset_orbit_node_at_level[level + 1] - f;
	return l;
}

long int poset_of_orbits::nb_flag_orbits_up_at_level(int level)
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

poset_orbit_node *poset_of_orbits::get_node_ij(int level, int node)
{
	int f;

	f = first_poset_orbit_node_at_level[level];
	return root + f + node;
}

int poset_of_orbits::node_get_nb_of_extensions(int node)
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

long int poset_of_orbits::count_extension_nodes_at_level(int lvl)
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

double poset_of_orbits::level_progress(int lvl)
{
	return
		((double)(nb_fusion_nodes_at_level[lvl] +
				nb_extension_nodes_at_level[lvl])) /
			(double) nb_extension_nodes_at_level_total[lvl];
}

void poset_of_orbits::change_extension_type(int level,
		int node, int cur_ext, int type, int verbose_level)
{
	if (type == EXTENSION_TYPE_EXTENSION) {
		// extension node
		if (root[node].get_E(cur_ext)->get_type() != EXTENSION_TYPE_UNPROCESSED &&
			root[node].get_E(cur_ext)->get_type() != EXTENSION_TYPE_PROCESSING) {
			cout << "poset_classification::change_extension_type trying to install "
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
			cout << "poset_classification::change_extension_type trying to install "
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

void poset_of_orbits::get_table_of_nodes(long int *&Table,
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

		ring_theory::longinteger_object go;

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

void poset_of_orbits::print_progress_by_level(int lvl)
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

	cout << "poset_classification::print_tree "
			"nb_poset_orbit_nodes_used="
			<< nb_poset_orbit_nodes_used << endl;
	for (i = 0; i < nb_poset_orbit_nodes_used; i++) {
		PC->print_node(i);
	}
}

void poset_of_orbits::init_root_node_from_base_case(int verbose_level)
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
				"storing strong poset_classifications" << endl;
	}
	root[PC->get_Base_case()->size].store_strong_generators(PC, PC->get_Base_case()->Stab_gens);
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

void poset_of_orbits::init_root_node(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::init_root_node" << endl;
	}
	if (PC->has_base_case()) {

		if (f_v) {
			cout << "poset_of_orbits::init_root_node before init_root_node_from_base_case" << endl;
		}
		init_root_node_from_base_case(verbose_level);
		if (f_v) {
			cout << "poset_of_orbits::init_root_node after init_root_node_from_base_case" << endl;
		}

	}
	else {
		if (f_v) {
			cout << "poset_of_orbits::init_root_node before root[0].init_root_node" << endl;
		}
		root[0].init_root_node(PC, verbose_level - 1);
		if (f_v) {
			cout << "poset_of_orbits::init_root_node after root[0].init_root_node" << endl;
		}
	}
	if (f_v) {
		cout << "poset_of_orbits::init_root_node done" << endl;
	}
}

void poset_of_orbits::make_tabe_of_nodes(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_of_orbits::make_tabe_of_nodes" << endl;
	}
	long int *Table;
	int nb_rows, nb_cols;
	string fname;
	orbiter_kernel_system::file_io Fio;

	get_table_of_nodes(Table,
		nb_rows, nb_cols, 0 /*verbose_level*/);

	fname.assign(PC->get_problem_label_with_path());
	fname.append("_table_of_orbits.csv");

	Fio.lint_matrix_write_csv(fname, Table, nb_rows, nb_cols);

	if (f_v) {
		cout << "poset_classification::post_processing written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}


	FREE_lint(Table);

	if (f_v) {
		cout << "poset_of_orbits::make_tabe_of_nodes done" << endl;
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
		orbiter_kernel_system::memory_object *m, int &nb_group_elements,
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
				orbiter_kernel_system::os_interface Os;

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
		orbiter_kernel_system::memory_object *m, int &nb_group_elements,
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
				orbiter_kernel_system::os_interface Os;

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

long int poset_of_orbits::calc_size_on_file(int depth_completed,
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
	orbiter_kernel_system::file_io Fio;

	f = first_node_at_level(level);
	nb_nodes = nb_orbits_at_level(level);
	if (f_v) {
		cout << "poset_of_orbits::read_sv_level_file_binary2 "
				<< nb_nodes << " nodes" << endl;
		cout << "f_recreate_extensions="
				<< f_recreate_extensions << endl;
		cout << "f_dont_keep_sv=" << f_dont_keep_sv << endl;
		if (f_split) {
			cout << "f_split is TRUE, split_mod=" << split_mod
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
	orbiter_kernel_system::file_io Fio;

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
	orbiter_kernel_system::file_io Fio;

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
	int f_v = FALSE;//(verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

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
		const char *fname_base,
		int lvl, int t0, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;
	char fname[1000];

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
	orbiter_kernel_system::file_io Fio;

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

void poset_of_orbits::read_level_file(int level,
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
	orbiter_kernel_system::file_io Fio;

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

		J = PC->find_poset_orbit_node_for_set(level - 1,
				sets[i], FALSE /* f_tolerant */,
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
				Aut.init_ascii_coding(data[i], verbose_level - 2);

				Aut.decode_ascii(FALSE);

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
				O->store_strong_generators(PC, Strong_gens);
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
	char str[1000];
	orbiter_kernel_system::file_io Fio;
	orbiter_kernel_system::os_interface Os;

	fname1.assign(fname_base);
	snprintf(str, sizeof(str), "_lvl_%d_candidates.txt", lvl);
	fname1.append(str);
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
		int lvl, long int *&Data, int &nb_reps, int verbose_level)
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
	orbiter_kernel_system::file_io Fio;

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

			orbiter_kernel_system::Orbiter->Lint_vec->create_string_with_quotes(S, Data + i * lvl, lvl);
			f << "," << S << endl;
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
	orbiter_kernel_system::file_io Fio;
	orbiter_kernel_system::os_interface Os;

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
		PC->compute_and_print_automorphism_group_orders(lvl, f);
		f << endl;
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
	orbiter_kernel_system::os_interface Os;


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
	PC->compute_and_print_automorphism_group_orders(lvl, f);
	f << endl;
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
		cout << "poset_classification::log_nodes_for_treefile "
				"cur=" << cur << endl;
	}
	if (PC->has_base_case() && cur < PC->get_Base_case()->size) {
		return; // !!!
	}

	node->log_current_node(PC, depth, f,
			FALSE /* f_with_strong_generators */, 0);

	if (f_recurse) {
		//cout << "recursing into dependent nodes" << endl;
		for (i = 0; i < node->get_nb_of_extensions(); i++) {
			if (node->get_E(i)->get_type() == EXTENSION_TYPE_EXTENSION) {
				if (node->get_E(i)->get_data() >= 0) {
					next = node->get_E(i)->get_data();
					log_nodes_for_treefile(next,
							depth + 1, f, TRUE, verbose_level);
				}
			}
		}
	}
}

void poset_of_orbits::save_representatives_at_level_to_csv(std::string &fname,
		int lvl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;
	long int *set;
	long int ago;

	if (f_v) {
		cout << "poset_classification::save_representatives_at_level_to_csv" << endl;
	}
	{
		ofstream ost(fname);

		set = NEW_lint(lvl);


		l = PC->nb_orbits_at_level(lvl);
		//cout << "The " << l << " representatives at level " << lvl << " are:" << endl;
		ost << "ROW,REP,AGO,OL" << endl;
		for (i = 0; i < l; i++) {
			get_node_ij(lvl, i)->store_set_to(PC, lvl - 1, set /*gen->S0*/);
			//Orbiter->Lint_vec.print(cout, set /*gen->S0*/, lvl);

			ost << i;
			{
				string str;
				ost << ",";
				orbiter_kernel_system::Orbiter->Lint_vec->create_string_with_quotes(str, set, lvl);
				ost << str;
			}

			ago = get_node_ij(lvl, i)->get_stabilizer_order_lint(PC);
			ost << "," << ago;

			ring_theory::longinteger_object len;

			PC->orbit_length(i, lvl, len);

			ost << "," << len;


			ost << endl;

		}

		ost << "END" << endl;


		FREE_lint(set);
	}
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "poset_classification::save_representatives_at_level_to_csv done" << endl;
	}
}




}}}


