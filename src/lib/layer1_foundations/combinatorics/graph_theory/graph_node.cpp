// graph_node.cpp
// 
// Anton Betten
// December 30, 2013
//
//
// 
//
//

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace graph_theory {


graph_node::graph_node()
{
	Record_birth();
	// std::string label;
	id = -1;
	f_has_data1 = false;
	data1 = -1;
	f_has_data2 = false;
	data2 = -1;
	f_has_data3 = false;
	data3 = -1;
	f_has_vec_data = false;
	vec_data = NULL;
	vec_data_len = 0;

	f_has_distinguished_element = false;
	distinguished_element_index = -1;


	layer = -1;
	neighbor_list_allocated = 0;
	nb_neighbors = 0;
	neighbor_list = NULL;
	Edge_color = NULL;
	x_coordinate = 0;

	nb_children = 0;
	nb_children_allocated = 0;
	child_id = NULL;
	weight_of_subtree = 1;
	width = 0;
	depth_first_node_rank = -1;
	radius_factor = 1.;
}


graph_node::~graph_node()
{
	Record_death();
	if (neighbor_list) {
		FREE_int(neighbor_list);
	}
	if (Edge_color) {
		FREE_int(Edge_color);
	}
	if (f_has_vec_data) {
		FREE_lint(vec_data);
	}
	if (child_id) {
		FREE_int(child_id);
	}
}

void graph_node::add_neighbor(
		int l, int n, int id, int edge_color)
{
	int i;
	
	if (nb_neighbors >= neighbor_list_allocated) {
		int new_neighbor_list_allocated;
		int *new_neighbor_list;
		int *new_edge_color;
		
		if (neighbor_list_allocated) {
			new_neighbor_list_allocated = 2 * neighbor_list_allocated;
		}
		else {
			new_neighbor_list_allocated = 16;
		}
		new_neighbor_list = NEW_int(new_neighbor_list_allocated);
		new_edge_color = NEW_int(new_neighbor_list_allocated);
		for (i = 0; i < nb_neighbors; i++) {
			new_neighbor_list[i] = neighbor_list[i];
			new_edge_color[i] = Edge_color[i];
		}
		if (neighbor_list) {
			FREE_int(neighbor_list);
		}
		if (Edge_color) {
			FREE_int(Edge_color);
		}
		neighbor_list = new_neighbor_list;
		Edge_color = new_edge_color;
		neighbor_list_allocated = new_neighbor_list_allocated;
	}
	neighbor_list[nb_neighbors] = id;
	Edge_color[nb_neighbors] = edge_color;
	nb_neighbors++;
}

void graph_node::add_text(
		std::string &text)
{

	label.assign(text);
}

void graph_node::add_vec_data(
		long int *v, int len)
{
	vec_data = NEW_lint(len);
	vec_data_len = len;
	Lint_vec_copy(v, vec_data, len);
	f_has_vec_data = true;
}

void graph_node::set_distinguished_element(
		int idx)
{
	f_has_distinguished_element = true;
	distinguished_element_index = idx;
}


void graph_node::add_data1(
		int data)
{
	f_has_data1 = true;
	data1 = data;
}

void graph_node::add_data2(
		int data)
{
	f_has_data2 = true;
	data2 = data;
}

void graph_node::add_data3(
		int data)
{
	f_has_data3 = true;
	data3 = data;
}

void graph_node::write_memory_object(
		other::orbiter_kernel_system::memory_object *m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "graph_node::write_memory_object" << endl;
	}
	m->write_string(label);

	m->write_int(id);
	m->write_int(f_has_data1);
	m->write_int(data1);
	m->write_int(f_has_data2);
	m->write_int(data2);
	m->write_int(f_has_data3);
	m->write_int(data3);
	m->write_int(f_has_vec_data);
	if (f_has_vec_data) {
		m->write_int(vec_data_len);
		for (i = 0; i < vec_data_len; i++) {
			m->write_lint(vec_data[i]);
		}
	}
	m->write_int(f_has_distinguished_element);
	m->write_int(distinguished_element_index);
	m->write_int(layer);
	m->write_int(nb_neighbors);
	for (i = 0; i < nb_neighbors; i++) {
		m->write_int(neighbor_list[i]);
		m->write_int(Edge_color[i]);
	}
	m->write_double(x_coordinate);
	m->write_double(radius_factor);
	if (f_v) {
		cout << "graph_node::write_memory_object "
				"finished, data size (in chars) = "
				<< m->used_length << endl;
	}
}

void graph_node::read_memory_object(
		other::orbiter_kernel_system::memory_object *m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "graph_node::read_memory_object" << endl;
	}
	m->read_string(label);
	m->read_int(&id);
	m->read_int(&f_has_data1);
	m->read_int(&data1);
	m->read_int(&f_has_data2);
	m->read_int(&data2);
	m->read_int(&f_has_data3);
	m->read_int(&data3);
	m->read_int(&f_has_vec_data);
	if (f_has_vec_data) {
		m->read_int(&vec_data_len);
		vec_data = NEW_lint(vec_data_len);
		for (i = 0; i < vec_data_len; i++) {
			m->read_lint(&vec_data[i]);
		}
	}
	else {
		vec_data_len = 0;
		vec_data = NULL;
	}
	
	m->read_int(&f_has_distinguished_element);
	m->read_int(&distinguished_element_index);


	m->read_int(&layer);
	m->read_int(&nb_neighbors);
	neighbor_list = NEW_int(nb_neighbors);
	Edge_color = NEW_int(nb_neighbors);
	for (i = 0; i < nb_neighbors; i++) {
		m->read_int(&neighbor_list[i]);
		m->read_int(&Edge_color[i]);
	}
	
	m->read_double(&x_coordinate);
	m->read_double(&radius_factor);
	if (f_v) {
		cout << "graph_node::read_memory_object finished" << endl;
	}
}

void graph_node::allocate_tree_structure(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "graph_node::allocate_tree_structure" << endl;
	}
	
	nb_children = 0;
	nb_children_allocated = 1000;
	child_id = NEW_int(nb_children_allocated);
	
	if (f_v) {
		cout << "graph_node::allocate_tree_structure done" << endl;
	}
}

int graph_node::remove_neighbor(
		layered_graph *G, int id,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "graph_node::remove_neighbor" << endl;
	}

	for (i = 0; i < nb_neighbors; i++) {
		if (neighbor_list[i] == id) {
			for (j = i + 1; j < nb_neighbors; j++) {
				neighbor_list[j - 1] = neighbor_list[j];
				Edge_color[j - 1] = Edge_color[j];
			}
			nb_neighbors--;
			return true;
		}
	}

	if (f_v) {
		cout << "graph_node::remove_neighbor done" << endl;
	}
	return false;
}

void graph_node::find_all_parents(
		layered_graph *G, std::vector<int> &All_Parents,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, id = 0, l, n, my_layer;

	if (f_v) {
		cout << "graph_node::find_parent layer = " << layer << endl;
	}

	G->find_node_by_id(graph_node::id, my_layer, n);
	if (f_v) {
		cout << "graph_node::find_parent my_layer = " << my_layer << endl;
	}

	for (i = 0; i < nb_neighbors; i++) {
		id = neighbor_list[i];
		G->find_node_by_id(id, l, n);

		if (f_v) {
			cout << "graph_node::find_parent i=" << i << " / " << nb_neighbors
					<< " id=" << id << " l=" << l << " n=" << n << endl;
		}

		if (l < my_layer) {
			All_Parents.push_back(id);
		}
	}

	if (f_v) {
		cout << "graph_node::find_parent done" << endl;
	}
}

int graph_node::find_parent(
		layered_graph *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, id = 0, l, n;
	
	if (f_v) {
		cout << "graph_node::find_parent" << endl;
	}
	
	for (i = 0; i < nb_neighbors; i++) {
		id = neighbor_list[i];
		G->find_node_by_id(id, l, n);
		if (l < layer) {
			if (f_v) {
				cout << "graph_node::find_parent done" << endl;
			}
			return id;
		}
	}
	cout << "graph_node::find_parent did not find "
			"a parent node" << endl;
	cout << "layer = " << layer << endl;
	cout << "id = " << id << endl;
	exit(1);
}

void graph_node::register_child(
		layered_graph *G,
		int id_child, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int l, n;
	
	if (f_v) {
		cout << "graph_node::register_child" << endl;
	}
	if (nb_children == nb_children_allocated) {
		nb_children_allocated = 2 * nb_children_allocated;
		int *child_id_new;

		child_id_new = NEW_int(nb_children_allocated);
		Int_vec_copy(child_id, child_id_new, nb_children);
		FREE_int(child_id);
		child_id = child_id_new;
	}
	child_id[nb_children++] = id_child;
	
	G->find_node_by_id(id_child, l, n);
	weight_of_subtree += G->L[l].Nodes[n].weight_of_subtree;
	
	if (f_v) {
		cout << "graph_node::register_child done" << endl;
	}
}

void graph_node::place_x_based_on_tree(
		layered_graph *G,
		double left, double right,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, w, w0, w1;
	int id_child, l, n;
	double lft, rgt;
	double dx;

	if (f_v) {
		cout << "graph_node::place_x_based_on_tree" << endl;
	}
	
	x_coordinate = (left + right) * .5;
	w = weight_of_subtree;
	width = right - left;
	dx = (double) width / (double) (w - 1);
		// the node itself counts for the
		// weight, so we subtract one
	w0 = 0;
	
	for (i = 0; i < nb_children; i++) {
		id_child = child_id[i];
		G->find_node_by_id(id_child, l, n);

		w1 = G->L[l].Nodes[n].weight_of_subtree;
		lft = left + (double)w0 * dx;
		rgt = left + ((double)(w0 + w1)) * dx;
	
		G->L[l].Nodes[n].place_x_based_on_tree(G,
				lft, rgt, verbose_level);
		w0 += w1;
	}


	if (f_v) {
		cout << "graph_node::place_x_based_on_tree done" << endl;
	}
}

void graph_node::depth_first_rank_recursion(
		layered_graph *G, int &r, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, id_child, l, n;

	if (f_v) {
		cout << "graph_node::depth_first_rank_recursion" << endl;
	}

	depth_first_node_rank = r++;

	for (i = 0; i < nb_children; i++) {
		id_child = child_id[i];
		G->find_node_by_id(id_child, l, n);

		G->L[l].Nodes[n].depth_first_rank_recursion(G, r, verbose_level);
	}
	if (f_v) {
		cout << "graph_node::depth_first_rank_recursion done" << endl;
	}
}

void graph_node::scale_x_coordinate(
		double x_stretch, int verbose_level)
{
	x_coordinate *= x_stretch;
}



}}}}




