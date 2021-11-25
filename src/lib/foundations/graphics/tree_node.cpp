// tree_node.cpp
//
// Anton Betten
//
// moved here from tree.cpp: October 12, 2013
//
// February 7, 2003

#include "foundations.h"


using namespace std;


#define DONT_DRAW_ROOT_NODE 0

namespace orbiter {
namespace foundations {



tree_node::tree_node()
{
	parent = NULL;
	depth = 0;
	f_value = FALSE;
	value = 0;

	f_int_data = FALSE;
	int_data = 0;
	char_data = NULL;
	nb_children = 0;
	children = NULL;


	weight = 0;
	placement_x = 0;
	placement_y = 0;
	width = 0;

	DFS_rank = 0;
}

tree_node::~tree_node()
{
}

void tree_node::init(int depth, tree_node *parent, int f_value, int value, 
	int f_i_data, int i_data, char *c_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "tree_node::init depth=" << depth << " value=" << value << endl;
		}
	tree_node::depth = depth;
	tree_node::parent = parent;
	
	tree_node::f_value = f_value;
	tree_node::value = value;
	
	f_int_data = f_i_data;
	int_data = i_data;
	if (c_data) {
		char_data = NEW_char(strlen(c_data) + 1);
		strcpy(char_data, c_data);
		}
	else 
		char_data = NULL;
	
	nb_children = 0;
	children = NULL;
}

void tree_node::print_path()
{
	if (parent) {
		parent->print_path();
		}
	if (f_value)
		cout << value << " ";
}

void tree_node::print_depth_first()
{
	int i;
	
	cout << "depth " << depth << " : ";
	print_path();
#if 0
	if (f_value) {
		cout << value;
		}
#endif
	cout << " : ";
	cout << weight;
	cout << " : (";
	cout << placement_x << "," << placement_y << "," << width << ")";
	cout << " : ";
	if (f_int_data) {
		cout << int_data;
		}
	cout << " : ";
	if (char_data)
		cout << char_data;
	cout << endl;
	for (i = 0; i < nb_children; i++) {
		children[i]->print_depth_first();
		}
}

void tree_node::compute_DFS_rank(int &rk)
{
	int i;
	
	DFS_rank = rk;
	rk++;
	for (i = 0; i < nb_children; i++) {
		children[i]->compute_DFS_rank(rk);
		}
}

void tree_node::get_coordinates(int &idx, int *coord_xy)
{
	int i;
	
	coord_xy[idx * 2 + 0] = placement_x;
	coord_xy[idx * 2 + 1] = placement_y;
	idx++;
	for (i = 0; i < nb_children; i++) {
		children[i]->get_coordinates(idx, coord_xy);
		}
}

void tree_node::get_coordinates_and_width(int &idx, int *coord_xyw)
{
	int i;
	
	coord_xyw[idx * 3 + 0] = placement_x;
	coord_xyw[idx * 3 + 1] = placement_y;
	coord_xyw[idx * 3 + 2] = width;
	idx++;
	for (i = 0; i < nb_children; i++) {
		children[i]->get_coordinates_and_width(idx, coord_xyw);
		}
}

void tree_node::calc_weight()
{
	int i;

	weight = 1;
	for (i = 0; i < nb_children; i++) {
		children[i]->calc_weight();
		weight += children[i]->weight;
		}
}

void tree_node::place_xy(int left, int right, int ymax, int max_depth)
{
	int i, w, w0, w1, lft, rgt;
	double dx;
	
	placement_x = (left + right) >> 1;
	placement_y = tree_node_calc_y_coordinate(ymax, depth, max_depth);
	w = weight;
	width = right - left;
	dx = (double) width / (double) (w - 1);
		// the node itself counts for the weight, so we subtract one
	w0 = 0;
	
	for (i = 0; i < nb_children; i++) {
		w1 = children[i]->weight;
		lft = left + (int)((double)w0 * dx);
		rgt = left + (int)((double)(w0 + w1) * dx);
		children[i]->place_xy(lft, rgt, ymax, max_depth);
		w0 += w1;
		}
}

void tree_node::place_on_circle(int xmax, int ymax, int max_depth)
{
	int i, dy;
	double x, y;
	double x1, y1;

	x = placement_x;
	y = placement_y;
	dy = (int)((double)ymax / (double)(max_depth + 1));
	y = ymax - y;
	y -= dy * 0.5;
	y /= ymax;
	x /= (double) xmax;
	x *= 2. * M_PI;
	x -= M_PI;
	x1 = y * cos(x) * xmax * 0.5 + xmax * 0.5;
	y1 = y * sin(x) * ymax * 0.5 + ymax * 0.5;
	placement_x = (int) x1;
	placement_y = (int) y1;
	for (i = 0; i < nb_children; i++) {
		children[i]->place_on_circle(xmax, ymax, max_depth);
		}
}

void tree_node::add_node(int l, int depth, int *path, int i_data, char *c_data, 
	int verbose_level)
{
	int i, idx;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "tree_node::add_node: depth=" << depth << " : ";
		Orbiter->Int_vec.print(cout, path, l);
		cout << endl;
		}
	if (l == 0) {
		if (f_vv) {
			cout << "add_node(): node of length 0" << endl;
			}
		init(0, NULL, TRUE, -1, TRUE, i_data, c_data, verbose_level);
		return;
		}
	idx = find_child(path[depth]);
	if (f_vv) {
		cout << "add_node(): find_child for " << path[depth] << " returns " << idx << endl;
		}
	if (idx == -1) {
		tree_node **new_children = new ptree_node[nb_children + 1];
		for (i = 0; i < nb_children; i++) {
			new_children[i] = children[i];
			}
		new_children[nb_children] = new tree_node;
		if (nb_children)
			delete [] children;
		children = new_children;
		nb_children++;
		if (f_vv) {
			cout << "nb_children increased to " << nb_children << endl;
			}
		
		if (l == depth + 1) {
			if (f_vv) {
				cout << "initializing terminal node" << endl;
				}
			children[nb_children - 1]->init(depth + 1, this, TRUE, path[depth], TRUE, i_data, c_data, verbose_level);
			return;
			}
		else {
			if (f_vv) {
				cout << "initializing intermediate node" << endl;
				}
			children[nb_children - 1]->init(depth + 1, this, TRUE, path[depth], FALSE, 0, NULL, verbose_level);
			idx = nb_children - 1;
			}
		}
	if (f_vv) {
		cout << "searching deeper" << endl;
		}
	children[idx]->add_node(l, depth + 1, path, i_data, c_data, verbose_level);
}

int tree_node::find_child(int val)
{
	int i;
	
	for (i = 0; i < nb_children; i++) {
		if (children[i]->value == val)
			return i;
		}
	return -1;
}

void tree_node::get_values(int *v)
{
	cout << "get_values depth=" << depth << " value=" << value << endl;
	if (depth) {
		v[depth - 1] = value;
		parent->get_values(v);
		}
}

void tree_node::draw_edges(mp_graphics &G,
		layered_graph_draw_options *Opt,
		int f_i,
	int f_has_parent, int parent_x, int parent_y, int max_depth,
	int f_has_draw_vertex_callback, 
	void (*draw_vertex_callback)(tree *T, mp_graphics *G, int *v, int layer, tree_node *N, int x, int y, int dx, int dy),
	tree *T)
{
	//int rad = 20;
	//int dx = rad; // / sqrt(2);
	//int dy = dx;
	int x, y, i;
	int Px[3], Py[3];
	

	int f_circle_text = TRUE;

	if (Opt->f_nodes_empty) {
		f_circle_text = FALSE;
	}


#if DONT_DRAW_ROOT_NODE
	if (!f_has_parent) {
		x = placement_x;
		y = placement_y;
		for (i = 0; i < nb_children; i++) {
			children[i]->draw_edges(G, rad, f_circletext, f_i, TRUE, x, y, max_depth, f_edge_labels, 
				f_has_draw_vertex_callback, draw_vertex_callback, T);
			}
		return;
		}
#endif
	x = placement_x;
	y = placement_y;
	// calc_y_coordinate(y, depth, max_depth);



	cout << "{" << x << "," << y << "}, // depth " << depth << " ";
	print_path();
	cout << endl;
	
	Px[1] = x;
	Py[1] = y;
	if (f_has_parent 
#if DONT_DRAW_ROOT_NODE
	 && depth >= 2 
#endif
	 ) {
		Px[0] = parent_x;
		Py[0] = parent_y;
		G.polygon2(Px, Py, 0, 1);

#if 0
		if (Opt->f_edge_labels && char_data) {
			Px[2] = (x + parent_x) >> 1;
			Py[2] = (y + parent_y) >> 1;
			G.aligned_text(Px[2], Py[2], "" /*"tl"*/, char_data);
			}
#endif
		}
	

	for (i = 0; i < nb_children; i++) {
		children[i]->draw_edges(G, Opt, f_i, TRUE, x, y, max_depth,
			f_has_draw_vertex_callback, draw_vertex_callback, T);
		}
}

void tree_node::draw_vertices(mp_graphics &G,
		layered_graph_draw_options *Opt,
		int f_i,
	int f_has_parent, int parent_x, int parent_y, int max_depth,
	int f_has_draw_vertex_callback, 
	void (*draw_vertex_callback)(tree *T, mp_graphics *G, int *v, int layer, tree_node *N, int x, int y, int dx, int dy),
	tree *T)
{
	//int rad = 20;
	int dx = Opt->rad; // / sqrt(2);
	int dy = dx;
	int x, y, i;
	int Px[3], Py[3];
	char str[1000];
	int *v;
	

#if DONT_DRAW_ROOT_NODE
	if (!f_has_parent) {
		x = placement_x;
		y = placement_y;
		for (i = 0; i < nb_children; i++) {
			children[i]->draw_vertices(G, rad, f_circletext, f_i, TRUE, x, y, max_depth, f_edge_labels, 
				f_has_draw_vertex_callback, draw_vertex_callback, T);
			}
		return;
		}
#endif
	x = placement_x;
	y = placement_y;
	// calc_y_coordinate(y, depth, max_depth);

	v = NEW_int(depth + 1);
	get_values(v);


	if (Opt->rad > 0) {
		if (Opt->f_circle) {
			if (depth == 0) {
				G.nice_circle(x, y, (int) (Opt->rad * 1.2));
				}
			if (FALSE) { // clearly something specific, here hyperoval in PG(2,16)
				G.nice_circle(x, y, (int) (Opt->rad * 3));
				}
			else {
				G.nice_circle(x, y, Opt->rad);
				}
			}
		}
	

	if (f_has_draw_vertex_callback) {
		cout << "calling draw_vertex_callback" << endl;
		(*draw_vertex_callback)(T, &G, v, depth, this, x, y, dx, dy);
		}
	FREE_int(v);


	cout << "{" << x << "," << y << "}, // depth " << depth << " ";
	print_path();
	cout << endl;
	
	Px[1] = x;
	Py[1] = y;
	if (f_has_parent 
#if DONT_DRAW_ROOT_NODE
	 && depth >= 2 
#endif
	 ) {
		Px[0] = parent_x;
		Py[0] = parent_y;
		//G.polygon2(Px, Py, 0, 1);
		
		}
	

	if (T->f_count_leaves) {
		if (nb_children == 0) {
		
			int dy, x0, y0;

			x0 = placement_x;
			y0 = placement_y;

			dy = parent->placement_y - y0;
			y0 -= dy;
		

			snprintf(str, 1000, "L%d", T->leaf_count);

			T->leaf_count++;
			G.aligned_text(x0, y0, "", str);
		
			}
		}

	for (i = 0; i < nb_children; i++) {
		children[i]->draw_vertices(G, Opt, f_i, TRUE, x, y, max_depth,
			f_has_draw_vertex_callback, draw_vertex_callback, T);
		}
	if (f_value) {
		snprintf(str, 1000, "%d", value);
		}
	else {
		snprintf(str, 1000, " ");
		}


	if (!Opt->f_nodes_empty) {
		//G.circle_text(x, y, str);
		G.aligned_text(x, y, "", str);
		}
	else {
		//G.aligned_text(x, y, 1, "tl", str);
		}
	if (f_i && !Opt->f_nodes_empty && f_int_data) {
		snprintf(str, 1000, "%d", int_data);
		G.aligned_text(Px[1], Py[1], "tl", str);
		}
}

void tree_node::draw_sideways(mp_graphics &G, int f_circletext, int f_i, 
	int f_has_parent, int parent_x, int parent_y, int max_depth, int f_edge_labels)
{
	int x, y, i;
	int xx, yy;
	int Px[3], Py[3];
	char str[1000];
	//int rad = 50;
	
#if DONT_DRAW_ROOT_NODE
	if (!f_has_parent) {
		x = placement_x;
		y = placement_y;
		xx = 10000 - y;
		yy = 10000 - x;
		for (i = 0; i < nb_children; i++) {
			children[i]->draw(G, f_circletext, f_i, TRUE, xx, yy, max_depth, f_edge_labels);
			}
		return;
		}
#endif
	x = placement_x;
	y = placement_y;
	xx = 10000 - y;
	yy = 10000 - x;
	// calc_y_coordinate(y, depth, max_depth);
	
	//G.circle(xx, yy, 20);

	cout << "{" << xx << "," << yy << "}, // depth " << depth << " ";
	print_path();
	cout << endl;
	
	Px[1] = xx;
	Py[1] = yy;
	if (f_has_parent 
#if DONT_DRAW_ROOT_NODE
	 && depth >= 2 
#endif
	 ) {
		Px[0] = parent_x;
		Py[0] = parent_y;
		G.polygon2(Px, Py, 0, 1);
		
		if (f_edge_labels && char_data) {
			Px[2] = (xx + parent_x) >> 1;
			Py[2] = (yy + parent_y) >> 1;
			G.aligned_text(Px[2], Py[2], "" /*"tl"*/, char_data);
			}
		}
	
	for (i = 0; i < nb_children; i++) {
		children[i]->draw_sideways(G, f_circletext, f_i, TRUE, xx, yy, max_depth, f_edge_labels);
		}
	if (f_value) {
		snprintf(str, 1000, "%d", value);
		}
	else {
		snprintf(str, 1000, " ");
		}
	if (f_circletext) {
#if 0
		//G.circle_text(xx, yy, str);
		G.sf_interior(100);
		G.sf_color(0); // 1 = black, 0 = white
		G.circle(xx, yy, rad);
		G.sf_interior(0);
		G.sf_color(1); // 1 = black, 0 = white
		G.circle(xx, yy, rad);
#endif
		G.aligned_text(Px[1], Py[1], "" /*"tl"*/, str);
		}
	else {
		//G.aligned_text(xx, yy, 1, "tl", str);
		}
	if (f_i && f_circletext && f_int_data) {
		snprintf(str, 1000, "%d", int_data);
		G.aligned_text(Px[1], Py[1], "tl", str);
		}
}


int tree_node_calc_y_coordinate(int ymax, int l, int max_depth)
{
	int dy, y;
	
	dy = (int)((double)ymax / (double)(max_depth + 1));
	y = (int)(dy * ((double)l + 0.5));
	y = ymax - y;
	return y;
}

}
}

