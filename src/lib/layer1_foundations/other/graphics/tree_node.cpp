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
namespace layer1_foundations {
namespace other {
namespace graphics {



tree_node::tree_node()
{
	Record_birth();
	parent = NULL;
	depth = 0;
	f_value = false;
	value = 0;

	f_has_color = false;
	color = 0;

	//label;

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
	Record_death();
}

void tree_node::init(
		int depth, tree_node *parent, int f_value, int value,
	int f_has_color, int color, std::string &label,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "tree_node::init "
				"depth=" << depth << " value=" << value << endl;
		}
	tree_node::depth = depth;
	tree_node::parent = parent;
	
	tree_node::f_value = f_value;
	tree_node::value = value;
	
	tree_node::f_has_color = f_has_color;
	tree_node::color = color;
	
	tree_node::label.assign(label);

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
	if (f_has_color) {
		cout << color;
		}
	cout << " : ";
	cout << label;
	cout << endl;
	for (i = 0; i < nb_children; i++) {
		children[i]->print_depth_first();
		}
}

void tree_node::compute_DFS_rank(
		int &rk)
{
	int i;
	
	DFS_rank = rk;
	rk++;
	for (i = 0; i < nb_children; i++) {
		children[i]->compute_DFS_rank(rk);
	}
}

int tree_node::find_node(
		int &DFS_rk, int *path, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tree_node::find_node ";
		cout << "my_path = ";
		Int_vec_print(cout, path, sz);
		cout << " value=" << value << endl;
	}
	int i;

	if (value == path[0]) {
		if (sz == 1) {
			DFS_rk = DFS_rank;
			return true;
		}
		else {
			for (i = 0; i < nb_children; i++) {
				if (children[i]->find_node(
						DFS_rk, path + 1, sz - 1, verbose_level)) {
					return true;
				}
			}
			cout << "tree_node::find_node did not find node" << endl;
			exit(1);
		}
	}
	else {
		return false;
	}
}

int tree_node::find_node_and_path(
		std::vector<int> &Rk,
		int *path, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tree_node::find_node_and_path ";
		cout << "my_path = ";
		Int_vec_print(cout, path, sz);
		cout << " value=" << value << endl;
	}
	int i;

	if (value == path[0]) {
		Rk.push_back(DFS_rank);
		if (sz == 1) {
			return true;
		}
		else {
			for (i = 0; i < nb_children; i++) {
				if (children[i]->find_node_and_path(
						Rk, path + 1, sz - 1, verbose_level)) {
					return true;
				}
			}
			cout << "tree_node::find_node_and_path did not find node" << endl;
			exit(1);
		}
	}
	else {
		return false;
	}
}


void tree_node::get_coordinates(
		int &idx, int *coord_xy)
{
	int i;
	
	coord_xy[idx * 2 + 0] = placement_x;
	coord_xy[idx * 2 + 1] = placement_y;
	idx++;
	for (i = 0; i < nb_children; i++) {
		children[i]->get_coordinates(idx, coord_xy);
		}
}

void tree_node::get_coordinates_and_width(
		int &idx, int *coord_xyw)
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

void tree_node::place_xy(
		int left, int right, int ymax, int max_depth)
{
	int i, w, w0, w1, lft, rgt;
	double dx;
	
	placement_x = (left + right) >> 1;
	placement_y = calc_y_coordinate(ymax, depth, max_depth);
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

void tree_node::place_on_circle(
		int xmax, int ymax, int max_depth)
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

void tree_node::add_node(
		int l,
		int depth, int *path, int color, std::string &label,
		int verbose_level)
{
	int i, idx;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "tree_node::add_node depth=" << depth << " : ";
		Int_vec_print(cout, path, l);
		cout << endl;
	}
	if (l == 0) {
		if (f_vv) {
			cout << "tree_node::add_node node of length 0" << endl;
		}
		init(0, NULL, true, -1, true, color, label, verbose_level);
		return;
	}
	idx = find_child(path[depth]);
	if (f_vv) {
		cout << "tree_node::add_node find_child for " << path[depth] << " returns " << idx << endl;
	}
	if (idx == -1) {
		tree_node **new_children = new ptree_node[nb_children + 1];
		for (i = 0; i < nb_children; i++) {
			new_children[i] = children[i];
		}
		new_children[nb_children] = new tree_node;
		if (nb_children) {
			delete [] children;
		}
		children = new_children;
		nb_children++;
		if (f_vv) {
			cout << "tree_node::add_node nb_children increased to " << nb_children << endl;
		}
		
		if (l == depth + 1) {
			if (f_vv) {
				cout << "tree_node::add_node initializing terminal node" << endl;
			}
			children[nb_children - 1]->init(depth + 1, this,
					true, path[depth], true, color, label,
					verbose_level);
			return;
		}
		else {
			if (f_vv) {
				cout << "initializing intermediate node" << endl;
			}
			children[nb_children - 1]->init(depth + 1, this,
					true, path[depth], false, 0, label,
					verbose_level);
			idx = nb_children - 1;
		}
	}
	if (f_vv) {
		cout << "searching deeper" << endl;
	}
	children[idx]->add_node(l, depth + 1, path, color, label, verbose_level);
}

int tree_node::find_child(
		int val)
{
	int i;
	
	for (i = 0; i < nb_children; i++) {
		if (children[i]->value == val) {
			return i;
		}
	}
	return -1;
}

void tree_node::get_values(
		int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tree_node::get_values" << endl;
		cout << "get_values depth=" << depth << " value=" << value << endl;
	}
	if (depth) {
		v[depth - 1] = value;
		parent->get_values(v, verbose_level);
	}
	if (f_v) {
		cout << "tree_node::get_values done" << endl;
	}
}

void tree_node::draw_edges(
		mp_graphics &G,
		tree_draw_options *Tree_draw_options,
		layered_graph_draw_options *Opt,
	int f_has_parent, int parent_x, int parent_y, int max_depth,
	tree *T, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tree_node::draw_edges" << endl;
	}
	//int rad = 20;
	//int dx = rad; // / sqrt(2);
	//int dy = dx;
	int x, y, i;
	int Px[3], Py[3];
	

	//int f_circle_text = true;

#if 0
	if (Opt->f_nodes_empty) {
		f_circle_text = false;
	}
#endif


#if DONT_DRAW_ROOT_NODE
	if (!f_has_parent) {
		x = placement_x;
		y = placement_y;
		for (i = 0; i < nb_children; i++) {
			children[i]->draw_edges(
					G, rad, f_circletext, true, x, y, max_depth, f_edge_labels,
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

	int f_show = false;

	if (Tree_draw_options->f_select_path) {
		int rk;

		rk = DFS_rank;
		if (f_v) {
			cout << "tree_node::draw_edges DFS_rank = " << rk << endl;
		}
		if (T->f_node_select[rk]) {
			f_show = true;
		}
	}
	else {
		f_show = true;
	}

	if (f_show) {

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
	}
	

	for (i = 0; i < nb_children; i++) {
		children[i]->draw_edges(
				G, Tree_draw_options, Opt, true, x, y, max_depth, T, verbose_level);
	}

	if (f_v) {
		cout << "tree_node::draw_edges done" << endl;
	}
}

void tree_node::draw_vertices(
		mp_graphics &G,
		tree_draw_options *Tree_draw_options,
		layered_graph_draw_options *Opt,
		int f_has_parent, int parent_x, int parent_y, int max_depth,
		tree *T, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tree_node::draw_vertices" << endl;
	}
	int dx = Opt->rad;
	int dy = dx;
	int x, y, i;
	int Px[3], Py[3];
	int *v;
	
#if DONT_DRAW_ROOT_NODE
	if (!f_has_parent) {
		x = placement_x;
		y = placement_y;
		for (i = 0; i < nb_children; i++) {
			children[i]->draw_vertices(
					G, rad, f_circletext, f_i, true, x, y, max_depth, f_edge_labels,
				f_has_draw_vertex_callback, draw_vertex_callback, T);
			}
		return;
		}
#endif
	x = placement_x;
	y = placement_y;
	// calc_y_coordinate(y, depth, max_depth);

	v = NEW_int(depth + 1);
	get_values(v, verbose_level);

#if 0
	if (Opt->rad > 0) {
		if (Opt->f_circle) {
			if (depth == 0) {
				G.nice_circle(x, y, (int) (Opt->rad * 1.2));
			}
			G.nice_circle(x, y, Opt->rad);
		}
	}
#endif
	
	int f_show = false;

	if (Tree_draw_options->f_select_path) {
		int rk;

		rk = DFS_rank;
		if (f_v) {
			cout << "tree_node::draw_vertices DFS_rank = " << rk << endl;
		}
		if (T->f_node_select[rk]) {
			f_show = true;
		}
	}
	else {
		f_show = true;
	}

	if (!f_has_parent) {
		f_show = false;
	}


	if (f_show) {
		if (Opt->f_nodes) {
			G.sf_color(color);
			//G.sf_interior(color /* fill_interior*/);
			G.nice_circle(x, y, Opt->rad);
		}

		if (f_has_color) {


			if (Opt->f_nodes_empty) {
				G.sf_color(color);
				//G.sf_interior(color /* fill_interior*/);
				G.nice_circle(x, y, Opt->rad);
			}
			else {
				string s;

				s = std::to_string(value);
				G.aligned_text(x, y, "", s);
			}
		}
		else {
			string s;

			s = std::to_string(value);
			G.aligned_text(x, y, "", s);
		}



		if (Tree_draw_options->f_has_draw_vertex_callback) {
			cout << "calling draw_vertex_callback" << endl;
			(*Tree_draw_options->draw_vertex_callback)(
					T, &G, v, depth, this, x, y, dx, dy);
		}

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
		

			string s;

			s = "L" + std::to_string(T->leaf_count);


			T->leaf_count++;
			G.aligned_text(x0, y0, "", s);
		
			}
		}

	for (i = 0; i < nb_children; i++) {
		children[i]->draw_vertices(
				G, Tree_draw_options, Opt, true, x, y, max_depth, T, verbose_level);
		}


	if (f_v) {
		cout << "tree_node::draw_vertices done" << endl;
	}

}

void tree_node::draw_sideways(
		mp_graphics &G, int f_circletext, int f_i,
	int f_has_parent, int parent_x, int parent_y, int max_depth, int f_edge_labels)
{
	int x, y, i;
	int xx, yy;
	int Px[3], Py[3];
	//int rad = 50;
	
#if DONT_DRAW_ROOT_NODE
	if (!f_has_parent) {
		x = placement_x;
		y = placement_y;
		xx = 10000 - y;
		yy = 10000 - x;
		for (i = 0; i < nb_children; i++) {
			children[i]->draw(G, f_circletext, f_i, true, xx, yy, max_depth, f_edge_labels);
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
		
		if (f_edge_labels && label.length()) {
			Px[2] = (xx + parent_x) >> 1;
			Py[2] = (yy + parent_y) >> 1;
			G.aligned_text(Px[2], Py[2], "" /*"tl"*/, label);
			}
		}
	
	for (i = 0; i < nb_children; i++) {
		children[i]->draw_sideways(
				G, f_circletext, f_i, true, xx, yy, max_depth, f_edge_labels);
		}

	string s;

	if (f_value) {
		s = std::to_string(value);
		}
	else {
		s = " ";
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
		G.aligned_text(Px[1], Py[1], "" /*"tl"*/, s);
		}
	else {
		//G.aligned_text(xx, yy, 1, "tl", str);
		}
	if (f_i && f_circletext && f_has_color) {
		s = std::to_string(color);
		G.aligned_text(Px[1], Py[1], "tl", s);
		}
}


int tree_node::calc_y_coordinate(
		int ymax, int l, int max_depth)
{
	int dy, y;
	
	dy = (int)((double)ymax / (double)(max_depth + 1));
	y = (int)(dy * ((double)l + 0.5));
	y = ymax - y;
	return y;
}


}}}}


