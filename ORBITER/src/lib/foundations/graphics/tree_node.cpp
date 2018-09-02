// tree_node.C
//
// Anton Betten
//
// moved here from tree.C: October 12, 2013
//
// February 7, 2003

#include "foundations.h"

#define DONT_DRAW_ROOT_NODE 0


tree_node::tree_node()
{
	depth = 0;
	nb_children = 0;
	children = NULL;
	f_int_data = FALSE;
	char_data = NULL;
}

tree_node::~tree_node()
{
}

void tree_node::init(INT depth, tree_node *parent, INT f_value, INT value, 
	INT f_i_data, INT i_data, char *c_data, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	
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
	INT i;
	
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

void tree_node::compute_DFS_rank(INT &rk)
{
	INT i;
	
	DFS_rank = rk;
	rk++;
	for (i = 0; i < nb_children; i++) {
		children[i]->compute_DFS_rank(rk);
		}
}

void tree_node::get_coordinates(INT &idx, INT *coord_xy)
{
	INT i;
	
	coord_xy[idx * 2 + 0] = placement_x;
	coord_xy[idx * 2 + 1] = placement_y;
	idx++;
	for (i = 0; i < nb_children; i++) {
		children[i]->get_coordinates(idx, coord_xy);
		}
}

void tree_node::get_coordinates_and_width(INT &idx, INT *coord_xyw)
{
	INT i;
	
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
	INT i;

	weight = 1;
	for (i = 0; i < nb_children; i++) {
		children[i]->calc_weight();
		weight += children[i]->weight;
		}
}

void tree_node::place_xy(INT left, INT right, INT ymax, INT max_depth)
{
	INT i, w, w0, w1, lft, rgt;
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
		lft = left + (INT)((double)w0 * dx);
		rgt = left + (INT)((double)(w0 + w1) * dx);
		children[i]->place_xy(lft, rgt, ymax, max_depth);
		w0 += w1;
		}
}

void tree_node::place_on_circle(INT xmax, INT ymax, INT max_depth)
{
	INT i, dy;
	double x, y;
	double x1, y1;

	x = placement_x;
	y = placement_y;
	dy = (INT)((double)ymax / (double)(max_depth + 1));
	y = ymax - y;
	y -= dy * 0.5;
	y /= ymax;
	x /= (double) xmax;
	x *= 2. * M_PI;
	x -= M_PI;
	x1 = y * cos(x) * xmax * 0.5 + xmax * 0.5;
	y1 = y * sin(x) * ymax * 0.5 + ymax * 0.5;
	placement_x = (INT) x1;
	placement_y = (INT) y1;
	for (i = 0; i < nb_children; i++) {
		children[i]->place_on_circle(xmax, ymax, max_depth);
		}
}

void tree_node::add_node(INT l, INT depth, INT *path, INT i_data, char *c_data, 
	INT verbose_level)
{
	INT i, idx;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "tree_node::add_node: depth=" << depth << " : ";
		INT_vec_print(cout, path, l);
		cout << endl;
		}
	if (l == 0) {
		if (f_vv) {
			cout << "add_node(): node of length 0" << endl;
			}
		init(0, NULL, FALSE, 0, TRUE, i_data, c_data, verbose_level);
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

INT tree_node::find_child(INT val)
{
	INT i;
	
	for (i = 0; i < nb_children; i++) {
		if (children[i]->value == val)
			return i;
		}
	return -1;
}

void tree_node::get_values(INT *v)
{
	cout << "get_values depth=" << depth << " value=" << value << endl;
	if (depth) {
		v[depth - 1] = value;
		parent->get_values(v);
		}
}

void tree_node::draw_edges(mp_graphics &G, INT rad, INT f_circle, INT f_circletext, INT f_i, 
	INT f_has_parent, INT parent_x, INT parent_y, INT max_depth, INT f_edge_labels, 
	INT f_has_draw_vertex_callback, 
	void (*draw_vertex_callback)(tree *T, mp_graphics *G, INT *v, INT layer, tree_node *N, INT x, INT y, INT dx, INT dy),
	tree *T
	)
{
	//INT rad = 20;
	//INT dx = rad; // / sqrt(2);
	//INT dy = dx;
	INT x, y, i;
	INT Px[3], Py[3];
	
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
		
		if (f_edge_labels && char_data) {
			Px[2] = (x + parent_x) >> 1;
			Py[2] = (y + parent_y) >> 1;
			G.aligned_text(Px[2], Py[2], "" /*"tl"*/, char_data);
			}
		}
	
	for (i = 0; i < nb_children; i++) {
		children[i]->draw_edges(G, rad, f_circle, f_circletext, f_i, TRUE, x, y, max_depth, f_edge_labels, 
			f_has_draw_vertex_callback, draw_vertex_callback, T);
		}
}

void tree_node::draw_vertices(mp_graphics &G, INT rad, INT f_circle, INT f_circletext, INT f_i, 
	INT f_has_parent, INT parent_x, INT parent_y, INT max_depth, INT f_edge_labels, 
	INT f_has_draw_vertex_callback, 
	void (*draw_vertex_callback)(tree *T, mp_graphics *G, INT *v, INT layer, tree_node *N, INT x, INT y, INT dx, INT dy),
	tree *T
	)
{
	//INT rad = 20;
	INT dx = rad; // / sqrt(2);
	INT dy = dx;
	INT x, y, i;
	INT Px[3], Py[3];
	char str[1000];
	INT *v;
	
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

	v = NEW_INT(depth + 1);
	get_values(v);


	if (rad > 0) {
		if (f_circle) {
			if (depth == 0) {
				G.nice_circle(x, y, (INT) (rad * 1.2));
				}
			if (depth == 18) { // clearly something specific, here hyperval in PG(2,16)
				G.nice_circle(x, y, (INT) (rad * 3));
				}
			else {
				G.nice_circle(x, y, rad);
				}
			}
		}
	

	if (f_has_draw_vertex_callback) {
		cout << "calling draw_vertex_callback" << endl;
		(*draw_vertex_callback)(T, &G, v, depth, this, x, y, dx, dy);
		}
	FREE_INT(v);


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
		
			INT dy, x0, y0;

			x0 = placement_x;
			y0 = placement_y;

			dy = parent->placement_y - y0;
			y0 -= dy;
		

			sprintf(str, "L%ld", T->leaf_count);

			T->leaf_count++;
			G.aligned_text(x0, y0, "", str);
		
			}
		}

	for (i = 0; i < nb_children; i++) {
		children[i]->draw_vertices(G, rad, f_circle, f_circletext, f_i, TRUE, x, y, max_depth, f_edge_labels, 
			f_has_draw_vertex_callback, draw_vertex_callback, T);
		}
	if (f_value) {
		sprintf(str, "%ld", value);
		}
	else {
		sprintf(str, " ");
		}
	if (f_circletext) {
		//G.circle_text(x, y, str);
		G.aligned_text(x, y, "", str);
		}
	else {
		//G.aligned_text(x, y, 1, "tl", str);
		}
	if (f_i && f_circletext && f_int_data) {
		sprintf(str, "%ld", int_data);
		G.aligned_text(Px[1], Py[1], "tl", str);
		}
}

void tree_node::draw_sideways(mp_graphics &G, INT f_circletext, INT f_i, 
	INT f_has_parent, INT parent_x, INT parent_y, INT max_depth, INT f_edge_labels)
{
	INT x, y, i;
	INT xx, yy;
	INT Px[3], Py[3];
	char str[1000];
	//INT rad = 50;
	
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
		sprintf(str, "%ld", value);
		}
	else {
		sprintf(str, " ");
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
		sprintf(str, "%ld", int_data);
		G.aligned_text(Px[1], Py[1], "tl", str);
		}
}


INT tree_node_calc_y_coordinate(INT ymax, INT l, INT max_depth)
{
	INT dy, y;
	
	dy = (INT)((double)ymax / (double)(max_depth + 1));
	y = (INT)(dy * ((double)l + 0.5));
	y = ymax - y;
	return y;
}


