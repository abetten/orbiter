// layered_graph_draw_options.C
// 
// Anton Betten
// December 15, 2015
//
//
// 
//
//

#include "galois.h"


layered_graph_draw_options::layered_graph_draw_options()
{
}

layered_graph_draw_options::~layered_graph_draw_options()
{
	x_max = 10000;
	y_max = 10000;
	xmax = 1000000;
	ymax = 1000000;
	rad = 50;

	f_circle = TRUE;
	f_corners = FALSE;
	f_nodes_empty = FALSE;
	f_select_layers = FALSE;
	nb_layer_select = 0;

	f_has_draw_begining_callback = FALSE;
	f_has_draw_ending_callback = FALSE;
	f_has_draw_vertex_callback = FALSE;

	f_show_level_info = FALSE;
	f_embedded = FALSE;
	f_sideways = FALSE;
	f_show_level_info = FALSE;
	f_label_edges = FALSE;
	f_rotated = FALSE;

	global_scale = .45;
	global_line_width = 1.5;
};

void layered_graph_draw_options::init(
	INT xmax, INT ymax, INT x_max, INT y_max, INT rad, 
	INT f_circle, INT f_corners, INT f_nodes_empty, 
	INT f_select_layers, INT nb_layer_select, INT *layer_select, 
	INT f_has_draw_begining_callback, 
	void (*draw_begining_callback)(layered_graph *LG, mp_graphics *G, INT x_max, INT y_max, INT f_rotated, INT dx, INT dy), 
	INT f_has_draw_ending_callback, 
	void (*draw_ending_callback)(layered_graph *LG, mp_graphics *G, INT x_max, INT y_max, INT f_rotated, INT dx, INT dy), 
	INT f_has_draw_vertex_callback, 
	void (*draw_vertex_callback)(layered_graph *LG, mp_graphics *G, INT layer, INT node, INT x, INT y, INT dx, INT dy), 
	INT f_show_level_info, 
	INT f_embedded, INT f_sideways, 
	INT f_label_edges, 
	INT f_rotated, 
	double global_scale, double global_line_width)
{
	layered_graph_draw_options *O;
	
	O = this;

	O->xmax = xmax;
	O->ymax = ymax;
	O->x_max = x_max;
	O->y_max = y_max;
	O->rad = rad;
	O->f_circle = f_circle;
	O->f_corners = f_corners;
	O->f_nodes_empty = f_nodes_empty;
	O->f_select_layers = f_select_layers;
	O->nb_layer_select = nb_layer_select;
	O->layer_select = layer_select;
	O->f_has_draw_begining_callback = f_has_draw_begining_callback;
	O->draw_begining_callback = draw_begining_callback;
	O->f_has_draw_ending_callback = f_has_draw_ending_callback;
	O->draw_ending_callback = draw_ending_callback;
	O->f_has_draw_vertex_callback = f_has_draw_vertex_callback;
	O->draw_vertex_callback = draw_vertex_callback;
	O->f_show_level_info = f_show_level_info;
	O->f_embedded = f_embedded;
	O->f_sideways = f_sideways;
	O->f_label_edges = f_label_edges;
	O->f_rotated = f_rotated;
	O->global_scale = global_scale;
	O->global_line_width = global_line_width;
}

