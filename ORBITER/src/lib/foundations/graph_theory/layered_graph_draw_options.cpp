// layered_graph_draw_options.C
// 
// Anton Betten
// December 15, 2015
//
//
// 
//
//

#include "foundations.h"

namespace orbiter {
namespace foundations {



layered_graph_draw_options::layered_graph_draw_options()
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
	layer_select = NULL;

	f_has_draw_begining_callback = FALSE;
	draw_begining_callback = NULL;
	f_has_draw_ending_callback = FALSE;
	draw_ending_callback = NULL;
	f_has_draw_vertex_callback = FALSE;
	draw_vertex_callback = NULL;

	f_show_level_info = FALSE;
	f_embedded = FALSE;
	f_sideways = FALSE;
	f_show_level_info = FALSE;
	f_label_edges = FALSE;
	f_rotated = FALSE;

	global_scale = .45;
	global_line_width = 1.5;
}

layered_graph_draw_options::~layered_graph_draw_options()
{
};

void layered_graph_draw_options::init(
	int xmax, int ymax, int x_max, int y_max, int rad, 
	int f_circle, int f_corners, int f_nodes_empty, 
	int f_select_layers, int nb_layer_select, int *layer_select, 
	int f_has_draw_begining_callback, 
	void (*draw_begining_callback)(layered_graph *LG, mp_graphics *G,
			int x_max, int y_max, int f_rotated, int dx, int dy),
	int f_has_draw_ending_callback, 
	void (*draw_ending_callback)(layered_graph *LG, mp_graphics *G,
			int x_max, int y_max, int f_rotated, int dx, int dy),
	int f_has_draw_vertex_callback, 
	void (*draw_vertex_callback)(layered_graph *LG, mp_graphics *G,
			int layer, int node, int x, int y, int dx, int dy),
	int f_show_level_info, 
	int f_embedded, int f_sideways, 
	int f_label_edges, 
	int f_rotated, 
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

}
}

