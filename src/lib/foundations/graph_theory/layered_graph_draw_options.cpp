// layered_graph_draw_options.cpp
// 
// Anton Betten
// December 15, 2015
//
//
// 
//
//

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {



layered_graph_draw_options::layered_graph_draw_options()
{
	f_file = FALSE;
	fname = NULL;

	xin = 10000;
	yin = 10000;
	xout = ONE_MILLION;
	yout = ONE_MILLION;

	f_spanning_tree = FALSE;

#if 0
	x_max = 10000;
	y_max = 10000;
	xmax = 1000000;
	ymax = 1000000;
#endif

	f_circle = TRUE;
	f_corners = FALSE;
	rad = 50;
	f_embedded = FALSE;
	f_sideways = FALSE;
	f_show_level_info = FALSE;
	f_label_edges = FALSE;
	f_y_stretch = FALSE;
	y_stretch = 1.;
	f_scale = FALSE;
	scale = .45;
	f_line_width = FALSE;
	line_width = 1.5;
	f_rotated = FALSE;


	f_nodes_empty = FALSE;
	f_select_layers = FALSE;
	select_layers = NULL;
	nb_layer_select = 0;
	layer_select = NULL;

	f_has_draw_begining_callback = FALSE;
	draw_begining_callback = NULL;
	f_has_draw_ending_callback = FALSE;
	draw_ending_callback = NULL;
	f_has_draw_vertex_callback = FALSE;
	draw_vertex_callback = NULL;

}

layered_graph_draw_options::~layered_graph_draw_options()
{
};

#if 0
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
#endif


int layered_graph_draw_options::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;
	//int f_v = (verbose_level >= 1);

	cout << "layered_graph_draw_options::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname = argv[++i];
			cout << "-file " << fname << endl;
			}
		if (strcmp(argv[i], "-xin") == 0) {
			xin = atoi(argv[++i]);
			cout << "-xin " << xin << endl;
			}
		else if (strcmp(argv[i], "-yin") == 0) {
			yin = atoi(argv[++i]);
			cout << "-yin " << yin << endl;
			}
		else if (strcmp(argv[i], "-xout") == 0) {
			xout = atoi(argv[++i]);
			cout << "-xout " << xout << endl;
			}
		else if (strcmp(argv[i], "-yout") == 0) {
			yout = atoi(argv[++i]);
			cout << "-yout " << yout << endl;
			}
		else if (strcmp(argv[i], "-spanning_tree") == 0) {
			f_spanning_tree = TRUE;
			cout << "-spanning_tree " << endl;
			}
		else if (strcmp(argv[i], "-circle") == 0) {
			f_circle = TRUE;
			cout << "-circle " << endl;
			}
		else if (strcmp(argv[i], "-corners") == 0) {
			f_corners = TRUE;
			cout << "-corners " << endl;
			}
		else if (strcmp(argv[i], "-rad") == 0) {
			rad = atoi(argv[++i]);
			cout << "-rad " << rad << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = atoi(argv[++i]);
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = atoi(argv[++i]);
			cout << "-sideways " << endl;
			}
		else if (strcmp(argv[i], "-show_level_info") == 0) {
			f_show_level_info = atoi(argv[++i]);
			cout << "-show_level_info " << endl;
			}
		else if (strcmp(argv[i], "-label_edges") == 0) {
			f_label_edges = atoi(argv[++i]);
			cout << "-label_edges " << endl;
			}
		else if (strcmp(argv[i], "-y_stretch") == 0) {
			f_y_stretch = atoi(argv[++i]);
			y_stretch = atof(argv[++i]);
			cout << "-y_stretch " << endl;
			}
		else if (strcmp(argv[i], "-scale") == 0) {
			f_scale = TRUE;
			sscanf(argv[++i], "%lf", &scale);
			cout << "-scale " << scale << endl;
			}
		else if (strcmp(argv[i], "-line_width") == 0) {
			f_line_width = TRUE;
			sscanf(argv[++i], "%lf", &line_width);
			cout << "-line_width " << line_width << endl;
			}
		else if (strcmp(argv[i], "-rotated") == 0) {
			f_rotated = atoi(argv[++i]);
			cout << "-rotated " << endl;
			}
		else if (strcmp(argv[i], "-nodes_empty") == 0) {
			f_nodes_empty = atoi(argv[++i]);
			cout << "-nodes_empty " << endl;
			}
		else if (strcmp(argv[i], "-select_layers") == 0) {
			f_select_layers = atoi(argv[++i]);
			select_layers = argv[++i];
			int_vec_scan(select_layers, layer_select, nb_layer_select);
			cout << "-select_layers ";
			int_vec_print(cout, layer_select, nb_layer_select);
			cout << endl;
			}

		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "layered_graph_draw_options::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "layered_graph_draw_options::read_arguments done" << endl;
	return i + 1;
}




}
}

