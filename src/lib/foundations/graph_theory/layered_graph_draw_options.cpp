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
		else if (strcmp(argv[i], "-xin") == 0) {
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
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
			}
		else if (strcmp(argv[i], "-show_level_info") == 0) {
			f_show_level_info = TRUE;
			cout << "-show_level_info " << endl;
			}
		else if (strcmp(argv[i], "-label_edges") == 0) {
			f_label_edges = TRUE;
			cout << "-label_edges " << endl;
			}
		else if (strcmp(argv[i], "-y_stretch") == 0) {
			f_y_stretch = TRUE;
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

