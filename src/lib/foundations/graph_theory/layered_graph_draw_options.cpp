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
	//fname = NULL;

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
	f_x_stretch = FALSE;
	x_stretch = 1.;
	f_y_stretch = FALSE;
	y_stretch = 1.;
	f_scale = FALSE;
	scale = .45;
	f_line_width = FALSE;
	line_width = 1.5;
	f_rotated = FALSE;


	f_nodes_empty = FALSE;
	f_select_layers = FALSE;
	//select_layers = NULL;
	nb_layer_select = 0;
	layer_select = NULL;

	f_has_draw_begining_callback = FALSE;
	draw_begining_callback = NULL;
	f_has_draw_ending_callback = FALSE;
	draw_ending_callback = NULL;
	f_has_draw_vertex_callback = FALSE;
	draw_vertex_callback = NULL;

	f_paths_in_between = FALSE;
	layer1 = 0;
	node1 = 0;;
	layer2 = 0;
	node2 = 0;
}

layered_graph_draw_options::~layered_graph_draw_options()
{
};

int layered_graph_draw_options::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	//int f_v = (verbose_level >= 1);

	cout << "layered_graph_draw_options::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname.assign(argv[++i]);
			cout << "-file " << fname << endl;
		}
		else if (stringcmp(argv[i], "-xin") == 0) {
			xin = strtoi(argv[++i]);
			cout << "-xin " << xin << endl;
		}
		else if (stringcmp(argv[i], "-yin") == 0) {
			yin = strtoi(argv[++i]);
			cout << "-yin " << yin << endl;
		}
		else if (stringcmp(argv[i], "-xout") == 0) {
			xout = strtoi(argv[++i]);
			cout << "-xout " << xout << endl;
		}
		else if (stringcmp(argv[i], "-yout") == 0) {
			yout = strtoi(argv[++i]);
			cout << "-yout " << yout << endl;
		}
		else if (stringcmp(argv[i], "-spanning_tree") == 0) {
			f_spanning_tree = TRUE;
			cout << "-spanning_tree " << endl;
		}
		else if (stringcmp(argv[i], "-circle") == 0) {
			f_circle = TRUE;
			cout << "-circle " << endl;
		}
		else if (stringcmp(argv[i], "-corners") == 0) {
			f_corners = TRUE;
			cout << "-corners " << endl;
		}
		else if (stringcmp(argv[i], "-rad") == 0) {
			rad = strtoi(argv[++i]);
			cout << "-rad " << rad << endl;
		}
		else if (stringcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
		}
		else if (stringcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
		}
		else if (stringcmp(argv[i], "-show_level_info") == 0) {
			f_show_level_info = TRUE;
			cout << "-show_level_info " << endl;
		}
		else if (stringcmp(argv[i], "-label_edges") == 0) {
			f_label_edges = TRUE;
			cout << "-label_edges " << endl;
		}
		else if (stringcmp(argv[i], "-x_stretch") == 0) {
			f_x_stretch = TRUE;
			x_stretch = strtof(argv[++i]);
			cout << "-x_stretch " << x_stretch << endl;
		}
		else if (stringcmp(argv[i], "-y_stretch") == 0) {
			f_y_stretch = TRUE;
			y_stretch = strtof(argv[++i]);
			cout << "-y_stretch " << y_stretch << endl;
		}
		else if (stringcmp(argv[i], "-scale") == 0) {
			f_scale = TRUE;
			scale = strtof(argv[++i]);
			cout << "-scale " << scale << endl;
		}
		else if (stringcmp(argv[i], "-line_width") == 0) {
			f_line_width = TRUE;
			line_width = strtof(argv[++i]);
			cout << "-line_width " << line_width << endl;
		}
		else if (stringcmp(argv[i], "-rotated") == 0) {
			f_rotated = TRUE;
			cout << "-rotated " << endl;
		}
		else if (stringcmp(argv[i], "-nodes_empty") == 0) {
			f_nodes_empty = TRUE;
			cout << "-nodes_empty " << endl;
		}
		else if (stringcmp(argv[i], "-select_layers") == 0) {
			f_select_layers = TRUE;
			select_layers.assign(argv[++i]);
			Orbiter->Int_vec.scan(select_layers, layer_select, nb_layer_select);
			cout << "-select_layers ";
			Orbiter->Int_vec.print(cout, layer_select, nb_layer_select);
			cout << endl;
		}
		else if (stringcmp(argv[i], "-paths_in_between") == 0) {
			f_paths_in_between = TRUE;
			layer1 = strtoi(argv[++i]);
			node1 = strtoi(argv[++i]);
			layer2 = strtoi(argv[++i]);
			node2 = strtoi(argv[++i]);
			cout << "-paths_in_between " << layer1 << " " << node1
					<< " " << layer2 << " " << node2 << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
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


void layered_graph_draw_options::print()
{
	cout << "layered_graph_draw_options::print:" << endl;

	if (f_file) {
		cout << "file name: " << fname << endl;
	}
	cout << "xin, xout, yin, yout=" << xin << ", " << xout << ", " << yin << ", " << yout << endl;
	cout << "f_spanning_tree=" << f_spanning_tree << endl;
	cout << "f_circle=" << f_circle << endl;
	cout << "f_corners=" << f_corners << endl;
	cout << "rad=" << rad << endl;
	cout << "f_embedded=" << f_embedded << endl;
	cout << "f_sideways=" << f_sideways << endl;
	cout << "f_show_level_info=" << f_show_level_info << endl;
	cout << "f_label_edges=" << f_label_edges << endl;
	if (f_x_stretch) {
		cout << "x_stretch=" << x_stretch << endl;
	}
	if (f_y_stretch) {
		cout << "y_stretch=" << y_stretch << endl;
	}
	if (f_scale) {
		cout << "scale=" << scale << endl;
	}
	if (f_line_width) {
		cout << "line_width=" << line_width << endl;
	}
	cout << "f_rotated=" << f_rotated << endl;
	cout << "f_nodes_empty=" << f_nodes_empty << endl;
	if (f_select_layers) {
		cout << "select_layers=" << select_layers << endl;
	}
	if (nb_layer_select) {
		cout << "layer_select=";
		Orbiter->Int_vec.print(cout, layer_select, nb_layer_select);
		cout << endl;
	}
	if (f_paths_in_between) {
		cout << "f_paths_in_between" << endl;
		cout << "layer1=" << layer1 << endl;
		cout << "node1=" << node1 << endl;
		cout << "layer2=" << layer2 << endl;
		cout << "node2=" << node2 << endl;
	}
}






}
}

