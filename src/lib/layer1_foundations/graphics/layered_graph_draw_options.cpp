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
namespace layer1_foundations {
namespace graphics {



layered_graph_draw_options::layered_graph_draw_options()
{

	f_paperheight = false;
	paperheight = 0;

	f_paperwidth = false;
	paperwidth = 0;


	xin = 10000;
	yin = 10000;
	xout = ONE_MILLION;
	yout = ONE_MILLION;

	f_spanning_tree = false;

	f_circle = true;
	f_corners = false;
	rad = 200;
	f_embedded = false;
	f_sideways = false;
	f_show_level_info = false;
	f_label_edges = false;
	f_x_stretch = false;
	x_stretch = 1.;
	f_y_stretch = false;
	y_stretch = 1.;
	f_scale = false;
	scale = .45;
	f_line_width = false;
	line_width = 1.5;
	f_rotated = false;


	f_nodes = false;
	f_nodes_empty = false;
	f_show_colors = false;

	f_select_layers = false;
	//select_layers = NULL;
	nb_layer_select = 0;
	layer_select = NULL;

	f_has_draw_begining_callback = false;
	draw_begining_callback = NULL;
	f_has_draw_ending_callback = false;
	draw_ending_callback = NULL;
	f_has_draw_vertex_callback = false;
	draw_vertex_callback = NULL;

	f_paths_in_between = false;
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
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "layered_graph_draw_options::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-paperheight") == 0) {
			f_paperheight = true;
			paperheight = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-paperheight " << paperheight << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-paperwidth") == 0) {
			f_paperwidth = true;
			paperwidth = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-paperwidth " << paperwidth << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-xin") == 0) {
			xin = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-xin " << xin << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-yin") == 0) {
			yin = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-yin " << yin << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-xout") == 0) {
			xout = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-xout " << xout << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-yout") == 0) {
			yout = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-yout " << yout << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-spanning_tree") == 0) {
			f_spanning_tree = true;
			if (f_v) {
				cout << "-spanning_tree " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-circle") == 0) {
			f_circle = true;
			if (f_v) {
				cout << "-circle " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-corners") == 0) {
			f_corners = true;
			if (f_v) {
				cout << "-corners " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-radius") == 0) {
			rad = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-radius " << rad << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-embedded") == 0) {
			f_embedded = true;
			if (f_v) {
				cout << "-embedded " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-sideways") == 0) {
			f_sideways = true;
			if (f_v) {
				cout << "-sideways " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-show_level_info") == 0) {
			f_show_level_info = true;
			if (f_v) {
				cout << "-show_level_info " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_edges") == 0) {
			f_label_edges = true;
			if (f_v) {
				cout << "-label_edges " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-x_stretch") == 0) {
			f_x_stretch = true;
			x_stretch = ST.strtof(argv[++i]);
			if (f_v) {
				cout << "-x_stretch " << x_stretch << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-y_stretch") == 0) {
			f_y_stretch = true;
			y_stretch = ST.strtof(argv[++i]);
			if (f_v) {
				cout << "-y_stretch " << y_stretch << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-scale") == 0) {
			f_scale = true;
			scale = ST.strtof(argv[++i]);
			if (f_v) {
				cout << "-scale " << scale << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-line_width") == 0) {
			f_line_width = true;
			line_width = ST.strtof(argv[++i]);
			if (f_v) {
				cout << "-line_width " << line_width << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-rotated") == 0) {
			f_rotated = true;
			if (f_v) {
				cout << "-rotated " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-nodes") == 0) {
			f_nodes = true;
			if (f_v) {
				cout << "-nodes " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-nodes_empty") == 0) {
			f_nodes_empty = true;
			if (f_v) {
				cout << "-nodes_empty " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-show_colors") == 0) {
			f_show_colors = true;
			if (f_v) {
				cout << "-show_colors " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-select_layers") == 0) {
			f_select_layers = true;
			select_layers.assign(argv[++i]);
			Int_vec_scan(select_layers, layer_select, nb_layer_select);
			if (f_v) {
				cout << "-select_layers ";
				Int_vec_print(cout, layer_select, nb_layer_select);
				cout << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-paths_in_between") == 0) {
			f_paths_in_between = true;
			layer1 = ST.strtoi(argv[++i]);
			node1 = ST.strtoi(argv[++i]);
			layer2 = ST.strtoi(argv[++i]);
			node2 = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-paths_in_between " << layer1 << " " << node1
					<< " " << layer2 << " " << node2 << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "layered_graph_draw_options::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "layered_graph_draw_options::read_arguments done" << endl;
	}
	return i + 1;
}


void layered_graph_draw_options::print()
{
	//cout << "layered_graph_draw_options::print:" << endl;


	if (f_paperheight) {
		cout << "-paperheight " << paperheight << endl;
	}
	if (f_paperwidth) {
		cout << "-paperwidth " << paperwidth << endl;
	}

	cout << "xin, xout, yin, yout=" << xin << ", " << xout << ", " << yin << ", " << yout << endl;
	cout << "radius=" << rad << endl;

	if (f_spanning_tree) {
		cout << "f_spanning_tree=" << endl;
	}
	if (f_circle) {
		cout << "f_circle" << endl;
	}
	if (f_corners) {
		cout << "f_corners" << endl;
	}
	if (f_embedded) {
		cout << "f_embedded" << endl;
	}
	if (f_sideways) {
		cout << "f_sideways" << endl;
	}
	if (f_show_level_info) {
		cout << "f_show_level_info" << endl;
	}
	if (f_label_edges) {
		cout << "f_label_edges" << endl;
	}
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
	if (f_rotated) {
		cout << "rotated" << endl;
	}
	if (f_nodes) {
		cout << "nodes" << endl;
	}
	if (f_nodes_empty) {
		cout << "nodes_empty" << endl;
	}
	if (f_show_colors) {
		cout << "show_colors" << endl;
	}
	if (f_select_layers) {
		cout << "select_layers=" << select_layers << endl;
	}
	if (nb_layer_select) {
		cout << "layer_select=";
		Int_vec_print(cout, layer_select, nb_layer_select);
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






}}}

