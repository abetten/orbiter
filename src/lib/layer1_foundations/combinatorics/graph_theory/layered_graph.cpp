// layered_graph.cpp
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


layered_graph::layered_graph()
{
	Record_birth();
	nb_layers = 0;
	nb_nodes_total = 0;
	id_of_first_node = 0;
	L = NULL;
	// fname_base
	f_has_data1 = false;
	data1 = -1;
}

layered_graph::~layered_graph()
{
	Record_death();
	if (L) {
		FREE_OBJECTS(L);
	}
}

void layered_graph::init(
		int nb_layers, int *Nb_nodes_layer,
	std::string &fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "layered_graph::init" << endl;
	}
	layered_graph::nb_layers = nb_layers;
	layered_graph::fname_base.assign(fname_base);

	L = NEW_OBJECTS(graph_layer, nb_layers);
	id_of_first_node = 0;
	for (i = 0; i < nb_layers; i++) {
		if (f_v) {
			cout << "layered_graph::init "
					"before L[i].init, i=" << i << endl;
		}
		L[i].init(
				Nb_nodes_layer[i],
				id_of_first_node,
				verbose_level);
		id_of_first_node += Nb_nodes_layer[i];
	}
	nb_nodes_total = id_of_first_node;
	if (f_v) {
		cout << "layered_graph::init done" << endl;
	}
}

int layered_graph::nb_nodes()
{
	int N = 0;
	int i;

	for (i = 0; i < nb_layers; i++) {
		N += L[i].nb_nodes;
	}
	return N;
}

void layered_graph::print_nb_nodes_per_level()
{
	int i;

	cout << "layered_graph::print_nb_nodes_per_level" << endl;
	cout << "level & number of nodes " << endl;
	for (i = 0; i < nb_layers; i++) {
		cout << i << " & " <<  L[i].nb_nodes << "\\\\" << endl;
	}
}

double layered_graph::average_word_length()
{
	double s = 0.;
	double avg;
	int N = 0;
	int i;

	for (i = 0; i < nb_layers; i++) {
		N += L[i].nb_nodes;
		s += L[i].nb_nodes * (i + 1);
	}
	if (N == 0) {
		cout << "layered_graph::average_word_length N == 0" << endl;
		exit(1);
	}
	avg = s / (double) N;
	return avg;
}

void layered_graph::place(
		int verbose_level)
{
	double dy, dy2;
	int i;

	dy = 1. / (double) nb_layers;
	dy2 = dy * .5;
	for (i = 0; i < nb_layers; i++) {
		L[i].y_coordinate = 1. - i * dy - dy2;
		L[i].place(verbose_level);
	}
}

void layered_graph::place_upside_down(
		int verbose_level)
{
	double dy, dy2;
	int i;

	dy = 1. / (double) nb_layers;
	dy2 = dy * .5;
	for (i = 0; i < nb_layers; i++) {
		L[i].y_coordinate = i * dy - dy2;
		L[i].place(verbose_level);
	}
}

void layered_graph::place_with_y_stretch(
		double y_stretch, int verbose_level)
{
	double dy, dy2;
	int i;

	dy = y_stretch / (double) nb_layers;
	dy2 = dy * .5;
	for (i = 0; i < nb_layers; i++) {
		L[i].y_coordinate = 1. - i * dy - dy2;
		//L[i].place(verbose_level);
	}
}

void layered_graph::scale_x_coordinates(
		double x_stretch, int verbose_level)
{
	int i;

	for (i = 0; i < nb_layers; i++) {
		L[i].scale_x_coordinates(x_stretch, verbose_level);
		//L[i].place(verbose_level);
	}
}

void layered_graph::place_with_grouping(
		int **Group_sizes,
		int *Nb_groups, double x_stretch,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double dy, dy2;
	int i;

	if (f_v) {
		cout << "layered_graph::place_with_grouping "
				"x_stretch= " << x_stretch << endl;
	}
	dy = 1. / (double) nb_layers;
	dy2 = dy * .5;
	for (i = 0; i < nb_layers; i++) {
		if (f_v) {
			cout << "layered_graph::place_with_grouping "
					"layer " << i << endl;
		}
		L[i].y_coordinate = 1. - i * dy - dy2;
		L[i].place_with_grouping(
				Group_sizes[i], Nb_groups[i],
				x_stretch, verbose_level);
		if (f_v) {
			cout << "layered_graph::place_with_grouping "
					"layer " << i << " done" << endl;
		}
	}
	if (f_v) {
		cout << "layered_graph::place_with_grouping done" << endl;
	}
}

void layered_graph::add_edge(
		int l1, int n1, int l2, int n2,
		int edge_color,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int id1, id2;

	if (f_v) {
		cout << "layered_graph::add_edge l1=" << l1
				<< " n1=" << n1 << " l2="
				<< l2 << " n2=" << n2 << endl;
	}
	if (n1 < 0) {
		cout << "layered_graph::add_edge "
				"n1 is negative, n1=" << n1 << endl;
	}
	if (n2 < 0) {
		cout << "layered_graph::add_edge "
				"n2 is negative, n2=" << n2 << endl;
	}
	if (n1 >= L[l1].nb_nodes) {
		cout << "layered_graph::add_edge "
				"n1 >= L[l1].nb_nodes" << endl;
		cout << "l1 = " << l1 << endl;
		cout << "n1 = " << n1 << endl;
		cout << "L[l1].nb_nodes = " << L[l1].nb_nodes << endl;
		exit(1);
	}
	id1 = L[l1].Nodes[n1].id;
	if (n2 >= L[l2].nb_nodes) {
		cout << "layered_graph::add_edge n2 >= L[l2].nb_nodes" << endl;
		cout << "l2 = " << l2 << endl;
		cout << "n2 = " << n2 << endl;
		cout << "L[l2].nb_nodes = " << L[l2].nb_nodes << endl;
		exit(1);
	}
	id2 = L[l2].Nodes[n2].id;
	L[l1].Nodes[n1].add_neighbor(l2, n2, id2, edge_color);
	L[l2].Nodes[n2].add_neighbor(l1, n1, id1, edge_color);
	if (f_v) {
		cout << "layered_graph::add_edge done" << endl;
	}
}

void layered_graph::add_text(
		int l, int n,
		std::string &text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::add_text l=" << l
				<< " n=" << n << endl;
	}
	L[l].Nodes[n].add_text(text);
}

void layered_graph::add_data1(
		int data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::add_data1" << endl;
	}
	f_has_data1 = true;
	data1 = data;
}

void layered_graph::add_node_vec_data(
		int l, int n,
		long int *v, int len,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::add_node_vec_data "
				"l=" << l << " n=" << n << endl;
	}
	L[l].Nodes[n].add_vec_data(v, len);
}

void layered_graph::set_distinguished_element_index(
		int l, int n, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::set_distinguished_element_index "
				"l=" << l << " n=" << n << endl;
	}
	L[l].Nodes[n].set_distinguished_element(index);
}


void layered_graph::add_node_data1(
		int l, int n, int data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::add_node_data1 "
				"l=" << l << " n=" << n << endl;
	}
	L[l].Nodes[n].add_data1(data);
}

void layered_graph::add_node_data2(
		int l, int n, int data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::add_node_data2 "
				"l=" << l << " n=" << n << endl;
	}
	L[l].Nodes[n].add_data2(data);
}

void layered_graph::add_node_data3(
		int l, int n, int data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::add_node_data3 "
				"l=" << l << " n=" << n << endl;
	}
	L[l].Nodes[n].add_data3(data);
}



void layered_graph::draw_with_options(
		std::string &fname,
		other::graphics::layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int factor_1000 = 1000;
	string fname_full;
	other::orbiter_kernel_system::file_io Fio;
	
	fname_full = fname + ".mp";

	if (f_v) {
		cout << "layered_graph::draw_with_options "
				"fname_full = " << fname_full << endl;
	}
	if (O == NULL) {
		cout << "layered_graph::draw_with_options "
				"O == NULL" << endl;
		exit(1);
	}

	{


		if (f_v) {
			cout << "layered_graph::draw_with_options xin = " << O->xin << endl;
			cout << "layered_graph::draw_with_options yin = " << O->yin << endl;
			cout << "layered_graph::draw_with_options xout = " << O->xout << endl;
			cout << "layered_graph::draw_with_options yout = " << O->yout << endl;
			cout << "layered_graph::draw_with_options f_embedded = " << O->f_embedded << endl;
		}

		other::graphics::mp_graphics G;

		G.init(fname_full, O, verbose_level - 1);


		G.header();
		G.begin_figure(factor_1000);

		//G.sl_thickness(30); // 100 is normal

	
	
		if (f_v) {
			cout << "layered_graph::draw" << endl;
			cout << "f_nodes_empty=" << O->f_nodes_empty << endl;
			}

		if (O->f_has_draw_begining_callback) {
			(*O->draw_begining_callback)(
					this, &G,
					O->xin, O->yin, O->f_rotated,
					O->rad * 4, O->rad * 4);
		}


		

		// draw edges:

		if (f_v) {
			cout << "layered_graph::draw before draw_edges" << endl;
		}

		draw_edges(O, &G, verbose_level - 2);

		if (f_v) {
			cout << "layered_graph::draw after draw_edges" << endl;
		}


		// now draw the vertices:
		if (f_v) {
			cout << "layered_graph::draw before draw_vertices" << endl;
		}

		draw_vertices(O, &G, verbose_level - 2);

		if (f_v) {
			cout << "layered_graph::draw after draw_vertices" << endl;
		}


		if (O->f_has_draw_ending_callback) {
			(*O->draw_ending_callback)(this, &G, O->xin, O->yin,
					O->f_rotated, O->rad * 4, O->rad * 4);
		}


		if (f_v) {
			cout << "layered_graph::draw before draw_level_info" << endl;
		}
		draw_level_info(O, &G, verbose_level - 2);
		if (f_v) {
			cout << "layered_graph::draw after draw_level_info" << endl;
		}

		if (O->f_corners) {
			double move_out = 0.01;

			G.frame(move_out);
		}



		G.end_figure();
		G.footer();
	}
	if (f_v) {
		cout << "layered_graph::draw written file " << fname_full
				<< " of size " << Fio.file_size(fname_full) << endl;
	}
	
}

void layered_graph::draw_edges(
		other::graphics::layered_graph_draw_options *O,
		other::graphics::mp_graphics *G,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::draw_edges" << endl;
	}
	other::orbiter_kernel_system::numerics Num;
	other::data_structures::sorting Sorting;
	int threshold = 50000;
	int own_id;
	int x, y, x2, y2;
	int h, id, l, n;
	int i, j;
	int Px[10], Py[10];
	int edge_label = 1;
	int xoffset = 3 * O->rad / 2;
	int yoffset = 0;
	int rad_x_twice, rad_y_twice;



	rad_x_twice = O->rad >> 3;
	rad_y_twice = O->rad >> 3;

	// draw the edges first:
	for (i = 0; i < nb_layers; i++) {

		if (O->f_select_layers) {
			int idx;

			if (!Sorting.int_vec_search_linear(
					O->layer_select,
					O->nb_layer_select, i, idx)) {
				continue;
			}

		}

		if (f_v) {
			cout << "layered_graph::draw drawing edges in "
					"layer " << i << "  with " << L[i].nb_nodes
					<< " nodes:" << endl;
		}

		for (j = 0; j < L[i].nb_nodes; j++) {
			if (f_v) {
				cout << "layered_graph::draw drawing edges in "
						"layer " << i << " node " << j
						<< " neighbors = "
						<< L[i].Nodes[j].nb_neighbors << endl;
			}
			if (f_v) {
				cout << "Vertex " << i << " " << j << " at ("
						<< L[i].Nodes[j].x_coordinate << ","
						<< L[i].y_coordinate << ")" << endl;
			}

			if (L[i].nb_nodes > threshold) {
				if (j > 0 && j < L[i].nb_nodes - 1) {
					if (f_v) {
						cout << "skipping node " << j << " in layer " << i << endl;
					}
					continue;
				}
			}
			coordinates(
					L[i].Nodes[j].id, O->xin, O->yin,
					O->f_rotated, x, y);
			//G.circle(x, y, rad);


			own_id = L[i].Nodes[j].id;

			int *up;
			int *down;
			int *up_color;
			int *down_color;
			int nb_up, nb_down;

			up = NEW_int(L[i].Nodes[j].nb_neighbors);
			down = NEW_int(L[i].Nodes[j].nb_neighbors);
			up_color = NEW_int(L[i].Nodes[j].nb_neighbors);
			down_color = NEW_int(L[i].Nodes[j].nb_neighbors);
			nb_up = 0;
			nb_down = 0;

			for (h = 0; h < L[i].Nodes[j].nb_neighbors; h++) {
				id = L[i].Nodes[j].neighbor_list[h];
				if (f_v) {
					cout << "layered_graph::draw drawing edges in "
						"layer " << i << " node " << j << " neighbor = "
						<< h << " / " << L[i].Nodes[j].nb_neighbors
						<< " own_id=" << own_id << " id=" << id << endl;
				}
				if (id < own_id) {
					continue;
				}
				find_node_by_id(id, l, n);
				if (f_v) {
					cout << "is in layer " << l << " mode " << n << endl;
				}
				if (O->f_select_layers) {
					int idx;

					if (!Sorting.int_vec_search_linear(
							O->layer_select,
							O->nb_layer_select, l, idx)) {
						continue;
					}
				}
				if (l < i) {
					up[nb_up] = id;
					up_color[nb_up] = L[i].Nodes[j].Edge_color[h];
					nb_up++;
					if (f_v) {
						cout << "added an up link" << endl;
					}
				}
				else {
					down[nb_down] = id;
					down_color[nb_down] = L[i].Nodes[j].Edge_color[h];
					nb_down++;
					if (f_v) {
						cout << "added a down link" << endl;
					}
				}
			}


			if (f_v) {
				cout << "layered_graph::draw drawing edges, node "
						<< j << ", nb_up = " << nb_up << endl;
			}
			if (nb_up > threshold) {
				if (f_v) {
					cout << "layered_graph::draw drawing "
							"edges nb_up > threshold" << endl;
				}
				for (h = 0; h < nb_up; h++) {

					int edge_color;

					id = up[h];
					edge_color = up_color[h];
					edge_color++;

#if 0
					if (edge_color == 0) {
						edge_color = 1;
					}
#endif

					find_node_by_id(id, l, n);
					coordinates(
							id, O->xin, O->yin,
							O->f_rotated, x2, y2);
					if (h > 0 && h < nb_up - 1) {
#if 1
						Px[0] = x;
						Px[1] = (int)(x + ((double)(x2 - x)) /
								Num.norm_of_vector_2D(x, x2, y, y2) * rad_x_twice);
						Py[0] = y;
						Py[1] = (int)(y + ((double)(y2 - y)) /
								Num.norm_of_vector_2D(x, x2, y, y2) * rad_y_twice);
#endif
					}
					else {
						Px[0] = x;
						Px[1] = x2;
						Py[0] = y;
						Py[1] = y2;
					}

					G->sl_color(edge_color);
					G->polygon2(Px, Py, 0, 1);
					G->sl_color(1);

					if (O->f_label_edges) {
						Px[2] = (Px[0] + Px[1]) >> 1;
						Py[2] = (Py[0] + Py[1]) >> 1;
						string s;
						s = std::to_string(edge_label);
						G->aligned_text_with_offset(Px[2], Py[2],
								xoffset, yoffset, "", s);
						edge_label++;
					}
				}
			}
			else {
				for (h = 0; h < nb_up; h++) {

					int edge_color;


					id = up[h];
					edge_color = up_color[h];

					edge_color++;
#if 0
					if (edge_color == 0) {
						edge_color = 1;
					}
#endif

					find_node_by_id(id, l, n);
					coordinates(
							id, O->xin, O->yin,
							O->f_rotated, x2, y2);
					Px[0] = x;
					Px[1] = x2;
					Py[0] = y;
					Py[1] = y2;

					G->sl_color(edge_color);
					G->polygon2(Px, Py, 0, 1);
					G->sl_color(1);

					if (O->f_label_edges) {
						Px[2] = (Px[0] + Px[1]) >> 1;
						Py[2] = (Py[0] + Py[1]) >> 1;
						string s;
						s = std::to_string(edge_label);
						G->aligned_text_with_offset(Px[2], Py[2],
								xoffset, yoffset, "", s);
						edge_label++;
					}
					if (l > i) {
						if (f_v) {
							cout << "edge " << i << " " << j
									<< " to " << l << " " << n << endl;
						}
					}
				}
			}

			if (f_v) {
				cout << "layered_graph::draw drawing edges, node "
						<< j << ", nb_down = " << nb_down << endl;
			}
			if (nb_down > threshold) {
				if (f_v) {
					cout << "layered_graph::draw drawing edges "
							"nb_down > threshold" << endl;
				}
				for (h = 0; h < nb_down; h++) {

					int edge_color;

					id = down[h];
					edge_color = down_color[h] + 1;

#if 0
					if (edge_color == 0) {
						edge_color = 1;
					}
#endif

					find_node_by_id(id, l, n);
					coordinates(
							id, O->xin, O->yin, O->f_rotated, x2, y2);
					if (h > 0 && h < nb_down - 1) {
#if 1
						Px[0] = x;
						Px[1] = x + ((double)(x2 - x)) /
								Num.norm_of_vector_2D(x, x2, y, y2) * rad_x_twice;
						Py[0] = y;
						Py[1] = y + ((double)(y2 - y)) /
								Num.norm_of_vector_2D(x, x2, y, y2) * rad_y_twice;
#endif
					}
					else {
						Px[0] = x;
						Px[1] = x2;
						Py[0] = y;
						Py[1] = y2;
					}

					G->sl_color(edge_color);
					G->polygon2(Px, Py, 0, 1);
					G->sl_color(1);

					if (O->f_label_edges) {
						Px[2] = (Px[0] + Px[1]) >> 1;
						Py[2] = (Py[0] + Py[1]) >> 1;
						string s;
						s = std::to_string(edge_label);
						G->aligned_text_with_offset(
								Px[2], Py[2],
								xoffset, yoffset, "", s);
						edge_label++;
					}
					if (l > i) {
						if (f_v) {
							cout << "edge " << i << " " << j
									<< " to " << l << " " << n << endl;
						}
					}
				}
			}
			else {
				for (h = 0; h < nb_down; h++) {


					int edge_color;

					id = down[h];
					edge_color = down_color[h];
					edge_color++;

#if 0
					if (edge_color == 0) {
						edge_color = 1;
					}
#endif

					find_node_by_id(id, l, n);
					coordinates(
							id, O->xin, O->yin,
							O->f_rotated, x2, y2);
					Px[0] = x;
					Px[1] = x2;
					Py[0] = y;
					Py[1] = y2;


					G->sl_color(edge_color);
					G->polygon2(Px, Py, 0, 1);
					G->sl_color(1);


					if (O->f_label_edges) {
						Px[2] = (Px[0] + Px[1]) >> 1;
						Py[2] = (Py[0] + Py[1]) >> 1;
						string s;
						s = std::to_string(edge_label);
						G->aligned_text_with_offset(
								Px[2], Py[2],
								xoffset, yoffset, "", s);
						edge_label++;
					}
					if (l > i) {
						if (f_v) {
							cout << "edge " << i << " " << j
									<< " to " << l << " " << n << endl;
						}
					}
				}
			}

			FREE_int(up);
			FREE_int(down);
			FREE_int(up_color);
			FREE_int(down_color);


#if 0
			for (h = 0; h < L[i].Nodes[j].nb_neighbors; h++) {
				id = L[i].Nodes[j].neighbor_list[h];
				find_node_by_id(id, l, n);
				coordinates(id, x_max, y_max, x2, y2);
				Px[0] = x;
				Px[1] = x2;
				Py[0] = y;
				Py[1] = y2;
				G.polygon2(Px, Py, 0, 1);
				if (l > i) {
					if (f_v) {
						cout << "edge " << i << " " << j
								<< " to " << l << " " << n << endl;
					}
				}
			}
#endif

		}
	}
	if (f_v) {
		cout << "layered_graph::draw_edges done" << endl;
	}

}


void layered_graph::draw_vertices(
		other::graphics::layered_graph_draw_options *O,
		other::graphics::mp_graphics *G,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::draw_vertices" << endl;
	}

	other::data_structures::sorting Sorting;
	int i, j;
	int x, y;
	int Px[10], Py[10];
	int threshold = 50000;

	for (i = 0; i < nb_layers; i++) {

		if (O->f_select_layers) {
			int idx;

			if (!Sorting.int_vec_search_linear(
					O->layer_select,
					O->nb_layer_select, i, idx)) {
				continue;
			}

		}

		if (f_v) {
			cout << "layered_graph::draw_vertices drawing nodes in layer "
					<< i << "  with " << L[i].nb_nodes << " nodes:" << endl;
		}

		if (L[i].nb_nodes > threshold) {
			coordinates(L[i].Nodes[0].id, O->xin, O->yin,
					O->f_rotated, x, y);
			Px[0] = x;
			Py[0] = y;
			coordinates(L[i].Nodes[L[i].nb_nodes - 1].id,
					O->xin, O->yin, O->f_rotated, x, y);
			Px[1] = x;
			Py[1] = y;
			G->polygon2(Px, Py, 0, 1);
		}
		for (j = 0; j < L[i].nb_nodes; j++) {
			if (f_v) {
				cout << "Vertex " << i << " " << j << " at ("
						<< L[i].Nodes[j].x_coordinate << ","
						<< L[i].y_coordinate << ")" << endl;
			}
			if (L[i].nb_nodes > threshold) {
				if (j > 0 && j < L[i].nb_nodes - 1) {
					continue;
				}
			}
			coordinates(
					L[i].Nodes[j].id, O->xin, O->yin,
					O->f_rotated, x, y);


			string label;


			if (L[i].Nodes[j].label.length()) {
				if (f_v) {
					cout << "Vertex " << i << " " << j
							<< " has the following label: "
							<< L[i].Nodes[j].label << endl;
				}
				label.assign(L[i].Nodes[j].label);
			}
			else {
				if (f_v) {
					cout << "Vertex " << i << " " << j
							<< " does not have a label" << endl;
				}
				//G.circle(x, y, rad);
			}


			if (L[i].Nodes[j].f_has_data1) {
				if (f_v) {
					cout << "Vertex " << i << " " << j
							<< " has the following data1 value: "
							<< L[i].Nodes[j].data1 << " radius_factor="
							<< L[i].Nodes[j].radius_factor << endl;
				}

				if (L[i].Nodes[j].radius_factor >= 1.) {
					label = "{\\scriptsize " + std::to_string(L[i].Nodes[j].data1) + "}";
				}
				else {
					label.assign("");
				}
			}
			else {
				//label.assign("");
			}

			G->nice_circle(
					x, y, O->rad * /*4 * */ L[i].Nodes[j].radius_factor);

			if (O->f_nodes_empty) {
				if (f_v) {
					cout << "Vertex " << i << " " << j
							<< " f_nodes_empty is true" << endl;
				}
			}
			else {
				if (O->f_has_draw_vertex_callback) {
					//cout << "Vertex " << i << " " << j
					//<< " before (*O->draw_vertex_callback)" << endl;
					(*O->draw_vertex_callback)(this, G, i, j, x, y,
							O->rad * /* 4 * */ L[i].Nodes[j].radius_factor,
							O->rad * /*4 * */ L[i].Nodes[j].radius_factor);
				}
				else {
					if (f_v) {
						cout << "layer " << i << " node " << j
								<< " label=" << label << endl;
					}

					if (label.length() /* L[i].Nodes[j].radius_factor >= 1.*/) {
						//G.circle_text(x, y, L[i].Nodes[j].label);
						G->aligned_text(x, y, "", label);
						//G.aligned_text(x, y, "", L[i].Nodes[j].label);
					}
				}
			}
		}
	}

	if (f_v) {
		cout << "layered_graph::draw_vertices done" << endl;
	}
}

void layered_graph::draw_level_info(
		other::graphics::layered_graph_draw_options *O,
		other::graphics::mp_graphics *G,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::draw_level_info" << endl;
	}

	int i;
	int x, y;
	int Px[10], Py[10];

	if (O->f_show_level_info) {
		// draw depth labels at the side:
		coordinates(L[0].Nodes[0].id,
				O->xin, O->yin, O->f_rotated, x, y);
		Px[0] = 1 * O->rad;
		Py[0] = y + 4 * O->rad;
		string s;
		s.assign("Level");
		G->aligned_text(Px[0], Py[0], "", s);
		for (i = 0; i < nb_layers - 1; i++) {
			coordinates(L[i].Nodes[0].id,
					O->xin, O->yin, O->f_rotated, x, y);
			Px[0] = 2 * O->rad;
			Py[0] = y;
			coordinates(L[i + 1].Nodes[0].id,
					O->xin, O->yin, O->f_rotated, x, y);
			Px[1] = 2 * O->rad;
			Py[1] = y;
			G->polygon2(Px, Py, 0, 1);
		}
		for (i = 0; i < nb_layers; i++) {
			coordinates(L[i].Nodes[0].id,
					O->xin, O->yin, O->f_rotated, x, y);
			Px[0] = 1 * O->rad;
			Py[0] = y;
			Px[1] = 3 * O->rad;
			Py[1] = y;
			G->polygon2(Px, Py, 0, 1);
		}
		for (i = 0; i < nb_layers; i++) {

			coordinates(L[i].Nodes[0].id,
					O->xin, O->yin, O->f_rotated, x, y);
			Px[0] = 0;
			Py[0] = y;
			//G.nice_circle(Px[0], Py[0], rad * 4);
			string s;
			s = std::to_string(i);
			G->aligned_text(Px[0], Py[0], "", s);
		}
	}

	if (f_v) {
		cout << "layered_graph::draw_level_info done" << endl;
	}

}


void layered_graph::coordinates_direct(
		double x_in, double y_in,
		int x_max, int y_max, int f_rotated,
		int &x, int &y)
{
	double x1, y1;

	if (f_rotated) {
		x1 = 1 - y_in;
		y1 = x_in;
	}
	else {
		x1 = x_in;
		y1 = y_in;
	}
	x = (int)(x1 * x_max);
	y = (int)(y1 * y_max);
}

void layered_graph::coordinates(
		int id,
		int x_max, int y_max, int f_rotated,
		int &x, int &y)
{
	int l, n;

	find_node_by_id(id, l, n);

	coordinates_direct(
			L[l].Nodes[n].x_coordinate,
			L[l].y_coordinate, x_max, y_max, f_rotated, x, y);
#if 0
	x = (int)(L[l].Nodes[n].x_coordinate * x_max);
	y = (int)(L[l].y_coordinate * y_max);
#endif
}

void layered_graph::find_node_by_id(
		int id, int &l, int &n)
{
	int i, id0;
	
	id0 = 0;
	for (i = 0; i < nb_layers; i++) {
		if (id >= id0 && id < id0 + L[i].nb_nodes) {
			l = i;
			n = id - id0;
			return;
		}
		id0 += L[i].nb_nodes;
	}
	cout << "layered_graph::find_node_by_id "
			"did not find node with id " << id << endl;
	exit(1);
}


void layered_graph::write_file(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::memory_object M;
	
	if (f_v) {
		cout << "layered_graph::write_file" << endl;
	}
	M.alloc(1024 /* length */, verbose_level - 1);
	M.used_length = 0;
	M.cur_pointer = 0;
	write_memory_object(&M, verbose_level - 1);
	M.write_file(fname, verbose_level - 1);
	if (f_v) {
		cout << "layered_graph::write_file done" << endl;
	}
}

void layered_graph::read_file(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::memory_object M;
	other::orbiter_kernel_system::file_io Fio;
	
	if (f_v) {
		cout << "layered_graph::read_file "
				"reading file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	M.read_file(fname, verbose_level - 1);
	if (f_v) {
		cout << "layered_graph::read_file "
				"read file " << fname << endl;
	}
	M.cur_pointer = 0;
	read_memory_object(&M, verbose_level - 1);
	if (f_v) {
		cout << "layered_graph::read_file done" << endl;
	}
}

void layered_graph::write_memory_object(
		other::orbiter_kernel_system::memory_object *m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	
	if (f_v) {
		cout << "layered_graph::write_memory_object" << endl;
	}
	m->write_int(1); // version number of this file format
	if (f_vv) {
		cout << "after m->write_int(1), "
				"m->used_length = " << m->used_length << endl;
	}
	m->write_int(nb_layers);
	if (f_vv) {
		cout << "after m->write_int(nb_layers), "
				"nb_layers=" << nb_layers
				<< " m->used_length = " << m->used_length << endl;
	}
	m->write_int(nb_nodes_total);
	m->write_int(id_of_first_node);

	//cout << "layered_graph::write_memory_object data1=" << data1 << endl;
	m->write_int(f_has_data1);
	m->write_int(data1);
	for (i = 0; i < nb_layers; i++) {
		L[i].write_memory_object(m, verbose_level - 1);
	}
	m->write_string(fname_base);
	m->write_int(MAGIC_SYNC); // a check to see if the file is not corrupt
	if (f_v) {
		cout << "layered_graph::write_memory_object "
				"finished, data size (in chars) = "
				<< m->used_length << endl;
	}
}

void layered_graph::read_memory_object(
		other::orbiter_kernel_system::memory_object *m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int version, a;
	
	if (f_v) {
		cout << "layered_graph::read_memory_object" << endl;
	}

	
	m->read_int(&version); // version number of this file format
	if (version != 1) {
		cout << "layered_graph::read_memory_object "
				"unknown version: version = " << version << endl;
		exit(1);
	}
	m->read_int(&nb_layers);
	m->read_int(&nb_nodes_total);
	m->read_int(&id_of_first_node);
	m->read_int(&f_has_data1);
	m->read_int(&data1);

	//cout << "layered_graph::read_memory_object
	// data1=" << data1 << endl;
	
	L = NEW_OBJECTS(graph_layer, nb_layers);

	for (i = 0; i < nb_layers; i++) {
		L[i].read_memory_object(m, verbose_level - 1);
	}
	
	m->read_string(fname_base);

	m->read_int(&a);
	if (a != MAGIC_SYNC) {
		cout << "layered_graph::read_memory_object "
				"unknown the file seems to be corrupt" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "layered_graph::read_memory_object "
				"finished" << endl;
	}
}

void layered_graph::remove_edges(
		int layer1, int node1,
		int layer2, int node2,
		std::vector<std::vector<int> > &All_Paths,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::remove_edges" << endl;
		cout << "layer1 = " << layer1 << " node1=" << node1
				<< " layer2=" << layer2 << " node2=" << node2 << endl;
	}
	int l, n, j, id, l1, n1;
	int f_found;
	int h, d;


	for (l = layer1; l < layer2; l++) {
		for (n = 0; n < L[l].nb_nodes; n++) {
			for (j = 0; j < L[l].Nodes[n].nb_neighbors; j++) {
				id = L[l].Nodes[n].neighbor_list[j];
				find_node_by_id(id, l1, n1);
				if (l1 < l) {
					continue;
				}
				f_found = false;
				d = layer2 - l1;
				for (h = 0; h < All_Paths.size(); h++) {
					if (All_Paths[h][d] == n1 && All_Paths[h][d + 1] == n) {
						f_found = true;
						break;
					}
				}
				if (!f_found) {
					// we need to remove the edge (l,n), (l+1, n1)
					remove_edge(l, n, l1, n1, verbose_level - 2);
					j--;
				}
			}
		}
	}
	if (f_v) {
		cout << "layered_graph::remove_edges done" << endl;
	}
}

void layered_graph::remove_edge(
		int layer1, int node1,
		int layer2, int node2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::remove_edge" << endl;
	}
	if (!L[layer1].Nodes[node1].remove_neighbor(this, L[layer2].Nodes[node2].id, verbose_level - 2)) {
		cout << "layered_graph::remove_edge could not remove neighbor (1)" << endl;
		cout << "layer1 = " << layer1 << " node1=" << node1 << " layer2=" << layer2 << " node2=" << node2 << endl;
		exit(1);
	}
	if (!L[layer2].Nodes[node2].remove_neighbor(this, L[layer1].Nodes[node1].id, verbose_level - 2)) {
		cout << "layered_graph::remove_edge could not remove neighbor (2)" << endl;
		cout << "layer1 = " << layer1 << " node1=" << node1 << " layer2=" << layer2 << " node2=" << node2 << endl;
		exit(1);
	}
	if (f_v) {
		cout << "layered_graph::remove_edge done" << endl;
	}
}

void layered_graph::find_all_paths_between(
		int layer1, int node1,
		int layer2, int node2,
		std::vector<std::vector<int> > &All_Paths,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::find_all_paths_between" << endl;
		cout << "layer1 = " << layer1 << " node1=" << node1 << " layer2=" << layer2 << " node2=" << node2 << endl;
	}

	vector<int> Path;

	Path.resize(layer2 - layer1 + 1);


	find_all_paths_between_recursion(
			layer1, node1, layer2, node2,
			layer2, node2,
			All_Paths, Path,
			verbose_level);

	cout << "We found the following " << All_Paths.size()
			<< " paths between node " << node2 << " at layer " << layer2
			<< " and node " << node1 << " at layer " << layer1 << ":" << endl;

	int i;

	for (i = 0; i < All_Paths.size(); i++) {
		cout << "path " << i << " is: ";

		other::orbiter_kernel_system::Orbiter->Int_vec->print(cout, All_Paths[i]);

		cout << "\\\\" << endl;

	}
	if (f_v) {
		cout << "layered_graph::find_all_paths_between done" << endl;
	}
}

void layered_graph::find_all_paths_between_recursion(
		int layer1, int node1,
		int layer2, int node2,
		int l0, int n0,
		std::vector<std::vector<int> > &All_Paths,
		std::vector<int> &Path,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int id, l1, n1;

	if (f_v) {
		cout << "layered_graph::find_all_paths_between_recursion" << endl;
		cout << "layer1 = " << layer1 << " node1=" << node1
				<< " layer2=" << layer2 << " node2=" << node2
				<< " l0=" << l0 << " n0=" << n0 << endl;
	}

	graph_node *N = &L[l0].Nodes[n0];

	Path[layer2 - l0] = n0;

	std::vector<int> All_Parents;
	int i;

	N->find_all_parents(this, All_Parents, verbose_level);
	if (f_v) {
		cout << "layered_graph::find_all_paths_between_recursion All_Parents=";
		other::orbiter_kernel_system::Orbiter->Int_vec->print(cout, All_Parents);
		cout << endl;
	}

	for (i = 0; i < All_Parents.size(); i++) {
		id = All_Parents[i];
		find_node_by_id(id, l1, n1);
		if (l1 == layer1 && n1 == node1) {
			Path[layer2 - l1] = n1;
			All_Paths.push_back(Path);
		}
		if (l1 > layer1) {
			find_all_paths_between_recursion(layer1, node1, layer2, node2,
					l1, n1,
					All_Paths, Path,
					verbose_level);
		}
	}




	if (f_v) {
		cout << "layered_graph::find_all_paths_between_recursion done" << endl;
	}
}

void layered_graph::create_spanning_tree(
		int f_place_x, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int l, n, id, id1, l1, n1;
	
	if (f_v) {
		cout << "layered_graph::create_spanning_tree" << endl;
	}

	for (l = 0; l < nb_layers; l++) {
		for (n = 0; n < L[l].nb_nodes; n++) {
			graph_node *N = &L[l].Nodes[n];
			N->layer = l;

			N->allocate_tree_structure(0 /*verbose_level */);
		}
	}
	for (l = nb_layers - 1; l > 0; l--) {
		for (n = 0; n < L[l].nb_nodes; n++) {
			graph_node *N = &L[l].Nodes[n];
			id = N->id;
			
			id1 = N->find_parent(this, 0 /*verbose_level */);
			find_node_by_id(id1, l1, n1);
			graph_node *N1 = &L[l1].Nodes[n1];
			N1->register_child(this, id, 0 /*verbose_level */);
		}
	}

	compute_depth_first_ranks(verbose_level);


	if (f_place_x) {
		double left = 0;
		double right = 1;
		L[0].Nodes[0].place_x_based_on_tree(this,
				left, right, 0 /*verbose_level*/);
	}


	if (f_v) {
		cout << "layered_graph::create_spanning_tree done" << endl;
	}
}


void layered_graph::compute_depth_first_ranks(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int r = 0;
	
	if (f_v) {
		cout << "layered_graph::compute_depth_first_ranks" << endl;
	}

	L[0].Nodes[0].depth_first_rank_recursion(this,
			r, 0 /*verbose_level*/);

	if (f_v) {
		cout << "layered_graph::compute_depth_first_ranks done" << endl;
	}
}



void layered_graph::set_radius_factor_for_all_nodes_at_level(
		int lvl, double radius_factor, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;

	if (f_v) {
		cout << "layered_graph::set_radius_factor_for_all_nodes_at_level "
				"level = " << lvl
				<< " radius_factor=" << radius_factor << endl;
	}
	for (j = 0; j < L[lvl].nb_nodes; j++) {
		L[lvl].Nodes[j].radius_factor = radius_factor;
	}
}


void layered_graph::make_subset_lattice(
		int n, int depth, int f_tree,
	int f_depth_first, int f_breadth_first,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_layers = n + 1;
	int *Nb;
	int i, k, r, a, b, r0;
	int *set1;
	int *set2;
	algebra::number_theory::number_theory_domain NT;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "layered_graph::make_subset_lattice n=" << n << endl;
	}

	Nb = NEW_int(nb_layers);
	for (i = 0; i <= n; i++) {
		Nb[i] = Combi.int_n_choose_k(n, i);
	}

	set1 = NEW_int(n);
	set2 = NEW_int(n);

	add_data1(0, 0/*verbose_level*/);

	string dummy;
	dummy.assign("");

	init(depth + 1 /*nb_layers*/, Nb, dummy, verbose_level);
	if (f_vv) {
		cout << "layered_graph::make_subset_lattice "
				"after init" << endl;
	}
	place(verbose_level);
	if (f_vv) {
		cout << "layered_graph::make_subset_lattice "
				"after place" << endl;
	}

	// create vertex labels:
	for (k = 0; k <= depth; k++) {
		for (r = 0; r < Nb[k]; r++) {
			Combi.unrank_k_subset(r, set1, n, k);
			add_node_data1(k, r, set1[k - 1], 0/*verbose_level*/);

			string text;
			int a, j, j0;
			if (f_depth_first) {
				cout << "k=" << k << " r=" << r << " set=";
				Int_vec_print(cout, set1, k);
				cout << endl;
				a = 0;
				for (i = k - 1; i >= 0; i--) {
					if (i) {
						j0 = set1[i - 1];
					}
					else {
						j0 = -1;
					}
					cout << "i=" << i << " set1[i]=" << set1[i] << endl;
					for (j = j0 + 1; j < set1[i]; j++) {
						cout << "i = " << i << " j=" << j << " adding "
								 << NT.i_power_j(2, n - j - 1) << endl;
						a += NT.i_power_j(2, n - j - 1);
					}
				}
				a += k;
				text = std::to_string(a);
			}
			else if (f_breadth_first) {
				a = 0;
				for (i = 0; i < k; i++) {
					a += Nb[i];
				}
				a += r;
				text = std::to_string(a);
			}
			else {
				if (k) {
					text = std::to_string(set1[k - 1]);
				}
				else {
					text = "";
				}
			}


			add_text(k, r, text, 0/*verbose_level*/);
		}
	}

	// create edges:
	for (k = 1; k <= depth; k++) {
		for (r = 0; r < Nb[k]; r++) {
			Combi.unrank_k_subset(r, set1, n, k);

			if (f_tree) {
				for (a = k - 1; a >= k - 1; a--) {
					Int_vec_copy(set1, set2, k);
					for (b = a; b < k - 1; b++) {
						set2[b] = set2[b + 1];
						}
					r0 = Combi.rank_k_subset(set2, n, k - 1);
					add_edge(k - 1, r0, k, r,
							1, // edge_color
							0 /*verbose_level*/);
				}
			}
			else {
				for (a = k - 1; a >= 0; a--) {
					Int_vec_copy(set1, set2, k);
					for (b = a; b < k - 1; b++) {
						set2[b] = set2[b + 1];
					}
					r0 = Combi.rank_k_subset(set2, n, k - 1);
					add_edge(k - 1, r0, k, r,
							1, // edge_color
							0 /*verbose_level*/);
				}
			}
		}
	}


	FREE_int(set1);
	FREE_int(set2);
	if (f_v) {
		cout << "layered_graph::make_subset_lattice done" << endl;
	}
}


void layered_graph::init_poset_from_file(
		std::string &fname,
		int f_grouping, double x_stretch,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_layer;
	int *Nb;
	int *Nb_orbits;
	int **Orbit_length;
	int i, l1, n1, l2, n2, nb_v = 0, c = 0, a;

	if (f_v) {
		cout << "layered_graph::init_poset_from_file" << endl;
	}
	{
		ifstream fp(fname);
		fp >> nb_layer;
		Nb = NEW_int(nb_layer);
		Nb_orbits = NEW_int(nb_layer);
		Orbit_length = NEW_pint(nb_layer);
		nb_v = 0;
		for (i = 0; i < nb_layer; i++) {
			fp >> Nb[i];
			nb_v += Nb[i];
		}
		add_data1(0, 0/*verbose_level*/);

		string dummy;
		dummy.assign("");

		init(nb_layer, Nb, dummy, 0);
		place(0 /*verbose_level*/);


		for (l1 = 0; l1 < nb_layer; l1++) {
			for (n1 = 0; n1 < Nb[l1]; n1++) {
				fp >> a;

				string text;

				text = std::to_string(a);


				add_text(l1, n1, text, 0/*verbose_level*/);
			}
		}

		for (l1 = 0; l1 < nb_layer; l1++) {
			fp >> Nb_orbits[l1];
			Orbit_length[l1] = NEW_int(Nb_orbits[l1]);
			for (i = 0; i < Nb_orbits[l1]; i++) {
				fp >> Orbit_length[l1][i];
			}
		}

		while (true) {
			fp >> l1;
			if (l1 == -1) {
				break;
				}
			fp >> n1;
			fp >> l2;
			fp >> n2;
			add_edge(l1, n1, l2, n2,
					1, // edge_color
					0 /*verbose_level*/);
			c++;
		}
	}
	if (f_grouping) {
		place_with_grouping(
				Orbit_length,
				Nb_orbits, x_stretch, 0 /*verbose_level*/);
		}
	if (f_v) {
		cout << "created a graph with " << nb_v
				<< " vertices and " << c << " edges" << endl;
	}

	if (f_v) {
		cout << "layered_graph::init_poset_from_file done" << endl;
	}
}

// example file created in DISCRETA/sgls2.cpp for the subgroup lattice of Sym(4):
#if 0
5
1 13 11 4 1
1 2 2 2 2 2 2 3 3 3 3 2 2 2 4 4 4 6 6 6 6 4 4 4 4 8 8 8 12 24
1 1
3 6 4 3
4 3 4 3 1
2 3 1
1 1
0 0 1 0
0 0 1 1
0 0 1 2
0 0 1 3
0 0 1 4
0 0 1 5
0 0 1 6
0 0 1 7
0 0 1 8
0 0 1 9
0 0 1 10
0 0 1 11
0 0 1 12
1 0 2 0
1 0 2 3
1 0 2 4
1 1 2 1
1 1 2 3
1 1 2 5
1 2 2 2
1 2 2 3
1 2 2 6
1 3 2 0
1 3 2 5
1 3 2 6
1 4 2 2
1 4 2 4
1 4 2 5
1 5 2 1
1 5 2 4
1 5 2 6
1 6 2 3
1 6 3 3
1 7 2 5
1 7 3 3
1 8 2 6
1 8 3 3
1 9 2 4
1 9 3 3
1 10 2 0
1 10 2 7
1 10 2 10
1 11 2 2
1 11 2 8
1 11 2 10
1 12 2 1
1 12 2 9
1 12 2 10
2 0 3 0
2 1 3 1
2 2 3 2
2 3 4 0
2 4 4 0
2 5 4 0
2 6 4 0
2 7 3 0
2 8 3 2
2 9 3 1
2 10 3 0
2 10 3 1
2 10 3 2
2 10 3 3
3 0 4 0
3 1 4 0
3 2 4 0
3 3 4 0
-1
#endif







}}}}





