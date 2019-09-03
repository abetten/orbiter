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
namespace foundations {


layered_graph::layered_graph()
{
	null();
}

layered_graph::~layered_graph()
{
	freeself();
}

void layered_graph::null()
{
	nb_nodes_total = 0;
	L = NULL;
	data1 = -1;
}

void layered_graph::freeself()
{
	if (L) {
		FREE_OBJECTS(L);
		}
	null();
}

void layered_graph::init(int nb_layers, int *Nb_nodes_layer, 
	const char *fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "layered_graph::init" << endl;
		}
	layered_graph::nb_layers = nb_layers;
	strcpy(layered_graph::fname_base, fname_base);
	L = NEW_OBJECTS(graph_layer, nb_layers);
	id_of_first_node = 0;
	for (i = 0; i < nb_layers; i++) {
		if (f_v) {
			cout << "layered_graph::init before L[i].init, i=" << i << endl;
			}
		L[i].init(Nb_nodes_layer[i], id_of_first_node, verbose_level);
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

void layered_graph::place(int verbose_level)
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

void layered_graph::place_with_y_stretch(double y_stretch, int verbose_level)
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

void layered_graph::place_with_grouping(int **Group_sizes,
		int *Nb_groups, double x_stretch, int verbose_level)
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
		L[i].place_with_grouping(Group_sizes[i], Nb_groups[i],
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

void layered_graph::add_edge(int l1, int n1, int l2, int n2,
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
	L[l1].Nodes[n1].add_neighbor(l2, n2, id2);
	L[l2].Nodes[n2].add_neighbor(l1, n1, id1);
	if (f_v) {
		cout << "layered_graph::add_edge done" << endl;
		}
}

void layered_graph::add_text(int l, int n,
		const char *text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::add_text l=" << l
				<< " n=" << n << endl;
		}
	L[l].Nodes[n].add_text(text);
}

void layered_graph::add_data1(int data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::add_data1" << endl;
		}
	data1 = data;
}

void layered_graph::add_node_vec_data(int l, int n,
		int *v, int len, int verbose_level)
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


void layered_graph::add_node_data1(int l, int n,
		int data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::add_node_data1 "
				"l=" << l << " n=" << n << endl;
		}
	L[l].Nodes[n].add_data1(data);
}

void layered_graph::add_node_data2(int l, int n,
		int data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::add_node_data2 "
				"l=" << l << " n=" << n << endl;
		}
	L[l].Nodes[n].add_data2(data);
}

void layered_graph::add_node_data3(int l, int n,
		int data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "layered_graph::add_node_data3 "
				"l=" << l << " n=" << n << endl;
		}
	L[l].Nodes[n].add_data3(data);
}



void layered_graph::draw_with_options(const char *fname,
		layered_graph_draw_options *O, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int x_min = 0; //, x_max = 10000;
	int y_min = 0; //, y_max = 10000;
	int factor_1000 = 1000;
	char fname_full[1000];
	double move_out = 0.01;
	int edge_label = 1;
	file_io Fio;
	
	strcpy(fname_full, fname);
	strcat(fname_full, ".mp");
	{
	mp_graphics G(fname_full, x_min, y_min,
			O->x_max, O->y_max,
			O->f_embedded, O->f_sideways);
	G.out_xmin() = 0;
	G.out_ymin() = 0;
	G.out_xmax() = O->xmax;
	G.out_ymax() = O->ymax;
	//cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
	
	G.tikz_global_scale = O->global_scale;
	G.tikz_global_line_width = O->global_line_width;

	G.header();
	G.begin_figure(factor_1000);

	int i, j, h, id, n, l;
	//int rad = 50;
	int rad_x_twice, rad_y_twice;
	int x, y, x2, y2;
	int Px[10], Py[10];
	int threshold = 50000;
	char text[1000];	
	int xoffset = 3 * O->rad / 2;
	int yoffset = 0;
	int own_id;
	numerics Num;
	sorting Sorting;

	rad_x_twice = O->rad >> 3;
	rad_y_twice = O->rad >> 3;
	G.user2dev_dist_x(rad_x_twice);
	G.user2dev_dist_y(rad_y_twice);

	//G.sl_thickness(30); // 100 is normal
	


	if (f_v) {
		cout << "layered_graph::draw" << endl;
		cout << "f_nodes_empty=" << O->f_nodes_empty << endl;
		}
	
	if (O->f_has_draw_begining_callback) {
		(*O->draw_begining_callback)(this, &G,
				O->x_max, O->y_max, O->f_rotated,
				O->rad * 4, O->rad * 4);
		}


	
	// draw the edges first:
	for (i = 0; i < nb_layers; i++) {

		if (O->f_select_layers) {
			int idx;
			
			if (!Sorting.int_vec_search_linear(O->layer_select,
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
						cout << "skipping node " << j
								<< " in layer " << i << endl;
						}
					continue;
					}
				}
			coordinates(L[i].Nodes[j].id, O->x_max, O->y_max,
					O->f_rotated, x, y);
			//G.circle(x, y, rad);


			own_id = L[i].Nodes[j].id;

			int *up;
			int *down;
			int nb_up, nb_down;

			up = NEW_int(L[i].Nodes[j].nb_neighbors);
			down = NEW_int(L[i].Nodes[j].nb_neighbors);
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
			
					if (!Sorting.int_vec_search_linear(O->layer_select,
							O->nb_layer_select, l, idx)) {
						continue;
						}			
					}
				if (l < i) {
					up[nb_up++] = id;
					if (f_v) {
						cout << "added an up link" << endl;
						}
					}
				else {
					down[nb_down++] = id;
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
					id = up[h];
					find_node_by_id(id, l, n);
					coordinates(id, O->x_max, O->y_max,
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
					G.polygon2(Px, Py, 0, 1);

					if (O->f_label_edges) {
						Px[2] = (Px[0] + Px[1]) >> 1;
						Py[2] = (Py[0] + Py[1]) >> 1;
						sprintf(text, "%d", edge_label);
						G.aligned_text_with_offset(Px[2], Py[2],
								xoffset, yoffset, "", text);
						edge_label++;
						}
					}
				}
			else {
				for (h = 0; h < nb_up; h++) {
					id = up[h];
					find_node_by_id(id, l, n);
					coordinates(id, O->x_max, O->y_max,
							O->f_rotated, x2, y2);
					Px[0] = x;
					Px[1] = x2;
					Py[0] = y;
					Py[1] = y2;
					G.polygon2(Px, Py, 0, 1);
					if (O->f_label_edges) {
						Px[2] = (Px[0] + Px[1]) >> 1;
						Py[2] = (Py[0] + Py[1]) >> 1;
						sprintf(text, "%d", edge_label);
						G.aligned_text_with_offset(Px[2], Py[2],
								xoffset, yoffset, "", text);
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
					id = down[h];
					find_node_by_id(id, l, n);
					coordinates(id, O->x_max, O->y_max, O->f_rotated, x2, y2);
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
					G.polygon2(Px, Py, 0, 1);
					if (O->f_label_edges) {
						Px[2] = (Px[0] + Px[1]) >> 1;
						Py[2] = (Py[0] + Py[1]) >> 1;
						sprintf(text, "%d", edge_label);
						G.aligned_text_with_offset(Px[2], Py[2],
								xoffset, yoffset, "", text);
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
					id = down[h];
					find_node_by_id(id, l, n);
					coordinates(id, O->x_max, O->y_max,
							O->f_rotated, x2, y2);
					Px[0] = x;
					Px[1] = x2;
					Py[0] = y;
					Py[1] = y2;
					G.polygon2(Px, Py, 0, 1);
					if (O->f_label_edges) {
						Px[2] = (Px[0] + Px[1]) >> 1;
						Py[2] = (Py[0] + Py[1]) >> 1;
						sprintf(text, "%d", edge_label);
						G.aligned_text_with_offset(Px[2], Py[2],
								xoffset, yoffset, "", text);
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

	// now draw the vertices:
	for (i = 0; i < nb_layers; i++) {

		if (O->f_select_layers) {
			int idx;
			
			if (!Sorting.int_vec_search_linear(O->layer_select,
					O->nb_layer_select, i, idx)) {
				continue;
				}
			
			}

		if (f_v) {
			cout << "layered_graph::draw drawing nodes in layer "
					<< i << "  with " << L[i].nb_nodes << " nodes:" << endl;
			}

		if (L[i].nb_nodes > threshold) {
			coordinates(L[i].Nodes[0].id, O->x_max, O->y_max,
					O->f_rotated, x, y);
			Px[0] = x;
			Py[0] = y;
			coordinates(L[i].Nodes[L[i].nb_nodes - 1].id,
					O->x_max, O->y_max, O->f_rotated, x, y);
			Px[1] = x;
			Py[1] = y;
			G.polygon2(Px, Py, 0, 1);
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
			coordinates(L[i].Nodes[j].id, O->x_max, O->y_max,
					O->f_rotated, x, y);


			char label[1000];

			
			if (L[i].Nodes[j].label) {
				if (f_v) {
					cout << "Vertex " << i << " " << j
							<< " has the following label: "
							<< L[i].Nodes[j].label << endl;
					}
				strcpy(label, L[i].Nodes[j].label);
				}
			else {
				if (f_v) {
					cout << "Vertex " << i << " " << j
							<< " does not have a label" << endl;
					}
				//G.circle(x, y, rad);
				}

			if (f_v) {
				cout << "Vertex " << i << " " << j
						<< " has the following data1 value: "
						<< L[i].Nodes[j].data1 << " radius_factor="
						<< L[i].Nodes[j].radius_factor << endl;
				}

			if (L[i].Nodes[j].radius_factor >= 1.) {
				sprintf(label, "{\\scriptsize %d}", L[i].Nodes[j].data1);
				}
			else {
				label[0] = 0;
				}

			G.nice_circle(x, y, O->rad * /*4 * */ L[i].Nodes[j].radius_factor);

			if (O->f_nodes_empty) {
				if (f_v) {
					cout << "Vertex " << i << " " << j
							<< " f_nodes_empty is TRUE" << endl;
					}
				}
			else {
				if (O->f_has_draw_vertex_callback) {
					//cout << "Vertex " << i << " " << j
					//<< " before (*O->draw_vertex_callback)" << endl;
					(*O->draw_vertex_callback)(this, &G, i, j, x, y,
							O->rad * /* 4 * */ L[i].Nodes[j].radius_factor,
							O->rad * /*4 * */ L[i].Nodes[j].radius_factor);
					}
				else {
					if (f_v) {
						cout << "layer " << i << " node " << j
								<< " label=" << label << endl;
						}

					if (TRUE /* L[i].Nodes[j].radius_factor >= 1.*/) {
						//G.circle_text(x, y, L[i].Nodes[j].label);
						G.aligned_text(x, y, "", label);
						//G.aligned_text(x, y, "", L[i].Nodes[j].label);
						}
					}
				}
			}
		}


	if (O->f_has_draw_ending_callback) {
		(*O->draw_ending_callback)(this, &G, O->x_max, O->y_max,
				O->f_rotated, O->rad * 4, O->rad * 4);
		}


	if (O->f_show_level_info) {
		// draw depth labels at the side:
		coordinates(L[0].Nodes[0].id,
				O->x_max, O->y_max, O->f_rotated, x, y);
		Px[0] = 1 * O->rad;
		Py[0] = y + 4 * O->rad;
		G.aligned_text(Px[0], Py[0], "", "Level");
		for (i = 0; i < nb_layers - 1; i++) {
			coordinates(L[i].Nodes[0].id,
					O->x_max, O->y_max, O->f_rotated, x, y);
			Px[0] = 2 * O->rad;
			Py[0] = y;
			coordinates(L[i + 1].Nodes[0].id,
					O->x_max, O->y_max, O->f_rotated, x, y);
			Px[1] = 2 * O->rad;
			Py[1] = y;
			G.polygon2(Px, Py, 0, 1);
			}
		for (i = 0; i < nb_layers; i++) {
			coordinates(L[i].Nodes[0].id,
					O->x_max, O->y_max, O->f_rotated, x, y);
			Px[0] = 1 * O->rad;
			Py[0] = y;
			Px[1] = 3 * O->rad;
			Py[1] = y;
			G.polygon2(Px, Py, 0, 1);
			}
		for (i = 0; i < nb_layers; i++) {
			char str[1000];
			
			coordinates(L[i].Nodes[0].id,
					O->x_max, O->y_max, O->f_rotated, x, y);
			Px[0] = 0;
			Py[0] = y;
			//G.nice_circle(Px[0], Py[0], rad * 4);
			sprintf(str, "%d", i);
			G.aligned_text(Px[0], Py[0], "", str);
			}
		}


	if (O->f_corners) {
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

void layered_graph::coordinates_direct(double x_in, double y_in,
		int x_max, int y_max, int f_rotated, int &x, int &y)
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

void layered_graph::coordinates(int id,
		int x_max, int y_max, int f_rotated, int &x, int &y)
{
	int l, n;

	find_node_by_id(id, l, n);

	coordinates_direct(L[l].Nodes[n].x_coordinate,
			L[l].y_coordinate, x_max, y_max, f_rotated, x, y);
#if 0
	x = (int)(L[l].Nodes[n].x_coordinate * x_max);
	y = (int)(L[l].y_coordinate * y_max);
#endif
}

void layered_graph::find_node_by_id(int id, int &l, int &n)
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


void layered_graph::write_file(char *fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	memory_object M;
	
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

void layered_graph::read_file(const char *fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	memory_object M;
	file_io Fio;
	
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
		memory_object *m, int verbose_level)
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
		memory_object *m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int version, a;
	char *p;
	
	if (f_v) {
		cout << "layered_graph::read_memory_object" << endl;
		}

	freeself();
	
	m->read_int(&version); // version number of this file format
	if (version != 1) {
		cout << "layered_graph::read_memory_object "
				"unknown version: version = " << version << endl;
		exit(1);
		}
	m->read_int(&nb_layers);
	m->read_int(&nb_nodes_total);
	m->read_int(&id_of_first_node);
	m->read_int(&data1);

	//cout << "layered_graph::read_memory_object
	// data1=" << data1 << endl;
	
	L = NEW_OBJECTS(graph_layer, nb_layers);

	for (i = 0; i < nb_layers; i++) {
		L[i].read_memory_object(m, verbose_level - 1);
		}
	
	m->read_string(p);
	strcpy(fname_base, p);
	FREE_char(p);

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


void layered_graph::compute_depth_first_ranks(int verbose_level)
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
		cout << "layered_graph::set_radius_factor_for_all_"
				"nodes_at_level level = " << lvl
				<< " radius_factor=" << radius_factor << endl;
		}
	for (j = 0; j < L[lvl].nb_nodes; j++) {
		L[lvl].Nodes[j].radius_factor = radius_factor;
		}
}










}
}




