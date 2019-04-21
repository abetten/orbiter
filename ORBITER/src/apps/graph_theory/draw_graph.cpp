// draw_graph.C
//
// Anton Betten
// May 9 2013

#include "orbiter.h"

using namespace std;


using namespace orbiter;


void draw_graph(mp_graphics &G,
	int nb_v, int nb_e, int *E, int f_directed,
	int f_no_point_labels, 
	int f_point_labels, int *point_labels, 
	int point_label_offset, 
	int f_edge_labels, 
	int f_bipartite, int size_of_bipartition, 
	int f_partition, const char *partition_text, 
	int f_vertex_selection, const char *vertex_selection_text, 
	int rad, 
	int verbose_level);


int main(int argc, char **argv)
{
	int i, j;
	int xmax = 500000;
	int ymax = 500000;
	int dx = 400000;
	int dy = 400000;
	int f_rad = TRUE;
	int rad = 40000;
	int verbose_level = 0;
	int E[1000];
	int f_nb_V = FALSE;
	int nb_V = 0, nb_E = 0;
	int f_fname = FALSE;
	const char *fname = NULL;
	int point_label_offset = 0; 
	int f_edge_labels = FALSE;
	int f_directed = FALSE;
	int f_on_grid = FALSE;
	int *coords_2D;
	int f_embedded = FALSE;
	int f_sideways = FALSE;
	int f_scale = FALSE;
	double tikz_global_scale = .45;
	int f_line_width = FALSE;
	double tikz_global_line_width = 1.5;
	int f_export = FALSE;
	const char *fname_export = NULL;
	int f_edge_set = FALSE;
	const char *edge_set = NULL;
	int f_colored_graph = FALSE;
	const char *colored_graph_fname = NULL;
	int f_bipartite = FALSE;
	int size_of_bipartition = 0;
	int x, y, e, u, v;
	int f_partition = FALSE;
	const char *partition_text = NULL;
	char partition_text2[10000];
	int f_partition_by_color_classes = FALSE;
	int *partition = NULL;
	int *partition_first = NULL;
	int partition_length = 0;
	int f_vertex_selection = FALSE;
	const char *vertex_selection_text = NULL;
	int f_no_point_labels = FALSE;
	int f_point_labels = FALSE;
	int *point_labels = NULL;
	combinatorics_domain Combi;


	//int t0 = os_ticks();
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-x") == 0) {
			xmax = atoi(argv[++i]);
			cout << "-x " << xmax << endl;
			}
		else if (strcmp(argv[i], "-y") == 0) {
			ymax = atoi(argv[++i]);
			cout << "-y " << ymax << endl;
			}
		else if (strcmp(argv[i], "-dx") == 0) {
			dx = atoi(argv[++i]);
			cout << "-dx " << dx << endl;
			}
		else if (strcmp(argv[i], "-dy") == 0) {
			dy = atoi(argv[++i]);
			cout << "-dy " << dy << endl;
			}
		else if (strcmp(argv[i], "-rad") == 0) {
			f_rad = TRUE;
			rad = atoi(argv[++i]);
			cout << "-rad " << rad << endl;
			}
		else if (strcmp(argv[i], "-nb_V") == 0) {
			f_nb_V = TRUE;
			nb_V = atoi(argv[++i]);
			cout << "-nb_V " << nb_V << endl;
			}
		else if (strcmp(argv[i], "-edge_set") == 0) {
			edge_set = argv[++i];
			cout << "-edge_set " << edge_set << endl;
			}
		else if (strcmp(argv[i], "-E") == 0) {
			while (TRUE) {
				e = atoi(argv[++i]);
				if (e == -1) {
					break;
					}
				Combi.k2ij(e, u, v, nb_V);
				E[2 * nb_E + 0] = u;
				E[2 * nb_E + 1] = v;
				nb_E++;
				}
			cout << "-E " << endl;
			int_matrix_print(E, nb_E, 2);
			}
		else if (strcmp(argv[i], "-edges") == 0) {
			while (TRUE) {
				x = atoi(argv[++i]);
				if (x == -1) {
					break;
					}
				y = atoi(argv[++i]);
				//a = ij2k(x, y, nb_V);
				E[2 * nb_E + 0] = x;
				E[2 * nb_E + 1] = y;
				nb_E++;
				}
			cout << "-edges " << endl;
			int_matrix_print(E, nb_E, 2);
			}
		else if (strcmp(argv[i], "-fname") == 0) {
			f_fname = TRUE;
			fname = argv[++i];
			cout << "-fname " << fname << endl;
			}
		else if (strcmp(argv[i], "-directed") == 0) {
			f_directed = TRUE;
			}
		else if (strcmp(argv[i], "-on_grid") == 0) {
			f_on_grid = TRUE;
			coords_2D = NEW_int(nb_V * 2);
			int j;
			for (j = 0; j < nb_V * 2; j++) {
				coords_2D[j] = atoi(argv[++i]);
				}
			cout << "-on_grid ";
			int_vec_print(cout, coords_2D, nb_V * 2);
			cout << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			}
		else if (strcmp(argv[i], "-scale") == 0) {
			f_scale = TRUE;
			sscanf(argv[++i], "%lf", &tikz_global_scale);
			cout << "-scale " << tikz_global_scale << endl;
			}
		else if (strcmp(argv[i], "-line_width") == 0) {
			f_line_width = TRUE;
			sscanf(argv[++i], "%lf", &tikz_global_line_width);
			cout << "-line_width " << tikz_global_line_width << endl;
			}
		else if (strcmp(argv[i], "-export") == 0) {
			f_export = TRUE;
			fname_export = argv[++i];
			cout << "-export " << fname_export << endl;
			}
		else if (strcmp(argv[i], "-point_label_offset") == 0) {
			point_label_offset = atoi(argv[++i]);
			cout << "-point_label_offset " << point_label_offset << endl;
			}
		else if (strcmp(argv[i], "-colored_graph") == 0) {
			f_colored_graph = TRUE;
			colored_graph_fname = argv[++i];
			cout << "-colored_graph " << colored_graph_fname << endl;
			}
		else if (strcmp(argv[i], "-bipartite") == 0) {
			f_bipartite = TRUE;
			size_of_bipartition = atoi(argv[++i]);
			cout << "-bipartite " << size_of_bipartition << endl;
			}
		else if (strcmp(argv[i], "-partition") == 0) {
			f_partition = TRUE;
			partition_text = argv[++i];
			cout << "-partition " << partition_text << endl;
			}
		else if (strcmp(argv[i], "-vertex_selection") == 0) {
			f_vertex_selection = TRUE;
			vertex_selection_text = argv[++i];
			cout << "-vertex_selection " << vertex_selection_text << endl;
			}
		else if (strcmp(argv[i], "-partition_by_color_classes") == 0) {
			f_partition_by_color_classes = TRUE;
			cout << "-partition_by_color_classes " << endl;
			}
		else if (strcmp(argv[i], "-no_point_labels") == 0) {
			f_no_point_labels = TRUE;
			cout << "-no_point_labels " << endl;
			}
		else if (strcmp(argv[i], "-point_labels") == 0) {
			f_point_labels = TRUE;
			cout << "-point_labels " << endl;
			}
		
		}

	if (f_partition) {
		strcpy(partition_text2, partition_text); 
		}

	if (f_fname == FALSE) {
		cout << "Please use option -fname <fname> to "
				"specify the base file name for the output file" << endl;
		exit(1);
		}

	if (f_edge_set) {
		int *Edges;
		int nb_edges;
		int_vec_scan(edge_set, Edges, nb_edges);
		nb_E = 0;
		for (i = 0; i < nb_edges; i++) {
			Combi.k2ij(Edges[i], u, v, nb_V);
			E[2 * nb_E + 0] = u;
			E[2 * nb_E + 1] = v;
			nb_E++;
			}
		cout << "done parsing the set of edges." << endl;
		}
	else if (f_colored_graph) {
		colored_graph *CG;

		CG = NEW_OBJECT(colored_graph);
		cout << "loading colored_graph from file "
				<< colored_graph_fname << endl;
		CG->load(colored_graph_fname, verbose_level);
		cout << "After loading the graph, "
				"CG->nb_points=" << CG->nb_points << endl;
		CG->print_points_and_colors();
		CG->print_adjacency_list();

		if (f_nb_V) {
			if (nb_V != CG->nb_points) {
				cout << "nb_V != CG->nb_points" << endl;
				exit(1);
				}
			}
		else {
			f_nb_V = TRUE;
			nb_V = CG->nb_points;
			}
		nb_E = 0;
		for (i = 0; i < nb_V; i++) {
			for (j = i + 1; j < nb_V; j++) {
				if (CG->is_adjacent(i, j)) {
					E[2 * nb_E + 0] = i;
					E[2 * nb_E + 1] = j;
					nb_E++;
					}
				}
			}
		if (f_partition_by_color_classes) {
			CG->partition_by_color_classes(
				partition, partition_first, 
				partition_length, 
				verbose_level);
			f_partition = TRUE;
			partition_text2[0] = 0;
			for (i = 0; i < partition_length; i++) {
				sprintf(partition_text2 + strlen(partition_text2), "%d", partition[i]);
				if (i < partition_length - 1) {
					sprintf(partition_text2 + strlen(partition_text2), ",");
					}
				}
			}
		if (f_point_labels) {
			point_labels = NEW_int(CG->nb_points);
			int_vec_copy(CG->points, point_labels, CG->nb_points);
			}
		}


	char fname2[1000];
	char ext[1000];

	sprintf(fname2, "%s", fname);
	
	get_extension_if_present_and_chop_off(fname2, ext);
	if (f_vertex_selection) {
		strcat(fname2, "_");
		strcat(fname2, vertex_selection_text);
		strcat(fname2, ext);
		}

	cout << "coutput file name: " << fname2 << endl;


#if 0
	cout << "The edges are:" << endl;
	int_matrix_print(E, nb_E, 2);
#endif

	{
	//mp_graphics G;

	int x_min = 0, y_min = 0;
	int factor_1000 = 1000;

	
	mp_graphics G(fname2, x_min, y_min, xmax, ymax, f_embedded, f_sideways);
	//G.setup(fname2, 0, 0, ONE_MILLION, ONE_MILLION, xmax, ymax, f_embedded, scale, line_width);
	G.out_xmin() = 0;
	G.out_ymin() = 0;
	G.out_xmax() = xmax;
	G.out_ymax() = ymax;
	//cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
	
	G.tikz_global_scale = tikz_global_scale;
	G.tikz_global_line_width = tikz_global_line_width;

	G.header();
	G.begin_figure(factor_1000);


	if (f_on_grid) {
		//int dx = 400000;
		//int dy = 400000;
		int Base[4];

		Base[0] = (int)(dx * 0.5);
		Base[1] = 0;
		//Base[2] = (int)(dx * 0.5);
		//Base[3] = (int)(dy * 0.866025); // sqrt(3) / 2
		Base[2] = (int)(dx * 0.0);
		Base[3] = (int)(dy * 0.5); // sqrt(3) / 2
		cout << "before draw_graph_on_2D_grid" << endl;
		G.draw_graph_on_2D_grid(0 /* x */, 0 /* y */,
			dx, dy, rad, nb_V, E, nb_E, coords_2D, Base, 
			f_point_labels, point_label_offset, f_directed);
		}
	else {
		int *E2;

		E2 = NEW_int(nb_E);
		for (i = 0; i < nb_E; i++) {
			E2[i] = Combi.ij2k(E[2 * i + 0], E[2 * i + 1], nb_V);
			}
		//cout << "before draw_graph" << endl;
		draw_graph(G, nb_V, nb_E, E2, f_directed,
			f_no_point_labels, 
			f_point_labels, point_labels, 
			point_label_offset, f_edge_labels, 
			f_bipartite, size_of_bipartition, 
			f_partition, partition_text2, 
			f_vertex_selection, vertex_selection_text, 
			rad, verbose_level);
			// not in GALOIS/draw.C

		FREE_int(E2);
		}

#if 0
	if (f_labels) {
		char str[1000];
		
		for (i = 0; i < nb_V; i++) {
			sprintf(str, "%d", i);
			
			}
		}
#endif
	
	if (f_export) {
		int *Adj;

		Adj = NEW_int(nb_V * nb_V);
		int_vec_zero(Adj, nb_V * nb_V);
		for (i = 0; i < nb_E; i++) {
			x = E[2 * i + 0];
			y = E[2 * i + 1];
			//k2ij(a, x, y, nb_V);
			if (f_directed) {
				Adj[x * nb_V + y] = 1;
				}
			else {
				Adj[x * nb_V + y] = 1;
				Adj[y * nb_V + x] = 1;
				}
			}

		{
			ofstream fp(fname_export);
			fp << "[";
			for (i = 0; i < nb_V; i++) {
				fp << "[";
				for (j = 0; j < nb_V; j++) {
					fp << Adj[i * nb_V + j];
					if (j < nb_V - 1) {
						fp << ",";
						}
					}
				fp << "]";
				if (i < nb_V - 1) {
					fp << ", ";
					}
				fp << endl;
				}
			fp << "]" << endl;

			
		}
		file_io Fio;
		cout << "Written file " << fname_export << " of size "
				<< Fio.file_size(fname_export) << endl;
		FREE_int(Adj);
		}
	G.finish(cout, TRUE);
	}
}

void draw_graph(mp_graphics &G,
	int nb_v, int nb_e, int *E, int f_directed,
	int f_no_point_labels, 
	int f_point_labels, int *point_labels, 
	int point_label_offset, 
	int f_edge_labels, 
	int f_bipartite, int size_of_bipartition, 
	int f_partition, const char *partition_text, 
	int f_vertex_selection, const char *vertex_selection_text, 
	int rad, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, h, e1, e2;
	int Px[1000], Py[1000];
	int Qx[1000], Qy[1000];
	int Rx[1000], Ry[1000];
	char str[1000];
	double phi = ((double) 360) / nb_v;
	double phi_half = phi * 0.5;
	int rad1 = 170000;
	int rad1a = rad1 * .8;
	int rad1b = rad1 * 1.2;
	int rad1c = rad1 * 1.3;
	//int rad = 30000;
	int dx, sz, sz0;
	int f_swap, a;
	int phi0 = 0;

	int *partition;
	int *partition_first = NULL;
	int partition_length = 0;
	int *vertex_selection;
	int *f_vertex_selected;
	int vertex_selection_length;
	int f_go = FALSE;
	numerics Num;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "draw_graph" << endl;
		}
	if (f_vertex_selection) {
		int_vec_scan(vertex_selection_text,
				vertex_selection, vertex_selection_length);
		cout << "vertex_selection: ";
		int_vec_print(cout, vertex_selection, vertex_selection_length);
		cout << endl;
		f_vertex_selected = NEW_int(nb_v);
		int_vec_zero(f_vertex_selected, nb_v);
		for (i = 0; i < vertex_selection_length; i++) {
			a = vertex_selection[i];
			f_vertex_selected[a] = TRUE;
			}
		}

	if (f_partition) {
		int_vec_scan(partition_text, partition, partition_length);
		cout << "partiton: ";
		int_vec_print(cout, partition, partition_length);
		cout << endl;
		partition_first = NEW_int(partition_length + 1);
		partition_first[0] = 0;
		for (i = 0; i < partition_length; i++) {
			partition_first[i + 1] = partition_first[i] + partition[i];
			}
		}
	
	if (f_bipartite) {
		for (h = 0; h < 2; h++) {
			if (h == 0) {
				sz0 = 0;
				sz = size_of_bipartition;
				}
			else {
				sz0 = size_of_bipartition;
				sz = nb_v - size_of_bipartition;
				}

			dx = (int)((double)(4 * rad1) / (sz - 1));	
			for (i = 0; i < sz; i++) {
				Px[sz0 + i] = i * dx;
				if (h == 0) {
					Py[sz0 + i] = rad1;
					}
				else {
					Py[sz0 + i] = -rad1;
					}
				}
			}
		}
	else {
		for (i = 0; i < nb_v; i++) {
			Num.on_circle_int(Px, Py, i, ((int)(phi0 + i * phi)) % 360, rad1);
			//cout << "i=" << i << " Px=" << Px[i] << " Py=" << Py[i] << endl;
			}
		for (i = 0; i < nb_v; i++) {
			Num.on_circle_int(Qx, Qy, 2 * i, ((int)(phi0 + i * phi - phi_half)) % 360, rad1a);
			//cout << "i=" << i << " Qx=" << Qx[2 * i] << " Qy=" << Qy[2 * i] << endl;
			}
		for (i = 0; i < nb_v; i++) {
			Num.on_circle_int(Qx, Qy, 2 * i + 1, ((int)(phi0 + i * phi - phi_half)) % 360, rad1b);
			//cout << "i=" << i << " Qx=" << Qx[2 * i + 1] << " Qy=" << Qy[2 * i + 1] << endl;
			}
		if (f_partition) {
			for (i = 0; i < partition_length; i++) {
				double m;

				m = (double) partition_first[i] + (double) partition[i] * 0.5;
				Num.on_circle_int(Rx, Ry, i, ((int)(phi0 + m * phi - phi_half)) % 360, rad1c);
				}
			}
		}
	G.sl_thickness(100);
	for (i = 0; i < nb_e; i++) {


		if (f_directed) {
			f_swap = E[i] % 2;
			a = E[i] >> 1;
			Combi.k2ij(a, e1, e2, nb_v);
			if (f_swap) {
				//cout << "directed edge " << i << " from " << e2 << " to " << e1 << endl;
				}
			else {
				//cout << "directed edge " << i << " from " << e1 << " to " << e2 << endl;
				}
			}
		else {
			Combi.k2ij(E[i], e1, e2, nb_v);
			//cout << "edge " << i << " from " << e1 << " to " << e2 << endl;
			}

		if (f_vertex_selection) {
			if (f_vertex_selected[e1] && f_vertex_selected[e2]) {
				f_go = TRUE;
				}
			else {
				f_go = FALSE;
				}
			}
		else {
			f_go = TRUE;
			}


		if (f_go) {
			if (f_directed) {
				int s, t;


				G.sl_ends(0, 1);
				if (f_swap) {
					s = e2;
					t = e1;
					}
				else {
					s = e1;
					t = e2;
					}
				Num.affine_pt1(Px, Py, s, s, t, 0.80, nb_v);
				G.polygon2(Px, Py, s, nb_v);
				G.sl_ends(0, 0);
				G.polygon2(Px, Py, s, t);
				}
			else {
				G.polygon2(Px, Py, e1, e2);
				}
			if (f_edge_labels) {

				Num.affine_pt1(Px, Py, e1, e1, e2, 0.25, nb_v);
				itoa(str, 1000, E[i]);
				G.aligned_text_array(Px, Py, nb_v, "", str);
				}
			}
		}
	G.sl_thickness(100);
	str[0] = 0;
	//cout << "circle_text:" << endl;
	for (i = 0; i < nb_v; i++) {


		if (f_vertex_selection) {
			if (f_vertex_selected[i]) {
				f_go = TRUE;
				}
			else {
				f_go = FALSE;
				}
			}
		else {
			f_go = TRUE;
			}

		if (f_go) {
			if (f_no_point_labels) {
				G.nice_circle(Px[i], Py[i], rad);
				}
			else {
				if (f_point_labels) {
					itoa(str, 1000, point_labels[i]);

					//G.nice_circle(Px[i], Py[i], rad);
					G.circle_text(Px[i], Py[i], rad, str);
					}
				else {
					itoa(str, 1000, i + point_label_offset);
					G.circle_text(Px[i], Py[i], rad, str);
					}
				}
			}
		}
	if (f_partition) {
		char str[1000];

		//G.sl_thickness(25);
		//G.circle(0, 0, rad1a);
		//G.sl_thickness(100);
		G.circle(0, 0, rad1b);
		for (i = 0; i < partition_length; i++) {
			sprintf(str, "${\\bf C}_{%d}$", i);
			a = partition_first[i];
			G.polygon2(Qx, Qy, 2 * a + 0, 2 * a + 1);
			G.aligned_text(Rx[i], Ry[i], "", str);
			}
		}
	if (f_vertex_selection) {
		FREE_int(vertex_selection);
		FREE_int(f_vertex_selected);
		}
	if (f_partition) {
		FREE_int(partition);
		FREE_int(partition_first);
		}
	if (f_v) {
		cout << "draw_graph done" << endl;
		}
}



