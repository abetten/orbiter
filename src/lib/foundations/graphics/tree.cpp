// tree.cpp
//
// Anton Betten
// February 7, 2003

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


tree::tree()
{
	root = NULL;
	nb_nodes = 0;
	max_depth = 0;
	path = NULL;
	f_count_leaves = FALSE;
	leaf_count = 0;
}

tree::~tree()
{
}

#define TREEPATHLEN 10000
#define BUFSIZE_TREE 100000

void tree::init(std::string &fname,
		int xmax, int ymax, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	char *buf;
	char *p_buf;
	int l, a, i, nb_nodes;
	char *c_data;
	int path[TREEPATHLEN];
	int color;
	string dummy;
	string label;
	string_tools ST;
	
	if (f_v) {
		cout << "reading tree from file " << fname << endl;
	}
	nb_nodes = 0;
	buf = NEW_char(BUFSIZE_TREE);
	{
		ifstream f(fname);
		//f.getline(buf, BUFSIZE_TREE);
		while (TRUE) {
			if (f.eof()) {
				cout << "premature end of file" << endl;
				exit(1);
			}
			f.getline(buf, BUFSIZE_TREE);

			if (f_vv) {
				cout << "read line '" << buf << "'" << endl;
			}

			p_buf = buf;
			if (buf[0] == '#') {
				continue;
			}
			ST.s_scan_int(&p_buf, &a);
			if (a == -1) {
				break;
			}
			nb_nodes++;
			}
		//s_scan_int(&p_buf, &nb_nodes);
	}
	if (f_v) {
		cout << "found " << nb_nodes << " nodes in file " << fname << endl;
	}
	
	if (f_v) {
		cout << "calling root->init" << endl;
	}
	root = NEW_OBJECT(tree_node);
	root->init(0 /* depth */,
			NULL, FALSE, 0, FALSE, 0, dummy,
			verbose_level - 1);
	
	if (f_v) {
		cout << "reading the file again" << endl;
	}
	{
		ifstream f(fname);
		//f.getline(buf, BUFSIZE_TREE);
		while (TRUE) {
			if (f.eof()) {
				cout << "premature end of file" << endl;
				exit(1);
			}
			f.getline(buf, BUFSIZE_TREE);
			p_buf = buf;
			if (buf[0] == '#') {
				continue;
			}
			ST.s_scan_int(&p_buf, &l);
			if (l == -1) {
				break;
			}
			if (l >= TREEPATHLEN) {
				cout << "tree::init overflow, please increase "
						"the value of TREEPATHLEN" << endl;
				cout << "l=" << l << endl;
				exit(1);
			}
			if (f_vv) {
				cout << "reading entry at depth " << l << endl;
			}
			for (i = 0; i < l; i++) {
				ST.s_scan_int(&p_buf, &path[i]);
			}

			// read one more entry, the color of the node:
			ST.s_scan_int(&p_buf, &color);

			// skip over whitespace:
			while (*p_buf == ' ') {
				p_buf++;
			}

			// read the label:
			c_data = p_buf;
			for (i = 0; c_data[i]; i++) {
				if (c_data[i] == '#') {
					c_data[i] = 0;
					break;
				}
			}
			label.assign(c_data);
			if (f_vv) {
				cout << "trying to add node: " << buf << endl;
			}
			root->add_node(l, 0, path, color, label, 0/*verbose_level - 1*/);
			if (f_vv) {
				cout << "node added: " << buf << endl;
			}
			tree::nb_nodes++;
			max_depth = MAXIMUM(max_depth, l);
		}
	}
	if (f_v) {
		cout << "finished adding nodes, max_depth = " << max_depth << endl;
		cout << "tree::nb_nodes=" << tree::nb_nodes << endl;
	}
	
	if (f_vv) {
		root->print_depth_first();
	}
	tree::path = NEW_int(max_depth + 1);

	int my_nb_nodes;
	
	compute_DFS_ranks(my_nb_nodes, verbose_level);
	
	root->calc_weight();
	root->place_xy(0, xmax, ymax, max_depth);
	if (f_v) {
		root->print_depth_first();
	}
	FREE_char(buf);

}

void tree::draw(std::string &fname,
		layered_graph_draw_options *Opt,
		int f_has_draw_vertex_callback,
		void (*draw_vertex_callback)(tree *T,
			mp_graphics *G, int *v, int layer, tree_node *N,
		int x, int y, int dx, int dy), 
		int verbose_level)
{
	string fname_full;
	
	fname_full.assign(fname);
	fname_full.append(".mp");

#if 0
	if (f_edge_labels) {
		strcat(fname_full, "e");
	}
#endif


	{
		//int x_min = 0;
		//int y_min = 0;
		int factor_1000 = 1000;

		mp_graphics G;
		G.init(fname_full, Opt, verbose_level);
#if 0
		mp_graphics G(fname_full, x_min, y_min,
				Opt->xin, Opt->yin,
				Opt->f_embedded, Opt->f_sideways,
				verbose_level - 1);
		G.out_xmin() = 0;
		G.out_ymin() = 0;
		G.out_xmax() = Opt->xout;
		G.out_ymax() = Opt->yout;
		//cout << "xmax/ymax = " << xmax << " / " << ymax << endl;

		G.tikz_global_scale = Opt->scale;
		G.tikz_global_line_width = Opt->line_width;
#endif

		G.header();
		G.begin_figure(factor_1000);
	
	
		//G.frame(0.05);

#if 0
		int x = 500000, y;
		calc_y_coordinate(y, 0, max_depth);

		if (f_circletext) {
			G.circle_text(x, y, "$\\emptyset$");
			}
		else {
			G.circle(x, y, 5);
			}
#endif

		//root->draw_sideways(G, f_circletext, f_i,
		//FALSE, 10000 - 0, 10000 - 0, max_depth, f_edge_labels);


#if 0
		int *radii = NULL;
		int x0, y0;
		if (f_on_circle) {
			int l;

#if 1
			G.sl_thickness(30); // 100 is normal
			//G.sl_thickness(200); // 100 is normal
			circle_center_and_radii(xmax_in, ymax_in, max_depth, x0, y0, radii);
			for (l = 1; l <= max_depth; l++) {
				G.circle(x0, y0, radii[l]);
				}
#endif
			}
#endif


		G.sl_thickness(30); // 100 is normal



		leaf_count = 0;

		//int f_circletext = TRUE;
		//int f_i = TRUE;


		root->draw_edges(G, Opt,
				FALSE, 0, 0,
				max_depth,
				f_has_draw_vertex_callback, draw_vertex_callback,
				this);

		G.sl_thickness(10); // 100 is normal

	
		root->draw_vertices(G, Opt,
				FALSE, 0, 0,
				max_depth,
				f_has_draw_vertex_callback, draw_vertex_callback,
				this);
	
#if 0
		if (f_on_circle) {
			FREE_int(radii);
		}
#endif


		G.end_figure();
		G.footer();
	}
	file_io Fio;

	cout << "written file " << fname_full << " of size "
			<< Fio.file_size(fname_full) << endl;
	
}

void tree::circle_center_and_radii(int xmax, int ymax,
		int max_depth, int &x0, int &y0, int *&rad)
{
	int l, dy;
	double y;

	x0 = xmax * 0.5;
	y0 = ymax * 0.5;
	rad = NEW_int(max_depth + 1);
	for (l = 0; l <= max_depth; l++) {
		dy = (int)((double)ymax / (double)(max_depth + 1));
		y = tree_node_calc_y_coordinate(ymax, l, max_depth);
		y = ymax - y;
		y -= dy * 0.5;
		y /= ymax;
		rad[l] = y * xmax * 0.5;
	}
}

void tree::compute_DFS_ranks(int &nb_nodes, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk;

	if (f_v) {
		cout << "tree::compute_DFS_ranks" << endl;
	}
	rk = 0;
	root->compute_DFS_rank(rk);
	nb_nodes = rk;
	if (f_v) {
		cout << "tree::compute_DFS_ranks done" << endl;
	}
}

}
}




