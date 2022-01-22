// tree.cpp
//
// Anton Betten
// February 7, 2003

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {


tree::tree()
{
	root = NULL;

	nb_nodes = 0;
	max_depth = 0;
	f_node_select = NULL;

	path = NULL;
	f_count_leaves = FALSE;
	leaf_count = 0;
}

tree::~tree()
{
	if (f_node_select) {
		FREE_int(f_node_select);
	}
}

#define TREEPATHLEN 10000
#define BUFSIZE_TREE 100000

void tree::init(tree_draw_options *Tree_draw_options,
		int xmax, int ymax, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tree::init reading tree from file " << Tree_draw_options->file_name << endl;
	}
	int f_vv = (verbose_level >= 1);
	char *buf;
	char *p_buf;
	int l, a, i, nb_nodes;
	char *c_data;
	int path[TREEPATHLEN];
	int color;
	string dummy;
	string label;
	data_structures::string_tools ST;
	
	nb_nodes = 0;
	buf = NEW_char(BUFSIZE_TREE);
	{
		ifstream f(Tree_draw_options->file_name);
		//f.getline(buf, BUFSIZE_TREE);
		while (TRUE) {
			if (f.eof()) {
				cout << "tree::init premature end of file" << endl;
				exit(1);
			}
			f.getline(buf, BUFSIZE_TREE);

			if (f_vv) {
				cout << "tree::init read line '" << buf << "'" << endl;
			}

			p_buf = buf;
			if (buf[0] == '#') {
				continue;
			}
			ST.s_scan_int(&p_buf, &a);
			if (a == -1) {
				break;
			}

			if (Tree_draw_options->f_restrict) {
				if (a == Tree_draw_options->restrict_excluded_color) {
					continue;
				}
			}
			nb_nodes++;
			}
		//s_scan_int(&p_buf, &nb_nodes);
	}
	if (f_v) {
		cout << "tree::init found " << nb_nodes
				<< " nodes in file " << Tree_draw_options->file_name << endl;
	}
	
	if (f_v) {
		cout << "tree::init calling root->init" << endl;
	}
	root = NEW_OBJECT(tree_node);
	root->init(0 /* depth */,
			NULL, FALSE, 0, FALSE, 0, dummy,
			verbose_level - 1);
	
	if (f_v) {
		cout << "tree::init reading the file again" << endl;
	}
	{
		ifstream f(Tree_draw_options->file_name);
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

			if (Tree_draw_options->f_restrict) {
				if (color == Tree_draw_options->restrict_excluded_color) {
					continue;
				}
			}


			if (f_vv) {
				cout << "tree::init trying to add node: " << buf << endl;
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
		cout << "tree::init finished adding nodes, max_depth = " << max_depth << endl;
		cout << "tree::init nb_nodes=" << tree::nb_nodes << endl;
	}
	
	if (f_vv) {
		root->print_depth_first();
	}
	tree::path = NEW_int(max_depth + 1);

	int my_nb_nodes;
	
	if (f_v) {
		cout << "tree::init before compute_DFS_ranks" << endl;
	}
	compute_DFS_ranks(my_nb_nodes, verbose_level);
	if (f_v) {
		cout << "tree::init after compute_DFS_ranks" << endl;
	}
	
	if (f_v) {
		cout << "tree::init before root->calc_weight" << endl;
	}
	root->calc_weight();
	if (f_v) {
		cout << "tree::init after root->calc_weight" << endl;
	}
	if (f_v) {
		cout << "tree::init before root->place_xy" << endl;
	}
	root->place_xy(0, xmax, ymax, max_depth);
	if (f_v) {
		cout << "tree::init after root->place_xy" << endl;
	}
	if (f_v) {
		cout << "tree::init before print_depth_first" << endl;
		root->print_depth_first();
		cout << "tree::init after print_depth_first" << endl;
	}
	FREE_char(buf);




	if (f_v) {
		cout << "tree::init done" << endl;
	}

}

void tree::draw(std::string &fname,
		tree_draw_options *Tree_draw_options,
		layered_graph_draw_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tree::draw" << endl;
	}
	string fname_full;
	
	fname_full.assign(fname);
	fname_full.append(".mp");

#if 0
	if (f_edge_labels) {
		strcat(fname_full, "e");
	}
#endif

	if (f_v) {
		cout << "tree::draw before draw_preprocess" << endl;
	}
	draw_preprocess(fname,
			Tree_draw_options,
			Opt,
			verbose_level);
	if (f_v) {
		cout << "tree::draw after draw_preprocess" << endl;
	}

	{
		int factor_1000 = 1000;

		mp_graphics G;
		G.init(fname_full, Opt, verbose_level);

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


		if (f_v) {
			cout << "tree::draw before root->draw_edges" << endl;
		}
		root->draw_edges(
				G, Tree_draw_options, Opt,
				FALSE, 0, 0,
				max_depth,
				this, verbose_level);
		if (f_v) {
			cout << "tree::draw after root->draw_edges" << endl;
		}

		G.sl_thickness(10); // 100 is normal

	
		if (f_v) {
			cout << "tree::draw before root->draw_vertices" << endl;
		}
		root->draw_vertices(G, Tree_draw_options, Opt,
				FALSE, 0, 0,
				max_depth,
				this, verbose_level);
		if (f_v) {
			cout << "tree::draw after root->draw_vertices" << endl;
		}
	
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
	if (f_v) {
		cout << "tree::draw done" << endl;
	}
	
}

void tree::draw_preprocess(std::string &fname,
		tree_draw_options *Tree_draw_options,
		layered_graph_draw_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tree::draw_preprocess" << endl;
	}
	if (Tree_draw_options->f_select_path) {
		int rk;
		int *my_path;
		int sz;

		rk = 0;

		if (f_v) {
			cout << "tree::draw_preprocess before root->compute_DFS_rank" << endl;
		}
		root->compute_DFS_rank(rk);
		nb_nodes = rk;

		f_node_select = NEW_int(nb_nodes);

		Orbiter->Int_vec->zero(f_node_select, nb_nodes);
		Orbiter->Int_vec->scan(Tree_draw_options->select_path_text, my_path, sz);

		if (f_v) {
			cout << "tree::draw_preprocess my_path = ";
			Orbiter->Int_vec->print(cout, my_path, sz);
			cout << endl;
		}

		if (FALSE) {
			int DFS_rk;
			root->find_node(DFS_rk, my_path, sz, verbose_level);
			if (f_v) {
				cout << "tree::draw_preprocess my_path = ";
				Orbiter->Int_vec->print(cout, my_path, sz);
				cout << " rk=" << DFS_rk << endl;
			}
			f_node_select[DFS_rk] = TRUE;
		}
		else {
			int i, a;
			std::vector<int> Rk;
			root->find_node_and_path(Rk, my_path, sz, verbose_level);
			for (i = 0; i < Rk.size(); i++) {
				a = Rk[i];
				f_node_select[a] = TRUE;
			}
		}


	}
	if (f_v) {
		cout << "tree::draw_preprocess done" << endl;
	}
}


void tree::circle_center_and_radii(int xmax, int ymax,
		int max_depth, int &x0, int &y0, int *&rad)
{
	int l, dy;
	double y;
	tree_node N;

	x0 = xmax * 0.5;
	y0 = ymax * 0.5;
	rad = NEW_int(max_depth + 1);
	for (l = 0; l <= max_depth; l++) {
		dy = (int)((double)ymax / (double)(max_depth + 1));
		y = N.calc_y_coordinate(ymax, l, max_depth);
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




