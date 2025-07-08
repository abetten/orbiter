// poset_classification_draw.cpp
//
// Anton Betten
// moved out of poset_classification.cpp  November 14, 2007

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

#define MAX_NODES_FOR_TREEFILE 25000
//#define MAX_NODES_FOR_TREEFILE 6500

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


static void print_table1_top(
		std::ostream &fp);
static void print_table1_bottom(
		std::ostream &fp);
static void print_table_top(
		std::ostream &fp, int f_permutation_degree_is_small);
static void print_table_bottom(
		std::ostream &fp);
static void print_set_special(
		std::ostream &fp, long int *set, int sz);

void poset_classification::draw_poset_fname_base_aux_poset(
		std::string &fname, int depth)
{

	fname = problem_label_with_path + "_aux_poset_lvl_" + std::to_string(depth);
}

void poset_classification::draw_poset_fname_base_poset_lvl(
		std::string &fname, int depth)
{

	fname = problem_label_with_path + "_poset_lvl_" + std::to_string(depth);
}

void poset_classification::draw_poset_fname_base_tree_lvl(
		std::string &fname, int depth)
{
	fname = problem_label_with_path + "_tree_lvl_" + std::to_string(depth);

}

void poset_classification::draw_poset_fname_base_poset_detailed_lvl(
		std::string &fname, int depth)
{
	fname = problem_label_with_path + "_poset_detailed_lvl_" + std::to_string(depth);

}

void poset_classification::draw_poset_fname_aux_poset(
		std::string &fname, int depth)
{
	draw_poset_fname_base_aux_poset(fname, depth);
	fname += ".layered_graph";
}

void poset_classification::draw_poset_fname_poset(
		std::string &fname, int depth)
{
	draw_poset_fname_base_poset_lvl(fname, depth);
	fname += ".layered_graph";
}

void poset_classification::draw_poset_fname_tree(
		std::string &fname, int depth)
{
	draw_poset_fname_base_tree_lvl(fname, depth);
	fname += ".layered_graph";
}

void poset_classification::draw_poset_fname_poset_detailed(
		std::string &fname, int depth)
{
	draw_poset_fname_base_poset_detailed_lvl(fname, depth);
	fname += ".layered_graph";
}


void poset_classification::write_treefile(
		std::string &fname_base, int lvl,
		other::graphics::layered_graph_draw_options *draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::write_treefile "
				"verbose_level=" << verbose_level << endl;
	}
	if (write_treefile(
			fname_base, lvl, verbose_level)) {
#if 0
		if (f_v) {
			cout << "poset_classification::write_treefile "
					"before draw_tree" << endl;
		}
		draw_tree(fname_base, lvl, xmax, ymax, rad, f_embedded,
				verbose_level);
#endif
	}
	if (f_v) {
		cout << "poset_classification::write_treefile done" << endl;
	}
}

int poset_classification::write_treefile(
		std::string &fname_base,
		int lvl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	string fname1;
	int i, level;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "poset_classification::write_treefile" << endl;
	}
	fname1 = fname_base + "_" + std::to_string(lvl) + ".tree";
	
	if  (Poo->first_node_at_level(lvl + 1) < MAX_NODES_FOR_TREEFILE) {
		{
			if (f_vv) {
				cout << "poset_classification::write_treefile "
						"writing treefile " << fname1 << endl;
			}
			ofstream f(fname1);
			
			f << "# " << lvl << endl;

			if (f_base_case) {
				level = 0; // starter_size;
			}
			else {
				level = 0;
			}
			for (i = Poo->first_node_at_level(level);
					i < Poo->first_node_at_level(level + 1); i++) {
				if (f_vv) {
					cout << "poset_classification::write_treefile "
							"node " << i << ":" << endl;
				}
				Poo->log_nodes_for_treefile(level, i, f,
						true /* f_recurse */, verbose_level);
			}

			f << "-1 " << Poo->first_node_at_level(lvl + 1) << endl;
		}
		if (f_vv) {
			cout << "written file " << fname1
					<< " of size " << Fio.file_size(fname1) << endl;
		}
		if (f_v) {
			cout << "poset_classification::write_treefile done" << endl;
		}
		return true;
	}
	else {
		cout << "poset_classification::write_treefile too many nodes, "
				"you may increase MAX_NODES_FOR_TREEFILE if you wish" << endl;
		cout << "MAX_NODES_FOR_TREEFILE=" << MAX_NODES_FOR_TREEFILE << endl;
		cout << "first_poset_orbit_node_at_level[lvl + 1]="
				<< Poo->first_node_at_level(lvl + 1) << endl;
		cout << "lvl=" << lvl << endl;
		return false;
	}
}

void poset_classification::draw_tree(
		std::string &fname_base, int lvl,
		other::graphics::tree_draw_options *Tree_draw_options,
		other::graphics::layered_graph_draw_options *Draw_options,
	int xmax, int ymax, int rad, int f_embedded,
	int f_sideways, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	string fname;
	string fname1;
	other::graphics::tree T;
	int idx = 0;
	int nb_nodes, i;
	int *coord_xyw;
	int *perm;
	int *perm_inv;
	int f_draw_points = true;
	int f_draw_extension_points = false;
	int f_draw_aut_group_order = false;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "poset_classification::draw_tree" << endl;
	}
	fname = fname_base + "_" + std::to_string(lvl);

	fname1 = fname_base + "_" + std::to_string(lvl) + ".tree";


	if (Fio.file_size(fname1)) {
		if (f_vv) {
			cout << "reading treefile" << endl;
		}
		Tree_draw_options->f_file = true;
		Tree_draw_options->file_name.assign(fname1);

		T.init(
				Tree_draw_options, xmax, ymax, verbose_level - 1);
			
		nb_nodes = T.nb_nodes;
		if (f_vv) {
			cout << "poset_classification::draw_tree read treefile "
					<< fname1 << " with " << nb_nodes
					<< " nodes" << endl;
			cout << "poset_classification::draw_tree "
					"first_poset_orbit_node_at_level"
					"level[lvl + 1] "
					<< Poo->first_node_at_level(lvl + 1)
					<< " nodes" << endl;
		}
		if (nb_nodes != Poo->first_node_at_level(lvl + 1)) {
			cout << "poset_classification::draw_tree nb_nodes != "
					"first_poset_orbit_node_at_level"
					"[lvl + 1]" << endl;
			cout << "nb_nodes=" << nb_nodes << endl;
			cout << "first_poset_orbit_node_at_level[lvl + 1]="
					<< Poo->first_node_at_level(lvl + 1) << endl;
			exit(1);
		}
		if (nb_nodes > 100) {
			f_draw_points = false;
			f_draw_aut_group_order = false;
		}
			
		coord_xyw = NEW_int(3 * nb_nodes);
			
		if (f_vv) {
			cout << "poset_classification::draw_tree "
					"calling get_coordinates" << endl;
		}
		T.root->get_coordinates_and_width(idx, coord_xyw);

#if 0
		for (i = 0; i < nb_nodes; i++) {
			coord_xyw[i * 3 + 2] = (int)sqrt((double)coord_xyw[i * 3 + 2]);
		}
#endif

		if (false) {
			cout << "poset_classification::draw_tree "
					"coord_xyw:" << endl;
			for (i = 0; i < nb_nodes; i++) {
				cout << i << " : (" 
					<< coord_xyw[i * 3 + 0] << ","
					<< coord_xyw[i * 3 + 1] << ","
					<< coord_xyw[i * 3 + 2] << ")" << endl;
			}
		}
		
		if (f_vv) {	
			cout << "poset_classification::draw_tree calling "
					"poset_orbit_node_depth_"
					"breadth_perm_and_inverse" << endl;
		}
		Poo->poset_orbit_node_depth_breadth_perm_and_inverse(
				lvl /* max_depth */,
			perm, perm_inv, verbose_level);
		if (false) {
			cout << "poset_classification::draw_tree "
					"depth_breadth_perm_and_inverse:" << endl;
			for (i = 0; i < nb_nodes; i++) {
				cout << i << " : (" 
					<< perm[i] << "," 
					<< perm_inv[i] << ")" << endl;
			}
		}
		
		if (f_vv) {	
			cout << "poset_classification::draw_tree "
					"before draw_tree_low_level" << endl;
		}
		draw_tree_low_level(
				fname, Tree_draw_options, Draw_options,
				nb_nodes,
			coord_xyw, perm_inv, perm, 
			f_draw_points, f_draw_extension_points,
			f_draw_aut_group_order,
			xmax, ymax, rad, f_embedded, f_sideways,
			0 /*verbose_level - 2*/);
			
		FREE_int(coord_xyw);
		FREE_int(perm);
		FREE_int(perm_inv);
	}
	else {
		cout << "poset_classification::draw_tree the file " << fname1
				<< " does not exist, cannot draw the tree" << endl;
	}
}

void poset_classification::draw_tree_low_level(
		std::string &fname,
		other::graphics::tree_draw_options *Tree_draw_options,
		other::graphics::layered_graph_draw_options *Draw_options,
		int nb_nodes,
	int *coord_xyw, int *perm, int *perm_inv, 
	int f_draw_points, int f_draw_extension_points,
	int f_draw_aut_group_order,
	int xmax, int ymax, int rad, int f_embedded,
	int f_sideways,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int factor_1000 = 1000;
	string fname_full;
	other::orbiter_kernel_system::file_io Fio;
	
	if (xmax == -1)
		xmax = 2000;
	if (ymax == -1)
		ymax = 3000;
	if (ymax == 0)
		ymax = 3000;

	fname_full = fname + ".mp";

	if (f_v) {
		cout << "poset_classification::draw_tree_low_level "
				"xmax = " << xmax << " ymax = " << ymax
				<< " fname=" << fname_full << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}
	{

		other::graphics::mp_graphics G;

		G.init(fname, Draw_options, verbose_level - 1);

		G.header();
		G.begin_figure(factor_1000);

		draw_tree_low_level1(
				G,
				nb_nodes, coord_xyw, perm, perm_inv,
			f_draw_points, f_draw_extension_points,
			f_draw_aut_group_order, rad,
			0 /*verbose_level - 1*/);

	
		//G.draw_boxes_final();
		//G.end_figure();
		//G.footer();
	
		G.finish(cout, verbose_level);
	}
	if (f_v) {
		cout << "poset_classification::draw_tree_low_level "
				"written file " << fname_full
				<< " of size " << Fio.file_size(fname_full) << endl;
	}
	
}

void poset_classification::draw_tree_low_level1(
		other::graphics::mp_graphics &G,
	int nb_nodes,
	int *coords, int *perm, int *perm_inv, 
	int f_draw_points, int f_draw_extension_points,
	int f_draw_aut_group_order,
	int radius, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int *Px, *Py, *Width;
	int *Qx, *Qy;
	int x, y;
	int i, j;
	int y_offset2 = 200;
	int nb_e, Nb_e, pt, dx, dx0, nxt, hdl, depth, hdl2;
	int rad = 200 >> 3;
	other::data_structures::sorting Sorting;
	int max_set_size;
	long int *set0;
	long int *set1;
	

	max_set_size = Poo->get_max_set_size();

	Px = NEW_int(nb_nodes);
	Py = NEW_int(nb_nodes);
	Width = NEW_int(nb_nodes);
	Qx = NEW_int(100);
	Qy = NEW_int(100);
	set0 = NEW_lint(max_set_size);
	set1 = NEW_lint(max_set_size);
	
	if (f_v) {	
		cout << "poset_classification::draw_tree_low_level1" << endl;
		cout << "nb_nodes = " << nb_nodes << endl;
		cout << "rad = " << rad << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}
	for (i = 0; i < nb_nodes; i++) {
		Px[i] = coords[i * 3 + 0];
		Py[i] = coords[i * 3 + 1];
		Width[i] = coords[i * 3 + 2];
	}
	
	for (i = 0; i < nb_nodes; i++) {
		if (f_vv) {
			cout << "poset_classification::draw_tree_low_level1 i=" << i << endl;
		}
		nb_e = Poo->node_get_nb_of_extensions(i);
		if (f_vv) {
			cout << "draw_tree_low_level1: nb_e=" << nb_e << endl;
		}
		if (nb_e) {
			dx = MINIMUM(Width[i] / nb_e, 200);
			dx0 = ((nb_e - 1) * dx) >> 1;
		}
		else {
			dx = 0;
			dx0 = 0;
		}
		for (j = 0; j < nb_e; j++) {
			Nb_e = Poo->node_get_nb_of_extensions(j);
			if (f_vv) {
				cout << "poset_classification::draw_tree_low_level1 "
						"i=" << i
						<< " j=" << j << " nb_e=" << nb_e <<
					" Nb_e=" << Nb_e << endl;
				cout << "root[i].get_E(j)->get_type()=" << Poo->get_node(i)->get_E(j)->get_type() << endl;
			}

#if 0
			if (Nb_e) {
				Dx = MINIMUM(Width[j] / Nb_e, 200);
				//Dx0 = ((Nb_e - 1) * Dx) >> 1;
			}
			else {
				Dx = 0;
				//Dx0 = 0;
			}
#endif

			if (Poo->get_node(i)->get_E(j)->get_type() == EXTENSION_TYPE_EXTENSION) {
				// extension node
				pt = Poo->get_node(i)->get_E(j)->get_pt();
				nxt = Poo->get_node(i)->get_E(j)->get_data();
				if (f_vv) {
					cout << "poset_classification::draw_tree_low_level1 "
							"extension node: pt=" << pt
							<< " nxt=" << nxt << endl;
				}
				if (nxt >= 0) {
					Qx[0] = Px[perm[i]];
					Qy[0] = Py[perm[i]];
				
					Qx[1] = Px[perm[nxt]];
					Qy[1] = Py[perm[nxt]];
					if (false) {
						cout << "poset_classification::draw_tree_low_level1 "
								"Qx[0]=" << Qx[0] << " Qy[0]=" << Qy[0] << endl;
						cout << "poset_classification::draw_tree_low_level1 "
								"Qx[1]=" << Qx[1] << " Qy[1]=" << Qy[1] << endl;
					}
				
					if (f_draw_points) {
						Qx[0] -= dx0;
						Qx[0] += j * dx;
						Qy[0] -= y_offset2;
				
						Qx[1] += 0;
						Qy[1] += y_offset2;
					}
					if (false) {
						cout << "poset_classification::draw_tree_low_level1 "
								"Qx[0]=" << Qx[0] << " Qy[0]=" << Qy[0] << endl;
						cout << "poset_classification::draw_tree_low_level1 "
								"Qx[1]=" << Qx[1] << " Qy[1]=" << Qy[1] << endl;
					}
				
					G.polygon2(Qx, Qy, 0, 1);
					if (false) {
						cout << "poset_classification::draw_tree_low_level1 "
								"after G.polygon2" << endl;
					}
				}
			}
			else if (Poo->get_node(i)->get_E(j)->get_type() == EXTENSION_TYPE_FUSION) {
				// fusion node
				if (f_vv) {
					cout << "poset_classification::draw_tree_low_level1 "
							"fusion node" << endl;
				}
				if (true /*root[i].E[j].get_pt() > root[i].get_pt()*/) {
					pt = Poo->get_node(i)->get_E(j)->get_pt();
					hdl = Poo->get_node(i)->get_E(j)->get_data();
					depth = Poo->get_node(i)->depth_of_node(this);
					Poo->get_node(i)->store_set_to(this, depth - 1, set0);
					set0[depth] = pt;
					if (f_vvv) {
						cout << "poset_classification::draw_tree_low_level1 "
								"fusion node i=" << i
								<< " j=" << j << " set = ";
						Lint_vec_print(cout, set0, depth + 1);
						cout << endl;
					}
				
					//Poset->A2->element_retrieve(hdl, Elt1, false);
	
					Poset->A2->map_a_set_based_on_hdl(set0, set1, depth + 1, Poset->A, hdl, 0);

					Sorting.lint_vec_heapsort(set1, depth + 1);
				
					if (f_vvv) {
						cout << "poset_classification::draw_tree_low_level1 "
								"mapping the set to = ";
						Lint_vec_print(cout, set1, depth + 1);
						cout << endl;
					}

					hdl2 = find_poset_orbit_node_for_set(depth + 1, set1,
							false /* f_tolerant */,
							0 /* verbose_level */);
					if (hdl2 >= 0) {
						if (f_vvv) {
							cout << "poset_classification::draw_tree_low_level1 "
									"which is node " << hdl2 << endl;
						}

						Qx[0] = Px[perm[i]];
						Qy[0] = Py[perm[i]];
						Qx[1] = Px[perm[i]] - dx0 + j * dx;
						Qy[1] = Py[perm[i]] - y_offset2 - 30;
					
						Qx[2] = Px[perm[hdl2]];
						Qy[2] = Py[perm[hdl2]] + y_offset2 + 30;
						Qx[3] = Px[perm[hdl2]];
						Qy[3] = Py[perm[hdl2]];
				
						if (f_draw_points) {
							Qx[0] -= dx0;
							Qx[0] += j * dx;
							Qy[0] -= y_offset2;
							Qx[3] += 0;
							Qy[3] += y_offset2;
						}
				
						G.sl_udsty(100);
						if (f_draw_points) {
							G.bezier2(Qx, Qy, 1, 2);
							//G.bezier4(Qx, Qy, 0, 1, 2, 3);
							//G.polygon4(Qx, Qy, 0, 1, 2, 3);
						}
						G.sl_udsty(0);
					} // if (hdl2 >= 0)
				} // if (root[i].E[j].get_pt() > root[i].get_pt())
			} // if fusion node
		} // next j
	} // next i

#if 0
	for (i = 0; i < nb_nodes; i++) {
		Qx[0] = Px[perm[i]];
		Qy[0] = Py[perm[i]];
		Qx[1] = Px[perm[i]];
		Qy[1] = Py[perm[i]];
		Qx[2] = Px[perm[i]];
		Qy[2] = Py[perm[i]];
		Qx[3] = Px[perm[i]];
		Qy[3] = Py[perm[i]];
		if (i >= first_poset_orbit_node_node_at_level[sz]) {
			Qx[0] -= 15 * delta_x;
			Qx[1] += 15 * delta_x;
			Qx[2] += 15 * delta_x;
			Qx[3] -= 15 * delta_x;
			Qy[0] -= 12 * delta_y;
			Qy[1] -= 12 * delta_y;
			Qy[2] -= 20 * delta_y;
			Qy[3] -= 20 * delta_y;
		}
		else {
			Qx[0] -= 15 * delta_x;
			Qx[1] += 15 * delta_x;
			Qx[2] += 15 * delta_x;
			Qx[3] -= 15 * delta_x;
			Qy[0] -= 12 * delta_y;
			Qy[1] -= 12 * delta_y;
			Qy[2] -= 25 * delta_y;
			Qy[3] -= 25 * delta_y;
		}
		G.polygon5(Qx, Qy, 0, 1, 2, 3, 0);
	}
#endif

	if (f_v) {
		cout << "poset_classification::draw_tree_low_level1 "
				"now drawing node labels" << endl;
	}
	
	G.sf_interior(100 /* fill_interior */);
	G.sf_color(1 /* fill_color */);
	

	string s;

	for (i = 0; i < nb_nodes; i++) {
		if (i == 0) {
			continue;
		}
		pt = Poo->get_node(perm_inv[i])->get_pt();
		s = std::to_string(pt);
		//G.aligned_text(Px, Py, i, "", str);
		if (f_draw_points) {
			G.circle_text(Px[i], Py[i], rad, s);
		}
		else {
			G.circle(Px[i], Py[i], rad);
		}
	}
		

	if (f_draw_points) {
		if (f_v) {
			cout << "poset_classification::draw_tree_low_level1 "
					"now drawing connections" << endl;
		}
		for (i = 0; i < nb_nodes; i++) {
			nb_e = Poo->node_get_nb_of_extensions(i);
			if (nb_e) {
				dx = MINIMUM(Width[i] / nb_e, 200);
				dx0 = ((nb_e - 1) * dx) >> 1;
			}
			else {
				dx = 0;
				dx0 = 0;
			}

			x = Px[perm[i]];
			y = Py[perm[i]] + y_offset2;
			G.circle(x, y, rad);

			for (j = 0; j < nb_e; j++) {
				pt = Poo->get_node(i)->get_E(j)->get_pt();
				x = Px[perm[i]] - dx0 + j * dx;
				y = Py[perm[i]] - y_offset2;
				if (f_draw_extension_points) {
					s = std::to_string(pt);
					G.circle_text(x, y, rad, s);
					//G.aligned_text_with_offset(Px, Py,
					// perm[i], x_offset, y_offset, "", "$48$");
				}
				else {
					G.circle(x, y, rad);
				}
			}
		}
	}
	
	FREE_int(Px);
	FREE_int(Py);
	FREE_int(Width);
	FREE_int(Qx);
	FREE_int(Qy);
	FREE_lint(set0);
	FREE_lint(set1);
	if (f_v) {
		cout << "poset_classification::draw_tree_low_level1 done" << endl;
	}
}

void poset_classification::draw_poset_full(
		std::string &fname_base,
		int depth, int data,
		other::graphics::layered_graph_draw_options *LG_Draw_options,
		double x_stretch,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::graph_theory::layered_graph *LG;

	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"fname_base=" << fname_base << " data=" << data << endl;
	}
	make_full_poset_graph(
			depth, LG, data, x_stretch, verbose_level);
	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"after make_full_poset_graph" << endl;
	}

	string fname1;
	string fname2;

	fname1 = fname_base + "_poset_full_lvl_" + std::to_string(depth) + ".layered_graph";
	
	LG->write_file(fname1, 0 /*verbose_level*/);
	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"after LG->write_file" << endl;
	}

	fname2 = fname_base + "_poset_full_lvl_" + std::to_string(depth);

	LG->draw_with_options(
			fname2, LG_Draw_options,
			0 /* verbose_level */);

	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"after LG->draw" << endl;
	}

	FREE_OBJECT(LG);
	
	if (f_v) {
		cout << "poset_classification::draw_poset_full done" << endl;
	}
}

void poset_classification::draw_poset(
		std::string &fname_base,
		int depth, int data,
		other::graphics::layered_graph_draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::graph_theory::layered_graph *LG1;
	combinatorics::graph_theory::layered_graph *LG2;
	combinatorics::graph_theory::layered_graph *LG3;
	combinatorics::graph_theory::layered_graph *LG4;

	if (f_v) {
		cout << "poset_classification::draw_poset "
				"data=" << data << " fname_base=" << fname_base << endl;
	}


	if (f_v) {
		cout << "poset_classification::draw_poset "
				"before make_auxiliary_graph" << endl;
	}
	make_auxiliary_graph(
			depth, LG1, data,
			0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "poset_classification::draw_poset "
				"before make_graph" << endl;
	}
	make_graph(
			depth, LG2, data, false /* f_tree */,
			0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "poset_classification::draw_poset "
				"before make_graph" << endl;
	}
	make_graph(
			depth, LG3, data, true /* f_tree */,
			0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "poset_classification::draw_poset "
				"before make_poset_graph_detailed" << endl;
	}
	make_poset_graph_detailed(
			LG4, data, depth,
			0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "poset_classification::draw_poset "
				"after make_poset_graph_detailed" << endl;
	}

	string fname_base1;
	string fname_base2;
	string fname_base3;
	string fname_base4;
	string fname1;
	string fname2;
	string fname3;
	string fname4;

	draw_poset_fname_base_aux_poset(fname_base1, depth);
	draw_poset_fname_base_poset_lvl(fname_base2, depth);
	draw_poset_fname_base_tree_lvl(fname_base3, depth);
	draw_poset_fname_base_poset_detailed_lvl(fname_base4, depth);

	fname1 = fname_base1 + ".layered_graph";
	fname2 = fname_base2 + ".layered_graph";
	fname3 = fname_base3 + ".layered_graph";
	fname4 = fname_base4 + ".layered_graph";

	if (f_v) {
		cout << "poset_classification::draw_poset "
				"writing file " << fname1 << endl;
	}


	
	LG1->write_file(fname1, 0 /*verbose_level*/);
	LG1->draw_with_options(
			fname_base1, LG_Draw_options,
			0 /* verbose_level */);

	if (f_v) {
		cout << "poset_classification::draw_poset "
				"writing file " << fname2 << endl;
	}

	LG2->write_file(fname2, 0 /*verbose_level*/);
	LG2->draw_with_options(
			fname_base2, LG_Draw_options,
			0 /* verbose_level */);

	if (f_v) {
		cout << "poset_classification::draw_poset "
				"writing file " << fname3 << endl;
	}

	LG3->write_file(fname3, 0 /*verbose_level*/);
	LG3->draw_with_options(
			fname_base3, LG_Draw_options,
			0 /* verbose_level */);

	if (f_v) {
		cout << "poset_classification::draw_poset "
				"writing file " << fname4 << endl;
	}

	LG4->write_file(fname4, 0 /*verbose_level*/);
	LG4->draw_with_options(
			fname_base4, LG_Draw_options,
			0 /* verbose_level */);

	FREE_OBJECT(LG1);
	FREE_OBJECT(LG2);
	FREE_OBJECT(LG3);
	FREE_OBJECT(LG4);
	
	if (f_v) {
		cout << "poset_classification::draw_poset done" << endl;
	}
}

void poset_classification::draw_level_graph(
		std::string &fname_base,
		int depth, int data, int level,
		other::graphics::layered_graph_draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::graph_theory::layered_graph *LG;

	if (f_v) {
		cout << "poset_classification::draw_level_graph "
				"data=" << data << endl;
	}


	make_level_graph(depth, LG, data, level, verbose_level - 1);


	string fname_base1;
	string fname;

	fname_base1 = "_lvl_" + std::to_string(depth) + "_bipartite_lvl_" + std::to_string(level);

	fname = fname_base + "_lvl_" + std::to_string(depth) + "_bipartite_lvl_" + std::to_string(level) + ".layered_graph";

	LG->write_file(fname, 0 /*verbose_level*/);

	
	LG->draw_with_options(
			fname_base1, LG_Draw_options, 0 /* verbose_level */);

	FREE_OBJECT(LG);

	if (f_v) {
		cout << "poset_classification::draw_level_graph done" << endl;
	}
}


void poset_classification::make_flag_orbits_on_relations(
		int depth,
		std::string &fname_prefix, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v5 = (verbose_level >= 5);
	int nb_layers;
	int *Nb_elements;
	int *Fst;
	int *Nb_orbits;
	int **Fst_element_per_orbit;
	int **Orbit_len;
	int i, j, lvl, po, po2, so, n1, n2, ol1, ol2, el1, el2, h;
	long int *set;
	long int *set1;
	long int *set2;
	int f_contained;
	//longinteger_domain D;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "poset_classification::make_flag_orbits_on_relations" << endl;
	}
	set = NEW_lint(depth + 1);
	set1 = NEW_lint(depth + 1);
	set2 = NEW_lint(depth + 1);
	nb_layers = depth + 1;
	Nb_elements = NEW_int(nb_layers);
	Nb_orbits = NEW_int(nb_layers);
	Fst = NEW_int(nb_layers + 1);
	Fst_element_per_orbit = NEW_pint(nb_layers);
	Orbit_len = NEW_pint(nb_layers);

	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {
		Nb_orbits[i] = nb_orbits_at_level(i);
		Fst_element_per_orbit[i] = NEW_int(Nb_orbits[i] + 1);
		Orbit_len[i] = NEW_int(Nb_orbits[i]);
		Nb_elements[i] = 0;

		Fst_element_per_orbit[i][0] = 0;
		for (j = 0; j < Nb_orbits[i]; j++) {
			Orbit_len[i][j] = orbit_length_as_int(j, i);
			Nb_elements[i] += Orbit_len[i][j];
			Fst_element_per_orbit[i][j + 1] =
					Fst_element_per_orbit[i][j] + Orbit_len[i][j];
		}
		Fst[i + 1] = Fst[i] + Nb_elements[i];
	}

	for (lvl = 0; lvl <= depth; lvl++) {
		string fname;
		other::orbiter_kernel_system::file_io Fio;

		fname = fname_prefix + "_depth_" + std::to_string(lvl) + "_orbit_lengths.csv";

		string label;

		label.assign("Orbit_length");
		Fio.Csv_file_support->int_vec_write_csv(
				Orbit_len[lvl], Nb_orbits[lvl],
			fname, label);

		cout << "poset_classification::make_flag_orbits_on_relations "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	for (lvl = 0; lvl < depth; lvl++) {
		if (f_vv) {
			cout << "poset_classification::make_flag_orbits_on_relations "
					"adding edges lvl=" << lvl << " / " << depth << endl;
			}
		//f = 0;

		int *F;
		int flag_orbit_idx;
		string fname;

		if (f_vv) {
			cout << "poset_classification::make_flag_orbits_on_relations allocating F" << endl;
		}
		F = NEW_int(Nb_elements[lvl] * Nb_elements[lvl + 1]);
		Int_vec_zero(F, Nb_elements[lvl] * Nb_elements[lvl + 1]);

		fname = fname_prefix + "_depth_" + std::to_string(lvl) + ".csv";

		flag_orbit_idx = 1;
		for (po = 0; po < nb_orbits_at_level(lvl); po++) {

			if (f_vv) {
				cout << "poset_classification::make_flag_orbits_on_relations "
						"adding edges lvl=" << lvl
						<< " po=" << po << " / " << nb_orbits_at_level(lvl)
						<< " Fst_element_per_orbit[lvl][po]="
						<< Fst_element_per_orbit[lvl][po] << endl;
			}

			ol1 = Orbit_len[lvl][po];
			//
			n1 = Poo->first_node_at_level(lvl) + po;


			int *Down_orbits;
			int nb_down_orbits;

			Down_orbits = NEW_int(Poo->node_get_nb_of_extensions(n1));
			nb_down_orbits = 0;

			for (so = 0; so < Poo->node_get_nb_of_extensions(n1); so++) {

				if (f_vv) {
					cout << "poset_classification::make_flag_orbits_on_relations "
							"adding edges lvl=" << lvl
							<< " po=" << po << " / " << nb_orbits_at_level(lvl)
							<< " so=" << so << " / " << Poo->node_get_nb_of_extensions(n1)
							<< endl;
				}


				extension *E = Poo->get_node(n1)->get_E(so);
				if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
					//cout << "extension node" << endl;
					n2 = E->get_data();

					Down_orbits[nb_down_orbits++] = n2;
				}
				else if (E->get_type() == EXTENSION_TYPE_FUSION) {
					//cout << "fusion node" << endl;
					// po = data1
					// so = data2
					int n0, so0;
					n0 = E->get_data1();
					so0 = E->get_data2();
					//cout << "fusion (" << n1 << "/" << so << ") "
					//"-> (" << n0 << "/" << so0 << ")" << endl;
					extension *E0;
					E0 = Poo->get_node(n0)->get_E(so0);
					if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
						cout << "warning: fusion node does not point "
								"to extension node" << endl;
						cout << "type = ";
						print_extension_type(cout, E0->get_type());
						cout << endl;
						exit(1);
					}
					n2 = E0->get_data();
					Down_orbits[nb_down_orbits++] = n2;
				}

			} // next so


			if (f_vv) {
				cout << "poset_classification::make_flag_orbits_on_relations adding edges "
						"lvl=" << lvl
						<< " po=" << po << " / " << nb_orbits_at_level(lvl)
						<< " so=" << so << " / " << Poo->node_get_nb_of_extensions(n1)
						<< " downorbits = ";
				Int_vec_print(cout, Down_orbits, nb_down_orbits);
				cout << endl;
			}

			Sorting.int_vec_sort_and_remove_duplicates(Down_orbits, nb_down_orbits);
			if (f_vv) {
				cout << "poset_classification::make_flag_orbits_on_relations adding edges "
						"lvl=" << lvl << " po=" << po
						<< " so=" << so << " unique downorbits = ";
				Int_vec_print(cout, Down_orbits, nb_down_orbits);
				cout << endl;
			}

			for (h = 0; h < nb_down_orbits; h++, flag_orbit_idx++) {
				n2 = Down_orbits[h];
				po2 = n2 - Poo->first_node_at_level(lvl + 1);
				ol2 = Orbit_len[lvl + 1][po2];
				if (f_v5) {
					cout << "poset_classification::make_flag_orbits_on_relations "
							"adding edges lvl=" << lvl
							<< " po=" << po << " / " << nb_orbits_at_level(lvl)
							<< " so=" << so << " / " << Poo->node_get_nb_of_extensions(n1)
							<< " downorbit = " << h << " / " << nb_down_orbits
							<< " n1=" << n1 << " n2=" << n2
							<< " po2=" << po2
							<< " ol1=" << ol1 << " ol2=" << ol2
							<< " Fst_element_per_orbit[lvl][po]="
							<< Fst_element_per_orbit[lvl][po]
							<< " Fst_element_per_orbit[lvl + 1][po2]="
							<< Fst_element_per_orbit[lvl + 1][po2] << endl;
				}
				for (el1 = 0; el1 < ol1; el1++) {
					if (f_v5) {
						cout << "unrank " << lvl << ", " << po
								<< ", " << el1 << endl;
					}
					orbit_element_unrank(lvl, po, el1, set1,
							0 /* verbose_level */);
					if (f_v5) {
						cout << "set1=";
						Lint_vec_print(cout, set1, lvl);
						cout << endl;
					}


					for (el2 = 0; el2 < ol2; el2++) {
						if (f_v5) {
							cout << "unrank " << lvl + 1 << ", "
									<< po2 << ", " << el2 << endl;
						}
						orbit_element_unrank(lvl + 1, po2, el2, set2,
								0 /* verbose_level */);
						if (f_v5) {
							cout << "set2=";
							Lint_vec_print(cout, set2, lvl + 1);
							cout << endl;
						}

						if (f_v5) {
							cout << "poset_classification::make_flag_orbits_on_relations "
									"adding edges lvl=" << lvl
									<< " po=" << po << " so=" << so
									<< " downorbit = " << h << " / "
									<< nb_down_orbits << " n1=" << n1
									<< " n2=" << n2 << " po2=" << po2
									<< " ol1=" << ol1 << " ol2=" << ol2
									<< " el1=" << el1 << " el2=" << el2
									<< endl;
							cout << "set1=";
							Lint_vec_print(cout, set1, lvl);
							cout << endl;
							cout << "set2=";
							Lint_vec_print(cout, set2, lvl + 1);
							cout << endl;
						}


						Lint_vec_copy(set1, set, lvl);

						//f_contained = int_vec_sort_and_test_if_contained(
						// set, lvl, set2, lvl + 1);
						f_contained = poset_structure_is_contained(
								set, lvl, set2, lvl + 1,
								0 /* verbose_level*/);


						if (f_contained) {
							if (f_v5) {
								cout << "is contained" << endl;
							}

#if 0
							LG->add_edge(lvl,
								Fst_element_per_orbit[lvl][po] + el1,
								lvl + 1,
								Fst_element_per_orbit[lvl + 1][po2] + el2,
								0 /*verbose_level*/);
#else
							F[(Fst_element_per_orbit[lvl][po] + el1) * Nb_elements[lvl + 1] + Fst_element_per_orbit[lvl + 1][po2] + el2] = flag_orbit_idx;
#endif
						}
						else {
							if (f_v5) {
								cout << "is NOT contained" << endl;
							}
						}

					} // next el2
				} // next el1
			} // next h


			FREE_int(Down_orbits);

		} // po

		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->int_matrix_write_csv(
				fname,
				F, Nb_elements[lvl], Nb_elements[lvl + 1]);
		FREE_int(F);

		cout << "poset_classification::make_flag_orbits_on_relations "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

	} // lvl






	FREE_lint(set);
	FREE_lint(set1);
	FREE_lint(set2);
	FREE_int(Nb_elements);
	FREE_int(Nb_orbits);
	FREE_int(Fst);
	for (i = 0; i <= depth; i++) {
		FREE_int(Fst_element_per_orbit[i]);
	}
	FREE_pint(Fst_element_per_orbit);
	for (i = 0; i <= depth; i++) {
		FREE_int(Orbit_len[i]);
	}
	FREE_pint(Orbit_len);
	if (f_v) {
		cout << "poset_classification::make_flag_orbits_on_relations done" << endl;
	}
}



void poset_classification::make_full_poset_graph(
		int depth, combinatorics::graph_theory::layered_graph *&LG,
		int data1, double x_stretch, int verbose_level)
// Draws the full poset: each element of each orbit is drawn.
// The orbits are indicated by grouping the elements closer together.
// Uses int_vec_sort_and_test_if_contained to test containment relation.
// This is only good for actions on sets, not for actions on subspaces
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_layers;
	int *Nb_elements;
	int *Fst;
	int *Nb_orbits;
	int **Fst_element_per_orbit;
	int **Orbit_len;
	int i, j, lvl, po, po2, so, n1, n2, ol1, ol2, el1, el2, h;
	long int *set;
	long int *set1;
	long int *set2;
	int f_contained;
	//longinteger_domain D;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "poset_classification::make_full_poset_graph" << endl;
	}
	set = NEW_lint(depth + 1);
	set1 = NEW_lint(depth + 1);
	set2 = NEW_lint(depth + 1);
	nb_layers = depth + 1;
	Nb_elements = NEW_int(nb_layers);
	Nb_orbits = NEW_int(nb_layers);
	Fst = NEW_int(nb_layers + 1);
	Fst_element_per_orbit = NEW_pint(nb_layers);
	Orbit_len = NEW_pint(nb_layers);

	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {
		Nb_orbits[i] = nb_orbits_at_level(i);
		Fst_element_per_orbit[i] = NEW_int(Nb_orbits[i] + 1);
		Orbit_len[i] = NEW_int(Nb_orbits[i]);
		Nb_elements[i] = 0;

		Fst_element_per_orbit[i][0] = 0;
		for (j = 0; j < Nb_orbits[i]; j++) {
			Orbit_len[i][j] = orbit_length_as_int(j, i);
			Nb_elements[i] += Orbit_len[i][j];
			Fst_element_per_orbit[i][j + 1] =
					Fst_element_per_orbit[i][j] + Orbit_len[i][j];
		}
		Fst[i + 1] = Fst[i] + Nb_elements[i];
	}

	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);
	LG->add_data1(data1, 0/*verbose_level*/);

	if (f_v) {
		cout << "poset_classification::make_full_poset_graph "
				"before LG->init" << endl;
		cout << "nb_layers=" << nb_layers << endl;
		for (lvl = 0; lvl < depth; lvl++) {
			cout << "Nb_elements[" << lvl << "]=" << Nb_elements[lvl] << endl;
		}
	}
	string dummy;
	dummy.assign("");
	LG->init(nb_layers, Nb_elements, dummy, verbose_level);

	if (f_v) {
		cout << "poset_classification::make_full_poset_graph "
				"after LG->init" << endl;
	}
	if (f_v) {
		cout << "poset_classification::make_full_poset_graph "
				"before LG->place_with_grouping" << endl;
	}
	LG->place_with_grouping(Orbit_len, Nb_orbits, x_stretch, verbose_level);
	//LG->place(verbose_level);
	if (f_v) {
		cout << "poset_classification::make_full_poset_graph "
				"after LG->place" << endl;
	}

	for (lvl = 0; lvl < depth; lvl++) {
		if (f_vv) {
			cout << "poset_classification::make_full_poset_graph "
					"adding edges lvl=" << lvl << " / " << depth << endl;
		}
		//f = 0;
		for (po = 0; po < nb_orbits_at_level(lvl); po++) {

			if (f_vv) {
				cout << "poset_classification::make_full_poset_graph "
						"adding edges lvl=" << lvl
						<< " po=" << po << " / "
						<< nb_orbits_at_level(lvl)
						<< " Fst_element_per_orbit[lvl][po]="
						<< Fst_element_per_orbit[lvl][po] << endl;
			}

			ol1 = Orbit_len[lvl][po];
			//
			n1 = Poo->first_node_at_level(lvl) + po;


			int *Down_orbits;
			int nb_down_orbits;

			Down_orbits = NEW_int(Poo->node_get_nb_of_extensions(n1));
			nb_down_orbits = 0;

			for (so = 0; so < Poo->node_get_nb_of_extensions(n1); so++) {

				if (f_vv) {
					cout << "poset_classification::make_full_poset_graph "
							"adding edges lvl=" << lvl
							<< " po=" << po << " so=" << so << endl;
				}


				extension *E = Poo->get_node(n1)->get_E(so);
				if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
					//cout << "extension node" << endl;
					n2 = E->get_data();

					Down_orbits[nb_down_orbits++] = n2;
				}
				else if (E->get_type() == EXTENSION_TYPE_FUSION) {
					//cout << "fusion node" << endl;
					// po = data1
					// so = data2
					int n0, so0;
					n0 = E->get_data1();
					so0 = E->get_data2();
					//cout << "fusion (" << n1 << "/" << so << ") "
					//"-> (" << n0 << "/" << so0 << ")" << endl;
					extension *E0;
					E0 = Poo->get_node(n0)->get_E(so0);
					if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
						cout << "warning: fusion node does not point "
								"to extension node" << endl;
						cout << "type = ";
						print_extension_type(cout, E0->get_type());
						cout << endl;
						exit(1);
					}
					n2 = E0->get_data();
					Down_orbits[nb_down_orbits++] = n2;
				}

			} // next so


			if (f_vv) {
				cout << "poset_classification::make_full_poset_graph adding edges "
						"lvl=" << lvl << " po=" << po
						<< " so=" << so << " downorbits = ";
				Int_vec_print(cout, Down_orbits, nb_down_orbits);
				cout << endl;
			}

			Sorting.int_vec_sort_and_remove_duplicates(Down_orbits, nb_down_orbits);
			if (f_vv) {
				cout << "poset_classification::make_full_poset_graph adding edges "
						"lvl=" << lvl << " po=" << po
						<< " so=" << so << " unique downorbits = ";
				Int_vec_print(cout, Down_orbits, nb_down_orbits);
				cout << endl;
			}

			for (h = 0; h < nb_down_orbits; h++) {
				n2 = Down_orbits[h];
				po2 = n2 - Poo->first_node_at_level(lvl + 1);
				ol2 = Orbit_len[lvl + 1][po2];
				if (f_vv) {
					cout << "poset_classification::make_full_poset_graph "
							"adding edges lvl=" << lvl << " po=" << po
							<< " so=" << so << " downorbit = " << h
							<< " / " << nb_down_orbits << " n1=" << n1
							<< " n2=" << n2 << " po2=" << po2
							<< " ol1=" << ol1 << " ol2=" << ol2
							<< " Fst_element_per_orbit[lvl][po]="
							<< Fst_element_per_orbit[lvl][po]
							<< " Fst_element_per_orbit[lvl + 1][po2]="
							<< Fst_element_per_orbit[lvl + 1][po2] << endl;
				}
				for (el1 = 0; el1 < ol1; el1++) {
					if (f_vv) {
						cout << "unrank " << lvl << ", " << po
								<< ", " << el1 << endl;
					}
					orbit_element_unrank(lvl, po, el1, set1,
							0 /* verbose_level */);
					if (f_vv) {
						cout << "set1=";
						Lint_vec_print(cout, set1, lvl);
						cout << endl;
					}


					for (el2 = 0; el2 < ol2; el2++) {
						if (f_vv) {
							cout << "unrank " << lvl + 1 << ", "
									<< po2 << ", " << el2 << endl;
						}
						orbit_element_unrank(lvl + 1, po2, el2, set2,
								0 /* verbose_level */);
						if (f_vv) {
							cout << "set2=";
							Lint_vec_print(cout, set2, lvl + 1);
							cout << endl;
						}

						if (f_vv) {
							cout << "poset_classification::make_full_poset_graph "
									"adding edges lvl=" << lvl
									<< " po=" << po << " so=" << so
									<< " downorbit = " << h << " / "
									<< nb_down_orbits << " n1=" << n1
									<< " n2=" << n2 << " po2=" << po2
									<< " ol1=" << ol1 << " ol2=" << ol2
									<< " el1=" << el1 << " el2=" << el2
									<< endl;
							cout << "set1=";
							Lint_vec_print(cout, set1, lvl);
							cout << endl;
							cout << "set2=";
							Lint_vec_print(cout, set2, lvl + 1);
							cout << endl;
						}
						

						Lint_vec_copy(set1, set, lvl);

						//f_contained = int_vec_sort_and_test_if_contained(
						// set, lvl, set2, lvl + 1);
						f_contained = poset_structure_is_contained(
								set, lvl, set2, lvl + 1,
								0 /* verbose_level*/);
						

						if (f_contained) {
							if (f_vv) {
								cout << "is contained" << endl;
							}
							LG->add_edge(lvl,
								Fst_element_per_orbit[lvl][po] + el1,
								lvl + 1,
								Fst_element_per_orbit[lvl + 1][po2] + el2,
								1, // edge_color
								0 /*verbose_level*/);
						}
						else {
							if (f_vv) {
								cout << "is NOT contained" << endl;
							}
						}
						
					} // next el2
				} // next el1
			} // next h


			FREE_int(Down_orbits);

		} // po
	} // lvl



	if (f_vv) {
		cout << "poset_classification::make_full_poset_graph "
				"now making vertex labels" << endl;
	}
	for (lvl = 0; lvl <= depth; lvl++) {
		if (f_vv) {
			cout << "poset_classification::make_full_poset_graph "
					"now making vertex labels lvl " << lvl
					<< " / " << depth << endl;
		}
		for (po = 0; po < nb_orbits_at_level(lvl); po++) {

			ol1 = Orbit_len[lvl][po];
			//
			n1 = Poo->first_node_at_level(lvl) + po;

			if (f_vv) {
				cout << "poset_classification::make_full_poset_graph "
						"now making vertex labels lvl " << lvl
						<< " / " << depth << " po=" << po << " / "
						<< nb_orbits_at_level(lvl)
						<< " ol1=" << ol1 << endl;
			}

			for (el1 = 0; el1 < ol1; el1++) {

				if (f_vv) {
					cout << "unrank " << lvl << ", "
							<< po << ", " << el1 << endl;
				}
				orbit_element_unrank(lvl, po, el1, set1,
						0 /* verbose_level */);
				if (f_vv) {
					cout << "set1=";
					Lint_vec_print(cout, set1, lvl);
					cout << endl;
				}

				LG->add_node_vec_data(lvl,
						Fst_element_per_orbit[lvl][po] + el1,
						set1, lvl,
						0 /* verbose_level */);
			}


		}
	}



	FREE_lint(set);
	FREE_lint(set1);
	FREE_lint(set2);
	FREE_int(Nb_elements);
	FREE_int(Nb_orbits);
	FREE_int(Fst);
	for (i = 0; i <= depth; i++) {
		FREE_int(Fst_element_per_orbit[i]);
	}
	FREE_pint(Fst_element_per_orbit);
	for (i = 0; i <= depth; i++) {
		FREE_int(Orbit_len[i]);
	}
	FREE_pint(Orbit_len);
	if (f_v) {
		cout << "poset_classification::make_full_poset_graph done" << endl;
	}
}

void poset_classification::make_auxiliary_graph(
		int depth,
		combinatorics::graph_theory::layered_graph *&LG, int data1, int verbose_level)
// makes a graph of the poset of orbits with 2 * depth + 1 layers.
// The middle layers represent the flag orbits.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 4);
	int nb_layers;
	int *Nb;
	int *Fst;
	int i, lvl, po, so, n, n1, f;
	algebra::ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "poset_classification::make_auxiliary_graph" << endl;
	}

	//print_fusion_nodes(depth);

	

	nb_layers = 2 * depth + 1;
	Nb = NEW_int(nb_layers);
	Fst = NEW_int(nb_layers);

	Fst[0] = 0;
	for (i = 0; i < depth; i++) {
		Nb[2 * i] = nb_orbits_at_level(i);
		Fst[2 * i + 1] = Fst[2 * i] + Nb[2 * i];
		count_extension_nodes_at_level(i);
		Nb[2 * i + 1] = Poo->get_nb_extension_nodes_at_level_total(i);
		Fst[2 * i + 2] = Fst[2 * i + 1] + Nb[2 * i + 1];
	}
	Nb[2 * depth] = nb_orbits_at_level(depth);
	
	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);
	if (f_vv) {
		cout << "poset_classification::make_auxiliary_graph "
				"before LG->init" << endl;
	}
	LG->add_data1(data1, 0/*verbose_level*/);

	string dummy;
	dummy.assign("");

	LG->init(nb_layers, Nb, dummy, verbose_level - 1);
	if (f_vv) {
		cout << "poset_classification::make_auxiliary_graph "
				"after LG->init" << endl;
	}
	LG->place(verbose_level - 1);
	if (f_vv) {
		cout << "poset_classification::make_auxiliary_graph "
				"after LG->place" << endl;
	}
	for (lvl = 0; lvl < depth; lvl++) {
		if (f_vv) {
			cout << "poset_classification::make_auxiliary_graph "
					"adding edges "
					"lvl=" << lvl << " / " << depth << endl;
		}
		f = 0;
		for (po = 0; po < nb_orbits_at_level(lvl); po++) {

			if (f_v3) {
				cout << "poset_classification::make_auxiliary_graph "
						"adding edges lvl=" << lvl << " po=" << po
						<< " / " << nb_orbits_at_level(lvl) << endl;
			}

			//
			n = Poo->first_node_at_level(lvl) + po;
			for (so = 0; so < Poo->node_get_nb_of_extensions(n); so++) {

				if (f_v4) {
					cout << "poset_classification::make_auxiliary_graph "
							"adding edges "
							"lvl=" << lvl << " po=" << po
							<< " so=" << so << endl;
				}
				LG->add_edge(
						2 * lvl, po, 2 * lvl + 1, f + so,
						1, // edge_color
						verbose_level - 4);

				extension *E = Poo->get_node(n)->get_E(so);
				if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
					if (f_v4) {
						cout << "extension node" << endl;
					}
					n1 = E->get_data();
					if (f_v4) {
						cout << "n1=" << n1 << endl;
					}
					LG->add_edge(
							2 * lvl + 1, f + so, 2 * lvl + 2,
							n1 - Poo->first_node_at_level(lvl + 1),
							1, // edge_color
							verbose_level - 4);
				}
				else if (E->get_type() == EXTENSION_TYPE_FUSION) {
					if (f_v4) {
						cout << "fusion node" << endl;
					}
					// po = data1
					// so = data2
					int n0, so0;
					n0 = E->get_data1();
					so0 = E->get_data2();
					if (f_v4) {
						cout << "fusion (" << n << "/" << so << ") -> ("
								<< n0 << "/" << so0 << ")" << endl;
					}
					extension *E0;
					E0 = Poo->get_node(n0)->get_E(so0);
					if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
						cout << "warning: fusion node does not point "
								"to extension node" << endl;
						cout << "type = ";
						print_extension_type(cout, E0->get_type());
						cout << endl;
						exit(1);
					}
					n1 = E0->get_data();
					if (f_v4) {
						cout << "n1=" << n1
								<< " first_poset_orbit_node_node_at_level[lvl + 1] = "
								<< Poo->first_node_at_level(lvl + 1) << endl;
					}
					LG->add_edge(
							2 * lvl + 1, f + so, 2 * lvl + 2,
							n1 - Poo->first_node_at_level(lvl + 1),
							1, // edge_color
							verbose_level - 4);
				}
			}
			
			f += Poo->node_get_nb_of_extensions(n);
		}
		if (f_vv) {
			cout << "poset_classification::make_auxiliary_graph "
					"after LG->add_edge (1)" << endl;
		}
	}


	if (f_vv) {
		cout << "poset_classification::make_auxiliary_graph "
				"now making vertex labels" << endl;
	}
	for (lvl = 0; lvl <= depth; lvl++) {
		f = 0;
		if (f_vv) {
			cout << "poset_classification::make_auxiliary_graph "
					"now making vertex "
					"labels lvl " << lvl << " / " << depth << endl;
		}
		for (po = 0; po < nb_orbits_at_level(lvl); po++) {


			if (f_v3) {
				cout << "poset_classification::make_auxiliary_graph "
						"now making vertex labels lvl " << lvl << " / "
						<< depth << " po=" << po << " / "
						<< nb_orbits_at_level(lvl) << endl;
			}


			string text1;
			string text2;
			algebra::ring_theory::longinteger_object go, go1;
			int n, so, len, r;
			
			n = Poo->first_node_at_level(lvl) + po;
			get_stabilizer_order(lvl, po, go);
			go.print_to_string(text1);
			if (lvl) {
				text2 = "$" + std::to_string(Poo->get_node(n)->get_pt()) + "_{" + text1 + "}$";
			}
			else {
				text2 = "$\\emptyset_{" + text1 + "}$";
			}

			// set label to be the automorphism group order:
			//LG->add_text(2 * lvl + 0, po, text1, 0/*verbose_level*/);

			// set label to be the pt:

			string text3;

			text3.assign(text2);
			LG->add_text(2 * lvl + 0, po, text3, 0/*verbose_level*/);


			LG->add_node_data1(
					2 * lvl + 0, po, Poo->get_node(n)->get_pt(),
					0/*verbose_level*/);
			if (lvl) {
				LG->add_node_data2(
						2 * lvl + 0, po, 2 * (lvl - 1),
						0/*verbose_level*/);
				LG->add_node_data3(
						2 * lvl + 0, po,
						Poo->get_node(n)->get_prev() - Poo->first_node_at_level(lvl - 1),
						0/*verbose_level*/);
			}
			else {
				LG->add_node_data2(
						2 * lvl + 0, po, -1, 0/*verbose_level*/);
				LG->add_node_data3(
						2 * lvl + 0, po, -1, 0/*verbose_level*/);
			}
			for (so = 0; so < Poo->node_get_nb_of_extensions(n); so++) {
				extension *E = Poo->get_node(n)->get_E(so);
				len = E->get_orbit_len();
				D.integral_division_by_int(go, len, go1, r);

				go1.print_to_string(text1);
				text2 = "$" + std::to_string(E->get_pt()) + "_{" + text1 + "}$";

				// set label to be the automorphism group order:
				//LG->add_text(2 * lvl + 1, f + so, text1, 0/*verbose_level*/);
				// set label to be the point:

				string text3;

				text3.assign(text2);
				LG->add_text(2 * lvl + 1, f + so, text3, 0/*verbose_level*/);


			}
			f += Poo->node_get_nb_of_extensions(n);
		}
	}
	FREE_int(Nb);
	FREE_int(Fst);
	
	if (f_v) {
		cout << "poset_classification::make_auxiliary_graph done" << endl;
	}
}

void poset_classification::make_graph(
		int depth,
		combinatorics::graph_theory::layered_graph *&LG,
		int data1, int f_tree, int verbose_level)
// makes a graph  of the poset of orbits with depth + 1 layers.
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int nb_layers;
	int *Nb;
	int *Fst;
	int i, lvl, po, so, n, n1;
	long int *the_set;
	//longinteger_domain D;

	if (f_v) {
		cout << "poset_classification::make_graph f_tree=" << f_tree << endl;
	}

	//print_fusion_nodes(depth);

	

	nb_layers = depth + 1;
	Nb = NEW_int(nb_layers);
	Fst = NEW_int(nb_layers);

	Fst[0] = 0;
	for (i = 0; i < depth; i++) {
		Nb[i] = nb_orbits_at_level(i);
		Fst[i + 1] = Fst[i] + Nb[i];
	}
	Nb[depth] = nb_orbits_at_level(depth);

	the_set = NEW_lint(depth);

	
	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);
	if (f_vv) {
		cout << "poset_classification::make_graph "
				"before LG->init" << endl;
	}
	LG->add_data1(data1, 0/*verbose_level*/);

	string dummy;
	dummy.assign("");

	LG->init(nb_layers, Nb, dummy, verbose_level);
	if (f_vv) {
		cout << "poset_classification::make_graph "
				"after LG->init" << endl;
	}
	LG->place(verbose_level);
	if (f_vv) {
		cout << "poset_classification::make_graph "
				"after LG->place" << endl;
	}


	// make edges:
	for (lvl = 0; lvl < depth; lvl++) {
		if (f_v) {
			cout << "poset_classification::make_graph "
					"adding edges "
					"lvl=" << lvl << " / " << depth << endl;
		}
		for (po = 0; po < nb_orbits_at_level(lvl); po++) {

			if (false /*f_v*/) {
				cout << "poset_classification::make_graph "
						"adding edges "
						"lvl=" << lvl << " po=" << po << " / "
						<< nb_orbits_at_level(lvl) << endl;
			}

			//
			n = Poo->first_node_at_level(lvl) + po;
			for (so = 0; so < Poo->node_get_nb_of_extensions(n); so++) {

				if (false /*f_v*/) {
					cout << "poset_classification::make_graph "
							"adding edges "
							"lvl=" << lvl << " po=" << po
							<< " so=" << so << endl;
				}
				//LG->add_edge(2 * lvl, po, 2 * lvl + 1,
				//f + so, 0 /*verbose_level*/);
				extension *E = Poo->get_node(n)->get_E(so);
				if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
					//cout << "extension node" << endl;
					n1 = E->get_data();
					//cout << "n1=" << n1 << endl;
					LG->add_edge(lvl, po, lvl + 1,
							n1 - Poo->first_node_at_level(lvl + 1),
							1, // edge_color
							0 /*verbose_level*/);
				}

				if (!f_tree) {
					if (E->get_type() == EXTENSION_TYPE_FUSION) {
						//cout << "fusion node" << endl;
						// po = data1
						// so = data2
						int n0, so0;
						n0 = E->get_data1();
						so0 = E->get_data2();
						//cout << "fusion (" << n << "/" << so << ") -> ("
						//<< n0 << "/" << so0 << ")" << endl;
						extension *E0;
						E0 = Poo->get_node(n0)->get_E(so0);
						if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
							cout << "warning: fusion node does not point to "
									"extension node" << endl;
							cout << "type = ";
							print_extension_type(cout, E0->get_type());
							cout << endl;
							exit(1);
						}
						n1 = E0->get_data();
						//cout << "n1=" << n1
						//<< " first_poset_orbit_node_at_level[lvl + 1] = "
						//<< first_poset_orbit_node_at_level[lvl + 1] << endl;
						LG->add_edge(lvl, po, lvl + 1,
								n1 - Poo->first_node_at_level(lvl + 1),
								1, // edge_color
								0 /*verbose_level*/);
					}
				}
			}
		}
		if (f_vv) {
			cout << "poset_classification::make_graph "
					"after LG->add_edge (1)" << endl;
		}
	}


	// create vertex labels:
	if (f_vv) {
		cout << "poset_classification::make_graph "
				"now making vertex labels" << endl;
	}
	for (lvl = 0; lvl <= depth; lvl++) {
		if (f_vv) {
			cout << "poset_classification::make_graph "
					"now making vertex labels "
					"lvl " << lvl << " / " << depth << endl;
		}
		for (po = 0; po < nb_orbits_at_level(lvl); po++) {


			if (f_vv) {
				cout << "poset_classification::make_graph "
						"now making vertex "
						"labels lvl " << lvl << " / " << depth << " po="
						<< po << " / " << nb_orbits_at_level(lvl) << endl;
			}


			string text;
			string text2;
			algebra::ring_theory::longinteger_object go, go1;
			int n;
			
			n = Poo->first_node_at_level(lvl) + po;


			get_set_by_level(lvl, po, the_set);


			get_stabilizer_order(lvl, po, go);
			go.print_to_string(text);
			if (lvl) {
				text2 = std::to_string(the_set[lvl - 1]);
			}
			else {
				text2 = "$\\emptyset$";
			}

			string text3;

			text3.assign(text2);
			LG->add_text(lvl, po, text3, 0/*verbose_level*/);

			// if no vector data, the text will be printed:
			//LG->add_node_vec_data(lvl, po, the_set, lvl, 0 /* verbose_level */);


			// ToDo:
			if (false /* Control->f_node_label_is_group_order */) {
				// label the node with the group order:
				LG->add_node_data1(lvl, po, go.as_int(), 0/*verbose_level*/);
			}
			// ToDo:
			else if (true /* Control->f_node_label_is_element*/) {
				// label the node with the point:
				if (lvl) {
					LG->add_node_data1(
							lvl, po, Poo->get_node(n)->get_pt(),
							0/*verbose_level*/);
				}
				else {
					// root node has no element
				}
			}
			else {
				LG->add_node_data1(lvl, po, n, 0/*verbose_level*/);
			}

			if (lvl) {
				LG->add_node_data2(lvl, po, lvl - 1, 0/*verbose_level*/);
				LG->add_node_data3(lvl, po,
						Poo->get_node(n)->get_prev() - Poo->first_node_at_level(lvl - 1),
						0/*verbose_level*/);
			}
			else {
				LG->add_node_data2(lvl, po, -1, 0/*verbose_level*/);
				LG->add_node_data3(lvl, po, -1, 0/*verbose_level*/);
			}
		}
	}
	FREE_int(Nb);
	FREE_int(Fst);
	FREE_lint(the_set);
	
	if (f_v) {
		cout << "poset_classification::make_graph done" << endl;
	}
}

void poset_classification::make_level_graph(
		int depth,
		combinatorics::graph_theory::layered_graph *&LG,
		int data1, int level, int verbose_level)
// makes a graph with 4 levels showing the relation between
// orbits at level 'level' and orbits at level 'level' + 1
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Nb;
	int *Fst;
	long int nb_middle;
	int i, lvl, po, so, n, n1, f, l;
	algebra::ring_theory::longinteger_domain D;
	int nb_layers = 4;
	long int *the_set;
	long int *the_set2;

	if (f_v) {
		cout << "poset_classification::make_level_graph "
				"verbose_level=" << verbose_level << endl;
	}

	//print_fusion_nodes(depth);

	
	the_set = NEW_lint(depth);
	the_set2 = NEW_lint(depth);
	Nb = NEW_int(4);
	Fst = NEW_int(2);

	Fst[0] = 0;
	for (i = 0; i < level; i++) {
		Fst[0] += nb_orbits_at_level(i);
	}
	nb_middle = count_extension_nodes_at_level(level);
	Fst[1] = Fst[0] + nb_orbits_at_level(level);

	Nb[0] = nb_orbits_at_level(level);
	Nb[1] = nb_middle;
	Nb[2] = nb_middle;
	Nb[3] = nb_orbits_at_level(level + 1);

	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);
	if (f_vv) {
		cout << "poset_classification::make_level_graph "
				"before LG->init" << endl;
		cout << "nb_layers=" << nb_layers << endl;
		cout << "Nb=";
		Int_vec_print(cout, Nb, 4);
		cout << endl;
	}
	LG->add_data1(data1, 0/*verbose_level*/);

	string dummy;
	dummy.assign("");

	LG->init(nb_layers, Nb, dummy, verbose_level);
	if (f_vv) {
		cout << "poset_classification::make_level_graph "
				"after LG->init" << endl;
	}
	LG->place(verbose_level);
	if (f_vv) {
		cout << "poset_classification::make_level_graph "
				"after LG->place" << endl;
	}
	f = 0;
	for (po = 0; po < nb_orbits_at_level(level); po++) {

		if (f_vv) {
			cout << "poset_classification::make_level_graph "
					"adding edges "
					"level=" << level << " po=" << po << " / "
					<< nb_orbits_at_level(level) << endl;
		}

		//
		n = Poo->first_node_at_level(level) + po;
		for (so = 0; so < Poo->node_get_nb_of_extensions(n); so++) {

			if (false /*f_v*/) {
				cout << "poset_classification::make_level_graph "
						"adding edges lvl=" << lvl << " po="
						<< po << " so=" << so << endl;
			}
			LG->add_edge(0, po, 1, f + so,
					1, // edge_color
					0 /*verbose_level*/);
			LG->add_edge(1, f + so, 2, f + so,
					1, // edge_color
					0 /*verbose_level*/);
			extension *E = Poo->get_node(n)->get_E(so);
			if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
				//cout << "extension node" << endl;
				n1 = E->get_data();
				//cout << "n1=" << n1 << endl;
				LG->add_edge(
						2, f + so, 3,
						n1 - Poo->first_node_at_level(level + 1),
						1, // edge_color
						0 /*verbose_level*/);
			}
			else if (E->get_type() == EXTENSION_TYPE_FUSION) {
				//cout << "fusion node" << endl;
				// po = data1
				// so = data2
				int n0, so0;
				n0 = E->get_data1();
				so0 = E->get_data2();
				//cout << "fusion (" << n << "/" << so
				//<< ") -> (" << n0 << "/" << so0 << ")" << endl;
				extension *E0;
				E0 = Poo->get_node(n0)->get_E(so0);
				if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
					cout << "warning: fusion node does not point to "
							"extension node" << endl;
					cout << "type = ";
					print_extension_type(cout, E0->get_type());
					cout << endl;
					exit(1);
				}
				n1 = E0->get_data();
				//cout << "n1=" << n1
				//<< " first_poset_orbit_node_at_level[lvl + 1] = "
				//<< first_poset_orbit_node_at_level[lvl + 1] << endl;
				LG->add_edge(
						2, f + so, 3,
						n1 - Poo->first_node_at_level(level + 1),
						1, // edge_color
						0 /*verbose_level*/);
			}
		}
			
		f += Poo->node_get_nb_of_extensions(n);
	}
	if (f_vv) {
		cout << "poset_classification::make_level_graph "
				"after LG->add_edge" << endl;
	}


	// creates vertex labels for orbits at level 'level' and 'level' + 1:
	if (f_vv) {
		cout << "poset_classification::make_level_graph "
				"now making vertex labels" << endl;
	}
	for (lvl = level; lvl <= level + 1; lvl++) {
		f = 0;
		if (f_vv) {
			cout << "poset_classification::make_level_graph "
					"now making vertex labels lvl " << lvl
					<< " / " << depth << endl;
		}

		if (lvl == level) {
			l = 0;
		}
		else {
			l = 3;
		}

		for (po = 0; po < nb_orbits_at_level(lvl); po++) {


			if (f_vv) {
				cout << "poset_classification::make_level_graph "
						"now making vertex labels lvl " << lvl
						<< " / " << depth << " po=" << po << " / "
						<< nb_orbits_at_level(lvl) << endl;
			}


			string text;
			algebra::ring_theory::longinteger_object go, go1;
			int n, so, len, r;
			
			n = Poo->first_node_at_level(lvl) + po;
			get_stabilizer_order(lvl, po, go);
			go.print_to_string(text);


			string text3;

			text3.assign(text);

			LG->add_text(l, po, text3, 0/*verbose_level*/);
			LG->add_node_data1(
					l, po, Poo->get_node(n)->get_pt(), 0/*verbose_level*/);
			
			get_set_by_level(lvl, po, the_set);
			LG->add_node_vec_data(
					l, po, the_set, lvl, 0 /* verbose_level */);
#if 0
			if (lvl) {
				LG->add_node_data2(2 * lvl + 0, po, 2 * (lvl - 1),
						0/*verbose_level*/);
				LG->add_node_data3(2 * lvl + 0, po,
						root[n].prev - first_poset_orbit_node_at_level[lvl - 1],
						0/*verbose_level*/);
			}
			else {
				LG->add_node_data2(2 * lvl + 0, po, -1, 0/*verbose_level*/);
				LG->add_node_data3(2 * lvl + 0, po, -1, 0/*verbose_level*/);
			}
#endif

			if (lvl == level) {
				for (so = 0; so < Poo->node_get_nb_of_extensions(n); so++) {
					extension *E = Poo->get_node(n)->get_E(so);
					len = E->get_orbit_len();
					D.integral_division_by_int(go, len, go1, r);
					go1.print_to_string(text);

					string text3;

					text3.assign(text);

					LG->add_text(1, f + so, text3, 0/*verbose_level*/);
					LG->add_text(2, f + so, text3, 0/*verbose_level*/);
					
					//get_set_by_level(lvl, po, the_set);
					the_set[lvl] = E->get_pt();
					LG->add_node_vec_data(
							l + 1, f + so, the_set, lvl + 1,
							0 /* verbose_level */);
					LG->set_distinguished_element_index(
							l + 1, f + so, lvl,
							0 /* verbose_level */);


					if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
						the_set[lvl] = E->get_pt();
						LG->add_node_vec_data(
								l + 2, f + so, the_set, lvl + 1,
								0 /* verbose_level */);
						LG->set_distinguished_element_index(
								l + 2, f + so, lvl,
								0 /* verbose_level */);
					}
					else if (E->get_type() == EXTENSION_TYPE_FUSION) {

						//Poset->A->element_retrieve(E->get_data(), Elt1, 0);

						Poset->A2->map_a_set_based_on_hdl(
								the_set, the_set2, lvl + 1, Poset->A, E->get_data(), 0);

						LG->add_node_vec_data(
								l + 2, f + so, the_set2, lvl + 1,
								0 /* verbose_level */);
						LG->set_distinguished_element_index(
								l + 2, f + so, lvl,
								0 /* verbose_level */);
					}
				}
				f += Poo->node_get_nb_of_extensions(n);
			}
		}
	}
	FREE_lint(the_set);
	FREE_lint(the_set2);
	FREE_int(Nb);
	FREE_int(Fst);
	
	if (f_v) {
		cout << "poset_classification::make_level_graph done" << endl;
	}
}

void poset_classification::make_poset_graph_detailed(
		combinatorics::graph_theory::layered_graph *&LG,
		int data1, int max_depth,
		int verbose_level)
// creates the poset graph, with two middle layers at each level.
// In total, the graph that is created will have 3 * depth + 1 layers.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Nb;
	int *Nb_middle;
	int i, po, so, n, n1, f, L;
	algebra::ring_theory::longinteger_domain D;
	int nb_layers = 3 * max_depth + 1;
	long int *the_set;
	long int *the_set2;

	if (f_v) {
		cout << "poset_classification::make_poset_graph_detailed "
				"verbose_level=" << verbose_level << endl;
		cout << "max_depth=" << max_depth << endl;
		cout << "nb_layers=" << nb_layers << endl;
	}

	//print_fusion_nodes(depth);

	
	the_set = NEW_lint(max_depth);
	the_set2 = NEW_lint(max_depth);
	Nb = NEW_int(nb_layers);
	Nb_middle = NEW_int(max_depth);

	for (i = 0; i < max_depth; i++) {
		Nb_middle[i] = count_extension_nodes_at_level(i);
	}

	for (i = 0; i < max_depth; i++) {
		
		Nb[i * 3 + 0] = nb_orbits_at_level(i);
		Nb[i * 3 + 1] = Nb_middle[i];
		Nb[i * 3 + 2] = Nb_middle[i];
	}
	Nb[max_depth * 3 + 0] = nb_orbits_at_level(max_depth);

	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);
	if (f_vv) {
		cout << "poset_classification::make_poset_graph_detailed "
				"before LG->init" << endl;
		cout << "nb_layers=" << nb_layers << endl;
		cout << "Nb=";
		Int_vec_print(cout, Nb, nb_layers);
		cout << endl;
	}
	LG->add_data1(data1, 0/*verbose_level*/);

	string dummy;
	dummy.assign("");

	LG->init(nb_layers, Nb, dummy, verbose_level);
	if (f_vv) {
		cout << "poset_classification::make_poset_graph_detailed "
				"after LG->init" << endl;
	}
	for (i = 0; i < nb_layers; i++) {
		if ((i % 3) == 0) {
			LG->set_radius_factor_for_all_nodes_at_level(
					i, .9 /* radius_factor */, 0 /* verbose_level */);
		}
		else {
			// .9 means we don't draw a label at that node
			//LG->set_radius_factor_for_all_nodes_at_level(
			// i, .9 /* radius_factor */, 0 /* verbose_level */);
			LG->set_radius_factor_for_all_nodes_at_level(
					i, 0.9 /* radius_factor */, 0 /* verbose_level */);
		}
	}
	
	LG->place(verbose_level);
	if (f_vv) {
		cout << "poset_classification::make_poset_graph_detailed "
				"after LG->place" << endl;
	}


	// adding edges:
	if (f_vv) {
		cout << "poset_classification::make_poset_graph_detailed "
				"adding edges" << endl;
	}
	for (L = 0; L < max_depth; L++) {
		if (f_vv) {
			cout << "poset_classification::make_poset_graph_detailed "
					"adding edges at level " << L << endl;
		}
		f = 0;
		for (po = 0; po < nb_orbits_at_level(L); po++) {

			if (f_vv) {
				cout << "poset_classification::make_poset_graph_detailed "
						"adding edges level=" << L << " po=" << po
						<< " / " << nb_orbits_at_level(L) << endl;
			}

			//
			n = Poo->first_node_at_level(L) + po;
			for (so = 0; so < Poo->node_get_nb_of_extensions(n); so++) {

				if (false /*f_v*/) {
					cout << "poset_classification::make_poset_graph_detailed "
							"adding edges level=" << L << " po=" << po
							<< " so=" << so << endl;
				}
				LG->add_edge(L * 3 + 0, po,
						L * 3 + 1, f + so,
						1, // edge_color
						0 /*verbose_level*/);
				LG->add_edge(L * 3 + 1, f + so,
						L * 3 + 2, f + so,
						1, // edge_color
						0 /*verbose_level*/);
				extension *E = Poo->get_node(n)->get_E(so);
				if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
					//cout << "extension node" << endl;
					n1 = E->get_data();
					//cout << "n1=" << n1 << endl;
					LG->add_edge(L * 3 + 2, f + so, L * 3 + 3,
							n1 - Poo->first_node_at_level(L + 1),
							1, // edge_color
							0 /*verbose_level*/);
				}
				else if (E->get_type() == EXTENSION_TYPE_FUSION) {
					//cout << "fusion node" << endl;
					// po = data1
					// so = data2
					int n0, so0;
					n0 = E->get_data1();
					so0 = E->get_data2();
					//cout << "fusion (" << n << "/" << so
					//<< ") -> (" << n0 << "/" << so0 << ")" << endl;
					extension *E0;
					E0 = Poo->get_node(n0)->get_E(so0);
					if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
						cout << "warning: fusion node does not point to "
								"extension node" << endl;
						cout << "type = ";
						print_extension_type(cout, E0->get_type());
						cout << endl;
						exit(1);
					}
					n1 = E0->get_data();
					//cout << "n1=" << n1
					//<< " first_poset_orbit_node_at_level[lvl + 1] = "
					//<< first_poset_orbit_node_at_level[lvl + 1] << endl;
					LG->add_edge(L * 3 + 2, f + so, L * 3 + 3,
							n1 - Poo->first_node_at_level(L + 1),
							1, // edge_color
							0 /*verbose_level*/);
				}
			}
			
			f += Poo->node_get_nb_of_extensions(n);
		}
		if (f_vv) {
			cout << "poset_classification::make_poset_graph_detailed "
					"after LG->add_edge" << endl;
		}
	} // next L
	if (f_vv) {
		cout << "poset_classification::make_poset_graph_detailed "
				"adding edges done" << endl;
	}


	// adding vertex labels:
	if (f_vv) {
		cout << "poset_classification::make_poset_graph_detailed "
				"now making vertex labels" << endl;
	}
	for (L = 0; L <= max_depth; L++) {
		f = 0;
		if (f_vv) {
			cout << "poset_classification::make_poset_graph_detailed "
					"now making vertex labels level " << L
					<< " / " << max_depth << endl;
		}

		for (po = 0; po < nb_orbits_at_level(L); po++) {


			if (f_vv) {
				cout << "poset_classification::make_poset_graph_detailed "
						"now making vertex labels level " << L
						<< " / " << max_depth << " po=" << po
						<< " / " << nb_orbits_at_level(L) << endl;
			}


			string text;
			algebra::ring_theory::longinteger_object go, go1;
			int n, so, len, r;
			
			n = Poo->first_node_at_level(L) + po;
			get_stabilizer_order(L, po, go);
			go.print_to_string(text);

			string text3;

			text3.assign(text);

			LG->add_text(3 * L, po, text3, 0/*verbose_level*/);
			if (L) {
				LG->add_node_data1(
						3 * L, po,
						Poo->get_node(n)->get_pt(),
						0/*verbose_level*/);
			}
			
			get_set_by_level(L, po, the_set);
			LG->add_node_vec_data(3 * L, po, the_set, L, 0 /* verbose_level */);
#if 0
			if (lvl) {
				LG->add_node_data2(2 * lvl + 0, po, 2 * (lvl - 1),
						0/*verbose_level*/);
				LG->add_node_data3(2 * lvl + 0, po,
						root[n].prev - first_poset_orbit_node_at_level[lvl - 1],
						0/*verbose_level*/);
			}
			else {
				LG->add_node_data2(2 * lvl + 0, po, -1, 0/*verbose_level*/);
				LG->add_node_data3(2 * lvl + 0, po, -1, 0/*verbose_level*/);
			}
#endif

			if (L < max_depth) {
				for (so = 0; so < Poo->node_get_nb_of_extensions(n); so++) {
					if (f_vv) {
						cout << "poset_classification::make_poset_graph_detailed "
								"now making vertex labels level " << L
								<< " / " << max_depth << " po=" << po
								<< " / " << nb_orbits_at_level(L)
								<< " so=" << so << endl;
					}
					extension *E = Poo->get_node(n)->get_E(so);
					len = E->get_orbit_len();
					D.integral_division_by_int(go, len, go1, r);
					go1.print_to_string(text);

					string text3;

					text3.assign(text);

					LG->add_text(3 * L + 1, f + so, text3, 0/*verbose_level*/);
					LG->add_text(3 * L + 2, f + so, text3, 0/*verbose_level*/);
					
					//get_set_by_level(lvl, po, the_set);
					the_set[L] = E->get_pt();
					LG->add_node_vec_data(
							3 * L + 1, f + so, the_set,
							L + 1, 0 /* verbose_level */);
					LG->set_distinguished_element_index(3 * L + 1,
							f + so, L, 0 /* verbose_level */);


					if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
						the_set[L] = E->get_pt();
						LG->add_node_vec_data(3 * L + 2, f + so,
								the_set, L + 1, 0 /* verbose_level */);
						LG->set_distinguished_element_index(3 * L + 2,
								f + so, L, 0 /* verbose_level */);
					}
					else if (E->get_type() == EXTENSION_TYPE_FUSION) {

						//Poset->A->element_retrieve(E->get_data(), Elt1, 0);
						Poset->A2->map_a_set_based_on_hdl(
								the_set, the_set2, L + 1,
								Poset->A, E->get_data(), 0);
						LG->add_node_vec_data(
								3 * L + 2, f + so,
								the_set2, L + 1, 0 /* verbose_level */);
						LG->set_distinguished_element_index(
								3 * L + 2,
								f + so, L, 0 /* verbose_level */);
					}
				}
				f += Poo->node_get_nb_of_extensions(n);
			} // if (L < max_depth)
		} // next po
	} // next L
	FREE_lint(the_set);
	FREE_lint(the_set2);
	FREE_int(Nb);
	FREE_int(Nb_middle);
	
	if (f_v) {
		cout << "poset_classification::make_poset_graph_detailed done" << endl;
	}
}


void poset_classification::print_data_structure_tex(
		int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 2);
	string fname_base1;
	string fname;
	int lvl, po, so, n, n1, cnt;
	long int *set;
	algebra::ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "poset_classification::print_data_structure_tex" << endl;
	}
	fname_base1 = problem_label_with_path + "_data_lvl_" + std::to_string(depth);

	fname = fname_base1 + ".tex";

	set = NEW_lint(depth);
	{
		ofstream fp(fname);
		other::l1_interfaces::latex_interface L;

		L.head_easy(fp);

		print_table1_top(fp);
		cnt = 0;
		for (lvl = 0; lvl <= depth; lvl++) {
			if (f_v) {
				cout << "poset_classification::print_data_structure_tex "
						"adding edges lvl=" << lvl << " / " << depth << endl;
			}
			//f = 0;
			for (po = 0; po < nb_orbits_at_level(lvl); po++, cnt++) {


				if (cnt == 25) {
					print_table1_bottom(fp);
					fp << endl;
					fp << "\\bigskip" << endl;
					fp << endl;
					print_table1_top(fp);
					cnt = 0;
				}
				n = Poo->first_node_at_level(lvl) + po;
				
				string text;
				algebra::ring_theory::longinteger_object go, go1;
			
				n = Poo->first_node_at_level(lvl) + po;
				get_stabilizer_order(lvl, po, go);
				go.print_to_string(text);

				Poo->get_node(n)->store_set_to(this, lvl - 1, set);

				fp << lvl << " & " << po << " & ";

				Lint_vec_print(fp, set, lvl);

				fp << " & " << go << "\\\\" << endl;
				
			} // next po
		} // next lvl
		print_table1_bottom(fp);

		fp << endl;
		fp << "\\bigskip" << endl;
		fp << endl;

		int f_permutation_degree_is_small;

		if (Poset->A2->degree < 15) {
			f_permutation_degree_is_small = true;
		}
		else {
			f_permutation_degree_is_small = false;
		}


		print_table_top(fp, f_permutation_degree_is_small);

		cnt = 0;

		for (lvl = 0; lvl < depth; lvl++) {
			if (f_v) {
				cout << "poset_classification::print_data_structure_tex "
						"adding edges lvl=" << lvl << " / " << depth << endl;
			}
			//f = 0;
			for (po = 0; po < nb_orbits_at_level(lvl); po++) {

				n = Poo->first_node_at_level(lvl) + po;

				algebra::ring_theory::longinteger_object go, go1;
				int ol, r, hdl;
			
				n = Poo->first_node_at_level(lvl) + po;
				get_stabilizer_order(lvl, po, go);


				for (so = 0; so < Poo->node_get_nb_of_extensions(n); so++, cnt++) {

					if (cnt == 25) {
						print_table_bottom(fp);
						fp << endl;
						fp << "\\bigskip" << endl;
						fp << endl;
						print_table_top(fp, f_permutation_degree_is_small);
						cnt = 0;
					}
					if (false /*f_v*/) {
						cout << "poset_classification::print_data_structure_tex "
								"adding edges lvl=" << lvl << " po="
								<< po << " so=" << so << endl;
					}
					extension *E = Poo->get_node(n)->get_E(so); // root[n].E + so;
					ol = E->get_orbit_len();

					D.integral_division_by_int(go, ol, go1, r);

					if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
						//cout << "extension node" << endl;
						n1 = E->get_data();


						fp << lvl << " & " << po << " & " << so << " & ";

						Poo->get_node(n)->store_set_to(this, lvl - 1, set);
						set[lvl] = E->get_pt();

						print_set_special(fp, set, lvl + 1);
						//int_vec_print(fp, set, lvl + 1);

						fp << " & ";

						fp << " $id$ ";

						fp << " & ";

						if (f_permutation_degree_is_small) {
							fp << " $id$ ";

							fp << " & ";
						}

						Poo->get_node(n1)->store_set_to(this, lvl + 1 - 1, set);
						//int_vec_print(fp, set, lvl + 1);
						print_set_special(fp, set, lvl + 1);

						fp << " & " << go1 << "\\\\" << endl;


						//cout << "n1=" << n1 << endl;
					}
					else if (E->get_type() == EXTENSION_TYPE_FUSION) {
						//cout << "fusion node" << endl;
						// po = data1
						// so = data2
						int n0, so0;
						n0 = E->get_data1();
						so0 = E->get_data2();
						//cout << "fusion (" << n << "/" << so
						//<< ") -> (" << n0 << "/" << so0 << ")" << endl;
						extension *E0;
						E0 = Poo->get_node(n0)->get_E(so0);
						if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
							cout << "warning: fusion node does not point "
									"to extension node" << endl;
							cout << "type = ";
							print_extension_type(cout, E0->get_type());
							cout << endl;
							exit(1);
						}
						n1 = E0->get_data();
						//cout << "n1=" << n1
						//<< " first_poset_orbit_node_at_level[lvl + 1] = "
						//<< first_poset_orbit_node_at_level[lvl + 1] << endl;




						fp << lvl << " & " << po << " & " << so << " & ";

						Poo->get_node(n)->store_set_to(this, lvl - 1, set);
						set[lvl] = E->get_pt();
						//int_vec_print(fp, set, lvl + 1);
						print_set_special(fp, set, lvl + 1);


						int *Elt;


						Elt = NEW_int(Poset->A->elt_size_in_int);

						fp << " & ";

						hdl = E->get_data();
						Poset->A->Group_element->element_retrieve(hdl, Elt, false);

						fp << "$";
						Poset->A->Group_element->element_print_latex(Elt, fp);
						fp << "$";

						fp << " & ";

						if (f_permutation_degree_is_small) {
							fp << "$";
							Poset->A2->Group_element->element_print_as_permutation(Elt, fp);
							fp << "$";

							fp << " & ";
						}

						FREE_int(Elt);


						Poo->get_node(n1)->store_set_to(this, lvl + 1 - 1, set);
						//int_vec_print(fp, set, lvl + 1);
						print_set_special(fp, set, lvl + 1);

						fp << " & " << go1 << "\\\\" << endl;


					}
				}
			
				//f += Poo->node_get_nb_of_extensions(n);
				

			} // next po
		} // next lvl

		print_table_bottom(fp);

		L.foot(fp);
	}
	FREE_lint(set);
	if (f_v) {
		cout << "poset_classification::print_data_structure_tex done" << endl;
	}
}

static void print_table1_top(
		std::ostream &fp)
{
	fp << "\\begin{tabular}{|r|r|r|r|}" << endl;
	fp << "\\hline" << endl;
	fp << "$i$ & $j$ & $R_{i,j}$ & $|G_{R_{i,j}}|$\\\\" << endl;
	fp << "\\hline" << endl;
	fp << "\\hline" << endl;
}

static void print_table1_bottom(
		std::ostream &fp)
{
	fp << "\\hline" << endl;
	fp << "\\end{tabular}" << endl;
}

static void print_table_top(
		std::ostream &fp, int f_permutation_degree_is_small)
{
	fp << "\\begin{tabular}{|r|r|r|r|r";
	if (f_permutation_degree_is_small) {
		fp << "|r";
	}
	fp << "|r|r|}" << endl;
	fp << "\\hline" << endl;
	fp << "$i$ & $j$ & $a$ & $U_{i,j,a}$ & $\\varphi$ & ";
	if (f_permutation_degree_is_small) {
		fp << "$\\varphi$ & ";
	}
	fp << "$U_{i+1,c,d}$ & $|\\Stab_{G_{R_{i,j}}}(U_{i,j,a})|$\\\\" << endl;
	fp << "\\hline" << endl;
	fp << "\\hline" << endl;
}

static void print_table_bottom(
		std::ostream &fp)
{
	fp << "\\hline" << endl;
	fp << "\\end{tabular}" << endl;
}

static void print_set_special(
		std::ostream &fp, long int *set, int sz)
{
	int i;

	fp << "(";
	for (i = 0; i < sz; i++) {
		fp << set[i];
		if (i < sz - 2) {
			fp << ", ";
		}
		if (i == sz - 2) {
			fp << "; ";
		}
	}
	fp << ")";
}


}}}


