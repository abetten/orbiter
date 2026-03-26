/*
 * pc_tree_interface.cpp
 *
 *  Created on: Feb 27, 2026
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


#define MAX_NODES_FOR_TREEFILE 25000
//#define MAX_NODES_FOR_TREEFILE 6500


pc_tree_interface::pc_tree_interface()
{
	Record_birth();

	PC = NULL;

	lvl = 0;

	Tree_draw_options = NULL;
	Draw_options = NULL;

	//std::string fname;
	//std::string fname_tree;

}

pc_tree_interface::~pc_tree_interface()
{
	Record_death();
}

void pc_tree_interface::init(
		poset_classification *PC,
		int lvl,
		other::graphics::draw_options *Draw_options,
		int verbose_level)
{
	pc_tree_interface::PC = PC;
	pc_tree_interface::lvl = lvl;
	pc_tree_interface::Draw_options = Draw_options;

	fname = PC->get_problem_label_with_path() + "_" + std::to_string(lvl);

	fname_tree = PC->get_problem_label_with_path() + "_" + std::to_string(lvl) + ".tree";

}




int pc_tree_interface::write_treefile(
		int lvl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, level;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "pc_tree_interface::write_treefile" << endl;
	}

	if  (PC->get_Poo()->first_node_at_level(lvl + 1) < MAX_NODES_FOR_TREEFILE) {
		{
			if (f_vv) {
				cout << "pc_tree_interface::write_treefile "
						"writing treefile " << fname_tree << endl;
			}
			ofstream f(fname_tree);

			f << "# " << lvl << endl;

			level = 0; // starter_size;

			for (i = PC->get_Poo()->first_node_at_level(level);
					i < PC->get_Poo()->first_node_at_level(level + 1); i++) {
				if (f_vv) {
					cout << "pc_tree_interface::write_treefile "
							"node " << i << ":" << endl;
				}
				PC->get_Poo()->log_nodes_for_treefile(level, i, f,
						true /* f_recurse */, verbose_level);
			}

			f << "-1 " << PC->get_Poo()->first_node_at_level(lvl + 1) << endl;
		}
		if (f_vv) {
			cout << "written file " << fname_tree
					<< " of size " << Fio.file_size(fname_tree) << endl;
		}
		if (f_v) {
			cout << "pc_tree_interface::write_treefile done" << endl;
		}
		return true;
	}
	else {
		cout << "pc_tree_interface::write_treefile too many nodes, "
				"you may increase MAX_NODES_FOR_TREEFILE if you wish" << endl;
		cout << "MAX_NODES_FOR_TREEFILE=" << MAX_NODES_FOR_TREEFILE << endl;
		cout << "first_poset_orbit_node_at_level[lvl + 1]="
				<< PC->get_Poo()->first_node_at_level(lvl + 1) << endl;
		cout << "lvl=" << lvl << endl;
		return false;
	}
}

void pc_tree_interface::draw_tree(
		int lvl,
	int xmax, int ymax, int rad, int f_embedded,
	int f_sideways, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
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
		cout << "pc_tree_interface::draw_tree" << endl;
	}


	if (Fio.file_size(fname_tree)) {
		if (f_vv) {
			cout << "reading treefile" << endl;
		}
		Tree_draw_options->f_file = true;
		Tree_draw_options->file_name.assign(fname_tree);

		T.init(
				Tree_draw_options, xmax, ymax, verbose_level - 1);

		nb_nodes = T.nb_nodes;
		if (f_vv) {
			cout << "pc_tree_interface::draw_tree read treefile "
					<< fname_tree << " with " << nb_nodes
					<< " nodes" << endl;
			cout << "pc_tree_interface::draw_tree "
					"first_poset_orbit_node_at_level"
					"level[lvl + 1] "
					<< PC->get_Poo()->first_node_at_level(lvl + 1)
					<< " nodes" << endl;
		}
		if (nb_nodes != PC->get_Poo()->first_node_at_level(lvl + 1)) {
			cout << "pc_tree_interface::draw_tree nb_nodes != "
					"first_poset_orbit_node_at_level"
					"[lvl + 1]" << endl;
			cout << "nb_nodes=" << nb_nodes << endl;
			cout << "first_poset_orbit_node_at_level[lvl + 1]="
					<< PC->get_Poo()->first_node_at_level(lvl + 1) << endl;
			exit(1);
		}
		if (nb_nodes > 100) {
			f_draw_points = false;
			f_draw_aut_group_order = false;
		}

		coord_xyw = NEW_int(3 * nb_nodes);

		if (f_vv) {
			cout << "pc_tree_interface::draw_tree "
					"calling get_coordinates" << endl;
		}
		T.root->get_coordinates_and_width(idx, coord_xyw);

#if 0
		for (i = 0; i < nb_nodes; i++) {
			coord_xyw[i * 3 + 2] = (int)sqrt((double)coord_xyw[i * 3 + 2]);
		}
#endif

		if (false) {
			cout << "pc_tree_interface::draw_tree "
					"coord_xyw:" << endl;
			for (i = 0; i < nb_nodes; i++) {
				cout << i << " : ("
					<< coord_xyw[i * 3 + 0] << ","
					<< coord_xyw[i * 3 + 1] << ","
					<< coord_xyw[i * 3 + 2] << ")" << endl;
			}
		}

		if (f_vv) {
			cout << "pc_tree_interface::draw_tree calling "
					"poset_orbit_node_depth_breadth_perm_and_inverse" << endl;
		}
		PC->get_Poo()->poset_orbit_node_depth_breadth_perm_and_inverse(
				lvl /* max_depth */,
			perm, perm_inv, verbose_level);
		if (false) {
			cout << "pc_tree_interface::draw_tree "
					"depth_breadth_perm_and_inverse:" << endl;
			for (i = 0; i < nb_nodes; i++) {
				cout << i << " : ("
					<< perm[i] << ","
					<< perm_inv[i] << ")" << endl;
			}
		}

		if (f_vv) {
			cout << "pc_tree_interface::draw_tree "
					"before draw_tree_low_level" << endl;
		}
		draw_tree_low_level(
				//fname,
				//Tree_draw_options, Draw_options,
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
		cout << "pc_tree_interface::draw_tree the file " << fname_tree
				<< " does not exist, cannot draw the tree" << endl;
	}
}

void pc_tree_interface::draw_tree_low_level(
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

	if (xmax == -1) {
		xmax = 2000;
	}
	if (ymax == -1) {
		ymax = 3000;
	}
	if (ymax == 0) {
		ymax = 3000;
	}

	fname_full = fname + ".mp";

	if (f_v) {
		cout << "pc_tree_interface::draw_tree_low_level "
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
		cout << "pc_tree_interface::draw_tree_low_level "
				"written file " << fname_full
				<< " of size " << Fio.file_size(fname_full) << endl;
	}

}

void pc_tree_interface::draw_tree_low_level1(
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
	actions::action_global Action_global;


	max_set_size = PC->get_Poo()->get_max_set_size();

	Px = NEW_int(nb_nodes);
	Py = NEW_int(nb_nodes);
	Width = NEW_int(nb_nodes);
	Qx = NEW_int(100);
	Qy = NEW_int(100);
	set0 = NEW_lint(max_set_size);
	set1 = NEW_lint(max_set_size);

	if (f_v) {
		cout << "pc_tree_interface::draw_tree_low_level1" << endl;
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
			cout << "pc_tree_interface::draw_tree_low_level1 i=" << i << endl;
		}
		nb_e = PC->get_Poo()->node_get_nb_of_extensions(i);
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
			Nb_e = PC->get_Poo()->node_get_nb_of_extensions(j);
			if (f_vv) {
				cout << "pc_tree_interface::draw_tree_low_level1 "
						"i=" << i
						<< " j=" << j << " nb_e=" << nb_e <<
					" Nb_e=" << Nb_e << endl;
				cout << "root[i].get_E(j)->get_type()="
						<< PC->get_Poo()->get_node(i)->get_E(j)->get_type() << endl;
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

			if (PC->get_Poo()->get_node(i)->get_E(j)->get_type() == EXTENSION_TYPE_EXTENSION) {
				// extension node
				pt = PC->get_Poo()->get_node(i)->get_E(j)->get_pt();
				nxt = PC->get_Poo()->get_node(i)->get_E(j)->get_data();
				if (f_vv) {
					cout << "pc_tree_interface::draw_tree_low_level1 "
							"extension node: pt=" << pt
							<< " nxt=" << nxt << endl;
				}
				if (nxt >= 0) {
					Qx[0] = Px[perm[i]];
					Qy[0] = Py[perm[i]];

					Qx[1] = Px[perm[nxt]];
					Qy[1] = Py[perm[nxt]];
					if (false) {
						cout << "pc_tree_interface::draw_tree_low_level1 "
								"Qx[0]=" << Qx[0] << " Qy[0]=" << Qy[0] << endl;
						cout << "pc_tree_interface::draw_tree_low_level1 "
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
						cout << "pc_tree_interface::draw_tree_low_level1 "
								"Qx[0]=" << Qx[0] << " Qy[0]=" << Qy[0] << endl;
						cout << "pc_tree_interface::draw_tree_low_level1 "
								"Qx[1]=" << Qx[1] << " Qy[1]=" << Qy[1] << endl;
					}

					G.polygon2(Qx, Qy, 0, 1);
					if (false) {
						cout << "pc_tree_interface::draw_tree_low_level1 "
								"after G.polygon2" << endl;
					}
				}
			}
			else if (PC->get_Poo()->get_node(i)->get_E(j)->get_type() == EXTENSION_TYPE_FUSION) {
				// fusion node
				if (f_vv) {
					cout << "pc_tree_interface::draw_tree_low_level1 "
							"fusion node" << endl;
				}
				if (true /*root[i].E[j].get_pt() > root[i].get_pt()*/) {
					pt = PC->get_Poo()->get_node(i)->get_E(j)->get_pt();
					hdl = PC->get_Poo()->get_node(i)->get_E(j)->get_data();
					depth = PC->get_Poo()->get_node(i)->depth_of_node(PC);
					PC->get_Poo()->get_node(i)->store_set_to(PC, depth - 1, set0);
					set0[depth] = pt;
					if (f_vvv) {
						cout << "pc_tree_interface::draw_tree_low_level1 "
								"fusion node i=" << i
								<< " j=" << j << " set = ";
						Lint_vec_print(cout, set0, depth + 1);
						cout << endl;
					}

					//Poset->A2->element_retrieve(hdl, Elt1, false);

					Action_global.map_a_set_based_on_hdl(
							set0, set1,
							depth + 1,
							PC->get_poset()->A, PC->get_poset()->A2,
							hdl, 0);

					Sorting.lint_vec_heapsort(set1, depth + 1);

					if (f_vvv) {
						cout << "pc_tree_interface::draw_tree_low_level1 "
								"mapping the set to = ";
						Lint_vec_print(cout, set1, depth + 1);
						cout << endl;
					}

					hdl2 = PC->get_Poo()->find_poset_orbit_node_for_set(
							depth + 1, set1,
							false /* f_tolerant */,
							0 /* verbose_level */);
					if (hdl2 >= 0) {
						if (f_vvv) {
							cout << "pc_tree_interface::draw_tree_low_level1 "
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
		cout << "pc_tree_interface::draw_tree_low_level1 "
				"now drawing node labels" << endl;
	}

	G.sf_interior(100 /* fill_interior */);
	G.sf_color(1 /* fill_color */);


	string s;

	for (i = 0; i < nb_nodes; i++) {
		if (i == 0) {
			continue;
		}
		pt = PC->get_Poo()->get_node(perm_inv[i])->get_pt();
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
			cout << "pc_tree_interface::draw_tree_low_level1 "
					"now drawing connections" << endl;
		}
		for (i = 0; i < nb_nodes; i++) {
			nb_e = PC->get_Poo()->node_get_nb_of_extensions(i);
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
				pt = PC->get_Poo()->get_node(i)->get_E(j)->get_pt();
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
		cout << "pc_tree_interface::draw_tree_low_level1 done" << endl;
	}
}



}}}


