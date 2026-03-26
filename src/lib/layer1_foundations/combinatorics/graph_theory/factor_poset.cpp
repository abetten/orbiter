/*
 * factor_poset.cpp
 *
 *  Created on: Feb 14, 2026
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace graph_theory {


factor_poset::factor_poset()
{
	Record_birth();

	nb_layers = 0;
	Nb_elements = NULL;
	Fst = NULL;
	Nb_orbits = NULL;
	Fst_element_per_orbit = NULL;
	Orbit_len = NULL;

	LG = NULL;

}


factor_poset::~factor_poset()
{
	Record_death();

	if (Nb_elements) {
		FREE_int(Nb_elements);
	}
#if 0
	if (Nb_orbits) {
		FREE_int(Nb_orbits);
	}
#endif
	if (Fst) {
		FREE_int(Fst);
	}
	if (Fst_element_per_orbit) {
		int i;
		for (i = 0; i < nb_layers; i++) {
			FREE_int(Fst_element_per_orbit[i]);
		}
		FREE_pint(Fst_element_per_orbit);
	}

	if (Nb_orbits) {
		FREE_int(Nb_orbits);
	}
	if (Orbit_len) {
		int i;
		for (i = 0; i < nb_layers; i++) {
			FREE_int(Orbit_len[i]);
		}
		FREE_pint(Orbit_len);
	}

	if (LG) {
		FREE_OBJECT(LG);
	}

}

void factor_poset::init(
		int depth,
		int *Nb_orbits,
		int **Orbit_len,
		int data1,
		double x_stretch,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "factor_poset::init" << endl;
	}

	int i, j;

	nb_layers = depth + 1;
	if (f_v) {
		cout << "factor_poset::init nb_layers = " << nb_layers << endl;
		cout << "factor_poset::init x_stretch = " << x_stretch << endl;
	}

	Nb_elements = NEW_int(nb_layers);
	factor_poset::Nb_orbits = Nb_orbits; // = NEW_int(nb_layers);

	Fst = NEW_int(nb_layers + 1);
	Fst_element_per_orbit = NEW_pint(nb_layers);
	factor_poset::Orbit_len = Orbit_len; // = NEW_pint(nb_layers);

	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {

		//Nb_orbits[i] = nb_orbits_at_level(i);
		Fst_element_per_orbit[i] = NEW_int(Nb_orbits[i] + 1);
		//Orbit_len[i] = NEW_int(Nb_orbits[i]);
		Nb_elements[i] = 0;

		Fst_element_per_orbit[i][0] = 0;

		for (j = 0; j < Nb_orbits[i]; j++) {

			//Orbit_len[i][j] = orbit_length_as_int(j, i);

			Nb_elements[i] += Orbit_len[i][j];

			Fst_element_per_orbit[i][j + 1] =
					Fst_element_per_orbit[i][j] + Orbit_len[i][j];
		}
		Fst[i + 1] = Fst[i] + Nb_elements[i];
	}


	int lvl;


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



	if (f_v) {
		cout << "factor_poset::init done" << endl;
	}
}

void factor_poset::print_nb_orbits_per_level()
{
	int i;

	cout << "factor_poset::print_nb_nodes_per_level" << endl;
	cout << "level & number of orbits " << endl;
	for (i = 0; i < nb_layers; i++) {
		cout << i << " & " <<  Nb_orbits[i] << "\\\\" << endl;
	}
}



#define FACTOR_POSET_MAGIC_SYNC 6577893

void factor_poset::write_memory_object(
		other::orbiter_kernel_system::memory_object *m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "factor_poset::write_memory_object" << endl;
	}

	int i, j;

	m->write_int(1); // version number of this file format

	m->write_int(FACTOR_POSET_MAGIC_SYNC); // a check to see if the file is not corrupt

	if (f_vv) {
		cout << "factor_poset::write_memory_object "
				"m->used_length = " << m->used_length << endl;
	}
	if (f_vv) {
		cout << "factor_poset::write_memory_object "
				"nb_layers=" << nb_layers
				<< " m->used_length = " << m->used_length << endl;
	}
	m->write_int(nb_layers);
	for (i = 0; i < nb_layers; i++) {
		m->write_int(Nb_elements[i]);
	}
	for (i = 0; i < nb_layers + 1; i++) {
		m->write_int(Fst[i]);
	}
	for (i = 0; i < nb_layers; i++) {
		m->write_int(Nb_orbits[i]);
	}
	for (i = 0; i < nb_layers; i++) {
		for (j = 0; j < Nb_orbits[i] + 1; j++) {
			m->write_int(Fst_element_per_orbit[i][j]);
		}
	}
	for (i = 0; i < nb_layers; i++) {
		for (j = 0; j < Nb_orbits[i]; j++) {
			m->write_int(Orbit_len[i][j]);
		}
	}

	LG->write_memory_object(m, verbose_level);

	m->write_int(FACTOR_POSET_MAGIC_SYNC); // a check to see if the file is not corrupt

	if (f_v) {
		cout << "factor_poset::write_memory_object "
				"data size (in chars) = "
				<< m->used_length << endl;
	}
	if (f_v) {
		cout << "factor_poset::write_memory_object done" << endl;
	}
}



void factor_poset::read_memory_object(
		other::orbiter_kernel_system::memory_object *m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "factor_poset::read_memory_object" << endl;
	}

	int i, j;
	int version, magic_sync;

	m->read_int(&version); // version number of this file format
	if (version != 1) {
		cout << "factor_poset::read_memory_object "
				"unknown version: version = " << version << endl;
		exit(1);
	}

	m->read_int(&magic_sync);
	if (magic_sync != FACTOR_POSET_MAGIC_SYNC) {
		cout << "factor_poset::read_memory_object "
				"the file is unreadable" << endl;
		exit(1);

	}
	m->read_int(&nb_layers);
	Nb_elements = NEW_int(nb_layers);
	for (i = 0; i < nb_layers; i++) {
		m->read_int(&Nb_elements[i]);
	}
	Fst = NEW_int(nb_layers + 1);
	for (i = 0; i < nb_layers + 1; i++) {
		m->read_int(&Fst[i]);
	}
	Nb_orbits = NEW_int(nb_layers);
	for (i = 0; i < nb_layers; i++) {
		m->read_int(&Nb_orbits[i]);
	}
	Fst_element_per_orbit = NEW_pint(nb_layers);
	for (i = 0; i < nb_layers; i++) {
		Fst_element_per_orbit[i] = NEW_int(Nb_orbits[i] + 1);
		for (j = 0; j < Nb_orbits[i] + 1; j++) {
			m->read_int(&Fst_element_per_orbit[i][j]);
		}
	}
	Orbit_len = NEW_pint(nb_layers);
	for (i = 0; i < nb_layers; i++) {
		Orbit_len[i] = NEW_int(Nb_orbits[i]);
		for (j = 0; j < Nb_orbits[i]; j++) {
			m->read_int(&Orbit_len[i][j]);
		}
	}

	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);

	LG->read_memory_object(m, verbose_level);

	m->read_int(&magic_sync);
	if (magic_sync != FACTOR_POSET_MAGIC_SYNC) {
		cout << "factor_poset::read_memory_object "
				"file is unreadable" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "factor_poset::read_memory_object "
				"finished" << endl;
	}
}

void factor_poset::write_file(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::memory_object M;

	if (f_v) {
		cout << "factor_poset::write_file" << endl;
	}
	M.alloc(1024 /* length */, verbose_level - 1);
	M.used_length = 0;
	M.cur_pointer = 0;
	write_memory_object(&M, verbose_level - 1);
	M.write_file(fname, verbose_level - 1);
	if (f_v) {
		cout << "factor_poset::write_file done" << endl;
	}
}

void factor_poset::read_file(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::memory_object M;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "factor_poset::read_file "
				"reading file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	M.read_file(fname, verbose_level - 1);
	if (f_v) {
		cout << "factor_poset::read_file "
				"read file " << fname << endl;
	}
	M.cur_pointer = 0;
	read_memory_object(&M, verbose_level - 1);
	if (f_v) {
		cout << "factor_poset::read_file done" << endl;
	}
}


void factor_poset::draw_with_options(
		std::string &fname,
		other::graphics::draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int factor_1000 = 1000;
	string fname_full;
	other::orbiter_kernel_system::file_io Fio;

	fname_full = fname + ".mp";

	if (f_v) {
		cout << "factor_poset::draw_with_options "
				"fname_full = " << fname_full << endl;
	}
	if (O == NULL) {
		cout << "factor_poset::draw_with_options "
				"O == NULL" << endl;
		exit(1);
	}

	{


		if (f_v) {
			cout << "factor_poset::draw_with_options xin = " << O->xin << endl;
			cout << "factor_poset::draw_with_options yin = " << O->yin << endl;
			cout << "factor_poset::draw_with_options xout = " << O->xout << endl;
			cout << "factor_poset::draw_with_options yout = " << O->yout << endl;
			cout << "factor_poset::draw_with_options f_embedded = " << O->f_embedded << endl;
		}

		other::graphics::mp_graphics G;

		G.init(fname_full, O, verbose_level - 1);


		G.header();
		G.begin_figure(factor_1000);

		//G.sl_thickness(30); // 100 is normal



		if (f_v) {
			cout << "factor_poset::draw" << endl;
			cout << "f_nodes_empty=" << O->f_nodes_empty << endl;
		}

		if (O->f_has_draw_begining_callback) {
			(*O->draw_begining_callback)(
					LG, &G,
					O->xin, O->yin, O->f_rotated,
					O->rad * 4, O->rad * 4);
		}




		// draw edges:

		if (f_v) {
			cout << "factor_poset::draw before draw_edges" << endl;
		}

		LG->draw_edges(O, &G, verbose_level - 2);

		if (f_v) {
			cout << "factor_poset::draw after draw_edges" << endl;
		}


		if (O->f_no_vertices) {
			if (f_v) {
				cout << "factor_poset::draw we are not drawing the vertices" << endl;
			}


		}

		else {
			// now draw the vertices:
			if (f_v) {
				cout << "factor_poset::draw before draw_vertices" << endl;
			}

			LG->draw_vertices(O, &G, verbose_level - 2);

			if (f_v) {
				cout << "factor_poset::draw after draw_vertices" << endl;
			}
		}


		if (O->f_has_draw_ending_callback) {
			(*O->draw_ending_callback)(LG, &G, O->xin, O->yin,
					O->f_rotated, O->rad * 4, O->rad * 4);
		}


		if (f_v) {
			cout << "factor_poset::draw before draw_level_info" << endl;
		}
		LG->draw_level_info(O, &G, verbose_level - 2);
		if (f_v) {
			cout << "factor_poset::draw after draw_level_info" << endl;
		}

		if (O->f_corners) {
			double move_out = 0.01;

			G.frame(move_out);
		}



		if (f_v) {
			cout << "factor_poset::draw before draw_orbit_info" << endl;
		}
		draw_orbit_info(
				O,
				&G,
				verbose_level);
		if (f_v) {
			cout << "factor_poset::draw after draw_orbit_info" << endl;
		}


		G.end_figure();
		G.footer();
	}
	if (f_v) {
		cout << "factor_poset::draw written file " << fname_full
				<< " of size " << Fio.file_size(fname_full) << endl;
	}

}

void factor_poset::draw_orbit_info(
		other::graphics::draw_options *O,
		other::graphics::mp_graphics *G,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "factor_poset::draw_orbit_info" << endl;
	}

	other::data_structures::sorting Sorting;
	int i, j;
	int x, y;
	//int Px[10], Py[10];

	int cur = 0;

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
			cout << "factor_poset::draw_orbit_info drawing nodes in layer "
					<< i << "  with " << Nb_orbits[i] << " orbits:" << endl;
		}

		for (j = 0; j < Nb_orbits[i]; j++) {
			if (f_v) {
				cout << "orbit " << i << " " << j << " at ("
						<< LG->L[i].Nodes[j].x_coordinate << ","
						<< LG->L[i].y_coordinate << ")" << endl;
			}
			coordinates(
					i, j,
					O->xin, O->yin,
					O->f_rotated, x, y);


			string label;

			int ol;

			ol = Orbit_len[i][j];

			label = std::to_string(cur + j) + ":" + std::to_string(ol);


			if (label.length() /* L[i].Nodes[j].radius_factor >= 1.*/) {
				//G.circle_text(x, y, L[i].Nodes[j].label);
				G->aligned_text(x, y, "", label);
				//G.aligned_text(x, y, "", L[i].Nodes[j].label);
			}


		}

		if (O->f_poset_with_horizontal_lines) {

			int Px[10], Py[10];
			int edge_color;

			for (j = 0; j < Nb_orbits[i]; j++) {
				if (f_v) {
					cout << "orbit " << i << " " << j << " at ("
							<< LG->L[i].Nodes[j].x_coordinate << ","
							<< LG->L[i].y_coordinate << ")" << endl;
				}
				int n;
				int x1, y1;
				int x2, y2;

				n = LG->L[i].group_start[j];
				LG->coordinates_direct(
						LG->L[i].Nodes[n].x_coordinate,
						LG->L[i].y_coordinate, O->xin, O->yin, O->f_rotated, x1, y1);

				n = LG->L[i].group_start[j + 1] - 1;
				LG->coordinates_direct(
						LG->L[i].Nodes[n].x_coordinate,
						LG->L[i].y_coordinate, O->xin, O->yin, O->f_rotated, x2, y2);

				Px[0] = x1;
				Px[1] = x2;
				Py[0] = y1;
				Py[1] = y2;

				edge_color = 1;

				G->sl_color(edge_color);
				G->polygon2(Px, Py, 0, 1);
				G->sl_color(1);


			}

		}

		cur += Nb_orbits[i];
	}

	if (f_v) {
		cout << "factor_poset::draw_orbit_info done" << endl;
	}
}

void factor_poset::coordinates(
		int layer, int orbit,
		int x_max, int y_max, int f_rotated,
		int &x, int &y)
{
	//int l, n;



	if (!LG->L[layer].f_has_grouping) {
		cout << "factor_poset::coordinates no grouping available" << endl;
		exit(1);
	}
	LG->coordinates_direct(
			LG->L[layer].group_x[orbit],
			LG->L[layer].y_coordinate, x_max, y_max, f_rotated, x, y);
}


#if 0
int f_has_grouping;
int nb_groups;
int *group_start; // [nb_groups + 1]
double *group_x; // [nb_groups]
double *group_dx; // [nb_groups]
int nb_elements;
#endif

}}}}


