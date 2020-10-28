/*
 * graphical_output.cpp
 *
 *  Created on: Oct 11, 2020
 *      Author: betten
 */

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


graphical_output::graphical_output()
{

}

graphical_output::~graphical_output()
{

}

void graphical_output::draw_layered_graph_from_file(std::string &fname,
		layered_graph_draw_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graphical_output::draw_layered_graph_from_file fname=" << fname << endl;
	}
	layered_graph *LG;
	file_io Fio;

	LG = NEW_OBJECT(layered_graph);
	if (Fio.file_size(fname) <= 0) {
		cout << "graphical_output::draw_layered_graph_from_file file " << fname << " does not exist" << endl;
		exit(1);
		}
	LG->read_file(fname, verbose_level - 1);

	if (f_v) {
		cout << "graphical_output::draw_layered_graph_from_file Layered graph read from file" << endl;
	}

	LG->print_nb_nodes_per_level();

	int data1;


	data1 = LG->data1;

	if (f_v) {
		cout << "graphical_output::draw_layered_graph_from_file data1=" << data1 << endl;
	}

	if (Opt->f_y_stretch) {
		LG->place_with_y_stretch(Opt->y_stretch, verbose_level - 1);
		}
	if (Opt->f_spanning_tree) {
		// create n e w x coordinates
		LG->create_spanning_tree(TRUE /* f_place_x */, verbose_level);
		}
#if 0
	if (Opt->f_numbering_on) {
		// create depth first ranks at each node:
		LG->create_spanning_tree(FALSE /* f_place_x */, verbose_level);
		}
#endif



	string fname_out;

	fname_out.assign(fname);
	chop_off_extension(fname_out);
	fname_out.append("_draw");
#if 0
	if (Opt->f_spanning_tree) {
		fname_out.append("_tree");
	}
#endif

	//fname_out.append(".mp");

	if (Opt->f_paths_in_between) {

		if (f_v) {
			cout << "graphical_output::draw_layered_graph_from_file f_paths_in_between" << endl;
		}
		std::vector<std::vector<int> > All_Paths;

		if (f_v) {
			cout << "graphical_output::draw_layered_graph_from_file before LG->find_all_paths_between" << endl;
		}
		LG->find_all_paths_between(Opt->layer1, Opt->node1, Opt->layer2, Opt->node2,
				All_Paths,
				verbose_level - 2);
		if (f_v) {
			cout << "graphical_output::draw_layered_graph_from_file after LG->find_all_paths_between" << endl;
		}

		if (f_v) {
			cout << "graphical_output::draw_layered_graph_from_file before LG->remove_edges" << endl;
		}
		LG->remove_edges(Opt->layer1, Opt->node1, Opt->layer2, Opt->node2,
				All_Paths,
				verbose_level - 2);
		if (f_v) {
			cout << "graphical_output::draw_layered_graph_from_file after LG->remove_edges" << endl;
		}


	}


	LG->draw_with_options(fname_out, Opt, verbose_level - 10);

	int n;
	double avg;
	n = LG->nb_nodes();
	avg = LG->average_word_length();
	if (f_v) {
		cout << "graphical_output::draw_layered_graph_from_file number of nodes = " << n << endl;
		cout << "graphical_output::draw_layered_graph_from_fileaverage word length = " << avg << endl;
	}


	if (f_v) {
		cout << "graphical_output::draw_layered_graph_from_file Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;
	}

	FREE_OBJECT(LG);

	if (f_v) {
		cout << "graphical_output::draw_layered_graph_from_file done" << endl;
	}
}

}}

