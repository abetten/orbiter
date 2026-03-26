// poset_classification_draw.cpp
//
// Anton Betten
// moved out of poset_classification.cpp  November 14, 2007

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"


using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {



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




void poset_classification::draw_poset_full(
		std::string &fname_base,
		int depth, int data,
		other::graphics::draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"fname_base=" << fname_base << " data=" << data << endl;
	}

	layer1_foundations::combinatorics::graph_theory::factor_poset *Factor_poset;

	double x_stretch = 1.0;

	if (LG_Draw_options->f_poset_orbits_x_stretch) {
		if (f_v) {
			cout << "poset_classification::draw_poset_full "
					"poset_orbits_x_stretch = "
					<< LG_Draw_options->poset_orbits_x_stretch << endl;
		}
		x_stretch = (float) LG_Draw_options->poset_orbits_x_stretch / (float) 1000;
	}
	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"x_stretch = " << x_stretch << endl;
	}

	int type_of_poset = 0;
	std::string suffix;

	if (LG_Draw_options->f_poset_type_Asup) {
		type_of_poset = 1;
		suffix = "_Asup";
	}
	else if (LG_Draw_options->f_poset_type_Ainf) {
		type_of_poset = 2;
		suffix = "_Ainf";
	}

	pc_convert_data_structure Pc_convert_data_structure;

	Pc_convert_data_structure.init(
			this,
			0 /* verbose_level*/);

	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"before make_factor_poset" << endl;
	}

	Pc_convert_data_structure.make_factor_poset(
			depth, data, x_stretch, Factor_poset,
			type_of_poset,
			LG_Draw_options->f_poset_with_horizontal_lines,
			verbose_level - 2);
	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"after make_factor_poset" << endl;
	}

	string base_name;
	string fname1;
	string fname2;

	base_name = fname_base + "_poset_full_lvl_" + std::to_string(depth) + suffix;
	fname1 = base_name + ".layered_graph";
	fname2 = base_name + ".factor_poset";
	
	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"before LG->write_file" << endl;
	}

	Factor_poset->LG->write_file(fname1, 0 /*verbose_level*/);

	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"after LG->write_file" << endl;
	}



	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"before Factor_poset->write_file" << endl;
	}

	Factor_poset->write_file(fname2, 0 /*verbose_level*/);

	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"after Factor_poset->write_file" << endl;
	}




	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"before Factor_poset->LG->draw_with_options" << endl;
	}

	Factor_poset->LG->draw_with_options(
			base_name, LG_Draw_options,
			0 /* verbose_level */);

	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"after Factor_poset->LG->draw_with_options" << endl;
	}


	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"before FREE_OBJECT(Factor_poset)" << endl;
	}
	FREE_OBJECT(Factor_poset);
	if (f_v) {
		cout << "poset_classification::draw_poset_full "
				"after FREE_OBJECT(Factor_poset)" << endl;
	}
	
	if (f_v) {
		cout << "poset_classification::draw_poset_full done" << endl;
	}
}

void poset_classification::draw_poset(
		std::string &fname_base,
		int depth, int data,
		other::graphics::draw_options *LG_Draw_options,
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

	pc_convert_data_structure Pc_convert_data_structure;

	Pc_convert_data_structure.init(
			this,
			0 /* verbose_level*/);

	if (f_v) {
		cout << "poset_classification::draw_poset "
				"before make_auxiliary_graph" << endl;
	}
	Pc_convert_data_structure.make_auxiliary_graph(
			depth, LG1, data,
			0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "poset_classification::draw_poset "
				"before make_graph" << endl;
	}
	Pc_convert_data_structure.make_graph(
			depth, LG2, data, false /* f_tree */,
			0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "poset_classification::draw_poset "
				"before make_graph" << endl;
	}
	Pc_convert_data_structure.make_graph(
			depth, LG3, data, true /* f_tree */,
			0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "poset_classification::draw_poset "
				"before make_poset_graph_detailed" << endl;
	}
	Pc_convert_data_structure.make_poset_graph_detailed(
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
		other::graphics::draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::graph_theory::layered_graph *LG;

	if (f_v) {
		cout << "poset_classification::draw_level_graph "
				"data=" << data << endl;
	}

	pc_convert_data_structure Pc_convert_data_structure;

	Pc_convert_data_structure.init(
			this,
			0 /* verbose_level*/);

	Pc_convert_data_structure.make_level_graph(depth, LG, data, level, verbose_level - 1);


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



}}}


