/*
 * orbiter_symbol_table_entry.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace orbiter_kernel_system {



orbiter_symbol_table_entry::orbiter_symbol_table_entry()
{
	Record_birth();

	//std::string label;
	type = t_nothing;
	object_type = t_nothing_object;
	vec = NULL;
	vec_len = 0;
	//std::string str;
	ptr = NULL;
}

orbiter_symbol_table_entry::~orbiter_symbol_table_entry()
{
	Record_death();
	if (type == t_intvec && vec) {
		FREE_int(vec);
		vec = 0;
	}
	type = t_nothing;
	object_type = t_nothing_object;
}

void orbiter_symbol_table_entry::freeself()
{
	if (type == t_intvec && vec) {
		FREE_int(vec);
		vec = 0;
	}
	type = t_nothing;
	object_type = t_nothing_object;
}


void orbiter_symbol_table_entry::init(
		std::string &str_label)
{
	label.assign(str_label);
}

void orbiter_symbol_table_entry::init_finite_field(
		std::string &label,
		algebra::field_theory::finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_finite_field" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_finite_field;
	ptr = F;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_finite_field done" << endl;
	}
}

void orbiter_symbol_table_entry::init_polynomial_ring(
		std::string &label,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_polynomial_ring" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_polynomial_ring;
	ptr = HPD;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_polynomial_ring done" << endl;
	}
}

void orbiter_symbol_table_entry::init_any_group(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_any_group" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_any_group;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_any_group done" << endl;
	}
}

void orbiter_symbol_table_entry::init_linear_group(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_linear_group" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_linear_group;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_linear_group done" << endl;
	}
}

void orbiter_symbol_table_entry::init_permutation_group(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_permutation_group" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_permutation_group;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_permutation_group done" << endl;
	}
}

void orbiter_symbol_table_entry::init_modified_group(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_modified_group" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_modified_group;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_modified_group done" << endl;
	}
}

void orbiter_symbol_table_entry::init_projective_space(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_projective_space" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_projective_space;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_projective_space done" << endl;
	}
}

void orbiter_symbol_table_entry::init_orthogonal_space(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_orthogonal_space" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_orthogonal_space;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_orthogonal_space done" << endl;
	}
}

void orbiter_symbol_table_entry::init_BLT_set_classify(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_BLT_set_classify" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_BLT_set_classify;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_BLT_set_classify done" << endl;
	}
}

void orbiter_symbol_table_entry::init_spread_classify(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_spread_classify" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_spread_classify;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_spread_classify done" << endl;
	}
}

void orbiter_symbol_table_entry::init_cubic_surface(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_cubic_surface" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_cubic_surface;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_cubic_surface done" << endl;
	}
}

void orbiter_symbol_table_entry::init_quartic_curve(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_quartic_curve" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_quartic_curve;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_quartic_curve done" << endl;
	}
}

void orbiter_symbol_table_entry::init_BLT_set(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_BLT_set" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_BLT_set;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_BLT_set done" << endl;
	}
}

void orbiter_symbol_table_entry::init_classification_of_cubic_surfaces_with_double_sixes(
		std::string &label,
		void *p, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_classification_of_cubic_surfaces_with_double_sixes" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_classification_of_cubic_surfaces_with_double_sixes;
	ptr = p;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_classification_of_cubic_surfaces_with_double_sixes done" << endl;
	}

}

void orbiter_symbol_table_entry::init_collection(
		std::string &label,
		std::string &list_of_objects, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_collection" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_collection;

	data_structures::string_tools ST;

	std::vector<std::string> *the_list;
	the_list = new std::vector<std::string>;

	ST.parse_comma_separated_list(
			list_of_objects, *the_list,
			verbose_level);

	ptr = the_list;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_collection done" << endl;
	}
}

void orbiter_symbol_table_entry::init_geometric_object(
		std::string &label,
		geometry::other_geometry::geometric_object_create *COC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_geometric_object" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_geometric_object;
	ptr = COC;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_geometric_object done" << endl;
	}
}

void orbiter_symbol_table_entry::init_graph(
		std::string &label,
		void *Gr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_graph" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_graph;
	ptr = Gr;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_graph done" << endl;
	}
}

void orbiter_symbol_table_entry::init_code(
		std::string &label,
		void *Code, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_code" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_code;
	ptr = Code;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_code done" << endl;
	}
}

void orbiter_symbol_table_entry::init_spread(
		std::string &label,
		void *Spread, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_spread" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_spread;
	ptr = Spread;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_spread done" << endl;
	}
}

void orbiter_symbol_table_entry::init_translation_plane(
		std::string &label,
		void *Tp, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_translation_plane" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_translation_plane;
	ptr = Tp;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_translation_plane done" << endl;
	}
}




void orbiter_symbol_table_entry::init_spread_table(
		std::string &label,
		void *Spread_table_with_selection,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_spread_table" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_spread_table;
	ptr = Spread_table_with_selection;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_spread_table done" << endl;
	}
}

void orbiter_symbol_table_entry::init_packing_classify(
		std::string &label,
		void *Packing_classify, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_classify" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_packing_classify;
	ptr = Packing_classify;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_classify done" << endl;
	}
}



void orbiter_symbol_table_entry::init_packing_was(
		std::string &label,
		void *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_was" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_packing_was;
	ptr = P;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_was done" << endl;
	}
}


void orbiter_symbol_table_entry::init_packing_was_choose_fixed_points(
		std::string &label,
		void *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_was_choose_fixed_points" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_packing_was_choose_fixed_points;
	ptr = P;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_was_choose_fixed_points done" << endl;
	}
}


void orbiter_symbol_table_entry::init_packing_long_orbits(
		std::string &label,
		void *PL, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_long_orbits" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_packing_long_orbits;
	ptr = PL;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_packing_long_orbits done" << endl;
	}
}

void orbiter_symbol_table_entry::init_graph_classify(
		std::string &label,
		void *GC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_graph_classify" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_graph_classify;
	ptr = GC;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_graph_classify done" << endl;
	}
}

void orbiter_symbol_table_entry::init_diophant(
		std::string &label,
		void *Dio, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_diophant" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_diophant;
	ptr = Dio;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_diophant done" << endl;
	}
}

void orbiter_symbol_table_entry::init_design(
		std::string &label,
		void *DC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_design" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_design;
	ptr = DC;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_design done" << endl;
	}
}

void orbiter_symbol_table_entry::init_design_table(
		std::string &label,
		void *DT, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_design_table" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_design_table;
	ptr = DT;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_design_table done" << endl;
	}
}

void orbiter_symbol_table_entry::init_large_set_was(
		std::string &label,
		void *LSW, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_large_set_was" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_large_set_was;
	ptr = LSW;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_large_set_was done" << endl;
	}
}

void orbiter_symbol_table_entry::init_set(
		std::string &label,
		void *SB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_set" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_set;
	ptr = SB;
	if (f_v) {
		cout << "or\\biter_symbol_table_entry::init_set done" << endl;
	}
}

void orbiter_symbol_table_entry::init_vector(
		std::string &label,
		void *VB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_vector" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_vector;
	ptr = VB;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_vector done" << endl;
	}
}

void orbiter_symbol_table_entry::init_symbolic_object(
		std::string &label,
		void *SB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_symbolic_object" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_symbolic_object;
	ptr = SB;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_symbolic_object done" << endl;
	}
}

void orbiter_symbol_table_entry::init_combinatorial_object(
		std::string &label,
		void *Combo, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_combinatorial_object" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_combinatorial_object;
	ptr = Combo;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_combinatorial_object done" << endl;
	}
}

void orbiter_symbol_table_entry::init_geometry_builder_object(
		std::string &label,
		combinatorics::geometry_builder::geometry_builder *GB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_geometry_builder_object" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_geometry_builder;
	ptr = GB;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_geometry_builder_object done" << endl;
	}
}

void orbiter_symbol_table_entry::init_vector_ge(
		std::string &label,
		void *V, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_vector_ge" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_vector_ge;
	ptr = V;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_vector_ge done" << endl;
	}
}

void orbiter_symbol_table_entry::init_action_on_forms(
		std::string &label,
		void *AF, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_action_on_forms" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_action_on_forms;
	ptr = AF;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_action_on_forms done" << endl;
	}
}

void orbiter_symbol_table_entry::init_orbits(
		std::string &label,
		void *OC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_orbits" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_orbits;
	ptr = OC;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_orbits done" << endl;
	}
}

void orbiter_symbol_table_entry::init_poset_classification_control(
		std::string &label,
		void *PCC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_poset_classification_control" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_poset_classification_control;
	ptr = PCC;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_poset_classification_control done" << endl;
	}
}

void orbiter_symbol_table_entry::init_poset_classification_report_options(
		std::string &label,
		void *PCRO, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_poset_classification_report_options" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_poset_classification_report_options;
	ptr = PCRO;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_poset_classification_report_options done" << endl;
	}
}

void orbiter_symbol_table_entry::init_draw_options(
		std::string &label,
		graphics::layered_graph_draw_options *Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_draw_options" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_draw_options;
	ptr = Draw_options;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_draw_options done" << endl;
	}
}


void orbiter_symbol_table_entry::init_draw_incidence_structure_options(
		std::string &label,
		graphics::draw_incidence_structure_description *Draw_incidence_structure_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_draw_options" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_draw_incidence_structure_options;
	ptr = Draw_incidence_structure_description;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_draw_options done" << endl;
	}
}




void orbiter_symbol_table_entry::init_arc_generator_control(
		std::string &label,
		void *AGC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_arc_generator_control" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_arc_generator_control;
	ptr = AGC;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_arc_generator_control done" << endl;
	}
}

void orbiter_symbol_table_entry::init_poset_classification_activity(
		std::string &label,
		void *PCA, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_poset_classification_activity" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_poset_classification_activity;
	ptr = PCA;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_poset_classification_activity done" << endl;
	}
}

void orbiter_symbol_table_entry::init_crc_code(
		std::string &label,
		void *Code, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_crc_code" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_crc_code;
	ptr = Code;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_crc_code done" << endl;
	}
}

void orbiter_symbol_table_entry::init_mapping(
		std::string &label,
		void *Mapping, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_mapping" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_mapping;
	ptr = Mapping;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_mapping done" << endl;
	}
}

void orbiter_symbol_table_entry::init_variety(
		std::string &label,
		void *variety, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_variety" << endl;
	}
	orbiter_symbol_table_entry::label.assign(label);
	type = t_object;
	object_type = t_variety;
	ptr = variety;
	if (f_v) {
		cout << "orbiter_symbol_table_entry::init_variety done" << endl;
	}
}







void orbiter_symbol_table_entry::print()
{
	if (type == t_intvec) {
		Int_vec_print(cout, vec, vec_len);
		cout << endl;
	}
	else if (type == t_object) {

		// group of ten:

#if 0
		// group of ten:
		t_nothing_object,
		t_finite_field,
		t_polynomial_ring,
		t_any_group,
		t_linear_group,
		t_permutation_group,
		t_modified_group,
		t_projective_space,
		t_orthogonal_space,
		t_BLT_set_classify,
#endif



		if (object_type == t_nothing_object) {
			cout << "nothing object" << endl;
		}
		else if (object_type == t_finite_field) {
			algebra::field_theory::finite_field *F;

			F = (algebra::field_theory::finite_field *) ptr;
			F->Io->print();
		}
		else if (object_type == t_polynomial_ring) {
			algebra::ring_theory::homogeneous_polynomial_domain *HPD;

			HPD = (algebra::ring_theory::homogeneous_polynomial_domain *) ptr;
			HPD->print();
		}
		else if (object_type == t_any_group) {
			cout << "any group" << endl;
		}
		else if (object_type == t_linear_group) {
			cout << "linear group" << endl;
		}
		else if (object_type == t_permutation_group) {
			cout << "permutation group" << endl;
		}
		else if (object_type == t_modified_group) {
			cout << "modified group" << endl;
		}
		else if (object_type == t_projective_space) {
			cout << "projective space" << endl;
		}
		else if (object_type == t_orthogonal_space) {
			cout << "orthogonal space" << endl;
		}
		else if (object_type == t_BLT_set_classify) {
			cout << "classification object for BLT-sets" << endl;
		}


#if 0
		// group of ten:
		t_spread_classify,
		t_cubic_surface,
		t_quartic_curve,
		t_BLT_set,
		t_classification_of_cubic_surfaces_with_double_sixes,
		t_collection,
		t_geometric_object,
		t_graph,
		t_code,
		t_spread,
#endif


		else if (object_type == t_spread_classify) {
			cout << "spread classify" << endl;
		}
		else if (object_type == t_cubic_surface) {
			cout << "cubic surface" << endl;
		}
		else if (object_type == t_quartic_curve) {
			cout << "quartic curve" << endl;
		}
		else if (object_type == t_BLT_set) {
			cout << "BLT-set" << endl;
		}
		else if (object_type == t_classification_of_cubic_surfaces_with_double_sixes) {
			cout << "classification_of_cubic_surfaces_with_double_sixes" << endl;
		}
		else if (object_type == t_collection) {
			cout << "collection" << endl;
			std::vector<std::string> *the_list;
			int i;

			the_list = (std::vector<std::string> *) ptr;
			for (i = 0; i < the_list->size(); i++) {
				cout << i << " : " << (*the_list)[i] << endl;
			}
		}
		else if (object_type == t_geometric_object) {
			cout << "geometric object" << endl;
		}
		else if (object_type == t_graph) {
			cout << "graph" << endl;
		}
		else if (object_type == t_code) {
			cout << "code" << endl;
		}
		else if (object_type == t_spread) {
			cout << "spread" << endl;
		}


#if 0
		// group of ten:
		t_translation_plane,
		t_spread_table,
		t_packing_classify,
		t_packing_was,
		t_packing_was_choose_fixed_points,
		t_packing_long_orbits,
		t_graph_classify,
		t_diophant,
		t_design,
		t_design_table,
#endif



		else if (object_type == t_translation_plane) {
			cout << "translation plane" << endl;
		}
		else if (object_type == t_spread_table) {
			cout << "spread table" << endl;
		}
		else if (object_type == t_packing_classify) {
			cout << "packing classify" << endl;
		}
		else if (object_type == t_packing_was) {
			cout << "packing with symmetry assumption" << endl;
		}
		else if (object_type == t_packing_was_choose_fixed_points) {
			cout << "packing with symmetry assumption, choice of fixed points" << endl;
		}
		else if (object_type == t_packing_long_orbits) {
			cout << "packing with symmetry assumption, choosing long orbits" << endl;
		}
		else if (object_type == t_graph_classify) {
			cout << "graph_classification" << endl;
		}
		else if (object_type == t_diophant) {
			cout << "diophant" << endl;
		}
		else if (object_type == t_design) {
			cout << "design" << endl;
		}
		else if (object_type == t_design_table) {
			cout << "design_table" << endl;
		}



#if 0
		// group of ten:
		t_large_set_was,
		t_set,
		t_vector,
		t_symbolic_object,
		t_combinatorial_object,
		t_geometry_builder,
		t_vector_ge,
		t_action_on_forms,
		t_orbits,
		t_poset_classification_control,
#endif

		else if (object_type == t_large_set_was) {
			cout << "large_set_was" << endl;
		}
		else if (object_type == t_set) {
			cout << "set" << endl;
		}
		else if (object_type == t_vector) {
			cout << "vector : ";

			data_structures::vector_builder *VB;

			VB = (data_structures::vector_builder *) ptr;
			VB->print(cout);
		}
		else if (object_type == t_symbolic_object) {
			cout << "symbolic object" << endl;
		}
		else if (object_type == t_combinatorial_object) {
			cout << "combinatorial_object" << endl;
		}
		else if (object_type == t_geometry_builder) {
			cout << "geometry_builder" << endl;
		}
		else if (object_type == t_vector_ge) {
			cout << "vector_ge" << endl;
		}
		else if (object_type == t_action_on_forms) {
			cout << "action_on_forms" << endl;
		}
		else if (object_type == t_orbits) {
			cout << "orbits" << endl;
		}
		else if (object_type == t_poset_classification_control) {
			cout << "poset_classification_control" << endl;
		}


#if 0
		// group of 8:
		t_poset_classification_report_options,
		t_draw_options,
		t_draw_incidence_structure_options,
		t_arc_generator_control,
		t_poset_classification_activity,
		t_crc_code,
		t_mapping,
		t_variety,
#endif


		else if (object_type == t_poset_classification_report_options) {
			cout << "poset_classification_report_options" << endl;
		}
		else if (object_type == t_draw_options) {
			cout << "draw options" << endl;
		}
		else if (object_type == t_draw_incidence_structure_options) {
			cout << "draw incidence structure options" << endl;
		}
		else if (object_type == t_arc_generator_control) {
			cout << "arc_generator_control" << endl;
		}
		else if (object_type == t_poset_classification_activity) {
			cout << "poset_classification_activity" << endl;
		}
		else if (object_type == t_crc_code) {
			cout << "crc_code" << endl;
		}
		else if (object_type == t_mapping) {
			cout << "mapping" << endl;
		}
		else if (object_type == t_variety) {
			cout << "variety" << endl;
		}

	}
}


}}}}


