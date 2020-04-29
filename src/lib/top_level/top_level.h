// top_level.h
//
// Anton Betten
//
// started:  September 23 2010
//
// based on global.h, which was taken from reader.h: 3/22/09



using namespace orbiter::foundations;
using namespace orbiter::group_actions;
using namespace orbiter::classification;
using namespace orbiter::discreta;


namespace orbiter {

//! classes for combinatorial objects and their classification

namespace top_level {


class representatives;
	// added 7/3/12
class isomorph;
	// added 3/22/09
class search_blocking_set;
	// added Nov 2, 2010
class choose_points_or_lines;
	// added Nov 29, 2010
class subspace_orbits;
	// added March 29, 2012 (started Jan 25, 2010)
struct factor_group;
class orbit_of_sets;
	// added March 27, 2013
class singer_cycle;
	// added March 27, 2013
class exact_cover;
	// added April 30, 2013
class recoordinatize;
	// added November 2, 2013
class spread_classify;
	// added November 2, 2013
class polar;
class orbit_of_subspaces;
	// added April 10, 2014
class young;
	// added March 16, 2015
class exact_cover_arguments;
	// added January 12, 2016
class translation_plane_via_andre_model;
	// added January 25, 2016
class isomorph_arguments;
	// added January 27, 2016
struct isomorph_worker_data;
class surface_classify_wedge;
	// added September 2, 2016
class arc_generator;
	// moved here February 23, 2017
class surface_with_action;
	// added March 22, 2017
class surface_object_with_action;
	// added October 4, 2017
class classify_trihedral_pairs;
	// added October 9, 2017
class classify_double_sixes;
	// added October 10, 2017
class surface_create_description;
	// added January 14, 2018
class surface_create;
	// added January 14, 2018
class arc_lifting;
	// added January 14, 2018
class six_arcs_not_on_a_conic;
	// added March 6, 2018
class BLT_set_create_description;
	// added March 17, 2018
class BLT_set_create;
	// added March 17, 2018
class spread_create_description;
	// added March 22, 2018
class spread_create;
	// added March 22, 2018
class spread_lifting;
	// added April 1, 2018
class k_arc_generator;
class arc_lifting_simeon;
	// added Jan 6, 2019
class blt_set_classify;
	// started 8/13/2006, added Jan 6, 2019
class surfaces_arc_lifting;
	// started 1/9/2019
class arc_orbits_on_pairs;
	// started 1/9/2019
class arc_partition;
	// started 1/9/2019
class packing_classify;
class packing_invariants;
class invariants_packing;
class classify_cubic_curves;
class cubic_curve_with_action;
class blt_set_with_action;
class semifield_classify;
class semifield_level_two;
class semifield_lifting;
class semifield_downstep_node;
class semifield_flag_orbit_node;
class semifield_trace;
class trace_record;
class semifield_substructure;
class semifield_classify_with_substructure;
class packing_was;
class packing_long_orbits;
class tactical_decomposition;
class tensor_classify;
class design_create;
class design_create_description;
class large_set_classify;
class code_classify;
class ovoid_classify;
class graph_generator;
class regular_ls_classify;
class difference_set_in_heisenberg_group;
class cayley_graph_search;
class hadamard_classify;
class linear_set_classify;
class delandtsheer_doyen;
class boolean_function;
class hermitian_spread_classify;
class surface_study;
class pentomino_puzzle;
class character_table_burnside;
class create_graph_description;
class create_graph;
class algebra_global_with_action;
class group_theoretic_activity_description;
class graph_theoretic_activity_description;


// #############################################################################
// representatives.cpp
// #############################################################################

//! auxiliary class for class isomorph



class representatives {
public:
	action *A;

	char prefix[1000];
	char fname_rep[1000];
	char fname_stabgens[1000];
	char fname_fusion[1000];
	char fname_fusion_ge[1000];



	// flag orbits:
	int nb_objects;
	int *fusion; // [nb_objects]
		// fusion[i] == -2 means that the flag orbit i
		// has not yet been processed by the
		// isomorphism testing procedure.
		// fusion[i] = i means that flag orbit [i]
		// in an orbit representative
		// Otherwise, fusion[i] is an earlier flag_orbit,
		// and handle[i] is a group element that maps
		// to it
	int *handle; // [nb_objects]
		// handle[i] is only relevant if fusion[i] != i,
		// i.e., if flag orbit i is not a representative
		// of an isomorphism type.
		// In this case, handle[i] is the (handle of a)
		// group element moving flag orbit i to flag orbit fusion[i].


	// classified objects:
	int count;
	int *rep; // [count]
	sims **stab; // [count]



	//char *elt;
	int *Elt1;
	int *tl; // [A->base_len]

	int nb_open;
	int nb_reps;
	int nb_fused;


	representatives();
	void null();
	~representatives();
	void free();
	void init(action *A, int nb_objects, char *prefix, int verbose_level);
	void write_fusion(int verbose_level);
	void read_fusion(int verbose_level);
	void write_representatives_and_stabilizers(int verbose_level);
	void read_representatives_and_stabilizers(int verbose_level);
	void save(int verbose_level);
	void load(int verbose_level);
	void calc_fusion_statistics();
	void print_fusion_statistics();
};

}}


#include "./algebra_and_number_theory/tl_algebra_and_number_theory.h"
#include "./combinatorics/tl_combinatorics.h"
#include "./geometry/tl_geometry.h"
#include "./isomorph/isomorph.h"
#include "./orbits/orbits.h"
#include "./solver/solver.h"

