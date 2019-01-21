// top_level.h
//
// Anton Betten
//
// started:  September 23 2010
//
// based on global.h, which was taken from reader.h: 3/22/09


namespace orbiter {


typedef class representatives representatives;
	// added 7/3/12
typedef class isomorph isomorph;
	// added 3/22/09
typedef class search_blocking_set search_blocking_set;
	// added Nov 2, 2010
typedef class choose_points_or_lines choose_points_or_lines;
	// added Nov 29, 2010
typedef class subspace_orbits subspace_orbits;
	// added March 29, 2012 (started Jan 25, 2010)
typedef struct factor_group factor_group;
typedef class orbit_rep orbit_rep;
typedef class orbit_of_sets orbit_of_sets;
	// added March 27, 2013
typedef class singer_cycle singer_cycle;
	// added March 27, 2013
typedef class exact_cover exact_cover;
	// added April 30, 2013
typedef class recoordinatize recoordinatize;
	// added November 2, 2013
typedef class spread spread;
	// added November 2, 2013
typedef class polar polar;
typedef class orbit_of_subspaces orbit_of_subspaces;
	// added April 10, 2014
typedef class young young;
	// added March 16, 2015
typedef class exact_cover_arguments exact_cover_arguments;
	// added January 12, 2016
typedef class translation_plane_via_andre_model 
	translation_plane_via_andre_model;
	// added January 25, 2016
typedef class isomorph_arguments isomorph_arguments;
	// added January 27, 2016
typedef struct isomorph_worker_data isomorph_worker_data;
typedef class surface_classify_wedge surface_classify_wedge;
	// added September 2, 2016
typedef class arc_generator arc_generator;
	// moved here February 23, 2017
typedef class surface_with_action surface_with_action;
	// added March 22, 2017
typedef class surface_object_with_action surface_object_with_action;
	// added October 4, 2017
typedef class classify_trihedral_pairs classify_trihedral_pairs;
	// added October 9, 2017
typedef class classify_double_sixes classify_double_sixes;
	// added October 10, 2017
typedef class surface_create_description surface_create_description;
	// added January 14, 2018
typedef class surface_create surface_create;
	// added January 14, 2018
typedef class arc_lifting arc_lifting;
	// added January 14, 2018
typedef class six_arcs_not_on_a_conic six_arcs_not_on_a_conic;
	// added March 6, 2018
typedef class BLT_set_create_description BLT_set_create_description;
	// added March 17, 2018
typedef class BLT_set_create BLT_set_create;
	// added March 17, 2018
typedef class spread_create_description spread_create_description;
	// added March 22, 2018
typedef class spread_create spread_create;
	// added March 22, 2018
typedef class spread_lifting spread_lifting;
	// added April 1, 2018
typedef class k_arc_generator k_arc_generator;
typedef class arc_lifting_simeon arc_lifting_simeon;
	// added Jan 6, 2019
typedef class blt_set blt_set;
	// started 8/13/2006, added Jan 6, 2019
typedef class surfaces_arc_lifting surfaces_arc_lifting;
	// started 1/9/2019
typedef class arc_orbits_on_pairs arc_orbits_on_pairs;
	// started 1/9/2019
typedef class arc_partition arc_partition;
	// started 1/9/2019


// #############################################################################
// representatives.C
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

}


#include "./algebra_and_number_theory/tl_algebra_and_number_theory.h"
#include "./geometry/tl_geometry.h"
#include "./isomorph/isomorph.h"
#include "./orbits/orbits.h"
#include "./solver/solver.h"

