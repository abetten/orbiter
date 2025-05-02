// foundations.h
//
// Anton Betten
//
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005
// renamed from galois.h: August 16, 2018


#ifndef ORBITER_SRC_LIB_FOUNDATIONS_FOUNDATIONS_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_FOUNDATIONS_H_



// History:
//
// added class unipoly_domain: November 16, 2002
// added class finite_field: October 23, 2002
// added class longinteger: October 26, 2002
// added class mp: March 6, 2003
// added partitionstack: July 3 2007
// added class orthogonal: July 9 2007
// added class vector_hashing: October 14, 2008
// added file tensor: Dec 25 2008
// added class grassmann: June 5, 2009
// added class unusual: June 10 2009
// added file memory: June 25, 2009
// added file geometry: July 9, 2009
// added class classify: Oct 31, 2009
// added class grassmann_embedded: Jan 24, 2010
// added class hermitian: March 19, 2010
// added class incidence_structure: June 20, 2010
// added class finite_ring: June 21, 2010
// added class hjelmslev: June 22, 2010
// added class fancy_set: June 29, 2010
// added class norm_tables: Sept 23, 2010 (started 11/28/2008)
// added struct grid_frame: Sept 8, 2011
// added class data_file: Oct 13, 2011
// added class subfield_structure: November 14, 2011
// added class clique_finder: December 13, 2011
// added class colored_graph: October 28, 2012
// added class rainbow_cliques: October 28, 2012
// added class set_of_sets: November 30, 2012
// added class decomposition: December 1, 2012
// added file dlx.cpp: April 7, 2013
// added class spreadsheet: March 15, 2013
// added class andre_construction andre_construction: June 2, 2013
// added class andre_construction_point_element: June 2, 2013
// added class andre_construction_line_element: June 2, 2013
// added class int_matrix: October 23, 2013
// added class gl_classes: October 23, 2013
// added class layered_graph: January 6, 2014
// added class graph_layer: January 6, 2014
// added class graph_node: January 6, 2014
// added class int_vector: August 12, 2014
// added class projective_space (moved here from ACTION): December 31, 2014
// added class buekenhout_metz (moved here from TOP_LEVEL): December 31, 2014
// added class a_domain March 14, 2015
// added class diophant (moved here from INCIDENCE) April 16, 2015
// added class null_polarity_generator December 11, 2015
// added class layered_graph_draw_options December 15, 2015
// added class klein_correspondence January 1, 2016
// added class file_output January 8, 2016
// added class generators_symplectic_group March 29, 2016
// added class flag May 20, 2016
// moved class knarr from TOP_LEVEL: Jul 29, 2016
// moved class w3q from TOP_LEVEL: Jul 29, 2016
// moved class surface from TOP_LEVEL: Aug 1, 2016
// added class homogeneous_polynomial_domain: Sept 9, 2016
// added class eckardt_point: January 12, 2017
// added class surface_object: March 18, 2017
// added class vector_space: December 2, 2018

#include <iostream>
#include <fstream>
#include <iomanip>

#include <cstring>
#include <math.h>
#include <limits.h>
#include <sstream>

#include <map>
#include <vector>
#include <deque>
#include <string>
#include <unordered_set>
#include <algorithm>
	// for sort


#include <cstdint>
	// for uint32_t



/*--------------------------------------------------------------------*/
/// Define some ANSI colour codes
/*--------------------------------------------------------------------*/
#if __cplusplus >= 201103L
#define COLOR_UNICODE "\u001b"
#else
#define COLOR_UNICODE "\x1b"
#endif

#define RESET_COLOR_SCHEME COLOR_UNICODE "[0m"
#define BLACK COLOR_UNICODE "[30m"
#define RED COLOR_UNICODE "[31m"
#define GREEN COLOR_UNICODE "[32m"
#define YELLOW COLOR_UNICODE "[33m"
#define BLUE COLOR_UNICODE "[34m"
#define MAGENTA COLOR_UNICODE "[35m"
#define CYAN COLOR_UNICODE "[36m"
#define WHITE COLOR_UNICODE "[37m"

#define BRIGHT_BLACK COLOR_UNICODE "[30;1m"
#define BRIGHT_RED COLOR_UNICODE "[31;1m"
#define BRIGHT_GREEN COLOR_UNICODE "[32;1m"
#define BRIGHT_YELLOW COLOR_UNICODE "[33;1m"
#define BRIGHT_BLUE COLOR_UNICODE "[34;1m"
#define BRIGHT_MAGENTA COLOR_UNICODE "[35;1m"
#define BRIGHT_CYAN COLOR_UNICODE "[36;1m"
#define BRIGHT_WHITE COLOR_UNICODE "[37;1m"
/*--------------------------------------------------------------------*/


/*--------------------------------------------------------------------*/
/// The following code block identifies the current operating system the code is
/// being executed on and turns on specific macros in order to use system calls
/// defined by that operating system.
/*--------------------------------------------------------------------*/
#if defined(unix) || defined(__unix) || defined(__unix__)
#define SYSTEMUNIX
#endif

#if defined(_WIN32) || defined(_WIN64)
#define SYSTEMWINDOWS
#endif

#if defined(__APPLE__) || defined(__MACH__)
#define SYSTEMUNIX
#define SYSTEM_IS_MACINTOSH
	// use Mac specific stuff like asking how much memory the process uses.
#endif

#if defined(__linux__) || defined(linux) || defined(__linux)
#define SYSTEM_LINUX
#endif
/*--------------------------------------------------------------------*/

#define SYSTEMUNIX


#define HAS_NAUTY 1

// need to be defined in nauty_interface.cpp also.


#ifdef SYSTEMWINDOWS
//#pragma warning(disable : 4996)
#include <string>
#endif

#define MEMORY_DEBUG

#define MAGIC_SYNC 762873656L



// define exactly one of the following to match your system:
#undef int_HAS_2_charS
#define int_HAS_4_charS
#undef int_HAS_8_charS

#ifdef int_HAS_2_charS
typedef short int_2;
typedef long int_4;
typedef long int_8;
typedef unsigned short uint_2;
typedef unsigned long uint_4;
typedef unsigned long uint_8;
#endif
#ifdef int_HAS_4_charS
typedef short int_2;
typedef int int_4;
typedef long int_8;
typedef unsigned short uint_2;
typedef unsigned int uint_4;
typedef unsigned long uint_8;
#endif
#ifdef int_HAS_8_charS
typedef short int_2;
typedef short int int_4;
typedef int int_8;
typedef unsigned short uint_2;
typedef unsigned short int uint_4;
typedef unsigned int uint_8;
#endif


typedef int *pint;
typedef long int *plint;
typedef int **ppint;
typedef long int **pplint;
typedef char *pchar;
typedef unsigned char uchar;
typedef uchar *puchar;
typedef void *pvoid;



#define PAGE_LENGTH_LOG 20
#define PAGE_STORAGE_MAX_PAGE_SIZE (5 * 1L << 20)
#define BUFSIZE 100000


#define MINIMUM(x, y)   ( ((x) < (y)) ?  (x) : (y) )
#define MAXIMUM(x, y)   ( ((x) > (y)) ?  (x) : (y) )
#define MIN(x, y)   ( ((x) < (y)) ?  (x) : (y) )
#define MAX(x, y)   ( ((x) > (y)) ?  (x) : (y) )
#define ABS(x)      ( ((x) <  0 ) ? (-(x)) : (x) )
#define EVEN(x)     ( ((x) % 2) == 0 )
#define ODD(x)      ( ((x) % 2) == 1 )
#define DOUBLYEVEN(x)     ( ((x) % 4) == 0 )
#define SINGLYEVEN(x)     ( ((x) % 4) == 2 )
#define ONE_MILLION 1000000
#define ONE_HUNDRED_THOUSAND 100000


#ifndef true
#define true 1
#endif
#ifndef false
#define false 0
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846264
#endif


//! the orbiter library for the classification of combinatorial objects

namespace orbiter {



//! algebra, combinatorics and graph theory, geometry, linear algebra, number theory, data structures, solvers, graphics; no group actions

namespace layer1_foundations {


//! algebra, linear algebra, ring theory and number theory

namespace algebra {

	//! basic algebraic algorithms, generators for groups, conjugacy classes in the general linear group

	namespace basic_algebra {

		// algebra:
		class a_domain;
		class algebra_global;
		class generators_symplectic_group;
		class group_generators_domain;
		class heisenberg;
		class matrix_group_element;
		class matrix_group;
		class module;
		class null_polarity_generator;
		class rank_checker;
	}

	//! expression parser, used to create an abstract syntax tree (AST) of a well-formed algebraic expression

	namespace expression_parser {

		// expression_parser:
		class formula_vector;
		class formula;
		class symbolic_object_activity_description;
		class symbolic_object_activity;
		class symbolic_object_builder_description;
		class symbolic_object_builder;
		class syntax_tree_node_terminal;
		class syntax_tree_node;
		class syntax_tree;
		class syntax_tree_latex;
	}

	//! finite fields, n-th roots, subfields, trace and norm.

	namespace field_theory {

		// finite_fields:
		class finite_field_activity_description;
		class finite_field_activity;
		class finite_field_description;
		class finite_field_io;
		class finite_field_properties;
		class finite_field;
		class finite_field_implementation_by_tables;
		class finite_field_implementation_wo_tables;
		class minimum_polynomial;
		class norm_tables;
		class normal_basis;
		class nth_roots;
		class related_fields;
		class square_nonsquare;
		class subfield_structure;
	}

	//! linear algebra and representation theory

	namespace linear_algebra {

		// linear_algebra:
		class gl_class_rep;
		class gl_classes;
		class linear_algebra;
		class matrix_block_data;
		class representation_theory_domain;
		class vector_space;

	}

	//! number theory, cyclotomic sets, elliptic curves, number theoretic transform (NTT)

	namespace number_theory {

		// number_theory:
		class cyclotomic_sets;
		class elliptic_curve;
		class number_theoretic_transform;
		class number_theory_domain;

	}


	//! ring theory, including polynomial rings and longinteger arithmetic.

	namespace ring_theory {
		// ring_theory:
		class finite_ring;
		class homogeneous_polynomial_domain;
		class longinteger_domain;
		class longinteger_object;
		class partial_derivative;
		class polynomial_double_domain;
		class polynomial_double;
		class polynomial_ring_activity_description;
		class polynomial_ring_description;
		class ring_theory_global;
		class table_of_irreducible_polynomials;
		class unipoly_domain;

		typedef ring_theory::longinteger_object *plonginteger_object;
		typedef void *unipoly_object;

	}


}



//! combinatorics: coding theory, graph theory, special functions, combinatorial objects, classification, tactical decompositions, various puzzles

namespace combinatorics {


	//! classification of combinatorial objects using canonical forms and Nauty

	namespace canonical_form_classification {

		class any_combinatorial_object;
		class classification_of_objects_description;
		class classification_of_objects;
		class classify_bitvectors;
		class classify_using_canonical_forms;
		class data_input_stream_description_element;
		class data_input_stream_description;
		class data_input_stream_output;
		class data_input_stream;
		class encoded_combinatorial_object;
		class objects_report_options;

	}


	//! coding theory, MacWilliams, weight enumerators, cyclic codes, BCH codes, Reed-Muller codes, etc.

	namespace coding_theory {

		// coding_theory:
		class code_diagram;
		class coding_theory_domain;
		class crc_code_description;
		class crc_codes;
		class crc_object;
		class crc_options_description;
		class create_BCH_code;
		class create_RS_code;
		class error_pattern_generator;
		class error_repository;
		class ttp_codes;

	}


	//! cryptography: Vigenere, Caesar, RSA, primality tests, elliptic curve, NTRU, square roots modulo n.

	namespace cryptography {

		// cryptography
		class cryptography_domain;

	}


	//! design theory

	namespace design_theory {

	class design_object;
	class design_theory_global;

	}


	//! construction and classification of configurations, linear spaces, and designs

	namespace geometry_builder {

		// geometry_builder:
		class cperm;
		class decomposition_with_fuse;
		class gen_geo_conf;
		class gen_geo;
		class geometric_backtrack_search;
		class geometry_builder_description;
		class geometry_builder;
		class girth_test;
		class inc_encoding;
		class incidence;
		class iso_type;
		class test_semicanonical;

	}



	//! graph theory: constructions, clique finding, drawing

	namespace graph_theory {

		// graph_theory
		class clique_finder_control;
		class clique_finder;
		class colored_graph_cliques;
		class colored_graph;
		class graph_layer;
		class graph_node;
		class graph_theory_domain;
		class graph_theory_subgraph_search;
		class layered_graph;
		class rainbow_cliques;

	}

	//! database of mathematical objects

	namespace knowledge_base {

		// knowledge_base:
		class knowledge_base;

	}



	//! general combinatorics

	namespace other_combinatorics {
		class combinatorics_domain;
	}

	//! combinatorial puzzles

	namespace puzzles {

		class brick_domain;
		class domino_assignment;
		class domino_change;
		class pentomino_puzzle;

	}

	//! Solvers for diophantine systems of equations.

	namespace solvers {

		class diophant_activity_description;
		class diophant_activity;
		class diophant_create;
		class diophant_description;
		class diophant;
		class dlx_problem_description;
		class dlx_solver;
		struct dlx_node;
		typedef struct dlx_node *pdlx_node;
	}



	//! combinatorics: special functions

	namespace special_functions {

		class apn_functions;
		class boolean_function_domain;
		class permutations;
		class polynomial_function_domain;
		class special_functions_domain;

	}

	//! tactical decompositions of incidence structures

	namespace tactical_decompositions {

		class decomposition_scheme;
		class decomposition;
		class geo_parameter;
		class row_and_col_partition;
		class tactical_decomposition_domain;
		class tdo_data;
		class tdo_refinement_description;
		class tdo_refinement;
		class tdo_scheme_compute;
		class tdo_scheme_synthetic;

		struct solution_file_data;

	}


}




//! projective geometry and other finite geometries

namespace geometry {

	//! cubic surfaces, quartic curves, Schlaefli labelings, Eckardt points,  Del Pezzo surfaces, Clebsch maps

	namespace algebraic_geometry {

		class algebraic_geometry_global;
		class arc_lifting_with_two_lines;
		class clebsch_map;
		class cubic_curve;
		class del_pezzo_surface_of_degree_two_domain;
		class del_pezzo_surface_of_degree_two_object;
		class eckardt_point_info;
		class eckardt_point;
		class kovalevski_points;
		class quartic_curve_domain;
		class quartic_curve_object_properties;
		class quartic_curve_object;
		class schlaefli_double_six;
		class schlaefli_labels;
		class schlaefli_trihedral_pairs;
		class schlaefli_tritangent_planes;
		class schlaefli;
		class seventytwo_cases;
		class smooth_surface_object_properties;
		class surface_domain;
		class surface_object_properties;
		class surface_object;
		class surface_polynomial_domains;
		class veriety_description;
		class variety_object;
		class web_of_cubic_curves;

	}


	//! generalized quadrangles, spreads, translation planes

	namespace finite_geometries {

		// geometry:
		class andre_construction_line_element;
		class andre_construction_point_element;
		class andre_construction;
		class buekenhout_metz;
		class desarguesian_spread;
		class knarr;
		class spread_domain;
		class spread_tables;
		class W3q;

		}

	//! orthogonal geometry: quadrics, BLT sets

	namespace orthogonal_geometry {

		// orthogonal:
		class blt_set_domain;
		class blt_set_invariants;
		class linear_complex;
		class orthogonal_global;
		class orthogonal_group;
		class orthogonal_indexing;
		class orthogonal_plane_invariant;
		class orthogonal;
		class quadratic_form_list_coding;
		class quadratic_form;
		class unusual_model;

	}

	//! other geometries or related topics

	namespace other_geometry {

		class arc_basic;
		class arc_in_projective_space;
		class flag;
		class geometric_object_create;
		class geometric_object_description;
		class geometry_global;
		class hermitian;
		class hjelmslev;
		class incidence_structure;
		class intersection_type;
		class point_line;
		class points_and_lines;
		class three_skew_subspaces;
	}

	//! projective geometry over a finite field and related topics


	namespace projective_geometry {

		class grassmann_embedded;
		class grassmann;
		class klein_correspondence;
		class polarity;
		class projective_space_basic;
		class projective_space_implementation;
		class projective_space_of_dimension_three;
		class projective_space_plane;
		class projective_space_reporting;
		class projective_space_subspaces;
		class projective_space;

	}


}

//! other things at level 1

namespace other {


	//! basic data structures used throughout the project

	namespace data_structures {

		// data_structures:
		class algorithms;
		class ancestry_family;
		class ancestry_indi;
		class ancestry_tree;
		class bitmatrix;
		class bitvector;
		class data_file;
		class data_structures_global;
		class fancy_set;
		class forest;
		class int_matrix;
		class int_vec;
		class int_vector;
		class lint_vec;
		class page_storage;
		class partitionstack;
		class set_builder_description;
		class set_builder;
		class set_of_sets_lint;
		class set_of_sets;
		class sorting;
		class spreadsheet;
		class string_tools;
		class tally_lint;
		class tally_vector_data;
		class tally;
		class text_builder_description;
		class text_builder;
		class vector_builder_description;
		class vector_builder;
		class vector_hashing;

	}



	//! graphical output interfaces: 2D graphics (BMP, TikZ, Metapost) and 3D graphics (povray)

	namespace graphics {

		// graphics:
		class animate;
		class draw_bitmap_control;
		class draw_incidence_structure_description;
		class draw_mod_n_description;
		class draw_projective_curve_description;
		class drawable_set_of_objects;
		class graphical_output;
		class layered_graph_draw_options;
		class mp_graphics;
		class parametric_curve_point;
		class parametric_curve;
		class plot_tools;
		class povray_job_description;
		class scene_element_of_type_edge;
		class scene_element_of_type_face;
		class scene_element_of_type_line;
		class scene_element_of_type_plane;
		class scene_element_of_type_point;
		class scene_element_of_type_surface;
		class scene;
		class tree_draw_options;
		class tree;
		class tree_node;
		class video_draw_options;
		// pointer types
		typedef tree_node *ptree_node;

	}


	//! interface to external software, including to Sajeeb's expression parser

	namespace l1_interfaces {

		class easy_BMP_interface;
		class eigen_interface;
		class expression_parser_sajeeb;
		class gnuplot_interface;
		class interface_gap_low;
		class interface_magma_low;
		class latex_interface;
		class nauty_interface_control;
		class nauty_interface_for_combo;
		class nauty_interface;
		class nauty_output;
		class povray_interface;
		class pugixml_interface;

	}



	//! the Orbiter kernel. It contains functions related to the symbol-table, memory management, os-interface, file-io, latex-interface etc.

	namespace orbiter_kernel_system {

		enum symbol_table_entry_type {
			t_nothing,
			t_intvec,
			t_object,
			t_string,
		};


		enum symbol_table_object_type {

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

			// group of ten:
			t_large_set_was,
			t_set,
			t_vector,
			t_text,
			t_symbolic_object,
			t_combinatorial_object,
			t_geometry_builder,
			t_vector_ge,
			t_action_on_forms,
			t_orbits,

			// group of ten:
			t_poset_classification_control,
			t_poset_classification_report_options,
			t_draw_options,
			t_draw_incidence_structure_options,
			t_arc_generator_control,
			t_poset_classification_activity,
			t_crc_code,
			t_mapping,
			t_variety,
			t_combo_with_group,


			// group of 2:
			t_isomorph_arguments,
			t_classify_cubic_surfaces,

		};
		// please maintain:
		// orbiter_kernel_system::orbiter_symbol_table::stringify_type
		// orbiter_kernel_system::orbiter_symbol_table_entry::print


		class activity_output;
		class create_file_description;
		class csv_file_support;
		class file_io;
		class file_output;
		class mem_object_registry_entry;
		class mem_object_registry;
		class memory_object;
		class numerics;
		class orbiter_data_file;
		class orbiter_session;
		class orbiter_symbol_table_entry;
		class orbiter_symbol_table;
		class os_interface;
		class override_double;
		class prepare_frames;

	}




	//! expressions in reverse Polish notation

	namespace polish {

		class function_command;
		class function_polish_description;
		class function_polish;

	}




}











#ifdef MEMORY_DEBUG
#define NEW_int(n) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_int(n, __FILE__, __LINE__)
#define NEW_int_with_tracking(n, file, line) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_int(n, file, line)
#define NEW_pint(n) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_pint(n, __FILE__, __LINE__)
#define NEW_lint(n) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_lint(n, __FILE__, __LINE__)
#define NEW_plint(n) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_plint(n, __FILE__, __LINE__)
#define NEW_ppint(n) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_ppint(n, __FILE__, __LINE__)
#define NEW_pplint(n) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_pplint(n, __FILE__, __LINE__)
#define NEW_char(n) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_char(n, __FILE__, __LINE__)
#define NEW_char_with_tracking(n, file, line) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_char(n, file, line)
#define NEW_uchar(n) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_uchar(n, __FILE__, __LINE__)
#define NEW_pchar(n) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_pchar(n, __FILE__, __LINE__)
#define NEW_puchar(n) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_puchar(n, __FILE__, __LINE__)
#define NEW_pvoid(n) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_pvoid(n, __FILE__, __LINE__)
#define NEW_OBJECT(type) (type *)other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_OBJECT(new type, (std::size_t) sizeof(type), #type, __FILE__, __LINE__)
#define NEW_OBJECTS(type, n) (type *)other::orbiter_kernel_system::Orbiter->global_mem_object_registry->allocate_OBJECTS(new type[n], n, (std::size_t) sizeof(type), #type, __FILE__, __LINE__)
#define FREE_int(p) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_int(p, __FILE__, __LINE__)
#define FREE_pint(p) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_pint(p, __FILE__, __LINE__)
#define FREE_lint(p) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_lint(p, __FILE__, __LINE__)
#define FREE_plint(p) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_plint(p, __FILE__, __LINE__)
#define FREE_ppint(p) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_ppint(p, __FILE__, __LINE__)
#define FREE_pplint(p) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_pplint(p, __FILE__, __LINE__)
#define FREE_char(p) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_char(p, __FILE__, __LINE__)
#define FREE_uchar(p) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_uchar(p, __FILE__, __LINE__)
#define FREE_pchar(p) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_pchar(p, __FILE__, __LINE__)
#define FREE_puchar(p) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_puchar(p, __FILE__, __LINE__)
#define FREE_pvoid(p) other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_pvoid(p, __FILE__, __LINE__)
#define FREE_OBJECT(p) {other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_OBJECT(p, __FILE__, __LINE__); delete p;}
#define FREE_OBJECTS(p) {other::orbiter_kernel_system::Orbiter->global_mem_object_registry->free_OBJECTS(p, __FILE__, __LINE__); delete [] p;}
#else
#define NEW_int(n) new int[n]
#define NEW_int_with_tracking(n, file, line) new int[n]
#define NEW_pint(n) new pint[n]
#define NEW_lint(n) new long int[n]
#define NEW_lint(n) new (long int *)[n]
#define NEW_ppint(n) new ppint[n]
#define NEW_pplint(n) new pplint[n]
#define NEW_char(n) new char[n]
#define NEW_char_with_tracking(n, file, line) new char[n]
#define NEW_uchar(n) new uchar[n]
#define NEW_pchar(n) new pchar[n]
#define NEW_puchar(n) new puchar[n]
#define NEW_pvoid(n) new pvoid[n]
#define NEW_OBJECT(type) new type
#define NEW_OBJECTS(type, n) new type[n]
#define FREE_int(p) delete [] p
#define FREE_pint(p) delete [] p
#define FREE_lint(p) delete [] p
#define FREE_plint(p) delete [] p
#define FREE_ppint(p) delete [] p
#define FREE_pplint(p) delete [] p
#define FREE_char(p) delete [] p
#define FREE_uchar(p) delete [] p
#define FREE_pchar(p) delete [] p
#define FREE_puchar(p) delete [] p
#define FREE_pvoid(p) delete [] p
#define FREE_OBJECT(p) delete p
#define FREE_OBJECTS(p) delete [] p
#endif


#define Int_vec_print_as_polynomial_in_algebraic_notation(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->print_as_polynomial_in_algebraic_notation(A, B, C)

#define Int_vec_print(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->print(A, B, C)
#define Int_vec_stl_print(A, B) other::orbiter_kernel_system::Orbiter->Int_vec->print_stl(A, B)
#define Lint_vec_print(A, B, C) other::orbiter_kernel_system::Orbiter->Lint_vec->print(A, B, C)
#define Lint_vec_stl_print(A, B) other::orbiter_kernel_system::Orbiter->Lint_vec->print_stl(A, B)
#define Int_vec_print_fully(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->print_fully(A, B, C)
#define Int_vec_stl_print_fully(A, B) other::orbiter_kernel_system::Orbiter->Int_vec->print_stl_fully(A, B)
#define Lint_vec_print_fully(A, B, C) other::orbiter_kernel_system::Orbiter->Lint_vec->print_fully(A, B, C)
#define Lint_vec_stl_print_fully(A, B) other::orbiter_kernel_system::Orbiter->Lint_vec->print_stl_fully(A, B)
#define Int_vec_print_bare_fully(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->print_bare_fully(A, B, C)

#define Int_vec_print_integer_matrix(A,B,C,D) other::orbiter_kernel_system::Orbiter->Int_vec->print_integer_matrix(A, B, C, D)
#define Int_vec_print_integer_matrix_width(A,B,C,D,E,F) other::orbiter_kernel_system::Orbiter->Int_vec->print_integer_matrix_width(A, B, C, D, E, F)
#define Int_matrix_print_width(A,B,C,D,E) other::orbiter_kernel_system::Orbiter->Int_vec->print_integer_matrix_width(A, B, C, D, D, E)

#define Int_vec_copy(from, to, len) other::orbiter_kernel_system::Orbiter->Int_vec->copy(from, to, len)
#define Lint_vec_copy(from, to, len) other::orbiter_kernel_system::Orbiter->Lint_vec->copy(from, to, len)

#define Int_vec_print_to_str(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->print_to_str(A, B, C)
#define Lint_vec_print_to_str(A, B, C) other::orbiter_kernel_system::Orbiter->Lint_vec->print_to_str(A, B, C)


#define Int_vec_print_bare_str(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->print_bare_str(A, B, C)


#define Int_vec_print_GAP(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->print_GAP(A, B, C)
#define Lint_vec_print_GAP(A, B, C) other::orbiter_kernel_system::Orbiter->Lint_vec->print_GAP(A, B, C)


#define Int_matrix_print(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->matrix_print(A, B, C)
#define Lint_matrix_print(A, B, C) other::orbiter_kernel_system::Orbiter->Lint_vec->matrix_print(A, B, C)

#define Int_matrix_print_comma_separated(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->matrix_print_comma_separated(A, B, C)

#define Int_matrix_print_nonzero_entries(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->matrix_print_nonzero_entries(A, B, C)
#define Lint_matrix_print_nonzero_entries(A, B, C) other::orbiter_kernel_system::Orbiter->Lint_vec->matrix_print_nonzero_entries(A, B, C)

#define Int_matrix_print_ost(A, B, C, D) other::orbiter_kernel_system::Orbiter->Int_vec->matrix_print_ost(A, B, C, D)

#define Int_matrix_print_bitwise(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->matrix_print_bitwise(A, B, C)

#define Make_block_matrix_2x2(Mtx,k,A,B,C,D) other::orbiter_kernel_system::Orbiter->Int_vec->matrix_make_block_matrix_2x2(Mtx, k, A, B, C, D)



#define Int_vec_zero(A, B) other::orbiter_kernel_system::Orbiter->Int_vec->zero(A, B);
#define Lint_vec_zero(A, B) other::orbiter_kernel_system::Orbiter->Lint_vec->zero(A, B)


#define Int_vec_is_zero(A, B) other::orbiter_kernel_system::Orbiter->Int_vec->is_zero(A, B)

#define Int_vec_scan(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->scan(A, B, C)
#define Lint_vec_scan(A, B, C) other::orbiter_kernel_system::Orbiter->Lint_vec->scan(A, B, C)


#define Int_vec_copy_to_lint(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->copy_to_lint(A, B, C)
#define Lint_vec_copy_to_int(A, B, C) other::orbiter_kernel_system::Orbiter->Lint_vec->copy_to_int(A, B, C)


#define Int_vec_print_integer_matrix_in_C_source(A, B, C, D) other::orbiter_kernel_system::Orbiter->Int_vec->print_integer_matrix_in_C_source(A, B, C, D)

#define Int_vec_apply_lint(A, B, C, D) other::orbiter_kernel_system::Orbiter->Int_vec->apply_lint(A, B, C, D)
#define Lint_vec_apply(from, through, to, len) other::orbiter_kernel_system::Orbiter->Lint_vec->apply(from, through, to, len)



#define Int_vec_one(A, B) other::orbiter_kernel_system::Orbiter->Int_vec->one(A, B)
#define Lint_vec_one(A, B) other::orbiter_kernel_system::Orbiter->Lint_vec->one(A, B)

#define Int_vec_mone(A, B) other::orbiter_kernel_system::Orbiter->Int_vec->mone(A, B)
#define Lint_vec_mone(A, B) other::orbiter_kernel_system::Orbiter->Lint_vec->mone(A, B)

#define Int_vec_of_Hamming_weight_one(V, IDX, LEN) other::orbiter_kernel_system::Orbiter->Int_vec->is_Hamming_weight_one(V, IDX, LEN)


#define Int_vec_print_Cpp(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->print_Cpp(A, B, C)

#define Int_vec_complement(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->complement(A, B, C)
#define Int_vec_complement_to(A, B, C, D) other::orbiter_kernel_system::Orbiter->Int_vec->complement(A, B, C, D)

#define Lint_vec_complement_to(A, B, C, D) other::orbiter_kernel_system::Orbiter->Lint_vec->complement(A, B, C, D)

#define Int_vec_maximum(v, len) other::orbiter_kernel_system::Orbiter->Int_vec->maximum(v, len)
#define Lint_vec_maximum(v, len) other::orbiter_kernel_system::Orbiter->Lint_vec->maximum(v, len)
#define Int_vec_minimum(v, len) other::orbiter_kernel_system::Orbiter->Int_vec->minimum(v, len)
#define Lint_vec_minimum(v, len) other::orbiter_kernel_system::Orbiter->Lint_vec->minimum(v, len)


#define Int_vec_set_print(A, B, C) other::orbiter_kernel_system::Orbiter->Int_vec->set_print(A, B, C)
#define Lint_vec_set_print(A, B, C) other::orbiter_kernel_system::Orbiter->Lint_vec->set_print(A, B, C)


#define Int_vec_print_classified_str(A, B, C, D) other::orbiter_kernel_system::Orbiter->Int_vec->print_classified_str(A, B, C, D)

#define Int_vec_distribution(A, B, C, D, E) other::orbiter_kernel_system::Orbiter->Int_vec->distribution(A, B, C, D, E)

#define Int_vec_find_first_nonzero_entry(A, B) other::orbiter_kernel_system::Orbiter->Int_vec->find_first_nonzero_entry(A, B)


#define Int_vec_compare(p, q, len) other::orbiter_kernel_system::Orbiter->Int_vec->compare(p, q, len)
#define Lint_vec_compare(p, q, len) other::orbiter_kernel_system::Orbiter->Lint_vec->compare(p, q, len)

#define Int_vec_create_string_with_quotes(str, v, len) other::orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, v, len)
#define Lint_vec_create_string_with_quotes(str, v, len) other::orbiter_kernel_system::Orbiter->Lint_vec->create_string_with_quotes(str, v, len)

#define Int_vec_stringify(v, len) other::orbiter_kernel_system::Orbiter->Int_vec->stringify(v, len)
#define Lint_vec_stringify(v, len) other::orbiter_kernel_system::Orbiter->Lint_vec->stringify(v, len)

#define Get_vector(A) other::orbiter_kernel_system::Orbiter->get_object_of_type_vector(A)
#define Has_text(A) other::orbiter_kernel_system::Orbiter->is_text_available(A)
#define Get_text(A) other::orbiter_kernel_system::Orbiter->get_text(A)
#define Get_string(A) other::orbiter_kernel_system::Orbiter->get_string(A)
#define Get_int_vector_from_label(A, B, C, D) other::orbiter_kernel_system::Orbiter->get_int_vector_from_label(A, B, C, D)
#define Get_lint_vector_from_label(A, B, C, D) other::orbiter_kernel_system::Orbiter->get_lint_vector_from_label(A, B, C, D)
#define Get_matrix(label, A, m, n) other::orbiter_kernel_system::Orbiter->get_matrix_from_label(label, A, m, n)
#define Get_ring(label) other::orbiter_kernel_system::Orbiter->get_object_of_type_polynomial_ring(label)
#define Get_finite_field(label) other::orbiter_kernel_system::Orbiter->get_object_of_type_finite_field(label)
#define Get_symbol(label) other::orbiter_kernel_system::Orbiter->get_object_of_type_symbolic_object(label)
#define Find_symbol(label) other::orbiter_kernel_system::Orbiter->find_object_of_type_symbolic_object(label)
#define Get_crc_code(A) other::orbiter_kernel_system::Orbiter->get_object_of_type_crc_code(A)
#define Get_projective_space_low_level(A) other::orbiter_kernel_system::Orbiter->get_projective_space_low_level(A)
#define Get_geometry_builder(A) other::orbiter_kernel_system::Orbiter->get_geometry_builder(A)
#define Get_graph(A) other::orbiter_kernel_system::Orbiter->get_object_of_type_graph(A)
#define Get_design(A) other::orbiter_kernel_system::Orbiter->get_object_of_type_design(A)
#define Get_draw_options(A) other::orbiter_kernel_system::Orbiter->get_draw_options(A)
#define Get_draw_incidence_structure_options(A) other::orbiter_kernel_system::Orbiter->get_draw_incidence_structure_options(A)
#define Get_any_group_opaque(A) other::orbiter_kernel_system::Orbiter->get_any_group_opaque(A)
#define Get_isomorph_arguments_opaque(A) other::orbiter_kernel_system::Orbiter->get_isomorph_arguments_opaque(A)
#define Get_geometric_object(A) other::orbiter_kernel_system::Orbiter->get_geometric_object(A)
#define Get_classify_cubic_surfaces_opaque() other::orbiter_kernel_system::Orbiter->get_classify_cubic_surfaces_opaque(A)



#define Global_export(ptr, v) other::orbiter_kernel_system::Orbiter->do_export(ptr, v)
#define Global_import(v) other::orbiter_kernel_system::Orbiter->do_import(v)


#define Record_birth() other::orbiter_kernel_system::Orbiter->record_birth(__func__)
#define Record_death() other::orbiter_kernel_system::Orbiter->record_death(__func__)



enum monomial_ordering_type {
	t_LEX, // lexicographical
	t_PART, // by partition type
};


enum object_with_canonical_form_type {
	t_PTS, // points
	t_LNS, // lines
	t_PNL, // points and lines
	t_PAC, // packing
	t_INC, // incidence geometry
	t_LS, // large set
	t_MMX, // multi matrix: a matrix with entries over small natural numbers
};

enum diophant_equation_type {
	t_EQ, // equal to the given value
	t_LE, // less than or equal to the given value
	t_INT, // must be within the given interval
	t_ZOR // zero or equal to the given value
}; 



typedef enum monomial_ordering_type monomial_ordering_type;
typedef enum diophant_equation_type diophant_equation_type;


enum syntax_tree_node_operation_type
{
	operation_type_nothing,
	operation_type_mult,
	operation_type_add
};


enum data_input_stream_type {
	t_data_input_stream_unknown,
	t_data_input_stream_set_of_points,
	t_data_input_stream_set_of_lines,
	t_data_input_stream_set_of_points_and_lines,
	t_data_input_stream_set_of_packing,
	t_data_input_stream_file_of_points,
	t_data_input_stream_file_of_points_csv,
	t_data_input_stream_file_of_lines,
	t_data_input_stream_file_of_packings,
	t_data_input_stream_file_of_packings_through_spread_table,
	t_data_input_stream_file_of_designs_through_block_orbits,
	t_data_input_stream_file_of_designs_through_blocks,
	t_data_input_stream_file_of_point_set,
	t_data_input_stream_file_of_designs,
	t_data_input_stream_file_of_incidence_geometries,
	t_data_input_stream_file_of_incidence_geometries_by_row_ranks,
	t_data_input_stream_incidence_geometry,
	t_data_input_stream_incidence_geometry_by_row_ranks,
	t_data_input_stream_from_parallel_search,
	t_data_input_stream_orbiter_file,
	t_data_input_stream_csv_file,
	t_data_input_stream_graph_by_adjacency_matrix,
	t_data_input_stream_graph_object,
	t_data_input_stream_design_object,
	t_data_input_stream_graph_by_adjacency_matrix_from_file,
	t_data_input_stream_multi_matrix,
	t_data_input_stream_geometric_object,
	t_data_input_stream_Kaempfer_file,

};


enum CRC_type {
	t_CRC_16,
	t_CRC_32,
	t_CRC_771_30,

};




}}


#include "algebra/basic_algebra/basic_algebra.h"
#include "algebra/expression_parser/expression_parser.h"
#include "algebra/field_theory/finite_fields.h"
#include "algebra/linear_algebra/linear_algebra.h"
#include "algebra/number_theory/number_theory.h"
#include "algebra/ring_theory/ring_theory.h"

#include "combinatorics/canonical_form_classification/canonical_form_classification.h"
#include "combinatorics/coding_theory/coding_theory.h"
#include "combinatorics/cryptography/cryptography.h"
#include "combinatorics/design_theory/design_theory.h"
#include "combinatorics/geometry_builder/geometry_builder.h"
#include "combinatorics/graph_theory/graph_theory.h"
#include "combinatorics/knowledge_base/knowledge_base.h"
#include "combinatorics/other_combinatorics/other.h"
#include "combinatorics/solvers/solvers.h"
#include "combinatorics/puzzles/puzzles.h"
#include "combinatorics/special_functions/special_functions.h"
#include "combinatorics/tactical_decompositions/tactical_decompositions.h"


#include "geometry/algebraic_geometry/algebraic_geometry.h"
#include "geometry/finite_geometries/finite_geometries.h"
#include "geometry/orthogonal/orthogonal.h"
#include "geometry/other_geometry/other_geometry.h"
#include "geometry/projective_geometry/projective_geometry.h"



#include "other/data_structures/data_structures.h"
#include "other/graphics/graphics.h"
#include "other/l1_interfaces/l1_interfaces.h"
#include "other/orbiter_kernel_system/orbiter_kernel_system.h"
#include "other/polish/polish.h"





#endif /* ORBITER_SRC_LIB_FOUNDATIONS_FOUNDATIONS_H_ */



