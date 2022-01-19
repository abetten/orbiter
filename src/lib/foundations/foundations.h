// foundations.h
//
// Anton Betten
//
// renamed from galois.h: August 16, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


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
typedef short SHORT;
typedef SHORT *PSHORT;
typedef char *pchar;
typedef unsigned char uchar;
typedef uchar *puchar;
typedef void *pvoid;



#define PAGE_LENGTH_LOG 20
#define MAX_PAGE_SIZE_IN_charS (5 * 1L << 20)
#define BUFSIZE 100000
#undef DEBUG_PAGE_STORAGE


#define MINIMUM(x, y)   ( ((x) < (y)) ?  (x) : (y) )
#define MAXIMUM(x, y)   ( ((x) > (y)) ?  (x) : (y) )
#define MIN(x, y)   ( ((x) < (y)) ?  (x) : (y) )
#define MAX(x, y)   ( ((x) > (y)) ?  (x) : (y) )
#define ABS(x)      ( ((x) <  0 ) ? (-(x)) : (x) )
#define EVEN(x)     ( ((x) % 2) == 0 )
#define ODD(x)      ( ((x) % 2) == 1 )
#define DOUBLYEVEN(x)     ( ((x) % 4) == 0 )
#define SINGLYEVEN(x)     ( ((x) % 4) == 2 )
#define ONE_char_int(a) (((a) > -126) && ((a) < 127))
#define ONE_MILLION 1000000
#define ONE_HUNDRED_THOUSAND 100000


#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846264
#endif


//! the orbiter library for the classification of combinatorial objects

namespace orbiter {



//! algebra, combinatorics and graph theory, geometry, data structures, solvers; no group theory

namespace foundations {




namespace algebra {

	// algebra:
	class a_domain;
	class algebra_global;
	class generators_symplectic_group;
	class gl_class_rep;
	class gl_classes;
	class group_generators_domain;
	class heisenberg;
	class matrix_block_data;
	class null_polarity_generator;
	class rank_checker;
	class vector_space;

}


namespace coding_theory {

	// coding_theory:
	class coding_theory_domain;
	class create_BCH_code;

}

namespace combinatorics {

	// combinatorics:
	class boolean_function_domain;
	class brick_domain;
	class classification_of_objects_description;
	class classification_of_objects_report_options;
	class classification_of_objects;
	class combinatorics_domain;
	class encoded_combinatorial_object;
	class geo_parameter;
	class pentomino_puzzle;
	class tdo_data;
	class tdo_refinement_description;
	class tdo_refinement;
	class tdo_scheme_compute;
	class tdo_scheme_synthetic;
	struct solution_file_data;

}

namespace cryptography {

	// cryptography
	class cryptography_domain;

}


namespace data_structures {

	// data_structures:
	class algorithms;
	class bitmatrix;
	class bitvector;
	class classify_bitvectors;
	class classify_using_canonical_forms;
	class data_file;
	class data_input_stream_description;
	class data_input_stream;
	class data_structures_global;
	class fancy_set;
	class int_matrix;
	class int_vec;
	class int_vector;
	class lint_vec;
	class nauty_output;
	class page_storage;
	class partitionstack;
	class set_builder_description;
	class set_builder;
	class set_of_sets_lint;
	class set_of_sets;
	class sorting;
	class spreadsheet;
	class string_tools;
	class vector_hashing;

}


// expression_parser:
class expression_parser_domain;
class expression_parser;
class formula;
class lexer;
class syntax_tree_node_terminal;
class syntax_tree_node;
class syntax_tree;


namespace field_theory {

	// finite_fields:
	class finite_field_activity_description;
	class finite_field_activity;
	class finite_field_description;
	class finite_field;
	class finite_field_implementation_by_tables;
	class finite_field_implementation_wo_tables;
	class norm_tables;
	class nth_roots;
	class subfield_structure;
}

// geometry:
class andre_construction_line_element;
class andre_construction_point_element;
class andre_construction;
class arc_basic;
class arc_in_projective_space;
class buekenhout_metz;
class cubic_curve;
class decomposition;
class desarguesian_spread;
class elliptic_curve;
class flag;
class geometric_object_create;
class geometric_object_description;
class geometry_global;
class grassmann_embedded;
class grassmann;
class hermitian;
class hjelmslev;
class incidence_structure;
class klein_correspondence;
class knarr;
class object_with_canonical_form;
class point_line;
class points_and_lines;
class polarity;
class projective_space_implementation;
class projective_space;
class spread_tables;
class W3q;


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

// globals:
class function_command;
class function_polish_description;
class function_polish;
class magma_interface;
class numerics;
class orbiter_session;
class orbiter_symbol_table_entry;
class orbiter_symbol_table;
class polynomial_double_domain;
class polynomial_double;


namespace graph_theory {

	// graph_theory
	class clique_finder_control;
	class clique_finder;
	class colored_graph;
	class graph_layer;
	class graph_node;
	class graph_theory_domain;
	class layered_graph;
	class rainbow_cliques;

}

// graph_theory_nauty
class nauty_interface;


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
class povray_interface;
class povray_job_description;
class scene;
class tree_draw_options;
class tree;
class tree_node;
class video_draw_options;

// io_and_os:
class create_file_description;
class file_io;
class file_output;
class latex_interface;
class mem_object_registry_entry;
class mem_object_registry;
class memory_object;
class orbiter_data_file;
class os_interface;
class override_double;
class prepare_frames;


// knowledge_base:
class knowledge_base;


namespace linear_algebra {

	// linear_algebra:
	class linear_algebra;
	class representation_theory_domain;

}

namespace number_theory {

	// number_theory:
	class cyclotomic_sets;
	class number_theoretic_transform;
	class number_theory_domain;

}

// orthogonal:
class blt_set_domain;
class blt_set_invariants;
class orthogonal_indexing;
class orthogonal;
class unusual_model;

namespace ring_theory {
	// ring_theory:
	class finite_ring;
	class homogeneous_polynomial_domain;
	class longinteger_domain;
	class longinteger_object;
	class partial_derivative;
	class ring_theory_global;
	class table_of_irreducible_polynomials;
	class unipoly_domain;

	typedef ring_theory::longinteger_object *plonginteger_object;
	typedef void *unipoly_object;

}

namespace solvers {
	// solvers
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

// statistics:
class tally_vector_data;
class tally;

namespace algebraic_geometry {

	// surfaces:
	class arc_lifting_with_two_lines;
	class clebsch_map;
	class del_pezzo_surface_of_degree_two_domain;
	class del_pezzo_surface_of_degree_two_object;
	class eckardt_point_info;
	class eckardt_point;
	class quartic_curve_domain;
	class quartic_curve_object_properties;
	class quartic_curve_object;
	class schlaefli_labels;
	class schlaefli;
	class seventytwo_cases;
	class surface_domain;
	class surface_object_properties;
	class surface_object;
	class web_of_cubic_curves;

}

// pointer types
typedef tree_node *ptree_node;



#ifdef MEMORY_DEBUG
#define NEW_int(n) Orbiter->global_mem_object_registry->allocate_int(n, __FILE__, __LINE__)
#define NEW_int_with_tracking(n, file, line) Orbiter->global_mem_object_registry->allocate_int(n, file, line)
#define NEW_pint(n) Orbiter->global_mem_object_registry->allocate_pint(n, __FILE__, __LINE__)
#define NEW_lint(n) Orbiter->global_mem_object_registry->allocate_lint(n, __FILE__, __LINE__)
#define NEW_plint(n) Orbiter->global_mem_object_registry->allocate_plint(n, __FILE__, __LINE__)
#define NEW_ppint(n) Orbiter->global_mem_object_registry->allocate_ppint(n, __FILE__, __LINE__)
#define NEW_pplint(n) Orbiter->global_mem_object_registry->allocate_pplint(n, __FILE__, __LINE__)
#define NEW_char(n) Orbiter->global_mem_object_registry->allocate_char(n, __FILE__, __LINE__)
#define NEW_char_with_tracking(n, file, line) Orbiter->global_mem_object_registry->allocate_char(n, file, line)
#define NEW_uchar(n) Orbiter->global_mem_object_registry->allocate_uchar(n, __FILE__, __LINE__)
#define NEW_pchar(n) Orbiter->global_mem_object_registry->allocate_pchar(n, __FILE__, __LINE__)
#define NEW_puchar(n) Orbiter->global_mem_object_registry->allocate_puchar(n, __FILE__, __LINE__)
#define NEW_pvoid(n) Orbiter->global_mem_object_registry->allocate_pvoid(n, __FILE__, __LINE__)
#define NEW_OBJECT(type) (type *)Orbiter->global_mem_object_registry->allocate_OBJECT(new type, (std::size_t) sizeof(type), #type, __FILE__, __LINE__)
#define NEW_OBJECTS(type, n) (type *)Orbiter->global_mem_object_registry->allocate_OBJECTS(new type[n], n, (std::size_t) sizeof(type), #type, __FILE__, __LINE__)
#define FREE_int(p) Orbiter->global_mem_object_registry->free_int(p, __FILE__, __LINE__)
#define FREE_pint(p) Orbiter->global_mem_object_registry->free_pint(p, __FILE__, __LINE__)
#define FREE_lint(p) Orbiter->global_mem_object_registry->free_lint(p, __FILE__, __LINE__)
#define FREE_plint(p) Orbiter->global_mem_object_registry->free_plint(p, __FILE__, __LINE__)
#define FREE_ppint(p) Orbiter->global_mem_object_registry->free_ppint(p, __FILE__, __LINE__)
#define FREE_pplint(p) Orbiter->global_mem_object_registry->free_pplint(p, __FILE__, __LINE__)
#define FREE_char(p) Orbiter->global_mem_object_registry->free_char(p, __FILE__, __LINE__)
#define FREE_uchar(p) Orbiter->global_mem_object_registry->free_uchar(p, __FILE__, __LINE__)
#define FREE_pchar(p) Orbiter->global_mem_object_registry->free_pchar(p, __FILE__, __LINE__)
#define FREE_puchar(p) Orbiter->global_mem_object_registry->free_puchar(p, __FILE__, __LINE__)
#define FREE_pvoid(p) Orbiter->global_mem_object_registry->free_pvoid(p, __FILE__, __LINE__)
#define FREE_OBJECT(p) {Orbiter->global_mem_object_registry->free_OBJECT(p, __FILE__, __LINE__); delete p;}
#define FREE_OBJECTS(p) {Orbiter->global_mem_object_registry->free_OBJECTS(p, __FILE__, __LINE__); delete [] p;}
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
	t_LS // large set
};

enum diophant_equation_type {
	t_EQ, // equal to the given value
	t_LE, // less than or equal to the given value
	t_INT, // must be within the given interval
	t_ZOR // zero or equal to the given value
}; 

enum symbol_table_object_type {
	t_nothing_object,
	t_finite_field,
	t_any_group,
	t_linear_group,
	t_permutation_group,
	t_modified_group,
	t_projective_space,
	t_orthogonal_space,
	t_formula,
	t_cubic_surface,
	t_quartic_curve,
	t_classification_of_cubic_surfaces_with_double_sixes,
	t_collection,
	t_geometric_object,
	t_graph,
	t_spread_table,
	t_packing_was,
	t_packing_was_choose_fixed_points,
	t_packing_long_orbits,
	t_graph_classify,
	t_diophant,
	t_design,
	t_design_table,
	t_large_set_was,
	t_set,
	t_vector,
	t_combinatorial_objects,
	t_geometry_builder,
	t_action,
	t_poset,
	t_poset_classification,
};



typedef enum monomial_ordering_type monomial_ordering_type;
typedef enum diophant_equation_type diophant_equation_type;
typedef enum symbol_table_object_type symbol_table_object_type;


enum TokenType
{
	NONE,
	NAME,
	NUMBER,
	END,
	PLUS='+',
	MINUS='-',
	MULTIPLY='*',
	DIVIDE='/',
	ASSIGN='=',
	LHPAREN='(',
	RHPAREN=')',
	COMMA=',',
	NOT='!',

	// comparisons
	LT='<',
	GT='>',
	LE,     // <=
	GE,     // >=
	EQ,     // ==
	NE,     // !=
	AND,    // &&
	OR,      // ||

	// special assignments

	ASSIGN_ADD,  //  +=
	ASSIGN_SUB,  //  +-
	ASSIGN_MUL,  //  +*
	ASSIGN_DIV   //  +/

};

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
	t_data_input_stream_file_of_lines,
	t_data_input_stream_file_of_packings,
	t_data_input_stream_file_of_packings_through_spread_table,
	t_data_input_stream_file_of_point_set,
	t_data_input_stream_file_of_designs,
	t_data_input_stream_file_of_incidence_geometries,
	t_data_input_stream_file_of_incidence_geometries_by_row_ranks,
	t_data_input_stream_incidence_geometry,
	t_data_input_stream_incidence_geometry_by_row_ranks,
	t_data_input_stream_from_parallel_search,

};




}}


#include "./algebra/algebra.h"
#include "./coding_theory/coding_theory.h"
#include "./combinatorics/combinatorics.h"
#include "./cryptography/cryptography.h"
#include "./data_structures/data_structures.h"
#include "./expression_parser/expression_parser.h"
#include "./finite_fields/finite_fields.h"
#include "./geometry/geometry.h"
#include "./geometry_builder/geometry_builder.h"
#include "./globals/globals.h"
#include "./graph_theory/graph_theory.h"
#include "./graph_theory_nauty/graph_theory_nauty.h"
#include "./graphics/graphics.h"
#include "./io_and_os/io_and_os.h"
#include "./knowledge_base/knowledge_base.h"
#include "./linear_algebra/linear_algebra.h"
#include "./number_theory/number_theory.h"
#include "./orthogonal/orthogonal.h"
#include "./ring_theory/ring_theory.h"
#include "./solvers/solvers.h"
#include "./statistics/statistics.h"
#include "./surfaces/surfaces.h"


// Eigen_interface:
void orbiter_eigenvalues(int *Mtx, int nb_points, double *E, int verbose_level);



#endif /* ORBITER_SRC_LIB_FOUNDATIONS_FOUNDATIONS_H_ */



