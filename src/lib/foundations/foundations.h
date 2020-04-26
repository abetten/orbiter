// foundations.h
//
// Anton Betten
//
// renamed from galois.h: August 16, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005



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

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sstream>

#include <map>
#include <vector>
#include <deque>


//using namespace std;


#include <iostream>
#include <map>
#include <vector>
#include <deque>

#ifndef FOUNDATIONS_H_
#define FOUNDATIONS_H_

/*----------------------------------------------------------------------------*/
/// Define some ANSI colour codes
/*----------------------------------------------------------------------------*/
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
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/// The following code block identifies the current operating system the code is
/// being executed on and turns on specific macros in order to use system calls
/// defined by that operating system.
/*----------------------------------------------------------------------------*/
#if defined(unix) || defined(__unix) || defined(__unix__)
#define SYSTEMUNIX
#endif

#if defined(_WIN32) || defined(_WIN64)
#define SYSTEMWINDOWS
#endif

#if defined(__APPLE__) || defined(__MACH__)
#define SYSTEMUNIX
#define SYSTEM_IS_MACintOSH
	// use Mac specific stuff like asking how much memory the process uses.
#endif

#if defined(__linux__) || defined(linux) || defined(__linux)
#define SYSTEM_LINUX
#endif
/*----------------------------------------------------------------------------*/

#define SYSTEMUNIX


#define HAS_NAUTY 1

// need to be defined in nauty_interface.cpp also.


#ifdef SYSTEMWINDOWS
#pragma warning(disable : 4996)
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


class finite_field;
class longinteger_object;
typedef longinteger_object *plonginteger_object;
class longinteger_domain;
class rank_checker;
class classify;
typedef void *unipoly_object;
class unipoly_domain;
class mp_graphics;
class partitionstack;
class orthogonal;
class vector_hashing;
class unusual_model;
class grassmann;
class grassmann_embedded;
class hermitian;
class incidence_structure;
class finite_ring;
class hjelmslev;
class norm_tables;
class coordinate_frame;
class data_file;
class subfield_structure;
class clique_finder;
class colored_graph;
class rainbow_cliques;
class set_of_sets;
class decomposition;
class brick_domain;
class spreadsheet;
class andre_construction;
class andre_construction_point_element;
class andre_construction_line_element;
class memory_object;
class tree_node;
typedef tree_node *ptree_node;
class tree;
class int_matrix;
class gl_classes;
class gl_class_rep;
class matrix_block_data;
class layered_graph;
class graph_layer;
class graph_node;
class int_vector;
class projective_space;
class buekenhout_metz;
class a_domain;
class diophant;
class null_polarity_generator;
class layered_graph_draw_options;
class klein_correspondence;
class file_output;
class generators_symplectic_group;
class flag;
class W3q; // added March 4, 2011
class knarr; // added March 30, 2011
class surface_domain; // added July 26, 2016
class homogeneous_polynomial_domain;
// added Sept 9, 2016
class eckardt_point;
class surface_object;
class heisenberg;
class desarguesian_spread;
class classify_bitvectors;
class object_in_projective_space;
class point_line;
typedef struct plane_data PLANE_DATA;
class tdo_scheme;
class tdo_data;
struct solution_file_data;
class geo_parameter;
class scene;
class mem_object_registry;
class mem_object_registry_entry;
class eckardt_point_info;
class vector_space;
class elliptic_curve;
// added November 19, 2014
class arc_lifting_with_two_lines;
class page_storage;
class video_draw_options;
class numerics;
class polynomial_double;
class polynomial_double_domain;
class spread_tables;
class cubic_curve;
class partial_derivative;
class animate;
class blt_set_domain;
class blt_set_invariants;
class magma_interface;
class povray_interface;
class latex_interface;
class number_theory_domain;
class group_generators_domain;
class knowledge_base;
class combinatorics_domain;
class sorting;
class coding_theory_domain;
class file_io;
class table_of_irreducible_polynomials;
class set_of_sets_lint;
class classify_vector_data;
class clebsch_map;
class bitmatrix;
class override_double;
class os_interface;
class plot_tools;
class tdo_refinement;
class algebra_global;
class combinatorial_object_description;
class combinatorial_object_create;
class drawable_set_of_objects;
class parametric_curve;
class parametric_curve_entry;
class function_command;
class function_polish_description;
class function_polish;

#ifdef MEMORY_DEBUG
#define NEW_int(n) global_mem_object_registry.allocate_int(n, __FILE__, __LINE__)
#define NEW_int_with_tracking(n, file, line) global_mem_object_registry.allocate_int(n, file, line)
#define NEW_pint(n) global_mem_object_registry.allocate_pint(n, __FILE__, __LINE__)
#define NEW_lint(n) global_mem_object_registry.allocate_lint(n, __FILE__, __LINE__)
#define NEW_plint(n) global_mem_object_registry.allocate_plint(n, __FILE__, __LINE__)
#define NEW_ppint(n) global_mem_object_registry.allocate_ppint(n, __FILE__, __LINE__)
#define NEW_pplint(n) global_mem_object_registry.allocate_pplint(n, __FILE__, __LINE__)
#define NEW_char(n) global_mem_object_registry.allocate_char(n, __FILE__, __LINE__)
#define NEW_char_with_tracking(n, file, line) global_mem_object_registry.allocate_char(n, file, line)
#define NEW_uchar(n) global_mem_object_registry.allocate_uchar(n, __FILE__, __LINE__)
#define NEW_pchar(n) global_mem_object_registry.allocate_pchar(n, __FILE__, __LINE__)
#define NEW_puchar(n) global_mem_object_registry.allocate_puchar(n, __FILE__, __LINE__)
#define NEW_pvoid(n) global_mem_object_registry.allocate_pvoid(n, __FILE__, __LINE__)
#define NEW_OBJECT(type) (type *)global_mem_object_registry.allocate_OBJECT(new type, (std::size_t) sizeof(type), #type, __FILE__, __LINE__)
#define NEW_OBJECTS(type, n) (type *)global_mem_object_registry.allocate_OBJECTS(new type[n], n, (std::size_t) sizeof(type), #type, __FILE__, __LINE__)
#define FREE_int(p) global_mem_object_registry.free_int(p, __FILE__, __LINE__)
#define FREE_pint(p) global_mem_object_registry.free_pint(p, __FILE__, __LINE__)
#define FREE_lint(p) global_mem_object_registry.free_lint(p, __FILE__, __LINE__)
#define FREE_plint(p) global_mem_object_registry.free_plint(p, __FILE__, __LINE__)
#define FREE_ppint(p) global_mem_object_registry.free_ppint(p, __FILE__, __LINE__)
#define FREE_pplint(p) global_mem_object_registry.free_pplint(p, __FILE__, __LINE__)
#define FREE_char(p) global_mem_object_registry.free_char(p, __FILE__, __LINE__)
#define FREE_uchar(p) global_mem_object_registry.free_uchar(p, __FILE__, __LINE__)
#define FREE_pchar(p) global_mem_object_registry.free_pchar(p, __FILE__, __LINE__)
#define FREE_puchar(p) global_mem_object_registry.free_puchar(p, __FILE__, __LINE__)
#define FREE_pvoid(p) global_mem_object_registry.free_pvoid(p, __FILE__, __LINE__)
#define FREE_OBJECT(p) {global_mem_object_registry.free_OBJECT(p, __FILE__, __LINE__); delete p;}
#define FREE_OBJECTS(p) {global_mem_object_registry.free_OBJECTS(p, __FILE__, __LINE__); delete [] p;}
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



enum object_in_projective_space_type {
	t_PTS, // points
	t_LNS, // lines
	t_PAC // packing
};

enum diophant_equation_type {
	t_EQ, // equal to the given value
	t_LE, // less than or equal to the given value
	t_INT, // must be within the given interval
	t_ZOR // zero or equal to the given value
}; 

typedef enum diophant_equation_type diophant_equation_type;


// we cannot move the following two declarations into their appropriate places,
// for otherwise we would create incomplete type compile errors:

// #############################################################################
// int_matrix.cpp:
// #############################################################################


//! matrices over int


class int_matrix {
public:

	int *M;
	int m;
	int n;

	int_matrix();
	~int_matrix();
	void null();
	void freeself();
	void allocate(int m, int n);
	void allocate_and_init(int m, int n, int *Mtx);
	int &s_ij(int i, int j);
	int &s_m();
	int &s_n();
	void print();

	

};

// #############################################################################
// longinteger_object.cpp:
// #############################################################################

extern int longinteger_f_print_scientific;

//! a class to represent aritrary precision integers


class longinteger_object {

private:
	char sgn; // TRUE if negative
	int l;
	char *r;
	
public:
	longinteger_object();
	~longinteger_object();
	void freeself();
	
	char &ith(int i) { return r[i]; };
	char &sign() { return sgn; };
	int &len() { return l; };
	char *&rep() { return r; };
	void create(long int i, const char *file, int line);
	void create_product(int nb_factors, int *factors);
	void create_power(int a, int e);
		// creates a^e
	void create_power_minus_one(int a, int e);
		// creates a^e  - 1
	void create_from_base_b_representation(int b, int *rep, int len);
	void create_from_base_10_string(const char *str, int verbose_level);
	void create_from_base_10_string(const char *str);
	int as_int();
	long int as_lint();
	void as_longinteger(longinteger_object &a);
	void assign_to(longinteger_object &b);
	void swap_with(longinteger_object &b);
	std::ostream& print(std::ostream& ost);
	std::ostream& print_not_scientific(std::ostream& ost);
	int log10();
	int output_width();
	void print_width(std::ostream& ost, int width);
	void print_to_string(char *str);
	void normalize();
	void negate();
	int is_zero();
	void zero();
	int is_one();
	int is_mone();
	int is_one_or_minus_one();
	void one();
	void increment();
	void decrement();
	void add_int(int a);
	void create_i_power_j(int i, int j);
	int compare_with_int(int a);
};

std::ostream& operator<<(std::ostream& ost, longinteger_object& p);

}
}

#include "./algebra_and_number_theory/algebra_and_number_theory.h"
#include "./coding_theory/coding_theory.h"
#include "./combinatorics/combinatorics.h"
#include "./data_structures/data_structures.h"
#include "./geometry/geometry.h"
#include "./globals/globals.h"
#include "./graph_theory/graph_theory.h"
#include "./graph_theory_nauty/graph_theory_nauty.h"
#include "./graphics/graphics.h"
#include "./io_and_os/io_and_os.h"
#include "./solvers/solvers.h"
#include "./statistics/statistics.h"

#endif
