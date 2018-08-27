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
// added file dlx.C: April 7, 2013
// added class spreadsheet: March 15, 2013
// added class andre_construction andre_construction: June 2, 2013
// added class andre_construction_point_element: June 2, 2013
// added class andre_construction_line_element: June 2, 2013
// added class INT_matrix: October 23, 2013
// added class gl_classes: October 23, 2013
// added class layered_graph: January 6, 2014
// added class graph_layer: January 6, 2014
// added class graph_node: January 6, 2014
// added class INT_vector: August 12, 2014
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

#include <iostream>
#include <fstream>
//#include <sstream>
#include <iomanip>

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include <map>
#include <vector>
#include <deque>


using namespace std;


#include <iostream>
#include <map>
#include <vector>
#include <deque>



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
#define SYSTEM_IS_MACINTOSH
	// use Mac specific stuff like asking how much memory the process uses.
#endif

#if defined(__linux__) || defined(linux) || defined(__linux)
#define SYSTEM_LINUX
#endif
/*----------------------------------------------------------------------------*/



#define HAS_NAUTY 1

// need to be defined in nauty_interface.C also.


#ifdef SYSTEMWINDOWS
#pragma warning(disable : 4996)
#include <string>
#endif

#define MEMORY_DEBUG

#define MAGIC_SYNC 762873656L



// define exactly one of the following to match your system:
#undef INT_HAS_2_BYTES
#define INT_HAS_4_BYTES
#undef INT_HAS_8_BYTES

#ifdef INT_HAS_2_BYTES
typedef short INT2;
typedef long INT4;
typedef long INT8;
typedef unsigned short UINT2;
typedef unsigned long UINT4;
typedef unsigned long UINT8;
#endif
#ifdef INT_HAS_4_BYTES
typedef short INT2;
typedef int INT4;
typedef long INT8;
typedef unsigned short UINT2;
typedef unsigned int UINT4;
typedef unsigned long UINT8;
#endif
#ifdef INT_HAS_8_BYTES
typedef short INT2;
typedef short int INT4;
typedef int INT8;
typedef unsigned short UINT2;
typedef unsigned short int UINT4;
typedef unsigned int UINT8;
#endif


typedef long int INT;
typedef INT *PINT;
typedef INT **PPINT;
typedef unsigned long UINT;
typedef UINT *PUINT;
typedef long LONG;
typedef LONG *PLONG;
typedef unsigned long ULONG;
typedef ULONG *PULONG;
typedef short SHORT;
typedef SHORT *PSHORT;
typedef char BYTE;
typedef BYTE *PBYTE;
typedef unsigned char UBYTE;
typedef UBYTE *PUBYTE;
typedef char SCHAR;
typedef SCHAR *PSCHAR;
typedef float FLOAT;
typedef FLOAT *PFLOAT;
typedef BYTE TSTRING;
typedef int *pint;
typedef void *pvoid;



#define PAGE_LENGTH_LOG 20
#define MAX_PAGE_SIZE_IN_BYTES (5 * 1L << 20)
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
#define ONE_BYTE_INT(a) (((a) > -126) && ((a) < 127))
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

typedef class finite_field finite_field;
typedef class longinteger_object longinteger_object;
typedef longinteger_object *plonginteger_object;
typedef class longinteger_domain longinteger_domain;
typedef class rank_checker rank_checker;
typedef class classify classify;
typedef void *unipoly_object;
typedef class unipoly_domain unipoly_domain;
typedef class mp_graphics mp_graphics;
typedef class partitionstack partitionstack;
typedef class orthogonal orthogonal;
typedef class vector_hashing vector_hashing;
typedef class unusual_model unusual_model;
typedef class grassmann grassmann;
typedef class grassmann_embedded grassmann_embedded;
typedef class hermitian hermitian;
typedef class incidence_structure incidence_structure;
typedef class finite_ring finite_ring;
typedef class hjelmslev hjelmslev;
typedef class norm_tables norm_tables;
typedef struct coordinate_frame coordinate_frame;
typedef class data_file data_file;
typedef class subfield_structure subfield_structure;
typedef class clique_finder clique_finder;
typedef class colored_graph colored_graph;
typedef class rainbow_cliques rainbow_cliques;
typedef class set_of_sets set_of_sets;
typedef class decomposition decomposition;
typedef class brick_domain brick_domain;
typedef class spreadsheet spreadsheet;
typedef class andre_construction andre_construction;
typedef class andre_construction_point_element andre_construction_point_element;
typedef class andre_construction_line_element andre_construction_line_element;
typedef class memory_object memory_object;
typedef class tree_node tree_node;
typedef tree_node *ptree_node;
typedef class tree tree;
typedef class INT_matrix INT_matrix;
typedef class gl_classes gl_classes;
typedef class gl_class_rep gl_class_rep;
typedef class matrix_block_data matrix_block_data;
typedef class layered_graph layered_graph;
typedef class graph_layer graph_layer;
typedef class graph_node graph_node;
typedef class INT_vector INT_vector;
typedef class projective_space projective_space;
typedef class buekenhout_metz buekenhout_metz;
typedef class a_domain a_domain;
typedef class diophant dophant;
typedef class null_polarity_generator null_polarity_generator;
typedef class layered_graph_draw_options layered_graph_draw_options;
typedef class klein_correspondence klein_correspondence;
typedef class file_output file_output;
typedef class generators_symplectic_group generators_symplectic_group;
typedef class flag flag;
typedef class W3q W3q; // added March 4, 2011
typedef class knarr knarr; // added March 30, 2011
typedef class surface surface; // added July 26, 2016
typedef class homogeneous_polynomial_domain homogeneous_polynomial_domain;
	// added Sept 9, 2016
typedef class eckardt_point eckardt_point;
typedef class surface_object surface_object;
typedef class heisenberg heisenberg;
typedef class desarguesian_spread desarguesian_spread;
typedef class classify_bitvectors classify_bitvectors;
typedef class object_in_projective_space object_in_projective_space;
typedef class point_line point_line;
typedef struct plane_data PLANE_DATA;
typedef class tdo_scheme tdo_scheme;
typedef class tdo_data tdo_data;
typedef struct solution_file_data solution_file_data;
typedef class geo_parameter geo_parameter;
typedef class scene scene;
typedef class mem_object_registry mem_object_registry;
typedef class mem_object_registry_entry mem_object_registry_entry;


#ifdef MEMORY_DEBUG
#define NEW_int(n) global_mem_object_registry.allocate_int(n, __FILE__, __LINE__)
#define NEW_pint(n) global_mem_object_registry.allocate_pint(n, __FILE__, __LINE__)
#define NEW_INT(n) global_mem_object_registry.allocate_INT(n, __FILE__, __LINE__)
#define NEW_PINT(n) global_mem_object_registry.allocate_PINT(n, __FILE__, __LINE__)
#define NEW_PPINT(n) global_mem_object_registry.allocate_PPINT(n, __FILE__, __LINE__)
#define NEW_BYTE(n) global_mem_object_registry.allocate_BYTE(n, __FILE__, __LINE__)
#define NEW_UBYTE(n) global_mem_object_registry.allocate_UBYTE(n, __FILE__, __LINE__)
#define NEW_PBYTE(n) global_mem_object_registry.allocate_PBYTE(n, __FILE__, __LINE__)
#define NEW_PUBYTE(n) global_mem_object_registry.allocate_PUBYTE(n, __FILE__, __LINE__)
#define NEW_pvoid(n) global_mem_object_registry.allocate_pvoid(n, __FILE__, __LINE__)
#define NEW_OBJECT(type) (type *)global_mem_object_registry.allocate_OBJECT(new type, sizeof(type), __FILE__, __LINE__)
#define NEW_OBJECTS(type, n) (type *)global_mem_object_registry.allocate_OBJECTS(new type[n], n, sizeof(type), __FILE__, __LINE__)
#define FREE_int(p) global_mem_object_registry.free_int(p, __FILE__, __LINE__)
#define FREE_pint(p) global_mem_object_registry.free_pint(p, __FILE__, __LINE__)
#define FREE_INT(p) global_mem_object_registry.free_INT(p, __FILE__, __LINE__)
#define FREE_PINT(p) global_mem_object_registry.free_PINT(p, __FILE__, __LINE__)
#define FREE_PPINT(p) global_mem_object_registry.free_PPINT(p, __FILE__, __LINE__)
#define FREE_BYTE(p) global_mem_object_registry.free_BYTE(p, __FILE__, __LINE__)
#define FREE_UBYTE(p) global_mem_object_registry.free_UBYTE(p, __FILE__, __LINE__)
#define FREE_PBYTE(p) global_mem_object_registry.free_PBYTE(p, __FILE__, __LINE__)
#define FREE_PUBYTE(p) global_mem_object_registry.free_PUBYTE(p, __FILE__, __LINE__)
#define FREE_pvoid(p) global_mem_object_registry.free_pvoid(p, __FILE__, __LINE__)
#define FREE_OBJECT(p) {global_mem_object_registry.free_OBJECT(p, __FILE__, __LINE__); delete p;}
#define FREE_OBJECTS(p) {global_mem_object_registry.free_OBJECTS(p, __FILE__, __LINE__); delete [] p;}
#else
#define NEW_int(n) new int[n]
#define NEW_pint(n) new pint[n]
#define NEW_INT(n) new INT[n]
#define NEW_PINT(n) new PINT[n]
#define NEW_PPINT(n) new PPINT[n]
#define NEW_BYTE(n) new BYTE[n]
#define NEW_UBYTE(n) new UBYTE[n]
#define NEW_PBYTE(n) new PBYTE[n]
#define NEW_PUBYTE(n) new PUBYTE[n]
#define NEW_pvoid(n) new pvoid[n]
#define NEW_OBJECT(type) new type
#define NEW_OBJECTS(type, n) new type[n]
#define FREE_int(p) delete [] p
#define FREE_pint(p) delete [] p
#define FREE_INT(p) delete [] p
#define FREE_PINT(p) delete [] p
#define FREE_PPINT(p) delete [] p
#define FREE_BYTE(p) delete [] p
#define FREE_UBYTE(p) delete [] p
#define FREE_PBYTE(p) delete [] p
#define FREE_PUBYTE(p) delete [] p
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
	t_EQ, 
	t_LE,
	t_ZOR
}; 

typedef enum diophant_equation_type diophant_equation_type;


// we cannot move the following two declarations into their appropriate places,
// for otherwise we would create incomplete type compile errors:

// #############################################################################
// INT_matrix.C:
// #############################################################################


//! a class to represent matrices over INT


class INT_matrix {
public:

	INT *M;
	INT m;
	INT n;

	INT_matrix();
	~INT_matrix();
	void null();
	void freeself();
	void allocate(INT m, INT n);
	void allocate_and_init(INT m, INT n, INT *Mtx);
	INT &s_ij(INT i, INT j);
	INT &s_m();
	INT &s_n();
	void print();

	

};

// #############################################################################
// longinteger_object.C:
// #############################################################################

extern INT longinteger_f_print_scientific;

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
	
	char &sign() { return sgn; };
	int &len() { return l; };
	char *&rep() { return r; };
	void create(INT i);
	void create_product(INT nb_factors, INT *factors);
	void create_power(INT a, INT e);
		// creates a^e
	void create_power_minus_one(INT a, INT e);
		// creates a^e  - 1
	void create_from_base_b_representation(INT b, INT *rep, INT len);
	void create_from_base_10_string(const BYTE *str, INT verbose_level);
	void create_from_base_10_string(const BYTE *str);
	INT as_INT();
	void as_longinteger(longinteger_object &a);
	void assign_to(longinteger_object &b);
	void swap_with(longinteger_object &b);
	ostream& print(ostream& ost);
	ostream& print_not_scientific(ostream& ost);
	INT output_width();
	void print_width(ostream& ost, INT width);
	void print_to_string(BYTE *str);
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
	void add_INT(INT a);
	void create_i_power_j(INT i, INT j);
	INT compare_with_INT(INT a);
};

ostream& operator<<(ostream& ost, longinteger_object& p);



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


