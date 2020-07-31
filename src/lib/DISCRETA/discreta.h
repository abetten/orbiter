// discreta.h
//
// Anton Betten
//
// started:  18.12.1998
// modified: 23.03.2000
// moved from D2 to ORBI Nov 15, 2007


#ifndef ORBITER_SRC_LIB_DISCRETA_DISCRETA_H_
#define ORBITER_SRC_LIB_DISCRETA_DISCRETA_H_



#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdlib.h>
#include <string.h>

using namespace orbiter::classification;


namespace orbiter {

//! legacy project DISCRETA provides typed objects

namespace discreta {






#define BITS_OF_int 32
#define SYSTEMUNIX
#undef SYSTEMMAC
#undef SYSTEMWINDOWS
#define SAVE_ASCII_USE_COMPRESS




/*********************************** macros ********************************/

#define NOT_EXISTING_FUNCTION(s)  cout << "The function " << s << " does not exist in this class\n";

/******************* Constants for type determination **********************/

enum kind { 
	BASE = 0,                      //!< BASE
	INTEGER = 1,                   //!< INTEGER
	VECTOR = 2,                    //!< VECTOR
	NUMBER_PARTITION = 3,          //!< NUMBER_PARTITION
	// RATIONAL /* BRUCH */ = 4, 
	PERMUTATION = 6,               //!< PERMUTATION
	
	
	// POLYNOM = 9, 
	
	MATRIX = 11,                   //!< MATRIX

	// MONOM = 21, 
	LONGINTEGER = 22,              //!< LONGINTEGER
	
	//SUBGROUP_LATTICE = 36, 
	//SUBGROUP_ORBIT = 37, 
	MEMORY = 39,                   //!< MEMORY
	
	HOLLERITH = 44,                //!< HOLLERITH
	
	DATABASE = 50,                 //!< DATABASE
	BTREE = 51,                    //!< BTREE
	
	PERM_GROUP = 56,               //!< PERM_GROUP
	PERM_GROUP_STAB_CHAIN = 57,    //!< PERM_GROUP_STAB_CHAIN

	BT_KEY = 61,                   //!< BT_KEY
	
	DESIGN_PARAMETER = 70,         //!< DESIGN_PARAMETER
	 
	//GROUP_SELECTION = 78,          //!< GROUP_SELECTION
	UNIPOLY = 79,                  //!< UNIPOLY

	DESIGN_PARAMETER_SOURCE = 83,  //!< DESIGN_PARAMETER_SOURCE
	SOLID = 84,                    //!< SOLID

	BITMATRIX = 90,                //!< BITMATRIX
	//PC_PRESENTATION = 91,
	//PC_SUBGROUP = 92,
	//GROUP_WORD = 93, 
	//GROUP_TABLE = 94,
	//ACTION = 95, 
	//GEOMETRY = 96                  //!< GEOMETRY
	
};

enum domain_type { 
	GFp = 1, 
	GFq = 2,
	Orbiter_finite_field = 3
	//PC_GROUP = 3 
};

enum action_kind { 
	vector_entries = 1, 
	vector_positions = 2 
};

enum actionkind { 
	on_sets, 
	on_subset_of_group_elements_by_conjugation, 
	on_subset_of_group_elements_by_conjugation_with_table,
	on_group_elements_via_conjugation_using_group_table,
	on_points
};

enum numeric_mult_type { 
	with_perm_group, 
	with_group_table 
};

enum printing_mode_enum { 
	printing_mode_ascii, 
	printing_mode_latex, 
	printing_mode_ascii_file, 
	printing_mode_gap 
};


enum bt_key_kind { 
	bt_key_int = 0, 
	bt_key_string = 1, 
	bt_key_int_vec = 2
};

enum design_parameter_rule {
	rule_complementary = 1,
	rule_reduced_t = 2,
	rule_derived = 3,
	rule_residual = 4,
	rule_alltop = 5,
	rule_supplementary_reduced_t = 6,
	rule_supplementary_derived = 7,
	rule_supplementary_residual = 8,
	rule_supplementary_alltop = 9,
	rule_trung_complementary = 10,
	rule_supplementary = 11,
	rule_trung_left = 12,
	rule_trung_right = 13
};




/****************** declaration of classes and types ***********************/

class discreta_base;



// classes derived from base:

class integer;			// derived from base
	// self contains the integer value as a C (long)integer (int)
		
class longinteger;		// derived from base
	// self is a pointer to LONGINTEGER_REPRESENTATION
	// which contains the sign, the length 
	// and a C array of chars containing 
	// the decimal representation of the signless longinteger value 

class discreta_matrix;			// derived from base
	// self is a pointer obtained from 
	// calloc_m_times_n_objects().
	// this means that we have an array of m * n + 2 objacts, 
	// self points to the m * n array of user entries 
	// and at offset [-2] we have m (as an integer object), 
	// at offset [-1] we have n (as an integer object).
	// matrix access (via s_ij or via operator[]) 
	// is range checked.

//class bitmatrix;		// derived from base
	// self is a pointer to BITMATRIX_REPRESENTATION
	// which contains integers m, n, N and an array p of uint4s 
	// holding the bits in a row by row fashion.

class Vector;			// derived from base
	// self is a pointer obtained from 
	// calloc_nobjects_plus_length().
	// this means that we have an array of n + 1 objacts, 
	// self points to the n array of user entries 
	// and at offset [-1] we have the length l (as an integer object), 
	// vector access (via s_i or via operator[]) 
	// is range checked.

class memory;			// derived from base
	// self is a pointer to char which has some additional 
	// information stored at offset -3, -2, -1 in int4s.
	// these are alloc_length, used_length and cur_pointer.

class hollerith;		// derived from base
// there are so many string classes around so that I call 
// my string class hollerith class!
// n.b.: Herman Hollerith (Buffalo 1860 - Washington 1929),
// American engeneer; he invented   
// statistical machines working with perforated cards
// In 1896, he founded the Tabulating Machine Corporation 
// which later became IBM.


// classes derived from vector:
	
	class permutation;		// derived from vector
		// a vector holding the images of the 
		// points 0, 1, ..., l-1 under the permutation.
		// Note that the images are in 0, 1, ... , l-1 again!
		// the length is already stored in the vector.
		
	class number_partition;		// derived from vector
		// a vector of length 2:
		// offset 0: the type (PARTITION_TYPE_VECTOR 
		//                      or PARTITION_TYPE_EXPONENT)
		// offset 1: the self part holding the parts
		
	//class pc_presentation;		// derived from vector
	//class pc_subgroup;		// derived from vector
	//class group_word;		// derived from vector
	//class group_table;		// derived from vector
	class unipoly;			// derived from vector
	//class perm_group;		// derived from vector
	//class perm_group_stab_chain;	// derived from vector
	//class action;			// derived from vector
	//class geometry;			// derived from vector
	//class group_selection;		// derived from vector
	class solid;			// derived from vector
	class bt_key;			// derived from vector
	class database;			// derived from vector
	class btree;			// derived from vector
	class design_parameter_source;	// derived from vector
	class design_parameter;		// derived from vector


// utility class, for the operator M[i][j] matrix access:

class matrix_access;



//


class domain;
class with;
class printing_mode;
//class mp_graphics;



// in global.cpp:

extern const char *discreta_home;
extern const char *discreta_arch;

typedef class labelled_branching labelled_branching;
typedef class base_change base_change;
typedef class point_orbits point_orbits;

/************************* Prototypes of global functions ******************/

void discreta_init();
discreta_base *callocobject(kind k);
void freeobject(discreta_base *p);
discreta_base *calloc_nobjects(int n, kind k);
void free_nobjects(discreta_base *p, int n);
discreta_base *calloc_nobjects_plus_length(int n, kind k);
void free_nobjects_plus_length(discreta_base *p);
discreta_base *calloc_m_times_n_objects(int m, int n, kind k);
void free_m_times_n_objects(discreta_base *p);
void printobjectkind(std::ostream& ost, kind k);
const char *kind_ascii(kind k);
const char *action_kind_ascii(action_kind k);
//void int_swap(int& x, int& y);
void uint4_swap(uint_4& x, uint_4& y);

std::ostream& operator<<(std::ostream& ost, class discreta_base& p);
// discreta_base operator * (discreta_base& x, discreta_base &y);
// discreta_base operator + (discreta_base& x, discreta_base &y);

int invert_mod_integer(int i, int p);
int remainder_mod(int i, int n);
void factor_integer(int n, Vector& primes, Vector& exponents);
void discreta_print_factorization(Vector& primes, Vector& exponents, std::ostream &o);
void print_factorization_hollerith(Vector& primes, Vector& exponents, hollerith &h);
int nb_primes(int n);
int factor_if_prime_power(int n, int *p, int *e);
int Euler(int n);
int Moebius(int i);
int NormRemainder(int a, int m);
int log2(int n);
int sqrt_mod(int a, int p, int verbose_level);
int sqrt_mod_involved(int a, int p, int verbose_level);
void html_head(std::ostream& ost, char *title_long, char *title_short);
void html_foot(std::ostream& ost);
void sieve(Vector &primes, int factorbase, int verbose_level);
void sieve_primes(Vector &v, int from, int to, int limit, int verbose_level);
void print_intvec_mod_10(Vector &v);
void stirling_second(int n, int k, int f_ordered, discreta_base &res, int verbose_level);
void stirling_first(int n, int k, int f_signless, discreta_base &res, int verbose_level);
void Catalan(int n, Vector &v, int verbose_level);
void Catalan_n(int n, Vector &v, discreta_base &res, int verbose_level);
void Catalan_nk_matrix(int n, discreta_matrix &Cnk, int verbose_level);
void Catalan_nk_star_matrix(int n, discreta_matrix &Cnk, int verbose_level);
void Catalan_nk_star(int n, int k, discreta_matrix &Cnk, discreta_base &res, int verbose_level);

void N_choose_K(discreta_base & n, int k, discreta_base & res);
void Binomial(int n, int k, discreta_base & n_choose_k);
void Krawtchouk(int n, int q, int i, int j, discreta_base & a);
// $\sum_{u=0}^{\min(i,j)} (-1)^u \cdot (q-1)^{i-u} \cdot {j \choose u} \cdot $
// ${n - j \choose i - u}$
//int ij2k(int i, int j, int n);
//void k2ij(int k, int & i, int & j, int n);
void tuple2_rank(int rank, int &i, int &j, int n, int f_injective);
int tuple2_unrank(int i, int j, int n, int f_injective);
void output_texable_string(std::ostream & ost, char *in);
void texable_string(char *in, char *out);
void the_first_n_primes(Vector &P, int n);
void midpoint_of_2(int *Px, int *Py, int i1, int i2, int idx);
void midpoint_of_5(int *Px, int *Py, int i1, int i2, int i3, int i4, int i5, int idx);
void ratio_int(int *Px, int *Py, int idx_from, int idx_to, int idx_result, double r);

void frobenius_in_PG(domain *dom, int n, permutation &p);
// n is the projective dimension
void frobenius_in_AG(domain *dom, int n, permutation &p);
// n is the dimension
void translation_in_AG(domain *dom, int n, int i, discreta_base & a, permutation &p);
enum printing_mode_enum current_printing_mode();
void call_system(char *cmd);
void fill_char(void *v, int cnt, int c);
int hash_int(int hash0, int a);
void queue_init(Vector &Q, int elt);
int queue_get_and_remove_first_element(Vector &Q);
int queue_length(Vector &Q);
void queue_append(Vector &Q, int elt);
void print_classification_tex(Vector &content, Vector &multiplicities);
void print_classification_tex(Vector &content, Vector &multiplicities, std::ostream& ost);
void perm2permutation(int *a, int n, permutation &p);
int Gauss_int(int *A, int f_special, int f_complete, int *base_cols,
	int f_P, int *P, int m, int n, int Pn, 
	int q, int *add_table, int *mult_table,
	int *negate_table, int *inv_table, int verbose_level);
// returns the rank which is the number of entries in base_cols
void uchar_move(uchar *p, uchar *q, int len);
void int_vector_realloc(int *&p, int old_length, int new_length);
void int_vector_shorten(int *&p, int new_length);
void int_matrix_realloc(int *&p, int old_m, int new_m, int old_n, int new_n);
int code_is_irreducible(int k, int nmk, int idx_zero, int *M);
void fine_tune(finite_field *F, int *mtxD, int verbose_level);


/************************************* base ********************************/

// internal representations:

typedef struct longinteger_representation LONGINTEGER_REPRESENTATION;
//typedef struct bitmatrix_representation BITMATRIX_REPRESENTATION;

//! DISCRETA internal class


typedef union {
	int integer_value;
	char *char_pointer;
	int *int_pointer;
	discreta_base *vector_pointer;
	discreta_base *matrix_pointer;
	LONGINTEGER_REPRESENTATION *longinteger_rep;
	//BITMATRIX_REPRESENTATION *bitmatrix_rep;
} OBJECTSELF;

//! DISCRETA internal class to represent long integers


struct longinteger_representation {
	int sign;
	int len;
	char p[1];
};


#if 0
//! DISCRETA internal class to represent bitmatrices


struct bitmatrix_representation {
	int m;
	int n;
	int N;
	uint4 p[1];
};
#endif


// public class definitions:


//! DISCRETA base class. All DISCRETA classes are derived from this class




class discreta_base
{
	private:
	
	public:

	kind k;
	OBJECTSELF self;
	
	discreta_base();
	discreta_base(const discreta_base& x);
		// copy constructor
	discreta_base& operator = (const discreta_base &x);
		// copy assignment
	virtual ~discreta_base();
		// destructor
	void freeself_discreta_base();
	void freeself();
	void freeself_kind(kind k);
	void clearself() { self.vector_pointer = NULL; }

	integer& as_integer() { return *(integer *)this; }
	longinteger& as_longinteger() { return *(longinteger *)this; }
	Vector& as_vector() { return *(Vector *)this; }
	permutation& as_permutation() { return *(permutation *)this; }
	
	number_partition& as_number_partition() { return *(number_partition *)this; }
	discreta_matrix& as_matrix() { return *(discreta_matrix *)this; }
	//bitmatrix& as_bitmatrix() { return *(bitmatrix *)this; }
	//pc_presentation& as_pc_presentation() { return *(pc_presentation *)this; }
	//pc_subgroup& as_pc_subgroup() { return *(pc_subgroup *)this; }
	//group_word& as_group_word() { return *(group_word *)this; }
	//group_table& as_group_table() { return *(group_table *)this; }
	unipoly& as_unipoly() { return *(unipoly *)this; }
	//perm_group& as_perm_group() { return *(perm_group *)this; }
	//perm_group_stab_chain& as_perm_group_stab_chain() { return *(perm_group_stab_chain *)this; }
	memory& as_memory() { return *(memory *)this; }
	action& as_action() { return *(action *)this; }
	//geometry& as_geometry() { return *(geometry *)this; }
	hollerith& as_hollerith() { return *(hollerith *)this; }
	//group_selection& as_group_selection() { return *(group_selection *)this; }
	solid& as_solid() { return *(solid *)this; }
	bt_key& as_bt_key() { return *(bt_key *)this; }
	database& as_database() { return *(database *)this; }
	btree& as_btree() { return *(btree *)this; }
	design_parameter_source& as_design_parameter_source() { return *(design_parameter_source *)this; }
	design_parameter& as_design_parameter() { return *(design_parameter *)this; }
	
	integer& change_to_integer() { freeself(); c_kind(INTEGER); return as_integer(); }
	longinteger& change_to_longinteger() { freeself(); c_kind(LONGINTEGER); return as_longinteger(); }
	Vector& change_to_vector() { freeself(); c_kind(VECTOR); return as_vector(); }
	permutation& change_to_permutation() { freeself(); c_kind(PERMUTATION); return as_permutation(); }
	number_partition& change_to_number_partition() { freeself(); c_kind(NUMBER_PARTITION); return as_number_partition(); }
	discreta_matrix& change_to_matrix() { freeself(); c_kind(MATRIX); return as_matrix(); }
	//bitmatrix& change_to_bitmatrix() { freeself(); c_kind(BITMATRIX); return as_bitmatrix(); }
	//pc_presentation& change_to_pc_presentation() { freeself(); c_kind(PC_PRESENTATION); return as_pc_presentation(); }
	//pc_subgroup& change_to_pc_subgroup() { freeself(); c_kind(PC_SUBGROUP); return as_pc_subgroup(); }
	//group_word& change_to_group_word() { freeself(); c_kind(GROUP_WORD); return as_group_word(); }
	//group_table& change_to_group_table() { freeself(); c_kind(GROUP_TABLE); return as_group_table(); }
	unipoly& change_to_unipoly() { freeself(); c_kind(UNIPOLY); return as_unipoly(); }
	//perm_group& change_to_perm_group() { freeself(); c_kind(PERM_GROUP); return as_perm_group(); }
	//perm_group_stab_chain& change_to_perm_group_stab_chain() { freeself(); c_kind(PERM_GROUP_STAB_CHAIN); return as_perm_group_stab_chain(); }
	memory& change_to_memory() { freeself(); c_kind(MEMORY); return as_memory(); }
	//action& change_to_action() { freeself(); c_kind(ACTION); return as_action(); }
	//geometry& change_to_geometry() { freeself(); c_kind(GEOMETRY); return as_geometry(); }
	hollerith& change_to_hollerith() { freeself(); c_kind(HOLLERITH); return as_hollerith(); }
	//group_selection& change_to_group_selection() { freeself(); c_kind(GROUP_SELECTION); return as_group_selection(); }
	solid& change_to_solid() { freeself(); c_kind(SOLID); return as_solid(); }
	bt_key& change_to_bt_key() { freeself(); c_kind(BT_KEY); return as_bt_key(); }
	database& change_to_database() { freeself(); c_kind(DATABASE); return as_database(); }
	btree& change_to_btree() { freeself(); c_kind(BTREE); return as_btree(); }
	design_parameter_source& change_to_design_parameter_source() { freeself(); c_kind(DESIGN_PARAMETER_SOURCE); return as_design_parameter_source(); }
	design_parameter& change_to_design_parameter() { freeself(); c_kind(DESIGN_PARAMETER); return as_design_parameter(); }

	void *operator new(size_t, void *p) { return p; } 
	void settype_base();

	kind s_kind();
		// select kind of object
	virtual kind s_virtual_kind();
	void c_kind(kind k);
		// compute kind of object:
		// changes the object kind to class k
		// preserves the self part of the object
	void swap(discreta_base &a);
	void copyobject(discreta_base &x);
		// this := x
	virtual void copyobject_to(discreta_base &x);
		// x := this

	virtual std::ostream& print(std::ostream&);
		// all kinds of printing, the current printing mode is determined 
		// by the global variable printing_mode
	void print_to_hollerith(hollerith& h);
	std::ostream& println(std::ostream&);
		// print() and newline
	std::ostream& printobjectkind(std::ostream&);
		// prints the type of the object
	std::ostream& printobjectkindln(std::ostream&);

	int& s_i_i();
		// select_as_integer_i
	void m_i_i(int i);
		// make_as_integer_i

	virtual int compare_with(discreta_base &a);
		// -1 iff this < a
		// 0 iff this = a
		// 1 iff this > a
	int eq(discreta_base &a);
	int neq(discreta_base &a);
	int le(discreta_base &a);
	int lt(discreta_base &a);
	int ge(discreta_base &a);
	int gt(discreta_base &a);
	int is_even();
	int is_odd();
	
	
	// mathematical functions:
    
	// multiplicative group:
	void mult(discreta_base &x, discreta_base &y);
		// this := x * y
	void mult_mod(discreta_base &x, discreta_base &y, discreta_base &p);
	virtual void mult_to(discreta_base &x, discreta_base &y);
		// y := this * x
	int invert();
		// this := this^(-1)
		// returns TRUE if the object was invertible,
		// FALSE otherwise
	int invert_mod(discreta_base &p);
	virtual int invert_to(discreta_base &x);
	void mult_apply(discreta_base &x);
		// this := this * x
	discreta_base& operator *= (discreta_base &y)
		{ mult_apply(y); return *this; }
	discreta_base& power_int(int l);
		// this := this^l, l >= 0
	discreta_base& power_int_mod(int l, discreta_base &p);
	discreta_base& power_longinteger(longinteger &l);
	discreta_base& power_longinteger_mod(longinteger &l, discreta_base &p);
	discreta_base& commutator(discreta_base &x, discreta_base &y);
		// this := x^{-1} * y^{-1} * x * y
	discreta_base& conjugate(discreta_base &x, discreta_base &y);
		// this := y^{-1} * x * y
	discreta_base& divide_by(discreta_base& x);
	discreta_base& divide_by_exact(discreta_base& x);
	int order();
	int order_mod(discreta_base &p);
	

	// additive group:
	void add(discreta_base &x, discreta_base &y);
		// this := x + y
	void add_mod(discreta_base &x, discreta_base &y, discreta_base &p);
	virtual void add_to(discreta_base &x, discreta_base &y);
		// y := this + x
	void negate();
		// this := -this;
	virtual void negate_to(discreta_base &x);
		// x := - this
	void add_apply(discreta_base &x);
		// this := this + x
	discreta_base& operator += (discreta_base &y)
		{ add_apply(y); return *this; }
	
	
	virtual void normalize(discreta_base &p);
	virtual void zero();
		// this := 0
	virtual void one();
		// this := 1
	virtual void m_one();
		// this := -1
	virtual void homo_z(int z);
		// this := z
	virtual void inc();
		// this := this + 1
	virtual void dec();
		// this := this - 1
	virtual int is_zero();
		// TRUE iff this = 0
	virtual int is_one();
		// TRUE iff this = 1
	virtual int is_m_one();
		// TRUE iff this = -1
	discreta_base& factorial(int z);
		// this := z!
	discreta_base& i_power_j(int i, int j);
		// this := i^j

	virtual int compare_with_euclidean(discreta_base &a);
		// -1 iff this < a
		// 0 iff this = a
		// 1 iff this > a
	virtual void integral_division(discreta_base &x,
			discreta_base &q, discreta_base &r, int verbose_level);
	void integral_division_exact(discreta_base &x, discreta_base &q);
	void integral_division_by_integer(int x, discreta_base &q, discreta_base &r);
	void integral_division_by_integer_exact(int x, discreta_base &q);
	void integral_division_by_integer_exact_apply(int x);
	int is_divisor(discreta_base &y);
	void modulo(discreta_base &p);
	void extended_gcd(discreta_base &n, discreta_base &u,
			discreta_base &v, discreta_base &g, int verbose_level);
	void write_memory(memory &m, int debug_depth);
	void read_memory(memory &m, int debug_depth);
	int calc_size_on_file();
	void pack(memory & M, int verbose_level, int debug_depth);
	void unpack(memory & M, int verbose_level, int debug_depth);
	void save_ascii(std::ostream & f);
	void load_ascii(std::istream & f);
	void save_file(char *fname);
	void load_file(char *fname);
};


//! DISCRETA class to serialize data structures


class memory: public discreta_base
{
	public:
	memory();
	memory(const discreta_base &x);
		// copy constructor
	memory& operator = (const discreta_base &x);
		// copy assignment
	void settype_memory();
	~memory();
	void freeself_memory();
	kind s_virtual_kind();
	void copyobject_to(discreta_base &x);
	std::ostream& print(std::ostream& ost);
	int & alloc_length() { return self.int_pointer[-3]; }
	int & used_length() { return self.int_pointer[-2]; }
	int & cur_pointer() { return self.int_pointer[-1]; }

	char & s_i(int i) { return self.char_pointer[i]; };
	char & operator [] (int i) { return s_i(i); }

	void init(int length, char *d);
	void alloc(int length);
	void append(int length, char *d);
	void realloc(int new_length);
	void write_char(char c);
	void read_char(char *c);
	void write_int(int i);
	void read_int(int *i);
	void read_file(char *fname, int verbose_level);
	void write_file(char *fname, int verbose_level);
	int multiplicity_of_character(char c);
	void compress(int f_v);
	void decompress(int f_v);
	int csf();
	void write_mem(memory & M, int debug_depth);
	void read_mem(memory & M, int debug_depth);
	
};


//! DISCRETA string class


class hollerith: public discreta_base
{
	public:
	hollerith();
		// constructor, sets the hollerith_pointer to NULL
	hollerith(char *p);
	hollerith(const discreta_base& x);
		// copy constructor
	hollerith& operator = (const discreta_base &x);
		// copy assignment

	void *operator new(size_t, void *p) { return p; } 
	void settype_hollerith();

	~hollerith();
	void freeself_hollerith();
		// delete the matrix
	kind s_virtual_kind();
	void copyobject_to(discreta_base &x);

	std::ostream& print(std::ostream&);
	int compare_with(discreta_base &a);

	char * s_unchecked() { return self.char_pointer; } 
	char * s() { if (self.char_pointer)
		return self.char_pointer; else return (char *) ""; }
	void init(const char *p);
	void append(const char *p);
	void append_i(int i);
	void write_mem(memory & m, int debug_depth);
	void read_mem(memory & m, int debug_depth);
	int csf();
	void chop_off_extension_if_present(char *ext);
	void get_extension_if_present(char *ext);
	void get_current_date();
};

//! DISCRETA integer class



class integer: public discreta_base
{
	public:
	integer();
	integer(char *p);
	integer(int i);
	integer(const discreta_base& x);
		// copy constructor
	integer& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_integer();

	~integer();
	void freeself_integer();
	kind s_virtual_kind();
	void copyobject_to(discreta_base &x);

	std::ostream& print(std::ostream&);

	integer& m_i(int i);				// make_integer
	int& s_i() { return self.integer_value; };	// select_integer

	int compare_with(discreta_base &a);

	void mult_to(discreta_base &x, discreta_base &y);
	int invert_to(discreta_base &x);
	
	void add_to(discreta_base &x, discreta_base &y);
	void negate_to(discreta_base &x);
	
	void normalize(discreta_base &p);
	void zero();
	void one();
	void m_one();
	void homo_z(int z);
	void inc();
	void dec();
	int is_zero();
	int is_one();
	int is_m_one();

	int compare_with_euclidean(discreta_base &a);
	void integral_division(discreta_base &x,
			discreta_base &q, discreta_base &r, int verbose_level);
	
	void rand(int low, int high);
	int log2();
};

#define LONGINTEGER_PRINT_DOTS
#define LONGINTEGER_DIGITS_FOR_DOT 6



//! DISCRETA  class for integers of arbitrary magnitude




class longinteger: public discreta_base
{
	public:
	longinteger();
	longinteger(int a);
	longinteger(const char *s);
	longinteger(const discreta_base& x);
		// copy constructor
	longinteger& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_longinteger();

	~longinteger();
	void freeself_longinteger();
	kind s_virtual_kind();
	void copyobject_to(discreta_base &x);

	std::ostream& print(std::ostream&);
	
	LONGINTEGER_REPRESENTATION *s_rep();
	int& s_sign();
	int& s_len();
	char& s_p(int i);
	void allocate(int sign, const char *p);
	void allocate_internal(int sign, int len, const char *p);
	void allocate_empty(int len);
	void normalize_representation();
	
	int compare_with(discreta_base &b);
	int compare_with_unsigned(longinteger &b);

	void mult_to(discreta_base &x, discreta_base &y);
	int invert_to(discreta_base &x);
	void add_to(discreta_base &x, discreta_base &y);
	void negate_to(discreta_base &x);
	
	void zero();
	void one();
	void m_one();
	void homo_z(int z);
	void inc();
	void dec();
	int is_zero();
	int is_one();
	int is_m_one();
	int is_even();
	int is_odd();

	int compare_with_euclidean(discreta_base &b);
	void integral_division(discreta_base &x,
			discreta_base &q, discreta_base &r, int verbose_level);
	void square_root_floor(discreta_base &x);
	longinteger& Mersenne(int n);
	longinteger& Fermat(int n);
	int s_i();
	int retract_to_integer_if_possible(integer &x);
	int modp(int p);
	int ny_p(int p);
	void divide_out_int(int d);

	int Lucas_test_Mersenne(int m, int verbose_level);
};


//! DISCRETA vector class for vectors of DISCRETA objects




class Vector: public discreta_base
{
	public:
	Vector();
		// constructor, sets the vector_pointer to NULL
	Vector(const discreta_base& x);
		// copy constructor
	Vector& operator = (const discreta_base &x);
		// copy assignment
	
	void *operator new(size_t, void *p) { return p; } 
	void settype_vector();
	~Vector();
	void freeself_vector();
		// delete the vector
	kind s_virtual_kind();
	void copyobject_to(discreta_base &x);
	
	std::ostream& Print(std::ostream&);
	std::ostream& print(std::ostream&);
	std::ostream& print_unformatted(std::ostream& ost);
	std::ostream& print_intvec(std::ostream& ost);
	
	discreta_base & s_i(int i);
		// select i-th vector element
	int& s_ii(int i) 
		{ return s_i(i).s_i_i(); }
		// select i-th vector element as integer
	void m_ii(int i, int a) { s_i(i).m_i_i(a); }
		// make i-th vector element as integer (set value)
	discreta_base & operator [] (int i)
		{ return s_i(i); }
	int s_l();			
		// select vector length, 
		// length is 0 if vector_pointer is NULL
	void m_l(int l);
		// make vector of length l
		// allocates the memory and sets the objects to type BASE
	void m_l_n(int l);
		// make vector of length l of integers, initializes with 0
	void m_l_e(int l);
		// make vector of length l of integers, initializes with 1
	void m_l_x(int l, discreta_base &x);
		// allocates a vector of l copies of x
	Vector& realloc(int l);
	void mult_to(discreta_base &x, discreta_base &y);
	void add_to(discreta_base &x, discreta_base &y);
	void inc();
	void dec();

	int compare_with(discreta_base &a);
	void append_vector(Vector &v);
	Vector& append_integer(int a);
	Vector& append(discreta_base &x);
	Vector& insert_element(int i, discreta_base& x);
	Vector& get_and_delete_element(int i, discreta_base& x);
	Vector& delete_element(int i);
	void get_first_and_remove(discreta_base & x);
	bool insert_sorted(discreta_base& x);
		// inserts x into the sorted vector x.
		// ifthere are already occurences of x, the new x is added 
		// behind the x already there.
		// returns true if the element was already in the vector.
	bool search(discreta_base& x, int *idx);
		// returns TRUE if the object x has been found. 
		// idx contains the position where the object which 
		// has been found lies. 
		// if there are more than one element equal to x in the vector, 
		// the last one will be found. 
		// if the element has not been found, idx contains the position of 
		// the next larger element. 
		// This is the position to insert x if required.
	Vector& sort();
	void sort_with_fellow(Vector &fellow);
	Vector& sort_with_logging(permutation& p);
		// the permutation p tells where the sorted elements 
		// lay before, i.e. p[i] is the position of the
		// sorted element i in the unsorted vector.


	void sum_of_all_entries(discreta_base &x);





	void n_choose_k_first(int n, int k);
		// computes the lexicographically first k-subset of {0,...,n-1}
	int n_choose_k_next(int n, int k);
		// computes the lexicographically next k-subset
		// returns FALSE if there is no further k-subset
		// example: n = 4, k = 2
		// first gives (0,1),
		// next gives (0,2), then (0,3), (1,2), (1,3), (2,3).
	void first_lehmercode(int n) 
		{ m_l_n(n); }
		// first lehmercode = 0...0 (n times)
	int next_lehmercode();
		// computes the next lehmercode,
		// returns FALSE iff there is no next lehmercode.
		// the last lehmercode is n-1,n-2,...,2,1,0
		// example: n = 3,
		// first_lehmercode gives 
		// 0,0,0,0
		// next_lehmercode gives
		// 0, 1, 0
		// 1, 0, 0
		// 1, 1, 0
		// 2, 0, 0
		// 2, 1, 0
	void lehmercode2perm(permutation& p);
	void q_adic(int n, int q);
	int q_adic_as_int(int q);
	void mult_scalar(discreta_base& a);
	void first_word(int n, int q);
	int next_word(int q);
	void first_regular_word(int n, int q);
	int next_regular_word(int q);
	int is_regular_word();

	void apply_permutation(permutation &p);
	void apply_permutation_to_elements(permutation &p);
	void content(Vector & c, Vector & where);
	void content_multiplicities_only(Vector & c, Vector & mult);

	int hip();
	int hip1();
	void write_mem(memory & m, int debug_depth);
	void read_mem(memory & m, int debug_depth);
	int csf();

	void conjugate(discreta_base & a);
	void conjugate_with_inverse(discreta_base & a);
	void replace(Vector &v);
	void vector_of_vectors_replace(Vector &v);
	void extract_subvector(Vector & v, int first, int len);

	void PG_element_normalize();
	void PG_element_rank(int &a);
	void PG_element_rank_modified(int &a);
	void PG_element_unrank(int a);
	void PG_element_unrank_modified(int a);
	void AG_element_rank(int &a);
	void AG_element_unrank(int a);
	
	int hamming_weight();
	void scalar_product(Vector &w, discreta_base & a);
	void hadamard_product(Vector &w);
	void intersect(Vector& b, Vector &c);
	int vector_of_vectors_overall_length();
	void first_divisor(Vector &exponents);
	int next_divisor(Vector &exponents);
	int next_non_trivial_divisor(Vector &exponents);
	void multiply_out(Vector &primes, discreta_base &x);
	int hash(int hash0);
	int is_subset_of(Vector &w);
	void concatenation(Vector &v1, Vector &v2);
	void print_word_nicely(std::ostream &ost,
			int f_generator_labels, Vector &generator_labels);
	void print_word_nicely2(std::ostream &ost);
	void print_word_nicely_with_generator_labels(
			std::ostream &ost, Vector &generator_labels);
	void vector_of_vectors_lengths(Vector &lengths);
	void get_element_orders(Vector &vec_of_orders);
};

void merge(Vector &v1, Vector &v2, Vector &v3);
void merge_with_fellows(Vector &v1, Vector &v1_fellow, 
	Vector &v2, Vector &v2_fellow, 
	Vector &v3, Vector &v3_fellow);
void merge_with_value(Vector &idx1, Vector &idx2, Vector &idx3, 
	Vector &val1, Vector &val2, Vector &val3);
//int nb_PG_elements(int n, int q);
//int nb_AG_elements(int n, int q);
void intersection_of_vectors(Vector& V, Vector& v);


//! DISCRETA permutation class



class permutation: public Vector
{
	public:
	permutation();
		// constructor, sets the vector_pointer to NULL
	permutation(const discreta_base& x);
		// copy constructor
	permutation& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_permutation();
	kind s_virtual_kind();
	~permutation();
	void freeself_permutation();
	void copyobject_to(discreta_base &x);
	std::ostream& print(std::ostream&);
	std::ostream& print_list(std::ostream& ost);
	std::ostream& print_cycle(std::ostream& ost);
	void sscan(const char *s, int verbose_level);
	void scan(std::istream & is, int verbose_level);

	void m_l(int l);
	int& s_i(int i);
	int& operator [] (int i) 
		{ return s_i(i); }

	void mult_to(discreta_base &x, discreta_base &y);
	int invert_to(discreta_base &x);
	void one();
	int is_one();
	int compare_with(discreta_base &a);

	void write_mem(memory & m, int debug_depth);
	void read_mem(memory & m, int debug_depth);
	int csf();
	void get_fixpoints(Vector &f);
	void induce_action_on_blocks(permutation & gg, Vector & B);
	void induce3(permutation & b);
	void induce2(permutation & b);
	void induce_on_2tuples(permutation & p, int f_injective);
	void add_n_fixpoints_in_front(permutation & b, int n);
	void add_n_fixpoints_at_end(permutation & b, int n);
	void add_fixpoint_in_front(permutation & b);
	void embed_at(permutation & b, int n, int at);
	void remove_fixpoint(permutation & b, int i);
	void join(permutation & a, permutation & b);
	void cartesian_product_action(permutation & a, permutation & b);
	void Add2Cycle(int i0, int i1);
	void Add3Cycle(int i0, int i1, int i2);
	void Add4Cycle(int i0, int i1, int i2, int i3);
	void Add5Cycle(int i0, int i1, int i2, int i3, int i4);
	// void Add6Cycle(int i0, int i1, int i2, int i3, int i4, int i5);
	// void Add7Cycle(int i0, int i1, int i2, int i3, int i4, int i5, int i6);
	// void Add8Cycle(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7);
	void AddNCycle(int first, int len);

	// influence the behaviour of printing of permutations:
	void set_print_type_integer_from_zero();
	void set_print_type_integer_from_one();
	void set_print_type_PG_1_q_element(domain *dom);

	void convert_digit(int i, hollerith &a);
	void cycle_type(Vector& type, int verbose_level);
	int nb_of_inversions(int verbose_level);
	int signum(int verbose_level);
	int is_even(int verbose_level);
	void cycles(Vector &cycles);
	void restrict_to_subset(permutation &q, int first, int len);
	void induce_on_lines_of_PG_k_q(int k, int q, permutation &per, int verbose_level);
	void singer_cycle_on_points_of_projective_plane(int p, int f_modified, int verbose_level);
	void Cn_in_Cnm(int n, int m);
	int preimage(int i);
};

void signum_map(discreta_base & x, discreta_base &d);

//! DISCRETA utility class for matrix access




class matrix_access {
public:
	int i;
	discreta_matrix *p;
	discreta_base & operator [] (int j);
};

//! DISCRETA matrix class



class discreta_matrix: public discreta_base
{
	public:
	discreta_matrix();
		// constructor, sets the matrix_pointer to NULL
	discreta_matrix(const discreta_base& x);
		// copy constructor
	discreta_matrix& operator = (const discreta_base &x);
		// copy assignment

	void *operator new(size_t, void *p) { return p; } 
	void settype_matrix();

	~discreta_matrix();
	void freeself_matrix();
		// delete the matrix
	kind s_virtual_kind();
	void copyobject_to(discreta_base &x);

	std::ostream& print(std::ostream&);
	int compare_with(discreta_base &a);

	discreta_matrix& m_mn(int m, int n);
		// make matrix of format m times n
		// allocates the memory and sets the objects to type BASE
	discreta_matrix& m_mn_n(int m, int n);
	discreta_matrix& realloc(int m, int n);


	int s_m();
	int s_n();
	discreta_base & s_ij(int i, int j);
		// select (i,j)-th matrix element
	int& s_iji(int i, int j)
		{ return s_ij(i, j).s_i_i(); }
	void m_iji(int i, int j, int a) 
		{ s_ij(i, j).m_i_i(a); }
		// make (i,j)-th vector element as integer (set value)
	
	matrix_access operator [] (int i) 
		{ matrix_access ma = { i, this };  return ma; }
		// overload access operator

	void mult_to(discreta_base &x, discreta_base &y);
	void matrix_mult_to(discreta_matrix &x, discreta_base &y);
	void vector_mult_to(Vector &x, discreta_base &y);
	void multiply_vector_from_left(Vector &x, Vector &y);
	int invert_to(discreta_base &x);
	void add_to(discreta_base &x, discreta_base &y);
	void negate_to(discreta_base &x);
	void one();
	void zero();
	int is_zero();
	int is_one();


	int Gauss(int f_special, int f_complete,
			Vector& base_cols, int f_P, discreta_matrix& P, int verbose_level);
	int rank();
	int get_kernel(Vector& base_cols, discreta_matrix& kernel);
	discreta_matrix& transpose();
	int Asup2Ainf();
	int Ainf2Asup();
	int Asup2Acover();
	int Acover2nl(Vector& nl);

	void Frobenius(unipoly& m, int p, int verbose_level);
	void Berlekamp(unipoly& m, int p, int verbose_level);
	void companion_matrix(unipoly& m, int verbose_level);
	
	void elements_to_unipoly();
	void minus_X_times_id();
	void X_times_id_minus_self();
	void smith_normal_form(discreta_matrix& P, discreta_matrix& Pv,
			discreta_matrix& Q, discreta_matrix& Qv, int verbose_level);
	int smith_eliminate_column(discreta_matrix& P, discreta_matrix& Pv, int i,
			int verbose_level);
	int smith_eliminate_row(discreta_matrix& Q, discreta_matrix& Qv, int i,
			int verbose_level);
	void multiply_2by2_from_left(int i, int j, 
		discreta_base& aii, discreta_base& aij,
		discreta_base& aji, discreta_base& ajj,
		int verbose_level);
	void multiply_2by2_from_right(int i, int j, 
		discreta_base& aii, discreta_base& aij,
		discreta_base& aji, discreta_base& ajj,
		int verbose_level);

	void to_vector_of_rows(Vector& v);
	void from_vector_of_rows(Vector& v);
	void to_vector_of_columns(Vector& v);
	void from_vector_of_columns(Vector& v);
	void evaluate_at(discreta_base& x);
	void KX_module_order_ideal(int i, unipoly& mue, int verbose_level);
	void KX_module_apply(unipoly& p, Vector& v);
	void KX_module_join(Vector& v1, unipoly& mue1, 
		Vector& v2, unipoly& mue2, Vector& v3, unipoly& mue3, int verbose_level);
	void KX_cyclic_module_generator(Vector& v, unipoly& mue, int verbose_level);
	void KX_module_minpol(unipoly& p, unipoly& m, unipoly& mue, int verbose_level);

	void binomial(int n_min, int n_max, int k_min, int k_max);
	void stirling_second(int n_min, int n_max, int k_min, int k_max, int f_ordered);
	void stirling_first(int n_min, int n_max, int k_min, int k_max, int f_signless);
	void binomial(int n_min, int n_max, int k_min, int k_max, int f_inverse);
	int hip();
	int hip1();
	void write_mem(memory & m, int debug_depth);
	void read_mem(memory & M, int debug_depth);
	int csf();

	void calc_theX(int & nb_X, int *&theX);
	void apply_perms(int f_row_perm, permutation &row_perm, 
		int f_col_perm, permutation &col_perm);
	void apply_col_row_perm(permutation &p);
	void apply_row_col_perm(permutation &p);
	void incma_print_ascii_permuted_and_decomposed(std::ostream &ost, int f_tex,
		Vector & decomp, permutation & p);
	void print_decomposed(std::ostream &ost, Vector &row_decomp, Vector &col_decomp);
	void incma_print_ascii(std::ostream &ost, int f_tex,
		int f_row_decomp, Vector &row_decomp, 
		int f_col_decomp, Vector &col_decomp);
	void incma_print_latex(std::ostream &f,
		int f_row_decomp, Vector &row_decomp, 
		int f_col_decomp, Vector &col_decomp, 
		int f_labelling_points, Vector &point_labels, 
		int f_labelling_blocks, Vector &block_labels);
	void incma_print_latex2(std::ostream &f,
		int width, int width_10, 
		int f_outline_thin, const char *unit_length, 
		const char *thick_lines, const char *thin_lines, const char *geo_line_width, 
		int f_row_decomp, Vector &row_decomp, 
		int f_col_decomp, Vector &col_decomp, 
		int f_labelling_points, Vector &point_labels, 
		int f_labelling_blocks, Vector &block_labels);
	void calc_hash_key(int key_len, hollerith & hash_key, int f_v);
	int is_in_center();
	void power_mod(int r, integer &P, discreta_matrix &C);
	int proj_order_mod(integer &P);
	void PG_rep(domain *dom, permutation &p, int f_action_from_right, int f_modified);
	void PG_rep(permutation &p, int f_action_from_right, int f_modified);
	void AG_rep(domain *dom, permutation &p, int f_action_from_right);
	void AG_rep(permutation &p, int f_action_from_right);
	void MacWilliamsTransform(int n, int q, int f_v);
	void weight_enumerator_brute_force(domain *dom, Vector &v);
	void Simplex_code_generator_matrix(domain *dom, int k, int f_v);
	void PG_design_point_vs_hyperplane(domain *dom, int k, int f_v);
	void PG_k_q_design(domain *dom, int k, int f_v, int f_vv);
	void determinant(discreta_base &d, int verbose_level);
	void det(discreta_base & d, int f_v, int f_vv);
	void det_modify_input_matrix(discreta_base & d, int f_v, int f_vv);
	void PG_line_rank(int &a, int f_v);
	void PG_line_unrank(int a);
	void PG_point_normalize(int i0, int j0, int di, int dj, int length);
	void PG_point_unrank(int i0, int j0, int di, int dj, int length, int a);
	void PG_point_rank(int i0, int j0, int di, int dj, int length, int &a);
	void PG_element_normalize();
	void AG_point_rank(int i0, int j0, int di, int dj, int length, int &a);
	void AG_point_unrank(int i0, int j0, int di, int dj, int length, int a);
	void save_as_inc_file(char *fname);
	void save_as_inc(std::ofstream &f);
};

void determinant_map(discreta_base & x, discreta_base &d);
int nb_PG_lines(int n, int q);


#if 0
//! DISCRETA bitmatrix class




class bitmatrix: public discreta_base
{
	public:
	bitmatrix();
		// constructor, sets the bitmatrix_pointer to NULL
	bitmatrix(const discreta_base& x);
		// copy constructor
	bitmatrix& operator = (const discreta_base &x);
		// copy assignment

	void *operator new(size_t, void *p) { return p; } 
	void settype_bitmatrix();

	~bitmatrix();
	void freeself_bitmatrix();
		// delete the matrix
	kind s_virtual_kind();
	void copyobject_to(discreta_base &x);

	ostream& print(ostream&);

	bitmatrix& m_mn(int m, int n);
		// make matrix of format m times n
		// allocates the memory
	bitmatrix& m_mn_n(int m, int n);


	int s_m();
	int s_n();
	int s_N();
	uint4& s_i(int i);
	int s_ij(int i, int j);
		// select (i,j)-th matrix element
	void m_iji(int i, int j, int a);
		// make (i,j)-th vector element as integer (set value)

	void mult_to(discreta_base &x, discreta_base &y);
	void bitmatrix_mult_to(bitmatrix &x, discreta_base &y);
	
	int gauss(int f_complete, Vector& base_cols, int f_v);
	int get_kernel(Vector& base_cols, bitmatrix& kernel);

	void write_mem(memory & M, int debug_depth);
	void read_mem(memory & M, int debug_depth);
	int csf();
};
#endif

//! DISCRETA class for poynomials in one variable



class unipoly: public Vector
{
	public:
	unipoly();
		// constructor, sets the vector_pointer to NULL
	unipoly(const discreta_base& x);
		// copy constructor
	unipoly& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_unipoly();
	kind s_virtual_kind();
	~unipoly();
	void freeself_unipoly();
	void copyobject_to(discreta_base &x);
	std::ostream& print(std::ostream&);
	std::ostream& print_as_vector(std::ostream& ost);

	void m_l(int l);
	int degree();

	void mult_to(discreta_base &x, discreta_base &y);
	void add_to(discreta_base &x, discreta_base &y);
	void negate_to(discreta_base &x);
	void one();
	void zero();
	void x();
	void x_to_the_i(int i);
	int is_one();
	int is_zero();
	int compare_with_euclidean(discreta_base &a);
	void integral_division(discreta_base &x,
			discreta_base &q, discreta_base &r, int verbose_level);
	void derive();
	int is_squarefree(int verbose_level);
	int is_irreducible_GFp(int p, int verbose_level);
	int is_irreducible(int q, int verbose_level);
	int is_primitive(int m, int p, Vector& vp, int verbose_level);
	void numeric_polynomial(int n, int q);
	int polynomial_numeric(int q);
	void singer_candidate(int p, int f, int b, int a);
	void Singer(int p, int f, int verbose_level);
	void get_an_irreducible_polynomial(int f, int verbose_level);
	void evaluate_at(discreta_base& x, discreta_base& y);
	void largest_divisor_prime_to(unipoly& q, unipoly& r);
	void monic();
	void normal_base(int p, discreta_matrix& F, discreta_matrix& N, int verbose_level);
	int first_irreducible_polynomial(int p,
			unipoly& m, discreta_matrix& F, discreta_matrix& N, Vector &v,
			int verbose_level);
	int next_irreducible_polynomial(int p,
			unipoly& m, discreta_matrix& F, discreta_matrix& N, Vector &v,
			int verbose_level);
	void normalize(discreta_base &p);
	void Xnm1(int n);
	void Phi(int n, int f_v);
	void weight_enumerator_MDS_code(int n, int k, int q,
			int verbose_level);
	void charpoly(int q, int size, int *mtx, int verbose_level);
	
};






#define PARTITION_TYPE_VECTOR 0
#define PARTITION_TYPE_EXPONENT 1

//! DISCRETA class for partitions of an integer



class number_partition: public Vector 
{
	public:
	number_partition();
		// constructor, sets the vector_pointer to NULL
	number_partition(int n);
	void allocate_number_partition();
	number_partition(const discreta_base& x);
		// copy constructor
	number_partition& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_number_partition();
	kind s_virtual_kind();
	~number_partition();
	void freeself_number_partition();
	void copyobject_to(discreta_base &x);
	std::ostream& print(std::ostream&);

	int & s_type() { return Vector::s_i(0).as_integer().s_i(); }
	Vector & s_self() { return Vector::s_i(1).as_vector(); }
	
	void m_l(int l) { s_self().m_l_n(l); }
	int s_l() { return s_self().s_l(); }
	int & s_i(int i) { return s_self().s_ii(i); }
	int & operator [] (int i) { return s_self().s_ii(i); }

	void first(int n);
	int next();
	int next_exponent();
	int next_vector();
	int first_into_k_parts(int n, int k);
	int next_into_k_parts(int n, int k);
	int first_into_at_most_k_parts(int n, int k);
	int next_into_at_most_k_parts(int n, int k);
	int nb_parts();
	void conjugate();
	void type(number_partition &q);
	void multinomial(discreta_base &res, int f_v);
	void multinomial_ordered(discreta_base &res, int f_v);
	int sum_of_decreased_parts();

};



//! DISCRETA class for influencing arithmetic operations




class domain {
	private:
		domain_type the_type;
		discreta_base the_prime;
		//pc_presentation *the_pres;
		unipoly *the_factor_poly;
		domain *the_sub_domain;
		finite_field *F;
	
	public:
		domain(int p);
		domain(finite_field *F);
	domain(unipoly *factor_poly, domain *sub_domain);
	//domain(pc_presentation *pres);
	
	domain_type type();
	finite_field *get_F();
	int order_int();
	int order_subfield_int();
	int characteristic();
	int is_Orbiter_finite_field_domain();
	//pc_presentation *pres();
	unipoly *factor_poly();
	domain *sub_domain();
};


// domain.cpp:

int has_domain();
domain *get_current_domain();
//domain *get_domain_if_pc_group();
int is_GFp_domain(domain *& d);
int is_GFq_domain(domain *& d);
int is_Orbiter_finite_field_domain(domain *& d);
int is_finite_field_domain(domain *& d);
int finite_field_domain_order_int(domain * d);
int finite_field_domain_characteristic(domain * d);
int finite_field_domain_primitive_root();
void finite_field_domain_base_over_subfield(Vector & b);
void push_domain(domain *d);
void pop_domain(domain *& d);
domain *allocate_finite_field_domain(int q, int verbose_level);
void free_finite_field_domain(domain *dom);




//! DISCRETA class related to class domain




class with {
	private:
	public:

	with(domain *dom);
	~with();
};


//! DISCRETA  class related to printing of objects



class printing_mode {
	private:
	public:
	printing_mode(enum printing_mode_enum printing_mode);
	~printing_mode();
};


//! DISCRETA class for polyhedra




class solid: public Vector
{
	public:
	solid();
		// constructor, sets the Vector_pointer to NULL
	void init();
		// initialize trivially all components of solid
	
	Vector& group_generators() { return s_i(0).as_vector(); }
	permutation& group_generators_i(int i) { return group_generators().s_i(i).as_permutation(); }
	int& nb_V() { return s_ii(1); }
	int& nb_E() { return s_ii(2); }
	int& nb_F() { return s_ii(3); }
	Vector& placement() { return s_i(4).as_vector(); } 	/* of vertex */
	Vector& x() { return placement().s_i(0).as_vector(); }	/* of vertex */
	int& x_i(int i) { return x().s_ii(i); }			/* of vertex */
	Vector& y() { return placement().s_i(1).as_vector(); }	/* of vertex */
	int& y_i(int i) { return y().s_ii(i); }			/* of vertex */
	Vector& z() { return placement().s_i(2).as_vector(); }	/* of vertex */
	int& z_i(int i) { return z().s_ii(i); }			/* of vertex */
	Vector& v1() { return s_i(5).as_vector(); }		/* at edge */
	int& v1_i(int i) { return v1().s_ii(i); }		/* at edge */
	Vector& v2() { return s_i(6).as_vector(); }		/* at edge */
	int& v2_i(int i) { return v2().s_ii(i); }		/* at edge */
	Vector& f1() { return s_i(7).as_vector(); }		/* at edge */
	int& f1_i(int i) { return f1().s_ii(i); }		/* at edge */
	Vector& f2() { return s_i(8).as_vector(); }		/* at edge */
	int& f2_i(int i) { return f2().s_ii(i); }		/* at edge */
	Vector& nb_e() { return s_i(9).as_vector(); }		/* at face */
	int& nb_e_i(int i) { return nb_e().s_ii(i); }		/* at face */
	Vector& edge() { return s_i(10).as_vector(); }		/* at face */
	Vector& edge_i(int i) 
		{ return edge().s_i(i).as_vector(); }		/* at face */
	int& edge_ij(int i, int j) 
		{ return edge_i(i).s_ii(j); }			/* at face */
	Vector& neighbour_faces() 
		{ return s_i(11).as_vector(); }			/* at face */
	Vector& neighbour_faces_i(int i) 
		{ return neighbour_faces().s_i(i).as_vector(); }/* at face */
	int& neighbour_faces_ij(int i, int j) 
		{ return neighbour_faces_i(i).s_ii(j); }	/* at face */
	int& f_vertex_labels() { return s_ii(12); }
	Vector& vertex_labels() { return s_i(13).as_vector(); } /* of vertex */
	hollerith& vertex_labels_i(int i) 
		{ return vertex_labels().s_i(i).as_hollerith(); }	/* of vertex */
	Vector& vertex_labels_numeric() { return s_i(14).as_vector(); } /* of vertex */
	int& vertex_labels_numeric_i(int i) 
		{ return vertex_labels_numeric().s_ii(i); } /* of vertex */
	int& f_oriented() { return s_ii(15); }
	
	void init_V(int nb_V);
		// initialize vertices
	void init_E(int nb_E);
		// initialize edges
	void init_F(int nb_F);
		// initialize faces
	solid(const discreta_base& x);
		// copy constructor
	solid& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_solid();
	kind s_virtual_kind();
	~solid();
	void freeself_solid();
	void copyobject_to(discreta_base &x);
	std::ostream& print_list(std::ostream& ost);
	std::ostream& print(std::ostream& ost);
	void standard_vertex_labels(int f_start_with_zero);
	void determine_neighbours();
	void find_face(int e, int& f1, int& j1, int& f2, int& j2);
	int find_face_2(int e1, int e2);
	int find_face_by_two_edges(int e1, int e2);
	void find_faces_at_edge(int e, int& f1, int& f2);
	int find_edge(int v1, int v2);
	void add_edge(int v1, int v2, int f1, int f2);
	int add_edge(int v1, int v2);
	int find_and_add_edge(int i1, int i2, int f_v);
	void add_face3(int e1, int e2, int e3, int i1, int i2, int i3);
	void add_face4(int i1, int i2, int i3, int i4);
	void add_face5(int i1, int i2, int i3, int i4, int i5);
	void add_face_n(Vector& vertices);
	void adjacency_list(int vertex, int *adj, int *nb_adj);
	void center(int f, Vector& Px, Vector& Py, Vector& Pz);
	void vertices_of_face(int i, Vector& V);
	void Ratio(int e, double r, int& x, int& y, int& z);
	int find_common_face(int e1, int e2, int& f);
	void dual(solid& A);
	void cut_vertices(double r, solid & A);
	void edge_midpoints(solid& A);
	void join_disjoint(solid& A, solid& J, int f_v);
	//void direct_sum(solid& B, solid& J, int f_v);
	//void direct_product(Vector& gen, solid& J, int f_v);
	void scale(double f);
	//void add_central_point(solid& A);
	void induced_action_on_edges(permutation& p, permutation& q);
	void induced_group_on_edges(Vector & gen, Vector & gen_e);
	
	void tetrahedron(int r);
	void cube(int r);
	//void cube4D(int r1, int r2);
	void octahedron(int r);
	void dodecahedron(int r);
	void icosahedron(int r);
	void make_placed_graph(discreta_matrix & incma, Vector& aut_gens, Vector& cycles);
		
	void write_graphfile(char *fname);
	void write_solidfile(char *fname);
};
//void vec_generators_aut_cube_nd(int n, Vector &gen);
void number_to_binary(int n, int *v, int digits);
int binary_to_number(int *v, int digits);




//! DISCRETA class for databases



class bt_key: public Vector  
{
	public:
	bt_key();
		// constructor, sets the vector_pointer to NULL
	bt_key(const discreta_base& x);
		// copy constructor
	bt_key& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_bt_key();
	kind s_virtual_kind();
	~bt_key();
	void freeself_bt_key();
	void copyobject_to(discreta_base &x);
	std::ostream& print(std::ostream&);

	enum bt_key_kind & type() { return (enum bt_key_kind&) Vector::s_i(0).as_integer().s_i(); }
	int & output_size() { return Vector::s_i(1).as_integer().s_i(); }
	int & int_vec_first() { return Vector::s_i(2).as_integer().s_i(); }
	int & int_vec_len() { return Vector::s_i(3).as_integer().s_i(); }
	int & field1() { return Vector::s_i(4).as_integer().s_i(); }
	int & field2() { return Vector::s_i(5).as_integer().s_i(); }
	int & f_ascending() { return Vector::s_i(6).as_integer().s_i(); }
	
	void init(enum bt_key_kind type, int output_size, int field1, int field2);
	void init_int4(int field1, int field2);
	void init_int2(int field1, int field2);
	void init_string(int output_size, int field1, int field2);
	void init_int4_vec(int field1, int field2, int vec_fst, int vec_len);
	void init_int2_vec(int field1, int field2, int vec_fst, int vec_len);
};

int bt_lexicographic_cmp(char *p1, char *p2);
int bt_key_int_cmp(char *p1, char *p2);
int bt_key_int2_cmp(char *p1, char *p2);
void bt_key_print_int4(char **key, std::ostream& ost);
void bt_key_print_int2(char **key, std::ostream& ost);
void bt_key_print(char *key, Vector& V, std::ostream& ost);
int bt_key_compare_int4(char **p_key1, char **p_key2);
int bt_key_compare_int2(char **p_key1, char **p_key2);
int bt_key_compare(char *key1, char *key2, Vector& V, int depth);
void bt_key_fill_in_int4(char **p_key, discreta_base& key_op);
void bt_key_fill_in_int2(char **p_key, discreta_base& key_op);
void bt_key_fill_in_string(char **p_key, int output_size, discreta_base& key_op);
void bt_key_fill_in(char *key, Vector& V, Vector& the_object);
void bt_key_get_int4(char **key, int_4 &i);
void bt_key_get_int2(char **key, int_2 &i);

#define BTREEMAXKEYLEN 24
//#define BTREEMAXKEYLEN 48
//#define BTREEMAXKEYLEN 512


//! DISCRETA auxiliary class related to the class database


typedef struct keycarrier {
	char c[BTREEMAXKEYLEN];
} KEYCARRIER;

typedef KEYCARRIER KEYTYPE;

//! DISCRETA auxiliary class related to the class database


typedef struct datatype {
	uint_4 datref;
	uint_4 data_size;
} DATATYPE;

//#define DB_SIZEOF_HEADER 16
//#define DB_SIZEOF_HEADER_LOG 4
#define DB_POS_FILESIZE 4

#define DB_FILE_TYPE_STANDARD 1
#define DB_FILE_TYPE_COMPACT 2


//! DISCRETA class for a database



class database: public Vector  
{
	public:
	database();
		// constructor, sets the vector_pointer to NULL
	database(const discreta_base& x);
		// copy constructor
	database& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_database();
	kind s_virtual_kind();
	~database();
	void freeself_database();
	void copyobject_to(discreta_base &x);
	std::ostream& print(std::ostream&);

	Vector & btree_access() { return Vector::s_i(0).as_vector(); }
	btree & btree_access_i(int i) { return btree_access().s_i(i).as_btree(); }
	hollerith & filename() { return Vector::s_i(1).as_hollerith(); }
	int & f_compress() { return Vector::s_i(2).as_integer().s_i(); }
	int & objectkind() { return Vector::s_i(3).as_integer().s_i(); }
	int & f_open() { return Vector::s_i(4).as_integer().s_i(); }
	int & stream() { return Vector::s_i(5).as_integer().s_i(); }
	int & file_size() { return Vector::s_i(6).as_integer().s_i(); }
	int & file_type() { return Vector::s_i(7).as_integer().s_i(); }

	void init(const char *filename, int objectkind, int f_compress);
	void init_with_file_type(const char *filename, 
		int objectkind, int f_compress, int file_type);
	
	void create(int verbose_level);
	void open(int verbose_level);
	void close(int verbose_level);
	void delete_files();
	void put_file_size();
	void get_file_size();
	void user2total(int user, int &total, int &pad);
	int size_of_header();
	int size_of_header_log();
	
	void add_object_return_datref(Vector &the_object,
			uint_4 &datref, int verbose_level);
	void add_object(Vector &the_object, int verbose_level);
	void delete_object(Vector& the_object, uint_4 datref, int verbose_level);
	void get_object(uint_4 datref, Vector &the_object, int verbose_level);
	void get_object(DATATYPE *data_type, Vector &the_object, int verbose_level);
	void get_object_by_unique_int4(int btree_idx, 
		int id, Vector& the_object, int verbose_level);
	int get_object_by_unique_int4_if_there(int btree_idx, 
		int id, Vector& the_object, int verbose_level);
	int get_highest_int4(int btree_idx);
	void ith_object(int i, int btree_idx, 
		Vector& the_object, int verbose_level);
	void ith(int i, int btree_idx, 
		KEYTYPE *key_type, DATATYPE *data_type,
		int verbose_level);
	void print_by_btree(int btree_idx, std::ostream& ost);
	void print_by_btree_with_datref(int btree_idx, std::ostream& ost);
	void print_subset(Vector& datrefs, std::ostream& ost);
	void extract_subset(Vector& datrefs, 
		char *out_path, int verbose_level);
	void search_int4(int btree_idx, 
		int imin, int imax, Vector &datrefs, 
		int verbose_level);
	void search_int4_2dimensional(int btree_idx0, 
		int imin0, int imax0, 
		int btree_idx1, int imin1, int imax1, 
		Vector &datrefs, int verbose_level);
	void search_int4_multi_dimensional(Vector& btree_idx, 
		Vector& i_min, Vector &i_max, Vector& datrefs, 
		int verbose_level);

	int get_size_from_datref(uint_4 datref, int verbose_level);
	void add_data_DB(void *d, 
		int size, uint_4 *datref, int verbose_level);
	void add_data_DB_standard(void *d, 
		int size, uint_4 *datref, int verbose_level);
	void add_data_DB_compact(void *d, 
		int size, uint_4 *datref, int verbose_level);
	void free_data_DB(uint_4 datref, int size, int verbose_level);

	void file_open(int verbose_level);
	void file_create(int verbose_level);
	void file_close(int verbose_level);
	void file_seek(int offset);
	void file_write(void *p, int size, int nb);
	void file_read(void *p, int size, int nb);
};


#define BTREEHALFPAGESIZE  128
#define BTREEMAXPAGESIZE (2 * BTREEHALFPAGESIZE)

#define BTREE_PAGE_LENGTH_LOG 7

/* Dateiformat:
 * In Block 0 sind AllocRec/NextFreeRec/RootRec gesetzt.
 * Block 1..AllocRec sind Datenpages.
 * Die freien Bloecke sind ueber NextFreeRec verkettet.
 * Der letzte freie Block hat NIL als Nachfolger.
 * Dateigroesse = (AllocRec + 1) * sizeof(PageTyp) */


//! DISCRETA auxiliary class related to the class database


typedef struct itemtyp {
	KEYTYPE Key;
	DATATYPE Data;
	int_4 Childs; // Anzahl der Nachfolger ueber Ref
	int_4 Ref;
} ItemTyp;


//! DISCRETA auxiliary class related to the class database


typedef struct pagetyp {
	int_4 AllocRec;
	int_4 NextFreeRec;
	int_4 RootRec;

	int_4 NumItems;
	ItemTyp Item[BTREEMAXPAGESIZE + 1];
/* Item[0]           enthaelt keine Daten, 
 *                   nur Ref/Childs ist verwendet.
 * Item[1..NumItems] fuer Daten und 
 *                   Ref/Childs verwendet. */
} PageTyp;


//! DISCRETA auxiliary class related to the class database


typedef struct buffer {
	int_4 PageNum;
	int_4 unused;
	PageTyp Page;
	long align;
} Buffer;


//! DISCRETA class for a database



class btree: public Vector 
{
	public:
	btree();
		// constructor, sets the vector_pointer to NULL
	btree(const discreta_base& x);
		// copy constructor
	btree& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_btree();
	kind s_virtual_kind();
	~btree();
	void freeself_btree();
	void copyobject_to(discreta_base &x);
	std::ostream& print(std::ostream&);
	
	int & f_duplicatekeys() { return Vector::s_i(0).as_integer().s_i(); }
	Vector & key() { return Vector::s_i(1).as_vector(); }
	hollerith & filename() { return Vector::s_i(2).as_hollerith(); }
	int & f_open() { return Vector::s_i(3).as_integer().s_i(); }
	int & stream() { return Vector::s_i(4).as_integer().s_i(); }
	int & buf_idx() { return Vector::s_i(5).as_integer().s_i(); }
	int & Root() { return Vector::s_i(6).as_integer().s_i(); }
	int & FreeRec() { return Vector::s_i(7).as_integer().s_i(); }
	int & AllocRec() { return Vector::s_i(8).as_integer().s_i(); }
	int & btree_idx() { return Vector::s_i(9).as_integer().s_i(); }
	int & page_table_idx() { return Vector::s_i(10).as_integer().s_i(); }

	void init(const char *file_name, int f_duplicatekeys, int btree_idx);
	void add_key_int4(int field1, int field2);
	void add_key_int2(int field1, int field2);
	void add_key_string(int output_size, int field1, int field2);
	void key_fill_in(char *the_key, Vector& the_object);
	void key_print(char *the_key, std::ostream& ost);

	void create(int verbose_level);
	void open(int verbose_level);
	void close(int verbose_level);

	void ReadInfo(int verbose_level);
	void WriteInfo(int verbose_level);
	int AllocateRec(int verbose_level);
	void ReleaseRec(int x);
	void LoadPage(Buffer *BF, int x, int verbose_level);
	void SavePage(Buffer *BF, int verbose_level);

	int search_string(discreta_base& key_op,
		int& pos, int verbose_level);
	void search_interval_int4(int i_min, int i_max, 
		int& first, int &len, int verbose_level);
	void search_interval_int4_int4(int l0, int u0, 
		int l1, int u1, 
		int& first, int &len, 
		int verbose_level);
	void search_interval_int4_int4_int4(int l0, int u0, 
		int l1, int u1, 
		int l2, int u2, 
		int& first, int &len, 
		int verbose_level);
	void search_interval_int4_int4_int4_int4(int l0, int u0, 
		int l1, int u1, 
		int l2, int u2, 
		int l3, int u3, 
		int& first, int &len, 
		int verbose_level);
	int search_int4_int4(int data1, int data2, int &idx, int verbose_level);
	int search_unique_int4(int i, int verbose_level);
	int search_unique_int4_int4_int4_int4(int i0, 
		int i1, int i2, int i3, int verbose_level);
		// returns -1 if an element whose key starts with [i0,i1,i2,i3] could not be found or is not unique.
		// otherwise, the idx of that element is returned
	int search_datref_of_unique_int4(int i, 
		int verbose_level);
	int search_datref_of_unique_int4_if_there(int i, 
		int verbose_level);
	int get_highest_int4();
	void get_datrefs(int first, 
		int len, Vector& datrefs);

	int search(void *pSearchKey, 
		DATATYPE *pData, int *idx, int key_depth, 
		int verbose_level);
	int SearchBtree(int page, 
		void *pSearchKey, DATATYPE *pData, 
		Buffer *Buf, int *idx, int key_depth,
		int verbose_level);
	int SearchPage(Buffer *buffer, 
		void *pSearchKey, DATATYPE *pSearchData, 
		int *cur, int *x, int key_depth, 
		int verbose_level);

	int length(int verbose_level);
	void ith(int l, 
		KEYTYPE *key, DATATYPE *data, int verbose_level);
	int page_i_th(int l, 
		Buffer *buffer, int *cur, int *i, 
		int verbose_level);
	
	void insert_key(KEYTYPE *pKey, 
		DATATYPE *pData, 
		int verbose_level);
	void Update(int Node, int *Rise, 
		ItemTyp *RisenItem, 
		int *RisenNeighbourChilds, 
		int f_v);
	void Split(Buffer *BF, 
		ItemTyp *Item, int x, 
		int *RisenNeighbourChilds, 
		int verbose_level);

	void delete_ith(int idx, int verbose_level);
	void Delete(int Node, int& Underflow, int verbose_level);
	void FindGreatest(int Node1, 
		int& Underflow, Buffer *DKBF, int x, 
		int verbose_level);
	void Compensate(int Precedent, 
		int Node, int Path, int& Underflow,
		int verbose_level);
	
	void print_all(std::ostream& ost);
	void print_range(int first, int len, std::ostream& ost);
	void print_page(int x, std::ostream& ost);
	void page_print(Buffer *BF, std::ostream& ost);
	void item_print(ItemTyp *item, int i, std::ostream& ost);
	
	void file_open();
	void file_create();
	void file_close();
	void file_write(PageTyp *page, const char *message);
	void file_read(PageTyp *page, const char *message);
	void file_seek(int page_no);
};

#define MAX_FSTREAM_TABLE 1000


extern int fstream_table_used[MAX_FSTREAM_TABLE];
extern std::fstream *fstream_table[MAX_FSTREAM_TABLE];

int fstream_table_get_free_entry();
void database_init(int verbose_level);
void database_exit(void);
int root_buf_alloc(void);
void root_buf_free(int i);



// #############################################################################
// class page_table
// #############################################################################



typedef struct btree_page_registry_key_pair btree_page_registry_key_pair;

//! DISCRETA internal class related to class database

struct btree_page_registry_key_pair {
	int x;
	int idx;
	int ref;
};


//! DISCRETA internal class related to class database

typedef class page_table page_table;

typedef page_table *ppage_table;


//! DISCRETA class for bulk storage



class page_table {
public:
	page_storage *btree_pages;
	int btree_page_registry_length;
	int btree_page_registry_allocated_length;
	btree_page_registry_key_pair *btree_table;


	page_table();
	~page_table();
	void init(int verbose_level);
	void reallocate_table(int verbose_level);
	void print();
	int search(int len, int btree_idx, int btree_x, int &idx);
	int search_key_pair(int len, btree_page_registry_key_pair *K, int &idx);
	void save_page(Buffer *BF, int buf_idx, int verbose_level);
	int load_page(Buffer *BF, int x, int buf_idx, int verbose_level);
	void allocate_rec(Buffer *BF, int buf_idx, int x, int verbose_level);
	void write_pages_to_file(btree *B, int buf_idx, int verbose_level);
};




void page_table_init(int verbose_level);
void page_table_exit(int verbose_level);
int page_table_alloc(int verbose_level);
void page_table_free(int idx, int verbose_level);
page_table *page_table_pointer(int slot);




//! DISCRETA class for the design parameters database




class design_parameter_source: public Vector 
{
	public:
	design_parameter_source();
		// constructor, sets the Vector_pointer to NULL
	design_parameter_source(const discreta_base& x);
		// copy constructor
	design_parameter_source& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_design_parameter_source();
	kind s_virtual_kind();
	~design_parameter_source();
	void freeself_design_parameter_source();
	void copyobject_to(discreta_base &x);
	std::ostream& print(std::ostream&);
	void print2(design_parameter& p, std::ostream& ost);
	
	int & prev() { return Vector::s_i(0).as_integer().s_i(); }
	int & rule() { return Vector::s_i(1).as_integer().s_i(); }
	hollerith & comment() { return Vector::s_i(2).as_hollerith(); }
	Vector & references() { return Vector::s_i(3).as_vector(); }
	hollerith & references_i(int i) { return references().s_i(i).as_hollerith(); }

	void init();
	void text(hollerith& h);
	void text2(design_parameter& p, hollerith& h);
	void text012(hollerith& s0, hollerith& s1, hollerith& s2);
	void text012_extended(design_parameter& p, hollerith& s0, hollerith& s1, hollerith& s2);
};

// design.cpp:
int design_parameters_admissible(int v, int t, int k, discreta_base &lambda);
int calc_delta_lambda(int v, int t, int k, int f_v);
void design_lambda_max(int t, int v, int k, discreta_base & lambda_max);
void design_lambda_max_half(int t, int v, int k, discreta_base & lambda_max_half);
void design_lambda_ijs_matrix(int t, int v, int k, discreta_base& lambda, int s, discreta_matrix & M);
void design_lambda_ijs(int t, int v, int k, discreta_base& lambda, int s, int i, int j, discreta_base & lambda_ijs);
void design_lambda_ij(int t, int v, int k, discreta_base& lambda, int i, int j, discreta_base & lambda_ij);
int is_trivial_clan(int t, int v, int k);
void print_clan_tex_int(int t, int v, int k);
void print_clan_tex_int(int t, int v, int k, int delta_lambda, discreta_base &m_max);
void print_clan_tex(discreta_base &t, discreta_base &v, discreta_base &k, int delta_lambda, discreta_base &m_max);
int is_ancestor(int t, int v, int k);
int is_ancestor(int t, int v, int k, int delta_lambda);
int calc_redinv(int t, int v, int k, int delta_lambda, int &c, int &T, int &V, int &K, int &Delta_lambda);
int calc_derinv(int t, int v, int k, int delta_lambda, int &c, int &T, int &V, int &K, int &Delta_lambda);
int calc_resinv(int t, int v, int k, int delta_lambda, int &c, int &T, int &V, int &K, int &Delta_lambda);
void design_mendelsohn_coefficient_matrix(int t, int m, discreta_matrix & M);
void design_mendelsohn_rhs(int v, int t, int k, discreta_base& lambda, int m, int s, Vector & rhs);
int design_parameter_database_already_there(database &D, design_parameter &p, int& idx);
void design_parameter_database_add_if_new(database &D, design_parameter &p, int& highest_id, int verbose_level);
void design_parameter_database_closure(database &D, int highest_id_already_closed, int minimal_t, int verbose_level);
void design_parameter_database_read_design_txt(char *fname_design_txt, char *path_db, int f_form_closure, int minimal_t, int verbose_level);
void design_parameter_database_export_tex(char *path_db);
int determine_restricted_number_of_designs_t(database &D, btree &B, 
	int btree_idx_tvkl, int t, int first, int len);
int determine_restricted_number_of_designs_t_v(database &D, btree &B, 
	int btree_idx_tvkl, int t, int v, int first, int len);
void prepare_design_parameters_from_id(database &D, int id, hollerith& h);
void prepare_link(hollerith& link, int id);
void design_parameter_database_clans(char *path_db, int f_html, int f_v, int f_vv);
void design_parameter_database_family_report(char *path_db, int t, int v, int k, int lambda, int minimal_t);
void design_parameter_database_clan_report(char *path_db, Vector &ancestor, Vector &clan_lambda, Vector & clan_member, Vector & clan_member_path);
int Maxfit(int i, int j);


//! DISCRETA class for design parameters



class design_parameter: public Vector 
{
	public:
	design_parameter();
		// constructor, sets the vector_pointer to NULL
	design_parameter(const discreta_base& x);
		// copy constructor
	design_parameter& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_design_parameter();
	kind s_virtual_kind();
	~design_parameter();
	void freeself_design_parameter();
	void copyobject_to(discreta_base &x);
	std::ostream& print(std::ostream&);
	
	int & id() { return Vector::s_i(0).as_integer().s_i(); }
	int & t() { return Vector::s_i(1).as_integer().s_i(); }
	int & v() { return Vector::s_i(2).as_integer().s_i(); }
	int & K() { return Vector::s_i(3).as_integer().s_i(); }
	discreta_base & lambda() { return Vector::s_i(4); }
	Vector & source() { return Vector::s_i(5).as_vector(); }
	design_parameter_source & source_i(int i) { return source().s_i(i).as_design_parameter_source(); }

	void init();
	void init(int t, int v, int k, int lambda);
	void init(int t, int v, int k, discreta_base& lambda);
	void text(hollerith& h);
	void text_parameter(hollerith& h);
	void reduced_t(design_parameter& p);
	int increased_t(design_parameter& p);
	void supplementary_reduced_t(design_parameter& p);
	void derived(design_parameter& p);
	int derived_inverse(design_parameter& p);
	void supplementary_derived(design_parameter& p);
	void residual(design_parameter& p);
	void ancestor(design_parameter& p, Vector & path, int f_v, int f_vv);
	void supplementary_residual(design_parameter& p);
	int residual_inverse(design_parameter& p);
	int trung_complementary(design_parameter& p);
	int trung_left_partner(int& t1, int& v1, int& k1, discreta_base& lambda1,
		int& t_new, int& v_new, int& k_new, discreta_base& lambda_new);
	int trung_right_partner(int& t1, int& v1, int& k1, discreta_base& lambda1,
		int& t_new, int& v_new, int& k_new, discreta_base& lambda_new);
	int alltop(design_parameter& p);
	void complementary(design_parameter& p);
	void supplementary(design_parameter& p);
	int is_selfsupplementary();
	void lambda_of_supplementary(discreta_base& lambda_supplementary);
	
	void init_database(database& D, char *path);
};



// discreta_global.cpp:
void free_global_data();
void the_end(int t0);
void the_end_quietly(int t0);


}}


#endif /* ORBITER_SRC_LIB_DISCRETA_DISCRETA_H_ */








