// discreta.h
//
// Anton Betten
//
// started:  18.12.1998
// modified: 23.03.2000
// moved from D2 to ORBI Nov 15, 2007

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdlib.h>
#include <string.h>


namespace orbiter {

#define BITS_OF_int 32
#define SYSTEMUNIX
#undef SYSTEMMAC
#undef SYSTEMWINDOWS
#define SAVE_ASCII_USE_COMPRESS




/*********************************** macros ********************************/

#define NOT_EXISTING_FUNCTION(s)  cout << "The function " << s << " does not exist in this class\n";

/******************* Constants for type determination **********************/

enum kind { 
	BASE = 0,
	INTEGER = 1,
	VECTOR = 2,
	NUMBER_PARTITION = 3, 
	// RATIONAL /* BRUCH */ = 4, 
	PERMUTATION = 6,
	
	
	// POLYNOM = 9, 
	
	MATRIX = 11,

	// MONOM = 21, 
	LONGINTEGER = 22,
	
	//SUBGROUP_LATTICE = 36, 
	//SUBGROUP_ORBIT = 37, 
	MEMORY = 39, 
	
	HOLLERITH = 44,
	
	DATABASE = 50, 
	BTREE = 51, 
	
	PERM_GROUP = 56,  
	PERM_GROUP_STAB_CHAIN = 57,  

	BT_KEY = 61,
	
	DESIGN_PARAMETER = 70,
	 
	GROUP_SELECTION = 78, 
	UNIPOLY = 79, 

	DESIGN_PARAMETER_SOURCE = 83,  
	SOLID = 84, 

	BITMATRIX = 90,
	//PC_PRESENTATION = 91,
	//PC_SUBGROUP = 92,
	//GROUP_WORD = 93, 
	//GROUP_TABLE = 94,
	//ACTION = 95, 
	GEOMETRY = 96
	
};

enum domain_type { 
	GFp = 1, 
	GFq = 2 
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

enum group_selection_type {
	// the linear groups:
	SL,
	GL,
	SSL,
	GGL,
	PSL,
	PGL,
	PSSL,
	PGGL,
	ASL,
	AGL,
	ASSL,
	AGGL,
	Affine_translations,
	PSU_3_Q2,
	Suzuki,
	A5_in_PSL,
	S4_in_PSL,
	On_projective_lines,
	
	// the well known groups:
	Trivial,
	Symmetric, 
	Alternating,
	Dihedral, 
	Cyclic,
	Holomorph_of_cyclic, 
	Subgroup_of_holomorph_of_cyclic_group,
	Sn_wreath_Sm,
	Mathieu,
	From_file,
	Permutation_generator, 
	Higman_Sims_176,

	// unary operators:
	On_2_sets,
	On_2_tuples,
	On_3_sets,
	On_injective_2_tuples,
	Add_fixpoint,
	Stabilize_point,
	Holomorph,
	Even_subgroup,

	// binary operators:
	Comma,
	Direct_sum,
	Direct_product,
	Wreath_product,
	Exponentiation,
	On_mappings,
	
	Solid_Tetrahedron,
	Solid_Cube,
	Solid_Octahedron,
	Solid_Dodecahedron,
	Solid_Icosahedron,
	Solid_Cube4D, 
	Solid_truncate,
	Solid_dual,
	Solid_truncate_dode,
	Solid_truncate_cube,
	Solid_relabel_points,
	Solid_induced_group_on_edges,
	Solid_midpoints_of_edges,
	Solid_add_central_point,
	Solid_add_central_involution,
	Solid_Cubussimus, 
	Solid_Dodesimum, 
	Solid_CubeEE,
	Solid_CubeEE_russian
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
	// self is a pointer to LONGintEGER_REPRESENTATION
	// which contains the sign, the length 
	// and a C array of chars containing 
	// the decimal representation of the signless longinteger value 

class matrix;			// derived from base
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
	class geometry;			// derived from vector
	class group_selection;		// derived from vector
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
class mp_graphics;



// in global.C:

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
void printobjectkind(ostream& ost, kind k);
const char *kind_ascii(kind k);
const char *action_kind_ascii(kind k);
//void int_swap(int& x, int& y);
void uint4_swap(uint_4& x, uint_4& y);

ostream& operator<<(ostream& ost, class discreta_base& p);
// discreta_base operator * (discreta_base& x, discreta_base &y);
// discreta_base operator + (discreta_base& x, discreta_base &y);

int lcm_int(int m, int n);
//void extended_gcd_int(int m, int n, int &u, int &v, int &g);
int invert_mod_integer(int i, int p);
int remainder_mod(int i, int n);
void factor_integer(int n, Vector& primes, Vector& exponents);
void print_factorization(Vector& primes, Vector& exponents, ostream &o);
void print_factorization_hollerith(Vector& primes, Vector& exponents, hollerith &h);
int nb_primes(int n);
//int is_prime(int n);
int factor_if_prime_power(int n, int *p, int *e);
int Euler(int n);
int Moebius(int i);
int NormRemainder(int a, int m);
int log2(int n);
int sqrt_mod(int a, int p);
int sqrt_mod_involved(int a, int p);
//void latex_head(ostream& ost, int f_book, int f_title, char *title, char *author, int f_toc, int f_landscape);
//void latex_foot(ostream& ost);
void html_head(ostream& ost, char *title_long, char *title_short);
void html_foot(ostream& ost);
void sieve(Vector &primes, int factorbase, int f_v);
void sieve_primes(Vector &v, int from, int to, int limit, int f_v);
void print_intvec_mod_10(Vector &v);
void stirling_second(int n, int k, int f_ordered, discreta_base &res, int f_v);
void stirling_first(int n, int k, int f_signless, discreta_base &res, int f_v);
void Catalan(int n, Vector &v, int f_v);
void Catalan_n(int n, Vector &v, discreta_base &res, int f_v);
void Catalan_nk_matrix(int n, matrix &Cnk, int f_v);
void Catalan_nk_star_matrix(int n, matrix &Cnk, int f_v);
void Catalan_nk_star(int n, int k, matrix &Cnk, discreta_base &res, int f_v);

int atoi(char *p);
void N_choose_K(discreta_base & n, int k, discreta_base & res);
void Binomial(int n, int k, discreta_base & n_choose_k);
void Krawtchouk(int n, int q, int i, int j, discreta_base & a);
// $\sum_{u=0}^{\min(i,j)} (-1)^u \cdot (q-1)^{i-u} \cdot {j \choose u} \cdot $
// ${n - j \choose i - u}$
//int ij2k(int i, int j, int n);
//void k2ij(int k, int & i, int & j, int n);
void tuple2_rank(int rank, int &i, int &j, int n, int f_injective);
int tuple2_unrank(int i, int j, int n, int f_injective);
void output_texable_string(ostream & ost, char *in);
void texable_string(char *in, char *out);
void the_first_n_primes(Vector &P, int n);
void midpoint_of_2(int *Px, int *Py, int i1, int i2, int idx);
void midpoint_of_5(int *Px, int *Py, int i1, int i2, int i3, int i4, int i5, int idx);
void ratio_int(int *Px, int *Py, int idx_from, int idx_to, int idx_result, double r);

void time_check_delta(int dt);
void time_check(int t0);
int nb_of_bits();
void bit_set(uint & g, int k);
void bit_clear(uint & g, int k);
int bit_test(uint & g, int k);
void bitset2vector(uint g, Vector &v);
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
void print_classification_tex(Vector &content, Vector &multiplicities, ostream& ost);
void perm2permutation(int *a, int n, permutation &p);
//void print_integer_matrix(ostream &ost, int *p, int m, int n);
//void print_longinteger_matrix(ostream &ost, LONGint *p, int m, int n); removed Anton Betten Nov 1, 2011
int Gauss_int(int *A, int f_special, int f_complete, int *base_cols,
	int f_P, int *P, int m, int n, int Pn, 
	int q, int *add_table, int *mult_table, int *negate_table, int *inv_table, int f_v);
// returns the rank which is the number of entries in base_cols
void uchar_move(uchar *p, uchar *q, int len);
void int_vector_realloc(int *&p, int old_length, int new_length);
void int_vector_shorten(int *&p, int new_length);
void int_matrix_realloc(int *&p, int old_m, int new_m, int old_n, int new_n);
int code_is_irreducible(int k, int nmk, int idx_zero, int *M);
void fine_tune(finite_field *F, int *mtxD, int verbose_level);

// in mindist.C:
int mindist(int n, int k, int q, int *G, 
	int f_v, int f_vv, int idx_zero, int idx_one, 
	int *add_table, int *mult_table);

// domain.C:

int has_domain();
domain *get_current_domain();
//domain *get_domain_if_pc_group();
int is_GFp_domain(domain *& d);
int is_GFq_domain(domain *& d);
int is_finite_field_domain(domain *& d);
int finite_field_domain_order_int(domain * d);
int finite_field_domain_characteristic(domain * d);
int finite_field_domain_primitive_root();
void finite_field_domain_base_over_subfield(Vector & b);
void push_domain(domain *d);
void pop_domain(domain *& d);
domain *allocate_finite_field_domain(int q, int f_v);
void free_finite_field_domain(domain *dom, int f_v);

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
	matrix& as_matrix() { return *(matrix *)this; }
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
	geometry& as_geometry() { return *(geometry *)this; }
	hollerith& as_hollerith() { return *(hollerith *)this; }
	group_selection& as_group_selection() { return *(group_selection *)this; }
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
	matrix& change_to_matrix() { freeself(); c_kind(MATRIX); return as_matrix(); }
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
	geometry& change_to_geometry() { freeself(); c_kind(GEOMETRY); return as_geometry(); }
	hollerith& change_to_hollerith() { freeself(); c_kind(HOLLERITH); return as_hollerith(); }
	group_selection& change_to_group_selection() { freeself(); c_kind(GROUP_SELECTION); return as_group_selection(); }
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

	virtual ostream& print(ostream&);
		// all kinds of printing, the current printing mode is determined 
		// by the global variable printing_mode
	void print_to_hollerith(hollerith& h);
	ostream& println(ostream&);
		// print() and newline
	ostream& printobjectkind(ostream&);
		// prints the type of the object
	ostream& printobjectkindln(ostream&);

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

	virtual int compare_with_euklidean(discreta_base &a);
		// -1 iff this < a
		// 0 iff this = a
		// 1 iff this > a
	virtual void integral_division(discreta_base &x, discreta_base &q, discreta_base &r, int verbose_level);
	void integral_division_exact(discreta_base &x, discreta_base &q);
	void integral_division_by_integer(int x, discreta_base &q, discreta_base &r);
	void integral_division_by_integer_exact(int x, discreta_base &q);
	void integral_division_by_integer_exact_apply(int x);
	int is_divisor(discreta_base &y);
	void modulo(discreta_base &p);
	void extended_gcd(discreta_base &n, discreta_base &u, discreta_base &v, discreta_base &g, int verbose_level);
	void write_memory(memory &m, int debug_depth);
	void read_memory(memory &m, int debug_depth);
	int calc_size_on_file();
	void pack(memory & M, int f_v, int debug_depth);
	void unpack(memory & M, int f_v, int debug_depth);
	void save_ascii(ostream & f);
	void load_ascii(istream & f);
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
	ostream& print(ostream& ost);
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
	void read_file(char *fname, int f_v);
	void write_file(char *fname, int f_v);
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

	ostream& print(ostream&);
	int compare_with(discreta_base &a);

	char * s_unchecked() { return self.char_pointer; } 
	char * s() { if (self.char_pointer) return self.char_pointer; else return (char *) ""; }
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

	ostream& print(ostream&);

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

	int compare_with_euklidean(discreta_base &a);
	void integral_division(discreta_base &x, discreta_base &q, discreta_base &r, int verbose_level);
	
	void rand(int low, int high);
	int log2();
};

#define LONGintEGER_PRint_DOTS
#define LONGintEGER_DIGITS_FOR_DOT 6



//! DISCRETA  class for integers of arbitrary magnitude




class longinteger: public discreta_base
{
	public:
	longinteger();
	longinteger(int a);
	//longinteger(LONGint a); removed Anton Betten Nov 1, 2011
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

	ostream& print(ostream&);
	
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
	// void homo_z(LONGint z); removed Anton Betten Nov 1, 2011
	void inc();
	void dec();
	int is_zero();
	int is_one();
	int is_m_one();
	int is_even();
	int is_odd();

	int compare_with_euklidean(discreta_base &b);
	void integral_division(discreta_base &x, discreta_base &q, discreta_base &r, int verbose_level);
	void square_root_floor(discreta_base &x);
	longinteger& Mersenne(int n);
	longinteger& Fermat(int n);
	int s_i();
	int retract_to_integer_if_possible(integer &x);
	int modp(int p);
	int ny_p(int p);
	void divide_out_int(int d);

	int Lucas_test_Mersenne(int m, int f_v);
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
	
	ostream& Print(ostream&);
	ostream& print(ostream&);
	ostream& print_unformatted(ostream& ost);
	ostream& print_intvec(ostream& ost);
	
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
	void print_word_nicely(ostream &ost, int f_generator_labels, Vector &generator_labels);
	void print_word_nicely2(ostream &ost);
	void print_word_nicely_with_generator_labels(ostream &ost, Vector &generator_labels);
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
	ostream& print(ostream&);
	ostream& print_list(ostream& ost);
	ostream& print_cycle(ostream& ost);
	void sscan(const char *s, int f_v);
	void scan(istream & is, int f_v);

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
	void cycle_type(Vector& type, int f_v);
	int nb_of_inversions(int f_v);
	int signum(int f_v);
	int is_even(int f_v);
	void cycles(Vector &cycles);
	void restrict_to_subset(permutation &q, int first, int len);
	void induce_on_lines_of_PG_k_q(int k, int q, permutation &per, int f_v, int f_vv);
	void singer_cycle_on_points_of_projective_plane(int p, int f_modified, int f_v);
	void Cn_in_Cnm(int n, int m);
	int preimage(int i);
};

void signum_map(discreta_base & x, discreta_base &d);
//char get_character(istream & is, int f_v);

//! DISCRETA utility class for matrix access




class matrix_access {
public:
	int i;
	matrix *p;
	discreta_base & operator [] (int j);
};

//! DISCRETA matrix class



class matrix: public discreta_base
{
	public:
	matrix();
		// constructor, sets the matrix_pointer to NULL
	matrix(const discreta_base& x);
		// copy constructor
	matrix& operator = (const discreta_base &x);
		// copy assignment

	void *operator new(size_t, void *p) { return p; } 
	void settype_matrix();

	~matrix();
	void freeself_matrix();
		// delete the matrix
	kind s_virtual_kind();
	void copyobject_to(discreta_base &x);

	ostream& print(ostream&);
	int compare_with(discreta_base &a);

	matrix& m_mn(int m, int n);
		// make matrix of format m times n
		// allocates the memory and sets the objects to type BASE
	matrix& m_mn_n(int m, int n);
	matrix& realloc(int m, int n);


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
	void matrix_mult_to(matrix &x, discreta_base &y);
	void vector_mult_to(Vector &x, discreta_base &y);
	void multiply_vector_from_left(Vector &x, Vector &y);
	int invert_to(discreta_base &x);
	void add_to(discreta_base &x, discreta_base &y);
	void negate_to(discreta_base &x);
	void one();
	void zero();
	int is_zero();
	int is_one();


	int Gauss(int f_special, int f_complete, Vector& base_cols, int f_P, matrix& P, int f_v);
	int rank();
	int get_kernel(Vector& base_cols, matrix& kernel);
	matrix& transpose();
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
	void smith_normal_form(matrix& P, matrix& Pv, matrix& Q, matrix& Qv, int verbose_level);
	int smith_eliminate_column(matrix& P, matrix& Pv, int i, int verbose_level);
	int smith_eliminate_row(matrix& Q, matrix& Qv, int i, int verbose_level);
	void multiply_2by2_from_left(int i, int j, 
		discreta_base& aii, discreta_base& aij, discreta_base& aji, discreta_base& ajj, int verbose_level);
	void multiply_2by2_from_right(int i, int j, 
		discreta_base& aii, discreta_base& aij, discreta_base& aji, discreta_base& ajj, int verbose_level);

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
#if 0
	void lexleast_incidence_matrix(int f_on_rows, 
		int f_row_decomp, Vector & row_decomp, 
		int f_col_decomp, Vector & col_decomp, 
		int f_ddp, Vector & DDp, 
		int f_ddb, Vector & DDb, 
		int f_group, perm_group & G, 
		permutation & p, permutation & q, 
		int f_print_backtrack_points, 
		int f_get_aut_group, int f_aut_group_on_lexleast, Vector & aut_gens, 
		int f_v, int f_vv);
#endif
	void apply_perms(int f_row_perm, permutation &row_perm, 
		int f_col_perm, permutation &col_perm);
	void apply_col_row_perm(permutation &p);
	void apply_row_col_perm(permutation &p);
	void incma_print_ascii_permuted_and_decomposed(ostream &ost, int f_tex, 
		Vector & decomp, permutation & p);
	void print_decomposed(ostream &ost, Vector &row_decomp, Vector &col_decomp);
	void incma_print_ascii(ostream &ost, int f_tex, 
		int f_row_decomp, Vector &row_decomp, 
		int f_col_decomp, Vector &col_decomp);
	void incma_print_latex(ostream &f, 
		int f_row_decomp, Vector &row_decomp, 
		int f_col_decomp, Vector &col_decomp, 
		int f_labelling_points, Vector &point_labels, 
		int f_labelling_blocks, Vector &block_labels);
	void incma_print_latex2(ostream &f, 
		int width, int width_10, 
		int f_outline_thin, const char *unit_length, 
		const char *thick_lines, const char *thin_lines, const char *geo_line_width, 
		int f_row_decomp, Vector &row_decomp, 
		int f_col_decomp, Vector &col_decomp, 
		int f_labelling_points, Vector &point_labels, 
		int f_labelling_blocks, Vector &block_labels);
	void calc_hash_key(int key_len, hollerith & hash_key, int f_v);
	int is_in_center();
	void power_mod(int r, integer &P, matrix &C);
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
#if 0
	void canon(int f_row_decomp, Vector & row_decomp, 
		int f_col_decomp, Vector & col_decomp, 
		int f_group, perm_group & G, 
		permutation & p, permutation & q, 
		int f_get_aut_group, int f_aut_group_on_lexleast, Vector & aut_gens, discreta_base &ago,
		int f_v, int f_vv, int f_vvv, int f_vvvv, int f_tree_file);
	void canon_partition_backtrack(int f_row_decomp, Vector & row_decomp, 
		int f_col_decomp, Vector & col_decomp, 
		int f_group, perm_group & G, 
		permutation & p, permutation & q, 
		int f_get_aut_group, int f_aut_group_on_lexleast, Vector & aut_gens, discreta_base &ago,
		int f_v, int f_vv, int f_vvv, int f_vvvv, int f_tree_file);
#endif
	void canon_nauty(int f_row_decomp, Vector & row_decomp, 
		int f_col_decomp, Vector & col_decomp, 
		int f_group, perm_group & G, 
		permutation & p, permutation & q, 
		int f_get_aut_group, int f_aut_group_on_lexleast, Vector & aut_gens, 
		int f_v, int f_vv, int f_vvv);
#if 0
	void canon_tonchev(int f_row_decomp, Vector & row_decomp, 
		int f_col_decomp, Vector & col_decomp, 
		int f_group, perm_group & G, 
		permutation & p, permutation & q, 
		int f_get_aut_group, int f_aut_group_on_lexleast, Vector & aut_gens, 
		int f_v, int f_vv, int f_vvv);
#endif
	void save_as_geometry(int number, char *label);
	void save_as_inc_file(char *fname);
	void save_as_inc(ofstream &f);
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
	ostream& print(ostream&);
	ostream& print_as_vector(ostream& ost);

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
	int compare_with_euklidean(discreta_base &a);
	void integral_division(discreta_base &x, discreta_base &q, discreta_base &r, int verbose_level);
	void derive();
	int is_squarefree(int verbose_level);
	int is_irreducible_GFp(int p, int verbose_level);
	int is_irreducible(int q, int f_v);
	int is_primitive(int m, int p, Vector& vp, int verbose_level);
	void numeric_polynomial(int n, int q);
	int polynomial_numeric(int q);
	void singer_candidate(int p, int f, int b, int a);
	void Singer(int p, int f, int f_v, int f_vv);
	void get_an_irreducible_polynomial(int f, int verbose_level);
	void evaluate_at(discreta_base& x, discreta_base& y);
	void largest_divisor_prime_to(unipoly& q, unipoly& r);
	void monic();
	void normal_base(int p, matrix& F, matrix& N, int verbose_level);
	int first_irreducible_polynomial(int p, unipoly& m, matrix& F, matrix& N, Vector &v, int verbose_level);
	int next_irreducible_polynomial(int p, unipoly& m, matrix& F, matrix& N, Vector &v, int verbose_level);
	void normalize(discreta_base &p);
	void Xnm1(int n);
	void Phi(int n, int f_v);
	void weight_enumerator_MDS_code(int n, int k, int q, int f_v, int f_vv, int f_vvv);
	void charpoly(int q, int size, int *mtx, int verbose_level);
	
};

// vbp.C:
void place_lattice(Vector& nl, Vector& orbit_size, 
	int size_x, int size_y, 
	Vector& Px, Vector& Py, Vector& O_dx, 
	int f_upside_down, int f_v);





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
	ostream& print(ostream&);

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

int first_passport(Vector &pass, int n, int k);
int next_passport(Vector &pass, int n, int k);
int first_passport_i(Vector &pass, int n, int k, int i, int & S);
int next_passport_i(Vector &pass, int n, int k, int i, int & S);

//! DISCRETA class for incidence matrices



class geometry: public Vector
{
	public:
	geometry();
		// constructor, sets the vector_pointer to NULL
	void allocate_geometry();
	geometry(const discreta_base& x);
		// copy constructor
	geometry& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_geometry();
	kind s_virtual_kind();
	~geometry();
	void freeself_geometry();
	void copyobject_to(discreta_base &x);
	ostream& print(ostream&);
	void print_latex(ostream& ost);
	void print_head_latex(ostream& ost);
	void print_incma_text_latex(ostream& ost);
	void print_labellings_latex(ostream& ost);
	void print_incma_latex_picture(ostream& ost);
	
	void print_inc(ostream &ost);
	void print_inc_only(ostream &ost);
	void print_inc_header(ostream &ost);
	void print_ascii(ostream& ost);
	int scan(istream&);
	void scan_body(istream& f, int geo_nr, char *geo_label);


	int & number() { return Vector::s_i(0).as_integer().s_i(); }
	hollerith & label() { return Vector::s_i(1).as_hollerith(); }
	matrix & X() { return Vector::s_i(2).as_matrix(); }
	int & f_incidence_matrix() { return Vector::s_i(3).as_integer().s_i(); }
	
	Vector & point_labels() { return Vector::s_i(4).as_vector(); }
	Vector & block_labels() { return Vector::s_i(5).as_vector(); }

	int & f_row_decomp() { return Vector::s_i(6).as_integer().s_i(); }
	Vector & row_decomp() { return Vector::s_i(7).as_vector(); }
	int & f_col_decomp() { return Vector::s_i(8).as_integer().s_i(); }
	Vector & col_decomp() { return Vector::s_i(9).as_vector(); }
	
	int & f_ddp() { return Vector::s_i(10).as_integer().s_i(); }
	Vector & ddp() { return Vector::s_i(11).as_vector(); }
	int & f_ddb() { return Vector::s_i(12).as_integer().s_i(); }
	Vector & ddb() { return Vector::s_i(13).as_vector(); }

	int & f_canonical_labelling_points() { return Vector::s_i(14).as_integer().s_i(); }
	permutation & canonical_labelling_points() { return Vector::s_i(15).as_permutation(); }
	int & f_canonical_labelling_blocks() { return Vector::s_i(16).as_integer().s_i(); }
	permutation & canonical_labelling_blocks() { return Vector::s_i(17).as_permutation(); }

	int & f_aut_gens() { return Vector::s_i(18).as_integer().s_i(); }
	Vector & aut_gens() { return Vector::s_i(19).as_vector(); }
	discreta_base & ago() { return Vector::s_i(20); }

	void transpose();
	int is_2design(int &r, int &lambda, int f_v);
#if 0
	void calc_lexleast_and_autgroup(int f_v, int f_vv, int f_print_backtrack_point);
	void calc_canon_and_autgroup(int f_v, int f_vv, int f_vvv, int f_vvvv, 
		int f_print_backtrack_points, int f_tree_file);
	void calc_canon_and_autgroup_partition_backtrack(int f_v, int f_vv, int f_vvv, int f_vvvv, 
		int f_print_backtrack_points, int f_tree_file);
#endif
	void calc_canon_nauty(int f_v, int f_vv, int f_vvv);
	//void calc_canon_tonchev(int f_v, int f_vv, int f_vvv);
	void get_lexleast_X(matrix & X0);
};

int search_geo_file(matrix & X0, char *fname, int geo_nr, char *geo_label, int f_v);

// geo_canon.C:

void perm_test(void);
void geo_canon_with_initial_decomposition_and_ddp_ddb(
	int f_maxtest, int *back_to, 
	int f_transposed, 
	int nrow, int ncol, int nb_X, int *theX, 
	int f_row_decomp, Vector & row_decomp, 
	int f_col_decomp, Vector & col_decomp, 
	int f_ddp, Vector & DDp, 
	int f_ddb, Vector & DDb, 
	int f_col_group, perm_group & col_group, 
	permutation & p, permutation & q, 
	int f_print_backtrack_points, 
	int f_get_aut_group, int f_aut_group_on_lexleast, Vector & aut_gens, 
	int f_v, int f_vv);


//! DISCRETA class for influencing arithmetic operations




class domain {
	private:
		domain_type the_type;
		discreta_base the_prime;
		//pc_presentation *the_pres;
		unipoly *the_factor_poly;
		domain *the_sub_domain;
	
	public:
	domain(int p);
	domain(unipoly *factor_poly, domain *sub_domain);
	//domain(pc_presentation *pres);
	
	domain_type type();
	int order_int();
	int order_subfield_int();
	int characteristic();
	//pc_presentation *pres();
	unipoly *factor_poly();
	domain *sub_domain();
};


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

//! DISCRETA class to choose a group from the command line or from a UI



class group_selection: public Vector
{
	public:
	group_selection();
		// constructor, sets the vector_pointer to NULL
	group_selection(const discreta_base& x);
		// copy constructor
	group_selection& operator = (const discreta_base &x);
		// copy assignment
	void *operator new(size_t, void *p) { return p; } 
	void settype_group_selection();
	kind s_virtual_kind();
	~group_selection();
	void freeself_group_selection();
	void copyobject_to(discreta_base &x);
	ostream& print(ostream&);

	int & type() { return Vector::s_i(0).as_integer().s_i(); }
	int & val1() { return Vector::s_i(1).as_integer().s_i(); }
	int & val2() { return Vector::s_i(2).as_integer().s_i(); }
	hollerith & s() { return Vector::s_i(3).as_hollerith(); }

	void init(group_selection_type type, int v1, int v2, char *str);
};

const char *group_selection_type_as_text(group_selection_type t);
void compose_gsel_from_strings(Vector &gsel, int num_args, char **args);
void compose_group(Vector & gsel, Vector & gens, 
	hollerith & group_label, hollerith & group_label_tex, hollerith & acting_on, int f_v);


// perm_group_gens.C:

int vec_generators_is_trivial_group(Vector & gen);
int is_abelian(Vector & gen);
void read_file_of_generators_xml(Vector & gen, char *fname, int &f_cyclic_notation, int f_v);
void write_file_of_generators_xml_group_label(Vector & gen, char *group_label, int f_cyclic_notation);
void write_file_of_generators_xml(Vector & gen, char *fname, int f_cyclic_notation);
void read_file_of_generators(Vector & G, char *fname);
void read_generators(Vector & G, ifstream & f);
void write_file_of_generators_group_label(Vector & gen, char *group_label);
void write_file_of_generators(Vector & G, char *fname);
void write_generators(Vector & G, ofstream & f);
void write_file_of_generators_gap_group_label(Vector & gen, char *group_label);
void write_file_of_generators_gap(Vector & G, char *fname);
void write_generators_gap(Vector & G, ofstream & f);
void vec_induced_group_on_subset(Vector & V, Vector & subset, Vector & W);
void vec_subgroup_of_hol_of_cyclic_group(Vector & V, int n, int i);
void vec_hol_of_cyclic_group(Vector & V, int n);
void vec_conjugate(Vector & gen, permutation & p);
void vec_induce_action_on_blocks(Vector & gen, Vector & B);
void vec_induce3(Vector & gen);
void vec_induce2(Vector & gen);
void vec_induce_on_2tuples(Vector & gen, int f_injective);
void vec_add_fixpoint_in_front(Vector & gen);
void vec_add_fixpoint_at_end(Vector & gen);
int vec_generators_degree(Vector & a);
//void vec_generators_stabilize_point(Vector & a, Vector & b);
//void vec_generators_group_order(Vector & gen, discreta_base & o);
void vec_generators_remove_fixpoint(Vector & gen, int i);
void vec_generators_raise_to_nth_power(Vector & gen, int n);
void vec_generators_induce_on_lines_of_PG_k_q(Vector & gen, int k, int q, int f_v, int f_vv);
void vec_generators_trivial_group(Vector & gen, int deg);
void vec_generators_cyclic_group(Vector & gen, int deg);
void vec_generators_Cn_in_Cnm(Vector & gen, int n, int m);
void vec_generators_AutCq_in_Cqm(Vector & gen, int q, int m);
void vec_generators_symmetric_group(Vector & gen, int deg);
void vec_generators_alternating_group(Vector & gen, int deg);
void vec_generators_alternating_group_huppert(Vector & gen, int deg);
void vec_generators_dihedral_group(Vector & gen, int deg);
void vec_generators_Mathieu_n(Vector & gen, int n);
void vec_generators_Mathieu_11(Vector & gen);
void vec_generators_Mathieu_12(Vector & gen);
void vec_generators_Mathieu_23(Vector & gen);
void vec_generators_Mathieu_24(Vector & gen);
void vec_generators_diagonal_sum(Vector & a, Vector & b, Vector & c);
void vec_generators_comma(Vector & a, Vector & b, Vector & c);
void vec_generators_direct_sum(Vector & a, Vector & b, Vector & c);
void vec_generators_direct_product(Vector & a, Vector & b, Vector & c);
void vec_generators_GL_n_q_as_matrices(Vector & gen, int n, domain *dom, int f_v);
void vec_generators_GL_n_q_subgroup_as_matrices(Vector & gen, int n, int subgroup_index, domain *dom, int f_v);
void vec_generators_SL_n_q_as_matrices(Vector & gen, int n, domain *dom, int f_v);
void vec_generators_frobenius_in_PG(Vector & gen, int n, domain *dom, int f_v);
void vec_generators_frobenius_in_AG(Vector & gen, int n, domain *dom, int f_v);
void vec_generators_affine_translations(Vector & gen, int n, domain *dom, int f_v);
void vec_generators_affine_translations(Vector & gen, int n, int q, int f_v);
void vec_generators_projective_representation(domain *dom, Vector & a, Vector & b, int f_action_from_right, int f_modified, int f_v);
void vec_generators_affine_representation(domain *dom, Vector & a, Vector & b, int f_v);
void vec_generators_GL_n_q_projective_representation(Vector & gen, int n, int q, int f_special, int f_frobenius, int f_modified, int f_v);
void vec_generators_GL_n_q_affine_representation(Vector & gen, int n, int q, int f_special, int f_frobenius, int f_translations, int f_v);
void vec_generators_GL_n_q_subgroup_affine_representation(Vector & gen, int n, int q, int subgroup_index, 
	int f_special, int f_frobenius, int f_translations, int f_v);
void kernel_of_homomorphism(Vector & gens, Vector & kernel_gens, 
	void (*hom)(discreta_base & x, discreta_base & image), int f_v, int f_vv);
void vec_generators_A5_in_PSL(Vector& G, int q, int f_v);
void vec_generators_S4_in_PSL(Vector& G, int q, int f_v);
void vec_generators_even_subgroup(Vector & gen, Vector & gen_even_subgroup, int f_v);
//void vec_generators_on_conjugacy_class_of_subgroups_by_conjugation(perm_group &G, 
	//Vector &LayerOrbit, int layer, int orbit, Vector &gens, Vector &induced_gens, int f_v, int f_vv);
void vec_generators_restrict_to_subset(Vector & gen, int first, int len);
void wreath_embedding(permutation & g, int n, int m, permutation & q);
void wreath_embedding_component(permutation & g, int n, int m, int j, permutation & q);
void vec_generators_wreath_product(Vector & G, Vector & H, Vector & W, int f_v);
void vec_generators_Sn_wreath_Sm(int n, int m, Vector & G);
void vec_generators_q1_q2(int q1, int q2, Vector & gen, hollerith &label, 
	int f_write_generators_to_file, int f_v, int f_vv);
void vec_generators_q1_q2_aubv(int q1, int q2, int u, int v, Vector & G, hollerith &label, 
	int f_write_generators_to_file, int f_v, int f_vv);
void vec_generators_q1_q2_au1bv1_au2bv2(int q1, int q2, int u1, int v1, int u2, int v2, 
	Vector & G, hollerith &label, int f_write_generators_to_file, int f_v, int f_vv);
void vec_generators_AGGL1q_subgroup(int q, int subgroup_index, 
	int f_special, int f_frobenius, int f_translations, int f_v);
//void vec_generators_cycle_index(Vector &gen, Vector &C, int f_v);
void vec_generators_singer_cycle_on_points_of_projective_plane(Vector &gen, int p, int f_modified, int f_v);


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
	ostream& print_list(ostream& ost);
	ostream& print(ostream& ost);
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
	void direct_sum(solid& B, solid& J, int f_v);
	void direct_product(Vector& gen, solid& J, int f_v);
	void scale(double f);
	void add_central_point(solid& A);
	void induced_action_on_edges(permutation& p, permutation& q);
	void induced_group_on_edges(Vector & gen, Vector & gen_e);
	
	void tetrahedron(int r);
	void cube(int r);
	void cube4D(int r1, int r2);
	void octahedron(int r);
	void dodecahedron(int r);
	void icosahedron(int r);
	void make_placed_graph(matrix & incma, Vector& aut_gens, Vector& cycles);
		
	void write_graphfile(char *fname);
	void write_solidfile(char *fname);
};
void vec_generators_aut_cube_nd(int n, Vector &gen);
void number_to_binary(int n, int *v, int digits);
int binary_to_number(int *v, int digits);



// kramer_mesner.C

extern char *discreta_copyright_text;

typedef struct design_data DESIGN_DATA;


//! DISCRETA class for Kramer Mesner type problems




struct design_data {
	char *KM_fname;
	int v, t, k;
	Vector gen;
	Vector MM;
	Vector RR;
	Vector stab_go;
	discreta_base go;
	
	
	int lambda;
	int nb_sol;
	Vector S;

	matrix P;
};


void write_KM_file(char *gsel, char *g_label, char *g_label_tex, char *km_fname, char *acting_on, 
	Vector & G_gen, discreta_base & go, int deg,
	matrix & M, int t, int k);
void write_KM_file2(char *gsel, char *g_label, char *g_label_tex, char *km_fname, char *acting_on, 
	Vector & G_gen, discreta_base & go, int deg,
	matrix & M, int t, int k, int f_right_hand_side_in_last_column_of_M);
void write_ascii_generators(char *km_fname, Vector & gen);
void write_ascii_representatives(char *km_fname, Vector & R);
void write_ascii_stabilizer_orders(char *km_fname, Vector & Ago);
void write_ascii_stabilizer_orders_k_sets(char *km_fname, Vector & Ago, int k);
void write_ascii_KM_matrices(char *km_fname, Vector & MM);
void km_read_ascii_vtk(char *KM_fname, int &v, int &t, int &k);
void km_read_ascii_strings(char *KM_fname, hollerith& group_construction, hollerith& group_label, hollerith& group_label_tex, hollerith& acting_on);
void km_read_generators(char *KM_fname, Vector & gen);
void km_read_KM_matrices(char *KM_fname, Vector & MM);
void km_read_orbit_representatives(char *KM_fname, Vector & RR);
void km_read_stabilizer_orders(char *KM_fname, Vector & stab_go);
void km_read_orbits_below(char *KM_fname, Vector & Orbits_below1, Vector & Orbits_below2);
void km_read_lambda_values(char *KM_fname, Vector & lambda_values, Vector & lambda_solution_count);
void km_get_solutions_from_solver(char *KM_fname, int lambda);
int km_nb_of_solutions(char *KM_fname, int lambda);
void km_get_solutions(char *KM_fname, int lambda, int from, int len, Vector& S);
void km_read_until_lambdaend(ifstream & f);
void Mtk_via_Mtr_Mrk(int t, int r, int k, matrix & Mtr, matrix & Mrk, matrix & Mtk, int f_v);
void Mtk_from_MM(Vector & MM, matrix & Mtk, int t, int k, int f_v);

DESIGN_DATA *prepare_for_intersection_numbers(char *KM_fname);
void design_load_all_solutions(DESIGN_DATA *dd, int lambda);
void design_prepare_orbit_lengths(DESIGN_DATA *dd);
void design_orbits_vector(Vector & X, Vector & orbits, int f_complement);

void global_intersection_numbers_prepare_data(char *KM_fname, int lambda, int s_max, 
	DESIGN_DATA *&dd, matrix& L, matrix& Z, matrix &Bv, matrix & S1t, matrix &D);
void global_intersection_numbers_compute(char *KM_fname, int lambda, int s_max, 
	DESIGN_DATA *&dd, matrix& L, matrix& Z, matrix &Bv, matrix & S1t, matrix &D, 
	Vector& sol, matrix& As1, matrix& As2, Vector& inv);
void global_intersection_numbers(char *KM_fname, int lambda, int s_max);
void extend_design_from_residual(char *KM_fname, int lambda);
void get_group_to_file(int arg_length, char **group_arg_list, int f_v);
void show_design(char *solid_fname, char *KM_fname, int lambda, int m);
void report(char *KM_fname, int s_max);
void canonical_set_reps(Vector& Set_reps, perm_group &G, int f_v, int f_vv, int f_vvv);
void normalizer_action_on_orbits(perm_group & G, Vector & Reps, 
	Vector &N_gens, Vector &N_gens_induced, int f_v);
void action_on_orbits_of_normalizing_element(perm_group & G, Vector & Reps,
	permutation & p, permutation & q, int f_v);
void fuse_orbits(perm_group & G, Vector & Reps, 
	Vector & fusion_map, Vector & new_reps, int f_v, int f_vv, int f_vvv);
void km_compute_KM_matrix(int arg_length, char **group_arg_list, int t, int k, int f_v, int f_vv, int f_vvv);
// interface for ladder:
int permutation_element_image_of(int a, void *elt, void *data, int f_v);
void permutation_element_retrieve(int hdl, void *elt, void *data, int f_v);
int permutation_element_store(void *elt, void *data, int f_v);
void permutation_element_mult(void *a, void *b, void *ab, void *data, int f_v);
void permutation_element_invert(void *a, void *av, void *data, int f_v);
void permutation_element_move(void *a, void *b, void *data, int f_v);
void permutation_element_dispose(int hdl, void *data, int f_v);
void permutation_element_print(void *elt, void *data, ostream &ost);



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
	ostream& print(ostream&);

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
void bt_key_print_int4(char **key, ostream& ost);
void bt_key_print_int2(char **key, ostream& ost);
void bt_key_print(char *key, Vector& V, ostream& ost);
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
	ostream& print(ostream&);

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
	void print_by_btree(int btree_idx, ostream& ost);
	void print_by_btree_with_datref(int btree_idx, ostream& ost);
	void print_subset(Vector& datrefs, ostream& ost);
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
	ostream& print(ostream&);
	
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
	void key_print(char *the_key, ostream& ost);

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
	
	void print_all(ostream& ost);
	void print_range(int first, int len, ostream& ost);
	void print_page(int x, ostream& ost);
	void page_print(Buffer *BF, ostream& ost);
	void item_print(ItemTyp *item, int i, ostream& ost);
	
	void file_open();
	void file_create();
	void file_close();
	void file_write(PageTyp *page, const char *message);
	void file_read(PageTyp *page, const char *message);
	void file_seek(int page_no);
};

#define MAX_FSTREAM_TABLE 1000


extern int fstream_table_used[MAX_FSTREAM_TABLE];
extern fstream *fstream_table[MAX_FSTREAM_TABLE];

int fstream_table_get_free_entry();
void database_init(int verbose_level);
void database_exit(void);
int root_buf_alloc(void);
void root_buf_free(int i);



// ##########################################################################################################
// class page_table
// ##########################################################################################################



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
	ostream& print(ostream&);
	void print2(design_parameter& p, ostream& ost);
	
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

// design.C:
int design_parameters_admissible(int v, int t, int k, discreta_base &lambda);
int calc_delta_lambda(int v, int t, int k, int f_v);
void design_lambda_max(int t, int v, int k, discreta_base & lambda_max);
void design_lambda_max_half(int t, int v, int k, discreta_base & lambda_max_half);
void design_lambda_ijs_matrix(int t, int v, int k, discreta_base& lambda, int s, matrix & M);
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
void design_mendelsohn_coefficient_matrix(int t, int m, matrix & M);
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
#if 0
void create_all_masks(char *label, 
	int nb_row_partitions, char *row_partitions[], 
	int nb_col_partitions, char *col_partitions[]);
int create_masks(char *label, 
	int nb_row_partitions, char *row_partitions[], 
	int nb_col_partitions, char *col_partitions[], 
	int ci, int cj);
#endif
#if 0
void orbits_in_product_action(int n1, int n2, int f_v, int f_vv);
void orbits_in_product_action_D_CC(int n1, int p1, int p2, int f_v, int f_vv);
void orbits_in_product_action_CC_D(int p1, int p2, int n2, int f_v, int f_vv);
void orbits_in_product_action_extended(int q1, int q2, int u, int v, int f_v, int f_vv);
void orbits_in_product_action_extended_twice(int q1, int q2, int u1, int v1, int u2, int v2, 
	int f_cycle_index, int f_cycle_index_on_pairs, int f_v, int f_vv);
void extract_subgroup(int q1, int q2, int u1, int v1, int f_cycle_index);
#endif


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
	ostream& print(ostream&);
	
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


// counting.C:
void cycle_index_perm_group(perm_group &G, Vector &C, int f_v, int f_vv);
void cycle_type_add_monomial(Vector &C, Vector &m, discreta_base &coeff);
void cycle_index_Zn(Vector &C, int n);
void cycle_index_elementary_abelian(Vector &C, int p, int f);
void make_monomial_of_equal_cycles(int i, int e, Vector &m);
void cycle_index_q1_q2_dot_aubv_product_action(int q1, int q2, int u, int v, Vector &C);
void cycle_index_direct_product(Vector &CG, Vector &CH, Vector &CGH, int f_v);
void cycle_index_monomial_direct_product(Vector &m1, Vector &m2, Vector &m3);
void cycle_index_on_pairs(Vector &CG, Vector &CG2, int f_v);
void cycle_index_number_of_orbits(Vector &CG, discreta_base &number_of_orbits);
void cycle_index_number_of_orbits_on_mappings(Vector &CG, int k, discreta_base &number_of_orbits);
void print_cycle_type(Vector &C);

// orbit.C:
void all_orbits(int nb_elements, Vector &generators, 
	Vector &orbit_no, 
	int f_schreier_vectors, Vector &schreier_last, Vector &schreier_label, 
	Vector &orbit_elts, Vector &orbit_elts_inv, 
	Vector &orbit_first, Vector &orbit_length, 
	actionkind k, discreta_base &action_data, int f_v, int f_vv, int f_v_image_of);
int image_of_using_integers(int elt, int gen, actionkind k, discreta_base &action_data, int f_v);
void orbit_of_element(Vector &gens, discreta_base &elt, Vector &orbit,
	int f_schreier_data, Vector &schreier_prev, Vector &schreier_label, 
	actionkind k, discreta_base &action_data, int f_v, int f_vv);
void image_of(discreta_base &elt, discreta_base &image, discreta_base &g, actionkind k, discreta_base &action_data);
void induced_permutations_on_orbit(Vector &gens, Vector &induced_gens, Vector &orbit, actionkind k, discreta_base &action_data);
void induced_permutation_on_orbit(discreta_base &gen, permutation &induced_gen, Vector &orbit, actionkind k, discreta_base &action_data);
void trace_schreier_vector(int i, Vector &gen, 
	Vector &schreier_prev, Vector &schreier_label, permutation &p);
void transversal_for_orbit(Vector &gen, 
	Vector &schreier_prev, Vector &schreier_label, Vector &T);
void trace_and_multiply(Vector &T, Vector &gen, 
	Vector &schreier_prev, Vector &schreier_label, int i);
void compute_transversal(Vector &gens, int discreta_base_pt, Vector &T, int f_v, int f_vv);
void allocate_orbit_on_pairs_data(int v, int *&orbits_on_pairs);
void calc_orbits_on_pairs(Vector & gens, int *&orbits_on_pairs, int &nb_orbits, 
	Vector & orbit_first_i, Vector & orbit_first_j, Vector & orbit_length, int f_v, int f_vv);
void write_orbits_on_pairs_to_file(char *group_label, int nb_points, 
	Vector &orbit_first_i, Vector &orbit_first_j, Vector &orbit_length, 
	int *orbits_on_pairs);
void prepare_2_orbits_in_product_action(char *group_label, 
	Vector &gen, int d, int c, int f_v, int f_vv);


// discreta_global.C:
void free_global_data();
void the_end(int t0);
void the_end_quietly(int t0);
void calc_Kramer_Mesner_matrix_neighboring(poset_classification *gen,
	int level, matrix &M, int verbose_level);
// we assume that we don't use implicit fusion nodes
void Mtk_from_MM(Vector & MM, matrix & Mtk, int t, int k, 
	int f_subspaces, int q,  int verbose_level);
void Mtk_via_Mtr_Mrk(int t, int r, int k, int f_subspaces, int q, 
	matrix & Mtr, matrix & Mrk, matrix & Mtk, int verbose_level);
// Computes $M_{tk}$ via a recursion formula:
// $M_{tk} = {{k - t} \choose {k - r}} \cdot M_{t,r} \cdot M_{r,k}$.
void Mtk_sup_to_inf(poset_classification *gen,
	int t, int k, matrix & Mtk_sup, matrix & Mtk_inf, int verbose_level);
void compute_Kramer_Mesner_matrix(poset_classification *gen,
	int t, int k, matrix &M, int f_subspaces, int q, int verbose_level);
void matrix_to_diophant(matrix& M, diophant *&D, int verbose_level);


}








