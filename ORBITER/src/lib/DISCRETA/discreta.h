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

#define BITS_OF_INT 32
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
	// self contains the integer value as a C (long)integer (INT)
		
class longinteger;		// derived from base
	// self is a pointer to LONGINTEGER_REPRESENTATION
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

class bitmatrix;		// derived from base
	// self is a pointer to BITMATRIX_REPRESENTATION
	// which contains integers m, n, N and an array p of UINT4s 
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
	// information stored at offset -3, -2, -1 in INT4s.
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

extern const BYTE *discreta_home;
extern const BYTE *discreta_arch;

typedef class labelled_branching labelled_branching;
typedef class base_change base_change;
typedef class point_orbits point_orbits;

/************************* Prototypes of global functions ******************/

void discreta_init();
discreta_base *callocobject(kind k);
void freeobject(discreta_base *p);
discreta_base *calloc_nobjects(INT n, kind k);
void free_nobjects(discreta_base *p, INT n);
discreta_base *calloc_nobjects_plus_length(INT n, kind k);
void free_nobjects_plus_length(discreta_base *p);
discreta_base *calloc_m_times_n_objects(INT m, INT n, kind k);
void free_m_times_n_objects(discreta_base *p);
void printobjectkind(ostream& ost, kind k);
const char *kind_ascii(kind k);
const char *action_kind_ascii(kind k);
//void INT_swap(INT& x, INT& y);
void UINT4_swap(UINT4& x, UINT4& y);

ostream& operator<<(ostream& ost, class discreta_base& p);
// discreta_base operator * (discreta_base& x, discreta_base &y);
// discreta_base operator + (discreta_base& x, discreta_base &y);

INT lcm_INT(INT m, INT n);
void extended_gcd_int(INT m, INT n, INT &u, INT &v, INT &g);
INT invert_mod_integer(INT i, INT p);
INT remainder_mod(INT i, INT n);
void factor_integer(INT n, Vector& primes, Vector& exponents);
void print_factorization(Vector& primes, Vector& exponents, ostream &o);
void print_factorization_hollerith(Vector& primes, Vector& exponents, hollerith &h);
INT nb_primes(INT n);
//INT is_prime(INT n);
INT factor_if_prime_power(INT n, INT *p, INT *e);
INT Euler(INT n);
INT Moebius(INT i);
INT NormRemainder(INT a, INT m);
INT log2(INT n);
INT sqrt_mod(INT a, INT p);
INT sqrt_mod_involved(INT a, INT p);
//void latex_head(ostream& ost, INT f_book, INT f_title, BYTE *title, BYTE *author, INT f_toc, INT f_landscape);
//void latex_foot(ostream& ost);
void html_head(ostream& ost, BYTE *title_long, BYTE *title_short);
void html_foot(ostream& ost);
void sieve(Vector &primes, INT factorbase, INT f_v);
void sieve_primes(Vector &v, INT from, INT to, INT limit, INT f_v);
void print_intvec_mod_10(Vector &v);
void stirling_second(INT n, INT k, INT f_ordered, discreta_base &res, INT f_v);
void stirling_first(INT n, INT k, INT f_signless, discreta_base &res, INT f_v);
void Catalan(INT n, Vector &v, INT f_v);
void Catalan_n(INT n, Vector &v, discreta_base &res, INT f_v);
void Catalan_nk_matrix(INT n, matrix &Cnk, INT f_v);
void Catalan_nk_star_matrix(INT n, matrix &Cnk, INT f_v);
void Catalan_nk_star(INT n, INT k, matrix &Cnk, discreta_base &res, INT f_v);

INT atoi(char *p);
void N_choose_K(discreta_base & n, INT k, discreta_base & res);
void Binomial(INT n, INT k, discreta_base & n_choose_k);
void Krawtchouk(INT n, INT q, INT i, INT j, discreta_base & a);
// $\sum_{u=0}^{\min(i,j)} (-1)^u \cdot (q-1)^{i-u} \cdot {j \choose u} \cdot $
// ${n - j \choose i - u}$
//INT ij2k(INT i, INT j, INT n);
//void k2ij(INT k, INT & i, INT & j, INT n);
void tuple2_rank(INT rank, INT &i, INT &j, INT n, INT f_injective);
INT tuple2_unrank(INT i, INT j, INT n, INT f_injective);
void output_texable_string(ostream & ost, char *in);
void texable_string(char *in, char *out);
void the_first_n_primes(Vector &P, INT n);
void midpoint_of_2(INT *Px, INT *Py, INT i1, INT i2, INT idx);
void midpoint_of_5(INT *Px, INT *Py, INT i1, INT i2, INT i3, INT i4, INT i5, INT idx);
void ratio_int(INT *Px, INT *Py, INT idx_from, INT idx_to, INT idx_result, double r);

void time_check_delta(INT dt);
void time_check(INT t0);
INT nb_of_bits();
void bit_set(UINT & g, INT k);
void bit_clear(UINT & g, INT k);
INT bit_test(UINT & g, INT k);
void bitset2vector(UINT g, Vector &v);
void frobenius_in_PG(domain *dom, INT n, permutation &p);
// n is the projective dimension
void frobenius_in_AG(domain *dom, INT n, permutation &p);
// n is the dimension
void translation_in_AG(domain *dom, INT n, INT i, discreta_base & a, permutation &p);
enum printing_mode_enum current_printing_mode();
void call_system(BYTE *cmd);
void fill_char(void *v, INT cnt, INT c);
INT hash_INT(INT hash0, INT a);
void queue_init(Vector &Q, INT elt);
INT queue_get_and_remove_first_element(Vector &Q);
INT queue_length(Vector &Q);
void queue_append(Vector &Q, INT elt);
void print_classification_tex(Vector &content, Vector &multiplicities);
void print_classification_tex(Vector &content, Vector &multiplicities, ostream& ost);
void perm2permutation(INT *a, INT n, permutation &p);
//void print_integer_matrix(ostream &ost, INT *p, INT m, INT n);
//void print_longinteger_matrix(ostream &ost, LONGINT *p, INT m, INT n); removed Anton Betten Nov 1, 2011
INT Gauss_INT(INT *A, INT f_special, INT f_complete, INT *base_cols,
	INT f_P, INT *P, INT m, INT n, INT Pn, 
	INT q, INT *add_table, INT *mult_table, INT *negate_table, INT *inv_table, INT f_v);
// returns the rank which is the number of entries in base_cols
void UBYTE_move(UBYTE *p, UBYTE *q, INT len);
void INT_vector_realloc(INT *&p, INT old_length, INT new_length);
void INT_vector_shorten(INT *&p, INT new_length);
void INT_matrix_realloc(INT *&p, INT old_m, INT new_m, INT old_n, INT new_n);
INT code_is_irreducible(INT k, INT nmk, INT idx_zero, INT *M);
void fine_tune(finite_field *F, INT *mtxD, INT verbose_level);

// in mindist.C:
int mindist(int n, int k, int q, int *G, 
	int f_v, int f_vv, int idx_zero, int idx_one, 
	int *add_table, int *mult_table);

// domain.C:

INT has_domain();
domain *get_current_domain();
//domain *get_domain_if_pc_group();
INT is_GFp_domain(domain *& d);
INT is_GFq_domain(domain *& d);
INT is_finite_field_domain(domain *& d);
INT finite_field_domain_order_int(domain * d);
INT finite_field_domain_characteristic(domain * d);
INT finite_field_domain_primitive_root();
void finite_field_domain_base_over_subfield(Vector & b);
void push_domain(domain *d);
void pop_domain(domain *& d);
domain *allocate_finite_field_domain(INT q, INT f_v);
void free_finite_field_domain(domain *dom, INT f_v);

/************************************* base ********************************/

// internal representations:

typedef struct longinteger_representation LONGINTEGER_REPRESENTATION;
typedef struct bitmatrix_representation BITMATRIX_REPRESENTATION;

typedef union {
	INT integer_value;
	char *char_pointer;
	INT *int_pointer;
	discreta_base *vector_pointer;
	discreta_base *matrix_pointer;
	LONGINTEGER_REPRESENTATION *longinteger_rep;
	BITMATRIX_REPRESENTATION *bitmatrix_rep;
} OBJECTSELF;

struct longinteger_representation {
	INT sign;
	INT len;
	char p[1];
};

struct bitmatrix_representation {
	INT m;
	INT n;
	INT N;
	UINT4 p[1];
};


// public class definitions:

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
	bitmatrix& as_bitmatrix() { return *(bitmatrix *)this; }
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
	bitmatrix& change_to_bitmatrix() { freeself(); c_kind(BITMATRIX); return as_bitmatrix(); }
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

	INT& s_i_i();
		// select_as_integer_i
	void m_i_i(INT i);
		// make_as_integer_i

	virtual INT compare_with(discreta_base &a);
		// -1 iff this < a
		// 0 iff this = a
		// 1 iff this > a
	INT eq(discreta_base &a);
	INT neq(discreta_base &a);
	INT le(discreta_base &a);
	INT lt(discreta_base &a);
	INT ge(discreta_base &a);
	INT gt(discreta_base &a);
	INT is_even();
	INT is_odd();
	
	
	// mathematical functions:
    
	// multiplicative group:
	void mult(discreta_base &x, discreta_base &y);
		// this := x * y
	void mult_mod(discreta_base &x, discreta_base &y, discreta_base &p);
	virtual void mult_to(discreta_base &x, discreta_base &y);
		// y := this * x
	INT invert();
		// this := this^(-1)
		// returns TRUE if the object was invertible,
		// FALSE otherwise
	INT invert_mod(discreta_base &p);
	virtual INT invert_to(discreta_base &x);
	void mult_apply(discreta_base &x);
		// this := this * x
	discreta_base& operator *= (discreta_base &y)
		{ mult_apply(y); return *this; }
	discreta_base& power_int(INT l);
		// this := this^l, l >= 0
	discreta_base& power_int_mod(INT l, discreta_base &p);
	discreta_base& power_longinteger(longinteger &l);
	discreta_base& power_longinteger_mod(longinteger &l, discreta_base &p);
	discreta_base& commutator(discreta_base &x, discreta_base &y);
		// this := x^{-1} * y^{-1} * x * y
	discreta_base& conjugate(discreta_base &x, discreta_base &y);
		// this := y^{-1} * x * y
	discreta_base& divide_by(discreta_base& x);
	discreta_base& divide_by_exact(discreta_base& x);
	INT order();
	INT order_mod(discreta_base &p);
	

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
	virtual void homo_z(INT z);
		// this := z
	virtual void inc();
		// this := this + 1
	virtual void dec();
		// this := this - 1
	virtual INT is_zero();
		// TRUE iff this = 0
	virtual INT is_one();
		// TRUE iff this = 1
	virtual INT is_m_one();
		// TRUE iff this = -1
	discreta_base& factorial(INT z);
		// this := z!
	discreta_base& i_power_j(INT i, INT j);
		// this := i^j

	virtual INT compare_with_euklidean(discreta_base &a);
		// -1 iff this < a
		// 0 iff this = a
		// 1 iff this > a
	virtual void integral_division(discreta_base &x, discreta_base &q, discreta_base &r, INT verbose_level);
	void integral_division_exact(discreta_base &x, discreta_base &q);
	void integral_division_by_integer(INT x, discreta_base &q, discreta_base &r);
	void integral_division_by_integer_exact(INT x, discreta_base &q);
	void integral_division_by_integer_exact_apply(INT x);
	INT is_divisor(discreta_base &y);
	void modulo(discreta_base &p);
	void extended_gcd(discreta_base &n, discreta_base &u, discreta_base &v, discreta_base &g, INT verbose_level);
	void write_memory(memory &m, INT debug_depth);
	void read_memory(memory &m, INT debug_depth);
	INT calc_size_on_file();
	void pack(memory & M, INT f_v, INT debug_depth);
	void unpack(memory & M, INT f_v, INT debug_depth);
	void save_ascii(ostream & f);
	void load_ascii(istream & f);
	void save_file(char *fname);
	void load_file(char *fname);
};


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
	INT & alloc_length() { return self.int_pointer[-3]; }
	INT & used_length() { return self.int_pointer[-2]; }
	INT & cur_pointer() { return self.int_pointer[-1]; }

	char & s_i(INT i) { return self.char_pointer[i]; };
	char & operator [] (INT i) { return s_i(i); }

	void init(INT length, char *d);
	void alloc(INT length);
	void append(INT length, char *d);
	void realloc(INT new_length);
	void write_char(char c);
	void read_char(char *c);
	void write_int(INT i);
	void read_int(INT *i);
	void read_file(BYTE *fname, INT f_v);
	void write_file(BYTE *fname, INT f_v);
	INT multiplicity_of_character(BYTE c);
	void compress(INT f_v);
	void decompress(INT f_v);
	INT csf();
	void write_mem(memory & M, INT debug_depth);
	void read_mem(memory & M, INT debug_depth);
	
};

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
	INT compare_with(discreta_base &a);

	char * s_unchecked() { return self.char_pointer; } 
	char * s() { if (self.char_pointer) return self.char_pointer; else return (char *) ""; }
	void init(const char *p);
	void append(const char *p);
	void append_i(INT i);
	void write_mem(memory & m, INT debug_depth);
	void read_mem(memory & m, INT debug_depth);
	INT csf();
	void chop_off_extension_if_present(BYTE *ext);
	void get_extension_if_present(BYTE *ext);
	void get_current_date();
};

class integer: public discreta_base
{
	public:
	integer();
	integer(BYTE *p);
	integer(INT i);
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

	integer& m_i(INT i);				// make_integer
	INT& s_i() { return self.integer_value; };	// select_integer

	INT compare_with(discreta_base &a);

	void mult_to(discreta_base &x, discreta_base &y);
	INT invert_to(discreta_base &x);
	
	void add_to(discreta_base &x, discreta_base &y);
	void negate_to(discreta_base &x);
	
	void normalize(discreta_base &p);
	void zero();
	void one();
	void m_one();
	void homo_z(INT z);
	void inc();
	void dec();
	INT is_zero();
	INT is_one();
	INT is_m_one();

	INT compare_with_euklidean(discreta_base &a);
	void integral_division(discreta_base &x, discreta_base &q, discreta_base &r, INT verbose_level);
	
	void rand(INT low, INT high);
	INT log2();
};

#define LONGINTEGER_PRINT_DOTS
#define LONGINTEGER_DIGITS_FOR_DOT 6

class longinteger: public discreta_base
{
	public:
	longinteger();
	longinteger(INT a);
	//longinteger(LONGINT a); removed Anton Betten Nov 1, 2011
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
	INT& s_sign();
	INT& s_len();
	char& s_p(INT i);
	void allocate(INT sign, const char *p);
	void allocate_internal(INT sign, INT len, const char *p);
	void allocate_empty(INT len);
	void normalize_representation();
	
	INT compare_with(discreta_base &b);
	INT compare_with_unsigned(longinteger &b);

	void mult_to(discreta_base &x, discreta_base &y);
	INT invert_to(discreta_base &x);
	void add_to(discreta_base &x, discreta_base &y);
	void negate_to(discreta_base &x);
	
	void zero();
	void one();
	void m_one();
	void homo_z(INT z);
	// void homo_z(LONGINT z); removed Anton Betten Nov 1, 2011
	void inc();
	void dec();
	INT is_zero();
	INT is_one();
	INT is_m_one();
	INT is_even();
	INT is_odd();

	INT compare_with_euklidean(discreta_base &b);
	void integral_division(discreta_base &x, discreta_base &q, discreta_base &r, INT verbose_level);
	void square_root_floor(discreta_base &x);
	longinteger& Mersenne(INT n);
	longinteger& Fermat(INT n);
	INT s_i();
	INT retract_to_integer_if_possible(integer &x);
	INT modp(INT p);
	INT ny_p(INT p);
	void divide_out_int(INT d);

	INT Lucas_test_Mersenne(INT m, INT f_v);
};


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
	
	discreta_base & s_i(INT i);
		// select i-th vector element
	INT& s_ii(INT i) 
		{ return s_i(i).s_i_i(); }
		// select i-th vector element as integer
	void m_ii(INT i, INT a) { s_i(i).m_i_i(a); }
		// make i-th vector element as integer (set value)
	discreta_base & operator [] (INT i)
		{ return s_i(i); }
	INT s_l();			
		// select vector length, 
		// length is 0 if vector_pointer is NULL
	void m_l(INT l);
		// make vector of length l
		// allocates the memory and sets the objects to type BASE
	void m_l_n(INT l);
		// make vector of length l of integers, initializes with 0
	void m_l_e(INT l);
		// make vector of length l of integers, initializes with 1
	void m_l_x(INT l, discreta_base &x);
		// allocates a vector of l copies of x
	Vector& realloc(INT l);
	void mult_to(discreta_base &x, discreta_base &y);
	void add_to(discreta_base &x, discreta_base &y);
	void inc();
	void dec();

	INT compare_with(discreta_base &a);
	void append_vector(Vector &v);
	Vector& append_integer(INT a);
	Vector& append(discreta_base &x);
	Vector& insert_element(INT i, discreta_base& x);
	Vector& get_and_delete_element(INT i, discreta_base& x);
	Vector& delete_element(INT i);
	void get_first_and_remove(discreta_base & x);
	bool insert_sorted(discreta_base& x);
		// inserts x into the sorted vector x.
		// ifthere are already occurences of x, the new x is added 
		// behind the x already there.
		// returns true if the element was already in the vector.
	bool search(discreta_base& x, INT *idx);
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





	void n_choose_k_first(INT n, INT k);
		// computes the lexicographically first k-subset of {0,...,n-1}
	INT n_choose_k_next(INT n, INT k);
		// computes the lexicographically next k-subset
		// returns FALSE if there is no further k-subset
		// example: n = 4, k = 2
		// first gives (0,1),
		// next gives (0,2), then (0,3), (1,2), (1,3), (2,3).
	void first_lehmercode(INT n) 
		{ m_l_n(n); }
		// first lehmercode = 0...0 (n times)
	INT next_lehmercode();
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
	void q_adic(INT n, INT q);
	INT q_adic_as_int(INT q);
	void mult_scalar(discreta_base& a);
	void first_word(INT n, INT q);
	INT next_word(INT q);
	void first_regular_word(INT n, INT q);
	INT next_regular_word(INT q);
	INT is_regular_word();

	void apply_permutation(permutation &p);
	void apply_permutation_to_elements(permutation &p);
	void content(Vector & c, Vector & where);
	void content_multiplicities_only(Vector & c, Vector & mult);

	INT hip();
	INT hip1();
	void write_mem(memory & m, INT debug_depth);
	void read_mem(memory & m, INT debug_depth);
	INT csf();

	void conjugate(discreta_base & a);
	void conjugate_with_inverse(discreta_base & a);
	void replace(Vector &v);
	void vector_of_vectors_replace(Vector &v);
	void extract_subvector(Vector & v, INT first, INT len);

	void PG_element_normalize();
	void PG_element_rank(INT &a);
	void PG_element_rank_modified(INT &a);
	void PG_element_unrank(INT a);
	void PG_element_unrank_modified(INT a);
	void AG_element_rank(INT &a);
	void AG_element_unrank(INT a);
	
	INT hamming_weight();
	void scalar_product(Vector &w, discreta_base & a);
	void hadamard_product(Vector &w);
	void intersect(Vector& b, Vector &c);
	INT vector_of_vectors_overall_length();
	void first_divisor(Vector &exponents);
	INT next_divisor(Vector &exponents);
	INT next_non_trivial_divisor(Vector &exponents);
	void multiply_out(Vector &primes, discreta_base &x);
	INT hash(INT hash0);
	INT is_subset_of(Vector &w);
	void concatenation(Vector &v1, Vector &v2);
	void print_word_nicely(ostream &ost, INT f_generator_labels, Vector &generator_labels);
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
//INT nb_PG_elements(INT n, INT q);
//INT nb_AG_elements(INT n, INT q);
void intersection_of_vectors(Vector& V, Vector& v);


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
	void sscan(const char *s, INT f_v);
	void scan(istream & is, INT f_v);

	void m_l(INT l);
	INT& s_i(INT i);
	INT& operator [] (INT i) 
		{ return s_i(i); }

	void mult_to(discreta_base &x, discreta_base &y);
	INT invert_to(discreta_base &x);
	void one();
	INT is_one();
	INT compare_with(discreta_base &a);

	void write_mem(memory & m, INT debug_depth);
	void read_mem(memory & m, INT debug_depth);
	INT csf();
	void get_fixpoints(Vector &f);
	void induce_action_on_blocks(permutation & gg, Vector & B);
	void induce3(permutation & b);
	void induce2(permutation & b);
	void induce_on_2tuples(permutation & p, INT f_injective);
	void add_n_fixpoints_in_front(permutation & b, INT n);
	void add_n_fixpoints_at_end(permutation & b, INT n);
	void add_fixpoint_in_front(permutation & b);
	void embed_at(permutation & b, INT n, INT at);
	void remove_fixpoint(permutation & b, INT i);
	void join(permutation & a, permutation & b);
	void cartesian_product_action(permutation & a, permutation & b);
	void Add2Cycle(INT i0, INT i1);
	void Add3Cycle(INT i0, INT i1, INT i2);
	void Add4Cycle(INT i0, INT i1, INT i2, INT i3);
	void Add5Cycle(INT i0, INT i1, INT i2, INT i3, INT i4);
	// void Add6Cycle(INT i0, INT i1, INT i2, INT i3, INT i4, INT i5);
	// void Add7Cycle(INT i0, INT i1, INT i2, INT i3, INT i4, INT i5, INT i6);
	// void Add8Cycle(INT i0, INT i1, INT i2, INT i3, INT i4, INT i5, INT i6, INT i7);
	void AddNCycle(INT first, INT len);

	// influence the behaviour of printing of permutations:
	void set_print_type_integer_from_zero();
	void set_print_type_integer_from_one();
	void set_print_type_PG_1_q_element(domain *dom);

	void convert_digit(INT i, hollerith &a);
	void cycle_type(Vector& type, INT f_v);
	INT nb_of_inversions(INT f_v);
	INT signum(INT f_v);
	INT is_even(INT f_v);
	void cycles(Vector &cycles);
	void restrict_to_subset(permutation &q, INT first, INT len);
	void induce_on_lines_of_PG_k_q(INT k, INT q, permutation &per, INT f_v, INT f_vv);
	void singer_cycle_on_points_of_projective_plane(INT p, INT f_modified, INT f_v);
	void Cn_in_Cnm(INT n, INT m);
	INT preimage(INT i);
};

void signum_map(discreta_base & x, discreta_base &d);
//char get_character(istream & is, INT f_v);



class matrix_access {
public:
	INT i;
	matrix *p;
	discreta_base & operator [] (INT j);
};


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
	INT compare_with(discreta_base &a);

	matrix& m_mn(INT m, INT n);
		// make matrix of format m times n
		// allocates the memory and sets the objects to type BASE
	matrix& m_mn_n(INT m, INT n);
	matrix& realloc(INT m, INT n);


	INT s_m();
	INT s_n();
	discreta_base & s_ij(INT i, INT j);
		// select (i,j)-th matrix element
	INT& s_iji(INT i, INT j)
		{ return s_ij(i, j).s_i_i(); }
	void m_iji(INT i, INT j, INT a) 
		{ s_ij(i, j).m_i_i(a); }
		// make (i,j)-th vector element as integer (set value)
	
	matrix_access operator [] (INT i) 
		{ matrix_access ma = { i, this };  return ma; }
		// overload access operator

	void mult_to(discreta_base &x, discreta_base &y);
	void matrix_mult_to(matrix &x, discreta_base &y);
	void vector_mult_to(Vector &x, discreta_base &y);
	void multiply_vector_from_left(Vector &x, Vector &y);
	INT invert_to(discreta_base &x);
	void add_to(discreta_base &x, discreta_base &y);
	void negate_to(discreta_base &x);
	void one();
	void zero();
	INT is_zero();
	INT is_one();


	INT Gauss(INT f_special, INT f_complete, Vector& base_cols, INT f_P, matrix& P, INT f_v);
	INT rank();
	INT get_kernel(Vector& base_cols, matrix& kernel);
	matrix& transpose();
	INT Asup2Ainf();
	INT Ainf2Asup();
	INT Asup2Acover();
	INT Acover2nl(Vector& nl);

	void Frobenius(unipoly& m, INT p, INT verbose_level);
	void Berlekamp(unipoly& m, INT p, INT verbose_level);
	void companion_matrix(unipoly& m, INT verbose_level);
	
	void elements_to_unipoly();
	void minus_X_times_id();
	void X_times_id_minus_self();
	void smith_normal_form(matrix& P, matrix& Pv, matrix& Q, matrix& Qv, INT verbose_level);
	INT smith_eliminate_column(matrix& P, matrix& Pv, INT i, INT verbose_level);
	INT smith_eliminate_row(matrix& Q, matrix& Qv, INT i, INT verbose_level);
	void multiply_2by2_from_left(INT i, INT j, 
		discreta_base& aii, discreta_base& aij, discreta_base& aji, discreta_base& ajj, INT verbose_level);
	void multiply_2by2_from_right(INT i, INT j, 
		discreta_base& aii, discreta_base& aij, discreta_base& aji, discreta_base& ajj, INT verbose_level);

	void to_vector_of_rows(Vector& v);
	void from_vector_of_rows(Vector& v);
	void to_vector_of_columns(Vector& v);
	void from_vector_of_columns(Vector& v);
	void evaluate_at(discreta_base& x);
	void KX_module_order_ideal(INT i, unipoly& mue, INT verbose_level);
	void KX_module_apply(unipoly& p, Vector& v);
	void KX_module_join(Vector& v1, unipoly& mue1, 
		Vector& v2, unipoly& mue2, Vector& v3, unipoly& mue3, INT verbose_level);
	void KX_cyclic_module_generator(Vector& v, unipoly& mue, INT verbose_level);
	void KX_module_minpol(unipoly& p, unipoly& m, unipoly& mue, INT verbose_level);

	void binomial(INT n_min, INT n_max, INT k_min, INT k_max);
	void stirling_second(INT n_min, INT n_max, INT k_min, INT k_max, INT f_ordered);
	void stirling_first(INT n_min, INT n_max, INT k_min, INT k_max, INT f_signless);
	void binomial(INT n_min, INT n_max, INT k_min, INT k_max, INT f_inverse);
	INT hip();
	INT hip1();
	void write_mem(memory & m, INT debug_depth);
	void read_mem(memory & M, INT debug_depth);
	INT csf();

	void calc_theX(INT & nb_X, INT *&theX);
#if 0
	void lexleast_incidence_matrix(INT f_on_rows, 
		INT f_row_decomp, Vector & row_decomp, 
		INT f_col_decomp, Vector & col_decomp, 
		INT f_ddp, Vector & DDp, 
		INT f_ddb, Vector & DDb, 
		INT f_group, perm_group & G, 
		permutation & p, permutation & q, 
		INT f_print_backtrack_points, 
		INT f_get_aut_group, INT f_aut_group_on_lexleast, Vector & aut_gens, 
		INT f_v, INT f_vv);
#endif
	void apply_perms(INT f_row_perm, permutation &row_perm, 
		INT f_col_perm, permutation &col_perm);
	void apply_col_row_perm(permutation &p);
	void apply_row_col_perm(permutation &p);
	void incma_print_ascii_permuted_and_decomposed(ostream &ost, INT f_tex, 
		Vector & decomp, permutation & p);
	void print_decomposed(ostream &ost, Vector &row_decomp, Vector &col_decomp);
	void incma_print_ascii(ostream &ost, INT f_tex, 
		INT f_row_decomp, Vector &row_decomp, 
		INT f_col_decomp, Vector &col_decomp);
	void incma_print_latex(ostream &f, 
		INT f_row_decomp, Vector &row_decomp, 
		INT f_col_decomp, Vector &col_decomp, 
		INT f_labelling_points, Vector &point_labels, 
		INT f_labelling_blocks, Vector &block_labels);
	void incma_print_latex2(ostream &f, 
		INT width, INT width_10, 
		INT f_outline_thin, const BYTE *unit_length, 
		const BYTE *thick_lines, const BYTE *thin_lines, const BYTE *geo_line_width, 
		INT f_row_decomp, Vector &row_decomp, 
		INT f_col_decomp, Vector &col_decomp, 
		INT f_labelling_points, Vector &point_labels, 
		INT f_labelling_blocks, Vector &block_labels);
	void calc_hash_key(INT key_len, hollerith & hash_key, INT f_v);
	INT is_in_center();
	void power_mod(INT r, integer &P, matrix &C);
	INT proj_order_mod(integer &P);
	void PG_rep(domain *dom, permutation &p, INT f_action_from_right, INT f_modified);
	void PG_rep(permutation &p, INT f_action_from_right, INT f_modified);
	void AG_rep(domain *dom, permutation &p, INT f_action_from_right);
	void AG_rep(permutation &p, INT f_action_from_right);
	void MacWilliamsTransform(INT n, INT q, INT f_v);
	void weight_enumerator_brute_force(domain *dom, Vector &v);
	void Simplex_code_generator_matrix(domain *dom, INT k, INT f_v);
	void PG_design_point_vs_hyperplane(domain *dom, INT k, INT f_v);
	void PG_k_q_design(domain *dom, INT k, INT f_v, INT f_vv);
	void determinant(discreta_base &d, INT verbose_level);
	void det(discreta_base & d, INT f_v, INT f_vv);
	void det_modify_input_matrix(discreta_base & d, INT f_v, INT f_vv);
	void PG_line_rank(INT &a, INT f_v);
	void PG_line_unrank(INT a);
	void PG_point_normalize(INT i0, INT j0, INT di, INT dj, INT length);
	void PG_point_unrank(INT i0, INT j0, INT di, INT dj, INT length, INT a);
	void PG_point_rank(INT i0, INT j0, INT di, INT dj, INT length, INT &a);
	void PG_element_normalize();
	void AG_point_rank(INT i0, INT j0, INT di, INT dj, INT length, INT &a);
	void AG_point_unrank(INT i0, INT j0, INT di, INT dj, INT length, INT a);
#if 0
	void canon(INT f_row_decomp, Vector & row_decomp, 
		INT f_col_decomp, Vector & col_decomp, 
		INT f_group, perm_group & G, 
		permutation & p, permutation & q, 
		INT f_get_aut_group, INT f_aut_group_on_lexleast, Vector & aut_gens, discreta_base &ago,
		INT f_v, INT f_vv, INT f_vvv, INT f_vvvv, INT f_tree_file);
	void canon_partition_backtrack(INT f_row_decomp, Vector & row_decomp, 
		INT f_col_decomp, Vector & col_decomp, 
		INT f_group, perm_group & G, 
		permutation & p, permutation & q, 
		INT f_get_aut_group, INT f_aut_group_on_lexleast, Vector & aut_gens, discreta_base &ago,
		INT f_v, INT f_vv, INT f_vvv, INT f_vvvv, INT f_tree_file);
#endif
	void canon_nauty(INT f_row_decomp, Vector & row_decomp, 
		INT f_col_decomp, Vector & col_decomp, 
		INT f_group, perm_group & G, 
		permutation & p, permutation & q, 
		INT f_get_aut_group, INT f_aut_group_on_lexleast, Vector & aut_gens, 
		INT f_v, INT f_vv, INT f_vvv);
#if 0
	void canon_tonchev(INT f_row_decomp, Vector & row_decomp, 
		INT f_col_decomp, Vector & col_decomp, 
		INT f_group, perm_group & G, 
		permutation & p, permutation & q, 
		INT f_get_aut_group, INT f_aut_group_on_lexleast, Vector & aut_gens, 
		INT f_v, INT f_vv, INT f_vvv);
#endif
	void save_as_geometry(INT number, BYTE *label);
	void save_as_inc_file(BYTE *fname);
	void save_as_inc(ofstream &f);
};

void determinant_map(discreta_base & x, discreta_base &d);
INT nb_PG_lines(INT n, INT q);


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

	bitmatrix& m_mn(INT m, INT n);
		// make matrix of format m times n
		// allocates the memory
	bitmatrix& m_mn_n(INT m, INT n);


	INT s_m();
	INT s_n();
	INT s_N();
	UINT4& s_i(INT i);
	INT s_ij(INT i, INT j);
		// select (i,j)-th matrix element
	void m_iji(INT i, INT j, INT a);
		// make (i,j)-th vector element as integer (set value)

	void mult_to(discreta_base &x, discreta_base &y);
	void bitmatrix_mult_to(bitmatrix &x, discreta_base &y);
	
	INT gauss(INT f_complete, Vector& base_cols, INT f_v);
	INT get_kernel(Vector& base_cols, bitmatrix& kernel);

	void write_mem(memory & M, INT debug_depth);
	void read_mem(memory & M, INT debug_depth);
	INT csf();
};


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

	void m_l(INT l);
	INT degree();

	void mult_to(discreta_base &x, discreta_base &y);
	void add_to(discreta_base &x, discreta_base &y);
	void negate_to(discreta_base &x);
	void one();
	void zero();
	void x();
	void x_to_the_i(INT i);
	INT is_one();
	INT is_zero();
	INT compare_with_euklidean(discreta_base &a);
	void integral_division(discreta_base &x, discreta_base &q, discreta_base &r, INT verbose_level);
	void derive();
	INT is_squarefree(INT verbose_level);
	INT is_irreducible_GFp(INT p, INT verbose_level);
	INT is_irreducible(INT q, INT f_v);
	INT is_primitive(INT m, INT p, Vector& vp, INT verbose_level);
	void numeric_polynomial(INT n, INT q);
	INT polynomial_numeric(INT q);
	void singer_candidate(INT p, INT f, INT b, INT a);
	void Singer(INT p, INT f, INT f_v, INT f_vv);
	void get_an_irreducible_polynomial(INT f, INT verbose_level);
	void evaluate_at(discreta_base& x, discreta_base& y);
	void largest_divisor_prime_to(unipoly& q, unipoly& r);
	void monic();
	void normal_base(INT p, matrix& F, matrix& N, INT verbose_level);
	INT first_irreducible_polynomial(INT p, unipoly& m, matrix& F, matrix& N, Vector &v, INT verbose_level);
	INT next_irreducible_polynomial(INT p, unipoly& m, matrix& F, matrix& N, Vector &v, INT verbose_level);
	void normalize(discreta_base &p);
	void Xnm1(INT n);
	void Phi(INT n, INT f_v);
	void weight_enumerator_MDS_code(INT n, INT k, INT q, INT f_v, INT f_vv, INT f_vvv);
	void charpoly(INT q, INT size, INT *mtx, INT verbose_level);
	
};

// vbp.C:
void place_lattice(Vector& nl, Vector& orbit_size, 
	INT size_x, INT size_y, 
	Vector& Px, Vector& Py, Vector& O_dx, 
	INT f_upside_down, INT f_v);





#define PARTITION_TYPE_VECTOR 0
#define PARTITION_TYPE_EXPONENT 1

class number_partition: public Vector 
{
	public:
	number_partition();
		// constructor, sets the vector_pointer to NULL
	number_partition(INT n);
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

	INT & s_type() { return Vector::s_i(0).as_integer().s_i(); }
	Vector & s_self() { return Vector::s_i(1).as_vector(); }
	
	void m_l(INT l) { s_self().m_l_n(l); }
	INT s_l() { return s_self().s_l(); }
	INT & s_i(INT i) { return s_self().s_ii(i); }
	INT & operator [] (INT i) { return s_self().s_ii(i); }

	void first(INT n);
	INT next();
	INT next_exponent();
	INT next_vector();
	INT first_into_k_parts(INT n, INT k);
	INT next_into_k_parts(INT n, INT k);
	INT first_into_at_most_k_parts(INT n, INT k);
	INT next_into_at_most_k_parts(INT n, INT k);
	INT nb_parts();
	void conjugate();
	void type(number_partition &q);
	void multinomial(discreta_base &res, INT f_v);
	void multinomial_ordered(discreta_base &res, INT f_v);
	INT sum_of_decreased_parts();

};

INT first_passport(Vector &pass, INT n, INT k);
INT next_passport(Vector &pass, INT n, INT k);
INT first_passport_i(Vector &pass, INT n, INT k, INT i, INT & S);
INT next_passport_i(Vector &pass, INT n, INT k, INT i, INT & S);

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
	INT scan(istream&);
	void scan_body(istream& f, INT geo_nr, BYTE *geo_label);


	INT & number() { return Vector::s_i(0).as_integer().s_i(); }
	hollerith & label() { return Vector::s_i(1).as_hollerith(); }
	matrix & X() { return Vector::s_i(2).as_matrix(); }
	INT & f_incidence_matrix() { return Vector::s_i(3).as_integer().s_i(); }
	
	Vector & point_labels() { return Vector::s_i(4).as_vector(); }
	Vector & block_labels() { return Vector::s_i(5).as_vector(); }

	INT & f_row_decomp() { return Vector::s_i(6).as_integer().s_i(); }
	Vector & row_decomp() { return Vector::s_i(7).as_vector(); }
	INT & f_col_decomp() { return Vector::s_i(8).as_integer().s_i(); }
	Vector & col_decomp() { return Vector::s_i(9).as_vector(); }
	
	INT & f_ddp() { return Vector::s_i(10).as_integer().s_i(); }
	Vector & ddp() { return Vector::s_i(11).as_vector(); }
	INT & f_ddb() { return Vector::s_i(12).as_integer().s_i(); }
	Vector & ddb() { return Vector::s_i(13).as_vector(); }

	INT & f_canonical_labelling_points() { return Vector::s_i(14).as_integer().s_i(); }
	permutation & canonical_labelling_points() { return Vector::s_i(15).as_permutation(); }
	INT & f_canonical_labelling_blocks() { return Vector::s_i(16).as_integer().s_i(); }
	permutation & canonical_labelling_blocks() { return Vector::s_i(17).as_permutation(); }

	INT & f_aut_gens() { return Vector::s_i(18).as_integer().s_i(); }
	Vector & aut_gens() { return Vector::s_i(19).as_vector(); }
	discreta_base & ago() { return Vector::s_i(20); }

	void transpose();
	INT is_2design(INT &r, INT &lambda, INT f_v);
#if 0
	void calc_lexleast_and_autgroup(INT f_v, INT f_vv, INT f_print_backtrack_point);
	void calc_canon_and_autgroup(INT f_v, INT f_vv, INT f_vvv, INT f_vvvv, 
		INT f_print_backtrack_points, INT f_tree_file);
	void calc_canon_and_autgroup_partition_backtrack(INT f_v, INT f_vv, INT f_vvv, INT f_vvvv, 
		INT f_print_backtrack_points, INT f_tree_file);
#endif
	void calc_canon_nauty(INT f_v, INT f_vv, INT f_vvv);
	//void calc_canon_tonchev(INT f_v, INT f_vv, INT f_vvv);
	void get_lexleast_X(matrix & X0);
};

INT search_geo_file(matrix & X0, BYTE *fname, INT geo_nr, BYTE *geo_label, INT f_v);

// geo_canon.C:

void perm_test(void);
void geo_canon_with_initial_decomposition_and_ddp_ddb(
	INT f_maxtest, INT *back_to, 
	INT f_transposed, 
	INT nrow, INT ncol, INT nb_X, INT *theX, 
	INT f_row_decomp, Vector & row_decomp, 
	INT f_col_decomp, Vector & col_decomp, 
	INT f_ddp, Vector & DDp, 
	INT f_ddb, Vector & DDb, 
	INT f_col_group, perm_group & col_group, 
	permutation & p, permutation & q, 
	INT f_print_backtrack_points, 
	INT f_get_aut_group, INT f_aut_group_on_lexleast, Vector & aut_gens, 
	INT f_v, INT f_vv);



class domain {
	private:
		domain_type the_type;
		discreta_base the_prime;
		//pc_presentation *the_pres;
		unipoly *the_factor_poly;
		domain *the_sub_domain;
	
	public:
	domain(INT p);
	domain(unipoly *factor_poly, domain *sub_domain);
	//domain(pc_presentation *pres);
	
	domain_type type();
	INT order_int();
	INT order_subfield_int();
	INT characteristic();
	//pc_presentation *pres();
	unipoly *factor_poly();
	domain *sub_domain();
};


class with {
	private:
	public:

	with(domain *dom);
	~with();
};

class printing_mode {
	private:
	public:
	printing_mode(enum printing_mode_enum printing_mode);
	~printing_mode();
};

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

	INT & type() { return Vector::s_i(0).as_integer().s_i(); }
	INT & val1() { return Vector::s_i(1).as_integer().s_i(); }
	INT & val2() { return Vector::s_i(2).as_integer().s_i(); }
	hollerith & s() { return Vector::s_i(3).as_hollerith(); }

	void init(group_selection_type type, INT v1, INT v2, BYTE *str);
};

const char *group_selection_type_as_text(group_selection_type t);
void compose_gsel_from_strings(Vector &gsel, INT num_args, char **args);
void compose_group(Vector & gsel, Vector & gens, 
	hollerith & group_label, hollerith & group_label_tex, hollerith & acting_on, INT f_v);


// perm_group_gens.C:

INT vec_generators_is_trivial_group(Vector & gen);
INT is_abelian(Vector & gen);
void read_file_of_generators_xml(Vector & gen, char *fname, INT &f_cyclic_notation, INT f_v);
void write_file_of_generators_xml_group_label(Vector & gen, char *group_label, INT f_cyclic_notation);
void write_file_of_generators_xml(Vector & gen, char *fname, INT f_cyclic_notation);
void read_file_of_generators(Vector & G, char *fname);
void read_generators(Vector & G, ifstream & f);
void write_file_of_generators_group_label(Vector & gen, char *group_label);
void write_file_of_generators(Vector & G, char *fname);
void write_generators(Vector & G, ofstream & f);
void write_file_of_generators_gap_group_label(Vector & gen, char *group_label);
void write_file_of_generators_gap(Vector & G, char *fname);
void write_generators_gap(Vector & G, ofstream & f);
void vec_induced_group_on_subset(Vector & V, Vector & subset, Vector & W);
void vec_subgroup_of_hol_of_cyclic_group(Vector & V, INT n, INT i);
void vec_hol_of_cyclic_group(Vector & V, INT n);
void vec_conjugate(Vector & gen, permutation & p);
void vec_induce_action_on_blocks(Vector & gen, Vector & B);
void vec_induce3(Vector & gen);
void vec_induce2(Vector & gen);
void vec_induce_on_2tuples(Vector & gen, INT f_injective);
void vec_add_fixpoint_in_front(Vector & gen);
void vec_add_fixpoint_at_end(Vector & gen);
INT vec_generators_degree(Vector & a);
//void vec_generators_stabilize_point(Vector & a, Vector & b);
//void vec_generators_group_order(Vector & gen, discreta_base & o);
void vec_generators_remove_fixpoint(Vector & gen, INT i);
void vec_generators_raise_to_nth_power(Vector & gen, INT n);
void vec_generators_induce_on_lines_of_PG_k_q(Vector & gen, INT k, INT q, INT f_v, INT f_vv);
void vec_generators_trivial_group(Vector & gen, INT deg);
void vec_generators_cyclic_group(Vector & gen, INT deg);
void vec_generators_Cn_in_Cnm(Vector & gen, INT n, INT m);
void vec_generators_AutCq_in_Cqm(Vector & gen, INT q, INT m);
void vec_generators_symmetric_group(Vector & gen, INT deg);
void vec_generators_alternating_group(Vector & gen, INT deg);
void vec_generators_alternating_group_huppert(Vector & gen, INT deg);
void vec_generators_dihedral_group(Vector & gen, INT deg);
void vec_generators_Mathieu_n(Vector & gen, INT n);
void vec_generators_Mathieu_11(Vector & gen);
void vec_generators_Mathieu_12(Vector & gen);
void vec_generators_Mathieu_23(Vector & gen);
void vec_generators_Mathieu_24(Vector & gen);
void vec_generators_diagonal_sum(Vector & a, Vector & b, Vector & c);
void vec_generators_comma(Vector & a, Vector & b, Vector & c);
void vec_generators_direct_sum(Vector & a, Vector & b, Vector & c);
void vec_generators_direct_product(Vector & a, Vector & b, Vector & c);
void vec_generators_GL_n_q_as_matrices(Vector & gen, INT n, domain *dom, INT f_v);
void vec_generators_GL_n_q_subgroup_as_matrices(Vector & gen, INT n, INT subgroup_index, domain *dom, INT f_v);
void vec_generators_SL_n_q_as_matrices(Vector & gen, INT n, domain *dom, INT f_v);
void vec_generators_frobenius_in_PG(Vector & gen, INT n, domain *dom, INT f_v);
void vec_generators_frobenius_in_AG(Vector & gen, INT n, domain *dom, INT f_v);
void vec_generators_affine_translations(Vector & gen, INT n, domain *dom, INT f_v);
void vec_generators_affine_translations(Vector & gen, INT n, INT q, INT f_v);
void vec_generators_projective_representation(domain *dom, Vector & a, Vector & b, INT f_action_from_right, INT f_modified, INT f_v);
void vec_generators_affine_representation(domain *dom, Vector & a, Vector & b, INT f_v);
void vec_generators_GL_n_q_projective_representation(Vector & gen, INT n, INT q, INT f_special, INT f_frobenius, INT f_modified, INT f_v);
void vec_generators_GL_n_q_affine_representation(Vector & gen, INT n, INT q, INT f_special, INT f_frobenius, INT f_translations, INT f_v);
void vec_generators_GL_n_q_subgroup_affine_representation(Vector & gen, INT n, INT q, INT subgroup_index, 
	INT f_special, INT f_frobenius, INT f_translations, INT f_v);
void kernel_of_homomorphism(Vector & gens, Vector & kernel_gens, 
	void (*hom)(discreta_base & x, discreta_base & image), INT f_v, INT f_vv);
void vec_generators_A5_in_PSL(Vector& G, INT q, INT f_v);
void vec_generators_S4_in_PSL(Vector& G, INT q, INT f_v);
void vec_generators_even_subgroup(Vector & gen, Vector & gen_even_subgroup, INT f_v);
//void vec_generators_on_conjugacy_class_of_subgroups_by_conjugation(perm_group &G, 
	//Vector &LayerOrbit, INT layer, INT orbit, Vector &gens, Vector &induced_gens, INT f_v, INT f_vv);
void vec_generators_restrict_to_subset(Vector & gen, INT first, INT len);
void wreath_embedding(permutation & g, INT n, INT m, permutation & q);
void wreath_embedding_component(permutation & g, INT n, INT m, INT j, permutation & q);
void vec_generators_wreath_product(Vector & G, Vector & H, Vector & W, INT f_v);
void vec_generators_Sn_wreath_Sm(INT n, INT m, Vector & G);
void vec_generators_q1_q2(INT q1, INT q2, Vector & gen, hollerith &label, 
	INT f_write_generators_to_file, INT f_v, INT f_vv);
void vec_generators_q1_q2_aubv(INT q1, INT q2, INT u, INT v, Vector & G, hollerith &label, 
	INT f_write_generators_to_file, INT f_v, INT f_vv);
void vec_generators_q1_q2_au1bv1_au2bv2(INT q1, INT q2, INT u1, INT v1, INT u2, INT v2, 
	Vector & G, hollerith &label, INT f_write_generators_to_file, INT f_v, INT f_vv);
void vec_generators_AGGL1q_subgroup(INT q, INT subgroup_index, 
	INT f_special, INT f_frobenius, INT f_translations, INT f_v);
//void vec_generators_cycle_index(Vector &gen, Vector &C, INT f_v);
void vec_generators_singer_cycle_on_points_of_projective_plane(Vector &gen, INT p, INT f_modified, INT f_v);




class solid: public Vector
{
	public:
	solid();
		// constructor, sets the Vector_pointer to NULL
	void init();
		// initialize trivially all components of solid
	
	Vector& group_generators() { return s_i(0).as_vector(); }
	permutation& group_generators_i(INT i) { return group_generators().s_i(i).as_permutation(); }
	INT& nb_V() { return s_ii(1); }
	INT& nb_E() { return s_ii(2); }
	INT& nb_F() { return s_ii(3); }
	Vector& placement() { return s_i(4).as_vector(); } 	/* of vertex */
	Vector& x() { return placement().s_i(0).as_vector(); }	/* of vertex */
	INT& x_i(INT i) { return x().s_ii(i); }			/* of vertex */
	Vector& y() { return placement().s_i(1).as_vector(); }	/* of vertex */
	INT& y_i(INT i) { return y().s_ii(i); }			/* of vertex */
	Vector& z() { return placement().s_i(2).as_vector(); }	/* of vertex */
	INT& z_i(INT i) { return z().s_ii(i); }			/* of vertex */
	Vector& v1() { return s_i(5).as_vector(); }		/* at edge */
	INT& v1_i(INT i) { return v1().s_ii(i); }		/* at edge */
	Vector& v2() { return s_i(6).as_vector(); }		/* at edge */
	INT& v2_i(INT i) { return v2().s_ii(i); }		/* at edge */
	Vector& f1() { return s_i(7).as_vector(); }		/* at edge */
	INT& f1_i(INT i) { return f1().s_ii(i); }		/* at edge */
	Vector& f2() { return s_i(8).as_vector(); }		/* at edge */
	INT& f2_i(INT i) { return f2().s_ii(i); }		/* at edge */
	Vector& nb_e() { return s_i(9).as_vector(); }		/* at face */
	INT& nb_e_i(INT i) { return nb_e().s_ii(i); }		/* at face */
	Vector& edge() { return s_i(10).as_vector(); }		/* at face */
	Vector& edge_i(INT i) 
		{ return edge().s_i(i).as_vector(); }		/* at face */
	INT& edge_ij(INT i, INT j) 
		{ return edge_i(i).s_ii(j); }			/* at face */
	Vector& neighbour_faces() 
		{ return s_i(11).as_vector(); }			/* at face */
	Vector& neighbour_faces_i(INT i) 
		{ return neighbour_faces().s_i(i).as_vector(); }/* at face */
	INT& neighbour_faces_ij(INT i, INT j) 
		{ return neighbour_faces_i(i).s_ii(j); }	/* at face */
	INT& f_vertex_labels() { return s_ii(12); }
	Vector& vertex_labels() { return s_i(13).as_vector(); } /* of vertex */
	hollerith& vertex_labels_i(INT i) 
		{ return vertex_labels().s_i(i).as_hollerith(); }	/* of vertex */
	Vector& vertex_labels_numeric() { return s_i(14).as_vector(); } /* of vertex */
	INT& vertex_labels_numeric_i(INT i) 
		{ return vertex_labels_numeric().s_ii(i); } /* of vertex */
	INT& f_oriented() { return s_ii(15); }
	
	void init_V(INT nb_V);
		// initialize vertices
	void init_E(INT nb_E);
		// initialize edges
	void init_F(INT nb_F);
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
	void standard_vertex_labels(INT f_start_with_zero);
	void determine_neighbours();
	void find_face(INT e, INT& f1, INT& j1, INT& f2, INT& j2);
	INT find_face_2(INT e1, INT e2);
	INT find_face_by_two_edges(INT e1, INT e2);
	void find_faces_at_edge(INT e, INT& f1, INT& f2);
	INT find_edge(INT v1, INT v2);
	void add_edge(INT v1, INT v2, INT f1, INT f2);
	INT add_edge(INT v1, INT v2);
	INT find_and_add_edge(INT i1, INT i2, INT f_v);
	void add_face3(INT e1, INT e2, INT e3, INT i1, INT i2, INT i3);
	void add_face4(INT i1, INT i2, INT i3, INT i4);
	void add_face5(INT i1, INT i2, INT i3, INT i4, INT i5);
	void add_face_n(Vector& vertices);
	void adjacency_list(INT vertex, INT *adj, INT *nb_adj);
	void center(INT f, Vector& Px, Vector& Py, Vector& Pz);
	void vertices_of_face(INT i, Vector& V);
	void Ratio(INT e, double r, INT& x, INT& y, INT& z);
	INT find_common_face(INT e1, INT e2, INT& f);
	void dual(solid& A);
	void cut_vertices(double r, solid & A);
	void edge_midpoints(solid& A);
	void join_disjoint(solid& A, solid& J, INT f_v);
	void direct_sum(solid& B, solid& J, INT f_v);
	void direct_product(Vector& gen, solid& J, INT f_v);
	void scale(double f);
	void add_central_point(solid& A);
	void induced_action_on_edges(permutation& p, permutation& q);
	void induced_group_on_edges(Vector & gen, Vector & gen_e);
	
	void tetrahedron(INT r);
	void cube(INT r);
	void cube4D(INT r1, INT r2);
	void octahedron(INT r);
	void dodecahedron(INT r);
	void icosahedron(INT r);
	void make_placed_graph(matrix & incma, Vector& aut_gens, Vector& cycles);
		
	void write_graphfile(BYTE *fname);
	void write_solidfile(BYTE *fname);
};
void vec_generators_aut_cube_nd(INT n, Vector &gen);
void number_to_binary(INT n, INT *v, INT digits);
INT binary_to_number(INT *v, INT digits);



// kramer_mesner.C

extern char *discreta_copyright_text;

typedef struct design_data DESIGN_DATA;

struct design_data {
	BYTE *KM_fname;
	INT v, t, k;
	Vector gen;
	Vector MM;
	Vector RR;
	Vector stab_go;
	discreta_base go;
	
	
	INT lambda;
	INT nb_sol;
	Vector S;

	matrix P;
};


void write_KM_file(BYTE *gsel, BYTE *g_label, BYTE *g_label_tex, BYTE *km_fname, BYTE *acting_on, 
	Vector & G_gen, discreta_base & go, INT deg,
	matrix & M, INT t, INT k);
void write_KM_file2(BYTE *gsel, BYTE *g_label, BYTE *g_label_tex, BYTE *km_fname, BYTE *acting_on, 
	Vector & G_gen, discreta_base & go, INT deg,
	matrix & M, INT t, INT k, INT f_right_hand_side_in_last_column_of_M);
void write_ascii_generators(BYTE *km_fname, Vector & gen);
void write_ascii_representatives(BYTE *km_fname, Vector & R);
void write_ascii_stabilizer_orders(BYTE *km_fname, Vector & Ago);
void write_ascii_stabilizer_orders_k_sets(BYTE *km_fname, Vector & Ago, INT k);
void write_ascii_KM_matrices(BYTE *km_fname, Vector & MM);
void km_read_ascii_vtk(BYTE *KM_fname, INT &v, INT &t, INT &k);
void km_read_ascii_strings(BYTE *KM_fname, hollerith& group_construction, hollerith& group_label, hollerith& group_label_tex, hollerith& acting_on);
void km_read_generators(BYTE *KM_fname, Vector & gen);
void km_read_KM_matrices(BYTE *KM_fname, Vector & MM);
void km_read_orbit_representatives(BYTE *KM_fname, Vector & RR);
void km_read_stabilizer_orders(BYTE *KM_fname, Vector & stab_go);
void km_read_orbits_below(BYTE *KM_fname, Vector & Orbits_below1, Vector & Orbits_below2);
void km_read_lambda_values(BYTE *KM_fname, Vector & lambda_values, Vector & lambda_solution_count);
void km_get_solutions_from_solver(BYTE *KM_fname, INT lambda);
INT km_nb_of_solutions(BYTE *KM_fname, INT lambda);
void km_get_solutions(BYTE *KM_fname, INT lambda, INT from, INT len, Vector& S);
void km_read_until_lambdaend(ifstream & f);
void Mtk_via_Mtr_Mrk(INT t, INT r, INT k, matrix & Mtr, matrix & Mrk, matrix & Mtk, INT f_v);
void Mtk_from_MM(Vector & MM, matrix & Mtk, INT t, INT k, INT f_v);

DESIGN_DATA *prepare_for_intersection_numbers(BYTE *KM_fname);
void design_load_all_solutions(DESIGN_DATA *dd, INT lambda);
void design_prepare_orbit_lengths(DESIGN_DATA *dd);
void design_orbits_vector(Vector & X, Vector & orbits, INT f_complement);

void global_intersection_numbers_prepare_data(BYTE *KM_fname, INT lambda, INT s_max, 
	DESIGN_DATA *&dd, matrix& L, matrix& Z, matrix &Bv, matrix & S1t, matrix &D);
void global_intersection_numbers_compute(BYTE *KM_fname, INT lambda, INT s_max, 
	DESIGN_DATA *&dd, matrix& L, matrix& Z, matrix &Bv, matrix & S1t, matrix &D, 
	Vector& sol, matrix& As1, matrix& As2, Vector& inv);
void global_intersection_numbers(BYTE *KM_fname, INT lambda, INT s_max);
void extend_design_from_residual(BYTE *KM_fname, INT lambda);
void get_group_to_file(INT arg_length, char **group_arg_list, INT f_v);
void show_design(BYTE *solid_fname, BYTE *KM_fname, INT lambda, INT m);
void report(BYTE *KM_fname, INT s_max);
void canonical_set_reps(Vector& Set_reps, perm_group &G, INT f_v, INT f_vv, INT f_vvv);
void normalizer_action_on_orbits(perm_group & G, Vector & Reps, 
	Vector &N_gens, Vector &N_gens_induced, INT f_v);
void action_on_orbits_of_normalizing_element(perm_group & G, Vector & Reps,
	permutation & p, permutation & q, INT f_v);
void fuse_orbits(perm_group & G, Vector & Reps, 
	Vector & fusion_map, Vector & new_reps, INT f_v, INT f_vv, INT f_vvv);
void km_compute_KM_matrix(INT arg_length, char **group_arg_list, INT t, INT k, INT f_v, INT f_vv, INT f_vvv);
// interface for ladder:
INT permutation_element_image_of(INT a, void *elt, void *data, INT f_v);
void permutation_element_retrieve(INT hdl, void *elt, void *data, INT f_v);
INT permutation_element_store(void *elt, void *data, INT f_v);
void permutation_element_mult(void *a, void *b, void *ab, void *data, INT f_v);
void permutation_element_invert(void *a, void *av, void *data, INT f_v);
void permutation_element_move(void *a, void *b, void *data, INT f_v);
void permutation_element_dispose(INT hdl, void *data, INT f_v);
void permutation_element_print(void *elt, void *data, ostream &ost);




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
	INT & output_size() { return Vector::s_i(1).as_integer().s_i(); }
	INT & int_vec_first() { return Vector::s_i(2).as_integer().s_i(); }
	INT & int_vec_len() { return Vector::s_i(3).as_integer().s_i(); }
	INT & field1() { return Vector::s_i(4).as_integer().s_i(); }
	INT & field2() { return Vector::s_i(5).as_integer().s_i(); }
	INT & f_ascending() { return Vector::s_i(6).as_integer().s_i(); }
	
	void init(enum bt_key_kind type, INT output_size, INT field1, INT field2);
	void init_INT4(INT field1, INT field2);
	void init_INT2(INT field1, INT field2);
	void init_string(INT output_size, INT field1, INT field2);
	void init_int4_vec(INT field1, INT field2, INT vec_fst, INT vec_len);
	void init_int2_vec(INT field1, INT field2, INT vec_fst, INT vec_len);
};

INT bt_lexicographic_cmp(BYTE *p1, BYTE *p2);
INT bt_key_int_cmp(BYTE *p1, BYTE *p2);
INT bt_key_int2_cmp(BYTE *p1, BYTE *p2);
void bt_key_print_INT4(BYTE **key, ostream& ost);
void bt_key_print_INT2(BYTE **key, ostream& ost);
void bt_key_print(BYTE *key, Vector& V, ostream& ost);
INT bt_key_compare_INT4(BYTE **p_key1, BYTE **p_key2);
INT bt_key_compare_INT2(BYTE **p_key1, BYTE **p_key2);
INT bt_key_compare(BYTE *key1, BYTE *key2, Vector& V, INT depth);
void bt_key_fill_in_INT4(BYTE **p_key, discreta_base& key_op);
void bt_key_fill_in_INT2(BYTE **p_key, discreta_base& key_op);
void bt_key_fill_in_string(BYTE **p_key, INT output_size, discreta_base& key_op);
void bt_key_fill_in(BYTE *key, Vector& V, Vector& the_object);
void bt_key_get_INT4(BYTE **key, INT4 &i);
void bt_key_get_INT2(BYTE **key, INT2 &i);

#define BTREEMAXKEYLEN 24
//#define BTREEMAXKEYLEN 48
//#define BTREEMAXKEYLEN 512

typedef struct keycarrier {
	BYTE c[BTREEMAXKEYLEN];
} KEYCARRIER;

typedef KEYCARRIER KEYTYPE;

typedef struct datatype {
	UINT4 datref;
	UINT4 data_size;
} DATATYPE;

//#define DB_SIZEOF_HEADER 16
//#define DB_SIZEOF_HEADER_LOG 4
#define DB_POS_FILESIZE 4

#define DB_FILE_TYPE_STANDARD 1
#define DB_FILE_TYPE_COMPACT 2

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
	btree & btree_access_i(INT i) { return btree_access().s_i(i).as_btree(); }
	hollerith & filename() { return Vector::s_i(1).as_hollerith(); }
	INT & f_compress() { return Vector::s_i(2).as_integer().s_i(); }
	INT & objectkind() { return Vector::s_i(3).as_integer().s_i(); }
	INT & f_open() { return Vector::s_i(4).as_integer().s_i(); }
	INT & stream() { return Vector::s_i(5).as_integer().s_i(); }
	INT & file_size() { return Vector::s_i(6).as_integer().s_i(); }
	INT & file_type() { return Vector::s_i(7).as_integer().s_i(); }

	void init(const BYTE *filename, INT objectkind, INT f_compress);
	void init_with_file_type(const BYTE *filename, 
		INT objectkind, INT f_compress, INT file_type);
	
	void create(INT verbose_level);
	void open(INT verbose_level);
	void close(INT verbose_level);
	void delete_files();
	void put_file_size();
	void get_file_size();
	void user2total(INT user, INT *total, INT *pad);
	INT size_of_header();
	INT size_of_header_log();
	
	void add_object_return_datref(Vector &the_object, UINT4 &datref, INT verbose_level);
	void add_object(Vector &the_object, INT verbose_level);
	void delete_object(Vector& the_object, UINT4 datref, INT verbose_level);
	void get_object(UINT4 datref, Vector &the_object, INT verbose_level);
	void get_object(DATATYPE *data_type, Vector &the_object, INT verbose_level);
	void get_object_by_unique_INT4(INT btree_idx, 
		INT id, Vector& the_object, INT verbose_level);
	INT get_object_by_unique_INT4_if_there(INT btree_idx, 
		INT id, Vector& the_object, INT verbose_level);
	INT get_highest_INT4(INT btree_idx);
	void ith_object(INT i, INT btree_idx, 
		Vector& the_object, INT verbose_level);
	void ith(INT i, INT btree_idx, 
		KEYTYPE *key_type, DATATYPE *data_type,
		INT verbose_level);
	void print_by_btree(INT btree_idx, ostream& ost);
	void print_by_btree_with_datref(INT btree_idx, ostream& ost);
	void print_subset(Vector& datrefs, ostream& ost);
	void extract_subset(Vector& datrefs, 
		BYTE *out_path, INT verbose_level);
	void search_INT4(INT btree_idx, 
		INT imin, INT imax, Vector &datrefs, 
		INT verbose_level);
	void search_INT4_2dimensional(INT btree_idx0, 
		INT imin0, INT imax0, 
		INT btree_idx1, INT imin1, INT imax1, 
		Vector &datrefs, INT verbose_level);
	void search_INT4_multi_dimensional(Vector& btree_idx, 
		Vector& i_min, Vector &i_max, Vector& datrefs, 
		INT verbose_level);

	INT get_size_from_datref(UINT4 datref, INT verbose_level);
	void add_data_DB(void *d, 
		INT size, UINT4 *datref, INT verbose_level);
	void add_data_DB_standard(void *d, 
		INT size, UINT4 *datref, INT verbose_level);
	void add_data_DB_compact(void *d, 
		INT size, UINT4 *datref, INT verbose_level);
	void free_data_DB(UINT4 datref, INT size, INT verbose_level);

	void file_open(INT verbose_level);
	void file_create(INT verbose_level);
	void file_close(INT verbose_level);
	void file_seek(INT offset);
	void file_write(void *p, INT size, INT nb);
	void file_read(void *p, INT size, INT nb);
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

typedef struct itemtyp {
	KEYTYPE Key;
	DATATYPE Data;
	INT4 Childs; // Anzahl der Nachfolger ueber Ref
	INT4 Ref;
} ItemTyp;

typedef struct pagetyp {
	INT4 AllocRec;
	INT4 NextFreeRec;
	INT4 RootRec;

	INT4 NumItems;
	ItemTyp Item[BTREEMAXPAGESIZE + 1];
/* Item[0]           enthaelt keine Daten, 
 *                   nur Ref/Childs ist verwendet.
 * Item[1..NumItems] fuer Daten und 
 *                   Ref/Childs verwendet. */
} PageTyp;

typedef struct buffer {
	INT4 PageNum;
	INT4 unused;
	PageTyp Page;
	long align;
} Buffer;

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
	
	INT & f_duplicatekeys() { return Vector::s_i(0).as_integer().s_i(); }
	Vector & key() { return Vector::s_i(1).as_vector(); }
	hollerith & filename() { return Vector::s_i(2).as_hollerith(); }
	INT & f_open() { return Vector::s_i(3).as_integer().s_i(); }
	INT & stream() { return Vector::s_i(4).as_integer().s_i(); }
	INT & buf_idx() { return Vector::s_i(5).as_integer().s_i(); }
	INT & Root() { return Vector::s_i(6).as_integer().s_i(); }
	INT & FreeRec() { return Vector::s_i(7).as_integer().s_i(); }
	INT & AllocRec() { return Vector::s_i(8).as_integer().s_i(); }
	INT & btree_idx() { return Vector::s_i(9).as_integer().s_i(); }
	INT & page_table_idx() { return Vector::s_i(10).as_integer().s_i(); }

	void init(const BYTE *file_name, INT f_duplicatekeys, INT btree_idx);
	void add_key_INT4(INT field1, INT field2);
	void add_key_INT2(INT field1, INT field2);
	void add_key_string(INT output_size, INT field1, INT field2);
	void key_fill_in(BYTE *the_key, Vector& the_object);
	void key_print(BYTE *the_key, ostream& ost);

	void create(INT verbose_level);
	void open(INT verbose_level);
	void close(INT verbose_level);

	void ReadInfo(INT verbose_level);
	void WriteInfo(INT verbose_level);
	INT AllocateRec(INT verbose_level);
	void ReleaseRec(INT x);
	void LoadPage(Buffer *BF, INT x, INT verbose_level);
	void SavePage(Buffer *BF, INT verbose_level);

	INT search_string(discreta_base& key_op,
		INT& pos, INT verbose_level);
	void search_interval_INT4(INT i_min, INT i_max, 
		INT& first, INT &len, INT verbose_level);
	void search_interval_INT4_INT4(INT l0, INT u0, 
		INT l1, INT u1, 
		INT& first, INT &len, 
		INT verbose_level);
	void search_interval_INT4_INT4_INT4(INT l0, INT u0, 
		INT l1, INT u1, 
		INT l2, INT u2, 
		INT& first, INT &len, 
		INT verbose_level);
	void search_interval_INT4_INT4_INT4_INT4(INT l0, INT u0, 
		INT l1, INT u1, 
		INT l2, INT u2, 
		INT l3, INT u3, 
		INT& first, INT &len, 
		INT verbose_level);
	INT search_INT4_INT4(INT data1, INT data2, INT &idx, INT verbose_level);
	INT search_unique_INT4(INT i, INT verbose_level);
	INT search_unique_INT4_INT4_INT4_INT4(INT i0, 
		INT i1, INT i2, INT i3, INT verbose_level);
		// returns -1 if an element whose key starts with [i0,i1,i2,i3] could not be found or is not unique.
		// otherwise, the idx of that element is returned
	INT search_datref_of_unique_INT4(INT i, 
		INT verbose_level);
	INT search_datref_of_unique_INT4_if_there(INT i, 
		INT verbose_level);
	INT get_highest_INT4();
	void get_datrefs(INT first, 
		INT len, Vector& datrefs);

	INT search(void *pSearchKey, 
		DATATYPE *pData, INT *idx, INT key_depth, 
		INT verbose_level);
	INT SearchBtree(INT page, 
		void *pSearchKey, DATATYPE *pData, 
		Buffer *Buf, INT *idx, INT key_depth,
		INT verbose_level);
	INT SearchPage(Buffer *buffer, 
		void *pSearchKey, DATATYPE *pSearchData, 
		INT *cur, INT *x, INT key_depth, 
		INT verbose_level);

	INT length(INT verbose_level);
	void ith(INT l, 
		KEYTYPE *key, DATATYPE *data, INT verbose_level);
	INT page_i_th(INT l, 
		Buffer *buffer, INT *cur, INT *i, 
		INT verbose_level);
	
	void insert_key(KEYTYPE *pKey, 
		DATATYPE *pData, 
		INT verbose_level);
	void Update(INT Node, INT *Rise, 
		ItemTyp *RisenItem, 
		INT *RisenNeighbourChilds, 
		INT f_v);
	void Split(Buffer *BF, 
		ItemTyp *Item, INT x, 
		INT *RisenNeighbourChilds, 
		INT verbose_level);

	void delete_ith(INT idx, INT verbose_level);
	void Delete(INT Node, INT& Underflow, INT verbose_level);
	void FindGreatest(INT Node1, 
		INT& Underflow, Buffer *DKBF, INT x, 
		INT verbose_level);
	void Compensate(INT Precedent, 
		INT Node, INT Path, INT& Underflow,
		INT verbose_level);
	
	void print_all(ostream& ost);
	void print_range(INT first, INT len, ostream& ost);
	void print_page(INT x, ostream& ost);
	void page_print(Buffer *BF, ostream& ost);
	void item_print(ItemTyp *item, INT i, ostream& ost);
	
	void file_open();
	void file_create();
	void file_close();
	void file_write(PageTyp *page, const BYTE *message);
	void file_read(PageTyp *page, const BYTE *message);
	void file_seek(INT page_no);
};

#define MAX_FSTREAM_TABLE 1000


extern INT fstream_table_used[MAX_FSTREAM_TABLE];
extern fstream *fstream_table[MAX_FSTREAM_TABLE];

INT fstream_table_get_free_entry();
void database_init(INT verbose_level);
void database_exit(void);
INT root_buf_alloc(void);
void root_buf_free(INT i);



// ##########################################################################################################
// class page_table
// ##########################################################################################################



typedef struct btree_page_registry_key_pair btree_page_registry_key_pair;

struct btree_page_registry_key_pair {
	INT x;
	INT idx;
	INT ref;
};


typedef class page_table page_table;
typedef page_table *ppage_table;

class page_table {
public:
	page_storage *btree_pages;
	INT btree_page_registry_length;
	INT btree_page_registry_allocated_length;
	btree_page_registry_key_pair *btree_table;


	page_table();
	~page_table();
	void init(INT verbose_level);
	void reallocate_table(INT verbose_level);
	void print();
	INT search(INT len, INT btree_idx, INT btree_x, INT &idx);
	INT search_key_pair(INT len, btree_page_registry_key_pair *K, INT &idx);
	void save_page(Buffer *BF, INT buf_idx, INT verbose_level);
	INT load_page(Buffer *BF, INT x, INT buf_idx, INT verbose_level);
	void allocate_rec(Buffer *BF, INT buf_idx, INT x, INT verbose_level);
	void write_pages_to_file(btree *B, INT buf_idx, INT verbose_level);
};




void page_table_init(INT verbose_level);
void page_table_exit(INT verbose_level);
INT page_table_alloc(INT verbose_level);
void page_table_free(INT idx, INT verbose_level);
page_table *page_table_pointer(INT slot);






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
	
	INT & prev() { return Vector::s_i(0).as_integer().s_i(); }
	INT & rule() { return Vector::s_i(1).as_integer().s_i(); }
	hollerith & comment() { return Vector::s_i(2).as_hollerith(); }
	Vector & references() { return Vector::s_i(3).as_vector(); }
	hollerith & references_i(INT i) { return references().s_i(i).as_hollerith(); }

	void init();
	void text(hollerith& h);
	void text2(design_parameter& p, hollerith& h);
	void text012(hollerith& s0, hollerith& s1, hollerith& s2);
	void text012_extended(design_parameter& p, hollerith& s0, hollerith& s1, hollerith& s2);
};

// design.C:
INT design_parameters_admissible(INT v, INT t, INT k, discreta_base &lambda);
INT calc_delta_lambda(INT v, INT t, INT k, INT f_v);
void design_lambda_max(INT t, INT v, INT k, discreta_base & lambda_max);
void design_lambda_max_half(INT t, INT v, INT k, discreta_base & lambda_max_half);
void design_lambda_ijs_matrix(INT t, INT v, INT k, discreta_base& lambda, INT s, matrix & M);
void design_lambda_ijs(INT t, INT v, INT k, discreta_base& lambda, INT s, INT i, INT j, discreta_base & lambda_ijs);
void design_lambda_ij(INT t, INT v, INT k, discreta_base& lambda, INT i, INT j, discreta_base & lambda_ij);
INT is_trivial_clan(INT t, INT v, INT k);
void print_clan_tex_INT(INT t, INT v, INT k);
void print_clan_tex_INT(INT t, INT v, INT k, INT delta_lambda, discreta_base &m_max);
void print_clan_tex(discreta_base &t, discreta_base &v, discreta_base &k, INT delta_lambda, discreta_base &m_max);
INT is_ancestor(INT t, INT v, INT k);
INT is_ancestor(INT t, INT v, INT k, INT delta_lambda);
INT calc_redinv(INT t, INT v, INT k, INT delta_lambda, INT &c, INT &T, INT &V, INT &K, INT &Delta_lambda);
INT calc_derinv(INT t, INT v, INT k, INT delta_lambda, INT &c, INT &T, INT &V, INT &K, INT &Delta_lambda);
INT calc_resinv(INT t, INT v, INT k, INT delta_lambda, INT &c, INT &T, INT &V, INT &K, INT &Delta_lambda);
void design_mendelsohn_coefficient_matrix(INT t, INT m, matrix & M);
void design_mendelsohn_rhs(INT v, INT t, INT k, discreta_base& lambda, INT m, INT s, Vector & rhs);
INT design_parameter_database_already_there(database &D, design_parameter &p, INT& idx);
void design_parameter_database_add_if_new(database &D, design_parameter &p, INT& highest_id, INT verbose_level);
void design_parameter_database_closure(database &D, INT highest_id_already_closed, INT minimal_t, INT verbose_level);
void design_parameter_database_read_design_txt(BYTE *fname_design_txt, BYTE *path_db, INT f_form_closure, INT minimal_t, INT verbose_level);
void design_parameter_database_export_tex(BYTE *path_db);
INT determine_restricted_number_of_designs_t(database &D, btree &B, 
	INT btree_idx_tvkl, INT t, INT first, INT len);
INT determine_restricted_number_of_designs_t_v(database &D, btree &B, 
	INT btree_idx_tvkl, INT t, INT v, INT first, INT len);
void prepare_design_parameters_from_id(database &D, INT id, hollerith& h);
void prepare_link(hollerith& link, INT id);
void design_parameter_database_clans(BYTE *path_db, INT f_html, INT f_v, INT f_vv);
void design_parameter_database_family_report(BYTE *path_db, INT t, INT v, INT k, INT lambda, INT minimal_t);
void design_parameter_database_clan_report(BYTE *path_db, Vector &ancestor, Vector &clan_lambda, Vector & clan_member, Vector & clan_member_path);
INT Maxfit(INT i, INT j);
#if 0
void create_all_masks(char *label, 
	int nb_row_partitions, char *row_partitions[], 
	int nb_col_partitions, char *col_partitions[]);
INT create_masks(char *label, 
	int nb_row_partitions, char *row_partitions[], 
	int nb_col_partitions, char *col_partitions[], 
	int ci, int cj);
#endif
#if 0
void orbits_in_product_action(INT n1, INT n2, INT f_v, INT f_vv);
void orbits_in_product_action_D_CC(INT n1, INT p1, INT p2, INT f_v, INT f_vv);
void orbits_in_product_action_CC_D(INT p1, INT p2, INT n2, INT f_v, INT f_vv);
void orbits_in_product_action_extended(INT q1, INT q2, INT u, INT v, INT f_v, INT f_vv);
void orbits_in_product_action_extended_twice(INT q1, INT q2, INT u1, INT v1, INT u2, INT v2, 
	INT f_cycle_index, INT f_cycle_index_on_pairs, INT f_v, INT f_vv);
void extract_subgroup(INT q1, INT q2, INT u1, INT v1, INT f_cycle_index);
#endif

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
	
	INT & id() { return Vector::s_i(0).as_integer().s_i(); }
	INT & t() { return Vector::s_i(1).as_integer().s_i(); }
	INT & v() { return Vector::s_i(2).as_integer().s_i(); }
	INT & K() { return Vector::s_i(3).as_integer().s_i(); }
	discreta_base & lambda() { return Vector::s_i(4); }
	Vector & source() { return Vector::s_i(5).as_vector(); }
	design_parameter_source & source_i(INT i) { return source().s_i(i).as_design_parameter_source(); }

	void init();
	void init(INT t, INT v, INT k, INT lambda);
	void init(INT t, INT v, INT k, discreta_base& lambda);
	void text(hollerith& h);
	void text_parameter(hollerith& h);
	void reduced_t(design_parameter& p);
	INT increased_t(design_parameter& p);
	void supplementary_reduced_t(design_parameter& p);
	void derived(design_parameter& p);
	INT derived_inverse(design_parameter& p);
	void supplementary_derived(design_parameter& p);
	void residual(design_parameter& p);
	void ancestor(design_parameter& p, Vector & path, INT f_v, INT f_vv);
	void supplementary_residual(design_parameter& p);
	INT residual_inverse(design_parameter& p);
	INT trung_complementary(design_parameter& p);
	INT trung_left_partner(INT& t1, INT& v1, INT& k1, discreta_base& lambda1,
		INT& t_new, INT& v_new, INT& k_new, discreta_base& lambda_new);
	INT trung_right_partner(INT& t1, INT& v1, INT& k1, discreta_base& lambda1,
		INT& t_new, INT& v_new, INT& k_new, discreta_base& lambda_new);
	INT alltop(design_parameter& p);
	void complementary(design_parameter& p);
	void supplementary(design_parameter& p);
	INT is_selfsupplementary();
	void lambda_of_supplementary(discreta_base& lambda_supplementary);
	
	void init_database(database& D, BYTE *path);
};


// counting.C:
void cycle_index_perm_group(perm_group &G, Vector &C, INT f_v, INT f_vv);
void cycle_type_add_monomial(Vector &C, Vector &m, discreta_base &coeff);
void cycle_index_Zn(Vector &C, INT n);
void cycle_index_elementary_abelian(Vector &C, INT p, INT f);
void make_monomial_of_equal_cycles(INT i, INT e, Vector &m);
void cycle_index_q1_q2_dot_aubv_product_action(INT q1, INT q2, INT u, INT v, Vector &C);
void cycle_index_direct_product(Vector &CG, Vector &CH, Vector &CGH, INT f_v);
void cycle_index_monomial_direct_product(Vector &m1, Vector &m2, Vector &m3);
void cycle_index_on_pairs(Vector &CG, Vector &CG2, INT f_v);
void cycle_index_number_of_orbits(Vector &CG, discreta_base &number_of_orbits);
void cycle_index_number_of_orbits_on_mappings(Vector &CG, INT k, discreta_base &number_of_orbits);
void print_cycle_type(Vector &C);

// orbit.C:
void all_orbits(INT nb_elements, Vector &generators, 
	Vector &orbit_no, 
	INT f_schreier_vectors, Vector &schreier_last, Vector &schreier_label, 
	Vector &orbit_elts, Vector &orbit_elts_inv, 
	Vector &orbit_first, Vector &orbit_length, 
	actionkind k, discreta_base &action_data, INT f_v, INT f_vv, INT f_v_image_of);
INT image_of_using_integers(INT elt, INT gen, actionkind k, discreta_base &action_data, INT f_v);
void orbit_of_element(Vector &gens, discreta_base &elt, Vector &orbit,
	INT f_schreier_data, Vector &schreier_prev, Vector &schreier_label, 
	actionkind k, discreta_base &action_data, INT f_v, INT f_vv);
void image_of(discreta_base &elt, discreta_base &image, discreta_base &g, actionkind k, discreta_base &action_data);
void induced_permutations_on_orbit(Vector &gens, Vector &induced_gens, Vector &orbit, actionkind k, discreta_base &action_data);
void induced_permutation_on_orbit(discreta_base &gen, permutation &induced_gen, Vector &orbit, actionkind k, discreta_base &action_data);
void trace_schreier_vector(INT i, Vector &gen, 
	Vector &schreier_prev, Vector &schreier_label, permutation &p);
void transversal_for_orbit(Vector &gen, 
	Vector &schreier_prev, Vector &schreier_label, Vector &T);
void trace_and_multiply(Vector &T, Vector &gen, 
	Vector &schreier_prev, Vector &schreier_label, INT i);
void compute_transversal(Vector &gens, INT discreta_base_pt, Vector &T, INT f_v, INT f_vv);
void allocate_orbit_on_pairs_data(INT v, INT *&orbits_on_pairs);
void calc_orbits_on_pairs(Vector & gens, INT *&orbits_on_pairs, INT &nb_orbits, 
	Vector & orbit_first_i, Vector & orbit_first_j, Vector & orbit_length, int f_v, int f_vv);
void write_orbits_on_pairs_to_file(BYTE *group_label, INT nb_points, 
	Vector &orbit_first_i, Vector &orbit_first_j, Vector &orbit_length, 
	INT *orbits_on_pairs);
void prepare_2_orbits_in_product_action(char *group_label, 
	Vector &gen, INT d, INT c, INT f_v, INT f_vv);


// discreta_global.C:
void free_global_data();
void the_end(INT t0);
void the_end_quietly(INT t0);
void calc_Kramer_Mesner_matrix_neighboring(generator *gen, 
	INT level, matrix &M, INT verbose_level);
// we assume that we don't use implicit fusion nodes
void Mtk_from_MM(Vector & MM, matrix & Mtk, INT t, INT k, 
	INT f_subspaces, INT q,  INT verbose_level);
void Mtk_via_Mtr_Mrk(INT t, INT r, INT k, INT f_subspaces, INT q, 
	matrix & Mtr, matrix & Mrk, matrix & Mtk, INT verbose_level);
// Computes $M_{tk}$ via a recursion formula:
// $M_{tk} = {{k - t} \choose {k - r}} \cdot M_{t,r} \cdot M_{r,k}$.
void Mtk_sup_to_inf(generator *gen, 
	INT t, INT k, matrix & Mtk_sup, matrix & Mtk_inf, INT verbose_level);
void compute_Kramer_Mesner_matrix(generator *gen, 
	INT t, INT k, matrix &M, INT f_subspaces, INT q, INT verbose_level);
void matrix_to_diophant(matrix& M, diophant *&D, INT verbose_level);









