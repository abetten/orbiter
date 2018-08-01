// coding_theory.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


// #############################################################################
// mindist.C:
// #############################################################################

int mindist(int n, int k, int q, int *G, 
	int f_v, int f_vv, int idx_zero, int idx_one, 
	INT *add_table, INT *mult_table);
//Main routine for the code minimum distance computation.
//The tables are only needed if $q = p^f$ with $f > 1$. 
//In the GF(p) case, just pass a NULL pointer. 

// #############################################################################
// tensor.C:
// #############################################################################

void twisted_tensor_product_codes(
	INT *&H_subfield, INT &m, INT &n, 
	finite_field *F, finite_field *f, 
	INT f_construction_A, INT f_hyperoval, 
	INT f_construction_B, INT verbose_level);
void create_matrix_M(
	INT *&M, 
	finite_field *F, finite_field *f,
	INT &m, INT &n, INT &beta, INT &r, INT *exponents, 
	INT f_construction_A, INT f_hyperoval, INT f_construction_B, 
	INT f_elements_exponential, const BYTE *symbol_for_print, 
	INT verbose_level);
	// INT exponents[9]
void create_matrix_H_subfield(finite_field *F, finite_field*f, 
	INT *H_subfield, INT *C, INT *C_inv, INT *M, INT m, INT n, 
	INT beta, INT beta_q, 
	INT f_elements_exponential, const BYTE *symbol_for_print, 
	const BYTE *symbol_for_print_subfield, 
	INT f_construction_A, INT f_hyperoval, INT f_construction_B, 
	INT verbose_level);
void tt_field_reduction(finite_field &F, finite_field &f, 
	INT m, INT n, INT *M, INT *MM, INT verbose_level);


void make_tensor_code_9dimensional_as_point_set(finite_field *F, 
	INT *&the_set, INT &length, 
	INT verbose_level);
void make_tensor_code_9_dimensional(INT q, 
	const BYTE *override_poly_Q, const BYTE *override_poly, 
	INT f_hyperoval, 
	INT *&code, INT &length, 
	INT verbose_level);



