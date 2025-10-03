/*
 * design_theory.h
 *
 *  Created on: Feb 15, 2025
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER1_FOUNDATIONS_COMBINATORICS_DESIGN_THEORY_DESIGN_THEORY_H_
#define SRC_LIB_LAYER1_FOUNDATIONS_COMBINATORICS_DESIGN_THEORY_DESIGN_THEORY_H_



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace design_theory {


// #############################################################################
// design_object.cpp:
// #############################################################################

//! representation of a design without information about the group


class design_object {
public:


	std::string prefix;
	std::string label_txt;
	std::string label_tex;

	int q;
	algebra::field_theory::finite_field *F;

	int k;


	int f_has_set;
	long int *set; // [sz]
		// The subsets are coded as ranks of k-subsets.
	int sz; // = b, the number of blocks


	int f_has_block_partition;
	int block_partition_class_size;

	int *block; // [k]

	int v;
	int b;
	int nb_inc;
	int f_has_incma;
	int *incma; // [v * b]

	// opaque pointer:

	void *DC; // apps_combinatorics::design_create *DC

	design_object();
	~design_object();
	void compute_incidence_matrix_from_blocks(
			int *blocks, int nb_blocks, int k, int verbose_level);
	void compute_blocks_from_incidence_matrix(
			long int *&blocks, int &nb_blocks, int &block_sz,
			int verbose_level);
	void make_Baker_elliptic_semiplane_1978(
			int verbose_level);
	void make_Mathon_elliptic_semiplane_1987(
			int verbose_level);
	void make_design_from_incidence_matrix(
			std::string &label, int verbose_level);
	void do_export_flags(
			int verbose_level);
	void do_export_incidence_matrix_csv(
			int verbose_level);
	void do_export_incidence_matrix_latex(
			other::graphics::draw_incidence_structure_description *Draw_incidence_structure_description,
			int verbose_level);
	void do_intersection_matrix(
			int f_save,
			int verbose_level);
	void do_export_blocks(
			int verbose_level);
	void do_row_sums(
			int verbose_level);
	void do_tactical_decomposition(
			int verbose_level);


};


// #############################################################################
// design_theory_global.cpp:
// #############################################################################

//! anything related to design theory


class design_theory_global {
public:

	design_theory_global();
	~design_theory_global();
	void make_Baker_elliptic_semiplane_1978_incma(
			int *&Inc, int &v, int &b,
			int verbose_level);
	void make_Mathon_elliptic_semiplane_1987_incma(
			int *&Inc, int &V, int &B,
			int verbose_level);
	void make_design_from_incidence_matrix(
		int *&Inc, int &v, int &b, int &k,
		std::string &label,
		int verbose_level);

	void compute_incidence_matrix(
			int v, int b, int k, long int *Blocks_coded,
			int *&M, int verbose_level);
	void compute_incidence_matrix_from_blocks(
			int v, int b, int k, int *Blocks,
			int *&M, int verbose_level);
	void compute_incidence_matrix_from_blocks_lint(
			int v, int b, int k, long int *Blocks,
			int *&M, int verbose_level);
	void compute_incidence_matrix_from_sets(
			int v, int b, long int *Sets_coded,
			int *&M,
			int verbose_level);
	void compute_blocks_from_coding(
			int v, int b, int k, long int *Blocks_coded,
			int *&Blocks, int verbose_level);
	void compute_blocks_from_incma(
			int v, int b, int k, int *incma,
			int *&Blocks, int verbose_level);
	void create_incidence_matrix_of_graph(
			int *Adj, int n,
			int *&M, int &nb_rows, int &nb_cols,
			int verbose_level);


	void create_wreath_product_design(
			int n, int k,
			long int *&Blocks, long int &nb_blocks,
			int verbose_level);
	void create_linear_space_from_latin_square(
			int *Mtx, int s,
			int &v, int &k,
			long int *&Blocks, long int &nb_blocks,
			int verbose_level);
	void report_large_set(
			std::ostream &ost, long int *coding, int nb_designs,
			int design_v, int design_k, int design_sz, int verbose_level);
	void report_large_set_compact(
			std::ostream &ost, long int *coding, int nb_designs,
			int design_v, int design_k, int design_sz, int verbose_level);

};



// #############################################################################
// incidence_structure_by_flags.cpp:
// #############################################################################

//! representation of an incidence structure


class incidence_structure_by_flags {
public:


	int *flags;
	int nb_flags;
	int nb_rows;
	int nb_cols;


	incidence_structure_by_flags();
	~incidence_structure_by_flags();
	void init(
			int *flags, int nb_flags, int nb_rows, int nb_cols,
			int verbose_level);
	void print();
	void print_latex(
			std::ostream &ost);
	void print_incma_latex(
			std::ostream &ost);

};


}}}}



#endif /* SRC_LIB_LAYER1_FOUNDATIONS_COMBINATORICS_DESIGN_THEORY_DESIGN_THEORY_H_ */
