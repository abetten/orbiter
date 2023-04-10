/*
 * geometry_builder.h
 *
 *  Created on: Aug 24, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_GEOMETRY_BUILDER_GEOMETRY_BUILDER_H_
#define SRC_LIB_FOUNDATIONS_GEOMETRY_BUILDER_GEOMETRY_BUILDER_H_


namespace orbiter {
namespace layer1_foundations {
namespace geometry_builder {


#define COLOR_RED 2
#define COLOR_GREEN 3

// pick color codes from the list below:
// the list is taken from void mp_graphics::color_tikz(ofstream &fp, int color)
// line 2600

#if 0
if (color == 0)
	fp << "white";
else if (color == 1)
	fp << "black";
else if (color == 2)
	fp << "red";
else if (color == 3)
	fp << "green";
else if (color == 4)
	fp << "blue";
else if (color == 5)
	fp << "cyan";
else if (color == 6)
	fp << "magenta";
else if (color == 7)
	fp << "pink";
else if (color == 8)
	fp << "orange";
else if (color == 9)
	fp << "lightgray";
else if (color == 10)
	fp << "brown";
else if (color == 11)
	fp << "lime";
else if (color == 12)
	fp << "olive";
else if (color == 13)
	fp << "gray";
else if (color == 14)
	fp << "purple";
else if (color == 15)
	fp << "teal";
else if (color == 16)
	fp << "violet";
else if (color == 17)
	fp << "darkgray";
else if (color == 18)
	fp << "lightgray";
else if (color == 19)
	fp << "yellow";
else if (color == 20)
	fp << "green!50!red";
else if (color == 21)
	fp << "violet!50!red";
else if (color == 22)
	fp << "cyan!50!red";
else if (color == 23)
	fp << "green!50!blue";
else if (color == 24)
	fp << "brown!50!red";
else if (color == 25)
	fp << "purple!50!red";
else {
#endif






// #############################################################################
// cperm.cpp
// #############################################################################

//! a permutation for use in class gen_geo


class cperm {

public:
	int l;
	int *data;
		// a permutation of { 0, 1 ... l - 1 }

	cperm();
	~cperm();
	void init_and_identity(int l);
	void free();
	void move_to(cperm *q);
	void identity();
	void mult(cperm *b, cperm *c);
	void inverse(cperm *b);
	void power(cperm *res, int exp);
	void print();
	void mult_apply_forwc_r(int i, int l);
	/* a := a (i i+1 ... i+l-1). */
	void mult_apply_tau_r(int i, int j);
	/* a := a (i j). */
	void mult_apply_tau_l(int i, int j);
	/* a := (i j) a. */
	void mult_apply_backwc_l(int i, int l);
	/* a := (i+l-1 i+l-2 ... i+1 i) a. */

};







// #############################################################################
// decomposition_with_fuse.cpp
// #############################################################################

//! a row-tactical decomposition with fuse, to be used by the geometry_builder

class decomposition_with_fuse {

public:

	gen_geo *gg;

	int nb_fuse;
	int *Fuse_first; // [nb_fuse]
	int *Fuse_len; // [nb_fuse]

	int *K0; // [gg->GB->v_len * gg->GB->b_len]
	int *KK; // [gg->GB->v_len * gg->GB->b_len]
	int *K1; // [gg->GB->v_len * gg->GB->b_len]
	int *F_last_k_in_col; // [gg->GB->v_len * gg->GB->b_len]


	gen_geo_conf *Conf; //[gg->GB->v_len * gg->GB->b_len]

	// partition for Nauty:
	int *row_partition;
		// row partition: 1111011110...
		// where the 0's indicate the end of a block
		// The blocks are defined by the initial TDO decomposition.
	int *col_partition;
		// likewise, but for columns
		// The blocks are defined by the initial TDO decomposition.
	int **Partition;
		// [gg->GB->V + 1]
		// combination of row and column partition,
		// but with only i rows, so that it can be used
		// for computing the canonical form of the partial geometry
		// consisting of the first i rows only
	int **Partition_fixing_last;


	decomposition_with_fuse();
	~decomposition_with_fuse();
	gen_geo_conf *get_conf_IJ(
			int I, int J);
	void init(
			gen_geo *gg, int verbose_level);
	void TDO_init(
			int *v, int *b, int *theTDO, int verbose_level);
	void init_tdo_line(
			int fuse_idx,
			int tdo_line, int v, int *b, int *r, int verbose_level);
	void print_conf();
	void init_fuse(int verbose_level);
	void init_k(int verbose_level);
	void conf_init_last_non_zero_flag(int verbose_level);
	void init_partition(int verbose_level);


};



// #############################################################################
// gen_geo_conf.cpp
// #############################################################################

//! description of a configuration which is part of class decomposition_with_fuse


class gen_geo_conf {

public:
	int fuse_idx;

	int v;
	int b;
	int r;

	int r0;
	int i0;
	int j0;
	int f_last_non_zero_in_fuse;
		// only valid if J=0,
		// that is, for those in the first column

	gen_geo_conf();
	~gen_geo_conf();
	void print(std::ostream &ost);

};

// #############################################################################
// gen_geo.cpp
// #############################################################################

//! classification of geometries with a given row-tactical decomposition


class gen_geo {

public:

	geometry_builder *GB;

	decomposition_with_fuse *Decomposition_with_fuse;

	incidence *inc;

	int forget_ivhbar_in_last_isot;

	std::string inc_file_name;

	// record the search tree in text files for later processing:
	std::string fname_search_tree;
	std::ofstream *ost_search_tree;
	std::string fname_search_tree_flags;
	std::ofstream *ost_search_tree_flags;

	girth_test *Girth_test;

	test_semicanonical *Test_semicanonical;

	geometric_backtrack_search *Geometric_backtrack_search;

	gen_geo();
	~gen_geo();
	void init(geometry_builder *GB, int verbose_level);
	void init_semicanonical(int verbose_level);
	void print_pairs(int line);
	void main2(int verbose_level);
	void generate_all(int verbose_level);
	void setup_output_files(int verbose_level);
	void close_output_files(int verbose_level);
	void record_tree(int i1, int f_already_there);
	void print_I_m(int I, int m);
	void print(int v);
	void increment_pairs_point(int i1, int col, int k);
	void decrement_pairs_point(int i1, int col, int k);
	void girth_test_add_incidence(int i, int j_idx, int j);
	void girth_test_delete_incidence(int i, int j_idx, int j);
	void girth_Floyd(int i, int verbose_level);
	int check_girth_condition(int i, int j_idx, int j, int verbose_level);
	int apply_tests(int I, int m, int J, int n, int j, int verbose_level);
	void print(std::ostream &ost, int v, int v_cut);
	void print_override_theX(std::ostream &ost,
			int *theX, int v, int v_cut);

};

// #############################################################################
// geometric_backtrack_search.cpp
// #############################################################################

//! classification of geometries with a given row-tactical decomposition


class geometric_backtrack_search {

public:

	gen_geo *gg;

	iso_type **Row_stabilizer_orbits;
	int *Row_stabilizer_orbit_idx;

	geometric_backtrack_search();
	~geometric_backtrack_search();
	void init(gen_geo *gg, int verbose_level);

	int First(int verbose_level);
	int Next(int verbose_level);
	int BlockFirst(int I, int verbose_level);
	int BlockNext(int I, int verbose_level);
	int RowFirstSplit(int I, int m, int verbose_level);
	int RowNextSplit(int I, int m, int verbose_level);
	int geo_back_test(int I, int verbose_level);
	int RowFirst0(int I, int m, int verbose_level);
	int RowNext0(int I, int m, int verbose_level);
	int RowFirst(int I, int m, int verbose_level);
	int RowNext(int I, int m, int verbose_level);
	int RowFirstLexLeast(int I, int m, int verbose_level);
	int RowNextLexLeast(int I, int m, int verbose_level);
	int RowFirstOrderly(int I, int m, int verbose_level);
	void place_row(int I, int m, int idx, int verbose_level);
	int RowNextOrderly(int I, int m, int verbose_level);
	void RowClear(int I, int m);
	int ConfFirst(int I, int m, int J, int verbose_level);
	int ConfNext(int I, int m, int J, int verbose_level);
	void ConfClear(int I, int m, int J);
	int XFirst(int I, int m, int J, int n, int verbose_level);
	int XNext(int I, int m, int J, int n, int verbose_level);
	void XClear(int I, int m, int J, int n);
	int X_First(int I, int m, int J, int n, int j, int verbose_level);
	int TryToPlace(int I, int m, int J, int n, int j, int verbose_level);

};



// #############################################################################
// geometry_builder_description.cpp
// #############################################################################

//! description of a geometry


class geometry_builder_description {
public:

	int f_V;
	std::string V_text;
	int f_B;
	std::string B_text;
	int f_TDO;
	std::string TDO_text;
	int f_fuse;
	std::string fuse_text;

	int f_girth_test;
	int girth;

	// f_lambda and f_find_square are mutually exclusive!

	int f_lambda;
	int lambda;

	int f_find_square;
	int f_simple;

	int f_search_tree;
	int f_search_tree_flags;

	int f_orderly;
	int f_special_test_not_orderly;

	std::vector<std::string> test_lines;

	std::vector<std::string> test2_lines;

	int f_split;
	int split_line;
	int split_remainder;
	int split_modulo;

	std::vector<int> print_at_line;

	int f_fname_GEO;
	std::string fname_GEO;

	int f_output_to_inc_file;
	int f_output_to_sage_file;
	int f_output_to_blocks_file;
	int f_output_to_blocks_latex_file;

	geometry_builder_description();
	~geometry_builder_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// geometry_builder.cpp
// #############################################################################

//! classification of geometries





class geometry_builder {

public:

	geometry_builder_description *Descr;


	// the row partition:
	int *v;
	int v_len;

	// the column partition:
	int *b;
	int b_len;

	// a coarse grain partition of the row partition
	int *fuse;
	int fuse_len;

	// the structure constants (# of incidences in a row)
	int *TDO;
	int TDO_len;


	int V;
		// = sum(i = 0; i < v_len; i++) v[i]
	int B;
		// = sum(i= 0; i < b_len; i++) b[i]
	int *V_partition; // [V + 1]

	int *R; // [V]


	int f_transpose_it;
	int f_save_file;
	std::string fname;

	std::string control_file_name;
	int no;
	int flag_numeric;
	int f_no_inc_files;
	gen_geo *gg;

	geometry_builder();
	~geometry_builder();
	void init_description(
			geometry_builder_description *Descr,
			int verbose_level);
	void compute_VBR(int verbose_level);
	void print_tdo();
	void isot(int line, int verbose_level);
	void isot_no_vhbars(int verbose_level);
	void isot2(int line, int verbose_level);
	void set_split(int line, int remainder, int modulo);

};





// #############################################################################
// girth_test.cpp
// #############################################################################

//! classification of geometries





class girth_test {

public:

	gen_geo *gg;

	int girth;
	int V; // = gg->GB->V

	int **S; // [V][V * V]
	int **D; // [V][V * V]

	girth_test();
	~girth_test();
	void init(gen_geo *gg, int girth, int verbose_level);
	void Floyd(int row, int verbose_level);
	void add_incidence(int i, int j_idx, int j);
	void delete_incidence(int i, int j_idx, int j);
	int check_girth_condition(
			int i, int j_idx, int j, int verbose_level);
	void print_Si(int i);
	void print_Di(int i);

};




// #############################################################################
// inc_encoding.cpp
// #############################################################################

//! row-by-row encoding of an incidence geometry

class inc_encoding {

public:
	int *theX; // [v * dim_n]
	int dim_n;
	int v; // # of rows
	int b; // # of columns
	int *R; // [v]
		// R[i] is the number of incidences in row i

	inc_encoding();
	~inc_encoding();
	int &theX_ir(int i, int r);
	void init(
			int v, int b, int *R, int verbose_level);
	long int rank_row(int row);
	void get_flags(
			int row, std::vector<int> &flags);
	int find_square(int m, int n);
	void print_horizontal_bar(
		std::ostream &ost,
		gen_geo *gg, int f_print_isot, iso_type *it);
	void print_partitioned(
			std::ostream &ost, int v_cur, int v_cut,
			gen_geo *gg, int f_print_isot);
	void print_partitioned_override_theX(
			std::ostream &ost, int v_cur, int v_cut,
			gen_geo *gg, int *the_X, int f_print_isot);
	void print_permuted(
			cperm *pv, cperm *qv);
	void apply_permutation(
			incidence *inc, int v,
		int *theY, cperm *p, cperm *q, int verbose_level);


};





// #############################################################################
// incidence.cpp
// #############################################################################

//! encoding of an incidence geometry during classification


class incidence {

public:

	gen_geo *gg;
	inc_encoding *Encoding;

	int *K; //[gg->GB->B]
		// K[j] is the current sum of incidences in column j

	int **theY; //[gg->GB->B][gg->GB->V];

	int **pairs;
		//[gg->GB->V][];
		// pairs[i][i1]
		// is the number of blocks containing {i1,i}
		// where 0 \le i1 < i.



	int gl_nb_GEN;

	iso_type **iso_type_at_line; // [gg->GB->V]
	iso_type *iso_type_no_vhbars;

	int back_to_line;


	incidence();
	~incidence();
	void init(gen_geo *gg,
			int v, int b, int *R, int verbose_level);
	void init_pairs(int verbose_level);
	void print_pairs(int v);
	int find_square(int m, int n);
	void print_param();
	void free_isot();
	void print_R(int v, cperm *p, cperm *q);
	void install_isomorphism_test_after_a_given_row(
			int i,
			int f_orderly, int verbose_level);
	void install_isomorphism_test_of_second_kind_after_a_given_row(
			int i,
			int f_orderly, int verbose_level);
	void set_split(int row, int remainder, int modulo);
	void print_geo(
			std::ostream &ost, int v, int *theGEO);
	void print_inc(
			std::ostream &ost, int v, long int *theInc);
	void print_sage(
			std::ostream &ost, int v, long int *theInc);
	void print_blocks(
			std::ostream &ost, int v, long int *theInc);
	void compute_blocks(
			long int *&Blocks, int *&K, int v, long int *theInc);
	void compute_blocks_ranked(
			long int *&Blocks, int v, long int *theInc);
	int compute_k(int v, long int *theInc);
	int is_block_tactical(int v, long int *theInc);
	void geo_to_inc(
			int v, int *theGEO, long int *theInc, int nb_flags);
	void inc_to_geo(
			int v, long int *theInc, int *theGEO, int nb_flags);


};





// #############################################################################
// iso_type.cpp
// #############################################################################

//! classification of geometries based on canonical forms

class iso_type {

public:

	gen_geo *gg;

	int v;
	int sum_R;
	int sum_R_before;

	int f_orderly;

	// test of the first or the second kind:
	// (second kind means we only check those geometries
	// that are completely realizable
	int f_generate_first;
	int f_beginning_checked;

	int f_split;
	int split_remainder;
	int split_modulo;


	std::string fname;

	data_structures::classify_using_canonical_forms *Canonical_forms;

	int f_print_mod;
	int print_mod;

	iso_type();
	~iso_type();
	void init(gen_geo *gg, int v,
			int f_orderly, int verbose_level);
	void add_geometry(
		inc_encoding *Encoding,
		int f_partition_fixing_last,
		int &f_already_there,
		int verbose_level);
	void find_and_add_geo(
		int *theY,
		int f_partition_fixing_last,
		int &f_new_object, int verbose_level);
	void second();
	void set_split(int remainder, int modulo);
	void print_geos(int verbose_level);
	void write_inc_file(std::string &fname, int verbose_level);
	void write_sage_file(std::string &fname, int verbose_level);
	void write_blocks_file(std::string &fname, int verbose_level);
	void write_blocks_file_long(std::string &fname, int verbose_level);
	void print_GEO(int *pc, int v, incidence *inc);
	void print_status(std::ostream &ost, int f_with_flags);
	void print_flags(std::ostream &ost);
	void print_geometry(
			inc_encoding *Encoding, int v, incidence *inc);

};





// #############################################################################
// test_semicanonical.cpp
// #############################################################################

//! classification of geometries





class test_semicanonical {

public:

	gen_geo *gg;

	int MAX_V;

	// initial vertical and horizontal bars
	// to create semi-canonical partial geometries
	int nb_i_vbar;
	int *i_vbar;
	int nb_i_hbar;
	int *i_hbar;


	int *f_vbar; // [gg->GB->V * gg->inc->Encoding->dim_n]
	int *vbar; // [gg->GB->V]
	int *hbar; // [gg->GB->B]

	test_semicanonical();
	~test_semicanonical();
	void init(
			gen_geo *gg, int MAX_V, int verbose_level);
	void init_bars(int verbose_level);
	void print();
	void markers_update(
			int I, int m, int J, int n, int j,
			int i1, int j0, int r,
			int verbose_level);
	void marker_move_on(
			int I, int m, int J, int n, int j,
			int i1, int j0, int r,
			int verbose_level);
	int row_starter(
			int I, int m, int J, int n,
			int i1, int j0, int r,
			int verbose_level);
	void row_init(int I, int m, int J,
			int i1,
			int verbose_level);
	int col_marker_test(
			int j0, int j, int i1);
	void col_marker_remove(
			int I, int m, int J, int n,
			int i1, int j0, int r, int old_x);
	void row_test_continue(
			int I, int m, int J, int i1);

};







}}}



#endif /* SRC_LIB_FOUNDATIONS_GEOMETRY_BUILDER_GEOMETRY_BUILDER_H_ */
