/*
 * geometry_builder.h
 *
 *  Created on: Aug 24, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_GEOMETRY_BUILDER_GEOMETRY_BUILDER_H_
#define SRC_LIB_FOUNDATIONS_GEOMETRY_BUILDER_GEOMETRY_BUILDER_H_


namespace orbiter {
namespace foundations {






#define MAX_V 300
#define MAX_B 300
#define MAX_VB 100   /* MAX(MAX_V, MAX_B) */
#define MAX_B2 6400    /* B ueber 2 */
#define MAX_R 80

#define MAX_GRID 100
#define MAX_TYPE 200
	/* at least 2 * MAX_VB + 1 */






// #############################################################################
// cperm.cpp
// #############################################################################

//! a permutation for use in class gen_geo


class cperm {

public:
	int l;
	int *data;
		/* a permutation of
		 * { 0, 1 ... l - 1 } */

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
// gen_geo_conf.cpp
// #############################################################################

//! description of a configuration which is part of the description of the geometry for class gen_geo


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

//! classification of geometries given a row-tactical decomposition


class gen_geo {

public:

	geometry_builder *GB;

	int nb_fuse;
	int *Fuse_first;
	int *Fuse_len;
	int *K0;
	int *KK;
	int *K1;
	int *F_last_k_in_col;


	gen_geo_conf *Conf; //[GB->v_len * GB->b_len];

	incidence *inc;

	int *K; //[GB->B];

	int *f_vbar; // [GB->V * inc->Encoding->dim_n]
	int *vbar; // [GB->V]
	int *hbar; // [GB->B]

	int f_do_iso_test;
	int f_do_aut_group;
	int f_do_aut_group_in_iso_type_without_vhbars;
	int forget_ivhbar_in_last_isot;
	int gen_print_intervall;

	std::string inc_file_name;


	gen_geo();
	~gen_geo();
	void init(geometry_builder *GB,
		int f_do_iso_test,
		int f_do_aut_group,
		int f_do_aut_group_in_iso_type_without_vhbars,
		int gen_print_intervall,
		int verbose_level);
	void TDO_init(int *v, int *b, int *theTDO, int verbose_level);
	void init_tdo_line(int fuse_idx, int tdo_line, int v, int *b, int *r, int verbose_level);
	void print_conf();
	void init_bars(int verbose_level);
	void init_fuse(int verbose_level);
	void init_k();
	void conf_init_last_non_zero_flag();
	void print_pairs(int line);
	void main2(int &nb_GEN, int &nb_GEO, int &ticks, int &tps, int verbose_level);
	void generate_all(int verbose_level);
	void print_I_m(int I, int m);
	void print(int v);
	int GeoFst(int verbose_level);
	int GeoNxt(int verbose_level);
	int GeoRowFst(int I, int verbose_level);
	int GeoRowNxt(int I, int verbose_level);
	int GeoLineFstRange(int I, int m, int verbose_level);
	int GeoLineNxtRange(int I, int m, int verbose_level);
	int geo_back_test(int I, int verbose_level);
	int GeoLineFst0(int I, int m, int verbose_level);
	int GeoLineNxt0(int I, int m, int verbose_level);
	int GeoLineFst(int I, int m);
	int GeoLineNxt(int I, int m);
	void GeoLineClear(int I, int m);
	int GeoConfFst(int I, int m, int J);
	int GeoConfNxt(int I, int m, int J);
	void GeoConfClear(int I, int m, int J);
	int GeoXFst(int I, int m, int J, int n);
	int GeoXNxt(int I, int m, int J, int n);
	void GeoXClear(int I, int m, int J, int n);
	int X_Fst(int I, int m, int J, int n, int j);

};


// #############################################################################
// geo_frame.cpp
// #############################################################################

//! partition of a geometry


class geo_frame {
public:
	int G_max;
	int first[MAX_GRID + 1];
	int len[MAX_GRID];
	int grid_entry[MAX_GRID];

	geo_frame();
	~geo_frame();

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

	std::vector<std::string> test_lines;
	std::vector<std::string> test_flags;

	std::vector<std::string> test2_lines;
	std::vector<std::string> test2_flags;

	std::vector<int> print_at_line;

	int f_fname_GEO;
	std::string fname_GEO;

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
	void init_description(geometry_builder_description *Descr,
			int verbose_level);
	void compute_VBR(int verbose_level);
	void print_tdo();
	void isot(int line, int tdo_flags, int verbose_level);
	void isot_no_vhbars(int tdo_flags, int verbose_level);
	void isot2(int line, int tdo_flags, int verbose_level);
	void range(int line, int first, int len);
	void flush_line(int line);

};






// globals.cpp:
void inc_transpose(int *R,
	int *theX, int f_full, int max_r,
	int v, int b,
	int **theY, int *theYdim_n, int **R_new);
int tuple_cmp(int *a, int *b, int l);
void print_theX(int *theX, int dim_n, int v, int b, int *R);
void print_theX_pq(
	int *theX, int dim_n, int v, int b, int *R, cperm *pv, cperm *qv);

void cperm_test(void);

void frame2grid(geo_frame *frame, grid *grid);
int tdos_cmp(tdo_scheme *t1, tdo_scheme *t2, int verbose_level);
int true_false_string_numeric(const char *p);


// #############################################################################
// grid.cpp
// #############################################################################

//! holds invariants during the TDO process


class grid {

public:
	int f_points;
	int m;
	// # of objects
	// = v if f_point, = b otherwise
	int n;
	// # of structure constants per object
	int G_max;
	int first[MAX_GRID + 1];
	int len[MAX_GRID];
	int type_idx[MAX_GRID];
	int grid_entry[MAX_GRID];
	// the index into first[] / len[] of the object
	int type[MAX_GRID][MAX_GRID];
	// the structure constants

	grid();
	~grid();
	void print();
	void init_derived_i_first(grid *G_old, int derive_at_i);
	void init_derived_ij_first(grid *G_old, int I, int J);
	void copy_frame_to(grid *G_to);
	int insert_idx(int f, int l, int radix, int search_this, int *idx);

};



// #############################################################################
// inc_encoding.cpp
// #############################################################################

//! row-by-row encoding of an incidence geometry

class inc_encoding {

public:
	int *theX;
	int dim_n;
	int v;
	int b;
	int *R;

	inc_encoding();
	~inc_encoding();
	void init(int v, int b, int *R, int verbose_level);
	int find_square(int m, int n);
	void print_horizontal_bar(
		std::ostream &ost, incidence *inc, int f_print_isot, iso_type *it);
	void print_partitioned(
			std::ostream &ost, int v_cur, incidence *inc, int f_print_isot);
	void print_partitioned_override_theX(
			std::ostream &ost, int v_cur, incidence *inc, int *the_X, int f_print_isot);
	void print_permuted(cperm *pv, cperm *qv);
	tactical_decomposition *calc_tdo_without_vhbar(
		int f_second_tactical_decomposition, int verbose_level);
	void apply_permutation(incidence *inc, int v,
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

	int theY[MAX_V * MAX_R];
	int pairs[MAX_V][MAX_V];
	int f_lambda;
	int lambda;
	int f_find_square;
	int f_simple;

	// initial vertical and horizontal bars:
	int nb_i_vbar;
	int *i_vbar;
	int nb_i_hbar;
	int *i_hbar;

	int gl_nb_GEN;

	iso_type *iso_type_at_line[MAX_V];
	iso_type *iso_type_no_vhbars;

	int back_to_line;


	incidence();
	~incidence();
	void init(gen_geo *gg, int v, int b, int *R, int verbose_level);
	void init_bars(int verbose_level);
	void init_pairs(int verbose_level);
	int find_square(int m, int n);
	void print_param();
	void free_isot();
	void print_R(int v, cperm *p, cperm *q);
	void print(std::ostream &ost, int v);
	void print_override_theX(std::ostream &ost, int *theX, int v);
	void stuetze_nach_zeile(int i, int tdo_flags, int verbose_level);
	void stuetze2_nach_zeile(int i, int tdo_flags, int verbose_level);
	void set_range(int i, int first, int len);
	void set_flush_to_inc_file(int i, std::string &fname);
	void set_flush_line(int i);
	void print_geo(std::ostream &ost, int v, int *theGEO);
	void print_inc(std::ostream &ost, int v, long int *theInc);
	void print_blocks(std::ostream &ost, int v, long int *theInc);
	void compute_blocks(long int *&Blocks, int v, long int *theInc);
	int compute_k(int v, long int *theInc);
	int is_block_tactical(int v, long int *theInc);
	void geo_to_inc(int v, int *theGEO, long int *theInc);


};



// #############################################################################
// iso_grid.cpp
// #############################################################################

//! decomposition of an incidence geometry

class iso_grid {

public:
	int m; // = iso->b_t
	int n; // = type_len


	cperm q;
		// column permutation
	cperm qv; // q^-1


	int type[MAX_VB][MAX_TYPE];
	int G_max;
	int first[MAX_GRID];
	int len[MAX_GRID];
	int type_idx[MAX_GRID];
	int grid_entry[MAX_GRID];


	iso_grid();
	~iso_grid();
	void print();

};


// #############################################################################
// iso_info.cpp
// #############################################################################

//! input for the geometric isomorphism tester



class iso_info {

public:
	int *AtheX;
	/* v x max_r;
	 * dimension v x max_r or
	 * v x MAX_R */
	int *BtheX;
	int Af_full;
	int Bf_full;

	int v;
	int b;
	int max_r;

	int *R; /* [MAX_V] */

	int tdo_m;
	int tdo_V[MAX_V];
	int tdo_n;
	int tdo_B[MAX_B];

	int nb_isomorphisms;
	int f_break_after_fst;
	int f_verbose;
	int f_very_verbose;
	int f_use_d;
	int f_use_ddp;
	int f_use_ddb;
	int f_transpose_it;

	// optionally:
	int *Ar; // [v]
	int *Br;
	int *Ad; // [v]
	int *Bd;
	short *Addp; // [v \atop 2]
	short *Bddp;
	short *Addb; // [b \atop 2]
	short *Bddb;

	iso_info();
	~iso_info();

	void init_A_int(int *theA, int f_full);
	void init_B_int(int *theB, int f_full);
	void init_ddp(int f_ddp, short *Addp, short *Bddp);
	void init_ddb(int f_ddb, short *Addb, short *Bddb);
	void init_tdo(tdo_scheme *tdos);
	void init_tdo_V_B(int V, int B, int *Vi, int *Bj);
	void iso_test(int verbose_level);

};

void init_ISO2(void);




// #############################################################################
// iso_type.cpp
// #############################################################################

//! classification of geometries based on using the TDO invariant

class iso_type {

public:

	int v;
	int sum_R;
	incidence *inc;

	// flags for the type of TDO used:
	int f_transpose_it; // first flag
	int f_snd_TDO;      // second flag
	int f_ddp;          // third flag
	int f_ddb;          // fourth flag

	/* test of the first
	 * or the second kind:
	 * (second kind means
	 * check completely realizable
	 * geometries only)
	 */
	int f_generate_first;
	int f_beginning_checked;

	int f_range;
	int range_first;
	int range_len;

	int f_flush_line;

	std::string fname;

	int sum_nb_GEN;
	int sum_nb_GEO;
	int sum_nb_TDO;

	int nb_GEN;
	int nb_GEO;
	int nb_TDO;

	int dim_GEO;
	int dim_TDO;

	int **theGEO1; // [dim_GEO]
	int **theGEO2; // [dim_GEO]
	int *GEO_TDO_idx; // [dim_GEO]
	tdo_scheme **theTDO; // [dim_TDO]

	classify_using_canonical_forms *Canonical_forms;

	int f_print_mod;
	int print_mod;

	iso_type();
	~iso_type();
	void init(int v, incidence *inc, int tdo_flags, int verbose_level);
	void init2();
#if 0
	int find_geometry(
		inc_encoding *Encoding,
		int v, incidence *inc,
		int verbose_level);
#endif
	void add_geometry(
		inc_encoding *Encoding,
		int v, incidence *inc,
		int *already_there,
		int verbose_level);
	void recalc_autgroup(
		int v, incidence *inc,
		int tdo_idx, int geo_idx,
		int f_print_isot_small,
		int f_print_isot, int verbose_level);
	void calc_theY_and_tdos_override_v(
		inc_encoding *Encoding, incidence *inc, int v,
		int *&theY, tdo_scheme *&tdos, int verbose_level);
	tdo_scheme *geo_calc_tdos(
		inc_encoding *Encoding,
		incidence *inc,
		int v,
		short *&ddp, short *&ddb,
		cperm *tdo_p, cperm *tdo_q,
		int verbose_level);
	int find_geo(
		int v, incidence *inc, tdo_scheme *tdos,
		int *theY, int tdo_idx, int verbose_level);
	void find_and_add_geo(
		int v, incidence *inc,
		int *theY, int &f_new_object, int verbose_level);
	int isomorphic(
		int v, incidence *inc, tdo_scheme *tdos,
		int *pcA, int *pcB, int verbose_level);
	void do_aut_group(
		int v, incidence *inc, tdo_scheme *tdos,
		int *pc, int *aut_group_order,
		int f_print_isot_small, int f_print_isot, int verbose_level);
	void scan_tdo_flags(int tdo_flags);
	void second();
	void set_range(int first, int len);
	void set_flush_line();
	void flush();
	void TDO_realloc();
	void find_tdos(tdo_scheme *tdos, int *tdo_idx, int *f_found);
	void add_tdos_and_geo(tdo_scheme *tdos, int tdo_idx,
			int *theX, int *theY, int verbose_level);
	void add_geo(int tdo_idx, int *theX, int *theY);
	int *get_theX(int *theGEO);
	void geo_free(int *theGEO);
	void print_geos(int verbose_level);
	void write_inc_file(std::string &fname, int verbose_level);
	void write_blocks_file(std::string &fname, int verbose_level);
	void print(std::ostream &ost, int f_with_TDO, int v, incidence *inc);
	void print_GEO(int *pc, int v, incidence *inc);
	void print_status(std::ostream &ost, int f_with_flags);
	void print_flags(std::ostream &ost);
	void print_geometry(inc_encoding *Encoding, int v, incidence *inc);

};



// os.cpp:


int os_ticks();
int os_ticks_system();
int os_ticks_per_second();
void os_ticks_to_hms(int ticks, int *h, int *m, int *s);
void print_delta_time(int l, char *str);
char *Eostr(char *s);
char *eostr(char *s);
int ij2k(int i, int j, int n);
void k2ij(int k, int *i, int *j, int n);




// #############################################################################
// tactical_decomposition.cpp
// #############################################################################

//! compute a geometric invariant called TDO

class tactical_decomposition {

public:

	inc_encoding *Encoding;

	int f_TDO_multiple;
	int f_TDO_d_multiple;
	cperm p;
		// row permutation of degree tdo->v
	cperm q;
		// column permutation of degree tdo->inc->B
	cperm pv; // p^-1
	cperm qv; // q^-1
		// given theX, applying p to the rows,
		// q to the columns, the matrix of the TDO is obtained
	grid *G_last;
	grid *G_current;
	grid *G_next;
	tdo_scheme *tdos;
	tdo_scheme *tdos2;

	tactical_decomposition();
	~tactical_decomposition();
	void init(inc_encoding *Encoding, int verbose_level);
	void tdo_calc(inc_encoding *Encoding, incidence *inc, int v,
		int f_second_tactical_decomposition, int verbose_level);
	void make_point_and_block_partition(grid *Gpoints, grid *Gblocks);
	void init_partition(grid *Gpoints, grid *Gblocks,
		incidence *inc, int v, int verbose_level);
	void print();
	void radix_sort(grid *G, int radix, int first, int last);
	void refine_types(grid *Gm1, grid *G1, int verbose_level);
	void recollect_types(int v, grid *G0, grid *G1, int verbose_level);
	void collect_types(int v, grid *G0, grid *G1, int verbose_level);
	void next(int v, int verbose_level);
	void calc2(int v, int verbose_level);
	tdo_scheme *get_tdos(grid *G0, grid *G1, int f_derived, int verbose_level);
	void dd_work(int v, int f_points,
		short *&dd, int &N, short *&dd_mult, int verbose_level);
	void tdo_dd(int v, int f_points, int f_blocks,
		short *&ddp, int &Np, short *&ddp_mult,
		short *&ddb, int &Nb, short *&ddb_mult, int verbose_level);
	void refine(int v,
			grid *G, grid *G_next,
		int f_points, geo_frame *frame,
		cperm *P, cperm *Pv, cperm *Q, cperm *Qv, int verbose_level);
	void second_order_tdo(int v, int verbose_level);


};








// #############################################################################
// tdo_gradient.cpp
// #############################################################################

//! compute a more refined geometric invariant called second TDO

class tdo_gradient {

public:

	int N;
	int nb_tdos;

	tdo_scheme **tdos; // [N]
	int *mult; // [N]
	int *type; // [N]


	tdo_gradient();
	~tdo_gradient();
	void allocate(int N);
	void add_tdos(tdo_scheme *tdos, int i, int verbose_level);
};




// #############################################################################
// tdo_scheme.cpp
// #############################################################################

//! a geometric invariant called TDO

class tdo_scheme {

public:
	int m, n;
	int *a;

	// nb_rows x nb_cols is the dimension of the  TDO matrix;
	// we add one column on the left and
	// one row on top for V[i] and B[j], respectively.
	// the very first entry is the size of the array,
	// which is (nb_rows + 1) * (nb_cols + 1)
	// m = nb_rows
	// n = nb_cols

	tdo_scheme();
	~tdo_scheme();
	void allocate(int nb_rows, int nb_cols);
	int &nb_rows();
	int &nb_cols();
	int &Vi(int i);
	int &Bj(int j);
	int &aij(int i, int j);
	void print();
};







}}



#endif /* SRC_LIB_FOUNDATIONS_GEOMETRY_BUILDER_GEOMETRY_BUILDER_H_ */
