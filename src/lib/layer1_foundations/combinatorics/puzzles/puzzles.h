/*
 * puzzles.h
 *
 *  Created on: Dec 1, 2024
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER1_FOUNDATIONS_COMBINATORICS_PUZZLES_PUZZLES_H_
#define SRC_LIB_LAYER1_FOUNDATIONS_COMBINATORICS_PUZZLES_PUZZLES_H_





namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace puzzles {



// #############################################################################
// brick_domain.cpp
// #############################################################################

//! a problem of Neil Sloane

class brick_domain {

public:
	algebra::field_theory::finite_field *F;
	int q;
	int nb_bricks;

	brick_domain();
	~brick_domain();
	void init(
			algebra::field_theory::finite_field *F, int verbose_level);
	void unrank(
			int rk, int &f_vertical,
		int &x0, int &y0, int verbose_level);
	int rank(
			int f_vertical, int x0, int y0, int verbose_level);
	void unrank_coordinates(
			int rk,
		int &x1, int &y1, int &x2, int &y2,
		int verbose_level);
	int rank_coordinates(
			int x1, int y1, int x2, int y2,
		int verbose_level);
};





// #############################################################################
// domino_assignment.cpp:
// #############################################################################

// the dimensions are (<D>+1) * <s>   x <D>*<s>
// so, for D=7 we would get
// s=4:   32x28
// s=5:   40x35

//! compute a domino portrait using an optimization algorithm

class domino_assignment {
public:
	int D;
	int s;
	int size_dom;
	int tot_dom;

	int M; // number of rows  = (D + 1) * s
	int N; // number of columns = D * s

	int *ij_posi; // [M * N * 2];
		// ij_posi[(i * N + j) * 2 + 0] = i
		// ij_posi[(i * N + j) * 2 + 1] = j

	int *assi; // [tot_dom * 5];
		// 0: m
		// 1: n
		// 2: o = orientation
		// 3: i
		// 4: j
		// where (i,j) is the place of the top or left half of the domino

	int *broken_dom; // [M * N]
		// broken_dom[i * n + j] is the index in the assi array
		// of the domino piece covering the place (i,j)
	int *matching; // [M * N]
		// matching[i * N + j] tells the direction
		// of the second half of the domino
	int *A; // [M * N], the domino matrix
	int *mphoto; // [M * N], the photo matrix

	int *North; // [M * N]
	int *South; // [M * N]
	int *West; // [M * N]
	int *East; // [M * N]

	int brake_cnt;
	int *brake; // [tot_dom], used as [brake_cnt]

	int nb_changes;

	std::vector<domino_change> Changes;


	domino_assignment();
	~domino_assignment();
	void stage0(
			int verbose_level);
	void stage1(
			int verbose_level);
	void stage2(
			int verbose_level);
	void initialize_assignment(
			int D, int s, int verbose_level);
	void init_matching(
			int verbose_level);
	int cost_function();
	int compute_cost_of_one_piece(
			int idx);
	int compute_cost_of_one_piece_directly(
			int m, int n, int o, int i, int j);
	int my_distance(
			int a, int b);
	void compute_domino_matrix(
			int depth);
	void move(
			domino_assignment *To);
	void draw_domino_matrix(
			std::string &fname,
			int depth,
			int f_has_cost, int cost,
			other::graphics::layered_graph_draw_options *Draw_options,
			int verbose_level);
	void draw_domino_matrix2(
			std::string &fname,
			int f_has_cost, int cost,
		int f_frame, int f_grid, int f_B, int *B,
		int f_numbers, int f_gray,
		other::graphics::layered_graph_draw_options *Draw_options,
		int verbose_level);
	void read_photo(
			std::string &photo_fname, int verbose_level);
	void scale_photo(
			double *dphoto, int verbose_level);
	void do_flip_recorded(
			int f2, int verbose_level);
	void do_flip(
			int f2, int verbose_level);
	void flip_each(
			int verbose_level);
	void flip_randomized(
			int verbose_level);
	void do_swap_recorded(
			int s1, int s2, int verbose_level);
	void do_swap(
			int s1, int s2, int verbose_level);
	int do_flipswap(
			int f2);
	void swap_randomized(
			int verbose_level);
	void swap_each(
			int verbose_level);
	void do_horizontal_rotate(
			int ro, int verbose_level);
	void do_vertical_rotate(
			int ro, int verbose_level);
	int modify_matching(
			int idx_first_broken,
			int ass_m, int ass_n,
			int ass_o, int ass_i, int ass_j,
			int verbose_level);
	void follow_the_matching(
			int l, int *used, int *reached,
			int *list, int *length, int *prec,
			int verbose_level);
	int find_match(
			int l,
		int *reached1, int *list1, int *length1, int *prec1,
		int *reached2, int *list2, int *length2, int *prec2,
		int verbose_level);
	int breadth_search(
			int l, int *used, int *reached,
			int *list, int *length, int *prec,
			int verbose_level);
	void rotate_once(
			int ro, int verbose_level);
	void rotate_randomized(
			int verbose_level);
	void do_horizontal_shift(
			int ro, int verbose_level);
	void do_vertical_shift(
			int ro, int verbose_level);
	void shift_once(
			int ro, int verbose_level);
	void shift_once_randomized(
			int verbose_level);
	void shift_randomized(
			int verbose_level);
	void flip_after_shift(
			int verbose_level);
	void print_matching(
			std::ostream &ost);
	void print(
			std::ostream &ost);
	void prepare_latex(
			std::string &photo_label, int verbose_level);
	void record_flip(
			int idx, int verbose_level);
	void record_swap(
			int s1, int s2, int verbose_level);
	void record_matching(
			int verbose_level);
	void drop_changes_to(
			int nb_changes_to_drop_to, int verbose_level);
	void classify_changes_by_type(
			int verbose_level);
	void get_cost_function(
			int *&Cost, int &len, int verbose_level);
};

// #############################################################################
// domino_change.cpp:
// #############################################################################


//! utility class for the domino portrait algorithm

class domino_change {
public:
	int type_of_change;
	int cost_after_change;

	domino_change();
	~domino_change();
	void init(
			domino_assignment *DA,
			int type_of_change, int verbose_level);
};








// #############################################################################
// pentomino_puzzle.cpp
// #############################################################################


#define NB_PIECES 18



//! generate all solutions of the pentomino puzzle


class pentomino_puzzle {

	public:
	int *S[NB_PIECES];
	int S_length[NB_PIECES];
	int *O[NB_PIECES];
	int O_length[NB_PIECES];
	int *T[NB_PIECES];
	int T_length[NB_PIECES];
	int *R[NB_PIECES];
	int R_length[NB_PIECES];
	int Rotate[4 * 25];
	int Rotate6[4 * 36];
	int var_start[NB_PIECES + 1];
	int var_length[NB_PIECES + 1];

	pentomino_puzzle();
	~pentomino_puzzle();
	void main(
			int verbose_level);
	int has_interlocking_Ps(
			long int *set);
	int has_interlocking_Pprime(
			long int *set);
	int has_interlocking_Ls(
			long int *set);
	int test_if_interlocking_Ps(
			int a1, int a2);
	int has_interlocking_Lprime(
			long int *set);
	int test_if_interlocking_Ls(
			int a1, int a2);
	int number_of_pieces_of_type(
			int t, long int *set);
	int does_it_contain_an_I(
			long int *set);
	void decode_assembly(
			long int *set);
	// input set[5]
	void decode_piece(
			int j, int &h, int &r, int &t);
	// h is the kind of piece
	// r is the rotation index
	// t is the translation index
	// to get the actual rotation and translation, use
	// R[h][r] and T[h][t].
	int code_piece(
			int h, int r, int t);
	void draw_it(
			std::ostream &ost, long int *sol);
	void compute_image_function(
			other::data_structures::set_of_sets *S,
			int elt_idx,
			int gen_idx, int &idx_of_image, int verbose_level);
	void turn_piece(
			int &h, int &r, int &t, int verbose_level);
	void flip_piece(
			int &h, int &r, int &t, int verbose_level);
	void setup_pieces();
	void setup_rotate();
	void setup_var_start();
	void make_coefficient_matrix(
			solvers::diophant *D);

};









}}}}


#endif /* SRC_LIB_LAYER1_FOUNDATIONS_COMBINATORICS_PUZZLES_PUZZLES_H_ */
