// hamming.cpp
// 
// Anton Betten
// Dec 9, 2010
//
//
// 
// 
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;

// global data:

int t0; // the system time when the program started
int n;
int nb_points;
int nb_lines;
int nb_planes;
int nb_solids;
int nb_points_folded;
int nb_lines_folded;
int nb_planes_folded;
int nb_solids_folded;

int nb_BLOCKS;
int nb_POintS;


void create_object(int verbose_level);
void print_solid(int *x, int b_1, int b_2, int b_3, int c_1, int c_2, int c_3);
void print_line(int *x, int d_1, int e_1);
int is_adjacent(int *v_solid, int b_1, int b_2, int b_3, int c_1, int c_2, int c_3, int *v_line, int d_1, int e_1);
void create_geometry(int verbose_level);
int point_rank(int *x);
void point_unrank(int *x, int rk);
int line_rank(int *x, int b_1, int verbose_level);
void line_unrank(int rk, int *x, int &b_1, int verbose_level);
int plane_rank(int *x, int b_1, int b_2, int verbose_level);
void plane_unrank(int rk, int *x, int &b_1, int &b_2, int verbose_level);
int solid_rank(int *x, int b_1, int b_2, int b_3, int verbose_level);
void solid_unrank(int rk, int *x, int &b_1, int &b_2, int &b_3, int verbose_level);
int line_vertex_pair_rank(int *x, int b_1, int c_1, int verbose_level);
void line_vertex_pair_unrank(int rk, int *x, int &b_1, int &c_1, int verbose_level);
int solid_diagonal_pair_rank(int *x, int b_1, int b_2, int b_3, int c_1, int c_2, int c_3, int verbose_level);
void solid_diagonal_pair_unrank(int rk, int *x, int &b_1, int &b_2, int &b_3, 
	int &c_1, int &c_2, int &c_3, int verbose_level);
int low_weight_3vec_rank(int *x);
void low_weight_3vec_unrank(int rk, int *x);
void compress1(int *x, int *x_compressed, int b_1);
void expand1(int *x, int *x_compressed, int b_1);
void compress2(int *x, int *x_compressed, int b_1, int b_2);
void expand2(int *x, int *x_compressed, int b_1, int b_2);
void compress3(int *x, int *x_compressed, int b_1, int b_2, int b_3);
void expand3(int *x, int *x_compressed, int b_1, int b_2, int b_3);
int is_incident_point_line(int *v_point, int *v_line, int b_1);
int is_incident_line_solid(int *v_line, int b_1, int *v_solid, int c_1, int c_2, int c_3);
int is_incident_point_edge_solid(int *v_line, int e_1, 
	int *v_point, int *v_solid, int b_1, int b_2, int b_3);
void representative_under_folding(int *x, int len);
void representative_under_folding_line(int *x, int b_1);
void representative_under_folding_plane(int *x, int b_1, int b_2);
void representative_under_folding_solid(int *x, int b_1, int b_2, int b_3);
void opposite_under_folding_line(int *x, int b_1);
void opposite_under_folding_plane(int *x, int b_1, int b_2);
void opposite_under_folding_solid(int *x, int b_1, int b_2, int b_3);
void invert(int *x, int len);

int main(int argc, char **argv)
{
	int verbose_level;
	int i;
	t0 = os_ticks();
	
	n = 4;
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}
	
	create_geometry(verbose_level);
	create_object(verbose_level);
	
	the_end_quietly(t0);
}

#if 0
void create_object(int verbose_level)
{
	int *x;
	int *y;
	int *Points;
	int *Blocks;
	int b_1, b_2, b_3;
	int coeff[3];
	int i, j, h, a, b;
	int POint_width = n + 1;
	int BLOCK_width = n + n + 3;
	
	cout << "create_object" << endl;
	cout << "nb_points_folded=" << nb_points_folded << endl;
	cout << "nb_lines_folded=" << nb_lines_folded << endl;
	x = NEW_int(n);
	y = NEW_int(n);
	nb_POintS = nb_lines_folded;
	nb_BLOCKS = nb_points_folded * nb_solids_folded;
	
	Points = NEW_int(nb_POintS * POint_width);
	Blocks = NEW_int(nb_BLOCKS * BLOCK_width);
	cout << "nb_POintS=" << nb_POintS << endl;
	cout << "nb_BLOCKS=" << nb_BLOCKS << endl;

	for (i = 0; i < nb_POintS; i++) {
		line_unrank(i, x, b_1, 0);
		representative_under_folding_line(x, b_1);
		for (h = 0; h < n; h++) {
			Points[i * POint_width + h] = x[h];
			}
		Points[i * POint_width + n] = b_1;
		}
	cout << "POINTS:" << endl;
	print_integer_matrix_width(cout, Points, nb_POintS, POint_width, POint_width, 1);

	for (i = 0; i < nb_BLOCKS; i++) {
		cout << "BLOCK " << i << ":";
		a = i / 8;
		b = i % 8;
		cout << " a=" << a << " b=" << b << endl;
		solid_unrank(a, x, b_1, b_2, b_3, 0);
		representative_under_folding_solid(x, b_1, b_2, b_3);
		for (h = 0; h < n; h++) {
			y[h] = x[h];
			}
		AG_element_unrank(2, coeff, 1, 3, b);
		if (coeff[0]) {
			y[b_1] = 1;
			}
		if (coeff[1]) {
			y[b_2] = 1;
			}
		if (coeff[2]) {
			y[b_3] = 1;
			}
		for (h = 0; h < n; h++) {
			Blocks[i * BLOCK_width + h] = y[h];
			}
		for (h = 0; h < n; h++) {
			Blocks[i * BLOCK_width + n + h] = x[h];
			}
		Blocks[i * BLOCK_width + n + n + 0] = b_1;
		Blocks[i * BLOCK_width + n + n + 1] = b_2;
		Blocks[i * BLOCK_width + n + n + 2] = b_3;
		}
	cout << "BLOCKS:" << endl;
	print_integer_matrix_width(cout, Blocks, nb_BLOCKS, BLOCK_width, BLOCK_width, 1);

	int *M1;
	int *v_line;
	int *v_point;
	int *v_solid;
	int e_1;
	//int a;

	M1 = NEW_int(nb_POintS * nb_BLOCKS);
	for (i = 0; i < nb_POintS * nb_BLOCKS; i++) {
		M1[i] = 0;
		}
	for (i = 0; i < nb_POintS; i++) {
		v_line = Points + i * POint_width;
		e_1 = Points[i * POint_width + n];
		
		cout << "i=" << i << " : line ";
		int_vec_print(cout, v_line, n);
		cout << "e_1=" << e_1 << endl;
		
		for (j = 0; j < nb_BLOCKS; j++) {
			v_point = Blocks + j * BLOCK_width;
			v_solid = Blocks + j * BLOCK_width + n;
			b_1 = Blocks[j * BLOCK_width + n + n + 0];
			b_2 = Blocks[j * BLOCK_width + n + n + 1];
			b_3 = Blocks[j * BLOCK_width + n + n + 2];

			cout << "j=" << j << " : point ";
			int_vec_print(cout, v_point, n);
			cout << " solid ";
			int_vec_print(cout, v_solid, n);
			cout << " b_1=" << b_1;
			cout << " b_2=" << b_2;
			cout << " b_3=" << b_3;
			cout << endl;
		
			
			a = 0;
			if (is_incident_point_edge_solid(v_line, e_1, 
				v_point, v_solid, b_1, b_2, b_3)) {
				a = 1;
				}
			cout << "a=" << a << endl;
			M1[i * nb_BLOCKS + j] = a;
			}
		}
	cout << "incidence matrix:" << endl;
	
	print_integer_matrix_width(cout, M1, nb_POintS, nb_BLOCKS, nb_BLOCKS, 1);
	
	int *AAt;

	AAt = NEW_int(nb_POintS * nb_POintS);
	for (i = 0; i < nb_POintS; i++) {
		for (j = 0; j < nb_POintS; j++) {
			a = 0;
			for (h = 0; h < nb_BLOCKS; h++) {
				a += M1[i * nb_BLOCKS + h] * M1[j * nb_BLOCKS + h];
				}
			AAt[i * nb_POintS + j] = a;
			}
		}

	cout << "AAt:" << endl;
	
	print_integer_matrix_width(cout, AAt, nb_POintS, nb_POintS, nb_POintS, 1);
	
	{
	incidence_structure Inc;
	char fname[1000];

	sprintf(fname, "HamG_%d_2.inc", n);
	
	Inc.init_by_matrix(nb_POintS, nb_BLOCKS, M1, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}
	
	
	FREE_int(x);
	FREE_int(y);
	FREE_int(M1);
	FREE_int(AAt);
}
#else
void create_object(int verbose_level)
{
	int *Points;
	int *Blocks;
	int *x;
	int *y;
	int b_1, b_2, b_3;
	int c_1, c_2, c_3;
	int d_1, e_1;
	int i, j, h, a, ii, jj;
	int POint_width = n + 6;
	int BLOCK_width = n + 2;
	
	cout << "create_object" << endl;
	nb_POintS = nb_solids * 4;
	nb_BLOCKS = nb_lines * 2;
	
	x = NEW_int(n);
	y = NEW_int(n);
	Points = NEW_int(nb_POintS * POint_width);
	Blocks = NEW_int(nb_BLOCKS * BLOCK_width);
	cout << "nb_POintS=" << nb_POintS << endl;
	cout << "nb_BLOCKS=" << nb_BLOCKS << endl;

	for (i = 0; i < nb_POintS; i++) {
		solid_diagonal_pair_unrank(i, x, b_1, b_2, b_3, 
			c_1, c_2, c_3, 0);
		for (h = 0; h < n; h++) {
			Points[i * POint_width + h] = x[h];
			}
		Points[i * POint_width + n + 0] = b_1;
		Points[i * POint_width + n + 1] = b_2;
		Points[i * POint_width + n + 2] = b_3;
		Points[i * POint_width + n + 3] = c_1;
		Points[i * POint_width + n + 4] = c_2;
		Points[i * POint_width + n + 5] = c_3;
		}
	cout << "POintS:" << endl;
	print_integer_matrix_width(cout, Points, nb_POintS, POint_width, POint_width, 1);


	for (i = 0; i < nb_BLOCKS; i++) {
		//cout << "BLOCK " << i << ":";
		line_vertex_pair_unrank(i, x, b_1, c_1, 0);
		for (h = 0; h < n; h++) {
			Blocks[i * BLOCK_width + h] = x[h];
			}
		Blocks[i * BLOCK_width + n + 0] = b_1;
		Blocks[i * BLOCK_width + n + 1] = c_1;
		}
	cout << "BLOCKS:" << endl;
	print_integer_matrix_width(cout, Blocks, nb_BLOCKS, BLOCK_width, BLOCK_width, 1);

	int *M1;
	int *v_line;
	int *v_solid;

	M1 = NEW_int(nb_POintS * nb_BLOCKS);
	for (i = 0; i < nb_POintS * nb_BLOCKS; i++) {
		M1[i] = 0;
		}
	for (i = 0; i < nb_POintS; i++) {
		v_solid = Points + i * POint_width;
		b_1 = Points[i * POint_width + n + 0];
		b_2 = Points[i * POint_width + n + 1];
		b_3 = Points[i * POint_width + n + 2];
		c_1 = Points[i * POint_width + n + 3];
		c_2 = Points[i * POint_width + n + 4];
		c_3 = Points[i * POint_width + n + 5];
		
		
		for (j = 0; j < nb_BLOCKS; j++) {
			v_line = Blocks + j * BLOCK_width;
			d_1 = Blocks[j * BLOCK_width + n + 0];
			e_1 = Blocks[j * BLOCK_width + n + 1];

		
			
			a = is_adjacent(v_solid, b_1, b_2, b_3, c_1, c_2, c_3, v_line, d_1, e_1);

			//cout << "a=" << a << endl;
			if (a) {
				cout << "solid i=" << i << " ";
				print_solid(v_solid, b_1, b_2, b_3, c_1, c_2, c_3);
				//int_vec_print(cout, v_solid, n);
				//cout << " b_1=" << b_1;
				//cout << " b_2=" << b_2;
				//cout << " b_3=" << b_3;
				//cout << " c_1=" << c_1;
				//cout << " c_2=" << c_2;
				//cout << " c_3=" << c_3;
				//cout << endl;
				cout << "and line j=" << j << " ";
				print_line(v_line, d_1, e_1);
				//int_vec_print(cout, v_line, n);
				//cout << " d_1=" << d_1;
				//cout << " e_1=" << e_1;
				cout << " are adjacent" << endl;
				}
			M1[i * nb_BLOCKS + j] = a;
			}
		}
	cout << "incidence matrix:" << endl;
	
	print_integer_matrix_width(cout, M1, nb_POintS, nb_BLOCKS, nb_BLOCKS, 1);
	
	int *AAt;

	AAt = NEW_int(nb_POintS * nb_POintS);
	for (i = 0; i < nb_POintS; i++) {
		for (j = 0; j < nb_POintS; j++) {
			a = 0;
			for (h = 0; h < nb_BLOCKS; h++) {
				a += M1[i * nb_BLOCKS + h] * M1[j * nb_BLOCKS + h];
				}
			AAt[i * nb_POintS + j] = a;
			}
		}

	cout << "AAt:" << endl;
	
	print_integer_matrix_width(cout, AAt, nb_POintS, nb_POintS, nb_POintS, 1);
	
	{
	incidence_structure Inc;
	char fname[1000];

	sprintf(fname, "HamG_%d_2.inc", n);
	
	Inc.init_by_matrix(nb_POintS, nb_BLOCKS, M1, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}



	int *opposite_point;
	int *opposite_block;
	int *pts_sorted;
	int *blocks_sorted;
	int *Mtx2;

	opposite_point = NEW_int(nb_POintS);
	opposite_block = NEW_int(nb_BLOCKS);
	pts_sorted = NEW_int(nb_POintS);
	blocks_sorted = NEW_int(nb_BLOCKS);
	Mtx2 = NEW_int(nb_POintS * nb_BLOCKS);
	
	for (i = 0; i < nb_POintS; i++) {
		solid_diagonal_pair_unrank(i, x, b_1, b_2, b_3, c_1, c_2, c_3, 0);
		for (h = 0; h < n; h++) {
			y[h] = 1 - x[h];
			}
		c_1 = 1 - c_1;
		c_2 = 1 - c_2;
		c_3 = 1 - c_3;
		j = solid_diagonal_pair_rank(y, b_1, b_2, b_3, c_1, c_2, c_3, 0);
		opposite_point[i] = j;
		}
	cout << "i : opposite_point[i]" << endl;
	for (i = 0; i < nb_POintS; i++) {
		cout << setw(3) << i << " : " << setw(3) << opposite_point[i] << endl;
		}


	for (i = 0; i < nb_BLOCKS; i++) {
		line_vertex_pair_unrank(i, x, d_1, e_1, 0);
		for (h = 0; h < n; h++) {
			y[h] = 1 - x[h];
			}
		e_1 = 1 - e_1;
		j = line_vertex_pair_rank(y, d_1, e_1, 0);
		opposite_block[i] = j;
		}
	
	cout << "i : opposite_block[i]" << endl;
	for (i = 0; i < nb_BLOCKS; i++) {
		cout << setw(3) << i << " : " << setw(3) << opposite_block[i] << endl;
		}
	

	j = 0;
	for (i = 0; i < nb_POintS; i++) {
		a = opposite_point[i];
		if (a > i) {
			pts_sorted[j++] = i;
			pts_sorted[j++] = a;
			}
		}
	if (j != nb_POintS) {
		cout << "j != nb_POintS" << endl;
		exit(1);
		}
	cout << "i : pts_sorted[i]" << endl;
	for (i = 0; i < nb_points; i++) {
		cout << setw(3) << i << " : " << setw(3) << pts_sorted[i] << endl;
		}
	j = 0;
	for (i = 0; i < nb_BLOCKS; i++) {
		a = opposite_block[i];
		if (a > i) {
			blocks_sorted[j++] = i;
			blocks_sorted[j++] = a;
			}
		}
	if (j != nb_BLOCKS) {
		cout << "j != nb_BLOCKS" << endl;
		exit(1);
		}
	cout << "i : blocks_sorted[i]" << endl;
	for (i = 0; i < nb_lines; i++) {
		cout << setw(3) << i << " : " << setw(3) << blocks_sorted[i] << endl;
		}
	for (i = 0; i < nb_POintS; i++) {
		ii = pts_sorted[i];
		for (j = 0; j < nb_BLOCKS; j++) {
			jj = blocks_sorted[j];
			Mtx2[i * nb_BLOCKS + j] = M1[ii * nb_BLOCKS + jj];
			}
		}
	cout << "reordered incidence matrix:" << endl;
	
	print_integer_matrix_width(cout, Mtx2, nb_POintS, nb_BLOCKS, nb_BLOCKS, 1);
	{
	incidence_structure Inc;
	char fname[1000];

	sprintf(fname, "HamG_%d_2.inc", n);
	
	Inc.init_by_matrix(nb_POintS, nb_BLOCKS, Mtx2, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}


	int *Mtx3;
	int nb_POintS_folded, nb_BLOCKS_folded;
	
	nb_POintS_folded = nb_POintS / 2;
	nb_BLOCKS_folded = nb_BLOCKS / 2;
	Mtx3 = NEW_int(nb_POintS_folded * nb_BLOCKS_folded);
	for (i = 0; i < nb_POintS_folded * nb_BLOCKS_folded; i++) {
		Mtx3[i] = 0;
		}
	for (i = 0; i < nb_POintS_folded; i++) {
		ii = 2 * i;
		for (j = 0; j < nb_BLOCKS_folded; j++) {
			jj = 2 * j;
			a = Mtx2[ii * nb_BLOCKS + jj];
			a += Mtx2[ii * nb_BLOCKS + jj + 1];
			//a += Mtx2[(ii + 1) * nb_BLOCKS + jj];
			//a += Mtx2[(ii + 1) * nb_BLOCKS + jj + 1];
			if (a > 1) {
				cout << "i=" << i << " j=" << j << " a=" << a << endl;
				}
			if (a) {
				Mtx3[i * nb_BLOCKS_folded + j] = 1;
				}
			}
		}
	cout << "folded incidence matrix:" << endl;
	
	print_integer_matrix_width(cout, Mtx3, nb_POintS_folded, nb_BLOCKS_folded, nb_BLOCKS_folded, 1);


	int *FFt;

	FFt = NEW_int(nb_POintS_folded * nb_POintS_folded);
	for (i = 0; i < nb_POintS_folded; i++) {
		for (j = 0; j < nb_POintS_folded; j++) {
			a = 0;
			for (h = 0; h < nb_BLOCKS_folded; h++) {
				a += Mtx3[i * nb_BLOCKS_folded + h] * Mtx3[j * nb_BLOCKS_folded + h];
				}
			FFt[i * nb_POintS_folded + j] = a;
			}
		}

	cout << "FFt:" << endl;
	
	print_integer_matrix_width(cout, FFt, nb_POintS_folded, nb_POintS_folded, nb_POintS_folded, 1);
	


	{
	incidence_structure Inc;
	char fname[1000];

	sprintf(fname, "HamG_%d_2.inc", n);
	
	Inc.init_by_matrix(nb_POintS_folded, nb_BLOCKS_folded, Mtx3, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}
	

	
	FREE_int(x);
	FREE_int(y);
	FREE_int(M1);
	FREE_int(AAt);
}

#endif

void print_solid(int *x, int b_1, int b_2, int b_3, int c_1, int c_2, int c_3)
{
	int *w;
	int c[3];
	int i;
	
	w = NEW_int(n);
	for (i = 0; i < n; i++) {
		w[i] = FALSE;
		}
	c[0] = c_1;
	c[1] = c_2;
	c[2] = c_3;
	w[b_1] = TRUE;
	w[b_2] = TRUE;
	w[b_3] = TRUE;
	for (i = 0; i < n; i++) {
		if (w[i]) {
			cout << "*";
			}
		else {
			cout << x[i];
			}
		}
	cout << ",";
	for (i = 0; i < n; i++) {
		w[i] = x[i];
		}
	w[b_1] = c[0];
	w[b_2] = c[1];
	w[b_3] = c[2];
	int_vec_print(cout, w, n);
	invert(c, 3);
	w[b_1] = c[0];
	w[b_2] = c[1];
	w[b_3] = c[2];
	int_vec_print(cout, w, n);
	FREE_int(w);
}

void print_line(int *x, int d_1, int e_1)
{
	int *w;
	int i;
	
	w = NEW_int(n);
	for (i = 0; i < n; i++) {
		w[i] = FALSE;
		}
	w[d_1] = TRUE;
	for (i = 0; i < n; i++) {
		if (w[i]) {
			cout << "*";
			}
		else {
			cout << x[i];
			}
		}
	cout << ",";
	for (i = 0; i < n; i++) {
		w[i] = x[i];
		}
	w[d_1] = e_1;
	int_vec_print(cout, w, n);
	FREE_int(w);
}

int is_adjacent(int *v_solid, int b_1, int b_2, int b_3, int c_1, int c_2, int c_3, int *v_line, int d_1, int e_1)
{
	int *x;
	int *y;
	int h;
	int c[3];
	int ret = FALSE;
	
	x = NEW_int(n);
	y = NEW_int(n);
	for (h = 0; h < n; h++) {
		x[h] = v_solid[h];
		}
	for (h = 0; h < n; h++) {
		y[h] = v_line[h];
		}

	if (d_1 == b_1 || d_1 == b_2 || d_1 == b_3) {
		c[0] = c_1;
		c[1] = c_2;
		c[2] = c_3;
		x[b_1] = c[0];
		x[b_2] = c[1];
		x[b_3] = c[2];
		y[d_1] = e_1;
		if (int_vec_compare(x, y, n) == 0) {
			ret = TRUE;
			goto done;
			}
		invert(c, 3);
		x[b_1] = c[0];
		x[b_2] = c[1];
		x[b_3] = c[2];
		if (int_vec_compare(x, y, n) == 0) {
			ret = TRUE;
			goto done;
			}
		}

done:
	FREE_int(x);
	FREE_int(y);
	return ret;
}

void create_geometry(int verbose_level)
{
	int i, j, h, a, b_1, b_2, b_3, ii, jj;
	int *x;
	int *y;
	number_theory_domain NT;
	
	x = NEW_int(n);
	y = NEW_int(n);
	nb_points = NT.i_power_j(2, n);
	nb_lines = NT.i_power_j(2, n - 1) * n;
	nb_planes = NT.i_power_j(2, n - 2) * n * (n - 1) / 2;
	nb_solids = NT.i_power_j(2, n - 3) * n * (n - 1) * (n - 2) / 6;
	
	cout << "nb_points=" << nb_points << endl;
	cout << "nb_lines=" << nb_lines << endl;
	cout << "nb_planes=" << nb_planes << endl;
	cout << "nb_solids=" << nb_solids << endl;
	


	cout << "lines:" << endl;
	for (i = 0; i < nb_lines; i++) {
		line_unrank(i, x, b_1, 0);
		cout << setw(3) << i << " : [";
		int_vec_print(cout, x, n);
		cout << " : " << b_1 << "]" << endl;
		j = line_rank(x, b_1, 0);
		if (j != i) {
			cout << "j != i" << endl;
			exit(1);
			}
		}


	cout << "planes:" << endl;
	for (i = 0; i < nb_planes; i++) {
		plane_unrank(i, x, b_1, b_2, 0);
		cout << setw(3) << i << " : [";
		int_vec_print(cout, x, n);
		cout << " : " << b_1 << "," << b_2 << "]" << endl;
		j = plane_rank(x, b_1, b_2, 0);
		if (j != i) {
			cout << "j != i" << endl;
			exit(1);
			}
		}

	cout << "solids:" << endl;
	for (i = 0; i < nb_solids; i++) {
		solid_unrank(i, x, b_1, b_2, b_3, 0);
		cout << setw(3) << i << " : [";
		int_vec_print(cout, x, n);
		cout << " : " << b_1 << "," << b_2 << "," << b_3 << "]" << endl;
		j = solid_rank(x, b_1, b_2, b_3, 0);
		if (j != i) {
			cout << "j != i" << endl;
			exit(1);
			}
		}

	int *Mtx;
	int *Mtx2;

	Mtx = NEW_int(nb_points * nb_lines);
	Mtx2 = NEW_int(nb_points * nb_lines);

	for (i = 0; i < nb_points * nb_lines; i++) {
		Mtx[i] = 0;
		}
	for (j = 0; j < nb_lines; j++) {
		line_unrank(j, x, b_1, 0);
		for (h = 0; h < n; h++) {
			y[h] = x[h];
			}
		i = point_rank(y);
		Mtx[i * nb_lines + j] = 1;
		y[b_1] = 1;
		i = point_rank(y);
		Mtx[i * nb_lines + j] = 1;
		}
	cout << "incidence matrix:" << endl;
	
	print_integer_matrix_width(cout, Mtx, nb_points, nb_lines, nb_lines, 1);
	
	{
	incidence_structure Inc;
	char fname[1000];

	sprintf(fname, "HamG_%d_2.inc", n);
	
	Inc.init_by_matrix(nb_points, nb_lines, Mtx, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}
	
	int *opposite_point;
	int *opposite_line;
	int *pts_sorted;
	int *lines_sorted;

	opposite_point = NEW_int(nb_points);
	opposite_line = NEW_int(nb_lines);
	pts_sorted = NEW_int(nb_points);
	lines_sorted = NEW_int(nb_lines);

	for (i = 0; i < nb_points; i++) {
		point_unrank(x, i);
		for (h = 0; h < n; h++) {
			y[h] = 1 - x[h];
			}
		j = point_rank(y);
		opposite_point[i] = j;
		}
	cout << "i : opposite_point[i]" << endl;
	for (i = 0; i < nb_points; i++) {
		cout << setw(3) << i << " : " << setw(3) << opposite_point[i] << endl;
		}
	for (i = 0; i < nb_lines; i++) {
		line_unrank(i, x, b_1, 0);
		for (h = 0; h < n; h++) {
			y[h] = 1 - x[h];
			}
		j = line_rank(y, b_1, 0);
		opposite_line[i] = j;
		}
	
	cout << "i : opposite_line[i]" << endl;
	for (i = 0; i < nb_lines; i++) {
		cout << setw(3) << i << " : " << setw(3) << opposite_line[i] << endl;
		}
	
	j = 0;
	for (i = 0; i < nb_points; i++) {
		a = opposite_point[i];
		if (a > i) {
			pts_sorted[j++] = i;
			pts_sorted[j++] = a;
			}
		}
	if (j != nb_points) {
		cout << "j != nb_points" << endl;
		exit(1);
		}
	cout << "i : pts_sorted[i]" << endl;
	for (i = 0; i < nb_points; i++) {
		cout << setw(3) << i << " : " << setw(3) << pts_sorted[i] << endl;
		}
	j = 0;
	for (i = 0; i < nb_lines; i++) {
		a = opposite_line[i];
		if (a > i) {
			lines_sorted[j++] = i;
			lines_sorted[j++] = a;
			}
		}
	if (j != nb_lines) {
		cout << "j != nb_lines" << endl;
		exit(1);
		}
	cout << "i : lines_sorted[i]" << endl;
	for (i = 0; i < nb_lines; i++) {
		cout << setw(3) << i << " : " << setw(3) << lines_sorted[i] << endl;
		}
	for (i = 0; i < nb_points; i++) {
		ii = pts_sorted[i];
		for (j = 0; j < nb_lines; j++) {
			jj = lines_sorted[j];
			Mtx2[i * nb_lines + j] = Mtx[ii * nb_lines + jj];
			}
		}
	cout << "reordered incidence matrix:" << endl;
	
	print_integer_matrix_width(cout, Mtx2, nb_points, nb_lines, nb_lines, 1);
	{
	incidence_structure Inc;
	char fname[1000];

	sprintf(fname, "HamG_%d_2.inc", n);
	
	Inc.init_by_matrix(nb_points, nb_lines, Mtx2, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}
	int *Mtx3;
	
	nb_points_folded = nb_points / 2;
	nb_lines_folded = nb_lines / 2;
	Mtx3 = NEW_int(nb_points_folded * nb_lines_folded);
	for (i = 0; i < nb_points_folded * nb_lines_folded; i++) {
		Mtx3[i] = 0;
		}
	for (i = 0; i < nb_points_folded; i++) {
		ii = 2 * i;
		for (j = 0; j < nb_lines_folded; j++) {
			jj = 2 * j;
			a = Mtx2[ii * nb_lines + jj];
			a += Mtx2[ii * nb_lines + jj + 1];
			a += Mtx2[(ii + 1) * nb_lines + jj];
			a += Mtx2[(ii + 1) * nb_lines + jj + 1];

			if (a) {
				Mtx3[i * nb_lines_folded + j] = 1;
				}
			}
		}
	cout << "folded incidence matrix:" << endl;
	
	print_integer_matrix_width(cout, Mtx3, nb_points_folded, nb_lines_folded, nb_lines_folded, 1);
	{
	incidence_structure Inc;
	char fname[1000];

	sprintf(fname, "HamG_%d_2.inc", n);
	
	Inc.init_by_matrix(nb_points_folded, nb_lines_folded, Mtx2, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}
	

	nb_points_folded = nb_points / 2;
	nb_lines_folded = nb_lines / 2;
	nb_planes_folded = nb_planes / 2;
	nb_solids_folded = nb_solids / 2;
	cout << "nb_points_folded=" << nb_points_folded << endl;
	cout << "nb_lines_folded=" << nb_lines_folded << endl;
	
	FREE_int(x);
	FREE_int(y);
	FREE_int(Mtx);
	FREE_int(Mtx2);
	FREE_int(Mtx3);
	FREE_int(opposite_point);
	FREE_int(opposite_line);
	FREE_int(pts_sorted);
	FREE_int(lines_sorted);
}

int point_rank(int *x)
{
	int rk;
	geometry_global Gg;
	
	Gg.AG_element_rank(2, x, 1, n, rk);
	return rk;
}

void point_unrank(int *x, int rk)
{
	geometry_global Gg;

	Gg.AG_element_unrank(2, x, 1, n, rk);
}

int line_rank(int *x, int b_1, int verbose_level)
{
	int *y;
	int rk, rk1, co_rank;
	geometry_global Gg;
	
	x[b_1] = 0;
	y = NEW_int(n);
	co_rank = b_1;
	compress1(x, y, b_1);
	Gg.AG_element_rank(2, y, 1, n - 1, rk1);
	rk = rk1 * n + co_rank;
	FREE_int(y);
	return rk;
}

void line_unrank(int rk, int *x, int &b_1, int verbose_level)
{
	int *y;
	int rk1, co_rank;
	geometry_global Gg;
	
	y = NEW_int(n);
	co_rank = rk % n;
	rk1 = rk / n;
	b_1 = co_rank;
	Gg.AG_element_unrank(2, y, 1, n - 1, rk1);
	expand1(x, y, b_1);
	x[b_1] = 0;
	FREE_int(y);
}

int plane_rank(int *x, int b_1, int b_2, int verbose_level)
{
	int *y;
	int rk, rk1, co_rank;
	int n2;
	int subset[2];
	geometry_global Gg;
	combinatorics_domain Combi;
	
	n2 = n * (n - 1) / 2;
	x[b_1] = 0;
	x[b_2] = 0;
	subset[0] = b_1;
	subset[1] = b_2;
	y = NEW_int(n);
	co_rank = Combi.rank_k_subset(subset, n, 2);
	compress2(x, y, b_1, b_2);
	Gg.AG_element_rank(2, y, 1, n - 2, rk1);
	rk = rk1 * n2 + co_rank;
	FREE_int(y);
	return rk;
}

void plane_unrank(int rk, int *x, int &b_1, int &b_2, int verbose_level)
{
	int *y;
	int rk1, co_rank;
	int n2;
	int subset[2];
	geometry_global Gg;
	combinatorics_domain Combi;

	n2 = n * (n - 1) / 2;
	
	y = NEW_int(n);
	co_rank = rk % n2;
	rk1 = rk / n2;
	Combi.unrank_k_subset(co_rank, subset, n, 2);
	b_1 = subset[0];
	b_2 = subset[1];
	Gg.AG_element_unrank(2, y, 1, n - 2, rk1);
	expand2(x, y, b_1, b_2);
	x[b_1] = 0;
	x[b_2] = 0;
	FREE_int(y);
}

int solid_rank(int *x, int b_1, int b_2, int b_3, int verbose_level)
{
	int *y;
	int rk, rk1, co_rank;
	int n3;
	int subset[3];
	geometry_global Gg;
	combinatorics_domain Combi;

	n3 = n * (n - 1) * (n - 2) / 6;
	x[b_1] = 0;
	x[b_2] = 0;
	x[b_3] = 0;
	subset[0] = b_1;
	subset[1] = b_2;
	subset[2] = b_3;
	y = NEW_int(n);
	co_rank = Combi.rank_k_subset(subset, n, 3);
	compress3(x, y, b_1, b_2, b_3);
	Gg.AG_element_rank(2, y, 1, n - 3, rk1);
	rk = rk1 * n3 + co_rank;
	FREE_int(y);
	return rk;
}

void solid_unrank(int rk, int *x, int &b_1, int &b_2, int &b_3,
		int verbose_level)
{
	int *y;
	int rk1, co_rank;
	int n3;
	int subset[3];
	geometry_global Gg;
	combinatorics_domain Combi;

	n3 = n * (n - 1) * (n - 2) / 6;
	
	y = NEW_int(n);
	co_rank = rk % n3;
	rk1 = rk / n3;
	Combi.unrank_k_subset(co_rank, subset, n, 3);
	b_1 = subset[0];
	b_2 = subset[1];
	b_3 = subset[2];
	Gg.AG_element_unrank(2, y, 1, n - 3, rk1);
	expand3(x, y, b_1, b_2, b_3);
	x[b_1] = 0;
	x[b_2] = 0;
	x[b_3] = 0;

	FREE_int(y);
}

int line_vertex_pair_rank(int *x, int b_1, int c_1, int verbose_level)
{
	int rk, rk1, co_rank;

	rk1 = line_rank(x, b_1, verbose_level);
	co_rank = c_1;
	rk = rk1 * 2 + co_rank;
	return rk;
}

void line_vertex_pair_unrank(int rk,
		int *x, int &b_1, int &c_1, int verbose_level)
{
	int rk1, co_rank;

	co_rank = rk % 2;
	rk1 = rk / 2;
	line_unrank(rk1, x, b_1, verbose_level);
	c_1 = co_rank;
}

int solid_diagonal_pair_rank(int *x,
		int b_1, int b_2, int b_3, int c_1, int c_2, int c_3,
		int verbose_level)
{
	int rk, rk1, co_rank;
	int c[3];

	c[0] = c_1;
	c[1] = c_2;
	c[2] = c_3;
	co_rank = low_weight_3vec_rank(c);
	rk1 = solid_rank(x, b_1, b_2, b_3, verbose_level);
	rk = rk1 * 4 + co_rank;
	return rk;
}

void solid_diagonal_pair_unrank(int rk, int *x, int &b_1, int &b_2, int &b_3, 
	int &c_1, int &c_2, int &c_3, int verbose_level)
{
	int rk1, co_rank;
	int c[3];

	co_rank = rk % 4;
	rk1 = rk / 4;
	low_weight_3vec_unrank(co_rank, c);
	c_1 = c[0];
	c_2 = c[1];
	c_3 = c[2];
	solid_unrank(rk1, x, b_1, b_2, b_3, verbose_level);
}

int low_weight_3vec_rank(int *x)
{
	representative_under_folding(x, 3);
	if (x[0] == 0) {
		if (x[1] == 0) {
			if (x[2] == 0) {
				return 0;
				}
			else {
				return 3;
				}
			}
		else {
			return 2;
			}
		}
	else {
		return 1;
		}
}

void low_weight_3vec_unrank(int rk, int *x)
{
	int i;

	for (i = 0; i < 3; i++) {
		x[i] = 0;
		}
	if (rk == 0) {
		return;
		}
	if (rk >= 4) {
		cout << "low_weight_3vec_unrank rk >= 4" << endl;
		exit(1);
		}
	x[rk - 1] = 1;
}



void compress1(int *x, int *x_compressed, int b_1)
{
	int i;

	for (i = 0; i < b_1; i++) {
		x_compressed[i] = x[i];
		}
	for (i = b_1 + 1; i < n; i++) {
		x_compressed[i - 1] = x[i];
		}
	
}

void expand1(int *x, int *x_compressed, int b_1)
{
	int i;

	for (i = 0; i < b_1; i++) {
		x[i] = x_compressed[i];
		}
	for (i = b_1 + 1; i < n; i++) {
		x[i] = x_compressed[i - 1];
		}
}

void compress2(int *x, int *x_compressed, int b_1, int b_2)
{
	int i;

	for (i = 0; i < b_1; i++) {
		x_compressed[i] = x[i];
		}
	for (i = b_1 + 1; i < b_2; i++) {
		x_compressed[i - 1] = x[i];
		}
	for (i = b_2 + 1; i < n; i++) {
		x_compressed[i - 2] = x[i];
		}
	
}

void expand2(int *x, int *x_compressed, int b_1, int b_2)
{
	int i;

	for (i = 0; i < b_1; i++) {
		x[i] = x_compressed[i];
		}
	for (i = b_1 + 1; i < b_2; i++) {
		x[i] = x_compressed[i - 1];
		}
	for (i = b_2 + 1; i < n; i++) {
		x[i] = x_compressed[i - 2];
		}
}

void compress3(int *x, int *x_compressed, int b_1, int b_2, int b_3)
{
	int i;

	for (i = 0; i < b_1; i++) {
		x_compressed[i] = x[i];
		}
	for (i = b_1 + 1; i < b_2; i++) {
		x_compressed[i - 1] = x[i];
		}
	for (i = b_2 + 1; i < b_3; i++) {
		x_compressed[i - 2] = x[i];
		}
	for (i = b_3 + 1; i < n; i++) {
		x_compressed[i - 3] = x[i];
		}
	
}

void expand3(int *x, int *x_compressed, int b_1, int b_2, int b_3)
{
	int i;

	for (i = 0; i < b_1; i++) {
		x[i] = x_compressed[i];
		}
	for (i = b_1 + 1; i < b_2; i++) {
		x[i] = x_compressed[i - 1];
		}
	for (i = b_2 + 1; i < b_3; i++) {
		x[i] = x_compressed[i - 2];
		}
	for (i = b_3 + 1; i < n; i++) {
		x[i] = x_compressed[i - 3];
		}
}

int is_incident_point_line(int *v_point, int *v_line, int b_1)
{
	int *c;
	int i;
	int ret = TRUE;

	c = NEW_int(n);
	for (i = 0; i < n; i++) {
		c[i] = TRUE;
		}
	c[b_1] = FALSE;
	for (i = 0; i < n; i++) {
		if (!c[i])
			continue;
		if (v_point[i] != v_line[i]) {
			ret = FALSE;
			break;
			}
		}
	FREE_int(c);
	return ret;
}

int is_incident_line_solid(int *v_line, int b_1, int *v_solid, int c_1, int c_2, int c_3)
{
	int *c;
	int i;
	int ret = TRUE;

	c = NEW_int(n);
	for (i = 0; i < n; i++) {
		c[i] = TRUE;
		}
	c[c_1] = FALSE;
	c[c_2] = FALSE;
	c[c_3] = FALSE;
	if (c[b_1]) {
		ret = FALSE;
		goto done;
		}
	for (i = 0; i < n; i++) {
		if (!c[i])
			continue;
		if (v_line[i] != v_solid[i]) {
			ret = FALSE;
			break;
			}
		}

done:
	FREE_int(c);
	return ret;
}


int is_incident_point_edge_solid(int *v_line, int e_1, 
	int *v_point, int *v_solid, int b_1, int b_2, int b_3)
{
	int i;
	
	if (is_incident_point_line(v_point, v_line, e_1) && 
		is_incident_line_solid(v_line, e_1, v_solid, b_1, b_2, b_3)) {
		return TRUE;
		}
	int ret = FALSE;
	int *w_point;
	int *w_solid;
	w_point = NEW_int(n);
	w_solid = NEW_int(n);

	for (i = 0; i < n; i++) {
		w_point[i] = v_point[i];
		w_solid[i] = v_solid[i];
		}
	opposite_under_folding_solid(w_point, b_1, b_2, b_3);
	opposite_under_folding_solid(w_solid, b_1, b_2, b_3);
	if (is_incident_point_line(w_point, v_line, e_1) && 
		is_incident_line_solid(v_line, e_1, w_solid, b_1, b_2, b_3)) {
		ret = TRUE;
		}
		
	FREE_int(w_point);
	FREE_int(w_solid);
	return ret;
#if 0
	int *c;
	int i;
	int *xx;
	int *yy;
	int ret = TRUE;

	xx = NEW_int(n);
	yy = NEW_int(n);
	c = NEW_int(n);
	for (i = 0; i < n; i++) {
		xx[i] = v_point[i];
		yy[i] = v_solid[i];
		}
	//representative_under_folding(xx, n);
	//representative_under_folding_solid(yy, b_1, b_2, b_3);
	cout << "yy=solid";
	int_vec_print(cout, yy, n);
	cout << " b_1=" << b_1;
	cout << " b_2=" << b_2;
	cout << " b_3=" << b_3;
	cout << endl;
	for (i = 0; i < n; i++) {
		c[i] = TRUE;
		}
	c[b_1] = FALSE;
	c[b_2] = FALSE;
	c[b_3] = FALSE;
	for (i = 0; i < n; i++) {
		if (!c[i])
			continue;
		if (xx[i] != yy[i]) {
			ret = FALSE;
			break;
			}
		}

#if 0
	if (!ret) {
		ret = TRUE; // second chance
		opposite_under_folding_solid(yy, b_1, b_2, b_3);
		cout << "opposite:";
		int_vec_print(cout, yy, n);
		cout << " b_1=" << b_1;
		cout << " b_2=" << b_2;
		cout << " b_3=" << b_3;
		cout << endl;
		for (i = 0; i < n; i++) {
			if (!c[i])
				continue;
			if (xx[i] != yy[i]) {
				ret = FALSE;
				break;
				}
			}
		}
#endif

	FREE_int(c);
	FREE_int(xx);
	FREE_int(yy);
	return ret;
#endif

}

void representative_under_folding(int *x, int len)
{
	int i, w;

	w = 0;
	for (i = 0; i < len; i++) {
		if (x[i]) {
			w++;
			}
		}
	if (w > (len >> 1)) {
		invert(x, len);
		}
}

void representative_under_folding_line(int *x, int b_1)
{
	int *y;

	y = NEW_int(n);
	compress1(x, y, b_1);
	representative_under_folding(y, n - 1);
	expand1(x, y, b_1);
	FREE_int(y);
}

void representative_under_folding_plane(int *x, int b_1, int b_2)
{
	int *y;

	y = NEW_int(n);
	compress2(x, y, b_1, b_2);
	representative_under_folding(y, n - 2);
	expand2(x, y, b_1, b_2);
	FREE_int(y);
}

void representative_under_folding_solid(int *x, int b_1, int b_2, int b_3)
{
	int *y;

	y = NEW_int(n);
	compress3(x, y, b_1, b_2, b_3);
	representative_under_folding(y, n - 3);
	expand3(x, y, b_1, b_2, b_3);
	FREE_int(y);
}

void opposite_under_folding_line(int *x, int b_1)
{
	int *y;

	y = NEW_int(n);
	compress1(x, y, b_1);
	invert(y, n - 1);
	expand1(x, y, b_1);
	FREE_int(y);
}

void opposite_under_folding_plane(int *x, int b_1, int b_2)
{
	int *y;

	y = NEW_int(n);
	compress2(x, y, b_1, b_2);
	invert(y, n - 2);
	expand2(x, y, b_1, b_2);
	FREE_int(y);
}

void opposite_under_folding_solid(int *x, int b_1, int b_2, int b_3)
{
	int *y;

	y = NEW_int(n);
	compress3(x, y, b_1, b_2, b_3);
	invert(y, n - 3);
	expand3(x, y, b_1, b_2, b_3);
	FREE_int(y);
}


void invert(int *x, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		if (x[i]) {
			x[i] = 0;
			}
		else {
			x[i] = 1;
			}
		}
}

