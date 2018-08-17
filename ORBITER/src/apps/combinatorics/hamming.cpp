// hamming.C
// 
// Anton Betten
// Dec 9, 2010
//
//
// 
// 
//

#include "orbiter.h"

// global data:

INT t0; // the system time when the program started
INT n;
INT nb_points;
INT nb_lines;
INT nb_planes;
INT nb_solids;
INT nb_points_folded;
INT nb_lines_folded;
INT nb_planes_folded;
INT nb_solids_folded;

INT nb_BLOCKS;
INT nb_POINTS;


void create_object(INT verbose_level);
void print_solid(INT *x, INT b_1, INT b_2, INT b_3, INT c_1, INT c_2, INT c_3);
void print_line(INT *x, INT d_1, INT e_1);
INT is_adjacent(INT *v_solid, INT b_1, INT b_2, INT b_3, INT c_1, INT c_2, INT c_3, INT *v_line, INT d_1, INT e_1);
void create_geometry(INT verbose_level);
INT point_rank(INT *x);
void point_unrank(INT *x, INT rk);
INT line_rank(INT *x, INT b_1, INT verbose_level);
void line_unrank(INT rk, INT *x, INT &b_1, INT verbose_level);
INT plane_rank(INT *x, INT b_1, INT b_2, INT verbose_level);
void plane_unrank(INT rk, INT *x, INT &b_1, INT &b_2, INT verbose_level);
INT solid_rank(INT *x, INT b_1, INT b_2, INT b_3, INT verbose_level);
void solid_unrank(INT rk, INT *x, INT &b_1, INT &b_2, INT &b_3, INT verbose_level);
INT line_vertex_pair_rank(INT *x, INT b_1, INT c_1, INT verbose_level);
void line_vertex_pair_unrank(INT rk, INT *x, INT &b_1, INT &c_1, INT verbose_level);
INT solid_diagonal_pair_rank(INT *x, INT b_1, INT b_2, INT b_3, INT c_1, INT c_2, INT c_3, INT verbose_level);
void solid_diagonal_pair_unrank(INT rk, INT *x, INT &b_1, INT &b_2, INT &b_3, 
	INT &c_1, INT &c_2, INT &c_3, INT verbose_level);
INT low_weight_3vec_rank(INT *x);
void low_weight_3vec_unrank(INT rk, INT *x);
void compress1(INT *x, INT *x_compressed, INT b_1);
void expand1(INT *x, INT *x_compressed, INT b_1);
void compress2(INT *x, INT *x_compressed, INT b_1, INT b_2);
void expand2(INT *x, INT *x_compressed, INT b_1, INT b_2);
void compress3(INT *x, INT *x_compressed, INT b_1, INT b_2, INT b_3);
void expand3(INT *x, INT *x_compressed, INT b_1, INT b_2, INT b_3);
INT is_incident_point_line(INT *v_point, INT *v_line, INT b_1);
INT is_incident_line_solid(INT *v_line, INT b_1, INT *v_solid, INT c_1, INT c_2, INT c_3);
INT is_incident_point_edge_solid(INT *v_line, INT e_1, 
	INT *v_point, INT *v_solid, INT b_1, INT b_2, INT b_3);
void representative_under_folding(INT *x, INT len);
void representative_under_folding_line(INT *x, INT b_1);
void representative_under_folding_plane(INT *x, INT b_1, INT b_2);
void representative_under_folding_solid(INT *x, INT b_1, INT b_2, INT b_3);
void opposite_under_folding_line(INT *x, INT b_1);
void opposite_under_folding_plane(INT *x, INT b_1, INT b_2);
void opposite_under_folding_solid(INT *x, INT b_1, INT b_2, INT b_3);
void invert(INT *x, INT len);

int main(int argc, char **argv)
{
	INT verbose_level;
	INT i;
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
void create_object(INT verbose_level)
{
	INT *x;
	INT *y;
	INT *Points;
	INT *Blocks;
	INT b_1, b_2, b_3;
	INT coeff[3];
	INT i, j, h, a, b;
	INT POINT_width = n + 1;
	INT BLOCK_width = n + n + 3;
	
	cout << "create_object" << endl;
	cout << "nb_points_folded=" << nb_points_folded << endl;
	cout << "nb_lines_folded=" << nb_lines_folded << endl;
	x = NEW_INT(n);
	y = NEW_INT(n);
	nb_POINTS = nb_lines_folded;
	nb_BLOCKS = nb_points_folded * nb_solids_folded;
	
	Points = NEW_INT(nb_POINTS * POINT_width);
	Blocks = NEW_INT(nb_BLOCKS * BLOCK_width);
	cout << "nb_POINTS=" << nb_POINTS << endl;
	cout << "nb_BLOCKS=" << nb_BLOCKS << endl;

	for (i = 0; i < nb_POINTS; i++) {
		line_unrank(i, x, b_1, 0);
		representative_under_folding_line(x, b_1);
		for (h = 0; h < n; h++) {
			Points[i * POINT_width + h] = x[h];
			}
		Points[i * POINT_width + n] = b_1;
		}
	cout << "POINTS:" << endl;
	print_integer_matrix_width(cout, Points, nb_POINTS, POINT_width, POINT_width, 1);

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

	INT *M1;
	INT *v_line;
	INT *v_point;
	INT *v_solid;
	INT e_1;
	//INT a;

	M1 = NEW_INT(nb_POINTS * nb_BLOCKS);
	for (i = 0; i < nb_POINTS * nb_BLOCKS; i++) {
		M1[i] = 0;
		}
	for (i = 0; i < nb_POINTS; i++) {
		v_line = Points + i * POINT_width;
		e_1 = Points[i * POINT_width + n];
		
		cout << "i=" << i << " : line ";
		INT_vec_print(cout, v_line, n);
		cout << "e_1=" << e_1 << endl;
		
		for (j = 0; j < nb_BLOCKS; j++) {
			v_point = Blocks + j * BLOCK_width;
			v_solid = Blocks + j * BLOCK_width + n;
			b_1 = Blocks[j * BLOCK_width + n + n + 0];
			b_2 = Blocks[j * BLOCK_width + n + n + 1];
			b_3 = Blocks[j * BLOCK_width + n + n + 2];

			cout << "j=" << j << " : point ";
			INT_vec_print(cout, v_point, n);
			cout << " solid ";
			INT_vec_print(cout, v_solid, n);
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
	
	print_integer_matrix_width(cout, M1, nb_POINTS, nb_BLOCKS, nb_BLOCKS, 1);
	
	INT *AAt;

	AAt = NEW_INT(nb_POINTS * nb_POINTS);
	for (i = 0; i < nb_POINTS; i++) {
		for (j = 0; j < nb_POINTS; j++) {
			a = 0;
			for (h = 0; h < nb_BLOCKS; h++) {
				a += M1[i * nb_BLOCKS + h] * M1[j * nb_BLOCKS + h];
				}
			AAt[i * nb_POINTS + j] = a;
			}
		}

	cout << "AAt:" << endl;
	
	print_integer_matrix_width(cout, AAt, nb_POINTS, nb_POINTS, nb_POINTS, 1);
	
	{
	incidence_structure Inc;
	BYTE fname[1000];

	sprintf(fname, "HamG_%ld_2.inc", n);
	
	Inc.init_by_matrix(nb_POINTS, nb_BLOCKS, M1, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}
	
	
	FREE_INT(x);
	FREE_INT(y);
	FREE_INT(M1);
	FREE_INT(AAt);
}
#else
void create_object(INT verbose_level)
{
	INT *Points;
	INT *Blocks;
	INT *x;
	INT *y;
	INT b_1, b_2, b_3;
	INT c_1, c_2, c_3;
	INT d_1, e_1;
	INT i, j, h, a, ii, jj;
	INT POINT_width = n + 6;
	INT BLOCK_width = n + 2;
	
	cout << "create_object" << endl;
	nb_POINTS = nb_solids * 4;
	nb_BLOCKS = nb_lines * 2;
	
	x = NEW_INT(n);
	y = NEW_INT(n);
	Points = NEW_INT(nb_POINTS * POINT_width);
	Blocks = NEW_INT(nb_BLOCKS * BLOCK_width);
	cout << "nb_POINTS=" << nb_POINTS << endl;
	cout << "nb_BLOCKS=" << nb_BLOCKS << endl;

	for (i = 0; i < nb_POINTS; i++) {
		solid_diagonal_pair_unrank(i, x, b_1, b_2, b_3, 
			c_1, c_2, c_3, 0);
		for (h = 0; h < n; h++) {
			Points[i * POINT_width + h] = x[h];
			}
		Points[i * POINT_width + n + 0] = b_1;
		Points[i * POINT_width + n + 1] = b_2;
		Points[i * POINT_width + n + 2] = b_3;
		Points[i * POINT_width + n + 3] = c_1;
		Points[i * POINT_width + n + 4] = c_2;
		Points[i * POINT_width + n + 5] = c_3;
		}
	cout << "POINTS:" << endl;
	print_integer_matrix_width(cout, Points, nb_POINTS, POINT_width, POINT_width, 1);


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

	INT *M1;
	INT *v_line;
	INT *v_solid;

	M1 = NEW_INT(nb_POINTS * nb_BLOCKS);
	for (i = 0; i < nb_POINTS * nb_BLOCKS; i++) {
		M1[i] = 0;
		}
	for (i = 0; i < nb_POINTS; i++) {
		v_solid = Points + i * POINT_width;
		b_1 = Points[i * POINT_width + n + 0];
		b_2 = Points[i * POINT_width + n + 1];
		b_3 = Points[i * POINT_width + n + 2];
		c_1 = Points[i * POINT_width + n + 3];
		c_2 = Points[i * POINT_width + n + 4];
		c_3 = Points[i * POINT_width + n + 5];
		
		
		for (j = 0; j < nb_BLOCKS; j++) {
			v_line = Blocks + j * BLOCK_width;
			d_1 = Blocks[j * BLOCK_width + n + 0];
			e_1 = Blocks[j * BLOCK_width + n + 1];

		
			
			a = is_adjacent(v_solid, b_1, b_2, b_3, c_1, c_2, c_3, v_line, d_1, e_1);

			//cout << "a=" << a << endl;
			if (a) {
				cout << "solid i=" << i << " ";
				print_solid(v_solid, b_1, b_2, b_3, c_1, c_2, c_3);
				//INT_vec_print(cout, v_solid, n);
				//cout << " b_1=" << b_1;
				//cout << " b_2=" << b_2;
				//cout << " b_3=" << b_3;
				//cout << " c_1=" << c_1;
				//cout << " c_2=" << c_2;
				//cout << " c_3=" << c_3;
				//cout << endl;
				cout << "and line j=" << j << " ";
				print_line(v_line, d_1, e_1);
				//INT_vec_print(cout, v_line, n);
				//cout << " d_1=" << d_1;
				//cout << " e_1=" << e_1;
				cout << " are adjacent" << endl;
				}
			M1[i * nb_BLOCKS + j] = a;
			}
		}
	cout << "incidence matrix:" << endl;
	
	print_integer_matrix_width(cout, M1, nb_POINTS, nb_BLOCKS, nb_BLOCKS, 1);
	
	INT *AAt;

	AAt = NEW_INT(nb_POINTS * nb_POINTS);
	for (i = 0; i < nb_POINTS; i++) {
		for (j = 0; j < nb_POINTS; j++) {
			a = 0;
			for (h = 0; h < nb_BLOCKS; h++) {
				a += M1[i * nb_BLOCKS + h] * M1[j * nb_BLOCKS + h];
				}
			AAt[i * nb_POINTS + j] = a;
			}
		}

	cout << "AAt:" << endl;
	
	print_integer_matrix_width(cout, AAt, nb_POINTS, nb_POINTS, nb_POINTS, 1);
	
	{
	incidence_structure Inc;
	BYTE fname[1000];

	sprintf(fname, "HamG_%ld_2.inc", n);
	
	Inc.init_by_matrix(nb_POINTS, nb_BLOCKS, M1, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}



	INT *opposite_point;
	INT *opposite_block;
	INT *pts_sorted;
	INT *blocks_sorted;
	INT *Mtx2;

	opposite_point = NEW_INT(nb_POINTS);
	opposite_block = NEW_INT(nb_BLOCKS);
	pts_sorted = NEW_INT(nb_POINTS);
	blocks_sorted = NEW_INT(nb_BLOCKS);
	Mtx2 = NEW_INT(nb_POINTS * nb_BLOCKS);
	
	for (i = 0; i < nb_POINTS; i++) {
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
	for (i = 0; i < nb_POINTS; i++) {
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
	for (i = 0; i < nb_POINTS; i++) {
		a = opposite_point[i];
		if (a > i) {
			pts_sorted[j++] = i;
			pts_sorted[j++] = a;
			}
		}
	if (j != nb_POINTS) {
		cout << "j != nb_POINTS" << endl;
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
	for (i = 0; i < nb_POINTS; i++) {
		ii = pts_sorted[i];
		for (j = 0; j < nb_BLOCKS; j++) {
			jj = blocks_sorted[j];
			Mtx2[i * nb_BLOCKS + j] = M1[ii * nb_BLOCKS + jj];
			}
		}
	cout << "reordered incidence matrix:" << endl;
	
	print_integer_matrix_width(cout, Mtx2, nb_POINTS, nb_BLOCKS, nb_BLOCKS, 1);
	{
	incidence_structure Inc;
	BYTE fname[1000];

	sprintf(fname, "HamG_%ld_2.inc", n);
	
	Inc.init_by_matrix(nb_POINTS, nb_BLOCKS, Mtx2, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}


	INT *Mtx3;
	INT nb_POINTS_folded, nb_BLOCKS_folded;
	
	nb_POINTS_folded = nb_POINTS / 2;
	nb_BLOCKS_folded = nb_BLOCKS / 2;
	Mtx3 = NEW_INT(nb_POINTS_folded * nb_BLOCKS_folded);
	for (i = 0; i < nb_POINTS_folded * nb_BLOCKS_folded; i++) {
		Mtx3[i] = 0;
		}
	for (i = 0; i < nb_POINTS_folded; i++) {
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
	
	print_integer_matrix_width(cout, Mtx3, nb_POINTS_folded, nb_BLOCKS_folded, nb_BLOCKS_folded, 1);


	INT *FFt;

	FFt = NEW_INT(nb_POINTS_folded * nb_POINTS_folded);
	for (i = 0; i < nb_POINTS_folded; i++) {
		for (j = 0; j < nb_POINTS_folded; j++) {
			a = 0;
			for (h = 0; h < nb_BLOCKS_folded; h++) {
				a += Mtx3[i * nb_BLOCKS_folded + h] * Mtx3[j * nb_BLOCKS_folded + h];
				}
			FFt[i * nb_POINTS_folded + j] = a;
			}
		}

	cout << "FFt:" << endl;
	
	print_integer_matrix_width(cout, FFt, nb_POINTS_folded, nb_POINTS_folded, nb_POINTS_folded, 1);
	


	{
	incidence_structure Inc;
	BYTE fname[1000];

	sprintf(fname, "HamG_%ld_2.inc", n);
	
	Inc.init_by_matrix(nb_POINTS_folded, nb_BLOCKS_folded, Mtx3, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}
	

	
	FREE_INT(x);
	FREE_INT(y);
	FREE_INT(M1);
	FREE_INT(AAt);
}

#endif

void print_solid(INT *x, INT b_1, INT b_2, INT b_3, INT c_1, INT c_2, INT c_3)
{
	INT *w;
	INT c[3];
	INT i;
	
	w = NEW_INT(n);
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
	INT_vec_print(cout, w, n);
	invert(c, 3);
	w[b_1] = c[0];
	w[b_2] = c[1];
	w[b_3] = c[2];
	INT_vec_print(cout, w, n);
	FREE_INT(w);
}

void print_line(INT *x, INT d_1, INT e_1)
{
	INT *w;
	INT i;
	
	w = NEW_INT(n);
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
	INT_vec_print(cout, w, n);
	FREE_INT(w);
}

INT is_adjacent(INT *v_solid, INT b_1, INT b_2, INT b_3, INT c_1, INT c_2, INT c_3, INT *v_line, INT d_1, INT e_1)
{
	INT *x;
	INT *y;
	INT h;
	INT c[3];
	INT ret = FALSE;
	
	x = NEW_INT(n);
	y = NEW_INT(n);
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
		if (INT_vec_compare(x, y, n) == 0) {
			ret = TRUE;
			goto done;
			}
		invert(c, 3);
		x[b_1] = c[0];
		x[b_2] = c[1];
		x[b_3] = c[2];
		if (INT_vec_compare(x, y, n) == 0) {
			ret = TRUE;
			goto done;
			}
		}

done:
	FREE_INT(x);
	FREE_INT(y);
	return ret;
}

void create_geometry(INT verbose_level)
{
	INT i, j, h, a, b_1, b_2, b_3, ii, jj;
	INT *x;
	INT *y;
	
	x = NEW_INT(n);
	y = NEW_INT(n);
	nb_points = i_power_j(2, n);
	nb_lines = i_power_j(2, n - 1) * n;
	nb_planes = i_power_j(2, n - 2) * n * (n - 1) / 2;
	nb_solids = i_power_j(2, n - 3) * n * (n - 1) * (n - 2) / 6;
	
	cout << "nb_points=" << nb_points << endl;
	cout << "nb_lines=" << nb_lines << endl;
	cout << "nb_planes=" << nb_planes << endl;
	cout << "nb_solids=" << nb_solids << endl;
	


	cout << "lines:" << endl;
	for (i = 0; i < nb_lines; i++) {
		line_unrank(i, x, b_1, 0);
		cout << setw(3) << i << " : [";
		INT_vec_print(cout, x, n);
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
		INT_vec_print(cout, x, n);
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
		INT_vec_print(cout, x, n);
		cout << " : " << b_1 << "," << b_2 << "," << b_3 << "]" << endl;
		j = solid_rank(x, b_1, b_2, b_3, 0);
		if (j != i) {
			cout << "j != i" << endl;
			exit(1);
			}
		}

	INT *Mtx;
	INT *Mtx2;

	Mtx = NEW_INT(nb_points * nb_lines);
	Mtx2 = NEW_INT(nb_points * nb_lines);

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
	BYTE fname[1000];

	sprintf(fname, "HamG_%ld_2.inc", n);
	
	Inc.init_by_matrix(nb_points, nb_lines, Mtx, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}
	
	INT *opposite_point;
	INT *opposite_line;
	INT *pts_sorted;
	INT *lines_sorted;

	opposite_point = NEW_INT(nb_points);
	opposite_line = NEW_INT(nb_lines);
	pts_sorted = NEW_INT(nb_points);
	lines_sorted = NEW_INT(nb_lines);

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
	BYTE fname[1000];

	sprintf(fname, "HamG_%ld_2.inc", n);
	
	Inc.init_by_matrix(nb_points, nb_lines, Mtx2, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}
	INT *Mtx3;
	
	nb_points_folded = nb_points / 2;
	nb_lines_folded = nb_lines / 2;
	Mtx3 = NEW_INT(nb_points_folded * nb_lines_folded);
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
	BYTE fname[1000];

	sprintf(fname, "HamG_%ld_2.inc", n);
	
	Inc.init_by_matrix(nb_points_folded, nb_lines_folded, Mtx2, 0 /*verbose_level*/);
	Inc.save_inc_file(fname);
	}
	

	nb_points_folded = nb_points / 2;
	nb_lines_folded = nb_lines / 2;
	nb_planes_folded = nb_planes / 2;
	nb_solids_folded = nb_solids / 2;
	cout << "nb_points_folded=" << nb_points_folded << endl;
	cout << "nb_lines_folded=" << nb_lines_folded << endl;
	
	FREE_INT(x);
	FREE_INT(y);
	FREE_INT(Mtx);
	FREE_INT(Mtx2);
	FREE_INT(Mtx3);
	FREE_INT(opposite_point);
	FREE_INT(opposite_line);
	FREE_INT(pts_sorted);
	FREE_INT(lines_sorted);
}

INT point_rank(INT *x)
{
	INT rk;
	
	AG_element_rank(2, x, 1, n, rk);
	return rk;
}

void point_unrank(INT *x, INT rk)
{
	AG_element_unrank(2, x, 1, n, rk);
}

INT line_rank(INT *x, INT b_1, INT verbose_level)
{
	INT *y;
	INT rk, rk1, co_rank;
	
	x[b_1] = 0;
	y = NEW_INT(n);
	co_rank = b_1;
	compress1(x, y, b_1);
	AG_element_rank(2, y, 1, n - 1, rk1);
	rk = rk1 * n + co_rank;
	FREE_INT(y);
	return rk;
}

void line_unrank(INT rk, INT *x, INT &b_1, INT verbose_level)
{
	INT *y;
	INT rk1, co_rank;
	
	y = NEW_INT(n);
	co_rank = rk % n;
	rk1 = rk / n;
	b_1 = co_rank;
	AG_element_unrank(2, y, 1, n - 1, rk1);
	expand1(x, y, b_1);
	x[b_1] = 0;
	FREE_INT(y);
}

INT plane_rank(INT *x, INT b_1, INT b_2, INT verbose_level)
{
	INT *y;
	INT rk, rk1, co_rank;
	INT n2;
	INT subset[2];
	
	n2 = n * (n - 1) / 2;
	x[b_1] = 0;
	x[b_2] = 0;
	subset[0] = b_1;
	subset[1] = b_2;
	y = NEW_INT(n);
	co_rank = rank_k_subset(subset, n, 2);
	compress2(x, y, b_1, b_2);
	AG_element_rank(2, y, 1, n - 2, rk1);
	rk = rk1 * n2 + co_rank;
	FREE_INT(y);
	return rk;
}

void plane_unrank(INT rk, INT *x, INT &b_1, INT &b_2, INT verbose_level)
{
	INT *y;
	INT rk1, co_rank;
	INT n2;
	INT subset[2];

	n2 = n * (n - 1) / 2;
	
	y = NEW_INT(n);
	co_rank = rk % n2;
	rk1 = rk / n2;
	unrank_k_subset(co_rank, subset, n, 2);
	b_1 = subset[0];
	b_2 = subset[1];
	AG_element_unrank(2, y, 1, n - 2, rk1);
	expand2(x, y, b_1, b_2);
	x[b_1] = 0;
	x[b_2] = 0;
	FREE_INT(y);
}

INT solid_rank(INT *x, INT b_1, INT b_2, INT b_3, INT verbose_level)
{
	INT *y;
	INT rk, rk1, co_rank;
	INT n3;
	INT subset[3];
	
	n3 = n * (n - 1) * (n - 2) / 6;
	x[b_1] = 0;
	x[b_2] = 0;
	x[b_3] = 0;
	subset[0] = b_1;
	subset[1] = b_2;
	subset[2] = b_3;
	y = NEW_INT(n);
	co_rank = rank_k_subset(subset, n, 3);
	compress3(x, y, b_1, b_2, b_3);
	AG_element_rank(2, y, 1, n - 3, rk1);
	rk = rk1 * n3 + co_rank;
	FREE_INT(y);
	return rk;
}

void solid_unrank(INT rk, INT *x, INT &b_1, INT &b_2, INT &b_3, INT verbose_level)
{
	INT *y;
	INT rk1, co_rank;
	INT n3;
	INT subset[3];

	n3 = n * (n - 1) * (n - 2) / 6;
	
	y = NEW_INT(n);
	co_rank = rk % n3;
	rk1 = rk / n3;
	unrank_k_subset(co_rank, subset, n, 3);
	b_1 = subset[0];
	b_2 = subset[1];
	b_3 = subset[2];
	AG_element_unrank(2, y, 1, n - 3, rk1);
	expand3(x, y, b_1, b_2, b_3);
	x[b_1] = 0;
	x[b_2] = 0;
	x[b_3] = 0;

	FREE_INT(y);
}

INT line_vertex_pair_rank(INT *x, INT b_1, INT c_1, INT verbose_level)
{
	INT rk, rk1, co_rank;

	rk1 = line_rank(x, b_1, verbose_level);
	co_rank = c_1;
	rk = rk1 * 2 + co_rank;
	return rk;
}

void line_vertex_pair_unrank(INT rk, INT *x, INT &b_1, INT &c_1, INT verbose_level)
{
	INT rk1, co_rank;

	co_rank = rk % 2;
	rk1 = rk / 2;
	line_unrank(rk1, x, b_1, verbose_level);
	c_1 = co_rank;
}

INT solid_diagonal_pair_rank(INT *x, INT b_1, INT b_2, INT b_3, INT c_1, INT c_2, INT c_3, INT verbose_level)
{
	INT rk, rk1, co_rank;
	INT c[3];

	c[0] = c_1;
	c[1] = c_2;
	c[2] = c_3;
	co_rank = low_weight_3vec_rank(c);
	rk1 = solid_rank(x, b_1, b_2, b_3, verbose_level);
	rk = rk1 * 4 + co_rank;
	return rk;
}

void solid_diagonal_pair_unrank(INT rk, INT *x, INT &b_1, INT &b_2, INT &b_3, 
	INT &c_1, INT &c_2, INT &c_3, INT verbose_level)
{
	INT rk1, co_rank;
	INT c[3];

	co_rank = rk % 4;
	rk1 = rk / 4;
	low_weight_3vec_unrank(co_rank, c);
	c_1 = c[0];
	c_2 = c[1];
	c_3 = c[2];
	solid_unrank(rk1, x, b_1, b_2, b_3, verbose_level);
}

INT low_weight_3vec_rank(INT *x)
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

void low_weight_3vec_unrank(INT rk, INT *x)
{
	INT i;

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



void compress1(INT *x, INT *x_compressed, INT b_1)
{
	INT i;

	for (i = 0; i < b_1; i++) {
		x_compressed[i] = x[i];
		}
	for (i = b_1 + 1; i < n; i++) {
		x_compressed[i - 1] = x[i];
		}
	
}

void expand1(INT *x, INT *x_compressed, INT b_1)
{
	INT i;

	for (i = 0; i < b_1; i++) {
		x[i] = x_compressed[i];
		}
	for (i = b_1 + 1; i < n; i++) {
		x[i] = x_compressed[i - 1];
		}
}

void compress2(INT *x, INT *x_compressed, INT b_1, INT b_2)
{
	INT i;

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

void expand2(INT *x, INT *x_compressed, INT b_1, INT b_2)
{
	INT i;

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

void compress3(INT *x, INT *x_compressed, INT b_1, INT b_2, INT b_3)
{
	INT i;

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

void expand3(INT *x, INT *x_compressed, INT b_1, INT b_2, INT b_3)
{
	INT i;

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

INT is_incident_point_line(INT *v_point, INT *v_line, INT b_1)
{
	INT *c;
	INT i;
	INT ret = TRUE;

	c = NEW_INT(n);
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
	FREE_INT(c);
	return ret;
}

INT is_incident_line_solid(INT *v_line, INT b_1, INT *v_solid, INT c_1, INT c_2, INT c_3)
{
	INT *c;
	INT i;
	INT ret = TRUE;

	c = NEW_INT(n);
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
	FREE_INT(c);
	return ret;
}


INT is_incident_point_edge_solid(INT *v_line, INT e_1, 
	INT *v_point, INT *v_solid, INT b_1, INT b_2, INT b_3)
{
	INT i;
	
	if (is_incident_point_line(v_point, v_line, e_1) && 
		is_incident_line_solid(v_line, e_1, v_solid, b_1, b_2, b_3)) {
		return TRUE;
		}
	INT ret = FALSE;
	INT *w_point;
	INT *w_solid;
	w_point = NEW_INT(n);
	w_solid = NEW_INT(n);

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
		
	FREE_INT(w_point);
	FREE_INT(w_solid);
	return ret;
#if 0
	INT *c;
	INT i;
	INT *xx;
	INT *yy;
	INT ret = TRUE;

	xx = NEW_INT(n);
	yy = NEW_INT(n);
	c = NEW_INT(n);
	for (i = 0; i < n; i++) {
		xx[i] = v_point[i];
		yy[i] = v_solid[i];
		}
	//representative_under_folding(xx, n);
	//representative_under_folding_solid(yy, b_1, b_2, b_3);
	cout << "yy=solid";
	INT_vec_print(cout, yy, n);
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
		INT_vec_print(cout, yy, n);
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

	FREE_INT(c);
	FREE_INT(xx);
	FREE_INT(yy);
	return ret;
#endif

}

void representative_under_folding(INT *x, INT len)
{
	INT i, w;

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

void representative_under_folding_line(INT *x, INT b_1)
{
	INT *y;

	y = NEW_INT(n);
	compress1(x, y, b_1);
	representative_under_folding(y, n - 1);
	expand1(x, y, b_1);
	FREE_INT(y);
}

void representative_under_folding_plane(INT *x, INT b_1, INT b_2)
{
	INT *y;

	y = NEW_INT(n);
	compress2(x, y, b_1, b_2);
	representative_under_folding(y, n - 2);
	expand2(x, y, b_1, b_2);
	FREE_INT(y);
}

void representative_under_folding_solid(INT *x, INT b_1, INT b_2, INT b_3)
{
	INT *y;

	y = NEW_INT(n);
	compress3(x, y, b_1, b_2, b_3);
	representative_under_folding(y, n - 3);
	expand3(x, y, b_1, b_2, b_3);
	FREE_INT(y);
}

void opposite_under_folding_line(INT *x, INT b_1)
{
	INT *y;

	y = NEW_INT(n);
	compress1(x, y, b_1);
	invert(y, n - 1);
	expand1(x, y, b_1);
	FREE_INT(y);
}

void opposite_under_folding_plane(INT *x, INT b_1, INT b_2)
{
	INT *y;

	y = NEW_INT(n);
	compress2(x, y, b_1, b_2);
	invert(y, n - 2);
	expand2(x, y, b_1, b_2);
	FREE_INT(y);
}

void opposite_under_folding_solid(INT *x, INT b_1, INT b_2, INT b_3)
{
	INT *y;

	y = NEW_INT(n);
	compress3(x, y, b_1, b_2, b_3);
	invert(y, n - 3);
	expand3(x, y, b_1, b_2, b_3);
	FREE_INT(y);
}


void invert(INT *x, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		if (x[i]) {
			x[i] = 0;
			}
		else {
			x[i] = 1;
			}
		}
}

