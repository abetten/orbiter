/*
 * domino_assignment.cpp
 *
 *  Created on: Mar 1, 2020
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {


#define SPEEDUP_BY_FACTOR  2
#define PHOTO_TYPE "ASCII"



domino_assignment::domino_assignment()
{
	D = 0;
	s = 0;
	size_dom = 0;
	tot_dom = 0;

	M = 0; // number of rows  = (D + 1) * s
	N = 0; // number of columns = D * s

	ij_posi = NULL;
	assi = NULL; // [tot_dom * 5];
	broken_dom = NULL; // [M * N]
	matching = NULL;
	A = NULL;
	mphoto = NULL;

	North = NULL;
	South = NULL;
	West = NULL;
	East = NULL;

	brake_cnt = 0;
	brake = NULL;
	//broken1 = 0;

	nb_changes = 0;

}


domino_assignment::~domino_assignment()
{
	if (ij_posi) {
		FREE_int(ij_posi);
	}
	if (assi) {
		FREE_int(assi);
	}
	if (broken_dom) {
		FREE_int(broken_dom);
	}
	if (matching) {
		FREE_int(matching);
	}
	if (A) {
		FREE_int(A);
	}
	if (mphoto) {
		FREE_int(mphoto);
	}
	if (North) {
		FREE_int(North);
	}
	if (South) {
		FREE_int(South);
	}
	if (West) {
		FREE_int(West);
	}
	if (East) {
		FREE_int(East);
	}
	if (brake) {
		FREE_int(brake);
	}
}


void domino_assignment::stage0(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c;

	if (f_v) {
		cout << "stage 0" << endl;
	}
	if (f_v) {
		cout << "stage 0: before rotate_randomized" << endl;
	}
	rotate_randomized(verbose_level - 1);
	if (f_v) {
		cout << "stage 0: after rotate_randomized" << endl;
	}

	if (f_v) {
		cout << "stage 0: before shift_randomized" << endl;
	}
	shift_randomized(verbose_level - 1);
	if (f_v) {
		cout << "stage 0: after shift_randomized" << endl;
	}


	if (f_v) {
		cout << "stage 0: before flip_randomized" << endl;
	}
	flip_randomized(verbose_level - 1);
	if (f_v) {
		cout << "stage 0: after flip_randomized" << endl;
	}


	if (f_v) {
		cout << "stage 0: before swap_randomized" << endl;
	}
	swap_randomized(verbose_level - 1);
	if (f_v) {
		cout << "stage 0: after swap_randomized" << endl;
	}


	c = cost_function();

	if (f_v) {
		cout << "stage 0: before flip_each, cost = " << c << endl;
	}
	flip_each(verbose_level - 1);
	if (f_v) {
		cout << "stage 0: after flip_each" << endl;
	}



	if (f_v) {
		cout << "stage 0: before swap_each" << endl;
	}
	swap_each(verbose_level - 1);
	if (f_v) {
		cout << "stage 0: after swap_each" << endl;
	}

	c = cost_function();
	if (f_v) {
		cout << "stage 0: done, cost=" << c << endl;
	}
}

void domino_assignment::stage1(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, c, c1, c2;
	domino_assignment *Domino_Assignment_save;
	int save_nb_changes;

	if (f_v) {
		cout << "stage 1" << endl;
	}

	Domino_Assignment_save = NEW_OBJECT(domino_assignment);
	Domino_Assignment_save->initialize_assignment(D, s, verbose_level - 1);

	if (f_v) {
		cout << "stage 1:" << endl;
	}
	for (i = 0; i < tot_dom; i++){
		if (i % SPEEDUP_BY_FACTOR) {
			continue;
		}

		cout << "stage 1: " << i << " / " << tot_dom << endl;

		c1 = cost_function();

		cout << "stage 1: " << i << " / " << tot_dom << " cost = " << c1 << endl;

		move(Domino_Assignment_save);

		save_nb_changes = nb_changes;

		//save_assi();


		cout << "stage 1: before rotate_once" << endl;
		rotate_once(i, verbose_level - 1);
		cout << "stage 1: after rotate_once" << endl;


		//shift_new(FALSE,i);

		cout << "stage 1: before flip_after_shift" << endl;
		flip_after_shift(verbose_level - 1);
		cout << "stage 1: after flip_after_shift" << endl;

		cout << "stage 1: before swap_each" << endl;
		swap_each(0 /*verbose_level - 1*/);
		cout << "stage 1: after swap_each" << endl;

		c2 = cost_function();
		if (c1 <= c2) {
			//cout << i << " : " << c2 << " skipping" << endl;
			cout << "stage 1: restoring assignment" << endl;
			Domino_Assignment_save->move(this);
			nb_changes = save_nb_changes;
			drop_changes_to(save_nb_changes, verbose_level);
			//restore_assi();
			}
		else {
			if (f_v) {
				//cout << "improvement " << Draw_cnt << endl;
			}
		}
	}

	FREE_OBJECT(Domino_Assignment_save);

	c = cost_function();
	if (f_v) {
		cout << "stage 1: done, cost=" << c << endl;
	}
}

void domino_assignment::stage2(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, c, c1, c2;
	domino_assignment *Domino_Assignment_save;
	int save_nb_changes;

	if (f_v) {
		cout << "stage 2" << endl;
	}

	Domino_Assignment_save = NEW_OBJECT(domino_assignment);
	Domino_Assignment_save->initialize_assignment(D, s, verbose_level - 1);

	if (f_v) {
		cout << "stage 2:" << endl;
	}


	for (i = 0; i < tot_dom; i++){
		if (i % SPEEDUP_BY_FACTOR) {
			continue;
		}
		c1 = cost_function();
		move(Domino_Assignment_save);
		save_nb_changes = nb_changes;

		shift_once(i, verbose_level - 1);
		flip_after_shift(verbose_level - 1);
		swap_each(0 /*verbose_level - 1*/);

		c2 = cost_function();
		if (c1 <= c2) {
			//cout << i << " : " << c2 << " skipping" << endl;
			Domino_Assignment_save->move(this);
			nb_changes = save_nb_changes;
			drop_changes_to(save_nb_changes, verbose_level);
		}
		else {
			cout << "stage 2:" << i << "/" << tot_dom << " : " << c2 << endl;
		}
	}



	FREE_OBJECT(Domino_Assignment_save);


	c = cost_function();
	if (f_v) {
		cout << "stage 2: done, cost=" << c << endl;
	}
}

void domino_assignment::initialize_assignment(int D, int s,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t = 0;
	int i = 0;
	int j = 0;
	int m, n, td;

	if (f_v) {
		cout << "domino_assignment::initialize_assignment" << endl;
	}
	domino_assignment::D = D;
	domino_assignment::s = s;
	size_dom = D + ((D * (D - 1)) >> 1);
	tot_dom = size_dom * s * s;
	M = (D + 1) * s;
	N = D * s;

	ij_posi = NEW_int(M * N * 2);
	assi = NEW_int(tot_dom * 5);
	broken_dom = NEW_int(M * N);
	matching = NEW_int(M * N);
	A = NEW_int(M * N);
	mphoto = NEW_int(M * N);
	brake = NEW_int(tot_dom);

	nb_changes = 0;

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			ij_posi[(i * N + j) * 2 + 0] = i;
			ij_posi[(i * N + j) * 2 + 1] = j;
		}
	}

	i = j = 0;
	t = 0;
	if (ODD(M)) {
		for (m = 0; m < D; m++) {
			for (td = 0; td < s * s; td++) {
				assi[t * 5 + 0] = m;
				assi[t * 5 + 1] = m;
				assi[t * 5 + 2] = 1;
				assi[t * 5 + 3] = i;
				assi[t * 5 + 4] = j;
				broken_dom[i * N + j] = t;
				broken_dom[i * N + j + 1] = t;
				t++;
				if (j == N - 2) {
					j = 0;
					i++;
				}
				else {
					j = j + 2;
				}
			}
		}
		for (m = 0; m < D; m++) {
			for (n = m + 1; n < D; n++) {
				for (td = 0; td < s * s; td++) {
					assi[t * 5 + 0] = m;
					assi[t * 5 + 1] = n;
					assi[t * 5 + 2] = 4;
					assi[t * 5 + 3] = i;
					assi[t * 5 + 4] = j;
					broken_dom[i * N + j] = t;
					broken_dom[i * N + j + 1] = t;
					t++;
					if (j == N - 2){
						j = 0;
						i++;
					}
					else {
						j = j + 2;
					}
				}
			}
		}
	}
	else {
		// case M is ODD, i.e., the in
		for (m = 0; m < D; m++) {
			for (td = 0; td < s * s; td++) {
				assi[t * 5 + 0] = m;
				assi[t * 5 + 1] = m;
				assi[t * 5 + 2] = 0;
				assi[t * 5 + 3] = i;
				assi[t * 5 + 4] = j;
				broken_dom[i * N + j] = t;
				broken_dom[(i + 1) * N + j] = t;
				t++;
				if (j == N - 1) {
					j = 0;
					i = 2 + i;
				}
				else {
					j++;
				}
			}
		}
		for (m = 0; m < D; m++) {
			for (n = m + 1; n < D; n++) {
				for (td = 0; td < s * s; td++) {
					assi[t * 5 + 0] = m;
					assi[t * 5 + 1] = n;
					assi[t * 5 + 2] = 2;
					assi[t * 5 + 3] = i;
					assi[t * 5 + 4] = j;
					broken_dom[i * N + j] = t;
					broken_dom[(i + 1) * N + j] = t;

					t++;
					if (j == N - 1) {
						j = 0;
						i = 2 + i;
					}
					else {
						j++;
					}
				}
			}
		}
	}

	init_matching(verbose_level);


	if (f_v) {
		cout << "domino_assignment::initialize_assignment done" << endl;
	}
}

void domino_assignment::init_matching(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "domino_assignment::init_matching" << endl;
	}
	North = new int[M * N];
	South = new int[M * N];
	West = new int[M * N];
	East = new int[M * N];

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			if (i == 0) {
				North[i * N + j] = -1;
			}
			else {
				North[i * N + j] = (i - 1) * N + j;
			}
			if (i == M - 1) {
				South[i * N + j] = -1;
			}
			else {
				South[i * N + j] = (i + 1) * N + j;
			}
			if (j == 0) {
				West[i * N + j] = -1;
			}
			else {
				West[i * N + j] = i * N + j - 1;
			}
			if (j == N - 1) {
				East[i * N + j] = -1;
			}
			else {
				East[i * N + j] = i * N + j + 1;
			}
		}
	}
/*
	print_matrix_MN(North, 3);
	cout << endl;
	print_matrix_MN(South, 3);
	cout << endl;
	print_matrix_MN(West, 3);
	cout << endl;
	print_matrix_MN(East, 3);
	cout << endl;
*/
	//matching = new int[M * N];
	//matching_save = new int[M * N];
	if (EVEN(M)) {
		for (i = 0; i < M; i++) {
			for (j = 0; j < N; j++) {
				if (EVEN(i))
					matching[i * N + j] = 6;
				else
					matching[i * N + j] = 12;
			}
		}
	}
	else {
		for (i = 0; i < M; i++) {
			for (j = 0; j < N; j++) {
				if (EVEN(j))
					matching[i * N + j] = 3;
				else
					matching[i * N + j] = 9;
			}
		}
	}

	cout << "The initial board is:" << endl;
	print(cout);

	compute_domino_matrix(tot_dom);
	cout << "recomputed from assignment:" << endl;
	print(cout);

}

int domino_assignment::cost_function()
{
	int i;
	int sum = 0;

	for (i = 0; i < tot_dom; i++) {
		sum = sum +
			compute_cost_of_one_piece(i);
	}
	return sum;
}

int domino_assignment::compute_cost_of_one_piece(int idx)
{
	int m, n, o, i, j;
	int d = 0;

	m = assi[idx * 5 + 0];
	n = assi[idx * 5 + 1];
	o = assi[idx * 5 + 2];
	i = assi[idx * 5 + 3];
	j = assi[idx * 5 + 4];
	d = compute_cost_of_one_piece_directly(m, n, o, i, j);
	return d;
}

int domino_assignment::compute_cost_of_one_piece_directly(
		int m, int n, int o, int i, int j)
{
	int d;

	if (o == 0) {
		d = my_distance(m - mphoto[i * N + j], m - mphoto[(i + 1) * N + j]);
	}
	else if (o == 1) {
		d = my_distance(m - mphoto[i * N + j], m - mphoto[i * N + j + 1]);
	}
	else if (o == 2) {
		d = my_distance(m - mphoto[i * N + j], n - mphoto[(i + 1) * N + j]);
	}
	else if (o == 3) {
		d = my_distance(n - mphoto[i * N + j], m - mphoto[(i + 1) * N + j]);
	}
	else if (o == 4) {
		d = my_distance(m - mphoto[i * N + j], n - mphoto[i * N + j + 1]);
	}
	else if (o == 5) {
		d = my_distance(n - mphoto[i * N + j], m - mphoto[i * N + j + 1]);
	}
	else {
		cout << "domino_assignment::compute_cost_of_one_piece_directly illegal value of o" << endl;
		exit(1);
	}
	return d;
}

int domino_assignment::my_distance(int a, int b)
{
	return a * a + b * b;
	//return ABS(a) + ABS(b);
}

void domino_assignment::compute_domino_matrix(int depth)
{
	int c, m, n, o, i, j, p1, p2;

	Int_vec_zero(matching, M * N);
	Int_vec_zero(A, M * N);

	for (c = 0; c < depth; c++) {
		m = assi[c * 5 + 0];
		n = assi[c * 5 + 1];
		o = assi[c * 5 + 2];
		i = assi[c * 5 + 3];
		j = assi[c * 5 + 4];
		if (o == 0) {
			A[i * N + j] = m;
			A[(i + 1) * N + j] = m;
		}
		else if (o == 1) {
			A[i * N + j] = m;
			A[i * N + j + 1] = m;
		}
		else if (o == 2) {
			A[i * N + j] = m;
			A[(i + 1) * N + j] = n;
		}
		else if (o == 3) {
			A[i * N + j] = n;
			A[(i + 1) * N + j] = m;
		}
		else if (o == 4) {
			A[i * N + j] = m;
			A[i * N + j + 1] = n;
		}
		else if (o == 5) {
			A[i * N + j] = n;
			A[i * N + j + 1] = m;
		}
		p1 = i * N + j;
		if (m == n) {
			if (o == 0) {
				matching[p1] = 6;
				p2 = (i + 1) * N + j;
				matching[p2] = 12;
			}
			else if (o == 1) {
				matching[p1] = 3;
				p2 = i * N + j + 1;
				matching[p2] = 9;
			}
			else {
				cout << "illegal o" << endl;
					exit(1);
			}
		}
		else {
			if (o == 2 || o == 3) {
				matching[p1] = 6;
				p2 = (i + 1) * N + j;
				matching[p2] = 12;
			}
			else if (o == 4 || o == 5) {
				matching[p1] = 3;
				p2 = i * N + j + 1;
				matching[p2] = 9;
			}
			else {
				cout << "illegal o" << endl;
				exit(1);
			}
		}
	}
}

void domino_assignment::move(domino_assignment *To)
{
	if (To->M != M) {
		cout << "domino_assignment::move To->M != M" << endl;
		exit(1);
	}
	if (To->N != N) {
		cout << "domino_assignment::move To->N != N" << endl;
		exit(1);
	}
	Int_vec_copy(assi, To->assi, tot_dom * 5);
	Int_vec_copy(broken_dom, To->broken_dom, M * N);
	Int_vec_copy(matching, To->matching, M * N);
}

void domino_assignment::draw_domino_matrix(std::string &fname,
		int depth,
		int f_has_cost, int cost,
		graphics::layered_graph_draw_options *Draw_options,
		int verbose_level)
{
	compute_domino_matrix(depth);
	//print_this_matching(matching);

	draw_domino_matrix2(fname,
			f_has_cost, cost,
			FALSE, FALSE, FALSE, NULL,
			FALSE, FALSE,
			Draw_options,
			verbose_level - 1);
}

void domino_assignment::draw_domino_matrix2(std::string &fname,
	int f_has_cost, int cost,
	int f_frame, int f_grid, int f_B, int *B,
	int f_numbers, int f_gray,
	graphics::layered_graph_draw_options *Draw_options,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int n = 100, dx = 200, dy = 200;
	int dx = 400, dy = 400;
	//int xmin = 0, xmax = 0;
	//int ymin = 0, ymax = 0;
	int factor_1000 = 1000;
	string fname_full;
	int rad = 45;
	int edge = 2;
	//int rad = 22;
	//int edge = 10;
	double scale;
	//int cost;
	int xmax, ymax;

	if (f_v) {
		cout << "domino_assignment::draw_domino_matrix2" << endl;
	}
	//cost = compute_objec_function();
	scale = (double) N * .1;
	if (dx * N * scale > 4000) {
		//scale = ((double)4000) / (N * dx);
		f_numbers = FALSE;
	}
	if (dy * M * scale > 4000) {
		//scale = ((double)4000)  / (M * dy);
		f_numbers = FALSE;
	}
	//cout << "scale=" << scale << endl;
	dx = (int) (((double) dx) * scale);
	dy = (int) (((double) dy) * scale);
	rad = (int) (((double) rad) * scale);
	edge = (int) (((double) edge) * scale);
	//cout << "dx=" << dx << "dy=" << dy << " rad=" << rad << " edge=" << edge << endl;
	xmax = N * dx;
	ymax = M * dy;
	//cout << "xmax=" << xmax << " ymax=" << ymax << endl;

	Draw_options->xin = xmax;
	Draw_options->yin = ymax;
	Draw_options->scale = 0.2;
	Draw_options->line_width = 0.75;

	fname_full.assign(fname);
	fname_full.append(".mp");

	if (f_v) {
		cout << "domino_assignment::draw_domino_matrix2 fname=" << fname_full << endl;
	}
	{
		//int f_embedded = FALSE;
		//int f_sideways = FALSE;
		//double scale = 0.2;
		//double line_width = 0.75;

		graphics::mp_graphics G;


		G.init(
				fname_full,
				Draw_options,
				verbose_level);


#if 0
		mp_graphics G(fname_full, xmin, ymin, xmax, ymax,
				f_embedded, f_sideways, verbose_level - 1);
		G.set_parameters(scale, line_width);
#endif
		G.out_xmin() = 0;
		G.out_ymin() = 0;
		G.out_xmax() = 2000000;
		G.out_ymax() = (int)((2000000. * 16.) / 14.);

		G.header();
		G.begin_figure(factor_1000);


		G.domino_draw_assignment(A, matching, B,
				M, N,
				dx, dy,
				rad, edge,
				f_grid, f_gray, f_numbers, f_frame,
				TRUE /* f_cost */, cost);

		G.end_figure();
		G.footer();
	}

	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname_full << " of size "
				<< Fio.file_size(fname_full) << endl;
	}
}

void domino_assignment::read_photo(std::string &photo_fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *R, *B, *G, r, g, b, m, n, i, j; // c
	double *dphoto;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "domino_assignment::read_photo" << endl;
	}
	dphoto = new double[M * N];

	if (strcmp(PHOTO_TYPE, "ASCII") == 0) {

		double f;
		string fname_R;
		string fname_G;
		string fname_B;

		R = new int[M * N];
		G = new int[M * N];
		B = new int[M * N];
		fname_R.assign(photo_fname);
		fname_G.assign(photo_fname);
		fname_B.assign(photo_fname);
		fname_R.append(".r");
		fname_G.append(".g");
		fname_B.append(".b");

		if (Fio.file_size(fname_R) <= 0) {
			cout << "cannot find input file " << fname_R << endl;
			exit(1);
		}



		{
			ifstream f_R(fname_R);
			ifstream f_G(fname_G);
			ifstream f_B(fname_B);
			f_R >> n >> m;
			f_G >> n >> m;
			f_B >> n >> m;
			if (m != M) {
				cout << "get_photo: m != M" << endl;
				cout << "m = " << m << endl;
				cout << "M = " << M << endl;
				exit(1);
			}
			if (n != N) {
				cout << "get_photo: n != N" << endl;
				cout << "n = " << n << endl;
				cout << "N = " << N << endl;
				exit(1);
			}
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {
					f_R >> r; R[i * N + j] = r;
					f_G >> g; G[i * N + j] = g;
					f_B >> b; B[i * N + j] = b;
					f = sqrt((double)(r * r + g * g + b * b));
					//c = (int)f;
					dphoto[i * N + j] = f;
				}
			}
			if (f_v) {
				cout << "read photo from files " << fname_R << ", " << fname_G << ", " << fname_B << endl;
			}
		}
		scale_photo(dphoto, verbose_level);

		for (i = 0; i < M * N; i++) {
			mphoto[i] = D - 1 - mphoto[i];
		}

		string fname_m;

		fname_m.assign(photo_fname);
		fname_m.append("_m.csv");

		Fio.int_matrix_write_csv(fname_m, mphoto, M, N);
		if (f_v) {
			cout << "Written file " << fname_m << " of size " << Fio.file_size(fname_m) << endl;
		}
	}

	if (strcmp(PHOTO_TYPE, "ASCII_GRAY") == 0) {

		string fname;
		int *pixel;

		pixel = new int[M * N];
		fname.assign(photo_fname);
		fname.append(".txt");
		{
			ifstream fp(fname);
			fp >> n >> m;
			if (m != M) {
				cout << "get_photo: m != M" << endl;
				cout << "m = " << m << endl;
				cout << "M = " << M << endl;
				exit(1);
				}
			if (n != N) {
				cout << "get_photo: n != N" << endl;
				cout << "n = " << n << endl;
				cout << "N = " << N << endl;
				exit(1);
				}
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {
					fp >> r;
					pixel[i * N + j] = r;
					dphoto[i * N + j] = (double)r;
					}
				}
			if (f_v) {
				cout << "read photo from files " << fname << endl;
			}
		} // close files
		scale_photo(dphoto, verbose_level);
	}



	delete [] dphoto;
	if (f_v) {
		cout << "domino_assignment::read_photo done" << endl;
	}
}

void domino_assignment::scale_photo(double *dphoto, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	double max_photo = 0;
	double scale = 0;

	if (f_v) {
		cout << "domino_assignment::scale_photo" << endl;
	}
	for (i = 0; i < M * N; i++) {
		if (dphoto[i] > max_photo) {
			max_photo = dphoto[i];
		}
	}
	if (f_v) {
		cout << "D= " << D << endl;
		cout << "max_photo= " << max_photo << endl;
	}
	scale = max_photo / (double) D;
	if (f_v) {
		cout << "scale= " << scale << endl;
	}
	for (i = 0; i < M * N; i++) {
		mphoto[i] = (int)(dphoto[i] / scale);
		if (mphoto[i] >= D) {
			mphoto[i] = D - 1;
		}
	}
	if (f_v) {
		cout << "domino_assignment::scale_photo done" << endl;
	}
}

void domino_assignment::do_flip_recorded(int f2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int assi_before[5];
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "domino_assignment::do_flip_recorded f2=" << f2 << endl;
	}
	Int_vec_copy(assi + f2 * 5, assi_before, 5);

	do_flip(f2, verbose_level);

	if (Sorting.int_vec_compare(assi + f2 * 5, assi_before, 5)) {
		record_flip(f2, verbose_level - 1);
	}
	if (f_v) {
		cout << "domino_assignment::do_flip_recorded done" << endl;
	}
}

void domino_assignment::do_flip(int f2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int of, c1, c2;

	if (f_v) {
		cout << "domino_assignment::do_flip f2=" << f2 << endl;
	}

	of = assi[f2 * 5 + 2];
	if (assi[f2 * 5 + 2] < 4) {
		//vertical
		assi[f2 * 5 + 2] = 3 - (assi[f2 * 5 + 2] - 2);
		//f(x)=3-(x-2)==>f(2)=3, f(3)=2,
	}
	else {
		//horiz
		assi[f2 * 5 + 2] = 5 - (assi[f2 * 5 + 2] - 4);
		//f(x)=5-(x-4)==>f(4)=5, f(5)=4,
	}
	c1 = compute_cost_of_one_piece_directly(
			assi[f2 * 5 + 0],
			assi[f2 * 5 + 1],
			of,
			assi[f2 * 5 + 3],
			assi[f2 * 5 + 4]);
	c2 = compute_cost_of_one_piece_directly(
			assi[f2 * 5 + 0],
			assi[f2 * 5 + 1],
			assi[f2 * 5 + 2],
			assi[f2 * 5 + 3],
			assi[f2 * 5 + 4]);
	//cout << " :flipc1 = " << c1 << ", c2 = " << c2 << endl;
	if (c2 < c1) {
		//cnt++;
	}
	else {
		assi[f2 * 5 + 2] = of;
	}
}

void domino_assignment::flip_each(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f2;

	if (f_v) {
		cout << "domino_assignment::flip_each " << endl;
	}
	for (f2 = 0; f2 < tot_dom; f2++) {
		if (f_vv) {
			cout << "f2=" << f2 << ", flipping domino " << endl;
		}
		if (assi[f2 * 5 + 2] > 1) {
			// if it is not a m:m domino
			do_flip_recorded(f2, verbose_level - 1);
		}
		else {
			if (f_vv) {
				cout << " :no flip" << endl;
			}
		}
	}
}

void domino_assignment::flip_randomized(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f1, f2, i;
	orbiter_kernel_system::os_interface Os;

	f1 = Os.random_integer(tot_dom * 2);//number of flips,

	if (f_v) {
		cout << "domino_assignment::flip_randomized flipping " << f1 << " times" << endl;
	}
	for (i = 0; i < f1; i++) {

		f2 = Os.random_integer(tot_dom);
			// which domino we flip
		if (f_vv) {
			cout << "flip # " << i << ", flipping domino " << f2 << " which is x("<<assi[f2*5+0]<<","<<assi[f2*5+1]<<","<<assi[f2*5+2]<<","<<assi[f2*5+3]<<","<<assi[f2*5+4]<<")";
		}
		if (assi[f2 * 5 + 2] > 1) {
			// if it is not a m:m domino
			do_flip_recorded(f2, verbose_level - 1);
		}
		else {
			if (f_vv) {
				cout << " :no flip" << endl;
			}
		}
	}
	if (f_v) {
		cout << "finish flipping" << endl;
	}
}

void domino_assignment::do_swap_recorded(int s1, int s2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int assi_before[10];
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "domino_assignment::do_swap_recorded" << endl;
	}

	Int_vec_copy(assi + s1 * 5, assi_before, 5);
	Int_vec_copy(assi + s2 * 5, assi_before + 5, 5);

	do_swap(s1, s2, verbose_level);

	if (Sorting.int_vec_compare(assi + s1 * 5, assi_before, 5) ||
			Sorting.int_vec_compare(assi + s2 * 5, assi_before + 5, 5)) {
		record_swap(s1, s2, verbose_level - 1);
	}



}

void domino_assignment::do_swap(int s1, int s2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m1, n1, m2, n2, o1, o2, c1, c2;

	if (f_v) {
		cout << "domino_assignment::do_swap" << endl;
	}
	m1 = assi[s1 * 5 + 0];
	n1 = assi[s1 * 5 + 1];
	o1 = assi[s1 * 5 + 2];
	m2 = assi[s2 * 5 + 0];
	n2 = assi[s2 * 5 + 1];
	o2 = assi[s2 * 5 + 2];

	assi[s1 * 5 + 0] = m2;
	assi[s1 * 5 + 1] = n2;
	//assi[s1 * 5 + 2] = o2;
	assi[s2 * 5 + 0] = m1;
	assi[s2 * 5 + 1] = n1;
	//assi[s2 * 5 + 2] = o1;
	if (o1 == 1 || o1 > 3) {
		// s1 is horizontal
		if (m2 == n2) {
			assi[s1 * 5 + 2] = 1;
		}
		else {
			assi[s1 * 5 + 2] = 4;
		}
	}
	else {
		// s1 is Vertical
		if (m2 == n2) {
			assi[s1 * 5 + 2] = 0;
		}
		else {
			assi[s1 * 5 + 2] = 2;
		}
	}

	if (o2 == 1 || o2 > 3) {
		// s2 is horizontal
		if (m1 == n1) {
			assi[s2 * 5 + 2] = 1;
		}
		else {
			assi[s2 * 5 + 2] = 4;
		}
	}
	else {
		// s2 is Vertical
		if (m1 == n1) {
			assi[s2 * 5 + 2] = 0;
		}
		else {
			assi[s2 * 5 + 2] = 2;
		}
	}

	int os1, os2, cs1, cs2, cfs1, cfs2;

	c1 = compute_cost_of_one_piece_directly(
			m1,
			n1,
			o1,
			assi[s1 * 5 + 3],
			assi[s1 * 5 + 4]) +
		compute_cost_of_one_piece_directly(
				m2,
				n2,
				o2,
				assi[s2 * 5 + 3],
				assi[s2 * 5 + 4]);
	cs1 = compute_cost_of_one_piece_directly(
			m2,
			n2,
			assi[s1 * 5 + 2],
			assi[s1 * 5 + 3],
			assi[s1 * 5 + 4]);
	cs2 = compute_cost_of_one_piece_directly(
			m1,
			n1,
			assi[s2 * 5 + 2],
			assi[s2 * 5 + 3],
			assi[s2 * 5 + 4]);
	c2 = cs1 + cs2;
	if (c2 < c1) {
		//cnt++;
		if (assi[s1 * 5 + 2] > 1) {
			// if it is not a m:m domino
			do_flip(s1, verbose_level - 1);
		}
		if (assi[s2 * 5 + 2] > 1) {
			// if it is not a m:m domino
			do_flip(s2, verbose_level - 1);
		}
	}
	else {
		os1 = assi[s1 * 5 + 2];
		os2 = assi[s2 * 5 + 2];
		if (os1 > 1) {
			cfs1 = do_flipswap(s1);
		}
		else {
			cfs1 = cs1;
		}
		if (os2 > 1) {
			cfs2 = do_flipswap(s2);
		}
		else {
			cfs2 = cs2;
		}
		if (cfs1 >= cs1 && cfs2 >= cs2){
			assi[s1 * 5 + 0] = m1;
			assi[s1 * 5 + 1] = n1;
			assi[s1 * 5 + 2] = o1;
			assi[s2 * 5 + 0] = m2;
			assi[s2 * 5 + 1] = n2;
			assi[s2 * 5 + 2] = o2;
		}
		else {
			if (cfs1 + cfs2 >= c1) {
				if (cfs1 + cs2 < c1) {
					assi[s2 * 5 + 2] = os2;
					//cnt++;
				}
				else {
					if (cs1 + cfs2 < c1) {
						assi[s1 * 5 + 2] = os1;
						//cnt++;
					}
					else {
						assi[s1 * 5 + 0] = m1;
						assi[s1 * 5 + 1] = n1;
						assi[s1 * 5 + 2] = o1;
						assi[s2 * 5 + 0] = m2;
						assi[s2 * 5 + 1] = n2;
						assi[s2 * 5 + 2] = o2;
					}
				}
			}
			else {
				//cnt++;
			}
		}
	}
	if (f_v) {
		cout << "domino_assignment::do_swap done" << endl;
	}
}

int domino_assignment::do_flipswap(int f2)
{
	int c2;

	if (assi[f2 * 5 + 2] < 4) {
		//vertical
		assi[f2 * 5 + 2] = 3 - (assi[f2 * 5 + 2] - 2);
		//f(x)=3-(x-2)==>f(2)=3, f(3)=2,
	}
	else {
		//horiz
		assi[f2 * 5 + 2] = 5 - (assi[f2 * 5 + 2] - 4);
		//f(x)=5-(x-4)==>f(4)=5, f(5)=4,
	}
	c2 = compute_cost_of_one_piece(f2);
	return c2;
}

void domino_assignment::swap_randomized(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int s1, s2, r1, i;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "domino_assignment::swap_randomized" << endl;
	}
	r1 = Os.random_integer(tot_dom * 2);
		//number of swaps
	if (f_v) {
		cout << "swap " << r1 << " times" << endl;
	}
	for (i = 0; i < r1; i++) {
		s1 = Os.random_integer(tot_dom);
		s2 = Os.random_integer(tot_dom);
		if (s1 != s2) {
			if (f_vv) {
				cout << "swap # " << i << ", swapping domino " << s1 << " and " << s2 << endl;
			}
			do_swap_recorded(s1, s2, verbose_level - 2);
		}
	}
	if (f_v) {
		cout << "domino_assignment::swap_randomized done" << endl;
	}
}

void domino_assignment::swap_each(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int s1, s2, N, N100, cnt = 0; // r1

	//r1 = 1;
	if (f_v) {
		cout << "domino_assignment::swap_each verbose_level = " << verbose_level << endl;
	}
	N = tot_dom * tot_dom;
	N100 = N / 100 + 1;
	for (s1 = 0; s1 < tot_dom; s1++) {
		for (s2 = 0; s2 < tot_dom; s2++, cnt++) {
			if (f_v && (cnt % N100) == 0) {
				cout << "domino_assignment::swap_each " << cnt / N100 << " %" << endl;
			}
			if (f_vv) {
				cout << "domino_assignment::swap_each swapping domino " << s1 << " and " << s2 << endl;
			}
			do_swap_recorded(s1, s2, verbose_level - 2);
		}
	}

	if (f_v) {
		cout << "domino_assignment::swap_each done" << endl;
	}
}

void domino_assignment::do_horizontal_rotate(int ro, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int o2, i1; // o1

	if (f_v) {
		cout << "domino_assignment::do_horizontal_rotate" << endl;
	}
	if (assi[ro * 5 + 3] != M - 1) {
		if (assi[ro * 5 + 2] >= 4) {
			modify_matching(
				ro,
				assi[ro * 5 + 0],
				assi[ro * 5 + 1],
				2,
				assi[ro * 5 + 3],
				assi[ro * 5 + 4],
				verbose_level - 2);
			o2 = 2;
		}
		else {
			modify_matching(
				ro,
				assi[ro * 5 + 0],
				assi[ro * 5 + 1],
				0,
				assi[ro * 5 + 3],
				assi[ro * 5 + 4],
				verbose_level - 2);
			o2 = 0;
		}
	}
	else {
		if (assi[ro * 5 + 2] >= 4) {
			modify_matching(
				ro,
				assi[ro * 5 + 0],
				assi[ro * 5 + 1],
				2,
				assi[ro * 5 + 3] - 1,
				assi[ro * 5 + 4],
				verbose_level - 2);
			o2 = 2;
			i1 = assi[ro * 5 + 3];
			assi[ro * 5 + 3] = i1 - 1;
		}
		else {
			modify_matching(
				ro,
				assi[ro * 5 + 0],
				assi[ro * 5 + 1],
				0,
				assi[ro * 5 + 3] - 1,
				assi[ro * 5 + 4],
				verbose_level - 2);
			o2 = 0;
			i1 = assi[ro * 5 + 3];
			assi[ro * 5 + 3] = i1 - 1;
		}
	}
	//o1 = assi[ro * 5 + 2];
	assi[ro * 5 + 2] = o2;
	if (f_v) {
		cout << "domino_assignment::do_horizontal_rotate done" << endl;
	}
}

void domino_assignment::do_vertical_rotate(int ro, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int o2, j1;

	if (f_v) {
		cout << "domino_assignment::do_vertical_rotate ro=" << ro << endl;
		cout << "assi[ro * 5 + 4]=" << assi[ro * 5 + 4] << endl;
		cout << "assi[ro * 5 + 2]=" << assi[ro * 5 + 2] << endl;
	}
	if (assi[ro * 5 + 4] != N - 1) {
		if (assi[ro * 5 + 2] >= 2) {
			modify_matching(
				ro,
				assi[ro * 5 + 0],
				assi[ro * 5 + 1],
				4,
				assi[ro * 5 + 3],
				assi[ro * 5 + 4],
				verbose_level - 1);
			o2 = 4;
		}
		else {
			modify_matching(
				ro,
				assi[ro * 5 + 0],
				assi[ro * 5 + 1],
				1,
				assi[ro * 5 + 3],
				assi[ro * 5 + 4],
				verbose_level - 1);
			o2 = 1;
		}
	}
	else {
		if (assi[ro * 5 + 2] >= 2) {
			modify_matching(
				ro,
				assi[ro * 5 + 0],
				assi[ro * 5 + 1],
				4,
				assi[ro * 5 + 3],
				assi[ro * 5 + 4] - 1,
				verbose_level - 1);
			o2 = 4;
			j1 = assi[ro * 5 + 4];
			assi[ro * 5 + 4] = j1 - 1;
		}
		else {
			modify_matching(
				ro,
				assi[ro * 5 + 0],
				assi[ro * 5 + 1],
				1,
				assi[ro * 5 + 3],
				assi[ro * 5 + 4] - 1,
				verbose_level - 1);
			o2 = 1;
			j1 = assi[ro * 5 + 4];
			assi[ro * 5 + 4] = j1 - 1;
		}
	}
	//o1 = assi[ro * 5 + 2];
	assi[ro * 5 + 2] = o2;
	//o1 = assi[ro * 5 + 2];
	assi[ro * 5 + 2] = o2;
	if (f_v) {
		cout << "domino_assignment::do_vertical_rotate done" << endl;
	}
}

int domino_assignment::modify_matching(
		int idx_first_broken,
		int ass_m, int ass_n,
		int ass_o, int ass_i, int ass_j,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int pos1, pos_i1, pos_j1, pos2, ret = TRUE, l, u, v, p, q, i;
	int q_count = 1;

	int *used;
	int *reached1, *reached2;
	int *list1, *list2;
	int *length1, *length2;
	int *prec1, *prec2;
	int *do_brake;
	int *point_q;
	int *point_p;

	if (f_v) {
		cout << "domino_assignment::modify_matching" << endl;
		cout << "m=" << ass_m << " n=" << ass_n << " o=" << ass_o << " i=" << ass_i << " j=" << ass_j << endl;
		print(cout);
	}

	used = new int [M * N];
	reached1 = new int [M * N];
	reached2 = new int [M * N];
	list1 = new int [M * N]; // list1[i] = position covered by the anchor of the i-th piece along the path
	list2 = new int [M * N]; // list2[i] = position covered by the trailer of the i-th piece along the path
	length1 = new int [M * N];
	length2 = new int [M * N];
	prec1 = new int [M * N];
	prec2 = new int [M * N];
	do_brake = new int [tot_dom]; // do_brake[i] is TRUE if the i-th domino is part of the path
	point_q = new int [M * N];
	point_p = new int [M * N];




	Int_vec_zero(used, M * N);
	Int_vec_zero(reached1, M * N);
	Int_vec_zero(reached2, M * N);

	Int_vec_zero(brake, tot_dom);
	Int_vec_zero(do_brake, tot_dom);

	brake[0] = idx_first_broken;
	do_brake[idx_first_broken] = TRUE;
	brake_cnt = 1;

	used[ass_i * N + ass_j] = TRUE;
	if (ass_o == 0 || ass_o == 2 || ass_o == 3) {
		used[(ass_i + 1) * N + ass_j] = TRUE;
	}
	else {
		used[ass_i * N + ass_j + 1] = TRUE;
	}
	pos_i1 = ass_i;
	pos_j1 = ass_j;
	pos1 = pos_i1 * N + pos_j1;
	if (ass_o == 0 || ass_o == 2 || ass_o == 3) {
		pos2 = South[pos1];
		if (matching[pos1] == 6) {
			goto finish;
			cout<<" goto finish1" << endl;
		}
	}
	else {
		pos2 = East[pos1];
		if (matching[pos1] == 3) {
			goto finish;
		}
	}
	if (f_vv) {
		cout << "matching needs to change" << endl;
		cout << "trying to connect " << pos1 << " to " << pos2 << endl;
	}

	broken_dom[pos1] = idx_first_broken;
	broken_dom[pos2] = idx_first_broken;





	reached1[pos1] = TRUE;
	reached2[pos2] = TRUE;
	list1[0] = pos1;
	list2[0] = pos2;
	length1[0] = 1;
	length2[0] = 1;
	prec1[0] = -1;
	prec2[0] = -1;
	l = 1;
	while (l < M * N) {
		if (f_v) {
			cout << "l=" << l << " before follow_the_matching" << endl;
		}
		follow_the_matching(l, used, reached1, list1, length1, prec1,
			verbose_level - 1);
		if (f_v) {
			cout << "l=" << l << " after follow_the_matching" << endl;
		}
		l++;

		if (f_v) {
			cout << "l=" << l << " before find_match" << endl;
		}
		u = find_match(l,
			reached1, list1, length1, prec1,
			reached2, list2, length2, prec2,
			verbose_level - 1);
		if (f_v) {
			cout << "l=" << l << " after find_match" << endl;
		}
		if (u != -1) {
			if (f_v) {
				cout << "found a match, u = " << u << " point " << list1[u] << " breaking" << endl;
			}
			break;
		}

		if (f_v) {
			cout << "l=" << l << " before breadth_search" << endl;
		}
		if (!breadth_search(l, used, reached1,
				list1, length1, prec1, verbose_level - 1)) {
			if (f_vv) {
				cout << "cannot change the matching, breaking off" << endl;
				}
			ret = FALSE;
			goto finish;
			}
		if (f_v) {
			cout << "l=" << l << " after breadth_search" << endl;
		}
		l++;
	}
	if (f_v) {
		cout << "post processing" << endl;
	}
	u = prec1[u];
	p = list1[u];
	if (f_v) {
		cout << "u=" << u << " p=" << p << endl;
	}
	while (prec1[u] != -1) {
		v = prec1[u];
		q = list1[v];
		if (f_v) {
			cout << "u=" << u << " q_count=" << q_count << " v=" << u << " q=" << p << endl;
		}
		if (f_v) {
			cout << "matching points " << q << " and " << p << endl;
		}
		if (f_v) {
			cout << "broken_dom[q]=" << broken_dom[q] << endl;
		}
		if (f_v) {
			cout << "do_brake[broken_dom[q]]=" << do_brake[broken_dom[q]] << endl;
		}

		if (do_brake[broken_dom[q]] == FALSE) {
			if (f_v) {
				cout << "brake_cnt=" << brake_cnt << endl;
			}
			brake[brake_cnt] = broken_dom[q];
			do_brake[broken_dom[q]] = TRUE;
			brake_cnt++;
		}

		if (do_brake[broken_dom[p]] == FALSE) {
			brake[brake_cnt] = broken_dom[p];
			do_brake[broken_dom[p]] = TRUE;
			brake_cnt++;
		}
		if (q < p) {
			point_q[q_count] = q;
			point_p[q_count] = p;
		}
		else{
			point_q[q_count] = p;
			point_p[q_count] = q;
		}
		q_count++;

		if (North[q] == p) {
			matching[q] = 12;
			matching[p] = 6;
		}
		else if (South[q] == p) {
			matching[q] = 6;
			matching[p] = 12;
		}
		else if (West[q] == p) {
			matching[q] = 9;
			matching[p] = 3;
		}
		else if (East[q] == p) {
			matching[q] = 3;
			matching[p] = 9;
		}
		else {
			cout << "domino_assignment::modify_matching something is wrong" << endl;
			exit(1);
		}
		u = prec1[v];
		if (u == -1) {
			cout << "prec1[v] == -1, something is wrong" << endl;
			exit(1);
			}
		p = list1[u];
		if (f_v) {
			cout << "u=" << u << endl;
		}
	}

	if (f_v) {
		cout << "finally, matching points " << pos1 << " and " << pos2 << endl;
		cout << "brake_cnt=" << brake_cnt << endl;
		cout << "i : point_p[i] : point_q[i]" << endl;
		for (i = 1; i < brake_cnt; i++) {
			cout << i << " : " << point_p[i] << " : " << point_q[i] << endl;
		}
	}


	for (i = 1; i < brake_cnt; i++) {

		if (f_v) {
			cout << "i=" << i << " / " << brake_cnt << ":" << endl;
		}

		if (f_v) {
			cout << "point_q[i]=" << point_q[i] << endl;
		}
		if (f_v) {
			cout << "ij_posi[point_q[i] * 2 + 0]=" << ij_posi[point_q[i] * 2 + 0] << endl;
			cout << "ij_posi[point_q[i] * 2 + 1]=" << ij_posi[point_q[i] * 2 + 1] << endl;
		}
		if (f_v) {
			cout << "brake[i]=" << brake[i] << endl;
		}
		assi[ brake[i] * 5 + 3] = ij_posi[point_q[i] * 2 + 0];
		if (f_v) {
			cout << "assi[ brake[i] * 5 + 3]=" << assi[ brake[i] * 5 + 3] << endl;
		}
		assi[ brake[i] * 5 + 4] = ij_posi[point_q[i] * 2 + 1];
		if (f_v) {
			cout << "assi[ brake[i] * 5 + 4]=" << assi[ brake[i] * 5 + 4] << endl;
		}

		if (ij_posi[point_q[i] * 2 + 0] == ij_posi[point_p[i] * 2 + 0]) {
			//the new domino is horizontal
			if (f_v) {
				cout << "the new domino is horizontal" << endl;
			}
			if (assi[ brake[i] * 5 + 0] == assi[ brake[i] * 5 + 1]) {
				assi[ brake[i] * 5 + 2] = 1;
			}
			else {
				assi[ brake[i] * 5 + 2] = 4;
			}
		}
		else {
			//the new domino is vertical
			if (f_v) {
				cout << "the new domino is vertical" << endl;
			}
			if (assi[ brake[i] * 5 + 0] == assi[ brake[i] * 5 + 1]) {
				assi[ brake[i] * 5 + 2] = 0;
			}
			else {
				assi[ brake[i] * 5 + 2] = 2;
			}
		}
		broken_dom[point_q[i]] = brake[i];
		broken_dom[point_p[i]] = brake[i];
		if (f_v) {
			cout << "broken_dom[point_q[i]]=" << broken_dom[point_q[i]] << endl;
			cout << "broken_dom[point_p[i]]=" << broken_dom[point_p[i]] << endl;
		}
	}

	if (f_v) {
		cout << "after the loop" << endl;
		cout << "pos1=" << pos1 << endl;
		cout << "pos2=" << pos2 << endl;
	}

	if (f_v) {
		cout << "North[pos1]=" << North[pos1] << endl;
		cout << "South[pos1]=" << South[pos1] << endl;
		cout << "West[pos1]=" << West[pos1] << endl;
		cout << "East[pos1]=" << East[pos1] << endl;
	}
	if (North[pos1] == pos2) {
		matching[pos1] = 12;
		matching[pos2] = 6;
	}
	else if (South[pos1] == pos2) {
		matching[pos1] = 6;
		matching[pos2] = 12;
	}
	else if (West[pos1] == pos2) {
		matching[pos1] = 9;
		matching[pos2] = 3;
	}
	else if (East[pos1] == pos2) {
		matching[pos1] = 3;
		matching[pos2] = 9;
	}

	record_matching(verbose_level - 1);


	if (f_vvv) {
		//print_matching(M, N, matching, cout);
	}
finish:
	if (f_v) {
		cout << "domino_assignment::modify_matching before delete []" << endl;
	}

	delete [] used;
	delete [] reached1;
	delete [] reached2;
	delete [] list1;
	delete [] list2;
	delete [] length1;
	delete [] length2;
	delete [] prec1;
	delete [] prec2;

	delete [] do_brake;
	delete [] point_q;
	delete [] point_p;

	if (f_v) {
		print(cout);
		cout << "domino_assignment::modify_matching done ret=" << ret << endl;
	}

	return ret;

}



void domino_assignment::follow_the_matching(
		int l, int *used, int *reached,
		int *list, int *length, int *prec,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int u, from, to, p, q;

	if (l > 1) {
		from = length[l - 2];
	}
	else {
		from = 0;
	}
	to = length[l - 1];
	length[l] = to;
	for (u = from; u < to; u++) {
		p = list[u];
		if (matching[p] == 12) {
			q = North[p];
		}
		else if (matching[p] == 6) {
			q = South[p];
		}
		else if (matching[p] == 3) {
			q = East[p];
		}
		else if (matching[p] == 9) {
			q = West[p];
		}
		else {
			cout << "domino_assignment::follow_the_matching illegal value in matching" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "following the matching from " << p << " to " << q << endl;
			}
		if (used[p] && l > 1) {
			cout << "in follow_the_matching, something is wrong, used[p]" << endl;
			exit(1);
		}
		list[length[l]] = q;
		prec[length[l]] = u;
		reached[q] = TRUE;
		length[l]++;
	}
	if (f_v) {
		cout << "after follow_the_matching" << endl;
		//print_list(l + 1, list, length, prec);
	}
}


int domino_assignment::find_match(int l,
	int *reached1, int *list1, int *length1, int *prec1,
	int *reached2, int *list2, int *length2, int *prec2,
	int verbose_level)
{
	int u, p;

	for (u = length1[l - 2]; u < length1[l - 1]; u++) {
		p = list1[u];
		//cout << "find_match, looking at point " << p << endl;
		if (reached2[p]) {
			//cout << "found a matching, point " << p << endl;
			return u;
		}
	}
	return -1;
}

int domino_assignment::breadth_search(
		int l, int *used, int *reached,
		int *list, int *length, int *prec,
		int verbose_level)
{
	int from, to, u, p, q;
	if (l > 1) {
		from = length[l - 2];
	}
	else {
		from = 0;
	}
	to = length[l - 1];
	length[l] = length[l - 1];
	for (u = from; u < to; u++) {
		p = list[u];
		q = North[p];
		if (q != -1 && !used[q] && !reached[q]) {
			reached[q] = TRUE;
			prec[length[l]] = u;
			list[length[l]] = q;
			length[l]++;
		}
		q = South[p];
		if (q != -1 && !used[q] && !reached[q]) {
			reached[q] = TRUE;
			prec[length[l]] = u;
			list[length[l]] = q;
			length[l]++;
		}
		q = West[p];
		if (q != -1 && !used[q] && !reached[q]) {
			reached[q] = TRUE;
			prec[length[l]] = u;
			list[length[l]] = q;
			length[l]++;
		}
		q = East[p];
		if (q != -1 && !used[q] && !reached[q]) {
			reached[q] = TRUE;
			prec[length[l]] = u;
			list[length[l]] = q;
			length[l]++;
		}
	}
	//cout << "after breadth_search()" << endl;
	//print_list(l + 1, list, length, prec);
	if (length[l] == length[l - 1]) {
		//cout << "breadth_search() did not find new points" << endl;
		return FALSE;
	}
	else {
		return TRUE;
	}
}

void domino_assignment::rotate_once(int ro, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "domino_assignment::rotate_once, verbose_level=" << verbose_level << endl;
		cout << "ro=" << ro << endl;
		cout << "assi[ro * 5 + 2]=" << assi[ro * 5 + 2] << endl;
	}
	if (assi[ro * 5 + 2] >= 4 || assi[ro * 5 + 2] == 1) {
		if (f_v) {
			cout << "domino_assignment::rotate_once, before do_horizontal_rotate" << endl;
		}
		do_horizontal_rotate(ro, verbose_level);
		if (f_v) {
			cout << "domino_assignment::rotate_once, after do_horizontal_rotate" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "domino_assignment::rotate_once, before do_vertical_rotate" << endl;
		}
		do_vertical_rotate(ro, verbose_level);
		if (f_v) {
			cout << "domino_assignment::rotate_once, after do_vertical_rotate" << endl;
		}
	}
	if (f_v) {
		cout << "domino_assignment::rotate_once done" << endl;
	}

}

void domino_assignment::rotate_randomized(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::os_interface Os;
	int ro, r1, i;

	if (f_v) {
		cout << "domino_assignment::rotate_randomized" << endl;
	}
	r1 = Os.random_integer(tot_dom);
		// always choose 0 as the first random number
	r1 = Os.random_integer(tot_dom);
		// number of rotate,
	if (f_v) {
		cout << "rotate " << r1 << " times" << endl;
	}

	for (i = 0; i < r1; i++){

		ro = Os.random_integer(tot_dom - 1);
			// always choose 0 as the first random number (Not true!)
		ro = Os.random_integer(tot_dom - 1);
			// pick a domino to rotate

		if (f_v) {
			cout << "rotate # " << i << ", rotating domino " << ro << ", which is x("<<assi[ro*5+0]<<","<<assi[ro*5+1]<<","<<assi[ro*5+2]<<","<<assi[ro*5+3]<<","<<assi[ro*5+4]<<")"<<endl;
		}
		if (assi[ro * 5 + 2] >= 4 || assi[ro * 5 + 2] == 1) {
				do_horizontal_rotate(ro, verbose_level - 1);
		}
		else {
			do_vertical_rotate(ro, verbose_level - 1);
		}
	}
	if (f_v) {
		cout << "domino_assignment::rotate_randomized" << endl;
	}

}

void domino_assignment::do_horizontal_shift(int ro, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j2;

	if (f_v) {
		cout << "domino_assignment::do_horizontal_shift, verbose_level=" << verbose_level << endl;
		cout << "assi[ro * 5 + 4]=" << assi[ro * 5 + 4] << endl;
	}
	if (assi[ro * 5 + 4] != N - 2) {
		modify_matching(
				ro,
				assi[ro * 5 + 0],
				assi[ro * 5 + 1],
				assi[ro * 5 + 2],
				assi[ro * 5 + 3],
				assi[ro * 5 + 4] + 1,
				verbose_level - 1);
		j2 = assi[ro * 5 + 4] + 1;
	}
	else {
		modify_matching(
				ro,
				assi[ro * 5 + 0],
				assi[ro * 5 + 1],
				assi[ro * 5 + 2],
				assi[ro * 5 + 3],
				assi[ro * 5 + 4] - 1,
				verbose_level - 1);
		j2 = assi[ro * 5 + 4] - 1;
	}
	//j1 = assi[ro * 5 + 4];
	assi[ro * 5 + 4] = j2;
}


void domino_assignment::do_vertical_shift(int ro, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i2;

	if (f_v) {
		cout << "domino_assignment::do_vertical_shift, verbose_level=" << verbose_level << endl;
		cout << "assi[ro * 5 + 3]=" << assi[ro * 5 + 3] << endl;
	}
	if (assi[ro * 5 + 3] != M - 2){
		modify_matching(
				ro,
				assi[ro * 5 + 0],
				assi[ro * 5 + 1],
				assi[ro * 5 + 2],
				assi[ro * 5 + 3] + 1,
				assi[ro * 5 + 4],
				verbose_level - 1);
		i2 = assi[ro * 5 + 3] + 1;
	}
	else {
		modify_matching(
				ro,
				assi[ro * 5 + 0],
				assi[ro * 5 + 1],
				assi[ro * 5 + 2],
				assi[ro * 5 + 3] - 1,
				assi[ro * 5 + 4],
				verbose_level - 1);
		i2 = assi[ro * 5 + 3] - 1;
	}
	//i1 = assi[ro * 5 + 3];
	assi[ro * 5 + 3] = i2;
}

void domino_assignment::shift_once(int ro, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "domino_assignment::shift_once shifting domino " << ro << " verbose_level=" << verbose_level << endl;
	}
	if (f_vv) {
		cout << "shifting domino " << ro << ", which is x(" << assi[ro * 5 + 0] << "," << assi[ro*5+1]<<","<<assi[ro*5+2]<<","<<assi[ro*5+3]<<","<<assi[ro*5+4]<<")"<<endl;
	}
	if ( assi[ro * 5 + 2] >= 4 || assi[ro * 5 + 2] == 1) {
		do_horizontal_shift(ro, verbose_level - 1);
	}
	else {
		do_vertical_shift(ro, verbose_level - 1);
	}


}

void domino_assignment::shift_once_randomized(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::os_interface Os;
	int ro;

	if (f_v) {
		cout << "domino_assignment::shift_once_randomized" << endl;
	}

	ro = Os.random_integer(tot_dom);

	shift_once(ro, verbose_level);

}

void domino_assignment::shift_randomized(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ro, r1, i;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "domino_assignment::shift_randomized" << endl;
	}
	r1 = Os.random_integer(tot_dom * 2);
		// random number of shifts
	if (f_v) {
		cout << "domino_assignment::shift_randomized " << r1 << " times" << endl;
	}
	for (i = 0; i < r1; i++){

		ro = Os.random_integer(tot_dom);

		shift_once(ro, verbose_level);
	}

}

void domino_assignment::flip_after_shift(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "domino_assignment::flip_after_shift" << endl;
	}
	for (i = 0; i < brake_cnt; i++){
		if (assi[brake[i] * 5 + 2] > 1) {
			do_flip(brake[i], verbose_level - 1);
		}
	}
}

void domino_assignment::print_matching(std::ostream &ost)
{
	int i, j, m;

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			m = matching[i * N + j];
			if (m == 12) {
				ost << "N";
			}
			else if (m == 6) {
				ost << "S";
			}
			else if (m == 3) {
				ost << "E";
			}
			else if (m == 9) {
				ost << "W";
			}
			else {
				cout << "domino_assignment::print_matching unexpected value in matching at position " << i << "," << j << " is " << m << endl;
				exit(1);
			}
		}
		ost << endl;
	}
}

void domino_assignment::print(std::ostream &ost)
{
	int i, j, a, m;

	//compute_domino_matrix(tot_dom);
	for (i = 0; i < M; i++) {
		ost << setw(3) << i << " ";
		for (j = 0; j < N; j++) {
			a = A[i * N + j];
			ost << a;
		}
		cout << " ";
		for (j = 0; j < N; j++) {
			m = matching[i * N + j];
			if (m == 12) {
				ost << "N";
			}
			else if (m == 6) {
				ost << "S";
			}
			else if (m == 3) {
				ost << "E";
			}
			else if (m == 9) {
				ost << "W";
			}
			else {
				cout << "domino_assignment::print unexpected value in matching at position " << i << "," << j << " is " << m << endl;
				exit(1);
			}
		}
		cout << " ";
		for (j = 0; j < N; j++) {
			a = broken_dom[i * N + j];
			ost << setw(3) << a;
			if (j < N - 1) {
				cout << ",";
			}
		}
		ost << endl;
	}
}


void domino_assignment::prepare_latex(std::string &photo_label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "domino_assignment::prepare_latex" << endl;
	}
	string photo_label_tex;
	string tex_file_name;
	string pdf_file_name;
	//int i, j, l;


#if 0
	string str_underscore("_");
	while (photo_label.find(str_underscore) != std::string::npos) {
		photo_label.replace(photo_label.find(str_underscore),str_underscore.length(),"\\_");
	}
#endif


	tex_file_name.assign(photo_label);
	pdf_file_name.assign(photo_label);
	tex_file_name.append(".tex");
	pdf_file_name.append(".pdf");

	{
		ofstream f(tex_file_name);

		f << "\\documentclass[]{article}" << endl;
		f << "\\usepackage{amsmath}" << endl;
		f << "\\usepackage{amssymb}" << endl;
		f << "\\usepackage{latexsym}" << endl;
		f << "\\usepackage{epsfig}" << endl;
		f << "\\usepackage{tikz}" << endl;
		f << "%%\\usepackage{supertabular}" << endl;
		f << "\\evensidemargin 0in" << endl;
		f << "\\oddsidemargin 0in" << endl;
		f << "\\marginparwidth 0pt" << endl;
		f << "\\marginparsep 0pt" << endl;
		f << "\\topmargin -1in" << endl;
		f << "\\headheight 0.7cm" << endl;
		f << "\\headsep 1.8cm" << endl;
		f << "%%\\footheight 0.7cm" << endl;
		f << "\\footskip 2cm" << endl;
		f << "\\textheight 22cm" << endl;
		f << "\\textwidth 6.2in" << endl;
		f << "\\marginparpush 0pt" << endl;
		f << "%%\\newcommand{\\dominowidth}{167mm}" << endl;
		f << "\\newcommand{\\dominowidth}{190mm}" << endl;
		f << "\\newcommand{\\dominowidthsmall}{90mm}" << endl;
		f << "\\title{" << photo_label_tex << "}" << endl;
		f << "%%\\author{{\\sc }}" << endl;
		f << "\\date{\\today}" << endl;
		f << "\\pagestyle{empty}" << endl;
		f << "\\begin{document}" << endl;
		f << "\\pagestyle{empty}" << endl;
		f << "%%\\maketitle" << endl;
		f << "\\begin{center}" << endl;
		f << "\\input " << photo_label << "_solution_0.tex" << endl;


#if 0
		if (f_intermediate) {
			for (i = 0; i < Draw_cnt; i++) {
				char fname[1000];
				snprintf(fname, sizeof(fname), "%s_c%d", photo_label, i);
				f << "\\epsfig{file=" << fname << ".1,width=\\dominowidthsmall}\\\\\\bigskip" << endl;
				f << "after " << record_nb_improvements[i] << " improvements\\\\" << endl;
				if (record_improvement_type[i] == IMPROVE_HORIZ_ROTATE) {
					f << "by horizontal rotation" << endl;
					}
				else if (record_improvement_type[i] == IMPROVE_VERTIC_ROTATE) {
					f << "by vertical rotation" << endl;
					}
				}
			}
#endif

		f << "\\end{center}" << endl;

		//print_assi_latex(f, tot_dom, assi);

		f << "\\end{document}" << endl;
	}
	string cmd1;
	string cmd3;

	cmd1.assign("pdflatex ");
	cmd1.append(tex_file_name);

	cmd3.assign("open ");
	cmd3.append(pdf_file_name);
	cmd3.append(" &");


	system(cmd1.c_str());
	system(cmd3.c_str());


}

void domino_assignment::record_flip(int idx, int verbose_level)
{
	domino_change C;

	C.init(this, 1 /* type_of_change */, verbose_level);

	Changes.push_back(C);
	nb_changes++;
}

void domino_assignment::record_swap(int s1, int s2, int verbose_level)
{
	domino_change C;

	C.init(this, 2 /* type_of_change */, verbose_level);

	Changes.push_back(C);

}

void domino_assignment::record_matching(int verbose_level)
{
	domino_change C;

	C.init(this, 3 /* type_of_change */, verbose_level);
	Changes.push_back(C);

	nb_changes++;
}

void domino_assignment::drop_changes_to(int nb_changes_to_drop_to, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int l;

	if (f_v) {
		cout << "domino_assignment::drop_changes_to" << endl;
	}
	l = Changes.size();
	while (Changes.size() > nb_changes_to_drop_to) {
		Changes.pop_back();
	}
	if (f_v) {
		cout << "domino_assignment::drop_changes_to dropping from " << l
				<< " down to " << nb_changes_to_drop_to << endl;
	}
}

void domino_assignment::classify_changes_by_type(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int l, i, t;
	int N[4];

	if (f_v) {
		cout << "domino_assignment::classify_changes_by_type" << endl;
	}
	Int_vec_zero(N, 4);
	l = Changes.size();
	for (i = 0; i < l; i++) {
		t = Changes[i].type_of_change;
		if (t > 3) {
			cout << "type of change is out of bounds, t=" << t << endl;
			exit(1);
		}
		N[t]++;
	}
	cout << "types:" << endl;
	for (i = 0; i < 4; i++) {
		cout << i << " : " << N[i] << endl;
	}
}

void domino_assignment::get_cost_function(int *&Cost, int &len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "domino_assignment::get_cost_function" << endl;
	}
	len = Changes.size();
	Cost = NEW_int(len);

	for (i = 0; i < len; i++) {
		Cost[i] = Changes[i].cost_after_change;
	}
	if (f_v) {
		cout << "domino_assignment::get_cost_function done" << endl;
	}

}


}}}

