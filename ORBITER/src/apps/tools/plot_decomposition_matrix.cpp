// plot_decomposition_matrix.C
// 
// Anton Betten
// 3/18/2013
//
// 
//
//

#include "orbiter.h"

// global data:

INT t0; // the system time when the program started

void split(const BYTE *fname_base, INT split_v, INT f_dots, UBYTE *D, INT m, INT n, INT xmax, INT ymax, 
	double scale, double line_width);
void read_orbit_data(const BYTE *fname, 
	INT &nb_orbits, INT &N, 
	INT *&orbit_fst, 
	INT *&orbit_len, 
	INT *&orbit_number, 
	INT *&orbit_perm, 
	INT *&orbit_perm_inv, 
	INT *&schreier_vector, 
	INT *&schreier_prev, 
	INT verbose_level);
void draw_it(const BYTE *fname_base, INT idx, INT f_dots, UBYTE *D, INT m, INT n, INT xmax, INT ymax, 
	double scale, double line_width);
void draw_it2(mp_graphics &G, INT f_dots, UBYTE *D, INT m, INT n, INT xmax, INT ymax);

int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	const BYTE *fname1 = NULL;
	const BYTE *fname2 = NULL;
	const BYTE *fname3 = NULL;
	INT f_output_fname = FALSE;
	const BYTE *output_fname = NULL;
	INT xmax = 1000;
	INT ymax = 1000;
	INT f_dots = FALSE;
	INT f_split_v = FALSE;
	INT split_v = 1;
	INT f_scale = FALSE;
	double scale = .45;
	INT f_line_width = FALSE;
	double line_width = 1.5;

	t0 = os_ticks();

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file1") == 0) {
			fname1 = argv[++i];
			cout << "-file1 " << fname1 << endl;
			}
		else if (strcmp(argv[i], "-file2") == 0) {
			fname2 = argv[++i];
			cout << "-file2 " << fname2 << endl;
			}
		else if (strcmp(argv[i], "-file3") == 0) {
			fname3 = argv[++i];
			cout << "-file3 " << fname3 << endl;
			}
		else if (strcmp(argv[i], "-output_fname") == 0) {
			f_output_fname = TRUE;
			output_fname = argv[++i];
			cout << "-output_fname " << output_fname << endl;
			}
		else if (strcmp(argv[i], "-x") == 0) {
			xmax = atoi(argv[++i]);
			cout << "-x " << xmax << endl;
			}
		else if (strcmp(argv[i], "-y") == 0) {
			ymax = atoi(argv[++i]);
			cout << "-y " << ymax << endl;
			}
		else if (strcmp(argv[i], "-dots") == 0) {
			f_dots = TRUE;
			cout << "-dots" << endl;
			}
		else if (strcmp(argv[i], "-split_v") == 0) {
			f_split_v = TRUE;
			split_v = atoi(argv[++i]);
			cout << "-split_v " << split_v << endl;
			}
		else if (strcmp(argv[i], "-scale") == 0) {
			f_scale = TRUE;
			sscanf(argv[++i], "%lf", &scale);
			cout << "-scale " << scale << endl;
			}
		else if (strcmp(argv[i], "-line_width") == 0) {
			f_line_width = TRUE;
			sscanf(argv[++i], "%lf", &line_width);
			cout << "-line_width " << line_width << endl;
			}
		}
	if (fname1 == NULL) {
		cout << "Please specify -file1 <fname1>" << endl;
		exit(1);
		}
	if (fname2 == NULL) {
		cout << "Please specify -file2 <fname2>" << endl;
		exit(1);
		}
	if (fname3 == NULL) {
		cout << "Please specify -file3 <fname3>" << endl;
		exit(1);
		}
	if (f_output_fname == FALSE) {
		cout << "Please specify -output_fname <output_fname>" << endl;
		exit(1);
		}
	
	INT *M1;
	INT m1, n1;
	INT *M2;
	INT m2, n2;
	UBYTE *D; // bitvector

	INT m, n;
	INT *up_fst;
	INT *up_len;
	INT j, a, col, orb, idx, row, len, sol_idx;

	cout << "reading file " << fname1 << endl;
	INT_matrix_read_csv(fname1, M1, m1, n1, verbose_level - 1);
	cout << "reading file " << fname2 << endl;
	INT_matrix_read_csv(fname2, M2, m2, n2, verbose_level - 1);

	m = m1;
	n = M2[(m2 - 1) * n2 + 0] + 1;

	cout << "m=" << m << endl;
	cout << "n=" << n << endl;

	up_fst = NEW_INT(m + 1);
	up_len = NEW_INT(m);
	j = 0;
	for (i = 0; i < m; i++) {
		a = M1[i];
		up_fst[i] = j;
		up_len[i] = a;
		j += a;
		}
	up_fst[m] = j;

	cout << "allocating D, of size " << m * n << endl;
	len = (m * n + 7) >> 3;
	cout << "len = " << len << endl;
	D = NEW_UBYTE(len);
	for (i = 0; i < len; i++) {
		D[i] = 0;
		}


	cout << "Reading file " << fname3 << endl;
	INT nb_orbits, N;
	INT *orbit_fst;
	INT *orbit_len;
	INT *orbit_number;
	INT *orbit_perm;
	INT *orbit_perm_inv;
	INT *schreier_vector;
	INT *schreier_prev;

	read_orbit_data(fname3, 
		nb_orbits, N, 
		orbit_fst, 
		orbit_len, 
		orbit_number, 
		orbit_perm, 
		orbit_perm_inv, 
		schreier_vector, 
		schreier_prev, 
		verbose_level);
	cout << "Reading file " << fname3 << " done" << endl;

	
	for (i = 0; i < m2; i++) {
		if (i && (i % 1000) == 0) {
			cout << i << " / " << m2 << endl;
			}
		col = M2[i * n2 + 0];
		sol_idx = M2[i * n2 + 2];
		orb = orbit_number[orbit_perm_inv[sol_idx]];
		if (!INT_vec_search(up_fst, m + 1, orb, idx)) {
			// find the last occurence 
			idx--;
			}
		row = idx;
		if (up_fst[row] <= orb && up_fst[row + 1] > orb) {
			}
		else {
			cout << "error, did not find the right row" << endl;
			cout << "i=" << i << endl;
			cout << "col = " << col << endl;
			cout << "orb = " << orb << endl;
			cout << "row=" << row << endl;
			cout << "up_fst[row] = " << up_fst[row] << endl;
			cout << "up_fst[row + 1] = " << up_fst[row + 1] << endl;
			exit(1);
			}
		bitvector_m_ii(D, row * n + col, 1);
		//D[row * n + col]++;
		}

	cout << "decomposition matrix computed" << endl;

	if (f_split_v) {
		split(output_fname, split_v, f_dots, D, m, n, xmax, ymax, scale, line_width);
		}
	else {
		draw_it(output_fname, 0/*idx*/, f_dots, D, m, n, xmax, ymax, scale, line_width);
		}
	
	the_end_quietly(t0);
}

void split(const BYTE *fname_base, INT split_v, INT f_dots, UBYTE *D, INT m, INT n, INT xmax, INT ymax, 
	double scale, double line_width)
{
	INT nb_pic;
	INT h;
	INT i0, i1, i, j, m1, len;
	UBYTE *D1;

	nb_pic = m / split_v + 1;
	for (h = 0; h < nb_pic; h++) {
		i0 = h * split_v;
		i1 = (h + 1) * split_v;
		i1 = MINIMUM(m, i1);
		m1 = i1 - i0;
		len = (m1 * n + 7) >> 3;
		cout << "len = " << len << endl;
		D1 = NEW_UBYTE(len);
		for (i = 0; i < len; i++) {
			D1[i] = 0;
			}
		for (i = 0; i < m1; i++) {
			for (j = 0; j < n; j++) {
				if (bitvector_s_i(D, (i0 + i) * n + j)) {
					bitvector_m_ii(D1, i * n + j, 1);
					}
				}
			}
		draw_it(fname_base, h/*idx*/, f_dots, D1, m1, n, xmax, ymax, scale, line_width);
		FREE_UBYTE(D1);
		}
}

void read_orbit_data(const BYTE *fname, 
	INT &nb_orbits, INT &N, 
	INT *&orbit_fst, 
	INT *&orbit_len, 
	INT *&orbit_number, 
	INT *&orbit_perm, 
	INT *&orbit_perm_inv, 
	INT *&schreier_vector, 
	INT *&schreier_prev, 
	INT verbose_level)
// Reads from the file 'fname_staborbits'
// Reads nb_orbits, N, 
// orbit_fst[nb_orbits + 1]
// orbit_len[nb_orbits]
// orbit_number[N]
// orbit_perm[N]
// schreier_vector[N]
// schreier_prev[N]
// and computed orbit_perm_inv[N]
{
	INT f_v = (verbose_level >= 1);
	
	ifstream f(fname);
	INT i, a;
	
	if (f_v) {
		cout << "read_orbit_data" << endl;
		}
	f >> nb_orbits >> N;
	if (f_v) {
		cout << "nb_orbits=" << nb_orbits << endl;
		cout << "N=" << N << endl;
		}
	
	orbit_fst = NEW_INT(nb_orbits + 1);
	orbit_len = NEW_INT(nb_orbits);
	orbit_number = NEW_INT(N);
	orbit_perm = NEW_INT(N);
	orbit_perm_inv = NEW_INT(N);
	schreier_vector = NEW_INT(N);
	schreier_prev = NEW_INT(N);
	
	for (i = 0; i < nb_orbits; i++) {
		f >> a;
		f >> orbit_fst[i];
		f >> orbit_len[i];
		}
	for (i = 0; i < N; i++) {
		f >> a;
		f >> orbit_number[i];
		f >> orbit_perm[i];
		f >> schreier_vector[i];
		f >> schreier_prev[i];
		}
	orbit_fst[nb_orbits] = N;
	perm_inverse(orbit_perm, orbit_perm_inv, N);
	f >> a;
	if (a != -1) {
		cout << "problem in read_orbit_data" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "read_orbit_data finished" << endl;
		}
}

void draw_it(const BYTE *fname_base, INT idx, INT f_dots, UBYTE *D, INT m, INT n, INT xmax, INT ymax, 
	double scale, double line_width)
{
	mp_graphics G;
	BYTE fname_base2[1000];
	BYTE fname[1000];
	INT f_embedded = TRUE;
	INT f_sideways = TRUE;
	
	sprintf(fname_base2, "%s_%ld", fname_base, idx);
	sprintf(fname, "%s.mp", fname_base2);
	{
	G.setup(fname_base2, 0, 0, ONE_MILLION, ONE_MILLION, xmax, ymax, f_embedded, f_sideways, 
		scale, line_width);

	//G.frame(0.05);
	
	draw_it2(G, f_dots, D, m, n, xmax, ymax);

	G.finish(cout, TRUE);
	}
	cout << "draw_it written file " << fname << " of size " << file_size(fname) << endl;
}

void draw_it2(mp_graphics &G, INT f_dots, UBYTE *D, INT m, INT n, INT xmax, INT ymax)
{
	grid_frame F;
	INT i, j, a, cnt, mn;
	
	mn = MAXIMUM(m, n);
	F.f_matrix_notation = TRUE;
	F.m = m;
	F.n = n;
	F.origin_x = 0.;
	F.origin_y = 0.;
	F.dx = ONE_MILLION / (10 * mn);
	F.dy = ONE_MILLION / (10 * mn);

	cout << "draw_it2" << endl;
	cout << "dx=" << F.dx << endl;
	cout << "dy=" << F.dy << endl;
	G.grid_polygon2(&F, 0 - 1, 0 - 1, 10 * m + 1, 0 - 1);
	G.grid_polygon2(&F, 10 * m + 1, 0 - 1, 10 * m + 1, 10 * n + 1);
	G.grid_polygon2(&F, 10 * m + 1, 10 * n + 1, 0 - 1, 10 * n + 1);
	G.grid_polygon2(&F, 0 - 1, 10 * n + 1, 0 - 1, 0 - 1);

	G.sf_interior(100);
	G.sf_color(1);
	
	cnt = 0;
	for (i = 0; i < m; i++) {
		if (i && (i % 1000) == 0) {
			cout << "draw_it2 " << i << " / " << m << endl;
			}
		for (j = 0; j < n; j++) {
			//a = Aij(i, j);
			a = bitvector_s_i(D, i * n + j);
			if (a == 0) {
				continue;
				}
			cnt++;

			// if (cnt > 4000)  continue;
			//G.grid_fill_polygon4(&F, i, j, i + 1, j, i + 1, j + 1, i, j + 1);

			if (f_dots) {
				G.grid_polygon2(&F, 10 * i, 10 * j, 10 * i, 10 * j);
				}
			else {
				G.sf_interior(100);
				G.sf_color(1);
#if 0
				G.grid_fill_polygon4(&F, 
					10 * i + 1, 10 * j + 1, 
					10 * (i + 1) - 1, 10 * j + 1, 
					10 * (i + 1) - 1, 10 * (j + 1) - 1, 
					10 * i + 1, 10 * (j + 1) - 1);
#else
				G.grid_polygon5(&F, 
					10 * i + 1, 10 * j + 1, 
					10 * (i + 1) - 1, 10 * j + 1, 
					10 * (i + 1) - 1, 10 * (j + 1) - 1, 
					10 * i + 1, 10 * (j + 1) - 1, 
					10 * i + 1, 10 * j + 1);
#endif
				//G.grid_polygon2(&F, i, j, i + 1, j);
				//G.grid_polygon2(&F, i + 1, j, i + 1, j + 1);
				//G.grid_polygon2(&F, i + 1, j + 1, i, j + 1);
				//G.grid_polygon2(&F, i, j + 1, i, j);
				}
			}
		}
	cout << "draw_it2 # of non-zero coefficients = " << cnt << endl;
}


