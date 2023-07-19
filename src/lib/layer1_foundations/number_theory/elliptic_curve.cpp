// elliptic_curve.cpp
// 
// Anton Betten
// Oct 27, 2009
//
//
// pulled out of crypto.cpp: November 19, 2014
//
//

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace number_theory {



elliptic_curve::elliptic_curve()
{
	q = 0;
	p = 0;
	e = 0;
	b = c = 0;
	nb = 0;
	T = NULL;
	A = NULL;
	F = NULL;
}

elliptic_curve::~elliptic_curve()
{
	if (T) {
		FREE_int(T);
	}
	if (A) {
		FREE_int(A);
	}
}


void elliptic_curve::init(
		field_theory::finite_field *F,
		int b, int c,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "elliptic_curve::init q=" << F->q
				<< " b=" << b << " c=" << c << endl;
	}
	elliptic_curve::F = F;
	q = F->q;
	p = F->p;
	e = F->e;
	elliptic_curve::b = b;
	elliptic_curve::c = c;


	if (f_v) {
		cout << "elliptic_curve::init before compute_points" << endl;
	}

	compute_points(verbose_level);

	if (f_v) {
		cout << "elliptic_curve::init after compute_points" << endl;
	}

#if 0
	if (E.nb < 20) {
		print_integer_matrix_width(cout, E.A, E.nb, E.nb, E.nb, 3);
		}
	
#endif


	
#if 0
	cout << "point : order" << endl;
	for (i = 0; i < E.nb; i++) {
		j = order_of_point(E, i);
		cout << setw(4) << i << " : " << setw(4) << j << endl;
		}
	
	cout << "the curve has " << E.nb << " points" << endl;
#endif

#if 0
	{
	int a, b, c;
	a = 1;
	b = 2;
	c = E.A[a * E.nb + b];
	cout << "P_" << a << " + P_" << b << " = P_" << c << endl;
	}

	j = multiple_of_point(E, 0, 37);
	cout << "37 * P_0 = P_" << j << endl;
#endif
	if (f_v) {
		cout << "elliptic_curve::init done" << endl;
	}
}


void elliptic_curve::compute_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int x, y, y1, y2;
	int r;
	int bound;

	if (f_v) {
		cout << "elliptic_curve::compute_points" << endl;
	}
	bound = q + 1 + 2 * ((int)(sqrt(q)) + 1); // Hasse Weil bound
	


	T = NEW_int(bound * 3);
	nb = 0;


	// the point at infinity comes first:
	add_point_to_table(0, 1, 0);


	for (x = 0; x < q; x++) {
		r = evaluate_RHS(x);
		if (r == 0) {
			add_point_to_table(x, 0, 1);
			if (nb == bound) {
				cout << "elliptic_curve::compute_points The number of points exceeds the bound" << endl;
				exit(1);
			}
			//cout << nb++ << " : (" << x << "," << 0 << ",1)" << endl;
		}
		else {
			if (F->is_square(r)) {

				y = F->square_root(r);

				y1 = y;
				y2 = F->negate(y);
				if (y2 == y1) {
					add_point_to_table(x, y1, 1);
					if (nb == bound) {
						cout << "elliptic_curve::compute_points The number of points "
								"exceeds the bound" << endl;
						exit(1);
					}
				}
				else {
					if (y2 < y1) {
						y1 = y2;
						y2 = y;
					}
					add_point_to_table(x, y1, 1);
					if (nb == bound) {
						cout << "elliptic_curve::compute_points The number of points "
								"exceeds the bound" << endl;
						exit(1);
					}
					add_point_to_table(x, y2, 1);
					if (nb == bound) {
						cout << "elliptic_curve::compute_points The number of points "
								"exceeds the bound" << endl;
						exit(1);
					}
				}
			}
			else {
				// no point for this x coordinate
			}

#if 0
			if (p != 2) {
				l = Legendre(r, q, 0);

				if (l == 1) {
					y = sqrt_mod_involved(r, q);
						// DISCRETA/global.cpp

					if (F->mult(y, y) != r) {
						cout << "elliptic_curve::compute_points There is a problem "
								"with the square root" << endl;
						exit(1);
					}
					y1 = y;
					y2 = F->negate(y);
					if (y2 < y1) {
						y1 = y2;
						y2 = y;
					}
					add_point_to_table(x, y1, 1);
					if (nb == bound) {
						cout << "elliptic_curve::compute_points The number of points "
								"exceeds the bound" << endl;
						exit(1);
					}
					add_point_to_table(x, y2, 1);
					if (nb == bound) {
						cout << "elliptic_curve::compute_points The number of points "
								"exceeds the bound" << endl;
						exit(1);
					}
					//cout << nb++ << " : (" << x << ","
					// << y << ",1)" << endl;
					//cout << nb++ << " : (" << x << ","
					// << F.negate(y) << ",1)" << endl;
				}
			}
			else {
				y = F->frobenius_power(r, e - 1);
				add_point_to_table(x, y, 1);
				if (nb == bound) {
					cout << "elliptic_curve::compute_points The number of points exceeds "
							"the bound" << endl;
					exit(1);
				}
				//cout << nb++ << " : (" << x << ","
				// << y << ",1)" << endl;
			}
#endif

		}
	}


	if (nb == bound) {
		cout << "elliptic_curve::compute_points The number of points exceeds the bound" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "elliptic_curve::compute_points done, "
				"we found " << nb << " points" << endl;
	}
}

void elliptic_curve::add_point_to_table(
		int x, int y, int z)
{
	T[nb * 3 + 0] = x;
	T[nb * 3 + 1] = y;
	T[nb * 3 + 2] = z;
	nb++;
}

int elliptic_curve::evaluate_RHS(int x)
// evaluates x^3 + bx + c
{
	int x2, x3, t;
	
	x2 = F->mult(x, x);
	x3 = F->mult(x2, x);
	t = F->add(x3, F->mult(b, x));
	t = F->add(t, c);
	return t;
}

void elliptic_curve::print_points()
{
	int i;
	
	cout << "i : point (x,y,z)" << endl;
	for (i = 0; i < nb; i++) {
		cout << setw(4) << i << " & " << T[i * 3 + 0] << ","
				<< T[i * 3 + 1] << "," << T[i * 3 + 2] << "\\\\" << endl;
	}
}

void elliptic_curve::print_points_affine()
{
	int i;
	
	cout << "i : point (x,y,z)" << endl;
	for (i = 0; i < nb; i++) {
		cout << setw(4) << i << " & ";
		if (T[i * 3 + 2] == 0) {
			cout << "\\cO";
		}
		else {
			cout << "(" << T[i * 3 + 0] << "," << T[i * 3 + 1] << ")";
		}
		cout << "\\\\" << endl;
	}
}


void elliptic_curve::addition(
	int x1, int y1, int z1,
	int x2, int y2, int z2,
	int &x3, int &y3, int &z3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "elliptic_curve::addition: ";
		cout << "(" << x1 << "," << y1 << "," << z1 << ")";
		cout << " + ";
		cout << "(" << x2 << "," << y2 << "," << z2 << ")";
		cout << endl;
	}

	number_theory::number_theory_domain NT;

	NT.elliptic_curve_addition(F, b, c,
			x1, y1, z1,
			x2, y2, z2,
			x3, y3, z3, verbose_level);

	if (f_v) {
		cout << "elliptic_curve::addition done";
	}
}

void elliptic_curve::save_incidence_matrix(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	int i, x, y, z;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "elliptic_curve::save_incidence_matrix" << endl;
	}
	M = NEW_int(q * q);
	Int_vec_zero(M, q * q);
	for (i = 0; i < nb; i++) {
		x = T[i * 3 + 0];
		y = T[i * 3 + 1];
		z = T[i * 3 + 2];
		if (z == 0) {
			continue;
		}
		if (z != 1) {
			cout << "elliptic_curve::save_incidence_matrix point is not normalized" << endl;
			exit(1);
		}
		M[(q - 1 - y) * q + x] = 1;
	}
	Fio.Csv_file_support->int_matrix_write_csv(
			fname, M, q, q);
	if (f_v) {
		cout << "elliptic_curve::save_incidence_matrix written file "
				<< fname << " of size " << Fio.file_size(fname) << endl;
	}
	FREE_int(M);
	if (f_v) {
		cout << "elliptic_curve::save_incidence_matrix done" << endl;
	}
}

void elliptic_curve::draw_grid(
		std::string &fname,
		graphics::layered_graph_draw_options *Draw_options,
		int f_with_grid, int f_with_points, int point_density,
		int f_path, int start_idx, int nb_steps,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int factor_1000 = 1000;
	string fname_full;
	
	if (f_v) {
		cout << "draw_grid" << endl;
	}
	fname_full = fname + ".mp";

	{

		graphics::mp_graphics G;

		G.init(fname_full, Draw_options, verbose_level - 1);

		G.header();
		G.begin_figure(factor_1000);

		draw_grid2(G, f_with_grid, f_with_points, point_density,
				f_path, start_idx, nb_steps,
				verbose_level);
	
	
		G.end_figure();
		G.footer();
	}
	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname_full << " of size "
			<< Fio.file_size(fname_full) << endl;
	if (f_v) {
		cout << "draw_grid done" << endl;
	}
	
}


void elliptic_curve::draw_grid2(
		graphics::mp_graphics &G,
		int f_with_grid, int f_with_points, int point_density,
		int f_path, int start_idx, int nb_steps,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, c, d;
	int x1, x2, x3;

	//int rad = 10000;
	int i, h;

	double *Dx, *Dy;
	int *Px, *Py;
	int dx = G.in_xmax() / q;
	int dy = G.in_ymax() / q;
	int N = 1000;


	Px = NEW_int(N);
	Py = NEW_int(N);
	Dx = new double[N];
	Dy = new double[N];

	
	if (f_v) {
		cout << "elliptic_curve::draw_grid2" << endl;
		cout << "dx=" << dx << " dy=" << dy << endl;
	}




	if (f_v) {
		cout << "elliptic_curve::draw_grid2 drawing grid" << endl;
	}


#if 0
	if (f_with_grid) {
		G.draw_axes_and_grid(
			0., (double)(q - 1), 0., (double)(q - 1),
			x_stretch, y_stretch,
			true /* f_x_axis_at_y_min */,
			true /* f_y_axis_at_x_min */,
			1 /* x_mod */, 1 /* y_mod */, 1, 1,
			-2. /* x_labels_offset */,
			-2. /* y_labels_offset */,
			0.5 /* x_tick_half_width */,
			0.5 /* y_tick_half_width */,
			true /* f_v_lines */, 1 /* subdivide_v */,
			true /* f_h_lines */, 1 /* subdivide_h */,
			verbose_level - 1);
	}
	else {
		G.draw_axes_and_grid(
			0., (double)(q - 1), 0., (double)(q - 1),
			x_stretch, y_stretch,
			true /* f_x_axis_at_y_min */,
			true /* f_y_axis_at_x_min */,
			1 /* x_mod */, 1 /* y_mod */, 1, 1,
			-2. /* x_labels_offset */,
			-2. /* y_labels_offset */,
			0.5 /* x_tick_half_width */,
			0.5 /* y_tick_half_width */,
			true /* f_v_lines */, q - 1 /* subdivide_v */,
			true /* f_h_lines */, q - 1 /* subdivide_h */,
			verbose_level - 1);

	}
#endif


	if (f_with_points) {

		G.sf_color(1 /* fill_color*/);
		G.sf_interior(point_density /* fill_interior */);
		if (f_v) {
			cout << "drawing points, nb=" << nb << endl;
		}

		if (nb >= 40) {
			//rad = 2000;
		}
		for (h = 0; h < nb; h++) {
			x1 = T[3 * h + 0];
			x2 = T[3 * h + 1];
			x3 = T[3 * h + 2];
			//get_ab(q, x1, x2, x3, a, b);
			make_affine_point(x1, x2, x3, a, b, 0);

			
			Dx[0] = a;
			Dy[0] = b;
			Dx[1] = a + 1;
			Dy[1] = b;
			Dx[2] = a + 1;
			Dy[2] = b + 1;
			Dx[3] = a;
			Dy[3] = b + 1;
			Dx[4] = a;
			Dy[4] = b;
			
			for (i = 0; i < 5; i++) {
				Px[i] = Dx[i] * dx;
				Py[i] = Dy[i] * dy;
			}

			cout << "point " << h << " : "
					<< x1 << ", " << x2 << ", " << x3
					<< " : " << a << ", " << b
					<< " : " << Px[0] << "," << Py[0]
					<< endl;

			//G.circle(Px[0], Py[0], rad);
			G.fill_polygon5(Px, Py, 0, 1, 2, 3, 4);
		}


	}
	else {
		cout << "elliptic_curve::draw_grid2 not drawing any points" << endl;
	}



	if (f_path) {
		G.sl_ends(0 /* line_beg_style */, 1 /* line_end_style*/);
		G.sl_thickness(100 /* line_thickness*/);

		int h, j;
		int y1, y2, y3;
	
		if (f_v) {
			cout << "drawing multiples of point" << endl;
			}
		i = start_idx;
		for (h = 0; h < nb_steps; h++) {
			x1 = T[3 * i + 0];
			x2 = T[3 * i + 1];
			x3 = T[3 * i + 2];
			j = A[i * nb + start_idx];
			y1 = T[3 * j + 0];
			y2 = T[3 * j + 1];
			y3 = T[3 * j + 2];
			make_affine_point(x1, x2, x3, a, b, 0);
			make_affine_point(y1, y2, y3, c, d, 0);

			Dx[0] = a;
			Dy[0] = b;
			Dx[1] = c;
			Dy[1] = d;

			for (i = 0; i < 2; i++) {
				Px[i] = Dx[i] * dx + dx / 2;
				Py[i] = Dy[i] * dy + dy / 2;
			}

			G.polygon2(Px, Py, 0, 1);
			i = j;
		}
	}


	// draw the outline box last, so it overlaps all other elements:

	G.sl_ends(0 /* line_beg_style */, 0 /* line_end_style*/);

	G.sl_thickness(100);
	Dx[0] = 0;
	Dy[0] = 0;
	Dx[1] = q + 1;
	Dy[1] = 0;
	Dx[2] = q + 1;
	Dy[2] = q + 1;
	Dx[3] = 0;
	Dy[3] = q + 1;
	Dx[4] = 0;
	Dy[4] = 0;

	for (i = 0; i < 5; i++) {
		Px[i] = Dx[i] * dx;
		Py[i] = Dy[i] * dy;
	}


	G.polygon5(Px, Py, 0, 1, 2, 3, 4);



#if 0
	G.sl_udsty(100);
	a = Xcoord(-1);
	b = Ycoord(-1);
	c = Xcoord(q + 1);
	d = Ycoord(q + 1);
	cout << a << "," << b << "," << c << "," << d << endl;
	G.polygon2(Px, Py, a * Q + b, c * Q + d);
	a = Xcoord(q + 1);
	b = Ycoord(-1);
	c = Xcoord(-1);
	d = Ycoord(q + 1);
	cout << a << "," << b << "," << c << "," << d << endl;
	G.polygon2(Px, Py, a * Q + b, c * Q + d);


	q2 = q >> 1;
	if (ODD(q))
		r = 1;
	else 
		r = 0;
	
	a = Xcoord(q2) + r;
	b = Ycoord(-1);
	c = Xcoord(q2) + r;
	d = Ycoord(q + 1);
	cout << a << "," << b << "," << c << "," << d << endl;
	G.polygon2(Px, Py, a * Q + b, c * Q + d);

	a = Xcoord(-1);
	b = Ycoord(q2) + r;
	c = Xcoord(q + 1);
	d = Ycoord(q2) + r;
	cout << a << "," << b << "," << c << "," << d << endl;
	G.polygon2(Px, Py, a * Q + b, c * Q + d);
#endif


	FREE_int(Px);
	FREE_int(Py);
	delete [] Dx;
	delete [] Dy;



	if (f_v) {
		cout << "draw_grid2 done" << endl;
	}
}

void elliptic_curve::make_affine_point(
		int x1, int x2, int x3,
		int &a, int &b, int verbose_level)
{
	if (x3 == 0) {
		a = q >> 1;
		b = q;
	}
	else {
		if (x3 != 1) {
			cout << "x3 != 1" << endl;
			exit(1);
		}
		a = x1;
		b = x2;
	}
}



#if 0
int multiple_of_point(elliptic_curve &E, int i, int n)
{
	int j, a;

	a = E.nb - 1;
	for (j = 0; j < n; j++) {
		a = E.A[a * E.nb + i];
		}
	return a;
}
#endif

void elliptic_curve::compute_addition_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_v3 = (verbose_level >= 3);
	int i, j, k;
	int x1, x2, x3;
	int y1, y2, y3;
	int z1, z2, z3;

	if (f_v) {
		cout << "elliptic_curve::compute_addition_table" << endl;
	}
	
	A = NEW_int(nb * nb);
	for (i = 0; i < nb; i++) {
		x1 = T[3 * i + 0];
		x2 = T[3 * i + 1];
		x3 = T[3 * i + 2];
		for (j = 0; j < nb; j++) {
			y1 = T[3 * j + 0];
			y2 = T[3 * j + 1];
			y3 = T[3 * j + 2];
			if (false) {
				cout << "add " << i << " " << j << endl;
			}
			addition(
				x1, x2, x3, 
				y1, y2, y3,
				z1, z2, z3, 0 /*verbose_level - 1*/);

			
			k = index_of_point(z1, z2, z3);
			A[i * nb + j] = k;
		}
	}
	if (f_v) {
		cout << "elliptic_curve::compute_addition_table done" << endl;
	}
}

void elliptic_curve::print_addition_table()
{
	l1_interfaces::latex_interface L;

	Int_matrix_print(A, nb, nb);
	L.int_matrix_print_tex(cout, A, nb, nb);
}

#if 0
int elliptic_curve::index_of_point(int x1, int x2, int x3)
{
	int a, i;
	
	if (x3 == 0) {
		return 0;
		}
	if (x3 != 1) {
		a = F->inverse(x3);
		x1 = F->mult(x1, a);
		x2 = F->mult(x2, a);
		x3 = 1;
		}
	for (i = 1; i < nb; i++) {
		if (T[3 * i + 0] == x1 && T[3 * i + 1] == x2) {
			return i;
			}
		}
	cout << "did not find point "
			<< x1 << "," << x2 << "," << x3 << " in table" << endl;
	exit(1);
}
#endif

int elliptic_curve::index_of_point(
		int x1, int x2, int x3)
//int int_vec_search(int *v, int len, int a, int &idx)
// This function finds the last occurence of the element a.
// If a is not found, it returns in idx
// the position where it should be inserted if
// the vector is assumed to be in increasing order.

{
	int l, r, m, res, a;
	int f_found = false;
	int f_v = false;
	
	if (nb == 0) {
		cout << "elliptic_curve::index_of_point "
				"nb == 0" << endl;
		exit(1);
	}

	if (x3 == 0) {
		return 0;
	}
	if (x3 != 1) {
		a = F->inverse(x3);
		x1 = F->mult(x1, a);
		x2 = F->mult(x2, a);
		x3 = 1;
	}

	l = 1;
	r = nb;
	// invariant:
	// v[i] <= a for i < l;
	// v[i] >  a for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle
		if (T[3 * m + 0] > x1) {
			res = 1;
		}
		else if (T[3 * m + 0] < x1) {
			res = -1;
		}
		else {
			if (T[3 * m + 1] > x2) {
				res = 1;
			}
			else if (T[3 * m + 1] < x2) {
				res = -1;
			}
			else {
				res = 0;
			}
		}
		//res = v[m] - a;
		if (f_v) {
			cout << "l=" << l << " r=" << r<< " m=" << m
					<< " T[3 * m + 0]=" << T[3 * m + 0]
					<< " T[3 * m + 1]=" << T[3 * m + 1]
											 << " res=" << res << endl;
		}
		//cout << "search l=" << l << " m=" << m << " r=" 
		//	<< r << "a=" << a << " v[m]=" << v[m]
		// << " res=" << res << endl;
		// so, res is 
		// positive if v[m] > a,
		// zero if v[m] == a,
		// negative if v[m] < a
		if (res <= 0) {
			l = m + 1;
			if (f_v) {
				cout << "elliptic_curve::index_of_point "
						"moving to the right" << endl;
			}
			if (res == 0) {
				f_found = true;
			}
		}
		else {
			if (f_v) {
				cout << "elliptic_curve::index_of_point "
						"moving to the left" << endl;
			}
			r = m;
		}
	}
	// now: l == r; 
	// and f_found is set accordingly */
	if (!f_found) {
		cout << "elliptic_curve::index_of_point "
				"did not find point" << endl;
		cout << "x1=" << x1 << " x2=" << x2 << " x3=" << x3 << endl;
		exit(1);
	}
#if 1
	if (f_found) {
		l--;
	}
#endif
	return l;
}

void elliptic_curve::latex_points_with_order(
		std::ostream &ost)
{
	vector<int> Ord;
	int *p;
	int i;
	l1_interfaces::latex_interface L;

	order_of_all_points(Ord);
	p = NEW_int(Ord.size());
	for (i = 0; i < (int) Ord.size(); i++) {
		p[i] = Ord[i];
	}


	ost << "\\begin{array}{|r|r|r|}" << endl;

	ost << "\\hline" << endl;
	for (i = 0; i < nb; i++) {
		ost <<  i << " & ";
		if (i) {
			ost << "(" << T[3 * i + 0] << "," << T[3 * i + 1] << ")";
		}
		else {
			ost << "{\\cal O}";
		}
		ost << " & " << p[i];
		ost << "\\\\";
		ost << endl;
		}
	ost << "\\end{array}" << endl;
}


void elliptic_curve::latex_order_of_all_points(
		std::ostream &ost)
{
	vector<int> Ord;
	int *p;
	int i;
	l1_interfaces::latex_interface L;

	order_of_all_points(Ord);
	p = NEW_int(Ord.size());
	for (i = 0; i < (int) Ord.size(); i++) {
		p[i] = Ord[i];
	}
	L.print_integer_matrix_with_standard_labels(ost,
			p, Ord.size(), 1, true /* f_tex */);
	FREE_int(p);
}

void elliptic_curve::order_of_all_points(
		vector<int> &Ord)
{
	int i;
	int ord;

	for (i = 0; i < nb; i++) {
		ord = order_of_point(i);
		Ord.push_back(ord);
	}
}


int elliptic_curve::order_of_point(int i)
{
	int j;
	int ord;

	j = i;
	ord = 1;
	while (j != 0) {
		j = A[i * nb + j];
		ord++;
	}
	return ord;
}

void elliptic_curve::print_all_powers(int i)
{
	int j;
	int ord;

	j = i;
	ord = 1;
	while (j != 0) {
		cout << ord << " & " << j << " & (" << T[3 * j + 0]
			<< ", " << T[3 * j + 1] << ", " << T[3 * j + 2]
			<< ")\\\\" << endl;
		j = A[i * nb + j];
		ord++;
	}
}

}}}



