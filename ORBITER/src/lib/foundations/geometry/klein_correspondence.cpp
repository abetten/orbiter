// klein_correspondence.C
//
// Anton Betten
// 
// January 1, 2016

#include "foundations.h"

klein_correspondence::klein_correspondence()
{
	null();
}

klein_correspondence::~klein_correspondence()
{
	freeself();
}

void klein_correspondence::null()
{
	P3 = NULL;
	P5 = NULL;
	Gr63 = NULL;
	Gr62 = NULL;
	Form = NULL;
	Line_to_point_on_quadric = NULL;
	Point_on_quadric_to_line = NULL;
	Point_on_quadric_embedded_in_P5 = NULL;
	coordinates_of_quadric_points = NULL;
	Pt_rk = NULL;
}

void klein_correspondence::freeself()
{
	if (P3) {
		delete P3;
		}
	if (P5) {
		delete P5;
		}
	if (Gr63) {
		delete Gr63;
		}
	if (Gr62) {
		delete Gr62;
		}
	if (Form) {
		FREE_int(Form);
		}
	if (Line_to_point_on_quadric) {
		FREE_int(Line_to_point_on_quadric);
		}
	if (Point_on_quadric_to_line) {
		FREE_int(Point_on_quadric_to_line);
		}
	if (Point_on_quadric_embedded_in_P5) {
		FREE_int(Point_on_quadric_embedded_in_P5);
		}
	if (coordinates_of_quadric_points) {
		FREE_int(coordinates_of_quadric_points);
		}
	if (Pt_rk) {
		FREE_int(Pt_rk);
		}
}

void klein_correspondence::init(finite_field *F,
		orthogonal *O, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int d = 6;
	int i, u, v;

	if (f_v) {
		cout << "klein_correspondence::init" << endl;
		}
	
	klein_correspondence::F = F;
	klein_correspondence::O = O;
	q = F->q;


	nb_Pts = O->nb_points;
	
	P3 = NEW_OBJECT(projective_space);
	
	if (f_v) {
		cout << "klein_correspondence::init before P3->init" << endl;
		}
	P3->init(3, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level - 2);

	P5 = NEW_OBJECT(projective_space);
	
	if (f_v) {
		cout << "klein_correspondence::init before P5->init" << endl;
		}
	P5->init(5, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level - 2);
	if (f_v) {
		cout << "klein_correspondence::after before P5->init" << endl;
		}

	
	Gr63 = NEW_OBJECT(grassmann);
	Gr62 = NEW_OBJECT(grassmann);

	Gr63->init(6, 3, F, 0 /* verbose_level */);
	Gr62->init(6, 2, F, 0 /* verbose_level */);


	Form = NEW_int(d * d);
	int_vec_zero(Form, d * d);
	// the matrix with blocks
	// [0 1]
	// [1 0]
	// along the diagonal:
	for (i = 0; i < 3; i++) {
		u = 2 * i + 0;
		v = 2 * i + 1;
		Form[u * d + v] = 1;
		Form[v * d + u] = 1;
		}
	if (f_v) {
		cout << "klein_correspondence::init Form matrix:" << endl;
		int_matrix_print(Form, d, d);
		}
	if (f_v) {
		cout << "klein_correspondence::init before allocate "
				"Line_to_point_on_quadric P3->N_lines="
				<< P3->N_lines << endl;
		}
	Line_to_point_on_quadric = NEW_int(P3->N_lines);
	if (f_v) {
		cout << "klein_correspondence::init before allocate "
				"Point_on_quadric_to_line P3->N_lines="
				<< P3->N_lines << endl;
		}
	Point_on_quadric_to_line = NEW_int(P3->N_lines);
	if (f_v) {
		cout << "klein_correspondence::init before allocate "
				"Point_on_quadric_embedded_in_P5 P3->N_lines="
				<< P3->N_lines << endl;
		}
	Point_on_quadric_embedded_in_P5 = NEW_int(P3->N_lines);

	int basis_line[8]; // [2 * 4]
	int v6[6];
	int *x4, *y4, a, b, c, val, j;

	//basis_line = NEW_int(8);
	for (i = 0; i < P3->N_lines; i++) {
		Point_on_quadric_to_line[i] = -1;
		}
	for (i = 0; i < P3->N_lines; i++) {
		P3->unrank_line(basis_line, i);
		x4 = basis_line;
		y4 = basis_line + 4;
		v6[0] = F->Pluecker_12(x4, y4);
		v6[1] = F->Pluecker_34(x4, y4);
		v6[2] = F->Pluecker_13(x4, y4);
		v6[3] = F->Pluecker_42(x4, y4);
		v6[4] = F->Pluecker_14(x4, y4);
		v6[5] = F->Pluecker_23(x4, y4);
		a = F->mult(v6[0], v6[1]);
		b = F->mult(v6[2], v6[3]);
		c = F->mult(v6[4], v6[5]);
		val = F->add3(a, b, c);
		//cout << "a=" << a << " b=" << b << " c=" << c << endl;
		//cout << "val=" << val << endl;
		if (val) {
			cout << "klein_correspondence::init point does "
					"not lie on quadric" << endl;
			exit(1);
			}
		//j = P5->rank_point(v6);
		j = O->rank_point(v6, 1, 0 /* verbose_level */);
		if (FALSE) {
			cout << "klein_correspondence::init i=" << i
					<< " / " << P3->N_lines << " v6 : ";
			int_vec_print(cout, v6, 6);
			cout << " : j=" << j << endl;
			}
		Line_to_point_on_quadric[i] = j;
		Point_on_quadric_to_line[j] = i;
		}
	for (i = 0; i < P3->N_lines; i++) {
		if (Point_on_quadric_to_line[i] == -1) {
			cout << "Something is wrong with "
					"Point_on_quadric_to_line" << endl;
			exit(1);
			}
		}
	for (i = 0; i < P3->N_lines; i++) {
		O->unrank_point(v6, 1, i, 0);
		Point_on_quadric_embedded_in_P5[i] = P5->rank_point(v6);
		}

	if (f_v) {
		cout << "klein_correspondence::init before coordinates_"
				"of_quadric_points P3->N_lines * d="
				<< P3->N_lines * d << endl;
		}
	coordinates_of_quadric_points = NEW_int(P3->N_lines * d);


	if (f_v) {
		cout << "klein_correspondence::init before allocate "
				"Pt_rk P3->N_lines=" << P3->N_lines << endl;
		}
	Pt_rk = NEW_int(P3->N_lines);

	for (i = 0; i < P3->N_lines; i++) {
		O->unrank_point(
			coordinates_of_quadric_points + i * d, 1, i, 0);
		int_vec_copy(
			coordinates_of_quadric_points + i * d, v6, 6);
		F->PG_element_rank_modified(v6, 1, d, a);
		Pt_rk[i] = a;
		}

	if (f_vv) {
		cout << "Points on the Klein quadric:" << endl;
		if (nb_Pts < 50) {
			for (i = 0; i < nb_Pts; i++) {
				P3->unrank_line(basis_line, i);

				cout << i << " & " << endl;
				cout << "\\left[" << endl;
				cout << "\\begin{array}{cccc}" << endl;
				for (u = 0; u < 2; u++) {
					for (v = 0; v < 4; v++) {
						cout << basis_line[u * 4 + v] << " ";
						if (v < 4 - 1) {
							cout << "&";
							}
						}
					cout << "\\\\" << endl;
					}
				cout << "\\end{array}" << endl;
				cout << "\\right] & " << endl;
				int_vec_print(cout,
						coordinates_of_quadric_points + i * d, d);
				//cout << " : " << Pt_rk[i] << endl;
				cout << "\\\\" << endl;
				}
			}
		else {
			cout << "too many points to print" << endl;
			}
		}

	nb_pts_PG = nb_PG_elements(d - 1, q);
	if (f_v) {
		cout << "klein_correspondence::init "
				"nb_pts_PG = " << nb_pts_PG << endl;
		}

#if 0
	// this array is only used by REGULAR_PACKING
	if (f_v) {
		cout << "klein_correspondence::init before "
				"allocate Pt_idx nb_pts_P=" << nb_pts_PG << endl;
		}
	Pt_idx = NEW_int(nb_pts_PG);
	for (i = 0; i < nb_pts_PG; i++) {
		Pt_idx[i] = -1;
		}
	for (i = 0; i < nb_Pts; i++) {
		a = Pt_rk[i];
		Pt_idx[a] = i;
		}
#endif


	if (f_v) {
		cout << "klein_correspondence::init done" << endl;
		}
}

void klein_correspondence::plane_intersections(
	int *lines_in_PG3, int nb_lines,
	longinteger_object *&R,
	int **&Pts_on_plane, 
	int *&nb_pts_on_plane, 
	int &nb_planes, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *pts;
	int i;

	if (f_v) {
		cout << "klein_correspondence::plane_intersections" << endl;
		}
	pts = NEW_int(nb_lines);
	
	P3->klein_correspondence(P5, 
		lines_in_PG3, nb_lines, pts, 0/*verbose_level*/);

	P5->plane_intersection_type_fast(Gr63, pts, nb_lines, 
		R, Pts_on_plane, nb_pts_on_plane, nb_planes, 
		verbose_level - 3);


	if (f_vv) {
		cout << "klein_correspondence::plane_intersections: "
				"We found " << nb_planes << " planes." << endl;
#if 1
		for (i = 0; i < nb_planes; i++) {
			cout << setw(3) << i << " : " << R[i] 
				<< " : " << setw(5) << nb_pts_on_plane[i] << " : ";
			int_vec_print(cout, Pts_on_plane[i], nb_pts_on_plane[i]);
			cout << endl; 
			}
#endif
		}
	
	FREE_int(pts);
	if (f_v) {
		cout << "klein_correspondence::plane_intersections done" << endl;
		}
}

