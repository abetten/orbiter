/*
 * clebsch_map.cpp
 *
 *  Created on: Jul 2, 2019
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


clebsch_map::clebsch_map()
{
	SO = NULL;
	Surf = NULL;
	F = NULL;

	hds = 0;
	ds = 0;
	ds_row = 0;

	line1 = 0;
	line2 = 0;
	transversal = 0;

	tritangent_plane_idx = 0;


	line_idx[0] = -1;
	line_idx[1] = -1;
	plane_rk_global = 0;

	Clebsch_map = NULL;
	Clebsch_coeff = NULL;
}

clebsch_map::~clebsch_map()
{
	freeself();
}


void clebsch_map::freeself()
{
	int f_v = FALSE;

	if (f_v) {
		cout << "clebsch_map::freeself" << endl;
	}
}

void clebsch_map::init_half_double_six(surface_object *SO,
		int hds, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "clebsch_map::init_half_double_six" << endl;
	}


	clebsch_map::SO = SO;
	Surf = SO->Surf;
	F = Surf->F;
	clebsch_map::hds = hds;
	ds = hds / 2;
	ds_row = hds % 2;
	if (f_v) {
		cout << "clebsch_map::init_half_double_six hds = " << hds
				<< " double six = " << ds << " row = " << ds_row << endl;
	}

	if (f_v) {
		cout << "clebsch_map::init_half_double_six "
				"before Surf->prepare_clebsch_map" << endl;
	}
	Surf->Schlaefli->prepare_clebsch_map(ds, ds_row, line1, line2,
			transversal, verbose_level);
	if (f_v) {
		cout << "clebsch_map::init_half_double_six "
				"after Surf->prepare_clebsch_map" << endl;
	}

	if (f_v) {
		cout << "clebsch_map::init_half_double_six "
			"line1=" << line1
			<< " = " << Surf->Schlaefli->Labels->Line_label_tex[line1]
			<< " line2=" << line2
			<< " = " << Surf->Schlaefli->Labels->Line_label_tex[line2]
			<< " transversal=" << transversal
			<< " = " << Surf->Schlaefli->Labels->Line_label_tex[transversal]
			<< endl;
	}

	line_idx[0] = line1;
	line_idx[1] = line2;

	tritangent_plane_idx = Surf->Schlaefli->choose_tritangent_plane_for_Clebsch_map(line1, line2,
			transversal, verbose_level);

	plane_rk_global = SO->SOP->Tritangent_plane_rk[tritangent_plane_idx];


	int u, a, h;
	int v[4];
	int coefficients[3];


	Surf->P->Grass_planes->unrank_lint_here(
			Plane, plane_rk_global, 0);
	F->Linear_algebra->Gauss_simple(Plane, 3, 4, base_cols,
			0 /* verbose_level */);

	if (f_v) {
		Int_matrix_print(Plane, 3, 4);
		cout << "surface_with_action::arc_lifting_and_classify "
				"base_cols: ";
		Int_vec_print(cout, base_cols, 3);
		cout << endl;
	}


	if (f_v) {
		cout << "surface_with_action::arc_lifting_and_classify "
				"Lines with points on them:" << endl;
		SO->SOP->print_lines_with_points_on_them(cout);
		cout << "The half double six is no " << hds
				<< "$ = " << Surf->Schlaefli->Half_double_six_label_tex[hds]
				<< "$ : $";
		Lint_vec_print(cout, Surf->Schlaefli->Half_double_sixes + hds * 6, 6);
		cout << " = \\{" << endl;
		for (h = 0; h < 6; h++) {
			cout << Surf->Schlaefli->Labels->Line_label_tex[
					Surf->Schlaefli->Half_double_sixes[hds * 6 + h]];
			if (h < 6 - 1) {
				cout << ", ";
			}
		}
		cout << "\\}$\\\\" << endl;
	}

	for (u = 0; u < 6; u++) {

		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify u="
					<< u << " / 6" << endl;
		}
		a = SO->Lines[Surf->Schlaefli->Half_double_sixes[hds * 6 + u]];
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify "
					"intersecting line " << a << " and plane "
					<< plane_rk_global << endl;
		}
		intersection_points[u] =
				Surf->P->point_of_intersection_of_a_line_and_a_plane_in_three_space(
						a, plane_rk_global,
						0 /* verbose_level */);
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify "
					"intersection point " << intersection_points[u] << endl;
		}
		Surf->P->unrank_point(v, intersection_points[u]);
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify "
					"which is ";
			Int_vec_print(cout, v, 4);
			cout << endl;
		}
		F->Linear_algebra->reduce_mod_subspace_and_get_coefficient_vector(
			3, 4, Plane, base_cols,
			v, coefficients,
			0 /* verbose_level */);
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify "
					"local coefficients ";
			Int_vec_print(cout, coefficients, 3);
			cout << endl;
		}
		intersection_points_local[u] = Surf->P2->rank_point(coefficients);
	}




	if (f_v) {
		cout << "clebsch_map::init_half_double_six done" << endl;
	}
}

void clebsch_map::compute_Clebsch_map_down(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "clebsch_map::compute_Clebsch_map_down" << endl;
	}

	if (SO == NULL) {
		cout << "clebsch_map::compute_Clebsch_map_down SO == NULL" << endl;
		exit(1);
	}


	Clebsch_map = NEW_lint(SO->nb_pts);
	Clebsch_coeff = NEW_int(SO->nb_pts * 4);

	if (f_v) {
		cout << "clebsch_map::compute_Clebsch_map_down before compute_Clebsch_map_down_worker" << endl;
	}

	if (!compute_Clebsch_map_down_worker(
			Clebsch_map,
			Clebsch_coeff,
			verbose_level)) {
		cout << "clebsch_map::compute_Clebsch_map_down The plane contains one of the lines, "
				"this should not happen" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "clebsch_map::compute_Clebsch_map_down after compute_Clebsch_map_down_worker" << endl;
	}

	if (f_v) {
		cout << "clebsch_map::compute_Clebsch_map_down done" << endl;
		}
}


int clebsch_map::compute_Clebsch_map_down_worker(
	long int *Image_rk, int *Image_coeff,
	int verbose_level)
// assuming:
// In:
// SO->nb_pts
// SO->Pts[SO->nb_pts]
// SO->Lines[27]
// Out:
// Image_rk[nb_pts]  (image point in the plane in local coordinates)
//   Note Image_rk[i] is -1 if Pts[i] does not have an image.
// Image_coeff[nb_pts * 4] (image point in the plane in PG(3,q) coordinates)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int Plane[4 * 4];
	int Line_a[2 * 4];
	int Line_b[2 * 4];
	int Dual_planes[4 * 4];
		// dual coefficients of three planes:
		// the first plane is line_a together with the surface point
		// the second plane is line_b together with the surface point
		// the third plane is the plane onto which we map.
		// the fourth row is for the image point.
	int M[4 * 4];
	int v[4];
	long int i, h, pt, r;
	int coefficients[3];
	int base_cols[4];

	if (f_v) {
		cout << "clebsch_map::compute_Clebsch_map_down_worker" << endl;
	}
	Surf->P->Grass_planes->unrank_lint_here(Plane, plane_rk_global,
			0 /* verbose_level */);
	r = F->Linear_algebra->Gauss_simple(Plane, 3, 4, base_cols,
			0 /* verbose_level */);
	if (f_v) {
		cout << "Plane rank " << plane_rk_global << " :" << endl;
		Int_matrix_print(Plane, 3, 4);
	}

	F->Linear_algebra->RREF_and_kernel(4, 3, Plane, 0 /* verbose_level */);

	if (f_v) {
		cout << "Plane (3 basis vectors and dual coordinates):" << endl;
		Int_matrix_print(Plane, 4, 4);
		cout << "base_cols: ";
		Int_vec_print(cout, base_cols, r);
		cout << endl;
	}

	// make sure the two lines are not contained in
	// the plane onto which we map:

	// test line_a:
	Surf->P->Grass_lines->unrank_lint_here(Line_a,
			SO->Lines[line_idx[0]], 0 /* verbose_level */);
	if (f_v) {
		cout << "Line a = " << Surf->Schlaefli->Labels->Line_label_tex[line_idx[0]]
			<< " = " << SO->Lines[line_idx[0]] << ":" << endl;
		Int_matrix_print(Line_a, 2, 4);
	}
	for (i = 0; i < 2; i++) {
		if (F->Linear_algebra->dot_product(4, Line_a + i * 4, Plane + 3 * 4)) {
			break;
		}
	}
	if (i == 2) {
		cout << "clebsch_map::compute_Clebsch_map_down_worker Line a lies "
				"inside the hyperplane" << endl;
		return FALSE;
	}

	// test line_b:
	Surf->P->Grass_lines->unrank_lint_here(Line_b,
			SO->Lines[line_idx[1]], 0 /* verbose_level */);
	if (f_v) {
		cout << "Line b = " << Surf->Schlaefli->Labels->Line_label_tex[line_idx[1]]
			<< " = " << SO->Lines[line_idx[1]] << ":" << endl;
		Int_matrix_print(Line_b, 2, 4);
	}
	for (i = 0; i < 2; i++) {
		if (F->Linear_algebra->dot_product(4, Line_b + i * 4, Plane + 3 * 4)) {
			break;
		}
	}
	if (i == 2) {
		cout << "clebsch_map::compute_Clebsch_map_down_worker Line b lies "
				"inside the hyperplane" << endl;
		return FALSE;
	}

	// and now, map all surface points:
	for (h = 0; h < SO->nb_pts; h++) {
		pt = SO->Pts[h];

		Surf->unrank_point(v, pt);

		Int_vec_zero(Image_coeff + h * 4, 4);
		if (f_v) {
			cout << "clebsch_map::compute_Clebsch_map_down_worker "
					"pt " << h << " / " << SO->nb_pts << " is " << pt << " = ";
			Int_vec_print(cout, v, 4);
			cout << ":" << endl;
		}

		// make sure the points do not lie on either line_a or line_b
		// because the map is undefined there:
		Int_vec_copy(Line_a, M, 2 * 4);
		Int_vec_copy(v, M + 2 * 4, 4);
		if (F->Linear_algebra->Gauss_easy(M, 3, 4) == 2) {
			if (f_vv) {
				cout << "The point is on line_a" << endl;
			}
			Image_rk[h] = -1;
			continue;
		}
		Int_vec_copy(Line_b, M, 2 * 4);
		Int_vec_copy(v, M + 2 * 4, 4);
		if (F->Linear_algebra->Gauss_easy(M, 3, 4) == 2) {
			if (f_vv) {
				cout << "The point is on line_b" << endl;
			}
			Image_rk[h] = -1;
			continue;
		}

		// The point is good:

		// Compute the first plane in dual coordinates:
		Int_vec_copy(Line_a, M, 2 * 4);
		Int_vec_copy(v, M + 2 * 4, 4);
		F->Linear_algebra->RREF_and_kernel(4, 3, M, 0 /* verbose_level */);
		Int_vec_copy(M + 3 * 4, Dual_planes, 4);
		if (f_vv) {
			cout << "clebsch_map::compute_Clebsch_map_down_worker First plane in dual coordinates: ";
			Int_vec_print(cout, M + 3 * 4, 4);
			cout << endl;
		}

		// Compute the second plane in dual coordinates:
		Int_vec_copy(Line_b, M, 2 * 4);
		Int_vec_copy(v, M + 2 * 4, 4);
		F->Linear_algebra->RREF_and_kernel(4, 3, M, 0 /* verbose_level */);
		Int_vec_copy(M + 3 * 4, Dual_planes + 4, 4);
		if (f_vv) {
			cout << "clebsch_map::compute_Clebsch_map_down_worker Second plane in dual coordinates: ";
			Int_vec_print(cout, M + 3 * 4, 4);
			cout << endl;
		}


		// The third plane is the image
		// plane, given by dual coordinates:
		Int_vec_copy(Plane + 3 * 4, Dual_planes + 8, 4);
		if (f_vv) {
			cout << "clebsch_map::compute_Clebsch_map_down_worker Dual coordinates for all three planes: " << endl;
			Int_matrix_print(Dual_planes, 3, 4);
			cout << endl;
		}

		r = F->Linear_algebra->RREF_and_kernel(4, 3,
				Dual_planes, 0 /* verbose_level */);
		if (f_vv) {
			cout << "clebsch_map::compute_Clebsch_map_down_worker Dual coordinates and perp: " << endl;
			Int_matrix_print(Dual_planes, 4, 4);
			cout << endl;
			cout << "clebsch_map::compute_Clebsch_map_down_worker matrix of dual coordinates has rank " << r << endl;
		}


		if (r < 3) {
			if (f_v) {
				cout << "clebsch_map::compute_Clebsch_map_down_worker The line is contained in the plane" << endl;
			}
			Image_rk[h] = -1;
			continue;
		}
		F->PG_element_normalize(Dual_planes + 12, 1, 4);
		if (f_vv) {
			cout << "clebsch_map::compute_Clebsch_map_down_worker intersection point normalized: ";
			Int_vec_print(cout, Dual_planes + 12, 4);
			cout << endl;
		}
		Int_vec_copy(Dual_planes + 12, Image_coeff + h * 4, 4);

		// compute local coordinates of the image point:
		F->Linear_algebra->reduce_mod_subspace_and_get_coefficient_vector(
			3, 4, Plane, base_cols,
			Dual_planes + 12, coefficients,
			0 /* verbose_level */);
		Image_rk[h] = Surf->P2->rank_point(coefficients);
		if (f_vv) {
			cout << "clebsch_map::compute_Clebsch_map_down_worker pt " << h << " / " << SO->nb_pts
				<< " is " << pt << " : image = ";
			Int_vec_print(cout, Image_coeff + h * 4, 4);
			cout << " image = " << Image_rk[h] << endl;
		}
	}

	if (f_v) {
		cout << "clebsch_map::compute_Clebsch_map_down_worker done" << endl;
	}
	return TRUE;
}

void clebsch_map::clebsch_map_print_fibers()
{
	int i, j, u, pt;

	cout << "clebsch_map::clebsch_map_print_fibers" << endl;
	{
		data_structures::tally_lint C2;

		C2.init(Clebsch_map, SO->nb_pts, TRUE, 0);
		cout << "clebsch_map::clebsch_map_print_fibers The fibers "
				"have the following sizes: ";
		C2.print_naked(TRUE);
		cout << endl;

		int t2, f2, l2, sz;
		int t1, f1, l1;

		for (t2 = 0; t2 < C2.second_nb_types; t2++) {
			f2 = C2.second_type_first[t2];
			l2 = C2.second_type_len[t2];
			sz = C2.second_data_sorted[f2];
			cout << "clebsch_map::clebsch_map_print_fibers fibers "
					"of size " << sz << ":" << endl;
			if (sz == 1) {
				continue;
			}
			for (i = 0; i < l2; i++) {
				t1 = C2.second_sorting_perm_inv[f2 + i];
				f1 = C2.type_first[t1];
				l1 = C2.type_len[t1];
				pt = C2.data_sorted[f1];
				cout << "arc point " << pt << " belongs to the " << l1
					<< " surface points in the list of Pts "
					"(local numbering): ";
				for (j = 0; j < l1; j++) {
					u = C2.sorting_perm_inv[f1 + j];
					cout << u;
					//cout << Pts[u];
					if (j < l1 - 1) {
						cout << ", ";
					}
				}
				cout << endl;
			}
		}
		cout << endl;
	}
}


void clebsch_map::clebsch_map_find_arc_and_lines(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, pt, nb_blow_up_lines;

	if (f_v) {
		cout << "clebsch_map::clebsch_map_find_arc_and_lines" << endl;
	}


	if (f_v) {
		cout << "lines_on_point:" << endl;
		SO->SOP->lines_on_point->print_table();
	}

	{
		data_structures::tally_lint C2;

		C2.init(Clebsch_map, SO->nb_pts, TRUE, 0);
		if (f_v) {
			cout << "clebsch_map::clebsch_map_find_arc_and_lines "
					"The fibers have the following sizes: ";
			C2.print_naked(TRUE);
			cout << endl;
		}

		int t2, f2, l2, sz;
		int t1, f1, l1;
		int fiber[2];
		int common_elt;
		int u; //, v, w;




		nb_blow_up_lines = 0;
		for (t2 = 0; t2 < C2.second_nb_types; t2++) {
			f2 = C2.second_type_first[t2];
			l2 = C2.second_type_len[t2];
			sz = C2.second_data_sorted[f2];
			if (f_v) {
				cout << "clebsch_map::clebsch_map_find_arc_and_lines "
						"fibers of size " << sz << ":" << endl;
			}
			if (sz == 1) {
				continue;
			}

			if (f_v) {
				for (i = 0; i < l2; i++) {
					t1 = C2.second_sorting_perm_inv[f2 + i];
					f1 = C2.type_first[t1];
					l1 = C2.type_len[t1];
					pt = C2.data_sorted[f1];
					cout << "arc point " << pt << " belongs to the " << l1
							<< " surface points in the list of Pts "
							"(point indices): ";
					for (j = 0; j < l1; j++) {
						u = C2.sorting_perm_inv[f1 + j];
						cout << u;
						//cout << Pts[u];
						if (j < l1 - 1) {
							cout << ", ";
						}
					}
					cout << endl;
				}
			}


			for (i = 0; i < l2; i++) {
				t1 = C2.second_sorting_perm_inv[f2 + i];
				f1 = C2.type_first[t1];
				l1 = C2.type_len[t1];
				pt = C2.data_sorted[f1];

				if (pt == -1) {
					continue;
				}
				fiber[0] = C2.sorting_perm_inv[f1 + 0];
				fiber[1] = C2.sorting_perm_inv[f1 + 1];

				if (f_v) {
					cout << "lines through point fiber[0]="
							<< fiber[0] << " : ";
					SO->Surf->Schlaefli->print_set_of_lines_tex(cout,
							SO->SOP->lines_on_point->Sets[fiber[0]],
							SO->SOP->lines_on_point->Set_size[fiber[0]]);
					cout << endl;
					cout << "lines through point fiber[1]="
							<< fiber[1] << " : ";
					SO->Surf->Schlaefli->print_set_of_lines_tex(cout,
							SO->SOP->lines_on_point->Sets[fiber[1]],
							SO->SOP->lines_on_point->Set_size[fiber[1]]);
					cout << endl;
				}

				// find the unique line which passes through
				// the surface points fiber[0] and fiber[1]:
				if (!SO->SOP->lines_on_point->find_common_element_in_two_sets(
						fiber[0], fiber[1], common_elt)) {
					cout << "The fiber does not seem to come "
							"from a line, i=" << i << endl;


#if 1
					cout << "The fiber does not seem to come "
							"from a line" << endl;
					cout << "i=" << i << " / " << l2 << endl;
					cout << "pt=" << pt << endl;
					cout << "fiber[0]=" << fiber[0] << endl;
					cout << "fiber[1]=" << fiber[1] << endl;
					cout << "lines through point fiber[0]=" << fiber[0] << " : ";
					SO->Surf->Schlaefli->print_set_of_lines_tex(cout,
							SO->SOP->lines_on_point->Sets[fiber[0]],
							SO->SOP->lines_on_point->Set_size[fiber[0]]);
					cout << endl;
					cout << "lines through point fiber[1]=" << fiber[1] << " : ";
					SO->Surf->Schlaefli->print_set_of_lines_tex(cout,
							SO->SOP->lines_on_point->Sets[fiber[1]],
							SO->SOP->lines_on_point->Set_size[fiber[1]]);
					cout << endl;
					//exit(1);
#endif
				}
				else {
					if (nb_blow_up_lines == 6) {
						cout << "too many long fibers" << endl;
						exit(1);
					}
					cout << "i=" << i << " fiber[0]=" << fiber[0]
						<< " fiber[1]=" << fiber[1]
						<< " common_elt=" << common_elt << endl;
					Arc[nb_blow_up_lines] = pt;
					Blown_up_lines[nb_blow_up_lines] = common_elt;
					nb_blow_up_lines++;
				}

#if 0
				w = 0;
				for (u = 0; u < l2; u++) {
					fiber[0] = C2.sorting_perm_inv[f1 + u];
					for (v = u + 1; v < l2; v++, w++) {
						fiber[1] = C2.sorting_perm_inv[f1 + v];
						Fiber_recognize[w] = -1;
						if (!lines_on_point->find_common_element_in_two_sets(
								fiber[0], fiber[1], common_elt)) {
#if 0
							cout << "The fiber does not seem to "
									"come from a line" << endl;
							cout << "i=" << i << " / " << l2 << endl;
							cout << "pt=" << pt << endl;
							cout << "fiber[0]=" << fiber[0] << endl;
							cout << "fiber[1]=" << fiber[1] << endl;
							cout << "lines_on_point:" << endl;
							lines_on_point->print_table();
							exit(1);
#endif
						}
						else {
							Fiber_recognize[w] = common_elt;
						}
					}
				}
				{
					tally C_fiber;

					C_fiber.init(Fiber_recognize, w, FALSE, 0);
					cout << "The fiber type is : ";
					C_fiber.print_naked(TRUE);
					cout << endl;
				}
				Blown_up_lines[i] = -1;
#endif

				} // next i
			}

		if (nb_blow_up_lines != 6) {
			cout << "nb_blow_up_lines != 6" << endl;
			cout << "nb_blow_up_lines = " << nb_blow_up_lines << endl;
			exit(1);
		}
	} // end of classify C2
	if (f_v) {
		cout << "clebsch_map::clebsch_map_find_arc_and_lines "
				"done" << endl;
	}
}

void clebsch_map::report(std::ostream &ost, int verbose_level)
{
	orbiter_kernel_system::latex_interface L;

	int h;

	ost << "The half double six is no " << hds << "$ = "
			<< Surf->Schlaefli->Half_double_six_label_tex[hds] << "$ : $";
	Lint_vec_print(ost, Surf->Schlaefli->Half_double_sixes + hds * 6, 6);
	ost << " = \\{" << endl;
	for (h = 0; h < 6; h++) {
		ost << Surf->Schlaefli->Labels->Line_label_tex[Surf->Schlaefli->Half_double_sixes[hds * 6 + h]];
		if (h < 6 - 1) {
			ost << ", ";
		}
	}
	ost << "\\}$\\\\" << endl;

	ost << "line1$=" << line1 << " = " << Surf->Schlaefli->Labels->Line_label_tex[line1]
			<< "$ line2$=" << line2 << " = " << Surf->Schlaefli->Labels->Line_label_tex[line2]
			<< "$ transversal$=" << transversal << " = "
			<< Surf->Schlaefli->Labels->Line_label_tex[transversal] << "$\\\\" << endl;

	ost << "transversal = " << transversal << " = $"
			<< Surf->Schlaefli->Labels->Line_label_tex[transversal] << "$\\\\" << endl;
	ost << "plane\\_rk = $\\pi_{" << tritangent_plane_idx << "} = \\pi_{"
			<< Surf->Schlaefli->Eckard_point_label_tex[tritangent_plane_idx] << "} = "
			<< plane_rk_global << "$\\\\" << endl;


	ost << "The plane is:" << endl;
	Surf->P->Grass_planes->print_set_tex(ost, &plane_rk_global, 1, 0 /* verbose_level */);

	ost << "Clebsch map for lines $" << line1
			<< " = " << Surf->Schlaefli->Labels->Line_label_tex[line1] << ", "
			<< line2 << " = "
			<< Surf->Schlaefli->Labels->Line_label_tex[line2]
			<< "$\\\\" << endl;

	//SOA->SO->clebsch_map_latex(fp, Clebsch_map, Clebsch_coeff);

	ost << "Clebsch map for lines $" << line1
		<< " = " << Surf->Schlaefli->Labels->Line_label_tex[line1] << ", "
		<< line2 << " = " << Surf->Schlaefli->Labels->Line_label_tex[line2]
		<< "$ yields arc = $";
	L.lint_set_print_tex(ost, Arc, 6);
	ost << "$ : blown up lines = ";
	Lint_vec_print(ost, Blown_up_lines, 6);
	ost << "\\\\" << endl;

	SO->SOP->clebsch_map_latex(ost, Clebsch_map, Clebsch_coeff);

}



}}}

