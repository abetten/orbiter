/*
 * interface_povray.cpp
 *
 *  Created on: Apr 6, 2020
 *      Author: betten
 */







#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace interfaces {


interface_povray::interface_povray()
{
	argc = 0;
	argv = NULL;

	f_output_mask = FALSE;
	output_mask = NULL;
	f_nb_frames_default = FALSE;
	nb_frames_default = 0;
	f_round = FALSE;
	round = 0;
	f_rounds = FALSE;
	rounds_as_string = NULL;
	Opt = NULL;

	S = NULL;
	A = NULL;
}


void interface_povray::print_help(int argc, const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-povray") == 0) {
		cout << "-povray" << endl;
	}
}

int interface_povray::recognize_keyword(int argc, const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-povray") == 0) {
		return true;
	}
	return false;
}

void interface_povray::read_arguments(int argc, const char **argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_povray::read_arguments" << endl;
	//return 0;

	interface_povray::argc = argc;
	interface_povray::argv = argv;

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-povray") == 0) {
			cout << "-povray " << endl;
			i++;

			S = NEW_OBJECT(scene);

			S->init(verbose_level);

			for (; i < argc; i++) {
				if (strcmp(argv[i], "-video_options") == 0) {
					Opt = NEW_OBJECT(video_draw_options);
					i += Opt->read_arguments(argc - (i - 1),
						argv + i, verbose_level);

					cout << "-video_options" << endl;
					cout << "done with -video_options " << endl;
					cout << "i = " << i << endl;
					cout << "argc = " << argc << endl;
					if (i < argc) {
						cout << "next argument is " << argv[i] << endl;
					}
				}
				else if (strcmp(argv[i], "-round") == 0) {
					f_round = TRUE;
					round = atoi(argv[++i]);
					cout << "-round " << round << endl;
				}

				else if (strcmp(argv[i], "-rounds") == 0) {
					f_rounds = TRUE;
					rounds_as_string = argv[++i];
					cout << "-rounds " << rounds_as_string << endl;
				}
				else if (strcmp(argv[i], "-nb_frames_default") == 0) {
					f_nb_frames_default = TRUE;
					nb_frames_default = atoi(argv[++i]);
					cout << "-nb_frames_default " << nb_frames_default << endl;
				}
				else if (strcmp(argv[i], "-output_mask") == 0) {
					f_output_mask = TRUE;
					output_mask = argv[++i];
					cout << "-output_mask " << output_mask << endl;
				}
				else if (strcmp(argv[i], "-scene_objects") == 0) {
					cout << "-scene_objects " << endl;
					i++;
					i = read_scene_objects(argc, argv, i, verbose_level);
					cout << "done with -scene_objects " << endl;
					cout << "i = " << i << endl;
					cout << "argc = " << argc << endl;
					if (i < argc) {
						cout << "next argument is " << argv[i] << endl;
					}
				}
				else if (strcmp(argv[i], "-povray_end") == 0) {
					cout << "-povray_end " << endl;
					i++;
					break;
				}
				else {
					cout << "unrecognized option " << argv[i] << endl;
				}
			}
			if (Opt == NULL) {
				cout << "Please use option -video_options .." << endl;
				exit(1);
				}
			if (!f_output_mask) {
				cout << "Please use option -output_mask <output_mask>" << endl;
				exit(1);
				}
			if (!f_nb_frames_default) {
				cout << "Please use option -nb_frames_default <nb_frames>" << endl;
				exit(1);
				}
			if (!f_round && !f_rounds ) {
				cout << "Please use option -round <round> or "
						"-rounds <first_round> <nb_rounds>" << endl;
				exit(1);
				}
		}
	}
	cout << "interface_povray::read_arguments done" << endl;
}

int interface_povray::read_scene_objects(int argc, const char **argv, int i0, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "interface_povray::read_scene_objects" << endl;
	}
	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-cubic_lex") == 0) {
			cout << "-cubic_lex" << endl;
			const char *coeff_text;
			double *coeff;
			int coeff_sz;
			numerics Numerics;

			coeff_text = argv[++i];
			Numerics.vec_scan(coeff_text, coeff, coeff_sz);
			if (coeff_sz != 20) {
				cout << "For -cubic_lex, number of coefficients must be 20; is " << coeff_sz << endl;
				exit(1);
			}
			S->cubic(coeff);
			delete [] coeff;
		}
		else if (strcmp(argv[i], "-cubic_orbiter") == 0) {
			cout << "-cubic_orbiter" << endl;
			const char *coeff_text;
			double *coeff;
			int coeff_sz;
			numerics Numerics;

			coeff_text = argv[++i];
			Numerics.vec_scan(coeff_text, coeff, coeff_sz);
			if (coeff_sz != 20) {
				cout << "For -cubic_orbiter, the number of coefficients must be 20; is " << coeff_sz << endl;
				exit(1);
			}
			S->cubic_in_orbiter_ordering(coeff);
			delete [] coeff;
		}
		else if (strcmp(argv[i], "-cubic_Goursat") == 0) {
			cout << "-cubic_Goursat" << endl;
			const char *coeff_text;
			double *coeff;
			int coeff_sz;
			numerics Numerics;

			coeff_text = argv[++i];
			Numerics.vec_scan(coeff_text, coeff, coeff_sz);
			if (coeff_sz != 3) {
				cout << "For -cubic_Goursat, number of coefficients must be 3; is " << coeff_sz << endl;
				exit(1);
			}
			S->cubic_Goursat_ABC(coeff[0], coeff[1], coeff[2]);
			delete [] coeff;
		}
		else if (strcmp(argv[i], "-quadric_lex_10") == 0) {
			cout << "-quadric_lex_10" << endl;
			const char *coeff_text;
			double *coeff;
			int coeff_sz;
			numerics Numerics;

			coeff_text = argv[++i];
			Numerics.vec_scan(coeff_text, coeff, coeff_sz);
			if (coeff_sz != 10) {
				cout << "For -quadric_lex_10, number of coefficients must be 10; is " << coeff_sz << endl;
				exit(1);
			}
			S->quadric(coeff);
			delete [] coeff;
		}
		else if (strcmp(argv[i], "-quartic_lex_35") == 0) {
			cout << "-quartic_lex_35" << endl;
			const char *coeff_text;
			double *coeff;
			int coeff_sz;
			numerics Numerics;

			coeff_text = argv[++i];
			Numerics.vec_scan(coeff_text, coeff, coeff_sz);
			if (coeff_sz != 35) {
				cout << "For -quartic_lex_35, number of coefficients must be 35; is " << coeff_sz << endl;
				exit(1);
			}
			S->quartic(coeff);
			delete [] coeff;
		}
		else if (strcmp(argv[i], "-octic_lex_165") == 0) {
			cout << "-octic_lex_165" << endl;
			const char *coeff_text;
			double *coeff;
			int coeff_sz;
			numerics Numerics;

			coeff_text = argv[++i];
			Numerics.vec_scan(coeff_text, coeff, coeff_sz);
			if (coeff_sz != 165) {
				cout << "For -octic_lex_165, number of coefficients must be 165; is " << coeff_sz << endl;
				exit(1);
			}
			S->octic(coeff);
			delete [] coeff;
		}
		else if (strcmp(argv[i], "-point") == 0) {
			cout << "-point" << endl;
			const char *coeff_text;
			double *coeff;
			int coeff_sz;
			numerics Numerics;
			int idx;

			coeff_text = argv[++i];
			Numerics.vec_scan(coeff_text, coeff, coeff_sz);
			if (coeff_sz != 3) {
				cout << "For -point, the number of coefficients must be 3; is " << coeff_sz << endl;
				exit(1);
			}
			idx = S->point(coeff[0], coeff[1], coeff[2]);
			cout << "created point " << idx << endl;
			delete [] coeff;
		}
		else if (strcmp(argv[i], "-point_list_from_csv_file") == 0) {
			cout << "-point_list_from_csv_file" << endl;
			const char *fname;
			double *M;
			int m, n, h;
			file_io Fio;

			fname = argv[++i];
			Fio.double_matrix_read_csv(fname, M,
					m, n, verbose_level);
			cout << "The file " << fname << " contains " << m << " point coordiunates, each with " << n << " coordinates" << endl;
			if (n == 2) {
				for (h = 0; h < m; h++) {
					S->point(M[h * 2 + 0], M[h * 2 + 1], 0);
				}
			}
			else if (n == 3) {
				for (h = 0; h < m; h++) {
					S->point(M[h * 3 + 0], M[h * 3 + 1], M[h * 3 + 2]);
				}
			}
			else if (n == 4) {
				for (h = 0; h < m; h++) {
					S->point(M[h * 4 + 0], M[h * 4 + 1], M[h * 4 + 2]);
				}
			}
			else {
				cout << "The file " << fname << " should have either 2 or three columns" << endl;
				exit(1);
			}
			delete [] M;
		}
		else if (strcmp(argv[i], "-line_through_two_points_recentered_from_csv_file") == 0) {
			cout << "-line_through_two_points_recentered_from_csv_file" << endl;
			const char *fname;
			double *M;
			int m, n, h;
			file_io Fio;

			fname = argv[++i];
			Fio.double_matrix_read_csv(fname, M,
					m, n, verbose_level);
			cout << "The file " << fname << " contains " << m << " point coordiunates, each with " << n << " coordinates" << endl;
			if (n != 6) {
				cout << "The file " << fname << " should have 6 columns" << endl;
				exit(1);
			}
			for (h = 0; h < m; h++) {
				S->line_after_recentering(M[h * 6 + 0], M[h * 6 + 1], M[h * 6 + 2], M[h * 6 + 3], M[h * 6 + 4], M[h * 6 + 5], 10);
			}
			delete [] M;
		}
		else if (strcmp(argv[i], "-line_through_two_points_from_csv_file") == 0) {
			cout << "-line_through_two_points_from_csv_file" << endl;
			const char *fname;
			double *M;
			int m, n, h;
			file_io Fio;

			fname = argv[++i];
			Fio.double_matrix_read_csv(fname, M,
					m, n, verbose_level);
			cout << "The file " << fname << " contains " << m << " point coordiunates, each with " << n << " coordinates" << endl;
			if (n != 6) {
				cout << "The file " << fname << " should have 6 columns" << endl;
				exit(1);
			}
			for (h = 0; h < m; h++) {
				S->line(M[h * 6 + 0], M[h * 6 + 1], M[h * 6 + 2], M[h * 6 + 3], M[h * 6 + 4], M[h * 6 + 5]);
			}
			delete [] M;
		}
		else if (strcmp(argv[i], "-point_as_intersection_of_two_lines") == 0) {
			cout << "-point_as_intersection_of_two_lines" << endl;
			const char *Idx_text;
			int *Idx;
			int Idx_sz;
			//numerics Numerics;

			Idx_text = argv[++i];
			int_vec_scan(Idx_text, Idx, Idx_sz);
			if (Idx_sz != 2) {
				cout << "For -point_as_intersection_of_two_lines, the number of indices must be 2; is " << Idx_sz << endl;
				exit(1);
			}
			S->point_as_intersection_of_two_lines(Idx[0], Idx[1]);
			FREE_int(Idx);
		}
		else if (strcmp(argv[i], "-edge") == 0) {
			cout << "-edge" << endl;
			const char *Idx_text;
			int *Idx;
			int Idx_sz;
			//numerics Numerics;

			Idx_text = argv[++i];
			int_vec_scan(Idx_text, Idx, Idx_sz);
			if (Idx_sz != 2) {
				cout << "For -edge, the number of indices must be 2; is " << Idx_sz << endl;
				exit(1);
			}
			S->edge(Idx[0], Idx[1]);
			FREE_int(Idx);
		}
		else if (strcmp(argv[i], "-label") == 0) {
			cout << "-label" << endl;
			int pt_idx;
			const char *text;
			//numerics Numerics;

			pt_idx = atoi(argv[++i]);
			text = argv[++i];
			S->label(pt_idx, text);
		}
		else if (strcmp(argv[i], "-triangular_face_given_by_three_lines") == 0) {
			cout << "-triangular_face_given_by_three_lines" << endl;
			const char *Idx_text;
			int *Idx;
			int Idx_sz;
			//numerics Numerics;

			Idx_text = argv[++i];
			int_vec_scan(Idx_text, Idx, Idx_sz);
			if (Idx_sz != 3) {
				cout << "For -triangular_face_given_by_three_lines, the number of indices must be 3; is " << Idx_sz << endl;
				exit(1);
			}
			S->triangle(Idx[0], Idx[1], Idx[2], 0 /* verbose_level */);
			FREE_int(Idx);
		}
		else if (strcmp(argv[i], "-face") == 0) {
			cout << "-face" << endl;
			const char *Idx_text;
			int *Idx;
			int Idx_sz;
			//numerics Numerics;

			Idx_text = argv[++i];
			int_vec_scan(Idx_text, Idx, Idx_sz);
			S->face(Idx, Idx_sz);
			FREE_int(Idx);
		}
		else if (strcmp(argv[i], "-quadric_through_three_skew_lines") == 0) {
			cout << "-quadric_through_three_skew_lines" << endl;
			const char *Idx_text;
			int *Idx;
			int Idx_sz;
			//numerics Numerics;

			Idx_text = argv[++i];
			int_vec_scan(Idx_text, Idx, Idx_sz);
			if (Idx_sz != 3) {
				cout << "For -quadric_through_three_skew_lines, the number of indices must be 3; is " << Idx_sz << endl;
				exit(1);
			}
			S->quadric_through_three_lines(Idx[0], Idx[1], Idx[2], 0 /* verbose_level */);
			FREE_int(Idx);
		}
		else if (strcmp(argv[i], "-plane_defined_by_three_points") == 0) {
			cout << "-plane_defined_by_three_points" << endl;
			const char *Idx_text;
			int *Idx;
			int Idx_sz;
			//numerics Numerics;

			Idx_text = argv[++i];
			int_vec_scan(Idx_text, Idx, Idx_sz);
			if (Idx_sz != 3) {
				cout << "For -plane_defined_by_three_points, the number of indices must be 3; is " << Idx_sz << endl;
				exit(1);
			}
			S->plane_through_three_points(Idx[0], Idx[1], Idx[2]);
			FREE_int(Idx);
		}
		else if (strcmp(argv[i], "-line_through_two_points_recentered") == 0) {
			cout << "-line_through_two_points_recentered" << endl;
			const char *coeff_text;
			double *coeff;
			int coeff_sz;
			numerics Numerics;

			coeff_text = argv[++i];
			Numerics.vec_scan(coeff_text, coeff, coeff_sz);
			if (coeff_sz != 6) {
				cout << "For -line_through_two_points_recentered, the number of coefficients must be 6; is " << coeff_sz << endl;
				exit(1);
			}
			//S->line(coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5]);
			S->line_after_recentering(coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5], 10);
			delete [] coeff;
		}
		else if (strcmp(argv[i], "-line_through_two_points") == 0) {
			cout << "-line_through_two_points" << endl;
			const char *coeff_text;
			double *coeff;
			int coeff_sz;
			numerics Numerics;

			coeff_text = argv[++i];
			Numerics.vec_scan(coeff_text, coeff, coeff_sz);
			if (coeff_sz != 6) {
				cout << "For -line_through_two_points, the number of coefficients must be 6; is " << coeff_sz << endl;
				exit(1);
			}
			S->line(coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5]);
			//S->line_after_recentering(coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5], 10);
			delete [] coeff;
		}
		else if (strcmp(argv[i], "-line_through_two_existing_points") == 0) {
			cout << "-line_through_two_existing_points" << endl;
			const char *Idx_text;
			int *Idx;
			int Idx_sz;
			//numerics Numerics;

			Idx_text = argv[++i];
			int_vec_scan(Idx_text, Idx, Idx_sz);
			if (Idx_sz != 2) {
				cout << "For -line_through_two_existing_points, the number of indices must be 2; is " << Idx_sz << endl;
				exit(1);
			}
			S->line_through_two_points(Idx[0], Idx[1], 0 /* verbose_level */);
			FREE_int(Idx);
		}
		else if (strcmp(argv[i], "-line_through_point_with_direction") == 0) {
			cout << "-line_through_point_with_direction" << endl;
			const char *coeff_text;
			double *coeff;
			int coeff_sz;
			numerics Numerics;

			coeff_text = argv[++i];
			Numerics.vec_scan(coeff_text, coeff, coeff_sz);
			if (coeff_sz != 6) {
				cout << "For -line_through_point_with_direction, the number of coefficients must be 6; is " << coeff_sz << endl;
				exit(1);
			}
			S->line(coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5]);
			delete [] coeff;
		}
		else if (strcmp(argv[i], "-plane_by_dual_coordinates") == 0) {
			cout << "-plane_by_dual_coordinates" << endl;
			const char *coeff_text;
			double *coeff;
			int coeff_sz;
			numerics Numerics;

			coeff_text = argv[++i];
			Numerics.vec_scan(coeff_text, coeff, coeff_sz);
			if (coeff_sz != 4) {
				cout << "For -plane_by_dual_coordinates, the number of coefficients must be 4; is " << coeff_sz << endl;
				exit(1);
			}
			S->plane_from_dual_coordinates(coeff);
			delete [] coeff;
		}
		else if (strcmp(argv[i], "-dodecahedron") == 0) {
			cout << "-dodecahedron" << endl;

			int first_pt_idx;

			first_pt_idx = S->nb_points;

			S->Dodecahedron_points();
			S->Dodecahedron_edges(first_pt_idx);
			//cout << "Found " << S->nb_edges << " edges of the Dodecahedron" << endl;
			S->Dodecahedron_planes(first_pt_idx);

			// 20 points
			// 30 edges
			// 12 faces

		}
		else if (strcmp(argv[i], "-Hilbert_Cohn_Vossen_surface") == 0) {
			cout << "-Hilbert_Cohn_Vossen_surface" << endl;

			S->create_Hilbert_Cohn_Vossen_surface(verbose_level);

			// 1 cubic surface
			// 45 planes
			// 27 lines

		}
		else if (strcmp(argv[i], "-obj_file") == 0) {
			cout << "-obj_file" << endl;
			const char *fname;

			fname = argv[++i];
			cout << "before reading file " << fname << endl;
			S->read_obj_file(fname, verbose_level - 1);
			cout << "after reading file " << fname << endl;
		}
		else if (strcmp(argv[i], "-group_of_things") == 0) {
			cout << "-group_of_things" << endl;
			const char *Idx_text;
			int *Idx;
			int Idx_sz;

			Idx_text = argv[++i];
			cout << "group: " << Idx_text << endl;
			int_vec_scan(Idx_text, Idx, Idx_sz);
			cout << "group: ";
			int_vec_print(cout, Idx, Idx_sz);
			cout << endl;
			S->add_a_group_of_things(Idx, Idx_sz, verbose_level);
			FREE_int(Idx);
			cout << "end of -group_of_things" << endl;
		}
		else if (strcmp(argv[i], "-group_of_things_with_offset") == 0) {
			cout << "-group_of_things" << endl;
			const char *Idx_text;
			int *Idx;
			int Idx_sz;
			int offset, h;

			offset = atoi(argv[++i]);
			Idx_text = argv[++i];
			int_vec_scan(Idx_text, Idx, Idx_sz);
			for (h = 0; h < Idx_sz; h++) {
				Idx[h] += offset;
			}
			S->add_a_group_of_things(Idx, Idx_sz, verbose_level);
			FREE_int(Idx);
		}
		else if (strcmp(argv[i], "-group_of_things_as_interval") == 0) {
			cout << "-group_of_things_as_interval" << endl;
			int start;
			int len;
			int h;
			int *Idx;

			start = atoi(argv[++i]);
			len = atoi(argv[++i]);
			Idx = NEW_int(len);
			for (h = 0; h < len; h++) {
				Idx[h] = start + h;
			}
			S->add_a_group_of_things(Idx, len, verbose_level);
			FREE_int(Idx);
		}
		else if (strcmp(argv[i], "-group_of_things_as_interval_with_exceptions") == 0) {
			cout << "-group_of_things_as_interval_with_exceptions" << endl;
			int start;
			int len;
			const char *exceptions_text;
			int h;
			int *Idx;
			int *exceptions;
			int exceptions_sz;
			sorting Sorting;

			start = atoi(argv[++i]);
			len = atoi(argv[++i]);
			exceptions_text = argv[++i];

			int_vec_scan(exceptions_text, exceptions, exceptions_sz);

			Idx = NEW_int(len);
			for (h = 0; h < len; h++) {
				Idx[h] = start + h;
			}

			for (h = 0; h < exceptions_sz; h++) {
				if (!Sorting.int_vec_search_and_remove_if_found(Idx, len, exceptions[h])) {
					cout << "-group_of_things_as_interval_with_exceptions exception not found, value = " << exceptions[h] << endl;
					exit(1);
				}
			}

			FREE_int(exceptions);

			cout << "creating a group of things of size " << len << endl;

			S->add_a_group_of_things(Idx, len, verbose_level);
			FREE_int(Idx);
		}
		else if (strcmp(argv[i], "-group_of_all_points") == 0) {
			cout << "-group_of_all_points" << endl;
			int *Idx;
			int Idx_sz;
			int h;

			Idx_sz = S->nb_points;
			Idx = NEW_int(Idx_sz);
			for (h = 0; h < Idx_sz; h++) {
				Idx[h] = h;
			}
			S->add_a_group_of_things(Idx, Idx_sz, verbose_level);
			FREE_int(Idx);
		}
		else if (strcmp(argv[i], "-group_of_all_faces") == 0) {
			cout << "-group_of_all_faces" << endl;
			int *Idx;
			int Idx_sz;
			int h;

			Idx_sz = S->nb_faces;
			Idx = NEW_int(Idx_sz);
			for (h = 0; h < Idx_sz; h++) {
				Idx[h] = h;
			}
			S->add_a_group_of_things(Idx, Idx_sz, verbose_level);
			cout << "created group " << S->group_of_things.size() - 1 << " consisting of " << Idx_sz << " faces" << endl;
			FREE_int(Idx);
		}
		else if (strcmp(argv[i], "-group_subset_at_random") == 0) {
			cout << "-group_subset_at_random" << endl;
			int group_idx;
			double percentage;
			int *Selection;
			int sz_old;
			int sz;
			int j, r;
			os_interface Os;
			sorting Sorting;

			group_idx = atoi(argv[++i]);
			percentage = atof(argv[++i]);


			sz_old = S->group_of_things[group_idx].size();
			if (f_v) {
				cout << "sz_old" << sz_old << endl;
			}
			sz = sz_old * percentage;
			Selection = NEW_int(sz);
			for (j = 0; j < sz; j++) {
				r = Os.random_integer(sz_old);
				Selection[j] = S->group_of_things[group_idx][r];
			}
			Sorting.int_vec_sort_and_remove_duplicates(Selection, sz);

			S->add_a_group_of_things(Selection, sz, verbose_level);

			FREE_int(Selection);
		}
		else if (strcmp(argv[i], "-create_regulus") == 0) {
			cout << "-create_regulus" << endl;
			int idx, nb_lines;

			idx = atoi(argv[++i]);
			nb_lines = atoi(argv[++i]);
			S->create_regulus(idx, nb_lines, verbose_level);
		}
		else if (strcmp(argv[i], "-spheres") == 0) {
			cout << "-spheres" << endl;
			int group_idx;
			double rad;
			const char *properties;

			group_idx = atoi(argv[++i]);
			rad = atof(argv[++i]);
			properties = argv[++i];

			drawable_set_of_objects D;

			D.init_spheres(group_idx, rad, properties, verbose_level);
			S->Drawables.push_back(D);
		}
		else if (strcmp(argv[i], "-cylinders") == 0) {
			cout << "-cylinders" << endl;
			int group_idx;
			double rad;
			const char *properties;

			group_idx = atoi(argv[++i]);
			rad = atof(argv[++i]);
			properties = argv[++i];

			drawable_set_of_objects D;

			D.init_cylinders(group_idx, rad, properties, verbose_level);
			S->Drawables.push_back(D);
		}
		else if (strcmp(argv[i], "-prisms") == 0) {
			cout << "-prisms" << endl;
			int group_idx;
			double thickness;
			const char *properties;

			group_idx = atoi(argv[++i]);
			thickness = atof(argv[++i]);
			properties = argv[++i];

			drawable_set_of_objects D;

			D.init_prisms(group_idx, thickness, properties, verbose_level);
			S->Drawables.push_back(D);
		}
		else if (strcmp(argv[i], "-planes") == 0) {
			cout << "-planes" << endl;
			int group_idx;
			//double thickness;
			const char *properties;

			group_idx = atoi(argv[++i]);
			//thickness = atof(argv[++i]);
			properties = argv[++i];

			drawable_set_of_objects D;

			D.init_planes(group_idx, properties, verbose_level);
			S->Drawables.push_back(D);
		}
		else if (strcmp(argv[i], "-lines") == 0) {
			cout << "-lines" << endl;
			int group_idx;
			double rad;
			const char *properties;

			group_idx = atoi(argv[++i]);
			rad = atof(argv[++i]);
			properties = argv[++i];

			drawable_set_of_objects D;

			D.init_lines(group_idx, rad, properties, verbose_level);
			S->Drawables.push_back(D);
		}
		else if (strcmp(argv[i], "-cubics") == 0) {
			cout << "-cubics" << endl;
			int group_idx;
			//double thickness;
			const char *properties;

			group_idx = atoi(argv[++i]);
			//thickness = atof(argv[++i]);
			properties = argv[++i];

			drawable_set_of_objects D;

			D.init_cubics(group_idx, properties, verbose_level);
			S->Drawables.push_back(D);
		}
		else if (strcmp(argv[i], "-quadrics") == 0) {
			cout << "-quadrics" << endl;
			int group_idx;
			//double thickness;
			const char *properties;

			group_idx = atoi(argv[++i]);
			//thickness = atof(argv[++i]);
			properties = argv[++i];

			drawable_set_of_objects D;

			D.init_quadrics(group_idx, properties, verbose_level);
			S->Drawables.push_back(D);
		}
		else if (strcmp(argv[i], "-quartics") == 0) {
			cout << "-quartics" << endl;
			int group_idx;
			//double thickness;
			const char *properties;

			group_idx = atoi(argv[++i]);
			//thickness = atof(argv[++i]);
			properties = argv[++i];

			drawable_set_of_objects D;

			D.init_quartics(group_idx, properties, verbose_level);
			S->Drawables.push_back(D);
		}
		else if (strcmp(argv[i], "-octics") == 0) {
			cout << "-octics" << endl;
			int group_idx;
			//double thickness;
			const char *properties;

			group_idx = atoi(argv[++i]);
			//thickness = atof(argv[++i]);
			properties = argv[++i];

			drawable_set_of_objects D;

			D.init_octics(group_idx, properties, verbose_level);
			S->Drawables.push_back(D);
		}
		else if (strcmp(argv[i], "-texts") == 0) {
			cout << "-texts" << endl;
			int group_idx;
			double thickness_half;
			double scale;
			const char *properties;

			group_idx = atoi(argv[++i]);
			thickness_half = atof(argv[++i]);
			scale = atof(argv[++i]);
			properties = argv[++i];

			drawable_set_of_objects D;

			D.init_labels(group_idx, thickness_half, scale, properties, verbose_level);
			S->Drawables.push_back(D);
		}
		else if (strcmp(argv[i], "-scene_objects_end") == 0) {
			cout << "-scene_object_end " << endl;
			break;
		}
		else {
			cout << "-scene_object: unrecognized option " << argv[i] << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "interface_povray::read_scene_objects done" << endl;
	}
	return i;
}

void interface_povray::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_povray::worker" << endl;
	}


	A = NEW_OBJECT(animate);

	A->init(S, output_mask, nb_frames_default, Opt,
			this /* extra_data */,
			verbose_level);


	A->draw_frame_callback = interface_povray_draw_frame;







	//char fname_makefile[1000];


	//sprintf(fname_makefile, "makefile_animation");

	{
	ofstream fpm(A->fname_makefile);

	A->fpm = &fpm;

	fpm << "all:" << endl;

	if (f_rounds) {

		int *rounds;
		int nb_rounds;

		int_vec_scan(rounds_as_string, rounds, nb_rounds);

		cout << "Doing the following " << nb_rounds << " rounds: ";
		int_vec_print(cout, rounds, nb_rounds);
		cout << endl;

		int r;

		for (r = 0; r < nb_rounds; r++) {


			round = rounds[r];

			cout << "round " << r << " / " << nb_rounds
					<< " is " << round << endl;

			//round = first_round + r;

			A->animate_one_round(
					round,
					verbose_level);

			}
		}
	else {
		cout << "round " << round << endl;


		A->animate_one_round(
				round,
				verbose_level);

		}

	fpm << endl;
	}
	file_io Fio;

	cout << "Written file " << A->fname_makefile << " of size "
			<< Fio.file_size(A->fname_makefile) << endl;



	FREE_OBJECT(A);
	//FREE_OBJECT(S);
	A = NULL;
	//S = NULL;
	if (f_v) {
		cout << "interface_povray::worker done" << endl;
	}
}


void interface_povray_draw_frame(
	animate *Anim, int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level)
{
	int i;



	Anim->Pov->union_start(fp);


	if (round == 0) {


		for (i = 0; i < Anim->S->Drawables.size(); i++) {
			drawable_set_of_objects D;

			cout << "drawable " << i << ":" << endl;
			D = Anim->S->Drawables[i];
			D.draw(Anim, fp, verbose_level);
		}


	}

	//Anim->S->clipping_by_cylinder(0, 1.7 /* r */, fp);

	Anim->rotation(h, nb_frames, round, fp);
	Anim->union_end(
			h, nb_frames, round,
			clipping_radius,
			fp);

}


}}

