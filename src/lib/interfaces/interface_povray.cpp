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
				else if (strcmp(argv[i], "-scene_object") == 0) {
					cout << "-scene_object " << endl;
					i++;
					for (; i < argc; i++) {
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
							S->cubic(coeff);
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
						else if (strcmp(argv[i], "-point") == 0) {
							cout << "-point" << endl;
							const char *coeff_text;
							double *coeff;
							int coeff_sz;
							numerics Numerics;

							coeff_text = argv[++i];
							Numerics.vec_scan(coeff_text, coeff, coeff_sz);
							if (coeff_sz != 3) {
								cout << "For -point, the number of coefficients must be 3; is " << coeff_sz << endl;
								exit(1);
							}
							S->point(coeff[0], coeff[1], coeff[2]);
							delete [] coeff;
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
						else if (strcmp(argv[i], "-obj_file") == 0) {
							cout << "-obj_file" << endl;
							const char *fname;

							fname = argv[++i];
							cout << "before reading file " << fname << endl;
							S->read_obj_file(fname, verbose_level - 1);
							cout << "after reading file " << fname << endl;
						}
						else if (strcmp(argv[i], "-scene_object_end") == 0) {
							cout << "-scene_object_end " << endl;
							i++;
							break;
						}
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
	povray_interface Pov;
	int i;



	Pov.union_start(fp);


	if (round == 0) {

		for (i = 0; i < Anim->S->nb_cubics; i++) {
			int selection[1];

			selection[0] = i;

			Anim->S->draw_cubic_with_selection(selection, 1 /* nb_select */,
					Pov.color_white, fp);
		}

		if (Anim->S->nb_lines) {
			int *Selection;

			Selection = NEW_int(Anim->S->nb_lines);
			for (i = 0; i < Anim->S->nb_lines; i++) {
				Selection[i] = i;
			}
			Anim->S->draw_lines_with_selection(Selection, Anim->S->nb_lines,
					Pov.color_red, fp);
			FREE_int(Selection);
		}

		if (Anim->S->nb_edges) {
			int *Selection;

			Selection = NEW_int(Anim->S->nb_edges);
			for (i = 0; i < Anim->S->nb_edges; i++) {
				Selection[i] = i;
			}
			Anim->S->draw_edges_with_selection(Selection, Anim->S->nb_edges,
					Pov.color_red, fp);
			FREE_int(Selection);
		}

		if (Anim->S->nb_faces) {
			int *Selection;
			double thickness_half = 0.02;

			Selection = NEW_int(Anim->S->nb_faces);
			for (i = 0; i < Anim->S->nb_faces; i++) {
				Selection[i] = i;
			}
			Anim->S->draw_faces_with_selection(Selection, Anim->S->nb_faces,
					thickness_half, Pov.color_red, fp);
			FREE_int(Selection);
		}

#if 0
		if (Anim->S->nb_points) {
			int *Selection;
			double rad = 0.6;

			Selection = NEW_int(Anim->S->nb_points);
			for (i = 0; i < Anim->S->nb_points; i++) {
				Selection[i] = i;
			}
			Anim->S->draw_points_with_selection(Selection, Anim->S->nb_points,
					rad, Pov.color_chrome, fp);
			FREE_int(Selection);
		}
#endif

		if (Anim->S->nb_planes) {
			int *Selection;

			Selection = NEW_int(Anim->S->nb_planes);
			for (i = 0; i < Anim->S->nb_planes; i++) {
				Selection[i] = i;
			}
			Anim->S->draw_planes_with_selection(Selection, Anim->S->nb_planes,
					Pov.color_orange_transparent, fp);
			FREE_int(Selection);
		}

		if (Anim->S->nb_quartics) {
			int *Selection;

			Selection = NEW_int(Anim->S->nb_quartics);
			for (i = 0; i < Anim->S->nb_quartics; i++) {
				Selection[i] = i;
			}
			Anim->S->draw_quartic_with_selection(Selection, Anim->S->nb_quartics,
					Pov.color_pink_transparent, fp);
			FREE_int(Selection);
		}


		if (Anim->S->nb_quadrics) {
			int *Selection;

			Selection = NEW_int(Anim->S->nb_quadrics);
			for (i = 0; i < Anim->S->nb_quadrics; i++) {
				Selection[i] = i;
			}
			Anim->S->draw_quadric_with_selection(Selection, Anim->S->nb_quadrics,
					Pov.color_pink_transparent, fp);
			FREE_int(Selection);
		}




		Pov.rotate_111(h, nb_frames, fp);
	}


	if (Anim->Opt->f_has_global_picture_scale) {
		cout << "scale=" << Anim->Opt->global_picture_scale << endl;
		Pov.union_end(fp, Anim->Opt->global_picture_scale, clipping_radius);
	}
	else {
		Pov.union_end(fp, 1.0, clipping_radius);

	}

}


}}

