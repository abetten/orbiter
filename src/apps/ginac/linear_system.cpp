/*
 * linear_system.cpp
 *
 *  Created on: Jan 21, 2020
 *      Author: betten
 */



#include "orbiter.h"
// Include Orbiter definitions


using namespace std;
using namespace orbiter;
using namespace orbiter::layer5_applications;
// use orbiter's namespaces


// We rely on a package called ginac.
// Ginac is a C++ package for computer algebra
// despite the misleading acronym "Ginac is not a computer algebra system"

#include "ginac/ginac.h"

using namespace GiNaC;
// use ginac's namespace

#include <iostream>
// standard C++ stuff
using namespace std;
// use namespace std which countains things like cout

#include "ginac_linear_algebra.cpp"



void linear_system(int argc, const char **argv);
void draw_frame_linear_system(
	animate *Anim, int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level);
int main(int argc, const char **argv);


void linear_system(int argc, const char **argv)
{
	int verbose_level = 0;
	int f_output_mask = FALSE;
	const char *output_mask = NULL;
	int f_nb_frames_default = FALSE;
	int nb_frames_default;
	int f_round = FALSE;
	int round;
	int f_rounds = FALSE;
	const char *rounds_as_string = NULL;
	video_draw_options *Opt = NULL;

	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-video_options") == 0) {
			Opt = NEW_OBJECT(video_draw_options);
			i += Opt->read_arguments(argc - (i - 1),
				argv + i, verbose_level);

			cout << "-linear" << endl;
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



	char fname[1000];
	char title[1000];
	const char *author = "Orbiter";
	const char *extras_for_preamble = "";

	vsnprintf(fname, 1000, "HCV_report.tex", 0);
	vsnprintf(title, 1000, "The Hilbert, Cohn-Vossen Cubic Surface", 0);

	{
		ofstream fp(fname);
		latex_interface L;
		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */, TRUE /* f_title */,
			title, author,
			FALSE /*f_toc*/, FALSE /* f_landscape*/, FALSE /* f_12pt*/,
			TRUE /*f_enlarged_page*/, TRUE /* f_pagenumbers*/,
			extras_for_preamble);




		{
			scene *S;

			S = NEW_OBJECT(scene);

			S->init(verbose_level);


			double planes[] = {
					3., 2., -1., 1.,
					2., -2., 4., -2.,
					-1., .5, -1., 0
			};

			for (i = 0; i < 3; i++) {

				double x, y, z, rhs, n;

				x = planes[4 * i + 0];
				y = planes[4 * i + 1];
				z = planes[4 * i + 2];
				rhs = planes[4 * i + 3];

				n = sqrt(x *x + y * y + z * z);

				x = x / n;
				y = y / n;
				z = z / n;
				rhs = rhs / n;

				S->plane(x, y, z, rhs);
						// A plane is called a polynomial shape because
						// it is defined by a first order polynomial equation.
						// Given a plane: plane { <A, B, C>, D }
						// it can be represented by the equation
						// A*x + B*y + C*z - D*sqrt(A^2 + B^2 + C^2) = 0.
						// see http://www.povray.org/documentation/view/3.6.1/297/

			}


			animate *A;

			A = NEW_OBJECT(animate);

			A->init(S, output_mask, nb_frames_default, Opt,
					NULL /* extra_data */,
					verbose_level);


			A->draw_frame_callback = draw_frame_linear_system;







			//char fname_makefile[1000];


			//strcpy(fname_makefile, "makefile_animation");

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



			FREE_OBJECT(S);
		}




		L.foot(fp);
	}

}

void draw_frame_linear_system(
	animate *Anim, int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level)
{
	int i;
	povray_interface Pov;



	Pov.union_start(fp);


	if (round == 0) {

		Anim->S->draw_plane(0, Pov.color_red, fp);
		Anim->S->draw_plane(1, Pov.color_blue, fp);
		Anim->S->draw_plane(2, Pov.color_yellow, fp);

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


int main(int argc, const char **argv)
{

	linear_system(argc, argv);

}
