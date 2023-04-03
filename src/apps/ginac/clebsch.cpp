/*
 * clebsch.cpp
 *
 *  Created on: Feb 14, 2020
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


void surface(int argc, const char **argv);
void draw_frame(
	animate *Anim, int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level);
matrix make_cij(matrix **AB, int i, int j, std::ostream &ost, int verbose_level);


void surface(int argc, const char **argv)
// Computes the equation of the Hilbert, Cohn-Vossen surface
{
	int verbose_level = 0;
	int f_output_mask = false;
	const char *output_mask = NULL;
	int f_nb_frames_default = false;
	int nb_frames_default;
	int f_round = false;
	int round;
	int f_rounds = false;
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
			f_round = true;
			round = atoi(argv[++i]);
			cout << "-round " << round << endl;
			}

		else if (strcmp(argv[i], "-rounds") == 0) {
			f_rounds = true;
			rounds_as_string = argv[++i];
			cout << "-rounds " << rounds_as_string << endl;
			}
		else if (strcmp(argv[i], "-nb_frames_default") == 0) {
			f_nb_frames_default = true;
			nb_frames_default = atoi(argv[++i]);
			cout << "-nb_frames_default " << nb_frames_default << endl;
			}
		else if (strcmp(argv[i], "-output_mask") == 0) {
			f_output_mask = true;
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

	vsnprintf(fname, 1000, "Clebsch_report.tex", 0);
	vsnprintf(title, 1000, "The Clebsch Cubic Surface", 0);

	{
		ofstream fp(fname);
		latex_interface L;
		//latex_head_easy(fp);
		L.head(fp,
			false /* f_book */, true /* f_title */,
			title, author,
			false /*f_toc*/, false /* f_landscape*/, false /* f_12pt*/,
			true /*f_enlarged_page*/, true /* f_pagenumbers*/,
			extras_for_preamble);

		//LG->report(fp, f_sylow, f_group_table, verbose_level);




		// create the homogeneous component of degree 3
		// in the polynomial ring F_2[X0,X1,X2,X3]:
		// We don't need the fact that it is over F_2,
		// but we have to pick a finite field in order to create the ring
		// in Orbiter:

		homogeneous_polynomial_domain *HPD;
		finite_field *F;
		int verbose_level = 1;

		// create the finite field F_2:

		F = NEW_OBJECT(finite_field);
		F->init(2);


		surface_domain *Surf;

		Surf = NEW_OBJECT(surface_domain);

		Surf->init(F, 0 /*verbose_level*/);
		//void init_polynomial_domains(int verbose_level);

			// create the homogeneous polynomial ring
		// of degree 3 in X0,X1,X2,X3:
		// Orbiter will create the 20 monomials

		HPD = NEW_OBJECT(homogeneous_polynomial_domain);
		HPD->init(F, 4 /* nb_vars */, 3 /* degree */,
				false /* f_init_incidence_structure */,
				t_PART,
				0 /*verbose_level*/);

		//print the monomials in Orbiter ordering:
		for (i = 0; i < HPD->get_nb_monomials(); i++) {
			cout << i << " : ";
			HPD->print_monomial(cout, i);
			cout << endl;
		}

		// Create the 20 monomials as objects in ginac:
		// We use Orbiter created strings which represent each monomial
		ex M[20];
		parser reader;

		for (i = 0; i < HPD->get_nb_monomials(); i++) {

			std::stringstream sstr;


			HPD->print_monomial_str(sstr, i);


			M[i] = reader(sstr.str().c_str());
			ostringstream s;
			s << latex << sstr.str();
			cout << "M[" << i << "]=" << s.str() << endl;
		}


		double F_Eqn[20] = {
				-3,7,7,1,7,-2,-14,7,-14,3,-3,7,1,7,-14,3,-3,1,3,-1,
				//0, -4, -4, -4, -4, -8, -8, -4, -8, -4, 0, -4, -4, -4, -8, -4, 0, -4, -4, 0,
#if 0
				-9, // 1 x^3
				21, // 2 x^2y
				21, // 3 x^2z
				3, // 4 x^2
				21, // 5 xy^2
				-6, // 6 xyz
				-42, // 7 xy
				21, // 8 xz^2
				-42, // 9 xz
				9, // 10 x
				-9, // 11 y^3
				21, // 12 y^2z
				3, // 13 y^2
				21, // 14 yz^2
				-42, // 15 yz
				9, // 16 y
				-9, // 17 z^3
				3, // 18 z^2
				9, // 19 z
				-3, // 20 1
#endif
		};

		//<-3,7,7,1,7,-2,-14,7,-14,3,-3,7,1,7,-14,3,-3,1,3,-1.>


		double H_Eqn[20] = {
				-262656, // 1 x^3
				96768, // 2 x^2y
				96768, // 3 x^2z
				214272, // 4 x^2
				96768, // 5 xy^2
				635904, // 6 xyz
				-532224, // 7 xy
				96768, // 8 xz^2
				-525312, // 9 xz
				240408, // 10 x
				-262656, // 11 y^3
				96768, // 12 y^2z
				179712, // 13 y^2
				96768, // 14 yz^2
				-518400, // 15 yz
				264600, // 16 y
				-262656, // 17 z^3
				145152, // 18 z^2
				281880, // 19 z
				-179928, // 20 1
				//-262656*x^3 + 96768*x^2*y + 96768*x^2*z + 96768*x*y^2 + 635904*x*y*z + 96768*x*z^2 - 262656*y^3 + 96768*y^2*z + 96768*y*z^2 - 262656*z^3 + 214272*x^2 - 532224*x*y - 525312*x*z + 179712*y^2 - 518400*y*z + 145152*z^2 + 240408*x + 264600*y + 281880*z - 179928

		};



		{
			scene *S;

			S = NEW_OBJECT(scene);

			S->init(verbose_level);


			// clebsch cubic version 1:

			S->cubic(F_Eqn); // cubic 0
			//clebsch_cubic(); // cubic 2
			// lines 33-59   (previously: 21, 21+1, ... 21+26=47)
			S->clebsch_cubic_lines_a();
			S->clebsch_cubic_lines_b();
			S->clebsch_cubic_lines_cij();
			S->Clebsch_Eckardt_points();
			S->cubic(H_Eqn); // cubic 1


			// and now: X0^3+X1^3+X2^3+X3^3-(X0+X1+X2+X3)^3
			S->clebsch_cubic_version2(); // cubic 2
			S->clebsch_cubic_version2_lines_a(); // starting at 27
			S->clebsch_cubic_version2_lines_b();
			S->clebsch_cubic_version2_lines_c();


			S->clebsch_cubic_version2_Hessian(); // cubic 3



			animate *A;

			A = NEW_OBJECT(animate);

			A->init(S, output_mask, nb_frames_default, Opt,
					NULL /* extra_data */,
					verbose_level);


			A->draw_frame_callback = draw_frame;







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

void draw_frame(
	animate *Anim, int h, int nb_frames, int round,
	double clipping_radius,
	ostream &fp,
	int verbose_level)
{
	int i;
	povray_interface Pov;




	Pov.union_start(fp);



	if (round == 0) {

		{
			int selection[] = {0};
			Anim->S->draw_cubic_with_selection(selection, 1,
					Pov.color_white_simple, fp);

			//Anim->S->draw_lines_cij(fp);
			//Anim->S->draw_lines_ai(fp);
			//Anim->S->draw_lines_bj(fp);
		}

		Pov.rotate_111(h, nb_frames, fp);
	}

	else if (round == 1) {

		{
			int selection[] = {0};
			Anim->S->draw_cubic_with_selection(selection, 1,
					Pov.color_white_simple, fp);

			Anim->S->draw_lines_cij(fp);
			Anim->S->draw_lines_ai(fp);
			Anim->S->draw_lines_bj(fp);

			int s[7] = {0,1,2,3,4,5,6};

			Anim->S->draw_points_with_selection(s, 7, 0.15, Pov.color_chrome, fp);
		}

		Pov.rotate_111(h, nb_frames, fp);
	}

	else if (round == 2) {

		{
			int selection[] = {0};
			//Anim->S->draw_cubic_with_selection(selection, 1,
			//		Pov.color_white_simple, fp);

			Anim->S->draw_lines_cij(fp);
			Anim->S->draw_lines_ai(fp);
			Anim->S->draw_lines_bj(fp);

			int s[7] = {0,1,2,3,4,5,6};

			Anim->S->draw_points_with_selection(s, 7, 0.15, Pov.color_chrome, fp);
		}

		Pov.rotate_111(h, nb_frames, fp);
	}


	else if (round == 3) {

		{
			int selection[] = {2};
			Anim->S->draw_cubic_with_selection(selection, 1,
					Pov.color_white_simple, fp);

			//Anim->S->draw_lines_cij(fp);
			//Anim->S->draw_lines_ai(fp);
			//Anim->S->draw_lines_bj(fp);
		}

		Pov.rotate_111(h, nb_frames, fp);
	}

	else if (round == 4) {

		{
			int selection[] = {2};
			Anim->S->draw_cubic_with_selection(selection, 1,
					Pov.color_white_simple, fp);

			Anim->S->draw_lines_cij_with_offset(27, 12, fp);
			Anim->S->draw_lines_ai_with_offset(27, fp);
			Anim->S->draw_lines_bj_with_offset(27, fp);
		}

		Pov.rotate_111(h, nb_frames, fp);
	}

	else if (round == 5) {

			// the Hessian:
		{
			int selection[] = {3};
			Anim->S->draw_cubic_with_selection(selection, 1,
					Pov.color_white_simple, fp);

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



matrix make_cij(matrix **AB, int i, int j, std::ostream &ost, int verbose_level)
// i and j are between 0 and 5
{
	int f_v = (verbose_level >= 1);
	matrix AiBj(4, 4);
	matrix AjBi(4, 4);
	matrix N1, N2, N3;
	matrix N(4, 4);
	matrix C(2, 4);

	if (f_v) {
		cout << "make_cij i=" << i << " j=" << j << endl;
	}
	AiBj = {
			{AB[i]->m[0 * 4 + 3], AB[i]->m[0 * 4 + 2], AB[i]->m[0 * 4 + 1], AB[i]->m[0 * 4 + 0] },
			{AB[i]->m[1 * 4 + 3], AB[i]->m[1 * 4 + 2], AB[i]->m[1 * 4 + 1], AB[i]->m[1 * 4 + 0] },
			{AB[6 + j]->m[0 * 4 + 3], AB[6 + j]->m[0 * 4 + 2], AB[6 + j]->m[0 * 4 + 1], AB[6 + j]->m[0 * 4 + 0] },
			{AB[6 + j]->m[1 * 4 + 3], AB[6 + j]->m[1 * 4 + 2], AB[6 + j]->m[1 * 4 + 1], AB[6 + j]->m[1 * 4 + 0] }
	};
	//ost << "$A_{" << i + 1 << "}B_{" << j + 1 << "}=" << AiBj << "$\\\\" << endl;
	AjBi = {
			{AB[j]->m[0 * 4 + 3], AB[j]->m[0 * 4 + 2], AB[j]->m[0 * 4 + 1], AB[j]->m[0 * 4 + 0] },
			{AB[j]->m[1 * 4 + 3], AB[j]->m[1 * 4 + 2], AB[j]->m[1 * 4 + 1], AB[j]->m[1 * 4 + 0] },
			{AB[6 + i]->m[0 * 4 + 3], AB[6 + i]->m[0 * 4 + 2], AB[6 + i]->m[0 * 4 + 1], AB[6 + i]->m[0 * 4 + 0] },
			{AB[6 + i]->m[1 * 4 + 3], AB[6 + i]->m[1 * 4 + 2], AB[6 + i]->m[1 * 4 + 1], AB[6 + i]->m[1 * 4 + 0] }
	};
	N1 = right_nullspace(&AiBj);
	N2 = right_nullspace(&AjBi);
	N = {
			{N1(3,0), N1(2,0), N1(1,0), N1(0,0) },
			{N2(3,0), N2(2,0), N2(1,0), N2(0,0) },
	};
	N3 = right_nullspace(&N);
	C = {
			{N3(0,1), N3(1,1), N3(2,1), N3(3,1) },
			{N3(0,0), N3(1,0), N3(2,0), N3(3,0) }
	};
	right_normalize_row(&C, 0);
	right_normalize_row(&C, 1);
	return C;
}


int main(int argc, const char **argv)
{

	surface(argc, argv);

}
