/*
 * interface_povray.cpp
 *
 *  Created on: Apr 6, 2020
 *      Author: betten
 */







#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


interface_povray::interface_povray()
{
	f_povray = FALSE;
	f_output_mask = FALSE;
	//output_mask;
	f_nb_frames_default = FALSE;
	nb_frames_default = 0;
	f_round = FALSE;
	round = 0;
	f_rounds = FALSE;
	//rounds_as_string;
	Opt = NULL;


	S = NULL;
	A = NULL;

	f_prepare_frames = FALSE;
	Prepare_frames = NULL;
}


void interface_povray::print_help(int argc, std::string *argv, int i, int verbose_level)
{
	if (stringcmp(argv[i], "-povray") == 0) {
		cout << "-povray" << endl;
	}
	else if (stringcmp(argv[i], "-prepare_frames") == 0) {
		cout << "-prepare_frames <description> -end" << endl;
	}
}

int interface_povray::recognize_keyword(int argc, std::string *argv, int i, int verbose_level)
{
	if (i >= argc) {
		return false;
	}
	if (stringcmp(argv[i], "-povray") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-prepare_frames") == 0) {
		return true;
	}
	return false;
}

void interface_povray::read_arguments(int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_povray::read_arguments" << endl;
	}

	if (f_v) {
		cout << "interface_povray::read_arguments the next argument is " << argv[i] << endl;
	}
	if (stringcmp(argv[i], "-povray") == 0) {
		f_povray = TRUE;
		if (f_v) {
			cout << "-povray " << endl;
		}
		i++;

		S = NEW_OBJECT(scene);

		S->init(verbose_level);

		for (; i < argc; i++) {
			if (stringcmp(argv[i], "-video_options") == 0) {
				Opt = NEW_OBJECT(video_draw_options);
				i += Opt->read_arguments(argc - (i - 1),
					argv + i + 1, verbose_level);

				if (f_v) {
					cout << "-video_options" << endl;
					cout << "done with -video_options " << endl;
					cout << "i = " << i << endl;
					cout << "argc = " << argc << endl;
					if (i < argc) {
						cout << "next argument is " << argv[i] << endl;
					}
				}
			}
			else if (stringcmp(argv[i], "-round") == 0) {
				f_round = TRUE;
				round = strtoi(argv[++i]);
				if (f_v) {
					cout << "-round " << round << endl;
				}
			}

			else if (stringcmp(argv[i], "-rounds") == 0) {
				f_rounds = TRUE;
				rounds_as_string.assign(argv[++i]);
				if (f_v) {
					cout << "-rounds " << rounds_as_string << endl;
				}
			}
			else if (stringcmp(argv[i], "-nb_frames_default") == 0) {
				f_nb_frames_default = TRUE;
				nb_frames_default = strtoi(argv[++i]);
				if (f_v) {
					cout << "-nb_frames_default " << nb_frames_default << endl;
				}
			}
			else if (stringcmp(argv[i], "-output_mask") == 0) {
				f_output_mask = TRUE;
				output_mask.assign(argv[++i]);
				if (f_v) {
					cout << "-output_mask " << output_mask << endl;
				}
			}
			else if (stringcmp(argv[i], "-scene_objects") == 0) {
				if (f_v) {
					cout << "-scene_objects " << endl;
				}
				i++;
				i = S->read_scene_objects(argc, argv, i, verbose_level);
				if (f_v) {
					cout << "done with -scene_objects " << endl;
					cout << "i = " << i << endl;
					cout << "argc = " << argc << endl;
					if (i < argc) {
						cout << "next argument is " << argv[i] << endl;
					}
				}
			}
			else if (stringcmp(argv[i], "-povray_end") == 0) {
				if (f_v) {
					cout << "-povray_end " << endl;
				}
				break;
			}
			else {
				cout << "unrecognized option " << argv[i] << endl;
				exit(1);
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
	else if (stringcmp(argv[i], "-prepare_frames") == 0) {
		f_prepare_frames = TRUE;
		Prepare_frames = NEW_OBJECT(prepare_frames);
		i += Prepare_frames->parse_arguments(argc - (i + 1), argv + i + 1, verbose_level);

		if (f_v) {
			cout << "done reading -prepare_frames " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	if (f_v) {
		cout << "interface_povray::read_arguments done" << endl;
	}
}

void interface_povray::print()
{
	if (f_povray) {
		cout << "-povray " << endl;
		S->print();
	}
	if (f_prepare_frames) {
		Prepare_frames->print();
	}
}

void interface_povray::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_povray::worker" << endl;
	}


	if (f_povray) {

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

				Orbiter->Int_vec.scan(rounds_as_string, rounds, nb_rounds);

				cout << "Doing the following " << nb_rounds << " rounds: ";
				Orbiter->Int_vec.print(cout, rounds, nb_rounds);
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
	}
	else if (f_prepare_frames) {
		Prepare_frames->do_the_work(verbose_level);
	}
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
	int i, j;



	Anim->Pov->union_start(fp);


	if (round == 0) {


		for (i = 0; i < (int) Anim->S->Drawables.size(); i++) {
			drawable_set_of_objects D;
			int f_group_is_animated = FALSE;

			if (FALSE) {
				cout << "drawable " << i << ":" << endl;
			}
			D = Anim->S->Drawables[i];

			for (j = 0; j < Anim->S->animated_groups.size(); j++) {
				if (Anim->S->animated_groups[j] == i) {
					break;
				}
			}
			if (j < Anim->S->animated_groups.size()) {
				f_group_is_animated = TRUE;
			}
			if (FALSE) {
				if (f_group_is_animated) {
					cout << "is animated" << endl;
				}
				else {
					cout << "is not animated" << endl;
				}
			}
			D.draw(Anim, fp, f_group_is_animated, h, verbose_level);
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

