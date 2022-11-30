/*
 * povray_job_description.cpp
 *
 *  Created on: Jan 1, 2022
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace graphics {



povray_job_description::povray_job_description()
{
	f_output_mask = FALSE;
	//output_mask;
	f_nb_frames_default = FALSE;
	nb_frames_default = 0;
	f_round = FALSE;
	round = 0;
	f_rounds = FALSE;
	//rounds_as_string;
	Video_draw_options = NULL;


	S = NULL;
}


povray_job_description::~povray_job_description()
{
}


int povray_job_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "povray_job_description::read_arguments" << endl;
	}

	for (i = 0; i < argc; i++) {


		if (ST.stringcmp(argv[i], "-video_options") == 0) {
			Video_draw_options = NEW_OBJECT(video_draw_options);
			i += Video_draw_options->read_arguments(argc - (i + 1),
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
		else if (ST.stringcmp(argv[i], "-round") == 0) {
			f_round = TRUE;
			round = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-round " << round << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-rounds") == 0) {
			f_rounds = TRUE;
			rounds_as_string.assign(argv[++i]);
			if (f_v) {
				cout << "-rounds " << rounds_as_string << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-nb_frames_default") == 0) {
			f_nb_frames_default = TRUE;
			nb_frames_default = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-nb_frames_default " << nb_frames_default << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output_mask") == 0) {
			f_output_mask = TRUE;
			output_mask.assign(argv[++i]);
			if (f_v) {
				cout << "-output_mask " << output_mask << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-scene_objects") == 0) {
			if (f_v) {
				cout << "-scene_objects " << endl;
			}
			i++;
			S = NEW_OBJECT(scene);

			S->init(verbose_level);
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
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "povray_job_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "povray_job_description::read_arguments done" << endl;
	}
	return i + 1;
}

void povray_job_description::print()
{
	if (Video_draw_options) {
		Video_draw_options->print();
	}
	if (f_round) {
		cout << "-round " << round << endl;
	}
	if (f_rounds) {
		cout << "-rounds " << rounds_as_string << endl;
	}
	if (f_nb_frames_default) {
		cout << "-nb_frames_default " << nb_frames_default << endl;
	}
	if (f_output_mask) {
		cout << "-output_mask " << output_mask << endl;
	}
	if (S) {
		S->print();
	}
}





}}}

