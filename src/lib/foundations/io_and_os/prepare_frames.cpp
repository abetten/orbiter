/*
 * prepare_frames.cpp
 *
 *  Created on: Jul 10, 2020
 *      Author: betten
 */




#include "foundations.h"

#include <sstream>

using namespace std;



namespace orbiter {
namespace foundations {


prepare_frames::prepare_frames()
{
	nb_inputs = 0;
	//int input_first[1000];
	//int input_len[1000];
	//const char *input_mask[1000];
	f_o = FALSE;
	output_mask = NULL;
	f_output_starts_at = FALSE;
	output_starts_at = 0;
	f_step = FALSE;
	step = 0;

}

prepare_frames::~prepare_frames()
{

}

int prepare_frames::parse_arguments(int argc, const char **argv)
{
	int i;

	cout << "prepare_frames::parse_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-i") == 0) {
			input_first[nb_inputs] = atoi(argv[++i]);
			input_len[nb_inputs] = atoi(argv[++i]);
			input_mask[nb_inputs] = argv[++i];
			cout << "-i " << input_first[nb_inputs] << " " << input_len[nb_inputs] << " " << input_mask[nb_inputs] << endl;
			nb_inputs++;
		}
		else if (strcmp(argv[i], "-step") == 0) {
			f_step = TRUE;
			step = atoi(argv[++i]);
			cout << "-step " << step << endl;
		}
		else if (strcmp(argv[i], "-o") == 0) {
			f_o = TRUE;
			output_mask = argv[++i];
			cout << "-o " << output_mask << endl;
		}
		else if (strcmp(argv[i], "-output_starts_at") == 0) {
			f_output_starts_at = TRUE;
			output_starts_at = atoi(argv[++i]);
			cout << "-output_starts_at " << output_starts_at << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "prepare_frames::parse_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	}
	return i;
}

void prepare_frames::do_the_work(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_frames;
	int i, j, h, u;
	char input_fname[1000];
	char output_fname[1000];
	char cmd[3000];

	if (f_v) {
		cout << "prepare_frames::do_the_work" << endl;
	}
	nb_frames = 0;
	for (i = 0; i < nb_inputs; i++) {
		nb_frames += input_len[i];
	}

	cout << "nb_frames = " << nb_frames << endl;
	h = output_starts_at;
	for (i = 0; i < nb_inputs; i++) {
		cout << "input " << i << " / " << nb_inputs << endl;
		for (u = 0; u < input_len[i]; u++) {
			j = input_first[i] + u;
			cout << "input " << i << " / " << nb_inputs << " frame " << j << " / " << input_len[i] << endl;

			if (f_step) {
				j *= step;
			}
			sprintf(input_fname, input_mask[i], (int) j);
			sprintf(output_fname, output_mask, (int) h);
			sprintf(cmd, "cp %s %s", input_fname, output_fname);
			system(cmd);
			h++;
		}
	}
	cout << "nb_frames = " << nb_frames << " copied" << endl;
#if 0
	if (h != nb_frames) {
		cout << "h != nb_frames" << endl;
		exit(1);
	}
#endif
	cout << "copied " << h - output_starts_at << " files, we are done" << endl;
	if (f_v) {
		cout << "prepare_frames::do_the_work done" << endl;
	}
}

}}
