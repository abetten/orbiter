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
namespace layer1_foundations {
namespace orbiter_kernel_system {


prepare_frames::prepare_frames()
{
	nb_inputs = 0;
	//int input_first[1000];
	//int input_len[1000];
	//const char *input_mask[1000];
	f_o = false;
	//output_mask;
	f_output_starts_at = false;
	output_starts_at = 0;
	f_step = false;
	step = 0;

}

prepare_frames::~prepare_frames()
{

}

int prepare_frames::parse_arguments(int argc, std::string *argv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "prepare_frames::parse_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-i") == 0) {
			input_first[nb_inputs] = ST.strtoi(argv[++i]);
			input_len[nb_inputs] = ST.strtoi(argv[++i]);
			input_mask[nb_inputs].assign(argv[++i]);
			if (f_v) {
				cout << "-i " << input_first[nb_inputs] << " " << input_len[nb_inputs] << " " << input_mask[nb_inputs] << endl;
			}
			nb_inputs++;
		}
		else if (ST.stringcmp(argv[i], "-step") == 0) {
			f_step = true;
			step = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-step " << step << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-o") == 0) {
			f_o = true;
			output_mask.assign(argv[++i]);
			if (f_v) {
				cout << "-o " << output_mask << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output_starts_at") == 0) {
			f_output_starts_at = true;
			output_starts_at = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-output_starts_at " << output_starts_at << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "prepare_frames::parse_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	}
	return i + 1;
}

void prepare_frames::print()
{
	int i;

	if (f_step) {
		cout << "-step " << step << endl;
	}
	if (f_o) {
		cout << "-o " << output_mask << endl;
	}
	if (f_output_starts_at) {
		cout << "-output_starts_at " << output_starts_at << endl;
	}
	for (i = 0; i < nb_inputs; i++) {
		print_item(i);
	}
}

void prepare_frames::print_item(int i)
{
	cout << "-i " << input_first[i] << " " << input_len[i] << " " << input_mask[i] << endl;
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
			snprintf(input_fname, sizeof(input_fname), input_mask[i].c_str(), (int) j);
			snprintf(output_fname, sizeof(output_fname), output_mask.c_str(), (int) h);
			snprintf(cmd, sizeof(cmd), "cp %s %s", input_fname, output_fname);
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

}}}


