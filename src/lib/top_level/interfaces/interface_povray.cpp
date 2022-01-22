/*
 * interface_povray.cpp
 *
 *  Created on: Apr 6, 2020
 *      Author: betten
 */







#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {


interface_povray::interface_povray()
{
	f_povray = FALSE;
	Povray_job_description = NULL;

	f_prepare_frames = FALSE;
	Prepare_frames = NULL;
}


void interface_povray::print_help(int argc, std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;

	if (ST.stringcmp(argv[i], "-povray") == 0) {
		cout << "-povray" << endl;
	}
	else if (ST.stringcmp(argv[i], "-prepare_frames") == 0) {
		cout << "-prepare_frames <description> -end" << endl;
	}
}

int interface_povray::recognize_keyword(int argc, std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;

	if (i >= argc) {
		return false;
	}
	if (ST.stringcmp(argv[i], "-povray") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-prepare_frames") == 0) {
		return true;
	}
	return false;
}

void interface_povray::read_arguments(int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_povray::read_arguments" << endl;
	}

	if (f_v) {
		cout << "interface_povray::read_arguments the next argument is " << argv[i] << endl;
	}
	if (ST.stringcmp(argv[i], "-povray") == 0) {
		f_povray = TRUE;
		if (f_v) {
			cout << "-povray " << endl;
		}
		Povray_job_description = NEW_OBJECT(povray_job_description);
		i += Povray_job_description->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

		if (f_v) {
			cout << "done reading -povray_job_description " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
	}
	else if (ST.stringcmp(argv[i], "-prepare_frames") == 0) {
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
		Povray_job_description->print();
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

		graphical_output GO;


		GO.animate_povray(
				Povray_job_description,
				verbose_level);

	}
	else if (f_prepare_frames) {
		Prepare_frames->do_the_work(verbose_level);
	}
	if (f_v) {
		cout << "interface_povray::worker done" << endl;
	}
}


}}

