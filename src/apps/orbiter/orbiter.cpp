// orbiter.cpp
//
// by Anton Betten
//
// started: 4/3/2020
//

#include "orbiter.h"

using namespace std;
using namespace orbiter;
using namespace orbiter::interfaces;

int build_number =
#include "../../../build_number"
;


int main(int argc, const char **argv)
{

	//cout << "orbiter.out main" << endl;

	orbiter_session Session;
	int i;


	// setup:


	Orbiter_session = &Session;

	i = Session.read_arguments(argc, argv, 1);


	int verbose_level;

	verbose_level = Session.verbose_level;

	int f_v = (Session.verbose_level > 1);

	if (f_v) {
		cout << "Welcome to Orbiter!  Your build number is " << build_number << endl;
	}

	if (Session.f_list_arguments) {
		int j;

		cout << "argument list:" << endl;
		for (j = 0; j < argc; j++) {
			cout << j << " : " << argv[j] << endl;
		}
#if 0
		string cmd;

		cmd.assign(Session.orbiter_path);
		cmd.append("orbiter.out");
		for (j = 1; j < argc; j++) {
			cmd.append(" \"");
			cmd.append(argv[j]);
			cmd.append("\" ");
		}
		cout << "system: " << cmd << endl;
		system(cmd.c_str());
		exit(1);
#endif
	}
	if (Session.f_fork) {
		cout << "forking with respect to " << Session.fork_variable << endl;
		int j, h, case_number;
		vector<int> places;

		for (j = 1; j < argc; j++) {
			if (strcmp(Session.fork_variable.c_str(), argv[j]) == 0) {
				if (j != Session.fork_argument_idx + 1) {
					places.push_back(j);
				}
			}
		}
		cout << "the variable appears in " << places.size() << " many places:" << endl;
		for (j = 0; j < places.size(); j++) {
			cout << "argument " << places[j] << " is " << argv[places[j]] << endl;
		}


		for (case_number = Session.fork_from; case_number < Session.fork_to; case_number += Session.fork_step) {

			cout << "forking case " << case_number << endl;

			string cmd;

			cmd.assign(Session.orbiter_path);
			cmd.append("orbiter.out");
			for (j = Session.fork_argument_idx + 6; j < argc; j++) {
				cmd.append(" \"");
				for (h = 0; h < places.size(); h++) {
					if (places[h] == j) {
						break;
					}
				}
				if (h < places.size()) {
					char str[1000];

					sprintf(str, "%d", case_number);
					cmd.append(str);
				}
				else {
					cmd.append(argv[j]);
				}
				cmd.append("\" ");
			}
			char str[1000];

			sprintf(str, Session.fork_logfile_mask.c_str(), case_number);
			cmd.append(" >");
			cmd.append(str);
			cmd.append(" &");
			cout << "system: " << cmd << endl;
			system(cmd.c_str());
		}
	}
	else {
		if (Session.f_seed) {
			os_interface Os;

			if (f_v) {
				cout << "seeding random number generator with " << Session.the_seed << endl;
			}
			srand(Session.the_seed);
			Os.random_integer(1000);
		}

		// main dispatch:

		Session.work(argc, argv, i, verbose_level);


		// finish:

		if (f_memory_debug) {
			global_mem_object_registry.dump();
		}
	}

	the_end(Session.t0);

}


