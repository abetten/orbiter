/*
 * control_everything.cpp
 *
 *  Created on: Mar 26, 2026
 *      Author: betten
 */

#include "orbiter_user_interface.h"

using namespace std;


namespace orbiter {
namespace layer6_user_interface {
namespace control_everything {



void orbiter_execute_command_line(
		layer5_applications::user_interface::core_system::orbiter_top_level_session
			*The_Orbiter_top_level_session,
		int argc, const char **argv, int verbose_level)
// called from do_orbiter_session in the front-end orbiter.cpp
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_top_level_session::execute_command_line" << endl;
		cout << "A user's guide is available here: " << endl;
		cout << "https://www.math.colostate.edu/~betten/orbiter/users_guide.pdf" << endl;
		cout << "The sources are available here: " << endl;
		cout << "https://github.com/abetten/orbiter" << endl;
		cout << "An example makefile with many commands from the user's guide is here: " << endl;
		cout << "https://github.com/abetten/orbiter/tree/master/examples/users_guide/makefile" << endl;
#ifdef SYSTEMUNIX
		cout << "SYSTEMUNIX is defined" << endl;
#endif
#ifdef SYSTEMWINDOWS
		cout << "SYSTEMWINDOWS is defined" << endl;
#endif
#ifdef SYSTEM_IS_MACINTOSH
		cout << "SYSTEM_IS_MACINTOSH is defined" << endl;
#endif
		cout << "sizeof(int)=" << sizeof(int) << endl;
		cout << "sizeof(long int)=" << sizeof(long int) << endl;
	}

	std::string *Argv;
	other::data_structures::string_tools ST;
	int i;

	//cout << "before ST.convert_arguments, argc=" << argc << endl;

	ST.convert_arguments(argc, argv, Argv);
		// argc has changed!

	//cout << "after ST.convert_arguments, argc=" << argc << endl;

	//cout << "before Top_level_session.startup_and_read_arguments" << endl;
	i = orbiter_startup_and_read_arguments(
			The_Orbiter_top_level_session,
			argc, Argv, 1, verbose_level - 1);
	//cout << "after Top_level_session.startup_and_read_arguments" << endl;


	int session_verbose_level;

	session_verbose_level = The_Orbiter_top_level_session->Orbiter_session->verbose_level;

	if (f_v) {
		cout << "session_verbose_level = " << session_verbose_level << endl;
	}


	//int f_v = (verbose_level > 1);

	if (f_v) {
		cout << "orbiter_top_level_session::execute_command_line "
				"before handle_everything" << endl;
		//cout << "argc=" << argc << endl;
	}


	orbiter_handle_everything(
			The_Orbiter_top_level_session,
			argc, Argv, i, session_verbose_level);

	if (f_v) {
		cout << "orbiter_top_level_session::execute_command_line "
				"after handle_everything" << endl;
	}

	if (f_v) {
		cout << "orbiter_top_level_session::execute_command_line "
				"done" << endl;
	}
}

int orbiter_startup_and_read_arguments(
		layer5_applications::user_interface::core_system::orbiter_top_level_session
			*The_Orbiter_top_level_session,
		int argc,
		std::string *argv, int i0, int verbose_level)
// called from execute_command_line
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_top_level_session::startup_and_read_arguments" << endl;
	}

	int i;

	if (f_v) {
		cout << "orbiter_top_level_session::startup_and_read_arguments "
			"before Orbiter_session->read_arguments" << endl;
	}

	i = The_Orbiter_top_level_session->Orbiter_session->read_arguments(argc, argv, i0, verbose_level);


	if (f_v) {
		cout << "orbiter_top_level_session::startup_and_read_arguments done" << endl;
	}
	return i;
}

void orbiter_handle_everything(
		layer5_applications::user_interface::core_system::orbiter_top_level_session
			*The_Orbiter_top_level_session,
		int argc, std::string *Argv, int i, int verbose_level)
// called from execute_command_line
{
	int f_v = (verbose_level >= 1);

	if (false) {
		cout << "orbiter_top_level_session::handle_everything" << endl;
	}
	if (The_Orbiter_top_level_session->Orbiter_session->f_list_arguments) {
		int j;

		cout << "argument list:" << endl;
		for (j = 0; j < argc; j++) {
			cout << j << " : " << Argv[j] << endl;
		}
	}


	if (The_Orbiter_top_level_session->Orbiter_session->f_fork &&
			!The_Orbiter_top_level_session->Orbiter_session->f_parse_commands_only) {
		if (f_v) {
			cout << "before Orbiter_session->fork" << endl;
		}
		The_Orbiter_top_level_session->Orbiter_session->fork(argc, Argv, verbose_level);
		if (f_v) {
			cout << "after Orbiter_session->fork" << endl;
		}
	}
	else {
		if (The_Orbiter_top_level_session->Orbiter_session->f_seed &&
				!The_Orbiter_top_level_session->Orbiter_session->f_parse_commands_only) {
			other::orbiter_kernel_system::os_interface Os;

			if (f_v) {
				cout << "seeding random number generator with "
						<< The_Orbiter_top_level_session->Orbiter_session->the_seed << endl;
			}
			srand(The_Orbiter_top_level_session->Orbiter_session->the_seed);
			Os.random_integer(1000);
		}
		if (The_Orbiter_top_level_session->Orbiter_session->f_memory_debug) {
			other::orbiter_kernel_system::Orbiter->f_memory_debug = true;
		}

		// main dispatch:

		if (f_v) {
			cout << "orbiter_top_level_session::handle_everything memory_debug "
					"before parse_and_execute" << endl;
		}

		orbiter_parse_and_execute(The_Orbiter_top_level_session, argc, Argv, i, verbose_level);

		if (f_v) {
			cout << "orbiter_top_level_session::handle_everything memory_debug "
					"after parse_and_execute" << endl;
		}


		// finish:

		if (f_v) {
			cout << "orbiter_top_level_session::handle_everything memory_debug "
					"before finish_session" << endl;
		}

		The_Orbiter_top_level_session->finish_session(verbose_level);

		if (f_v) {
			cout << "orbiter_top_level_session::handle_everything memory_debug "
					"after finish_session" << endl;
		}


	}
	if (f_v) {
		cout << "orbiter_top_level_session::handle_everything done" << endl;
	}

}


void orbiter_parse_and_execute(
		layer5_applications::user_interface::core_system::orbiter_top_level_session
			*The_Orbiter_top_level_session,
		int argc, std::string *Argv, int i, int verbose_level)
// called from handle_everything
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int f_vv = false;

	if (false) {
		cout << "orbiter_top_level_session::parse_and_execute, "
				"parsing the orbiter dash code" << endl;
	}


	vector<void *> program;

	if (f_vv) {
		cout << "orbiter_top_level_session::parse_and_execute before parse" << endl;
	}
	orbiter_parse(The_Orbiter_top_level_session, argc, Argv, i, program, verbose_level);
	if (f_vv) {
		cout << "orbiter_top_level_session::parse_and_execute after parse" << endl;
	}

	if (f_v) {
		cout << "orbiter_top_level_session::parse_and_execute, "
				"we have parsed the following orbiter dash code program:" << endl;
	}
	for (i = 0; i < program.size(); i++) {

		orbiter_command *OC;

		OC = (orbiter_command *) program[i];

		cout << "Command " << i << ":" << endl;
		OC->print();
	}

	if (f_v) {
		cout << "################################################################################################" << endl;
	}

	if (The_Orbiter_top_level_session->Orbiter_session->f_parse_commands_only) {
		cout << "not executing the commands because of option -parse_commands_only" << endl;
	}
	else {
		if (f_v) {
			cout << "Executing commands:" << endl;
		}
		for (i = 0; i < program.size(); i++) {

			orbiter_command *OC;

			OC = (orbiter_command *) program[i];

			if (f_v) {
				cout << "################################################################################################" << endl;
				cout << "Executing command " << i << ":" << endl;
				OC->print();
				cout << "################################################################################################" << endl;
			}

			OC->execute(verbose_level);

		}
		if (f_v) {
			cout << "Executing commands done" << endl;
		}
	}



	if (f_v) {
		cout << "orbiter_top_level_session::parse_and_execute done" << endl;
	}
}

void orbiter_parse(
		layer5_applications::user_interface::core_system::orbiter_top_level_session
			*The_Orbiter_top_level_session,
		int argc, std::string *Argv,
		int &i, std::vector<void *> &program, int verbose_level)
// called from parse_and_execute
// program is a vector of pointers of type orbiter_command
{
	int cnt = 0;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i_prev = -1;

	if (f_v) {
		cout << "orbiter_top_level_session::parse "
				"parsing the orbiter dash code" << endl;
	}

	while (i < argc) {
		if (f_vv) {
			cout << "orbiter_top_level_session::parse "
					"cnt = " << cnt << ", i = " << i << endl;
			if (i < argc) {
				if (f_vv) {
					cout << "orbiter_top_level_session::parse i=" << i
							<< ", next argument is " << Argv[i] << endl;
				}
			}
		}
		if (i_prev == i) {
			cout << "orbiter_top_level_session::parse "
					"we seem to be stuck in a look" << endl;
			exit(1);
		}
		i_prev = i;
		if (f_v) {
			cout << "orbiter_top_level_session::parse "
					"i = " << i << endl;
		}

		orbiter_command *OC;

		OC = NEW_OBJECT(orbiter_command);
		if (f_vv) {
			cout << "orbiter_top_level_session::parse "
					"before OC->parse" << endl;
		}
		OC->parse(
				The_Orbiter_top_level_session,
				argc, Argv, i,
				verbose_level);
		if (f_vv) {
			cout << "orbiter_top_level_session::parse "
					"after OC->parse" << endl;
		}
		if (f_vv) {
			cout << "orbiter_top_level_session::parse "
					"command " << program.size() << " starting at token " << i << " is:" << endl;
			OC->print();
		}

		program.push_back(OC);

#if 0
		if (f_v) {
			cout << "orbiter_top_level_session::parse "
					"before OC->execute" << endl;
		}
		OC->execute(verbose_level);
		if (f_v) {
			cout << "orbiter_top_level_session::parse "
					"after OC->execute" << endl;
		}
#endif




		//cout << "Command is unrecognized " << Argv[i] << endl;
		//exit(1);
		cnt++;
	}

	if (f_v) {
		cout << "orbiter_top_level_session::parse "
				"parsing the orbiter dash code done" << endl;
	}
}


}}}



