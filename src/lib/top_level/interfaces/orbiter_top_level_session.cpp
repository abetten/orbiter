/*
 * orbiter_top_level_session.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace user_interface {


orbiter_top_level_session *The_Orbiter_top_level_session; // global top level Orbiter session



orbiter_top_level_session::orbiter_top_level_session()
{
	Orbiter_session = NULL;
}

orbiter_top_level_session::~orbiter_top_level_session()
{
	int verbose_level = 1;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_top_level_session::~orbiter_top_level_session" << endl;
	}
	if (Orbiter_session) {
		if (f_v) {
			cout << "orbiter_top_level_session::~orbiter_top_level_session before delete Orbiter_session" << endl;
		}
		delete Orbiter_session;
		if (f_v) {
			cout << "orbiter_top_level_session::~orbiter_top_level_session after delete Orbiter_session" << endl;
		}
	}
	if (f_v) {
		cout << "orbiter_top_level_session::~orbiter_top_level_session done" << endl;
	}
}

int orbiter_top_level_session::startup_and_read_arguments(int argc,
		std::string *argv, int i0)
{
	int i;

	cout << "orbiter_top_level_session::startup_and_read_arguments before new orbiter_session" << endl;

	Orbiter_session = new orbiter_kernel_system::orbiter_session;

	cout << "orbiter_top_level_session::startup_and_read_arguments after new orbiter_session" << endl;
	cout << "orbiter_top_level_session::startup_and_read_arguments before Orbiter_session->read_arguments" << endl;

	i = Orbiter_session->read_arguments(argc, argv, i0);



	cout << "orbiter_top_level_session::startup_and_read_arguments done" << endl;
	return i;
}

void orbiter_top_level_session::handle_everything(int argc, std::string *Argv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (FALSE) {
		cout << "orbiter_top_level_session::handle_everything" << endl;
	}
	if (Orbiter_session->f_list_arguments) {
		int j;

		cout << "argument list:" << endl;
		for (j = 0; j < argc; j++) {
			cout << j << " : " << Argv[j] << endl;
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


	if (Orbiter_session->f_fork) {
		if (f_v) {
			cout << "before Top_level_session.Orbiter_session->fork" << endl;
		}
		Orbiter_session->fork(argc, Argv, verbose_level);
		if (f_v) {
			cout << "after Session.fork" << endl;
		}
	}
	else {
		if (Orbiter_session->f_seed) {
			orbiter_kernel_system::os_interface Os;

			if (f_v) {
				cout << "seeding random number generator with " << Orbiter_session->the_seed << endl;
			}
			srand(Orbiter_session->the_seed);
			Os.random_integer(1000);
		}
		if (Orbiter_session->f_memory_debug) {
			orbiter_kernel_system::Orbiter->f_memory_debug = TRUE;
		}

		// main dispatch:

		if (f_v) {
			cout << "orbiter_top_level_session::handle_everything memory_debug "
					"before parse_and_execute" << endl;
		}

		parse_and_execute(argc, Argv, i, verbose_level);

		if (f_v) {
			cout << "orbiter_top_level_session::handle_everything memory_debug "
					"after parse_and_execute" << endl;
		}


		// finish:

		if (Orbiter_session->f_memory_debug) {
			if (f_v) {
				cout << "orbiter_top_level_session::handle_everything memory_debug "
						"before global_mem_object_registry.dump" << endl;
			}
			orbiter_kernel_system::Orbiter->global_mem_object_registry->dump();
			orbiter_kernel_system::Orbiter->global_mem_object_registry->dump_to_csv_file("orbiter_memory_dump.cvs");
			orbiter_kernel_system::Orbiter->global_mem_object_registry->sort_by_location_and_get_frequency(verbose_level);
			if (f_v) {
				cout << "orbiter_top_level_session::handle_everything memory_debug "
						"after global_mem_object_registry.dump" << endl;
			}
		}
	}
	if (f_v) {
		cout << "orbiter_top_level_session::handle_everything done" << endl;
	}

}

void orbiter_top_level_session::parse_and_execute(int argc, std::string *Argv, int i, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;

	if (FALSE) {
		cout << "orbiter_top_level_session::parse_and_execute, "
				"parsing the orbiter dash code" << endl;
	}


	vector<void *> program;

	if (f_vv) {
		cout << "orbiter_top_level_session::parse_and_execute before parse" << endl;
	}
	parse(argc, Argv, i, program, verbose_level);
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

	if (f_v) {
		cout << "orbiter_top_level_session::parse_and_execute done" << endl;
	}
}

void orbiter_top_level_session::parse(int argc, std::string *Argv,
		int &i, std::vector<void *> &program, int verbose_level)
{
	int cnt = 0;
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;
	int i_prev = -1;

	if (f_v) {
		cout << "orbiter_top_level_session::parse, parsing the orbiter dash code" << endl;
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
			cout << "orbiter_top_level_session::parse we seem to be stuck in a look" << endl;
			exit(1);
		}
		i_prev = i;
		if (f_v) {
			cout << "orbiter_top_level_session::parse before Interface_symbol_table, i = " << i << endl;
		}

		orbiter_command *OC;

		OC = NEW_OBJECT(orbiter_command);
		if (f_vv) {
			cout << "orbiter_top_level_session::parse before OC->parse" << endl;
		}
		OC->parse(this, argc, Argv, i, verbose_level);
		if (f_vv) {
			cout << "orbiter_top_level_session::parse after OC->parse" << endl;
		}

		program.push_back(OC);

#if 0
		if (f_v) {
			cout << "orbiter_top_level_session::parse before OC->execute" << endl;
		}
		OC->execute(verbose_level);
		if (f_v) {
			cout << "orbiter_top_level_session::parse after OC->execute" << endl;
		}
#endif




		//cout << "Command is unrecognized " << Argv[i] << endl;
		//exit(1);
		cnt++;
	}

	if (f_v) {
		cout << "orbiter_top_level_session::parse, parsing the orbiter dash code done" << endl;
	}
}

void *orbiter_top_level_session::get_object(int idx)
{
	return Orbiter_session->get_object(idx);
}

symbol_table_object_type orbiter_top_level_session::get_object_type(int idx)
{
	return Orbiter_session->get_object_type(idx);
}

int orbiter_top_level_session::find_symbol(std::string &label)
{
	return Orbiter_session->find_symbol(label);
}

void orbiter_top_level_session::find_symbols(std::vector<std::string> &Labels, int *&Idx)
{

	Orbiter_session->find_symbols(Labels, Idx);
}

void orbiter_top_level_session::print_symbol_table()
{
	Orbiter_session->print_symbol_table();
}

void orbiter_top_level_session::add_symbol_table_entry(std::string &label,
		orbiter_kernel_system::orbiter_symbol_table_entry *Symb, int verbose_level)
{
	Orbiter_session->add_symbol_table_entry(label, Symb, verbose_level);
}

apps_algebra::any_group *orbiter_top_level_session::get_object_of_type_any_group(std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_any_group cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_any_group) {
		cout << "orbiter_top_level_session::get_object_of_type_any_group object type != t_any_group" << endl;
		exit(1);
	}
	return (apps_algebra::any_group *) get_object(idx);
}

projective_geometry::projective_space_with_action *orbiter_top_level_session::get_object_of_type_projective_space(std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_projective_space cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_projective_space) {
		cout << "orbiter_top_level_session::get_object_of_type_projective_space object type != t_projective_space" << endl;
		exit(1);
	}
	return (projective_geometry::projective_space_with_action *) get_object(idx);
}



}}}


