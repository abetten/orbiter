/*
 * orbiter_session.cpp
 *
 *  Created on: May 26, 2020
 *      Author: betten
 */






#include "foundations.h"


using namespace std;

namespace orbiter {
namespace foundations {


orbiter_session *Orbiter = NULL;


orbiter_session::orbiter_session()
{
	if (Orbiter) {
		cout << "orbiter_session::orbiter_session The_Orbiter_session is non NULL" << endl;
		exit(1);
	}
	Orbiter = this;

	verbose_level = 0;

	t0 = 0;

	f_draw_options = TRUE;
	draw_options = NEW_OBJECT(layered_graph_draw_options);


	f_list_arguments = FALSE;

	f_seed = FALSE;
	the_seed = TRUE;

	f_memory_debug = FALSE;
	memory_debug_verbose_level = 0;

	f_override_polynomial = FALSE;
	//override_polynomial = NULL;

	f_orbiter_path = FALSE;
	//orbiter_path;

	f_magma_path = FALSE;
	//magma_path

	f_fork = FALSE;
	fork_argument_idx = 0;
	// fork_variable
	// fork_logfile_mask
	fork_from = 0;
	fork_to = 0;
	fork_step = 0;

	Orbiter_symbol_table = NULL;
}


orbiter_session::~orbiter_session()
{
	Orbiter = NULL;
	if (Orbiter_symbol_table) {
		FREE_OBJECT(Orbiter_symbol_table);
	}
}


void orbiter_session::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (stringcmp(argv[i], "-v") == 0) {
		cout << "-v <int : verbosity>" << endl;
	}
	else if (stringcmp(argv[i], "-draw_options") == 0) {
		cout << "-draw_options ... -end" << endl;
	}
	else if (stringcmp(argv[i], "-list_arguments") == 0) {
		cout << "-list_arguments" << endl;
	}
	else if (stringcmp(argv[i], "-seed") == 0) {
		cout << "-seed <int : seed>" << endl;
	}
	else if (stringcmp(argv[i], "-memory_debug") == 0) {
		cout << "-memory_debug <int : memory_debug_verbose_level>" << endl;
	}
	else if (stringcmp(argv[i], "-override_polynomial") == 0) {
		cout << "-override_polynomial <string : polynomial in decimal>" << endl;
	}
	else if (stringcmp(argv[i], "-orbiter_path") == 0) {
		cout << "-orbiter_path <string : path>" << endl;
	}
	else if (stringcmp(argv[i], "-magma_path") == 0) {
		cout << "-magma_path <string : path>" << endl;
	}
	else if (stringcmp(argv[i], "-fork") == 0) {
		cout << "-fork <string : variable> <string : logfile_mask> <int : from> <int : to> <int : step>" << endl;
	}
}

int orbiter_session::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (stringcmp(argv[i], "-v") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-draw_options") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-list_arguments") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-seed") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-memory_debug") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-override_polynomial") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-orbiter_path") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-magma_path") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-fork") == 0) {
		return true;
	}
	return false;
}

int orbiter_session::read_arguments(int argc,
		std::string *argv, int i0)
{
	int i;

	//cout << "orbiter_session::read_arguments" << endl;

	os_interface Os;

	t0 = Os.os_ticks();

	Orbiter_symbol_table = NEW_OBJECT(orbiter_symbol_table);

	for (i = i0; i < argc; i++) {
		if (stringcmp(argv[i], "-v") == 0) {
			verbose_level = strtoi(argv[++i]);
			//cout << "-v " << verbose_level << endl;
		}
		else if (stringcmp(argv[i], "-draw_options") == 0) {
			f_draw_options = TRUE;

			draw_options = NEW_OBJECT(layered_graph_draw_options);
			//cout << "-draw_options " << endl;
			i += draw_options->read_arguments(argc - (i + 1),
				argv + i + 1, 0 /*verbose_level*/);

#if 0
			cout << "done reading -draw_options " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-f_draw_options " << endl;
#endif
		}
		else if (stringcmp(argv[i], "-list_arguments") == 0) {
			f_list_arguments = TRUE;
			//cout << "-list_arguments " << endl;
		}
		else if (stringcmp(argv[i], "-seed") == 0) {
			f_seed = TRUE;
			the_seed = strtoi(argv[++i]);
			//cout << "-seed " << the_seed << endl;
		}
		else if (stringcmp(argv[i], "-memory_debug") == 0) {
			f_memory_debug = TRUE;
			memory_debug_verbose_level = strtoi(argv[++i]);
			//cout << "-memory_debug " << memory_debug_verbose_level << endl;
		}
		else if (stringcmp(argv[i], "-override_polynomial") == 0) {
			f_override_polynomial = TRUE;
			override_polynomial.assign(argv[++i]);
			//cout << "-override_polynomial " << override_polynomial << endl;
		}
		else if (stringcmp(argv[i], "-orbiter_path") == 0) {
			f_orbiter_path = TRUE;
			orbiter_path.assign(argv[++i]);
			//cout << "-orbiter_path " << orbiter_path << endl;
		}
		else if (stringcmp(argv[i], "-magma_path") == 0) {
			f_magma_path = TRUE;
			magma_path.assign(argv[++i]);
			cout << "-magma_path " << magma_path << endl;
		}
		else if (stringcmp(argv[i], "-fork") == 0) {
			f_fork = TRUE;
			fork_argument_idx = i;
			fork_variable.assign(argv[++i]);
			fork_logfile_mask.assign(argv[++i]);
			fork_from = strtoi(argv[++i]);
			fork_to = strtoi(argv[++i]);
			fork_step = strtoi(argv[++i]);
			//cout << "-fork " << fork_variable << " " << fork_logfile_mask << " " << fork_from << " " << fork_to << " " << fork_step << endl;
		}
		else {
			break;
		}
	}

	//cout << "orbiter_session::read_arguments done" << endl;
	return i;
}

void orbiter_session::fork(int argc, std::string *argv, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_session::fork" << endl;
	}

	cout << "forking with respect to " << fork_variable << endl;
	int j, h, case_number;
	vector<int> places;

	for (j = 1; j < argc; j++) {
		if (stringcmp(argv[j], fork_variable.c_str()) == 0) {
			if (j != fork_argument_idx + 1) {
				places.push_back(j);
			}
		}
	}
	cout << "the variable appears in " << places.size() << " many places:" << endl;
	for (j = 0; j < places.size(); j++) {
		cout << "argument " << places[j] << " is " << argv[places[j]] << endl;
	}


	for (case_number = fork_from; case_number < fork_to; case_number += fork_step) {

		cout << "forking case " << case_number << endl;

		string cmd;

		cmd.assign(orbiter_path);
		cmd.append("orbiter.out");
		for (j = fork_argument_idx + 6; j < argc; j++) {
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

		sprintf(str, fork_logfile_mask.c_str(), case_number);
		cmd.append(" >");
		cmd.append(str);
		cmd.append(" &");
		cout << "system: " << cmd << endl;
		system(cmd.c_str());
	}
	if (f_v) {
		cout << "orbiter_session::fork done" << endl;
	}

}


void *orbiter_session::get_object(int idx)
{
	return Orbiter_symbol_table->get_object(idx);
}

symbol_table_object_type orbiter_session::get_object_type(int idx)
{
	return Orbiter_symbol_table->get_object_type(idx);
}

int orbiter_session::find_symbol(std::string &label)
{
	return Orbiter_symbol_table->find_symbol(label);
}

void orbiter_session::find_symbols(std::vector<std::string> &Labels, int *&Idx)
{
	int i, idx;

	Idx = NEW_int(Labels.size());

	for (i = 0; i < Labels.size(); i++) {
		idx = find_symbol(Labels[i]);
		if (idx == -1) {
			cout << "cannot find symbol " << Labels[i] << endl;
			exit(1);
		}
		Idx[i] = idx;
	}
}

void orbiter_session::print_symbol_table()
{
	Orbiter_symbol_table->print_symbol_table();
}

void orbiter_session::add_symbol_table_entry(std::string &label,
		orbiter_symbol_table_entry *Symb, int verbose_level)
{
	Orbiter_symbol_table->add_symbol_table_entry(label, Symb, verbose_level);
}

void orbiter_session::get_lint_vec(std::string &label,
		long int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_session::get_lint_vec" << endl;
	}
	if (isalpha(label.c_str()[0])) {
		int idx;

		if (f_v) {
			cout << "orbiter_session::get_lint_vec" << endl;
			cout << "object label " << label << endl;
		}


		idx = Orbiter->Orbiter_symbol_table->find_symbol(label);
		if (f_v) {
			cout << "orbiter_session::get_lint_vec" << endl;
			cout << "idx = " << idx << endl;
		}
		if (idx == -1) {
			cout << "orbiter_session::get_lint_vec cannot find symbol " << label << endl;
			exit(1);
		}
		if (Orbiter->Orbiter_symbol_table->get_object_type(idx) != t_set) {
			cout << "orbiter_session::get_lint_vec object not of type set" << endl;
			exit(1);
		}
		set_builder *SB;

		SB = (set_builder *) Orbiter->Orbiter_symbol_table->get_object(idx);

		set_size = SB->sz;
		the_set = NEW_lint(SB->sz);
		Orbiter->Lint_vec.copy(SB->set, the_set, set_size);

		if (f_v) {
			cout << "orbiter_session::get_lint_vec" << endl;
			cout << "set : ";
			Orbiter->Lint_vec.print(cout, the_set, set_size);
			cout << endl;
		}

	}
	else {
		Orbiter->Lint_vec.scan(label.c_str(), the_set, set_size);
	}
	if (f_v) {
		cout << "orbiter_session::get_lint_vec done" << endl;
	}

}


}}

