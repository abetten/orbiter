/*
 * orbiter_session.cpp
 *
 *  Created on: May 26, 2020
 *      Author: betten
 */






#include "foundations.h"


using namespace std;
using namespace orbiter::layer1_foundations::other::data_structures;

namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace orbiter_kernel_system {

//! global Orbiter session

orbiter_kernel_system::orbiter_session *Orbiter;


orbiter_session::orbiter_session()
{

	cout << "orbiter_session::orbiter_session" << endl;

	if (Orbiter) {
		cout << "orbiter_session::orbiter_session "
				"The_Orbiter_session is non NULL" << endl;
		exit(1);
	}
	Orbiter = this;


	// this needs to come first, because NEW_OBJECT requires it:

	global_mem_object_registry = new mem_object_registry;



	verbose_level = 0;

	t0 = 0;

#if 0
	f_draw_options = true;
	draw_options = NEW_OBJECT(graphics::layered_graph_draw_options);

	f_draw_incidence_structure_description = true;
	Draw_incidence_structure_description = NEW_OBJECT(graphics::draw_incidence_structure_description);
#endif

	f_list_arguments = false;

	f_seed = false;
	the_seed = true;

	f_memory_debug = false;
	memory_debug_verbose_level = 0;

	f_override_polynomial = false;
	//override_polynomial = NULL;

	f_orbiter_path = false;
	//orbiter_path;

	f_magma_path = false;
	//magma_path

	f_fork = false;
	fork_argument_idx = 0;
	// fork_variable
	// fork_logfile_mask
	fork_from = 0;
	fork_to = 0;
	fork_step = 0;

	f_parse_commands_only = false;

	//Orbiter_symbol_table = NULL;
	Orbiter_symbol_table = NEW_OBJECT(orbiter_symbol_table);

	nb_times_finite_field_created = 0;
	nb_times_projective_space_created = 0;
	nb_times_action_created = 0;
	nb_calls_to_densenauty = 0;


	Int_vec = NEW_OBJECT(int_vec);
	Lint_vec = NEW_OBJECT(lint_vec);


	longinteger_f_print_scientific = false;
	syntax_tree_node_index = 0;

	f_has_get_projective_space_low_level_function = false;
	get_projective_space_low_level_function = NULL;

	//std::vector<void *> export_import_stack;

	cout << "orbiter_session::orbiter_session done" << endl;

}


orbiter_session::~orbiter_session()
{
	int verbose_level = 1;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_session::~orbiter_session" << endl;
	}


	cout << "nb_times_finite_field_created=" << nb_times_finite_field_created << endl;
	cout << "nb_times_projective_space_created=" << nb_times_projective_space_created << endl;
	cout << "nb_times_action_created=" << nb_times_action_created << endl;
	cout << "nb_calls_to_densenauty=" << nb_calls_to_densenauty << endl;

	if (Orbiter_symbol_table) {
		FREE_OBJECT(Orbiter_symbol_table);
	}

	Orbiter = NULL;

	if (f_v) {
		cout << "orbiter_session::~orbiter_session done" << endl;
	}
}


void orbiter_session::print_help(
		int argc,
		std::string *argv, int i, int verbose_level)
{
	string_tools ST;

	if (ST.stringcmp(argv[i], "-v") == 0) {
		cout << "-v <int : verbosity>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-draw_options") == 0) {
		cout << "-draw_options ... -end" << endl;
	}
	else if (ST.stringcmp(argv[i], "-draw_incidence_structure_description") == 0) {
		cout << "-draw_incidence_structure_description ... -end" << endl;
	}
	else if (ST.stringcmp(argv[i], "-list_arguments") == 0) {
		cout << "-list_arguments" << endl;
	}
	else if (ST.stringcmp(argv[i], "-seed") == 0) {
		cout << "-seed <int : seed>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-memory_debug") == 0) {
		cout << "-memory_debug <int : memory_debug_verbose_level>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-override_polynomial") == 0) {
		cout << "-override_polynomial <string : polynomial in decimal>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-orbiter_path") == 0) {
		cout << "-orbiter_path <string : path>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-magma_path") == 0) {
		cout << "-magma_path <string : path>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-fork") == 0) {
		cout << "-fork <string : variable> <string : logfile_mask> <int : from> <int : to> <int : step>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-parse_commands_only") == 0) {
		cout << "-parse_commands_only" << endl;
	}
}

int orbiter_session::recognize_keyword(
		int argc,
		std::string *argv, int i, int verbose_level)
{
	string_tools ST;

	if (ST.stringcmp(argv[i], "-v") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-draw_options") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-draw_incidence_structure_description") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-list_arguments") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-seed") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-memory_debug") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-override_polynomial") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-orbiter_path") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-magma_path") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-fork") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-parse_commands_only") == 0) {
		return true;
	}
	return false;
}

int orbiter_session::read_arguments(
		int argc,
		std::string *argv, int i0, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	string_tools ST;

	if (f_v) {
		cout << "orbiter_session::read_arguments" << endl;
	}

	os_interface Os;

	t0 = Os.os_ticks();



	for (i = i0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-v") == 0) {
			orbiter_session::verbose_level = ST.strtoi(argv[++i]);
			//f_v = (verbose_level >= 1);
			if (f_v) {
				cout << "-v " << verbose_level << endl;
			}
		}
#if 0
		else if (ST.stringcmp(argv[i], "-draw_options") == 0) {
			f_draw_options = true;

			draw_options = NEW_OBJECT(graphics::layered_graph_draw_options);
			//cout << "-draw_options " << endl;
			i += draw_options->read_arguments(argc - (i + 1),
				argv + i + 1, 0 /*verbose_level*/);

			if (f_v) {
				cout << "done reading -draw_options " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-f_draw_options " << endl;
				draw_options->print();
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_incidence_structure_description") == 0) {
			f_draw_incidence_structure_description = true;

			Draw_incidence_structure_description = NEW_OBJECT(graphics::draw_incidence_structure_description);
			//cout << "-draw_incidence_structure_description " << endl;
			i += Draw_incidence_structure_description->read_arguments(argc - (i + 1),
				argv + i + 1, 0 /*verbose_level*/);

			if (f_v) {
				cout << "done reading -draw_incidence_structure_description " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-draw_incidence_structure_description " << endl;
			}
		}
#endif


		else if (ST.stringcmp(argv[i], "-list_arguments") == 0) {
			f_list_arguments = true;
			if (f_v) {
				cout << "-list_arguments " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-seed") == 0) {
			f_seed = true;
			the_seed = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-seed " << the_seed << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-memory_debug") == 0) {
			f_memory_debug = true;
			memory_debug_verbose_level = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-memory_debug " << memory_debug_verbose_level << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-override_polynomial") == 0) {
			f_override_polynomial = true;
			override_polynomial.assign(argv[++i]);
			if (f_v) {
				cout << "-override_polynomial " << override_polynomial << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orbiter_path") == 0) {
			f_orbiter_path = true;
			orbiter_path.assign(argv[++i]);
			if (f_v) {
				cout << "-orbiter_path " << orbiter_path << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-magma_path") == 0) {
			f_magma_path = true;
			magma_path.assign(argv[++i]);
			if (f_v) {
				cout << "-magma_path " << magma_path << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-fork") == 0) {
			f_fork = true;
			fork_argument_idx = i;
			fork_variable.assign(argv[++i]);
			fork_logfile_mask.assign(argv[++i]);
			fork_from = ST.strtoi(argv[++i]);
			fork_to = ST.strtoi(argv[++i]);
			fork_step = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-fork " << fork_variable
						<< " " << fork_logfile_mask
						<< " " << fork_from
						<< " " << fork_to
						<< " " << fork_step
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-parse_commands_only") == 0) {
			f_parse_commands_only = true;
			if (f_v) {
				cout << "-parse_commands_only" << endl;
			}
		}
		else {
			if (f_v) {
				cout << "unrecognized command " << argv[i]
					<< " finished scanning global Orbiter options" << endl;
			}
			break;
		}
	}

	if (f_v) {
		cout << "orbiter_session::read_arguments done" << endl;
	}
	return i;
}

void orbiter_session::fork(
		int argc, std::string *argv, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_session::fork" << endl;
	}

	if (f_v) {
		cout << "forking with respect to " << fork_variable << endl;
	}
	int j, h, case_number;
	vector<int> places;
	string_tools ST;

	for (j = 1; j < argc; j++) {
		if (ST.stringcmp(argv[j], fork_variable.c_str()) == 0) {
			if (j != fork_argument_idx + 1) {
				places.push_back(j);
			}
		}
	}
	if (f_v) {
		cout << "the variable appears in " << places.size() << " many places:" << endl;
		for (j = 0; j < places.size(); j++) {
			cout << "argument " << places[j] << " is " << argv[places[j]] << endl;
		}
	}


	for (case_number = fork_from; case_number < fork_to; case_number += fork_step) {

		if (f_v) {
			cout << "forking case " << case_number << endl;
		}

		string cmd;

		cmd = orbiter_path + "orbiter.out";
		for (j = fork_argument_idx + 6; j < argc; j++) {
			cmd += " \"";
			for (h = 0; h < places.size(); h++) {
				if (places[h] == j) {
					break;
				}
			}
			if (h < places.size()) {
				cmd += std::to_string(case_number);
			}
			else {
				cmd += argv[j];
			}
			cmd += "\" ";
		}
		data_structures::string_tools ST;


		string str;

		str = ST.printf_d(fork_logfile_mask, case_number);

		cmd += " >" + str + " &";
		cout << "system: " << cmd << endl;
		system(cmd.c_str());
	}
	if (f_v) {
		cout << "orbiter_session::fork done" << endl;
	}

}


void *orbiter_session::get_object(
		int idx)
{
	return Orbiter_symbol_table->get_object(idx);
}

symbol_table_object_type orbiter_session::get_object_type(
		int idx)
{
	return Orbiter_symbol_table->get_object_type(idx);
}

int orbiter_session::find_symbol(
		std::string &label)
{
	return Orbiter_symbol_table->find_symbol(label);
}

void orbiter_session::get_vector_from_label(
		std::string &label,
		long int *&v, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_session::get_vector_from_label" << endl;
	}
	if (isalpha(label[0])) {
		if (f_v) {
			cout << "orbiter_session::get_vector_from_label "
					"searching label " << label << endl;
		}
		int idx;

		idx = Orbiter->find_symbol(label);

		if (Orbiter->get_object_type(idx) == t_vector) {

			vector_builder *VB;

			VB = (vector_builder *) Orbiter->get_object(idx);

			sz = VB->len;
			v = NEW_lint(sz);
			Lint_vec_copy(VB->v, v, sz);
		}
		else if (Orbiter->get_object_type(idx) == t_set) {

			set_builder *SB;

			SB = (set_builder *) Orbiter->get_object(idx);

			sz = SB->sz;
			v = NEW_lint(sz);
			Lint_vec_copy(SB->set, v, sz);
			//Lint_vec_copy_to_int(SB->set, v, sz);
		}
	}
	else {

		Lint_vec_scan(label, v, sz);
	}

	if (f_v) {
		cout << "orbiter_session::get_vector_from_label done" << endl;
	}
}

void orbiter_session::get_int_vector_from_label(
		std::string &label,
		int *&v, int &sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_session::get_int_vector_from_label" << endl;
	}

	long int *vl;

	if (f_v) {
		cout << "orbiter_session::get_int_vector_from_label "
				"before get_lint_vector_from_label" << endl;
	}
	get_lint_vector_from_label(
			label,
			vl, sz,
			verbose_level);
	if (f_v) {
		cout << "orbiter_session::get_int_vector_from_label "
				"after get_lint_vector_from_label" << endl;
	}

	v = NEW_int(sz);

	Lint_vec_copy_to_int(vl, v, sz);

	FREE_lint(vl);


	if (f_v) {
		cout << "orbiter_session::get_int_vector_from_label done" << endl;
	}
}


void orbiter_session::get_lint_vector_from_label(
		std::string &label,
		long int *&v, int &sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_session::get_lint_vector_from_label" << endl;
	}
	if (f_v) {
		cout << "orbiter_session::get_lint_vector_from_label label = " << label << endl;
	}
	if (isalpha(label[0])) {
		if (f_v) {
			cout << "orbiter_session::get_lint_vector_from_label "
					"searching label " << label << endl;
		}
		int idx;

		idx = Orbiter->find_symbol(label);

		if (Orbiter->get_object_type(idx) == t_vector) {

			vector_builder *VB;

			VB = (vector_builder *) Orbiter->get_object(idx);

			sz = VB->len;
			v = NEW_lint(sz);
			Lint_vec_copy(VB->v, v, sz);
		}
		else if (Orbiter->get_object_type(idx) == t_set) {

			set_builder *SB;

			SB = (set_builder *) Orbiter->get_object(idx);

			sz = SB->sz;
			v = NEW_lint(sz);
			Lint_vec_copy(SB->set, v, sz);
		}
	}
	else {
		if (f_v) {
			cout << "orbiter_session::get_lint_vector_from_label "
					"scanning equation" << endl;
		}

		Lint_vec_scan(label, v, sz);
	}
	if (f_v) {
		cout << "orbiter_session::get_lint_vector_from_label found a vector of size " << sz << endl;
	}

	if (f_v) {
		cout << "orbiter_session::get_lint_vector_from_label done" << endl;
	}
}

void orbiter_session::get_matrix_from_label(
		std::string &label,
		int *&v, int &m, int &n)
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_session::get_matrix_from_label" << endl;
	}
	if (isalpha(label[0])) {
		if (f_v) {
			cout << "orbiter_session::get_matrix_from_label "
					"searching label " << label << endl;
		}
		int idx;

		idx = Orbiter->find_symbol(label);

		if (Orbiter->get_object_type(idx) == t_vector) {

			vector_builder *VB;

			VB = (vector_builder *) Orbiter->get_object(idx);

			int sz;

			sz = VB->len;
			v = NEW_int(sz);
			Lint_vec_copy_to_int(VB->v, v, sz);

			if (!VB->f_has_k) {
				cout << "orbiter_session::get_matrix_from_label "
						"the vector does not have matrix formatting information" << endl;
				exit(1);
			}
			m = VB->k;
			n = (VB->len + m - 1) / m;
		}
		else if (Orbiter->get_object_type(idx) == t_set) {
			cout << "orbiter_session::get_matrix_from_label "
						"the object must be of type vector" << endl;
			exit(1);
		}
	}
	else {

		cout << "orbiter_session::get_matrix_from_label "
					"an object label must be given, starting with a letter" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbiter_session::get_matrix_from_label done" << endl;
	}
}


void orbiter_session::find_symbols(
		std::vector<std::string> &Labels, int *&Idx)
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

void orbiter_session::add_symbol_table_entry(
		std::string &label,
		orbiter_symbol_table_entry *Symb, int verbose_level)
{
	Orbiter_symbol_table->add_symbol_table_entry(label, Symb, verbose_level);
}

void orbiter_session::get_lint_vec(
		std::string &label,
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
		Lint_vec_copy(SB->set, the_set, set_size);

		if (f_v) {
			cout << "orbiter_session::get_lint_vec" << endl;
			cout << "set : ";
			Lint_vec_print(cout, the_set, set_size);
			cout << endl;
		}

	}
	else {
		Lint_vec_scan(label, the_set, set_size);
	}
	if (f_v) {
		cout << "orbiter_session::get_lint_vec done" << endl;
	}

}

void orbiter_session::print_type(
		symbol_table_object_type t)
{
	Orbiter_symbol_table->print_type(t);
}

algebra::field_theory::finite_field *orbiter_session::get_object_of_type_finite_field(
		std::string &label)
{
	int idx;

	idx = Orbiter_symbol_table->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_object_of_type_finite_field cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_finite_field) {
		cout << "orbiter_session::get_object_of_type_finite_field object type != t_finite_field" << endl;
		exit(1);
	}
	return (algebra::field_theory::finite_field *) get_object(idx);


}

algebra::ring_theory::homogeneous_polynomial_domain
	*orbiter_session::get_object_of_type_polynomial_ring(
		std::string &label)
{
	int idx;

	idx = Orbiter_symbol_table->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_object_of_type_polynomial_ring cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_polynomial_ring) {
		cout << "orbiter_session::get_object_of_type_polynomial_ring object type != t_polynomial_ring" << endl;
		exit(1);
	}
	return (algebra::ring_theory::homogeneous_polynomial_domain *) get_object(idx);

}



vector_builder *orbiter_session::get_object_of_type_vector(
		std::string &label)
{
	int idx;

	idx = Orbiter_symbol_table->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_object_of_type_vector cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_vector) {
		cout << "orbiter_session::get_object_of_type_vector object type != t_vector" << endl;
		exit(1);
	}
	return (vector_builder *) get_object(idx);


}

int orbiter_session::is_text_available(
		std::string &label)
{
	int idx;

	idx = Orbiter_symbol_table->find_symbol(label);
	if (idx == -1) {
		//cout << "orbiter_session::is_text_available cannot find symbol " << label << endl;
		return false;
	}
	if (get_object_type(idx) != t_text) {
		//cout << "orbiter_session::get_text object type != t_text" << endl;
		return false;
	}

	text_builder *TB;

	TB = (text_builder *) get_object(idx);
	if (!TB->f_has_text) {
		cout << "orbiter_session::get_text text is not available" << endl;
		exit(1);
	}
	return true;


}

std::string orbiter_session::get_text(
		std::string &label)
{
	int idx;

	idx = Orbiter_symbol_table->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_text cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_text) {
		cout << "orbiter_session::get_text object type != t_text" << endl;
		exit(1);
	}

	text_builder *TB;

	TB = (text_builder *) get_object(idx);
	if (!TB->f_has_text) {
		cout << "orbiter_session::get_text text is not available" << endl;
		exit(1);
	}
	return TB->text;


}

std::string orbiter_session::get_string(
		std::string &label)
{
	int idx;

	idx = Orbiter_symbol_table->find_symbol(label);
	if (idx == -1) {
		return label;
	}
	if (get_object_type(idx) != t_text) {
		cout << "orbiter_session::get_string object type != t_text" << endl;
		exit(1);
	}

	text_builder *TB;

	TB = (text_builder *) get_object(idx);
	if (!TB->f_has_text) {
		cout << "orbiter_session::get_string text is not available" << endl;
		exit(1);
	}
	return TB->text;


}


algebra::expression_parser::symbolic_object_builder
	*orbiter_session::get_object_of_type_symbolic_object(
		std::string &label)
{
	int idx;

	idx = Orbiter_symbol_table->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_object_of_type_symbolic_object "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_symbolic_object) {
		cout << "orbiter_session::get_object_of_type_symbolic_object "
				"object type != t_symbolic_object" << endl;
		exit(1);
	}
	return (algebra::expression_parser::symbolic_object_builder *) get_object(idx);


}


combinatorics::coding_theory::crc_object *orbiter_session::get_object_of_type_crc_code(
		std::string &label)
{
	int idx;

	idx = Orbiter_symbol_table->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_object_of_type_crc_code cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_crc_code) {
		cout << "orbiter_session::get_object_of_type_crc_code object type != t_crc_code" << endl;
		exit(1);
	}
	return (combinatorics::coding_theory::crc_object *) get_object(idx);


}


geometry::projective_geometry::projective_space *orbiter_session::get_projective_space_low_level(
		std::string &label)
{
	int idx;

	idx = Orbiter_symbol_table->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_projective_space_low_level cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_projective_space) {
		cout << "orbiter_session::get_projective_space_low_level object type != t_projective_space" << endl;
		exit(1);
	}

	if (!f_has_get_projective_space_low_level_function) {
		cout << "orbiter_session::get_projective_space_low_level !f_has_get_projective_space_low_level_function" << endl;
		exit(1);
	}
	return (* get_projective_space_low_level_function)(get_object(idx));

}

combinatorics::geometry_builder::geometry_builder *orbiter_session::get_geometry_builder(
		std::string &label)
{
	int idx;

	idx = Orbiter_symbol_table->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_geometry_builder cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_geometry_builder) {
		cout << "orbiter_session::get_geometry_builder object type != t_geometry_builder" << endl;
		exit(1);
	}

	return (combinatorics::geometry_builder::geometry_builder *) get_object(idx);

}




int orbiter_session::find_object_of_type_symbolic_object(
		std::string &label)
{
	int idx;

	idx = Orbiter_symbol_table->find_symbol(label);
	if (idx == -1) {
		return -1;
	}
	if (get_object_type(idx) != t_symbolic_object) {
		return -1;
	}
	return idx;


}


combinatorics::graph_theory::colored_graph
	*orbiter_session::get_object_of_type_graph(
			std::string &label)
{
	int idx;

	idx = find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_object_of_type_graph "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_graph) {
		cout << "orbiter_session::get_object_of_type_graph "
				"object type != t_graph" << endl;
		exit(1);
	}


	return (combinatorics::graph_theory::colored_graph *) get_object(idx);
}

combinatorics::design_theory::design_object
	*orbiter_session::get_object_of_type_design(
			std::string &label)
{
	int idx;

	idx = find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_object_of_type_design "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_design) {
		cout << "orbiter_session::get_object_of_type_design "
				"object type != t_design" << endl;
		exit(1);
	}


	return (combinatorics::design_theory::design_object *) get_object(idx);
}


graphics::layered_graph_draw_options
	*orbiter_session::get_draw_options(
			std::string &label)
{
	int idx;

	idx = find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_draw_options "
				"cannot find symbol " << label << endl;
		cout << "symbol table:" << endl;
		print_symbol_table();
		exit(1);
	}
	if (get_object_type(idx) != t_draw_options) {
		cout << "orbiter_session::get_draw_options "
				"object type != t_draw_options" << endl;
		exit(1);
	}


	return (graphics::layered_graph_draw_options *) get_object(idx);
}

graphics::draw_incidence_structure_description
	*orbiter_session::get_draw_incidence_structure_options(
			std::string &label)
{
	int idx;

	idx = find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_draw_incidence_structure_options "
				"cannot find symbol " << label << endl;
		cout << "symbol table:" << endl;
		print_symbol_table();
		exit(1);
	}
	if (get_object_type(idx) != t_draw_incidence_structure_options) {
		cout << "orbiter_session::get_draw_incidence_structure_options "
				"object type != t_draw_incidence_structure_options" << endl;
		exit(1);
	}


	return (graphics::draw_incidence_structure_description *) get_object(idx);
}

geometry::other_geometry::geometric_object_create
	*orbiter_session::get_geometric_object(
			std::string &label)
{
	int idx;

	idx = find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_geometric_object "
				"cannot find symbol " << label << endl;
		cout << "symbol table:" << endl;
		print_symbol_table();
		exit(1);
	}
	if (get_object_type(idx) != t_geometric_object) {
		cout << "orbiter_session::get_geometric_object "
				"object type != t_geometric_object" << endl;
		exit(1);
	}


	return (geometry::other_geometry::geometric_object_create *) get_object(idx);
}




void *orbiter_session::get_isomorph_arguments_opaque(
		std::string &label)
{
	int idx;

	idx = find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_isomorph_arguments_opaque "
				"cannot find symbol " << label << endl;
		cout << "symbol table:" << endl;
		print_symbol_table();
		exit(1);
	}
	if (get_object_type(idx) != t_isomorph_arguments) {
		cout << "orbiter_session::get_isomorph_arguments_opaque "
				"object type != t_isomorph_arguments" << endl;
		exit(1);
	}


	return get_object(idx);
}


void *orbiter_session::get_classify_cubic_surfaces_opaque(
		std::string &label)
{
	int idx;

	idx = find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_classify_cubic_surfaces_opaque "
				"cannot find symbol " << label << endl;
		cout << "symbol table:" << endl;
		print_symbol_table();
		exit(1);
	}
	if (get_object_type(idx) != t_classify_cubic_surfaces) {
		cout << "orbiter_session::get_classify_cubic_surfaces_opaque "
				"object type != t_classify_cubic_surfaces" << endl;
		exit(1);
	}


	return get_object(idx);
}




void
	*orbiter_session::get_any_group_opaque(
			std::string &label)
// the return type should be groups::any_group *
{
	int idx;

	idx = find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_session::get_any_group_opaque "
				"cannot find symbol " << label << endl;
		cout << "symbol table:" << endl;
		print_symbol_table();
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::other::orbiter_kernel_system::symbol_table_object_type::t_any_group) {
		cout << "orbiter_session::get_any_group_opaque "
				"object type != t_any_group" << endl;
		exit(1);
	}
	return (void *) get_object(idx);
}


void orbiter_session::start_memory_debug()
{
	f_memory_debug = true;
	cout << "memory debugging started" << endl;
}

void orbiter_session::stop_memory_debug()
{
	f_memory_debug = false;
	cout << "memory debugging stopped" << endl;
}

void orbiter_session::do_export(
		void *ptr, int verbose_level)
{
	export_import_stack.push_back(ptr);
}

void *orbiter_session::do_import(
		int verbose_level)
{
	if (export_import_stack.size() == 0) {
		cout << "orbiter_session::do_import export/import stack is empty" << endl;
		exit(1);
	}
	void *ptr;

	ptr = export_import_stack[export_import_stack.size() - 1];
	export_import_stack.pop_back();
	return ptr;
}

void orbiter_session::record_birth(
		const char *func_name)
{
	string s;

	s.assign(func_name);
	Births[s]++;
	//cout << "We are in function " << func_name << " for birth" << endl;
}

void orbiter_session::record_death(
		const char *func_name)
{
	string s;

	s.assign(func_name);
	Deaths[s]++;
	//cout << "We are in function " << func_name << " for death" << endl;
}

void orbiter_session::do_statistics()
{
	print_births();
}

void orbiter_session::print_births()
{
	int verbose_level = 0;

	int f_v (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_session::print_births" << endl;
	}
	map<string, int>::iterator iter;
	int nb_births;
	int nb_deaths;

	nb_births = 0;
	for (iter = Births.begin(); iter != Births.end(); iter++) {
		nb_births++;
	}

	nb_deaths = 0;
	for (iter = Deaths.begin(); iter != Deaths.end(); iter++) {
		nb_deaths++;
	}

	if (f_v) {
		int cnt;
		cout << "Births:" << endl;
		cnt = 0;
		for (iter = Births.begin(); iter != Births.end(); iter++, cnt++) {
			cout << cnt << ":" << iter->first << "," << iter->second << endl;
		}

		cout << "Deaths:" << endl;
		cnt = 0;
		for (iter = Deaths.begin(); iter != Deaths.end(); iter++, cnt++) {
			cout << cnt << " : ";
			cout << iter->first << "," << iter->second << endl;
		}
	}
	int_matrix *Births_sorted;
	vector<string> Births_string;
	vector<string> Deaths_string;
	int *Mtx;
	int *Mtx_deaths;
	int cnt;

	Mtx = NEW_int(nb_births * 3);
	Mtx_deaths = NEW_int(nb_deaths * 3);

	cnt = 0;
	for (iter = Births.begin(); iter != Births.end(); iter++, cnt++) {
		Mtx[3 * cnt + 0] = iter->second;
		Mtx[3 * cnt + 1] = cnt;
		Mtx[3 * cnt + 2] = -1;
		Births_string.push_back(iter->first);
	}

	cnt = 0;
	for (iter = Deaths.begin(); iter != Deaths.end(); iter++, cnt++) {
		Mtx_deaths[3 * cnt + 0] = iter->second;
		Mtx_deaths[3 * cnt + 1] = cnt;
		Mtx_deaths[3 * cnt + 2] = -1;
		Deaths_string.push_back(iter->first);
	}

	int i;
	long int total_births = 0;
	long int total_deaths = 0;

	for (i = 0; i < nb_births; i++) {
		total_births += Mtx[3 * i + 0];
	}

	for (i = 0; i < nb_deaths; i++) {
		total_deaths += Mtx_deaths[3 * i + 0];
	}

	for (i = 0; i < nb_deaths; i++) {
		string s;

		s = Deaths_string[i].substr(1, Deaths_string[i].length() - 1);
		Deaths_string[i] = s;
	}

	cout << "total_births = " << total_births << endl;
	cout << "total_deaths = " << total_deaths << endl;
	cout << "difference = " << total_births - total_deaths << endl;



	Births_sorted = NEW_OBJECT(int_matrix);
	Births_sorted->allocate_and_init(
			nb_births, 3, Mtx);

	if (f_v) {
		int w = 5;
		int j;

		cout << "Births_sorted before sorting:" << endl;
		//Births_sorted->print();

		for (i = 0; i < Births_sorted->m; i++) {
			for (j = 0; j < Births_sorted->n; j++) {
				cout << setw((int) w) << Births_sorted->M[i * Births_sorted->n + j];
				if (w) {
					cout << " ";
				}
			}
			cout << Births_string[i];
			cout << endl;
		}

	}

	Births_sorted->sort_rows(verbose_level);

	if (f_v) {
		cout << "Births_sorted after sorting:" << endl;
		Births_sorted->print();
	}

	//int nb_b;
	//int idx;
	int j, a, b, c;


	for (i = 0; i < nb_deaths; i++) {

		string s;

		s = Deaths_string[i];

		for (j = 0; j < nb_births; j++) {
			if (Births_string[j] == s) {
				Mtx_deaths[3 * i + 2] = j;
				Mtx[3 * j + 2] = i;
				break;
			}
		}
	}

#if 0
	string s;

	s = "algorithms";
	nb_b = Births[s];

	cout << "test: " << s << " nb_b = " << nb_b << endl;


	for (i = 0; i < nb_deaths; i++) {


		nb_b = Births[Deaths_string[i]];
		// nb_b may have changed by now!

		if (f_v) {
			cout << "checking on deaths of class " << Deaths_string[i] << " nb_b = " << nb_b << endl;
		}

		if (!Births_sorted->search_first_column_only(
				nb_b, idx, 0 /*verbose_level */)) {
			cout << "We have a death without a recorded birth! class name = " << Deaths_string[i]
				<< " !Births_sorted->search_first_column_only" << endl;
			cout << "nb_b=" << nb_b << endl;
			exit(1);
		}

		int h;

		for (h = idx; h >= 0; h--) {
			a = Births_sorted->M[h * 3 + 0];
			b = Births_sorted->M[h * 3 + 1];
			if (a != nb_b) {
				cout << "We have a death without a recorded birth! class name = " << Deaths_string[i] << endl;
				exit(1);
				break;
			}
			if (Births_string[b] == Deaths_string[i]) {
				Mtx_deaths[3 * cnt + 2] = b;
				Mtx[3 * b + 2] = i;
				break;
			}
		}


	}

#endif

	//int idx;

	int nb_d;

	cout << "Births sorted (top 20 only):" << endl;
	for (j = nb_births - 1; j >= 0; j--) {
		a = Births_sorted->M[j * 3 + 0];
		b = Births_sorted->M[j * 3 + 1];
		c = Mtx[3 * b + 2];
		if (c >= 0) {
			nb_d = Mtx_deaths[3 * c + 0];
		}
		else {
			nb_d = 0;
		}
		//idx = Births_sorted->M[j * 2 + 1];
		cout << Births_string[b] << "," << a << "," << nb_d << endl;
		if (j < nb_births - 20) {
			break;
		}
	}

	cout << "objects that are not freed:" << endl;
	for (j = nb_births - 1; j >= 0; j--) {
		a = Births_sorted->M[j * 3 + 0];
		b = Births_sorted->M[j * 3 + 1];
		c = Mtx[3 * b + 2];
		if (c >= 0) {
			nb_d = Mtx_deaths[3 * c + 0];
		}
		else {
			nb_d = 0;
		}
		//idx = Births_sorted->M[j * 2 + 1];
		if (nb_d == a) {
			continue;
		}
		cout << Births_string[b] << "," << a << "," << nb_d << endl;
	}

	if (f_v) {
		cout << "orbiter_session::print_births done" << endl;
	}
}


}}}}


