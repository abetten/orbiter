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
	cout << "orbiter_top_level_session::orbiter_top_level_session "
			"before new orbiter_session" << endl;

	Orbiter_session = new orbiter_kernel_system::orbiter_session;

	cout << "orbiter_top_level_session::orbiter_top_level_session "
			"after new orbiter_session" << endl;

	The_Orbiter_top_level_session = this;

	The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->f_has_free_entry_callback = true;
	The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->free_entry_callback = free_symbol_table_entry_callback;

	The_Orbiter_top_level_session->Orbiter_session->f_has_get_projective_space_low_level_function = true;
	The_Orbiter_top_level_session->Orbiter_session->get_projective_space_low_level_function = get_projective_space_low_level_function;

	//Orbiter_session = NULL;
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
			cout << "orbiter_top_level_session::~orbiter_top_level_session "
					"before delete Orbiter_session" << endl;
		}
		delete Orbiter_session;
		if (f_v) {
			cout << "orbiter_top_level_session::~orbiter_top_level_session "
					"after delete Orbiter_session" << endl;
		}
	}
	if (f_v) {
		cout << "orbiter_top_level_session::~orbiter_top_level_session done" << endl;
	}
}

void orbiter_top_level_session::execute_command_line(
		int argc, const char **argv, int verbose_level)
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
	data_structures::string_tools ST;
	int i;

	//cout << "before ST.convert_arguments, argc=" << argc << endl;

	ST.convert_arguments(argc, argv, Argv);
		// argc has changed!

	//cout << "after ST.convert_arguments, argc=" << argc << endl;

	//cout << "before Top_level_session.startup_and_read_arguments" << endl;
	i = startup_and_read_arguments(
			argc, Argv, 1, verbose_level - 1);
	//cout << "after Top_level_session.startup_and_read_arguments" << endl;



	int session_verbose_level;

	session_verbose_level = Orbiter_session->verbose_level;

	if (f_v) {
		cout << "session_verbose_level = " << session_verbose_level << endl;
	}


	//int f_v = (verbose_level > 1);

	if (f_v) {
		cout << "orbiter_top_level_session::execute_command_line "
				"before handle_everything" << endl;
		//cout << "argc=" << argc << endl;
	}


	handle_everything(
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

int orbiter_top_level_session::startup_and_read_arguments(
		int argc,
		std::string *argv, int i0, int verbose_level)
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

	i = Orbiter_session->read_arguments(argc, argv, i0, verbose_level);


	if (f_v) {
		cout << "orbiter_top_level_session::startup_and_read_arguments done" << endl;
	}
	return i;
}

void orbiter_top_level_session::handle_everything(
		int argc, std::string *Argv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (false) {
		cout << "orbiter_top_level_session::handle_everything" << endl;
	}
	if (Orbiter_session->f_list_arguments) {
		int j;

		cout << "argument list:" << endl;
		for (j = 0; j < argc; j++) {
			cout << j << " : " << Argv[j] << endl;
		}
	}


	if (Orbiter_session->f_fork) {
		if (f_v) {
			cout << "before Orbiter_session->fork" << endl;
		}
		Orbiter_session->fork(argc, Argv, verbose_level);
		if (f_v) {
			cout << "after Orbiter_session->fork" << endl;
		}
	}
	else {
		if (Orbiter_session->f_seed) {
			orbiter_kernel_system::os_interface Os;

			if (f_v) {
				cout << "seeding random number generator with "
						<< Orbiter_session->the_seed << endl;
			}
			srand(Orbiter_session->the_seed);
			Os.random_integer(1000);
		}
		if (Orbiter_session->f_memory_debug) {
			orbiter_kernel_system::Orbiter->f_memory_debug = true;
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

			string fname;

			fname = "orbiter_memory_dump.cvs";

			orbiter_kernel_system::Orbiter->global_mem_object_registry->dump();
			orbiter_kernel_system::Orbiter->global_mem_object_registry->dump_to_csv_file(fname);
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

void orbiter_top_level_session::parse_and_execute(
		int argc, std::string *Argv, int i, int verbose_level)
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

void orbiter_top_level_session::parse(
		int argc, std::string *Argv,
		int &i, std::vector<void *> &program, int verbose_level)
{
	int cnt = 0;
	int f_v = (verbose_level >= 1);
	int f_vv = false;
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
					"before Interface_symbol_table, i = " << i << endl;
		}

		orbiter_command *OC;

		OC = NEW_OBJECT(orbiter_command);
		if (f_vv) {
			cout << "orbiter_top_level_session::parse "
					"before OC->parse" << endl;
		}
		OC->parse(this, argc, Argv, i, verbose_level);
		if (f_vv) {
			cout << "orbiter_top_level_session::parse "
					"after OC->parse" << endl;
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

void *orbiter_top_level_session::get_object(
		int idx)
{
	return Orbiter_session->get_object(idx);
}

layer1_foundations::orbiter_kernel_system::symbol_table_object_type
	orbiter_top_level_session::get_object_type(
		int idx)
{
	return Orbiter_session->get_object_type(idx);
}

int orbiter_top_level_session::find_symbol(
		std::string &label)
{
	return Orbiter_session->find_symbol(label);
}

void orbiter_top_level_session::find_symbols(
		std::vector<std::string> &Labels, int *&Idx)
{

	Orbiter_session->find_symbols(Labels, Idx);
}

void orbiter_top_level_session::print_symbol_table()
{
	Orbiter_session->print_symbol_table();
}

void orbiter_top_level_session::add_symbol_table_entry(
		std::string &label,
		orbiter_kernel_system::orbiter_symbol_table_entry *Symb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_top_level_session::add_symbol_table_entry "
				"label = " << label << endl;
	}
	Orbiter_session->add_symbol_table_entry(label, Symb, verbose_level);
	if (f_v) {
		cout << "orbiter_top_level_session::add_symbol_table_entry done" << endl;
	}
}


groups::any_group
	*orbiter_top_level_session::get_any_group(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_any_group "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_any_group) {
		cout << "orbiter_top_level_session::get_any_group "
				"object type != t_any_group" << endl;
		exit(1);
	}
	return (groups::any_group *) get_object(idx);
}

projective_geometry::projective_space_with_action
	*orbiter_top_level_session::get_object_of_type_projective_space(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_projective_space "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_projective_space) {
		cout << "orbiter_top_level_session::get_object_of_type_projective_space "
				"object type != t_projective_space" << endl;
		exit(1);
	}
	return (projective_geometry::projective_space_with_action *) get_object(idx);
}

spreads::spread_table_with_selection
	*orbiter_top_level_session::get_object_of_type_spread_table(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_spread_table "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_spread_table) {
		cout << "orbiter_top_level_session::get_object_of_type_spread_table "
				"object type != t_spread_table" << endl;
		exit(1);
	}
	return (spreads::spread_table_with_selection *) get_object(idx);
}


packings::packing_classify
	*orbiter_top_level_session::get_packing_classify(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_packing_classify "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_packing_classify) {
		cout << "orbiter_top_level_session::get_packing_classify "
				"object type != t_packing_classify" << endl;
		exit(1);
	}
	return (packings::packing_classify *) get_object(idx);
}

poset_classification::poset_classification_control
	*orbiter_top_level_session::get_poset_classification_control(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_poset_classification_control "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_poset_classification_control) {
		cout << "orbiter_top_level_session::get_poset_classification_control "
				"object type != t_poset_classification_control" << endl;
		exit(1);
	}
	return (poset_classification::poset_classification_control *) get_object(idx);
}

poset_classification::poset_classification_report_options
	*orbiter_top_level_session::get_poset_classification_report_options(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_poset_classification_report_options "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_poset_classification_report_options) {
		cout << "orbiter_top_level_session::get_poset_classification_report_options "
				"object type != t_poset_classification_report_options" << endl;
		exit(1);
	}
	return (poset_classification::poset_classification_report_options *) get_object(idx);
}


apps_geometry::arc_generator_description
	*orbiter_top_level_session::get_object_of_type_arc_generator_control(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_arc_generator_control "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_arc_generator_control) {
		cout << "orbiter_top_level_session::get_object_of_type_arc_generator_control "
				"object type != t_arc_generator_control" << endl;
		exit(1);
	}
	return (apps_geometry::arc_generator_description *) get_object(idx);
}


poset_classification::poset_classification_activity_description
	*orbiter_top_level_session::get_object_of_type_poset_classification_activity(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_poset_classification_activity "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_poset_classification_activity) {
		cout << "orbiter_top_level_session::get_object_of_type_poset_classification_activity "
				"object type != t_poset_classification_activity" << endl;
		exit(1);
	}
	return (poset_classification::poset_classification_activity_description *) get_object(idx);
}




void orbiter_top_level_session::get_vector_or_set(
		std::string &label,
		long int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_top_level_session::get_vector_or_set" << endl;
	}

	if (isalpha(label[0])) {

		if (f_v) {
			cout << "orbiter_top_level_session::get_vector_or_set "
					"searching label " << label << endl;
		}
		int idx;

		idx = Orbiter_session->find_symbol(label);

		if (Orbiter_session->get_object_type(idx) == layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_vector) {

			if (f_v) {
				cout << "orbiter_top_level_session::get_vector_or_set "
						"found a vector " << label << endl;
			}
			data_structures::vector_builder *VB;

			VB = (data_structures::vector_builder *)
					Orbiter_session->get_object(idx);

			nb_pts = VB->len;
			Pts = NEW_lint(nb_pts);
			Lint_vec_copy(VB->v, Pts, nb_pts);

		}
		else if (Orbiter_session->get_object_type(idx) == layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_set) {

			if (f_v) {
				cout << "orbiter_top_level_session::get_vector_or_set "
						"found a set " << label << endl;
			}
			data_structures::set_builder *SB;

			SB = (data_structures::set_builder *)
					Orbiter_session->get_object(idx);

			nb_pts = SB->sz;
			Pts = NEW_lint(nb_pts);
			Lint_vec_copy(SB->set, Pts, nb_pts);
		}
	}
	else {

		Lint_vec_scan(label, Pts, nb_pts);
	}
	if (f_v) {
		cout << "orbiter_top_level_session::get_vector_or_set "
				"we found a set of size " << nb_pts << endl;
	}

	if (f_v) {
		cout << "orbiter_top_level_session::get_vector_or_set "
				"done" << endl;
	}

}


apps_algebra::vector_ge_builder
	*orbiter_top_level_session::get_object_of_type_vector_ge(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_vector_ge "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_vector_ge) {
		cout << "orbiter_top_level_session::get_object_of_type_vector_ge "
				"object type != t_vector_ge" << endl;
		cout << "object type = ";
		orbiter_kernel_system::Orbiter->print_type(get_object_type(idx));
		exit(1);
	}
	return (apps_algebra::vector_ge_builder *) get_object(idx);
}


orthogonal_geometry_applications::orthogonal_space_with_action
	*orbiter_top_level_session::get_object_of_type_orthogonal_space_with_action(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_orthogonal_space_with_action "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_orthogonal_space) {
		cout << "orbiter_top_level_session::get_object_of_type_orthogonal_space_with_action "
				"object type != t_orthogonal_space" << endl;
		exit(1);
	}


	return (orthogonal_geometry_applications::orthogonal_space_with_action *) get_object(idx);
}

#if 0
field_theory::finite_field
	*orbiter_top_level_session::get_object_of_type_finite_field(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_finite_field "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != t_finite_field) {
		cout << "orbiter_top_level_session::get_object_of_type_finite_field "
				"object type != t_finite_field" << endl;
		exit(1);
	}


	return (field_theory::finite_field *) get_object(idx);
}
#endif

spreads::spread_create
	*orbiter_top_level_session::get_object_of_type_spread(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_spread "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_spread) {
		cout << "orbiter_top_level_session::get_object_of_type_spread "
				"object type != t_spread" << endl;
		exit(1);
	}


	return (spreads::spread_create *) get_object(idx);
}

applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create
	*orbiter_top_level_session::get_object_of_type_cubic_surface(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_cubic_surface "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_cubic_surface) {
		cout << "orbiter_top_level_session::get_object_of_type_cubic_surface "
				"object type != t_cubic_surface" << endl;
		exit(1);
	}


	return (applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *)
			get_object(idx);
}

apps_coding_theory::create_code
	*orbiter_top_level_session::get_object_of_type_code(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_code "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_code) {
		cout << "orbiter_top_level_session::get_object_of_type_code "
				"object type != t_code" << endl;
		exit(1);
	}


	return (apps_coding_theory::create_code *) get_object(idx);
}


graph_theory::colored_graph
	*orbiter_top_level_session::get_object_of_type_graph(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_object_of_type_graph "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_graph) {
		cout << "orbiter_top_level_session::get_object_of_type_graph "
				"object type != t_graph" << endl;
		exit(1);
	}


	return (graph_theory::colored_graph *) get_object(idx);
}




orthogonal_geometry_applications::orthogonal_space_with_action
	*orbiter_top_level_session::get_orthogonal_space(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_orthogonal_space "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_orthogonal_space) {
		cout << "orbiter_top_level_session::get_orthogonal_space "
				"object type != t_orthogonal_space" << endl;
		exit(1);
	}


	return (orthogonal_geometry_applications::orthogonal_space_with_action *)
			get_object(idx);
}


orbits::orbits_create
	*orbiter_top_level_session::get_orbits(
		std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_orbits "
				"cannot find symbol " << label << endl;
		exit(1);
	}
	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_orbits) {
		cout << "orbiter_top_level_session::get_orbits "
				"object type != t_orbits" << endl;
		exit(1);
	}

	return (orbits::orbits_create *)
			get_object(idx);
}


canonical_form::variety_object_with_action
	*orbiter_top_level_session::get_variety(
			std::string &label)
{
	int idx;

	idx = Orbiter_session->find_symbol(label);
	if (idx == -1) {
		cout << "orbiter_top_level_session::get_variety "
				"cannot find symbol " << label << endl;
		exit(1);
	}

	if (get_object_type(idx) != layer1_foundations::orbiter_kernel_system::symbol_table_object_type::t_variety) {
		cout << "orbiter_top_level_session::get_variety "
				"object type != t_variety" << endl;
		exit(1);
	}


	return (canonical_form::variety_object_with_action *)
			get_object(idx);
}




void free_symbol_table_entry_callback(
		orbiter_kernel_system::orbiter_symbol_table_entry *Symb, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "free_symbol_table_entry_callback" << endl;
	}

	orbiter_kernel_system::symbol_table_object_type t;

	t = Symb->object_type;

	string s;

	s = The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->stringify_type(t);
	if (f_v) {
		cout << "free_symbol_table_entry_callback object of type " << s << endl;
	}

	if (t == orbiter_kernel_system::t_nothing_object) {
		if (f_v) {
			cout << "t_nothing_object" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_finite_field) {
		if (f_v) {
			cout << "t_finite_field" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_any_group) {
		if (f_v) {
			cout << "t_any_group" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_linear_group) {
		if (f_v) {
			cout << "t_linear_group" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_permutation_group) {
		if (f_v) {
			cout << "t_permutation_group" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_modified_group) {
		if (f_v) {
			cout << "t_modified_group" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_projective_space) {
		if (f_v) {
			cout << "t_projective_space" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_orthogonal_space) {
		if (f_v) {
			cout << "t_orthogonal_space" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_BLT_set_classify) {
		if (f_v) {
			cout << "t_BLT_set_classify" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_spread_classify) {
		if (f_v) {
			cout << "t_spread_classify" << endl;
		}
	}
#if 0
	else if (t == orbiter_kernel_system::t_formula) {
		if (f_v) {
			cout << "t_formula" << endl;
		}
	}
#endif
	else if (t == orbiter_kernel_system::t_cubic_surface) {
		if (f_v) {
			cout << "t_cubic_surface" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_quartic_curve) {
		if (f_v) {
			cout << "t_quartic_curve" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_BLT_set) {
		if (f_v) {
			cout << "t_BLT_set" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_classification_of_cubic_surfaces_with_double_sixes) {
		if (f_v) {
			cout << "t_classification_of_cubic_surfaces_with_double_sixes" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_collection) {
		if (f_v) {
			cout << "t_collection" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_geometric_object) {
		if (f_v) {
			cout << "t_geometric_object" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_graph) {
		if (f_v) {
			cout << "t_graph" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_code) {
		if (f_v) {
			cout << "t_code" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_spread_table) {
		if (f_v) {
			cout << "t_spread_table" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_packing_classify) {
		if (f_v) {
			cout << "t_packing_classify" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_packing_was) {
		if (f_v) {
			cout << "t_packing_was" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_packing_was_choose_fixed_points) {
		if (f_v) {
			cout << "t_packing_was_choose_fixed_points" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_packing_long_orbits) {
		if (f_v) {
			cout << "t_packing_long_orbits" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_graph_classify) {
		if (f_v) {
			cout << "t_graph_classify" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_diophant) {
		if (f_v) {
			cout << "t_diophant" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_design) {
		if (f_v) {
			cout << "t_design" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_design_table) {
		if (f_v) {
			cout << "t_design_table" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_large_set_was) {
		if (f_v) {
			cout << "t_large_set_was" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_set) {
		if (f_v) {
			cout << "t_set" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_vector) {
		if (f_v) {
			cout << "t_vector" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_symbolic_object) {
		if (f_v) {
			cout << "t_symbolic_object" << endl;
		}

		expression_parser::symbolic_object_builder *SB;

		SB = (expression_parser::symbolic_object_builder *) Symb->ptr;

		FREE_OBJECT(SB);
		Symb->ptr = NULL;
		Symb->object_type = orbiter_kernel_system::t_nothing_object;

		if (f_v) {
			cout << "symbolic_object freed" << endl;
		}

	}
	else if (t == orbiter_kernel_system::t_combinatorial_object) {
		if (f_v) {
			cout << "t_combinatorial_object" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_geometry_builder) {
		if (f_v) {
			cout << "t_geometry_builder" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_vector_ge) {
		if (f_v) {
			cout << "t_vector_ge" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_action_on_forms) {
		if (f_v) {
			cout << "t_action_on_forms" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_orbits) {
		if (f_v) {
			cout << "t_orbits" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_poset_classification_control) {
		if (f_v) {
			cout << "t_poset_classification_control" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_poset_classification_report_options) {
		if (f_v) {
			cout << "t_poset_classification_report_options" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_poset_classification_activity) {
		if (f_v) {
			cout << "t_poset_classification_activity" << endl;
		}
	}
	else if (t == orbiter_kernel_system::t_crc_code) {
		if (f_v) {
			cout << "t_crc_code" << endl;
		}
	}
	else {
		cout << "type is unknown" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "free_symbol_table_entry_callback done" << endl;
	}
}

geometry::projective_geometry::projective_space *get_projective_space_low_level_function(
		void *ptr)
{
	projective_geometry::projective_space_with_action *PA;

	PA = (projective_geometry::projective_space_with_action *) ptr;

	return PA->P;

}


}}}


