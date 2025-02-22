// exact_cover_arguments.cpp
//
// Anton Betten
// January 12, 2016

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace solvers_package {


exact_cover_arguments::exact_cover_arguments()
{
	Record_birth();
	null();
}

exact_cover_arguments::~exact_cover_arguments()
{
	Record_death();
	freeself();
}

void exact_cover_arguments::null()
{
	f_lift = false;
	f_has_base_fname = false;
	//base_fname = "";
	f_has_input_prefix = false;
	//input_prefix = "";
	f_has_output_prefix = false;
	//output_prefix = "";
	f_has_solution_prefix = false;
	//solution_prefix = "";
	f_lift = false;
	f_starter_size = false;
	starter_size = 0;
	f_lex = false;
	f_split = false;
	split_r = 0;
	split_m = 1;
	f_solve = false;
	f_save = false;
	f_read = false;
	f_draw_system = false;
	//std::string draw_options;
	//fname_system = NULL;
	f_write_tree = false;
	//fname_tree = NULL;
	f_has_solution_test_function = false;
	f_has_late_cleanup_function = false;
	prepare_function_new = NULL;
	early_test_function = NULL;
	early_test_function_data = NULL;
	solution_test_func = NULL;
	solution_test_func_data = NULL;
	late_cleanup_function = NULL;
	f_randomized = false;
	//random_permutation_fname = NULL;
}

void exact_cover_arguments::freeself()
{
	null();
}

int exact_cover_arguments::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int i;
	other::data_structures::string_tools ST;

	for (i = 1; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-starter_size") == 0) {
			f_starter_size = true;
			starter_size = ST.strtoi(argv[++i]);
			cout << "-starter_size " << starter_size << endl;
		}
		else if (ST.stringcmp(argv[i], "-lift") == 0) {
			f_lift = true;
			//lift_prefix = argv[++i]; 
			cout << "-lift " << endl;
		}
		else if (ST.stringcmp(argv[i], "-lex") == 0) {
			f_lex = true;
			cout << "-lex" << endl;
		}
		else if (ST.stringcmp(argv[i], "-solve") == 0) {
			f_solve = true;
			cout << "-solve" << endl;
		}
		else if (ST.stringcmp(argv[i], "-save") == 0) {
			f_save = true;
			cout << "-save" << endl;
		}
		else if (ST.stringcmp(argv[i], "-read") == 0) {
			f_read = true;
			cout << "-read" << endl;
		}
		else if (ST.stringcmp(argv[i], "-split") == 0) {
			f_split = true;
			split_r = ST.strtoi(argv[++i]);
			split_m = ST.strtoi(argv[++i]);
			cout << "-split " << split_r << " " << split_m << endl;
		}
		else if (ST.stringcmp(argv[i], "-draw_system") == 0) {
			f_draw_system = true;
			fname_system.assign(argv[++i]);
			cout << "-draw_system " << fname_system << endl;
		}
		else if (ST.stringcmp(argv[i], "-write_tree") == 0) {
			f_write_tree = true;
			fname_tree.assign(argv[++i]);
			cout << "-write_tree " << fname_tree << endl;
		}
		else if (ST.stringcmp(argv[i], "-base_fname") == 0) {
			f_has_base_fname = true;
			base_fname = argv[++i];
			cout << "-base_fname " << base_fname << endl;
		}
		else if (ST.stringcmp(argv[i], "-input_prefix") == 0) {
			f_has_input_prefix = true;
			input_prefix.assign(argv[++i]);
			cout << "-input_prefix " << input_prefix << endl;
		}
		else if (ST.stringcmp(argv[i], "-output_prefix") == 0) {
			f_has_output_prefix = true;
			output_prefix.assign(argv[++i]);
			cout << "-output_prefix " << output_prefix << endl;
		}
		else if (ST.stringcmp(argv[i], "-solution_prefix") == 0) {
			f_has_solution_prefix = true;
			solution_prefix.assign(argv[++i]);
			cout << "-solution_prefix " << solution_prefix << endl;
		}
		else if (ST.stringcmp(argv[i], "-randomized") == 0) {
			f_randomized = true;
			random_permutation_fname.assign(argv[++i]);
			cout << "-randomized " << random_permutation_fname << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "exact_cover_arguments::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "exact_cover_arguments::read_arguments done" << endl;
	return i + 1;
}

void exact_cover_arguments::compute_lifts(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "exact_cover_arguments::compute_lifts" << endl;
		cout << "exact_cover_arguments::compute_lifts verbose_level=" << verbose_level << endl;
		cout << "exact_cover_arguments::compute_lifts base_fname=" << base_fname << endl;
		cout << "exact_cover_arguments::compute_lifts input_prefix=" << input_prefix << endl;
		cout << "exact_cover_arguments::compute_lifts output_prefix=" << output_prefix << endl;
		cout << "exact_cover_arguments::compute_lifts solution_prefix=" << solution_prefix << endl;
	}

	if (!f_has_base_fname) {
		cout << "exact_cover_arguments::compute_lifts no base_fname" << endl;
		exit(1);
	}
	if (!f_has_input_prefix) {
		cout << "exact_cover_arguments::compute_lifts no input_prefix" << endl;
		exit(1);
	}
	if (!f_has_output_prefix) {
		cout << "exact_cover_arguments::compute_lifts no output_prefix" << endl;
		exit(1);
	}
	if (!f_has_solution_prefix) {
		cout << "exact_cover_arguments::compute_lifts no solution_prefix" << endl;
		exit(1);
	}
	if (!f_starter_size) {
		cout << "exact_cover_arguments::compute_lifts no starter_size" << endl;
		exit(1);
	}

	if (target_size == 0) {
		cout << "exact_cover_arguments::compute_lifts target_size == 0" << endl;
		exit(1);
	}

	exact_cover *E;

	E = NEW_OBJECT(exact_cover);

 
	E->init_basic(user_data, 
		A, A2, 
		target_size, starter_size, 
		input_prefix, output_prefix, solution_prefix, base_fname, 
		f_lex, 
		verbose_level - 1);

	E->init_early_test_func(
		early_test_function, early_test_function_data,
		verbose_level);

	E->init_prepare_function_new(
		prepare_function_new, 
		verbose_level);

	if (f_split) {
		E->set_split(split_r, split_m, verbose_level - 1);
	}

	if (f_has_solution_test_function) {
		E->add_solution_test_function(
			solution_test_func, 
			(void *) solution_test_func_data,
			verbose_level - 1);
	}

	if (f_has_late_cleanup_function) {
		E->add_late_cleanup_function(late_cleanup_function);
	}

	if (f_randomized) {
		E->randomize(random_permutation_fname, verbose_level);
	}
	
	if (f_v) {
		cout << "exact_cover_arguments::compute_lifts "
				"before compute_liftings_new" << endl;
	}

	E->compute_liftings_new(
			f_solve, f_save, f_read,
		f_draw_system, draw_options, fname_system,
		f_write_tree, fname_tree,
		verbose_level);

	FREE_OBJECT(E);
	
	if (f_v) {
		cout << "exact_cover_arguments::compute_lifts done" << endl;
	}
	
}

}}}



