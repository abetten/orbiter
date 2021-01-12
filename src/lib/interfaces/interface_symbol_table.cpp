/*
 * interface_symbol_table.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace interfaces {




interface_symbol_table::interface_symbol_table()
{
	f_define = FALSE;
	//define_label
	f_finite_field = FALSE;
	Finite_field_description = NULL;

	f_projective_space = FALSE;
	Projective_space_with_action_description = NULL;

	f_orthogonal_space = FALSE;
	Orthogonal_space_with_action_description = NULL;

	f_linear_group = FALSE;
	Linear_group_description = NULL;

	f_print_symbols = FALSE;
	f_with = FALSE;
	//std::vector<std::string> with_labels;

	f_finite_field_activity = FALSE;
	Finite_field_activity_description = NULL;

	f_projective_space_activity = FALSE;
	Projective_space_activity_description = NULL;

	f_orthogonal_space_activity = FALSE;
	Orthogonal_space_activity_description = NULL;

	f_group_theoretic_activity = FALSE;
	Group_theoretic_activity_description = NULL;
}


void interface_symbol_table::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (stringcmp(argv[i], "-define") == 0) {
		cout << "-define <string : label> description -end" << endl;
	}
	else if (stringcmp(argv[i], "-with") == 0) {
		cout << "-with <string : label> *[ -and <string : label> ] -do ... -end" << endl;
	}
}

int interface_symbol_table::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (i >= argc) {
		return false;
	}
	if (stringcmp(argv[i], "-define") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-with") == 0) {
		return true;
	}
	return false;
}

int interface_symbol_table::read_arguments(int argc,
		std::string *argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_symbol_table::read_arguments" << endl;

	for (i = i0; i < argc; i++) {
		if (stringcmp(argv[i], "-define") == 0) {
			f_define = TRUE;
			define_label.assign(argv[++i]);
			i++;
			cout << "-define " << define_label << endl;
			if (stringcmp(argv[i], "-finite_field") == 0) {
				f_finite_field = TRUE;
				Finite_field_description = NEW_OBJECT(finite_field_description);
				cout << "reading -finite_field" << endl;
				i += Finite_field_description->read_arguments(argc - (i + 1),
					argv + i + 1, verbose_level);

				cout << "-finite_field" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
			else if (stringcmp(argv[i], "-projective_space") == 0) {
				f_projective_space = TRUE;
				Projective_space_with_action_description = NEW_OBJECT(projective_space_with_action_description);
				cout << "reading -projective_space" << endl;
				i += Projective_space_with_action_description->read_arguments(argc - (i + 1),
					argv + i + 1, verbose_level);
				cout << "-projective_space" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
			else if (stringcmp(argv[i], "-orthogonal_space") == 0) {
				f_orthogonal_space = TRUE;
				Orthogonal_space_with_action_description = NEW_OBJECT(orthogonal_space_with_action_description);
				cout << "reading -orthogonal_space" << endl;
				i += Orthogonal_space_with_action_description->read_arguments(argc - (i + 1),
					argv + i + 1, verbose_level);
				cout << "-orthogonal_space" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
			else if (stringcmp(argv[i], "-linear_group") == 0) {
				f_linear_group = TRUE;
				Linear_group_description = NEW_OBJECT(linear_group_description);
				cout << "reading -linear_group" << endl;
				i += Linear_group_description->read_arguments(argc - (i + 1),
					argv + i + 1, verbose_level);

				cout << "-linear_group" << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
			else {
				cout << "unrecognized command after -define" << endl;
				exit(1);
			}
		}
		else if (stringcmp(argv[i], "-print_symbols") == 0) {
			f_print_symbols = TRUE;
			cout << "-print_symbols" << endl;
		}
		else if (stringcmp(argv[i], "-with") == 0) {
			f_with = TRUE;
			string s;

			s.assign(argv[++i]);
			with_labels.push_back(s);

			while (TRUE) {
				i++;
				if (stringcmp(argv[i], "-and") == 0) {
					string s;

					s.assign(argv[++i]);
					with_labels.push_back(s);
				}
				else if (stringcmp(argv[i], "-do") == 0) {
					i++;
					read_activity_arguments(argc, argv, i, verbose_level);
					break;
				}
				else {
					cout << "syntax error after -with, seeing " << argv[i] << endl;
					exit(1);
				}
			}
			cout << "-with ..." << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			i++;
			break;
		}
		else {
			cout << "interface_symbol_table::read_arguments: unrecognized option "
					<< argv[i] << ", skipping" << endl;
			break;
		}
	}
	cout << "interface_symbol_table::read_arguments done" << endl;
	return i;
}

void interface_symbol_table::read_activity_arguments(int argc,
		std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::read_activity_arguments" << endl;
	}
	if (stringcmp(argv[i], "-finite_field_activity") == 0) {
		f_finite_field_activity = TRUE;
		Finite_field_activity_description =
				NEW_OBJECT(finite_field_activity_description);
		cout << "reading -finite_field_activity" << endl;
		i += Finite_field_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		cout << "-finite_field_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		i++;
	}
	else if (stringcmp(argv[i], "-projective_space_activity") == 0) {
		f_projective_space_activity = TRUE;
		Projective_space_activity_description =
				NEW_OBJECT(projective_space_activity_description);
		cout << "reading -projective_space_activity" << endl;
		i += Projective_space_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		cout << "-projective_space_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		i++;
	}
	else if (stringcmp(argv[i], "-orthogonal_space_activity") == 0) {
		f_orthogonal_space_activity = TRUE;
		Orthogonal_space_activity_description =
				NEW_OBJECT(orthogonal_space_activity_description);
		cout << "reading -orthogonal_space_activity" << endl;
		i += Orthogonal_space_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		cout << "-orthogonal_space_activity" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		i++;
	}
	else if (stringcmp(argv[i], "-group_theoretic_activities") == 0) {
		f_group_theoretic_activity = TRUE;
		Group_theoretic_activity_description =
				NEW_OBJECT(group_theoretic_activity_description);
		cout << "reading -group_theoretic_activities" << endl;
		i += Group_theoretic_activity_description->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		cout << "-group_theoretic_activities" << endl;
		cout << "i = " << i << endl;
		cout << "argc = " << argc << endl;
		if (i < argc) {
			cout << "next argument is " << argv[i] << endl;
		}
		i++;
	}
	else {
		cout << "expecting activity after -do but seeing " << argv[i] << endl;
		exit(1);
	}

}

void interface_symbol_table::worker(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::worker" << endl;
	}

	if (f_define) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_define" << endl;
		}
		definition(Orbiter_top_level_session, verbose_level);
	}
	else if (f_print_symbols) {

		Orbiter_top_level_session->print_symbol_table();
	}
	else if (f_finite_field_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_finite_field_activity" << endl;
		}
		do_finite_field_activity(Orbiter_top_level_session, verbose_level);

	}
	else if (f_projective_space_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_projective_space_activity" << endl;
		}
		do_projective_space_activity(Orbiter_top_level_session, verbose_level);

	}
	else if (f_orthogonal_space_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_orthogonal_space_activity" << endl;
		}
		do_orthogonal_space_activity(Orbiter_top_level_session, verbose_level);

	}
	else if (f_group_theoretic_activity) {

		if (f_v) {
			cout << "interface_symbol_table::worker f_group_theoretic_activity" << endl;
		}
		do_group_theoretic_activity(Orbiter_top_level_session, verbose_level);

	}

	if (f_v) {
		cout << "interface_symbol_table::worker done" << endl;
	}
}

void interface_symbol_table::definition(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition" << endl;
	}
	if (f_finite_field) {

		if (f_v) {
			cout << "interface_symbol_table::definition f_finite_field" << endl;
		}
		Finite_field_description->print();
		finite_field *F;

		F = NEW_OBJECT(finite_field);
		F->init(Finite_field_description, verbose_level);

		orbiter_symbol_table_entry Symb;
		Symb.init_finite_field(define_label, F, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::definition before add_symbol_table_entry" << endl;
		}
		Orbiter_top_level_session->add_symbol_table_entry(
				define_label, &Symb, verbose_level);

	}
	else if (f_projective_space) {

		if (f_v) {
			cout << "interface_symbol_table::definition f_projective_space" << endl;
		}

		definition_of_projective_space(Orbiter_top_level_session, verbose_level);


	}
	else if (f_orthogonal_space) {

		if (f_v) {
			cout << "interface_symbol_table::definition f_orthogonal_space" << endl;
		}

		definition_of_orthogonal_space(Orbiter_top_level_session, verbose_level);


	}
	else if (f_linear_group) {

		if (f_v) {
			cout << "interface_symbol_table::definition f_linear_group" << endl;
		}


#if 0
		finite_field *F;
		F = NEW_OBJECT(finite_field);

		if (Linear_group_description->f_override_polynomial) {
			cout << "interface_symbol_table::definition "
					"creating finite field of order q=" << Linear_group_description->input_q
					<< " using override polynomial " << Linear_group_description->override_polynomial << endl;
			F->init_override_polynomial(Linear_group_description->input_q,
					Linear_group_description->override_polynomial, verbose_level - 3);
		}
		else {
			cout << "interface_symbol_table::definition creating finite field "
					"of order q=" << Linear_group_description->input_q
					<< " using the default polynomial (if necessary)" << endl;
			F->finite_field_init(Linear_group_description->input_q, 0);
		}
#else
		finite_field *F;

		if (string_starts_with_a_number(Linear_group_description->input_q)) {
			int q;

			q = strtoi(Linear_group_description->input_q);
			if (f_v) {
				cout << "interface_symbol_table::definition "
						"creating finite field of order " << q << endl;
			}
			F = NEW_OBJECT(finite_field);
			F->finite_field_init(q, 0);
		}
		else {
			if (f_v) {
				cout << "interface_symbol_table::definition "
						"using existing finite field " << Linear_group_description->input_q << endl;
			}
			int idx;
			idx = Orbiter_top_level_session->find_symbol(Linear_group_description->input_q);
			F = (finite_field *) Orbiter_top_level_session->get_object(idx);
		}


#endif

		Linear_group_description->F = F;
		//q = Descr->input_q;

		linear_group *LG;

		LG = NEW_OBJECT(linear_group);
		if (f_v) {
			cout << "interface_symbol_table::definition before LG->init, "
					"creating the group" << endl;
		}

		LG->linear_group_init(Linear_group_description, verbose_level - 5);

		orbiter_symbol_table_entry Symb;
		Symb.init_linear_group(define_label, LG, verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::definition before add_symbol_table_entry" << endl;
		}
		Orbiter_top_level_session->add_symbol_table_entry(
				define_label, &Symb, verbose_level);

	}
	if (f_v) {
		cout << "interface_symbol_table::definition done" << endl;
	}
}


void interface_symbol_table::definition_of_projective_space(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_projective_space" << endl;
	}
	finite_field *F;

	if (string_starts_with_a_number(Projective_space_with_action_description->input_q)) {
		int q;

		q = strtoi(Projective_space_with_action_description->input_q);
		if (f_v) {
			cout << "interface_symbol_table::definition_of_projective_space "
					"creating finite field of order " << q << endl;
		}
		F = NEW_OBJECT(finite_field);
		F->finite_field_init(q, 0);
	}
	else {
		if (f_v) {
			cout << "interface_symbol_table::definition_of_projective_space "
					"using existing finite field " << Projective_space_with_action_description->input_q << endl;
		}
		int idx;
		idx = Orbiter_top_level_session->find_symbol(Projective_space_with_action_description->input_q);
		F = (finite_field *) Orbiter_top_level_session->get_object(idx);
	}

	Projective_space_with_action_description->F = F;

	int f_semilinear;
	number_theory_domain NT;


	if (NT.is_prime(F->q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}

	projective_space_with_action *PA;

	PA = NEW_OBJECT(projective_space_with_action);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_projective_space before PA->init" << endl;
	}
	PA->init(Projective_space_with_action_description->F, Projective_space_with_action_description->n,
		f_semilinear,
		TRUE /*f_init_incidence_structure*/,
		0 /* verbose_level */);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_projective_space after PA->init" << endl;
	}

	orbiter_symbol_table_entry Symb;
	Symb.init_projective_space(define_label, PA, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_projective_space before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_projective_space done" << endl;
	}
}

void interface_symbol_table::definition_of_orthogonal_space(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_orthogonal_space" << endl;
	}
	finite_field *F;

	if (string_starts_with_a_number(Orthogonal_space_with_action_description->input_q)) {
		int q;

		q = strtoi(Orthogonal_space_with_action_description->input_q);
		if (f_v) {
			cout << "interface_symbol_table::definition_of_orthogonal_space "
					"creating finite field of order " << q << endl;
		}
		F = NEW_OBJECT(finite_field);
		F->finite_field_init(q, 0);
	}
	else {
		if (f_v) {
			cout << "interface_symbol_table::definition_of_orthogonal_space "
					"using existing finite field " << Orthogonal_space_with_action_description->input_q << endl;
		}
		int idx;
		idx = Orbiter_top_level_session->find_symbol(Orthogonal_space_with_action_description->input_q);
		F = (finite_field *) Orbiter_top_level_session->get_object(idx);
	}

	Orthogonal_space_with_action_description->F = F;

	int f_semilinear;
	number_theory_domain NT;


	if (NT.is_prime(F->q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}

	orthogonal_space_with_action *OA;

	OA = NEW_OBJECT(orthogonal_space_with_action);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_orthogonal_space before OA->init" << endl;
	}
	OA->init(Orthogonal_space_with_action_description,
		0 /* verbose_level */);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_orthogonal_space after OA->init" << endl;
	}

	orbiter_symbol_table_entry Symb;
	Symb.init_orthogonal_space(define_label, OA, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_orthogonal_space before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_orthogonal_space done" << endl;
	}
}

void interface_symbol_table::do_finite_field_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::do_finite_field_activity "
				"finite field activity for " << with_labels.size() << " objects" << endl;
	}

	int *Idx;


	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-finite_field_activity requires at least one input" << endl;
		exit(1);
	}
	finite_field *F;

	F = (finite_field *) Orbiter_top_level_session->get_object(Idx[0]);

	finite_field_activity FA;
	//FA.init(Finite_field_activity_description, verbose_level);
	Finite_field_activity_description->f_q = TRUE;
	Finite_field_activity_description->q = F->q;
	FA.Descr = Finite_field_activity_description;
	FA.F = F;

	if (with_labels.size() == 2) {
		cout << "-finite_field_activity has two inputs" << endl;
		FA.F_secondary = (finite_field *) Orbiter_top_level_session->get_object(Idx[1]);
	}



	if (f_v) {
		cout << "interface_symbol_table::do_finite_field_activity "
				"before FA.perform_activity" << endl;
	}
	FA.perform_activity(verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::do_finite_field_activity "
				"after FA.perform_activity" << endl;
	}

	FREE_int(Idx);

}

void interface_symbol_table::do_projective_space_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::do_projective_space_activity "
				"projective space activity for " << with_labels.size() << " objects" << endl;
	}

	int *Idx;


	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-finite_field_activity requires at least one input" << endl;
		exit(1);
	}
	projective_space_with_action *PA;

	PA = (projective_space_with_action *) Orbiter_top_level_session->get_object(Idx[0]);

	projective_space_activity Activity;
	Activity.Descr = Projective_space_activity_description;
	Activity.PA = PA;

#if 0
	if (with_labels.size() == 2) {
		cout << "-finite_field_activity has two inputs" << endl;
		FA.F_secondary = (finite_field *) Orbiter_top_level_session->get_object(Idx[1]);
	}
#endif


	if (f_v) {
		cout << "interface_symbol_table::do_projective_space_activity "
				"before Activity.perform_activity" << endl;
	}
	Activity.perform_activity(verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::do_projective_space_activity "
				"after Activity.perform_activity" << endl;
	}

	FREE_int(Idx);

}

void interface_symbol_table::do_orthogonal_space_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::do_orthogonal_space_activity "
				"orthogonal space activity for " << with_labels.size() << " objects" << endl;
	}

	int *Idx;


	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-finite_field_activity requires at least one input" << endl;
		exit(1);
	}
	orthogonal_space_with_action *OA;

	OA = (orthogonal_space_with_action *) Orbiter_top_level_session->get_object(Idx[0]);

	orthogonal_space_activity Activity;
	Activity.Descr = Orthogonal_space_activity_description;
	Activity.OA = OA;

#if 0
	if (with_labels.size() == 2) {
		cout << "-finite_field_activity has two inputs" << endl;
		FA.F_secondary = (finite_field *) Orbiter_top_level_session->get_object(Idx[1]);
	}
#endif


	if (f_v) {
		cout << "interface_symbol_table::do_orthogonal_space_activity "
				"before Activity.perform_activity" << endl;
	}
	Activity.perform_activity(verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::do_orthogonal_space_activity "
				"after Activity.perform_activity" << endl;
	}

	FREE_int(Idx);

}

void interface_symbol_table::do_group_theoretic_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::do_group_theoretic_activity "
				"finite field activity for " << with_labels.size() << " objects" << endl;
	}

	int *Idx;

	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-group_theoretic_activity requires at least one input" << endl;
		exit(1);
	}

	linear_group *LG;

	LG = (linear_group *) Orbiter_top_level_session->get_object(Idx[0]);
	{
		group_theoretic_activity Activity;

		Activity.init(Group_theoretic_activity_description, LG->F, LG, verbose_level);

		if (with_labels.size() == 2) {
			cout << "-group_theoretic_activity has two inputs" << endl;
			linear_group *LG2;
			LG2 = (linear_group *) Orbiter_top_level_session->get_object(Idx[1]);

			Activity.A2 = LG2->A_linear;
		}

		if (f_v) {
			cout << "interface_symbol_table::do_group_theoretic_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::do_group_theoretic_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "interface_symbol_table::do_group_theoretic_activity done" << endl;
	}

}

}}
