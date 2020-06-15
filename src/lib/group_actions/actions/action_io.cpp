/*
 * action_io.cpp
 *
 *  Created on: Mar 30, 2019
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"
#include <cstring>
	// for memcpy

using namespace std;


namespace orbiter {
namespace group_actions {


void action::report(ostream &ost, int f_sims, sims *S,
		int f_strong_gens, strong_generators *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::report" << endl;
	}
	ost << "Group action $" << label_tex
			<< "$ of degree " << degree << "\\\\" << endl;


	if (type_G == wreath_product_t) {
		wreath_product *W;

		W = G.wreath_product_group;
		if (f_v) {
			cout << "action::report before W->report" << endl;
		}
		W->report(ost, verbose_level);
		if (f_v) {
			cout << "action::report after W->report" << endl;
		}
	}

	if (f_sims) {
		if (f_v) {
			cout << "action::report printing group order" << endl;
		}
		longinteger_object go;

		S->group_order(go);
		ost << "Group order " << go << "\\\\" << endl;
		ost << "tl=$";
		//int_vec_print(ost, S->orbit_len, base_len());
		for (int t = 0; t < S->A->base_len(); t++) {
			ost << S->get_orbit_length(t);
			if (t < S->A->base_len()) {
				ost << ", ";
			}
		}
		ost << "$\\\\" << endl;
		if (f_v) {
			cout << "action::report printing group order done" << endl;
		}
	}

	if (Stabilizer_chain) {
		if (base_len()) {
			ost << "Base: $";
			lint_vec_print(ost, get_base(), base_len());
			ost << "$\\\\" << endl;
		}
		if (f_strong_gens) {
			ost << "{\\small\\arraycolsep=2pt" << endl;
			SG->print_generators_tex(ost);
			ost << "}" << endl;
		}
		else {
			ost << "Does not have strong generators.\\\\" << endl;
		}
	}
	if (f_sims) {
		if (f_v) {
			cout << "action::report before S->report" << endl;
		}
		S->report(ost, verbose_level);
		if (f_v) {
			cout << "action::report after S->report" << endl;
		}
	}
	if (f_v) {
		cout << "action::report done" << endl;
	}
}

void action::read_orbit_rep_and_candidates_from_files_and_process(
	const char *prefix,
	int level, int orbit_at_level, int level_of_candidates_file,
	void (*early_test_func_callback)(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level),
	void *early_test_func_callback_data,
	long int *&starter,
	int &starter_sz,
	sims *&Stab,
	strong_generators *&Strong_gens,
	long int *&candidates,
	int &nb_candidates,
	int &nb_cases,
	int verbose_level)
// A needs to be the base action
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *candidates1;
	int nb_candidates1;
	int h; //, i;

	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files_and_process" << endl;
		}

	read_orbit_rep_and_candidates_from_files(prefix,
		level, orbit_at_level, level_of_candidates_file,
		starter,
		starter_sz,
		Stab,
		Strong_gens,
		candidates1,
		nb_candidates1,
		nb_cases,
		verbose_level - 1);

	for (h = level_of_candidates_file; h < level; h++) {

		long int *candidates2;
		int nb_candidates2;

		if (f_vv) {
			cout << "action::read_orbit_rep_and_candidates_from_files_and_process "
					"testing candidates at level " << h
					<< " number of candidates = " << nb_candidates1 << endl;
			}
		candidates2 = NEW_lint(nb_candidates1);

		(*early_test_func_callback)(starter, h + 1,
			candidates1, nb_candidates1,
			candidates2, nb_candidates2,
			early_test_func_callback_data, 0 /*verbose_level - 1*/);

		if (f_vv) {
			cout << "action::read_orbit_rep_and_candidates_from_files_and_process "
					"number of candidates at level " << h + 1
					<< " reduced from " << nb_candidates1 << " to "
					<< nb_candidates2 << " by "
					<< nb_candidates1 - nb_candidates2 << endl;
			}

		lint_vec_copy(candidates2, candidates1, nb_candidates2);
		nb_candidates1 = nb_candidates2;

		FREE_lint(candidates2);
		}

	candidates = candidates1;
	nb_candidates = nb_candidates1;

	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files_and_process "
				"done" << endl;
		}
}

void action::read_orbit_rep_and_candidates_from_files(
	const char *prefix,
	int level, int orbit_at_level, int level_of_candidates_file,
	long int *&starter,
	int &starter_sz,
	sims *&Stab,
	strong_generators *&Strong_gens,
	long int *&candidates,
	int &nb_candidates,
	int &nb_cases,
	int verbose_level)
// A needs to be the base action
{
	int f_v = (verbose_level >= 1);
	int orbit_at_candidate_level = -1;
	file_io Fio;


	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files" << endl;
		}

	{
	candidates = NULL;
	//longinteger_object stab_go;

	char fname1[1000];
	sprintf(fname1, "%s_lvl_%d", prefix, level);

	read_set_and_stabilizer(fname1,
		orbit_at_level, starter, starter_sz, Stab,
		Strong_gens,
		nb_cases,
		verbose_level);



	//Stab->group_order(stab_go);

	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"Read starter " << orbit_at_level << " / "
				<< nb_cases << " : ";
		lint_vec_print(cout, starter, starter_sz);
		cout << endl;
		//cout << "read_orbit_rep_and_candidates_from_files "
		//"Group order=" << stab_go << endl;
		}

	if (level == level_of_candidates_file) {
		orbit_at_candidate_level = orbit_at_level;
		}
	else {
		// level_of_candidates_file < level
		// Now, we need to find out the orbit representative
		// at level_of_candidates_file
		// that matches with the prefix of starter
		// so that we can retrieve it's set of candidates.
		// Once we have the candidates for the prefix, we run it through the
		// test function to find the candidate set of starter as a subset
		// of this set.

		orbit_at_candidate_level =
				Fio.find_orbit_index_in_data_file(prefix,
				level_of_candidates_file, starter,
				verbose_level);
		}
	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"Found starter, orbit_at_candidate_level="
				<< orbit_at_candidate_level << endl;
		}


	// read the set of candidates from the binary file:

	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"before generator_read_candidates_of_orbit" << endl;
		}
	char fname2[1000];
	sprintf(fname2, "%s_lvl_%d_candidates.bin", prefix,
			level_of_candidates_file);
	Fio.poset_classification_read_candidates_of_orbit(
		fname2, orbit_at_candidate_level,
		candidates, nb_candidates, verbose_level - 1);

	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"generator_read_candidates_of_orbit done" << endl;
		}


	if (candidates == NULL) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"cound not read the candidates" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"Found " << nb_candidates << " candidates at level "
				<< level_of_candidates_file << endl;
		}
	}
	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"done" << endl;
		}
}


void action::read_representatives(char *fname,
		int *&Reps, int &nb_reps, int &size, int verbose_level)
{
	int f_casenumbers = FALSE;
	int nb_cases;
	int *Set_sizes;
	long int **Sets;
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	int i, j;
	file_io Fio;

	cout << "action::read_file_and_print_representatives "
			"reading file " << fname << endl;

	Fio.read_and_parse_data_file_fancy(fname,
		f_casenumbers,
		nb_cases,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		0/*verbose_level*/);
	nb_reps = nb_cases;
	size = Set_sizes[0];
	Reps = NEW_int(nb_cases * size);
	for (i = 0; i < nb_cases; i++) {
		for (j = 0; j < size; j++) {
			Reps[i * size + j] = Sets[i][j];
			}
		}
	Fio.free_data_fancy(nb_cases,
		Set_sizes, Sets,
		Ago_ascii, Aut_ascii,
		Casenumbers);
}

void action::read_representatives_and_strong_generators(
	char *fname, int *&Reps,
	char **&Aut_ascii, int &nb_reps, int &size, int verbose_level)
{
	int f_casenumbers = FALSE;
	int nb_cases;
	int *Set_sizes;
	long int **Sets;
	char **Ago_ascii;
	//char **Aut_ascii;
	int *Casenumbers;
	int i, j;
	file_io Fio;


	cout << "action::read_file_and_print_representatives "
			"reading file " << fname << endl;

	Fio.read_and_parse_data_file_fancy(fname,
		f_casenumbers,
		nb_cases,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		0/*verbose_level*/);
	nb_reps = nb_cases;
	size = Set_sizes[0];
	Reps = NEW_int(nb_cases * size);
	for (i = 0; i < nb_cases; i++) {
		for (j = 0; j < size; j++) {
			Reps[i * size + j] = Sets[i][j];
			}
		}
	Fio.free_data_fancy(nb_cases,
		Set_sizes, Sets,
		Ago_ascii, NULL /*Aut_ascii*/,
		Casenumbers);
}

void action::read_file_and_print_representatives(
		char *fname, int f_print_stabilizer_generators, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_casenumbers = FALSE;
	int nb_cases;
	int *Set_sizes;
	long int **Sets;
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	int i;
	file_io Fio;

	if (f_v) {
		cout << "action::read_file_and_print_representatives "
				"reading file "
				<< fname << endl;
	}

	Fio.read_and_parse_data_file_fancy(fname,
		f_casenumbers,
		nb_cases,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		0/*verbose_level*/);
	for (i = 0; i < nb_cases; i++) {
		cout << "Orbit " << i << " representative ";
		lint_vec_print(cout, Sets[i], Set_sizes[i]);
		cout << endl;

		group *G;
		vector_ge *gens;
		int *tl;

		G = NEW_OBJECT(group);
		G->init(this, verbose_level - 2);
		G->init_ascii_coding_to_sims(Aut_ascii[i], verbose_level - 2);


		longinteger_object go;

		G->S->group_order(go);

		gens = NEW_OBJECT(vector_ge);
		tl = NEW_int(base_len());
		G->S->extract_strong_generators_in_order(*gens, tl,
				0 /* verbose_level */);
		cout << "Stabilizer has order " << go << " tl=";
		int_vec_print(cout, tl, base_len());
		cout << endl;

		if (f_print_stabilizer_generators) {
			cout << "The stabilizer is generated by:" << endl;
			gens->print(cout);
			}

		FREE_OBJECT(G);
		FREE_OBJECT(gens);
		FREE_int(tl);

		}
	Fio.free_data_fancy(nb_cases,
		Set_sizes, Sets,
		Ago_ascii, Aut_ascii,
		Casenumbers);

}

void action::read_set_and_stabilizer(const char *fname,
	int no, long int *&set, int &set_sz, sims *&stab,
	strong_generators *&Strong_gens,
	int &nb_cases,
	int verbose_level)
{
	int f_v = (verbose_level  >= 1);
	int f_vv = (verbose_level  >= 2);
	int f_casenumbers = FALSE;
	//int nb_cases;
	int *Set_sizes;
	long int **Sets;
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	group *G;
	int i;
	file_io Fio;


	if (f_v) {
		cout << "action::read_set_and_stabilizer "
				"reading file " << fname
				<< " no=" << no << endl;
		}

	Fio.read_and_parse_data_file_fancy(fname,
		f_casenumbers,
		nb_cases,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		verbose_level - 1);

	if (f_vv) {
		cout << "action::read_set_and_stabilizer "
				"after read_and_parse_data_file_fancy" << endl;
		cout << "Aut_ascii[no]=" << Aut_ascii[no] << endl;
		cout << "Set_sizes[no]=" << Set_sizes[no] << endl;
		}

	set_sz = Set_sizes[no];
	set = NEW_lint(set_sz);
	for (i = 0; i < set_sz; i ++) {
		set[i] = Sets[no][i];
		}


	G = NEW_OBJECT(group);
	G->init(this, verbose_level - 2);
	if (f_vv) {
		cout << "action::read_set_and_stabilizer "
				"before G->init_ascii_coding_to_sims" << endl;
		}
	G->init_ascii_coding_to_sims(Aut_ascii[no], verbose_level - 2);
	if (f_vv) {
		cout << "action::read_set_and_stabilizer "
				"after G->init_ascii_coding_to_sims" << endl;
		}

	stab = G->S;
	G->S = NULL;
	G->f_has_sims = FALSE;

	longinteger_object go;

	stab->group_order(go);


	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init_from_sims(stab, 0);
	f_has_strong_generators = TRUE;

	if (f_vv) {
		cout << "action::read_set_and_stabilizer "
				"Group order=" << go << endl;
		}

	FREE_OBJECT(G);
	if (f_vv) {
		cout << "action::read_set_and_stabilizer "
				"after FREE_OBJECT  G" << endl;
		}
	Fio.free_data_fancy(nb_cases,
		Set_sizes, Sets,
		Ago_ascii, Aut_ascii,
		Casenumbers);
	if (f_v) {
		cout << "action::read_set_and_stabilizer done" << endl;
		}

}


void action::list_elements_as_permutations_vertically(
		vector_ge *gens,
		ostream &ost)
{
	int i, j, a, len;

	len = gens->len;
	for (j = 0; j < len; j++) {
		ost << " & \\alpha_{" << j << "}";
	}
	ost << "\\\\" << endl;
	for (i = 0; i < degree; i++) {
		ost << setw(3) << i;
		for (j = 0; j < len; j++) {
			a = element_image_of(i,
					gens->ith(j), 0 /* verbose_level */);
			ost << " & " << setw(3) << a;
		}
		ost << "\\\\" << endl;
	}
}

void action::print_symmetry_group_type(ostream &ost)
{
	action_global AG;

	AG.action_print_symmetry_group_type(ost, type_G);
	if (f_has_subaction) {
		ost << "->";
		subaction->print_symmetry_group_type(ost);
		}
	//else {
		//ost << "no subaction";
		//}

}


void action::print_info()
{
	cout << "ACTION " << label << " degree=" << degree << " of type ";
	print_symmetry_group_type(cout);
	cout << endl;
	cout << "low_level_point_size=" << low_level_point_size;
	cout << ", f_has_sims=" << f_has_sims;
	cout << ", f_has_strong_generators=" << f_has_strong_generators;
	cout << endl;

	if (f_is_linear) {
		cout << "linear of dimension " << dimension << endl;
		}
	else {
		cout << "the action is not linear" << endl;
	}

	if (Stabilizer_chain) {
		if (base_len()) {
			cout << "base: ";
			lint_vec_print(cout, get_base(), base_len());
			cout << endl;
		}
	}
	else {
		cout << "The action does not have a stabilizer chain" << endl;
	}
	if (f_has_sims) {
		cout << "has sims" << endl;
		longinteger_object go;

		Sims->group_order(go);
		cout << "Order " << go << " = ";
		//int_vec_print(cout, Sims->orbit_len, base_len());
		for (int t = 0; t < base_len(); t++) {
			cout << Sims->get_orbit_length(t);
			if (t < base_len()) {
				cout << " * ";
			}
		}
		cout << endl;
	}
	cout << endl;

}

void action::report_basic_orbits(ostream &ost)
{
	int i;

	if (Stabilizer_chain) {
		ost << "The base has length " << base_len() << "\\\\" << endl;
		ost << "The basic orbits are: \\\\" << endl;
		for (i = 0; i < base_len(); i++) {
			ost << "Basic orbit " << i << " is orbit of " << base_i(i)
				<< " of length " << transversal_length_i(i) << "\\\\" << endl;
		}
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
	}
}

void action::print_base()
{
	if (Stabilizer_chain) {
		cout << "action " << label << " has base ";
		lint_vec_print(cout, get_base(), base_len());
		cout << endl;
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
	}
}

void action::print_points(ostream &ost)
{
	int i;
	int *v;

	cout << "action::print_points "
			"low_level_point_size=" << low_level_point_size <<  endl;
	v = NEW_int(low_level_point_size);
	ost << "{\\renewcommand*{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & P_{i}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < degree; i++) {
		unrank_point(i, v);
		ost << i << " & ";
		int_vec_print(ost, v, low_level_point_size);
		ost << "\\\\" << endl;
		if (((i + 1) % 10) == 0) {
			ost << "\\hline" << endl;
			ost << "\\end{array}" << endl;
			if (((i + 1) % 50) == 0) {
				ost << "$$" << endl;
				ost << "$$" << endl;
			}
			else {
				ost << ", \\;" << endl;
			}
			ost << "\\begin{array}{|c|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "i & P_{i}\\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\hline" << endl;
		}
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}%" << endl;
	FREE_int(v);
	cout << "action::print_points done" << endl;
}

void action::print_group_order(ostream &ost)
{
	longinteger_object go;
	group_order(go);
	cout << go;
}

void action::print_group_order_long(ostream &ost)
{
	int i;

	longinteger_object go;
	group_order(go);
	cout << go << " =";
	if (Stabilizer_chain) {
		for (i = 0; i < base_len(); i++) {
			cout << " " << transversal_length_i(i);
			}
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
	}

}

void action::print_vector(vector_ge &v)
{
	int i, l;

	l = v.len;
	cout << "vector of " << l << " group elements:" << endl;
	for (i = 0; i < l; i++) {
		cout << i << " : " << endl;
		element_print_quick(v.ith(i), cout);
		cout << endl;
		}
}

void action::print_vector_as_permutation(vector_ge &v)
{
	int i, l;

	l = v.len;
	cout << "vector of " << l << " group elements:" << endl;
	for (i = 0; i < l; i++) {
		cout << i << " : ";
		element_print_as_permutation(v.ith(i), cout);
		cout << endl;
		}
}



}}
