/*
 * action_io.cpp
 *
 *  Created on: Mar 30, 2019
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {


void action::report(
		std::ostream &ost, int f_sims, groups::sims *S,
		int f_strong_gens, groups::strong_generators *SG,
		graphics::layered_graph_draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::report" << endl;
	}

	ost << "\\section*{The Action}" << endl;

	ost << "Group action $" << label_tex
			<< "$ of degree " << degree << "\\\\" << endl;

	report_what_we_act_on(ost,
			LG_Draw_options,
			verbose_level);

	if (is_matrix_group()) {
		ost << "The group is a matrix group.\\\\" << endl;

#if 0
		field_theory::finite_field *F;
		groups::matrix_group *M;

		M = get_matrix_group();
		F = M->GFq;

		{
			geometry::projective_space *P;

			P = NEW_OBJECT(geometry::projective_space);

			P->projective_space_init(M->n - 1, F, TRUE, verbose_level);

			ost << "The base action is on projective space ${\\rm PG}(" << M->n - 1 << ", " << F->q << ")$\\\\" << endl;

			P->Reporting->report_summary(ost);



			FREE_OBJECT(P);
		}
#endif



	}

	if (type_G == wreath_product_t) {
		groups::wreath_product *W;

		W = G.wreath_product_group;
		if (f_v) {
			cout << "action::report before W->report" << endl;
		}
		W->report(ost, verbose_level);
		if (f_v) {
			cout << "action::report after W->report" << endl;
		}
	}

	ost << "\\subsection*{Base and Stabilizer Chain}" << endl;

	if (f_sims) {
		if (f_v) {
			cout << "action::report printing group order" << endl;
		}
		ring_theory::longinteger_object go;

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
			Lint_vec_print(ost, get_base(), base_len());
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
		S->report(ost, label, LG_Draw_options, 0 /*verbose_level - 2*/);
		if (f_v) {
			cout << "action::report after S->report" << endl;
		}
	}
	if (Stabilizer_chain) {
		if (f_strong_gens) {

			ost << "GAP export: \\\\" << endl;
			ost << "\\begin{verbatim}" << endl;
			SG->print_generators_gap(ost);
			ost << "\\end{verbatim}" << endl;


			ost << "Magma export: \\\\" << endl;
			ost << "\\begin{verbatim}" << endl;
			SG->export_magma(this, ost, verbose_level);
			ost << "\\end{verbatim}" << endl;

			ost << "Compact form: \\\\" << endl;
			ost << "\\begin{verbatim}" << endl;
			SG->print_generators_compact(ost);
			ost << "\\end{verbatim}" << endl;

		}
	}
	if (f_v) {
		cout << "action::report done" << endl;
	}
}

void action::report_what_we_act_on(
		std::ostream &ost,
		graphics::layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::report_what_we_act_on" << endl;
	}


	std::string txt;
	std::string tex;
	action_global AcGl;

	AcGl.get_symmetry_group_type_text(txt, tex, type_G);


	ost << "The action is of type " << tex << "\\\\" << endl;

	ost << "\\bigskip" << endl;

	if (is_matrix_group()) {

		field_theory::finite_field *F;
		groups::matrix_group *M;

		M = get_matrix_group();
		F = M->GFq;

#if 0
		{
			geometry::projective_space *P;

			P = NEW_OBJECT(geometry::projective_space);

			P->projective_space_init(M->n - 1, F, TRUE, verbose_level);

			ost << "\\section*{The Group Acts on Projective Space ${\\rm PG}(" << M->n - 1 << ", " << F->q << ")$}" << endl;

			P->Reporting->report(ost, O, verbose_level);



			FREE_OBJECT(P);
		}
#endif

		if (type_G == action_on_orthogonal_t) {

			if (G.AO->f_on_points) {
				ost << "acting on points only\\\\" << endl;
				ost << "Number of points = " << G.AO->O->Hyperbolic_pair->nb_points << "\\\\" << endl;
			}
			else if (G.AO->f_on_lines) {
				ost << "acting on lines only\\\\" << endl;
				ost << "Number of lines = " << G.AO->O->Hyperbolic_pair->nb_lines << "\\\\" << endl;
			}
			else if (G.AO->f_on_points_and_lines) {
				ost << "acting on points and lines\\\\" << endl;
				ost << "Number of points = " << G.AO->O->Hyperbolic_pair->nb_points << "\\\\" << endl;
				ost << "Number of lines = " << G.AO->O->Hyperbolic_pair->nb_lines << "\\\\" << endl;
			}

			G.AO->O->Quadratic_form->report_quadratic_form(ost, 0 /* verbose_level */);

			ost << "Tactical decomposition induced by a hyperbolic pair:\\\\" << endl;
			G.AO->O->report_schemes_easy(ost);

			G.AO->O->report_points(ost, 0 /* verbose_level */);

			G.AO->O->report_lines(ost, 0 /* verbose_level */);

		}

		ost << "Group Action $" << label_tex << "$ on Projective Space ${\\rm PG}(" << M->n - 1 << ", " << F->q << ")$\\\\" << endl;

		ost << "The finite field ${\\mathbb F}_{" << F->q << "}$:\\\\" << endl;

		F->cheat_sheet(ost, verbose_level);

		ost << endl << "\\bigskip" << endl << endl;


	}


	if (degree < 100) {
		ost << "The group acts on the following set of size " << degree << ":\\\\" << endl;
		latex_all_points(ost);
	}



	if (f_v) {
		cout << "action::report_what_we_act_on done" << endl;
	}
}

void action::read_orbit_rep_and_candidates_from_files_and_process(
		std::string &prefix,
	int level, int orbit_at_level, int level_of_candidates_file,
	void (*early_test_func_callback)(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level),
	void *early_test_func_callback_data,
	long int *&starter,
	int &starter_sz,
	groups::sims *&Stab,
	groups::strong_generators *&Strong_gens,
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

	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files_and_process before read_orbit_rep_and_candidates_from_files" << endl;
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
		verbose_level);
	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files_and_process after read_orbit_rep_and_candidates_from_files" << endl;
	}

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
			early_test_func_callback_data, verbose_level - 1);

		if (f_vv) {
			cout << "action::read_orbit_rep_and_candidates_from_files_and_process "
					"number of candidates at level " << h + 1
					<< " reduced from " << nb_candidates1 << " to "
					<< nb_candidates2 << " by "
					<< nb_candidates1 - nb_candidates2 << endl;
			}

		Lint_vec_copy(candidates2, candidates1, nb_candidates2);
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
		std::string &prefix,
	int level, int orbit_at_level, int level_of_candidates_file,
	long int *&starter,
	int &starter_sz,
	groups::sims *&Stab,
	groups::strong_generators *&Strong_gens,
	long int *&candidates,
	int &nb_candidates,
	int &nb_cases,
	int verbose_level)
// A needs to be the base action
{
	int f_v = (verbose_level >= 1);
	int orbit_at_candidate_level = -1;
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files prefix=" << prefix << endl;
	}

	{
		candidates = NULL;
		//longinteger_object stab_go;

		string fname1;
		char str[1000];
		snprintf(str, sizeof(str), "_lvl_%d", level);
		fname1.assign(prefix);
		fname1.append(str);

		if (f_v) {
			cout << "action::read_orbit_rep_and_candidates_from_files before read_set_and_stabilizer fname1=" << fname1 << endl;
		}
		read_set_and_stabilizer(fname1,
			orbit_at_level, starter, starter_sz, Stab,
			Strong_gens,
			nb_cases,
			verbose_level);
		if (f_v) {
			cout << "action::read_orbit_rep_and_candidates_from_files after read_set_and_stabilizer" << endl;
		}



		//Stab->group_order(stab_go);

		if (f_v) {
			cout << "action::read_orbit_rep_and_candidates_from_files "
					"Read starter " << orbit_at_level << " / "
					<< nb_cases << " : ";
			Lint_vec_print(cout, starter, starter_sz);
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

			orbit_at_candidate_level = Fio.find_orbit_index_in_data_file(prefix,
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
		string fname2;
		fname2.assign(prefix);
		snprintf(str, sizeof(str), "_lvl_%d_candidates.bin", level_of_candidates_file);
		fname2.append(str);


		if (f_v) {
			cout << "action::read_orbit_rep_and_candidates_from_files "
					"before Fio.poset_classification_read_candidates_of_orbit" << endl;
		}
		Fio.poset_classification_read_candidates_of_orbit(
			fname2, orbit_at_candidate_level,
			candidates, nb_candidates, verbose_level - 1);

		if (f_v) {
			cout << "action::read_orbit_rep_and_candidates_from_files "
					"after Fio.poset_classification_read_candidates_of_orbit" << endl;
		}


		if (candidates == NULL) {
			cout << "action::read_orbit_rep_and_candidates_from_files "
					"could not read the candidates" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "action::read_orbit_rep_and_candidates_from_files "
					"Found " << nb_candidates << " candidates at level "
					<< level_of_candidates_file << endl;
		}
	}
	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files done" << endl;
	}
}


void action::read_representatives(
		std::string &fname,
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
	orbiter_kernel_system::file_io Fio;

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
	std::string &fname, int *&Reps,
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
	orbiter_kernel_system::file_io Fio;


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
		std::string &fname,
		int f_print_stabilizer_generators, int verbose_level)
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
	orbiter_kernel_system::file_io Fio;

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
		Lint_vec_print(cout, Sets[i], Set_sizes[i]);
		cout << endl;

		data_structures_groups::group_container *G;
		data_structures_groups::vector_ge *gens;
		int *tl;

		G = NEW_OBJECT(data_structures_groups::group_container);
		G->init(this, verbose_level - 2);

		string s;

		s.assign(Aut_ascii[i]);
		G->init_ascii_coding_to_sims(s, verbose_level - 2);


		ring_theory::longinteger_object go;

		G->S->group_order(go);

		gens = NEW_OBJECT(data_structures_groups::vector_ge);
		tl = NEW_int(base_len());
		G->S->extract_strong_generators_in_order(*gens, tl,
				0 /* verbose_level */);
		cout << "Stabilizer has order " << go << " tl=";
		Int_vec_print(cout, tl, base_len());
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

void action::read_set_and_stabilizer(
		std::string &fname,
	int no, long int *&set, int &set_sz, groups::sims *&stab,
	groups::strong_generators *&Strong_gens,
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
	data_structures_groups::group_container *G;
	int i;
	orbiter_kernel_system::file_io Fio;


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


	G = NEW_OBJECT(data_structures_groups::group_container);
	G->init(this, verbose_level - 2);
	if (f_vv) {
		cout << "action::read_set_and_stabilizer "
				"before G->init_ascii_coding_to_sims" << endl;
		}

	string s;

	s.assign(Aut_ascii[no]);
	G->init_ascii_coding_to_sims(s, verbose_level - 2);
	if (f_vv) {
		cout << "action::read_set_and_stabilizer "
				"after G->init_ascii_coding_to_sims" << endl;
		}

	stab = G->S;
	G->S = NULL;
	G->f_has_sims = FALSE;

	ring_theory::longinteger_object go;

	stab->group_order(go);


	Strong_gens = NEW_OBJECT(groups::strong_generators);
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
		data_structures_groups::vector_ge *gens,
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

void action::print_symmetry_group_type(std::ostream &ost)
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
			Lint_vec_print(cout, get_base(), base_len());
			cout << endl;
		}
	}
	else {
		cout << "The action does not have a stabilizer chain" << endl;
	}
	if (f_has_sims) {
		cout << "has sims" << endl;
		ring_theory::longinteger_object go;

		Sims->group_order(go);
		cout << "Order " << go << " = ";
		//int_vec_print(cout, Sims->orbit_len, base_len());
		for (int t = 0; t < base_len(); t++) {
			cout << Sims->get_orbit_length(t);
			if (t < base_len() - 1) {
				cout << " * ";
			}
		}
		//cout << endl;
	}
	cout << endl;

}

void action::report_basic_orbits(std::ostream &ost)
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
		Lint_vec_print(cout, get_base(), base_len());
		cout << endl;
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
	}
}

void action::print_bare_base(std::ofstream &ost)
{
	if (Stabilizer_chain) {
		orbiter_kernel_system::Orbiter->Lint_vec->print_bare_fully(ost, get_base(), base_len());
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
		exit(1);
	}
}

void action::latex_all_points(std::ostream &ost)
{
	int i;
	int *v;


	if (ptr->ptr_unrank_point == NULL) {
		cout << "action::latex_all_points ptr->ptr_unrank_point == NULL" << endl;
		return;
	}
	v = NEW_int(low_level_point_size);
#if 0
	cout << "action::latex_all_points "
			"low_level_point_size=" << low_level_point_size <<  endl;
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
			ost << "\\begin{array}{|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "i & P_{i}\\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\hline" << endl;
		}
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}%" << endl;
	cout << "action::latex_all_points done" << endl;
#else
	if (low_level_point_size < 10) {
		ost << "\\begin{multicols}{2}" << endl;
	}
	ost << "\\noindent" << endl;
	for (i = 0; i < degree; i++) {
		unrank_point(i, v);
		ost << i << " = ";
		Int_vec_print(ost, v, low_level_point_size);
		ost << "\\\\" << endl;
	}
	if (low_level_point_size < 10) {
		ost << "\\end{multicols}" << endl;
	}

#endif

	FREE_int(v);
}

void action::latex_point_set(
		std::ostream &ost,
		long int *set, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *v;

	if (f_v) {
		cout << "action::print_points "
				"low_level_point_size=" << low_level_point_size <<  endl;
	}
	v = NEW_int(low_level_point_size);
#if 0
	ost << "{\\renewcommand*{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & P_{i} \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < sz; i++) {
		unrank_point(set[i], v);
		ost << i << " & ";
		ost << set[i] << " = ";
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
			ost << "\\begin{array}{|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "i & P_{i}\\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\hline" << endl;
		}
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}%" << endl;
#else


	if (ptr->ptr_unrank_point) {
		if (low_level_point_size < 10) {
			ost << "\\begin{multicols}{2}" << endl;
		}
		ost << "\\noindent" << endl;
		for (i = 0; i < sz; i++) {
			unrank_point(set[i], v);
			ost << i << " : ";
			ost << set[i] << " = ";
			Int_vec_print(ost, v, low_level_point_size);
			ost << "\\\\" << endl;
		}
		if (low_level_point_size < 10) {
			ost << "\\end{multicols}" << endl;
		}
	}
#endif

	FREE_int(v);
	if (f_v) {
		cout << "action::print_points done" << endl;
	}
}


void action::print_group_order(std::ostream &ost)
{
	ring_theory::longinteger_object go;
	group_order(go);
	cout << go;
}

void action::print_group_order_long(std::ostream &ost)
{
	int i;

	ring_theory::longinteger_object go;
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

void action::print_vector(
		data_structures_groups::vector_ge &v)
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

void action::print_vector_as_permutation(
		data_structures_groups::vector_ge &v)
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


void action::write_set_of_elements_latex_file(
		std::string &fname,
		std::string &title, int *Elt, int nb_elts)
{
	{
		ofstream ost(fname);
		number_theory::number_theory_domain NT;

		string author, extra_praeamble;

		orbiter_kernel_system::latex_interface L;

		L.head(ost,
				FALSE /* f_book*/,
				TRUE /* f_title */,
				title, author,
				FALSE /* f_toc */,
				FALSE /* f_landscape */,
				TRUE /* f_12pt */,
				TRUE /* f_enlarged_page */,
				TRUE /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);


		//Nth->report(ost, verbose_level);

		int i;

		for (i = 0; i < nb_elts; i++) {
			ost << "$$" << endl;
			element_print_latex(Elt + i * elt_size_in_int, ost);
			ost << "$$" << endl;
		}

		L.foot(ost);


	}

	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

}

void action::export_to_orbiter(
		std::string &fname, std::string &label,
		groups::strong_generators *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int a;
	orbiter_kernel_system::file_io Fio;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "action::export_to_orbiter" << endl;
	}

	SG->group_order(go);
	if (f_v) {
		cout << "action::export_to_orbiter go = " << go << endl;
		cout << "action::export_to_orbiter number of generators = " << SG->gens->len << endl;
		cout << "action::export_to_orbiter degree = " << degree << endl;
	}
	{
		ofstream fp(fname);

		fp << "GENERATORS_" << label << " = \\" << endl;
		for (i = 0; i < SG->gens->len; i++) {
			fp << "\t\"";
			for (j = 0; j < degree; j++) {
				if (FALSE) {
					cout << "action::export_to_orbiter computing image of " << j << " under generator " << i << endl;
				}
				a = element_image_of(j, SG->gens->ith(i), 0 /* verbose_level*/);
				fp << a;
				if (j < degree - 1 || i < SG->gens->len - 1) {
					fp << ",";
				}
			}
			fp << "\"";
			if (i < SG->gens->len - 1) {
				fp << "\\" << endl;
			}
			else {
				fp << endl;
			}
		}

		fp << endl;
		fp << label << ":" << endl;
		fp << "\t$(ORBITER_PATH)orbiter.out -v 2 \\" << endl;
		fp << "\t-define G -permutation_group -symmetric_group " << degree << " \\" << endl;
		fp << "\t-subgroup_by_generators " << label << " " << go << " " << SG->gens->len << " $(GENERATORS_" << label << ") -end \\" << endl;

		//		$(ORBITER_PATH)orbiter.out -v 10
		//			-define G -permutation_group -symmetric_group 13
		//				-subgroup_by_generators H5 5 1 $(GENERATORS_H5) -end
		// with backslashes at the end of the line

	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "action::export_to_orbiter" << endl;
	}
}


void action::export_to_orbiter_as_bsgs(
		std::string &fname,
		std::string &label,
		std::string &label_tex,
		groups::strong_generators *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::export_to_orbiter_as_bsgs" << endl;
	}


	if (f_v) {
		cout << "action::export_to_orbiter_as_bsgs "
				"before SG->export_to_orbiter_as_bsgs" << endl;
	}
	SG->export_to_orbiter_as_bsgs(
			this,
			fname, label, label_tex,
			verbose_level);
	if (f_v) {
		cout << "action::export_to_orbiter_as_bsgs "
				"after SG->export_to_orbiter_as_bsgs" << endl;
	}


	if (f_v) {
		cout << "action::export_to_orbiter_as_bsgs" << endl;
	}
}


void action::print_one_element_tex(
		std::ostream &ost,
		int *Elt, int f_with_permutation)
{
	ost << "$$" << endl;
	element_print_latex(Elt, ost);
	ost << "$$" << endl;

	if (f_with_permutation) {
		element_print_as_permutation(Elt, ost);
		ost << "\\\\" << endl;

		int *perm;
		int h, j;

		perm = NEW_int(degree);

		element_as_permutation(
				Elt,
				perm, 0 /* verbose_level */);

		for (h = 0; h < degree; h++) {
			j = perm[h];

			ost << j;
			if (j < degree) {
				ost << ", ";
			}
		}
		ost << "\\\\" << endl;

		FREE_int(perm);

	}

}


}}}

