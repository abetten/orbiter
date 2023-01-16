// orbit_transversal.cpp
//
// Anton Betten
//
// November 26, 2017

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {


orbit_transversal::orbit_transversal()
{
	A = NULL;
	A2 = NULL;
	nb_orbits = 0;
	Reps = NULL;
}

orbit_transversal::~orbit_transversal()
{
	if (Reps) {
		FREE_OBJECTS(Reps);
		}
}

void orbit_transversal::init_from_schreier(
		groups::schreier *Sch,
		actions::action *default_action,
		ring_theory::longinteger_object &full_group_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit_idx;
	//set_and_stabilizer *R;

	if (f_v) {
		cout << "orbit_transversal::init_from_schreier" << endl;
	}
	A = default_action;
	A2 = Sch->A;
	nb_orbits = Sch->nb_orbits;
	Reps = NEW_OBJECTS(set_and_stabilizer, nb_orbits);
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		Sch->get_orbit_rep_to(default_action,
				full_group_order,
				orbit_idx,
				Reps + orbit_idx,
				verbose_level);
		//memcpy(Reps + orbit_idx, R, sizeof(set_and_stabilizer));
		//ToDo

		//Reps[orbit_idx] = R;
		//R->null();
		//FREE_OBJECT(R);
	}
	if (f_v) {
		cout << "orbit_transversal::init_from_schreier done" << endl;
	}

}

void orbit_transversal::read_from_file(
		actions::action *A,
		actions::action *A2,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_transversal::read_from_file fname = " << fname << endl;
	}
	
	orbit_transversal::A = A;
	orbit_transversal::A2 = A2;

	int *Set_sizes;
	long int **Sets;
	char **Ago_ascii;
	char **Aut_ascii; 
	int *Casenumbers;
	int nb_cases, nb_cases_mod;
	int i;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "orbit_transversal::read_from_file "
				"before read_and_parse_data_file_fancy" << endl;
	}
	Fio.read_and_parse_data_file_fancy(fname,
		FALSE /*f_casenumbers */, 
		nb_cases, 
		Set_sizes, Sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		verbose_level - 1);

	nb_orbits = nb_cases;


	if (f_v) {
		cout << "orbit_transversal::read_from_file "
				"processing " << nb_orbits
				<< " orbit representatives" << endl;
	}


	Reps = NEW_OBJECTS(set_and_stabilizer, nb_orbits);

	nb_cases_mod = (nb_cases / 100) + 1;
	
	for (i = 0; i < nb_cases; i++) {
		
		if (f_v && ((i + 1) % nb_cases_mod) == 0) {
			cout << "orbit_transversal::read_from_file processing "
					"case " << i << " / " << nb_orbits << " : "
					<< 100. * (double) i / (double) nb_cases
					<< "%" << endl;
		}
		groups::strong_generators *gens;
		long int *set;
		string s;

		gens = NEW_OBJECT(groups::strong_generators);

		s.assign(Aut_ascii[i]);
		gens->init_from_ascii_coding(A,
				s, 0 /* verbose_level */);
		
		set = NEW_lint(Set_sizes[i]);
		Lint_vec_copy(Sets[i], set, Set_sizes[i]);
		Reps[i].init_everything(A, A2, set, Set_sizes[i], 
			gens, 0 /* verbose_level */);

		FREE_OBJECT(Reps[i].Stab);
		Reps[i].Stab = NULL;

		// gens and set is now part of Reps[i], so we don't free them here.
	}
	

	Fio.free_data_fancy(nb_cases,
		Set_sizes, Sets, 
		Ago_ascii, Aut_ascii, 
		Casenumbers);

	if (f_v) {
		cout << "orbit_transversal::read_from_file done" << endl;
	}
}

void orbit_transversal::read_from_file_one_case_only(
		actions::action *A,
		actions::action *A2,
		std::string &fname,
		int case_nr,
		set_and_stabilizer *&Rep,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_transversal::read_from_file_one_case_only "
				"fname = " << fname << " case_nr = " << case_nr << endl;
	}

	orbit_transversal::A = A;
	orbit_transversal::A2 = A2;

	int *Set_sizes;
	long int **Sets;
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	int nb_cases; //, nb_cases_mod;
	int i;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "orbit_transversal::read_from_file_one_case_only "
				"before read_and_parse_data_file_fancy" << endl;
	}
	Fio.read_and_parse_data_file_fancy(fname,
		FALSE /*f_casenumbers */,
		nb_cases,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		verbose_level - 1);

	nb_orbits = nb_cases;


	if (f_v) {
		cout << "orbit_transversal::read_from_file_one_case_only "
				"processing " << nb_orbits
				<< " orbit representatives" << endl;
	}


	Rep = NEW_OBJECT(set_and_stabilizer);

	//nb_cases_mod = (nb_cases / 100) + 1;

	i = case_nr;

	groups::strong_generators *gens;
	long int *set;

	gens = NEW_OBJECT(groups::strong_generators);

	string s;

	s.assign(Aut_ascii[i]);
	gens->init_from_ascii_coding(A,
			s, 0 /* verbose_level */);

	set = NEW_lint(Set_sizes[i]);

	Lint_vec_copy(Sets[i], set, Set_sizes[i]);

	Rep->init_everything(A, A2, set, Set_sizes[i],
		gens, 0 /* verbose_level */);

	FREE_OBJECT(Rep->Stab);
	Rep->Stab = NULL;

	// gens and set is now part of Reps[i], so we don't free them here.


	Fio.free_data_fancy(nb_cases,
		Set_sizes, Sets,
		Ago_ascii, Aut_ascii,
		Casenumbers);

	if (f_v) {
		cout << "orbit_transversal::read_from_file_one_case_only done" << endl;
	}
}

data_structures::tally *orbit_transversal::get_ago_distribution(long int *&ago,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "orbit_transversal::get_ago_distribution" << endl;
	}
	if (f_v) {
		cout << "orbit_transversal::get_ago_distribution "
				"nb_orbits = " << nb_orbits << endl;
	}
	ago = NEW_lint(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		ago[i] = Reps[i].group_order_as_lint();
	}
	data_structures::tally *C;
	C = NEW_OBJECT(data_structures::tally);
	C->init_lint(ago, nb_orbits, FALSE, 0);
	if (f_v) {
		cout << "orbit_transversal::get_ago_distribution done" << endl;
	}
	return C;
}

void orbit_transversal::report_ago_distribution(std::ostream &ost)
{
	data_structures::tally *C;
	long int *Ago;
	int i, f, l, a;

	C = get_ago_distribution(Ago, 0 /*verbose_level*/);

	for (i = C->nb_types - 1; i >= 0; i--) {
		f = C->type_first[i];
		l = C->type_len[i];
		a = C->data_sorted[f];
		//ost << "$" << a;
		ost << "There are " << l << " orbits with a stabilizer of order " << a << "\\\\" << endl;
		}

	FREE_lint(Ago);
}

void orbit_transversal::print_table_latex(
		ostream &f,
		int f_has_callback,
		void (*callback_print_function)(
				stringstream &ost, void *data, void *callback_data),
		void *callback_data,
		int f_has_callback2,
		void (*callback_print_function2)(
				stringstream &ost, void *data, void *callback_data),
		void *callback_data2,
		int verbose_level)
{
	int I, i, row;
	int nb_rows_per_page = 40, nb_tables;

	f << "The " << nb_orbits << " orbits are :\\\\" << endl;

	nb_tables = (nb_orbits + nb_rows_per_page - 1) / nb_rows_per_page;

	for (I = 0; I < nb_tables; I++) {
		f << "$$" << endl;
		f << "\\begin{array}{r|rr";
		if (f_has_callback) {
			f << "r";
		}
		if (f_has_callback2) {
			f << "r";
		}
		f << "}" << endl;
		f << "&&&\\\\" << endl;
		f << "\\hline" << endl;
		for (row = 0; row < nb_rows_per_page; row++) {
			i = I * nb_rows_per_page + row;
			if (i < nb_orbits) {

				ring_theory::longinteger_object go;
				Reps[i].Strong_gens->group_order(go);

				f << i << " & ";
				Lint_vec_print(f, Reps[i].data, Reps[i].sz);
				f << " & " << go;
				if (f_has_callback) {
					f << " & ";
					stringstream ost;
					(*callback_print_function)(ost,
							Reps[i].data, callback_data);
					string s = ost.str();
					f << s;
				}
				if (f_has_callback2) {
					f << " & ";
					stringstream ost;
					(*callback_print_function2)(ost,
							Reps[i].data, callback_data2);
					string s = ost.str();
					f << s;
				}
				//f << " & ";
				//Reps[i].Strong_gens->print_generators_tex(f);
				f << "\\\\" << endl;
			}
		}
		f << "\\end{array}" << endl;
		f << "$$" << endl;
	}
}

void orbit_transversal::export_data_in_source_code_inside_tex(
		std::string &prefix,
		std::string &label_of_structure, std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int h, i;
	int size;

	if (f_v) {
		cout << "orbit_transversal::export_data_in_source_code_inside_tex" << endl;
		}

	ost << "\\section{The " << label_of_structure
			<< " in Numeric Form}" << endl << endl;

	if (nb_orbits == 0) {
		cout << "orbit_transversal::export_data_in_source_code_inside_tex "
				"nb_orbits == 0" << endl;
		return;
	}
	size = Reps[0].sz;
	//fp << "\\clearpage" << endl << endl;
	for (h = 0; h < nb_orbits; h++) {
		if (Reps[h].sz != size) {
			cout << "the data has different sizes" << endl;
			exit(1);
		}
		for (i = 0; i < size; i++) {
			ost << Reps[h].data[i];
			if (i < Reps[h].sz - 1) {
				ost << ", ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\begin{verbatim}" << endl << endl;
	ost << "int " << prefix << "_size = " << size << ";" << endl;
	ost << "int " << prefix << "_nb_reps = " << nb_orbits << ";" << endl;
	ost << "int " << prefix << "_reps[] = {" << endl;
	for (h = 0; h < nb_orbits; h++) {
		ost << "\t";
		for (i = 0; i < size; i++) {
			ost << Reps[h].data[i];
			ost << ", ";
			}
		ost << endl;
		}
	ost << "};" << endl;
	ost << "const char *" << prefix << "_stab_order[] = {" << endl;
	for (h = 0; h < nb_orbits; h++) {

		ring_theory::longinteger_object go;

		if (Reps[h].Stab) {
			Reps[h].Stab->group_order(go);
			ost << "\"";
			go.print_not_scientific(ost);
			ost << "\"," << endl;
			}
		else {
			ost << "\"";
			ost << "1";
			ost << "\"," << endl;
			}
		}
	ost << "};" << endl;

	{
	int *stab_gens_first;
	int *stab_gens_len;
	int fst, j;

	stab_gens_first = NEW_int(nb_orbits);
	stab_gens_len = NEW_int(nb_orbits);
	fst = 0;
	ost << "int " << prefix << "_stab_gens[] = {" << endl;
	for (h = 0; h < nb_orbits; h++) {

		stab_gens_first[h] = fst;
		stab_gens_len[h] = Reps[h].Strong_gens->gens->len;
		fst += Reps[h].Strong_gens->gens->len;

		for (j = 0; j < Reps[h].Strong_gens->gens->len; j++) {
			if (f_vv) {
				cout << "isomorph_report_data_in_source_code_inside_"
						"tex_with_selection before extract_strong_"
						"generators_in_order generator " << j
						<< " / " << Reps[h].Strong_gens->gens->len << endl;
				}
			ost << "";
			A->element_print_for_make_element(
					Reps[h].Strong_gens->gens->ith(j), ost);
			ost << endl;
			}
		}
	ost << "};" << endl;
	ost << "int " << prefix << "_stab_gens_fst[] = { ";
	for (h = 0; h < nb_orbits; h++) {
		ost << stab_gens_first[h];
		if (h < nb_orbits - 1) {
			ost << ", ";
			}
		}
	ost << "};" << endl;
	ost << "int " << prefix << "_stab_gens_len[] = { ";
	for (h = 0; h < nb_orbits; h++) {
		ost << stab_gens_len[h];
		if (h < nb_orbits - 1) {
			ost << ", ";
			}
		}
	ost << "};" << endl;
	ost << "int " << prefix << "_make_element_size = "
			<< A->make_element_size << ";" << endl;
	}
	ost << "\\end{verbatim}" << endl << endl;


	if (f_v) {
		cout << "orbit_transversal::export_data_in_source_code_inside_tex" << endl;
		}

}




}}}



