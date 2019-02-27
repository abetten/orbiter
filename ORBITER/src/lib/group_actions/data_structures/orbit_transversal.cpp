// orbit_transversal.C
//
// Anton Betten
//
// November 26, 2017

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {

orbit_transversal::orbit_transversal()
{
	null();
}

orbit_transversal::~orbit_transversal()
{
	freeself();
}

void orbit_transversal::null()
{
	A = NULL;
	A2 = NULL;
	nb_orbits = 0;
	Reps = NULL;
}

void orbit_transversal::freeself()
{
	if (Reps) {
		FREE_OBJECTS(Reps);
		}
	null();
}

void orbit_transversal::init_from_schreier(
		schreier *Sch,
		action *default_action,
		longinteger_object &full_group_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit_idx;
	set_and_stabilizer *R;

	if (f_v) {
		cout << "orbit_transversal::init_from_schreier" << endl;
	}
	A = default_action;
	A2 = Sch->A;
	nb_orbits = Sch->nb_orbits;
	Reps = NEW_OBJECTS(set_and_stabilizer, nb_orbits);
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		R = Sch->get_orbit_rep(default_action,
				full_group_order,
				orbit_idx, verbose_level);
		memcpy(Reps + orbit_idx, R, sizeof(set_and_stabilizer));
		//Reps[orbit_idx] = R;
		R->null();
		FREE_OBJECT(R);
	}
	if (f_v) {
		cout << "orbit_transversal::init_from_schreier done" << endl;
	}

}

void orbit_transversal::read_from_file(
		action *A, action *A2, const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_transversal::read_from_file fname = " << fname << endl;
		}
	
	orbit_transversal::A = A;
	orbit_transversal::A2 = A2;

	int *Set_sizes;
	int **Sets;
	char **Ago_ascii;
	char **Aut_ascii; 
	int *Casenumbers;
	int nb_cases, nb_cases_mod;
	int i;

	if (f_v) {
		cout << "orbit_transversal::read_from_file "
				"before read_and_parse_data_file_fancy" << endl;
		}
	read_and_parse_data_file_fancy(fname, 
		FALSE /*f_casenumbers */, 
		nb_cases, 
		Set_sizes, Sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		verbose_level - 1);
		// GALOIS/util.C

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
		strong_generators *gens;
		int *set;

		gens = NEW_OBJECT(strong_generators);
		gens->init_from_ascii_coding(A,
				Aut_ascii[i], 0 /* verbose_level */);
		
		set = NEW_int(Set_sizes[i]);
		int_vec_copy(Sets[i], set, Set_sizes[i]);
		Reps[i].init_everything(A, A2, set, Set_sizes[i], 
			gens, 0 /* verbose_level */);

		FREE_OBJECT(Reps[i].Stab);
		Reps[i].Stab = NULL;

		// gens and set is now part of Reps[i], so we don't free them here.
		}
	

	free_data_fancy(nb_cases, 
		Set_sizes, Sets, 
		Ago_ascii, Aut_ascii, 
		Casenumbers);

	if (f_v) {
		cout << "orbit_transversal::read_from_file done" << endl;
		}
}

classify *orbit_transversal::get_ago_distribution(int *&ago,
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
	ago = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		ago[i] = Reps[i].group_order_as_int();
	}
	classify *C;
	C = NEW_OBJECT(classify);
	C->init(ago, nb_orbits, FALSE, 0);
	if (f_v) {
		cout << "orbit_transversal::get_ago_distribution done" << endl;
	}
	return C;
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

				longinteger_object(go);
				Reps[i].Strong_gens->group_order(go);

				f << i << " & ";
				int_vec_print(f, Reps[i].data, Reps[i].sz);
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


}}


