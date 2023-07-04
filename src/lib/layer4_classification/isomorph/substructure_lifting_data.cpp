/*
 * substructure_lifting_data.cpp
 *
 *  Created on: Sep 5, 2022
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace isomorph {


substructure_lifting_data::substructure_lifting_data()
{
	Iso = NULL;

	nb_flag_orbits = 0;

	N = 0;
		// the number of solutions,
		// computed in init_cases_from_file
		// or read from file in read_case_len



	starter_solution_first = NULL;
		// [nb_starter + 1] the beginning of solutions
		// belonging to a given starter
		// previously called case_first
	starter_solution_len = NULL;
		// [nb_starter + 1] the number of solutions
		// belonging to a given starter
		// previously called case_len



	starter_number_of_solution = NULL;


	flag_orbit_solution_first = NULL;
	flag_orbit_solution_len = NULL;
	flag_orbit_of_solution = NULL;
	orbit_perm = NULL;
	orbit_perm_inv = NULL;
	schreier_vector = NULL;
	schreier_prev = NULL;

	f_use_table_of_solutions = false;
	table_of_solutions = NULL;

	first_flag_orbit_of_starter = NULL;
	nb_flag_orbits_of_starter = NULL;

	stats_nb_backtrack = NULL;
	stats_nb_backtrack_decision = NULL;
	stats_graph_size = NULL;
	stats_time = NULL;

	v = NULL;
	DB_sol = NULL;
	id_to_datref = NULL;
	id_to_hash = NULL;
	hash_vs_id_hash = NULL;
	hash_vs_id_id = NULL;
}

substructure_lifting_data::~substructure_lifting_data()
{
	if (table_of_solutions) {
		FREE_lint(table_of_solutions);
		table_of_solutions = NULL;
		f_use_table_of_solutions = false;
	}
	if (v) {
		delete [] v;
	}
	if (DB_sol) {
		freeobject(DB_sol);
		DB_sol = NULL;
	}
	if (id_to_datref) {
		FREE_lint(id_to_datref);
		id_to_datref = NULL;
	}
	if (id_to_hash) {
		FREE_lint(id_to_hash);
		id_to_hash = NULL;
	}
	if (hash_vs_id_hash) {
		FREE_lint(hash_vs_id_hash);
		hash_vs_id_hash = NULL;
	}
	if (hash_vs_id_id) {
		FREE_lint(hash_vs_id_id);
		hash_vs_id_id = NULL;
	}
}


void substructure_lifting_data::init(isomorph *Iso, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "substructure_lifting_data::init" << endl;
	}

	substructure_lifting_data::Iso = Iso;

	fname_flag_orbits = Iso->prefix + "_flag_orbits.csv";

	fname_stab_orbits = Iso->prefix + "_stab_orbits.csv";


	fname_case_len = Iso->prefix + "_case_len.txt";


	fname_statistics = Iso->prefix + "_statistics.txt";


	fname_hash_and_datref = Iso->prefix + "_hash_and_datref.csv";



	fname_db1 = Iso->prefix + "_solutions.db";
	fname_db2 = Iso->prefix + "_solutions_a.idx";
	fname_db3 = Iso->prefix + "_solutions_b.idx";
	fname_db4 = Iso->prefix + "_solutions_c.idx";
	fname_db5 = Iso->prefix + "_solutions_d.idx";


	fname_orbits_of_stabilizer_csv = Iso->prefix + "_orbits_of_stabilizer.csv";



	v = new layer2_discreta::typed_objects::Vector[1];


	if (f_v) {
		cout << "substructure_lifting_data::init done" << endl;
	}
}

void substructure_lifting_data::write_solution_first_and_len(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ofstream f(fname_case_len);
	int i;

	if (f_v) {
		cout << "substructure_lifting_data::write_solution_first_and_len " << endl;
	}
	f << N << " " << Iso->Sub->nb_starter << endl;
	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		f << setw(4) << i << " " << setw(4) << starter_solution_first[i]
			<< " " << setw(4) << starter_solution_len[i] << endl;
	}
	f << "-1" << endl;
	if (f_v) {
		cout << "substructure_lifting_data::write_solution_first_and_len done" << endl;
	}
}

void substructure_lifting_data::read_solution_first_and_len(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "substructure_lifting_data::read_solution_first_and_len "
				"reading from file "
			<< fname_case_len << " of size "
			<< Fio.file_size(fname_case_len) << endl;
	}


	{
		ifstream f(fname_case_len);
		int i, a;

		f >> N >> Iso->Sub->nb_starter;
		starter_solution_first = NEW_int(Iso->Sub->nb_starter + 1);
		starter_solution_len = NEW_int(Iso->Sub->nb_starter + 1);
		for (i = 0; i < Iso->Sub->nb_starter; i++) {
			f >> a;
			f >> starter_solution_first[i];
			f >> starter_solution_len[i];
		}
		starter_solution_first[Iso->Sub->nb_starter] =
				starter_solution_first[Iso->Sub->nb_starter - 1] +
				starter_solution_len[Iso->Sub->nb_starter - 1];
		f >> a;
		if (a != -1) {
			cout << "problem in read_solution_first_and_len" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "substructure_lifting_data::read_solution_first_and_len:" << endl;
		Sorting.int_vec_print_classified(cout, starter_solution_len, Iso->Sub->nb_starter);
		cout << endl;
	}
	if (f_v) {
		cout << "substructure_lifting_data::read_solution_first_and_len done" << endl;
	}
}

void substructure_lifting_data::init_starter_number(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	int i, j, f, l;

	if (f_v) {
		cout << "substructure_lifting_data::init_starter_number N=" << N << endl;
	}
	starter_number_of_solution = NEW_int(N);
	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		f = starter_solution_first[i];
		l = starter_solution_len[i];
		for (j = 0; j < l; j++) {
			starter_number_of_solution[f + j] = i;
		}
	}
	if (f_vv) {
		cout << "substructure_lifting_data::init_starter_number:" << endl;
		Int_vec_print(cout, starter_number_of_solution, N);
		cout << endl;
	}
	if (f_v) {
		cout << "substructure_lifting_data::init_starter_number done" << endl;
	}
}

void substructure_lifting_data::init_solution(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "substructure_lifting_data::init_solution" << endl;
	}

	if (f_v) {
		cout << "substructure_lifting_data::init_solution "
				"before read_solution_first_and_len" << endl;
	}
	read_solution_first_and_len(verbose_level - 2);
	if (f_v) {
		cout << "substructure_lifting_data::init_solution "
				"after read_solution_first_and_len" << endl;
	}

	if (f_v) {
		cout << "substructure_lifting_data::init_solution "
				"before init_starter_number" << endl;
	}
	init_starter_number(verbose_level - 2);
	if (f_v) {
		cout << "substructure_lifting_data::init_solution "
				"after init_starter_number" << endl;
	}

	if (f_v) {
		cout << "substructure_lifting_data::init_solution "
				"before init_starter_number" << endl;
	}
	read_hash_and_datref_file(verbose_level - 2);
	if (f_v) {
		cout << "substructure_lifting_data::init_solution "
				"after init_starter_number" << endl;
	}

	if (f_v) {
		cout << "substructure_lifting_data::init_solution done" << endl;
	}
}

void substructure_lifting_data::load_table_of_solutions(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int id, j;
	long int data[1000];

	if (f_v) {
		cout << "substructure_lifting_data::load_table_of_solutions N=" << N << endl;
	}
	setup_and_open_solution_database(verbose_level - 2);
	table_of_solutions = NEW_lint(N * Iso->size);
	for (id = 0; id < N; id++) {
		load_solution(id, data, verbose_level - 1);
		for (j = 0; j < Iso->size; j++) {
			table_of_solutions[id * Iso->size + j] = data[j];
		}
#if 0
		cout << "solution " << id << " : ";
		int_vec_print(cout, table_of_solutions + id * size, size);
		cout << endl;
#endif
	}
	f_use_table_of_solutions = true;
	close_solution_database(verbose_level - 2);
	if (f_v) {
		cout << "substructure_lifting_data::load_table_of_solutions done" << endl;
	}
}



void substructure_lifting_data::list_solutions_by_starter(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, idx, id, f, l, fst, len, h, pos, u;
	long int data[1000];
	long int data2[1000];
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "substructure_lifting_data::list_solutions_by_starter" << endl;
	}
	setup_and_open_solution_database(verbose_level - 2);

	j = 0;
	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		f = starter_solution_first[i];
		l = starter_solution_len[i];
		cout << "starter " << i << " solutions from="
				<< f << " len=" << l << endl;
		pos = f;
		while (pos < f + l) {
			fst = flag_orbit_solution_first[j];
			len = flag_orbit_solution_len[j];
			cout << "orbit " << j << " from=" << fst
					<< " len=" << len << endl;
			for (u = 0; u < len; u++) {
				idx = fst + u;
				id = orbit_perm[idx];
				load_solution(id, data, verbose_level - 1);
				for (h = 0; h < Iso->size; h++) {
					data2[h] = data[h];
				}
				Sorting.lint_vec_heapsort(data2, Iso->size);
				cout << i << " : " << j << " : "
						<< idx << " : " << id << endl;
				Lint_vec_print(cout, data, Iso->size);
				cout << endl;
				Lint_vec_print(cout, data2, Iso->size);
				cout << endl;
			}
			pos += len;
			j++;
		}
	}
	close_solution_database(verbose_level - 2);
	if (f_v) {
		cout << "substructure_lifting_data::list_solutions_by_starter done" << endl;
	}
}


void substructure_lifting_data::list_solutions_by_orbit(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, idx, id, f, l, h;
	long int data[1000];
	long int data2[1000];
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "substructure_lifting_data::list_solutions_by_orbit" << endl;
	}
	setup_and_open_solution_database(verbose_level - 1);

	for (i = 0; i < nb_flag_orbits; i++) {
		f = flag_orbit_solution_first[i];
		l = flag_orbit_solution_len[i];
		cout << "orbit " << i << " from=" << f
				<< " len=" << l << endl;
		for (j = 0; j < l; j++) {
			idx = f + j;
			id = orbit_perm[idx];
			load_solution(id, data, verbose_level - 1);
			for (h = 0; h < Iso->size; h++) {
				data2[h] = data[h];
			}
			Sorting.lint_vec_heapsort(data2, Iso->size);
			cout << j << " : " << idx << " : " << id << endl;
			Lint_vec_print(cout, data, Iso->size);
			cout << endl;
			Lint_vec_print(cout, data2, Iso->size);
			cout << endl;
		}
	}

	close_solution_database(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::list_solutions_by_orbit done" << endl;
	}
}

void substructure_lifting_data::orbits_of_stabilizer(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvvv = (verbose_level >= 4);
	int f_v5 = (verbose_level >= 5);
	int i, j, f, l, nb_orbits_prev = 0;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer" << endl;
		cout << "number of starters = nb_starter = "
				<< Iso->Sub->nb_starter << endl;
		cout << "number of solutions (= N) = " << N << endl;
		cout << "action A_base=";
		Iso->A_base->print_info();
		cout << endl;
		cout << "action A=";
		Iso->A->print_info();
		cout << endl;
		}

	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer "
				"before setup_and_open_solution_database" << endl;
	}
	setup_and_open_solution_database(verbose_level - 2);
	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer "
				"after setup_and_open_solution_database" << endl;
	}
	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer "
				"before setup_and_open_level_database" << endl;
	}
	Iso->Sub->setup_and_open_level_database(verbose_level - 2);
	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer "
				"after setup_and_open_level_database" << endl;
	}


	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer "
				"before prepare_database_access" << endl;
	}
	Iso->Sub->prepare_database_access(Iso->level, verbose_level - 2);
	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer "
				"after prepare_database_access" << endl;
	}


	nb_flag_orbits = 0;
	flag_orbit_solution_first = NEW_int(N + 1);
	flag_orbit_solution_len = NEW_int(N);
	flag_orbit_of_solution = NEW_int(N);
	orbit_perm = NEW_int(N);
	orbit_perm_inv = NEW_int(N);
	schreier_vector = NEW_int(N);
	schreier_prev = NEW_int(N);

	// added Dec 25, 2012:

	first_flag_orbit_of_starter = NEW_int(Iso->Sub->nb_starter);
	nb_flag_orbits_of_starter = NEW_int(Iso->Sub->nb_starter);

	for (i = 0; i < N; i++) {
		schreier_vector[i] = -2;
		schreier_prev[i] = -1;
	}

	flag_orbit_solution_first[0] = 0;
	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		if (f_v) {
			cout << "substructure_lifting_data::orbits_of_stabilizer case "
					"i=" << i << " / " << Iso->Sub->nb_starter << endl;
		}

		first_flag_orbit_of_starter[i] = nb_flag_orbits;
		nb_flag_orbits_of_starter[i] = 0;

		data_structures_groups::vector_ge gens;



		Iso->Sub->load_strong_generators(Iso->level,
			i,
			gens, go, verbose_level - 2);
		if (f_v5) {
			cout << "substructure_lifting_data::orbits_of_stabilizer "
					"after load_strong_generators" << endl;
			cout << "substructure_lifting_data::orbits_of_stabilizer "
					"The stabilizer is a group of order "
					<< go << " with " << gens.len
					<< " strong generators" << endl;
			gens.print_with_given_action(cout, Iso->A_base);
		}

		f = starter_solution_first[i];
		l = starter_solution_len[i];
		if (f_v && ((i % 5000) == 0)) {
			cout << "substructure_lifting_data::orbits_of_stabilizer "
					"Case " << i
					<< " / " << Iso->Sub->nb_starter << endl;
		}
		if (f_vv) {
			cout << "substructure_lifting_data::orbits_of_stabilizer "
					"nb_orbits = "
					<< nb_flag_orbits << endl;
			cout << "substructure_lifting_data::orbits_of_stabilizer "
					"case " << i
					<< " starts at " << f << " with " << l
					<< " solutions" << endl;
		}
		if (gens.len == 0 /*O->nb_strong_generators == 0*/) {
			if (f_vv) {
				cout << "substructure_lifting_data::orbits_of_stabilizer "
						"the stabilizer is trivial" << endl;
			}
			for (j = 0; j < l; j++) {
				flag_orbit_solution_len[nb_flag_orbits] = 1;
				schreier_vector[f + j] = -1;
				flag_orbit_of_solution[f + j] = nb_flag_orbits;
				orbit_perm[f + j] = f + j;
				orbit_perm_inv[f + j] = f + j;
				nb_flag_orbits++;
				flag_orbit_solution_first[nb_flag_orbits] =
						flag_orbit_solution_first[nb_flag_orbits - 1] +
						flag_orbit_solution_len[nb_flag_orbits - 1];
				nb_flag_orbits_of_starter[i]++;
			}
		}
		else {
			if (f_vv) {
				cout << "substructure_lifting_data::orbits_of_stabilizer "
						"the stabilizer is non trivial" << endl;
			}
			if (starter_solution_len[i] != 0) {
				if (f_vv) {
					cout << "substructure_lifting_data::orbits_of_stabilizer "
							"before orbits_of_stabilizer_case" << endl;
				}
				orbits_of_stabilizer_case(i, gens, verbose_level - 2);
				if (f_vv) {
					cout << "substructure_lifting_data::orbits_of_stabilizer "
							"after orbits_of_stabilizer_case" << endl;
					cout << "substructure_lifting_data::orbits_of_stabilizer "
							"the " << l << " solutions in case " << i
							<< " fall into " << nb_flag_orbits - nb_orbits_prev
							<< " orbits" << endl;
				}
				nb_flag_orbits_of_starter[i] = nb_flag_orbits - nb_orbits_prev;
			}
		}
		if (f_v) {
			cout << "substructure_lifting_data::orbits_of_stabilizer "
					"Case " << i
					<< " / " << Iso->Sub->nb_starter << " finished, we found "
					<< nb_flag_orbits - nb_orbits_prev << " orbits : ";
			if (nb_flag_orbits - nb_orbits_prev) {
				data_structures::tally C;

				C.init(flag_orbit_solution_len + nb_orbits_prev,
						nb_flag_orbits - nb_orbits_prev, false, 0);
				C.print_naked(true /* f_backwards */);
				cout << endl;
			}
			else {
				cout << endl;
			}
		}
		if (false && f_vvvv) {
			cout << "i : orbit_perm : orbit_number : schreier_vector : "
					"schreier_prev" << endl;
			for (j = 0; j < l; j++) {
				cout << f + j << " : "
					<< orbit_perm[f + j] << " : "
					<< flag_orbit_of_solution[f + j] << " : "
					<< schreier_vector[f + j] << " : "
					<< schreier_prev[f + j] << endl;
			}
			cout << "j : orbit_fst : orbit_len" << endl;
			for (j = nb_orbits_prev; j < nb_flag_orbits; j++) {
				cout << j << " : " << flag_orbit_solution_first[j] << " : "
						<< flag_orbit_solution_len[j] << endl;
			}
			cout << j << " : " << flag_orbit_solution_first[j] << endl;
			if (flag_orbit_solution_first[nb_flag_orbits] != starter_solution_first[i + 1]) {
				cout << "orbit_fst[nb_orbits] != "
						"solution_first[i + 1]" << endl;
				cout << "orbit_fst[nb_orbits]="
						<< flag_orbit_solution_first[nb_flag_orbits] << endl;
				cout << "solution_first[i + 1]="
						<< starter_solution_first[i + 1] << endl;
				exit(1);
			}
		}
		nb_orbits_prev = nb_flag_orbits;
	} // next i

	if (flag_orbit_solution_first[nb_flag_orbits] != N) {
		cout << "orbit_fst[nb_orbits] != N" << endl;
		cout << "orbit_fst[nb_orbits]=" << flag_orbit_solution_first[nb_flag_orbits] << endl;
		cout << "N=" << N << endl;
		cout << "nb_orbits=" << nb_flag_orbits << endl;
		cout << "nb_starter=" << Iso->Sub->nb_starter << endl;
	}

	close_solution_database(verbose_level - 2);
	Iso->Sub->close_level_database(verbose_level - 2);

	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer "
				"We found " << nb_flag_orbits
				<< " orbits : ";
		data_structures::tally C;

		C.init(flag_orbit_solution_len, nb_flag_orbits, false, 0);
		C.print_naked(true /* f_backwards */);
		cout << endl;
	}

#if 0
	if (false && f_vv) {
		cout << "nb_starter=" << nb_starter << endl;
		cout << "i : solution_first[i] : solution_len[i]" << endl;
		for (i = 0; i < nb_starter; i++) {
			f = solution_first[i];
			l = solution_len[i];
			cout << setw(9) << i << setw(9) << f << setw(9) << l << endl;
		}
		cout << "nb_orbits=" << nb_orbits << endl;
		cout << "i : orbit_fst[i] : orbit_len[i]" << endl;
		for (i = 0; i < nb_orbits; i++) {
			cout << setw(9) << i << " "
				<< setw(9) << orbit_fst[i] << " "
				<< setw(9) << orbit_len[i] << endl;
		}
		cout << "N=" << N << endl;
		cout << "i : orbit_number[i] : orbit_perm[i] : schreier_vector[i] : "
				"schreier_prev[i]" << endl;
		for (i = 0; i < N; i++) {
			cout << setw(9) << i << " "
				<< setw(9) << orbit_number[i] << " "
				<< setw(9) << orbit_perm[i] << " "
				<< setw(9) << schreier_vector[i] << " "
				<< setw(9) << schreier_prev[i] << " "
				<< endl;
		}
	}
#endif

	if (f_v) {
		cout << "Number of flag orbits by starter orbit:" << endl;
		for (i = 0; i < Iso->Sub->nb_starter; i++) {
			cout << i << " : " << nb_flag_orbits_of_starter[i] << endl;
		}
		cout << "Total number of flag orbits: "
				<< first_flag_orbit_of_starter[Iso->Sub->nb_starter - 1]
					+ nb_flag_orbits_of_starter[Iso->Sub->nb_starter - 1] << endl;
	}

	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer "
				"before write_starter_nb_orbits" << endl;
	}
	write_starter_nb_orbits(verbose_level - 2);
	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer "
				"after write_starter_nb_orbits" << endl;
	}

	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer done" << endl;
	}
}

void substructure_lifting_data::orbits_of_stabilizer_case(
		int the_case,
		data_structures_groups::vector_ge &gens,
		int verbose_level)
{
	layer2_discreta::typed_objects::Vector v;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v4 = (verbose_level >= 4);
	int j, f, l, k, ff, ll;

	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				<< the_case << " / " << Iso->Sub->nb_starter << endl;
	}

	f = starter_solution_first[the_case];
	l = starter_solution_len[the_case];
	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"solution_first[the_case] = " << f << endl;
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"solution_len[the_case] = " << l << endl;
	}

	ring_theory::longinteger_object S_go;
	groups::sims *S;
	actions::action *A_induced;
	groups::schreier *Schreier;
	long int *sets;
	int h, p, prev, b, hdl;
	data_structures::sorting Sorting;

	sets = NEW_lint(l * Iso->size);
	S = NEW_OBJECT(groups::sims);
	A_induced = NEW_OBJECT(actions::action);
	Schreier = NEW_OBJECT(groups::schreier);


	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"generators as permutations (skipped)" << endl;
		//gens.print_as_permutation(cout);
	}
	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"before S->init" << endl;
	}
	S->init(Iso->A_base, verbose_level - 2);
	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"after S->init" << endl;
	}
	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"before S->init_generators" << endl;
	}
	S->init_generators(gens, verbose_level - 2);
	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"after S->init_generators" << endl;
	}
	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"before S->compute_base_orbits" << endl;
	}
	S->compute_base_orbits(verbose_level - 2);
	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"after S->compute_base_orbits" << endl;
	}
	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"before S->group_order" << endl;
	}
	S->group_order(S_go);
	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"The starter has a stabilizer of order "
				<< S_go << endl;
	}

	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"loading all solutions, number of solutions = " << l << endl;
	}
	for (j = 0; j < l; j++) {

		load_solution(f + j, sets + j * Iso->size, verbose_level - 1);
		if (false && f_vv) {
			cout << "solution " << j << "        : ";
			Lint_vec_print(cout, sets + j * Iso->size, Iso->size);
			cout << endl;
		}
		Sorting.lint_vec_heapsort(sets + j * Iso->size, Iso->size);
		if (false && f_vv) {
			cout << "solution " << j << " sorted : ";
			Lint_vec_print(cout, sets + j * Iso->size, Iso->size);
			cout << endl;
		}
	}
	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"loading all solutions done" << endl;
	}

	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"computing induced action on " << l << " sets of size " << Iso->size << endl;
	}

	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"before Iso->A->Induced_action->induced_action_on_sets" << endl;
	}

	A_induced = Iso->A->Induced_action->induced_action_on_sets(S, //K,
		l, Iso->size, sets, false /*true*/ /* A Betten 1/26/13*/,
		verbose_level /*- 2*/);

	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"computing induced action finished" << endl;
	}

#if 0
	A_induced->group_order(AA_go);
	A_induced->Kernel->group_order(K_go);
	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case orbit "
				<< nb_orbits << " induced action has order "
				<< AA_go << ", kernel has order " << K_go << endl;
	}
#endif

	if (f_vv) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"induced action computed" << endl;
		cout << "number of generators is " << gens.len << endl;
#if 0
		for (k = 0; k < gens.len; k++) {
			cout << k << " : ";
			//AA->element_print_as_permutation(gens.ith(k), cout);
			cout << endl;
		}
#endif
	}

	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"before AA->compute_all_point_orbits" << endl;
	}
	A_induced->compute_all_point_orbits(*Schreier, gens, 0 /*verbose_level - 4*/);
	//AA->all_point_orbits(*Schreier, verbose_level - 2);

	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				"after AA->compute_all_point_orbits" << endl;
	}
	if (f_v4) {
		Schreier->print_tables(cout, true);
	}

	for (k = 0; k < l; k++) {
		p = Schreier->orbit[k];
		prev = Schreier->prev[k];
		hdl = Schreier->label[k];
		//cout << "coset " << k << " point p=" << p
		// << " prev=" << prev << " label " << hdl << endl;
		if (prev != -1) {
			//A->element_retrieve(O->hdl_strong_generators[hdl],
			// A->Elt1, false);
			b = A_induced->Group_element->element_image_of(
					prev, gens.ith(hdl), false);
			//cout << "image of " << prev << " results in =" << b << endl;
			if (b != p) {
				cout << "b != p" << endl;
				exit(1);
			}
			if (!Iso->A->Group_element->check_if_transporter_for_set(
					gens.ith(hdl), Iso->size,
				sets + prev * Iso->size, sets + p * Iso->size,
				0 /*verbose_level - 2*/)) {
				exit(1);
			}
		}
	}
	for (k = 0; k < Schreier->nb_orbits; k++) {
		ff = Schreier->orbit_first[k];
		ll = Schreier->orbit_len[k];
		for (h = 0; h < ll; h++) {
			p = f + Schreier->orbit[ff + h];
			flag_orbit_of_solution[f + ff + h] = nb_flag_orbits;
			orbit_perm[f + ff + h] = p;
			orbit_perm_inv[p] = f + ff + h;
			schreier_vector[f + ff + h] = Schreier->label[ff + h];
			if (h == 0) {
				schreier_prev[f + ff + h] = -1;
			}
			else {
				schreier_prev[f + ff + h] =
						f + Schreier->prev[ff + h];
			}
		}
		flag_orbit_solution_len[nb_flag_orbits] = ll;
		nb_flag_orbits++;
		flag_orbit_solution_first[nb_flag_orbits] =
				flag_orbit_solution_first[nb_flag_orbits - 1] + ll;
	}

	FREE_lint(sets);
	FREE_OBJECT(S);
	FREE_OBJECT(A_induced);
	FREE_OBJECT(Schreier);
	if (f_v) {
		cout << "substructure_lifting_data::orbits_of_stabilizer_case "
				<< the_case << " / " << Iso->Sub->nb_starter << " done" << endl;
	}

}


void substructure_lifting_data::orbit_representative(
		int i, int &i0,
	int &orbit, int *transporter, int verbose_level)
// slow because it calls load_strong_generators
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int c, p, i_loc, l; //, hdl;
	int *Elt1, *Elt2;
	data_structures_groups::vector_ge gens;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "substructure_lifting_data::orbit_representative" << endl;
	}


	if (f_v) {
		cout << "substructure_lifting_data::orbit_representative "
				"before prepare_database_access" << endl;
	}
	Iso->Sub->prepare_database_access(Iso->level, verbose_level - 2);
	if (f_v) {
		cout << "substructure_lifting_data::orbit_representative "
				"after prepare_database_access" << endl;
	}




	Elt1 = Iso->Folding->orbit_representative_Elt1;
	Elt2 = Iso->Folding->orbit_representative_Elt2;
	c = starter_number_of_solution[i];
	if (f_v) {
		cout << "substructure_lifting_data::orbit_representative "
				"before load_strong_generators" << endl;
	}
	Iso->Sub->load_strong_generators(Iso->level, c,
		gens, go, verbose_level - 2);
	if (f_v) {
		cout << "substructure_lifting_data::orbit_representative "
				"after load_strong_generators" << endl;
	}
	Iso->A->Group_element->element_one(transporter, false);
	if (f_vv) {
		cout << "substructure_lifting_data::orbit_representative "
				"i=" << i << endl;
	}
	while (true) {
		i_loc = orbit_perm_inv[i];
		p = schreier_prev[i_loc];
		if (f_vv) {
			cout << "substructure_lifting_data::orbit_representative "
					"i=" << i << " i_loc=" << i_loc
					<< " p=" << p << endl;
		}
		if (p == -1) {
			i0 = i;
			orbit = flag_orbit_of_solution[i_loc];
			break;
		}
		l = schreier_vector[i_loc];
		//cout << "l=" << l << endl;
		//hdl = O->hdl_strong_generators[l];
		//A->element_retrieve(hdl, Elt1, false);
		Iso->A->Group_element->element_invert(gens.ith(l), Elt2, false);
		Iso->A->Group_element->element_mult(transporter, Elt2, Elt1, false);
		Iso->A->Group_element->element_move(Elt1, transporter, false);
		i = p;
	}
	if (f_v) {
		cout << "substructure_lifting_data::orbit_representative "
				"The representative of solution " << i << " is "
				<< i0 << " in orbit " << orbit << endl;
	}
	if (f_v) {
		cout << "substructure_lifting_data::orbit_representative done" << endl;
	}
}

void substructure_lifting_data::test_orbit_representative(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int r, r0, orbit, k;
	long int data1[1000];
	long int data2[1000];
	int *transporter;

	if (f_v) {
		cout << "substructure_lifting_data::test_orbit_representative" << endl;
	}
	transporter = NEW_int(Iso->A->elt_size_in_int);

	setup_and_open_solution_database(verbose_level - 1);

	for (k = 0; k < N; k++) {
		r = k;
		//r = random_integer(N);
		//cout << "k=" << k << " r=" << r << endl;

		load_solution(r, data1, verbose_level - 1);

		orbit_representative(r, r0, orbit,
				transporter, verbose_level);
		if (r != r0) {
			cout << "k=" << k << " r=" << r << " r0=" << r0 << endl;
		}

		load_solution(r0, data2, verbose_level - 1);
		if (!Iso->A->Group_element->check_if_transporter_for_set(transporter,
				Iso->size, data1, data2, 0 /*verbose_level*/)) {
			exit(1);
		}
	}

	close_solution_database(verbose_level - 1);
	FREE_int(transporter);
	if (f_v) {
		cout << "substructure_lifting_data::test_orbit_representative done" << endl;
	}
}

void substructure_lifting_data::test_identify_solution(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int r, r0, id, id0;
	long int data1[1000];
	long int data2[1000];
	int perm[1000];
	int i, k;
	int *transporter;
	combinatorics::combinatorics_domain Combi;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "substructure_lifting_data::test_identify_solution" << endl;
	}
	transporter = NEW_int(Iso->A->elt_size_in_int);


	setup_and_open_solution_database(verbose_level - 1);

	for (k = 0; k < 10; k++) {
		r = Os.random_integer(nb_flag_orbits);
		id = orbit_perm[flag_orbit_solution_first[r]];
		if (schreier_prev[flag_orbit_solution_first[r]] != -1) {
			cout << "schreier_prev[orbit_fst[r]] != -1" << endl;
			exit(1);
		}
		//cout << "k=" << k << " r=" << r << endl;

		load_solution(id, data1, verbose_level - 1);
		Combi.random_permutation(perm, Iso->size);
		for (i = 0; i < Iso->size; i++) {
			data2[i] = data1[perm[i]];
		}

		int f_failure_to_find_point;
		r0 = Iso->Folding->identify_solution(data2, transporter,
				Iso->Sub->f_use_implicit_fusion, f_failure_to_find_point,
				verbose_level - 2);

		if (f_failure_to_find_point) {
			cout << "f_failure_to_find_point" << endl;
		}
		else {
			cout << "k=" << k << " r=" << r << " r0=" << r0 << endl;
			id0 = orbit_perm[flag_orbit_solution_first[r0]];

			load_solution(id0, data1, verbose_level - 1);
			if (!Iso->A->Group_element->check_if_transporter_for_set(transporter,
					Iso->size, data2, data1, 0 /*verbose_level*/)) {
				cout << "test_identify_solution, "
						"check fails, stop" << endl;
				exit(1);
			}
		}
	}

	close_solution_database(verbose_level - 1);
	FREE_int(transporter);
	if (f_v) {
		cout << "substructure_lifting_data::test_identify_solution done" << endl;
	}
}

void substructure_lifting_data::setup_and_open_solution_database(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "substructure_lifting_data::setup_and_open_solution_database" << endl;
	}
	if (DB_sol) {
		layer2_discreta::typed_objects::freeobject(DB_sol);
		DB_sol = NULL;
	}
	DB_sol = (layer2_discreta::typed_objects::database *)
			layer2_discreta::typed_objects::callocobject(
					layer2_discreta::typed_objects::DATABASE);
	DB_sol->change_to_database();

	init_DB_sol(0 /*verbose_level - 1*/);

	DB_sol->open(0 /*verbose_level - 1*/);
}

void substructure_lifting_data::setup_and_create_solution_database(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "substructure_lifting_data::setup_and_create_solution_database" << endl;
	}
	if (DB_sol) {
		layer2_discreta::typed_objects::freeobject(DB_sol);
		DB_sol = NULL;
	}
	DB_sol = (layer2_discreta::typed_objects::database *)
			layer2_discreta::typed_objects::callocobject(
					layer2_discreta::typed_objects::DATABASE);
	DB_sol->change_to_database();

	init_DB_sol(0 /*verbose_level - 1*/);

	DB_sol->create(0 /*verbose_level - 1*/);
}

void substructure_lifting_data::close_solution_database(int verbose_level)
{
	DB_sol->close(0/*verbose_level - 1*/);
}

void substructure_lifting_data::init_DB_sol(int verbose_level)
// We assume that the starter is of size 'level' and that
// fields 4,..., 4+level-1 are the starter values
{
	int f_v = (verbose_level >= 1);
	layer2_discreta::typed_objects::database &D = *DB_sol;
	layer2_discreta::typed_objects::btree B1, B2, B3, B4;
	int f_compress = true;
	int f_duplicatekeys = true;
	int i;

	if (f_v) {
		cout << "substructure_lifting_data::init_DB_sol" << endl;
	}
	//cout << "substructure_lifting_data::init_DB_sol before D.init" << endl;
	D.init(fname_db1.c_str(), layer2_discreta::typed_objects::VECTOR, f_compress);


	//cout << "substructure_lifting_data::init_DB_sol before B1.init" << endl;
	B1.init(fname_db2.c_str(), f_duplicatekeys, 0 /* btree_idx */);
	B1.add_key_int4(0, 0);
		// the index of the starter
	B1.add_key_int4(1, 0);
		// the number of this solution within the solutions
		// of the same starter
	D.btree_access().append(B1);


	//cout << "substructure_lifting_data::init_DB_sol before B2.init" << endl;
	B2.init(fname_db3.c_str(), f_duplicatekeys, 1 /* btree_idx */);
		// entries 4, 5, ... 4 + level - 1 are the starter values
	for (i = 0; i < Iso->level; i++) {
		B2.add_key_int4(4 + i, 0);
	}
	//B2.add_key_int4(3, 0);
	//B2.add_key_int4(4, 0);
	//B2.add_key_int4(5, 0);
	//B2.add_key_int4(6, 0);
	//B2.add_key_int4(7, 0);
	//B2.add_key_int4(8, 0);
	D.btree_access().append(B2);


	//cout << "substructure_lifting_data::init_DB_sol before B3.init" << endl;
	B3.init(fname_db4.c_str(), f_duplicatekeys, 2 /* btree_idx */);
	B3.add_key_int4(2, 0);
		// the id
	D.btree_access().append(B3);


	B4.init(fname_db5.c_str(), f_duplicatekeys, 3 /* btree_idx */);
	B4.add_key_int4(0, 0);
		// the index of the starter
	B4.add_key_int4(3, 0);
		// the hash value
	D.btree_access().append(B4);


	//cout << "substructure_lifting_data::init_DB_sol done" << endl;
}

void substructure_lifting_data::add_solution_to_database(
		long int *data,
	int nb, int id, int no, int nb_solutions, long int h, uint_4 &datref,
	int print_mod, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vvv = (verbose_level >= 3);


	if (f_v) {
		cout << "substructure_lifting_data::add_solution_to_database" << endl;
	}

	layer2_discreta::typed_objects::Vector v;
	int j;

	//h = int_vec_hash_after_sorting(data + 1, size);
	v.m_l_n(4 + Iso->size);
	v.m_ii(0, data[0]); // starter number
	v.m_ii(1, nb); // solution number within this starter
	v.m_ii(2, id); // global solution number
	v.m_ii(3, h); // the hash number
	for (j = 0; j < Iso->size; j++) {
		v.m_ii(4 + j, data[1 + j]);
	}
	if (f_vvv || ((no % print_mod) == 0)) {
		cout << "Solution no " << no << " / " << nb_solutions
				<< " starter case " << data[0] << " nb " << nb
				<< " id=" << id << " : " << v << " : " << endl;
	}

	DB_sol->add_object_return_datref(v, datref, 0/*verbose_level - 3*/);
	if (f_v) {
		cout << "substructure_lifting_data::add_solution_to_database done" << endl;
	}
}

void substructure_lifting_data::load_solution(
		int id, long int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "substructure_lifting_data::load_solution id=" << id << endl;
	}
	int i, j;
	long int datref;
	layer2_discreta::typed_objects::Vector v;
	//int verbose_level = 0;

	if (f_use_table_of_solutions) {
		for (j = 0; j < Iso->size; j++) {
			data[j] = table_of_solutions[id * Iso->size + j];
		}
		return;
	}
	//DB_sol->get_object_by_unique_int4(2, id, v, verbose_level);
	datref = id_to_datref[id];
	DB_sol->get_object((uint_4) datref, v, 0/*verbose_level*/);

	//cout << v << endl;

	for (i = 0; i < Iso->size; i++) {
		data[i] = v.s_ii(4 + i);
	}

	if (f_v) {
		cout << "substructure_lifting_data::load_solution done" << endl;
	}
}

void substructure_lifting_data::load_solution_by_btree(
		int btree_idx, int idx, int &id, long int *data)
{
	//int i;
	layer2_discreta::typed_objects::Vector v;

	cout << "substructure_lifting_data::load_solution_by_btree" << endl;
	exit(1);
#if 0
	DB_sol->ith_object(idx, btree_idx, v, 0 /*verbose_level*/);
	for (i = 0; i < size; i++) {
		data[i] = v.s_ii(4 + i);
	}
	id = v.s_ii(2);
#endif
}


void substructure_lifting_data::count_solutions(
		int nb_files,
		std::string *fname,
		int *List_of_cases, int *&Nb_sol_per_file,
		int f_get_statistics,
		int f_has_final_test_function,
		int (*final_test_function)(
				long int *data, int sz,
				void *final_test_data, int verbose_level),
		void *final_test_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	//int total_days, total_hours, total_minutes;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "substructure_lifting_data::count_solutions" << endl;
	}
	if (Iso->Sub->nb_starter == 0) {
		cout << "substructure_lifting_data::count_solutions "
				"nb_starter == 0" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions "
				"nb_starter = " << Iso->Sub->nb_starter << endl;
	}
	starter_solution_first = NEW_int(Iso->Sub->nb_starter + 1);
	starter_solution_len = NEW_int(Iso->Sub->nb_starter);

	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		starter_solution_first[i] = 0;
		starter_solution_len[i] = 0;
	}

	stats_nb_backtrack = NEW_int(Iso->Sub->nb_starter);
	stats_nb_backtrack_decision = NEW_int(Iso->Sub->nb_starter);
	stats_graph_size = NEW_int(Iso->Sub->nb_starter);
	stats_time = NEW_int(Iso->Sub->nb_starter);

	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		stats_nb_backtrack[i] = -1;
		stats_nb_backtrack_decision[i] = -1;
		stats_graph_size[i] = -1;
		stats_time[i] = -1;
	}

	int a, b;

	if (f_v) {
		cout << "substructure_lifting_data::count_solutions "
				"before Fio.count_solutions_in_list_of_files" << endl;
	}
	Fio.count_solutions_in_list_of_files(
			nb_files, fname, List_of_cases, Nb_sol_per_file,
			Iso->size,
			f_has_final_test_function,
			final_test_function,
			final_test_data,
			verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions "
				"after Fio.count_solutions_in_list_of_files" << endl;
	}

	for (i = 0; i < nb_files; i++) {
		a = List_of_cases[i];
		b = Nb_sol_per_file[i];
		starter_solution_len[a] = b;
	}
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions "
				"after Fio.count_solutions_in_list_of_files" << endl;
		cout << "case_len: ";
		Int_vec_print(cout, starter_solution_len, Iso->Sub->nb_starter);
		cout << "substructure_lifting_data::count_solutions "
				"solution_len[]:" << endl;
		for (i = 0; i < Iso->Sub->nb_starter; i++) {
			cout << i << " : " << starter_solution_len[i] << endl;
		}
	}


	starter_solution_first[0] = 0;
	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		starter_solution_first[i + 1] =
				starter_solution_first[i] + starter_solution_len[i];
	}



	N = starter_solution_first[Iso->Sub->nb_starter];
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions "
				"N=" << N << endl;
	}

	init_starter_number(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions "
				"after init_starter_number" << endl;
	}

	write_solution_first_and_len(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions "
				"after write_solution_first_and_len" << endl;
	}

	if (f_get_statistics) {
		get_statistics(nb_files, fname, List_of_cases, verbose_level);
		write_statistics();
		evaluate_statistics(verbose_level);
	}
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions done" << endl;
	}
}

void substructure_lifting_data::add_solutions_to_database(
		long int *Solutions,
	int the_case, int nb_solutions, int nb_solutions_total,
	int print_mod, int &no,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int u, v;
	long int *data;
	data_structures::data_structures_global Data;

	if (f_v) {
		cout << "substructure_lifting_data::add_solutions_to_database "
				"case " << the_case << endl;
	}
	data = NEW_lint(Iso->size + 1);
	for (u = 0; u < nb_solutions; u++) {

		uint_4 datref;
		long int hs, id;

		data[0] = the_case;
		for (v = 0; v < Iso->size; v++) {
			data[1 + v] = Solutions[u * Iso->size + v];
		}
		id = starter_solution_first[data[0]] + u;

		hs = Data.lint_vec_hash_after_sorting(data + 1, Iso->size);
		if ((u % 1000) == 0) {
			if (f_vv) {
				cout << "substructure_lifting_data::add_solutions_to_database "
						"case " << the_case << " u=" << u << " id=" << id
						<< " hs=" << hs << " no=" << no << endl;
			}
		}


		add_solution_to_database(data,
			u, id, no, nb_solutions_total, hs, datref,
			print_mod, verbose_level - 2);
			// in isomorph_database.cpp

		id_to_datref[id] = datref;
		id_to_hash[id] = hs;
		hash_vs_id_hash[id] = hs;
		hash_vs_id_id[id] = id;

		no++;
	}
	FREE_lint(data);
	if (f_v) {
		cout << "substructure_lifting_data::add_solutions_to_database "
				"case " << the_case << " done, added "
				<< nb_solutions << " solutions; "
						"updated database length is " << no << endl;
	}
}


void substructure_lifting_data::init_solutions(long int **Solutions, int *Nb_sol,
	int verbose_level)
// Solutions[nb_starter], Nb_sol[nb_starter]
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "substructure_lifting_data::init_solutions "
				"nb_starter = " << Iso->Sub->nb_starter << endl;
	}
	starter_solution_first = NEW_int(Iso->Sub->nb_starter + 1);
	starter_solution_len = NEW_int(Iso->Sub->nb_starter);
	N = 0;
	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		starter_solution_first[i] = 0;
		starter_solution_len[i] = Nb_sol[i];
		N += starter_solution_len[i];
	}
	if (f_v) {
		cout << "substructure_lifting_data::init_solutions "
				"N = " << N << endl;
	}
	starter_solution_first[0] = 0;
	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		starter_solution_first[i + 1] =
				starter_solution_first[i] + starter_solution_len[i];
	}
	if (starter_solution_first[Iso->Sub->nb_starter] != N) {
		cout << "substructure_lifting_data::init_solutions "
				"solution_first[nb_starter] != N" << endl;
		exit(1);
	}

	init_starter_number(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::init_solutions "
				"after init_starter_number" << endl;
	}

	write_solution_first_and_len(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::init_solutions "
				"after write_solution_first_and_len" << endl;
	}


	setup_and_create_solution_database(0/*verbose_level - 1*/);

	int h;
	int no = 0;
	int print_mod = 1000;

	id_to_datref_allocate(verbose_level);

	if (f_v) {
		cout << "substructure_lifting_data::init_solutions "
				"before add_solutions_to_database" << endl;
	}

	for (h = 0; h < Iso->Sub->nb_starter; h++) {
		if (starter_solution_len[h]) {
			add_solutions_to_database(
					Solutions[h],
				h,
				starter_solution_len[h], N, print_mod, no,
				verbose_level);
		}
	}

	write_hash_and_datref_file(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::init_solutions "
				"written hash and datref file" << endl;
		cout << "substructure_lifting_data::init_solutions "
				"sorting hash_vs_id_hash" << endl;
	}
	{
		data_structures::tally_lint C;

		C.init(hash_vs_id_hash, N, true, 0);
		cout << "substructure_lifting_data::init_solutions "
				"Classification of hash values:" << endl;
		C.print(false /*f_backwards*/);
	}
	Sorting.lint_vec_heapsort_with_log(
			hash_vs_id_hash, hash_vs_id_id, N);
	if (f_v) {
		cout << "substructure_lifting_data::init_solutions "
				"after sorting hash_vs_id_hash" << endl;
	}

	close_solution_database(0 /*verbose_level - 1*/);



	if (f_v) {
		cout << "substructure_lifting_data::init_solutions done" << endl;
	}
}

void substructure_lifting_data::count_solutions_from_clique_finder_case_by_case(
		int nb_files,
		long int *list_of_cases, std::string *fname,
		int verbose_level)
// Called from isomorph_read_solution_files_from_clique_finder
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, h;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "substructure_lifting_data::count_solutions_from_clique_finder_case_by_case "
				"nb_starter = " << Iso->Sub->nb_starter << " nb_files=" << nb_files << endl;
	}
	starter_solution_first = NEW_int(Iso->Sub->nb_starter + 1);
	starter_solution_len = NEW_int(Iso->Sub->nb_starter);

	Int_vec_zero(starter_solution_len, Iso->Sub->nb_starter);
	N = 0;
	for (i = 0; i < nb_files; i++) {
		int nb_solutions;

		Fio.count_number_of_solutions_in_file(fname[i],
			nb_solutions,
			verbose_level - 2);

		if (f_vv) {
			cout << "substructure_lifting_data::count_solutions_from_clique_finder_case_by_case file "
					<< i << " / " << nb_files << " = "
					<< fname[i] << " read, nb_solutions="
					<< nb_solutions << endl;
		}

		h = list_of_cases[i];
		starter_solution_len[h] = nb_solutions;

		N += nb_solutions;
	}
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions_from_clique_finder_case_by_case "
				"done counting solutions, "
				"total number of solutions = " << N << endl;
		cout << "h : solution_len[h]" << endl;
		for (h = 0; h < Iso->Sub->nb_starter; h++) {
			cout << h << " : " << starter_solution_len[h] << endl;
		}
	}
	starter_solution_first[0] = 0;
	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		starter_solution_first[i + 1] = starter_solution_first[i] + starter_solution_len[i];
	}
	if (starter_solution_first[Iso->Sub->nb_starter] != N) {
		cout << "substructure_lifting_data::count_solutions_from_clique_finder_case_by_case "
				"solution_first[nb_starter] != N" << endl;
		exit(1);
	}

	init_starter_number(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions_from_clique_finder_case_by_case "
				"after init_starter_number" << endl;
	}

	write_solution_first_and_len(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions_from_clique_finder_case_by_case "
				"done" << endl;
	}
}



void substructure_lifting_data::count_solutions_from_clique_finder(
		int nb_files,
		std::string *fname, int verbose_level)
// Called from isomorph_read_solution_files_from_clique_finder
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, h, c, n;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "substructure_lifting_data::count_solutions_from_clique_finder "
				"nb_starter = " << Iso->Sub->nb_starter << " nb_files="
				<< nb_files << endl;
	}
	starter_solution_first = NEW_int(Iso->Sub->nb_starter + 1);
	starter_solution_len = NEW_int(Iso->Sub->nb_starter);
	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		starter_solution_first[i] = 0;
		starter_solution_len[i] = 0;
	}
	N = 0;
	for (i = 0; i < nb_files; i++) {
		int *nb_solutions;
		int *case_nb;
		int nb_cases;

		Fio.count_number_of_solutions_in_file_by_case(fname[i],
			nb_solutions, case_nb, nb_cases,
			verbose_level - 2);

		if (f_vv) {
			cout << "substructure_lifting_data::count_solutions_from_clique_finder "
					"file " << i << " / " << nb_files << " = " << fname[i]
					<< " read, nb_cases=" << nb_cases << endl;
		}

		for (h = 0; h < nb_cases; h++) {
			c = case_nb[h];
			n = nb_solutions[h];
			starter_solution_len[c] = n;
			N += n;
		}
		FREE_int(nb_solutions);
		FREE_int(case_nb);
	}
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions_from_clique_finder "
				"done counting solutions, total number of "
				"solutions = " << N << endl;
		cout << "h : solution_len[h]" << endl;
		for (h = 0; h < Iso->Sub->nb_starter; h++) {
			cout << h << " : " << starter_solution_len[h] << endl;
		}
	}
	starter_solution_first[0] = 0;
	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		starter_solution_first[i + 1] =
				starter_solution_first[i] + starter_solution_len[i];
	}
	if (starter_solution_first[Iso->Sub->nb_starter] != N) {
		cout << "substructure_lifting_data::count_solutions_from_clique_finder "
				"solution_first[nb_starter] != N" << endl;
		exit(1);
	}

	init_starter_number(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions_from_clique_finder "
				"after init_starter_number" << endl;
	}

	write_solution_first_and_len(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::count_solutions_from_clique_finder "
				"after write_solution_first_and_len" << endl;
	}
}



void substructure_lifting_data::read_solutions_from_clique_finder_case_by_case(
		int nb_files, long int *list_of_cases, std::string *fname,
		int verbose_level)
// Called from isomorph_read_solution_files_from_clique_finder
// Called after count_solutions_from_clique_finder
// We assume that N, the number of solutions is known
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	data_structures::sorting Sorting;
	orbiter_kernel_system::file_io Fio;

	int i, no = 0;
	int print_mod = 1000;


	if (f_v) {
		cout << "substructure_lifting_data::read_solutions_from_clique_finder_case_by_case "
				"nb_files=" << nb_files
				<< " N=" << N << endl;
	}


	setup_and_create_solution_database(0/*verbose_level - 1*/);

	if (f_v) {
		cout << "substructure_lifting_data::read_solutions_from_clique_finder_"
				"case_by_case after setup_and_create_solution_"
				"database" << endl;
	}

	id_to_datref_allocate(verbose_level);

	for (i = 0; i < nb_files; i++) {

		if (f_vv) {
			cout << "substructure_lifting_data::read_solutions_from_clique_finder_"
					"case_by_case, file " << i << " / " << nb_files
					<< " which is " << fname[i] << endl;
		}
		int nb_solutions;
		long int *Solutions;

		Fio.read_solutions_from_file(fname[i],
			nb_solutions,
			Solutions, Iso->size /* solution_size */,
			verbose_level - 2);

		if (f_vv) {
			cout << "substructure_lifting_data::read_solutions_from_clique_finder_"
					"case_by_case file " << fname[i] << " number of "
							"solutions read: " << nb_solutions << endl;
		}

		add_solutions_to_database(Solutions,
			list_of_cases[i], nb_solutions, N, print_mod, no,
			verbose_level);


		FREE_lint(Solutions);
	}


	write_hash_and_datref_file(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::read_solutions_from_clique_finder_case_by_case "
				"written hash and datref file" << endl;
		cout << "substructure_lifting_data::read_solutions_from_clique_finder_case_by_case "
				"sorting hash_vs_id_hash" << endl;
	}
	{
		data_structures::tally_lint C;

		C.init(hash_vs_id_hash, N, true, 0);
		cout << "substructure_lifting_data::read_solutions_from_clique_finder_case_by_case "
				"Classification of hash values:" << endl;
		C.print(false /*f_backwards*/);
	}
	Sorting.lint_vec_heapsort_with_log(hash_vs_id_hash, hash_vs_id_id, N);
	if (f_v) {
		cout << "substructure_lifting_data::read_solutions_from_clique_finder_case_by_case "
				"after sorting hash_vs_id_hash" << endl;
	}

	close_solution_database(0 /*verbose_level - 1*/);


	if (f_v) {
		cout << "substructure_lifting_data::read_solutions_from_clique_finder_case_by_case "
				"done" << endl;
	}
}


void substructure_lifting_data::read_solutions_from_clique_finder(
		int nb_files,
		std::string *fname,
		int *substructure_case_number, int *Nb_sol_per_file,
		int verbose_level)
// Called from isomorph_read_solution_files_from_clique_finder
// Called after count_solutions_from_clique_finder
// We assume that N, the number of solutions is known
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	int i, no = 0;
	//int *data;
	int print_mod = 1000;
	int nb_solutions_total;
	data_structures::sorting Sorting;
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "substructure_lifting_data::read_solutions_from_clique_finder "
				"nb_files=" << nb_files << " N=" << N << endl;
	}


	setup_and_create_solution_database(0/*verbose_level - 1*/);

	//data = NEW_int(size + 1);

	if (f_v) {
		cout << "substructure_lifting_data::read_solutions_from_clique_finder "
				"after setup_and_create_solution_database" << endl;
	}

	id_to_datref_allocate(verbose_level);

	nb_solutions_total = 0;


	for (i = 0; i < nb_files; i++) {

		if (f_vv) {
			cout << "substructure_lifting_data::read_solutions_from_clique_finder "
					"reading file " << fname[i] << endl;
		}
		int the_case;
		int nb_solutions;
		//int *case_nb;
		//int nb_cases;
		long int *Solutions;
		int solution_size;
		//int the_case, h; //, u, v;
		//int nb_solutions_total;
		//string fname_summary;
		//char extension[1000];
		data_structures::string_tools ST;

#if 0
		fname_summary = fname[i];
		ST.chop_off_extension_if_present(fname_summary, ".txt");
		fname_summary += "_summary.csv";

		Fio.count_number_of_solutions_in_file_by_case(fname[i],
			nb_solutions, case_nb, nb_cases,
			verbose_level - 2);
#endif


		the_case = substructure_case_number[i];

		Fio.read_solutions_from_file_and_get_solution_size(fname[i],
			nb_solutions, Solutions, solution_size,
			verbose_level);

		if (f_vv) {
			cout << "substructure_lifting_data::read_solutions_from_clique_finder "
					"The file " << fname[i] << " case number = "
					<< the_case << " contains "
					<< nb_solutions << " solutions" << endl;
		}

		if (nb_solutions != Nb_sol_per_file[i]) {
			cout << "substructure_lifting_data::read_solutions_from_clique_finder nb_solutions != Nb_sol_per_file[i]" << endl;
			exit(1);
		}

#if 0
		Fio.int_vecs_write_csv(case_nb, nb_solutions, nb_cases,
			fname_summary, "Case_nb", "Nb_sol");
#endif

		nb_solutions_total += nb_solutions;

#if 0
		std::vector<std::vector<int> > &Solutions;
		int solution_size;

		Fio.read_solutions_from_file_size_is_known(fname[i],
				Solutions, solution_size,
				verbose_level);
#endif


#if 0
		Fio.read_solutions_from_file_by_case(fname[i],
			nb_solutions, //case_nb, nb_cases,
			Solutions, size /* solution_size */,
			verbose_level - 2);
#endif

		if (f_vv) {
			cout << "substructure_lifting_data::read_solutions_from_clique_finder "
					"adding solutions" << endl;
		}


		add_solutions_to_database(
				Solutions,
			the_case, nb_solutions, nb_solutions_total,
			print_mod, no,
			verbose_level);


		if (f_vv) {
			cout << "substructure_lifting_data::read_solutions_from_clique_finder "
					"file " << fname[i] << " done number = " << no << endl;
		}
	}

	if (no != N) {
		cout << "substructure_lifting_data::read_solutions_from_clique_finder no != N" << endl;
		cout << "no=" << no << endl;
		cout << "N=" << N << endl;
		exit(1);
	}


	write_hash_and_datref_file(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::read_solutions_from_clique_finder "
				"written hash and datref file" << endl;
		cout << "substructure_lifting_data::read_solutions_from_clique_finder "
				"sorting hash_vs_id_hash" << endl;
	}
	{
		data_structures::tally_lint C;

		C.init(hash_vs_id_hash, N, true, 0);
		cout << "substructure_lifting_data::read_solutions_from_clique_finder "
				"Classification of hash values:" << endl;
		C.print(false /*f_backwards*/);
	}
	Sorting.lint_vec_heapsort_with_log(hash_vs_id_hash, hash_vs_id_id, N);
	if (f_v) {
		cout << "substructure_lifting_data::read_solutions_from_clique_finder "
				"after sorting hash_vs_id_hash" << endl;
	}

	close_solution_database(0 /*verbose_level - 1*/);

	//FREE_int(data);

	if (f_v) {
		cout << "substructure_lifting_data::read_solutions_from_clique_finder done" << endl;
	}
}


#define MY_BUFSIZE 1000000


void substructure_lifting_data::build_up_database(
		int nb_files,
	std::string *fname,
	int f_has_final_test_function,
	int (*final_test_function)(
			long int *data, int sz,
			void *final_test_data, int verbose_level),
	void *final_test_data,
	int verbose_level)
// We assume that N, the number of solutions is known
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	int i, nb_total = 0, j, nb = 0, prev = 0, id = 0;
	long int a;
	long int h;
	char *p_buf;
	long int data[1000];
	char buf[MY_BUFSIZE];
	int print_mod = 1000;
	uint_4 datref;
	int nb_fail = 0;
	data_structures::sorting Sorting;
	orbiter_kernel_system::file_io Fio;
	data_structures::data_structures_global Data;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "substructure_lifting_data::build_up_database "
				"nb_files=" << nb_files << " N=" << N << endl;
	}


	setup_and_create_solution_database(verbose_level - 1);


	if (f_v) {
		cout << "substructure_lifting_data::build_up_database "
				"after setup_and_create_solution_database" << endl;
	}

	id_to_datref_allocate(verbose_level);

	for (i = 0; i < nb_files; i++) {

		ifstream f(fname[i]);
		if (f_v) {
			cout << "substructure_lifting_data::build_up_database "
					"reading file " << fname[i] << " of size "
					<< Fio.file_size(fname[i]) << endl;
		}

		while (true) {

			if (f.eof()) {
				break;
			}

#if 0
			{
				string S;
				int l;
				getline(f, S);
				l = S.length();
				if (f_vvv) {
					cout << "substructure_lifting_data::build_up_database "
							"read line of length " << l << " : " << S << endl;
				}
				for (j = 0; j < l; j++) {
					buf[j] = S[j];
				}
				buf[l] = 0;
			}
#else
			{
				f.getline(buf, MY_BUFSIZE, '\n');
			}
#endif
			if (f_vvv) {
				cout << "substructure_lifting_data::build_up_database "
						"read: " << buf << endl;
			}

			p_buf = buf;


			ST.s_scan_lint(&p_buf, &a);

			data[0] = a;
				// starter number;

			if (data[0] != prev) {
				prev = data[0];
				nb = 0;
			}
			if (a == -1) {
				break;
			}

			for (j = 0; j < Iso->size; j++) {
				ST.s_scan_lint(&p_buf, &a);
				data[j + 1] = a;
			}


			if (f_has_final_test_function) {
				if (!(*final_test_function)(data + 1, Iso->size,
						final_test_data, verbose_level - 1)) {
					if (f_vvv) {
						cout << "substructure_lifting_data::build_up_database "
								"solution fails the final test, "
								"skipping" << endl;
					}
					nb_fail++;
					continue;
				}
				else {
					cout << nb_total << " : " << data[0] << " : ";
					Lint_vec_print(cout, data + 1, Iso->size);
					cout << endl;
				}
			}

			id = starter_solution_first[data[0]] + nb;


			h = Data.lint_vec_hash_after_sorting(data + 1, Iso->size);

			add_solution_to_database(data,
				nb, id, nb_total, N, h, datref, print_mod,
				verbose_level - 3);

			id_to_datref[id] = datref;
			id_to_hash[id] = h;
			hash_vs_id_hash[id] = h;
			hash_vs_id_id[id] = id;

			nb_total++; // number of solutions total
			nb++; // number of solutions within the starter case
		} // end while
		if (f_v) {
			cout << "substructure_lifting_data::build_up_database "
					"finished reading file " << fname[i]
					<< " nb=" << nb << " nb_total=" << nb_total << endl;
		}
	}	// next i


	if (f_v) {
		cout << "substructure_lifting_data::build_up_database "
				"finished number of solutions total = " << nb_total
				<< " nb_fail = " << nb_fail << endl;
	}


	write_hash_and_datref_file(verbose_level);
	if (f_v) {
		cout << "substructure_lifting_data::build_up_database "
				"written hash and datref file" << endl;
		cout << "substructure_lifting_data::build_up_database "
				"sorting hash_vs_id_hash" << endl;
	}
	{
		data_structures::tally_lint C;

		C.init(hash_vs_id_hash, N, true, 0);
		cout << "substructure_lifting_data::build_up_database "
				"Classification of hash values:" << endl;
		C.print(false /*f_backwards*/);
	}
	Sorting.lint_vec_heapsort_with_log(
			hash_vs_id_hash, hash_vs_id_id, N);
	if (f_v) {
		cout << "substructure_lifting_data::build_up_database "
				"after sorting hash_vs_id_hash" << endl;
	}

	close_solution_database(verbose_level - 1);
	if (f_v) {
		cout << "substructure_lifting_data::build_up_database done" << endl;
	}
}

void substructure_lifting_data::get_statistics(int nb_files,
		std::string *fname, int *List_of_cases,
		int verbose_level)
{
	int i, the_case, nb_sol, nb_backtrack, nb_backtrack_decision;
	int nb_points, dt[5], dt_total;
	int f_v = (verbose_level >= 1);
	string fname_summary;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "substructure_lifting_data::get_statistics: reading "
				<< nb_files << " files" << endl;
	}
#if 0
	for (i = 0; i < nb_files; i++) {
		cout << fname[i] << endl;
	}
#endif


	for (i = 0; i < nb_files; i++) {


		fname_summary.assign(fname[i]);

		// ToDo:
#if 0
		if (strcmp(fname_summary + strlen(fname_summary) - 4, ".txt")) {
			cout << "substructure_lifting_data::get_statistics: "
					"file name does not end in .txt" << endl;
			return;
			}
		strcpy(fname_summary + strlen(fname_summary) - 4, ".summary");
#endif

		ifstream fp(fname_summary);

		if (f_v) {
			cout << "substructure_lifting_data::get_statistics "
					"file " << i << " / " << nb_files
					<< ", reading file " << fname_summary
					<< " of size " << Fio.file_size(fname[i]) << endl;
		}
		if (Fio.file_size(fname_summary) <= 0) {
			cout << "problems reading file " << fname_summary << endl;
			return;
		}
		while (true) {
			fp >> the_case;
			if (the_case == -1)
				break;
			fp >> nb_sol;
			fp >> nb_backtrack;
			fp >> nb_backtrack_decision;
			fp >> nb_points;
			fp >> dt[0];
			fp >> dt[1];
			fp >> dt[2];
			fp >> dt[3];
			fp >> dt[4];
			fp >> dt_total;
			stats_nb_backtrack[the_case] = nb_backtrack;
			stats_nb_backtrack_decision[the_case] = nb_backtrack_decision;
			stats_graph_size[the_case] = nb_points;
			stats_time[the_case] = dt_total;
		}
	} // next i
	if (f_v) {
		cout << "substructure_lifting_data::get_statistics: done" << endl;
	}

}

void substructure_lifting_data::write_statistics()
{
	orbiter_kernel_system::file_io Fio;

	{
		ofstream f(fname_statistics);
		int i;

		f << Iso->Sub->nb_starter << endl;
		for (i = 0; i < Iso->Sub->nb_starter; i++) {
			f << setw(7) << i << " "
				<< setw(4) << stats_nb_backtrack[i]
				<< setw(4) << stats_nb_backtrack_decision[i]
				<< setw(4) << stats_graph_size[i]
				<< setw(4) << stats_time[i] << endl;
		}
		f << "-1" << endl;
	}
	cout << "substructure_lifting_data::write_statistics "
			"written file " << fname_statistics << " of size "
			<< Fio.file_size(fname_statistics) << endl;
}

void substructure_lifting_data::evaluate_statistics(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int nb_backtrack_max;
	int nb_backtrack_min;
	int graph_size_max;
	int graph_size_min;
	int time_max;
	int time_min;
	ring_theory::longinteger_object a, b, c, a1, b1, c1, d, n, q1, q2, q3, r1, r2, r3;
	ring_theory::longinteger_domain D;

	nb_backtrack_max = nb_backtrack_min = stats_nb_backtrack[0];
	graph_size_max = graph_size_min = stats_graph_size[0];
	time_max = time_min = stats_time[0];

	a.create(0);
	b.create(0);
	c.create(0);
	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		nb_backtrack_max =
				MAXIMUM(nb_backtrack_max, stats_nb_backtrack[i]);
		nb_backtrack_min =
				MINIMUM(nb_backtrack_min, stats_nb_backtrack[i]);
		graph_size_max =
				MAXIMUM(graph_size_max, stats_graph_size[i]);
		graph_size_min =
				MINIMUM(graph_size_min, stats_graph_size[i]);
		time_max =
				MAXIMUM(time_max, stats_time[i]);
		time_min =
				MINIMUM(time_min, stats_time[i]);
		a1.create(stats_nb_backtrack[i]);
		b1.create(stats_graph_size[i]);
		c1.create(stats_time[i]);
		D.add(a, a1, d);
		d.assign_to(a);
		D.add(b, b1, d);
		d.assign_to(b);
		D.add(c, c1, d);
		d.assign_to(c);
	}
	if (f_v) {
		cout << "evaluate_statistics" << endl;
		cout << "nb_backtrack_max=" << nb_backtrack_max << endl;
		cout << "nb_backtrack_min=" << nb_backtrack_min << endl;
		cout << "graph_size_max=" << graph_size_max << endl;
		cout << "graph_size_min=" << graph_size_min << endl;
		cout << "time_max=" << time_max << endl;
		cout << "time_min=" << time_min << endl;
		cout << "sum nb_backtrack = " << a << endl;
		cout << "sum graph_size = " << b << endl;
		cout << "sum time = " << c << endl;
		n.create(Iso->Sub->nb_starter);
		D.integral_division(a, n, q1, r1, 0);
		D.integral_division(b, n, q2, r2, 0);
		D.integral_division(c, n, q3, r3, 0);
		cout << "average nb_backtrack = " << q1 << endl;
		cout << "average graph_size = " << q2 << endl;
		cout << "average time = " << q3 << endl;
	}
}






void substructure_lifting_data::write_starter_nb_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "substructure_lifting_data::write_starter_nb_orbits" << endl;
	}

	string label;

	label.assign("Stab_orbits");
	Fio.int_vec_write_csv(
			first_flag_orbit_of_starter, Iso->Sub->nb_starter,
			fname_orbits_of_stabilizer_csv, label);

	if (f_v) {
		cout << "substructure_lifting_data::write_starter_nb_orbits "
				"Written file "
				<< fname_orbits_of_stabilizer_csv << " of size "
				<< Fio.file_size(fname_orbits_of_stabilizer_csv) << endl;
	}

	if (f_v) {
		cout << "substructure_lifting_data::write_starter_nb_orbits done" << endl;
	}
}

void substructure_lifting_data::read_starter_nb_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "substructure_lifting_data::read_starter_nb_orbits" << endl;
	}
	int *M;
	int m, n, i;

	if (f_v) {
		cout << "substructure_lifting_data::read_starter_nb_orbits "
				"Reading file "
				<< fname_orbits_of_stabilizer_csv << " of size "
				<< Fio.file_size(fname_orbits_of_stabilizer_csv) << endl;
	}

	Fio.int_matrix_read_csv(fname_orbits_of_stabilizer_csv,
			M, m, n, verbose_level - 1);

	if (m != Iso->Sub->nb_starter) {
		cout << "substructure_lifting_data::read_starter_nb_orbits "
				"m != Iso->nb_starter" << endl;
		exit(1);
	}
	if (n != 1) {
		cout << "substructure_lifting_data::read_starter_nb_orbits "
				"n != 1" << endl;
		exit(1);
	}

	first_flag_orbit_of_starter = NEW_int(Iso->Sub->nb_starter + 1);
	nb_flag_orbits_of_starter = NEW_int(Iso->Sub->nb_starter);
	first_flag_orbit_of_starter[0] = 0;
	for (i = 0; i < m; i++) {
		nb_flag_orbits_of_starter[i] = M[i];
		first_flag_orbit_of_starter[i + 1] =
				first_flag_orbit_of_starter[i] + nb_flag_orbits_of_starter[i];
	}

	FREE_int(M);

	if (f_v) {
		cout << "substructure_lifting_data::read_starter_nb_orbits done" << endl;
	}
}


void substructure_lifting_data::write_hash_and_datref_file(
		int verbose_level)
// Writes the file 'fname_hash_and_datref'
// containing id_to_hash[] and id_to_datref[]
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "substructure_lifting_data::write_hash_and_datref_file" << endl;
	}
	{
		//ofstream f(fname_hash_and_datref);
		orbiter_kernel_system::file_io Fio;
		long int *T;
		int i;

		T = NEW_lint(N * 2);

		for (i = 0; i < N; i++) {
			T[2 * i + 0] = id_to_hash[i];
			T[2 * i + 1] = id_to_datref[i];
		}
		Fio.lint_matrix_write_csv(fname_hash_and_datref, T, N, 2);


		data_structures::tally_lint TA;

		TA.init(id_to_hash, N, true, 0);
		cout << "substructure_lifting_data::write_hash_and_datref_file "
				"id_to_hash tallied:" << endl;
		TA.print_second(false /* f_backwards */);
		cout << endl;

#if 0
		f << N << endl;
		for (i = 0; i < N; i++) {
			f << setw(3) << i << " "
				<< setw(3) << id_to_hash[i] << " "
				<< setw(3) << id_to_datref[i] << endl;
			}
		f << -1 << endl;
#endif

		FREE_lint(T);
	}
	if (f_v) {
		cout << "substructure_lifting_data::write_hash_and_datref_file finished" << endl;
		cout << "substructure_lifting_data::write_hash_and_datref_file written file "
				<< fname_hash_and_datref << " of size "
				<< Fio.file_size(fname_hash_and_datref) << endl;
	}
}

void substructure_lifting_data::read_hash_and_datref_file(int verbose_level)
// Reads the file 'fname_hash_and_datref'
// containing id_to_hash[] and id_to_datref[]
// Also initializes hash_vs_id_hash and hash_vs_id_id
// Called from init_solution
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "substructure_lifting_data::read_hash_and_datref_file" << endl;
	}

	{
		//ifstream f(fname_hash_and_datref);
		orbiter_kernel_system::file_io Fio;
		long int *T;
		int m, n;

		Fio.lint_matrix_read_csv(fname_hash_and_datref, T, m, n, verbose_level);


		if (m != N) {
			cout << "substructure_lifting_data::read_hash_and_datref_file, m != N" << endl;
			exit(1);
		}
		if (n != 2) {
			cout << "substructure_lifting_data::read_hash_and_datref_file, n != 2" << endl;
			exit(1);
		}

		long int id, h, d;

#if 0
		f >> N1;
		if (N1 != N) {
			cout << "substructure_lifting_data::read_hash_and_datref_file "
					"N1 != N" << endl;
			cout << "N=" << N << endl;
			cout << "N1=" << N1 << endl;
			exit(1);
		}
#endif

		id_to_datref_allocate(verbose_level);

		for (id = 0; id < N; id++) {
			//f >> a >> h >> d;

			h = T[2 * id + 0];
			d = T[2 * id + 1];

#if 0
			if (a != id) {
				cout << "substructure_lifting_data::read_hash_and_datref_file "
						"a != id" << endl;
				exit(1);
			}
#endif
			id_to_hash[id] = h;
			id_to_datref[id] = d;
			hash_vs_id_hash[id] = h;
			hash_vs_id_id[id] = id;
		}

		FREE_lint(T);

#if 0
		f >> a;
		if (a != -1) {
			cout << "substructure_lifting_data::read_hash_and_datref_file "
					"EOF marker missing" << endl;
			exit(1);
		}
#endif
	}
	Sorting.lint_vec_heapsort_with_log(hash_vs_id_hash, hash_vs_id_id, N);
	if (f_vv) {
		cout << "substructure_lifting_data::read_hash_and_datref_file" << endl;
		print_hash_vs_id();
	}

	if (f_v) {
		cout << "substructure_lifting_data::read_hash_and_datref_file done" << endl;
	}
}

void substructure_lifting_data::print_hash_vs_id()
{
	int i;

	cout << "substructure_lifting_data::print_hash_vs_id" << endl;
	cout << "i : hash_vs_id_hash[i] : hash_vs_id_id[i]" << endl;
	for (i = 0; i < N; i++) {
		cout << i << " : " << hash_vs_id_hash[i]
			<< " : " << hash_vs_id_id[i] << endl;
	}
}

void substructure_lifting_data::write_orbit_data(int verbose_level)
// Writes the file 'fname_staborbits'
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "substructure_lifting_data::write_orbit_data" << endl;
	}

	{
		long int *T;
		int i;

		T = NEW_lint(nb_flag_orbits * 2);

		for (i = 0; i < nb_flag_orbits; i++) {
			T[2 * i + 0] = flag_orbit_solution_first[i];
			T[2 * i + 1] = flag_orbit_solution_len[i];
		}
		Fio.lint_matrix_write_csv(fname_flag_orbits, T, nb_flag_orbits, 2);
		FREE_lint(T);


		if (f_v) {
			cout << "substructure_lifting_data::write_orbit_data finished" << endl;
			cout << "substructure_lifting_data::write_orbit_data written file "
					<< fname_flag_orbits << " of size "
					<< Fio.file_size(fname_flag_orbits) << endl;
		}
	}


	{
		long int *T;
		int i;

		T = NEW_lint(N * 4);

		for (i = 0; i < N; i++) {
			T[4 * i + 0] = flag_orbit_of_solution[i];
			T[4 * i + 1] = orbit_perm[i];
			T[4 * i + 2] = schreier_vector[i];
			T[4 * i + 3] = schreier_prev[i];
		}
		Fio.lint_matrix_write_csv(fname_stab_orbits, T, N, 4);
		FREE_lint(T);


		if (f_v) {
			cout << "substructure_lifting_data::write_orbit_data finished" << endl;
			cout << "substructure_lifting_data::write_orbit_data written file "
					<< fname_stab_orbits << " of size "
					<< Fio.file_size(fname_stab_orbits) << endl;
		}
	}

#if 0
	{
		//ofstream f(fname_staborbits);
		int i;

		f << nb_flag_orbits << " " << N << endl;
		for (i = 0; i < nb_flag_orbits; i++) {
			f << setw(3) << i << " "
				<< setw(3) << orbit_fst[i] << " "
				<< setw(3) << orbit_len[i] << endl;
		}
		for (i = 0; i < N; i++) {
			f << setw(3) << i << " "
				<< setw(3) << orbit_number[i] << " "
				<< setw(3) << orbit_perm[i] << " "
				<< setw(3) << schreier_vector[i] << " "
				<< setw(3) << schreier_prev[i] << " "
				<< endl;
		}
		f << "-1" << endl;
	}
	if (f_v) {
		cout << "substructure_lifting_data::write_orbit_data finished" << endl;
		cout << "written file " << fname_staborbits << " of size "
				<< Fio.file_size(fname_staborbits) << endl;
	}
#endif
}

void substructure_lifting_data::read_orbit_data(
		int verbose_level)
// Reads from the files fname_flag_orbits and fname_stab_orbits
// Reads nb_orbits, N,
// orbit_fst[nb_flag_orbits + 1]
// orbit_len[nb_flag_orbits]
// orbit_number[N]
// orbit_perm[N]
// schreier_vector[N]
// schreier_prev[N]
// and computed orbit_perm_inv[N]
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "substructure_lifting_data::read_orbit_data" << endl;
	}


	{
		orbiter_kernel_system::file_io Fio;
		long int *T;
		int m, n;

		Fio.lint_matrix_read_csv(fname_flag_orbits, T, m, n, verbose_level);

		nb_flag_orbits = m;

		if (n != 2) {
			cout << "substructure_lifting_data::read_orbit_data, n != 2" << endl;
			exit(1);
		}

		long int i, a, b;

		flag_orbit_solution_first = NEW_int(nb_flag_orbits + 1);
		flag_orbit_solution_len = NEW_int(nb_flag_orbits);
		for (i = 0; i < nb_flag_orbits; i++) {
			a = T[2 * i + 0];
			b = T[2 * i + 1];
			flag_orbit_solution_first[i] = a;
			flag_orbit_solution_len[i] = b;
		}

		FREE_lint(T);


	}


	{
		orbiter_kernel_system::file_io Fio;
		long int *T;
		int m, n;

		Fio.lint_matrix_read_csv(fname_stab_orbits, T, m, n, verbose_level);

		N = m;

		if (n != 4) {
			cout << "substructure_lifting_data::read_orbit_data, n != 4" << endl;
			exit(1);
		}

		long int i, a, b, c, d;

		flag_orbit_of_solution = NEW_int(N);
		orbit_perm = NEW_int(N);
		orbit_perm_inv = NEW_int(N);
		schreier_vector = NEW_int(N);
		schreier_prev = NEW_int(N);

		for (i = 0; i < N; i++) {
			a = T[4 * i + 0];
			b = T[4 * i + 1];
			c = T[4 * i + 2];
			d = T[4 * i + 3];
			flag_orbit_of_solution[i] = a;
			orbit_perm[i] = b;
			schreier_vector[i] = c;
			schreier_prev[i] = d;
		}

		FREE_lint(T);

		combinatorics::combinatorics_domain Combi;

		Combi.perm_inverse(orbit_perm, orbit_perm_inv, N);

	}

	flag_orbit_solution_first[nb_flag_orbits] = N;


#if 0
	{
		ifstream f(fname_stab_orbits);
		int i, a;
		f >> nb_flag_orbits >> N;
		if (f_v) {
			cout << "nb_orbits=" << nb_flag_orbits << endl;
			cout << "N=" << N << endl;
		}

		orbit_fst = NEW_int(nb_flag_orbits + 1);
		orbit_len = NEW_int(nb_flag_orbits);
		orbit_number = NEW_int(N);
		orbit_perm = NEW_int(N);
		orbit_perm_inv = NEW_int(N);
		schreier_vector = NEW_int(N);
		schreier_prev = NEW_int(N);

		for (i = 0; i < nb_flag_orbits; i++) {
			f >> a;
			f >> orbit_fst[i];
			f >> orbit_len[i];
		}
		for (i = 0; i < N; i++) {
			f >> a;
			f >> orbit_number[i];
			f >> orbit_perm[i];
			f >> schreier_vector[i];
			f >> schreier_prev[i];
		}
		orbit_fst[nb_flag_orbits] = N;
		Combi.perm_inverse(orbit_perm, orbit_perm_inv, N);
		f >> a;
		if (a != -1) {
			cout << "problem in read_orbit_data" << endl;
			exit(1);
		}
	}
#endif
	if (f_v) {
		cout << "substructure_lifting_data::read_orbit_data finished" << endl;
	}
}


void substructure_lifting_data::test_hash(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	long int data[1000];
	int id, case_nb, f, l, i;
	long int *H;
	data_structures::sorting Sorting;
	data_structures::data_structures_global Data;


	if (f_v) {
		cout << "substructure_lifting_data::test_hash" << endl;
	}
	setup_and_open_solution_database(verbose_level - 1);
	for (case_nb = 0; case_nb < Iso->Sub->nb_starter; case_nb++) {
		f = starter_solution_first[case_nb];
		l = starter_solution_len[case_nb];
		if (l == 1) {
			continue;
			}
		cout << "starter " << case_nb << " f=" << f << " l=" << l << endl;
		H = NEW_lint(l);
		for (i = 0; i < l; i++) {
			//id = orbit_perm[f + i];
			id = f + i;
			load_solution(id, data, verbose_level - 1);
			Sorting.lint_vec_heapsort(data, Iso->size);
			H[i] = Data.lint_vec_hash(data, Iso->size);
			}
		{
			data_structures::tally_lint C;
			C.init(H, l, true, 0);
			C.print(false /*f_backwards*/);
		}
		FREE_lint(H);
		}

	close_solution_database(verbose_level - 1);
	if (f_v) {
		cout << "substructure_lifting_data::test_hash done" << endl;
	}
}

void substructure_lifting_data::id_to_datref_allocate(int verbose_level)
{
	id_to_datref = NEW_lint(N);
	id_to_hash = NEW_lint(N);
	hash_vs_id_hash = NEW_lint(N);
	hash_vs_id_id = NEW_lint(N);
}


}}}

