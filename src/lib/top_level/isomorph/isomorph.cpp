// isomorph.cpp
// 
// Anton Betten
// started 2007
// moved here from reader2.cpp: 3/22/09
// renamed isomorph.cpp from global.cpp: 7/14/11
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


isomorph::isomorph()
{
	null();
}

void isomorph::null()
{
	solution_first = NULL;
	solution_len = NULL;
	starter_number = NULL;
	
	A_base = NULL;
	A = NULL;
	gen = NULL;
	
	orbit_fst = NULL;
	orbit_len = NULL;
	orbit_number = NULL;
	orbit_perm = NULL;
	orbit_perm_inv = NULL;
	schreier_vector = NULL;
	schreier_prev = NULL;
	
	flag_orbit_fst = NULL;
	flag_orbit_len = NULL;
	
	Reps = NULL;

	gens_perm = NULL;
	AA = NULL;
	AA_perm = NULL;
	AA_on_k_subsets = NULL;
	UF = NULL;
	
	subset = NULL;
	subset_witness = NULL;
	rearranged_set = NULL;
	rearranged_set_save = NULL;
	canonical_set = NULL;
	tmp_set = NULL;
	Elt_transporter = NULL;
	tmp_Elt = NULL;
	Elt1 = NULL;
	transporter = NULL;

	null_tmp_data();

	D1 = NULL;
	D2 = NULL;
	fp_ge1 = NULL;
	fp_ge2 = NULL;
	fp_ge = NULL;
	DB_sol = NULL;
	id_to_datref = NULL;
	id_to_hash = NULL;
	hash_vs_id_hash = NULL;
	hash_vs_id_id = NULL;
	f_use_table_of_solutions = FALSE;
	table_of_solutions = NULL;
	
	DB_level = NULL;
	stabilizer_recreated = NULL;
	print_set_function = NULL;
	
	nb_times_make_set_smaller_called = 0;
}

isomorph::~isomorph()
{
	free();
	null();
}

void isomorph::free()
{
	//int i;
	int f_v = FALSE;

	if (f_v) {
		cout << "isomorph::free" << endl;
		}

#if 0
	if (f_v) {
		cout << "isomorph::free before deleting A" << endl;
		}
	if (A) {
		FREE_OBJECT(A);
		}
#endif
	if (f_v) {
		cout << "isomorph::free before deleting AA" << endl;
		}
	if (AA) {
		FREE_OBJECT(AA);
		AA = NULL;
		}
#if 0
	if (f_v) {
		cout << "isomorph::free before deleting gen" << endl;
		}
	if (gen) {
		FREE_OBJECT(gen);
		gen = NULL;
		}
#endif

	if (f_v) {
		cout << "isomorph::free "
				"before deleting stabilizer_recreated" << endl;
		}
	if (stabilizer_recreated) {
		delete stabilizer_recreated;
		stabilizer_recreated = NULL;
		}
	if (f_v) {
		cout << "isomorph::free "
				"before deleting DB_sol" << endl;
		}
	if (DB_sol) {
		freeobject(DB_sol);
		DB_sol = NULL;
		}
	if (f_v) {
		cout << "isomorph::free before deleting D1" << endl;
		}
	if (D1) {
		freeobject(D1);
		D1 = NULL;
		}
	if (f_v) {
		cout << "isomorph::free before deleting D2" << endl;
		}
	if (D2) {
		freeobject(D2);
		D2 = NULL;
		}
	if (f_tmp_data_has_been_allocated) {
		if (f_v) {
			cout << "isomorph::free before free_tmp_data" << endl;
			}
		free_tmp_data();
		}
	if (id_to_datref) {
		FREE_int(id_to_datref);
		id_to_datref = NULL;
		}
	if (id_to_hash) {
		FREE_int(id_to_hash);
		id_to_hash = NULL;
		}
	if (hash_vs_id_hash) {
		FREE_int(hash_vs_id_hash);
		hash_vs_id_hash = NULL;
		}
	if (hash_vs_id_id) {
		FREE_int(hash_vs_id_id);
		hash_vs_id_id = NULL;
		}
	if (table_of_solutions) {
		FREE_lint(table_of_solutions);
		table_of_solutions = NULL;
		f_use_table_of_solutions = FALSE;
		}
	if (f_v) {
		cout << "isomorph::free done" << endl;
		}
}

void isomorph::null_tmp_data()
{
	f_tmp_data_has_been_allocated = FALSE;
	tmp_set1 = NULL;
	tmp_set2 = NULL;
	tmp_set3 = NULL;
	tmp_Elt1 = NULL;
	tmp_Elt2 = NULL;
	tmp_Elt3 = NULL;
	trace_set_recursion_tmp_set1 = NULL;
	trace_set_recursion_Elt1 = NULL;
	apply_fusion_tmp_set1 = NULL;
	apply_fusion_Elt1 = NULL;
	find_extension_set1 = NULL;
	make_set_smaller_set = NULL;
	make_set_smaller_Elt1 = NULL;
	make_set_smaller_Elt2 = NULL;
	orbit_representative_Elt1 = NULL;
	orbit_representative_Elt2 = NULL;
	handle_automorphism_Elt1 = NULL;
	v = NULL;
}

void isomorph::allocate_tmp_data()
// called by init_action_BLT() in isomorph_BLT()
{
	f_tmp_data_has_been_allocated = TRUE;
	tmp_set1 = NEW_lint(size);
	tmp_set2 = NEW_lint(size);
	tmp_set3 = NEW_lint(size);
	tmp_Elt1 = NEW_int(A->elt_size_in_int);
	tmp_Elt2 = NEW_int(A->elt_size_in_int);
	tmp_Elt3 = NEW_int(A->elt_size_in_int);

	trace_set_recursion_tmp_set1 = NEW_lint(size);
	trace_set_recursion_Elt1 = NEW_int(A->elt_size_in_int);
	
	apply_fusion_tmp_set1 = NEW_lint(size);
	apply_fusion_Elt1 = NEW_int(A->elt_size_in_int);
	
	find_extension_set1 = NEW_lint(size);

	make_set_smaller_set = NEW_lint(size);
	make_set_smaller_Elt1 = NEW_int(A->elt_size_in_int);
	make_set_smaller_Elt2 = NEW_int(A->elt_size_in_int);

	orbit_representative_Elt1 = NEW_int(A->elt_size_in_int);
	orbit_representative_Elt2 = NEW_int(A->elt_size_in_int);

	handle_automorphism_Elt1 = NEW_int(A->elt_size_in_int);
	
	v = new Vector[1];

}

void isomorph::free_tmp_data()
{
	int f_v = FALSE;
	
	if (f_v) {
		cout << "isomorph::free_tmp_data" << endl;
		}
	if (f_tmp_data_has_been_allocated) {
		f_tmp_data_has_been_allocated = FALSE;
		FREE_lint(tmp_set1);
		FREE_lint(tmp_set2);
		FREE_lint(tmp_set3);
		FREE_int(tmp_Elt1);
		FREE_int(tmp_Elt2);
		FREE_int(tmp_Elt3);
		FREE_lint(trace_set_recursion_tmp_set1);
		FREE_int(trace_set_recursion_Elt1);
		FREE_lint(apply_fusion_tmp_set1);
		FREE_int(apply_fusion_Elt1);
		FREE_lint(make_set_smaller_set);
		FREE_int(make_set_smaller_Elt1);
		FREE_int(make_set_smaller_Elt2);
		FREE_int(orbit_representative_Elt1);
		FREE_int(orbit_representative_Elt2);
		FREE_int(handle_automorphism_Elt1);
		delete [] v;
		}
	null_tmp_data();
	if (f_v) {
		cout << "isomorph::free_tmp_data finished" << endl;
		}
}

void isomorph::init(const char *prefix, 
	action *A_base, action *A, poset_classification *gen,
	int size, int level, 
	int f_use_database_for_starter, 
	int f_implicit_fusion, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char cmd[1000];

	if (f_v) {
		cout << "isomorph::init" << endl;
		cout << "prefix=" << prefix << endl;
		cout << "A_base=" << A_base->label << endl;
		cout << "A=" << A->label << endl;
		cout << "size=" << size << endl;
		cout << "level=" << level << endl;
		cout << "f_use_database_for_starter="
				<< f_use_database_for_starter << endl;
		cout << "f_implicit_fusion=" << f_implicit_fusion << endl;
		}

	strcpy(isomorph::prefix, prefix);
	isomorph::A_base = A_base;
	isomorph::A = A;
	isomorph::gen = gen;
	isomorph::size = size;
	isomorph::level = level;
	isomorph::f_use_database_for_starter = f_use_database_for_starter;


	nb_starter = 0;
	f_use_implicit_fusion = FALSE;
	
#if 0
	if (f_use_database_for_starter) {
		sprintf(fname_data_file, "%s_%d.data", prefix, level - 1);
		}
	else {
		sprintf(fname_data_file, "%s_%d.data", prefix, level);
		}
	if (f_v) {
		cout << "fname_data_file=" << fname_data_file << endl;
		}
	sprintf(fname_level_file, "%s_lvl_%d", prefix, level);
#endif
	sprintf(fname_staborbits, "%sstaborbits.txt", prefix);
	sprintf(fname_case_len, "%scase_len.txt", prefix);
	sprintf(fname_statistics, "%sstatistics.txt", prefix);
	sprintf(fname_hash_and_datref, "%shash_and_datref.txt", prefix);
	sprintf(fname_db1, "%ssolutions.db", prefix);
	sprintf(fname_db2, "%ssolutions_a.idx", prefix);
	sprintf(fname_db3, "%ssolutions_b.idx", prefix);
	sprintf(fname_db4, "%ssolutions_c.idx", prefix);
	sprintf(fname_db5, "%ssolutions_d.idx", prefix);

	sprintf(event_out_fname, "%sevent.txt", prefix);
	sprintf(fname_orbits_of_stabilizer_csv,
			"%sorbits_of_stabilizer.csv", prefix);
	sprintf(prefix_invariants, "%sINVARIANTS/", prefix);
	sprintf(prefix_tex, "%sTEX/", prefix);
	sprintf(cmd, "mkdir %s", prefix);
	system(cmd);
	sprintf(cmd, "mkdir %sINVARIANTS/", prefix);
	system(cmd);
	sprintf(cmd, "mkdir %sTEX/", prefix);
	system(cmd);

	allocate_tmp_data();

	if (f_v) {
		cout << "isomorph::init done" << endl;
		}
}







void isomorph::init_solution(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "isomorph::init_solution" << endl;
		}
	read_solution_first_and_len();
	init_starter_number(verbose_level);
	read_hash_and_datref_file(verbose_level);
	if (f_v) {
		cout << "isomorph::init_solution done" << endl;
		}
}

void isomorph::load_table_of_solutions(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int id, j;
	long int data[1000];

	if (f_v) {
		cout << "isomorph::load_table_of_solutions N=" << N << endl;
		}
	setup_and_open_solution_database(verbose_level);
	table_of_solutions = NEW_lint(N * size);
	for (id = 0; id < N; id++) {
		load_solution(id, data);
		for (j = 0; j < size; j++) {
			table_of_solutions[id * size + j] = data[j];
			}
#if 0
		cout << "solution " << id << " : ";
		int_vec_print(cout, table_of_solutions + id * size, size);
		cout << endl;
#endif
		}
	f_use_table_of_solutions = TRUE;
	close_solution_database(verbose_level);
	if (f_v) {
		cout << "isomorph::load_table_of_solutions done" << endl;
		}
}

void isomorph::init_starter_number(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, f, l;
	
	if (f_v) {
		cout << "isomorph::init_starter_number N=" << N << endl;
		}
	starter_number = NEW_int(N);
	for (i = 0; i < nb_starter; i++) {
		f = solution_first[i];
		l = solution_len[i];
		for (j = 0; j < l; j++) {
			starter_number[f + j] = i;
			}
		}
	if (f_v) {
		cout << "starter_number:" << endl;
		int_vec_print(cout, starter_number, N);
		cout << endl;
		}
}


void isomorph::list_solutions_by_starter()
{
	int i, j, idx, id, f, l, fst, len, h, pos, u;
	long int data[1000];
	long int data2[1000];
	int verbose_level = 0;
	sorting Sorting;
	
	setup_and_open_solution_database(verbose_level - 1);
	
	j = 0;
	for (i = 0; i < nb_starter; i++) {
		f = solution_first[i];
		l = solution_len[i];
		cout << "starter " << i << " solutions from="
				<< f << " len=" << l << endl;
		pos = f;
		while (pos < f + l) {
			fst = orbit_fst[j];
			len = orbit_len[j];
			cout << "orbit " << j << " from=" << fst
					<< " len=" << len << endl;
			for (u = 0; u < len; u++) {
				idx = fst + u;
				id = orbit_perm[idx];
				load_solution(id, data);
				for (h = 0; h < size; h++) {
					data2[h] = data[h];
					}
				Sorting.lint_vec_heapsort(data2, size);
				cout << i << " : " << j << " : "
						<< idx << " : " << id << endl;
				lint_vec_print(cout, data, size);
				cout << endl;
				lint_vec_print(cout, data2, size);
				cout << endl;
				}
			pos += len;
			j++;
			}
		}
	close_solution_database(verbose_level);
}


void isomorph::list_solutions_by_orbit()
{
	int i, j, idx, id, f, l, h;
	long int data[1000];
	long int data2[1000];
	int verbose_level = 0;
	sorting Sorting;
	
	setup_and_open_solution_database(verbose_level - 1);

	for (i = 0; i < nb_orbits; i++) {
		f = orbit_fst[i];
		l = orbit_len[i];
		cout << "orbit " << i << " from=" << f
				<< " len=" << l << endl;
		for (j = 0; j < l; j++) {
			idx = f + j;
			id = orbit_perm[idx];
			load_solution(id, data);
			for (h = 0; h < size; h++) {
				data2[h] = data[h];
				}
			Sorting.lint_vec_heapsort(data2, size);
			cout << j << " : " << idx << " : " << id << endl;
			lint_vec_print(cout, data, size);
			cout << endl;
			lint_vec_print(cout, data2, size);
			cout << endl;
			}
		}

	close_solution_database(verbose_level);
}

void isomorph::orbits_of_stabilizer(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvvv = (verbose_level >= 4);
	int f_v5 = (verbose_level >= 5);
	int i, j, f, l, nb_orbits_prev = 0;
	longinteger_object go;

	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer" << endl;
		cout << "number of starters = nb_starter = "
				<< nb_starter << endl;
		cout << "number of solutions (= N) = " << N << endl;
		cout << "action A_base=";
		A_base->print_info();
		cout << endl;
		cout << "action A=";
		A->print_info();
		cout << endl;
		}

	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer before setup_and_open_solution_database" << endl;
	}
	setup_and_open_solution_database(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer after setup_and_open_solution_database" << endl;
	}
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer before setup_and_open_level_database" << endl;
	}
	setup_and_open_level_database(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer after setup_and_open_level_database" << endl;
	}


	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer before prepare_database_access" << endl;
	}
	prepare_database_access(level, verbose_level - 1);
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer after prepare_database_access" << endl;
	}


	nb_orbits = 0;
	orbit_fst = NEW_int(N + 1);
	orbit_len = NEW_int(N);
	orbit_number = NEW_int(N);
	orbit_perm = NEW_int(N);
	orbit_perm_inv = NEW_int(N);
	schreier_vector = NEW_int(N);
	schreier_prev = NEW_int(N);

	// added Dec 25, 2012:

	flag_orbit_fst = NEW_int(nb_starter);
	flag_orbit_len = NEW_int(nb_starter);

	for (i = 0; i < N; i++) {
		schreier_vector[i] = -2;
		schreier_prev[i] = -1;
		}
	
	orbit_fst[0] = 0;
	for (i = 0; i < nb_starter; i++) {
		if (f_v) {
			cout << "isomorph::orbits_of_stabilizer case "
					"i=" << i << " / " << nb_starter << endl;
			}

		flag_orbit_fst[i] = nb_orbits;
		flag_orbit_len[i] = 0;

		//oracle *O;
		vector_ge gens;
		
		//O = &gen->root[gen->first_oracle_node_at_level[level] + i];
		
		
		load_strong_generators(level, 
			i, 
			gens, go, verbose_level - 2);
		if (f_v5) {
			cout << "isomorph::orbits_of_stabilizer "
					"after load_strong_generators" << endl;
			cout << "isomorph::orbits_of_stabilizer "
					"The stabilizer is a group of order "
					<< go << " with " << gens.len
					<< " strong generators" << endl;
			gens.print_with_given_action(cout, A_base);
			}
		
		f = solution_first[i];
		l = solution_len[i];
		if (f_v && ((i % 5000) == 0)) {
			cout << "isomorph::orbits_of_stabilizer Case " << i
					<< " / " << nb_starter << endl;
			}
		if (f_vv) {
			cout << "isomorph::orbits_of_stabilizer nb_orbits = "
					<< nb_orbits << endl;
			cout << "isomorph::orbits_of_stabilizer case " << i
					<< " starts at " << f << " with " << l
					<< " solutions" << endl;
			}
		if (gens.len == 0 /*O->nb_strong_generators == 0*/) {
			if (f_vv) {
				cout << "isomorph::orbits_of_stabilizer "
						"the stabilizer is trivial" << endl;
				}
			for (j = 0; j < l; j++) {
				orbit_len[nb_orbits] = 1;
				schreier_vector[f + j] = -1;
				orbit_number[f + j] = nb_orbits;
				orbit_perm[f + j] = f + j;
				orbit_perm_inv[f + j] = f + j;
				nb_orbits++;
				orbit_fst[nb_orbits] =
						orbit_fst[nb_orbits - 1] +
						orbit_len[nb_orbits - 1];
				flag_orbit_len[i]++;
				}
			}
		else {
			if (f_vv) {
				cout << "isomorph::orbits_of_stabilizer "
						"the stabilizer is non trivial" << endl;
				}
			if (solution_len[i] != 0) {
				if (f_vv) {
					cout << "isomorph::orbits_of_stabilizer "
							"before orbits_of_stabilizer_case" << endl;
					}
				orbits_of_stabilizer_case(i, gens, verbose_level - 2);
				if (f_vv) {
					cout << "isomorph::orbits_of_stabilizer "
							"after orbits_of_stabilizer_case" << endl;
					cout << "isomorph::orbits_of_stabilizer "
							"the " << l << " solutions in case " << i
							<< " fall into " << nb_orbits - nb_orbits_prev
							<< " orbits" << endl;
					}
				flag_orbit_len[i] = nb_orbits - nb_orbits_prev;
				}
			}
		if (f_v) {
			cout << "isomorph::orbits_of_stabilizer Case " << i
					<< " / " << nb_starter << " finished, we found "
					<< nb_orbits - nb_orbits_prev << " orbits : ";
			if (nb_orbits - nb_orbits_prev) {
				classify C;

				C.init(orbit_len + nb_orbits_prev,
						nb_orbits - nb_orbits_prev, FALSE, 0);
				C.print_naked(TRUE /* f_backwards */);
				cout << endl;
				}
			else {
				cout << endl;
				}
			}
		if (FALSE && f_vvvv) {
			cout << "i : orbit_perm : orbit_number : schreier_vector : "
					"schreier_prev" << endl;
			for (j = 0; j < l; j++) {
				cout << f + j << " : " 
					<< orbit_perm[f + j] << " : " 
					<< orbit_number[f + j] << " : " 
					<< schreier_vector[f + j] << " : " 
					<< schreier_prev[f + j] << endl;
				}
			cout << "j : orbit_fst : orbit_len" << endl;
			for (j = nb_orbits_prev; j < nb_orbits; j++) {
				cout << j << " : " << orbit_fst[j] << " : "
						<< orbit_len[j] << endl;
				}
			cout << j << " : " << orbit_fst[j] << endl;
			if (orbit_fst[nb_orbits] != solution_first[i + 1]) {
				cout << "orbit_fst[nb_orbits] != "
						"solution_first[i + 1]" << endl;
				cout << "orbit_fst[nb_orbits]="
						<< orbit_fst[nb_orbits] << endl;
				cout << "solution_first[i + 1]="
						<< solution_first[i + 1] << endl;
				exit(1);
				}
			}			
		nb_orbits_prev = nb_orbits;
		} // next i
	
	if (orbit_fst[nb_orbits] != N) {
		cout << "orbit_fst[nb_orbits] != N" << endl;
		cout << "orbit_fst[nb_orbits]=" << orbit_fst[nb_orbits] << endl;
		cout << "N=" << N << endl;
		cout << "nb_orbits=" << nb_orbits << endl;
		cout << "nb_starter=" << nb_starter << endl;
		}
	
	close_solution_database(verbose_level);
	close_level_database(verbose_level);

	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer Case " << i << " / "
				<< nb_starter << " finished, we found " << nb_orbits
				<< " orbits : ";
		classify C;

		C.init(orbit_len, nb_orbits, FALSE, 0);
		C.print_naked(TRUE /* f_backwards */);
		cout << endl;
		}

#if 0
	if (FALSE && f_vv) {
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


	write_starter_nb_orbits(verbose_level);
	
}

void isomorph::orbits_of_stabilizer_case(int the_case,
		vector_ge &gens, int verbose_level)
{
	Vector v;
	//oracle *O;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v4 = (verbose_level >= 4);
	int j, f, l, k, ff, ll;
	
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer_case "
				<< the_case << " / " << nb_starter << endl;
		}
	
	//O = &gen->root[gen->first_oracle_node_at_level[level] + the_case];
	f = solution_first[the_case];
	l = solution_len[the_case];
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer_case "
				"solution_first[the_case] = " << f << endl;
		cout << "isomorph::orbits_of_stabilizer_case "
				"solution_len[the_case] = " << l << endl;
		}

	longinteger_object S_go;
	sims *S;
	action *AA;
	schreier *Schreier;
	long int *sets;
	int h, p, prev, b, hdl;
	sorting Sorting;
			
	sets = NEW_lint(l * size);
	S = NEW_OBJECT(sims);
	AA = NEW_OBJECT(action);
	Schreier = NEW_OBJECT(schreier);
			
		
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case "
				"generators as permutations (skipped)" << endl;
		//gens.print_as_permutation(cout);
	}
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case before S->init" << endl;
	}
	S->init(A_base, verbose_level - 2);
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case after S->init" << endl;
	}
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case before S->init_generators" << endl;
	}
	S->init_generators(gens, verbose_level - 2);
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case after S->init_generators" << endl;
	}
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case before S->compute_base_orbits" << endl;
	}
	S->compute_base_orbits(verbose_level - 2);
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case after S->compute_base_orbits" << endl;
	}
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case before S->group_order" << endl;
	}
	S->group_order(S_go);
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer_case "
				"The starter has a stabilizer of order "
				<< S_go << endl;
		}
			
	for (j = 0; j < l; j++) {

		load_solution(f + j, sets + j * size);
		if (FALSE && f_vv) {
			cout << "solution " << j << "        : ";
			lint_vec_print(cout, sets + j * size, size);
			cout << endl;
			}
		Sorting.lint_vec_heapsort(sets + j * size, size);
		if (FALSE && f_vv) {
			cout << "solution " << j << " sorted : ";
			lint_vec_print(cout, sets + j * size, size);
			cout << endl;
			}
		}
	
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case "
				"computing induced action on " << l << " sets of size " << size << endl;
		}
			
	AA->induced_action_on_sets(*A, S, //K, 
		l, size, sets, FALSE /*TRUE*/ /* A Betten 1/26/13*/,
		verbose_level /*- 2*/);

	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case "
				"computing induced action finished" << endl;
		}
		
#if 0	
	AA->group_order(AA_go);
	AA->Kernel->group_order(K_go);
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer_case orbit "
				<< nb_orbits << " induced action has order "
				<< AA_go << ", kernel has order " << K_go << endl;
		}
#endif
	
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case "
				"induced action computed" << endl;
		cout << "generators:" << endl;
		for (k = 0; k < gens.len; k++) {
			cout << k << " : ";
			//AA->element_print_as_permutation(gens.ith(k), cout);
			cout << endl;
			}
		}
	
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer_case "
				"computing point orbits" << endl;
		}
	AA->compute_all_point_orbits(*Schreier, gens, verbose_level - 4);
	//AA->all_point_orbits(*Schreier, verbose_level - 2);
			
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer_case "
				"Point orbits computed" << endl;
		}
	if (f_v4) {
		Schreier->print_tables(cout, TRUE);
		}

	for (k = 0; k < l; k++) {
		p = Schreier->orbit[k];
		prev = Schreier->prev[k];
		hdl = Schreier->label[k];
		//cout << "coset " << k << " point p=" << p
		// << " prev=" << prev << " label " << hdl << endl;
		if (prev != -1) {
			//A->element_retrieve(O->hdl_strong_generators[hdl],
			// A->Elt1, FALSE);
			b = AA->element_image_of(prev, gens.ith(hdl), FALSE);
			//cout << "image of " << prev << " results in =" << b << endl;
			if (b != p) {
				cout << "b != p" << endl;
				exit(1);
				}
			if (!A->check_if_transporter_for_set(
					gens.ith(hdl), size,
				sets + prev * size, sets + p * size,
				verbose_level - 2)) {
				exit(1);
				}
			}
		}
	for (k = 0; k < Schreier->nb_orbits; k++) {
		ff = Schreier->orbit_first[k];
		ll = Schreier->orbit_len[k];
		for (h = 0; h < ll; h++) {
			p = f + Schreier->orbit[ff + h];
			orbit_number[f + ff + h] = nb_orbits;
			orbit_perm[f + ff + h] = p;
			orbit_perm_inv[p] = f + ff + h;
			schreier_vector[f + ff + h] =
					Schreier->label[ff + h];
			if (h == 0) {
				schreier_prev[f + ff + h] = -1;
				}
			else {
				schreier_prev[f + ff + h] =
						f + Schreier->prev[ff + h];
				}
			}
		orbit_len[nb_orbits] = ll;
		nb_orbits++;
		orbit_fst[nb_orbits] = orbit_fst[nb_orbits - 1] + ll;
		}
			
	FREE_lint(sets);
	FREE_OBJECT(S);
	FREE_OBJECT(AA);
	FREE_OBJECT(Schreier);
	
}


void isomorph::orbit_representative(int i, int &i0, 
	int &orbit, int *transporter, int verbose_level)
// slow because it calls load_strong_generators
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int c, p, i_loc, l; //, hdl;
	int *Elt1, *Elt2;
	vector_ge gens;
	longinteger_object go;
	
	if (f_v) {
		cout << "isomorph::orbit_representative" << endl;
		}


	prepare_database_access(level, verbose_level);




	Elt1 = orbit_representative_Elt1;
	Elt2 = orbit_representative_Elt2;
	c = starter_number[i];
	if (f_v) {
		cout << "isomorph::orbit_representative "
				"before load_strong_generators" << endl;
		}
	load_strong_generators(level, c, 
		gens, go, verbose_level);
	if (f_v) {
		cout << "isomorph::orbit_representative "
				"after load_strong_generators" << endl;
		}
	A->element_one(transporter, FALSE);
	if (f_vv) {
		cout << "isomorph::orbit_representative "
				"i=" << i << endl;
		}
	while (TRUE) {
		i_loc = orbit_perm_inv[i];
		p = schreier_prev[i_loc];
		if (f_vv) {
			cout << "isomorph::orbit_representative "
					"i=" << i << " i_loc=" << i_loc
					<< " p=" << p << endl;
			}
		if (p == -1) {
			i0 = i;
			orbit = orbit_number[i_loc];
			break;
			}
		l = schreier_vector[i_loc];
		//cout << "l=" << l << endl;
		//hdl = O->hdl_strong_generators[l];
		//A->element_retrieve(hdl, Elt1, FALSE);
		A->element_invert(gens.ith(l), Elt2, FALSE);
		A->element_mult(transporter, Elt2, Elt1, FALSE);
		A->element_move(Elt1, transporter, FALSE);
		i = p;
		}
	if (f_v) {
		cout << "isomorph::orbit_representative "
				"The representative of solution " << i << " is "
				<< i0 << " in orbit " << orbit << endl;
		}
}

void isomorph::test_orbit_representative(int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int r, r0, orbit, k;
	long int data1[1000];
	long int data2[1000];
	int *transporter;
	
	transporter = NEW_int(A->elt_size_in_int);

	setup_and_open_solution_database(verbose_level - 1);
	
	for (k = 0; k < N; k++) {
		r = k;
		//r = random_integer(N);
		//cout << "k=" << k << " r=" << r << endl;
	
		load_solution(r, data1);

		orbit_representative(r, r0, orbit,
				transporter, verbose_level);
		if (r != r0) {
			cout << "k=" << k << " r=" << r << " r0=" << r0 << endl;
			}
			
		load_solution(r0, data2);
		if (!A->check_if_transporter_for_set(transporter,
				size, data1, data2, verbose_level)) {
			exit(1);
			}
		}
	
	close_solution_database(verbose_level - 1);
	FREE_int(transporter);
}

void isomorph::test_identify_solution(int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int r, r0, id, id0;
	long int data1[1000];
	long int data2[1000];
	int perm[1000];
	int i, k;
	int *transporter;
	combinatorics_domain Combi;
	os_interface Os;

	transporter = NEW_int(A->elt_size_in_int);


	setup_and_open_solution_database(verbose_level - 1);
	
	for (k = 0; k < 10; k++) {
		r = Os.random_integer(nb_orbits);
		id = orbit_perm[orbit_fst[r]];
		if (schreier_prev[orbit_fst[r]] != -1) {
			cout << "schreier_prev[orbit_fst[r]] != -1" << endl;
			exit(1);
			}
		//cout << "k=" << k << " r=" << r << endl;
	
		load_solution(id, data1);
		Combi.random_permutation(perm, size);
		for (i = 0; i < size; i++) {
			data2[i] = data1[perm[i]];
			}

		int f_failure_to_find_point;
		r0 = identify_solution(data2, transporter,
				f_use_implicit_fusion, f_failure_to_find_point,
				verbose_level - 2);
		
		if (f_failure_to_find_point) {
			cout << "f_failure_to_find_point" << endl;
			}
		else {
			cout << "k=" << k << " r=" << r << " r0=" << r0 << endl;
			id0 = orbit_perm[orbit_fst[r0]];
			
			load_solution(id0, data1);
			if (!A->check_if_transporter_for_set(transporter,
					size, data2, data1, verbose_level)) {
				cout << "test_identify_solution, "
						"check fails, stop" << endl;
				exit(1);
				}
			}
		}
	
	close_solution_database(verbose_level - 1);
	FREE_int(transporter);
}

void isomorph::compute_stabilizer(sims *&Stab,
		int verbose_level)
// Called from do_iso_test
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	//int f_vvvv = (verbose_level >= 4);
	longinteger_object AA_go, K_go;
	sims *S; //, *K; //, *stab;
	action *AA;
	vector_ge *gens;
	schreier *Schreier;
	long int *sets;
	int j, first, f, l, c, first_orbit_this_case, orb_no;
	longinteger_object go, so, so1;
	sorting Sorting;

	if (f_v) {
		cout << "isomorph::compute_stabilizer "
				"iso_node " << iso_nodes << endl;
		}
	
	first = orbit_fst[orbit_no];
	c = starter_number[first];
	f = solution_first[c];
	l = solution_len[c];
	first_orbit_this_case = orbit_number[f];
	orb_no = orbit_no - first_orbit_this_case;
	
	if (f_vv) {
		cout << "isomorph::compute_stabilizer "
				"orbit_no=" << orbit_no << " starting at "
				<< first << " case number " << c
			<< " first_orbit_this_case=" << first_orbit_this_case 
			<< " local orbit number " << orb_no << endl;
		}
	
	if (f_v) {
		cout << "isomorph::compute_stabilizer "
				"f=" << f << " l=" << l << endl;
		}

	S = NEW_OBJECT(sims);
	AA = NEW_OBJECT(action);
	gens = NEW_OBJECT(vector_ge);
	Schreier = NEW_OBJECT(schreier);
	sets = NEW_lint(l * size);

	prepare_database_access(level, verbose_level);
	
	load_strong_generators(level, c, 
		*gens, go, verbose_level - 1);
	if (f_v) {
		cout << "isomorph::compute_stabilizer "
				"orbit_no=" << orbit_no
				<< " after load_strong_generators" << endl;
		cout << "isomorph::compute_stabilizer "
				"Stabilizer of starter has order " << go << endl;
		}

	
	S->init(A_base, verbose_level - 2);
	S->init_generators(*gens, FALSE);
	S->compute_base_orbits(0/*verbose_level - 4*/);
	
	if (f_v) {
		cout << "isomorph::compute_stabilizer "
				"The action in the stabilizer sims object is:" << endl;
		S->A->print_info();
		}
	if (f_v) {
		cout << "isomorph::compute_stabilizer "
				"loading " << l
			<< " solutions associated to starter " << c 
			<< " (representative of isomorphism type "
			<< orbit_no << ")" << endl;
		}
	for (j = 0; j < l; j++) {
		load_solution(f + j, sets + j * size);
		Sorting.lint_vec_heapsort(sets + j * size, size);
		}
	if (f_v) {
		cout << "isomorph::compute_stabilizer "
				"The " << l << " solutions are:" << endl;
		if (l < 20) {
			lint_matrix_print(sets, l, size);
			}
		else {
			cout << "isomorph::compute_stabilizer "
					"Too big to print, we print only 20" << endl;
			lint_matrix_print(sets, 20, size);
			}
		}

#if 0	
	gens->init(A);
	gens->allocate(O->nb_strong_generators);
	
	for (j = 0; j < O->nb_strong_generators; j++) {
		A->element_retrieve(O->hdl_strong_generators[j], gens->ith(j), FALSE);
		}
#endif

	if (f_v) {
		cout << "isomorph::compute_stabilizer "
				"computing induced action" << endl;
		}
			
	AA->induced_action_on_sets(*A, S, l, size,
			sets, TRUE, verbose_level - 2);
	
	if (f_v) {
		cout << "isomorph::compute_stabilizer "
				"computing induced action done" << endl;
		}
	AA->group_order(AA_go);
	AA->Kernel->group_order(K_go);
	if (f_v) {
		cout << "isomorph::compute_stabilizer "
				"induced action has order " << AA_go << endl;
		cout << "isomorph::compute_stabilizer "
				"induced action has a kernel of order " << K_go << endl;
		}

	if (f_v) {
		cout << "isomorph::compute_stabilizer "
				"computing all point orbits" << endl;
		}
			
	AA->compute_all_point_orbits(*Schreier, *gens,
			0/*verbose_level - 2*/);


	if (f_v) {
		cout << "isomorph::compute_stabilizer orbit "
				<< orbit_no << " found " << Schreier->nb_orbits
				<< " orbits" << endl;
		}
	
	//Schreier->point_stabilizer(AA, AA_go, stab,
	// orb_no, verbose_level - 2);
	Schreier->point_stabilizer(A_base, go, Stab,
			orb_no, 0 /*verbose_level - 2*/);
	Stab->group_order(so);

	if (f_v) {
		cout << "isomorph::compute_stabilizer "
				"starter set has stabilizer of order "
				<< go << endl;
		cout << "isomorph::compute_stabilizer "
				"orbit " << orb_no << " has length "
				<< Schreier->orbit_len[orb_no] << endl;
		cout << "isomorph::compute_stabilizer "
				"n e w stabilizer has order " << so << endl;
		cout << "isomorph::compute_stabilizer "
				"orbit_no=" << orbit_no << " finished" << endl;
		}

	FREE_OBJECT(S);
	FREE_OBJECT(AA);
	FREE_OBJECT(gens);
	FREE_OBJECT(Schreier);
	FREE_lint(sets);
}

void isomorph::test_compute_stabilizer(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int orbit_no;
	sims *Stab;
	int k;
	os_interface Os;

	if (f_v) {
		cout << "isomorph::test_compute_stabilizer" << endl;
		}
	setup_and_open_solution_database(verbose_level - 1);
	
	for (k = 0; k < 100; k++) {
		orbit_no = Os.random_integer(nb_orbits);
		
		cout << "k=" << k << " orbit_no=" << orbit_no << endl;
		
		compute_stabilizer(Stab, verbose_level);
		
		FREE_OBJECT(Stab);
		}
	
	close_solution_database(verbose_level - 1);
}

void isomorph::test_memory()
{
	orbit_no = 0;
	int verbose_level = 0;
	int id;
	//action *AA;
	sims *Stab;
	long int data[1000];
	
	
	setup_and_open_solution_database(verbose_level - 1);

	compute_stabilizer(Stab, verbose_level);
		
	
	id = orbit_perm[orbit_fst[orbit_no]];
	
	load_solution(id, data);
	
	//cout << "calling induced_action_on_set" << endl;
	//AA = NULL;
	
	while (TRUE) {
		induced_action_on_set(Stab, data, 0/*verbose_level*/);
		}

}

void isomorph::test_edges(int verbose_level)
{
	int *transporter1;
	int *transporter2;
	int *Elt1, *Elt2;
	//int r1, r2;
	int id1, id2;
	long int data1[1000];
	long int data2[1000];
	int subset[1000];
	int i, j, a, b;
	long int subset1[] = {0, 1, 2, 3, 4, 8};

	transporter1 = NEW_int(A->elt_size_in_int);
	transporter2 = NEW_int(A->elt_size_in_int);
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	
	/*r1 =*/ test_edge(1, subset1, transporter1, verbose_level);
	id1 = orbit_perm[orbit_fst[1]];
	
	long int subset2[] = {0, 1, 2, 3, 4, 6 };
	
	/*r2 =*/ test_edge(74, subset2, transporter2, verbose_level);
	id2 = orbit_perm[orbit_fst[74]];
	
	A->element_invert(transporter2, Elt1, FALSE);
	A->element_mult(transporter1, Elt1, Elt2, FALSE);
	A->element_invert(Elt2, Elt1, FALSE);

	setup_and_open_solution_database(verbose_level - 1);

	load_solution(id1, data1);
	load_solution(id2, data2);
	close_solution_database(verbose_level - 1);
	
	if (!A->check_if_transporter_for_set(Elt2,
			size, data1, data2, verbose_level)) {
		cout << "does not map data1 to data2" << endl;
		exit(1);
		}
	for (j = 0; j < level; j++) {
		b = data2[j];
		a = A->element_image_of(b, Elt1, FALSE);
		for (i = 0; i < size; i++) {
			if (data1[i] == a) {
				subset[j] = i;
				break;
				}
			}
		if (i == size) {
			cout << "did not find element a in data1" << endl;
			exit(1);
			}
		}
	cout << "subset: ";
	int_vec_print(cout, subset, level);
	cout << endl;

	FREE_int(transporter1);
	FREE_int(transporter2);
	FREE_int(Elt1);
	FREE_int(Elt2);
	
}

int isomorph::test_edge(int n1,
		long int *subset1, int *transporter, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int r, r0, id, id0;
	long int data1[1000];
	long int data2[1000];
	sorting Sorting;



	setup_and_open_solution_database(verbose_level - 1);
	
	r = n1;
	id = orbit_perm[orbit_fst[r]];
	if (schreier_prev[orbit_fst[r]] != -1) {
		cout << "schreier_prev[orbit_fst[r]] != -1" << endl;
		exit(1);
		}
	//cout << "k=" << k << " r=" << r << endl;
	
	load_solution(id, data1);
		
	Sorting.rearrange_subset_lint_all(size, level, data1,
			subset1, data2, verbose_level - 1);
		
	int f_failure_to_find_point;

	r0 = identify_solution(data2, transporter, 
		f_use_implicit_fusion, f_failure_to_find_point, verbose_level);
	
	if (f_failure_to_find_point) {
		cout << "f_failure_to_find_point" << endl;
		}
	else {
		cout << "r=" << r << " r0=" << r0 << endl;
		id0 = orbit_perm[orbit_fst[r0]];
			
		load_solution(id0, data1);
		if (!A->check_if_transporter_for_set(
				transporter, size, data2, data1, verbose_level)) {
			cout << "test_identify_solution, check fails, stop" << endl;	
			exit(1);
			}
		}
	
	close_solution_database(verbose_level - 1);
	return r0;
}



void isomorph::read_data_files_for_starter(int level, 
	const char *prefix, int verbose_level)
// Calls gen->read_level_file_binary for all levels i from 0 to level
// Uses letter a files for i from 0 to level - 1
// and letter b file for i = level.
// If gen->f_starter is TRUE, we start from i = gen->starter_size instead.
// Finally, it computes nb_starter.
{
	int f_v = (verbose_level >= 1);
	char fname_base_a[1000];
	char fname_base_b[1000];
	int i, i0;
	
	if (f_v) {
		cout << "isomorph::read_data_files_for_starter" << endl;
		cout << "prefix=" << prefix << endl;
		cout << "level=" << level << endl;
		}
	
	sprintf(fname_base_a, "%sa", prefix);
	sprintf(fname_base_b, "%sb", prefix);
	
	if (gen->has_base_case()) {
		i0 = gen->get_Base_case()->size;
		}
	else {
		i0 = 0;
		}
	if (f_v) {
		cout << "isomorph::read_data_files_for_starter "
				"i0=" << i0 << endl;
		}
	for (i = i0; i < level; i++) {
		if (f_v) {
			cout << "reading data file for level "
					<< i << " with prefix " << fname_base_b << endl;
			}
		gen->read_level_file_binary(i, fname_base_b,
				MINIMUM(1, verbose_level - 1));
		}

	if (f_v) {
		cout << "reading data file for level " << level
				<< " with prefix " << fname_base_a << endl;
		}
	gen->read_level_file_binary(level, fname_base_a,
			MINIMUM(1, verbose_level - 1));

	compute_nb_starter(level, verbose_level);

	if (f_v) {
		cout << "isomorph::read_data_files_for_starter finished, "
				"number of starters = " << nb_starter << endl;
		}
}

void isomorph::compute_nb_starter(int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	nb_starter = gen->nb_orbits_at_level(level);
	if (f_v) {
		cout << "isomorph::compute_nb_starter finished, "
				"number of starters = " << nb_starter << endl;
		}

}

void isomorph::print_node_local(int level, int node_local)
{
	int n;

	n = gen->first_node_at_level(level) + node_local;
	cout << n << "=" << level << "/" << node_local;
}

void isomorph::print_node_global(int level, int node_global)
{
	int node_local;

	node_local = node_global - gen->first_node_at_level(level);
	cout << node_global << "=" << level << "/" << node_local;
}

void isomorph::test_hash(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	long int data[1000];
	int id, case_nb, f, l, i;
	int *H;
	sorting Sorting;


	if (f_v) {
		cout << "isomorph::test_hash" << endl;
		}
	setup_and_open_solution_database(verbose_level - 1);
	for (case_nb = 0; case_nb < nb_starter; case_nb++) {
		f = solution_first[case_nb];
		l = solution_len[case_nb];
		if (l == 1) {
			continue;
			}
		cout << "starter " << case_nb << " f=" << f << " l=" << l << endl;
		H = NEW_int(l);
		for (i = 0; i < l; i++) {
			//id = orbit_perm[f + i];
			id = f + i;
			load_solution(id, data);
			Sorting.lint_vec_heapsort(data, size);
			H[i] = lint_vec_hash(data, size);
			}
		{
		classify C;
		C.init(H, l, TRUE, 0);
		C.print(FALSE /*f_backwards*/);
		}
		FREE_int(H);
		}

	close_solution_database(verbose_level - 1);	
}


void isomorph::compute_Ago_Ago_induced(longinteger_object *&Ago,
		longinteger_object *&Ago_induced, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int h, rep, first, /*c,*/ id;
	long int data[1000];
	
	if (f_v) {
		cout << "isomorph::compute_Ago_Ago_induced" << endl;
		}
	Ago = NEW_OBJECTS(longinteger_object, Reps->count);
	Ago_induced = NEW_OBJECTS(longinteger_object, Reps->count);


	for (h = 0; h < Reps->count; h++) {
		if (f_vv) {
			cout << "isomorph::compute_Ago_Ago_induced orbit "
					<< h << " / " << Reps->count << endl;
			}
		rep = Reps->rep[h];
		first = orbit_fst[rep];
		//c = starter_number[first];
		id = orbit_perm[first];		
		load_solution(id, data);

		sims *Stab;
		
		Stab = Reps->stab[h];

		Stab->group_order(Ago[h]);
		//f << "Stabilizer has order $";
		//go.print_not_scientific(f);
		if (f_vvv) {
			cout << "isomorph::compute_Ago_Ago_induced computing "
					"induced action on the set (in data)" << endl;
			}
		induced_action_on_set_basic(Stab, data, 0 /*verbose_level*/);
		
			
		AA->group_order(Ago_induced[h]);
		}

	if (f_v) {
		cout << "isomorph::compute_Ago_Ago_induced done" << endl;
		}

}

void isomorph::init_high_level(action *A, poset_classification *gen,
	int size, char *prefix_classify, char *prefix, int level,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph::init_high_level" << endl;
		}

	
	discreta_init();

	int f_use_database_for_starter = FALSE;
	int f_implicit_fusion = FALSE;
	
	if (f_v) {
		cout << "isomorph::init_high_level before init" << endl;
		}
	init(prefix, A, A, gen, 
		size, level, 
		f_use_database_for_starter, 
		f_implicit_fusion, 
		verbose_level);
		// sets q, level and initializes file names
	if (f_v) {
		cout << "isomorph::init_high_level after init" << endl;
		}


	

	if (f_v) {
		cout << "isomorph::init_high_level "
				"before read_data_files_for_starter" << endl;
		}

	read_data_files_for_starter(level,
			prefix_classify, verbose_level);

	if (f_v) {
		cout << "isomorph::init_high_level "
				"before init_solution" << endl;
		}

	init_solution(verbose_level);
	
	if (f_v) {
		cout << "isomorph::init_high_level "
				"after init_solution" << endl;
		}


	if (f_v) {
		cout << "isomorph::init_high_level "
				"before read_orbit_data" << endl;
		}

	read_orbit_data(verbose_level);

	if (f_v) {
		cout << "isomorph::init_high_level "
				"after read_orbit_data" << endl;
		}


	depth_completed = level /*- 2*/;

	if (f_v) {
		cout << "isomorph::init_high_level "
				"before iso_test_init" << endl;
		}
	iso_test_init(verbose_level);
	if (f_v) {
		cout << "isomorph::init_high_level "
				"after iso_test_init" << endl;
		}

	if (f_v) {
		cout << "isomorph::init_high_level "
				"before Reps->load" << endl;
		}
	Reps->load(verbose_level);
	if (f_v) {
		cout << "isomorph::init_high_level "
				"after Reps->load" << endl;
		}

	if (f_v) {
		cout << "isomorph::init_high_level "
				"before setup_and_open_solution_database" << endl;
		}
	setup_and_open_solution_database(verbose_level - 1);

	if (f_v) {
		cout << "isomorph::init_high_level "
				"before setup_and_open_level_database" << endl;
		}
	setup_and_open_level_database(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::init_high_level done" << endl;
		}
}


void isomorph::get_orbit_transversal(orbit_transversal *&T,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph::get_orbit_transversal" << endl;
		}
	int h, rep, first, id;
	longinteger_object go;

	T = NEW_OBJECT(orbit_transversal);

	T->A = A_base;
	T->A2 = A;
	T->nb_orbits = Reps->count;
	T->Reps = NEW_OBJECTS(set_and_stabilizer, nb_orbits);


	for (h = 0; h < Reps->count; h++) {
		rep = Reps->rep[h];
		first = orbit_fst[rep];
		id = orbit_perm[first];

		long int *data;
		data = NEW_lint(size);

		load_solution(id, data);

		sims *Stab;

		Stab = Reps->stab[h];
		//T->Reps[h].init_data(data, size, 0 /* verbose_level */);

		strong_generators *SG;

		SG = NEW_OBJECT(strong_generators);

		SG->init_from_sims(Stab, 0 /* verbose_level */);
		T->Reps[h].init_everything(A_base, A, data, size,
				SG, verbose_level);

	}
	if (f_v) {
		cout << "isomorph::get_orbit_transversal done" << endl;
		}
}


}}

