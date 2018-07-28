// isomorph.C
// 
// Anton Betten
// started 2007
// moved here from reader2.C: 3/22/09
// renamed isomorph.C from global.C: 7/14/11
//
// 
//
//

#include "orbiter.h"


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
	
	starter_orbit_fst = NULL;
	starter_nb_orbits = NULL;
	
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
	//INT i;
	INT f_v = FALSE;

	if (f_v) {
		cout << "isomorph::free" << endl;
		}

#if 0
	if (f_v) {
		cout << "isomorph::free before deleting A" << endl;
		}
	if (A) {
		delete A;
		}
#endif
	if (f_v) {
		cout << "isomorph::free before deleting AA" << endl;
		}
	if (AA) {
		delete AA;
		AA = NULL;
		}
#if 0
	if (f_v) {
		cout << "isomorph::free before deleting gen" << endl;
		}
	if (gen) {
		delete gen;
		gen = NULL;
		}
#endif

	if (f_v) {
		cout << "isomorph::free before deleting stabilizer_recreated" << endl;
		}
	if (stabilizer_recreated) {
		delete stabilizer_recreated;
		stabilizer_recreated = NULL;
		}
	if (f_v) {
		cout << "isomorph::free before deleting DB_sol" << endl;
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
		FREE_INT(id_to_datref);
		id_to_datref = NULL;
		}
	if (id_to_hash) {
		FREE_INT(id_to_hash);
		id_to_hash = NULL;
		}
	if (hash_vs_id_hash) {
		FREE_INT(hash_vs_id_hash);
		hash_vs_id_hash = NULL;
		}
	if (hash_vs_id_id) {
		FREE_INT(hash_vs_id_id);
		hash_vs_id_id = NULL;
		}
	if (table_of_solutions) {
		FREE_INT(table_of_solutions);
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
	tmp_set1 = NEW_INT(size);
	tmp_set2 = NEW_INT(size);
	tmp_set3 = NEW_INT(size);
	tmp_Elt1 = NEW_INT(A->elt_size_in_INT);
	tmp_Elt2 = NEW_INT(A->elt_size_in_INT);
	tmp_Elt3 = NEW_INT(A->elt_size_in_INT);

	trace_set_recursion_tmp_set1 = NEW_INT(size);
	trace_set_recursion_Elt1 = NEW_INT(A->elt_size_in_INT);
	
	apply_fusion_tmp_set1 = NEW_INT(size);
	apply_fusion_Elt1 = NEW_INT(A->elt_size_in_INT);
	
	find_extension_set1 = NEW_INT(size);

	make_set_smaller_set = NEW_INT(size);
	make_set_smaller_Elt1 = NEW_INT(A->elt_size_in_INT);
	make_set_smaller_Elt2 = NEW_INT(A->elt_size_in_INT);

	orbit_representative_Elt1 = NEW_INT(A->elt_size_in_INT);
	orbit_representative_Elt2 = NEW_INT(A->elt_size_in_INT);

	handle_automorphism_Elt1 = NEW_INT(A->elt_size_in_INT);
	
	v = new Vector[1];

}

void isomorph::free_tmp_data()
{
	INT f_v = FALSE;
	
	if (f_v) {
		cout << "isomorph::free_tmp_data" << endl;
		}
	if (f_tmp_data_has_been_allocated) {
		f_tmp_data_has_been_allocated = FALSE;
		FREE_INT(tmp_set1);
		FREE_INT(tmp_set2);
		FREE_INT(tmp_set3);
		FREE_INT(tmp_Elt1);
		FREE_INT(tmp_Elt2);
		FREE_INT(tmp_Elt3);
		FREE_INT(trace_set_recursion_tmp_set1);
		FREE_INT(trace_set_recursion_Elt1);
		FREE_INT(apply_fusion_tmp_set1);
		FREE_INT(apply_fusion_Elt1);
		FREE_INT(make_set_smaller_set);
		FREE_INT(make_set_smaller_Elt1);
		FREE_INT(make_set_smaller_Elt2);
		FREE_INT(orbit_representative_Elt1);
		FREE_INT(orbit_representative_Elt2);
		FREE_INT(handle_automorphism_Elt1);
		delete [] v;
		}
	null_tmp_data();
	if (f_v) {
		cout << "isomorph::free_tmp_data finished" << endl;
		}
}

void isomorph::init(const BYTE *prefix, 
	action *A_base, action *A, generator *gen, 
	INT size, INT level, 
	INT f_use_database_for_starter, 
	INT f_implicit_fusion, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE cmd[1000];

	if (f_v) {
		cout << "isomorph::init" << endl;
		cout << "prefix=" << prefix << endl;
		cout << "A_base=" << A_base->label << endl;
		cout << "A=" << A->label << endl;
		cout << "size=" << size << endl;
		cout << "level=" << level << endl;
		cout << "f_use_database_for_starter=" << f_use_database_for_starter << endl;
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
		sprintf(fname_data_file, "%s_%ld.data", prefix, level - 1);
		}
	else {
		sprintf(fname_data_file, "%s_%ld.data", prefix, level);
		}
	if (f_v) {
		cout << "fname_data_file=" << fname_data_file << endl;
		}
	sprintf(fname_level_file, "%s_lvl_%ld", prefix, level);
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
	sprintf(fname_orbits_of_stabilizer_csv, "%sorbits_of_stabilizer.csv", prefix);
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







void isomorph::init_solution(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
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

void isomorph::load_table_of_solutions(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT id, j;
	INT data[1000];

	if (f_v) {
		cout << "isomorph::load_table_of_solutions N=" << N << endl;
		}
	setup_and_open_solution_database(verbose_level);
	table_of_solutions = NEW_INT(N * size);
	for (id = 0; id < N; id++) {
		load_solution(id, data);
		for (j = 0; j < size; j++) {
			table_of_solutions[id * size + j] = data[j];
			}
#if 0
		cout << "solution " << id << " : ";
		INT_vec_print(cout, table_of_solutions + id * size, size);
		cout << endl;
#endif
		}
	f_use_table_of_solutions = TRUE;
	close_solution_database(verbose_level);
	if (f_v) {
		cout << "isomorph::load_table_of_solutions done" << endl;
		}
}

void isomorph::init_starter_number(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, f, l;
	
	if (f_v) {
		cout << "isomorph::init_starter_number N=" << N << endl;
		}
	starter_number = NEW_INT(N);
	for (i = 0; i < nb_starter; i++) {
		f = solution_first[i];
		l = solution_len[i];
		for (j = 0; j < l; j++) {
			starter_number[f + j] = i;
			}
		}
	if (f_v) {
		cout << "starter_number:" << endl;
		INT_vec_print(cout, starter_number, N);
		cout << endl;
		}
}


void isomorph::list_solutions_by_starter()
{
	INT i, j, idx, id, f, l, fst, len, h, pos, u;
	INT data[1000];
	INT data2[1000];
	INT verbose_level = 0;
	
	setup_and_open_solution_database(verbose_level - 1);
	
	j = 0;
	for (i = 0; i < nb_starter; i++) {
		f = solution_first[i];
		l = solution_len[i];
		cout << "starter " << i << " solutions from=" << f << " len=" << l << endl;
		pos = f;
		while (pos < f + l) {
			fst = orbit_fst[j];
			len = orbit_len[j];
			cout << "orbit " << j << " from=" << fst << " len=" << len << endl;
			for (u = 0; u < len; u++) {
				idx = fst + u;
				id = orbit_perm[idx];
				load_solution(id, data);
				for (h = 0; h < size; h++) {
					data2[h] = data[h];
					}
				INT_vec_heapsort(data2, size);
				cout << i << " : " << j << " : " << idx << " : " << id << endl;
				INT_vec_print(cout, data, size);
				cout << endl;
				INT_vec_print(cout, data2, size);
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
	INT i, j, idx, id, f, l, h;
	INT data[1000];
	INT data2[1000];
	INT verbose_level = 0;
	
	setup_and_open_solution_database(verbose_level - 1);

	for (i = 0; i < nb_orbits; i++) {
		f = orbit_fst[i];
		l = orbit_len[i];
		cout << "orbit " << i << " from=" << f << " len=" << l << endl;
		for (j = 0; j < l; j++) {
			idx = f + j;
			id = orbit_perm[idx];
			load_solution(id, data);
			for (h = 0; h < size; h++) {
				data2[h] = data[h];
				}
			INT_vec_heapsort(data2, size);
			cout << j << " : " << idx << " : " << id << endl;
			INT_vec_print(cout, data, size);
			cout << endl;
			INT_vec_print(cout, data2, size);
			cout << endl;
			}
		}

	close_solution_database(verbose_level);
}

void isomorph::orbits_of_stabilizer(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvvv = (verbose_level >= 4);
	INT f_v5 = (verbose_level >= 5);
	INT i, j, f, l, nb_orbits_prev = 0;
	longinteger_object go;

	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer" << endl;
		cout << "number of starters = nb_starter = " << nb_starter << endl;
		cout << "number of solutions (= N) = " << N << endl;
		cout << "action A_base=";
		A_base->print_info();
		cout << endl;
		cout << "action A=";
		A->print_info();
		cout << endl;
		}

	setup_and_open_solution_database(verbose_level - 1);
	setup_and_open_level_database(verbose_level - 1);


	prepare_database_access(level, verbose_level - 1);


	nb_orbits = 0;
	orbit_fst = NEW_INT(N + 1);
	orbit_len = NEW_INT(N);
	orbit_number = NEW_INT(N);
	orbit_perm = NEW_INT(N);
	orbit_perm_inv = NEW_INT(N);
	schreier_vector = NEW_INT(N);
	schreier_prev = NEW_INT(N);

	// added Dec 25, 2012:

	starter_orbit_fst = NEW_INT(nb_starter);
	starter_nb_orbits = NEW_INT(nb_starter);

	for (i = 0; i < N; i++) {
		schreier_vector[i] = -2;
		schreier_prev[i] = -1;
		}
	
	orbit_fst[0] = 0;
	for (i = 0; i < nb_starter; i++) {
		if (f_v) {
			cout << "isomorph::orbits_of_stabilizer case i=" << i << " / " << nb_starter << endl;
			}

		starter_orbit_fst[i] = nb_orbits;
		starter_nb_orbits[i] = 0;

		//oracle *O;
		vector_ge gens;
		
		//O = &gen->root[gen->first_oracle_node_at_level[level] + i];
		
		
		load_strong_generators(level, 
			i, 
			gens, go, verbose_level - 2);
		if (f_v5) {
			cout << "isomorph::orbits_of_stabilizer after load_strong_generators" << endl;
			cout << "isomorph::orbits_of_stabilizer The stabilizer is a group of order " << go << " with " << gens.len << " strong generators" << endl;
			gens.print_with_given_action(cout, A_base);
			}
		
		f = solution_first[i];
		l = solution_len[i];
		if (f_v && ((i % 5000) == 0)) {
			cout << "isomorph::orbits_of_stabilizer Case " << i << " / " << nb_starter << endl;
			}
		if (f_vv) {
			cout << "isomorph::orbits_of_stabilizer nb_orbits = " << nb_orbits << endl;
			cout << "isomorph::orbits_of_stabilizer case " << i << " starts at " << f << " with " << l << " solutions" << endl;
			}
		if (gens.len == 0 /*O->nb_strong_generators == 0*/) {
			if (f_vv) {
				cout << "isomorph::orbits_of_stabilizer the stabilizer is trivial" << endl;
				}
			for (j = 0; j < l; j++) {
				orbit_len[nb_orbits] = 1;
				schreier_vector[f + j] = -1;
				orbit_number[f + j] = nb_orbits;
				orbit_perm[f + j] = f + j;
				orbit_perm_inv[f + j] = f + j;
				nb_orbits++;
				orbit_fst[nb_orbits] = orbit_fst[nb_orbits - 1] + orbit_len[nb_orbits - 1];
				starter_nb_orbits[i]++;
				}
			}
		else {
			if (f_vv) {
				cout << "isomorph::orbits_of_stabilizer the stabilizer is non trivial" << endl;
				}
			if (solution_len[i] != 0) {
				if (f_vv) {
					cout << "isomorph::orbits_of_stabilizer before orbits_of_stabilizer_case" << endl;
					}
				orbits_of_stabilizer_case(i, gens, verbose_level - 2);
				if (f_vv) {
					cout << "isomorph::orbits_of_stabilizer after orbits_of_stabilizer_case" << endl;
					cout << "isomorph::orbits_of_stabilizer the " << l << " solutions in case " << i << " fall into " << nb_orbits - nb_orbits_prev << " orbits" << endl;
					}
				starter_nb_orbits[i] = nb_orbits - nb_orbits_prev;
				}
			}
		if (f_v) {
			cout << "isomorph::orbits_of_stabilizer Case " << i << " / " << nb_starter << " finished, we found " << nb_orbits - nb_orbits_prev << " orbits : ";
			if (nb_orbits - nb_orbits_prev) {
				classify C;

				C.init(orbit_len + nb_orbits_prev, nb_orbits - nb_orbits_prev, FALSE, 0);
				C.print_naked(TRUE /* f_backwards */);
				cout << endl;
				}
			else {
				cout << endl;
				}
			}
		if (FALSE && f_vvvv) {
			cout << "i : orbit_perm : orbit_number : schreier_vector : schreier_prev" << endl;
			for (j = 0; j < l; j++) {
				cout << f + j << " : " 
					<< orbit_perm[f + j] << " : " 
					<< orbit_number[f + j] << " : " 
					<< schreier_vector[f + j] << " : " 
					<< schreier_prev[f + j] << endl;
				}
			cout << "j : orbit_fst : orbit_len" << endl;
			for (j = nb_orbits_prev; j < nb_orbits; j++) {
				cout << j << " : " << orbit_fst[j] << " : " << orbit_len[j] << endl;
				}
			cout << j << " : " << orbit_fst[j] << endl;
			if (orbit_fst[nb_orbits] != solution_first[i + 1]) {
				cout << "orbit_fst[nb_orbits] != solution_first[i + 1]" << endl;
				cout << "orbit_fst[nb_orbits]=" << orbit_fst[nb_orbits] << endl;
				cout << "solution_first[i + 1]=" << solution_first[i + 1] << endl;
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
		cout << "isomorph::orbits_of_stabilizer Case " << i << " / " << nb_starter << " finished, we found " << nb_orbits << " orbits : ";
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
		cout << "i : orbit_number[i] : orbit_perm[i] : schreier_vector[i] : schreier_prev[i]" << endl;
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

void isomorph::orbits_of_stabilizer_case(INT the_case, vector_ge &gens, INT verbose_level)
{
	Vector v;
	//oracle *O;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_v4 = (verbose_level >= 4);
	INT j, f, l, k, ff, ll;
	
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer_case " << the_case << " / " << nb_starter << endl;
		}
	
	//O = &gen->root[gen->first_oracle_node_at_level[level] + the_case];
	f = solution_first[the_case];
	l = solution_len[the_case];
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer_case solution_first[the_case] = " << f << endl;
		cout << "isomorph::orbits_of_stabilizer_case solution_len[the_case] = " << l << endl;
		}

	longinteger_object S_go;
	sims *S;
	action *AA;
	schreier *Schreier;
	INT *sets;
	INT h, p, prev, b, hdl;
			
	sets = NEW_INT(l * size);
	S = new sims;
	AA = new action;
	Schreier = new schreier;
			
		
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case generators as permutations (skipped)" << endl;
		//gens.print_as_permutation(cout);
		}	
	S->init(A_base);
	S->init_generators(gens, FALSE);
	S->compute_base_orbits(0 /*verbose_level - 2*/);
	S->group_order(S_go);
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer_case The starter has a stabilizer of order " << S_go << endl;
		}
			
	for (j = 0; j < l; j++) {

		load_solution(f + j, sets + j * size);
		if (FALSE && f_vv) {
			cout << "solution " << j << "        : ";
			INT_vec_print(cout, sets + j * size, size);
			cout << endl;
			}
		INT_vec_heapsort(sets + j * size, size);
		if (FALSE && f_vv) {
			cout << "solution " << j << " sorted : ";
			INT_vec_print(cout, sets + j * size, size);
			cout << endl;
			}
		}
	
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case computing induced action" << endl;
		}
			
	AA->induced_action_on_sets(*A, S, //K, 
		l, size, sets, FALSE /*TRUE*/ /* A Betten 1/26/13*/, verbose_level /*- 2*/);

	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case computing induced action finished" << endl;
		}
		
#if 0	
	AA->group_order(AA_go);
	AA->Kernel->group_order(K_go);
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer_case orbit " << nb_orbits << " induced action has order " << AA_go << ", kernel has order " << K_go << endl;
		}
#endif
	
	if (f_vv) {
		cout << "isomorph::orbits_of_stabilizer_case induced action computed" << endl;
		cout << "generators:" << endl;
		for (k = 0; k < gens.len; k++) {
			cout << k << " : ";
			//AA->element_print_as_permutation(gens.ith(k), cout);
			cout << endl;
			}
		}
	
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer_case computing point orbits" << endl;
		}
	AA->compute_all_point_orbits(*Schreier, gens, verbose_level - 4);
	//AA->all_point_orbits(*Schreier, verbose_level - 2);
			
	if (f_v) {
		cout << "isomorph::orbits_of_stabilizer_case Point orbits computed" << endl;
		}
	if (f_v4) {
		Schreier->print_tables(cout, TRUE);
		}

	for (k = 0; k < l; k++) {
		p = Schreier->orbit[k];
		prev = Schreier->prev[k];
		hdl = Schreier->label[k];
		//cout << "coset " << k << " point p=" << p << " prev=" << prev << " label " << hdl << endl;
		if (prev != -1) {
			//A->element_retrieve(O->hdl_strong_generators[hdl], A->Elt1, FALSE);
			b = AA->element_image_of(prev, gens.ith(hdl), FALSE);
			//cout << "image of " << prev << " results in =" << b << endl;
			if (b != p) {
				cout << "b != p" << endl;
				exit(1);
				}
			if (!A->check_if_transporter_for_set(gens.ith(hdl), size, 
				sets + prev * size, sets + p * size, verbose_level - 2)) {
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
			schreier_vector[f + ff + h] = Schreier->label[ff + h];
			if (h == 0) {
				schreier_prev[f + ff + h] = -1;
				}
			else {
				schreier_prev[f + ff + h] = f + Schreier->prev[ff + h];
				}
			}
		orbit_len[nb_orbits] = ll;
		nb_orbits++;
		orbit_fst[nb_orbits] = orbit_fst[nb_orbits - 1] + ll;
		}
			
	FREE_INT(sets);
	delete S;
	delete AA;
	delete Schreier;
	
}


void isomorph::orbit_representative(INT i, INT &i0, 
	INT &orbit, INT *transporter, INT verbose_level)
// slow because it calls load_strong_generators
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT c, p, i_loc, l; //, hdl;
	//oracle *O;
	INT *Elt1, *Elt2;
	vector_ge gens;
	longinteger_object go;
	
	if (f_v) {
		cout << "isomorph::orbit_representative" << endl;
		}


	prepare_database_access(level, verbose_level);




	Elt1 = orbit_representative_Elt1;
	Elt2 = orbit_representative_Elt2;
	c = starter_number[i];
	//O = &gen->root[gen->first_oracle_node_at_level[level] + c];
	if (f_v) {
		cout << "isomorph::orbit_representative before load_strong_generators" << endl;
		}
	load_strong_generators(level, c, 
		gens, go, verbose_level);
	if (f_v) {
		cout << "isomorph::orbit_representative after load_strong_generators" << endl;
		}
	A->element_one(transporter, FALSE);
	if (f_vv) {
		cout << "isomorph::orbit_representative i=" << i << endl;
		}
	while (TRUE) {
		i_loc = orbit_perm_inv[i];
		p = schreier_prev[i_loc];
		if (f_vv) {
			cout << "isomorph::orbit_representative i=" << i << " i_loc=" << i_loc << " p=" << p << endl;
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
		cout << "isomorph::orbit_representative The representative of solution " << i << " is " << i0 << " in orbit " << orbit << endl;
		}
}

void isomorph::test_orbit_representative(INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT r, r0, orbit, k;
	INT data1[1000];
	INT data2[1000];
	INT *transporter;
	
	transporter = NEW_INT(A->elt_size_in_INT);

	setup_and_open_solution_database(verbose_level - 1);
	
	for (k = 0; k < N; k++) {
		r = k;
		//r = random_integer(N);
		//cout << "k=" << k << " r=" << r << endl;
	
		load_solution(r, data1);

		orbit_representative(r, r0, orbit, transporter, verbose_level);
		if (r != r0) {
			cout << "k=" << k << " r=" << r << " r0=" << r0 << endl;
			}
			
		load_solution(r0, data2);
		if (!A->check_if_transporter_for_set(transporter, size, data1, data2, verbose_level)) {
			exit(1);
			}
		}
	
	close_solution_database(verbose_level - 1);
	FREE_INT(transporter);
}

void isomorph::test_identify_solution(INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT r, r0, id, id0;
	INT data1[1000];
	INT data2[1000];
	INT perm[1000];
	INT i, k;
	INT *transporter;
	
	transporter = NEW_INT(A->elt_size_in_INT);


	setup_and_open_solution_database(verbose_level - 1);
	
	for (k = 0; k < 10; k++) {
		r = random_integer(nb_orbits);
		id = orbit_perm[orbit_fst[r]];
		if (schreier_prev[orbit_fst[r]] != -1) {
			cout << "schreier_prev[orbit_fst[r]] != -1" << endl;
			exit(1);
			}
		//cout << "k=" << k << " r=" << r << endl;
	
		load_solution(id, data1);
		random_permutation(perm, size);
		for (i = 0; i < size; i++) {
			data2[i] = data1[perm[i]];
			}

		INT f_failure_to_find_point;
		r0 = identify_solution(data2, transporter, f_use_implicit_fusion, f_failure_to_find_point, verbose_level - 2);
		
		if (f_failure_to_find_point) {
			cout << "f_failure_to_find_point" << endl;
			}
		else {
			cout << "k=" << k << " r=" << r << " r0=" << r0 << endl;
			id0 = orbit_perm[orbit_fst[r0]];
			
			load_solution(id0, data1);
			if (!A->check_if_transporter_for_set(transporter, size, data2, data1, verbose_level)) {
				cout << "test_identify_solution, check fails, stop" << endl;
				exit(1);
				}
			}
		}
	
	close_solution_database(verbose_level - 1);
	FREE_INT(transporter);
}

void isomorph::compute_stabilizer(sims *&Stab, INT verbose_level)
// Called from do_iso_test
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	//INT f_vvvv = (verbose_level >= 4);
	longinteger_object AA_go, K_go;
	sims *S; //, *K; //, *stab;
	action *AA;
	vector_ge *gens;
	schreier *Schreier;
	INT *sets;
	INT j, first, f, l, c, first_orbit_this_case, orb_no;
	//oracle *O;
	longinteger_object go, so, so1;
	//longinteger_domain DO;

	if (f_v) {
		cout << "isomorph::compute_stabilizer iso_node " << iso_nodes << endl;
		}
	
	first = orbit_fst[orbit_no];
	c = starter_number[first];
	f = solution_first[c];
	l = solution_len[c];
	first_orbit_this_case = orbit_number[f];
	orb_no = orbit_no - first_orbit_this_case;
	
	if (f_vv) {
		cout << "isomorph::compute_stabilizer orbit_no=" << orbit_no << " starting at " << first << " case number " << c 
			<< " first_orbit_this_case=" << first_orbit_this_case 
			<< " local orbit number " << orb_no << endl;
		}
	
	if (f_v) {
		cout << "isomorph::compute_stabilizer f=" << f << " l=" << l << endl;
		}

	S = new sims;
	AA = new action;
	gens = new vector_ge;
	Schreier = new schreier;
	sets = NEW_INT(l * size);

	prepare_database_access(level, verbose_level);
	
	load_strong_generators(level, c, 
		*gens, go, verbose_level - 1);
#if 0
	O = &gen->root[gen->first_oracle_node_at_level[level] + c];

	if (O->nb_strong_generators)
		DO.multiply_up(go, O->tl, A->base_len);
	else
		go.create(1);
#endif
	if (f_v) {
		cout << "isomorph::compute_stabilizer orbit_no=" << orbit_no << " after load_strong_generators" << endl;
		cout << "isomorph::compute_stabilizer Stabilizer of starter has order " << go << endl;
		}

	
	S->init(A_base);
	S->init_generators(*gens, FALSE);
	S->compute_base_orbits(0/*verbose_level - 4*/);
	
	if (f_v) {
		cout << "isomorph::compute_stabilizer The action in the stabilizer sims object is:" << endl;
		S->A->print_info();
		}
	if (f_v) {
		cout << "isomorph::compute_stabilizer loading " << l 
			<< " solutions associated to starter " << c 
			<< " (representative of isomorphism type " << orbit_no << ")" << endl;
		}
	for (j = 0; j < l; j++) {
		load_solution(f + j, sets + j * size);
		INT_vec_heapsort(sets + j * size, size);
		}
	if (f_v) {
		cout << "isomorph::compute_stabilizer The " << l << " solutions are:" << endl;
		if (l < 20) {
			INT_matrix_print(sets, l, size);
			}
		else {
			cout << "isomorph::compute_stabilizer Too big to print, we print only 20" << endl;
			INT_matrix_print(sets, 20, size);
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
		cout << "isomorph::compute_stabilizer computing induced action" << endl;
		}
			
	AA->induced_action_on_sets(*A, S, l, size, sets, TRUE, verbose_level - 2);
	
	if (f_v) {
		cout << "isomorph::compute_stabilizer computing induced action done" << endl;
		}
	AA->group_order(AA_go);
	AA->Kernel->group_order(K_go);
	if (f_v) {
		cout << "isomorph::compute_stabilizer induced action has order " << AA_go << endl;
		cout << "isomorph::compute_stabilizer induced action has a kernel of order " << K_go << endl;
		}

	if (f_v) {
		cout << "isomorph::compute_stabilizer computing all point orbits" << endl;
		}
			
	AA->compute_all_point_orbits(*Schreier, *gens, 0/*verbose_level - 2*/);


	if (f_v) {
		cout << "isomorph::compute_stabilizer orbit " << orbit_no << " found " << Schreier->nb_orbits << " orbits" << endl;
		}
	
	//Schreier->point_stabilizer(AA, AA_go, stab, orb_no, verbose_level - 2);
	Schreier->point_stabilizer(A_base, go, Stab, orb_no, 0 /*verbose_level - 2*/);
	Stab->group_order(so);

	if (f_v) {
		cout << "isomorph::compute_stabilizer starter set has stabilizer of order " << go << endl;
		cout << "isomorph::compute_stabilizer orbit " << orb_no << " has length " << Schreier->orbit_len[orb_no] << endl;
		cout << "isomorph::compute_stabilizer new stabilizer has order " << so << endl;
		cout << "isomorph::compute_stabilizer orbit_no=" << orbit_no << " finished" << endl;
		}

	delete S;
	delete AA;
	delete gens;
	delete Schreier;
	FREE_INT(sets);
}

void isomorph::test_compute_stabilizer(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT orbit_no;
	sims *Stab;
	INT k;
	
	if (f_v) {
		cout << "isomorph::test_compute_stabilizer" << endl;
		}
	setup_and_open_solution_database(verbose_level - 1);
	
	for (k = 0; k < 100; k++) {
		orbit_no = random_integer(nb_orbits);
		
		cout << "k=" << k << " orbit_no=" << orbit_no << endl;
		
		compute_stabilizer(Stab, verbose_level);
		
		delete Stab;
		}
	
	close_solution_database(verbose_level - 1);
}

void isomorph::test_memory()
{
	orbit_no = 0;
	INT verbose_level = 0;
	INT id;
	//action *AA;
	sims *Stab;
	INT data[1000];
	
	
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

void isomorph::test_edges(INT verbose_level)
{
	INT *transporter1;
	INT *transporter2;
	INT *Elt1, *Elt2;
	//INT r1, r2;
	INT id1, id2;
	INT data1[1000];
	INT data2[1000];
	INT subset[1000];
	INT i, j, a, b;
	INT subset1[] = {0, 1, 2, 3, 4, 8};

	transporter1 = NEW_INT(A->elt_size_in_INT);
	transporter2 = NEW_INT(A->elt_size_in_INT);
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	
	/*r1 =*/ test_edge(1, subset1, transporter1, verbose_level);
	id1 = orbit_perm[orbit_fst[1]];
	
	INT subset2[] = {0, 1, 2, 3, 4, 6 };
	
	/*r2 =*/ test_edge(74, subset2, transporter2, verbose_level);
	id2 = orbit_perm[orbit_fst[74]];
	
	A->element_invert(transporter2, Elt1, FALSE);
	A->element_mult(transporter1, Elt1, Elt2, FALSE);
	A->element_invert(Elt2, Elt1, FALSE);

	setup_and_open_solution_database(verbose_level - 1);

	load_solution(id1, data1);
	load_solution(id2, data2);
	close_solution_database(verbose_level - 1);
	
	if (!A->check_if_transporter_for_set(Elt2, size, data1, data2, verbose_level)) {
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
	INT_vec_print(cout, subset, level);
	cout << endl;

	FREE_INT(transporter1);
	FREE_INT(transporter2);
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	
}

INT isomorph::test_edge(INT n1, INT *subset1, INT *transporter, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT r, r0, id, id0;
	INT data1[1000];
	INT data2[1000];
	


	setup_and_open_solution_database(verbose_level - 1);
	
	r = n1;
	id = orbit_perm[orbit_fst[r]];
	if (schreier_prev[orbit_fst[r]] != -1) {
		cout << "schreier_prev[orbit_fst[r]] != -1" << endl;
		exit(1);
		}
	//cout << "k=" << k << " r=" << r << endl;
	
	load_solution(id, data1);
		
	rearrange_subset(size, level, data1, subset1, data2, verbose_level - 1);
		
	INT f_failure_to_find_point;

	r0 = identify_solution(data2, transporter, 
		f_use_implicit_fusion, f_failure_to_find_point, verbose_level);
	
	if (f_failure_to_find_point) {
		cout << "f_failure_to_find_point" << endl;
		}
	else {
		cout << "r=" << r << " r0=" << r0 << endl;
		id0 = orbit_perm[orbit_fst[r0]];
			
		load_solution(id0, data1);
		if (!A->check_if_transporter_for_set(transporter, size, data2, data1, verbose_level)) {
			cout << "test_identify_solution, check fails, stop" << endl;	
			exit(1);
			}
		}
	
	close_solution_database(verbose_level - 1);
	return r0;
}

#if 0
void isomorph::read_data_file(INT f_recompute_schreier, 
	INT verbose_level)
// Reads the data for starters.
// First, it reads the data file whose name is fname_data_file,
// which contains the data up to level - 1
// and sets depth_completed to one level less, which is level - 2.
// Then, the bottom two levels are read from the level databases 
// through the object database D.
// In this step, it calls init_DB_level.
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i;
	
	if (f_v) {
		cout << "isomorph::read_data_file: reading file " << fname_data_file << endl;
		}
	depth_completed = 0;
	
	generator_read_data_file(gen, 
		depth_completed, 
		fname_data_file, 
		verbose_level);
		// in DISCRETA/snakesandladders.C
		// fname_data_file correspond to the data file level - 1 
	
	//  reads up to level - 1 
	// (because this is the one which has the downstep info for level - 2)
	
	// ignore level - 1
	depth_completed--;
	
	
	if (f_v) {
		cout << "isomorph::read_data_file: after reading file " << fname_data_file << endl;
		cout << "depth_completed = " << depth_completed << ", level = " << level << endl;
		}

	if (f_recompute_schreier) {
		if (f_v) {
			cout << "isomorph::read_data_file recomputing Schreier vectors" << endl;
			}
		gen->recreate_schreier_vectors_up_to_level(
			level - 2, 
			TRUE /* f_compact */, 
			MINIMUM(verbose_level, 1));
		}

	// and now, initialize gen->first_oracle_node_at_level[i + 1]
	// for i = depth_completed + 1, ..., level
	// We find out how many nodes there are from the length of the btree structure

	if (f_v) {
		cout << "isomorph::read_data_file: reading the number of nodes at each level from the level database" << endl;
		}
	for (i = depth_completed + 1; i <= level; i++) {
		database D;
		Vector v;
		INT f, nb_nodes;
		
		if (f_v) {
			cout << "isomorph::read_data_file: level " << i << endl;
			}
		init_DB_level(D, i, verbose_level - 1);
		D.open(verbose_level - 3);
		nb_nodes = D.btree_access_i(0).length(verbose_level - 3);
		if (f_v) {
			cout << "nb_nodes = " << nb_nodes << " (from the btree)" << endl;
			cout << "Now loading object 0" << endl;
			}
		
		D.ith_object(0, 0/* btree_idx*/, v, verbose_level - 1);
		if (f_v) {
			cout << "Object 0 is " << endl;
			cout << v << endl;
			}
		f = v.s_ii(0);
		if (f_v) {
			cout << "f=" << f << " nb_nodes=" << nb_nodes << endl;
			}
		if (f != gen->first_oracle_node_at_level[i]) {
			cout << "f != gen->first_oracle_node_at_level[i]" << endl;
			cout << "f=" << f << endl;
			cout << "gen->first_oracle_node_at_level[i]=" << gen->first_oracle_node_at_level[i] << endl;
			cout << "i=" << i << endl;
			exit(1);
			}

		D.close(verbose_level - 3);

		gen->first_oracle_node_at_level[i + 1] = f + nb_nodes;
		if (f_v) {
			cout << "gen->first_oracle_node_at_level[" << i + 1 << "]=" << gen->first_oracle_node_at_level[i + 1] << endl;
			}
		}
	nb_starter = gen->first_oracle_node_at_level[level + 1] - gen->first_oracle_node_at_level[level];
	if (f_v) {
		cout << "isomorph::read_data_file: nb_starter = " << nb_starter << endl;
		}
#if 0
	if (depth_completed != level) {
		BYTE fname_base[1000];
		
		cout << "warning: depth_completed = " << depth_completed << ", level = " << level << endl;
		//exit(1);
		generator_read_level_file(gen, level, fname_level_file, verbose_level);
		cout << "after generator_read_level_file" << endl;
		sprintf(fname_base, "BLT_%ld", q);
		generator_write_level_file_binary(gen, level, fname_base, verbose_level);
		}
#endif

	if (f_v) {
		cout << "read_data_file done, depth_completed = " << depth_completed << endl;
		}

}
#endif


void isomorph::read_data_files_for_starter(INT level, 
	const BYTE *prefix, INT verbose_level)
// Calls gen->read_level_file_binary for all levels i from 0 to level
// Uses letter a files for i from 0 to level - 1
// and letter b file for i = level.
// If gen->f_starter is TRUE, we start from i = gen->starter_size instead.
// Finally, it computes nb_starter.
{
	INT f_v = (verbose_level >= 1);
	BYTE fname_base_a[1000];
	BYTE fname_base_b[1000];
	INT i, i0;
	
	if (f_v) {
		cout << "isomorph::read_data_files_for_starter" << endl;
		cout << "prefix=" << prefix << endl;
		cout << "level=" << level << endl;
		}
	
	sprintf(fname_base_a, "%sa", prefix);
	sprintf(fname_base_b, "%sb", prefix);
	
	if (gen->f_starter) {
		i0 = gen->starter_size;
		}
	else {
		i0 = 0;
		}
	if (f_v) {
		cout << "isomorph::read_data_files_for_starter i0=" << i0 << endl;
		}
	for (i = i0; i < level; i++) {
		if (f_v) {
			cout << "reading data file for level " << i << " with prefix " << fname_base_b << endl;
			}
		gen->read_level_file_binary(i, fname_base_b, MINIMUM(1, verbose_level - 1));
		}

	if (f_v) {
		cout << "reading data file for level " << level << " with prefix " << fname_base_a << endl;
		}
	gen->read_level_file_binary(level, fname_base_a, MINIMUM(1, verbose_level - 1));

	compute_nb_starter(level, verbose_level);
	//nb_starter = gen->first_oracle_node_at_level[level + 1] - gen->first_oracle_node_at_level[level];

	if (f_v) {
		cout << "isomorph::read_data_files_for_starter finished, number of starters = " << nb_starter << endl;
		}
}

void isomorph::compute_nb_starter(INT level, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	nb_starter = gen->nb_orbits_at_level(level);
		//gen->first_oracle_node_at_level[level + 1] - gen->first_oracle_node_at_level[level];
	if (f_v) {
		cout << "isomorph::compute_nb_starter finished, number of starters = " << nb_starter << endl;
		}

}

void isomorph::print_node_local(INT level, INT node_local)
{
	INT n;

	n = gen->first_oracle_node_at_level[level] + node_local;
	cout << n << "=" << level << "/" << node_local;
}

void isomorph::print_node_global(INT level, INT node_global)
{
	INT node_local;

	node_local = node_global - gen->first_oracle_node_at_level[level];
	cout << node_global << "=" << level << "/" << node_local;
}

void isomorph::test_hash(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT data[1000];
	INT id, case_nb, f, l, i;
	INT *H;


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
		H = NEW_INT(l);
		for (i = 0; i < l; i++) {
			//id = orbit_perm[f + i];
			id = f + i;
			load_solution(id, data);
			INT_vec_heapsort(data, size);
			H[i] = INT_vec_hash(data, size);
			}
		{
		classify C;
		C.init(H, l, TRUE, 0);
		C.print(FALSE /*f_backwards*/);
		}
		FREE_INT(H);
		}

	close_solution_database(verbose_level - 1);	
}


void isomorph::compute_Ago_Ago_induced(longinteger_object *&Ago, longinteger_object *&Ago_induced, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT h, rep, first, /*c,*/ id;
	INT data[1000];
	
	if (f_v) {
		cout << "isomorph::compute_Ago_Ago_induced" << endl;
		}
	Ago = new longinteger_object[Reps->count];
	Ago_induced = new longinteger_object[Reps->count];


	for (h = 0; h < Reps->count; h++) {
		if (f_vv) {
			cout << "isomorph::compute_Ago_Ago_induced orbit " << h << " / " << Reps->count << endl;
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
			cout << "isomorph::compute_Ago_Ago_induced computing induced action on the set (in data)" << endl;
			}
		induced_action_on_set_basic(Stab, data, 0 /*verbose_level*/);
		
			
		AA->group_order(Ago_induced[h]);
		}

	if (f_v) {
		cout << "isomorph::compute_Ago_Ago_induced done" << endl;
		}

}

void isomorph::init_high_level(action *A, generator *gen, 
	INT size, BYTE *prefix_classify, BYTE *prefix, INT level, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph::init_high_level" << endl;
		}

	
	discreta_init();

	INT f_use_database_for_starter = FALSE;
	INT f_implicit_fusion = FALSE;
	
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
		cout << "isomorph::init_high_level before read_data_files_for_starter" << endl;
		}

	read_data_files_for_starter(level, prefix_classify, verbose_level);

	if (f_v) {
		cout << "isomorph::init_high_level before init_solution" << endl;
		}

	init_solution(verbose_level);
	
	read_orbit_data(verbose_level);


	depth_completed = level /*- 2*/;

	if (f_v) {
		cout << "isomorph::init_high_level before iso_test_init" << endl;
		}
	iso_test_init(verbose_level);

	if (f_v) {
		cout << "isomorph::init_high_level before Reps->load" << endl;
		}
	Reps->load(verbose_level);

	setup_and_open_solution_database(verbose_level - 1);
	setup_and_open_level_database(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::init_high_level done" << endl;
		}
}



