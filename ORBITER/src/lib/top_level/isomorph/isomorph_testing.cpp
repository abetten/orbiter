// isomorph_testing.C
// 
// Anton Betten
// Oct 21, 2008
//
// moved here from reader2.C 3/22/09
// renamend isomorph_testing.C from iso.C 7/14/11
//
//

#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace top_level {

void isomorph::iso_test_init(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "isomorph::iso_test_init" << endl;
		}
	
	iso_test_init2(verbose_level);


	Reps = NEW_OBJECT(representatives);

	Reps->init(gen->Poset->A, nb_orbits, prefix, verbose_level);

	if (f_v) {
		cout << "isomorph::iso_test_init done" << endl;
		}
}

void isomorph::iso_test_init2(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i;
	combinatorics_domain Combi;
	
	if (f_v) {
		cout << "isomorph::iso_test_init2" << endl;
		}
	
	subset = NEW_int(level);
	subset_witness = NEW_int(level);
	rearranged_set = NEW_int(size);
	rearranged_set_save = NEW_int(size);
	canonical_set = NEW_int(size);
	tmp_set = NEW_int(size);
	Elt_transporter = NEW_int(A->elt_size_in_int);
	tmp_Elt = NEW_int(A->elt_size_in_int);
	Elt1 = NEW_int(A->elt_size_in_int);
	transporter = NEW_int(A->elt_size_in_int);
	
	if (f_v) {
		cout << "isomorph::iso_test_init2 "
				"before int_n_choose_k" << endl;
		}
	NCK = Combi.int_n_choose_k(size, level);
	if (f_v) {
		cout << "isomorph::iso_test_init2 "
				"after int_n_choose_k" << endl;
		}


	if (f_v) {
		cout << "isomorph::iso_test_init2 done" << endl;
		}
}

void isomorph::probe(int flag_orbit, int subset_rk,
		int f_implicit_fusion, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *Stab;
	longinteger_object go;
	int data[1000];
	int i, id;
	combinatorics_domain Combi;
	sorting Sorting;

	if (f_v) {
		cout << "isomorph::probe for flag orbit " << flag_orbit
				<< " and subset " << subset_rk << endl;
		}
	
	setup_and_open_solution_database(verbose_level - 1);
	setup_and_open_level_database(MINIMUM(1, verbose_level - 1));

	if (f_v) {
		cout << "isomorph::probe for flag orbit " << flag_orbit
				<< " and subset " << subset_rk
				<< " before compute_stabilizer" << endl;
		}
	iso_nodes = 0;
	orbit_no = flag_orbit;
	subset_rank = subset_rk;
	compute_stabilizer(Stab, verbose_level - 1);
		
	Stab->group_order(go);
	
	if (f_v) {
		cout << "isomorph::probe for flag orbit " << flag_orbit
				<< " and subset " << subset_rk
				<< ", known stab order " << go << endl;
		}
	
	
	id = orbit_perm[orbit_fst[flag_orbit]];
	
	load_solution(id, data);
	if (f_v) {
		cout << "isomorph::probe flag orbit " << flag_orbit << " : ";
		int_vec_print(cout, data, size);
		cout << endl;
		}
	
	if (f_v) {
		cout << "isomorph::probe calling "
				"induced_action_on_set" << endl;
		}
	induced_action_on_set(Stab, data, verbose_level - 2);

	if (f_v) {
		cout << "isomorph::probe induced_action_on_set "
				"finished" << endl;
		}
	

	stabilizer_action_init(verbose_level - 1);
	
	Reps->calc_fusion_statistics();

	Combi.unrank_k_subset(subset_rk, subset, size, level);

	if (f_v) {
		cout << "isomorph::probe the subset with rank "
				<< subset_rk  << " is ";
		int_vec_print(cout, subset, level);
		cout << endl;
		cout << "size=" << size << endl;
		cout << "level=" << level << endl;
		}
	Sorting.rearrange_subset(size, level, data,
			subset, rearranged_set, verbose_level - 3);
		// in GALOIS/sorting.C
		
	
	for (i = 0; i < size; i++) {
		rearranged_set_save[i] = rearranged_set[i];
		}

	if (f_v) {
		cout << "The rearranged set is ";
		int_vec_print(cout, rearranged_set, size);
		cout << endl;
		}


	if (f_v) {
		cout << "isomorph::probe before process_rearranged_set" << endl;
		}

	process_rearranged_set(
		Stab, data, 
		f_implicit_fusion, verbose_level - 1);

	if (f_v) {
		cout << "isomorph::probe after process_rearranged_set" << endl;
		}

	close_level_database(verbose_level - 1);
	close_solution_database(verbose_level - 1);
	
	stabilizer_action_exit();
}

void isomorph::isomorph_testing(int t0, 
	int f_play_back, const char *play_back_file_name, 
	int f_implicit_fusion, int print_mod, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_v4 = FALSE;// (verbose_level >= 1);
	sims *Stab;
	longinteger_object go;
	int f_eof;
	
	if (f_v) {
		cout << "isomorph::isomorph_testing" << endl;
		}
	//list_solutions_by_starter();
	
	
	
	//list_solutions_by_orbit();
	
	fp_event_out = new ofstream;

	fp_event_out->open(event_out_fname);
	//ofstream fe("event.txt");
	
	ifstream *play_back_file = NULL;
	
	
	if (f_play_back) {
		play_back_file = new ifstream;
		play_back_file->open(play_back_file_name);

#if 0
		skip_through_event_file(*play_back_file, verbose_level);
		play_back_file->close();
		delete play_back_file;
		f_play_back = FALSE;
#endif
		}


	iso_nodes = 0;
	
	if (f_v) {
		cout << "isomorph::isomorph_testing nb_orbits="
				<< nb_orbits << endl;
		}
	for (orbit_no = 0; orbit_no < nb_orbits; orbit_no++) {
		if (f_v4) {
			cout << "isomorph::isomorph_testing orbit_no=" << orbit_no
					<< " fusion=" << Reps->fusion[orbit_no] << endl;
			}
		if (Reps->fusion[orbit_no] == -2) {
			
			cout << "isomorphism type " << Reps->count 
				<< " is represented by solution orbit " << orbit_no << endl;
			
			Reps->rep[Reps->count] = orbit_no;
			Reps->fusion[orbit_no] = orbit_no;
			
			*fp_event_out << "BEGIN isomorphism type "
					<< Reps->count  << endl;
			*fp_event_out << "O " << orbit_no << endl;
			
			if (f_play_back) {
				f_eof = FALSE;
				}
			
			do_iso_test(t0, Stab, 
				f_play_back, play_back_file, 
				f_eof, print_mod, 
				f_implicit_fusion, verbose_level - 1);
			
			if (f_play_back && f_eof) {
				play_back_file->close();
				f_play_back = FALSE;
				delete play_back_file;
				}


			Reps->stab[Reps->count] = Stab;
			*fp_event_out << "END isomorphism type "
					<< Reps->count << endl;
			Reps->count++;
			}
		//break;
		}
	if (f_v) {
		cout << "isomorph::isomorph_testing done" << endl;
		}

	if (f_play_back) {
		play_back_file->close();
		delete play_back_file;
		}

	*fp_event_out << "-1" << endl;
	delete fp_event_out;
	cout << "Written file " << event_out_fname << " of size "
			<< file_size(event_out_fname) << endl;
	
	cout << "We found " << Reps->count << " isomorphism types" << endl;

	//write_classification_matrix(verbose_level);
	//write_classification_graph(verbose_level);

	if (f_v) {
		cout << "isomorph::isomorph_testing done" << endl;
		}
}

void isomorph::write_classification_matrix(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Mtx;
	int nb_rows, nb_cols;
	int *starter_idx;
	int i, j, f, l, h;
	

	if (f_v) {
		cout << "isomorph::write_classification_matrix" << endl;
		}

	nb_rows = nb_starter;
	nb_cols = Reps->count;

	Mtx = NEW_int(nb_rows * nb_cols);
	int_vec_zero(Mtx, nb_rows * nb_cols);
	starter_idx = NEW_int(nb_orbits);

	for (i = 0; i < nb_starter; i++) {
		f = flag_orbit_fst[i];
		l = flag_orbit_len[i];
		for (j = 0; j < l; j++) {
			starter_idx[f + j] = i;
			}
		}

	int *down_link;

	compute_down_link(down_link, verbose_level);

	if (f_v) {
		cout << "starter_idx=";
		int_vec_print(cout, starter_idx, nb_orbits);
		cout << endl;
		}

	for (h = 0; h < nb_orbits; h++) {
		i = starter_idx[h];
		j = down_link[h];
		Mtx[i * nb_cols + j]++;
		}

	if (f_v) {
		cout << "isomorph::write_classification_matrix" << endl;
		cout << "The classification matrix is:" << endl;
		int_matrix_print(Mtx, nb_rows, nb_cols);
		}

	FREE_int(Mtx);
	FREE_int(starter_idx);
	FREE_int(down_link);
	
	if (f_v) {
		cout << "isomorph::write_classification_matrix done" << endl;
		}
}

void isomorph::write_classification_graph(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_layers;
	int *Nb;
	int *Fst;
	int i, j, f, l, d;

	if (f_v) {
		cout << "isomorph::write_classification_graph" << endl;
		}
	layered_graph *LG;


	nb_layers = 3;
	Nb = NEW_int(nb_layers);
	Fst = NEW_int(nb_layers + 1);

	Fst[0] = 0;
	Nb[0] = nb_starter;
	
	Fst[1] = Fst[0] + Nb[0];
	Nb[1] = nb_orbits;

	Fst[2] = Fst[1] + Nb[1];
	Nb[2] = Reps->count;

	Fst[3] = Fst[2] + Nb[2];



	LG = NEW_OBJECT(layered_graph);
	
	LG->init(nb_layers, Nb, "", verbose_level);
	if (f_vv) {
		cout << "isomorph::write_classification_graph "
				"after LG->init" << endl;
		}
	LG->place(verbose_level);
	if (f_vv) {
		cout << "isomorph::write_classification_graph "
				"after LG->place" << endl;
		}

	// make the first set of edges (upper part)

	if (f_vv) {
		cout << "isomorph::write_classification_graph "
				"making the first set of edges" << endl;
		}

	for (i = 0; i < nb_starter; i++) {
		f = flag_orbit_fst[i];
		l = flag_orbit_len[i];
		if (f_vv) {
			if (l) {
				cout << "starter orbit " << i << " f=" << f
						<< " l=" << l << endl;
				}
			}
		for (j = 0; j < l; j++) {
			LG->add_edge(0, i, 1, f + j, 0 /*verbose_level*/);
			}
		}


	// make the second set of edges (lower part)

	if (f_vv) {
		cout << "isomorph::write_classification_graph "
				"making the second set of edges" << endl;
		}

	int *down_link;

	compute_down_link(down_link, verbose_level);


	for (i = 0; i < nb_orbits; i++) {
		d = down_link[i];
		LG->add_edge(1, i, 2, d, 0 /*verbose_level*/);
		}


	char fname_base1[1000];
	char fname[1000];
	sprintf(fname_base1, "%sclassification_graph", prefix);
	sprintf(fname, "%s.layered_graph", fname_base1);
	LG->write_file(fname, 0 /*verbose_level*/);
	if (f_v) {
		cout << "isomorph::write_classification_graph "
				"Written file " << fname << " of size "
				<< file_size(fname) << endl;
		}


	FREE_int(down_link);
	FREE_OBJECT(LG);
	if (f_v) {
		cout << "isomorph::write_classification_graph done" << endl;
		}
}

void isomorph::decomposition_matrix(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int m, n, i, j, a, b, f, l;
	int *M;

	if (f_v) {
		cout << "isomorph::decomposition_matrix" << endl;
		}
	m = nb_starter;
	n = Reps->count;
	M = NEW_int(m * n);
	for (i = 0; i < m * n; i++) {
		M[i] = 0;
		}

	int *down_link;

	compute_down_link(down_link, verbose_level);

	for (i = 0; i < nb_starter; i++) {
		f = flag_orbit_fst[i];
		l = flag_orbit_len[i];
		for (j = 0; j < l; j++) {
			a = f + j;
			b = down_link[a];
			M[i * n + b]++;
			}
		}

	char fname_base1[1000];
	char fname[1000];
	
	sprintf(fname_base1, "%sdecomposition_matrix", prefix);
	sprintf(fname, "%s.csv", fname_base1);
	int_matrix_write_csv(fname, M, m, n);

	FREE_int(down_link);
	FREE_int(M);
	if (f_v) {
		cout << "isomorph::decomposition_matrix done" << endl;
		}
}


void isomorph::compute_down_link(int *&down_link,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, f, flag_orbit;
	
	if (f_v) {
		cout << "isomorph::compute_down_link" << endl;
		}
	down_link = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		down_link[i] = -1;
		}
	
	for (i = 0; i < Reps->count; i++) {
		flag_orbit = Reps->rep[i];
		down_link[flag_orbit] = i;
		}
	for (i = 0; i < nb_orbits; i++) {
		f = Reps->fusion[i];
		if (f == i) {
			if (down_link[i] == -1) {
				cout << "data structure is inconsistent" << endl;
				exit(1);
				}
			}
		else {
			if (down_link[f] == -1) {
				cout << "data structure is inconsistent" << endl;
				exit(1);
				}
			down_link[i] = down_link[f];
			}
		}

	if (f_vv) {
		cout << "down_link: ";
		int_vec_print(cout, down_link, nb_orbits);
		cout << endl;
		}
	if (f_v) {
		cout << "isomorph::compute_down_link done" << endl;
		}
}

void isomorph::do_iso_test(int t0, sims *&Stab, 
	int f_play_back, ifstream *play_back_file, 
	int &f_eof, int print_mod, 
	int f_implicit_fusion, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	longinteger_object go;
	int id;
	int data[1000];
	int f_continue;
	combinatorics_domain Combi;
	
	
	if (f_v) {
		cout << "###############################################################" << endl;
		cout << "isomorph::do_iso_test orbit_no=" << orbit_no << endl;
		}

	setup_and_open_solution_database(verbose_level - 1);
	setup_and_open_level_database(MINIMUM(1, verbose_level - 1));

	compute_stabilizer(Stab, verbose_level - 1);
		
	Stab->group_order(go);
	
	if (f_v) {
		cout << "isomorph::do_iso_test for isomorphism type "
				<< Reps->count
			<< " which is represented by flag orbit " << orbit_no
			<< ", known stab order " << go
			<< " orbit_fst=" << orbit_fst[orbit_no] << " orbit_len="
			<< orbit_len[orbit_no] << " " << endl;
		}
	
	id = orbit_perm[orbit_fst[orbit_no]];
	
	load_solution(id, data);
	if (f_vv) {
		cout << "isomorph::do_iso_test orbit_no = " << orbit_no << " : ";
		int_vec_print(cout, data, size);
		cout << endl;
		}
	
	if (f_vv) {
		cout << "isomorph::do_iso_test "
				"calling induced_action_on_set" << endl;
		}
	induced_action_on_set(Stab, data, verbose_level - 2);

	if (f_vv) {
		cout << "isomorph::do_iso_test "
				"induced_action_on_set finished" << endl;
		}
	

	stabilizer_action_init(verbose_level - 2);
	
	if (f_v3) {
		cout << "base for AA: ";
		AA->print_base();
		cout << "base for A:" << endl;
		A->print_base();
		}
	
	cnt_minimal = 0;
	Reps->calc_fusion_statistics();

	Combi.first_k_subset(subset, size, level);
	subset_rank = Combi.rank_k_subset(subset, size, level);

	f_continue = FALSE;

	while (TRUE) {

#if 0
		if ((iso_nodes % 500000) == 0) {
			registry_dump_sorted();
			}
#endif

		if ((iso_nodes % print_mod) == 0 && !f_continue) {
			print_statistics_iso_test(t0, Stab);
			}
		
		if (!next_subset(t0, 
			f_continue, Stab, data, 
			f_play_back, play_back_file, f_eof, 
			verbose_level - 2)) {
			break;
			}

		
		if (f_continue) {
			continue;
			}
		
		
		iso_nodes++;

		if (f_v3) {
			cout << "isomorph::do_iso_test "
					"before process_rearranged_set" << endl;
			}

#if 0
		if (iso_nodes ==43041) {
			process_rearranged_set(
				Stab, data, 
				f_implicit_fusion, verbose_level - 2 + 12);
			}
		else {
#endif

			process_rearranged_set(
				Stab, data, 
				f_implicit_fusion, verbose_level - 2);

#if 0
			}
#endif
		
		Reps->calc_fusion_statistics();

		if (!Combi.next_k_subset(subset, size, level)) {
			break;
			}
		}
	if (f_v) {
		cout << "isomorph::do_iso_test "
				"cnt_minimal=" << cnt_minimal << endl;
		}

	print_statistics_iso_test(t0, Stab);

	Stab->group_order(go);
	if (f_v) {
		cout << "isomorph::do_iso_test "
				"the full stabilizer has order " << go << endl;
		}	

	close_level_database(verbose_level - 1);
	close_solution_database(verbose_level - 1);
	
	stabilizer_action_exit();
	if (f_v) {
		cout << "isomorph::do_iso_test done" << endl;
		}
}


int isomorph::next_subset(int t0, 
	int &f_continue, sims *Stab, int *data, 
	int f_play_back, ifstream *play_back_file, int &f_eof, 
	int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v6 = (verbose_level >= 6);
	int f_is_minimal;
	int i;
	combinatorics_domain Combi;
	sorting Sorting;

	f_continue = FALSE;
	
	if (f_play_back) {
		if (!next_subset_play_back(subset_rank, play_back_file, 
			f_eof, verbose_level)) {
			return FALSE;
			}
		}
	subset_rank = Combi.rank_k_subset(subset, size, level);

		
		
	
	if (f_play_back) {
		f_is_minimal = TRUE;
		}
	else {
		f_is_minimal = is_minimal(verbose_level);
		}
	
		
		
	if (!f_is_minimal) {
	
		//cout << "next subset at backtrack_level="
		//<< backtrack_level << endl;
		if (!Combi.next_k_subset(subset, size, level)) {
			// in GALOIS/combinatorics.C

			return FALSE;
			}
		f_continue = TRUE;
		return TRUE;
		}
		
	if (f_vvv) {
		cout << "iso_node " << iso_nodes << " found minimal subset no " 
			<< cnt_minimal << ", rank = " << subset_rank << " : ";
		int_set_print(cout, subset, level);
		cout << endl;
		}
	cnt_minimal++;
		
	if (f_v6) {
		cout << "after is_minimal: A: ";
		A->print_base();
		cout << "after is_minimal: AA: ";
		AA->print_base();
		}



	if (FALSE) {
		print_statistics_iso_test(t0, Stab);
		}
	if (f_v6) {
		cout << "current stabilizer:" << endl;
		AA->print_vector(Stab->gens);
		//AA->print_vector_as_permutation(Stab->gens);
		}
	
	Sorting.rearrange_subset(size, level, data,
		subset, rearranged_set, verbose_level - 3);
		// in GALOIS/sorting.C
		
	
	for (i = 0; i < size; i++) {
		rearranged_set_save[i] = rearranged_set[i];
		}



	return TRUE;
}

void isomorph::process_rearranged_set(
	sims *Stab, int *data,
	int f_implicit_fusion, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int f_v6 = (verbose_level >= 6);
	int orbit_no0, id0, hdl, i, j;
	int data0[1000];
	longinteger_object new_go;
	int f_found;
	
	if (f_v) {
		cout << "isomorph::process_rearranged_set "
				"flag orbit " << orbit_no << " subset "
				<< subset_rank << endl;
		//cout << "verbose_level=" << verbose_level << endl;
		//cout << "before identify_solution" << endl;
		}
	
	int f_failure_to_find_point;

	f_found = identify_solution_relaxed(
		rearranged_set, transporter,
		f_implicit_fusion, orbit_no0, f_failure_to_find_point,
		verbose_level - 2);

	if (f_failure_to_find_point) {
		if (f_vv) {
			cout << "isomorph::process_rearranged_set flag orbit "
					<< orbit_no << " subset " << subset_rank
					<< " : f_failure_to_find_point" << endl;
			}
		return;
		}	
	if (!f_found) {
#if 0
		if (TRUE /*f_vv*/) {
			cout << "isomorph::process_rearranged_set flag orbit "
					<< orbit_no << " subset " << subset_rank
					<< " : not found" << endl;
			cout << "Original set: ";
			int_vec_print(cout, data, size);
			cout << endl;
			cout << "subset: ";
			int_vec_print(cout, subset, level);
			cout << endl;
			cout << "Rearranged set: ";
			int_vec_print(cout, rearranged_set_save, size);
			cout << endl;
			cout << "After trace: ";
			int_vec_print(cout, rearranged_set, size);
			cout << endl;
			int_vec_copy(rearranged_set_save, rearranged_set, size);
			f_found = identify_solution_relaxed(
				rearranged_set, transporter,
				f_implicit_fusion, orbit_no0, f_failure_to_find_point,
				verbose_level + 10);
			cout << "f_found=" << f_found << endl;
			}
		exit(1);
#endif
		if (FALSE) {
			cout << "isomorph::process_rearranged_set flag orbit "
					<< orbit_no << " subset " << subset_rank
					<< " : not found" << endl;
		}
		return;
		}
	if (f_v) {
		cout << "isomorph::process_rearranged_set flag orbit " << orbit_no
				<< " subset " << subset_rank << endl;
		cout << "after identify_solution, needs to be joined with "
				"flag orbit = " << orbit_no0 << endl;
		}

	id0 = orbit_perm[orbit_fst[orbit_no0]];
	
	load_solution(id0, data0);

	if (!A->check_if_transporter_for_set(transporter, size, 
		data, data0, verbose_level)) {
		cout << "the element does not map set1 to set2" << endl;
		exit(1);
		}
	else {
		//cout << "the element does map set1 to set2" << endl;
		}

		

	if (f_vv) {
		cout << "fusion[orbit_no0] = " << Reps->fusion[orbit_no0] << endl;
		}

	if (orbit_no0 == orbit_no) {
		if (f_v) {
			cout << "isomorph::process_rearranged_set flag orbit "
					<< orbit_no << " subset " << subset_rank
					<<  " automorphism" << endl;
			//A->element_print(transporter, cout);
			}
			
		if (handle_automorphism(data, Stab, transporter, 
			verbose_level - 2)) {
			Stab->group_order(new_go);
			*fp_event_out << "A " << orbit_no << " " << subset_rank
					<< " " << new_go << endl;
			cout << "event: A " << orbit_no << " " << subset_rank
					<< " " << new_go << endl;
			}
			

		}
	else if (Reps->fusion[orbit_no0] == -2) {
		Reps->fusion[orbit_no0] = orbit_no;
		if (f_v) {
			cout << "isomorph::process_rearranged_set flag orbit "
					<< orbit_no << " subset " << subset_rank
					<<  " fusion" << endl;
			}
		A->element_invert(transporter, tmp_Elt, FALSE);
		if (FALSE && f_v6) {
			cout << "fusion element:" << endl;
			A->element_print(tmp_Elt, cout);
			}
		hdl = A->element_store(tmp_Elt, FALSE);
		if (f_v6) {
			//cout << "hdl=" << hdl << endl;
			}
		Reps->handle[orbit_no0] = hdl;
		*fp_event_out << "F " << orbit_no << " " << subset_rank
				<< " " << orbit_no0 << endl;
		if (f_v) {
			cout << "event: F " << orbit_no << " " << subset_rank
					<< " " << orbit_no0 << endl;
			}
		}
	else {
		if (f_v) {
			cout << "isomorph::process_rearranged_set flag orbit "
					<< orbit_no << " subset " << subset_rank
					<<  " automorphism due to repeated fusion" << endl;
			}
		if (Reps->fusion[orbit_no0] != orbit_no) {
			cout << "COLLISION-ERROR!!!" << endl;
			cout << "automorphism due to repeated fusion" << endl;
			cout << "fusion[orbit_no0] != orbit_no" << endl;
			cout << "orbit_no = " << orbit_no << endl;
			cout << "orbit_no0 = " << orbit_no0 << endl;
			cout << "fusion[orbit_no0] = " << Reps->fusion[orbit_no0] << endl;
			cout << "handle[orbit_no0] = " << Reps->handle[orbit_no0] << endl;
			A->element_retrieve(Reps->handle[orbit_no0], Elt1, FALSE);
			cout << "old transporter inverse:" << endl;
			A->element_print(Elt1, cout);
			cout << "n e w transporter:" << endl;
			A->element_print(transporter, cout);
			A->element_mult(transporter, Elt1, tmp_Elt, FALSE);
			cout << "n e w transporter times old transporter inverse:" << endl;
			A->element_print(tmp_Elt, cout);
			cout << "subset: ";
			int_vec_print(cout, subset, level);
			cout << endl;

			int my_data[1000];
			int my_data0[1000];
			int original_orbit;
			
			original_orbit = Reps->fusion[orbit_no0];
			load_solution(orbit_perm[orbit_fst[orbit_no]], my_data);
			load_solution(orbit_perm[orbit_fst[original_orbit]], my_data0);
		

			cout << "i : data[i] : rearranged_set_save[i] : image under "
					"group element : data0[i]" << endl;
			for (i = 0; i < size; i++) {
				j = rearranged_set_save[i];
				cout << setw(3) << i << " : " 
					<< setw(6) << data[i] << " : " 
					<< setw(3) << j << " : " 
					<< setw(6) << A->image_of(tmp_Elt, j) << " : " 
					<< setw(6) << data0[i] 
					<< endl;
				}
			cout << "COLLISION-ERROR!!! exit" << endl;
			exit(1);
			}
		hdl = Reps->handle[orbit_no0];
		//cout << "hdl=" << hdl << endl;
		A->element_retrieve(hdl, Elt1, FALSE);
		//A->element_print(Elt1, cout);
		A->element_mult(transporter, Elt1, tmp_Elt, FALSE);
		
		if (handle_automorphism(data, Stab, tmp_Elt, verbose_level)) {
			Stab->group_order(new_go);
			*fp_event_out << "AF " << orbit_no << " " << subset_rank
					<< " " << orbit_no0 << " " << new_go << endl;
			if (f_v) {
				cout << "event: AF " << orbit_no << " " << subset_rank
						<< " " << orbit_no0 << " " << new_go << endl;
				}
			}
		}
}

int isomorph::is_minimal(int verbose_level)
{
	int rk, rk0;
	combinatorics_domain Combi;
	
	rk = Combi.rank_k_subset(subset, size, level);
	rk0 = UF->ancestor(rk);
	if (rk0 == rk) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}


void isomorph::stabilizer_action_exit()
{
	int h;
	
	for (h = 0; h < stabilizer_nb_generators; h++) {
		FREE_int(stabilizer_generators[h]);
		}
	FREE_pint(stabilizer_generators);
	FREE_int(stabilizer_orbit);
	stabilizer_generators = NULL;
	stabilizer_orbit = NULL;
}



void isomorph::stabilizer_action_init(int verbose_level)
// Computes the permutations of the set that are induced by the 
// generators for the stabilizer in AA
{
	int f_v = (verbose_level >= 1);
	int h, i, j;
	int *Elt;
	combinatorics_domain Combi;
	
	nb_sets_reached = 0;
	nb_is_minimal_called = 0;
	nb_is_minimal = 0;
	
	AA->group_order(stabilizer_group_order);
	if (f_v) {
		cout << "stabilizer of order " << stabilizer_group_order << endl;
		}

	stabilizer_nb_generators = AA->Strong_gens->gens->len;
	//stabilizer_nb_generators = AA->strong_generators->len;
	stabilizer_generators = NEW_pint(stabilizer_nb_generators);
	
	stabilizer_orbit = NEW_int(NCK);
	for (i = 0; i < NCK; i++) {
		stabilizer_orbit[i] = -2;
		}
	
	for (h = 0; h < stabilizer_nb_generators; h++) {
		stabilizer_generators[h] = NEW_int(size);
		Elt = AA->Strong_gens->gens->ith(h);
		//Elt = AA->strong_generators->ith(h);
		for (i = 0; i < size; i++) {
			j = AA->image_of(Elt, i);
			stabilizer_generators[h][i] = j;
			}
		if (f_v) {
			cout << "generator " << h << ":" << endl;
			A->element_print_quick(Elt, cout);
			Combi.perm_print(cout, stabilizer_generators[h], size);
			cout << endl;
			}
		}
}

void isomorph::stabilizer_action_add_generator(int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int **new_gens;
	int h, i, j;
	combinatorics_domain Combi;
	
	AA->group_order(stabilizer_group_order);
	if (f_v) {
		cout << "stabilizer_action_add_generator, group of order "
				<< stabilizer_group_order << endl;
		}

	new_gens = NEW_pint(stabilizer_nb_generators + 1);
	for (h = 0; h < stabilizer_nb_generators; h++) {
		new_gens[h] = stabilizer_generators[h];
		}
	h = stabilizer_nb_generators;
	new_gens[h] = NEW_int(size);
	for (i = 0; i < size; i++) {
		j = AA->image_of(Elt, i);
		new_gens[h][i] = j;
		}
	if (f_vv) {
		cout << "generator " << h << ":" << endl;
		A->element_print_quick(Elt, cout);
		Combi.perm_print(cout, new_gens[h], size);
		cout << endl;
		}
	stabilizer_nb_generators++;
	FREE_pint(stabilizer_generators);
	stabilizer_generators = new_gens;

	int *Elt1;
	int len, nb, N;
	double f;

	len = gens_perm->len;

	gens_perm->reallocate(len + 1);

	Elt1 = NEW_int(AA_perm->elt_size_in_int);

	AA_perm->make_element(Elt1, stabilizer_generators[h],
			0 /* verbose_level */);
	AA_perm->element_move(Elt1, gens_perm->ith(len),
			0 /* verbose_level */);
	UF->add_generator(Elt1, 0 /* verbose_level */);
	
	nb = UF->count_ancestors_above(subset_rank);
	N = AA_on_k_subsets->degree - subset_rank;
	f = ((double)nb / (double)N) * 100;
	if (f_v) {
		cout << "stabilizer_action_add_generator: number of ancestors = "
				<< nb << " / " << N << " (" << f << "%)" << endl;
		}
	if (f_v) {
		cout << "isomorph::stabilizer_action_add_generator finished" << endl;
		}



	FREE_int(Elt1);
}

void isomorph::print_statistics_iso_test(int t0, sims *Stab)
{
	//double progress;
	longinteger_object go; 
	longinteger_object AA_go;
	int subset_rank;
	int t1, dt;
	int nb, N;
	double f1; //, f2;
	combinatorics_domain Combi;
	
	t1 = os_ticks();
	dt = t1 - t0;
	//cout << "time_check t0=" << t0 << endl;
	//cout << "time_check t1=" << t1 << endl;
	//cout << "time_check dt=" << dt << endl;
	time_check_delta(cout, dt);
	subset_rank = Combi.rank_k_subset(subset, size, level);
	Stab->group_order(go);
	AA->group_order(AA_go);
	//progress = (double)nb_sets_reached / (double)NCK;
	cout 
		<< " iso_node " << iso_nodes 
		<< " iso-type " << Reps->count /*isomorph_cnt*/ 
		<< " cnt_minimal=" << cnt_minimal 
		<< " subset " << subset_rank 
		<< " / " << NCK << " : ";

	nb = UF->count_ancestors_above(subset_rank);
	N = AA_on_k_subsets->degree; // - subset_rank;
	f1 = ((double)nb / (double)N) * 100;
#if 0
	N = AA_on_k_subsets->degree - subset_rank;
	f2 = ((double)nb / (double)N) * 100;
#endif
	cout << "ancestors left = " << nb << " / " << N
			<< " (" << f1 << "%): ";
	int_set_print(cout, subset, level);
	cout << " current stabilizer order " << go 
		<< " induced action order " << AA_go 
		<< " nb_reps=" << Reps->nb_reps 
		<< " nb_fused=" << Reps->nb_fused 
		<< " nb_open=" << Reps->nb_open;

	if (nb_times_make_set_smaller_called) {
		cout << " nb_times_make_set_smaller_called="
				<< nb_times_make_set_smaller_called;
		}
		//<< " nb_is_minimal_called=" << nb_is_minimal_called 
		//<< " nb_is_minimal=" << nb_is_minimal 
		//<< " nb_sets_reached=" << nb_sets_reached 
		//<< " progress = " << progress
	cout << endl;
}


int isomorph::identify(int *set, int f_implicit_fusion,
		int verbose_level)
// opens and closes the solution database and the level database.
// Hence this function is slow.
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int idx;
	
	if (f_v) {
		cout << "isomorph::identify" << endl;
		}
	setup_and_open_solution_database(verbose_level - 1);
	setup_and_open_level_database(verbose_level - 2);

	idx = identify_database_is_open(set, f_implicit_fusion, verbose_level);

	close_level_database(verbose_level - 2);
	close_solution_database(verbose_level - 2);
	
	if (f_v) {
		cout << "isomorph::identify done" << endl;
		}
	return idx;
}

int isomorph::identify_database_is_open(int *set,
		int f_implicit_fusion, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//database DD1, DD2;
	int data0[1000];
	int orbit_no0, id0, f;
	sorting Sorting;

	if (f_v) {
		cout << "isomorph::identify_database_is_open" << endl;
		}
	//setup_and_open_solution_database(verbose_level - 1);
	//setup_and_open_level_database(verbose_level - 2);

	int f_failure_to_find_point;
	
	orbit_no0 = identify_solution(set, transporter,
			f_implicit_fusion, f_failure_to_find_point,
			verbose_level - 3);

	if (f_vv) {
		cout << "identify_solution returns orbit_no0 = " << orbit_no0 << endl;
		}

	if (f_failure_to_find_point) {
		cout << "isomorph::identify_database_is_open: "
				"f_failure_to_find_point" << endl;
		exit(1);
		}
	id0 = orbit_perm[orbit_fst[orbit_no0]];
	
	load_solution(id0, data0);
		
	if (!A->check_if_transporter_for_set(transporter, size, 
		set, data0, verbose_level)) {
		cout << "the element does not map set to canonical set (1)" << endl;
		exit(1);
		}
	else {
		if (f_v) {
			cout << "the element does map set1 to set2" << endl;
			}
		}

	f = Reps->fusion[orbit_no0];
	if (f_vv) {
		cout << "identify_solution f = fusion[orbit_no0] = " << f << endl;
		}
	if (f != orbit_no0) {

		// A Betten 10/25/2014
		// why do we load the fusion element from file?
		// this seems to slow down the process.

		FILE *f2;
		int *Elt1, *Elt2;
		
		Elt1 = NEW_int(gen->Poset->A->elt_size_in_int);
		Elt2 = NEW_int(gen->Poset->A->elt_size_in_int);
		
		f2 = fopen(Reps->fname_fusion_ge, "rb");
		fseek(f2, orbit_no0 * gen->Poset->A->coded_elt_size_in_char, SEEK_SET);
		gen->Poset->A->element_read_file_fp(Elt1, f2, 0/* verbose_level*/);
		
		fclose(f2);
		gen->Poset->A->mult(transporter, Elt1, Elt2);
		gen->Poset->A->move(Elt2, transporter);
		FREE_int(Elt1);
		FREE_int(Elt2);
		}

	id0 = orbit_perm[orbit_fst[f]];
	
	load_solution(id0, data0);
		
	if (!A->check_if_transporter_for_set(transporter, size, 
		set, data0, verbose_level)) {
		cout << "the element does not map set to canonical set (2)" << endl;
		exit(1);
		}
	else {
		//cout << "the element does map set1 to set2" << endl;
		}
	
	if (f_vv) {
		cout << "canonical set is " << f << endl;
		cout << "transporter:" << endl;
		A->print(cout, transporter);
		}

	int idx;

	if (!Sorting.int_vec_search(Reps->rep, Reps->count, f, idx)) {
		cout << "representative not found f=" << f << endl;
		exit(1);
		}
	

	//close_level_database(verbose_level - 2);
	//close_solution_database(verbose_level - 2);
	
	if (f_v) {
		cout << "isomorph::identify_database_is_open done" << endl;
		}
	return idx;
}


void isomorph::induced_action_on_set_basic(sims *S,
		int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object go, K_go;
	
	if (f_v) {
		cout << "isomorph::induced_action_on_set_basic" << endl;
		}
	if (AA) {
		FREE_OBJECT(AA);
		AA = NULL;
		}

	//AA = NEW_OBJECT(action);

	if (f_vv) {
		cout << "isomorph::induced_action_on_set_basic "
				"before induced_action_by_restriction" << endl;
		}
	AA = gen->Poset->A2->create_induced_action_by_restriction(
		S,
		size, set,
		TRUE,
		0/*verbose_level*/);
	if (f_vv) {
		cout << "isomorph::induced_action_on_set_basic "
				"after induced_action_by_restriction" << endl;
		}
	AA->group_order(go);
	AA->Kernel->group_order(K_go);
	if (f_vv) {
		cout << "isomorph::induced_action_on_set_basic "
				"induced action by restriction: group order = "
				<< go << endl;
		cout << "isomorph::induced_action_on_set_basic "
				"kernel group order = " << K_go << endl;
		}
	if (f_v) {
		cout << "isomorph::induced_action_on_set_basic done" << endl;
		}
}

void isomorph::induced_action_on_set(
		sims *S, int *set, int verbose_level)
// Called by do_iso_test and print_isomorphism_types
// Creates the induced action on the set from the given action.
// The given action is gen->A2
// The induced action is computed to AA
// The set is in set[].
// Allocates a n e w union_find data structure and initializes it
// using the generators in S.
// Calls action::induced_action_by_restriction()
{
	longinteger_object go, K_go;
	//sims *K;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "isomorph::induced_action_on_set" << endl;
		}
	if (gens_perm) {
		FREE_OBJECT(gens_perm);
		gens_perm = NULL;
		}
	if (AA) {
		FREE_OBJECT(AA);
		AA = NULL;
		}
	if (AA_perm) {
		FREE_OBJECT(AA_perm);
		AA_perm = NULL;
		}
	if (AA_on_k_subsets) {
		FREE_OBJECT(AA_on_k_subsets);
		AA_on_k_subsets = NULL;
		}
	if (UF) {
		FREE_OBJECT(UF);
		UF = NULL;
		}
	//AA = NEW_OBJECT(action);
	AA_perm = NEW_OBJECT(action);
	AA_on_k_subsets = NEW_OBJECT(action);
	
	
	if (f_v) {
		cout << "isomorph::induced_action_on_set "
				"before induced_action_by_restriction" << endl;
		}
	AA = gen->Poset->A2->create_induced_action_by_restriction(
			S,
			size, set,
			TRUE,
			0/*verbose_level*/);
	if (f_v) {
		cout << "isomorph::induced_action_on_set "
				"after induced_action_by_restriction" << endl;
		}
	AA->group_order(go);
	AA->Kernel->group_order(K_go);
	if (f_v) {
		cout << "isomorph::induced_action_on_set "
				"induced action by restriction: group order = "
				<< go << endl;
		cout << "isomorph::induced_action_on_set "
				"kernel group order = " << K_go << endl;
		}
	
	if (f_vv) {
		cout << "isomorph::induced_action_on_set "
				"induced action:" << endl;
		//AA->Sims->print_generators();
		//AA->Sims->print_generators_as_permutations();
		//AA->Sims->print_basic_orbits();
	
		longinteger_object go;
		AA->Sims->group_order(go);
		cout << "isomorph::induced_action_on_set "
				"AA->Sims go=" << go << endl;

		//cout << "induced action, in the original action:" << endl;
		//AA->Sims->print_generators_as_permutations_override_action(A);
		}	
	
	//cout << "kernel:" << endl;
	//K->print_generators();
	//K->print_generators_as_permutations();

	if (f_v) {
		cout << "isomorph::induced_action_on_set "
				"before init_permutation_group" << endl;
		}

	AA_perm->init_permutation_group(size, 0/*verbose_level*/);
	if (f_v) {
		cout << "AA_perm:" << endl;
		AA_perm->print_info();
		}
	
	if (f_v) {
		cout << "isomorph::induced_action_on_set "
				"before induced_action_on_k_subsets" << endl;
		}
	AA_on_k_subsets->induced_action_on_k_subsets(
		*AA_perm, level /* k */,
		0/*verbose_level*/);
	if (f_v) {
		cout << "isomorph::induced_action_on_set "
				"AA_on_k_subsets:" << endl;
		AA_on_k_subsets->print_info();
		}

	if (f_v) {
		cout << "isomorph::induced_action_on_set "
				"creating gens_perm" << endl;
		}

	if (AA->Strong_gens == NULL) {
		cout << "AA->Strong_gens == NULL" << endl;
		exit(1);
		}

	vector_ge *gens = AA->Strong_gens->gens;
	//vector_ge *gens = AA->strong_generators;
	int len, h, i, j;
	int *data1;
	int *data2;
	int *Elt1;

	len = gens->len;
	gens_perm = NEW_OBJECT(vector_ge);

	gens_perm->init(AA_perm);
	gens_perm->allocate(len);

	data1 = NEW_int(size);
	data2 = NEW_int(size);
	Elt1 = NEW_int(AA_perm->elt_size_in_int);

	for (h = 0; h < len; h++) {
		if (FALSE /*f_v*/) {
			cout << "isomorph::induced_action_on_set "
					"generator " << h << " / " << len << ":" << endl;
			}
		for (i = 0; i < size; i++) {
			j = AA->image_of(gens->ith(h), i);
			data1[i] = j;
			}
		if (FALSE /*f_v*/) {
			cout << "isomorph::induced_action_on_set permutation: ";
			int_vec_print(cout, data1, size);
			cout << endl;
			}
		AA_perm->make_element(Elt1, data1, 0 /* verbose_level */);
		AA_perm->element_move(Elt1, gens_perm->ith(h),
				0 /* verbose_level */);
		}
	if (f_v) {
		cout << "isomorph::induced_action_on_set "
				"created gens_perm" << endl;
		}

	UF = NEW_OBJECT(union_find);
	UF->init(AA_on_k_subsets, verbose_level);
	if (f_v) {
		cout << "isomorph::induced_action_on_set "
				"after UF->init" << endl;
		}
	UF->add_generators(gens_perm, 0 /* verbose_level */);
	if (f_v) {
		cout << "isomorph::induced_action_on_set "
				"after UF->add_generators" << endl;
		}
	if (f_v) {
		int nb, N;
		double f;
		nb = UF->count_ancestors();
		N = AA_on_k_subsets->degree;
		f = ((double)nb / (double)N) * 100;
		cout << "isomorph::induced_action_on_set number of ancestors = "
				<< nb << " / " << N << " (" << f << "%)" << endl;
		}
	if (f_v) {
		cout << "isomorph::induced_action_on_set finished" << endl;
		}

	FREE_int(data1);
	FREE_int(data2);
	FREE_int(Elt1);
}

int isomorph::handle_automorphism(int *set, sims *Stab,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v6 = (verbose_level >= 6);
	int *Elt1;
	longinteger_object go, go1;
	int ret;
	
	if (f_v) {
		cout << "isomorph::handle_automorphism orbit " << orbit_no
				<< " subset " << subset_rank <<  endl;
		}
	Elt1 = handle_automorphism_Elt1;
#if 0
	if (f_vvv) {
		A->element_print(Elt, cout);
		}
#endif
			
	Stab->group_order(go);
	if (Stab->strip_and_add(Elt, Elt1 /* residue */,
			0/*verbose_level +4*//*- 2*/)) {
		Stab->closure_group(2000 /* nb_times */,
				0/*verbose_level*/);
		Stab->group_order(go1);
		if (f_v) {
			cout << "isomorph::handle_automorphism orbit " << orbit_no
					<< " subset " << subset_rank <<  " : ";
			cout << "the stabilizer has been extended, old order " 
				<< go << " n e w group order " << go1 << endl;
			}
		if (f_v) {
			cout << "isomorph::handle_automorphism orbit " << orbit_no
					<< " subset " << subset_rank
					<< "n e w automorphism:" << endl;
			A->element_print(Elt, cout);
			}
		//induced_action_on_set(Stab, set, verbose_level - 2);
		
		stabilizer_action_add_generator(Elt, verbose_level);
		
		if (f_v6) {
			//AA->element_print_as_permutation(Elt, cout);
			//cout << endl;
			//A->element_print_as_permutation(Elt, cout);
			//cout << endl;
			}
		if (f_v6) {
			cout << "isomorph::handle_automorphism orbit " << orbit_no
					<< " subset " << subset_rank <<  " : ";
			cout << "current stabilizer:" << endl;
			AA->print_vector(Stab->gens);
			//AA->print_vector_as_permutation(Stab->gens);
			Stab->print_transversals();
			Stab->print_transversal_lengths();
			}
		Stab->group_order(go);
		if (Stab->closure_group(200 /* nb_times */, 0/*verbose_level*/)) {
			Stab->group_order(go1);
			if (f_v) {
				cout << "isomorph::handle_automorphism orbit " << orbit_no
						<< " subset " << subset_rank <<  " : ";
				cout << "the stabilizer has been extended during "
						"closure_group, old order "
					<< go << " n e w group order " << go1 << endl;
				}
			induced_action_on_set(Stab, set, 0/*verbose_level - 1*/);
			}
		ret = TRUE;
		}
	else {
		if (f_vvv) {	
			cout << "isomorph::handle_automorphism orbit " << orbit_no
					<< " subset " << subset_rank <<  " : ";
			cout << "already known" << endl;
			}
		ret = FALSE;
		}
	return ret;
}


}}


