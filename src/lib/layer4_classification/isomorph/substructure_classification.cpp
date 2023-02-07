/*
 * substructure_classification.cpp
 *
 *  Created on: Sep 6, 2022
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


substructure_classification::substructure_classification()
{
	Iso = NULL;

	f_use_database_for_starter = FALSE;
	depth_completed = 0;
	f_use_implicit_fusion = FALSE;



	//std::string fname_db_level_ge;

	//std::string fname_db_level;
	//std::string fname_db_level_idx1;
	//std::string fname_db_level_idx2;


	nb_starter = 0;

	gen = NULL;

	D1 = NULL;
	D2 = NULL;
	fp_ge1 = NULL;
	fp_ge2 = NULL;
	fp_ge = NULL;

	DB_level = NULL;


}

substructure_classification::~substructure_classification()
{
	int f_v = FALSE;

	if (f_v) {
		cout << "substructure_classification::~substructure_classification before deleting D1" << endl;
		}
	if (D1) {
		freeobject(D1);
		D1 = NULL;
		}
	if (f_v) {
		cout << "substructure_classification::~substructure_classification before deleting D2" << endl;
		}
	if (D2) {
		freeobject(D2);
		D2 = NULL;
		}

}

void substructure_classification::init(isomorph *Iso,
		poset_classification::poset_classification *gen,
		int f_use_database_for_starter,
		int f_implicit_fusion,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "substructure_classification::init" << endl;
		cout << "f_use_database_for_starter="
				<< f_use_database_for_starter << endl;
		cout << "f_implicit_fusion=" << f_implicit_fusion << endl;
	}

	substructure_classification::Iso = Iso;
	substructure_classification::f_use_database_for_starter = f_use_database_for_starter;
	substructure_classification::gen = gen;

	f_use_implicit_fusion = f_implicit_fusion;

	nb_starter = 0;



	if (f_v) {
		cout << "substructure_classification::init done" << endl;
	}
}

void substructure_classification::read_data_files_for_starter(int level,
	std::string &prefix, int verbose_level)
// Calls gen->read_level_file_binary for all levels i from 0 to level
// Uses letter a files for i from 0 to level - 1
// and letter b file for i = level.
// If gen->f_starter is TRUE, we start from i = gen->starter_size instead.
// Finally, it computes nb_starter.
{
	int f_v = (verbose_level >= 1);
	string fname_base_a;
	string fname_base_b;
	int i, i0;

	if (f_v) {
		cout << "substructure_classification::read_data_files_for_starter" << endl;
		cout << "prefix=" << prefix << endl;
		cout << "level=" << level << endl;
	}

	fname_base_a.assign(prefix);
	fname_base_a.append("a");
	fname_base_b.assign(prefix);
	fname_base_b.append("b");

	if (gen->has_base_case()) {
		i0 = gen->get_Base_case()->size;
	}
	else {
		i0 = 0;
	}
	if (f_v) {
		cout << "substructure_classification::read_data_files_for_starter "
				"i0=" << i0 << endl;
	}
	for (i = i0; i < level; i++) {
		if (f_v) {
			cout << "substructure_classification::read_data_files_for_starter "
					"reading data file for level "
					<< i << " with prefix " << fname_base_b << endl;
		}
		gen->read_level_file_binary(i, fname_base_b,
				MINIMUM(1, verbose_level - 1));
	}

	if (f_v) {
		cout << "substructure_classification::read_data_files_for_starter "
				"reading data file for level " << level
				<< " with prefix " << fname_base_a << endl;
	}
	gen->read_level_file_binary(level, fname_base_a,
			MINIMUM(1, verbose_level - 1));

	if (f_v) {
		cout << "substructure_classification::read_data_files_for_starter "
				"before compute_nb_starter" << endl;
	}
	compute_nb_starter(level, verbose_level);

	if (f_v) {
		cout << "substructure_classification::read_data_files_for_starter finished, "
				"number of starters = " << nb_starter << endl;
	}
}

void substructure_classification::compute_nb_starter(int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	nb_starter = gen->nb_orbits_at_level(level);
	if (f_v) {
		cout << "substructure_classification::compute_nb_starter finished, "
				"number of starters = " << nb_starter << endl;
	}

}

void substructure_classification::print_node_local(int level, int node_local)
{
	int n;

	n = gen->first_node_at_level(level) + node_local;
	cout << n << "=" << level << "/" << node_local;
}

void substructure_classification::print_node_global(int level, int node_global)
{
	int node_local;

	node_local = node_global - gen->first_node_at_level(level);
	cout << node_global << "=" << level << "/" << node_local;
}

void substructure_classification::setup_and_open_level_database(int verbose_level)
// Called from do_iso_test, identify and test_hash
// (Which are all in isomorph_testing.cpp)
// Calls init_DB_level for D1 and D2 and D1->open and D2->open.
// Calls fopen for fp_ge1 and fp_ge2.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "substructure_classification::setup_and_open_level_database" << endl;
	}

	if (D1) {
		freeobject(D1);
		D1 = NULL;
	}
	if (D2) {
		freeobject(D2);
		D2 = NULL;
	}
	D1 = (layer2_discreta::typed_objects::database *) callocobject(layer2_discreta::typed_objects::DATABASE);
	D1->change_to_database();
	D2 = (layer2_discreta::typed_objects::database *) callocobject(layer2_discreta::typed_objects::DATABASE);
	D2->change_to_database();

	if (f_v) {
		cout << "substructure_classification::setup_and_open_level_database before init_DB_level D1" << endl;
	}
	init_DB_level(*D1, Iso->level - 1, verbose_level - 1);
	if (f_v) {
		cout << "substructure_classification::setup_and_open_level_database after init_DB_level D1" << endl;
	}
	fname_ge1.assign(fname_db_level_ge);
	if (f_v) {
		cout << "substructure_classification::setup_and_open_level_database fname_ge1=" << fname_ge1 << endl;
	}

	if (f_v) {
		cout << "substructure_classification::setup_and_open_level_database before init_DB_level D2" << endl;
	}
	init_DB_level(*D2, Iso->level, verbose_level - 1);
	if (f_v) {
		cout << "substructure_classification::setup_and_open_level_database after init_DB_level D2" << endl;
	}
	fname_ge2.assign(fname_db_level_ge);
	if (f_v) {
		cout << "substructure_classification::setup_and_open_level_database fname_ge2=" << fname_ge2 << endl;
	}

	if (f_v) {
		cout << "substructure_classification::setup_and_open_level_database before D1->open" << endl;
	}
	D1->open(0/*verbose_level - 1*/);
	D2->open(0/*verbose_level - 1*/);

	if (f_v) {
		cout << "substructure_classification::setup_and_open_level_database before fp_ge1" << endl;
	}
	fp_ge1 = new ifstream(fname_ge1, ios::binary);
	fp_ge2 = new ifstream(fname_ge2, ios::binary);
	//fp_ge1 = fopen(fname_ge1, "r");
	//fp_ge2 = fopen(fname_ge2, "r");
	if (f_v) {
		cout << "substructure_classification::setup_and_open_level_database done" << endl;
	}
}

void substructure_classification::close_level_database(int verbose_level)
// Closes D1, D2, fp_ge1, fp_ge2.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "substructure_classification::close_level_database" << endl;
	}
	D1->close(0/*verbose_level - 1*/);
	D2->close(0/*verbose_level - 1*/);
	freeobject(D1);
	freeobject(D2);
	D1 = NULL;
	D2 = NULL;
	delete fp_ge1;
	delete fp_ge2;
	//fclose(fp_ge1);
	//fclose(fp_ge2);
	fp_ge1 = NULL;
	fp_ge2 = NULL;
}

void substructure_classification::prepare_database_access(
		int cur_level, int verbose_level)
// sets DB_level to be D1 or D2, depending on cur_level
// Called from make_set_smaller_database
// and load_strong_generators
// and trace_next_point_database
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "substructure_classification::prepare_database_access "
				"cur_level=" << cur_level << endl;
	}
	if (cur_level == Iso->level - 1) {
		//first_node = gen->first_poset_orbit_node_at_level[level - 1];
		DB_level = D1;
		fp_ge = fp_ge1;
	}
	else if (cur_level == Iso->level) {
		//first_node = gen->first_poset_orbit_node_at_level[level];
		DB_level = D2;
		fp_ge = fp_ge2;
	}
	else {
		cout << "iso_node " << Iso->Folding->iso_nodes
				<< " substructure_classification::prepare_database_access "
						"cur_level = " << cur_level << endl;
		exit(1);
	}
}



void substructure_classification::find_extension_easy(
		long int *set, int case_nb,
		int &idx, int &f_found, int verbose_level)
// case_nb is the starter that is associated with the given set.
// We wish to find out if the set is a solution that has been stored
// with that starter.
// If so, we wish to determine the number of that solution amongst all
// solutions for that starter (returned in idx).
// Otherwise, we return FALSE.

// returns TRUE if found, FALSE otherwise
// Called from identify_solution
// Linear search through all solutions at a given starter.
// calls load solution for each of the solutions
// stored with the case and compares the vectors.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "substructure_classification::find_extension_easy "
				"case_nb=" << case_nb << endl;
	}
	f_found = FALSE;
#if 0
	int ret1, idx1;
	int ret2, idx2;
	ret1 = find_extension_easy_old(D, set, case_nb, idx1, verbose_level);
	if (f_v) {
		cout << "substructure_classification::find_extension_easy idx1=" << idx1 << endl;
	}
	ret2 = find_extension_easy_new(D, set, case_nb, idx2, verbose_level);
	if (f_v) {
		cout << "substructure_classification::find_extension_easy idx2=" << idx2 << endl;
	}
	if (ret1 != ret2) {
		cout << "substructure_classification::find_extension_easy ret1 != ret2" << endl;
		exit(1);
	}
	if (ret1 && (idx1 != idx2)) {
		cout << "substructure_classification::find_extension_easy "
				"ret1 && (idx1 != idx2)" << endl;
		exit(1);
	}
	idx = idx1;
	return ret1;
#else

	find_extension_easy_new(set, case_nb, idx, f_found, verbose_level);

	#endif
}

int substructure_classification::find_extension_search_interval(long int *set,
	int first, int len, int &idx,
	int f_btree_idx, int btree_idx,
	int f_through_hash, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "substructure_classification::find_extension_search_interval" << endl;
	}

	long int *data = Iso->Folding->find_extension_set1;
	int i, id = 0;
	data_structures::sorting Sorting;

	for (i = 0; i < len; i++) {
		if (f_btree_idx) {
			Iso->Lifting->load_solution_by_btree(btree_idx, first + i, id, data);
		}
		else {
			if (f_through_hash) {
				id = Iso->Lifting->hash_vs_id_id[first + i];
			}
			else {
				id = first + i;
			}
			Iso->Lifting->load_solution(id, data, verbose_level - 1);
		}
		Sorting.lint_vec_heapsort(data + Iso->level, Iso->size - Iso->level);
		if (Sorting.lint_vec_compare(set + Iso->level, data + Iso->level, Iso->size - Iso->level) == 0) {
			break;
		}
	}
	if (i == len) {
		return FALSE;
		//cout << "isomorph::find_extension_search_interval "
		//"did not find extension" << endl;
		//exit(1);
	}
	idx = id;
	if (f_v) {
		cout << "substructure_classification::find_extension_search_interval done" << endl;
	}
	return TRUE;
}

int substructure_classification::find_extension_easy_old(long int *set,
		int case_nb, int &idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int first, len, ret;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "substructure_classification::find_extension_easy_old" << endl;
		cout << "case_nb=" << case_nb << endl;
	}
	Sorting.lint_vec_heapsort(set + Iso->level, Iso->size - Iso->level);
	first = Iso->Lifting->solution_first[case_nb];
	len = Iso->Lifting->solution_len[case_nb];
	ret = find_extension_search_interval(set,
		first, len, idx, FALSE, 0, FALSE, verbose_level);
	if (f_v) {
		if (ret) {
			cout << "substructure_classification::find_extension_easy_old "
					"solution found at idx=" << idx << endl;
		}
		else {
			cout << "substructure_classification::find_extension_easy_old "
					"solution not found" << endl;
		}
	}
	return ret;
}

void substructure_classification::find_extension_easy_new(long int *set,
		int case_nb, int &idx, int &f_found, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = FALSE; // (verbose_level >= 2);
	//int ret;
	int first, idx2, len;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "substructure_classification::find_extension_easy_new" << endl;
	}
	Sorting.lint_vec_heapsort(set + Iso->level, Iso->size - Iso->level);

	long int h;
	data_structures::data_structures_global Data;

	h = Data.lint_vec_hash_after_sorting(set, Iso->size);
	if (f_v) {
		cout << "substructure_classification::find_extension_easy_new h=" << h << endl;
	}


	if (f_v) {
		cout << "substructure_classification::find_extension_easy_new before "
				"int_vec_search_first_occurrence(h)" << endl;
	}
	f_found = Sorting.lint_vec_search_first_occurrence(
			Iso->Lifting->hash_vs_id_hash,
			Iso->Lifting->N, h, first, 0 /*verbose_level*/);
	if (f_v) {
		cout << "substructure_classification::find_extension_easy_new after "
				"int_vec_search_first_occurrence(h) f_found=" << f_found << endl;
	}

	if (!f_found) {
		goto finish;
	}
	if (f_v) {
		cout << "substructure_classification::find_extension_easy_new before "
				"int_vec_search_first_occurrence(h + 1) h+1=" << h + 1 << endl;
	}
	f_found = Sorting.lint_vec_search_first_occurrence(
			Iso->Lifting->hash_vs_id_hash,
			Iso->Lifting->N, h + 1, idx2, 0 /*verbose_level*/);
	if (f_v) {
		cout << "substructure_classification::find_extension_easy_new after "
				"int_vec_search_first_occurrence(h+1) f_found=" << f_found << endl;
	}
	len = idx2 - first;
	if (f_v) {
		cout << "substructure_classification::find_extension_easy_new len=" << len << endl;
	}
#if 0

	if (f_vv) {
		cout << "case_nb=" << case_nb << " h=" << h << endl;
	}
	btree &B4 = DB_sol->btree_access_i(3);
	//int l0 = case_nb;
	//int u0 = case_nb;
	//int l1 = h;
	//int u1 = h + 1;
	int first, last, len;
	int f_found1, f_found2;
	f_found1 = B4.search_int4_int4(case_nb, h, first,
			0 /*verbose_level */);

#if 0
	B4.search_interval_int4_int4(l0, u0,
		l1, u1,
		first, len,
		3 /*verbose_level*/);
#endif
	if (f_vv) {
		cout << "f_found1=" << f_found1 << " first=" << first << endl;
	}
	f_found2 = B4.search_int4_int4(case_nb, h + 1, last,
			0 /*verbose_level */);
	if (f_vv) {
		cout << "f_found2=" << f_found2 << " last=" << last << endl;
	}
	len = last - first + 1;
#endif

	if (len == 0) {
		f_found = FALSE;
	}
	else {
		if (f_v) {
			cout << "substructure_classification::find_extension_easy_new before "
					"find_extension_search_interval" << endl;
		}

		f_found = find_extension_search_interval(set,
			first, len, idx, FALSE, 3, TRUE, 0 /*verbose_level*/);

		if (f_v) {
			cout << "substructure_classification::find_extension_easy_new after "
					"find_extension_search_interval f_found=" << f_found << endl;
		}
	}

finish:


	if (f_v) {
		if (f_found) {
			cout << "substructure_classification::find_extension_easy_new "
					"solution found at idx=" << idx << endl;
		}
		else {
			cout << "substructure_classification::find_extension_easy_new "
					"solution not found" << endl;
		}
	}

}

int substructure_classification::open_database_and_identify_object(long int *set,
	int *transporter,
	int f_implicit_fusion, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int r;
	int f_failure_to_find_point;

	if (f_v) {
		cout << "substructure_classification::open_database_and_identify_object" << endl;
	}

	Iso->Lifting->setup_and_open_solution_database(0/*verbose_level - 1*/);
	setup_and_open_level_database(0/*verbose_level - 1*/);

	r = Iso->Folding->identify_solution(set, transporter,
		f_implicit_fusion, f_failure_to_find_point, verbose_level - 2);

	if (f_failure_to_find_point) {
		cout << "substructure_classification::open_database_and_identify_object: "
				"f_failure_to_find_point" << endl;
		r = -1;
 	}

	else {
		if (f_v) {
			cout << "substructure_classification::open_database_and_identify_object: "
					"object identified as belonging to isomorphism class "
					<< r << endl;
		}
	}

	Iso->Lifting->close_solution_database(0/*verbose_level - 1*/);
	close_level_database(0/*verbose_level - 1*/);
	if (f_v) {
		cout << "substructure_classification::open_database_and_identify_object done" << endl;
	}
	return r;
}



void substructure_classification::init_DB_level(
		layer2_discreta::typed_objects::database &D,
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	layer2_discreta::typed_objects::btree B1, B2;
	int f_compress = TRUE;
	int f_duplicatekeys = TRUE;
	int i;
	char str[1000];

	if (f_v) {
		cout << "substructure_classification::init_DB_level level=" << level << endl;
	}

	if (f_v) {
		cout << "substructure_classification::init_DB_level Iso->prefix=" << Iso->prefix << endl;
	}
	fname_db_level.assign(Iso->prefix);
	snprintf(str, sizeof(str), "starter_lvl_%d.db", level);
	fname_db_level.append(str);

	if (f_v) {
		cout << "substructure_classification::init_DB_level fname_db_level=" << fname_db_level << endl;
	}

	fname_db_level_idx1.assign(Iso->prefix);
	snprintf(str, sizeof(str), "starter_lvl_%d_a.idx", level);
	fname_db_level_idx1.append(str);

	if (f_v) {
		cout << "substructure_classification::init_DB_level fname_db_level_idx1=" << fname_db_level_idx1 << endl;
	}

	fname_db_level_idx2.assign(Iso->prefix);
	snprintf(str, sizeof(str), "starter_lvl_%d_b.idx", level);
	fname_db_level_idx2.append(str);

	if (f_v) {
		cout << "substructure_classification::init_DB_level fname_db_level_idx2=" << fname_db_level_idx2 << endl;
	}

	fname_db_level_ge.assign(Iso->prefix);
	snprintf(str, sizeof(str), "starter_lvl_%d_ge.bin", level);
	fname_db_level_ge.append(str);

	if (f_v) {
		cout << "substructure_classification::init_DB_level fname_db_level_ge=" << fname_db_level_ge << endl;
	}

	if (f_v) {
		cout << "substructure_classification::init_DB_level before D.init" << endl;
	}
	D.init(fname_db_level.c_str(), layer2_discreta::typed_objects::VECTOR, f_compress);
	if (f_v) {
		cout << "substructure_classification::init_DB_level after D.init" << endl;
	}

	if (f_v) {
		cout << "substructure_classification::init_DB_level before B1.init" << endl;
	}
	B1.init(fname_db_level_idx1.c_str(), f_duplicatekeys, 0 /* btree_idx */);
	if (f_v) {
		cout << "substructure_classification::init_DB_level after B1.init" << endl;
	}

	if (f_v) {
		cout << "substructure_classification::init_DB_level before B1.add_key_int4" << endl;
	}
	B1.add_key_int4(0, 0);
	if (f_v) {
		cout << "substructure_classification::init_DB_level after B1.add_key_int4" << endl;
	}

	if (f_v) {
		cout << "substructure_classification::init_DB_level before D.btree_access" << endl;
	}
	D.btree_access().append(B1);
	if (f_v) {
		cout << "substructure_classification::init_DB_level after D.btree_access" << endl;
	}

	if (f_v) {
		cout << "substructure_classification::init_DB_level before B2.init" << endl;
	}
	B2.init(fname_db_level_idx2.c_str(), f_duplicatekeys, 1 /* btree_idx */);
		// 2 up to 2+level-1 are the values of the starter (of size level)
	if (f_v) {
		cout << "substructure_classification::init_DB_level after B2.init" << endl;
	}
	for (i = 0; i < level; i++) {
		B2.add_key_int4(2 + i, 0);
	}
	D.btree_access().append(B2);
	if (f_v) {
		cout << "substructure_classification::init_DB_level level=" << level << " done" << endl;
	}
}

void substructure_classification::create_level_database(int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;//(verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int f, nb_nodes, I, J, i, j, idx, print_mod = 1;
	poset_classification::poset_orbit_node *O;
	long int set1[1000];
	long int set2[1000];
	//char *elt;
	int *Elt;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "substructure_classification::create_level_database "
				"level = " << level << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}

	Elt = NEW_int(gen->get_A()->elt_size_in_int);
	f = gen->first_node_at_level(level);
	nb_nodes = gen->nb_orbits_at_level(level);

	if (f_vv) {
		cout << "f=" << f << endl;
		cout << "nb_nodes=" << nb_nodes << endl;
	}

	layer2_discreta::typed_objects::database D;
	//FILE *fp;
	int cnt = 0;

	//elt = NEW_char(gen->A->coded_elt_size_in_char);

	if (f_v) {
		cout << "substructure_classification::create_level_database before init_DB_level" << endl;
	}
	init_DB_level(D, level, verbose_level - 1);
	if (f_v) {
		cout << "substructure_classification::create_level_database after init_DB_level" << endl;
	}

	if (f_v) {
		cout << "substructure_classification::create_level_database before D.create" << endl;
	}
	D.create(0/*verbose_level - 1*/);
	if (f_v) {
		cout << "substructure_classification::create_level_database after D.create" << endl;
	}
	//fp = fopen(fname_db_level_ge, "wb");
	{
		ofstream fp(fname_db_level_ge, ios::binary);

		//if (nb_nodes > 10000) {
			print_mod = 1000;
			//}
		for (i = 0; i < nb_nodes; i++) {
			I = f + i;
			O = gen->get_node(I);
			O->store_set_to(gen, level - 1, set1);
			if (f_v && ((i % print_mod) == 0) && i) {
				cout << "substructure_classification::create_level_database level "
						<< level << " i=" << i << " / " << nb_nodes
						<< " set=";
				Lint_vec_print(cout, set1, level);
				cout << endl;
			}

			int len, nb_fusion;
			layer2_discreta::typed_objects::Vector v;


				// # ints   description
				// 1         global ID
				// 1         ancestor global ID
				// level     the set itself
				// 1         # strong generators
				// A->base_len: tl  (only if # strong generators is != 0)
				// 1         nb_extensions
				// for each extension:
				// 1         pt
				// 1         orbit_len
				// 1         type
				// 1         global ID of descendant node
				//           (if type == 1 EXTENSION or type == 2 FUSION)

				// and finally:
				// 1         ref of first group element
				// (altogether, we are storing
				// # strong_generators  +
				// # fusion nodes
				// group elements.
				// they have refs d, d+1,...


				// A->coded_elt_size_in_char


			nb_fusion = 0;
			for (j = 0; j < O->get_nb_of_extensions(); j++) {
				if (O->get_E(j)->get_type() == EXTENSION_TYPE_FUSION) {
					nb_fusion++;
				}
			}

			len = 1 + 1 + level + 1;
			if (O->get_nb_strong_generators()) {
				len += gen->get_A()->base_len();
			}
			len += 1;
			len += 4 * O->get_nb_of_extensions();
			len += 1; // for the reference of the first group element
			//len += O->nb_strong_generators;
			//len += nb_fusion;

			v.m_l_n(len);
			idx = 0;
			v.m_ii(idx++, I);
			v.m_ii(idx++, O->get_prev());
			for (j = 0; j < level; j++) {
				v.m_ii(idx++, set1[j]);
				}
			v.m_ii(idx++, O->get_nb_strong_generators());
			if (O->get_nb_strong_generators()) {
				for (j = 0; j < gen->get_A()->base_len(); j++) {
					v.m_ii(idx++, O->get_tl(j));
				}
			}
			v.m_ii(idx++, O->get_nb_of_extensions());
			for (j = 0; j < O->get_nb_of_extensions(); j++) {
				v.m_ii(idx++, O->get_E(j)->get_pt());
				set1[level] = O->get_E(j)->get_pt();
				v.m_ii(idx++, O->get_E(j)->get_orbit_len());
				v.m_ii(idx++, O->get_E(j)->get_type());
				if (O->get_E(j)->get_type() == EXTENSION_TYPE_EXTENSION) {
					v.m_ii(idx++, O->get_E(j)->get_data());
				}
				else if (O->get_E(j)->get_type() == EXTENSION_TYPE_FUSION) {
					//gen->get_A()->element_retrieve(O->get_E(j)->get_data(), gen->get_Elt1(), FALSE);


					gen->get_A2()->map_a_set_based_on_hdl(set1, set2, level + 1, gen->get_A(), O->get_E(j)->get_data(), 0);
					Sorting.lint_vec_heapsort(set2, level + 1);

					if (f_vv /*f_vv && (i % print_mod) == 0*/) {
						cout << "mapping ";
						Lint_vec_print(cout, set1, level + 1);
						cout << " to ";
						Lint_vec_print(cout, set2, level + 1);
						cout << endl;
					}


					J = gen->find_poset_orbit_node_for_set(level + 1,
							set2, FALSE /* f_tolerant */, 0);
					v.m_ii(idx++, J);
				}
				else {
					cout << "unknown type " << O->get_E(j)->get_type()
							<< " i=" << i << " j=" << j << endl;
					exit(1);
				}
			}
#if 0
			int len_mem, h, idx1;
			char *mem;
			if (idx != len - 1) {
				cout << "idx != len - 1, idx=" << idx << " len=" << len
						<< " i=" << i << " j=" << j << endl;
				exit(1);
			}
			len_mem = (O->nb_strong_generators + nb_fusion) *
					gen->A->coded_elt_size_in_char;
			mem = NEW_char(len_mem);
			idx1 = 0;
			for (j = 0; j < O->nb_strong_generators; j++) {
				gen->A->element_retrieve(O->hdl_strong_generators[j],
						gen->Elt1, FALSE);
				gen->A->element_pack(gen->Elt1, elt, FALSE);
				for (h = 0; h < gen->A->coded_elt_size_in_char; h++) {
					mem[idx1++] = elt[h];
				}
			}
			for (j = 0; j < O->nb_extensions; j++) {
				if (O->E[j].type == 1)
					continue;
				gen->A->element_retrieve(O->E[j].data, gen->Elt1, FALSE);
				gen->A->element_pack(gen->Elt1, elt, FALSE);
				for (h = 0; h < gen->A->coded_elt_size_in_char; h++) {
					mem[idx1++] = elt[h];
				}
			}
			if (idx1 != len_mem) {
				cout << "idx1 != len_mem idx=" << idx << " len_mem=" << len_mem
						<< " i=" << i << " j=" << j << endl;
				exit(1);
			}
			memory M;

			M.init(len_mem, mem);
			M.swap(v.s_i(idx));
#else
			v.m_ii(idx++, cnt);

			std::vector<int> gen_hdl;

			O->get_strong_generators_handle(gen_hdl, verbose_level);

			if (f_v) {
				cout << "substructure_classification::create_level_database before writing generators, gen_hdl.size()=" << gen_hdl.size() << endl;
			}

			for (j = 0; j < gen_hdl.size(); j++) {
				if (f_v) {
					cout << "substructure_classification::create_level_database j=" << j << " / " << gen_hdl.size()
							<< " gen_hdl[j]=" << gen_hdl[j] << endl;
				}
				if (f_v) {
					cout << "substructure_classification::create_level_database before element_retrieve" << endl;
				}
				gen->get_A()->Group_element->element_retrieve(
						gen_hdl[j], Iso->Folding->Elt1,
						0/*verbose_level*/);
				if (f_v) {
					cout << "substructure_classification::create_level_database before element_write_file_fp" << endl;
				}
				gen->get_A()->Group_element->element_write_file_fp(Iso->Folding->Elt1, fp,
						0/* verbose_level*/);
				cnt++;
			}


			if (f_v) {
				cout << "substructure_classification::create_level_database before writing fusion elements, O->get_nb_of_extensions()=" << O->get_nb_of_extensions() << endl;
			}

			for (j = 0; j < O->get_nb_of_extensions(); j++) {
				if (O->get_E(j)->get_type() == EXTENSION_TYPE_EXTENSION) {
					continue;
				}
				gen->get_A()->Group_element->element_retrieve(O->get_E(j)->get_data(), Iso->Folding->Elt1, FALSE);
				gen->get_A()->Group_element->element_write_file_fp(Iso->Folding->Elt1, fp,
						0/* verbose_level*/);
				cnt++;
			}

			if (idx != len) {
				cout << "idx != len, idx=" << idx << " len=" << len << endl;
				exit(1);
			}
#endif

			if (f_v) {
				cout << "substructure_classification::create_level_database before D.add_object" << endl;
			}


			D.add_object(v, 0 /*verbose_level - 2*/);

			if (f_v && ((i % print_mod) == 0)) {
				cout << "object " << i << " / " << nb_nodes << " added : ";
				int sz;
				sz = v.csf();
				cout << "size on file = " << sz << ", group element "
						"counter = " << cnt << endl;
			}
		}


		//fclose(fp);

	}

	D.close(0/*verbose_level - 1*/);

	orbiter_kernel_system::file_io Fio;
	if (f_v) {
		cout << "number of group elements in " << fname_db_level_ge
				<< " is " << cnt << endl;
		cout << "file size is " << Fio.file_size(fname_db_level_ge) << endl;
		cout << "gen->A->coded_elt_size_in_char="
				<< gen->get_A()->coded_elt_size_in_char << endl;
	}

	FREE_int(Elt);

	//FREE_char(elt);

}

void substructure_classification::load_strong_generators(int cur_level,
		int cur_node_local,
		data_structures_groups::vector_ge &gens,
		ring_theory::longinteger_object &go,
		int verbose_level)
// Called from compute_stabilizer and from orbit_representative
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v5 = (verbose_level >= 5);

	if (f_v) {
		cout << "substructure_classification::load_strong_generators "
				"cur_level=" << cur_level << " cur_node_local="
				<< cur_node_local << endl;
	}
	if (f_use_database_for_starter) {
		if (f_vv) {
			cout << "substructure_classification::load_strong_generators "
					"using database" << endl;
		}
		load_strong_generators_database(cur_level, cur_node_local,
			gens, go, verbose_level);
		if (f_v5) {
			cout << "substructure_classification::load_strong_generators "
					"found the following strong generators:" << endl;
			gens.print(cout);
		}
	}
	else {
		load_strong_generators_tree(cur_level, cur_node_local,
			gens, go, verbose_level);
	}
	if (f_v) {
		cout << "substructure_classification::load_strong_generators done" << endl;
	}
}

void substructure_classification::load_strong_generators_tree(int cur_level,
	int cur_node_local,
	data_structures_groups::vector_ge &gens,
	ring_theory::longinteger_object &go,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	poset_classification::poset_orbit_node *O;
	int i, node;
	//longinteger_domain Dom;

	if (f_v) {
		cout << "substructure_classification::load_strong_generators_tree "
				"cur_level=" << cur_level << " cur_node_local="
				<< cur_node_local << endl;
	}

	node = gen->first_node_at_level(cur_level) + cur_node_local;
	O = gen->get_node(node);


	std::vector<int> gen_hdl;

	O->get_strong_generators_handle(gen_hdl, verbose_level);



#if 0
	if (O->nb_strong_generators == 0) {
		gens.init(gen->get_A(), verbose_level - 2);
		gens.allocate(0, verbose_level - 2);
		go.create(1, __FILE__, __LINE__);
		goto finish;
	}
	int *tl;
	tl = NEW_int(gen->get_A()->base_len());
	for (i = 0; i < gen->get_A()->base_len(); i++) {
		tl[i] = O->get_tl(i);
	}

	Dom.multiply_up(go, tl, gen->get_A()->base_len(), 0 /* verbose_level */);

	FREE_int(tl);
#else
	O->get_stabilizer_order(gen, go);
#endif

	gens.init(gen->get_A(), verbose_level - 2);
	gens.allocate(gen_hdl.size(), verbose_level - 2);

	for (i = 0; i < gen_hdl.size(); i++) {
		gen->get_A()->Group_element->element_retrieve(
				gen_hdl[i],
				gens.ith(i), FALSE);
	}
//finish:
	if (f_v) {
		cout << "substructure_classification::load_strong_generators_tree "
				"cur_level=" << cur_level << " cur_node_local="
				<< cur_node_local << " done" << endl;
	}
}

void substructure_classification::load_strong_generators_database(int cur_level,
		int cur_node_local,
		data_structures_groups::vector_ge &gens,
		ring_theory::longinteger_object &go,
		int verbose_level)
// Reads node cur_node (global index) from database D through btree 0
// Reads generators from file fp_ge
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int *tmp_ELT;
	layer2_discreta::typed_objects::Vector v;
	int i;
	int set[1000];
	int *tl;
	ring_theory::longinteger_domain Dom;


	if (f_v) {
		cout << "substructure_classification::load_strong_generators_database "
				"cur_level=" << cur_level << " cur_node_local="
				<< cur_node_local << endl;
	}

	prepare_database_access(cur_level, verbose_level);

	tmp_ELT = NEW_int(gen->get_A()->elt_size_in_int);

	//cur_node_local = cur_node - first_node;
	if (f_v) {
		cout << "substructure_classification::load_strong_generators_database "
				"loading object " << cur_node_local << endl;
	}
	DB_level->ith_object(cur_node_local, 0/* btree_idx*/, v,
			0 /*MINIMUM(1, verbose_level - 2)*/);

	if (f_vvv) {
		cout << "substructure_classification::load_strong_generators_database "
				"v=" << v << endl;
	}
	for (i = 0; i < cur_level; i++) {
		set[i] = v.s_ii(2 + i);
	}
	if (f_vv) {
		cout << "substructure_classification::load_strong_generators_database set: ";
		Int_vec_print(cout, set, cur_level);
		cout << endl;
	}
	int nb_strong_generators;
	int pos, ref;
	pos = 2 + cur_level;
	nb_strong_generators = v.s_ii(pos++);
	if (f_vv) {
		cout << "substructure_classification::load_strong_generators_database "
				"nb_strong_generators="
				<< nb_strong_generators << endl;
	}
	if (nb_strong_generators == 0) {
		gens.init(gen->get_A(), verbose_level - 2);
		gens.allocate(0, verbose_level - 2);
		go.create(1, __FILE__, __LINE__);
		goto finish;
	}
	tl = NEW_int(gen->get_A()->base_len());
	for (i = 0; i < gen->get_A()->base_len(); i++) {
		tl[i] = v.s_ii(pos++);
	}
	Dom.multiply_up(go, tl, gen->get_A()->base_len(), 0 /* verbose_level */);
	FREE_int(tl);
	pos = v.s_l() - 1;
	ref = v.s_ii(pos++);
	if (f_vv) {
		cout << "substructure_classification::load_strong_generators_database "
				"ref = " << ref << endl;
	}

	gens.init(gen->get_A(), verbose_level - 2);
	gens.allocate(nb_strong_generators, verbose_level - 2);

	//fseek(fp_ge, ref * gen->Poset->A->coded_elt_size_in_char, SEEK_SET);
	fp_ge->seekg(ref * gen->get_A()->coded_elt_size_in_char, ios::beg);
	for (i = 0; i < nb_strong_generators; i++) {
		gen->get_A()->Group_element->element_read_file_fp(gens.ith(i), *fp_ge,
				0/* verbose_level*/);
	}
finish:
	FREE_int(tmp_ELT);

	if (f_v) {
		cout << "substructure_classification::load_strong_generators_database "
				"cur_level=" << cur_level << " cur_node_local="
				<< cur_node_local << " done" << endl;
	}

}



}}}

