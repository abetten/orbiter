/*
 * packings_global.cpp
 *
 *  Created on: Oct 18, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {
namespace packings {


packings_global::packings_global()
{
}


packings_global::~packings_global()
{
}



void packings_global::merge_packings(
		std::string *fnames, int nb_files,
		std::string &file_of_spreads,
		data_structures::classify_bitvectors *&CB,
		int verbose_level)
{
#if 0
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "packings_global::merge_packings" << endl;
	}

	CB = NEW_OBJECT(classify_bitvectors);


	// for use if INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE
	long int *Spread_table;
	int nb_spreads;
	int spread_size;

	if (f_v) {
		cout << "packings_global::merge_packings "
				"Reading spread table from file "
				<< file_of_spreads << endl;
	}
	Fio.lint_matrix_read_csv(file_of_spreads,
			Spread_table, nb_spreads, spread_size,
			0 /* verbose_level */);
	if (f_v) {
		cout << "Reading spread table from file "
				<< file_of_spreads << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
	}

	int f, g, N, table_length, nb_reject = 0;

	N = 0;

	if (f_v) {
		cout << "packings_global::merge_packings "
				"counting the overall number of input packings" << endl;
	}

	for (f = 0; f < nb_files; f++) {

		if (f_v) {
			cout << "packings_global::merge_packings file "
					<< f << " / " << nb_files << " : " << fnames[f] << endl;
		}

		spreadsheet *S;

		S = NEW_OBJECT(spreadsheet);

		S->read_spreadsheet(fnames[f], 0 /*verbose_level*/);

		table_length = S->nb_rows - 1;
		N += table_length;



		FREE_OBJECT(S);

	}

	if (f_v) {
		cout << "packings_global::merge_packings file "
				<< "we have " << N << " packings in "
				<< nb_files << " files" << endl;
	}

	for (f = 0; f < nb_files; f++) {

		if (f_v) {
			cout << "packings_global::merge_packings file "
					<< f << " / " << nb_files << " : " << fnames[f] << endl;
		}

		spreadsheet *S;

		S = NEW_OBJECT(spreadsheet);

		S->read_spreadsheet(fnames[f], 0 /*verbose_level*/);
		if (FALSE /*f_v3*/) {
			S->print_table(cout, FALSE);
			}

		int ago_idx, original_file_idx, input_idx_idx, input_set_idx;
		int nb_rows_idx, nb_cols_idx, canonical_form_idx;

		ago_idx = S->find_by_column("ago");
		original_file_idx = S->find_by_column("original_file");
		input_idx_idx = S->find_by_column("input_idx");
		input_set_idx = S->find_by_column("input_set");
		nb_rows_idx = S->find_by_column("nb_rows");
		nb_cols_idx = S->find_by_column("nb_cols");
		canonical_form_idx = S->find_by_column("canonical_form");

		table_length = S->nb_rows - 1;

		//rep,ago,original_file,input_idx,input_set,nb_rows,nb_cols,canonical_form


		for (g = 0; g < table_length; g++) {

			int ago;
			char *text;
			long int *the_set_in;
			int set_size_in;
			long int *canonical_labeling;
			int canonical_labeling_sz;
			int nb_rows, nb_cols;
			object_in_projective_space *OiP;


			ago = S->get_int(g + 1, ago_idx);
			nb_rows = S->get_int(g + 1, nb_rows_idx);
			nb_cols = S->get_int(g + 1, nb_cols_idx);

			text = S->get_string(g + 1, input_set_idx);
			Orbiter->Lint_vec.scan(text, the_set_in, set_size_in);


			if (f_v) {
				cout << "File " << f << " / " << nb_files
						<< ", input set " << g << " / "
						<< table_length << endl;
				//int_vec_print(cout, the_set_in, set_size_in);
				//cout << endl;
				}

			if (FALSE) {
				cout << "canonical_form_idx=" << canonical_form_idx << endl;
			}
			text = S->get_string(g + 1, canonical_form_idx);
			if (FALSE) {
				cout << "text=" << text << endl;
			}
			Orbitr->Lint_vec.scan(text, canonical_labeling, canonical_labeling_sz);
			if (FALSE) {
				cout << "File " << f << " / " << nb_files
						<< ", input set " << g << " / "
						<< table_length << " canonical_labeling = ";
				Orbiter->Lint_vec.print(cout, canonical_labeling, canonical_labeling_sz);
				cout << endl;
				}

			if (canonical_labeling_sz != nb_rows + nb_cols) {
				cout << "packings_global::merge_packings "
						"canonical_labeling_sz != nb_rows + nb_cols" << endl;
				exit(1);
			}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (FALSE) {
				cout << "packings_global::merge_packings "
						"before init_packing_from_spread_table" << endl;
			}
			OiP->init_packing_from_spread_table(P, the_set_in,
				Spread_table, nb_spreads, spread_size,
				0 /*verbose_level*/);
			if (FALSE) {
				cout << "packings_global::merge_packings "
						"after init_packing_from_spread_table" << endl;
			}
			OiP->f_has_known_ago = TRUE;
			OiP->known_ago = ago;

			int *Incma_in;
			int *Incma_out;
			int nb_rows1, nb_cols1;
			int *partition;
			uchar *canonical_form;
			int canonical_form_len;


			if (FALSE) {
				cout << "packings_global::merge_packings "
						"before encode_incma" << endl;
			}
			OiP->encode_incma(Incma_in, nb_rows1, nb_cols1,
					partition, 0 /*verbose_level - 1*/);
			if (FALSE) {
				cout << "packings_global::merge_packings "
						"after encode_incma" << endl;
			}
			if (nb_rows1 != nb_rows) {
				cout << "packings_global::merge_packings "
						"nb_rows1 != nb_rows" << endl;
				exit(1);
			}
			if (nb_cols1 != nb_cols) {
				cout << "packings_global::merge_packings "
						"nb_cols1 != nb_cols" << endl;
				exit(1);
			}

			OiP->input_fname = S->get_string(g + 1, original_file_idx);
			OiP->input_idx = S->get_int(g + 1, input_idx_idx);

			text = S->get_string(g + 1, input_set_idx);

			OiP->set_as_string.assign(text);

			int i, j, ii, jj, a;
			int L = nb_rows * nb_cols;

			Incma_out = NEW_int(L);
			for (i = 0; i < nb_rows; i++) {
				ii = canonical_labeling[i];
				for (j = 0; j < nb_cols; j++) {
					jj = canonical_labeling[nb_rows + j] - nb_rows;
					//cout << "i=" << i << " j=" << j << " ii=" << ii
					//<< " jj=" << jj << endl;
					Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
					}
				}
			if (FALSE) {
				cout << "packings_global::merge_packings "
						"before bitvector_allocate_and_coded_length" << endl;
			}
			canonical_form = bitvector_allocate_and_coded_length(
					L, canonical_form_len);
			for (i = 0; i < nb_rows; i++) {
				for (j = 0; j < nb_cols; j++) {
					if (Incma_out[i * nb_cols + j]) {
						a = i * nb_cols + j;
						bitvector_set_bit(canonical_form, a);
						}
					}
				}

			if (CB->n == 0) {
				if (f_v) {
					cout << "packings_global::merge_packings "
							"before CB->init" << endl;
				}
				CB->init(N, canonical_form_len, verbose_level);
				}
			if (f_v) {
				cout << "packings_global::merge_packings "
						"before CB->add" << endl;
			}
			int idx;
			int f_found;

			CB->search_and_add_if_new(canonical_form, OiP, f_found, idx, 0 /*verbose_level*/);
			if (f_found) {
				nb_reject++;
			}
			if (f_v) {
				cout << "packings_global::merge_packings "
						"CB->add returns f_found = " << f_found
						<< " nb iso = " << CB->nb_types
						<< " nb_reject=" << nb_reject << endl;
			}


			//int idx;

			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init(OiP, ago, nb_rows, nb_cols,
					canonical_labeling, 0 /*verbose_level*/);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;


			FREE_lint(the_set_in);
			//FREE_int(canonical_labeling);
			FREE_int(Incma_in);
			FREE_int(Incma_out);
			FREE_int(partition);
			//FREE_uchar(canonical_form);

		} // next g



	} // next f

	if (f_v) {
		cout << "packings_global::merge_packings done, "
				"we found " << CB->nb_types << " isomorphism types "
				"of packings" << endl;
		}


	//FREE_OBJECT(CB);
	FREE_lint(Spread_table);

	if (f_v) {
		cout << "packings_global::merge_packings done" << endl;
	}
#endif
}

void packings_global::select_packings(
		std::string &fname,
		std::string &file_of_spreads_original,
		spread_tables *Spread_tables,
		int f_self_polar,
		int f_ago, int select_ago,
		data_structures::classify_bitvectors *&CB,
		int verbose_level)
{
#if 0
	int f_v = (verbose_level >= 1);


	int nb_accept = 0;
	file_io Fio;

	if (f_v) {
		cout << "packings_global::select_packings" << endl;
	}

	CB = NEW_OBJECT(classify_bitvectors);



	long int *Spread_table;
	int nb_spreads;
	int spread_size;
	int packing_size;
	int a, b;

	if (f_v) {
		cout << "packings_global::select_packings "
				"Reading spread table from file "
				<< file_of_spreads_original << endl;
	}
	Fio.lint_matrix_read_csv(file_of_spreads_original,
			Spread_table, nb_spreads, spread_size,
			0 /* verbose_level */);
	if (nb_spreads != Spread_tables->nb_spreads) {
		cout << "packings_global::select_packings "
				"nb_spreads != Spread_tables->nb_spreads" << endl;
		exit(1);
	}
	if (spread_size != Spread_tables->spread_size) {
		cout << "packings_global::select_packings "
				"spread_size != Spread_tables->spread_size" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Reading spread table from file "
				<< file_of_spreads_original << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
	}



	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s" << endl;
	}

	int *s2l, *l2s;
	int i, idx;
	long int *set;
	long int extra_data[1];
	sorting Sorting;

	extra_data[0] = spread_size;

	set = NEW_lint(spread_size);
	s2l = NEW_int(nb_spreads);
	l2s = NEW_int(nb_spreads);
	for (i = 0; i < nb_spreads; i++) {
		Orbiter->Lint_vec.copy(Spread_tables->spread_table +
				i * spread_size, set, spread_size);
		Sorting.lint_vec_heapsort(set, spread_size);
		if (!Sorting.search_general(Spread_tables->spread_table,
				nb_spreads, (void *) set, idx,
				table_of_sets_compare_func,
				extra_data, 0 /*verbose_level*/)) {
			cout << "packings_global::select_packings "
					"cannot find spread " << i << " = ";
			Orbiter->Lint_vec.print(cout, set, spread_size);
			cout << endl;
			exit(1);
		}
		s2l[i] = idx;
		l2s[idx] = i;
	}
	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s done" << endl;
	}

	int g, table_length, nb_reject = 0;


	if (f_v) {
		cout << "packings_global::select_packings file "
				<< fname << endl;
	}

	spreadsheet *S;

	S = NEW_OBJECT(spreadsheet);

	S->read_spreadsheet(fname, 0 /*verbose_level*/);
	if (FALSE /*f_v3*/) {
		S->print_table(cout, FALSE);
		}

	int ago_idx, original_file_idx, input_idx_idx, input_set_idx;
	int nb_rows_idx, nb_cols_idx, canonical_form_idx;

	ago_idx = S->find_by_column("ago");
	original_file_idx = S->find_by_column("original_file");
	input_idx_idx = S->find_by_column("input_idx");
	input_set_idx = S->find_by_column("input_set");
	nb_rows_idx = S->find_by_column("nb_rows");
	nb_cols_idx = S->find_by_column("nb_cols");
	canonical_form_idx = S->find_by_column("canonical_form");

	table_length = S->nb_rows - 1;

	//rep,ago,original_file,input_idx,
	//input_set,nb_rows,nb_cols,canonical_form
	int f_first = TRUE;


	for (g = 0; g < table_length; g++) {

		int ago;
		char *text;
		long int *the_set_in;
		int set_size_in;
		long int *canonical_labeling;
		int canonical_labeling_sz;
		int nb_rows, nb_cols;
		object_in_projective_space *OiP;
		int f_accept = FALSE;
		int *set1;
		int *set2;

		ago = S->get_int(g + 1, ago_idx);
		nb_rows = S->get_int(g + 1, nb_rows_idx);
		nb_cols = S->get_int(g + 1, nb_cols_idx);

		text = S->get_string(g + 1, input_set_idx);
		Orbiter->Lint_vec.scan(text, the_set_in, set_size_in);

		packing_size = set_size_in;

		if (f_v && (g % 1000) == 0) {
			cout << "File " << fname
					<< ", input set " << g << " / "
					<< table_length << endl;
			//int_vec_print(cout, the_set_in, set_size_in);
			//cout << endl;
			}


		if (f_self_polar) {
			set1 = NEW_int(packing_size);
			set2 = NEW_int(packing_size);

			// test if self-polar:
			for (i = 0; i < packing_size; i++) {
				a = the_set_in[i];
				b = s2l[a];
				set1[i] = b;
			}
			Sorting.int_vec_heapsort(set1, packing_size);
			for (i = 0; i < packing_size; i++) {
				a = set1[i];
				b = Spread_tables->dual_spread_idx[a];
				set2[i] = b;
			}
			Sorting.int_vec_heapsort(set2, packing_size);

#if 0
			cout << "set1: ";
			int_vec_print(cout, set1, packing_size);
			cout << endl;
			cout << "set2: ";
			int_vec_print(cout, set2, packing_size);
			cout << endl;
#endif
			if (int_vec_compare(set1, set2, packing_size) == 0) {
				cout << "The packing is self-polar" << endl;
				f_accept = TRUE;
			}
			else {
				f_accept = FALSE;
			}
			FREE_int(set1);
			FREE_int(set2);
		}
		if (f_ago) {
			if (ago == select_ago) {
				f_accept = TRUE;
			}
			else {
				f_accept = FALSE;
			}
		}



		if (f_accept) {

			nb_accept++;


			if (FALSE) {
				cout << "canonical_form_idx=" << canonical_form_idx << endl;
			}
			text = S->get_string(g + 1, canonical_form_idx);
			if (FALSE) {
				cout << "text=" << text << endl;
			}
			Orbiter->Lint_vec.scan(text, canonical_labeling, canonical_labeling_sz);
			if (FALSE) {
				cout << "File " << fname
						<< ", input set " << g << " / "
						<< table_length << " canonical_labeling = ";
				Orbiter->Lint_vec.print(cout, canonical_labeling, canonical_labeling_sz);
				cout << endl;
				}

			if (canonical_labeling_sz != nb_rows + nb_cols) {
				cout << "packings_global::select_packings "
						"canonical_labeling_sz != nb_rows + nb_cols" << endl;
				exit(1);
			}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (FALSE) {
				cout << "packings_global::select_packings "
						"before init_packing_from_spread_table" << endl;
			}
			OiP->init_packing_from_spread_table(P, the_set_in,
				Spread_table, nb_spreads, spread_size,
				0 /*verbose_level*/);
			if (FALSE) {
				cout << "packings_global::merge_packings "
						"after init_packing_from_spread_table" << endl;
			}
			OiP->f_has_known_ago = TRUE;
			OiP->known_ago = ago;

			int *Incma_in;
			int *Incma_out;
			int nb_rows1, nb_cols1;
			int *partition;
			uchar *canonical_form;
			int canonical_form_len;


			if (FALSE) {
				cout << "packings_global::select_packings "
						"before encode_incma" << endl;
			}
			OiP->encode_incma(Incma_in, nb_rows1, nb_cols1,
					partition, 0 /*verbose_level - 1*/);
			if (FALSE) {
				cout << "packings_global::select_packings "
						"after encode_incma" << endl;
			}
			if (nb_rows1 != nb_rows) {
				cout << "packings_global::select_packings "
						"nb_rows1 != nb_rows" << endl;
				exit(1);
			}
			if (nb_cols1 != nb_cols) {
				cout << "packings_global::select_packings "
						"nb_cols1 != nb_cols" << endl;
				exit(1);
			}

			OiP->input_fname = S->get_string(g + 1, original_file_idx);
			OiP->input_idx = S->get_int(g + 1, input_idx_idx);

			text = S->get_string(g + 1, input_set_idx);

			OiP->set_as_string.assign(text);

			int i, j, ii, jj, a;
			int L = nb_rows * nb_cols;

			Incma_out = NEW_int(L);
			for (i = 0; i < nb_rows; i++) {
				ii = canonical_labeling[i];
				for (j = 0; j < nb_cols; j++) {
					jj = canonical_labeling[nb_rows + j] - nb_rows;
					//cout << "i=" << i << " j=" << j << " ii=" << ii
					//<< " jj=" << jj << endl;
					Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
					}
				}
			if (FALSE) {
				cout << "packings_global::select_packings "
						"before bitvector_allocate_and_coded_length" << endl;
			}
			canonical_form = bitvector_allocate_and_coded_length(
					L, canonical_form_len);
			for (i = 0; i < nb_rows; i++) {
				for (j = 0; j < nb_cols; j++) {
					if (Incma_out[i * nb_cols + j]) {
						a = i * nb_cols + j;
						bitvector_set_bit(canonical_form, a);
						}
					}
				}

			if (f_first) {
				if (f_v) {
					cout << "packings_global::select_packings "
							"before CB->init" << endl;
				}
				CB->init(table_length, canonical_form_len, verbose_level);
				f_first = FALSE;
			}


			if (f_v) {
				cout << "packings_global::select_packings "
						"before CB->add" << endl;
			}

			int idx;
			int f_found;

			CB->search_and_add_if_new(canonical_form, OiP, f_found, idx, 0 /*verbose_level*/);
			if (f_found) {
				cout << "reject" << endl;
				nb_reject++;
			}
			if (f_v) {
				cout << "packings_global::select_packings "
						"CB->add returns f_found = " << f_found
						<< " nb iso = " << CB->nb_types
						<< " nb_reject=" << nb_reject
						<< " nb_accept=" << nb_accept
						<< " CB->n=" << CB->n
						<< " CB->nb_types=" << CB->nb_types
						<< endl;
			}


			//int idx;

			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init(OiP, ago, nb_rows, nb_cols,
					canonical_labeling, 0 /*verbose_level*/);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;

			FREE_int(Incma_in);
			FREE_int(Incma_out);
			FREE_int(partition);
			//FREE_int(canonical_labeling);
			//FREE_uchar(canonical_form);
		} // if (f_accept)



		FREE_lint(the_set_in);

	} // next g




	if (f_v) {
		cout << "packings_global::select_packings done, "
				"we found " << CB->nb_types << " isomorphism types "
				"of packings. nb_accept = " << nb_accept
				<< " CB->n = " << CB->n
				<< " CB->nb_types = " << CB->nb_types
				<< endl;
		}


	//FREE_OBJECT(CB);
	FREE_lint(Spread_table);

	if (f_v) {
		cout << "packings_global::select_packings done" << endl;
	}
#endif
}



void packings_global::select_packings_self_dual(
		std::string &fname,
		std::string &file_of_spreads_original,
		int f_split, int split_r, int split_m,
		spread_tables *Spread_tables,
		data_structures::classify_bitvectors *&CB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packings_global::select_packings_self_dual" << endl;
	}


#if 0
	int nb_accept = 0;
	file_io Fio;

	CB = NEW_OBJECT(classify_bitvectors);



	long int *Spread_table_original;
	int nb_spreads;
	int spread_size;
	int packing_size;
	int a, b;

	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"Reading spread table from file "
				<< file_of_spreads_original << endl;
	}
	Fio.lint_matrix_read_csv(file_of_spreads_original,
			Spread_table_original, nb_spreads, spread_size,
			0 /* verbose_level */);
	if (nb_spreads != Spread_tables->nb_spreads) {
		cout << "packings_global::select_packings_self_dual "
				"nb_spreads != Spread_tables->nb_spreads" << endl;
		exit(1);
	}
	if (spread_size != Spread_tables->spread_size) {
		cout << "packings_global::select_packings_self_dual "
				"spread_size != Spread_tables->spread_size" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Reading spread table from file "
				<< file_of_spreads_original << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
	}



	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s" << endl;
	}

	int *s2l, *l2s;
	int i, idx;
	long int *set;
	long int extra_data[1];
	sorting Sorting;

	extra_data[0] = spread_size;

	set = NEW_lint(spread_size);
	s2l = NEW_int(nb_spreads);
	l2s = NEW_int(nb_spreads);
	for (i = 0; i < nb_spreads; i++) {
		Orbiter->Lint_vec.copy(Spread_table_original +
				i * spread_size, set, spread_size);
		Sorting.lint_vec_heapsort(set, spread_size);
		if (!Sorting.search_general(Spread_tables->spread_table,
				nb_spreads, (int *) set, idx,
				table_of_sets_compare_func,
				extra_data, 0 /*verbose_level*/)) {
			cout << "packings_global::select_packings_self_dual "
					"cannot find spread " << i << " = ";
			Orbiter->Lint_vec.print(cout, set, spread_size);
			cout << endl;
			exit(1);
		}
		s2l[i] = idx;
		l2s[idx] = i;
	}
	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s done" << endl;
	}

	int g, table_length, nb_reject = 0;


	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"file " << fname << endl;
	}

	spreadsheet *S;

	S = NEW_OBJECT(spreadsheet);

	S->read_spreadsheet(fname, 0 /*verbose_level*/);
	if (FALSE /*f_v3*/) {
		S->print_table(cout, FALSE);
		}

	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"read file " << fname << endl;
	}


	int ago_idx, original_file_idx, input_idx_idx, input_set_idx;
	int nb_rows_idx, nb_cols_idx, canonical_form_idx;

	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"finding column indices" << endl;
	}

	ago_idx = S->find_by_column("ago");
	original_file_idx = S->find_by_column("original_file");
	input_idx_idx = S->find_by_column("input_idx");
	input_set_idx = S->find_by_column("input_set");
	nb_rows_idx = S->find_by_column("nb_rows");
	nb_cols_idx = S->find_by_column("nb_cols");
	canonical_form_idx = S->find_by_column("canonical_form");

	table_length = S->nb_rows - 1;

	//rep,ago,original_file,input_idx,
	//input_set,nb_rows,nb_cols,canonical_form
	int f_first = TRUE;


	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"first pass, table_length=" << table_length << endl;
	}

	// first pass: build up the database:

	for (g = 0; g < table_length; g++) {

		int ago;
		char *text;
		long int *the_set_in;
		int set_size_in;
		long int *canonical_labeling;
		int canonical_labeling_sz;
		int nb_rows, nb_cols;
		object_in_projective_space *OiP;
		int f_accept;

		ago = S->get_int(g + 1, ago_idx);
		nb_rows = S->get_int(g + 1, nb_rows_idx);
		nb_cols = S->get_int(g + 1, nb_cols_idx);

		text = S->get_string(g + 1, input_set_idx);
		Orbiter->Lint_vec.scan(text, the_set_in, set_size_in);

		packing_size = set_size_in;

		if (f_v && (g % 1000) == 0) {
			cout << "File " << fname
					<< ", input set " << g << " / "
					<< table_length << endl;
			//int_vec_print(cout, the_set_in, set_size_in);
			//cout << endl;
			}


		f_accept = TRUE;



		if (f_accept) {

			nb_accept++;


			if (FALSE) {
				cout << "canonical_form_idx=" << canonical_form_idx << endl;
			}
			text = S->get_string(g + 1, canonical_form_idx);
			if (FALSE) {
				cout << "text=" << text << endl;
			}
			Orbiter->Lint_vec.scan(text, canonical_labeling, canonical_labeling_sz);
			if (FALSE) {
				cout << "File " << fname
						<< ", input set " << g << " / "
						<< table_length << " canonical_labeling = ";
				Orbiter->Lint_vec.print(cout,
						canonical_labeling, canonical_labeling_sz);
				cout << endl;
				}

			if (canonical_labeling_sz != nb_rows + nb_cols) {
				cout << "packings_global::select_packings_self_dual "
						"canonical_labeling_sz != nb_rows + nb_cols" << endl;
				exit(1);
			}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (FALSE) {
				cout << "packings_global::select_packings_self_dual "
						"before init_packing_from_spread_table" << endl;
			}
			OiP->init_packing_from_spread_table(P, the_set_in,
					Spread_table_original, nb_spreads, spread_size,
				0 /*verbose_level*/);
			if (FALSE) {
				cout << "packings_global::select_packings_self_dual "
						"after init_packing_from_spread_table" << endl;
			}
			OiP->f_has_known_ago = TRUE;
			OiP->known_ago = ago;

			int *Incma_in;
			int *Incma_out;
			int nb_rows1, nb_cols1;
			int *partition;
			uchar *canonical_form;
			int canonical_form_len;


			if (FALSE) {
				cout << "packings_global::select_packings_self_dual "
						"before encode_incma" << endl;
			}
			OiP->encode_incma(Incma_in, nb_rows1, nb_cols1,
					partition, 0 /*verbose_level - 1*/);
			if (FALSE) {
				cout << "packings_global::select_packings_self_dual "
						"after encode_incma" << endl;
			}
			if (nb_rows1 != nb_rows) {
				cout << "packings_global::select_packings_self_dual "
						"nb_rows1 != nb_rows" << endl;
				exit(1);
			}
			if (nb_cols1 != nb_cols) {
				cout << "packings_global::select_packings_self_dual "
						"nb_cols1 != nb_cols" << endl;
				exit(1);
			}

			OiP->input_fname = S->get_string(g + 1, original_file_idx);
			OiP->input_idx = S->get_int(g + 1, input_idx_idx);

			text = S->get_string(g + 1, input_set_idx);

			OiP->set_as_string.assign(text);

			int i, j, ii, jj, a;
			int L = nb_rows * nb_cols;

			Incma_out = NEW_int(L);
			for (i = 0; i < nb_rows; i++) {
				ii = canonical_labeling[i];
				for (j = 0; j < nb_cols; j++) {
					jj = canonical_labeling[nb_rows + j] - nb_rows;
					//cout << "i=" << i << " j=" << j << " ii=" << ii
					//<< " jj=" << jj << endl;
					Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
					}
				}
			if (FALSE) {
				cout << "packings_global::select_packings_self_dual "
						"before bitvector_allocate_and_coded_length" << endl;
			}
			canonical_form = bitvector_allocate_and_coded_length(
					L, canonical_form_len);
			for (i = 0; i < nb_rows; i++) {
				for (j = 0; j < nb_cols; j++) {
					if (Incma_out[i * nb_cols + j]) {
						a = i * nb_cols + j;
						bitvector_set_bit(canonical_form, a);
						}
					}
				}

			if (f_first) {
				if (f_v) {
					cout << "packings_global::select_packings_self_dual "
							"before CB->init" << endl;
				}
				CB->init(table_length, canonical_form_len, verbose_level);
				f_first = FALSE;
			}


			if (FALSE) {
				cout << "packings_global::select_packings_self_dual "
						"before CB->add" << endl;
			}

			int idx;
			int f_found;

			CB->search_and_add_if_new(canonical_form, OiP, f_found, idx, 0 /*verbose_level*/);
			if (f_found) {
				cout << "reject" << endl;
				nb_reject++;
			}
			if (FALSE) {
				cout << "packings_global::select_packings_self_dual "
						"CB->add f_found = " << f_found
						<< " nb iso = " << CB->nb_types
						<< " nb_reject=" << nb_reject
						<< " nb_accept=" << nb_accept
						<< " CB->n=" << CB->n
						<< " CB->nb_types=" << CB->nb_types
						<< endl;
			}


			//int idx;

			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init(OiP, ago, nb_rows, nb_cols,
					canonical_labeling, 0 /*verbose_level*/);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;

			FREE_int(Incma_in);
			FREE_int(Incma_out);
			FREE_int(partition);
			//FREE_int(canonical_labeling);
			//FREE_uchar(canonical_form);
		} // if (f_accept)



		FREE_lint(the_set_in);

	} // next g




	if (f_v) {
		cout << "packings_global::select_packings_self_dual done, "
				"we found " << CB->nb_types << " isomorphism types "
				"of packings. nb_accept = " << nb_accept
				<< " CB->n = " << CB->n
				<< " CB->nb_types = " << CB->nb_types
				<< endl;
		}


	// second pass:

	int nb_self_dual = 0;
	int g1 = 0;
	int *self_dual_cases;
	int nb_self_dual_cases = 0;


	self_dual_cases = NEW_int(table_length);


	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"second pass, table_length="
				<< table_length << endl;
	}


	for (g = 0; g < table_length; g++) {

		int ago;
		char *text;
		int *the_set_in;
		int set_size_in;
		int *canonical_labeling1;
		int *canonical_labeling2;
		//int canonical_labeling_sz;
		int nb_rows, nb_cols;
		object_in_projective_space *OiP1;
		object_in_projective_space *OiP2;
		long int *set1;
		long int *set2;

		ago = S->get_int(g + 1, ago_idx);
		nb_rows = S->get_int(g + 1, nb_rows_idx);
		nb_cols = S->get_int(g + 1, nb_cols_idx);

		text = S->get_string(g + 1, input_set_idx);
		Orbiter->Int_vec.scan(text, the_set_in, set_size_in);

		packing_size = set_size_in;


		if (f_split) {
			if ((g % split_m) != split_r) {
				continue;
			}
		}
		g1++;
		if (f_v && (g1 % 100) == 0) {
			cout << "File " << fname
					<< ", case " << g1 << " input set " << g << " / "
					<< table_length
					<< " nb_self_dual=" << nb_self_dual << endl;
			//int_vec_print(cout, the_set_in, set_size_in);
			//cout << endl;
			}


		set1 = NEW_lint(packing_size);
		set2 = NEW_lint(packing_size);

		for (i = 0; i < packing_size; i++) {
			a = the_set_in[i];
			b = s2l[a];
			set1[i] = b;
		}
		Sorting.lint_vec_heapsort(set1, packing_size);
		for (i = 0; i < packing_size; i++) {
			a = set1[i];
			b = Spread_tables->dual_spread_idx[a];
			set2[i] = l2s[b];
		}
		for (i = 0; i < packing_size; i++) {
			a = set1[i];
			b = l2s[a];
			set1[i] = b;
		}
		Sorting.lint_vec_heapsort(set1, packing_size);
		Sorting.lint_vec_heapsort(set2, packing_size);

#if 0
		cout << "set1: ";
		int_vec_print(cout, set1, packing_size);
		cout << endl;
		cout << "set2: ";
		int_vec_print(cout, set2, packing_size);
		cout << endl;
#endif




		OiP1 = NEW_OBJECT(object_in_projective_space);
		OiP2 = NEW_OBJECT(object_in_projective_space);

		if (FALSE) {
			cout << "packings_global::select_packings_self_dual "
					"before init_packing_from_spread_table" << endl;
		}
		OiP1->init_packing_from_spread_table(P, set1,
				Spread_table_original, nb_spreads, spread_size,
				0 /*verbose_level*/);
		OiP2->init_packing_from_spread_table(P, set2,
				Spread_table_original, nb_spreads, spread_size,
				0 /*verbose_level*/);
		if (FALSE) {
			cout << "packings_global::select_packings_self_dual "
					"after init_packing_from_spread_table" << endl;
		}
		OiP1->f_has_known_ago = TRUE;
		OiP1->known_ago = ago;



		uchar *canonical_form1;
		uchar *canonical_form2;
		int canonical_form_len;



		int *Incma_in1;
		int *Incma_out1;
		int *Incma_in2;
		int *Incma_out2;
		int nb_rows1, nb_cols1;
		int *partition;
		//uchar *canonical_form1;
		//uchar *canonical_form2;
		//int canonical_form_len;


		if (FALSE) {
			cout << "packings_global::select_packings_self_dual "
					"before encode_incma" << endl;
		}
		OiP1->encode_incma(Incma_in1, nb_rows1, nb_cols1,
				partition, 0 /*verbose_level - 1*/);
		OiP2->encode_incma(Incma_in2, nb_rows1, nb_cols1,
				partition, 0 /*verbose_level - 1*/);
		if (FALSE) {
			cout << "packings_global::select_packings_self_dual "
					"after encode_incma" << endl;
		}
		if (nb_rows1 != nb_rows) {
			cout << "packings_global::select_packings_self_dual "
					"nb_rows1 != nb_rows" << endl;
			exit(1);
		}
		if (nb_cols1 != nb_cols) {
			cout << "packings_global::select_packings_self_dual "
					"nb_cols1 != nb_cols" << endl;
			exit(1);
		}


		if (FALSE) {
			cout << "packings_global::select_packings_self_dual "
					"before PA->set_stabilizer_of_object" << endl;
			}


		canonical_labeling1 = NEW_int(nb_rows * nb_cols);
		canonical_labeling2 = NEW_int(nb_rows * nb_cols);

		canonical_labeling(
				OiP1,
				canonical_labeling1,
				0 /*verbose_level - 2*/);
		canonical_labeling(
				OiP2,
				canonical_labeling2,
				0 /*verbose_level - 2*/);


		OiP1->input_fname = S->get_string(g + 1, original_file_idx);
		OiP1->input_idx = S->get_int(g + 1, input_idx_idx);
		OiP2->input_fname = S->get_string(g + 1, original_file_idx);
		OiP2->input_idx = S->get_int(g + 1, input_idx_idx);

		text = S->get_string(g + 1, input_set_idx);

		OiP1->set_as_string.assign(text);

		OiP2->set_as_string.assign(text);

		int i, j, ii, jj, a, ret;
		int L = nb_rows * nb_cols;

		Incma_out1 = NEW_int(L);
		Incma_out2 = NEW_int(L);
		for (i = 0; i < nb_rows; i++) {
			ii = canonical_labeling1[i];
			for (j = 0; j < nb_cols; j++) {
				jj = canonical_labeling1[nb_rows + j] - nb_rows;
				//cout << "i=" << i << " j=" << j << " ii=" << ii
				//<< " jj=" << jj << endl;
				Incma_out1[i * nb_cols + j] = Incma_in1[ii * nb_cols + jj];
				}
			}
		for (i = 0; i < nb_rows; i++) {
			ii = canonical_labeling2[i];
			for (j = 0; j < nb_cols; j++) {
				jj = canonical_labeling2[nb_rows + j] - nb_rows;
				//cout << "i=" << i << " j=" << j << " ii=" << ii
				//<< " jj=" << jj << endl;
				Incma_out2[i * nb_cols + j] = Incma_in2[ii * nb_cols + jj];
				}
			}
		if (FALSE) {
			cout << "packings_global::select_packings_self_dual "
					"before bitvector_allocate_and_coded_length" << endl;
		}
		canonical_form1 = bitvector_allocate_and_coded_length(
				L, canonical_form_len);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (Incma_out1[i * nb_cols + j]) {
					a = i * nb_cols + j;
					bitvector_set_bit(canonical_form1, a);
					}
				}
			}
		canonical_form2 = bitvector_allocate_and_coded_length(
				L, canonical_form_len);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (Incma_out2[i * nb_cols + j]) {
					a = i * nb_cols + j;
					bitvector_set_bit(canonical_form2, a);
					}
				}
			}


		if (FALSE) {
			cout << "packings_global::select_packings_self_dual "
					"before CB->search" << endl;
		}

		int idx1, idx2;

		ret = CB->search(canonical_form1, idx1, 0 /*verbose_level*/);

		if (ret == FALSE) {
			cout << "cannot find the dual packing, "
					"something is wrong" << endl;
			ret = CB->search(canonical_form1, idx1, 5 /* verbose_level*/);
#if 0
			cout << "CB:" << endl;
			CB->print_table();
			cout << "canonical form1: ";
			for (int j = 0; j < canonical_form_len; j++) {
				cout << (int) canonical_form1[j];
				if (j < canonical_form_len - 1) {
					cout << ", ";
					}
				}
			cout << endl;
#endif
			exit(1);
		}
		if (FALSE) {
			cout << "packings_global::select_packings_self_dual "
					"CB->search returns idx1=" << idx1 << endl;
		}
		ret = CB->search(canonical_form2, idx2, 0 /*verbose_level*/);

		if (ret == FALSE) {
			cout << "cannot find the dual packing, "
					"something is wrong" << endl;
			ret = CB->search(canonical_form2, idx2, 5 /* verbose_level*/);
#if 0
			cout << "CB:" << endl;
			CB->print_table();
			cout << "canonical form2: ";
			for (int j = 0; j < canonical_form_len; j++) {
				cout << (int) canonical_form2[j];
				if (j < canonical_form_len - 1) {
					cout << ", ";
					}
				}
#endif
			exit(1);
		}
		if (FALSE) {
			cout << "packings_global::select_packings_self_dual "
					"CB->search returns idx2=" << idx2 << endl;
		}

		FREE_int(Incma_in1);
		FREE_int(Incma_out1);
		FREE_int(Incma_in2);
		FREE_int(Incma_out2);
		FREE_int(partition);
		FREE_int(canonical_labeling1);
		FREE_int(canonical_labeling2);
		FREE_uchar(canonical_form1);
		FREE_uchar(canonical_form2);

		FREE_lint(set1);
		FREE_lint(set2);

		if (idx1 == idx2) {
			cout << "self-dual" << endl;
			nb_self_dual++;
			self_dual_cases[nb_self_dual_cases++] = g;
		}

		FREE_int(the_set_in);

	} // next g

	string fname_base;
	string fname_self_dual;
	char str[1000];
	string_tools String;

	fname_base.assign(fname);
	String.chop_off_extension(fname_base);
	fname_self_dual.assign(fname);
	String.chop_off_extension(fname_self_dual);
	if (f_split) {
		sprintf(str, "_self_dual_r%d_m%d.csv", split_r, split_m);
	}
	else {
		sprintf(str, "_self_dual.csv");
	}
	fname_self_dual.append(str);
	cout << "saving self_dual_cases to file " << fname_self_dual << endl;
	Fio.int_vec_write_csv(self_dual_cases, nb_self_dual_cases,
			fname_self_dual, "self_dual_idx");
	cout << "written file " << fname_self_dual
			<< " of size " << Fio.file_size(fname_self_dual) << endl;



	//FREE_OBJECT(CB);
	FREE_lint(Spread_table_original);

	if (f_v) {
		cout << "packings_global::select_packings_self_dual "
				"done, nb_self_dual = " << nb_self_dual << endl;
	}
#endif

}



}}}


