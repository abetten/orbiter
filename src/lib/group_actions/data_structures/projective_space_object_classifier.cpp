/*
 * projective_space_object_classifier.cpp
 *
 *  Created on: Sep 13, 2020
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {


projective_space_object_classifier::projective_space_object_classifier()
{

	Descr = NULL;

	PA = NULL;

	nb_objects_to_test = 0;

	CB = NULL;

}

projective_space_object_classifier::~projective_space_object_classifier()
{
}

void projective_space_object_classifier::do_the_work(
		projective_space_object_classifier_description *Descr,
		projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_object_classifier::do_the_work" << endl;
	}


	projective_space_object_classifier::PA = PA;
	projective_space_object_classifier::Descr = Descr;

	//int i;

	CB = NEW_OBJECT(classify_bitvectors);





	if (f_v) {
		cout << "projective_space_object_classifier::do_the_work "
				"before PA->classify_objects_using_nauty" << endl;
	}

	classify_objects_using_nauty(verbose_level - 1);

	if (f_v) {
		cout << "projective_space_object_classifier::do_the_work "
			"after PA->classify_objects_using_nauty" << endl;
	}


	cout << "projective_space_object_classifier::do_the_work We found "
			<< CB->nb_types << " types" << endl;


#if 0

	compute_and_print_ago_distribution_with_classes(cout,
			CB, verbose_level);


	cout << "projective_space_object_classifier::do_the_work "
			"In the ordering of canonical forms, they are" << endl;
	CB->print_reps();
	cout << "We found " << CB->nb_types << " types:" << endl;
	for (i = 0; i < CB->nb_types; i++) {

		object_in_projective_space_with_action *OiPA;
		object_in_projective_space *OiP;

		cout << i << " / " << CB->nb_types << " is "
			<< CB->Type_rep[i] << " : " << CB->Type_mult[i] << " : ";
		OiPA = (object_in_projective_space_with_action *) CB->Type_extra_data[i];
		OiP = OiPA->OiP;
		if (OiP->type != t_PAC) {
			OiP->print(cout);
		}

#if 0
		for (j = 0; j < rep_len; j++) {
			cout << (int) Type_data[i][j];
			if (j < rep_len - 1) {
				cout << ", ";
				}
			}
#endif
		cout << endl;
	}

#endif




	if (Descr->f_save_classification) {
		cout << "projective_space_object_classifier::do_the_work "
				"Saving the classification with save_prefix " << Descr->save_prefix << endl;
		save(Descr->save_prefix, verbose_level);
		CB->save(Descr->save_prefix,
			OiPA_encode, OiPA_group_order,
			NULL /* void *global_data */,
			verbose_level);

#if 0
		void save(const char *prefix,
			void (*encode_function)(void *extra_data,
				int *&encoding, int &encoding_sz, void *global_data),
			void (*get_group_order_or_NULL)(void *extra_data,
				longinteger_object &go, void *global_data),
			void *global_data,
			int verbose_level);
#endif
	}




	if (Descr->f_report) {

		cout << "projective_space_object_classifier::do_the_work Producing a latex report:" << endl;


		if (Descr->f_classification_prefix == FALSE) {
			cout << "please use option -classification_prefix <prefix> to set the "
					"prefix for the output file" << endl;
			exit(1);
			}

		string fname;

		fname.assign(Descr->classification_prefix);
		fname.append("_classification.tex");


		latex_report(fname,
				Descr->report_prefix,
				Descr->fixed_structure_order_list_sz,
				Descr->fixed_structure_order_list,
				Descr->max_TDO_depth,
				verbose_level);


	}// f_report

	if (f_v) {
		cout << "projective_space_object_classifier::do_the_work done" << endl;
	}


}


void projective_space_object_classifier::classify_objects_using_nauty(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int input_idx;
	int t0, t1, dt;
	file_io Fio;
	os_interface Os;

	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty" << endl;
	}


	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty "
				"before count_number_of_objects_to_test" << endl;
	}
	nb_objects_to_test = Descr->Data->count_number_of_objects_to_test(
		verbose_level - 1);

	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty "
				"nb_objects_to_test = " << nb_objects_to_test << endl;
	}


	t0 = Os.os_ticks();

	vector<long int> Cumulative_Ago;
	vector<vector<int> > Cumulative_canonical_labeling;
	vector<vector<int> > Cumulative_data;
	std::vector<std::vector<std::pair<int, int> > > Fibration;


	for (input_idx = 0; input_idx < Descr->Data->nb_inputs; input_idx++) {
		if (f_v) {
			cout << "projective_space_object_classifier::classify_objects_using_nauty "
					"input " << input_idx << " / " << Descr->Data->nb_inputs
					<< " is:" << endl;
		}

		if (Descr->Data->input_type[input_idx] == INPUT_TYPE_SET_OF_POINTS) {

			process_set_of_points(
					Descr->Data->input_string[input_idx],
					verbose_level);



		}
		else if (Descr->Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINT_SET) {

			process_set_of_points_from_file(
					Descr->Data->input_string[input_idx],
					verbose_level);

			}
		else if (Descr->Data->input_type[input_idx] == INPUT_TYPE_SET_OF_LINES) {

			process_set_of_lines_from_file(
					Descr->Data->input_string[input_idx],
					verbose_level);

			}
		else if (Descr->Data->input_type[input_idx] == INPUT_TYPE_SET_OF_PACKING) {


			process_set_of_packing(
					Descr->Data->input_string[input_idx],
					verbose_level);

		}
		else if (Descr->Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINTS ||
				Descr->Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_LINES ||
				Descr->Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_PACKINGS ||
				Descr->Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {

			if (f_v) {
				cout << "projective_space_object_classifier::classify_objects_using_nauty "
						"before process_multiple_objects_from_file" << endl;
			}
			process_multiple_objects_from_file(
					Descr->Data->input_type[input_idx] /* file_type */,
					input_idx,
					Descr->Data->input_string[input_idx],
					Descr->Data->input_string2[input_idx],
					Cumulative_data,
					Cumulative_Ago,
					Cumulative_canonical_labeling,
					Fibration,
					verbose_level);
			if (f_v) {
				cout << "projective_space_object_classifier::classify_objects_using_nauty "
						"after process_multiple_objects_from_file" << endl;
			}

		}
		else {
			cout << "projective_space_object_classifier::classify_objects_using_nauty "
					"unknown input type" << endl;
			exit(1);
		}
	}

	t1 = Os.os_ticks();
	dt = t1 - t0;

	cout << "projective_space_object_classifier::classify_objects_using_nauty Time ";
	Os.time_check_delta(cout, dt);
	cout << endl;


	if (Descr->f_save_cumulative_canonical_labeling) {

		if (f_v) {
			cout << "projective_space_object_classifier::classify_objects_using_nauty f_save_cumulative_canonical_labeling" << endl;
		}
		string canonical_labeling_fname;
		int canonical_labeling_len;
		int u, v;
		long int *M;
		file_io Fio;

		if (Cumulative_canonical_labeling.size()) {
			canonical_labeling_len = Cumulative_canonical_labeling[0].size();
		}
		else {
			canonical_labeling_len = 0;
		}

		canonical_labeling_fname.assign(Descr->cumulative_canonical_labeling_fname);

		M = NEW_lint(Cumulative_canonical_labeling.size() * canonical_labeling_len);
		for (u = 0; u < Cumulative_canonical_labeling.size(); u++) {
			for (v = 0; v < canonical_labeling_len; v++) {
				M[u * canonical_labeling_len + v] = Cumulative_canonical_labeling[u][v];
			}
		}
		Fio.lint_matrix_write_csv(canonical_labeling_fname,
				M, Cumulative_canonical_labeling.size(), canonical_labeling_len);

		cout << "Written file " << canonical_labeling_fname << " of size " << Fio.file_size(canonical_labeling_fname) << endl;
		FREE_lint(M);
	}

	if (Descr->f_save_cumulative_ago) {


		if (f_v) {
			cout << "projective_space_object_classifier::classify_objects_using_nauty f_save_cumulative_ago" << endl;
		}

		string ago_fname;
		int u;
		long int *M;
		file_io Fio;

		ago_fname.assign(Descr->cumulative_ago_fname);

		M = NEW_lint(Cumulative_Ago.size());
		for (u = 0; u < Cumulative_Ago.size(); u++) {
			M[u] = Cumulative_Ago[u];
		}
		Fio.lint_vec_write_csv(M, Cumulative_Ago.size(),
				ago_fname, "Ago");

		tally T;

		T.init_lint(M, Cumulative_Ago.size(), FALSE, 0);
		cout << "Written file " << ago_fname << " of size " << Fio.file_size(ago_fname) << endl;

		cout << "Ago distribution: ";
		T.print(TRUE);
		cout << endl;

		string ago_fname1;

		ago_fname1.assign(ago_fname);
		replace_extension_with(ago_fname1, "_class_of_");
		T.save_classes_individually(ago_fname1);

		FREE_lint(M);
	}


	if (Descr->f_save_cumulative_data) {


		if (f_v) {
			cout << "projective_space_object_classifier::classify_objects_using_nauty f_save_cumulative_data" << endl;
		}

		string data_fname;
		int data_len;
		int u, v;
		long int *M;
		file_io Fio;

		if (Cumulative_data.size()) {
			data_len = Cumulative_data[0].size();
		}
		else {
			data_len = 0;
		}

		data_fname.assign(Descr->cumulative_data_fname);

		M = NEW_lint(Cumulative_data.size() * data_len);
		for (u = 0; u < Cumulative_data.size(); u++) {
			for (v = 0; v < data_len; v++) {
				M[u * data_len + v] = Cumulative_data[u][v];
			}
		}
		Fio.lint_matrix_write_csv(data_fname,
				M, Cumulative_data.size(), data_len);

		cout << "Written file " << data_fname << " of size " << Fio.file_size(data_fname) << endl;
		FREE_lint(M);
	}


	if (Descr->f_save_fibration) {

		if (f_v) {
			cout << "projective_space_object_classifier::classify_objects_using_nauty f_save_fibration" << endl;
		}
		string data_fname1;
		string data_fname2;
		set_of_sets *File_idx;
		set_of_sets *Obj_idx;
		file_io Fio;
		int nb_sets;
		int *Sz;
		int i, j, l, a, b;

		nb_sets = Fibration.size();
		Sz = NEW_int(nb_sets);
		for (i = 0; i < nb_sets; i++) {
			Sz[i] = Fibration[i].size();
		}

		File_idx = NEW_OBJECT(set_of_sets);
		Obj_idx = NEW_OBJECT(set_of_sets);

		File_idx->init_basic_with_Sz_in_int(INT_MAX /* underlying_set_size */,
					nb_sets, Sz, verbose_level);
		Obj_idx->init_basic_with_Sz_in_int(INT_MAX /* underlying_set_size */,
					nb_sets, Sz, verbose_level);
		for (i = 0; i < nb_sets; i++) {
			l = Fibration[i].size();
			for (j = 0; j < l; j++) {
				a = Fibration[i][j].first;
				b = Fibration[i][j].second;
				File_idx->Sets[i][j] = a;
				Obj_idx->Sets[i][j] = b;
			}
		}


		data_fname1.assign(Descr->fibration_fname);
		replace_extension_with(data_fname1, "1.csv");
		data_fname2.assign(Descr->fibration_fname);
		replace_extension_with(data_fname2, "2.csv");

		if (f_v) {
			cout << "projective_space_object_classifier::classify_objects_using_nauty before File_idx->save_csv" << endl;
		}
		File_idx->save_csv(data_fname1, TRUE, verbose_level);
		if (f_v) {
			cout << "projective_space_object_classifier::classify_objects_using_nauty before Obj_idx->save_csv" << endl;
		}
		Obj_idx->save_csv(data_fname2, TRUE, verbose_level);
		if (f_v) {
			cout << "projective_space_object_classifier::classify_objects_using_nauty after Obj_idx->save_csv" << endl;
		}


		cout << "Written file " << data_fname1 << " of size " << Fio.file_size(data_fname1) << endl;
		cout << "Written file " << data_fname2 << " of size " << Fio.file_size(data_fname2) << endl;
		FREE_int(Sz);
		FREE_OBJECT(File_idx);
		FREE_OBJECT(Obj_idx);
	}



#if 0
	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty "
				"before compute_and_print_ago_distribution" << endl;
	}

	compute_and_print_ago_distribution(cout, CB, verbose_level);

	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty "
				"after compute_and_print_ago_distribution" << endl;
	}
#endif

	t1 = Os.os_ticks();
	dt = t1 - t0;

	cout << "projective_space_object_classifier::classify_objects_using_nauty Time ";
	Os.time_check_delta(cout, dt);



	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty "
				"before CB->finalize" << endl;
	}

	CB->finalize(verbose_level); // computes C_type_of and perm


	t1 = Os.os_ticks();
	dt = t1 - t0;

	cout << "projective_space_object_classifier::classify_objects_using_nauty Time ";
	Os.time_check_delta(cout, dt);


	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty done" << endl;
		}
}

void projective_space_object_classifier::process_multiple_objects_from_file(
		int file_type, int file_idx,
		std::string &input_data,
		std::string &input_data2,
		std::vector<std::vector<int> > &Cumulative_data,
		std::vector<long int> &Cumulative_Ago,
		std::vector<std::vector<int> > &Cumulative_canonical_labeling,
		std::vector<std::vector<std::pair<int, int> > > &Fibration,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "projective_space_object_classifier::process_multiple_objects_from_file" << endl;
	}
	int t0, t1, dt;
	file_io Fio;
	os_interface Os;

	cout << "projective_space_object_classifier::process_multiple_objects_from_file "
			"input from file " << input_data
		<< ":" << endl;

	t0 = Os.os_ticks();

	set_of_sets *SoS;

	SoS = NEW_OBJECT(set_of_sets);

	cout << "projective_space_object_classifier::process_multiple_objects_from_file "
			"Reading the file " << input_data << endl;
	SoS->init_from_file(
			PA->P->N_points /* underlying_set_size */,
			input_data, verbose_level);
	cout << "Read the file " << input_data << endl;

	int h;


	// for use if INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE
	long int *Spread_table;
	int nb_spreads;
	int spread_size;

	if (file_type ==
			INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
		cout << "projective_space_object_classifier::process_multiple_objects_from_file "
				"Reading spread table from file "
			<< input_data2 << endl;
		Fio.lint_matrix_read_csv(input_data2,
				Spread_table, nb_spreads, spread_size,
				0 /* verbose_level */);
		cout << "Reading spread table from file "
				<< input_data2 << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
		}

	cout << "projective_space_object_classifier::process_multiple_objects_from_file "
			"processing " << SoS->nb_sets << " objects" << endl;

	{
		vector<long int> Ago;
		vector<vector<int>> The_canonical_labeling;
		int canonical_labeling_len;

		long int *Known_ago = NULL;
		long int *Known_canonical_labeling = NULL;
		int m = 0;

		if (Descr->f_load_canonical_labeling) {

			if (f_v) {
				cout << "projective_space_object_classifier::process_multiple_objects_from_file f_load_canonical_labeling" << endl;
			}
			string load_canonical_labeling_fname;

			load_canonical_labeling_fname.assign(input_data);
			replace_extension_with(load_canonical_labeling_fname, "_can_lab.csv");


			Fio.lint_matrix_read_csv(load_canonical_labeling_fname,
					Known_canonical_labeling, m, canonical_labeling_len, verbose_level);

			if (m != SoS->nb_sets) {
				cout << "after loading the canonical labeling, the number of lines in the file don't match" << endl;
				exit(1);
			}
		}

		if (Descr->f_load_ago) {
			int n;

			if (f_v) {
				cout << "projective_space_object_classifier::process_multiple_objects_from_file f_load_ago" << endl;
			}
			string load_ago_fname;

			load_ago_fname.assign(input_data);
			replace_extension_with(load_ago_fname, "_ago.csv");
			Fio.lint_matrix_read_csv(load_ago_fname,
					Known_ago, m, n, verbose_level);

			if (n != 1) {
				cout << "after loading the ago, n != 1" << endl;
				exit(1);
			}
			if (m != SoS->nb_sets) {
				cout << "after loading the ago, the number of lines in the file don't match" << endl;
				exit(1);
			}
		}

		for (h = 0; h < SoS->nb_sets; h++) {

			cout << "loop " << h << " / " << SoS->nb_sets << ":" << endl;

			long int *the_set_in;
			int set_size_in;
			object_in_projective_space *OiP;


			set_size_in = SoS->Set_size[h];
			the_set_in = SoS->Sets[h];

			if (f_vv || ((h % 1024) == 0)) {
				cout << "projective_space_object_classifier::process_multiple_objects_from_file "
						"The input set " << h << " / " << SoS->nb_sets
					<< " has size " << set_size_in << ":" << endl;
				}

			if (f_vvv) {
				cout << "projective_space_object_classifier::process_multiple_objects_from_file "
						"The input set is:" << endl;
				lint_vec_print(cout, the_set_in, set_size_in);
				cout << endl;
				}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (file_type == INPUT_TYPE_FILE_OF_POINTS) {
				OiP->init_point_set(PA->P, the_set_in, set_size_in,
						0 /* verbose_level*/);
				}
			else if (file_type == INPUT_TYPE_FILE_OF_LINES) {
				OiP->init_line_set(PA->P, the_set_in, set_size_in,
						0 /* verbose_level*/);
				}
			else if (file_type == INPUT_TYPE_FILE_OF_PACKINGS) {
				OiP->init_packing_from_set(PA->P,
						the_set_in, set_size_in, 0 /*verbose_level*/);
				}
			else if (file_type == INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
				OiP->init_packing_from_spread_table(PA->P, the_set_in,
					Spread_table, nb_spreads, spread_size,
					0 /*verbose_level*/);
				}
			else {
				cout << "projective_space_object_classifier::process_multiple_objects_from_file "
						"unknown type" << endl;
				exit(1);
				}

			stringstream sstr;
			//sstr << set_size_in;
			for (int k = 0; k < set_size_in; k++) {
				sstr << the_set_in[k];
				if (k < set_size_in - 1) {
					sstr << ",";
				}
			}
			string s = sstr.str();
			//cout << s << endl;
			//Ago_text[clique_no] = NEW_char(strlen(s.c_str()) + 1);
			//strcpy(Ago_text[clique_no], s.c_str());

			OiP->input_fname.assign(input_data);
			OiP->input_idx = h;

			OiP->set_as_string.assign(s);





			int nb_rows, nb_cols;
			long int *canonical_labeling;
			int ret, u, idx;

			OiP->encoding_size(nb_rows, nb_cols, 0 /*verbose_level*/);
			canonical_labeling = NEW_lint(nb_rows + nb_cols);

			strong_generators *SG;

			SG = NULL;

			if (Descr->f_load_canonical_labeling) {
				ret = process_object_with_known_canonical_labeling(
					OiP,
					Known_canonical_labeling + h * canonical_labeling_len, canonical_labeling_len,
					idx,
					verbose_level - 3);
				// we don't have strong generators !
				//SG = NULL;

				lint_vec_copy(Known_canonical_labeling + h * canonical_labeling_len,
						canonical_labeling, canonical_labeling_len);

				if (Descr->f_load_ago) {
					Ago.push_back(Known_ago[h]);
				}
				else {
					Ago.push_back(0);
				}
			}
			else {
				ret = process_object(OiP,
						SG,
						canonical_labeling, canonical_labeling_len,
						idx,
						verbose_level - 3);
				Ago.push_back(SG->group_order_as_lint());
				//FREE_OBJECT(SG);
			}


			vector<int> the_canonical_labeling;
			for (u = 0; u < canonical_labeling_len; u++) {
				the_canonical_labeling.push_back(canonical_labeling[u]);
			}

			The_canonical_labeling.push_back(the_canonical_labeling);

			if (ret) {

				FREE_OBJECT(OiP);
				FREE_OBJECT(SG);
				FREE_lint(canonical_labeling);
				Fibration[idx].push_back(make_pair(file_idx, h));
				}
			else {
				t1 = Os.os_ticks();
				dt = t1 - t0;

				cout << "Time ";
				Os.time_check_delta(cout, dt);
				cout << " --- New isomorphism type! input set " << h
						<< " / " << SoS->nb_sets << " The n e w number of "
						"isomorphism types is " << CB->nb_types << endl;



				cout << "initializing data of size " << set_size_in << endl;
				vector<int> data;
				for (u = 0; u < set_size_in; u++) {
					data.push_back(the_set_in[u]);
				}
				cout << "pushing Cumulative_data" << endl;
				Cumulative_data.push_back(data);

				cout << "pushing Cumulative_Ago" << endl;
				Cumulative_Ago.push_back(Ago[h]);

				cout << "pushing Cumulative_canonical_labeling" << endl;


				Cumulative_canonical_labeling.push_back(the_canonical_labeling);

				vector<pair<int, int> > v;
				v.push_back(make_pair(file_idx, h));
				Fibration.push_back(v);

				cout << "pushing Cumulative_canonical_labeling done" << endl;

				int idx;

				if (f_v) {
					cout << "storing OiPA" << endl;
				}

				object_in_projective_space_with_action *OiPA;

				OiPA = NEW_OBJECT(object_in_projective_space_with_action);

				//cout << "before OiPA->init" << endl;
				OiPA->init(OiP, Ago[h], SG, nb_rows, nb_cols,
						canonical_labeling, 0/*verbose_level*/);
				//cout << "after OiPA->init" << endl;
				idx = CB->type_of[CB->n - 1];
				CB->Type_extra_data[idx] = OiPA;


				}

			if (f_vv) {
				cout << "projective_space_object_classifier::process_multiple_objects_from_file "
						"after input set " << h << " / "
						<< SoS->nb_sets
						<< ", we have " << CB->nb_types
						<< " isomorphism types of objects" << endl;
				}

			}


		if (Descr->f_save_canonical_labeling) {
			if (f_v) {
				cout << "projective_space_object_classifier::process_multiple_objects_from_file "
						"f_save_canonical_labeling is TRUE" << endl;
			}
			string canonical_labeling_fname;
			int u, v;
			long int *M;
			file_io Fio;

			canonical_labeling_fname.assign(input_data);
			replace_extension_with(canonical_labeling_fname, "_can_lab.csv");

			M = NEW_lint(The_canonical_labeling.size() * canonical_labeling_len);
			for (u = 0; u < The_canonical_labeling.size(); u++) {
				for (v = 0; v < canonical_labeling_len; v++) {
					M[u * canonical_labeling_len + v] = The_canonical_labeling[u][v];
				}
			}
			Fio.lint_matrix_write_csv(canonical_labeling_fname,
					M, The_canonical_labeling.size(), canonical_labeling_len);

			FREE_lint(M);
		}

		if (Descr->f_save_ago) {
			if (f_v) {
				cout << "projective_space_object_classifier::process_multiple_objects_from_file "
						"f_save_ago is TRUE" << endl;
			}
			string ago_fname;
			int u;
			long int *M;
			file_io Fio;

			ago_fname.assign(input_data);
			replace_extension_with(ago_fname, "_ago.csv");

			M = NEW_lint(The_canonical_labeling.size());
			for (u = 0; u < The_canonical_labeling.size(); u++) {
				M[u] = Ago[u];
			}
			Fio.lint_vec_write_csv(M, The_canonical_labeling.size(),
					ago_fname, "Ago");

			FREE_lint(M);
		}

		if (Descr->f_load_canonical_labeling) {
			FREE_lint(Known_canonical_labeling);
		}

		if (Descr->f_load_ago) {
			FREE_lint(Known_ago);
		}



		if (file_type ==
				INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
			FREE_lint(Spread_table);
			}
		FREE_OBJECT(SoS);
		if (f_v) {
			cout << "projective_space_object_classifier::process_multiple_objects_from_file done" << endl;
		}
	}
	if (f_v) {
		cout << "projective_space_object_classifier::process_multiple_objects_from_file really done" << endl;
	}
}


void projective_space_object_classifier::process_set_of_points(
		std::string &input_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_object_classifier::process_set_of_points" << endl;
	}
	int f_found;
	string dummy;

	cout << "projective_space_object_classifier::classify_objects_using_nauty "
			"input set of points "
		<< input_data << ":" << endl;

	object_in_projective_space *OiP;
	strong_generators *SG;

	dummy.assign("command_line");
	OiP = PA->create_object_from_string(t_PTS,
			dummy, CB->n,
			input_data, verbose_level);

	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty "
				"before process_object" << endl;
		}
	int nb_rows, nb_cols;
	long int *canonical_labeling;
	int canonical_labeling_len;
	int idx;

	OiP->encoding_size(
			nb_rows, nb_cols,
			verbose_level);
	canonical_labeling = NEW_lint(nb_rows + nb_cols);

	f_found = process_object(OiP,
			SG,
			canonical_labeling, canonical_labeling_len,
			idx,
			verbose_level);


	if (f_v) {
		cout << "projective_space_object_classifier::classify_objects_using_nauty "
				"after process_object INPUT_TYPE_SET_OF_POINTS, f_found=" << f_found << endl;
		}


	if (f_found) {
		cout << "before FREE_OBJECT(SG)" << endl;
		FREE_OBJECT(SG);
		cout << "before FREE_OBJECT(OiP)" << endl;
		FREE_OBJECT(OiP);
		//cout << "before FREE_OBJECT(canonical_labeling)" << endl;
		//FREE_int(canonical_labeling);
		//cout << "after FREE_OBJECT(canonical_labeling)" << endl;
		FREE_lint(canonical_labeling);
		}
	else {
		cout << "projective_space_object_classifier::classify_objects_using_nauty "
				"New isomorphism type! The n e w number of "
			"isomorphism types is " << CB->nb_types << endl;
		int idx;

		object_in_projective_space_with_action *OiPA;

		OiPA = NEW_OBJECT(object_in_projective_space_with_action);

		OiPA->init(OiP, SG->group_order_as_lint(), SG, nb_rows, nb_cols,
				canonical_labeling, verbose_level);
		//FREE_OBJECT(SG);
		idx = CB->type_of[CB->n - 1];
		CB->Type_extra_data[idx] = OiPA;

		//compute_and_print_ago_distribution(cout,
		//		CB, verbose_level);
		}
	if (f_v) {
		cout << "projective_space_object_classifier::process_set_of_points done" << endl;
	}
}

void projective_space_object_classifier::process_set_of_points_from_file(
		std::string &input_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_object_classifier::process_set_of_points_from_file" << endl;
	}
	int f_found;
	file_io Fio;


	cout << "projective_space_object_classifier::process_set_of_points_from_file "
			"input set of points from file "
		<< input_data << ":" << endl;

	object_in_projective_space *OiP;
	strong_generators *SG;
	long int *the_set;
	int set_size;

	Fio.read_set_from_file(input_data, the_set, set_size, verbose_level);

	OiP = PA->create_object_from_int_vec(t_PTS,
			input_data, CB->n,
			the_set, set_size, verbose_level);

	if (f_v) {
		cout << "projective_space_object_classifier::process_set_of_points_from_file "
				"before encoding_size" << endl;
		}
	int nb_rows, nb_cols;
	long int *canonical_labeling;
	int canonical_labeling_len;
	int idx;

	OiP->encoding_size(
			nb_rows, nb_cols,
			verbose_level);
	canonical_labeling = NEW_lint(nb_rows + nb_cols);

	if (f_v) {
		cout << "projective_space_object_classifier::process_set_of_points_from_file "
				"before process_object" << endl;
		}
	f_found = process_object(OiP,
			SG,
			canonical_labeling, canonical_labeling_len,
			idx,
			verbose_level);


	if (f_v) {
		cout << "projective_space_object_classifier::process_set_of_points_from_file "
				"f_found=" << f_found << endl;
		}


	if (f_found) {
		FREE_OBJECT(SG);
		FREE_OBJECT(OiP);
		//FREE_int(canonical_labeling);
		FREE_lint(canonical_labeling);
		}
	else {
		cout << "projective_space_object_classifier::process_set_of_points_from_file "
				"New isomorphism type! The n e w number of "
			"isomorphism types is " << CB->nb_types << endl;
		int idx;

		object_in_projective_space_with_action *OiPA;

		OiPA = NEW_OBJECT(object_in_projective_space_with_action);

		OiPA->init(OiP, SG->group_order_as_lint(), SG, nb_rows, nb_cols,
				canonical_labeling, verbose_level);
		//FREE_OBJECT(SG);
		idx = CB->type_of[CB->n - 1];
		CB->Type_extra_data[idx] = OiPA;

		//compute_and_print_ago_distribution(cout,
		//		CB, verbose_level);
		}
	if (f_v) {
		cout << "projective_space_object_classifier::process_set_of_points_from_file done" << endl;
	}
}

void projective_space_object_classifier::process_set_of_lines_from_file(
		std::string &input_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_object_classifier::process_set_of_lines_from_file" << endl;
	}
	cout << "projective_space_with_action::process_set_of_lines_from_file "
			"input set of lines " << input_data
		<< ":" << endl;

	object_in_projective_space *OiP;
	strong_generators *SG;
	string dummy;

	dummy.assign("command_line");
	OiP = PA->create_object_from_string(t_LNS,
			dummy, CB->n,
			input_data, verbose_level);

	int nb_rows, nb_cols;
	long int *canonical_labeling;

	OiP->encoding_size(
			nb_rows, nb_cols,
			verbose_level);
	canonical_labeling = NEW_lint(nb_rows + nb_cols);
	int canonical_labeling_len;
	int idx;


	if (process_object(OiP,
		SG,
		canonical_labeling, canonical_labeling_len,
		idx,
		verbose_level)) {

		FREE_OBJECT(SG);
		FREE_OBJECT(OiP);
		FREE_lint(canonical_labeling);
		}
	else {
		cout << "projective_space_object_classifier::process_set_of_lines_from_file "
				"New isomorphism type! The n e w number of "
				"isomorphism types is " << CB->nb_types << endl;
		int idx;

		object_in_projective_space_with_action *OiPA;

		OiPA = NEW_OBJECT(object_in_projective_space_with_action);

		OiPA->init(OiP, SG->group_order_as_lint(), SG, nb_rows, nb_cols,
				canonical_labeling, verbose_level);
		//FREE_OBJECT(SG);
		idx = CB->type_of[CB->n - 1];
		CB->Type_extra_data[idx] = OiPA;

		//compute_and_print_ago_distribution(cout,
		//	CB, verbose_level);
		}
	if (f_v) {
		cout << "projective_space_object_classifier::process_set_of_lines_from_file done" << endl;
	}
}

void projective_space_object_classifier::process_set_of_packing(
		std::string &input_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_object_classifier::process_set_of_packing" << endl;
	}

	cout << "projective_space_object_classifier::process_set_of_packing "
			"input set of packing "
		<< input_data << ":" << endl;

	object_in_projective_space *OiP;
	strong_generators *SG;
	string dummy;

	dummy.assign("command_line");
	OiP = PA->create_object_from_string(t_PAC,
			dummy, CB->n,
			input_data, verbose_level);

	int nb_rows, nb_cols;
	long int *canonical_labeling;

	OiP->encoding_size(
			nb_rows, nb_cols,
			verbose_level);
	canonical_labeling = NEW_lint(nb_rows + nb_cols);
	int canonical_labeling_len;
	int idx;

	if (process_object(OiP,
		SG,
		canonical_labeling, canonical_labeling_len,
		idx,
		verbose_level)) {

		FREE_OBJECT(SG);
		FREE_OBJECT(OiP);
		FREE_lint(canonical_labeling);
		}
	else {
		cout << "projective_space_object_classifier::process_set_of_packing "
				"New isomorphism type! The n e w number of "
			"isomorphism types is " << CB->nb_types << endl;
		int idx;

		object_in_projective_space_with_action *OiPA;

		OiPA = NEW_OBJECT(object_in_projective_space_with_action);

		OiPA->init(OiP, SG->group_order_as_lint(), SG, nb_rows, nb_cols,
				canonical_labeling, verbose_level);
		//FREE_OBJECT(SG);
		idx = CB->type_of[CB->n - 1];
		CB->Type_extra_data[idx] = OiPA;

		//compute_and_print_ago_distribution(cout,
		//		CB, verbose_level);
		}
	if (f_v) {
		cout << "projective_space_object_classifier::process_set_of_packing done" << endl;
	}
}


int projective_space_object_classifier::process_object(
	object_in_projective_space *OiP,
	strong_generators *&SG,
	long int *canonical_labeling, int &canonical_labeling_len,
	int &idx,
	int verbose_level)
// returns f_found, which is TRUE if the object is rejected
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_object_classifier::process_object "
				"n=" << CB->n << endl;
	}

	longinteger_object go;

	bitvector *Canonical_form;
	//uchar *canonical_form;
	//int canonical_form_len;


	if (f_v) {
		cout << "projective_space_object_classifier::process_object "
				"before PA->set_stabilizer_of_object" << endl;
	}

	SG = PA->set_stabilizer_of_object(
		OiP,
		//Descr->f_save_incma_in_and_out, Descr->save_incma_in_and_out_prefix,
		TRUE /* f_compute_canonical_form */, Canonical_form,
		//canonical_form, canonical_form_len,
		canonical_labeling, canonical_labeling_len,
		verbose_level - 2);


	if (f_v) {
		cout << "projective_space_object_classifier::process_object "
				"after PA->set_stabilizer_of_object" << endl;
	}


	SG->group_order(go);

	if (CB->n == 0) {
		CB->init(nb_objects_to_test, Canonical_form->get_allocated_length(), verbose_level);
	}
	int f_found;

	CB->search_and_add_if_new(Canonical_form->get_data(), OiP, f_found, idx, verbose_level);


	//delete SG;

	if (f_v) {
		cout << "projective_space_object_classifier::process_object done" << endl;
	}
	return f_found;
}

int projective_space_object_classifier::process_object_with_known_canonical_labeling(
	object_in_projective_space *OiP,
	long int *canonical_labeling, int canonical_labeling_len,
	int &idx,
	int verbose_level)
// returns f_found, which is TRUE if the object is rejected
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_object_classifier::process_object_with_known_canonical_labeling "
				"n=" << CB->n << endl;
	}

	//longinteger_object go;
	//int *Extra_data;


	//uchar *canonical_form;
	//int canonical_form_len;


#if 0
	if (f_v) {
		cout << "projective_space_object_classifier::process_object_with_known_canonical_labeling "
				"before PA->set_stabilizer_of_object" << endl;
	}


	SG = PA->set_stabilizer_of_object(
		OiP,
		Descr->f_save_incma_in_and_out, Descr->save_incma_in_and_out_prefix,
		TRUE /* f_compute_canonical_form */,
		canonical_form, canonical_form_len,
		canonical_labeling, canonical_labeling_len,
		verbose_level - 2);


	if (f_v) {
		cout << "projective_space_object_classifier::process_object_with_known_canonical_labeling "
				"after PA->set_stabilizer_of_object" << endl;
	}
#else

	if (f_v) {
		cout << "projective_space_object_classifier::process_object_with_known_canonical_labeling "
				"before OiP->canonical_form_given_canonical_labeling" << endl;
		}

	bitvector *Bitvec_canonical_form;

	OiP->canonical_form_given_canonical_labeling(
				canonical_labeling,
				Bitvec_canonical_form,
				//canonical_form, canonical_form_len,
				verbose_level);


	if (f_v) {
		cout << "projective_space_object_classifier::process_object_with_known_canonical_labeling "
				"after OiP->canonical_form_given_canonical_labeling" << endl;
		}
#endif


	//SG->group_order(go);



	if (CB->n == 0) {
		CB->init(nb_objects_to_test, Bitvec_canonical_form->get_allocated_length() /*canonical_form_len*/, verbose_level);
	}
	int f_found;

	CB->search_and_add_if_new(Bitvec_canonical_form->get_data(), OiP, f_found, idx, verbose_level);


	//delete SG;

	FREE_OBJECT(Bitvec_canonical_form);

	if (f_v) {
		cout << "projective_space_object_classifier::process_object_with_known_canonical_labeling done" << endl;
	}
	return f_found;
}

void projective_space_object_classifier::save(
		std::string &output_prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname;
	file_io Fio;

	if (f_v) {
		cout << "projective_space_object_classifier::save" << endl;
	}
	fname.assign(output_prefix);
	fname.append("_classified.cvs");

	{
		ofstream fp(fname);
		int i, j;

		fp << "rep,ago,original_file,input_idx,input_set,"
				"nb_rows,nb_cols,canonical_form" << endl;
		for (i = 0; i < CB->nb_types; i++) {

			object_in_projective_space_with_action *OiPA;
			object_in_projective_space *OiP;

			//cout << i << " / " << CB->nb_types << " is "
			//	<< CB->Type_rep[i] << " : " << CB->Type_mult[i] << " : ";
			OiPA = (object_in_projective_space_with_action *) CB->Type_extra_data[i];
			OiP = OiPA->OiP;
			if (OiP == NULL) {
				cout << "OiP == NULL" << endl;
				exit(1);
			}
			if (OiP->type != t_PAC) {
				//OiP->print(cout);
				}
			//OiP->print(cout);

	#if 0
			for (j = 0; j < rep_len; j++) {
				cout << (int) Type_data[i][j];
				if (j < rep_len - 1) {
					cout << ", ";
					}
				}
	#endif
			//cout << "before writing OiP->set_as_string:" << endl;

			int ago;

			if (OiP->f_has_known_ago) {
				ago = OiP->known_ago;
			}
			else {
				ago = 0; //OiPA->Aut_gens->group_order_as_lint();
			}
			fp << i << "," << ago
					<< "," << OiP->input_fname
					<< "," << OiP->input_idx
					<< ",\"" << OiP->set_as_string << "\",";
			//cout << "before writing OiPA->nb_rows:" << endl;
			fp << OiPA->nb_rows << "," << OiPA->nb_cols<< ",";

			//cout << "before writing canonical labeling:" << endl;
			fp << "\"";
			for (j = 0; j < OiPA->nb_rows + OiPA->nb_cols; j++) {
				fp << OiPA->canonical_labeling[j];
				if (j < OiPA->nb_rows + OiPA->nb_cols - 1) {
					fp << ",";
				}
			}
			fp << "\"";
			fp << endl;
			}
		fp << "END" << endl;
	}
	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	if (f_v) {
		cout << "projective_space_object_classifier::save done" << endl;
	}
}


void projective_space_object_classifier::latex_report(
		std::string &fname,
		std::string &prefix,
		int fixed_structure_order_list_sz,
		int *fixed_structure_order_list,
		int max_TDO_depth,
		int verbose_level)
{
	int i, j;
	int f_v = (verbose_level >= 1);
	sorting Sorting;
	file_io Fio;
	latex_interface L;

	if (f_v) {
		cout << "projective_space_object_classifier::latex_report" << endl;
	}
	{
		ofstream fp(fname);
		latex_interface L;

		L.head_easy(fp);

		int *Table;
		int width = 4;
		int *row_labels;
		int *col_labels;
		int row_part_first[2], row_part_len[1];
		int nb_row_parts = 1;
		int col_part_first[2], col_part_len[1];
		int nb_col_parts = 1;



		row_part_first[0] = 0;
		row_part_first[1] = CB->nb_types;
		row_part_len[0] = CB->nb_types;

		col_part_first[0] = 0;
		col_part_first[1] = width;
		col_part_len[0] = width;

		Table = NEW_int(CB->nb_types * width);
		int_vec_zero(Table, CB->nb_types * width);

		row_labels = NEW_int(CB->nb_types);
		col_labels = NEW_int(width);
		for (i = 0; i < CB->nb_types; i++) {
			row_labels[i] = i;
		}
		for (j = 0; j < width; j++) {
			col_labels[j] = j;
		}

		for (i = 0; i < CB->nb_types; i++) {
			j = CB->perm[i];
			Table[i * width + 0] = CB->Type_rep[j];
			Table[i * width + 1] = CB->Type_mult[j];
			Table[i * width + 2] = 0; // group order
			Table[i * width + 3] = 0; // object list
		}

		fp << "\\section{Summary of Orbits}" << endl;
		fp << "$$" << endl;
		L.int_matrix_print_with_labels_and_partition(fp,
				Table, CB->nb_types, 4,
			row_labels, col_labels,
			row_part_first, row_part_len, nb_row_parts,
			col_part_first, col_part_len, nb_col_parts,
			print_summary_table_entry,
			CB /*void *data*/,
			TRUE /* f_tex */);
		fp << "$$" << endl;

		compute_and_print_ago_distribution_with_classes(fp,
				CB, verbose_level);

		for (i = 0; i < CB->nb_types; i++) {

			j = CB->perm[i];
			object_in_projective_space_with_action *OiPA;
			object_in_projective_space *OiP;

			cout << "###################################################"
					"#############################" << endl;
			cout << "Orbit " << i << " / " << CB->nb_types
					<< " is canonical form no " << j
					<< ", original object no " << CB->Type_rep[j]
					<< ", frequency " << CB->Type_mult[j]
					<< " : " << endl;


			{
				int *Input_objects;
				int nb_input_objects;
				CB->C_type_of->get_class_by_value(Input_objects,
					nb_input_objects, j, 0 /*verbose_level */);

				cout << "This isomorphism type appears " << nb_input_objects
						<< " times, namely for the following "
								"input objects:" << endl;
				L.int_vec_print_as_matrix(cout, Input_objects,
						nb_input_objects, 10 /* width */,
						FALSE /* f_tex */);

				FREE_int(Input_objects);
			}

			OiPA = (object_in_projective_space_with_action *) CB->Type_extra_data[j];
			OiP = OiPA->OiP;
			if (OiP->type != t_PAC) {
				OiP->print(cout);
			}

			//OiP->init_point_set(PA->P, (int *)CB->Type_extra_data[j],
			//sz, 0 /* verbose_level*/);



			strong_generators *SG;
			longinteger_object go;


			bitvector *Canonical_form;
			//uchar *canonical_form;
			//int canonical_form_len;

			int nb_r, nb_c;
			long int *canonical_labeling;
			int canonical_labeling_len;

			OiP->encoding_size(
					nb_r, nb_c,
					verbose_level);
			canonical_labeling = NEW_lint(nb_r + nb_c);

#if 1
			if (f_v) {
				cout << "projective_space_object_classifier::latex_report before PA->set_stabilizer_of_object" << endl;
			}
			SG = PA->set_stabilizer_of_object(
				OiP,
				//Descr->f_save_incma_in_and_out, Descr->save_incma_in_and_out_prefix,
				TRUE /* f_compute_canonical_form */, Canonical_form,
				//canonical_form, canonical_form_len,
				canonical_labeling, canonical_labeling_len,
				verbose_level - 2);
			if (f_v) {
				cout << "projective_space_object_classifier::latex_report after PA->set_stabilizer_of_object" << endl;
			}

			FREE_lint(canonical_labeling);

			SG->group_order(go);
#endif

			fp << "\\section*{Isomorphism type " << i << " / " << CB->nb_types << "}" << endl;
			fp << "Isomorphism type " << i << " / " << CB->nb_types
				//<<  " stored at " << j
				<< " is original object "
				<< CB->Type_rep[j] << " and appears "
				<< CB->Type_mult[j] << " times: \\\\" << endl;
			//if (OiP->type != t_PAC) {
				OiP->print_tex(fp);
				fp << endl;
				fp << "\\bigskip" << endl;
				fp << endl;
			//	}

			if (OiP->type == t_PAC) {
				long int *Sets;
				int nb_sets;
				int set_size;
				action *A_on_spreads;
				schreier *Sch;

				OiP->get_packing_as_set_system(Sets, nb_sets, set_size, verbose_level);


				A_on_spreads = PA->A_on_lines->create_induced_action_on_sets(nb_sets,
						set_size, Sets,
						verbose_level);


				Sch = SG->orbits_on_points_schreier(A_on_spreads, verbose_level);

				fp << "Orbits on spreads:\\\\" << endl;
				Sch->print_and_list_orbits_tex(fp);


				FREE_OBJECT(Sch);
				FREE_OBJECT(A_on_spreads);
				FREE_lint(Sets);
			}
			//int_vec_print(fp, OiP->set, OiP->sz);
			fp << "Group order " << go << "\\\\" << endl;

			//fp << "Stabilizer:" << endl;
			//SG->print_generators_tex(fp);

			{
				int *Input_objects;
				int nb_input_objects;
				CB->C_type_of->get_class_by_value(Input_objects,
						nb_input_objects, j, 0 /*verbose_level */);
				Sorting.int_vec_heapsort(Input_objects, nb_input_objects);

				fp << "This isomorphism type appears " << nb_input_objects
						<< " times, namely for the following "
						<< nb_input_objects << " input objects: " << endl;
				if (nb_input_objects < 10) {
					fp << "$" << endl;
					L.int_set_print_tex(fp, Input_objects, nb_input_objects);
					fp << "$\\\\" << endl;
				}
				else {
					fp << "$$" << endl;
					L.int_vec_print_as_matrix(fp, Input_objects,
						nb_input_objects, 10 /* width */, TRUE /* f_tex */);
					fp << "$$" << endl;
				}

				FREE_int(Input_objects);
			}



#if 0
			if (OiP->type == t_PTS) {
				//long int *set;
				//int sz;

				OiP->print_tex(fp);


				cout << "printing generators in restricted action:" << endl;
				action *A_restricted;

				A_restricted = SG->A->restricted_action(OiP->set, OiP->sz,
						verbose_level);
				SG->print_with_given_action(
						fp, A_restricted);
				FREE_OBJECT(A_restricted);
			}
#endif


			fp << "Stabilizer:\\\\" << endl;
			SG->print_generators_tex(fp);


#if 0
			//fp << "Stabilizer, all elements:\\\\" << endl;
			//SG->print_elements_ost(fp);
			//SG->print_elements_with_special_orthogonal_action_ost(fp);

			{
				action *A_conj;
				sims *Base_group;

				Base_group = SG->create_sims(verbose_level);

				A_conj = PA->A->create_induced_action_by_conjugation(
					Base_group, FALSE /* f_ownership */,
					verbose_level);

				fp << "Generators in conjugation action action on the group itself:\\\\" << endl;
				SG->print_with_given_action(fp, A_conj);

				fp << "Elements in conjugation action action on the group itself:\\\\" << endl;
				SG->print_elements_with_given_action(fp, A_conj);

				string fname_gap;
				char str[1000];

				fname_gap.assign("class_");

				sprintf(str, "%d", i);

				fname_gap.append(str);
				fname_gap.append(".gap");

				SG->export_permutation_group_to_GAP(fname_gap.c_str(), verbose_level);
				schreier *Sch;

				Sch = SG->orbits_on_points_schreier(A_conj, verbose_level);

				fp << "Orbits on itself by conjugation:\\\\" << endl;
				Sch->print_and_list_orbits_tex(fp);


				FREE_OBJECT(Sch);
				FREE_OBJECT(A_conj);
				FREE_OBJECT(Base_group);
			}
#endif


			int *Incma;
			int nb_rows, nb_cols;
			int *partition;
			incidence_structure *Inc;
			partitionstack *Stack;


			OiP->encode_incma_and_make_decomposition(
				Incma, nb_rows, nb_cols, partition,
				Inc,
				Stack,
				verbose_level);
			FREE_int(Incma);
			FREE_int(partition);
		#if 0
			cout << "set ";
			int_vec_print(cout, OiP->set, OiP->sz);
			cout << " go=" << go << endl;



			incidence_structure *Inc;
			partitionstack *Stack;

			int Sz[1];
			int *Subsets[1];

			Sz[0] = OiP->sz;
			Subsets[0] = OiP->set;

			cout << "computing decomposition:" << endl;
			PA->P->decomposition(1 /* nb_subsets */, Sz, Subsets,
				Inc,
				Stack,
				verbose_level);

		#if 0
			cout << "the decomposition is:" << endl;
			Inc->get_and_print_decomposition_schemes(*Stack);
			Stack->print_classes(cout);
		#endif




		#if 0
			fp << "canonical form: ";
			for (i = 0; i < canonical_form_len; i++) {
				fp << (int)canonical_form[i];
				if (i < canonical_form_len - 1) {
					fp << ", ";
					}
				}
			fp << "\\\\" << endl;
		#endif
		#endif


			Inc->get_and_print_row_tactical_decomposition_scheme_tex(
				fp, TRUE /* f_enter_math */,
				TRUE /* f_print_subscripts */, *Stack);

		#if 0
			Inc->get_and_print_tactical_decomposition_scheme_tex(
				fp, TRUE /* f_enter_math */,
				*Stack);
		#endif



			int f_refine_prev, f_refine, h;
			int f_print_subscripts = TRUE;

			f_refine_prev = TRUE;
			for (h = 0; h < max_TDO_depth; h++) {
				if (EVEN(h)) {
					f_refine = Inc->refine_column_partition_safe(
							*Stack, verbose_level - 3);
				}
				else {
					f_refine = Inc->refine_row_partition_safe(
							*Stack, verbose_level - 3);
				}

				if (f_v) {
					cout << "incidence_structure::compute_TDO_safe "
							"h=" << h << " after refine" << endl;
				}
				if (EVEN(h)) {
					//int f_list_incidences = FALSE;
					Inc->get_and_print_column_tactical_decomposition_scheme_tex(
						fp, TRUE /* f_enter_math */,
						f_print_subscripts, *Stack);
					//get_and_print_col_decomposition_scheme(
					//PStack, f_list_incidences, FALSE);
					//PStack.print_classes_points_and_lines(cout);
				}
				else {
					//int f_list_incidences = FALSE;
					Inc->get_and_print_row_tactical_decomposition_scheme_tex(
						fp, TRUE /* f_enter_math */,
						f_print_subscripts, *Stack);
					//get_and_print_row_decomposition_scheme(
					//PStack, f_list_incidences, FALSE);
					//PStack.print_classes_points_and_lines(cout);
				}

				if (!f_refine_prev && !f_refine) {
					break;
				}
				f_refine_prev = f_refine;
			}

			cout << "Classes of the partition:\\\\" << endl;
			Stack->print_classes_tex(fp);



			OiP->klein(verbose_level);

#if 1
			sims *Stab;
			int *Elt;
			int nb_trials;
			int max_trials = 100;

			Stab = SG->create_sims(verbose_level);
			Elt = NEW_int(PA->A->elt_size_in_int);

			for (h = 0; h < fixed_structure_order_list_sz; h++) {
				if (Stab->find_element_of_given_order_int(Elt,
						fixed_structure_order_list[h], nb_trials, max_trials,
						verbose_level)) {
					fp << "We found an element of order "
							<< fixed_structure_order_list[h] << ", which is:" << endl;
					fp << "$$" << endl;
					PA->A->element_print_latex(Elt, fp);
					fp << "$$" << endl;
					PA->report_fixed_objects_in_PG_3_tex(
						Elt, fp,
						verbose_level);
				}
				else {
					fp << "We could not find an element of order "
						<< fixed_structure_order_list[h] << "\\\\" << endl;
				}
			}

			FREE_int(Elt);
#endif

			FREE_OBJECT(SG);

			FREE_OBJECT(Stack);
			FREE_OBJECT(Inc);

		}


		L.foot(fp);
	}

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	//FREE_int(perm);
	//FREE_int(v);
	if (f_v) {
		cout << "projective_space_object_classifier::latex_report done" << endl;
	}
}



}}

