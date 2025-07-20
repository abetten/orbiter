/*
 * combinatorial_object_stream.cpp
 *
 *  Created on: Dec 17, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {



combinatorial_object_stream::combinatorial_object_stream()
{
	Record_birth();
	//Data_input_stream_description = NULL;

	IS = NULL;

	f_projective_space = false;
	PA = NULL;

	Classification_of_objects = NULL;
	Objects_after_classification = NULL;

}

combinatorial_object_stream::~combinatorial_object_stream()
{
	Record_death();
	if (Classification_of_objects) {
		FREE_OBJECT(Classification_of_objects);
	}
	if (Objects_after_classification) {
		FREE_OBJECT(Objects_after_classification);
	}
}


void combinatorial_object_stream::init(
		combinatorics::canonical_form_classification::data_input_stream_description
				*Data_input_stream_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::init" << endl;
	}
	//combinatorial_object::Data_input_stream_description = Data_input_stream_description;


	IS = NEW_OBJECT(combinatorics::canonical_form_classification::data_input_stream);

	if (f_v) {
		cout << "combinatorial_object_stream::init "
				"before IS->init" << endl;
	}

	IS->init(Data_input_stream_description, verbose_level);

	if (f_v) {
		cout << "combinatorial_object_stream::init "
				"after IS->init" << endl;
	}


	if (f_v) {
		cout << "combinatorial_object_stream::init done" << endl;
	}

}



void combinatorial_object_stream::do_canonical_form(
		combinatorics::canonical_form_classification::classification_of_objects_description
				*Canonical_form_Descr,
		int f_projective_space,
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
// called from combinatorial_object_activity::perform_activity_combo
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_canonical_form" << endl;
	}

	geometry::projective_geometry::projective_space *P;


	if (f_projective_space) {
		if (f_v) {
			cout << "combinatorial_object_stream::do_canonical_form we have a projectve space" << endl;
			cout << "combinatorial_object_stream::do_canonical_form PA = " << PA->P->label_txt << endl;
		}
		P = PA->P;
	}
	else {
		if (f_v) {
			cout << "combinatorial_object_stream::do_canonical_form no projective space" << endl;
		}
		P = NULL;
	}
	combinatorial_object_stream::f_projective_space = f_projective_space;
	combinatorial_object_stream::PA = PA;


	if (f_v) {
		cout << "combinatorial_object_stream::do_canonical_form allocating Classification_of_objects" << endl;
	}

	Classification_of_objects = NEW_OBJECT(combinatorics::canonical_form_classification::classification_of_objects);

	if (f_v) {
		cout << "combinatorial_object_stream::do_canonical_form "
				"before Classification_of_objects->perform_classification" << endl;
	}
	Classification_of_objects->perform_classification(
			Canonical_form_Descr,
			f_projective_space,
			P,
			IS,
			verbose_level - 2);
	if (f_v) {
		cout << "combinatorial_object_stream::do_canonical_form "
				"after Classification_of_objects->perform_classification" << endl;
	}




	if (f_v) {
		cout << "combinatorial_object_stream::do_canonical_form done" << endl;
	}

}

void combinatorial_object_stream::write_canonical_form_data(
		std::string fname_base,
		int verbose_level)
// creates a csv file containing the result from the canonization as nauty output.
// The input object is repeated together with the input_idx.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::write_canonical_form_data" << endl;
	}

	std::string fname;

	fname = fname_base + "_c.csv";
	//fname = Classification_of_objects->IS->Descr->label_txt + "_canonicized.csv";

	if (f_v) {
		cout << "combinatorial_object_stream::write_canonical_form_data fname = " << fname << endl;
	}

	int iso_type;



	std::string *Headers;
	std::string *Table;

	int nb_rows, nb_cols;

	nb_rows = Classification_of_objects->Output->nb_orbits;
	nb_cols = 11;
	Headers = new std::string[nb_cols];
	Table = new std::string[nb_rows * nb_cols];

	Headers[0] = "Row";
	Headers[1] = "object";
	Headers[2] = "n";
	Headers[3] = "ago";
	Headers[4] = "base_length";
	Headers[5] = "aut_counter";
	Headers[6] = "base";
	Headers[7] = "tl";
	Headers[8] = "aut";
	Headers[9] = "cl";
	Headers[10] = "stats";

	int input_idx;

	for (iso_type = 0; iso_type < Classification_of_objects->Output->nb_orbits; iso_type++) {

		if (f_v) {
			cout << "combinatorial_object_stream::write_canonical_form_data "
					"iso_type = " << iso_type << " / " << Classification_of_objects->Output->nb_orbits << endl;
			//cout << "NO=" << endl;
			//CO->NO_transversal[iso_type]->print();
		}

		input_idx = Classification_of_objects->Output->Idx_transversal[iso_type];

		//Classification_of_objects->Output->OWCF[iso_type];
		//Classification_of_objects->Output->NO[iso_type];

		std::string s_idx;


		s_idx = std::to_string(Classification_of_objects->Output->OWCF[input_idx]->input_idx);

		string s1;


		s1 = Classification_of_objects->Output->OWCF[input_idx]->stringify(
				verbose_level);

		// 9 strings:
		std::string s_n;
		std::string s_ago;
		std::string s_base_length;
		std::string s_aut_counter;
		std::string s_base;
		std::string s_tl;
		std::string s_aut;
		std::string s_cl;
		std::string s_stats;

		Classification_of_objects->Output->NO[input_idx]->stringify(
				s_n, s_ago,
				s_base_length, s_aut_counter,
				s_base, s_tl,
				s_aut, s_cl,
				s_stats,
				verbose_level - 2);


		Table[iso_type * nb_cols + 0] = "\"" + s_idx + "\"";
		Table[iso_type * nb_cols + 1] = "\"" + s1 + "\"";
		Table[iso_type * nb_cols + 2] = "\"" + s_n + "\"";
		Table[iso_type * nb_cols + 3] = "\"" + s_ago + "\"";
		Table[iso_type * nb_cols + 4] = "\"" + s_base_length + "\"";
		Table[iso_type * nb_cols + 5] = "\"" + s_aut_counter + "\"";
		Table[iso_type * nb_cols + 6] = "\"" + s_base + "\"";
		Table[iso_type * nb_cols + 7] = "\"" + s_tl + "\"";
		Table[iso_type * nb_cols + 8] = "\"" + s_aut + "\"";
		Table[iso_type * nb_cols + 9] = "\"" + s_cl + "\"";
		Table[iso_type * nb_cols + 10] = "\"" + s_stats + "\"";



	}

	other::orbiter_kernel_system::file_io Fio;


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_rows, nb_cols, Table,
			Headers,
			verbose_level - 1);

	if (f_v) {
		cout << "combinatorial_object_stream::write_canonical_form_data "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	delete [] Headers;
	delete [] Table;



	if (f_v) {
		cout << "combinatorial_object_stream::write_canonical_form_data done" << endl;
	}
}

void combinatorial_object_stream::write_canonical_form_data_non_trivial_group(
		std::string fname_base,
		int verbose_level)
// creates a csv file containing the result from the canonization as nauty output.
// The input object is repeated together with the input_idx.
// This function only writes the objects whose automorphism group is non-trivial.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::write_canonical_form_data_non_trivial_group" << endl;
	}

	std::string fname;

	fname = fname_base + "_c_nt.csv";
	//fname = Classification_of_objects->IS->Descr->label_txt + "_canonicized.csv";

	if (f_v) {
		cout << "combinatorial_object_stream::write_canonical_form_data_non_trivial_group fname = " << fname << endl;
	}

	int iso_type;



	std::string *Headers;
	std::string *Table;

	int nb_rows, nb_cols;
	int nb_r;

	nb_rows = Classification_of_objects->Output->nb_orbits;
	nb_cols = 11;
	Headers = new std::string[nb_cols];
	Table = new std::string[nb_rows * nb_cols];

	Headers[0] = "Row";
	Headers[1] = "object";
	Headers[2] = "n";
	Headers[3] = "ago";
	Headers[4] = "base_length";
	Headers[5] = "aut_counter";
	Headers[6] = "base";
	Headers[7] = "tl";
	Headers[8] = "aut";
	Headers[9] = "cl";
	Headers[10] = "stats";

	int input_idx;

	nb_r = 0;
	for (iso_type = 0; iso_type < Classification_of_objects->Output->nb_orbits; iso_type++) {

		if (f_v) {
			cout << "combinatorial_object_stream::write_canonical_form_data_non_trivial_group "
					"iso_type = " << iso_type << " / " << Classification_of_objects->Output->nb_orbits << endl;
			cout << "NO=" << endl;
			//CO->NO_transversal[iso_type]->print();
		}

		//Classification_of_objects->Output->OWCF[iso_type];
		//Classification_of_objects->Output->NO[iso_type];

		// skip if the automorphism group is trivial:

		if (Classification_of_objects->Output->NO[iso_type]->Ago->is_one()) {
			continue;
		}

		input_idx = Classification_of_objects->Output->Idx_transversal[iso_type];

		std::string s_idx;


		s_idx = std::to_string(Classification_of_objects->Output->OWCF[input_idx]->input_idx);

		string s1;


		s1 = Classification_of_objects->Output->OWCF[input_idx]->stringify(
				verbose_level);

		// 9 strings:
		std::string s_n;
		std::string s_ago;
		std::string s_base_length;
		std::string s_aut_counter;
		std::string s_base;
		std::string s_tl;
		std::string s_aut;
		std::string s_cl;
		std::string s_stats;

		Classification_of_objects->Output->NO[input_idx]->stringify(
				s_n, s_ago,
				s_base_length, s_aut_counter,
				s_base, s_tl,
				s_aut, s_cl,
				s_stats,
				verbose_level - 2);


		Table[nb_r * nb_cols + 0] = "\"" + s_idx + "\"";
		Table[nb_r * nb_cols + 1] = "\"" + s1 + "\"";
		Table[nb_r * nb_cols + 2] = "\"" + s_n + "\"";
		Table[nb_r * nb_cols + 3] = "\"" + s_ago + "\"";
		Table[nb_r * nb_cols + 4] = "\"" + s_base_length + "\"";
		Table[nb_r * nb_cols + 5] = "\"" + s_aut_counter + "\"";
		Table[nb_r * nb_cols + 6] = "\"" + s_base + "\"";
		Table[nb_r * nb_cols + 7] = "\"" + s_tl + "\"";
		Table[nb_r * nb_cols + 8] = "\"" + s_aut + "\"";
		Table[nb_r * nb_cols + 9] = "\"" + s_cl + "\"";
		Table[nb_r * nb_cols + 10] = "\"" + s_stats + "\"";

		nb_r++;


	}

	other::orbiter_kernel_system::file_io Fio;


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_r, nb_cols, Table,
			Headers,
			verbose_level - 1);

	if (f_v) {
		cout << "combinatorial_object_stream::write_canonical_form_data_non_trivial_group "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	delete [] Headers;
	delete [] Table;



	if (f_v) {
		cout << "combinatorial_object_stream::write_canonical_form_data_non_trivial_group done" << endl;
	}
}



void combinatorial_object_stream::do_post_processing(
		int verbose_level)
// called from combinatorial_object_activity::perform_activity_combo
// creates Objects_after_classification, which is memory intense
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_post_processing" << endl;
	}

	Objects_after_classification = NEW_OBJECT(canonical_form::objects_after_classification);

	if (!IS->Descr->f_label) {
		cout << "please use -label <label_txt> <label_tex> "
				"when defining the input stream for the combinatorial object" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "combinatorial_object_stream::do_post_processing "
				"before Objects_after_classification->init_after_nauty" << endl;
	}
	Objects_after_classification->init_after_nauty(
			Classification_of_objects,
			f_projective_space, PA,
			verbose_level);
	if (f_v) {
		cout << "combinatorial_object_stream::do_post_processing "
				"after Objects_after_classification->init_after_nauty" << endl;
	}

	if (f_v) {
		cout << "combinatorial_object_stream::do_post_processing done" << endl;
	}

}

void combinatorial_object_stream::do_test_distinguishing_property(
		combinatorics::graph_theory::colored_graph *CG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_test_distinguishing_property" << endl;
	}

	int input_idx;
	int *F_distinguishing;

	F_distinguishing = NEW_int(IS->Objects.size());


	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

		combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

		F_distinguishing[input_idx] = CG->test_distinguishing_property(
				OwCF->set, OwCF->sz, verbose_level);
	}

	other::data_structures::tally T;

	T.init(F_distinguishing, IS->Objects.size(), false, 0);
	cout << "classification : ";
	T.print_first(true /* f_backwards*/);
	cout << endl;

	cout << "distinguishing sets are:";
	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {
		if (F_distinguishing[input_idx]) {
			cout << input_idx << ", ";
		}
	}
	cout << endl;

	cout << "distinguishing sets are:";
	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {
		if (!F_distinguishing[input_idx]) {
			continue;
		}

		combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

		OwCF->print_brief(cout);

	}
	cout << endl;


	FREE_int(F_distinguishing);

	if (f_v) {
		cout << "combinatorial_object_stream::do_test_distinguishing_property done" << endl;
	}

}


void combinatorial_object_stream::do_covering_type(
		orbits::orbits_create *Orb,
		int subset_sz,
		int f_filter_by_Steiner_property,
		other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_covering_type verbose_level = " << verbose_level << endl;
		cout << "combinatorial_object_stream::do_covering_type "
				"subset_sz = " << subset_sz << endl;
		cout << "combinatorial_object_stream::do_covering_type "
				"f_filter_by_Steiner_property = " << f_filter_by_Steiner_property << endl;
	}

	AO = NEW_OBJECT(other::orbiter_kernel_system::activity_output);
	//AO->nb_rows = IS->Objects.size();

	AO->headings = "set,idx,covering_wrt" + std::to_string(subset_sz) + ",steiner_p";
	AO->nb_cols = 4;
	AO->description_txt = "covering_wrt" + std::to_string(subset_sz);


	combinatorics::other_combinatorics::combinatorics_domain Combi;

	int nCk;

	if (IS->Objects.size() == 0) {

		if (f_v) {
			cout << "combinatorial_object_stream::do_covering_type "
					"breaking off early because there is no input" << endl;
		}
		return;
	}

	if (f_v) {
		cout << "combinatorial_object_stream::do_covering_type "
				"IS->Objects.size() = " << IS->Objects.size() << endl;

	}


	combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

	OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[0];


	if (OwCF->type != t_PTS) {
		cout << "combinatorial_object_stream::do_covering_type OwCF->type != t_PTS" << endl;
		exit(1);
	}

	int sz;
	sz = OwCF->sz;

	if (f_v) {
		cout << "combinatorial_object_stream::do_covering_type sz = " << sz << endl;

	}



	//degree = Orb->Group->A->degree;

	poset_classification::poset_classification *PC;

	if (!Orb->f_has_On_subsets) {
		cout << "combinatorial_object_stream::do_covering_type "
				"the orbit structure has no subset orbits" << endl;
		exit(1);
	}
	PC = Orb->On_subsets;

	int nb_orbits;

	nb_orbits = PC->nb_orbits_at_level(subset_sz);

	if (f_v) {
		cout << "combinatorial_object_stream::do_covering_type nb_orbits = " << nb_orbits << endl;

	}

	if (nb_orbits < 0) {
		cout << "combinatorial_object_stream::do_covering_type "
				"the orbits on subsets of size " << subset_sz << " are not available" << endl;
		exit(1);
	}

	nCk = Combi.int_n_choose_k(sz, subset_sz);

	if (f_v) {
		cout << "combinatorial_object_stream::do_covering_type nCk = " << nCk << endl;

	}

	int *covering;
	char *covering_char;
	int *index_subset;
	long int *subset;
	long int *canonical_subset;
	int *Elt;

	covering = NEW_int(nb_orbits);
	covering_char = NEW_char(nb_orbits);
	index_subset = NEW_int(subset_sz);
	subset = NEW_lint(subset_sz);
	canonical_subset = NEW_lint(subset_sz);
	Elt = NEW_int(PC->get_poset()->A->elt_size_in_int);

	long int input_idx;

	other::data_structures::algorithms Algorithms;


	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

		combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];


		if (OwCF->type != t_PTS) {
			cout << "combinatorial_object_stream::do_covering_type OwCF->type != t_PTS" << endl;
			exit(1);
		}

		long int *set;

		set = OwCF->set;
		if (OwCF->sz != sz) {
			cout << "the objects must have the same size" << endl;
			exit(1);
		}

		int rk, i, local_idx;
		Int_vec_zero(covering, nb_orbits);

		for (rk = 0; rk < nCk; rk++) {
			Combi.unrank_k_subset(rk, index_subset, sz, subset_sz);
			for (i = 0; i < subset_sz; i++) {
				subset[i] = set[index_subset[i]];
			}

			local_idx = PC->trace_set(subset, subset_sz, subset_sz,
				canonical_subset, Elt,
				0 /*verbose_level - 3*/);

			covering[local_idx]++;
		}

		int m, f_steiner;

		m = Int_vec_maximum(covering, nb_orbits);

		if (m == 1) {
			f_steiner = true;
		}
		else {
			f_steiner = false;
		}

		int f_keep = true;

		if (f_filter_by_Steiner_property && f_steiner == false) {
			f_keep = false;
		}
		else {
			f_keep = true;
		}

		if (f_keep) {
			std::vector<std::string> feedback;

			string str;


			str = "\"" + Lint_vec_stringify(set, sz) + "\"";

			feedback.push_back(str);

			str = std::to_string(input_idx);

			feedback.push_back(str);

			unsigned long int m;

			for (i = 0; i < nb_orbits; i++) {
				covering_char[i] = covering[i];
			}

			m = Algorithms.make_bitword(
					covering_char, nb_orbits);

			str = "\"" + std::to_string(m) + "\"";
			//str = "\"" + Int_vec_stringify(covering, nb_orbits) + "\"";

			feedback.push_back(str);

			str = std::to_string(f_steiner);

			feedback.push_back(str);

			AO->Feedback.push_back(feedback);
		}

	}

	FREE_int(covering);
	FREE_char(covering_char);
	FREE_int(index_subset);
	FREE_lint(subset);
	FREE_lint(canonical_subset);

	FREE_int(Elt);

	if (f_v) {
		cout << "combinatorial_object_stream::do_covering_type done" << endl;
	}
}

void combinatorial_object_stream::do_compute_frequency_graph(
		combinatorics::graph_theory::colored_graph *CG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_compute_frequency_graph" << endl;
	}


	int input_idx;
	//int N;
	int *code = NULL;
	//int sz = 0;


	code = NEW_int(CG->nb_points);

	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

		combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

#if 0
		if (input_idx == 0) {

			sz = OwCF->sz;
			N = 1 << OwCF->sz;
			frequency = NEW_int(N);
		}
		else {
			if (OwCF->sz != sz) {
				cout << "the size of the sets must be constant" << endl;
				exit(1);
			}
		}
#endif

		CG->all_distinguishing_codes(
				OwCF->set, OwCF->sz, code, verbose_level);
		//CG->distinguishing_code_frequency(
		//		OwCF->set, OwCF->sz, frequency, N, verbose_level);

		other::data_structures::tally T;

		T.init(code, CG->nb_points, false, 0);
		cout << "frequency tally : ";
		T.print_first(true /* f_backwards*/);
		cout << endl;
		T.print_types();

	}


	FREE_int(code);

	if (f_v) {
		cout << "combinatorial_object_stream::do_compute_frequency_graph done" << endl;
	}
}

void combinatorial_object_stream::do_compute_ideal(
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_compute_ideal" << endl;
	}




	int input_idx;


	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

		combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

		HPD->explore_vanishing_ideal(OwCF->set, OwCF->sz, verbose_level);

	}

	if (f_v) {
		cout << "combinatorial_object_stream::do_compute_ideal done" << endl;
	}
}

void combinatorial_object_stream::do_save(
		std::string &save_as_fname,
		int f_extract,
		long int *extract_idx_set, int extract_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_save" << endl;
	}
	int input_idx;
	int sz;
	int N;

	N = IS->Objects.size();

	combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

	OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[0];

	//OwCF->set;
	sz = OwCF->sz;

	for (input_idx = 0; input_idx < N; input_idx++) {

		if (false) {
			cout << "combinatorial_object_stream::do_save "
					"input_idx = " << input_idx
					<< " / " << IS->Objects.size() << endl;
		}

		combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *)
				IS->Objects[input_idx];

		//OwCF->set;
		if (OwCF->sz != sz) {
			cout << "the objects have different sizes, cannot save" << endl;
			exit(1);
		}


	}

	long int *Sets;

	Sets = NEW_lint(N * sz);

	for (input_idx = 0; input_idx < N; input_idx++) {
		combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

		Lint_vec_copy(OwCF->set, Sets + input_idx * sz, sz);
	}

	cout << "The combined number of objects is " << N << endl;


	if (f_extract) {
		if (f_v) {
			cout << "extracting subset of size " << extract_size << endl;
		}
		long int *Sets2;
		int h, i;

		Sets2 = NEW_lint(extract_size * sz);
		for (h = 0; h < extract_size; h++) {
			i = extract_idx_set[h];
			Lint_vec_copy(Sets + i * sz, Sets2 + h * sz, sz);
		}
		FREE_lint(Sets);
		Sets = Sets2;
		if (f_v) {
			cout << "number of sets is reduced from "
					<< N << " to " << extract_size << endl;
		}
		N = extract_size;
	}
	else {

	}
	other::orbiter_kernel_system::file_io Fio;

	string fname_out;

	fname_out.assign(save_as_fname);

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname_out, Sets, N, sz);

	cout << "Written file " << fname_out
			<< " of size " << Fio.file_size(fname_out) << endl;

	if (f_v) {
		cout << "combinatorial_object::do_save done" << endl;
	}
}


void combinatorial_object_stream::draw_incidence_matrices(
		std::string &prefix,
		other::graphics::draw_incidence_structure_description *Draw_incidence_structure_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;
	other::l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_stream::draw_incidence_matrices" << endl;
	}



	string fname;

	fname = prefix + "_incma.tex";

	if (f_v) {
		cout << "combinatorial_object_stream::draw_incidence_matrices "
				"before latex_report" << endl;
	}


	{
		ofstream ost(fname);
		other::l1_interfaces::latex_interface L;

		L.head_easy(ost);


		int N;

		N = IS->Objects.size();



		if (f_v) {
			cout << "combinatorial_object_stream::draw_incidence_matrices "
					"before loop" << endl;
		}

		int i;

		ost << "\\noindent" << endl;

		for (i = 0; i < N; i++) {

			if (f_v) {
				cout << "combinatorial_object_stream::draw_incidence_matrices "
						"object " << i << " / " << N << endl;
			}
			combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

			OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[i];


			combinatorics::canonical_form_classification::encoded_combinatorial_object *Enc;

			if (f_v) {
				cout << "combinatorial_object_stream::draw_incidence_matrices "
						"before OwCF->encode_incma" << endl;
			}
			OwCF->encode_incma(Enc, verbose_level);
			if (f_v) {
				cout << "combinatorial_object_stream::draw_incidence_matrices "
						"after OwCF->encode_incma" << endl;
			}

			//Enc->latex_set_system_by_rows_and_columns(ost, verbose_level);


			if (f_v) {
				cout << "combinatorial_object_stream::draw_incidence_matrices "
						"before OwCF->latex_incma" << endl;
			}
			Enc->latex_incma(ost, Draw_incidence_structure_description, verbose_level);
			if (f_v) {
				cout << "combinatorial_object_stream::draw_incidence_matrices "
						"after OwCF->latex_incma" << endl;
			}

			FREE_OBJECT(Enc);


		}



		L.foot(ost);
	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "combinatorial_object_stream::draw_incidence_matrices done" << endl;
	}
}

void combinatorial_object_stream::unpack_from_restricted_action(
		std::string &prefix,
		groups::any_group *G,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;
	other::l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_stream::unpack_from_restricted_action" << endl;
	}



	if (G->A->type_G != action_by_restriction_t) {
		cout << "combinatorial_object_stream::unpack_from_restricted_action "
				"must be a restricted action" << endl;
		exit(1);
	}
	induced_actions::action_by_restriction *ABR;
	ABR = G->A->G.ABR;


	string fname;

	fname = prefix + "_unpacked.txt";

	if (f_v) {
		cout << "combinatorial_object_stream::unpack_from_restricted_action "
				"before latex_report" << endl;
	}


	{

		ofstream ost(fname);

		int N;

		N = IS->Objects.size();



		if (f_v) {
			cout << "combinatorial_object_stream::unpack_from_restricted_action "
					"before loop" << endl;
		}

		int i, h;
		long int a, b;


		for (i = 0; i < N; i++) {

			combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

			OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[i];


			//encoded_combinatorial_object *Enc;

			//OwCF->encode_incma(Enc, verbose_level);

			for (h = 0; h < OwCF->sz; h++) {

				a = OwCF->set[h];
				b = ABR->original_point(a);
				OwCF->set[h] = b;
			}

			//Enc->latex_incma(ost, verbose_level);

			//FREE_OBJECT(Enc);

			ost << OwCF->sz;
			for (h = 0; h < OwCF->sz; h++) {
				ost << " " << OwCF->set[h];
			}
			ost << endl;

		}

		ost << -1 << endl;


	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "combinatorial_object_stream::unpack_from_restricted_action done" << endl;
	}
}


void combinatorial_object_stream::line_covering_type(
		std::string &prefix,
		projective_geometry::projective_space_with_action *PA,
		std::string &lines,
		int verbose_level)
// calls P->Subspaces->line_intersection_type_basic_given_a_set_of_lines
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::line_covering_type" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	other::l1_interfaces::latex_interface L;

	geometry::projective_geometry::projective_space *P;

	P = PA->P;

	long int *the_lines;
	int nb_lines;

	Get_lint_vector_from_label(lines, the_lines, nb_lines, verbose_level);

	string fname;

	fname = prefix + "_line_covering_type.txt";

	if (f_v) {
		cout << "combinatorial_object_stream::line_covering_type before latex_report" << endl;
	}


	{

		ofstream ost(fname);

		int N;

		N = IS->Objects.size();



		if (f_v) {
			cout << "combinatorial_object_stream::line_covering_type before loop" << endl;
		}

		int i, h;

		int *type;

		type = NEW_int(nb_lines);


		for (i = 0; i < N; i++) {

			combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

			OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[i];


			P->Subspaces->line_intersection_type_basic_given_a_set_of_lines(
					the_lines, nb_lines,
					OwCF->set, OwCF->sz, type, 0 /*verbose_level */);

			ost << OwCF->sz;
			for (h = 0; h < nb_lines; h++) {
				ost << " " << type[h];
			}
			ost << endl;

		}

		ost << -1 << endl;


	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "combinatorial_object_stream::line_covering_type done" << endl;
	}
}

void combinatorial_object_stream::line_type(
		std::string &prefix,
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;
	other::l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_stream::line_type" << endl;
	}


	geometry::projective_geometry::projective_space *P;

	P = PA->P;

	string fname;

	fname = prefix + "_line_type.txt";

	if (f_v) {
		cout << "combinatorial_object_stream::line_type before latex_report" << endl;
	}


	{

		ofstream ost(fname);

		int N;

		N = IS->Objects.size();



		if (f_v) {
			cout << "combinatorial_object_stream::line_type before loop" << endl;
		}

		int i, h;

		int *type;

		type = NEW_int(P->Subspaces->N_lines);


		for (i = 0; i < N; i++) {

			combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

			OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[i];


			P->Subspaces->line_intersection_type(
					OwCF->set, OwCF->sz, type, 0 /* verbose_level */);
				// type[N_lines]

			ost << OwCF->sz;
			for (h = 0; h < P->Subspaces->N_lines; h++) {
				ost << " " << type[h];
			}
			ost << endl;

#if 1
			other::data_structures::tally T;

			T.init(type, P->Subspaces->N_lines, false, 0);

			if (f_v) {
				cout << "combinatorial_object_stream::perform_activity_GOC line type:" << endl;
				T.print(true /* f_backwards*/);
				cout << endl;
			}

			std::string fname_line_type;


			fname_line_type = prefix + "_line_type_set_partition_" + std::to_string(i) + ".csv";

			T.save_classes_individually(fname_line_type);
			if (true) {
				cout << "Written file " << fname_line_type << " of size "
						<< Fio.file_size(fname_line_type) << endl;
			}

#endif



		}

		ost << -1 << endl;


	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "combinatorial_object_stream::line_type done" << endl;
	}
}


void combinatorial_object_stream::do_activity(
		user_interface::activity_description *Activity_description,
		other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
// front end for graph theoretic activities
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_activity" << endl;
	}


	if (Activity_description->f_graph_theoretic_activity) {

		if (f_v) {
			cout << "combinatorial_object_stream::do_activity f_graph_theoretic_activity" << endl;
		}
		do_graph_theoretic_activity(
				Activity_description->Graph_theoretic_activity_description,
				AO,
				verbose_level);

	}
	else {
		cout << "combinatorial_object_stream::do_activity activity is not recognized" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "combinatorial_object_stream::do_activity done" << endl;
	}
}

void combinatorial_object_stream::do_graph_theoretic_activity(
		apps_graph_theory::graph_theoretic_activity_description
				*Graph_theoretic_activity_description,
				other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_graph_theoretic_activity" << endl;
	}


	AO = NEW_OBJECT(other::orbiter_kernel_system::activity_output);
	//AO->nb_rows = IS->Objects.size();

	{
		apps_graph_theory::graph_theoretic_activity Activity;


		Activity.feedback_headings(
				Graph_theoretic_activity_description,
				AO->headings,
				AO->nb_cols,
				verbose_level);


		Activity.get_label(
				Graph_theoretic_activity_description,
				AO->description_txt,
				verbose_level);

	}


	int input_idx;


	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

		combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

		if (OwCF->type != t_INC) {
			cout << "combinatorial_object_stream::do_graph_theoretic_activity "
					"expecting an object of type incidence geometry" << endl;
			exit(1);
		}

		int *Adj;
		int N;
		string label, label_tex;

		label = IS->Descr->label_txt + "_obj" + std::to_string(input_idx);
		label_tex = IS->Descr->label_tex + "\\_obj" + std::to_string(input_idx);

		OwCF->collinearity_graph(
				Adj, N,
				verbose_level);

		combinatorics::graph_theory::colored_graph *CG;

		CG = NEW_OBJECT(combinatorics::graph_theory::colored_graph);
		if (f_v) {
			cout << "combinatorial_object_stream::do_graph_theoretic_activity "
					"before CG->init_from_adjacency_no_colors" << endl;
		}
		CG->init_from_adjacency_no_colors(
				N, Adj, label, label_tex,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_stream::do_graph_theoretic_activity "
					"after CG->init_from_adjacency_no_colors" << endl;
		}

		{
			apps_graph_theory::graph_theoretic_activity Activity;

			std::vector<std::string> feedback;

			Activity.init(Graph_theoretic_activity_description, 1, &CG, verbose_level);


			if (f_v) {
				cout << "combinatorial_object_stream::do_graph_theoretic_activity "
						"before Activity.perform_activity" << endl;
			}
			Activity.perform_activity(feedback, verbose_level);
			if (f_v) {
				cout << "combinatorial_object_stream::do_graph_theoretic_activity "
						"after Activity.perform_activity" << endl;
			}

			AO->Feedback.push_back(feedback);


		}

		FREE_int(Adj);
		FREE_OBJECT(CG);

	}

	if (f_v) {
		cout << "combinatorial_object_stream::do_graph_theoretic_activity done" << endl;
	}
}


void combinatorial_object_stream::do_algebraic_degree(
		projective_geometry::projective_space_with_action *PA,
		int *&Algebraic_degree, std::string *&Reduced_equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_algebraic_degree" << endl;
	}


	algebra::ring_theory::homogeneous_polynomial_domain *HPD;

	int n, q, m, i, qm1;

	n = PA->P->Subspaces->n + 1;
	q = PA->F->q;
	qm1 = q - 1;

	m = (q - 1) * n;
	if (f_v) {
		cout << "combinatorial_object_stream::do_algebraic_degree "
				"m = " << m << endl;
	}

	if (f_v) {
		cout << "combinatorial_object_stream::do_algebraic_degree "
				"we will now create the polynomial rings" << endl;
	}

	HPD = NEW_OBJECTS(algebra::ring_theory::homogeneous_polynomial_domain, m + 1);

	for (i = 1; i <= m; i++) {

		//Descr->;

		if (f_v) {
			cout << "combinatorial_object_stream::do_algebraic_degree "
					"before HPD->init i=" << i << endl;
		}
		HPD[i].init(
				PA->F,
				n /* nb_vars */, i /* degree */,
				t_PART,
				verbose_level - 2);

	}


	if (f_v) {
		for (i = 1; i <= m; i++) {
			cout << "degree " << i << ":" << endl;
			HPD[i].print_monomials(verbose_level);
		}
	}




	combinatorics::special_functions::special_functions_domain Special_functions_domain;

	if (f_v) {
		cout << "combinatorial_object_stream::do_algebraic_degree "
				"before Special_functions_domain.init" << endl;
	}

	Special_functions_domain.init(
			PA->P,
			verbose_level);

	if (f_v) {
		cout << "combinatorial_object_stream::do_algebraic_degree "
				"after Special_functions_domain.init" << endl;
	}




	int d;

	other::data_structures::int_matrix **Subspace_wgr;
	int *Subspace_wgr_rk;

	Subspace_wgr = (other::data_structures::int_matrix **) NEW_pvoid(m + 1);

	Subspace_wgr_rk = NEW_int(m + 1);
	Subspace_wgr_rk[0] = 0;

	if (f_v) {
		cout << "combinatorial_object_stream::do_algebraic_degree "
				"before computing subspaces with good reduction" << endl;
	}


	for (d = 1; d <= m; d++) {

		if (f_v) {
			cout << "combinatorial_object_stream::do_algebraic_degree "
					"before HPD[m].subspace_with_good_reduction, d=" << d << endl;
		}
		HPD[m].subspace_with_good_reduction(
				d /* degree */, qm1 /* modulus */,
				Subspace_wgr[d],
				verbose_level - 2);
		if (f_v) {
			cout << "combinatorial_object_stream::do_algebraic_degree "
					"after HPD[m].subspace_with_good_reduction" << endl;
		}

		Subspace_wgr_rk[d] = Subspace_wgr[d]->m;

	}


	if (f_v) {
		cout << "combinatorial_object_stream::do_algebraic_degree "
				"after computing subspaces with good reduction" << endl;
	}

	if (f_v) {
		cout << "combinatorial_object_stream::do_algebraic_degree Subspace_wgr_rk = ";
		Int_vec_print(cout, Subspace_wgr_rk, m + 1);
		cout << endl;
	}

	int input_idx;


	Reduced_equation = new string[IS->Objects.size()];
	Algebraic_degree = NEW_int(IS->Objects.size());

	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

		combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

		if (f_v) {
			cout << "combinatorial_object_stream::do_algebraic_degree "
					"input_idx =  " << input_idx << " / " << IS->Objects.size() << endl;
			cout << "combinatorial_object_stream::do_algebraic_degree "
					"the input set has size " << OwCF->sz << endl;
			cout << "combinatorial_object_stream::do_algebraic_degree "
					"the input set is: " << endl;
			Lint_vec_print(cout, OwCF->set, OwCF->sz);
			cout << endl;
		}


		std::string poly_rep;

		if (f_v) {
			cout << "combinatorial_object_stream::do_algebraic_degree "
					"before Special_functions_domain.make_polynomial_representation" << endl;
		}
		Special_functions_domain.make_polynomial_representation(
				OwCF->set, OwCF->sz,
				poly_rep,
				verbose_level - 1);
		if (f_v) {
			cout << "combinatorial_object_stream::do_algebraic_degree "
					"after Special_functions_domain.make_polynomial_representation" << endl;
		}

		if (f_v) {
			cout << "combinatorial_object_stream::do_algebraic_degree "
					"poly_rep = " << poly_rep << endl;
		}


		int *eqn;
		int eqn_size;
		std::string name_of_formula;
		std::string name_of_formula_tex;



		if (f_v) {
			cout << "combinatorial_object_stream::do_algebraic_degree "
					"before HPD[m].parse_equation_wo_parameters" << endl;
		}
		HPD[m].parse_equation_wo_parameters(
				name_of_formula,
				name_of_formula_tex,
				poly_rep /*equation_text*/,
				eqn, eqn_size,
				0 /*verbose_level - 5*/);
		if (f_v) {
			cout << "combinatorial_object_stream::do_algebraic_degree "
					"after HPD[m].parse_equation_wo_parameters" << endl;
		}
		if (f_v) {
			cout << "combinatorial_object_stream::do_algebraic_degree "
					"eqn = ";
			Int_vec_print(cout, eqn, eqn_size);
			cout << endl;
			HPD[m].print_equation(cout, eqn);
			cout << endl;
		}


		int *eqn_reduced;
		int *eqn_kernel;

		eqn_reduced = NEW_int(eqn_size);
		eqn_kernel = NEW_int(eqn_size);

		for (d = 1; d <= m; d++) {

			if (f_v) {
				cout << "combinatorial_object_stream::do_algebraic_degree "
						"testing if degree = " << d << endl;
			}
			if (f_v) {
				cout << "combinatorial_object_stream::do_algebraic_degree "
						"before HPD[m].test_potential_algebraic_degree" << endl;
			}

			if (HPD[m].test_potential_algebraic_degree(
				eqn, eqn_size,
				OwCF->set, OwCF->sz,
				d,
				Subspace_wgr[d],
				eqn_reduced,
				eqn_kernel,
				verbose_level - 2)) {

				if (f_v) {
					cout << "combinatorial_object_stream::do_algebraic_degree "
							"HPD[m].test_potential_algebraic_degree returns true, algebraic degree = " << d << endl;
				}
				break;
			}


			if (f_v) {
				cout << "combinatorial_object_stream::do_algebraic_degree "
						"HPD[m].test_potential_algebraic_degree returns false "
						"degree = " << d << " is not possible" << endl;
			}


		}

		int *eqn_reduced2;

		eqn_reduced2 = NEW_int(HPD[d].get_nb_monomials());


		cout << "input " << input_idx << " / " << IS->Objects.size() << " : ";
		Lint_vec_print(cout, OwCF->set, OwCF->sz);
		cout << " algebraic degree is " << d << endl;

		cout << "combinatorial_object_stream::do_algebraic_degree "
				"eqn_reduced=";
		Int_vec_print_fully(cout, eqn_reduced, eqn_size);
		cout << endl;
		cout << "eqn_size=" << eqn_size << endl;

		{
			string s;

			s = HPD[d].stringify_equation(eqn_reduced);

			cout << "combinatorial_object_stream::do_algebraic_degree "
					"eqn_reduced=" << s << endl;
		}


		if (d < m) {
			HPD[m].equation_reduce(
					q - 1 /*modulus*/,
					&HPD[d],
					eqn_reduced,
					eqn_reduced2,
					verbose_level - 1);

		}
		else {
			Int_vec_copy(eqn_reduced, eqn_reduced2, eqn_size);
		}

		cout << "combinatorial_object_stream::do_algebraic_degree "
				"eqn_reduced2=";
		HPD[d].print_equation(cout, eqn_reduced2);
		cout << endl;

		string s;

		s = HPD[d].stringify_equation(eqn_reduced2);

		cout << "combinatorial_object_stream::do_algebraic_degree "
				"eqn_reduced2=" << s << endl;

		Algebraic_degree[input_idx] = d;
		Reduced_equation[input_idx] = s;

		FREE_int(eqn_reduced);
		FREE_int(eqn_kernel);

	}


	cout << "combinatorial_object_stream::do_algebraic_degree Algebraic_degree=";
	Int_vec_print(cout, Algebraic_degree, IS->Objects.size());
	cout << endl;


	for (d = 1; d <= m; d++) {
		FREE_OBJECT(Subspace_wgr[d]);
	}
	FREE_pvoid((void **) Subspace_wgr);



	if (f_v) {
		cout << "combinatorial_object_stream::do_algebraic_degree done" << endl;
	}
}



void combinatorial_object_stream::make_polynomial_representation(
		projective_geometry::projective_space_with_action *PA,
		std::string *&Equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::make_polynomial_representation" << endl;
	}


	algebra::ring_theory::homogeneous_polynomial_domain *HPD;

	int n, q, m, i; // qm1;

	n = PA->P->Subspaces->n + 1;
	q = PA->F->q;
	//qm1 = q - 1;

	m = (q - 1) * n;
	if (f_v) {
		cout << "combinatorial_object_stream::do_algebraic_degree "
				"m = " << m << endl;
	}


	HPD = NEW_OBJECTS(algebra::ring_theory::homogeneous_polynomial_domain, m + 1);

	for (i = 1; i <= m; i++) {


		if (f_v) {
			cout << "combinatorial_object_stream::make_polynomial_representation "
					"before HPD->init i=" << i << endl;
		}
		HPD[i].init(
				PA->F,
				n /* nb_vars */, i /* degree */,
				t_PART,
				verbose_level - 2);

	}





	combinatorics::special_functions::special_functions_domain Special_functions_domain;

	if (f_v) {
		cout << "combinatorial_object_stream::make_polynomial_representation "
				"before Special_functions_domain.init" << endl;
	}

	Special_functions_domain.init(
			PA->P,
			verbose_level);

	if (f_v) {
		cout << "combinatorial_object_stream::make_polynomial_representation "
				"after Special_functions_domain.init" << endl;
	}




	int input_idx;


	Equation = new string[IS->Objects.size()];

	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

		combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

		if (f_v) {
			cout << "combinatorial_object_stream::make_polynomial_representation "
					"input_idx =  " << input_idx << " / " << IS->Objects.size() << endl;
			cout << "combinatorial_object_stream::make_polynomial_representation "
					"the input set has size " << OwCF->sz << endl;
			cout << "combinatorial_object_stream::make_polynomial_representation "
					"the input set is: " << endl;
			Lint_vec_print(cout, OwCF->set, OwCF->sz);
			cout << endl;
		}


		std::string poly_rep;

		if (f_v) {
			cout << "combinatorial_object_stream::make_polynomial_representation "
					"before Special_functions_domain.make_polynomial_representation" << endl;
		}
		Special_functions_domain.make_polynomial_representation(
				OwCF->set, OwCF->sz,
				poly_rep,
				verbose_level - 1);
		if (f_v) {
			cout << "combinatorial_object_stream::make_polynomial_representation "
					"after Special_functions_domain.make_polynomial_representation" << endl;
		}

		if (f_v) {
			cout << "combinatorial_object_stream::make_polynomial_representation "
					"poly_rep = " << poly_rep << endl;
			cout << "combinatorial_object_stream::make_polynomial_representation "
					"string length = " << poly_rep.length() << endl;
		}


		other::orbiter_kernel_system::file_io Fio;
		std::string fname;

		if (IS->Descr->f_label) {
			fname = IS->Descr->label_txt + "_" + std::to_string(input_idx) + "_polynomial.txt";
		}
		else {
			fname = "object_" + std::to_string(input_idx) + "_polynomial.txt";
		}
		{
			std::ofstream ost(fname);

			ost << poly_rep << endl;
		}
		cout << "combinatorial_object_stream::make_polynomial_representation "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;


		Equation[input_idx] = poly_rep;

		int *eqn;
		int eqn_size;
		std::string name_of_formula;
		std::string name_of_formula_tex;



		if (f_v) {
			cout << "combinatorial_object_stream::make_polynomial_representation "
					"before HPD[m].parse_equation_wo_parameters" << endl;
		}
		HPD[m].parse_equation_wo_parameters(
				name_of_formula,
				name_of_formula_tex,
				poly_rep /*equation_text*/,
				eqn, eqn_size,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_stream::make_polynomial_representation "
					"after HPD[m].parse_equation_wo_parameters" << endl;
		}
		if (f_v) {
			cout << "combinatorial_object_stream::make_polynomial_representation "
					"eqn = ";
			Int_vec_print(cout, eqn, eqn_size);
			cout << endl;
			HPD[m].print_equation(cout, eqn);
			cout << endl;
		}


	}





	if (f_v) {
		cout << "combinatorial_object_stream::make_polynomial_representation done" << endl;
	}
}




}}}


