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
	//Data_input_stream_description = NULL;

	IS = NULL;

	Classification_of_objects = NULL;
	Objects_after_classification = NULL;

}

combinatorial_object_stream::~combinatorial_object_stream()
{
	if (Classification_of_objects) {
		FREE_OBJECT(Classification_of_objects);
	}
	if (Objects_after_classification) {
		FREE_OBJECT(Objects_after_classification);
	}
}


void combinatorial_object_stream::init(
		canonical_form_classification::data_input_stream_description
				*Data_input_stream_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::init" << endl;
	}
	//combinatorial_object::Data_input_stream_description = Data_input_stream_description;


	IS = NEW_OBJECT(canonical_form_classification::data_input_stream);

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
		canonical_form_classification::classification_of_objects_description
				*Canonical_form_Descr,
		int f_projective_space,
		projective_geometry::projective_space_with_action *PA,
		geometry::projective_space *P,
		int verbose_level)
// called from combinatorial_object_activity::perform_activity_combo
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::geometry::projective_space *P" << endl;
	}



	Classification_of_objects = NEW_OBJECT(canonical_form_classification::classification_of_objects);

	if (f_v) {
		cout << "combinatorial_object_stream::geometry::projective_space *P "
				"before Classification_of_objects->perform_classification" << endl;
	}
	Classification_of_objects->perform_classification(
			Canonical_form_Descr,
			true /* f_projective_space */, P,
			IS,
			verbose_level);
	if (f_v) {
		cout << "combinatorial_object_stream::geometry::projective_space *P "
				"after Classification_of_objects->perform_classification" << endl;
	}


	Objects_after_classification = NEW_OBJECT(canonical_form::objects_after_classification);

	if (!IS->Descr->f_label) {
		cout << "please use -label <label_txt> <label_tex> "
				"when defining the input stream for the combinatorial object" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "combinatorial_object_stream::geometry::projective_space *P "
				"before Objects_after_classification->init_after_nauty" << endl;
	}
	Objects_after_classification->init_after_nauty(
			Classification_of_objects,
			f_projective_space, PA,
			verbose_level);
	if (f_v) {
		cout << "combinatorial_object_stream::geometry::projective_space *P "
				"after Objects_after_classification->init_after_nauty" << endl;
	}


	if (f_v) {
		cout << "combinatorial_object_stream::geometry::projective_space *P done" << endl;
	}

}

#if 0
void combinatorial_object_stream::do_canonical_form_not_PG(
		canonical_form_classification::classification_of_objects_description
			*Canonical_form_Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_canonical_form_not_PG" << endl;
	}

	Classification = NEW_OBJECT(canonical_form_classification::classification_of_objects);

	if (f_v) {
		cout << "combinatorial_object_stream::do_canonical_form_not_PG "
				"before Classification->perform_classification" << endl;
	}
	Classification->perform_classification(
			Canonical_form_Descr,
			false /* f_projective_space */, NULL /* P */,
			IS,
			verbose_level);
	if (f_v) {
		cout << "combinatorial_object_stream::do_canonical_form_not_PG "
				"after Combo->Classification->perform_classification" << endl;
	}


	Objects_after_classification = NEW_OBJECT(canonical_form::objects_after_classification);

	if (f_v) {
		cout << "combinatorial_object_stream::do_canonical_form_not_PG "
				"before Objects_after_classification->init_after_nauty" << endl;
	}
	Objects_after_classification->init_after_nauty(
			Classification,
			false /* f_projective_space */, NULL,
			verbose_level);
	if (f_v) {
		cout << "combinatorial_object_stream::do_canonical_form_not_PG "
				"after Objects_after_classification->init_after_nauty" << endl;
	}


	if (f_v) {
		cout << "combinatorial_object_stream::do_canonical_form_not_PG done" << endl;
	}

}
#endif

void combinatorial_object_stream::do_test_distinguishing_property(
		graph_theory::colored_graph *CG,
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

		canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

		F_distinguishing[input_idx] = CG->test_distinguishing_property(
				OwCF->set, OwCF->sz, verbose_level);
	}

	data_structures::tally T;

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

		canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

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
		orbiter_kernel_system::activity_output *&AO,
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

	AO = NEW_OBJECT(orbiter_kernel_system::activity_output);
	//AO->nb_rows = IS->Objects.size();

	AO->headings = "set,idx,covering_wrt" + std::to_string(subset_sz) + ",steiner_p";
	AO->nb_cols = 4;
	AO->description_txt = "covering_wrt" + std::to_string(subset_sz);


	combinatorics::combinatorics_domain Combi;

	int nCk;

	if (IS->Objects.size() == 0) {

		if (f_v) {
			cout << "combinatorial_object_stream::do_covering_type "
					"breaking off early because there is no input" << endl;
		}
		return;
	}

	if (f_v) {
		cout << "combinatorial_object_stream::do_covering_type IS->Objects.size() = " << IS->Objects.size() << endl;

	}


	canonical_form_classification::any_combinatorial_object *OwCF;

	OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[0];


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

	data_structures::algorithms Algorithms;


	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

		canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];


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
		graph_theory::colored_graph *CG,
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

		canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

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

		data_structures::tally T;

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
		ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_compute_ideal" << endl;
	}




	int input_idx;


	for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

		canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

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

	canonical_form_classification::any_combinatorial_object *OwCF;

	OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[0];

	//OwCF->set;
	sz = OwCF->sz;

	for (input_idx = 0; input_idx < N; input_idx++) {

		if (false) {
			cout << "combinatorial_object_stream::do_save "
					"input_idx = " << input_idx
					<< " / " << IS->Objects.size() << endl;
		}

		canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (canonical_form_classification::any_combinatorial_object *)
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
		canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

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
	orbiter_kernel_system::file_io Fio;

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
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	l1_interfaces::latex_interface L;

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
		l1_interfaces::latex_interface L;

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
			canonical_form_classification::any_combinatorial_object *OwCF;

			OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[i];


			canonical_form_classification::encoded_combinatorial_object *Enc;

			if (f_v) {
				cout << "combinatorial_object_stream::draw_incidence_matrices "
						"before OwCF->encode_incma" << endl;
			}
			OwCF->encode_incma(Enc, verbose_level);
			if (f_v) {
				cout << "combinatorial_object_stream::draw_incidence_matrices "
						"after OwCF->encode_incma" << endl;
			}

			//Enc->latex_set_system_by_columns(ost, verbose_level);

			//Enc->latex_set_system_by_rows(ost, verbose_level);

			if (f_v) {
				cout << "combinatorial_object_stream::draw_incidence_matrices "
						"before OwCF->latex_incma" << endl;
			}
			Enc->latex_incma(ost, verbose_level);
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
		apps_algebra::any_group *G,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	l1_interfaces::latex_interface L;

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

			canonical_form_classification::any_combinatorial_object *OwCF;

			OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[i];


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

	orbiter_kernel_system::file_io Fio;
	l1_interfaces::latex_interface L;

	geometry::projective_space *P;

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

			canonical_form_classification::any_combinatorial_object *OwCF;

			OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[i];


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
	orbiter_kernel_system::file_io Fio;
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_stream::line_type" << endl;
	}


	geometry::projective_space *P;

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

			canonical_form_classification::any_combinatorial_object *OwCF;

			OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[i];


			P->Subspaces->line_intersection_type(
					OwCF->set, OwCF->sz, type, 0 /* verbose_level */);
				// type[N_lines]

			ost << OwCF->sz;
			for (h = 0; h < P->Subspaces->N_lines; h++) {
				ost << " " << type[h];
			}
			ost << endl;

#if 1
			data_structures::tally T;

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
		orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
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

	if (f_v) {
		cout << "combinatorial_object_stream::do_activity done" << endl;
	}
}

void combinatorial_object_stream::do_graph_theoretic_activity(
		apps_graph_theory::graph_theoretic_activity_description
				*Graph_theoretic_activity_description,
		orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_stream::do_graph_theoretic_activity" << endl;
	}


	AO = NEW_OBJECT(orbiter_kernel_system::activity_output);
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

		canonical_form_classification::any_combinatorial_object *OwCF;

		OwCF = (canonical_form_classification::any_combinatorial_object *) IS->Objects[input_idx];

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

		graph_theory::colored_graph *CG;

		CG = NEW_OBJECT(graph_theory::colored_graph);
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






}}}


