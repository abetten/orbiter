/*
 * combinatorial_object_activity.cpp
 *
 *  Created on: Mar 20, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {



combinatorial_object_activity::combinatorial_object_activity()
{
	Descr = NULL;

	f_has_geometric_object = FALSE;
	GOC = NULL;

	f_has_input_stream = FALSE;
	IS = NULL;

}

combinatorial_object_activity::~combinatorial_object_activity()
{
}


void combinatorial_object_activity::init(
		combinatorial_object_activity_description *Descr,
		geometry::geometric_object_create *GOC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::init" << endl;
	}

	combinatorial_object_activity::Descr = Descr;
	f_has_geometric_object = TRUE;
	combinatorial_object_activity::GOC = GOC;

	if (f_v) {
		cout << "combinatorial_object_activity::init done" << endl;
	}
}



void combinatorial_object_activity::init_input_stream(
		combinatorial_object_activity_description *Descr,
		data_structures::data_input_stream *IS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::init_input_stream" << endl;
	}

	combinatorial_object_activity::Descr = Descr;
	f_has_input_stream = TRUE;
	combinatorial_object_activity::IS = IS;


	if (f_v) {
		cout << "combinatorial_object_activity::init_input_stream done" << endl;
	}
}

void combinatorial_object_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity" << endl;
	}
	if (f_has_geometric_object) {
		perform_activity_geometric_object(verbose_level);
	}
	else if (f_has_input_stream) {
		perform_activity_input_stream(verbose_level);
	}
	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity done" << endl;
	}
}


void combinatorial_object_activity::perform_activity_geometric_object(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity_geometric_object" << endl;
	}

	if (Descr->f_line_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object f_line_type" << endl;
		}

		geometry::projective_space *P;

		P = GOC->Descr->P;

		int *type;

		type = NEW_int(P->N_lines);


		P->line_intersection_type(
				GOC->Pts, GOC->nb_pts, type, 0 /* verbose_level */);
			// type[N_lines]


		data_structures::tally T;

		T.init(type, P->N_lines, FALSE, 0);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object line type:" << endl;
			T.print(TRUE /* f_backwards*/);
			cout << endl;
		}


	}

	if (Descr->f_conic_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object f_conic_type" << endl;
		}

		geometry::projective_space *P;

		P = GOC->Descr->P;

		long int **Pts_on_conic;
		int **Conic_eqn;
		int *nb_pts_on_conic;
		int len;
		int i;

		P->conic_type(
				GOC->Pts, GOC->nb_pts,
				Descr->conic_type_threshold,
				Pts_on_conic, Conic_eqn, nb_pts_on_conic, len,
				verbose_level);


		cout << "We found " << len << " conics" << endl;
		for (i = 0; i < len; i++) {
			cout << i << " : " << nb_pts_on_conic << endl;
		}
		//

	}

	if (Descr->f_non_conical_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object f_conic_type" << endl;
		}

		geometry::projective_space *P;

		P = GOC->Descr->P;

		std::vector<int> Rk;

		P->determine_nonconical_six_subsets(
				GOC->Pts, GOC->nb_pts,
				Rk,
				verbose_level);

		cout << "We found " << Rk.size() << " non-conical 6 subsets" << endl;

	}


	if (Descr->f_ideal) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object f_ideal" << endl;
		}

		ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = orbiter_kernel_system::Orbiter->get_object_of_type_polynomial_ring(Descr->ideal_ring_label);


		HPD->explore_vanishing_ideal(GOC->Pts, GOC->nb_pts, verbose_level);

#if 0
		int *Kernel;
		int r;

		Kernel = NEW_int(HPD->get_nb_monomials() * HPD->get_nb_monomials());


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_GOC the input set has size " << GOC->nb_pts << endl;
			cout << "combinatorial_object_activity::perform_activity_GOC the input set is: " << endl;
			Lint_vec_print(cout, GOC->Pts, GOC->nb_pts);
			cout << endl;
			//P->print_set_numerical(cout, GOC->Pts, GOC->nb_pts);
		}


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_GOC before HPD->vanishing_ideal" << endl;
		}
		HPD->vanishing_ideal(GOC->Pts, GOC->nb_pts,
				r, Kernel, verbose_level - 1);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_GOC after HPD->vanishing_ideal" << endl;
		}


		int h; //, ns;
		int nb_pts;
		long int *Pts;

		for (h = 0; h < r; h++) {
			cout << "generator " << h << " / " << r << " is ";
			HPD->print_equation_relaxed(cout, Kernel + h * HPD->get_nb_monomials());
			cout << endl;

		}

		//ns = HPD->get_nb_monomials() - r; // dimension of null space

		cout << "looping over all generators of the ideal:" << endl;
		for (h = 0; h < r; h++) {
			cout << "generator " << h << " / " << r << " is ";
			Int_vec_print(cout, Kernel + h * HPD->get_nb_monomials(), HPD->get_nb_monomials());
			cout << " : " << endl;

			vector<long int> Points;
			int i;

			HPD->enumerate_points(Kernel + h * HPD->get_nb_monomials(),
					Points, verbose_level);
			nb_pts = Points.size();

			Pts = NEW_lint(nb_pts);
			for (i = 0; i < nb_pts; i++) {
				Pts[i] = Points[i];
			}


			cout << "We found " << nb_pts << " points on the generator of the ideal" << endl;
			cout << "They are : ";
			Lint_vec_print(cout, Pts, nb_pts);
			cout << endl;
			//P->print_set_numerical(cout, Pts, nb_pts);

			FREE_lint(Pts);

		} // next h
#endif

	}


	if (Descr->f_save) {

		orbiter_kernel_system::file_io Fio;
		string fname;

		fname.assign(GOC->label_txt);
		fname.append(".txt");

		if (f_v) {
			cout << "We will write to the file " << fname << endl;
		}
		Fio.write_set_to_file(fname, GOC->Pts, GOC->nb_pts, verbose_level);
		if (f_v) {
			cout << "Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity_geometric_object done" << endl;
	}
}


void combinatorial_object_activity::perform_activity_input_stream(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity_input_stream" << endl;
	}

	if (Descr->f_canonical_form_PG) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream f_canonical_form_PG" << endl;
		}

		projective_geometry::projective_space_with_action *PA;


		if (Descr->f_canonical_form_PG_has_PA) {
			PA = Descr->Canonical_form_PG_PA;
		}
		else {

			PA = Get_object_of_projective_space(Descr->canonical_form_PG_PG_label);
		}

		combinatorics::classification_of_objects *CO;

		CO = NEW_OBJECT(combinatorics::classification_of_objects);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"before CO->perform_classification" << endl;
		}
		CO->perform_classification(
				Descr->Canonical_form_PG_Descr,
				TRUE /* f_projective_space */, PA->P,
				IS,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"after CO->perform_classification" << endl;
		}



		object_with_properties *OwP;

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"before post_process_classification" << endl;
		}
		post_process_classification(
					CO,
					OwP,
					TRUE /* f_projective_space */, PA,
					CO->Descr->label,
					verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"after post_process_classification" << endl;
		}

		if (Descr->f_report) {
			if (f_v) {
				cout << "combinatorial_object_activity::perform_activity_input_stream "
						"before classification_report" << endl;
			}
			classification_report(
					CO,
					OwP, verbose_level);
			if (f_v) {
				cout << "combinatorial_object_activity::perform_activity_input_stream "
						"after classification_report" << endl;
			}
		}

		FREE_OBJECTS(OwP);
		FREE_OBJECT(CO);



	}
	else if (Descr->f_canonical_form) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"f_canonical_form" << endl;
		}



		combinatorics::classification_of_objects *CO;

		CO = NEW_OBJECT(combinatorics::classification_of_objects);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"before CO->perform_classification" << endl;
		}
		CO->perform_classification(
				Descr->Canonical_form_Descr,
				FALSE /* f_projective_space */, NULL /* P */,
				IS,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"after CO->perform_classification" << endl;
		}


		object_with_properties *OwP;

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"before post_process_classification" << endl;
		}
		post_process_classification(
					CO,
					OwP,
					FALSE /* f_projective_space */, NULL,
					CO->Descr->label,
					verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"after post_process_classification" << endl;
		}

		if (Descr->f_report) {
			if (f_v) {
				cout << "combinatorial_object_activity::perform_activity_input_stream "
						"before classification_report" << endl;
			}
			classification_report(
					CO,
					OwP, verbose_level);
			if (f_v) {
				cout << "combinatorial_object_activity::perform_activity_input_stream "
						"after classification_report" << endl;
			}
		}

		FREE_OBJECTS(OwP);
		FREE_OBJECT(CO);



	}
	else if (Descr->f_draw_incidence_matrices) {
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"f_draw_incidence_matrices" << endl;
		}
		draw_incidence_matrices(
				Descr->draw_incidence_matrices_prefix,
				IS,
				verbose_level);

	}
	else if (Descr->f_test_distinguishing_property) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"f_test_distinguishing_property" << endl;
		}

		int idx;

		idx = orbiter_kernel_system::Orbiter->find_symbol(Descr->test_distinguishing_property_graph);

		symbol_table_object_type t;

		t = orbiter_kernel_system::Orbiter->get_object_type(idx);
		if (t != t_graph) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
				<< Descr->test_distinguishing_property_graph << " is not of type graph" << endl;
			exit(1);
		}

		//create_graph *Gr;
		graph_theory::colored_graph *CG;

		//Gr = (create_graph *) Orbiter->get_object(idx);
		CG = (graph_theory::colored_graph *) orbiter_kernel_system::Orbiter->get_object(idx);

#if 0
		if (!Gr->f_has_CG) {
			cout << "combinatorial_object_activity::perform_activity_input_stream !Gr->f_has_CG" << endl;
			exit(1);
		}
#endif
		int input_idx;
		int *F_distinguishing;

		F_distinguishing = NEW_int(IS->Objects.size());


		for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) IS->Objects[input_idx];

			F_distinguishing[input_idx] = CG->test_distinguishing_property(OwCF->set, OwCF->sz, verbose_level);
		}

		data_structures::tally T;

		T.init(F_distinguishing, IS->Objects.size(), FALSE, 0);
		cout << "classification : ";
		T.print_first(TRUE /* f_backwards*/);
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

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) IS->Objects[input_idx];

			OwCF->print(cout);

		}
		cout << endl;


		FREE_int(F_distinguishing);

	}
	if (Descr->f_ideal) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream f_ideal" << endl;
		}

		ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = orbiter_kernel_system::Orbiter->get_object_of_type_polynomial_ring(Descr->ideal_ring_label);


		int input_idx;


		for (input_idx = 0; input_idx < IS->Objects.size(); input_idx++) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) IS->Objects[input_idx];

			HPD->explore_vanishing_ideal(OwCF->set, OwCF->sz, verbose_level);

		}

	}
	else if (Descr->f_save_as) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"f_save_as " << Descr->save_as_fname << endl;
		}

		do_save(Descr->save_as_fname,
				FALSE, NULL, 0,
				verbose_level);


	}
	else if (Descr->f_extract_subset) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"f_extract_subset " << Descr->extract_subset_fname << endl;
		}
		long int *extract_idx_set;
		int extract_size;

		orbiter_kernel_system::Orbiter->get_lint_vector_from_label(Descr->extract_subset_set,
				extract_idx_set, extract_size, 0 /* verbose_level */);
		do_save(Descr->extract_subset_fname,
				TRUE /* f_extract */, extract_idx_set, extract_size,
				verbose_level);


	}
	else if (Descr->f_unpack_from_restricted_action) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"unpack_from_restricted_action "
					<< Descr->unpack_from_restricted_action_prefix
					<< " " << Descr->unpack_from_restricted_action_group_label << endl;
		}

		unpack_from_restricted_action(
				Descr->unpack_from_restricted_action_prefix,
				Descr->unpack_from_restricted_action_group_label,
				IS,
				verbose_level);

	}

	else if (Descr->f_line_covering_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"line_covering_type "
					<< Descr->line_covering_type_prefix
					<< " " << Descr->line_covering_type_projective_space
					<< " " << Descr->line_covering_type_lines
					<< endl;
		}

		line_covering_type(
				Descr->line_covering_type_prefix,
				Descr->line_covering_type_projective_space,
				Descr->line_covering_type_lines,
				IS,
				verbose_level);

	}
	else if (Descr->f_line_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_input_stream "
					"f_line_type" << endl;
		}


		line_type(
					Descr->line_type_prefix,
					Descr->line_type_projective_space_label,
					IS,
					verbose_level);



	}



}


void combinatorial_object_activity::do_save(std::string &save_as_fname,
		int f_extract, long int *extract_idx_set, int extract_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::do_save" << endl;
	}
	int input_idx;
	int sz;
	int N;

	N = IS->Objects.size();

	geometry::object_with_canonical_form *OwCF;

	OwCF = (geometry::object_with_canonical_form *) IS->Objects[0];

	//OwCF->set;
	sz = OwCF->sz;

	for (input_idx = 0; input_idx < N; input_idx++) {

		if (FALSE) {
			cout << "combinatorial_object_activity::perform_activity_IS "
					"input_idx = " << input_idx << " / " << IS->Objects.size() << endl;
		}

		geometry::object_with_canonical_form *OwCF;

		OwCF = (geometry::object_with_canonical_form *) IS->Objects[input_idx];

		//OwCF->set;
		if (OwCF->sz != sz) {
			cout << "the objects have different sizes, cannot save" << endl;
			exit(1);
		}


	}

	long int *Sets;

	Sets = NEW_lint(N * sz);

	for (input_idx = 0; input_idx < N; input_idx++) {
		geometry::object_with_canonical_form *OwCF;

		OwCF = (geometry::object_with_canonical_form *) IS->Objects[input_idx];

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
			cout << "number of sets is reduced from " << N << " to " << extract_size << endl;
		}
		N = extract_size;
	}
	else {

	}
	orbiter_kernel_system::file_io Fio;

	string fname_out;

	fname_out.assign(save_as_fname);

	Fio.lint_matrix_write_csv(fname_out, Sets, N, sz);

	cout << "Written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;

	if (f_v) {
		cout << "combinatorial_object_activity::do_save done" << endl;
	}
}

void combinatorial_object_activity::post_process_classification(
		combinatorics::classification_of_objects *CO,
		object_with_properties *&OwP,
		int f_projective_space,
		projective_geometry::projective_space_with_action *PA,
		std::string &prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::post_process_classification" << endl;
	}


	OwP = NEW_OBJECTS(object_with_properties, CO->nb_orbits);

	int iso_type;


	for (iso_type = 0; iso_type < CO->nb_orbits; iso_type++) {

		cout << "iso_type = " << iso_type << " / " << CO->nb_orbits << endl;
		cout << "NO=" << endl;
		CO->NO_transversal[iso_type]->print();

		std::string label;
		char str[1000];

		snprintf(str, sizeof(str), "_object%d", iso_type);
		label.assign(prefix);
		label.append(str);

		OwP[iso_type].init(
				CO->OWCF_transversal[iso_type],
				CO->NO_transversal[iso_type],
				f_projective_space, PA,
				CO->Descr->max_TDO_depth,
				label,
				verbose_level);


	}

	if (f_v) {
		cout << "combinatorial_object_activity::post_process_classification done" << endl;
	}
}

void combinatorial_object_activity::classification_report(
		combinatorics::classification_of_objects *CO,
		object_with_properties *OwP,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::classification_report" << endl;
	}

	combinatorics::classification_of_objects_report_options *Report_options;

	if (CO->Descr->f_classification_prefix == FALSE) {
		cout << "please use option -classification_prefix <prefix> to set the "
				"prefix for the output file" << endl;
		exit(1);
	}

	Report_options = Descr->Classification_of_objects_report_options;

	if (f_v) {
		cout << "combinatorial_object_activity::classification_report before latex_report" << endl;
	}
	latex_report(Report_options,
			CO,
			OwP,
			verbose_level);

	if (f_v) {
		cout << "combinatorial_object_activity::classification_report after latex_report" << endl;
	}

}

void combinatorial_object_activity::latex_report(
		combinatorics::classification_of_objects_report_options *Report_options,
		combinatorics::classification_of_objects *CO,
		object_with_properties *OwP,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	orbiter_kernel_system::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_activity::latex_report" << endl;
	}



	string fname;

	fname.assign(Report_options->prefix);
	fname.append("_classification.tex");

	if (f_v) {
		cout << "combinatorial_object_activity::classification_report before latex_report" << endl;
	}



	if (f_v) {
		cout << "combinatorial_object_activity::latex_report, CB->nb_types=" << CO->CB->nb_types << endl;
	}
	{
		ofstream fp(fname);
		orbiter_kernel_system::latex_interface L;

		L.head_easy(fp);


		CO->report_summary_of_orbits(fp, verbose_level);


		fp << "Ago : ";
		CO->T_Ago->print_file_tex(fp, FALSE /* f_backwards*/);
		fp << "\\\\" << endl;

		if (f_v) {
			cout << "combinatorial_object_activity::latex_report before loop" << endl;
		}


		report_all_isomorphism_types(fp, Report_options, CO, OwP,
				verbose_level);

		L.foot(fp);
	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	//FREE_int(perm);
	//FREE_int(v);
	if (f_v) {
		cout << "combinatorial_object_activity::latex_report done" << endl;
	}
}

void combinatorial_object_activity::report_all_isomorphism_types(
		std::ostream &fp,
		combinatorics::classification_of_objects_report_options *Report_options,
		combinatorics::classification_of_objects *CO,
		object_with_properties *OwP,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::report_all_isomorphism_types" << endl;
	}
	int i;

	orbiter_kernel_system::latex_interface L;

	for (i = 0; i < CO->CB->nb_types; i++) {

		fp << "\\section*{Isomorphism type " << i << " / " << CO->CB->nb_types << "}" << endl;
		fp << "Isomorphism type " << i << " / " << CO->CB->nb_types
			//<<  " stored at " << j
			<< " is original object "
			<< CO->CB->Type_rep[i] << " and appears "
			<< CO->CB->Type_mult[i] << " times: \\\\" << endl;

		{
			data_structures::sorting Sorting;
			int *Input_objects;
			int nb_input_objects;
			CO->CB->C_type_of->get_class_by_value(Input_objects,
					nb_input_objects, i, 0 /*verbose_level */);
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
				fp << "Too big to print. \\\\" << endl;
#if 0
				fp << "$$" << endl;
				L.int_vec_print_as_matrix(fp, Input_objects,
					nb_input_objects, 10 /* width */, TRUE /* f_tex */);
				fp << "$$" << endl;
#endif
			}

			FREE_int(Input_objects);
		}

		if (f_v) {
			cout << "combinatorial_object_activity::report_all_isomorphism_types before report_isomorphism_type" << endl;
		}
		report_isomorphism_type(fp, Report_options, CO, OwP, i, verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::report_all_isomorphism_types after report_isomorphism_type" << endl;
		}


	} // next i
	if (f_v) {
		cout << "combinatorial_object_activity::report_all_isomorphism_types done" << endl;
	}

}


void combinatorial_object_activity::report_isomorphism_type(
		std::ostream &fp,
		combinatorics::classification_of_objects_report_options *Report_options,
		combinatorics::classification_of_objects *CO,
		object_with_properties *OwP,
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::report_isomorphism_type i=" << i << endl;
	}
	int j;
	orbiter_kernel_system::latex_interface L;

	//j = CB->perm[i];
	//j = CB->Type_rep[i];
	j = i;

	cout << "###################################################"
			"#############################" << endl;
	cout << "Orbit " << i << " / " << CO->CB->nb_types
			<< " is canonical form no " << j
			<< ", original object no " << CO->CB->Type_rep[i]
			<< ", frequency " << CO->CB->Type_mult[i]
			<< " : " << endl;


	{
		int *Input_objects;
		int nb_input_objects;
		CO->CB->C_type_of->get_class_by_value(Input_objects,
			nb_input_objects, j, 0 /*verbose_level */);

		cout << "This isomorphism type appears " << nb_input_objects
				<< " times, namely for the following "
						"input objects:" << endl;
		if (nb_input_objects < 10) {
			L.int_vec_print_as_matrix(cout, Input_objects,
					nb_input_objects, 10 /* width */,
					FALSE /* f_tex */);
		}
		else {
			cout << "too many to print" << endl;
		}

		FREE_int(Input_objects);
	}



	if (f_v) {
		cout << "combinatorial_object_activity::report_isomorphism_type i=" << i << " before report_object" << endl;
	}
	report_object(fp,
			Report_options,
			CO,
			OwP,
			i /* object_idx */,
			verbose_level);
	if (f_v) {
		cout << "combinatorial_object_activity::report_isomorphism_type i=" << i << " after report_object" << endl;
	}




	if (f_v) {
		cout << "combinatorial_object_activity::report_isomorphism_type i=" << i << " done" << endl;
	}
}

void combinatorial_object_activity::report_object(std::ostream &fp,
		combinatorics::classification_of_objects_report_options *Report_options,
		combinatorics::classification_of_objects *CO,
		object_with_properties *OwP,
		int object_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::report_object" << endl;
	}


	geometry::object_with_canonical_form *OwCF = CO->OWCF_transversal[object_idx];

	OwCF->print_tex_detailed(fp, Report_options->f_show_incidence_matrices, verbose_level);

	if (FALSE /*CO->f_projective_space*/) {

#if 0
		object_in_projective_space_with_action *OiPA;

		OiPA = (object_in_projective_space_with_action *)
				CB->Type_extra_data[object_idx];

		OiPA->report(fp, PA, max_TDO_depth, verbose_level);
#endif

	}
	else {
		OwP[object_idx].latex_report(fp, Report_options, verbose_level);
	}



}

void combinatorial_object_activity::draw_incidence_matrices(
		std::string &prefix,
		data_structures::data_input_stream *IS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	orbiter_kernel_system::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_activity::draw_incidence_matrices" << endl;
	}



	string fname;

	fname.assign(prefix);
	fname.append("_incma.tex");

	if (f_v) {
		cout << "combinatorial_object_activity::draw_incidence_matrices before latex_report" << endl;
	}


	{
		ofstream ost(fname);
		orbiter_kernel_system::latex_interface L;

		L.head_easy(ost);


		int N;

		N = IS->Objects.size();



		if (f_v) {
			cout << "combinatorial_object_activity::draw_incidence_matrices before loop" << endl;
		}

		int i;

		ost << "\\noindent" << endl;

		for (i = 0; i < N; i++) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) IS->Objects[i];


			combinatorics::encoded_combinatorial_object *Enc;

			OwCF->encode_incma(Enc, verbose_level);

			//Enc->latex_set_system_by_columns(ost, verbose_level);

			//Enc->latex_set_system_by_rows(ost, verbose_level);

			Enc->latex_incma(ost, verbose_level);

			FREE_OBJECT(Enc);


		}



		L.foot(ost);
	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "combinatorial_object_activity::draw_incidence_matrices done" << endl;
	}
}

void combinatorial_object_activity::unpack_from_restricted_action(
		std::string &prefix,
		std::string &group_label,
		data_structures::data_input_stream *IS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	orbiter_kernel_system::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_activity::unpack_from_restricted_action" << endl;
	}


	apps_algebra::any_group *G;

	G = Get_object_of_type_any_group(group_label);
	groups::linear_group *LG;

	LG = G->LG;


	if (!G->f_linear_group) {
		cout << "combinatorial_object_activity::unpack_from_restricted_action must be a linear group" << endl;
		exit(1);
	}

	if (LG->A2->type_G != action_by_restriction_t) {
		cout << "combinatorial_object_activity::unpack_from_restricted_action must be a restricted action" << endl;
		exit(1);
	}
	induced_actions::action_by_restriction *ABR;
	ABR = LG->A2->G.ABR;


	string fname;

	fname.assign(prefix);
	fname.append("_unpacked.txt");

	if (f_v) {
		cout << "combinatorial_object_activity::unpack_from_restricted_action before latex_report" << endl;
	}


	{

		ofstream ost(fname);

		int N;

		N = IS->Objects.size();



		if (f_v) {
			cout << "combinatorial_object_activity::unpack_from_restricted_action before loop" << endl;
		}

		int i, h;
		long int a, b;


		for (i = 0; i < N; i++) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) IS->Objects[i];


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
		cout << "combinatorial_object_activity::unpack_from_restricted_action done" << endl;
	}
}


void combinatorial_object_activity::line_covering_type(
		std::string &prefix,
		std::string &projective_space_label,
		std::string &lines,
		data_structures::data_input_stream *IS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	orbiter_kernel_system::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_activity::line_covering_type" << endl;
	}

	projective_geometry::projective_space_with_action *PA;

	PA = Get_object_of_projective_space(projective_space_label);

	geometry::projective_space *P;

	P = PA->P;

	long int *the_lines;
	int nb_lines;

	orbiter_kernel_system::Orbiter->get_lint_vector_from_label(lines, the_lines, nb_lines, verbose_level);

	string fname;

	fname.assign(prefix);
	fname.append("_line_covering_type.txt");

	if (f_v) {
		cout << "combinatorial_object_activity::line_covering_type before latex_report" << endl;
	}


	{

		ofstream ost(fname);

		int N;

		N = IS->Objects.size();



		if (f_v) {
			cout << "combinatorial_object_activity::line_covering_type before loop" << endl;
		}

		int i, h;

		int *type;

		type = NEW_int(nb_lines);


		for (i = 0; i < N; i++) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) IS->Objects[i];


			P->line_intersection_type_basic_given_a_set_of_lines(
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
		cout << "combinatorial_object_activity::line_covering_type done" << endl;
	}
}

void combinatorial_object_activity::line_type(
		std::string &prefix,
		std::string &projective_space_label,
		data_structures::data_input_stream *IS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	orbiter_kernel_system::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_activity::line_type" << endl;
	}

	projective_geometry::projective_space_with_action *PA;

	PA = Get_object_of_projective_space(projective_space_label);

	geometry::projective_space *P;

	P = PA->P;

	string fname;

	fname.assign(prefix);
	fname.append("_line_type.txt");

	if (f_v) {
		cout << "combinatorial_object_activity::line_type before latex_report" << endl;
	}


	{

		ofstream ost(fname);

		int N;

		N = IS->Objects.size();



		if (f_v) {
			cout << "combinatorial_object_activity::line_type before loop" << endl;
		}

		int i, h;

		int *type;

		type = NEW_int(P->N_lines);


		for (i = 0; i < N; i++) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) IS->Objects[i];


			P->line_intersection_type(
					OwCF->set, OwCF->sz, type, 0 /* verbose_level */);
				// type[N_lines]

			ost << OwCF->sz;
			for (h = 0; h < P->N_lines; h++) {
				ost << " " << type[h];
			}
			ost << endl;

#if 1
			data_structures::tally T;

			T.init(type, P->N_lines, FALSE, 0);

			if (f_v) {
				cout << "combinatorial_object_activity::perform_activity_GOC line type:" << endl;
				T.print(TRUE /* f_backwards*/);
				cout << endl;
			}

			std::string fname_line_type;
			char str[1000];

			snprintf(str, sizeof(str), "_line_type_set_partition_%d.csv", i);

			fname_line_type.assign(prefix);
			fname_line_type.append(str);

			T.save_classes_individually(fname_line_type);
			if (TRUE) {
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
		cout << "combinatorial_object_activity::line_type done" << endl;
	}
}





}}}




