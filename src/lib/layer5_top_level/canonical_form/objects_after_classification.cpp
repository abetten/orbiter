/*
 * objects_after_classification.cpp
 *
 *  Created on: Jul 9, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {




objects_after_classification::objects_after_classification()
{
	Record_birth();

	Classification_of_objects = NULL;

	OwP = NULL; // [CO->nb_orbits]

	f_projective_space = false;
	PA = NULL;

}

objects_after_classification::~objects_after_classification()
{
	Record_death();
}


void objects_after_classification::init_after_nauty(
		combinatorics::canonical_form_classification::classification_of_objects *Classification_of_objects,
		int f_projective_space,
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "objects_after_classification::init_after_nauty" << endl;
	}

	objects_after_classification::Classification_of_objects = Classification_of_objects;
	objects_after_classification::f_projective_space = f_projective_space;
	objects_after_classification::PA = PA;

	if (!Classification_of_objects->IS) {
		cout << "objects_after_classification::init_after_nauty "
				"no input stream" << endl;
		exit(1);
	}


	if (!Classification_of_objects->IS->Descr->f_label) {
		cout << "objects_after_classification::init_after_nauty "
				"input stream does not have a label" << endl;
		exit(1);
	}


	OwP = NEW_OBJECTS(combinatorial_object_with_properties, Classification_of_objects->Output->nb_orbits);

	int iso_type;


	for (iso_type = 0; iso_type < Classification_of_objects->Output->nb_orbits; iso_type++) {

		if (f_v) {
			cout << "objects_after_classification::init_after_nauty "
					"iso_type = " << iso_type << " / " << Classification_of_objects->Output->nb_orbits << endl;
			cout << "NO=" << endl;
			//CO->NO_transversal[iso_type]->print();
		}

		std::string label;
		std::string label_tex;

		int input_idx;

		input_idx = Classification_of_objects->Output->Idx_transversal[iso_type];

		label = Classification_of_objects->IS->Descr->label_txt + "_object_" + std::to_string(iso_type);
		label_tex = Classification_of_objects->IS->Descr->label_tex + "\\_object\\_" + std::to_string(iso_type);



		if (f_v) {
			cout << "objects_after_classification::init_after_nauty "
					"before OwP[iso_type].init" << endl;
		}
		OwP[iso_type].init(
				Classification_of_objects->Output->OWCF[input_idx],
				Classification_of_objects->Output->NO[input_idx],
				f_projective_space, PA,
				Classification_of_objects->Descr->max_TDO_depth,
				label,
				label_tex,
				verbose_level);
		if (f_v) {
			cout << "objects_after_classification::init_after_nauty "
					"after OwP[iso_type].init" << endl;
		}


	}

	if (f_v) {
		cout << "objects_after_classification::init_after_nauty done" << endl;
	}
}

void objects_after_classification::classification_write_file(
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "objects_after_classification::classification_write_file" << endl;
	}

	other::l1_interfaces::latex_interface L;
	int iso;

	combinatorics::canonical_form_classification::encoded_combinatorial_object **Enc;

	Enc = (combinatorics::canonical_form_classification::encoded_combinatorial_object **) NEW_pvoid(Classification_of_objects->Output->CB->nb_types);

	for (iso = 0; iso < Classification_of_objects->Output->CB->nb_types; iso++) {

		if (f_v ) {
			cout << "Isomorphism type " << iso << " / " << Classification_of_objects->Output->CB->nb_types << endl;
		}
		cout << " is original object "
			<< Classification_of_objects->Output->CB->Type_rep[iso] << " and appears "
			<< Classification_of_objects->Output->CB->Type_mult[iso] << " times: \\\\" << endl;


		other::data_structures::sorting Sorting;
		int *Input_objects;
		int nb_input_objects;
		//int object_idx;

		Classification_of_objects->Output->CB->C_type_of->get_class_by_value(
				Input_objects,
				nb_input_objects, iso, 0 /*verbose_level */);
		Sorting.int_vec_heapsort(
				Input_objects, nb_input_objects);

		cout << "This isomorphism type appears " << nb_input_objects
				<< " times, namely for the following "
				<< nb_input_objects << " input objects: " << endl;
		if (nb_input_objects < 10) {
			cout << "$" << endl;
			L.int_set_print_tex(
					cout, Input_objects, nb_input_objects);
			cout << "$\\\\" << endl;
		}
		else {
			cout << "Too big to print. \\\\" << endl;
#if 0
			fp << "$$" << endl;
			L.int_vec_print_as_matrix(fp, Input_objects,
				nb_input_objects, 10 /* width */, true /* f_tex */);
			fp << "$$" << endl;
#endif
		}

		int input_idx;

		input_idx = Classification_of_objects->Output->Idx_transversal[iso];


		//object_idx = Input_objects[0];
		combinatorics::canonical_form_classification::any_combinatorial_object *OwCF
			= Classification_of_objects->Output->OWCF[input_idx];


		if (f_v) {
			cout << "object_with_canonical_form::print_tex_detailed "
					"before encode_incma" << endl;
		}
		OwCF->encode_incma(Enc[iso], verbose_level);

		FREE_int(Input_objects);

	}

	string *Table;
	int nb_c = 6;

	Table = new string[Classification_of_objects->Output->CB->nb_types * nb_c];

	for (iso = 0; iso < Classification_of_objects->Output->CB->nb_types; iso++) {


		//int nb_points, nb_blocks;
		//int *Inc;

		//nb_points = Enc[iso]->nb_rows;
		//nb_blocks = Enc[iso]->nb_cols;
		//Inc = Enc[iso]->get_Incma();

		string s_input_idx;
		string s_nb_rows;
		string s_nb_cols;
		string s_nb_flags;
		string s_ago;
		string s_incma;

		s_input_idx = std::to_string(Classification_of_objects->Output->CB->Type_rep[iso]);
		s_nb_rows = std::to_string(Enc[iso]->nb_rows);
		s_nb_cols = std::to_string(Enc[iso]->nb_cols);
		s_nb_flags = std::to_string(Enc[iso]->get_nb_flags());
		s_ago = OwP[iso].A_perm->group_order_as_string();
		s_incma = Enc[iso]->stringify_incma();

		Table[iso * nb_c + 0] = s_input_idx;
		Table[iso * nb_c + 1] = s_nb_rows;
		Table[iso * nb_c + 2] = s_nb_cols;
		Table[iso * nb_c + 3] = s_nb_flags;
		Table[iso * nb_c + 4] = s_ago;
		Table[iso * nb_c + 5] = "\"" + s_incma + "\"";


	}

	string fname;
	string headings;

	headings.assign("input,nb_rows,nb_cols,nb_flags,ago,flags");
	other::orbiter_kernel_system::file_io Fio;

	fname = fname_base + "_classification_data.csv";

	Fio.Csv_file_support->write_table_of_strings(
			fname,
			Classification_of_objects->Output->CB->nb_types, nb_c, Table,
			headings,
			verbose_level);


	if (f_v) {
		cout << "objects_after_classification::classification_write_file done" << endl;
	}
}

void objects_after_classification::classification_report(
		combinatorics::canonical_form_classification::objects_report_options
					*Report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "objects_after_classification::classification_report" << endl;
	}

	if (Classification_of_objects == NULL) {
		cout << "objects_after_classification::classification_report "
				"Classification_of_objects == NULL" << endl;
		exit(1);
	}

	if (Classification_of_objects->Descr == NULL) {
		cout << "objects_after_classification::classification_report "
				"Classification_of_objects->Descr == NULL" << endl;
		exit(1);
	}



	if (f_v) {
		cout << "objects_after_classification::classification_report "
				"before latex_report" << endl;
	}
	latex_report(
			Report_options,
			verbose_level);

	if (f_v) {
		cout << "objects_after_classification::classification_report "
				"after latex_report" << endl;
	}

	if (f_v) {
		cout << "objects_after_classification::classification_report done" << endl;
	}
}

void objects_after_classification::latex_report(
		combinatorics::canonical_form_classification::objects_report_options
			*Report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "objects_after_classification::latex_report" << endl;
	}

	string fname;


	if (Report_options->f_canonical_forms) {

		fname = Classification_of_objects->get_label() + "_canonical_forms.tex";
	}
	else {

		fname = Classification_of_objects->get_label() + "_classification.tex";
	}



	if (f_v) {
		cout << "objects_after_classification::latex_report, "
				"CB->nb_types=" << Classification_of_objects->Output->CB->nb_types << endl;
	}

	other::orbiter_kernel_system::file_io Fio;





	{
		other::l1_interfaces::latex_interface L;

		ofstream ost(fname);

		//L.head_easy(ost);
		L.head_easy_and_enlarged(ost);


		Classification_of_objects->Output->report_summary_of_iso_types(
				ost, verbose_level);


		ost << "Ago : ";
		Classification_of_objects->Output->T_Ago->print_file_tex(
				ost, false /* f_backwards*/);
		ost << "\\\\" << endl;

		if (f_v) {
			cout << "objects_after_classification::latex_report "
					"before report_all_isomorphism_types" << endl;
		}

		report_all_isomorphism_types(
				ost,
				Report_options,
				verbose_level);

		if (f_v) {
			cout << "objects_after_classification::latex_report "
					"after report_all_isomorphism_types" << endl;
		}


		if (Report_options->f_canonical_forms) {

			if (f_v) {
				cout << "objects_after_classification::latex_report "
						"before report_all_canonical_forms" << endl;
			}
			report_all_canonical_forms(
					ost,
					Report_options,
					verbose_level);
			if (f_v) {
				cout << "objects_after_classification::latex_report "
						"after report_all_canonical_forms" << endl;
			}

		}

		L.foot(ost);
	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	//FREE_int(perm);
	//FREE_int(v);
	if (f_v) {
		cout << "objects_after_classification::latex_report done" << endl;
	}
}

void objects_after_classification::report_all_isomorphism_types(
		std::ostream &ost,
		combinatorics::canonical_form_classification::objects_report_options
			*Report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "objects_after_classification::report_all_isomorphism_types" << endl;
	}
	int iso_idx, j;

	ost << "\\subsubsection*{objects\\_after\\_classification::report\\_all\\_isomorphism\\_types}" << endl;


	other::l1_interfaces::latex_interface L;

	for (iso_idx = 0; iso_idx < Classification_of_objects->Output->CB->nb_types; iso_idx++) {

		j = Classification_of_objects->Output->CB->perm[iso_idx];

		ost << "\\section*{Isomorphism type " << iso_idx << " / "
				<< Classification_of_objects->Output->CB->nb_types << "}" << endl;
		ost << "Isomorphism type " << iso_idx << " / " << Classification_of_objects->Output->CB->nb_types
			<<  " stored at " << j
			<< " is original object "
			<< Classification_of_objects->Output->CB->Type_rep[j] << " and appears "
			<< Classification_of_objects->Output->CB->Type_mult[j] << " times: \\\\" << endl;

		{
			other::data_structures::sorting Sorting;
			int *Input_objects;
			int nb_input_objects;
			Classification_of_objects->Output->CB->C_type_of->get_class_by_value(
					Input_objects,
					nb_input_objects, j,
					0 /*verbose_level */);
			Sorting.int_vec_heapsort(Input_objects, nb_input_objects);

			ost << "This isomorphism type appears " << nb_input_objects
					<< " times, namely for the following "
					<< nb_input_objects << " input objects: " << endl;
			if (nb_input_objects < 10) {
				ost << "$" << endl;
				L.int_set_print_tex(
						ost, Input_objects, nb_input_objects);
				ost << "$\\\\" << endl;
			}
			else {
				ost << "Too big to print. \\\\" << endl;
#if 0
				fp << "$$" << endl;
				L.int_vec_print_as_matrix(fp, Input_objects,
					nb_input_objects, 10 /* width */, true /* f_tex */);
				fp << "$$" << endl;
#endif
			}

			FREE_int(Input_objects);
		}

		if (f_v) {
			cout << "objects_after_classification::report_all_isomorphism_types "
					"before report_isomorphism_type" << endl;
		}
		report_isomorphism_type(
				ost,
				Report_options,
				iso_idx,
				verbose_level);
		if (f_v) {
			cout << "objects_after_classification::report_all_isomorphism_types "
					"after report_isomorphism_type" << endl;
		}


	} // next iso_idx



	if (f_v) {
		cout << "objects_after_classification::report_all_isomorphism_types done" << endl;
	}

}

void objects_after_classification::report_all_canonical_forms(
		std::ostream &ost,
		combinatorics::canonical_form_classification::objects_report_options
			*Report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "objects_after_classification::report_all_canonical_forms" << endl;
	}


	other::l1_interfaces::latex_interface L;
	int input_idx;


	for (input_idx = 0; input_idx < Classification_of_objects->Output->nb_input; input_idx++) {

		ost << "\\section*{Input Object " << input_idx << " / "
				<< Classification_of_objects->Output->nb_input << "}" << endl;


		if (f_v) {
			cout << "objects_after_classification::report_all_canonical_forms "
					<< " input object " << input_idx
					<< ", before report_input_object_only" << endl;
		}

		report_input_object_only(
				ost,
				Report_options,
				input_idx /* object_idx */,
				verbose_level);

		if (f_v) {
			cout << "objects_after_classification::report_all_canonical_forms "
					"after report_input_object_only" << endl;
		}



	}


	if (f_v) {
		cout << "objects_after_classification::report_all_canonical_forms done" << endl;
	}
}


void objects_after_classification::report_isomorphism_type(
		std::ostream &ost,
		combinatorics::canonical_form_classification::objects_report_options
			*Report_options,
		int iso_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "objects_after_classification::report_isomorphism_type "
				"iso_idx=" << iso_idx << endl;
	}
	int j;
	other::l1_interfaces::latex_interface L;

	ost << "\\subsubsection*{objects\\_after\\_classification::report\\_isomorphism\\_type}" << endl;


	//j = CB->perm[i];
	//j = CB->Type_rep[i];
	j = Classification_of_objects->Output->CB->perm[iso_idx];
	//j = i;

	cout << "###################################################"
			"#############################" << endl;
	cout << "Orbit " << iso_idx << " / " << Classification_of_objects->Output->CB->nb_types
			<< " is canonical form no " << j
			<< ", original object no " << Classification_of_objects->Output->CB->Type_rep[j]
			<< ", frequency " << Classification_of_objects->Output->CB->Type_mult[j]
			<< " : " << endl;


	{
		int *Input_objects;
		int nb_input_objects;
		Classification_of_objects->Output->CB->C_type_of->get_class_by_value(
				Input_objects,
				nb_input_objects,
				j,
				0 /*verbose_level */);

		cout << "This isomorphism type appears " << nb_input_objects
				<< " times, namely for the following "
						"input objects:" << endl;
		if (nb_input_objects < 10) {
			L.int_vec_print_as_matrix(
					cout,
					Input_objects,
					nb_input_objects,
					10 /* width */,
					false /* f_tex */);
		}
		else {
			cout << "too many to print" << endl;
		}

		FREE_int(Input_objects);
	}


	int input_idx;

	input_idx = Classification_of_objects->Output->Idx_transversal[iso_idx];

	if (f_v) {
		cout << "objects_after_classification::report_isomorphism_type "
				"iso type " << iso_idx
				<< " is input object " << input_idx
				<< ", before report_object_with_properties" << endl;
	}
	report_object_with_properties(
			ost,
			Report_options,
			input_idx /* object_idx */,
			iso_idx,
			verbose_level);
	if (f_v) {
		cout << "objects_after_classification::report_isomorphism_type "
				"iso_idx=" << iso_idx << " after report_object_with_properties" << endl;
	}




	if (f_v) {
		cout << "objects_after_classification::report_isomorphism_type "
				"iso_idx=" << iso_idx << " done" << endl;
	}
}

void objects_after_classification::report_object_with_properties(
		std::ostream &ost,
		combinatorics::canonical_form_classification::objects_report_options
			*Report_options,
		int input_idx, int iso_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "objects_after_classification::report_object_with_properties "
				"input_idx=" << input_idx << endl;
	}

	ost << "\\subsubsection*{objects\\_after\\_classification::report\\_object\\_with\\_properties}" << endl;

	//int j;

	//j = CO->CB->perm[i];


	report_input_object_only(
			ost, Report_options, input_idx,
			verbose_level);

#if 0
	combinatorics::canonical_form_classification::any_combinatorial_object
		*OwCF = Classification_of_objects->Output->OWCF[input_idx];

	if (f_v) {
		cout << "objects_after_classification::report_object_with_properties "
				"before OwCF->print_tex_detailed" << endl;
	}

	ost << "\\subsubsection*{objects\\_after\\_classification::report\\_object "
			"print\\_tex\\_detailed}" << endl;

	OwCF->print_tex_detailed(
			ost,
			Report_options,
			verbose_level);
	if (f_v) {
		cout << "objects_after_classification::report_object_with_properties "
				"after OwCF->print_tex_detailed" << endl;
	}
#endif

	if (false /*CO->f_projective_space*/) {

#if 0
		object_in_projective_space_with_action *OiPA;

		OiPA = (object_in_projective_space_with_action *)
				CB->Type_extra_data[object_idx];

		OiPA->report(fp, PA, max_TDO_depth, verbose_level);
#endif

	}
	else {
		if (f_v) {
			cout << "objects_after_classification::report_object_with_properties "
					"before OwP[iso_idx].latex_report" << endl;
		}
		ost << "\\subsubsection*{objects\\_after\\_classification::report\\_object\\_with\\_properties latex\\_report}" << endl;

		OwP[iso_idx].latex_report(
				ost,
				Report_options,
				verbose_level);
		if (f_v) {
			cout << "objects_after_classification::report_object_with_properties "
					"after OwP[iso_idx].latex_report" << endl;
		}
	}



}


void objects_after_classification::report_input_object_only(
		std::ostream &ost,
		combinatorics::canonical_form_classification::objects_report_options
			*Report_options,
		int input_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "objects_after_classification::report_input_object_only "
				"input_idx=" << input_idx << endl;
	}

	combinatorics::canonical_form_classification::any_combinatorial_object
		*Any_Combo = Classification_of_objects->Output->OWCF[input_idx];

	other::l1_interfaces::nauty_output *NO =
			Classification_of_objects->Output->NO[input_idx];
	if (f_v) {
		cout << "objects_after_classification::report_input_object_only "
				"before OwCF->print_tex_detailed" << endl;
	}

	int iso_type;
	int iso_rep_idx;
	int N;

	N = NO->N;
	int *Iso;
	int *Clv;

	iso_type = Classification_of_objects->Output->CB->type_of[input_idx];
	iso_rep_idx = Classification_of_objects->Output->Idx_transversal[iso_type];

	ost << "isomorphism type = " << iso_type << "\\\\" << endl;
	ost << "isomorphism representative = " << iso_rep_idx << "\\\\" << endl;

	Clv = NEW_int(NO->N);
	Iso = NEW_int(NO->N);

	int *perm1;
	int *perm2;


	perm1 = NO->canonical_labeling;
	perm2 = Classification_of_objects->Output->NO[iso_rep_idx]->canonical_labeling;

	ost << "canonical labeling: \\\\" << endl;
	Int_vec_print_fully(ost, perm1, NO->N);
	ost << "\\\\" << endl;



	ost << "canonical labeling of iso representative: \\\\" << endl;
	Int_vec_print_fully(ost,
			perm2,
			N);
	ost << "\\\\" << endl;

	combinatorics::other_combinatorics::combinatorics_domain Combo;

	Combo.Permutations->perm_inverse(
			perm1,
			Clv,
			N);

	Combo.Permutations->perm_mult(
			Clv,
			perm2,
			Iso,
			N);


	ost << "Isomorphism to the chosen representative: iso=\\\\" << endl;
	Int_vec_print_fully(ost,
			Iso,
			N);
	ost << "\\\\" << endl;



	if (Any_Combo->type == t_INC) {
		if (f_v) {
			cout << "any_combinatorial_object::print_tex t_INC" << endl;
		}
		ost << "Input flags: \\\\" << endl;
		Lint_vec_print_fully(ost, Any_Combo->set, Any_Combo->sz);
		ost << "\\\\" << endl;


		long int *flags_out;

		flags_out = NEW_lint(Any_Combo->sz);

		Combo.Permutations->apply_in_product_action_lint(
				Any_Combo->v, Any_Combo->b, Iso,
				Any_Combo->set /* flags_in */, flags_out, Any_Combo->sz /* nb_flags*/,
				verbose_level);

		other::data_structures::sorting Sorting;

		ost << "Output flags before sorting: \\\\" << endl;
		Lint_vec_print_fully(ost, flags_out, Any_Combo->sz);
		ost << "\\\\" << endl;


		Sorting.lint_vec_heapsort(
				flags_out, Any_Combo->sz);

		ost << "Output flags after sorting: \\\\" << endl;
		Lint_vec_print_fully(ost, flags_out, Any_Combo->sz);
		ost << "\\\\" << endl;


		FREE_lint(flags_out);
	}

	FREE_int(Clv);
	FREE_int(Iso);

	Any_Combo->print_tex_detailed(
			ost,
			Report_options,
			verbose_level);

	if (f_v) {
		cout << "objects_after_classification::report_input_object_only "
				"after OwCF->print_tex_detailed" << endl;
	}

	if (f_v) {
		cout << "objects_after_classification::report_input_object_only done" << endl;
	}
}


}}}


