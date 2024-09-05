/*
 * classification_of_combinatorial_objects.cpp
 *
 *  Created on: Jul 9, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {




classification_of_combinatorial_objects::classification_of_combinatorial_objects()
{

	CO = NULL;

	OwP = NULL; // [CO->nb_orbits]

	f_projective_space = false;
	PA = NULL;

}

classification_of_combinatorial_objects::~classification_of_combinatorial_objects()
{
}


void classification_of_combinatorial_objects::init_after_nauty(
		canonical_form_classification::classification_of_objects *CO,
		int f_projective_space,
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_combinatorial_objects::init_after_nauty" << endl;
	}

	classification_of_combinatorial_objects::CO = CO;
	classification_of_combinatorial_objects::f_projective_space = f_projective_space;
	classification_of_combinatorial_objects::PA = PA;

	if (!CO->IS) {
		cout << "classification_of_combinatorial_objects::init_after_nauty "
				"no input stream" << endl;
		exit(1);
	}


	if (!CO->IS->Descr->f_label) {
		cout << "classification_of_combinatorial_objects::init_after_nauty "
				"input stream does not have a label" << endl;
		exit(1);
	}


	OwP = NEW_OBJECTS(combinatorial_object_with_properties, CO->nb_orbits);

	int iso_type;


	for (iso_type = 0; iso_type < CO->nb_orbits; iso_type++) {

		if (f_v) {
			cout << "classification_of_combinatorial_objects::init_after_nauty "
					"iso_type = " << iso_type << " / " << CO->nb_orbits << endl;
			cout << "NO=" << endl;
			//CO->NO_transversal[iso_type]->print();
		}

		std::string label;

		label = CO->IS->Descr->label_txt + "_object" + std::to_string(iso_type);

		if (f_v) {
			cout << "classification_of_combinatorial_objects::init_after_nauty "
					"before OwP[iso_type].init" << endl;
		}
		OwP[iso_type].init(
				CO->OWCF_transversal[iso_type],
				CO->NO_transversal[iso_type],
				f_projective_space, PA,
				CO->Descr->max_TDO_depth,
				label,
				verbose_level);
		if (f_v) {
			cout << "classification_of_combinatorial_objects::init_after_nauty "
					"after OwP[iso_type].init" << endl;
		}


	}

	if (f_v) {
		cout << "classification_of_combinatorial_objects::init_after_nauty done" << endl;
	}
}

void classification_of_combinatorial_objects::classification_write_file(
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_combinatorial_objects::classification_write_file" << endl;
	}

	l1_interfaces::latex_interface L;
	int iso;

	canonical_form_classification::encoded_combinatorial_object **Enc;

	Enc = (canonical_form_classification::encoded_combinatorial_object **) NEW_pvoid(CO->CB->nb_types);

	for (iso = 0; iso < CO->CB->nb_types; iso++) {

		if (f_v ) {
			cout << "Isomorphism type " << iso << " / " << CO->CB->nb_types << endl;
		}
		cout << " is original object "
			<< CO->CB->Type_rep[iso] << " and appears "
			<< CO->CB->Type_mult[iso] << " times: \\\\" << endl;


		data_structures::sorting Sorting;
		int *Input_objects;
		int nb_input_objects;
		//int object_idx;

		CO->CB->C_type_of->get_class_by_value(
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

		//object_idx = Input_objects[0];
		canonical_form_classification::any_combinatorial_object *OwCF = CO->OWCF_transversal[iso];


		if (f_v) {
			cout << "object_with_canonical_form::print_tex_detailed "
					"before encode_incma" << endl;
		}
		OwCF->encode_incma(Enc[iso], verbose_level);

		FREE_int(Input_objects);

	}

	string *Table;
	int nb_c = 6;

	Table = new string[CO->CB->nb_types * nb_c];

	for (iso = 0; iso < CO->CB->nb_types; iso++) {


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

		s_input_idx = std::to_string(CO->CB->Type_rep[iso]);
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
	orbiter_kernel_system::file_io Fio;

	fname = fname_base + "_classification_data.csv";

	Fio.Csv_file_support->write_table_of_strings(
			fname,
			CO->CB->nb_types, nb_c, Table,
			headings,
			verbose_level);


	if (f_v) {
		cout << "classification_of_combinatorial_objects::classification_write_file done" << endl;
	}
}

void classification_of_combinatorial_objects::classification_report(
		canonical_form_classification::classification_of_objects_report_options
					*Report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_combinatorial_objects::classification_report" << endl;
	}

	if (CO == NULL) {
		cout << "classification_of_combinatorial_objects::classification_report "
				"CO == NULL" << endl;
		exit(1);
	}

	if (CO->Descr == NULL) {
		cout << "classification_of_combinatorial_objects::classification_report "
				"CO->Descr == NULL" << endl;
		exit(1);
	}



	if (f_v) {
		cout << "classification_of_combinatorial_objects::classification_report "
				"before latex_report" << endl;
	}
	latex_report(Report_options,
			verbose_level);

	if (f_v) {
		cout << "classification_of_combinatorial_objects::classification_report "
				"after latex_report" << endl;
	}

	if (f_v) {
		cout << "classification_of_combinatorial_objects::classification_report done" << endl;
	}
}

void classification_of_combinatorial_objects::latex_report(
		canonical_form_classification::classification_of_objects_report_options
			*Report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_combinatorial_objects::latex_report" << endl;
	}


	string fname;

	fname = CO->get_label() + "_classification.tex";


	if (f_v) {
		cout << "classification_of_combinatorial_objects::classification_report "
				"before latex_report" << endl;
	}



	if (f_v) {
		cout << "classification_of_combinatorial_objects::latex_report, "
				"CB->nb_types=" << CO->CB->nb_types << endl;
	}

	orbiter_kernel_system::file_io Fio;

	{
		l1_interfaces::latex_interface L;

		ofstream ost(fname);

		//L.head_easy(ost);
		L.head_easy_and_enlarged(ost);


		CO->report_summary_of_orbits(ost, verbose_level);


		ost << "Ago : ";
		CO->T_Ago->print_file_tex(ost, false /* f_backwards*/);
		ost << "\\\\" << endl;

		if (f_v) {
			cout << "classification_of_combinatorial_objects::latex_report before loop" << endl;
		}

		report_all_isomorphism_types(
				ost, Report_options,
				verbose_level);

		L.foot(ost);
	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	//FREE_int(perm);
	//FREE_int(v);
	if (f_v) {
		cout << "classification_of_combinatorial_objects::latex_report done" << endl;
	}
}

void classification_of_combinatorial_objects::report_all_isomorphism_types(
		std::ostream &ost,
		canonical_form_classification::classification_of_objects_report_options
			*Report_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_combinatorial_objects::report_all_isomorphism_types" << endl;
	}
	int i, j;

	l1_interfaces::latex_interface L;

	for (i = 0; i < CO->CB->nb_types; i++) {

		j = CO->CB->perm[i];

		ost << "\\section*{Isomorphism type " << i << " / " << CO->CB->nb_types << "}" << endl;
		ost << "Isomorphism type " << i << " / " << CO->CB->nb_types
			<<  " stored at " << j
			<< " is original object "
			<< CO->CB->Type_rep[j] << " and appears "
			<< CO->CB->Type_mult[j] << " times: \\\\" << endl;

		{
			data_structures::sorting Sorting;
			int *Input_objects;
			int nb_input_objects;
			CO->CB->C_type_of->get_class_by_value(Input_objects,
					nb_input_objects, j, 0 /*verbose_level */);
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
			cout << "classification_of_combinatorial_objects::report_all_isomorphism_types "
					"before report_isomorphism_type" << endl;
		}
		report_isomorphism_type(
				ost, Report_options, i, verbose_level);
		if (f_v) {
			cout << "classification_of_combinatorial_objects::report_all_isomorphism_types "
					"after report_isomorphism_type" << endl;
		}


	} // next i
	if (f_v) {
		cout << "classification_of_combinatorial_objects::report_all_isomorphism_types done" << endl;
	}

}


void classification_of_combinatorial_objects::report_isomorphism_type(
		std::ostream &ost,
		canonical_form_classification::classification_of_objects_report_options
			*Report_options,
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_combinatorial_objects::report_isomorphism_type "
				"i=" << i << endl;
	}
	int j;
	l1_interfaces::latex_interface L;

	//j = CB->perm[i];
	//j = CB->Type_rep[i];
	j = CO->CB->perm[i];
	//j = i;

	cout << "###################################################"
			"#############################" << endl;
	cout << "Orbit " << i << " / " << CO->CB->nb_types
			<< " is canonical form no " << j
			<< ", original object no " << CO->CB->Type_rep[j]
			<< ", frequency " << CO->CB->Type_mult[j]
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
					false /* f_tex */);
		}
		else {
			cout << "too many to print" << endl;
		}

		FREE_int(Input_objects);
	}



	if (f_v) {
		cout << "classification_of_combinatorial_objects::report_isomorphism_type "
				"i=" << i << " before report_object" << endl;
	}
	report_object(
			ost,
			Report_options,
			i /* object_idx */,
			verbose_level);
	if (f_v) {
		cout << "classification_of_combinatorial_objects::report_isomorphism_type "
				"i=" << i << " after report_object" << endl;
	}




	if (f_v) {
		cout << "classification_of_combinatorial_objects::report_isomorphism_type "
				"i=" << i << " done" << endl;
	}
}

void classification_of_combinatorial_objects::report_object(
		std::ostream &ost,
		canonical_form_classification::classification_of_objects_report_options
			*Report_options,
		int i,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classification_of_combinatorial_objects::report_object "
				"object_idx=" << i << endl;
	}

	//int j;

	//j = CO->CB->perm[i];

	canonical_form_classification::any_combinatorial_object *OwCF = CO->OWCF_transversal[i];

	if (f_v) {
		cout << "classification_of_combinatorial_objects::report_object "
				"before OwCF->print_tex_detailed" << endl;
	}

	ost << "\\subsubsection*{classification\\_of\\_combinatorial\\_objects::report\\_object print\\_tex\\_detailed}" << endl;

	OwCF->print_tex_detailed(
			ost,
			Report_options,
			verbose_level);
	if (f_v) {
		cout << "classification_of_combinatorial_objects::report_object "
				"after OwCF->print_tex_detailed" << endl;
	}

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
			cout << "classification_of_combinatorial_objects::report_object "
					"before OwP[object_idx].latex_report" << endl;
		}
		ost << "\\subsubsection*{classification\\_of\\_combinatorial\\_objects::report\\_object latex\\_report}" << endl;
		OwP[i].latex_report(
				ost,
				Report_options,
				verbose_level);
		if (f_v) {
			cout << "classification_of_combinatorial_objects::report_object "
					"after OwP[object_idx].latex_report" << endl;
		}
	}



}


}}}


