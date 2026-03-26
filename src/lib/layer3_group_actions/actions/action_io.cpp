/*
 * action_io.cpp
 *
 *  Created on: Mar 30, 2019
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {




void action::print_symmetry_group_type(
		std::ostream &ost)
{
	action_global AG;

	AG.action_print_symmetry_group_type(ost, type_G);
	if (f_has_subaction) {
		ost << "->";
		subaction->print_symmetry_group_type(ost);
	}
	//else {
		//ost << "no subaction";
	//}

}

std::string action::stringify_subaction_labels()
{
	string s, s1;

	if (subaction) {
		s1 = subaction->stringify_subaction_labels();
		s = label + " -> " + s1;
	}
	else {
		s = label;
	}
	return s;
}

void action::print_info()
{
	action_global AG;

	string s_type;
	string s_base;
	string s_tl;
	string s_go;

	s_type = AG.stringify_symmetry_group_type(type_G);
	s_base = stringify_base();
	s_tl = stringify_tl();

	if (f_has_sims) {
		s_go = Sims->stringify_group_order();
	}
	else {
		s_go = "no sims";
	}


	string s_sub;

	s_sub = stringify_subaction_labels();

	cout << "ACTION " << label << " : " << label_tex
			<< " degree = " << degree << " of type " << s_type
			<< " : base = (" << s_base << ")"
			<< " : tl = (" << s_tl << ")"
			<< " : go = " << s_go
			<< " : label = " << ptr->label
			<< " : sub = " << s_sub << endl;


#if 0
	//print_symmetry_group_type(cout);
	cout << endl;
	cout << "low_level_point_size=" << low_level_point_size;
	cout << ", f_has_sims=" << f_has_sims;
	cout << ", f_has_strong_generators=" << f_has_strong_generators;
	cout << endl;
	cout << "make_element_size=" << make_element_size << endl;
	cout << "elt_size_in_int=" << elt_size_in_int << endl;

	if (f_is_linear) {
		cout << "linear of dimension " << dimension << endl;
		}
	else {
		cout << "the action is not linear" << endl;
	}
#endif
	//print_base();


}


void action::print_base()
{

	print_base(cout);
#if 0
	if (Stabilizer_chain) {
		cout << "action " << label << " has base ";
		Lint_vec_print(cout, get_base(), base_len());
		cout << " basic orbits have the following length: ";
		Int_vec_print(cout, get_transversal_length(), base_len());
		cout << endl;
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
	}
#endif
}

std::string action::stringify_base()
{
	string s;

	if (Stabilizer_chain) {
		if (base_len() == 0) {
			s =  "base of length zero";
		}
		else {
			int i;
			for (i = 0; i < base_len(); i++) {
				s += std::to_string(base_i(i));
				if (i < base_len() - 1) {
					s += ", ";
				}
			}
		}
	}
	else {
		s = " - ";
	}
	return s;
}

std::string action::stringify_tl()
{
	string s;

	if (Stabilizer_chain) {
		if (base_len() == 0) {
			s =  "N/A";
		}
		else {
			int i;
			for (i = 0; i < base_len(); i++) {
				s += std::to_string(transversal_length_i(i));
				if (i < base_len() - 1) {
					s += ", ";
				}
			}
		}
	}
	else {
		s = " - ";
	}
	return s;
}


void action::print_base(
		std::ostream &ost)
{
	if (Stabilizer_chain) {
		int i;

		ost << "action " << label << " has base ";
		if (base_len() == 0) {
			ost << "of length zero" << endl;
		}
		else {
			for (i = 0; i < base_len(); i++) {
				ost << base_i(i);
				if (i < base_len() - 1) {
					ost << ", ";
				}
			}
			ost << " and transversal_length: ";
			for (i = 0; i < base_len(); i++) {
				ost << transversal_length_i(i);
				if (i < base_len() - 1) {
					ost << ", ";
				}
			}
		}
		ost << endl;
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
	}
}


void action::print_bare_base(
		std::ofstream &ost)
{
	if (Stabilizer_chain) {
		other::orbiter_kernel_system::Orbiter->Lint_vec->print_bare_fully(ost, get_base(), base_len());
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
		exit(1);
	}
}



void action::print_group_order(
		std::ostream &ost)
{
	algebra::ring_theory::longinteger_object go;
	group_order(go);
	cout << go;
}


void action::print_group_order_long(
		std::ostream &ost)
{
	int i;

	algebra::ring_theory::longinteger_object go;
	group_order(go);
	cout << go << " =";
	if (Stabilizer_chain) {
		for (i = 0; i < base_len(); i++) {
			cout << " " << transversal_length_i(i);
		}
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
	}

}

void action::print_vector(
		data_structures_groups::vector_ge &v)
{
	int i, l;

	l = v.len;
	cout << "vector of " << l << " group elements:" << endl;
	for (i = 0; i < l; i++) {
		cout << i << " : " << endl;
		Group_element->element_print_quick(v.ith(i), cout);
		cout << endl;
	}
}

void action::print_vector_as_permutation(
		data_structures_groups::vector_ge &v)
{
	int i, l;

	l = v.len;
	cout << "vector of " << l << " group elements:" << endl;
	for (i = 0; i < l; i++) {
		cout << i << " : ";
		Group_element->element_print_as_permutation(v.ith(i), cout);
		cout << endl;
	}
}



void action::export_to_orbiter(
		std::string &fname, std::string &label,
		groups::strong_generators *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int a;
	other::orbiter_kernel_system::file_io Fio;
	algebra::ring_theory::longinteger_object go;

	if (f_v) {
		cout << "action::export_to_orbiter" << endl;
	}

	SG->group_order(go);
	if (f_v) {
		cout << "action::export_to_orbiter go = " << go << endl;
		cout << "action::export_to_orbiter number of generators = " << SG->gens->len << endl;
		cout << "action::export_to_orbiter degree = " << degree << endl;
	}
	{
		ofstream fp(fname);

		fp << "GENERATORS_" << label << " = \\" << endl;
		for (i = 0; i < SG->gens->len; i++) {
			fp << "\t\"";
			for (j = 0; j < degree; j++) {
				if (false) {
					cout << "action::export_to_orbiter "
							"computing image of " << j << " under generator " << i << endl;
				}
				a = Group_element->element_image_of(
						j, SG->gens->ith(i), 0 /* verbose_level*/);
				fp << a;
				if (j < degree - 1 || i < SG->gens->len - 1) {
					fp << ",";
				}
			}
			fp << "\"";
			if (i < SG->gens->len - 1) {
				fp << "\\" << endl;
			}
			else {
				fp << endl;
			}
		}

		fp << endl;
		fp << label << ":" << endl;
		fp << "\t$(ORBITER_PATH)orbiter.out -v 2 \\" << endl;
		fp << "\t-define G -permutation_group -symmetric_group " << degree << " \\" << endl;
		fp << "\t-subgroup_by_generators " << label << " " << go << " " << SG->gens->len << " $(GENERATORS_" << label << ") -end \\" << endl;

		//		$(ORBITER_PATH)orbiter.out -v 10
		//			-define G -permutation_group -symmetric_group 13
		//				-subgroup_by_generators H5 5 1 $(GENERATORS_H5) -end
		// with backslashes at the end of the line

	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "action::export_to_orbiter" << endl;
	}
}


void action::export_to_orbiter_as_bsgs(
		std::string &fname,
		std::string &label,
		std::string &label_tex,
		groups::strong_generators *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::export_to_orbiter_as_bsgs" << endl;
	}


	if (f_v) {
		cout << "action::export_to_orbiter_as_bsgs "
				"before SG->export_to_orbiter_as_bsgs" << endl;
	}
	SG->export_to_orbiter_as_bsgs(
			this,
			fname, label, label_tex,
			verbose_level);
	if (f_v) {
		cout << "action::export_to_orbiter_as_bsgs "
				"after SG->export_to_orbiter_as_bsgs" << endl;
	}


	if (f_v) {
		cout << "action::export_to_orbiter_as_bsgs" << endl;
	}
}




}}}

