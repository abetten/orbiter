// schreier_io.cpp
//
// Anton Betten
// moved here from schreier.cpp: November 3, 2018
// originally started as schreier.cpp: December 9, 2003

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {

void schreier::latex(
		std::string &fname)
{
	int f_with_cosetrep = true;

	{
		ofstream fp(fname);
		other::l1_interfaces::latex_interface L;

		L.head_easy(fp);

		Generators_and_images->print_generators_latex(fp);

		fp << "Orbit lengths: $";
		Forest->print_orbit_length_distribution(fp);
		fp << "$\\\\" << endl;

		Forest->print_and_list_orbits_tex(fp);

		if (Generators_and_images->A->degree < 100) {
			print_tables_latex(fp, f_with_cosetrep);
		}

		L.foot(fp);
	}
}


void schreier::print_and_list_orbits_and_stabilizer(
		std::ostream &ost,
		actions::action *default_action,
		algebra::ring_theory::longinteger_object &go,
	void (*print_point)(
			std::ostream &ost, int pt, void *data),
	void *data)
{
	int i;

	ost << Forest->nb_orbits << " orbits" << endl;
	ost << "orbit group with " << Generators_and_images->gens.len << " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (i = 0; i < Forest->nb_orbits; i++) {

		sims *Stab;
		strong_generators *SG;

		ost << "Orbit " << i << " / " << Forest->nb_orbits
				<< " : of length " << Forest->orbit_len[i];
		ost << " is:" << endl;
		Forest->print_orbit(ost, i);
		ost << endl;
		ost << "Which is:" << endl;
		Forest->print_orbit_using_callback(ost, i, print_point, data);
		//ost << endl;
		ost << "The stabilizer of the element "
				<< Forest->orbit[Forest->orbit_first[i]] << " is:" << endl;
		point_stabilizer(default_action, go, Stab, i, 0 /* verbose_level */);

		SG = NEW_OBJECT(strong_generators);

		SG->init_from_sims(Stab, 0 /* verbose_level*/);
		SG->print_generators(ost, 0 /* verbose_level*/);
		FREE_OBJECT(SG);
		FREE_OBJECT(Stab);
	}
	ost << endl;
}



void schreier::print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
		std::ostream &ost,
		actions::action *default_action,
		strong_generators *gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "schreier::print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex" << endl;
	}
	for (i = 0; i < Forest->nb_orbits; i++) {
		print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
				i, default_action,
				gens, ost);
	}
}

void schreier::print_and_list_orbit_and_stabilizer_tex(
		int i,
		actions::action *default_action,
		algebra::ring_theory::longinteger_object &full_group_order,
	std::ostream &ost)
{
	ost << " Orbit " << i << " / " << Forest->nb_orbits << " : ";
	Forest->print_orbit_tex(ost, i);
	ost << " of length " << Forest->orbit_len[i] << "\\\\" << endl;

	strong_generators *gens;

	gens = stabilizer_orbit_rep(default_action,
		full_group_order, i, 0 /*verbose_level */);

	gens->print_generators_tex(ost);

	FREE_OBJECT(gens);
}

void schreier::write_orbit_summary(
		std::string &fname,
		actions::action *default_action,
		algebra::ring_theory::longinteger_object &full_group_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::write_orbit_summary" << endl;
	}
	long int *Rep;
	long int *Stab_order;
	long int *Orbit_length;
	int orbit_no;

	Rep = NEW_lint(Forest->nb_orbits);
	Stab_order = NEW_lint(Forest->nb_orbits);
	Orbit_length = NEW_lint(Forest->nb_orbits);

	for (orbit_no = 0; orbit_no < Forest->nb_orbits; orbit_no++) {
		Rep[orbit_no] = Forest->orbit[Forest->orbit_first[orbit_no]];
		Orbit_length[orbit_no] = Forest->orbit_len[orbit_no];

		strong_generators *gens;

		gens = stabilizer_orbit_rep(
				default_action,
			full_group_order, orbit_no,
			0 /*verbose_level */);

		Stab_order[orbit_no] = gens->group_order_as_lint();

	}

	other::orbiter_kernel_system::file_io Fio;
	long int *Vec[3];
	string *column_label;


	column_label = new string[3];
	column_label[0] = "Rep";
	column_label[1] = "StabOrder";
	column_label[2] = "OrbitLength";

	Vec[0] = Rep;
	Vec[1] = Stab_order;
	Vec[2] = Orbit_length;

	Fio.Csv_file_support->lint_vec_array_write_csv(
			3 /* nb_vecs */, Vec, Forest->nb_orbits,
			fname, column_label);

	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}

	delete [] column_label;
	FREE_lint(Rep);
	FREE_lint(Stab_order);
	FREE_lint(Orbit_length);
	if (f_v) {
		cout << "schreier::write_orbit_summary done" << endl;
	}
}



void schreier::get_stabilizer_orbit_rep(
	int orbit_idx, actions::action *default_action,
	strong_generators *gens,
	strong_generators *&gens_stab)
{
	algebra::ring_theory::longinteger_object full_group_order;

	gens->group_order(full_group_order);


	gens_stab = stabilizer_orbit_rep(
		default_action,
		full_group_order, orbit_idx,
		0 /*verbose_level */);

}

void schreier::print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
	int i, actions::action *default_action,
	strong_generators *gens, std::ostream &ost)
{
	other::data_structures::sorting Sorting;
	other::l1_interfaces::latex_interface L;
	algebra::ring_theory::longinteger_object full_group_order;

	gens->group_order(full_group_order);

	ost << " Orbit " << i << " / " << Forest->nb_orbits << " : ";
	Forest->print_orbit_tex(ost, i);
	ost << " of length " << Forest->orbit_len[i] << "\\\\" << endl;

	strong_generators *gens_stab;

	gens_stab = stabilizer_orbit_rep(
		default_action,
		full_group_order, i, 0 /*verbose_level */);

	gens_stab->print_generators_tex(ost);

#if 0
	long int *Subgroup_elements_by_index;
	long int sz_subgroup;

	sz_subgroup = gens_stab->group_order_as_lint();

	if (sz_subgroup < 20) {
		gens->list_of_elements_of_subgroup(gens_stab,
			Subgroup_elements_by_index, sz_subgroup,
			0 /* verbose_level */);

		Sorting.lint_vec_heapsort(Subgroup_elements_by_index,
				sz_subgroup);

		ost << "The subgroup consists of the following "
				<< sz_subgroup << " elements:" << endl;
		ost << "$$" << endl;
		L.lint_vec_print_as_matrix(ost,
				Subgroup_elements_by_index, sz_subgroup,
				10 /* width */, true /* f_tex */);
		ost << "$$" << endl;

		FREE_lint(Subgroup_elements_by_index);

	}
#endif

	FREE_OBJECT(gens_stab);
}

void schreier::print_and_list_orbits_sorted_by_length_tex(
		std::ostream &ost)
{
	Forest->print_and_list_orbits_sorted_by_length(ost, true);
}

void schreier::print_and_list_orbits_sorted_by_length(
		std::ostream &ost)
{
	Forest->print_and_list_orbits_sorted_by_length(ost, false);
}


void schreier::print_and_list_orbits_and_stabilizer_sorted_by_length(
	std::ostream &ost, int f_tex,
	actions::action *default_action,
	algebra::ring_theory::longinteger_object &full_group_order)
{
	int i, h;
	int *Len;
	int *Perm;
	int *Perm_inv;
	other::data_structures::sorting Sorting;

	Len = NEW_int(Forest->nb_orbits);
	Perm = NEW_int(Forest->nb_orbits);
	Perm_inv = NEW_int(Forest->nb_orbits);
	Int_vec_copy(Forest->orbit_len, Len, Forest->nb_orbits);
	Sorting.int_vec_sorting_permutation(Len, Forest->nb_orbits,
			Perm, Perm_inv, true /*f_increasingly*/);

	ost << "There are " << Forest->nb_orbits << " orbits under a group with "
			<< Generators_and_images->gens.len << " generators:";
	if (f_tex) {
		ost << "\\\\" << endl;
	}
	else {
		ost << endl;
	}
	ost << "Orbit lengths: ";
	Int_vec_print(ost, Forest->orbit_len, Forest->nb_orbits);
	if (f_tex) {
		ost << "\\\\" << endl;
	}
	else {
		ost << endl;
	}
	if (!f_tex) {
		ost << "i : orbit_len[i]" << endl;
	}
	for (h = 0; h < Forest->nb_orbits; h++) {
		i = Perm_inv[h];
		if (f_tex) {
			print_and_list_orbit_and_stabilizer_tex(i,
					default_action, full_group_order, ost);
		}
		else {
			ost << " Orbit " << h << " / " << Forest->nb_orbits
					<< " is " << i << " : " << Forest->orbit_len[i];
			ost << " : ";
			Forest->print_orbit(ost, i);
			ost << endl;
		}
	}
	ost << endl;

	FREE_int(Len);
	FREE_int(Perm);
	FREE_int(Perm_inv);
}

void schreier::print_fancy(
	std::ostream &ost, int f_tex,
	actions::action *default_action,
	strong_generators *gens_full_group)
{
	int i, h;
	int *Len;
	int *Perm;
	int *Perm_inv;
	algebra::ring_theory::longinteger_object full_group_order;
	other::data_structures::sorting Sorting;

	gens_full_group->group_order(full_group_order);
	Len = NEW_int(Forest->nb_orbits);
	Perm = NEW_int(Forest->nb_orbits);
	Perm_inv = NEW_int(Forest->nb_orbits);
	Int_vec_copy(Forest->orbit_len, Len, Forest->nb_orbits);
	Sorting.int_vec_sorting_permutation(
			Len, Forest->nb_orbits,
			Perm, Perm_inv, true /*f_increasingly*/);

	ost << "There are " << Forest->nb_orbits << " orbits under a group with "
			<< Generators_and_images->gens.len << " generators:";
	if (f_tex) {
		ost << "\\\\" << endl;
	}
	else {
		ost << endl;
	}
	ost << "Orbit lengths: ";
	Int_vec_print(ost, Forest->orbit_len, Forest->nb_orbits);
	if (f_tex) {
		ost << "\\\\" << endl;
	}
	else {
		ost << endl;
	}
	if (!f_tex) {
		ost << "i : orbit_len[i]" << endl;
	}
	for (h = 0; h < Forest->nb_orbits; h++) {
		i = Perm_inv[h];
		if (f_tex) {
			//print_and_list_orbit_and_stabilizer_tex(
			// i, default_action, full_group_order, ost);
			print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
				i, default_action,
				gens_full_group, ost);
		}
		else {
			ost << " Orbit " << h << " / " << Forest->nb_orbits
					<< " is " << i << " : " << Forest->orbit_len[i];
			ost << " : ";
			Forest->print_orbit(ost, i);
			ost << endl;
		}
	}
	ost << endl;

	FREE_int(Len);
	FREE_int(Perm);
	FREE_int(Perm_inv);
}

void schreier::print_and_list_orbits_using_labels(
		std::ostream &ost, long int *labels)
{
	int i;

	ost << Forest->nb_orbits << " orbits" << endl;
	ost << "orbit group with " << Generators_and_images->gens.len << " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (i = 0; i < Forest->nb_orbits; i++) {
		ost << i << " : " << Forest->orbit_first[i]
			<< " : " << Forest->orbit_len[i];
		ost << " : ";
		Forest->print_orbit_using_labels(ost, i, labels);
		ost << endl;
	}
	ost << endl;
}

void schreier::print_tables(
		std::ostream &ost,
	int f_with_cosetrep)
{
	int i;
	int w; //  j, k;
	algebra::number_theory::number_theory_domain NT;

#if 0
	ost << gens.len << " generators:" << endl;
	for (i = 0; i < A->degree; i++) {
		ost << i;
		for (j = 0; j < gens.len; j++) {
			k = A->element_image_of(i, gens.ith(j), false);
			ost << " : " << k;
		}
		ost << endl;
	}
	ost << endl;
#endif
	w = NT.int_log10(Generators_and_images->A->degree) + 1;
	ost << "i : orbit[i] : orbit_inv[i] : prev[i] : label[i]";
	if (f_with_cosetrep)
		ost << " : coset_rep";
	ost << endl;

	if (Generators_and_images->A->degree < 100) {
		for (i = 0; i < Generators_and_images->A->degree; i++) {
			Generators_and_images->coset_rep(i, 0 /* verbose_level */);
			//coset_rep_inv(i);
			ost << setw(w) << i << " : " << " : "
				<< setw(w) << Forest->orbit[i] << " : "
				<< setw(w) << Forest->orbit_inv[i] << " : "
				<< setw(w) << Forest->prev[i] << " : "
				<< setw(w) << Forest->label[i];
			if (f_with_cosetrep) {
				ost << " : ";
				//A->element_print(Elt1, cout);
				Generators_and_images->A->Group_element->element_print_as_permutation(
						Generators_and_images->cosetrep, ost);
				ost << endl;
				Generators_and_images->A->Group_element->element_print_quick(
						Generators_and_images->cosetrep, ost);
			}
			ost << endl;
		}
	}
	else {
		cout << "too large to print" << endl;
	}
	ost << endl;
}

void schreier::print_tables_latex(
		std::ostream &ost,
	int f_with_cosetrep)
{
    int i;
    int w; //  j, k;
	algebra::number_theory::number_theory_domain NT;

#if 0
	ost << gens.len << " generators:" << endl;
	for (i = 0; i < A->degree; i++) {
		ost << i;
		for (j = 0; j < gens.len; j++) {
			k = A->element_image_of(i, gens.ith(j), false);
			ost << " : " << k;
			}
		ost << endl;
		}
	ost << endl;
#endif
	w = NT.int_log10(Generators_and_images->A->degree) + 1;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|c|c|" << endl;
	if (f_with_cosetrep) {
		ost << "c|";
	}
	ost << "}" << endl;
	ost << "\\hline" << endl;
	ost << "i & orbit & orbitinv & prev & label";
	if (f_with_cosetrep) {
		ost << "& cosetrep";
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < Generators_and_images->A->degree; i++) {
		Generators_and_images->coset_rep(i, 0 /* verbose_level */);
		//coset_rep_inv(i);
		ost << i << " & "
			<< setw(w) << Forest->orbit[i] << " & "
			<< setw(w) << Forest->orbit_inv[i] << " & "
			<< setw(w) << Forest->prev[i] << " & "
			<< setw(w) << Forest->label[i];
		if (f_with_cosetrep) {
			ost << " & ";
			//A->element_print(Elt1, cout);
			//A->element_print_as_permutation(cosetrep, ost);
			//ost << endl;
			Generators_and_images->A->Group_element->element_print_latex(
					Generators_and_images->cosetrep, ost);
		}
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;

		if (((i + 1) % 10) == 0) {
			ost << "\\end{array}" << endl;
			ost << "$$" << endl;
			ost << "$$" << endl;
			ost << "\\begin{array}{|c|c|c|c|c|" << endl;
			if (f_with_cosetrep) {
				ost << "c|";
			}
			ost << "}" << endl;
			ost << "\\hline" << endl;
			ost << "i & orbit & orbitinv & prev & label";
			if (f_with_cosetrep) {
				ost << "& cosetrep";
			}
			ost << "\\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\hline" << endl;
		}
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
	ost << endl;
}

void schreier::print_orbit_with_original_labels(
		std::ostream &ost, int orbit_no)
{
	int i, first, len;
	long int *v;
	long int *w;

	first = Forest->orbit_first[orbit_no];
	len = Forest->orbit_len[orbit_no];
	v = NEW_lint(len);
	for (i = 0; i < len; i++) {
		v[i] = Forest->orbit[first + i];
	}

	Generators_and_images->A->Induced_action->original_point_labels(
			v, len, w, 0 /*verbose_level*/);


	//int_vec_print(ost, v, len);
	//int_vec_heapsort(v, len);
	Lint_vec_print_fully(ost, w, len);

	FREE_lint(v);
	FREE_lint(w);
}

void schreier::print_orbit_sorted_with_original_labels_tex(
		std::ostream &ost,
		int orbit_no, int f_truncate, int max_length)
{
	other::l1_interfaces::latex_interface L;
	int i, first, len;
	long int *v;
	long int *w;
	other::data_structures::sorting Sorting;

	first = Forest->orbit_first[orbit_no];
	len = Forest->orbit_len[orbit_no];
	v = NEW_lint(len);
	for (i = 0; i < len; i++) {
		v[i] = Forest->orbit[first + i];
	}

	//int_vec_print(ost, v, len);
	Sorting.lint_vec_heapsort(v, len);
	//int_vec_print_fully(ost, v, len);

	Generators_and_images->A->Induced_action->original_point_labels(
			v, len, w, 0 /*verbose_level*/);

	if (f_truncate && len > max_length) {
		L.lint_set_print_tex(ost, w, max_length);
		ost << "truncated after " << max_length << " elements";
	}
	else {
		L.lint_set_print_tex(ost, w, len);
	}

	FREE_lint(v);
	FREE_lint(w);
}


void schreier::print_and_list_orbits_with_original_labels_tex(
		std::ostream &ost)
{
	int orbit_no;

	ost << Forest->nb_orbits << " orbits:\\\\" << endl;
	ost << "orbits under a group acting on a set of size "
			<< Generators_and_images->degree << ":\\\\" << endl;
	//ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (orbit_no = 0; orbit_no < Forest->nb_orbits; orbit_no++) {
		ost << " Orbit " << orbit_no << " / " << Forest->nb_orbits
				<< " of size " << Forest->orbit_len[orbit_no] << " : ";
		//print_and_list_orbit_tex(i, ost);
		print_orbit_sorted_with_original_labels_tex(ost,
				orbit_no, false /* f_truncate */, 0 /* max_length*/);
		ost << "\\\\" << endl;
	}
	ost << endl;
}





}}}

