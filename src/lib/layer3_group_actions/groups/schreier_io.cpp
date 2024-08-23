// schreier_io.cpp
//
// Anton Betten
// moved here from schreier.cpp: November 3, 2018
// originally started as schreier.cpp: December 9, 2003

#include "layer1_foundations/foundations.h"
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
		l1_interfaces::latex_interface L;

		L.head_easy(fp);

		print_generators_latex(fp);

		fp << "Orbit lengths: $";
		print_orbit_length_distribution(fp);
		fp << "$\\\\" << endl;

		print_and_list_orbits_tex(fp);

		if (A->degree < 100) {
			print_tables_latex(fp, f_with_cosetrep);
		}

		L.foot(fp);
	}
}


void schreier::print_orbit_lengths(
		std::ostream &ost)
{
	int i, f, l, m;
	int *orbit_len_sorted;
	int *sorting_perm;
	int *sorting_perm_inv;
	int nb_types;
	int *type_first;
	int *type_len;
	data_structures::sorting Sorting;

	Sorting.int_vec_classify(
			nb_orbits, orbit_len, orbit_len_sorted,
		sorting_perm, sorting_perm_inv,
		nb_types, type_first, type_len);

	ost << nb_orbits << " orbits: " << endl;
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		m = orbit_len_sorted[f];
		if (l > 1) {
			cout << l << " \\times ";
		}
		cout << m;
		if (i < nb_types - 1) {
			cout << ", ";
		}
	}
	ost << endl;
	FREE_int(orbit_len_sorted);
	FREE_int(sorting_perm);
	FREE_int(sorting_perm_inv);
	FREE_int(type_first);
	FREE_int(type_len);

}

void schreier::print_orbit_lengths_tex(
		std::ostream &ost)
{
	int i, f, l, m;
	int *orbit_len_sorted;
	int *sorting_perm;
	int *sorting_perm_inv;
	int nb_types;
	int *type_first;
	int *type_len;
	data_structures::sorting Sorting;

	Sorting.int_vec_classify(
			nb_orbits, orbit_len, orbit_len_sorted,
		sorting_perm, sorting_perm_inv,
		nb_types, type_first, type_len);

	ost << "There are " << nb_orbits << " orbits, the orbit lengths are $";
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		m = orbit_len_sorted[f];
		ost << m;
		if (l > 1) {
			ost << "^{" << l << "}";
		}
		if (i < nb_types - 1) {
			ost << ", ";
		}
	}
	ost << "$ \\\\" << endl;
	FREE_int(orbit_len_sorted);
	FREE_int(sorting_perm);
	FREE_int(sorting_perm_inv);
	FREE_int(type_first);
	FREE_int(type_len);

}

void schreier::print_fixed_points_tex(
		std::ostream &ost)
{
	int i, f, l, m, idx, h, fst, j, a;
	int *orbit_len_sorted;
	int *sorting_perm;
	int *sorting_perm_inv;
	int nb_types;
	int *type_first;
	int *type_len;
	data_structures::sorting Sorting;

	Sorting.int_vec_classify(
			nb_orbits, orbit_len, orbit_len_sorted,
		sorting_perm, sorting_perm_inv,
		nb_types, type_first, type_len);

	idx = -1;
	for (i = 0; i < nb_types; i++) {
		fst = type_first[i];
		m = orbit_len_sorted[fst];
		if (m == 1) {
			idx = i;
		}
	}
	if (idx >= 0) {
		fst = type_first[idx];
		l = type_len[idx];
		ost << "There are " << l << " fixed elements, they are:\\\\";
		for (h = 0; h < l; h++) {
			j = sorting_perm_inv[fst + h];
			f = orbit_first[j];
			a = orbit[f];
			ost << a << "\\\\" << endl;
		}
	}
	FREE_int(orbit_len_sorted);
	FREE_int(sorting_perm);
	FREE_int(sorting_perm_inv);
	FREE_int(type_first);
	FREE_int(type_len);

}


void schreier::print_orbit_length_distribution(
		std::ostream &ost)
{
	int *val, *mult, len;

	orbiter_kernel_system::Orbiter->Int_vec->distribution(
			orbit_len, nb_orbits, val, mult, len);
	orbiter_kernel_system::Orbiter->Int_vec->distribution_print(
			ost, val, mult, len);
	ost << endl;

	FREE_int(val);
	FREE_int(mult);
}

void schreier::print_orbit_length_distribution_to_string(
		std::string &str)
{
	int *val, *mult, len;

	orbiter_kernel_system::Orbiter->Int_vec->distribution(
			orbit_len, nb_orbits, val, mult, len);
	orbiter_kernel_system::Orbiter->Int_vec->distribution_print_to_string(
			str, val, mult, len);

	FREE_int(val);
	FREE_int(mult);
}


void schreier::print_orbit_reps(
		std::ostream &ost)
{
	int i, c, r;

	ost << nb_orbits << " orbits" << endl;
	ost << "orbits of a group with " << gens.len
			<< " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i] : rep" << endl;
	for (i = 0; i < nb_orbits; i++) {
		ost << setw(3) << i << " : " << setw(6)
				<< orbit_first[i] << " : " << setw(6) << orbit_len[i];
		c = orbit_first[i];
		r = orbit[c];
		ost << " : " << setw(6) << r << endl;
		//<< " : ";
		//print_orbit(ost, i);
		//ost << endl;
	}
	ost << endl;
}

void schreier::print(
		std::ostream &ost)
{
	int i;

	ost << nb_orbits << " orbits" << endl;
	ost << "orbit group with " << gens.len << " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {
		ost << i << " : " << orbit_first[i] << " : "
				<< orbit_len[i] << endl;
		//<< " : ";
		//print_orbit(ost, i);
		//ost << endl;
	}
	ost << endl;
}

void schreier::print_and_list_orbits_and_stabilizer(
		std::ostream &ost,
		actions::action *default_action,
		ring_theory::longinteger_object &go,
	void (*print_point)(
			std::ostream &ost, int pt, void *data),
	void *data)
{
	int i;

	ost << nb_orbits << " orbits" << endl;
	ost << "orbit group with " << gens.len << " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {

		sims *Stab;
		strong_generators *SG;

		ost << "Orbit " << i << " / " << nb_orbits
				<< " : of length " << orbit_len[i];
		ost << " is:" << endl;
		print_orbit(ost, i);
		ost << endl;
		ost << "Which is:" << endl;
		print_orbit_using_callback(ost, i, print_point, data);
		//ost << endl;
		ost << "The stabilizer of the element "
				<< orbit[orbit_first[i]] << " is:" << endl;
		point_stabilizer(default_action, go, Stab, i, 0 /* verbose_level */);

		SG = NEW_OBJECT(strong_generators);

		SG->init_from_sims(Stab, 0 /* verbose_level*/);
		SG->print_generators(ost, 0 /* verbose_level*/);
		FREE_OBJECT(SG);
		FREE_OBJECT(Stab);
	}
	ost << endl;
}

void schreier::print_and_list_orbits(
		std::ostream &ost)
{
	int i;

	ost << nb_orbits << " orbits" << endl;
	ost << "orbit group with " << gens.len << " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {
		ost << " Orbit " << i << " / " << nb_orbits
				<< " : " << orbit_first[i] << " : " << orbit_len[i];
		ost << " : ";
		print_orbit(ost, i);
		ost << endl;
	}
	ost << endl;
}

void schreier::print_and_list_orbits_with_original_labels(
		std::ostream &ost)
{
	int i;

	ost << nb_orbits << " orbits" << endl;
	ost << "orbit group with " << gens.len << " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {
		ost << " Orbit " << i << " / " << nb_orbits
				<< " : " << orbit_first[i] << " : " << orbit_len[i];
		ost << " : ";
		print_orbit(ost, i);
		ost << endl;
	}
	ost << endl;

}


void schreier::print_and_list_orbits_tex(
		std::ostream &ost)
{
	int orbit_no;

	ost << nb_orbits << " orbits:\\\\" << endl;
	ost << "orbits under a group with " << gens.len
			<< " generators acting on a set of size "
			<< A->degree << ":\\\\" << endl;
	//ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (orbit_no = 0; orbit_no < nb_orbits; orbit_no++) {
		print_and_list_orbit_tex(orbit_no, ost);
	}
	ost << endl;
}

void schreier::print_and_list_non_trivial_orbits_tex(
		std::ostream &ost)
{
	int orbit_no;

	ost << nb_orbits << " orbits:\\\\" << endl;
	ost << "orbits under a group with " << gens.len
			<< " generators acting on a set of size "
			<< A->degree << ":\\\\" << endl;
	//ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (orbit_no = 0; orbit_no < nb_orbits; orbit_no++) {

		if (orbit_len[orbit_no] > 1) {
			print_and_list_orbit_tex(orbit_no, ost);
		}
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
	for (i = 0; i < nb_orbits; i++) {
		print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
				i, default_action,
				gens, ost);
	}
}

void schreier::make_orbit_trees(
		std::ostream &ost,
		std::string &fname_mask,
		graphics::layered_graph_draw_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "schreier::make_orbit_trees" << endl;
	}
	int f_has_point_labels = false;
	long int *point_labels = NULL;

	if (f_v) {
		cout << "schreier::make_orbit_trees before draw_forest" << endl;
	}
	draw_forest(fname_mask,
		Opt,
		f_has_point_labels, point_labels,
		verbose_level - 1);
	if (f_v) {
		cout << "schreier::make_orbit_trees after draw_forest" << endl;
	}

	data_structures::string_tools ST;



	int i;
	for (i = 0; i < nb_orbits; i++) {
		string fname;

		fname = ST.printf_d(fname_mask, i);

		ost << "" << endl;
		ost << "\\bigskip" << endl;
		ost << "" << endl;
		ost << "Orbit " << i << " consisting of the following "
				<< orbit_len[i]
				<< " elements:" << endl;
		ost << "$$" << endl;
		L.int_set_print_tex(ost,
			orbit + orbit_first[i], orbit_len[i]);
		ost << "$$" << endl;
		ost << "" << endl;
		ost << "\\begin{center}" << endl;
		ost << "\\input " << fname << endl;
		ost << "\\end{center}" << endl;
		ost << "" << endl;
	}


	if (f_v) {
		cout << "schreier::make_orbit_trees" << endl;
	}
}

void schreier::print_and_list_orbits_with_original_labels_tex(
		std::ostream &ost)
{
	int orbit_no;

	ost << nb_orbits << " orbits:\\\\" << endl;
	ost << "orbits under a group with " << gens.len
			<< " generators acting on a set of size "
			<< A->degree << ":\\\\" << endl;
	//ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (orbit_no = 0; orbit_no < nb_orbits; orbit_no++) {
		ost << " Orbit " << orbit_no << " / " << nb_orbits
				<< " of size " << orbit_len[orbit_no] << " : ";
		//print_and_list_orbit_tex(i, ost);
		print_orbit_sorted_with_original_labels_tex(ost,
				orbit_no, false /* f_truncate */, 0 /* max_length*/);
		ost << "\\\\" << endl;
	}
	ost << endl;
}

void schreier::print_and_list_orbit_tex(
		int i, std::ostream &ost)
{
	ost << " Orbit " << i << " / " << nb_orbits
			<< " of size " << orbit_len[i] << " : ";
	//print_orbit_tex(ost, i);
	print_orbit_sorted_tex(ost, i, false /* f_truncate */, 0 /* max_length*/);
	ost << "\\\\" << endl;
}

void schreier::print_and_list_orbit_and_stabilizer_tex(
		int i,
		actions::action *default_action,
	ring_theory::longinteger_object &full_group_order,
	std::ostream &ost)
{
	ost << " Orbit " << i << " / " << nb_orbits << " : ";
	print_orbit_tex(ost, i);
	ost << " of length " << orbit_len[i] << "\\\\" << endl;

	strong_generators *gens;

	gens = stabilizer_orbit_rep(default_action,
		full_group_order, i, 0 /*verbose_level */);

	gens->print_generators_tex(ost);

	FREE_OBJECT(gens);
}

void schreier::write_orbit_summary(
		std::string &fname,
		actions::action *default_action,
		ring_theory::longinteger_object &full_group_order,
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

	Rep = NEW_lint(nb_orbits);
	Stab_order = NEW_lint(nb_orbits);
	Orbit_length = NEW_lint(nb_orbits);

	for (orbit_no = 0; orbit_no < nb_orbits; orbit_no++) {
		Rep[orbit_no] = orbit[orbit_first[orbit_no]];
		Orbit_length[orbit_no] = orbit_len[orbit_no];

		strong_generators *gens;

		gens = stabilizer_orbit_rep(
				default_action,
			full_group_order, orbit_no,
			0 /*verbose_level */);

		Stab_order[orbit_no] = gens->group_order_as_lint();

	}

	orbiter_kernel_system::file_io Fio;
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
			3 /* nb_vecs */, Vec, nb_orbits,
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

void schreier::print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
	int i, actions::action *default_action,
	strong_generators *gens, std::ostream &ost)
{
	data_structures::sorting Sorting;
	l1_interfaces::latex_interface L;
	ring_theory::longinteger_object full_group_order;

	gens->group_order(full_group_order);

	ost << " Orbit " << i << " / " << nb_orbits << " : ";
	print_orbit_tex(ost, i);
	ost << " of length " << orbit_len[i] << "\\\\" << endl;

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
	print_and_list_orbits_sorted_by_length(ost, true);
}

void schreier::print_and_list_orbits_sorted_by_length(
		std::ostream &ost)
{
	print_and_list_orbits_sorted_by_length(ost, false);
}

void schreier::print_and_list_orbits_sorted_by_length(
	std::ostream &ost, int f_tex)
{
	int i, h;
	int *Len;
	int *Perm;
	int *Perm_inv;
	data_structures::sorting Sorting;

	Len = NEW_int(nb_orbits);
	Perm = NEW_int(nb_orbits);
	Perm_inv = NEW_int(nb_orbits);
	Int_vec_copy(orbit_len, Len, nb_orbits);
	Sorting.int_vec_sorting_permutation(Len, nb_orbits,
			Perm, Perm_inv, true /*f_increasingly*/);

	ost << "There are " << nb_orbits
			<< " orbits under a group with "
			<< gens.len << " generators:";
	if (f_tex) {
		ost << "\\\\" << endl;
	}
	else {
		ost << endl;
	}
	ost << "Orbit lengths: ";
	Int_vec_print(ost, orbit_len, nb_orbits);
	if (f_tex) {
		ost << "\\\\" << endl;
	}
	else {
		ost << endl;
	}
	if (!f_tex) {
		ost << "i : orbit_len[i]" << endl;
	}
	for (h = 0; h < nb_orbits; h++) {
		i = Perm_inv[h];
		if (f_tex) {
			print_and_list_orbit_tex(i, ost);
		}
		else {
			ost << " Orbit " << h << " / " << nb_orbits
					<< " is " << i << " : " << orbit_len[i];
			ost << " : ";
			print_orbit(ost, i);
			ost << endl;
		}
	}
	ost << endl;

	FREE_int(Len);
	FREE_int(Perm);
	FREE_int(Perm_inv);
}

void schreier::print_and_list_orbits_and_stabilizer_sorted_by_length(
	std::ostream &ost, int f_tex,
	actions::action *default_action,
	ring_theory::longinteger_object &full_group_order)
{
	int i, h;
	int *Len;
	int *Perm;
	int *Perm_inv;
	data_structures::sorting Sorting;

	Len = NEW_int(nb_orbits);
	Perm = NEW_int(nb_orbits);
	Perm_inv = NEW_int(nb_orbits);
	Int_vec_copy(orbit_len, Len, nb_orbits);
	Sorting.int_vec_sorting_permutation(Len, nb_orbits,
			Perm, Perm_inv, true /*f_increasingly*/);

	ost << "There are " << nb_orbits << " orbits under a group with "
			<< gens.len << " generators:";
	if (f_tex) {
		ost << "\\\\" << endl;
	}
	else {
		ost << endl;
	}
	ost << "Orbit lengths: ";
	Int_vec_print(ost, orbit_len, nb_orbits);
	if (f_tex) {
		ost << "\\\\" << endl;
	}
	else {
		ost << endl;
	}
	if (!f_tex) {
		ost << "i : orbit_len[i]" << endl;
	}
	for (h = 0; h < nb_orbits; h++) {
		i = Perm_inv[h];
		if (f_tex) {
			print_and_list_orbit_and_stabilizer_tex(i,
					default_action, full_group_order, ost);
		}
		else {
			ost << " Orbit " << h << " / " << nb_orbits
					<< " is " << i << " : " << orbit_len[i];
			ost << " : ";
			print_orbit(ost, i);
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
	ring_theory::longinteger_object full_group_order;
	data_structures::sorting Sorting;

	gens_full_group->group_order(full_group_order);
	Len = NEW_int(nb_orbits);
	Perm = NEW_int(nb_orbits);
	Perm_inv = NEW_int(nb_orbits);
	Int_vec_copy(orbit_len, Len, nb_orbits);
	Sorting.int_vec_sorting_permutation(Len, nb_orbits,
			Perm, Perm_inv, true /*f_increasingly*/);

	ost << "There are " << nb_orbits << " orbits under a group with "
			<< gens.len << " generators:";
	if (f_tex) {
		ost << "\\\\" << endl;
	}
	else {
		ost << endl;
	}
	ost << "Orbit lengths: ";
	Int_vec_print(ost, orbit_len, nb_orbits);
	if (f_tex) {
		ost << "\\\\" << endl;
	}
	else {
		ost << endl;
	}
	if (!f_tex) {
		ost << "i : orbit_len[i]" << endl;
	}
	for (h = 0; h < nb_orbits; h++) {
		i = Perm_inv[h];
		if (f_tex) {
			//print_and_list_orbit_and_stabilizer_tex(
			// i, default_action, full_group_order, ost);
			print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
				i, default_action,
				gens_full_group, ost);
		}
		else {
			ost << " Orbit " << h << " / " << nb_orbits
					<< " is " << i << " : " << orbit_len[i];
			ost << " : ";
			print_orbit(ost, i);
			ost << endl;
		}
	}
	ost << endl;

	FREE_int(Len);
	FREE_int(Perm);
	FREE_int(Perm_inv);
}

void schreier::print_and_list_orbits_of_given_length(
	std::ostream &ost, int len)
{
	int i;


	ost << "Orbits of length " << len << ":" << endl;
	cout << "i : orbit_len[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {
		if (orbit_len[i] != len) {
			continue;
		}
		ost << " Orbit " << i << " / "
				<< nb_orbits << " : " << orbit_len[i];
		ost << " : ";
		print_orbit(ost, i);
		ost << endl;
	}
	ost << endl;
}

void schreier::print_and_list_orbits_using_labels(
		std::ostream &ost, long int *labels)
{
	int i;

	ost << nb_orbits << " orbits" << endl;
	ost << "orbit group with " << gens.len << " generators:" << endl;
	ost << "i : orbit_first[i] : orbit_len[i]" << endl;
	for (i = 0; i < nb_orbits; i++) {
		ost << i << " : " << orbit_first[i]
			<< " : " << orbit_len[i];
		ost << " : ";
		print_orbit_using_labels(ost, i, labels);
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
    number_theory::number_theory_domain NT;

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
	w = NT.int_log10(A->degree) + 1;
	ost << "i : orbit[i] : orbit_inv[i] : prev[i] : label[i]";
	if (f_with_cosetrep)
		ost << " : coset_rep";
	ost << endl;

	if (A->degree < 100) {
		for (i = 0; i < A->degree; i++) {
			coset_rep(i, 0 /* verbose_level */);
			//coset_rep_inv(i);
			ost << setw(w) << i << " : " << " : "
				<< setw(w) << orbit[i] << " : "
				<< setw(w) << orbit_inv[i] << " : "
				<< setw(w) << prev[i] << " : "
				<< setw(w) << label[i];
			if (f_with_cosetrep) {
				ost << " : ";
				//A->element_print(Elt1, cout);
				A->Group_element->element_print_as_permutation(cosetrep, ost);
				ost << endl;
				A->Group_element->element_print_quick(cosetrep, ost);
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
    number_theory::number_theory_domain NT;

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
	w = NT.int_log10(A->degree) + 1;
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
	for (i = 0; i < A->degree; i++) {
		coset_rep(i, 0 /* verbose_level */);
		//coset_rep_inv(i);
		ost << i << " & "
			<< setw(w) << orbit[i] << " & "
			<< setw(w) << orbit_inv[i] << " & "
			<< setw(w) << prev[i] << " & "
			<< setw(w) << label[i];
		if (f_with_cosetrep) {
			ost << " & ";
			//A->element_print(Elt1, cout);
			//A->element_print_as_permutation(cosetrep, ost);
			//ost << endl;
			A->Group_element->element_print_latex(cosetrep, ost);
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

void schreier::print_generators()
{
	int j;

	cout << gens.len << " generators in action "
			<< A->label << " of degree " << A->degree << ":" << endl;
	for (j = 0; j < gens.len; j++) {
		cout << "generator " << j << ":" << endl;
		//A->element_print(gens.ith(j), cout);
		A->Group_element->element_print_quick(gens.ith(j), cout);
		//A->element_print_as_permutation(gens.ith(j), cout);
		if (j < gens.len - 1) {
			cout << ", " << endl;
		}
	}
}

void schreier::print_generators_latex(
		std::ostream &ost)
{
	int j;

	ost << gens.len << " generators in action $"
			<< A->label_tex << "$ of degree "
			<< A->degree << ":\\\\" << endl;
	for (j = 0; j < gens.len; j++) {
		ost << "generator " << j << ":" << endl;
		//A->element_print(gens.ith(j), cout);
		ost << "$$" << endl;
		A->Group_element->element_print_latex(gens.ith(j), ost);
		//A->element_print_as_permutation(gens.ith(j), cout);
		if (j < gens.len - 1) {
			ost << ", " << endl;
		}
		ost << "$$" << endl;
	}
}

void schreier::print_generators_with_permutations()
{
	int j;

	cout << gens.len << " generators in action "
			<< A->label << " of degree "
			<< A->degree << ":" << endl;
	for (j = 0; j < gens.len; j++) {
		cout << "generator " << j << ":" << endl;
		//A->element_print(gens.ith(j), cout);
		A->Group_element->element_print_quick(gens.ith(j), cout);
		A->Group_element->element_print_as_permutation(gens.ith(j), cout);
		cout << endl;
		if (j < gens.len - 1) {
			cout << ", " << endl;
		}
	}
}

void schreier::print_orbit(
		int orbit_no)
{
	print_orbit(cout, orbit_no);
}

void schreier::print_orbit_using_labels(
		int orbit_no, long int *labels)
{
	print_orbit_using_labels(cout, orbit_no, labels);
}

void schreier::print_orbit(
		std::ostream &ost, int orbit_no)
{
	int i, first, len;
	long int *v;

	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	v = NEW_lint(len);
	for (i = 0; i < len; i++) {
		v[i] = orbit[first + i];
	}
	//int_vec_print(ost, v, len);
	//int_vec_heapsort(v, len);
	Lint_vec_print_fully(ost, v, len);

	FREE_lint(v);
}

void schreier::print_orbit_with_original_labels(
		std::ostream &ost, int orbit_no)
{
	int i, first, len;
	long int *v;
	long int *w;

	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	v = NEW_lint(len);
	for (i = 0; i < len; i++) {
		v[i] = orbit[first + i];
	}

	A->Induced_action->original_point_labels(
			v, len, w, 0 /*verbose_level*/);


	//int_vec_print(ost, v, len);
	//int_vec_heapsort(v, len);
	Lint_vec_print_fully(ost, w, len);

	FREE_lint(v);
	FREE_lint(w);
}

void schreier::print_orbit_tex(
		std::ostream &ost, int orbit_no)
{
	l1_interfaces::latex_interface L;
	int i, first, len;
	int *v;
	data_structures::sorting Sorting;

	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	v = NEW_int(len);
	for (i = 0; i < len; i++) {
		v[i] = orbit[first + i];
	}
	//int_vec_print(ost, v, len);
	Sorting.int_vec_heapsort(v, len);
	//int_vec_print_fully(ost, v, len);
	L.int_set_print_tex(ost, v, len);

	FREE_int(v);
}

void schreier::print_orbit_sorted_tex(
		std::ostream &ost,
		int orbit_no, int f_truncate, int max_length)
{
	l1_interfaces::latex_interface L;
	int i, first, len;
	int *v;
	data_structures::sorting Sorting;

	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	v = NEW_int(len);
	for (i = 0; i < len; i++) {
		v[i] = orbit[first + i];
	}
	//int_vec_print(ost, v, len);
	Sorting.int_vec_heapsort(v, len);
	//int_vec_print_fully(ost, v, len);
	if (f_truncate && len > max_length) {
		L.int_set_print_tex(ost, v, max_length);
		ost << "truncated after " << max_length << " elements";
	}
	else {
		L.int_set_print_tex(ost, v, len);
	}

	FREE_int(v);
}

void schreier::get_orbit_sorted(
		int *&v, int &len, int orbit_no)
{
	int i, first;
	data_structures::sorting Sorting;

	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	v = NEW_int(len);
	for (i = 0; i < len; i++) {
		v[i] = orbit[first + i];
	}
	//int_vec_print(ost, v, len);
	Sorting.int_vec_heapsort(v, len);

}

void schreier::print_orbit_sorted_with_original_labels_tex(
		std::ostream &ost,
		int orbit_no, int f_truncate, int max_length)
{
	l1_interfaces::latex_interface L;
	int i, first, len;
	long int *v;
	long int *w;
	data_structures::sorting Sorting;

	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	v = NEW_lint(len);
	for (i = 0; i < len; i++) {
		v[i] = orbit[first + i];
	}

	//int_vec_print(ost, v, len);
	Sorting.lint_vec_heapsort(v, len);
	//int_vec_print_fully(ost, v, len);

	A->Induced_action->original_point_labels(
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


void schreier::print_orbit_using_labels(
		std::ostream &ost,
	int orbit_no, long int *labels)
{
	int i, first, len;
	int *v;
	data_structures::sorting Sorting;

	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	v = NEW_int(len);
	for (i = 0; i < len; i++) {
		v[i] = labels[orbit[first + i]];
	}
	//int_vec_print(ost, v, len);
	Sorting.int_vec_heapsort(v, len);
	Int_vec_print_fully(ost, v, len);

	FREE_int(v);
}

void schreier::print_orbit_using_callback(
		std::ostream &ost,
	int orbit_no,
	void (*print_point)(
			std::ostream &ost, int pt, void *data),
	void *data)
{
	int i, first, len;

	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	for (i = 0; i < len; i++) {
		ost << orbit[first + i] << " which is " << endl;
		(*print_point)(ost, orbit[first + i], data);
	}
}

void schreier::print_orbit_type(
		int f_backwards)
{
	data_structures::tally C;

	C.init(orbit_len, nb_orbits, false, 0);
	C.print_bare(f_backwards);
}

void schreier::list_all_orbits_tex(
		std::ostream &ost)
{
	int i, j, f, l, a;

	ost << "$";
	for (i = 0; i < nb_orbits; i++) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			a = orbit[f + j];
			ost << a;
			if (j < l - 1) {
				ost << ", ";
			}
		}
		if (i < nb_orbits - 1) {
			ost << " \\mid ";
		}
	}
	ost << "$";
}

void schreier::print_orbit_through_labels(
		std::ostream &ost,
	int orbit_no, long int *point_labels)
{
	int i, first, len;
	long int *v;
	data_structures::sorting Sorting;

	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	v = NEW_lint(len);
	for (i = 0; i < len; i++) {
		v[i] = point_labels[orbit[first + i]];
	}
	Sorting.lint_vec_heapsort(v, len);
	Lint_vec_print_fully(ost, v, len);
	FREE_lint(v);
}

void schreier::print_orbit_sorted(
		std::ostream &ost, int orbit_no)
{
	int i, len;
	int *v;
	data_structures::sorting Sorting;

	len = orbit_first[orbit_no + 1] - orbit_first[orbit_no];
	v = NEW_int(len);
	for (i = 0; i < len; i++) {
		v[i] = orbit[orbit_first[orbit_no] + i];
	}
	Sorting.int_vec_heapsort(v, len);

	ost << "{ ";
	for (i = 0; i < len; i++) {
		if (f_print_function) {
			ost << v[i] << "=";
			(*print_function)(ost, v[i], print_function_data);
		}
		else {
			ost << v[i];
		}
		if (i < len - 1) {
			ost << ", ";
		}
	}
	ost << " }";
	FREE_int(v);
}

void schreier::print_orbit(
		int cur, int last)
{
	int i;

	cout << "schreier::print_orbit degree=" << A->degree << endl;
	cout << "i : orbit[i] : orbit_inv[i]" << endl;
	for (i = 0; i < A->degree; i++) {
		if (i == cur) {
			cout << ">";
		}
		if (i == last) {
			cout << ">";
		}
		cout << i << " : " << orbit[i]
			<< " : " << orbit_inv[i] << endl;
	}
	cout << endl;
}

void schreier::print_tree(
		int orbit_no)
{
	int *path;
	int i, j, l;

	path = NEW_int(A->degree);
	i = orbit_first[orbit_no];
	while (i < orbit_first[orbit_no + 1]) {
		trace_back_and_record_path(path, orbit[i], l);
		// now l is the distance from the root
		cout << l;
		for (j = 0; j < l; j++) {
			cout << " " << path[j];
		}
		cout << " 0 ";
		if (label[i] != -1) {
			cout << " $s_{" << label[i] << "}$";
		}
		cout << endl;
		i++;
	}
	FREE_int(path);
}

void schreier::draw_forest(
		std::string &fname_mask,
		graphics::layered_graph_draw_options *Opt,
		int f_has_point_labels, long int *point_labels,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "schreier::draw_forest" << endl;
	}

	data_structures::string_tools ST;



	for (i = 0; i < nb_orbits; i++) {
		string fname;

		fname = ST.printf_d(fname_mask, i);

		if (f_v) {
			cout << "schreier::draw_forest drawing orbit "
					<< i << " / " << nb_orbits << endl;
		}
		draw_tree(fname,
				Opt,
				i /* orbit_no */,
				f_has_point_labels, point_labels,
				verbose_level);
	}
	if (f_v) {
		cout << "schreier::draw_forest done" << endl;
	}
}

void schreier::get_orbit_by_levels(
		int orbit_no,
		data_structures::set_of_sets *&SoS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int fst, len;
	int *depth;
	int *horizontal_position;
	int i, j, l, max_depth;

	if (f_v) {
		cout << "schreier::get_orbit_by_levels" << endl;
		cout << "schreier::get_orbit_by_levels "
				"nb_gen = " << gens.len << endl;
	}

	fst = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	depth = NEW_int(len);
	horizontal_position = NEW_int(len);
	max_depth = 0;
	for (j = 0; j < len; j++) {
		trace_back(orbit[fst + j], l);
		l--;
		depth[j] = l;
		max_depth = MAX(max_depth, l);
	}
	int nb_layers;
	nb_layers = max_depth + 1;
	int *Nb;
	int *Nb1;
	int **Node;


	//classify C;
	//C.init(depth, len, false, 0);
	Nb = NEW_int(nb_layers);
	Nb1 = NEW_int(nb_layers);
	Int_vec_zero(Nb, nb_layers);
	Int_vec_zero(Nb1, nb_layers);
	for (j = 0; j < len; j++) {
		trace_back(orbit[fst + j], l);
		l--;
		horizontal_position[j] = Nb[l];
		Nb[l]++;
	}
	if (f_v) {
		cout << "schreier::get_orbit_by_levels" << endl;
		cout << "number of nodes at depth:" << endl;
		for (i = 0; i <= max_depth; i++) {
			cout << i << " : " << Nb[i] << endl;
		}
	}
	Node = NEW_pint(nb_layers);
	for (i = 0; i <= max_depth; i++) {
		Node[i] = NEW_int(Nb[i]);
	}
	for (j = 0; j < len; j++) {
		trace_back(orbit[fst + j], l);
		l--;
		Node[l][Nb1[l]] = j;
		Nb1[l]++;
	}
	SoS = NEW_OBJECT(data_structures::set_of_sets);

	SoS->init_basic_with_Sz_in_int(
			A->degree /* underlying_set_size */,
			nb_layers /* nb_sets */,
			Nb, verbose_level);

	for (i = 0; i <= max_depth; i++) {
		for (j = 0; j < Nb[i]; j++) {
			SoS->Sets[i][j] = orbit[fst + Node[i][j]];
		}
	}


	if (f_v) {
		cout << "schreier::get_orbit_by_levels done" << endl;
	}

}

void schreier::export_tree_as_layered_graph_and_save(
		int orbit_no,
		std::string &fname_mask,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph_and_save" << endl;
		cout << "schreier::export_tree_as_layered_graph_and_save "
				"degree = " << degree << endl;
		cout << "schreier::export_tree_as_layered_graph_and_save "
				"orbit_no = " << orbit_no << endl;
		cout << "schreier::export_tree_as_layered_graph_and_save "
				"nb_gen = " << gens.len << endl;
	}



	graph_theory::layered_graph *LG;

	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph_and_save "
				"before export_tree_as_layered_graph" << endl;
	}

	export_tree_as_layered_graph(
			orbit_no,
			LG,
			verbose_level);

	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph_and_save "
				"after export_tree_as_layered_graph" << endl;
	}


	data_structures::string_tools ST;



	string fname;

	fname = ST.printf_d(fname_mask, orbit_no);


	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph_and_save "
				"before LG->write_file" << endl;
	}
	LG->write_file(fname, 0 /*verbose_level*/);
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph_and_save "
				"after LG->write_file" << endl;
	}


	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph_and_save "
				"before FREE_OBJECT(LG)" << endl;
	}
	FREE_OBJECT(LG);
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph_and_save "
				"after FREE_OBJECT(LG)" << endl;
	}


	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph_and_save done" << endl;
	}
}


void schreier::export_tree_as_layered_graph(
		int orbit_no,
		graph_theory::layered_graph *&LG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int fst, len;
	int *depth;
	int *horizontal_position;
	int i, j, l, max_depth;

	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph" << endl;
		cout << "schreier::export_tree_as_layered_graph "
				"degree = " << degree << endl;
		cout << "schreier::export_tree_as_layered_graph "
				"orbit_no = " << orbit_no << endl;
		cout << "schreier::export_tree_as_layered_graph "
				"nb_gen = " << gens.len << endl;
	}


	if (f_v) {
		cout << "    i : orbit : o_inv :  prev : label" << endl;
		for (i = 0; i < degree; i++) {
			cout << setw(5) << i << " : " << setw(5) << orbit[i]
				<< " : " << setw(5) << orbit_inv[i]
				<< " : " << setw(5) << prev[i]
				<< " : " << setw(5) << label[i]
				<< endl;
		}

	}
	fst = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	depth = NEW_int(len);
	horizontal_position = NEW_int(len);
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph fst = " << fst << endl;
		cout << "schreier::export_tree_as_layered_graph len = " << len << endl;
	}

	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"computing max_depth" << endl;
	}
	max_depth = 0;
	for (j = 0; j < len; j++) {
		trace_back(orbit[fst + j], l);
		l--;
		depth[j] = l;
		max_depth = MAX(max_depth, l);
	}
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"max_depth = " << max_depth << endl;
	}

	int nb_layers;
	nb_layers = max_depth + 1;
	int *Nb;
	int *Nb1;
	int **Node;


	//classify C;
	//C.init(depth, len, false, 0);
	Nb = NEW_int(nb_layers);
	Nb1 = NEW_int(nb_layers);
	Int_vec_zero(Nb, nb_layers);
	Int_vec_zero(Nb1, nb_layers);


	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"computing number of nodes per level" << endl;
	}

	for (j = 0; j < len; j++) {
		l = depth[j];
		horizontal_position[j] = Nb[l];
		Nb[l]++;
	}
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph" << endl;
		cout << "number of nodes at depth:" << endl;
		for (i = 0; i <= max_depth; i++) {
			cout << i << " : " << Nb[i] << endl;
		}
	}
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"collecting nodes by level" << endl;
	}
	Node = NEW_pint(nb_layers);
	for (i = 0; i <= max_depth; i++) {
		Node[i] = NEW_int(Nb[i]);
	}
	for (j = 0; j < len; j++) {
		l = depth[j];
		Node[l][Nb1[l]] = j;
		Nb1[l]++;
	}
	for (i = 0; i <= max_depth; i++) {
		if (Nb[i] != Nb1[i]) {
			cout << "Nb[i] != Nb1[i]" << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "i : depth" << endl;
		for (i = 0; i < len; i++) {
			cout << i << " : " << depth[i] << endl;
		}
		cout << "depth : number of nodes" << endl;
		for (i = 0; i <= max_depth; i++) {
			cout << i << " : " << Nb[i] << endl;
		}
		cout << "j : depth : horizontal_position" << endl;
		for (j = 0; j < len; j++) {
			cout << j << " : " << depth[j] << " : " << horizontal_position[j] << endl;
		}
		cout << "depth : j : node" << endl;
		for (i = 0; i <= max_depth; i++) {
			for (j = 0; j < Nb[i]; j++) {
				cout << i << " : " << j << " : " << Node[i][j] << endl;
			}
		}
	}
	//data_structures::sorting Sorting;
	//graph_theory::layered_graph *LG;
	int n1, j2, N2;

	LG = NEW_OBJECT(graph_theory::layered_graph);

	string dummy;
	dummy.assign("");

	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"before LG->init" << endl;
	}
	//LG->add_data1(data1, 0/*verbose_level*/);
	LG->init(nb_layers, Nb, dummy, verbose_level);
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"after LG->init" << endl;
	}
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"before LG->place" << endl;
	}
	LG->place(verbose_level);
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"after LG->place" << endl;
	}
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"before adding edges" << endl;
	}
	for (i = 0; i <= max_depth; i++) {
		if (f_vv) {
			cout << "schreier::export_tree_as_layered_graph "
					"adding edges at depth "
					"i=" << i << " / " << max_depth
					<< " Nb[i]=" << Nb[i] << endl;
		}
		for (j = 0; j < Nb[i]; j++) {
			n1 = Node[i][j];
			if (f_vv) {
				cout << "schreier::export_tree_as_layered_graph "
						"adding edges "
						"i=" << i << " / " << max_depth
						<< " j=" << j
						<< " n1=" << n1
						<< " prev=" << prev[fst + n1]
						<< endl;
			}
			if (prev[fst + n1] != -1) {
				N2 = orbit_inv[prev[fst + n1]] - fst;
#if 0
				if (!Sorting.int_vec_search_linear(
						Node[i - 1], Nb[i - 1], N2, n2)) {
					cout << "cannot find ancestor node in level i - 1" << endl;
					exit(1);
				}
#endif
				j2 = horizontal_position[N2];
				if (f_vv) {
					cout << "schreier::export_tree_as_layered_graph "
							"adding edges "
							"i=" << i << " / " << max_depth
							<< " j=" << j << " n1=" << n1
							<< " N2=" << N2 << " j2=" << j2 << endl;
				}
				if (f_vv) {
					cout << "adding edge ("<< i - 1 << "," << j2 << ") "
							"-> (" << i << "," << j << ")" << endl;
				}
				LG->add_edge(i - 1, j2, i, j,
						0 /*verbose_level*/);
				//int l1, int n1, int l2, int n2,
			}
		}
	}
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"after adding edges" << endl;
	}
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"before adding node text" << endl;
	}
	for (j = 0; j < len; j++) {
		int a;

		a = orbit[fst + j];
		trace_back(a, l);
		l--;

		string text2;

		text2 = std::to_string(a);

		LG->add_text(
				l, horizontal_position[j], text2,
				0/*verbose_level*/);
		LG->add_node_data1(
				l, horizontal_position[j], a,
				0/*verbose_level*/);
	}
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"after adding node text" << endl;
	}

	data_structures::string_tools ST;


#if 0
	string fname;

	fname = ST.printf_d(fname_mask, orbit_no);


	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"before LG->write_file" << endl;
	}
	LG->write_file(fname, 0 /*verbose_level*/);
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"after LG->write_file" << endl;
	}
#endif

	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"before FREE_OBJECT(LG)" << endl;
	}
	//FREE_OBJECT(LG);
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"after FREE_OBJECT(LG)" << endl;
	}

	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"before FREE_int" << endl;
	}
	FREE_int(Nb);
	FREE_int(Nb1);
	FREE_int(depth);
	FREE_int(horizontal_position);
	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph "
				"after FREE_int" << endl;
	}

	if (f_v) {
		cout << "schreier::export_tree_as_layered_graph done" << endl;
	}
}

void schreier::draw_tree(
		std::string &fname,
		graphics::layered_graph_draw_options *Opt,
		int orbit_no,
		int f_has_point_labels, long int *point_labels,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *path;
	int *weight;
	int *placement_x;
	int i, j, last, max_depth = 0, len;


	if (f_v) {
		cout << "schreier::draw_tree" << endl;
	}
	if (f_v) {
		cout << "schreier::draw_tree Opt:" << endl;
		Opt->print();
	}

	if (orbit_no >= nb_orbits) {
		cout << "schreier::draw_tree orbit_no out of range" << endl;
		exit(1);
	}
	path = NEW_int(A->degree);
	weight = NEW_int(A->degree);
	placement_x = NEW_int(A->degree);

	i = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	last = i + len; //orbit_first[orbit_no + 1];
	if (f_v) {
		cout << "schreier::draw_tree first=" << i << endl;
		cout << "schreier::draw_tree len=" << len << endl;
		cout << "schreier::draw_tree last=" << last << endl;
	}

	for (j = 0; j < A->degree; j++) {
		weight[j] = 0;
		placement_x[j] = 0;
	}
	subtree_calc_weight(
			weight, max_depth, i, last);
	if (false) {
		cout << "the weights: " << endl;
		for (j = i; j < last; j++) {
			cout << j << " : " << weight[j] << " : " << endl;
		}
		cout << endl;
		cout << "max_depth = " << max_depth << endl;
	}

	if (f_v) {
		cout << "max_depth = " << max_depth << endl;
	}
	subtree_place(
			weight, placement_x, 0, Opt->xin, i, last);
	if (false) {
		for (j = i; j < last; j++) {
			cout << j << " : " << placement_x[j] << endl;
		}
		cout << endl;
	}
#if 0
	if (orbit_len[orbit_no] > 100) {
		f_circletext = false;
	}
#endif

	if (f_v) {
		cout << "schreier::draw_tree before draw_tree2" << endl;
	}
	draw_tree2(
			fname,
			Opt,
			weight, placement_x, max_depth, i, last,
			f_has_point_labels, point_labels,
			verbose_level - 2);
	if (f_v) {
		cout << "schreier::draw_tree after draw_tree2" << endl;
	}

	FREE_int(path);
	FREE_int(weight);
	FREE_int(placement_x);
	if (f_v) {
		cout << "schreier::draw_tree done" << endl;
	}
}

static void calc_y_coordinate(
		int &y, int l, int max_depth, int y_max)
{
	int dy;

	dy = (int)((double)y_max / (double)max_depth);
	//dy = (int)((double)1000000 / (double)max_depth);
	y = (int)(dy * ((double)l + 0.5));
	y = y_max - y;
}

void schreier::draw_tree2(
		std::string &fname,
		graphics::layered_graph_draw_options *Opt,
		int *weight, int *placement_x,
		int max_depth,
		int i,
		int last,
		int f_has_point_labels, long int *point_labels,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int x_min = 0, x_max = Opt->xin;
	//int y_min = 0, y_max = Opt->yin;
	int factor_1000 = 1000;
	string fname_full;

	if (f_v) {
		cout << "schreier::draw_tree2" << endl;
	}

	fname_full = fname + ".mp";

	if (f_v) {
		cout << "schreier::draw_tree2 before creating G" << endl;
	}

	graphics::mp_graphics G;

	G.init(fname_full, Opt, verbose_level - 1);

	G.header();
	G.begin_figure(factor_1000);

	int x = Opt->yin / 2;
	int y;
	if (f_v) {
		cout << "schreier::draw_tree2 "
				"before calc_y_coordinate" << endl;
	}
	calc_y_coordinate(
			y, 0, max_depth, Opt->yin);
	if (f_v) {
		cout << "schreier::draw_tree2 "
				"after calc_y_coordinate" << endl;
	}


	if (f_v) {
		cout << "schreier::draw_tree2 "
				"before subtree_draw_lines" << endl;
	}
	subtree_draw_lines(
			G, Opt,
			x, y, weight,
			placement_x, max_depth, i, last,
			Opt->yin,
			verbose_level);
	if (f_v) {
		cout << "schreier::draw_tree2 "
				"after subtree_draw_lines" << endl;
	}

	if (f_v) {
		cout << "schreier::draw_tree2 "
				"before subtree_draw_vertices" << endl;
	}
	subtree_draw_vertices(
			G, Opt,
			x, y, weight,
			placement_x, max_depth, i, last,
			f_has_point_labels, point_labels,
			Opt->yin,
			verbose_level);
	if (f_v) {
		cout << "schreier::draw_tree2 "
				"after subtree_draw_vertices" << endl;
	}

	int j, L, l, N;
	double avg;

	N = last - i;
	L = 0;
	for (j = i; j < last; j++) {
		trace_back(orbit[j], l);
		L += l;
	}
	avg = (double) L / (double)N;
	// x = 500000;
	x = Opt->xin / 2;
	calc_y_coordinate(y, max_depth + 1, max_depth, Opt->yin);
	int nb_gens;
	double H; // entropy

	nb_gens = gens.len;
	if (nb_gens) {
		H = log(N) / log(nb_gens);
	}
	else {
		H = 0.;
	}

	string s;
	s = "N=" + std::to_string(N) + ", avg=" + std::to_string(avg) + ",  gens=" + std::to_string(nb_gens) + ", H=" + std::to_string(H);
	G.aligned_text(x, y, "", s);


	G.end_figure();
	G.footer();
	if (f_v) {
		cout << "schreier::draw_tree2 done" << endl;
	}
}

void schreier::subtree_draw_lines(
		graphics::mp_graphics &G,
		graphics::layered_graph_draw_options *Opt,
		int parent_x, int parent_y, int *weight,
		int *placement_x, int max_depth, int i, int last,
		int y_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int pt = orbit[i];
	int x, y, l, ii;
	int Px[3], Py[3];

	if (f_v) {
		cout << "schreier::subtree_draw_lines" << endl;
	}
	trace_back(pt, l);
	// l is 1 if pt is the root.
	x = placement_x[pt];
	calc_y_coordinate(y, l, max_depth, y_max);

	//G.circle(x, y, 2000);
	Px[0] = parent_x;
	Py[0] = parent_y;
	Px[1] = x;
	Py[1] = y;
	Px[2] = (Px[0] + Px[1]) >> 1;
	Py[2] = (Py[0] + Py[1]) >> 1;
	//cout << "schreier::subtree_draw_lines "
	// << parent_x << "," << parent_y << " - "
	// << x << "," << y << endl;


#if 0
	int y1;
	calc_y_coordinate(y1, 0, max_depth);
	if (parent_x == 500000 && parent_y == y1) {
		}
	else {
		G.polygon2(Px, Py, 0, 1);
		}
#endif

	if (l > 1) {
		G.polygon2(Px, Py, 0, 1);
	}

	if (Opt->f_label_edges) {
		if (l > 1) {
			// if pt is not the root node:
			string s;
			s = "$\\alpha_{" + std::to_string(label[i]) + "}$";
			G.aligned_text(Px[2], Py[2], "", s);
		}
	}

	for (ii = i + 1; ii < last; ii++) {
		if (prev[ii] == pt) {
			subtree_draw_lines(G, Opt,
					x, y, weight, placement_x,
					max_depth, ii, last,
					y_max,
					verbose_level);
		}
	}

	if (f_v) {
		cout << "schreier::subtree_draw_lines done" << endl;
	}
}

void schreier::subtree_draw_vertices(
		graphics::mp_graphics &G,
		graphics::layered_graph_draw_options *Opt,
		int parent_x, int parent_y, int *weight,
		int *placement_x, int max_depth, int i, int last,
		int f_has_point_labels, long int *point_labels,
		int y_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int pt = orbit[i];
	int x, y, l, ii;
	//int Px[2], Py[2];
	string str;
	string s;

	if (f_v) {
		cout << "schreier::subtree_draw_vertices" << endl;
	}
	trace_back(pt, l);
	x = placement_x[pt];
	calc_y_coordinate(y, l, max_depth, y_max);

#if 0
	Px[0] = parent_x;
	Py[0] = parent_y;
	Px[1] = x;
	Py[1] = y;
	//cout << "schreier::subtree_draw_vertices "
	// << parent_x << "," << parent_y << " - " << x << "," << y << endl;
	//G.polygon2(Px, Py, 0, 1);
#endif

	for (ii = i + 1; ii < last; ii++) {
		if (prev[ii] == pt) {
			subtree_draw_vertices(G, Opt,
				x, y, weight, placement_x,
				max_depth, ii, last,
				f_has_point_labels, point_labels,
				y_max,
				verbose_level);
		}
	}
#if 0
	if (pt == 169303 || pt == 91479) {
		G.circle(x, y, 4 * rad);
		}
#endif
	if (f_has_point_labels) {
		str = std::to_string(point_labels[pt]);
	}
	else {
		str = std::to_string(pt);
	}
	if (Opt->f_nodes_empty) {
		s.assign("");
		G.circle_text(x, y, Opt->rad, s);
		//G.circle(x, y, rad);
		//G.aligned_text(Px, Py, 1, "tl", str);
	}
	else {
		s.assign(str);
		G.circle_text(x, y, Opt->rad, s);
	}
	if (f_v) {
		cout << "schreier::subtree_draw_vertices done" << endl;
	}
}

void schreier::subtree_place(
		int *weight, int *placement_x,
	int left, int right, int i, int last)
{
	int pt = orbit[i];
	int ii, l, w, w0, w1, lft, rgt, width;
	double dx;

	placement_x[pt] = (left + right) >> 1;
	w = weight[pt];
	width = right - left;
	dx = width / (double) (w - 1);
		// the node itself counts for the weight, so we subtract one
	w0 = 0;

	trace_back(pt, l);
	for (ii = i + 1; ii < last; ii++) {
		if (prev[ii] == pt) {
			w1 = weight[orbit[ii]];
			lft = left + (int)((double)w0 * dx);
			rgt = left + (int)((double)(w0 + w1) * dx);
			subtree_place(weight, placement_x, lft, rgt, ii, last);
			w0 += w1;
		}
	}
}

int schreier::subtree_calc_weight(
		int *weight,
	int &max_depth, int i, int last)
{
	int pt = orbit[i];
	int ii, l, w = 1, w1;

	trace_back(pt, l);
	if (l > max_depth)
		max_depth = l;
	for (ii = i + 1; ii < last; ii++) {
		if (prev[ii] == pt) {
			w1 = subtree_calc_weight(weight, max_depth, ii, last);
			w += w1;
		}
	}
	weight[pt] = w;
	return w;
}

int schreier::subtree_depth_first(
		std::ostream &ost,
		int *path, int i, int last)
{
	int pt = orbit[i];
	int ii, l, w = 1, w1;

	for (ii = i + 1; ii < last; ii++) {
		if (prev[ii] == pt) {


			trace_back_and_record_path(path, orbit[ii], l);
			// now l is the distance from the root
			print_path(ost, path, l);

			w1 = subtree_depth_first(ost, path, ii, last);
			w += w1;
		}
	}
	return w;
}

void schreier::print_path(
		std::ostream &ost, int *path, int l)
{
	int j;

	ost << l;
	for (j = 0; j < l; j++) {
		ost << " " << path[j];
	}
	ost << endl;
}

void schreier::write_to_file_csv(
		std::string &fname_csv, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::write_to_file_csv" << endl;
	}
	data_structures::spreadsheet S;

	int nb_rows;
	int nb_cols;
	int i;

	nb_rows = 1 + nb_orbits;
	nb_cols = 6;
	S.init_empty_table(nb_rows, nb_cols);

	std::string text;
	text.assign("OrbitNumber");
	S.fill_entry_with_text(0, 0, text);
	text.assign("OrbitLength");
	S.fill_entry_with_text(0, 1, text);
	text.assign("OrbitRep");
	S.fill_entry_with_text(0, 2, text);
	text.assign("OrbitElements");
	S.fill_entry_with_text(0, 3, text);
	text.assign("OrbitSVPrev");
	S.fill_entry_with_text(0, 4, text);
	text.assign("OrbitSVLabel");
	S.fill_entry_with_text(0, 5, text);
	for (i = 0; i < nb_orbits; i++) {

		int len;

		len = orbit_len[i];

		S.set_entry_lint(1 + i, 0, i);
		S.set_entry_lint(1 + i, 1, len);
		S.set_entry_lint(1 + i, 2, orbit[orbit_first[i]]);

		string str;

		str = "\"" + Int_vec_stringify(orbit + orbit_first[i], len) + "\"";
		S.fill_entry_with_text(1 + i, 3, str);


		str = "\"" + Int_vec_stringify(prev + orbit_first[i], len) + "\"";
		S.fill_entry_with_text(1 + i, 4, str);


		str = "\"" + Int_vec_stringify(label + orbit_first[i], len) + "\"";
		S.fill_entry_with_text(1 + i, 5, str);

	}
	S.save(fname_csv, 0/* verbose_level*/);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname_csv
				<< " of size " << Fio.file_size(fname_csv) << endl;
	}

	if (f_v) {
		cout << "schreier::write_to_file_csv done" << endl;
	}
}


void schreier::write_to_file_binary(
		std::ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a = 0, version = 1;

	if (f_v) {
		cout << "schreier::write_to_file_binary" << endl;
	}
	fp.write((char *) &a, sizeof(int));
	fp.write((char *) &version, sizeof(int));
	fp.write((char *) &A->degree, sizeof(int));
	fp.write((char *) &nb_orbits, sizeof(int));
	for (i = 0; i < nb_orbits; i++) {
		fp.write((char *) &orbit_first[i], sizeof(int));
		fp.write((char *) &orbit_len[i], sizeof(int));
	}
	for (i = 0; i < A->degree; i++) {
		fp.write((char *) &orbit[i], sizeof(int));
		fp.write((char *) &prev[i], sizeof(int));
		fp.write((char *) &label[i], sizeof(int));
		//fp.write((char *) &orbit_no[i], sizeof(int));
	}

	// ToDo
#if 0
	if (f_v) {
		cout << "schreier::write_to_file_binary before gens.write_to_file_binary" << endl;
	}
	gens.write_to_file_binary(fp, verbose_level - 1);
	if (f_v) {
		cout << "schreier::write_to_file_binary after gens.write_to_file_binary" << endl;
	}
	if (f_v) {
		cout << "schreier::write_to_file_binary before gens_inv.write_to_file_binary" << endl;
	}
	gens_inv.write_to_file_binary(fp, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "schreier::write_to_file_binary after gens_inv.write_to_file_binary" << endl;
	}
#endif


	if (f_v) {
		cout << "schreier::write_to_file_binary done" << endl;
	}
}

void schreier::read_from_file_binary(
		std::ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, deg, dummy, a, version;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "schreier::read_from_file_binary" << endl;
	}
	init2();
	fp.read((char *) &a, sizeof(int));
	if (a == 0) {
		fp.read((char *) &version, sizeof(int));
		fp.read((char *) &deg, sizeof(int));
	}
	else {
		version = 0;
		deg = a;
	}
	//fp.read((char *) &deg, sizeof(int));
	fp.read((char *) &nb_orbits, sizeof(int));
	if (deg != A->degree) {
		cout << "schreier::read_from_file_binary "
				"deg != A->degree" << endl;
		exit(1);
	}
	orbit_first = NEW_int(nb_orbits);
	orbit_len = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		fp.read((char *) &orbit_first[i], sizeof(int));
		fp.read((char *) &orbit_len[i], sizeof(int));
	}
	orbit = NEW_int(A->degree);
	orbit_inv = NEW_int(A->degree);
	prev = NEW_int(A->degree);
	label = NEW_int(A->degree);
	//orbit_no = NEW_int(A->degree);
	for (i = 0; i < A->degree; i++) {
		fp.read((char *) &orbit[i], sizeof(int));
		fp.read((char *) &prev[i], sizeof(int));
		fp.read((char *) &label[i], sizeof(int));
		if (version == 0) {
			fp.read((char *) &dummy, sizeof(int));
			//fp.read((char *) &orbit_no[i], sizeof(int));
		}
	}
	Combi.perm_inverse(orbit, orbit_inv, A->degree);

	// ToDo
#if 0
	gens.init(A, 0 /*verbose_level - 1*/);
	gens.read_from_file_binary(fp, 0 /*verbose_level - 1*/);
	gens_inv.init(A, 0 /*verbose_level - 1*/);
	gens_inv.read_from_file_binary(fp, 0 /*verbose_level - 1*/);
#endif

	if (f_v) {
		cout << "schreier::read_from_file_binary done" << endl;
	}
}


void schreier::write_file_binary(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "schreier::write_file_binary" << endl;
	}
	{
		ofstream fp(fname, ios::binary);

		write_to_file_binary(fp, verbose_level - 1);
	}
	cout << "schreier::write_file_binary Written file "
			<< fname << " of size " << Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "schreier::write_file_binary done" << endl;
	}
}

void schreier::read_file_binary(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "schreier::read_file_binary reading file "
				<< fname << " of size " << Fio.file_size(fname) << endl;
	}
	cout << "schreier::read_file_binary Reading file "
			<< fname << " of size " << Fio.file_size(fname) << endl;
	{
		ifstream fp(fname, ios::binary);

		read_from_file_binary(fp, verbose_level - 1);
	}
	if (f_v) {
		cout << "schreier::read_file_binary done" << endl;
	}
}

void schreier::list_elements_as_permutations_vertically(
		std::ostream &ost)
{
	if (f_images_only) {
		cout << "schreier::list_elements_as_permutations_vertically is not "
				"allowed if f_images_only is true" << endl;
		exit(1);
	}
	A->list_elements_as_permutations_vertically(&gens, ost);
}


}}}

