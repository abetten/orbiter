/*
 * group_element.cpp
 *
 *  Created on: Feb 7, 2023
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"



using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {



group_element::group_element()
{
	A = NULL;

	Elt1 = Elt2 = Elt3 = Elt4 = Elt5 = NULL;
	eltrk1 = eltrk2 = eltrk3 = NULL;
	elt_mult_apply = NULL;
	elt1 = NULL;
	element_rw_memory_object = NULL;

}

group_element::~group_element()
{
	int f_v = false;

	if (f_v) {
		cout << "group_element::~group_element" << endl;
	}

	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Elt2) {
		FREE_int(Elt2);
	}
	if (Elt3) {
		FREE_int(Elt3);
	}
	if (Elt4) {
		FREE_int(Elt4);
	}
	if (Elt5) {
		FREE_int(Elt5);
	}
	if (eltrk1) {
		FREE_int(eltrk1);
	}
	if (eltrk2) {
		FREE_int(eltrk2);
	}
	if (eltrk3) {
		FREE_int(eltrk3);
	}
	if (elt_mult_apply) {
		FREE_int(elt_mult_apply);
	}
	if (elt1) {
		FREE_uchar(elt1);
	}
	if (element_rw_memory_object) {
		FREE_char(element_rw_memory_object);
	}

	if (f_v) {
		cout << "group_element::~group_element done" << endl;
	}
}

void group_element::init(
		action *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_element::init" << endl;
	}
	group_element::A = A;


	if (f_v) {
		cout << "group_element::init done" << endl;
	}
}

void group_element::null_element_data()
{
	Elt1 = Elt2 = Elt3 = Elt4 = Elt5 = NULL;
	eltrk1 = eltrk2 = eltrk3 = NULL;
	elt_mult_apply = NULL;
	elt1 = NULL;
	element_rw_memory_object = NULL;

}

void group_element::allocate_element_data()
// this cannot go to init because we don't have A->elt_size_in_int yet.
{
	Elt1 = Elt2 = Elt3 = Elt4 = Elt5 = NULL;
	eltrk1 = eltrk2 = eltrk3 = NULL;
	elt_mult_apply = NULL;
	elt1 = NULL;
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);
	Elt5 = NEW_int(A->elt_size_in_int);
	eltrk1 = NEW_int(A->elt_size_in_int);
	eltrk2 = NEW_int(A->elt_size_in_int);
	eltrk3 = NEW_int(A->elt_size_in_int);
	elt_mult_apply = NEW_int(A->elt_size_in_int);
	elt1 = NEW_uchar(A->coded_elt_size_in_char);
	element_rw_memory_object = NEW_char(A->coded_elt_size_in_char);

}

int group_element::image_of(
		void *elt, int a)
{
	A->ptr->nb_times_image_of_called++;
	return (*A->ptr->ptr_element_image_of)(*A, a, elt, 0);
}

void group_element::image_of_low_level(
		void *elt,
		int *input, int *output,
		int verbose_level)
{
	A->ptr->nb_times_image_of_low_level_called++;
	(*A->ptr->ptr_element_image_of_low_level)(
			*A,
			input, output, elt, verbose_level);
}

int group_element::linear_entry_ij(
		void *elt, int i, int j)
{
	return (*A->ptr->ptr_element_linear_entry_ij)(*A, elt, i, j, 0);
}

int group_element::linear_entry_frobenius(
		void *elt)
{
	return (*A->ptr->ptr_element_linear_entry_frobenius)(*A, elt, 0);
}

void group_element::one(
		void *elt)
{
	(*A->ptr->ptr_element_one)(*A, elt, 0);
}

int group_element::is_one(
		void *elt)
{
	return element_is_one(elt, 0);
	//return (*ptr_element_is_one)(*A, elt, false);
}

void group_element::unpack(
		void *elt, void *Elt)
{
	A->ptr->nb_times_unpack_called++;
	(*A->ptr->ptr_element_unpack)(*A, elt, Elt, 0);
}

void group_element::pack(
		void *Elt, void *elt)
{
	A->ptr->nb_times_pack_called++;
	(*A->ptr->ptr_element_pack)(*A, Elt, elt, 0);
}

void group_element::retrieve(
		void *elt, int hdl)
{
	A->ptr->nb_times_retrieve_called++;
	(*A->ptr->ptr_element_retrieve)(*A, hdl, elt, 0);
}

int group_element::store(
		void *elt)
{
	A->ptr->nb_times_store_called++;
	return (*A->ptr->ptr_element_store)(*A, elt, 0);
}

void group_element::mult(
		void *a, void *b, void *ab)
{
	A->ptr->nb_times_mult_called++;
	(*A->ptr->ptr_element_mult)(*A, a, b, ab, 0);
}

void group_element::mult_apply_from_the_right(
		void *a, void *b)
// a := a * b
{
	(*A->ptr->ptr_element_mult)(*A, a, b, elt_mult_apply, 0);
	(*A->ptr->ptr_element_move)(*A, elt_mult_apply, a, 0);
}

void group_element::mult_apply_from_the_left(
		void *a, void *b)
// b := a * b
{
	(*A->ptr->ptr_element_mult)(*A, a, b, elt_mult_apply, 0);
	(*A->ptr->ptr_element_move)(*A, elt_mult_apply, b, 0);
}

void group_element::invert(
		void *a, void *av)
{
	A->ptr->nb_times_invert_called++;
	(*A->ptr->ptr_element_invert)(*A, a, av, 0);
}

void group_element::invert_in_place(
		void *a)
{
	(*A->ptr->ptr_element_invert)(*A, a, elt_mult_apply, 0);
	(*A->ptr->ptr_element_move)(*A, elt_mult_apply, a, 0);
}

void group_element::transpose(
		void *a, void *at)
{
	(*A->ptr->ptr_element_transpose)(*A, a, at, 0);
}

void group_element::move(
		void *a, void *b)
{
	(*A->ptr->ptr_element_move)(*A, a, b, 0);
}

void group_element::dispose(
		int hdl)
{
	(*A->ptr->ptr_element_dispose)(*A, hdl, 0);
}

void group_element::print(
		std::ostream &ost, void *elt)
{
	(*A->ptr->ptr_element_print)(*A, elt, ost);
}

void group_element::print_quick(
		std::ostream &ost, void *elt)
{
	(*A->ptr->ptr_element_print_quick)(*A, elt, ost);
}

void group_element::print_as_permutation(
		std::ostream &ost, void *elt)
{
	element_print_as_permutation(elt, ost);
}

void group_element::print_point(
		int a, std::ostream &ost)
{
	int verbose_level = 0;

	//cout << "action::print_point" << endl;
	(*A->ptr->ptr_print_point)(*A, a, ost, verbose_level);
}

void group_element::unrank_point(
		long int rk, int *v)
// v[low_level_point_size]
{
	int verbose_level = 0;

	if (A->ptr->ptr_unrank_point == NULL) {
		cout << "group_element::unrank_point "
				"ptr_unrank_point == NULL, "
				"label=" << A->ptr->label << endl;
		//exit(1);
		Int_vec_zero(v, A->low_level_point_size);
	}
	else {
		(*A->ptr->ptr_unrank_point)(*A, rk, v, verbose_level);
	}
}

long int group_element::rank_point(
		int *v)
// v[low_level_point_size]
{
	int verbose_level = 0;

	if (A->ptr->ptr_rank_point == NULL) {
		cout << "group_element::rank_point "
				"ptr_rank_point == NULL, "
				"label=" << A->ptr->label << endl;
		//exit(1);
		return 0;
	}
	return (*A->ptr->ptr_rank_point)(*A, v, verbose_level);
}

void group_element::code_for_make_element(
		int *data, void *elt)
{
	(*A->ptr->ptr_element_code_for_make_element)(*A, elt, data);
}

void group_element::print_for_make_element(
		ostream &ost, void *elt)
{
	(*A->ptr->ptr_element_print_for_make_element)(*A, elt, ost);
}

void group_element::print_for_make_element_no_commas(
		ostream &ost, void *elt)
{
	(*A->ptr->ptr_element_print_for_make_element_no_commas)(*A, elt, ost);
}



// #############################################################################

long int group_element::element_image_of(
		long int a, void *elt, int verbose_level)
{
	if (A->ptr == NULL) {
		cout << "group_element::element_image_of A->ptr == NULL" << endl;
		exit(1);
	}
	A->ptr->nb_times_image_of_called++;
	return (*A->ptr->ptr_element_image_of)(*A, a, elt, verbose_level);
}

void group_element::element_image_of_low_level(
		int *input, int *output, void *elt,
		int verbose_level)
{
	if (A->ptr->ptr_element_image_of_low_level == NULL) {
		cout << "group_element::element_image_of_low_level "
				"A->ptr is NULL" << endl;
		exit(1);
	}
	A->ptr->nb_times_image_of_low_level_called++;
	(*A->ptr->ptr_element_image_of_low_level)(
			*A,
			input, output, elt, verbose_level);
}

void group_element::element_one(
		void *elt, int verbose_level)
{
	(*A->ptr->ptr_element_one)(*A, elt, verbose_level);
}

int group_element::element_linear_entry_ij(
		void *elt,
		int i, int j, int verbose_level)
{
	return (*A->ptr->ptr_element_linear_entry_ij)(
			*A,
			elt, i, j, verbose_level);
}

int group_element::element_linear_entry_frobenius(
		void *elt,
		int verbose_level)
{
	return (*A->ptr->ptr_element_linear_entry_frobenius)(
			*A,
			elt, verbose_level);
}

int group_element::element_is_one(
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "group_element::element_is_one "
				"in action " << A->label << endl;
	}
	if (A->f_has_kernel && A->Kernel->A->base_len()) {
		int *Elt1;
		int drop_out_level, image;
		Elt1 = NEW_int(A->elt_size_in_int);
		if (f_v) {
			cout << "group_element::element_is_one "
					"before Kernel->strip" << endl;
		}
		ret = A->Kernel->strip(
				(int *)elt, Elt1 /* *residue */,
			drop_out_level, image,
			0 /*verbose_level*/);

		FREE_int(Elt1);
		if (f_v) {
			cout << "group_element::element_is_one "
					"returning " << ret << endl;
		}
		return ret;
	}
	ret = (*A->ptr->ptr_element_is_one)(*A, elt, verbose_level);
	if (f_v) {
		cout << "group_element::element_is_one "
				"returning " << ret << endl;
	}

	return ret;
}

void group_element::element_unpack(
		void *elt, void *Elt, int verbose_level)
{
	A->ptr->nb_times_unpack_called++;
	(*A->ptr->ptr_element_unpack)(*A, elt, Elt, verbose_level);
}

void group_element::element_pack(
		void *Elt, void *elt, int verbose_level)
{
	A->ptr->nb_times_pack_called++;
	(*A->ptr->ptr_element_pack)(*A, Elt, elt, verbose_level);
}

void group_element::element_retrieve(
		int hdl, void *elt, int verbose_level)
{
	A->ptr->nb_times_retrieve_called++;
	(*A->ptr->ptr_element_retrieve)(*A, hdl, elt, verbose_level);
}

int group_element::element_store(
		void *elt, int verbose_level)
{
	A->ptr->nb_times_store_called++;
	return (*A->ptr->ptr_element_store)(*A, elt, verbose_level);
}

void group_element::element_mult(
		void *a, void *b, void *ab, int verbose_level)
{
	A->ptr->nb_times_mult_called++;
	(*A->ptr->ptr_element_mult)(*A, a, b, ab, verbose_level);
}

void group_element::element_invert(
		void *a, void *av, int verbose_level)
{
	A->ptr->nb_times_invert_called++;
	(*A->ptr->ptr_element_invert)(*A, a, av, verbose_level);
}

void group_element::element_transpose(
		void *a, void *at, int verbose_level)
{
	(*A->ptr->ptr_element_transpose)(*A, a, at, verbose_level);
}

void group_element::element_move(
		void *a, void *b, int verbose_level)
{
	(*A->ptr->ptr_element_move)(*A, a, b, verbose_level);
}

void group_element::element_dispose(
		int hdl, int verbose_level)
{
	(*A->ptr->ptr_element_dispose)(*A, hdl, verbose_level);
}

void group_element::element_print(
		void *elt, std::ostream &ost)
{
	(*A->ptr->ptr_element_print)(*A, elt, ost);
}

void group_element::element_print_quick(
		void *elt, std::ostream &ost)
{
	if (A->ptr->ptr_element_print_quick == NULL) {
		cout << "group_element::element_print_quick "
				"ptr_element_print_quick == NULL" << endl;
		exit(1);
	}
	(*A->ptr->ptr_element_print_quick)(*A, elt, ost);
}

void group_element::element_print_latex(
		void *elt, std::ostream &ost)
{
	(*A->ptr->ptr_element_print_latex)(*A, elt, ost);
}

void group_element::element_print_latex_with_extras(
		void *elt, std::string &label, std::ostream &ost)
{
	//int *fp,;
	int n, ord;

	//fp = NEW_int(A->degree);
	//n = count_fixed_points(elt, fp, 0);
	n = count_fixed_points(elt, 0);
	//cout << "with " << n << " fixed points" << endl;
	//FREE_int(fp);

	ord = element_order(elt);

	ost << "$$" << label << endl;
	element_print_latex(elt, ost);
	ost << "$$" << endl << "of order $" << ord << "$ and with "
			<< n << " fixed points." << endl;
}


void group_element::element_print_latex_with_point_labels(
	void *elt, std::ostream &ost,
	std::string *Point_labels, void *data)
{
	(*A->ptr->ptr_element_print_latex_with_point_labels)(
				*A, elt, ost, Point_labels, data);
}

void group_element::element_print_verbose(
		void *elt, std::ostream &ost)
{
	(*A->ptr->ptr_element_print_verbose)(*A, elt, ost);
}

void group_element::element_code_for_make_element(
		void *elt, int *data)
{
	(*A->ptr->ptr_element_code_for_make_element)(*A, elt, data);
}

std::string group_element::element_stringify_code_for_make_element(
		void *elt)
{
	int *data;
	string s;

	data = NEW_int(A->make_element_size);
	(*A->ptr->ptr_element_code_for_make_element)(*A, elt, data);
	s = Int_vec_stringify(data, A->make_element_size);
	FREE_int(data);
	return s;
}

void group_element::element_print_for_make_element(
		void *elt, std::ostream &ost)
{
	(*A->ptr->ptr_element_print_for_make_element)(*A, elt, ost);
}

void group_element::element_print_for_make_element_no_commas(
		void *elt, std::ostream &ost)
{
	(*A->ptr->ptr_element_print_for_make_element_no_commas)(*A, elt, ost);
}

void group_element::element_print_as_permutation(
		void *elt, std::ostream &ost)
{
	element_print_as_permutation_with_offset(
			elt, ost, 0, false, true, 0);
}

void group_element::element_print_as_permutation_verbose(
		void *elt,
		std::ostream &ost, int verbose_level)
{
	element_print_as_permutation_with_offset(elt,
			ost, 0, false, true, verbose_level);
}


void group_element::compute_permutation(
		void *elt,
		int *perm, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;

	if (f_v) {
		cout << "group_element::compute_permutation" << endl;
	}
	for (i = 0; i < A->degree; i++) {

		if (f_vv) {
			cout << "group_element::compute_permutation" << i << endl;
		}
		j = element_image_of(i, elt, verbose_level - 2);
		perm[i] = j;
		if (f_vv) {
			cout << "group_element::compute_permutation "
					<< i << "->" << j << endl;
		}
	}
	if (f_v) {
		cout << "group_element::compute_permutation done" << endl;
	}
}

void group_element::cycle_type(
		void *elt,
		int *cycles, int &nb_cycles,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_element::cycle_type" << endl;
	}

	combinatorics::combinatorics_domain Combi;
	int *perm;

	perm = NEW_int(A->degree);
	if (f_v) {
		cout << "group_element::cycle_type "
				"before compute_permutation" << endl;
	}
	compute_permutation(
			elt,
			perm, 0 /* verbose_level */);
	if (f_v) {
		cout << "group_element::cycle_type "
				"after compute_permutation" << endl;
	}


	Combi.Permutations->perm_cycle_type(
			perm, A->degree, cycles, nb_cycles);

	FREE_int(perm);

	if (f_v) {
		cout << "group_element::cycle_type done" << endl;
	}
}

void group_element::element_print_as_permutation_with_offset(
	void *elt, std::ostream &ost,
	int offset, int f_do_it_anyway_even_for_big_degree,
	int f_print_cycles_of_length_one, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 2);
	int *perm;
	int f_cycle_length = false;
	int f_max_cycle_length = false;
	int max_cycle_length = 50;
	int f_orbit_structure = false;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "group_element::element_print_as_permutation_with_offset "
				"degree=" << A->degree << endl;
	}
	if (A->degree > 5000) {
		cout << "group_element::element_print_as_permutation_with_offset "
				"the degree is too large, we won't print the permutation" << endl;
		return;
	}

	perm = NEW_int(A->degree);

	if (f_v) {
		cout << "group_element::element_print_as_permutation_with_offset "
				"before compute_permutation" << endl;
	}
	compute_permutation(
			elt,
			perm, 0 /* verbose_level */);
	if (f_v) {
		cout << "group_element::element_print_as_permutation_with_offset "
				"after compute_permutation" << endl;
	}

	//perm_print(ost, perm, degree);
	if (f_v) {
		cout << "group_element::element_print_as_permutation_with_offset "
				"before Combi.perm_print_offset" << endl;
	}
	Combi.Permutations->perm_print_offset(
			ost, perm, A->degree, offset,
			f_print_cycles_of_length_one,
			f_cycle_length,
			f_max_cycle_length, max_cycle_length,
			f_orbit_structure,
			NULL, NULL);
	if (f_v) {
		cout << "group_element::element_print_as_permutation_with_offset "
				"after Combi.perm_print_offset" << endl;
	}
	//ost << endl;
	//perm_print_cycles_sorted_by_length(ost, degree, v);


#if 0
	if (degree) {
		if (f_v) {
			cout << "group_element::element_print_as_permutation_with_offset: "
					"calling perm_print_cycles_sorted_by_length_offset" << endl;
			}
		//ost << "perm of degree " << degree << " : ";
		//int_vec_print_fully(ost, v, degree);
		//ost << " = ";

		perm_print_cycles_sorted_by_length_offset(ost, degree, v, offset,
			f_do_it_anyway_even_for_big_degree, f_print_cycles_of_length_one,
			verbose_level);
		}
#endif


	//ost << endl;
	FREE_int(perm);
}

void group_element::element_print_as_permutation_with_offset_and_max_cycle_length(
	void *elt,
	std::ostream &ost, int offset,
	int max_cycle_length,
	int f_orbit_structure)
{
	int *perm;
	int f_print_cycles_of_length_one = false;
	int f_cycle_length = false;
	int f_max_cycle_length = true;
	combinatorics::combinatorics_domain Combi;

	perm = NEW_int(A->degree);

	compute_permutation(
			elt,
			perm, 0 /* verbose_level */);
#if 0
	for (i = 0; i < A->degree; i++) {
		j = element_image_of(i, elt, false);
		perm[i] = j;
	}
#endif
	//perm_print(ost, v, degree);

	Combi.Permutations->perm_print_offset(
			ost, perm, A->degree, offset,
			f_print_cycles_of_length_one,
			f_cycle_length,
			f_max_cycle_length, max_cycle_length, f_orbit_structure,
			NULL, NULL);

	FREE_int(perm);
}

void group_element::element_print_image_of_set(
		void *elt, int size, long int *set)
{
	long int i, j;

	for (i = 0; i < size; i++) {
		j = element_image_of(set[i], elt, false);
		cout << i << " -> " << j << endl;
	}
}

int group_element::element_signum_of_permutation(
		void *elt)
{
	int *perm;
	int sgn;
	combinatorics::combinatorics_domain Combi;

	perm = NEW_int(A->degree);

	compute_permutation(
			elt,
			perm, 0 /* verbose_level */);
#if 0
	for (i = 0; i < A->degree; i++) {
		j = element_image_of(i, elt, false);
		perm[i] = j;
	}
#endif

	sgn = Combi.Permutations->perm_signum(perm, A->degree);

	FREE_int(perm);

	return sgn;
}



void group_element::element_write_file_fp(
		int *Elt,
		std::ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;

	elt = element_rw_memory_object;
	if (elt == NULL) {
		cout << "group_element::element_write_file_fp elt == NULL" << endl;
		exit(1);
	}
	if (f_v) {
		element_print(Elt, cout);
		Int_vec_print(cout, Elt, A->elt_size_in_int);
		cout << endl;
	}
	element_pack(Elt, elt, false);
	fp.write(elt, A->coded_elt_size_in_char);
	//fwrite(elt, 1 /* size */, coded_elt_size_in_char /* items */, fp);
}

void group_element::element_read_file_fp(
		int *Elt,
		std::ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;

	elt = element_rw_memory_object;
	fp.read(elt, A->coded_elt_size_in_char);
	//fread(elt, 1 /* size */, coded_elt_size_in_char /* items */, fp);
	element_unpack(elt, Elt, false);
	if (f_v) {
		element_print(Elt, cout);
		Int_vec_print(cout, Elt, A->elt_size_in_int);
		cout << endl;
	}
}

void group_element::element_write_file(
		int *Elt,
		std::string &fname, int verbose_level)
// opens and closes the file
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

#if 0
	FILE *f2;
	f2 = fopen(fname, "wb");
	element_write_file_fp(Elt, f2, 0/* verbose_level*/);
	fclose(f2);
#else
	{
		ofstream fp(fname, ios::binary);

		element_write_file_fp(Elt, fp, 0/* verbose_level*/);
	}
#endif

	if (f_v) {
		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
}

void group_element::element_read_file(
		int *Elt,
		std::string &fname, int verbose_level)
// opens and closes the file
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "group_element::element_read_file: "
				"reading from file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}
#if 0
	FILE *f2;
	f2 = fopen(fname, "rb");
	element_read_file_fp(Elt, f2, 0/* verbose_level*/);

	fclose(f2);
#else
	{
		ifstream fp(fname, ios::binary);

		element_read_file_fp(Elt, fp, 0/* verbose_level*/);
	}

#endif
}

void group_element::element_write_to_memory_object(
		int *Elt,
		orbiter_kernel_system::memory_object *m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;

	if (f_v) {
		cout << "group_element::element_write_to_memory_object" << endl;
	}
	elt = element_rw_memory_object;

	element_pack(Elt, elt, false);
	m->append(A->coded_elt_size_in_char, elt, 0);
}


void group_element::element_read_from_memory_object(
		int *Elt,
		orbiter_kernel_system::memory_object *m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;
	int i;


	if (f_v) {
		cout << "group_element::element_read_from_memory_object" << endl;
	}
	elt = element_rw_memory_object;

	for (i = 0; i < A->coded_elt_size_in_char; i++) {
		m->read_char(elt + i);
	}
	element_unpack(elt, Elt, false);
}

void group_element::element_write_to_file_binary(
		int *Elt,
		std::ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;

	if (f_v) {
		cout << "group_element::element_write_to_file_binary" << endl;
	}
	if (f_v) {
		cout << "group_element::element_write_to_file_binary "
				"coded_elt_size_in_char=" << A->coded_elt_size_in_char << endl;
	}
	if (A->coded_elt_size_in_char == 0) {
		cout << "group_element::element_write_to_file_binary "
				"A->coded_elt_size_in_char == 0" << endl;
		exit(1);
	}
	//elt = NEW_char(coded_elt_size_in_char);
		// memory allocation should be avoided in a low-level function
	elt = element_rw_memory_object;
	if (elt == NULL) {
		cout << "group_element::element_write_to_file_binary "
				"elt == NULL" << endl;
		A->print_info();
		exit(1);
	}

	element_pack(Elt, elt, verbose_level);
	fp.write(elt, A->coded_elt_size_in_char);
	//FREE_char(elt);
	if (f_v) {
		cout << "group_element::element_write_to_file_binary done" << endl;
	}
}

void group_element::element_read_from_file_binary(
		int *Elt,
		std::ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;


	if (f_v) {
		cout << "group_element::element_read_from_file_binary" << endl;
	}
	//elt = NEW_char(coded_elt_size_in_char);
		// memory allocation should be avoided in a low-level function
	elt = element_rw_memory_object;

	if (f_v) {
		cout << "group_element::element_read_from_file_binary "
				"coded_elt_size_in_char=" << A->coded_elt_size_in_char << endl;
	}
	fp.read(elt, A->coded_elt_size_in_char);
	element_unpack(elt, Elt, verbose_level);
	//FREE_char(elt);
	if (f_v) {
		cout << "group_element::element_read_from_file_binary done" << endl;
	}
}

void group_element::random_element(
		groups::sims *S, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_element::random_element" << endl;
	}

	S->random_element(Elt, verbose_level - 1);

	if (f_v) {
		cout << "group_element::random_element done" << endl;
	}
}

int group_element::element_has_order_two(
		int *E1, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "group_element::element_has_order_two" << endl;
	}

	element_mult(E1, E1, Elt1, 0);
	if (is_one(Elt1)) {
		ret = true;
	}
	else {
		ret = false;
	}

	if (f_v) {
		cout << "group_element::element_has_order_two done" << endl;
	}
	return ret;
}

int group_element::product_has_order_two(
		int *E1,
		int *E2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "group_element::product_has_order_two" << endl;
	}

	element_mult(E1, E2, Elt1, 0);
	element_mult(Elt1, Elt1, Elt2, 0);
	if (is_one(Elt2)) {
		ret = true;
	}
	else {
		ret = false;
	}

	if (f_v) {
		cout << "group_element::product_has_order_two done" << endl;
	}
	return ret;
}

int group_element::product_has_order_three(
		int *E1,
		int *E2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "group_element::product_has_order_three" << endl;
	}

	element_mult(E1, E2, Elt1, 0);
	element_mult(Elt1, Elt1, Elt2, 0);
	element_mult(Elt2, Elt1, Elt3, 0);
	if (is_one(Elt3)) {
		ret = true;
	}
	else {
		ret = false;
	}

	if (f_v) {
		cout << "group_element::product_has_order_three done" << endl;
	}
	return ret;
}

int group_element::element_order(
		void *elt)
{
	int *cycle_type;
	int order;

	cycle_type = NEW_int(A->degree);
	order = element_order_and_cycle_type_verbose(
			elt, cycle_type, 0);
	FREE_int(cycle_type);
	return order;
}

int group_element::element_order_and_cycle_type(
		void *elt, int *cycle_type)
{
	return element_order_and_cycle_type_verbose(
			elt, cycle_type, 0);
}

int group_element::element_order_and_cycle_type_verbose(
		void *elt, int *cycle_type, int verbose_level)
// cycle_type[i - 1] is the number of cycles of length i for 1 le i le n
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *have_seen;
	long int l, l1, first, next, len, g, n, order = 1;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "group_element::element_order_verbose" << endl;
	}
	if (f_vv) {
		cout << "The element is:" << endl;
		element_print_quick(elt, cout);
		cout << "as permutation:" << endl;
		element_print_as_permutation(elt, cout);
	}
	n = A->degree;
	Int_vec_zero(cycle_type, A->degree);
	have_seen = NEW_int(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = false;
	}
	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
		}
		// work on cycle, starting with l:
		first = l;
		l1 = l;
		len = 1;
		while (true) {
			have_seen[l1] = true;
			next = element_image_of(l1, elt, 0);
			if (next > n) {
				cout << "group_element::element_order_verbose: next = "
					<< next << " > n = " << n << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "group_element::element_order_verbose "
						"have_seen[next]" << endl;
				exit(1);
			}
			l1 = next;
			len++;
		}
		cycle_type[len - 1]++;
		if (len == 1) {
			continue;
		}
		g = NT.gcd_lint(len, order);
		order *= len / g;
	}
	FREE_int(have_seen);
	if (f_v) {
		cout << "group_element::element_order_verbose "
				"done order=" << order << endl;
	}
	return order;
}

int group_element::element_order_if_divisor_of(
		void *elt, int o)
// returns the order of the element if o == 0
// if o != 0, returns the order of the element provided it divides o,
// 0 otherwise.
{
	int *have_seen;
	long int l, l1, first, next, len, g, n, order = 1;
	number_theory::number_theory_domain NT;

	n = A->degree;
	have_seen = NEW_int(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = false;
	}
	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
		}
		// work on cycle, starting with l:
		first = l;
		l1 = l;
		len = 1;
		while (true) {
			have_seen[l1] = true;
			next = element_image_of(l1, elt, 0);
			if (next > n) {
				cout << "group_element::element_order_if_divisor_of next = "
					<< next << " > n = " << n << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "group_element::element_order_if_divisor_of "
						"have_seen[next]" << endl;
				exit(1);
				}
			l1 = next;
			len++;
		}
		if (len == 1) {
			continue;
		}
		if (o && (o % len)) {
			FREE_int(have_seen);
			return 0;
		}
		g = NT.gcd_lint(len, order);
		order *= len / g;
	}
	FREE_int(have_seen);
	return order;
}

void group_element::element_print_base_images(
		int *Elt)
{
	element_print_base_images(Elt, cout);
}

void group_element::element_print_base_images(
		int *Elt, std::ostream &ost)
{
	element_print_base_images_verbose(Elt, cout, 0);
}

void group_element::element_print_base_images_verbose(
		int *Elt, std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *base_images;

	if (f_v) {
		cout << "group_element::element_print_base_images_verbose" << endl;
	}
	base_images = NEW_int(A->base_len());
	element_base_images_verbose(Elt, base_images, verbose_level - 1);
	ost << "base images: ";
	Int_vec_print(ost, base_images, A->base_len());
	FREE_int(base_images);
}

void group_element::element_base_images(
		int *Elt, int *base_images)
{
	element_base_images_verbose(Elt, base_images, 0);
}

void group_element::element_base_images_verbose(
		int *Elt, int *base_images, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, bi;

	if (f_v) {
		cout << "group_element::element_base_images_verbose" << endl;
	}
	for (i = 0; i < A->base_len(); i++) {
		bi = A->base_i(i);
		if (f_vv) {
			cout << "the " << i << "-th base point is "
					<< bi << " is mapped to:" << endl;
		}
		base_images[i] = element_image_of(bi, Elt, verbose_level - 2);
		if (f_vv) {
			cout << "the " << i << "-th base point is "
					<< bi << " is mapped to: " << base_images[i] << endl;
		}
	}
}

void group_element::minimize_base_images(
		int level,
		groups::sims *S, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *base_images1;
	int *base_images2;
	int *Elt1, *Elt2, *Elt3;
	int i, j, /*bi,*/ oj, j0 = 0, image0 = 0, image;


	if (f_v) {
		cout << "group_element::minimize_base_images" << endl;
		cout << "level=" << level << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	base_images1 = NEW_int(A->base_len());
	base_images2 = NEW_int(A->base_len());

	element_move(Elt, Elt1, 0);
	for (i = level; i < A->base_len(); i++) {
		element_base_images(Elt1, base_images1);
		//bi = base[i];
		if (f_vv) {
			cout << "level " << i << " S->orbit_len[i]="
					<< S->get_orbit_length(i) << endl;
		}
		for (j = 0; j < S->get_orbit_length(i); j++) {
			oj = S->get_orbit(i, j);
			image = element_image_of(oj, Elt1, 0);
			if (f_vv) {
				cout << "level " << i << " j=" << j
						<< " oj=" << oj << " image="
						<< image << endl;
			}
			if (j == 0) {
				image0 = image;
				j0 = 0;
			}
			else {
				if (image < image0) {
					if (f_vv) {
						cout << "level " << i << " coset j="
								<< j << " image=" << image
								<< "less that image0 = "
								<< image0 << endl;
					}
					image0 = image;
					j0 = j;
				}
			}
		}
		if (f_vv) {
			cout << "level " << i << " S->orbit_len[i]="
					<< S->get_orbit_length(i) << " j0=" << j0 << endl;
		}
		S->coset_rep(Elt3, i, j0, 0 /*verbose_level*/);
		if (f_vv) {
			cout << "cosetrep=" << endl;
			element_print_quick(Elt3, cout);
			if (A->degree < 500) {
				element_print_as_permutation(Elt3, cout);
				cout << endl;
			}
		}
		element_mult(Elt3, Elt1, Elt2, 0);
		element_move(Elt2, Elt1, 0);
		element_base_images(Elt1, base_images2);
		if (f_vv) {
			cout << "level " << i << " j0=" << j0 << endl;
			cout << "before: ";
			Int_vec_print(cout, base_images1, A->base_len());
			cout << endl;
			cout << "after : ";
			Int_vec_print(cout, base_images2, A->base_len());
			cout << endl;
		}
	}

	element_move(Elt1, Elt, 0);

	FREE_int(base_images1);
	FREE_int(base_images2);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
}

void group_element::element_conjugate_bvab(
		int *Elt_A,
		int *Elt_B, int *Elt_C,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Elt1, *Elt2;


	if (f_v) {
		cout << "group_element::element_conjugate_bvab" << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	if (f_v) {
		cout << "group_element::element_conjugate_bvab A=" << endl;
		element_print_quick(Elt_A, cout);
		cout << "group_element::element_conjugate_bvab B=" << endl;
		element_print_quick(Elt_B, cout);
	}

	element_invert(Elt_B, Elt1, 0);
	element_mult(Elt1, Elt_A, Elt2, 0);
	element_mult(Elt2, Elt_B, Elt_C, 0);
	if (f_v) {
		cout << "group_element::element_conjugate_bvab C=B^-1 * A * B" << endl;
		element_print_quick(Elt_C, cout);
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "group_element::element_conjugate_bvab done" << endl;
	}
}

void group_element::element_conjugate_babv(
		int *Elt_A,
		int *Elt_B, int *Elt_C,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Elt1, *Elt2;


	if (f_v) {
		cout << "group_element::element_conjugate_babv" << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	element_invert(Elt_B, Elt1, 0);
	element_mult(Elt_B, Elt_A, Elt2, 0);
	element_mult(Elt2, Elt1, Elt_C, 0);

	FREE_int(Elt1);
	FREE_int(Elt2);
}

void group_element::element_commutator_abavbv(
		int *Elt_A,
		int *Elt_B, int *Elt_C,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Elt1, *Elt2, *Elt3, *Elt4;


	if (f_v) {
		cout << "group_element::element_commutator_abavbv" << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);

	element_invert(Elt_A, Elt1, 0);
	element_invert(Elt_B, Elt2, 0);
	element_mult(Elt_A, Elt_B, Elt3, 0);
	element_mult(Elt3, Elt1, Elt4, 0);
	element_mult(Elt4, Elt2, Elt_C, 0);

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
}

int group_element::find_non_fixed_point(
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "group_element::find_non_fixed_point" << endl;
		cout << "degree=" << A->degree << endl;
	}
	for (i = 0; i < A->degree; i++) {
		j = element_image_of(i, elt, verbose_level - 1);
		if (j != i) {
			if (f_v) {
				cout << "moves " << i << " to " << j << endl;
			}
			return i;
		}
	}
	if (f_v) {
		cout << "cannot find non fixed point" << endl;
	}
	return -1;
}

#if 0
int group_element::find_fixed_points(
		void *elt,
		int *fixed_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, n = 0;

	if (f_v) {
		cout << "computing fixed points in action "
				<< A->label << " of degree " << A->degree << endl;
	}
	for (i = 0; i < A->degree; i++) {
		j = element_image_of(i, elt, 0);
		if (j == i) {
			fixed_points[n++] = i;
		}
	}
	if (f_v) {
		cout << "found " << n << " fixed points" << endl;
	}
	return n;
}
#endif

void group_element::compute_fixed_points(
		void *elt,
		std::vector<long int> &fixed_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j;

	if (f_v) {
		cout << "group_element::compute_fixed_points "
				"computing fixed points in action "
				<< A->label << " of degree " << A->degree << endl;
	}
	for (i = 0; i < A->degree; i++) {
		j = element_image_of(i, elt, 0);
		if (j == i) {
			fixed_points.push_back(i);
		}
	}
	if (f_v) {
		cout << "group_element::compute_fixed_points "
				"found " << fixed_points.size() << " fixed points" << endl;
	}
}

int group_element::count_fixed_points(
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, cnt = 0;

	if (f_v) {
		cout << "group_element::count_fixed_points "
				"counting fixed points in action "
				<< A->label << " of degree " << A->degree << endl;
	}
	for (i = 0; i < A->degree; i++) {
		j = element_image_of(i, elt, 0);
		if (j == i) {
			cnt++;
		}
	}
	if (f_v) {
		cout << "group_element::count_fixed_points done, "
				"found " << cnt << " fixed points" << endl;
	}
	return cnt;
}

int group_element::test_if_set_stabilizes(
		int *Elt,
		int size, long int *set, int verbose_level)
{
	long int *set1, *set2;
	int cmp;
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "group_element::test_if_set_stabilizes" << endl;
	}
	set1 = NEW_lint(size);
	set2 = NEW_lint(size);
	Lint_vec_copy(set, set1, size);

	Sorting.lint_vec_quicksort_increasingly(set1, size);

	map_a_set(set1, set2, size, Elt, 0);

	Sorting.lint_vec_quicksort_increasingly(set2, size);
	cmp = Sorting.lint_vec_compare(set1, set2, size);
	if (f_v) {
		cout << "the elements takes " << endl;
		Lint_vec_print(cout, set1, size);
		cout << endl << "to" << endl;
		Lint_vec_print(cout, set2, size);
		cout << endl;
		cout << "cmp = " << cmp << endl;
	}
	FREE_lint(set1);
	FREE_lint(set2);
	if (cmp == 0) {
		if (f_v) {
			cout << "group_element::test_if_set_stabilizes "
					"done, returning true" << endl;
		}
		return true;
	}
	else {
		if (f_v) {
			cout << "group_element::test_if_set_stabilizes "
					"done, returning false" << endl;
		}
		return false;
	}
}

void group_element::map_a_set(
		long int *set,
		long int *image_set,
		int n, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "group_element::map_a_set" << endl;
	}
	if (f_vv) {
		cout << "group element:" << endl;
		element_print_quick(Elt, cout);
		cout << endl;
		cout << "set: " << endl;
		Lint_vec_print(cout, set, n);
		cout << endl;
	}
	for (i = 0; i < n; i++) {
		if (f_vv) {
			cout << "group_element::map_a_set "
					"i=" << i << " / " << n
					<< ", computing image of " << set[i] << endl;
		}
		image_set[i] = element_image_of(set[i], Elt, verbose_level - 2);
		if (f_vv) {
			cout << "group_element::map_a_set "
					"i=" << i << " / " << n << ", image of "
					<< set[i] << " is " << image_set[i] << endl;
		}
	}
}

void group_element::map_a_set_and_reorder(
		long int *set,
		long int *image_set,
		int n, int *Elt, int verbose_level)
{
	data_structures::sorting Sorting;

	map_a_set(set, image_set, n, Elt, verbose_level);
	Sorting.lint_vec_heapsort(image_set, n);
}

void group_element::make_element_from_permutation_representation(
		int *Elt, groups::sims *S, int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *base_image;
	int i, a;

	if (f_v) {
		cout << "group_element::make_element_from_permutation_representation" << endl;
		cout << "group_element::make_element_from_permutation_representation A = ";
		A->print_info();
		cout << endl;
	}
	base_image = NEW_int(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		a = A->base_i(i);
		base_image[i] = data[a];
		if (base_image[i] >= A->degree) {
			cout << "group_element::make_element_from_permutation_representation "
					"base_image[i] >= degree" << endl;
			cout << "i=" << i << " base[i] = " << a
					<< " base_image[i]=" << base_image[i] << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "group_element::make_element_from_permutation_representation "
				"before make_element_from_base_image" << endl;
	}
	make_element_from_base_image(
			Elt, S, base_image, verbose_level - 1);
	if (f_v) {
		cout << "group_element::make_element_from_permutation_representation "
				"after make_element_from_base_image" << endl;
	}

	FREE_int(base_image);
	if (f_v) {
		cout << "group_element::make_element_from_permutation_representation done"
				<< endl;
	}
}

void group_element::make_element_from_base_image(
		int *Elt, groups::sims *S,
		int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *base_image;
	int *base_image_cur;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Elt4;
	int *Elt5;
	//sims *S;
#if 1
	int offset = 0;
	int f_do_it_anyway_even_for_big_degree = true;
	int f_print_cycles_of_length_one = false;
#endif

	int i, j, yi, z, b, c, bi;

	if (f_v) {
		cout << "group_element::make_element_from_base_image" << endl;
	}

	if (f_v) {
		cout << "group_element::make_element_from_base_image" << endl;
		cout << "base images: ";
		Int_vec_print(cout, data, A->base_len());
		cout << endl;
		A->print_info();
	}
#if 0
	if (!f_has_sims) {
		cout << "group_element::make_element_from_base_image "
				"fatal: does not have sims" << endl;
		exit(1);
	}
	S = Sims;
#endif
	if (f_v) {
		cout << "group_element::make_element_from_base_image action in Sims:" << endl;
		S->A->print_info();
	}
	base_image = NEW_int(A->base_len());
	base_image_cur = NEW_int(A->base_len());
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);
	Elt5 = NEW_int(A->elt_size_in_int);

	Int_vec_copy(data, base_image, A->base_len());

#if 0
	for (j = 0; j < A->base_len(); j++) {
		base_image[j] = data[j];
	}
#endif

	element_one(Elt3, 0);

	for (i = 0; i < A->base_len(); i++) {

		if (f_v) {
			cout << "group_element::make_element_from_base_image i = " << i << " / " << A->base_len() << endl;
		}


		compute_base_images(Elt3, base_image_cur, 0);

		if (f_v) {
			cout << "group_element::make_element_from_base_image wanted:";
			Int_vec_print(cout, data, A->base_len());
			cout << endl;
			cout << "group_element::make_element_from_base_image have  :";
			Int_vec_print(cout, base_image_cur, A->base_len());
			cout << endl;
		}

		element_invert(Elt3, Elt4, 0);

		bi = A->base_i(i);

		yi = base_image[i];

		z = element_image_of(yi, Elt4, 0);

		j = S->get_orbit_inv(i, z);
		//j = S->orbit_inv[i][z];

		if (f_vv) {
			cout << "group_element::make_element_from_base_image i=" << i << endl;
			cout << "group_element::make_element_from_base_image Elt3=" << endl;

			element_print_quick(Elt3, cout);

			element_print_base_images(Elt3);
			cout << endl;

			element_print_as_permutation_with_offset(
					Elt3, cout,
					offset, f_do_it_anyway_even_for_big_degree,
					f_print_cycles_of_length_one,
					0 /*verbose_level*/);
			cout << endl;

			cout << "group_element::make_element_from_base_image Elt3^-1=" << endl;

			element_print_quick(Elt4, cout);

			element_print_base_images(Elt4);
			cout << endl;

			element_print_as_permutation_with_offset(
					Elt4, cout,
					offset, f_do_it_anyway_even_for_big_degree,
					f_print_cycles_of_length_one,
					0 /*verbose_level*/);
			cout << endl;

			cout << "group_element::make_element_from_base_image i=" << i << " bi=" << bi
					<< " desired base image = yi = " << yi << " yi reverse = z = "
					<< z << " j=orbit_inv(i,z)=" << j << endl;
		}

		S->coset_rep(Elt5, i, j, 0);

		if (f_vv) {
			cout << "group_element::make_element_from_base_image cosetrep_i_j_=cosetrep_" << i << "_" << j << "_=" << endl;

			element_print_quick(Elt5, cout);

			element_print_base_images(Elt5);
			cout << endl;

			element_print_as_permutation_with_offset(
					Elt5, cout,
				offset, f_do_it_anyway_even_for_big_degree,
				f_print_cycles_of_length_one,
				0/*verbose_level*/);
			cout << endl;
		}

		element_mult(Elt5, Elt3, Elt4, 0);

		element_move(Elt4, Elt3, 0);

		if (f_vv) {
			cout << "group_element::make_element_from_base_image Elt3 = cosetrep*Elt3 = " << endl;
		}
		if (f_vv) {
			cout << "group_element::make_element_from_base_image after left multiplying, Elt3=" << endl;
			element_print_quick(Elt3, cout);

			element_print_base_images(Elt3);
			cout << endl;

			element_print_as_permutation_with_offset(
					Elt3, cout,
				offset, f_do_it_anyway_even_for_big_degree,
				f_print_cycles_of_length_one,
				0/*verbose_level*/);
			cout << endl;

			cout << "group_element::make_element_from_base_image computing image of bi=" << bi << endl;
		}

		c = element_image_of(bi, Elt3, 0);
		if (f_vv) {
			cout << "group_element::make_element_from_base_image bi=" << bi << " -> " << c << endl;
		}
		if (c != yi) {
			cout << "group_element::make_element_from_base_image "
					"fatal: bi does not map to y1, but to c where" << endl;
			cout << "bi=" << bi << endl;
			cout << "c=" << c << endl;
			cout << "yi=" << yi << endl;
			exit(1);
		}
	}

	// finshed

	element_move(Elt3, Elt, 0);

	// final test is the base images of Elt agree:

	for (i = 0; i < A->base_len(); i++) {
		yi = data[i];
		b = element_image_of(A->base_i(i), Elt, 0);
		if (yi != b) {
			cout << "group_element::make_element_from_base_image "
					"fatal: yi != b"
					<< endl;
			cout << "i=" << i << endl;
			cout << "base[i]=" << A->base_i(i) << endl;
			cout << "yi=" << yi << endl;
			cout << "b=" << b << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "group_element::make_element_from_base_image "
				"created element:" << endl;
		element_print_quick(Elt, cout);
	}
	FREE_int(base_image);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
	FREE_int(Elt5);
	if (f_v) {
		cout << "group_element::make_element_from_base_image done" << endl;
	}
}

void group_element::compute_base_images(
		int *Elt, int *base_images, int verbose_level)
{
	int i;

	for (i = 0; i < A->base_len(); i++) {
		base_images[i] = element_image_of(A->base_i(i), Elt, 0);
	}
}

void group_element::make_element_2x2(
		int *Elt, int a0, int a1, int a2, int a3)
{
	int data[4];

	data[0] = a0;
	data[1] = a1;
	data[2] = a2;
	data[3] = a3;
	make_element(Elt, data, 0);
}

void group_element::make_element_from_string(
		int *Elt,
		std::string &data_string, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_element::make_element_from_string" << endl;
	}
	int *data;
	int data_len;

	Int_vec_scan(data_string, data, data_len);

	if (f_v) {
		cout << "group_element::make_element_from_string data = ";
		Int_vec_print(cout, data, data_len);
		cout << endl;
	}

	make_element(Elt, data, verbose_level);

	FREE_int(data);

	if (f_v) {
		cout << "group_element::make_element_from_string Elt = " << endl;
		element_print_quick(Elt, cout);
	}

	if (f_v) {
		cout << "action::make_element_from_string done" << endl;
	}
}

void group_element::make_element(
		int *Elt, int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_element::make_element" << endl;
	}
	if (A->type_G == product_action_t) {

		if (f_v) {
			cout << "group_element::make_element product_action_t" << endl;
		}

		induced_actions::product_action *PA;

		PA = A->G.product_action_data;
		PA->make_element(Elt, data, verbose_level);
		//PA->A1->make_element(Elt, data, verbose_level);
		//PA->A2->make_element(Elt + PA->A1->elt_size_in_int,
		//	data + PA->A1->make_element_size, verbose_level);
	}
	else if (A->type_G == action_on_sets_t) {
		if (f_v) {
			cout << "group_element::make_element action_on_sets_t" << endl;
		}
		A->subaction->Group_element->make_element(
				Elt, data, verbose_level);
	}
	else if (A->type_G == action_on_pairs_t) {
		if (f_v) {
			cout << "group_element::make_element action_on_pairs_t" << endl;
		}
		A->subaction->Group_element->make_element(
				Elt, data, verbose_level);
	}
	else if (A->type_G == matrix_group_t) {
		if (f_v) {
			cout << "group_element::make_element matrix_group_t" << endl;
		}
		A->G.matrix_grp->Element->make_element(
				Elt, data, verbose_level);
	}
	else if (A->type_G == wreath_product_t) {
		if (f_v) {
			cout << "group_element::make_element wreath_product_t" << endl;
		}
		A->G.wreath_product_group->make_element(
				Elt, data, verbose_level);
	}
	else if (A->type_G == direct_product_t) {
		if (f_v) {
			cout << "group_element::make_element direct_product_t" << endl;
		}
		A->G.direct_product_group->make_element(
				Elt, data, verbose_level);
	}
	else if (A->type_G == polarity_extension_t) {
		if (f_v) {
			cout << "group_element::make_element polarity_extension_t" << endl;
		}
		A->G.Polarity_extension->make_element(
				Elt, data, verbose_level);
	}
	else if (A->f_has_subaction) {
		if (f_v) {
			cout << "group_element::make_element subaction" << endl;
		}
		A->subaction->Group_element->make_element(
				Elt, data, verbose_level);
	}
	else if (A->type_G == perm_group_t) {
		if (f_v) {
			cout << "group_element::make_element perm_group_t" << endl;
		}
		A->G.perm_grp->make_element(
				Elt, data, verbose_level);
	}
	else {
		cout << "group_element::make_element unknown type_G: ";
		A->print_symmetry_group_type(cout);
		cout << endl;
		exit(1);
	}
}

void group_element::element_power_int_in_place(
		int *Elt,
		int n, int verbose_level)
{
	int *Elt2;
	int *Elt3;
	int *Elt4;

	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);
	move(Elt, Elt2);
	one(Elt3);
	while (n) {
		if (ODD(n)) {
			mult(Elt2, Elt3, Elt4);
			move(Elt4, Elt3);
		}
		mult(Elt2, Elt2, Elt4);
		move(Elt4, Elt2);
		n >>= 1;
	}
	move(Elt3, Elt);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
}

void group_element::word_in_ab(
		int *Elt1, int *Elt2, int *Elt3,
		const char *word, int verbose_level)
{
	int *Elt4;
	int *Elt5;
	int l, i;


	Elt4 = NEW_int(A->elt_size_in_int);
	Elt5 = NEW_int(A->elt_size_in_int);
	one(Elt4);
	l = strlen(word);
	for (i = 0; i < l; i++) {
		if (word[i] == 'a') {
			mult(Elt4, Elt1, Elt5);
			move(Elt5, Elt4);
		}
		else if (word[i] == 'b') {
			mult(Elt4, Elt2, Elt5);
			move(Elt5, Elt4);
		}
		else {
			cout << "word must consist of a and b" << endl;
			exit(1);
		}
	}
	move(Elt4, Elt3);

	FREE_int(Elt4);
	FREE_int(Elt5);
}

void group_element::evaluate_word(
		int *Elt, int *word, int len,
		data_structures_groups::vector_ge *gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_element::evaluate_word" << endl;
	}
	int *Elt1;
	int *Elt2;
	int i;


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	one(Elt1);

	for (i = 0; i < len; i++) {
		mult(Elt1, gens->ith(word[i]), Elt2);
		move(Elt2, Elt1);
	}
	move(Elt1, Elt);

	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "group_element::evaluate_word done" << endl;
	}
}

int group_element::check_if_in_set_stabilizer(
		int *Elt,
		int size, long int *set, int verbose_level)
{
	int i, a, b, idx;
	long int *ordered_set;
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;

	ordered_set = NEW_lint(size);
	Lint_vec_copy(set, ordered_set, size);

	Sorting.lint_vec_heapsort(ordered_set, size);

	for (i = 0; i < size; i++) {
		a = ordered_set[i];
		b = element_image_of(a, Elt, 0);
		if (!Sorting.lint_vec_search(
				ordered_set, size, b, idx, 0)) {
			if (f_v) {
				cout << "group_element::check_if_in_set_stabilizer fails" << endl;
				cout << "set: ";
				Lint_vec_print(cout, set, size);
				cout << endl;
				cout << "ordered_set: ";
				Lint_vec_print(cout, ordered_set, size);
				cout << endl;
				cout << "image of " << i << "-th element "
						<< a << " is " << b
						<< " is not found" << endl;
			}
			FREE_lint(ordered_set);
			return false;
		}
	}
	FREE_lint(ordered_set);
	return true;

}

void group_element::check_if_in_set_stabilizer_debug(
		int *Elt,
		int size, long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_element::check_if_in_set_stabilizer_debug" << endl;
		cout << "group_element::check_if_in_set_stabilizer_debug "
				"size = " << size << endl;
	}

	int i, a, b, idx;
	long int *ordered_set;
	data_structures::sorting Sorting;

	ordered_set = NEW_lint(size);
	Lint_vec_copy(set, ordered_set, size);

	Sorting.lint_vec_heapsort(ordered_set, size);
	if (f_v) {
		cout << "group_element::check_if_in_set_stabilizer_debug "
				"sorted set:" << endl;
		Lint_vec_print(cout, ordered_set, size);
		cout << endl;
	}
	for (i = 0; i < size; i++) {
		a = ordered_set[i];
		b = element_image_of(a, Elt, 0);
		if (f_v) {
			cout << "group_element::check_if_in_set_stabilizer_debug "
					<< a << " -> " << b << endl;
			cout << "a=" << a << " = ";
			print_point(a, cout);
			cout << endl;
			cout << "b=" << b << " = ";
			print_point(b, cout);
			cout << endl;
		}
		if (!Sorting.lint_vec_search(
					ordered_set, size, b, idx, 0)) {
			if (f_v) {
				cout << "group_element::check_if_in_set_stabilizer fails" << endl;
				cout << "set: ";
				Lint_vec_print(cout, set, size);
				cout << endl;
				cout << "ordered_set: ";
				Lint_vec_print(cout, ordered_set, size);
				cout << endl;
				cout << "image of " << i << "-th element "
						<< a << " is " << b
						<< " is not found" << endl;
			}
			FREE_lint(ordered_set);
			exit(1);
		}
	}
	FREE_lint(ordered_set);

}


int group_element::check_if_transporter_for_set(
		int *Elt,
		int size,
		long int *set1, long int *set2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 4);
	int i, a, b, idx;
	long int *ordered_set2;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "group_element::check_if_transporter_for_set "
				"size=" << size << endl;
	}
	if (f_vv) {
		Lint_vec_print(cout, set1, size);
		cout << endl;
		Lint_vec_print(cout, set2, size);
		cout << endl;
		element_print(Elt, cout);
		cout << endl;
	}
	ordered_set2 = NEW_lint(size);
	Lint_vec_copy(set2, ordered_set2, size);
#if 0
	for (i = 0; i < size; i++) {
		ordered_set2[i] = set2[i];
	}
#endif
	Sorting.lint_vec_heapsort(ordered_set2, size);
	if (f_vv) {
		cout << "sorted target set:" << endl;
		Lint_vec_print(cout, ordered_set2, size);
		cout << endl;
	}
	for (i = 0; i < size; i++) {
		a = set1[i];
		if (false) {
			cout << "i=" << i << " a=" << a << endl;
		}
		b = element_image_of(a, Elt, 0);
		if (false) {
			cout << "i=" << i << " a=" << a << " b=" << b << endl;
		}
		if (!Sorting.lint_vec_search(
				ordered_set2, size, b, idx, 0)) {
			if (f_v) {
				cout << "group_element::check_if_transporter_for_set fails" << endl;
				cout << "set1   : ";
				Lint_vec_print(cout, set1, size);
				cout << endl;
				cout << "set2   : ";
				Lint_vec_print(cout, set2, size);
				cout << endl;
				cout << "ordered: ";
				Lint_vec_print(cout, ordered_set2, size);
				cout << endl;
				cout << "image of " << i << "-th element "
						<< a << " is " << b
						<< " is not found" << endl;
			}
			FREE_lint(ordered_set2);
			return false;
		}
	}
	FREE_lint(ordered_set2);
	return true;

}

#if 0
void group_element::compute_fixed_objects_in_PG(
		int up_to_which_rank,
		geometry::projective_space *P,
	int *Elt,
	std::vector<std::vector<long int> > &Fix,
	int verbose_level)
// creates temporary actions using induced_action_on_grassmannian
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_element::compute_fixed_objects_in_PG" << endl;
	}

	if (up_to_which_rank < 1) {
		cout << "group_element::compute_fixed_objects_in_PG "
				"up_to_which_rank < 1" << endl;
		exit(1);
	}




	if (f_v) {
		cout << "group_element::compute_fixed_objects_in_PG "
				"computing fixed points" << endl;
	}
	{
		vector<long int> fixed_points;

		compute_fixed_points(
				Elt,
				fixed_points, 0 /* verbose_level */);

		Fix.push_back(fixed_points);
	}

	int dimension;

	for (dimension = 2; dimension <= up_to_which_rank; dimension++) {

		if (f_v) {
			cout << "group_element::compute_fixed_objects_in_PG "
					"computing fixed subspaces of rank " << dimension << endl;
		}

		vector<long int> fixpoints;


		compute_fixed_points_in_induced_action_on_grassmannian(
			Elt,
			dimension,
			fixpoints,
			0 /*verbose_level*/);

		Fix.push_back(fixpoints);


	}


	if (f_v) {
		cout << "group_element::compute_fixed_objects_in_PG done" << endl;
	}
}

void group_element::compute_fixed_points_in_induced_action_on_grassmannian(
	int *Elt,
	int dimension,
	std::vector<long int> &fixpoints,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_element::compute_fixed_points_in_induced_action_on_grassmannian" << endl;
	}

	action *A_induced;

	if (f_v) {
		cout << "group_element::compute_fixed_points_in_induced_action_on_grassmannian "
				"before A->Induced_action->induced_action_on_grassmannian" << endl;
	}
	A_induced = A->Induced_action->induced_action_on_grassmannian(
			dimension, 0 /* verbose_level*/);
	if (f_v) {
		cout << "group_element::compute_fixed_points_in_induced_action_on_grassmannian "
				"after A->Induced_action->induced_action_on_grassmannian" << endl;
	}

	long int a, b;

	for (a = 0; a < A_induced->degree; a++) {
		b = A_induced->Group_element->element_image_of(
				a, Elt, 0 /* verbose_level */);
		if (b == a) {
			fixpoints.push_back(a);
		}
	}


	FREE_OBJECT(A_induced);

	if (f_v) {
		cout << "group_element::compute_fixed_points_in_induced_action_on_grassmannian done" << endl;
	}
}
#endif

void group_element::report_fixed_objects_in_PG(
		std::ostream &ost,
		geometry::projective_space *P,
	int *Elt,
	int verbose_level)
// creates temporary actions using induced_action_on_grassmannian
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "group_element::report_fixed_objects_in_PG" << endl;
	}


	combinatorics_with_groups::fixed_objects_in_PG *Fixed_objects_in_PG;

	int up_to_which_rank = P->Subspaces->n;

	Fixed_objects_in_PG = NEW_OBJECT(combinatorics_with_groups::fixed_objects_in_PG);

	if (f_v) {
		cout << "group_element::report_fixed_objects_in_PG "
				"before Fixed_objects_in_PG->init" << endl;
	}
	Fixed_objects_in_PG->init(
			A /* actions::action *A_base */,
			A,
			Elt,
			up_to_which_rank,
			P,
			verbose_level - 2);
	if (f_v) {
		cout << "group_element::report_fixed_objects_in_PG "
				"after Fixed_objects_in_PG->init" << endl;
	}

	Fixed_objects_in_PG->report(ost, verbose_level - 2);

	FREE_OBJECT(Fixed_objects_in_PG);

#if 0
	int j, h, cnt;
	int v[4];
	//field_theory::finite_field *F;

	//ost << "\\section{Fixed Objects}" << endl;

	//F = PG->F;


	int up_to_which_rank = P->Subspaces->n;
	std::vector<std::vector<long int>> Fix;
	long int a;

	if (f_v) {
		cout << "group_element::report_fixed_objects_in_PG "
				"before compute_fixed_objects_in_PG" << endl;
	}
	compute_fixed_objects_in_PG(
			up_to_which_rank,
			P,
			Elt,
			Fix,
			verbose_level);


	ost << "\\bigskip" << endl;

	ost << "The element" << endl;
	ost << "$$" << endl;
	A->Group_element->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following fixed objects:\\\\" << endl;


	ost << "\\bigskip" << endl;
	//ost << "Fixed Points:\\" << endl;


	cnt = Fix[0].size();
	ost << "There are " << cnt << " / " << P->Subspaces->N_points
			<< " fixed points, they are: \\\\" << endl;
	for (j = 0; j < cnt; j++) {
		a = Fix[0][j];

		P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified_lint(
				v, 1, 4, a);

		ost << j << " / " << cnt << " = " << a << " : ";
		Int_vec_print(ost, v, 4);
		ost << "\\\\" << endl;
	}

	ost << "\\bigskip" << endl;

	for (h = 2; h <= up_to_which_rank; h++) {

		if (f_v) {
			cout << "group_element::compute_fixed_objects_in_PG "
					"listing fixed subspaces of rank " << h << endl;
		}
		vector<long int> fix;
		action *Ah;

		if (f_v) {
			cout << "group_element::compute_fixed_objects_in_PG "
					"before A->Induced_action->induced_action_on_grassmannian" << endl;
		}
		Ah = A->Induced_action->induced_action_on_grassmannian(
				h, 0 /* verbose_level*/);
		if (f_v) {
			cout << "group_element::compute_fixed_objects_in_PG "
					"after A->Induced_action->induced_action_on_grassmannian" << endl;
		}

		cnt = Fix[h - 1].size();
		ost << "There are " << cnt << " / " << Ah->degree
				<< " fixed subspaces of "
				"rank " << h << ", they are: \\\\" << endl;

		for (j = 0; j < cnt; j++) {
			a = Fix[h - 1][j];

			ost << j << " / " << cnt << " = " << a << " : $";
			Ah->G.AG->G->print_single_generator_matrix_tex(ost, a);
			ost << "$\\\\" << endl;
		}
		FREE_OBJECT(Ah);
	}
#endif


	if (f_v) {
		cout << "group_element::report_fixed_objects_in_P3 done" << endl;
	}
}

int group_element::test_if_it_fixes_the_polynomial(
	int *Elt,
	int *input,
	ring_theory::homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *input1;
	int *output;
	int ret = true;
	algebra::matrix_group *mtx;

	if (f_v) {
		cout << "group_element::test_if_it_fixes_the_polynomial" << endl;
	}

	if (A->type_G != matrix_group_t) {
		cout << "group_element::test_if_it_fixes_the_polynomial "
				"A->type_G != matrix_group_t" << endl;
		exit(1);
	}

	mtx = A->G.matrix_grp;

	input1 = NEW_int(HPD->get_nb_monomials());
	output = NEW_int(HPD->get_nb_monomials());

	Int_vec_copy(input, input1, HPD->get_nb_monomials());


	mtx->GFq->Projective_space_basic->PG_element_normalize_from_front(
			input1, 1, HPD->get_nb_monomials());


	if (f_v) {
		cout << "group_element::test_if_it_fixes_the_polynomial "
				"before action_on_polynomial" << endl;
	}
	action_on_polynomial(
		Elt,
		input1, output,
		HPD,
		verbose_level - 1);
	if (f_v) {
		cout << "group_element::test_if_it_fixes_the_polynomial "
				"after action_on_polynomial" << endl;
	}

	mtx->GFq->Projective_space_basic->PG_element_normalize_from_front(
			output, 1, HPD->get_nb_monomials());


	data_structures::sorting Sorting;
	int cmp;

	cmp = Sorting.integer_vec_compare(
			input1, output, HPD->get_nb_monomials());
	if (f_v) {
		cout << "group_element::test_if_it_fixes_the_polynomial "
				"cmp = " << cmp << endl;

		Int_vec_print(cout, input1, HPD->get_nb_monomials());
		cout << endl;

		Int_vec_print(cout, output, HPD->get_nb_monomials());
		cout << endl;
	}
	if (cmp) {
		if (f_v) {
			cout << "group_element::test_if_it_fixes_the_polynomial "
					"the element does not fix the equation" << endl;
		}
		ret = false;
	}
	else {
		if (f_v) {
			cout << "group_element::test_if_it_fixes_the_polynomial "
					"the element fixes the equation" << endl;
		}
		ret = true;
	}


	FREE_int(input1);
	FREE_int(output);

	if (f_v) {
		cout << "group_element::test_if_it_fixes_the_polynomial done" << endl;
	}
	return ret;
}

void group_element::action_on_polynomial(
	int *Elt,
	int *input, int *output,
	ring_theory::homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_semilinear;
	algebra::matrix_group *mtx;
	int n;
	int *Elt1;

	if (f_v) {
		cout << "group_element::action_on_polynomial" << endl;
	}

	if (A->type_G != matrix_group_t) {
		cout << "group_element::action_on_polynomial "
				"A->type_G != matrix_group_t" << endl;
		exit(1);
	}

	Elt1 = NEW_int(A->elt_size_in_int);

	mtx = A->G.matrix_grp;
	f_semilinear = mtx->f_semilinear;
	n = mtx->n;

	if (f_vv) {
		cout << "group_element::action_on_polynomial "
				"input = ";
		Int_vec_print(cout, input, HPD->get_nb_monomials());
		cout << endl;
	}

	A->Group_element->element_invert(Elt, Elt1, 0);


	if (f_semilinear) {
		HPD->substitute_semilinear(
				input, output,
				f_semilinear, Elt[n * n], Elt1,
				0 /* verbose_level */);
	}
	else {
		HPD->substitute_linear(
				input, output, Elt1,
				0 /* verbose_level */);
	}

	if (f_vv) {
		cout << "group_element::action_on_polynomial "
				"output = ";
		Int_vec_print(cout, output, HPD->get_nb_monomials());
		cout << endl;
	}

	FREE_int(Elt1);

	if (f_v) {
		cout << "group_element::action_on_polynomial done" << endl;
	}
}

std::string group_element::stringify(
	int *Elt)
{
	return Int_vec_stringify(Elt, A->make_element_size);
}


}}}

