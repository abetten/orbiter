// action_cb.C
//
// Anton Betten
// 1/1/2009

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

namespace orbiter {

int action::image_of(
		void *elt, int a)
{
	nb_times_image_of_called++;
	return (*ptr_element_image_of)(
			*this, a, elt, 0);
}

void action::image_of_low_level(
		void *elt,
		int *input, int *output,
		int verbose_level)
{
	nb_times_image_of_low_level_called++;
	(*ptr_element_image_of_low_level)(
			*this,
			input, output, elt, verbose_level);
}

int action::linear_entry_ij(
		void *elt, int i, int j)
{
	return (*ptr_element_linear_entry_ij)(
			*this, elt, i, j, 0);
}

int action::linear_entry_frobenius(
		void *elt)
{
	return (*ptr_element_linear_entry_frobenius)(
			*this, elt, 0);
}

void action::one(void *elt)
{
	(*ptr_element_one)(*this, elt, 0);
}

int action::is_one(void *elt)
{
	return element_is_one(elt, 0);
	//return (*ptr_element_is_one)(*this, elt, FALSE);
}

void action::unpack(void *elt, void *Elt)
{
	nb_times_unpack_called++;
	(*ptr_element_unpack)(
			*this, elt, Elt, 0);
}

void action::pack(void *Elt, void *elt)
{
	nb_times_pack_called++;
	(*ptr_element_pack)(
			*this, Elt, elt, 0);
}

void action::retrieve(void *elt, int hdl)
{
	nb_times_retrieve_called++;
	(*ptr_element_retrieve)(
			*this, hdl, elt, 0);
}

int action::store(void *elt)
{
	nb_times_store_called++;
	return (*ptr_element_store)(
			*this, elt, 0);
}

void action::mult(
		void *a, void *b, void *ab)
{
	nb_times_mult_called++;
	(*ptr_element_mult)(
			*this, a, b, ab, 0);
}

void action::mult_apply_from_the_right(
		void *a, void *b)
// a := a * b
{
	(*ptr_element_mult)(
			*this, a, b, elt_mult_apply, 0);
	(*ptr_element_move)(
			*this, elt_mult_apply, a, 0);
}

void action::mult_apply_from_the_left(
		void *a, void *b)
// b := a * b
{
	(*ptr_element_mult)(
			*this, a, b, elt_mult_apply, 0);
	(*ptr_element_move)(
			*this, elt_mult_apply, b, 0);
}

void action::invert(void *a, void *av)
{
	nb_times_invert_called++;
	(*ptr_element_invert)(
			*this, a, av, 0);
}

void action::invert_in_place(void *a)
{
	(*ptr_element_invert)(
			*this, a, elt_mult_apply, 0);
	(*ptr_element_move)(
			*this, elt_mult_apply, a, 0);
}

void action::transpose(void *a, void *at)
{
	(*ptr_element_transpose)(
			*this, a, at, 0);
}

void action::move(void *a, void *b)
{
	(*ptr_element_move)(
			*this, a, b, 0);
}

void action::dispose(int hdl)
{
	(*ptr_element_dispose)(
			*this, hdl, 0);
}

void action::print(
		ostream &ost, void *elt)
{
	(*ptr_element_print)(
			*this, elt, ost);
}

void action::print_quick(
		ostream &ost, void *elt)
{
	(*ptr_element_print_quick)(
			*this, elt, ost);
}

void action::print_as_permutation(
		ostream &ost, void *elt)
{
	element_print_as_permutation(elt, ost);
}

void action::print_point(
		int a, ostream &ost)
{
	return (*ptr_print_point)(
			*this, a, ost);
}

void action::code_for_make_element(
		int *data, void *elt)
{
	(*ptr_element_code_for_make_element)(
			*this, elt, data);
}

void action::print_for_make_element(
		ostream &ost, void *elt)
{
	(*ptr_element_print_for_make_element)(
			*this, elt, ost);
}

void action::print_for_make_element_no_commas(
		ostream &ost, void *elt)
{
	(*ptr_element_print_for_make_element_no_commas)(
			*this, elt, ost);
}



// #############################################################################

int action::element_image_of(
		int a, void *elt, int verbose_level)
{
	nb_times_image_of_called++;
	return (*ptr_element_image_of)(
			*this, a, elt, verbose_level);
}

void action::element_image_of_low_level(
		int *input, int *output, void *elt,
		int verbose_level)
{
	if (ptr_element_image_of_low_level == NULL) {
		cout << "action::element_image_of_low_level "
				"ptr is NULL" << endl;
		exit(1);
		}
	nb_times_image_of_low_level_called++;
	(*ptr_element_image_of_low_level)(
			*this,
			input, output, elt, verbose_level);
}

void action::element_one(
		void *elt, int verbose_level)
{
	(*ptr_element_one)(
			*this, elt, verbose_level);
}

int action::element_linear_entry_ij(
		void *elt,
		int i, int j, int verbose_level)
{
	return (*ptr_element_linear_entry_ij)(
			*this,
			elt, i, j, verbose_level);
}

int action::element_linear_entry_frobenius(
		void *elt,
		int verbose_level)
{
	return (*ptr_element_linear_entry_frobenius)(
			*this,
			elt, verbose_level);
}

int action::element_is_one(
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;
	
	if (f_v) {
		cout << "action::element_is_one "
				"in action " << label << endl;
		}
	if (f_has_kernel && Kernel->A->base_len) {
		int *Elt1;
		int drop_out_level, image;
		Elt1 = NEW_int(elt_size_in_int); // this should be avoided
		if (f_v) {
			cout << "action::element_is_one "
					"before Kernel->strip" << endl;
			}
		ret = Kernel->strip((int *)elt, Elt1 /* *residue */, 
			drop_out_level, image, 0 /*verbose_level*/);
		FREE_int(Elt1);
		if (f_v) {
			cout << "action::element_is_one "
					"returning " << ret << endl;
			}
		if (ret)
			return TRUE;
		else
			return FALSE;
		}
	ret = (*ptr_element_is_one)(*this, elt, verbose_level);
	if (f_v) {
		cout << "action::element_is_one "
				"returning " << ret << endl;
		}

	return ret;
}

void action::element_unpack(
		void *elt, void *Elt, int verbose_level)
{
	nb_times_unpack_called++;
	(*ptr_element_unpack)(
			*this, elt, Elt, verbose_level);
}

void action::element_pack(
		void *Elt, void *elt, int verbose_level)
{
	nb_times_pack_called++;
	(*ptr_element_pack)(
			*this, Elt, elt, verbose_level);
}

void action::element_retrieve(
		int hdl, void *elt, int verbose_level)
{
	nb_times_retrieve_called++;
	(*ptr_element_retrieve)(
			*this, hdl, elt, verbose_level);
}

int action::element_store(
		void *elt, int verbose_level)
{
	nb_times_store_called++;
	return (*ptr_element_store)(
			*this, elt, verbose_level);
}

void action::element_mult(
		void *a, void *b, void *ab, int verbose_level)
{
	nb_times_mult_called++;
	(*ptr_element_mult)(
			*this, a, b, ab, verbose_level);
}

void action::element_invert(
		void *a, void *av, int verbose_level)
{
	nb_times_invert_called++;
	(*ptr_element_invert)(
			*this, a, av, verbose_level);
}

void action::element_transpose(
		void *a, void *at, int verbose_level)
{
	(*ptr_element_transpose)(
			*this, a, at, verbose_level);
}

void action::element_move(
		void *a, void *b, int verbose_level)
{
	(*ptr_element_move)(
			*this, a, b, verbose_level);
}

void action::element_dispose(
		int hdl, int verbose_level)
{
	(*ptr_element_dispose)(
			*this, hdl, verbose_level);
}

void action::element_print(
		void *elt, ostream &ost)
{
	(*ptr_element_print)(
			*this, elt, ost);
}

void action::element_print_quick(
		void *elt, ostream &ost)
{
	if (ptr_element_print_quick == NULL) {
		cout << "action::element_print_quick "
				"ptr_element_print_quick == NULL" << endl;
		exit(1);
		}
	(*ptr_element_print_quick)(
			*this, elt, ost);
}

void action::element_print_latex(
		void *elt, ostream &ost)
{
	(*ptr_element_print_latex)(
			*this, elt, ost);
}

void action::element_print_verbose(
		void *elt, ostream &ost)
{
	(*ptr_element_print_verbose)(
			*this, elt, ost);
}

void action::element_code_for_make_element(
		void *elt, int *data)
{
	(*ptr_element_code_for_make_element)(
			*this, elt, data);
}

void action::element_print_for_make_element(
		void *elt, ostream &ost)
{
	(*ptr_element_print_for_make_element)(
			*this, elt, ost);
}

void action::element_print_for_make_element_no_commas(
		void *elt, ostream &ost)
{
	(*ptr_element_print_for_make_element_no_commas)(
			*this, elt, ost);
}

void action::element_print_as_permutation(
		void *elt, ostream &ost)
{
	element_print_as_permutation_with_offset(
			elt, ost, 0, FALSE, TRUE, 0);
}

void action::element_print_as_permutation_verbose(
		void *elt,
		ostream &ost, int verbose_level)
{
	element_print_as_permutation_with_offset(elt,
			ost, 0, FALSE, TRUE, verbose_level);
}

void action::element_as_permutation(
		void *elt,
		int *perm, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	
	if (f_v) {
		cout << "action::element_as_permutation" << endl;
		}	
	for (i = 0; i < degree; i++) {
		if (f_vv) {
			cout << "action::element_as_permutation" << i << endl;
			}
		j = element_image_of(i, elt, verbose_level - 2);
		perm[i] = j;
		if (f_vv) {
			cout << "action::element_as_permutation "
					<< i << "->" << j << endl;
			}
		}
	if (f_v) {
		cout << "action::element_as_permutation done" << endl;
		}	
}

void action::element_print_as_permutation_with_offset(
	void *elt, ostream &ost,
	int offset, int f_do_it_anyway_even_for_big_degree, 
	int f_print_cycles_of_length_one, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *v, i, j;
	int f_cycle_length = FALSE;
	int f_max_cycle_length = FALSE;
	int max_cycle_length = 50;
	int f_orbit_structure = FALSE;
	
	if (f_v) {
		cout << "action::element_print_as_permutation_with_offset "
				"degree=" << degree << endl;
		}
	v = NEW_int(degree);
	for (i = 0; i < degree; i++) {
		if (f_vv) {
			cout << "action::element_print_as_permutation_with_offset "
					"computing image of " << i << endl;
			}
		j = element_image_of(i,
				elt, verbose_level - 2);
		if (f_vv) {
			cout << "action::element_print_as_permutation_with_offset "
					<< i << "->" << j << endl;
			}
		v[i] = j;
		}
	//perm_print(ost, v, degree);
	perm_print_offset(ost, v, degree, offset,
			f_cycle_length,
			f_max_cycle_length, max_cycle_length,
			f_orbit_structure);
	//ost << endl;
	//perm_print_cycles_sorted_by_length(ost, degree, v);


#if 0
	if (degree) {
		if (f_v) {
			cout << "action::element_print_as_permutation_with_offset: "
					"calling perm_print_cycles_sorted_by_length_offset" << endl;
			}
		//ost << "perm of degree " << degree << " : ";
		//int_vec_print_fully(ost, v, degree);
		//ost << " = ";

		perm_print_cycles_sorted_by_length_offset(ost, degree, v, offset,
			f_do_it_anyway_even_for_big_degree, f_print_cycles_of_length_one,
			verbose_level);
			// in action_global.C
		}
#endif


	//ost << endl;
	FREE_int(v);
}

void action::element_print_as_permutation_with_offset_and_max_cycle_length(
	void *elt,
	ostream &ost, int offset,
	int max_cycle_length,
	int f_orbit_structure)
{
	int *v, i, j;
	int f_cycle_length = FALSE;
	int f_max_cycle_length = TRUE;
	
	v = NEW_int(degree);
	for (i = 0; i < degree; i++) {
		j = element_image_of(i, elt, FALSE);
		v[i] = j;
		}
	//perm_print(ost, v, degree);
	perm_print_offset(ost, v, degree, offset, f_cycle_length,
			f_max_cycle_length, max_cycle_length, f_orbit_structure);
	FREE_int(v);
}

void action::element_print_image_of_set(
		void *elt, int size, int *set)
{
	int i, j;
	
	for (i = 0; i < size; i++) {
		j = element_image_of(set[i], elt, FALSE);
		cout << i << " -> " << j << endl;
		}
}

int action::element_signum_of_permutation(void *elt)
{
	int *v;
	int i, j, sgn;
	
	v = NEW_int(degree);
	for (i = 0; i < degree; i++) {
		j = element_image_of(i, elt, FALSE);
		v[i] = j;
		}
	sgn = perm_signum(v, degree);
	FREE_int(v);
	return sgn;
}



void action::element_write_file_fp(int *Elt,
		FILE *fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;
	
	elt = element_rw_memory_object;
	if (f_v) {
		element_print(Elt, cout);
		int_vec_print(cout, Elt, elt_size_in_int);
		cout << endl;
		}
	element_pack(Elt, elt, FALSE);
	fwrite(elt, 1 /* size */,
			coded_elt_size_in_char /* items */, fp);
}

void action::element_read_file_fp(int *Elt,
		FILE *fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;
	
	elt = element_rw_memory_object;
	fread(elt, 1 /* size */,
			coded_elt_size_in_char /* items */, fp);
	element_unpack(elt, Elt, FALSE);
	if (f_v) {
		element_print(Elt, cout);
		int_vec_print(cout, Elt, elt_size_in_int);
		cout << endl;
		}
}

void action::element_write_file(int *Elt,
		const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	FILE *f2;
	f2 = fopen(fname, "wb");
	element_write_file_fp(Elt, f2, 0/* verbose_level*/);
	fclose(f2);

	if (f_v) {
		cout << "written file " << fname << " of size "
				<< file_size(fname) << endl;
		}
}

void action::element_read_file(int *Elt,
		const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "element_read_file: "
				"reading from file " << fname
				<< " of size " << file_size(fname) << endl;
		}
	FILE *f2;
	f2 = fopen(fname, "rb");
	element_read_file_fp(Elt, f2, 0/* verbose_level*/);
	
	fclose(f2);
}

void action::element_write_to_memory_object(int *Elt,
		memory_object *m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;

	if (f_v) {
		cout << "action::element_write_to_memory_object" << endl;
		}
	elt = element_rw_memory_object;

	element_pack(Elt, elt, FALSE);
	m->append(coded_elt_size_in_char, elt, 0);
}


void action::element_read_from_memory_object(int *Elt,
		memory_object *m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;
	int i;

	
	if (f_v) {
		cout << "action::element_read_from_memory_object" << endl;
		}
	elt = element_rw_memory_object;

	for (i = 0; i < coded_elt_size_in_char; i++) {
		m->read_char(elt + i);
		}
	element_unpack(elt, Elt, FALSE);
}

void action::element_write_to_file_binary(int *Elt,
		ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;

	if (f_v) {
		cout << "action::element_write_to_file_binary" << endl;
		}
	if (coded_elt_size_in_char == 0) {
		cout << "action::element_write_to_file_binary "
				"coded_elt_size_in_char == 0" << endl;
		exit(1);
	}
	//elt = NEW_char(coded_elt_size_in_char);
		// memory allocation should be avoided in a low-level function
	elt = element_rw_memory_object;

	element_pack(Elt, elt, FALSE);
	fp.write(elt, coded_elt_size_in_char);
	//FREE_char(elt);
}

void action::element_read_from_file_binary(int *Elt,
		ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;

	
	if (f_v) {
		cout << "action::element_read_from_memory_object" << endl;
		}
	//elt = NEW_char(coded_elt_size_in_char);
		// memory allocation should be avoided in a low-level function
	elt = element_rw_memory_object;

	fp.read(elt, coded_elt_size_in_char);
	element_unpack(elt, Elt, FALSE);
	//FREE_char(elt);
}

void action::random_element(sims *S, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::random_element" << endl;
		}

	S->random_element(Elt, verbose_level - 1);

	if (f_v) {
		cout << "action::random_element done" << endl;
		}
}

}
