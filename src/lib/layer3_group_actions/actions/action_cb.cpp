// action_cb.cpp
//
// Anton Betten
// 1/1/2009

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;

namespace orbiter {
namespace layer3_group_actions {
namespace actions {


int action::image_of(void *elt, int a)
{
	ptr->nb_times_image_of_called++;
	return (*ptr->ptr_element_image_of)(*this, a, elt, 0);
}

void action::image_of_low_level(
		void *elt,
		int *input, int *output,
		int verbose_level)
{
	ptr->nb_times_image_of_low_level_called++;
	(*ptr->ptr_element_image_of_low_level)(
			*this,
			input, output, elt, verbose_level);
}

int action::linear_entry_ij(
		void *elt, int i, int j)
{
	return (*ptr->ptr_element_linear_entry_ij)(*this, elt, i, j, 0);
}

int action::linear_entry_frobenius(
		void *elt)
{
	return (*ptr->ptr_element_linear_entry_frobenius)(*this, elt, 0);
}

void action::one(void *elt)
{
	(*ptr->ptr_element_one)(*this, elt, 0);
}

int action::is_one(void *elt)
{
	return element_is_one(elt, 0);
	//return (*ptr_element_is_one)(*this, elt, FALSE);
}

void action::unpack(void *elt, void *Elt)
{
	ptr->nb_times_unpack_called++;
	(*ptr->ptr_element_unpack)(*this, elt, Elt, 0);
}

void action::pack(void *Elt, void *elt)
{
	ptr->nb_times_pack_called++;
	(*ptr->ptr_element_pack)(*this, Elt, elt, 0);
}

void action::retrieve(void *elt, int hdl)
{
	ptr->nb_times_retrieve_called++;
	(*ptr->ptr_element_retrieve)(*this, hdl, elt, 0);
}

int action::store(void *elt)
{
	ptr->nb_times_store_called++;
	return (*ptr->ptr_element_store)(*this, elt, 0);
}

void action::mult(
		void *a, void *b, void *ab)
{
	ptr->nb_times_mult_called++;
	(*ptr->ptr_element_mult)(*this, a, b, ab, 0);
}

void action::mult_apply_from_the_right(
		void *a, void *b)
// a := a * b
{
	(*ptr->ptr_element_mult)(*this, a, b, elt_mult_apply, 0);
	(*ptr->ptr_element_move)(*this, elt_mult_apply, a, 0);
}

void action::mult_apply_from_the_left(
		void *a, void *b)
// b := a * b
{
	(*ptr->ptr_element_mult)(*this, a, b, elt_mult_apply, 0);
	(*ptr->ptr_element_move)(*this, elt_mult_apply, b, 0);
}

void action::invert(void *a, void *av)
{
	ptr->nb_times_invert_called++;
	(*ptr->ptr_element_invert)(*this, a, av, 0);
}

void action::invert_in_place(void *a)
{
	(*ptr->ptr_element_invert)(*this, a, elt_mult_apply, 0);
	(*ptr->ptr_element_move)(*this, elt_mult_apply, a, 0);
}

void action::transpose(void *a, void *at)
{
	(*ptr->ptr_element_transpose)(*this, a, at, 0);
}

void action::move(void *a, void *b)
{
	(*ptr->ptr_element_move)(*this, a, b, 0);
}

void action::dispose(int hdl)
{
	(*ptr->ptr_element_dispose)(*this, hdl, 0);
}

void action::print(
		ostream &ost, void *elt)
{
	(*ptr->ptr_element_print)(*this, elt, ost);
}

void action::print_quick(
		ostream &ost, void *elt)
{
	(*ptr->ptr_element_print_quick)(*this, elt, ost);
}

void action::print_as_permutation(
		ostream &ost, void *elt)
{
	element_print_as_permutation(elt, ost);
}

void action::print_point(
		int a, std::ostream &ost)
{
	//cout << "action::print_point" << endl;
	(*ptr->ptr_print_point)(*this, a, ost);
}

void action::unrank_point(long int rk, int *v)
// v[low_level_point_size]
{
	if (ptr->ptr_unrank_point == NULL) {
		cout << "action::unrank_point ptr_unrank_point == NULL, label=" << ptr->label << endl;
		exit(1);
	}
	(*ptr->ptr_unrank_point)(*this, rk, v);
}

long int action::rank_point(int *v)
// v[low_level_point_size]
{
	if (ptr->ptr_rank_point == NULL) {
		cout << "action::rank_point ptr_rank_point == NULL, label=" << ptr->label << endl;
		exit(1);
	}
	return (*ptr->ptr_rank_point)(*this, v);
}

void action::code_for_make_element(
		int *data, void *elt)
{
	(*ptr->ptr_element_code_for_make_element)(*this, elt, data);
}

void action::print_for_make_element(
		ostream &ost, void *elt)
{
	(*ptr->ptr_element_print_for_make_element)(*this, elt, ost);
}

void action::print_for_make_element_no_commas(
		ostream &ost, void *elt)
{
	(*ptr->ptr_element_print_for_make_element_no_commas)(*this, elt, ost);
}



// #############################################################################

long int action::element_image_of(
		long int a, void *elt, int verbose_level)
{
	if (ptr == NULL) {
		cout << "action::element_image_of ptr == NULL" << endl;
		exit(1);
	}
	ptr->nb_times_image_of_called++;
	return (*ptr->ptr_element_image_of)(*this, a, elt, verbose_level);
}

void action::element_image_of_low_level(
		int *input, int *output, void *elt,
		int verbose_level)
{
	if (ptr->ptr_element_image_of_low_level == NULL) {
		cout << "action::element_image_of_low_level "
				"ptr is NULL" << endl;
		exit(1);
		}
	ptr->nb_times_image_of_low_level_called++;
	(*ptr->ptr_element_image_of_low_level)(
			*this,
			input, output, elt, verbose_level);
}

void action::element_one(
		void *elt, int verbose_level)
{
	(*ptr->ptr_element_one)(*this, elt, verbose_level);
}

int action::element_linear_entry_ij(
		void *elt,
		int i, int j, int verbose_level)
{
	return (*ptr->ptr_element_linear_entry_ij)(
			*this,
			elt, i, j, verbose_level);
}

int action::element_linear_entry_frobenius(
		void *elt,
		int verbose_level)
{
	return (*ptr->ptr_element_linear_entry_frobenius)(
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
	if (f_has_kernel && Kernel->A->base_len()) {
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
	ret = (*ptr->ptr_element_is_one)(*this, elt, verbose_level);
	if (f_v) {
		cout << "action::element_is_one "
				"returning " << ret << endl;
		}

	return ret;
}

void action::element_unpack(
		void *elt, void *Elt, int verbose_level)
{
	ptr->nb_times_unpack_called++;
	(*ptr->ptr_element_unpack)(*this, elt, Elt, verbose_level);
}

void action::element_pack(
		void *Elt, void *elt, int verbose_level)
{
	ptr->nb_times_pack_called++;
	(*ptr->ptr_element_pack)(*this, Elt, elt, verbose_level);
}

void action::element_retrieve(
		int hdl, void *elt, int verbose_level)
{
	ptr->nb_times_retrieve_called++;
	(*ptr->ptr_element_retrieve)(*this, hdl, elt, verbose_level);
}

int action::element_store(
		void *elt, int verbose_level)
{
	ptr->nb_times_store_called++;
	return (*ptr->ptr_element_store)(*this, elt, verbose_level);
}

void action::element_mult(
		void *a, void *b, void *ab, int verbose_level)
{
	ptr->nb_times_mult_called++;
	(*ptr->ptr_element_mult)(*this, a, b, ab, verbose_level);
}

void action::element_invert(
		void *a, void *av, int verbose_level)
{
	ptr->nb_times_invert_called++;
	(*ptr->ptr_element_invert)(*this, a, av, verbose_level);
}

void action::element_transpose(
		void *a, void *at, int verbose_level)
{
	(*ptr->ptr_element_transpose)(*this, a, at, verbose_level);
}

void action::element_move(
		void *a, void *b, int verbose_level)
{
	(*ptr->ptr_element_move)(*this, a, b, verbose_level);
}

void action::element_dispose(
		int hdl, int verbose_level)
{
	(*ptr->ptr_element_dispose)(*this, hdl, verbose_level);
}

void action::element_print(
		void *elt, std::ostream &ost)
{
	(*ptr->ptr_element_print)(*this, elt, ost);
}

void action::element_print_quick(
		void *elt, std::ostream &ost)
{
	if (ptr->ptr_element_print_quick == NULL) {
		cout << "action::element_print_quick "
				"ptr_element_print_quick == NULL" << endl;
		exit(1);
		}
	(*ptr->ptr_element_print_quick)(*this, elt, ost);
}

void action::element_print_latex(
		void *elt, std::ostream &ost)
{
	(*ptr->ptr_element_print_latex)(*this, elt, ost);
}

void action::element_print_latex_with_extras(void *elt, std::string &label, std::ostream &ost)
{
	int *fp, n, ord;

	fp = NEW_int(degree);
	n = find_fixed_points(elt, fp, 0);
	//cout << "with " << n << " fixed points" << endl;
	FREE_int(fp);

	ord = element_order(elt);

	ost << "$$" << label << endl;
	element_print_latex(elt, ost);
	ost << "$$" << endl << "of order $" << ord << "$ and with "
			<< n << " fixed points." << endl;
}


void action::element_print_latex_with_print_point_function(
	void *elt, std::ostream &ost,
	void (*point_label)(std::stringstream &sstr, long int pt, void *data),
	void *point_label_data)
{
	(*ptr->ptr_element_print_latex_with_print_point_function)(
				*this, elt, ost, point_label, point_label_data);
}

void action::element_print_verbose(
		void *elt, std::ostream &ost)
{
	(*ptr->ptr_element_print_verbose)(*this, elt, ost);
}

void action::element_code_for_make_element(
		void *elt, int *data)
{
	(*ptr->ptr_element_code_for_make_element)(*this, elt, data);
}

void action::element_print_for_make_element(
		void *elt, std::ostream &ost)
{
	(*ptr->ptr_element_print_for_make_element)(*this, elt, ost);
}

void action::element_print_for_make_element_no_commas(
		void *elt, std::ostream &ost)
{
	(*ptr->ptr_element_print_for_make_element_no_commas)(*this, elt, ost);
}

void action::element_print_as_permutation(
		void *elt, std::ostream &ost)
{
	element_print_as_permutation_with_offset(
			elt, ost, 0, FALSE, TRUE, 0);
}

void action::element_print_as_permutation_verbose(
		void *elt,
		std::ostream &ost, int verbose_level)
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
	void *elt, std::ostream &ost,
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
	combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "action::element_print_as_permutation_with_offset "
				"degree=" << degree << endl;
		}
	if (degree > 5000) {
		cout << "action::element_print_as_permutation_with_offset "
				"the degree is too large, we won't print the permutation" << endl;
		return;
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
	Combi.perm_print_offset(ost, v, degree, offset,
			f_print_cycles_of_length_one,
			f_cycle_length,
			f_max_cycle_length, max_cycle_length,
			f_orbit_structure,
			NULL, NULL);
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
		}
#endif


	//ost << endl;
	FREE_int(v);
}

void action::element_print_as_permutation_with_offset_and_max_cycle_length(
	void *elt,
	std::ostream &ost, int offset,
	int max_cycle_length,
	int f_orbit_structure)
{
	int *v, i, j;
	int f_print_cycles_of_length_one = FALSE;
	int f_cycle_length = FALSE;
	int f_max_cycle_length = TRUE;
	combinatorics::combinatorics_domain Combi;
	
	v = NEW_int(degree);
	for (i = 0; i < degree; i++) {
		j = element_image_of(i, elt, FALSE);
		v[i] = j;
		}
	//perm_print(ost, v, degree);
	Combi.perm_print_offset(ost, v, degree, offset,
			f_print_cycles_of_length_one,
			f_cycle_length,
			f_max_cycle_length, max_cycle_length, f_orbit_structure,
			NULL, NULL);
	FREE_int(v);
}

void action::element_print_image_of_set(
		void *elt, int size, long int *set)
{
	long int i, j;
	
	for (i = 0; i < size; i++) {
		j = element_image_of(set[i], elt, FALSE);
		cout << i << " -> " << j << endl;
		}
}

int action::element_signum_of_permutation(void *elt)
{
	int *v;
	int i, j, sgn;
	combinatorics::combinatorics_domain Combi;

	v = NEW_int(degree);
	for (i = 0; i < degree; i++) {
		j = element_image_of(i, elt, FALSE);
		v[i] = j;
		}
	sgn = Combi.perm_signum(v, degree);
	FREE_int(v);
	return sgn;
}



void action::element_write_file_fp(int *Elt,
		ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;
	
	elt = element_rw_memory_object;
	if (elt == NULL) {
		cout << "action::element_write_file_fp elt == NULL" << endl;
		exit(1);
	}
	if (f_v) {
		element_print(Elt, cout);
		Int_vec_print(cout, Elt, elt_size_in_int);
		cout << endl;
		}
	element_pack(Elt, elt, FALSE);
	fp.write(elt, coded_elt_size_in_char);
	//fwrite(elt, 1 /* size */, coded_elt_size_in_char /* items */, fp);
}

void action::element_read_file_fp(int *Elt,
		ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;
	
	elt = element_rw_memory_object;
	fp.read(elt, coded_elt_size_in_char);
	//fread(elt, 1 /* size */, coded_elt_size_in_char /* items */, fp);
	element_unpack(elt, Elt, FALSE);
	if (f_v) {
		element_print(Elt, cout);
		Int_vec_print(cout, Elt, elt_size_in_int);
		cout << endl;
		}
}

void action::element_write_file(int *Elt,
		std::string &fname, int verbose_level)
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

void action::element_read_file(int *Elt,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	
	if (f_v) {
		cout << "element_read_file: "
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

void action::element_write_to_memory_object(int *Elt,
		orbiter_kernel_system::memory_object *m, int verbose_level)
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
		orbiter_kernel_system::memory_object *m, int verbose_level)
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
	if (f_v) {
		cout << "action::element_write_to_file_binary coded_elt_size_in_char=" << coded_elt_size_in_char << endl;
		}
	if (coded_elt_size_in_char == 0) {
		cout << "action::element_write_to_file_binary "
				"coded_elt_size_in_char == 0" << endl;
		exit(1);
	}
	//elt = NEW_char(coded_elt_size_in_char);
		// memory allocation should be avoided in a low-level function
	elt = element_rw_memory_object;
	if (elt == NULL) {
		cout << "action::element_write_to_file_binary elt == NULL" << endl;
		print_info();
		exit(1);
	}

	element_pack(Elt, elt, verbose_level);
	fp.write(elt, coded_elt_size_in_char);
	//FREE_char(elt);
	if (f_v) {
		cout << "action::element_write_to_file_binary done" << endl;
		}
}

void action::element_read_from_file_binary(int *Elt,
		ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *elt;

	
	if (f_v) {
		cout << "action::element_read_from_file_binary" << endl;
		}
	//elt = NEW_char(coded_elt_size_in_char);
		// memory allocation should be avoided in a low-level function
	elt = element_rw_memory_object;

	if (f_v) {
		cout << "action::element_read_from_file_binary coded_elt_size_in_char=" << coded_elt_size_in_char << endl;
		}
	fp.read(elt, coded_elt_size_in_char);
	element_unpack(elt, Elt, verbose_level);
	//FREE_char(elt);
	if (f_v) {
		cout << "action::element_read_from_file_binary done" << endl;
		}
}

void action::random_element(groups::sims *S, int *Elt, int verbose_level)
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

void action::all_elements(data_structures_groups::vector_ge *&vec, int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::all_elements" << endl;
	}

	if (!f_has_sims) {
		cout << "action::all_elements !f_has_sims" << endl;
		exit(1);
	}

	ring_theory::longinteger_object go;
	long int i, goi;

	group_order(go);
	goi = go.as_int();

	vec = NEW_OBJECT(data_structures_groups::vector_ge);
	vec->init(this, 0 /*verbose_level*/);
	vec->allocate(goi, verbose_level);


	for (i = 0; i < goi; i++) {
		Sims->element_unrank_lint(i, vec->ith(i));
	}

	if (f_v) {
		cout << "action::all_elements done" << endl;
	}
}


void action::all_elements_save_csv(std::string &fname, int verbose_level)
{

	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "action::all_elements_save_csv" << endl;
	}

	if (!f_has_sims) {
		cout << "action::all_elements_save_csv !f_has_sims" << endl;
		exit(1);
	}
	data_structures_groups::vector_ge *vec;
	int i;
	int *data;
	int *Elt;

	all_elements(vec, verbose_level);
	data = NEW_int(make_element_size);


	{
		ofstream ost(fname);

		ost << "Row,Element" << endl;
		for (i = 0; i < vec->len; i++) {
			Elt = vec->ith(i);

			element_code_for_make_element(Elt, data);

			stringstream ss;
			Int_vec_print_str_naked(ss, data, make_element_size);
			ost << i << ",\"" << ss.str() << "\"" << endl;
		}
		ost << "END" << endl;
	}
	if (f_v) {
		cout << "action::all_elements_save_csv Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_OBJECT(vec);
	FREE_int(data);

	if (f_v) {
		cout << "action::all_elements_save_csv done" << endl;
	}
}



}}}


