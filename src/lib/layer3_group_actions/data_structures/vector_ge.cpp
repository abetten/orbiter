// vector_ge.cpp
//
// Anton Betten
// December 9, 2003

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


#undef PRINT_WITH_TYPE
#define RANGE_CHECKING

namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {


vector_ge::vector_ge()
{
	A = NULL;
	data = NULL;
	len = 0;
	//null();
}

vector_ge::~vector_ge()
{
	if (data) {
		FREE_int(data);
		data = NULL;
	}
}

void vector_ge::null()
{
	A = NULL;
	data = NULL;
	len = 0;
}



void vector_ge::init(actions::action *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::init" << endl;
	}
	vector_ge::A = A;
	data = NULL;
	len = 0;
	if (f_v) {
		cout << "vector_ge::init done" << endl;
	}
}

void vector_ge::copy(vector_ge *&vector_copy, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::copy" << endl;
	}
	if (A == NULL) {
		cout << "vector_ge::copy A == NULL" << endl;
		exit(1);
	}

	vector_copy = NEW_OBJECT(vector_ge);
	vector_copy->null();
	vector_copy->init(A, verbose_level);
	if (f_v) {
		cout << "vector_ge::copy before vector_copy->allocate" << endl;
	}
	vector_copy->allocate(len, verbose_level);
	if (f_v) {
		cout << "vector_ge::copy before loop" << endl;
	}
	for (i = 0; i < len; i++) {
		A->Group_element->element_move(ith(i), vector_copy->ith(i), 0);
	}
	if (f_v) {
		cout << "vector_ge::copy done" << endl;
	}
}

void vector_ge::init_by_hdl(actions::action *A,
		int *gen_hdl, int nb_gen, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::init_by_hdl" << endl;
	}
	init(A, verbose_level);
	allocate(nb_gen, verbose_level);
	for (i = 0; i < nb_gen; i++) {
		A->Group_element->element_retrieve(gen_hdl[i], ith(i), 0);
	}
	if (f_v) {
		cout << "vector_ge::init_by_hdl done" << endl;
	}
}

void vector_ge::init_single(actions::action *A,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::init_single" << endl;
	}
	init(A, verbose_level);
	allocate(1, verbose_level);
	A->Group_element->element_move(Elt, ith(0), 0);
	if (f_v) {
		cout << "vector_ge::init_single done" << endl;
	}
}

void vector_ge::init_double(actions::action *A,
		int *Elt1, int *Elt2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::init_double" << endl;
	}
	init(A, verbose_level);
	allocate(2, verbose_level);
	A->Group_element->element_move(Elt1, ith(0), 0);
	A->Group_element->element_move(Elt2, ith(1), 0);
	if (f_v) {
		cout << "vector_ge::init_double done" << endl;
	}
}

void vector_ge::init_from_permutation_representation(
		actions::action *A, groups::sims *S, int *data,
	int nb_elements, int verbose_level)
// data[nb_elements * A->degree]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int *Elt;

	if (f_v) {
		cout << "vector_ge::init_from_permutation_representation" << endl;
	}
	Elt = NEW_int(A->elt_size_in_int);
	init(A, verbose_level);
	allocate(nb_elements, verbose_level);
	for (i = 0; i < nb_elements; i++) {
		A->Group_element->make_element_from_permutation_representation(
				Elt, S, data + i * A->degree, 0/*verbose_level*/);
		if (f_vv) {
			cout << "vector_ge::init_from_permutation_representation "
					"generator " << i << ": " << endl;
			A->Group_element->element_print_quick(Elt, cout);
			A->Group_element->element_print_latex(Elt, cout);
		}
		A->Group_element->element_move(Elt, ith(i), 0);
	}
	
	FREE_int(Elt);
	if (f_v) {
		cout << "vector_ge::init_from_permutation_representation done" << endl;
	}
}

void vector_ge::init_from_data(actions::action *A, int *data,
	int nb_elements, int elt_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int *Elt;

	if (f_v) {
		cout << "vector_ge::init_from_data" << endl;
	}
	Elt = NEW_int(A->elt_size_in_int);
	init(A, verbose_level);
	allocate(nb_elements, verbose_level);
	for (i = 0; i < nb_elements; i++) {
		A->Group_element->make_element(Elt, data + i * elt_size, verbose_level);
		if (f_vv) {
			cout << "vector_ge::init_from_data "
					"generator " << i << ": " << endl;
			A->Group_element->element_print_quick(Elt, cout);
		}
		A->Group_element->element_move(Elt, ith(i), 0);
	}
	
	FREE_int(Elt);
	if (f_v) {
		cout << "vector_ge::init_from_data done" << endl;
	}
}

void vector_ge::init_conjugate_svas_of(vector_ge *v,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *Elt1, *Elt2, *Elt3;

	if (f_v) {
		cout << "vector_ge::init_conjugate_svas_of" << endl;
	}

	init(v->A, verbose_level);
	allocate(v->len, verbose_level);

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	A->Group_element->invert(Elt, Elt1);
	for (i = 0; i < len; i++) {
		A->Group_element->element_mult(Elt1, v->ith(i), Elt2, false);
		A->Group_element->element_mult(Elt2, Elt, Elt3, false);
		A->Group_element->element_move(Elt3, ith(i), false);
	}
	
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_v) {
		cout << "vector_ge::init_conjugate_svas_of done" << endl;
	}
}

void vector_ge::init_conjugate_sasv_of(vector_ge *v,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *Elt1, *Elt2, *Elt3;

	if (f_v) {
		cout << "vector_ge::init_conjugate_svas_of" << endl;
	}

	init(v->A, verbose_level);
	allocate(v->len, verbose_level);

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	A->Group_element->invert(Elt, Elt1);
	for (i = 0; i < len; i++) {
		A->Group_element->element_mult(Elt, v->ith(i), Elt2, false);
		A->Group_element->element_mult(Elt2, Elt1, Elt3, false);
		A->Group_element->element_move(Elt3, ith(i), false);
	}
	
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_v) {
		cout << "vector_ge::init_conjugate_sasv_of done" << endl;
	}
}

int *vector_ge::ith(int i)
{
#ifdef RANGE_CHECKING
	if (i < 0 || i >= len) {
		cout << "vector_ge::ith() access error "
				"i = " << i << " len = " << len << endl;
		exit(1);
		}
#endif
	return data + i * A->elt_size_in_int; 
}

void vector_ge::print(std::ostream &ost)
{
	int i;

	for (i = 0; i < len; i++) {
		ost << "Element " << i << " / " << len << " is:" << endl;
		A->Group_element->element_print_quick(ith(i), ost);
		ost << endl;
	}
}

void vector_ge::print_quick(std::ostream& ost)
{
	int i;
	
	ost << "(" << endl;
	//ost << "len=" << len << " A->elt_size_in_int=" << A->elt_size_in_int << " data=" << data << endl;
	for (i = 0; i < len; i++) {
		if (data == NULL) {
			cout << "vector_ge::print fatal: data == NULL" << endl;
			exit(1);
		}
		A->Group_element->element_print_quick(ith(i), ost);
		if (i < len - 1) {
			ost << ", " << endl;
		}
	}
	ost << ")" << endl;
}

void vector_ge::print_tex(std::ostream &ost)
{
	int i;

	ost << "$$" << endl;
	for (i = 0; i < len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		A->Group_element->element_print_latex(ith(i), ost);
		if (i < len - 1) {
			ost << ", " << endl;
		}
		if (((i + 1) % 3) == 0 && i < len - 1) {
			ost << "$$" << endl;
			ost << "$$" << endl;
			}
		}
	ost << "$$" << endl;
}



void vector_ge::print_generators_tex(
		ring_theory::longinteger_object &go, std::ostream &ost)
{
	int i;

	ost << "Generators for a group of order " << go << ":" << endl;
	ost << "$$" << endl;
	for (i = 0; i < len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		A->Group_element->element_print_latex(ith(i), ost);
		if (((i + 1) % 4) == 0 && i < len - 1) {
			ost << "$$" << endl;
			ost << "$$" << endl;
		}
	}
	ost << "$$" << endl;
	for (i = 0; i < len; i++) {
		A->Group_element->element_print_for_make_element(ith(i), ost);
		ost << "\\\\" << endl;
	}
}

void vector_ge::print_as_permutation(std::ostream& ost)
{
	int i;
	
	ost << "(" << endl;
	for (i = 0; i < len; i++) {
		A->Group_element->element_print(ith(i), ost);
		if (A->degree < 1000) {
			cout << "is the permutation" << endl;
			A->Group_element->element_print_as_permutation(ith(i), ost);
		}
		else {
			cout << "vector_ge::print_as_permutation "
					"the degree is too large, we won't print "
					"the permutation" << endl;
		}
		if (i < len - 1) {
			ost << ", " << endl;
		}
	}
	ost << ")" << endl;
}

void vector_ge::allocate(int length, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::allocate" << endl;
		cout << "vector_ge::allocate length = " << length << endl;
	}
	if (f_v) {
		cout << "vector_ge::allocate A = " << endl;
		A->print_info();
	}
	if (data) {
		FREE_int(data);
		//cout << "vector_ge::allocate warning, data != NULL, "
		//"we seem to be having a memory leak here" << endl;
	}
	len = length;
	data = NEW_int(length * A->elt_size_in_int);
	if (f_v) {
		cout << "vector_ge::allocate done" << endl;
	}
}

void vector_ge::reallocate(int new_length, int verbose_level)
{
	int *data2;
	int *elt, *elt2, i, l;
	
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::reallocate" << endl;
	}
	data2 = NEW_int(new_length * A->elt_size_in_int);

	l = MINIMUM(len, new_length);
	for (i = 0; i < l; i++) {
		elt = ith(i);
		elt2 = data2 + i * A->elt_size_in_int;
		A->Group_element->element_move(elt, elt2, false);
	}
	if (data) {
		FREE_int(data);
		data = NULL;
	}
	data = data2;
	len = new_length;
	if (f_v) {
		cout << "vector_ge::reallocate done" << endl;
	}
}

void vector_ge::reallocate_and_insert_at(
		int position, int *elt, int verbose_level)
{
	int *data2;
	int *elt1, *elt2, i;
	
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::reallocate_and_insert_at len=" << len << endl;
	}
	data2 = NEW_int((len + 1) * A->elt_size_in_int);
	for (i = 0; i < len; i++) {
		elt1 = ith(i);
		if (i >= position) {
			elt2 = data2 + (i + 1) * A->elt_size_in_int;
		}
		else {
			elt2 = data2 + i * A->elt_size_in_int;
		}
		A->Group_element->element_move(elt1, elt2, false);
	}
	if (data) {
		FREE_int(data);
		data = NULL;
	}
	else {
		if (f_v) {
			cout << "vector_ge::reallocate_and_insert_at data == NULL" << endl;
		}
	}
	data = data2;
	len = len + 1;
	if (position < 0 || position >= len) {
		cout << "vector_ge::reallocate_and_insert_at position out of bounds, position=" << position << endl;
		exit(1);
	}
	copy_in(position, elt);
	if (f_v) {
		cout << "vector_ge::reallocate_and_insert_at done" << endl;
	}
}

void vector_ge::insert_at(int length_before,
		int position, int *elt, int verbose_level)
// does not reallocate, but shifts elements up to make space.
// the last element might be lost if there is no space.
{
	int *elt1, *elt2, i;
	
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::insert_at" << endl;
	}
	for (i = length_before; i >= position; i--) {
		if (i + 1 >= len) {
			continue;
		}
		
		elt1 = ith(i);
		elt2 = ith(i + 1);
		A->Group_element->element_move(elt1, elt2, false);
	}
	copy_in(position, elt);
	if (f_v) {
		cout << "vector_ge::insert_at done" << endl;
	}
}

void vector_ge::append(int *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::append" << endl;
	}
	reallocate_and_insert_at(len, elt, verbose_level);
	if (f_v) {
		cout << "vector_ge::append done" << endl;
	}
}

void vector_ge::copy_in(int i, int *elt)
{
	int *elt2 = ith(i);
	A->Group_element->element_move(elt, elt2, false);
};

void vector_ge::copy_out(int i, int *elt)
{
	int *elt2 = ith(i);
	A->Group_element->element_move(elt2, elt, false);
}

void vector_ge::conjugate_svas(int *Elt)
{
	int i;
	int *Elt1, *Elt2, *Elt3;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	A->Group_element->invert(Elt, Elt1);
	for (i = 0; i < len; i++) {
		A->Group_element->element_mult(Elt1, ith(i), Elt2, false);
		A->Group_element->element_mult(Elt2, Elt, Elt3, false);
		A->Group_element->element_move(Elt3, ith(i), false);
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
}

void vector_ge::conjugate_sasv(int *Elt)
{
	int i;
	int *Elt1, *Elt2, *Elt3;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	A->Group_element->invert(Elt, Elt1);
	for (i = 0; i < len; i++) {
		A->Group_element->element_mult(Elt, ith(i), Elt2, false);
		A->Group_element->element_mult(Elt2, Elt1, Elt3, false);
		A->Group_element->element_move(Elt3, ith(i), false);
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
}

void vector_ge::print_with_given_action(
		std::ostream &ost, actions::action *A2)
{
	int i, l;

	l = len;
	for (i = 0; i < l; i++) {
		ost << "generator " << i << ":" << endl;
		A->Group_element->element_print_quick(ith(i), ost);
		ost << endl;
		A2->Group_element->element_print_as_permutation(ith(i), ost);
		ost << endl;
	}
}

void vector_ge::print(std::ostream &ost,
		int f_print_as_permutation,
	int f_offset, int offset,
	int f_do_it_anyway_even_for_big_degree,
	int f_print_cycles_of_length_one,
	int verbose_level)
{
	int i, l;
	
	l = len;
	if (!f_offset) {
		offset = 0;
	}
	ost << "vector of " << l << " group elements:" << endl;
	ost << "f_print_as_permutation=" << f_print_as_permutation << endl;
	for (i = 0; i < l; i++) {
		ost << "generator " << i << " / " << l << ":" << endl;
		A->Group_element->element_print_quick(ith(i), ost);
		ost << endl;
		if (f_print_as_permutation) {
			//A->element_print_as_permutation(ith(i), ost);
			A->Group_element->element_print_as_permutation_with_offset(ith(i), ost,
				offset, f_do_it_anyway_even_for_big_degree, 
				f_print_cycles_of_length_one, verbose_level - 1);
			ost << endl;
		}
	}
}

void vector_ge::print_for_make_element(std::ostream &ost)
{
	int i, l;

	l = len;
	ost << "Strong generators: (" << l << " of them)" << endl;
	for (i = 0; i < l; i++) {
		A->Group_element->element_print_for_make_element(ith(i), ost);
		ost << "\\\\" << endl;
	}
}


void vector_ge::write_to_memory_object(
		orbiter_kernel_system::memory_object *m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "vector_ge::write_to_memory_object" << endl;
	}
	m->write_int(len);
	for (i = 0; i < len; i++) {
		A->Group_element->element_write_to_memory_object(ith(i), m, 0);
	}
}

void vector_ge::read_from_memory_object(
		orbiter_kernel_system::memory_object *m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;

	if (f_v) {
		cout << "vector_ge::read_from_memory_object" << endl;
	}
	m->read_int(&l);
	allocate(l, verbose_level);
	for (i = 0; i < len; i++) {
		A->Group_element->element_read_from_memory_object(ith(i), m, 0);
	}
}

void vector_ge::write_to_file_binary(ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "vector_ge::write_to_file_binary" << endl;
	}
	fp.write((char *) &len, sizeof(int));
	for (i = 0; i < len; i++) {
		if (f_v) {
			cout << "vector_ge::write_to_file_binary writing element " << i << " / " << len << endl;
		}
		A->Group_element->element_write_to_file_binary(ith(i), fp, verbose_level);
	}
}

void vector_ge::read_from_file_binary(ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;

	if (f_v) {
		cout << "vector_ge::read_from_file_binary" << endl;
	}
	fp.read((char *) &l, sizeof(int));
	allocate(l, verbose_level);
	for (i = 0; i < len; i++) {
		if (f_v) {
			cout << "vector_ge::read_from_file_binary reading element " << i << " / " << len << endl;
		}
		A->Group_element->element_read_from_file_binary(ith(i), fp, verbose_level);
	}
}

void vector_ge::write_to_csv_file_coded(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "vector_ge::write_to_csv_file_coded" << endl;
	}
	int *Table;

	Table = NEW_int(len * A->make_element_size);
	for (i = 0; i < len; i++) {
		for (j = 0; j < A->make_element_size; j++) {
			Table[i * A->make_element_size + j] = ith(i)[j];
		}
	}
	orbiter_kernel_system::file_io Fio;

	Fio.int_matrix_write_csv(fname, Table, len, A->make_element_size);
	if (f_v) {
		cout << "vector_ge::write_to_csv_file_coded written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "vector_ge::write_to_csv_file_coded done" << endl;
	}
}

void vector_ge::save_csv(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::save_csv" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	int i;
	int *Elt;
	int *data;

	data = NEW_int(A->make_element_size);
	{
		ofstream ost(fname);

		ost << "Row,Element" << endl;
		for (i = 0; i < len; i++) {
			Elt = ith(i);

			A->Group_element->element_code_for_make_element(Elt, data);

			stringstream ss;
			Int_vec_print_str_naked(ss, data, A->make_element_size);
			ost << i << ",\"" << ss.str() << "\"" << endl;
		}
		ost << "END" << endl;
	}
	FREE_int(data);

	if (f_v) {
		cout << "sims::save_csv Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

}

void vector_ge::export_inversion_graphs(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::export_inversion_graphs" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	int h;
	int *Elt;
	int *perm;
	int N2;
	combinatorics::combinatorics_domain Combi;

	N2 = Combi.int_n_choose_k(A->degree, 2);

	perm = NEW_int(A->degree);

	{
		ofstream ost(fname);

		ost << "Row";
		for (h = 0; h < N2; h++) {
			ost << ",C" << h;
		}
		ost << endl;
		for (h = 0; h < len; h++) {
			Elt = ith(h);


			int *Adj;
			int N;
			int i, j;

			A->Group_element->element_as_permutation(Elt, perm, 0 /* verbose_level*/);

			graph_theory::graph_theory_domain GT;


			if (f_v) {
				cout << "vector_ge::export_inversion_graphs before GT.make_inversion_graph" << endl;
			}
			GT.make_inversion_graph(Adj, N, perm, A->degree, verbose_level);
			if (f_v) {
				cout << "vector_ge::export_inversion_graphs after GT.make_inversion_graph" << endl;
			}

			ost << h;
			for (i = 0; i < N; i++) {
				for (j = i + 1; j < N; j++) {
					ost << "," << Adj[i * N + j];
				}
			}
			ost << endl;


			FREE_int(Adj);


		}
		ost << "END" << endl;
	}
	FREE_int(perm);

	if (f_v) {
		cout << "sims::export_inversion_graphs Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

}


void vector_ge::read_column_csv(std::string &fname,
		actions::action *A, int col_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::read_column_csv" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	data_structures::spreadsheet S;
	int n, i, me_sz;


	init(A, verbose_level);

	me_sz = A->make_element_size;

	if (f_v) {
		cout << "vector_ge::read_column_csv reading file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}

	S.read_spreadsheet(fname, verbose_level);

	n = S.nb_rows - 1;

	allocate(n, verbose_level);
	for (i = 0; i < n; i++) {
		string s;
		int *data;
		int sz;

		S.get_string(s, i + 1, col_idx);
		Int_vec_scan(s, data, sz);
		if (sz != me_sz) {
			cout << "vector_ge::read_column_csv sz != me_sz" << endl;
			cout << "vector_ge::read_column_csv sz = " << sz << endl;
			cout << "vector_ge::read_column_csv me_sz = " << me_sz << endl;
			exit(1);
		}
		A->Group_element->make_element(ith(i), data, 0 /*verbose_level*/);
		FREE_int(data);
	}

	if (f_v) {
		cout << "vector_ge::read_column_csv done" << endl;
	}
}

void vector_ge::read_column_csv_using_column_label(
		std::string &fname,
		actions::action *A,
		std::string &column_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::read_column_csv column_label = " << column_label << endl;
	}


	orbiter_kernel_system::file_io Fio;
	data_structures::spreadsheet S;
	int n, i, me_sz;


	init(A, verbose_level);

	me_sz = A->make_element_size;

	if (f_v) {
		cout << "vector_ge::read_column_csv reading file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}

	S.read_spreadsheet(fname, verbose_level);


	int col_idx;

	col_idx = S.find_column(column_label);


	n = S.nb_rows - 1;

	allocate(n, verbose_level);
	for (i = 0; i < n; i++) {
		string s;
		int *data;
		int sz;

		S.get_string(s, i + 1, col_idx);

		if (s.length() >= 2 && s[0] == '"' && s[s.length() - 1] == '"') {
			string s2;

			s2 = s.substr(1, s.length() - 2);
			s = s2;
		}
		Int_vec_scan(s, data, sz);
		if (sz != me_sz) {
			cout << "vector_ge::read_column_csv sz != me_sz" << endl;
			cout << "vector_ge::read_column_csv sz = " << sz << endl;
			cout << "vector_ge::read_column_csv me_sz = " << me_sz << endl;
			exit(1);
		}
		A->Group_element->make_element(ith(i), data, 0 /*verbose_level*/);
		FREE_int(data);
	}

	if (f_v) {
		cout << "vector_ge::read_column_csv done" << endl;
	}
}



void vector_ge::extract_subset_of_elements_by_rank_text_vector(
		std::string &rank_vector_text, groups::sims *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::extract_subset_of_elements_"
				"by_rank_text_vector" << endl;
	}
	int *v;
	int len;

	Int_vec_scan(rank_vector_text, v, len);
	if (f_v) {
		cout << "vector_ge::extract_subset_of_elements_"
				"by_rank_text_vector after scanning: ";
		Int_vec_print(cout, v, len);
		cout << endl;
	}
	extract_subset_of_elements_by_rank(v, len, S, verbose_level);
	FREE_int(v);
	if (f_v) {
		cout << "vector_ge::extract_subset_of_elements_"
				"by_rank_text_vector done" << endl;
	}
}

void vector_ge::extract_subset_of_elements_by_rank(
		int *rank_vector, int len, groups::sims *S,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, r;
	int *Elt;

	if (f_v) {
		cout << "vector_ge::extract_subset_of_elements_by_rank" << endl;
	}
	Elt = NEW_int(A->elt_size_in_int);
	allocate(len, verbose_level);
	for (i = 0; i < len; i++) {
		r = rank_vector[i];
		S->element_unrank_lint(r, Elt);

		if (f_v) {
			cout << "vector_ge::extract_subset_of_elements_by_rank "
					"element " << i << " = " << r << " / " << len << endl;
			A->Group_element->element_print_quick(Elt, cout);
		}
		A->Group_element->element_move(Elt, ith(i), 0);
	}
	FREE_int(Elt);
	if (f_v) {
		cout << "vector_ge::extract_subset_of_elements_by_rank done" << endl;
	}
}

int vector_ge::test_if_all_elements_stabilize_a_point(
		actions::action *A2, int pt)
{
	int i;
	
	for (i = 0; i < len; i++) {
		if (A2->Group_element->element_image_of(pt, ith(i), 0) != pt) {
			return false;
		}
	}
	return true;
}

int vector_ge::test_if_all_elements_stabilize_a_set(
		actions::action *A2,
		long int *set, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "vector_ge::test_if_all_elements_stabilize_a_set" << endl;
	}
	
	for (i = 0; i < len; i++) {
		if (f_v) {
			cout << "testing element " << i << " / " << len << endl;
		}
		if (!A2->Group_element->test_if_set_stabilizes(ith(i),
				sz, set, 0 /* verbose_level*/)) {
			return false;
		}
	}
	if (f_v) {
		cout << "vector_ge::test_if_all_elements_stabilize_a_set done" << endl;
	}
	return true;
}


groups::schreier *vector_ge::orbits_on_points_schreier(
		actions::action *A_given, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::schreier *Sch;

	if (f_v) {
		cout << "vector_ge::orbits_on_points_schreier "
				"degree = " << A_given->degree << endl;
	}
	if (f_v) {
		cout << "vector_ge::orbits_on_points_schreier "
				"action ";
		A_given->print_info();
		cout << endl;
	}

	Sch = NEW_OBJECT(groups::schreier);

	Sch->init(A_given, verbose_level - 2);
	Sch->initialize_tables();
	Sch->init_generators(*this, verbose_level - 2);
	if (f_v) {
		cout << "vector_ge::orbits_on_points_schreier "
				"before Sch->compute_all_point_orbits" << endl;
	}
	Sch->compute_all_point_orbits(verbose_level);
	if (f_v) {
		cout << "vector_ge::orbits_on_points_schreier "
				"after Sch->compute_all_point_orbits" << endl;
	}

	if (f_v) {
		cout << "vector_ge::orbits_on_points_schreier "
				"we found " << Sch->nb_orbits << " orbits" << endl;
	}

	if (f_v) {
		cout << "vector_ge::orbits_on_points_schreier done" << endl;
	}
	return Sch;
}

void vector_ge::reverse_isomorphism_exterior_square(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	field_theory::finite_field *F;

	if (f_v) {
		cout << "vector_ge::reverse_isomorphism_exterior_square" << endl;
	}

	geometry::klein_correspondence *K;
	orthogonal_geometry::orthogonal *O;
	int A4[17];


	F = A->matrix_group_finite_field();

	O = NEW_OBJECT(orthogonal_geometry::orthogonal);
	O->init(1 /* epsilon */, 6 /* n */, F, verbose_level);

	K = NEW_OBJECT(geometry::klein_correspondence);
	K->init(F, O, verbose_level);


	for (i = 0; i < len; i++) {

		K->reverse_isomorphism(ith(i), A4, verbose_level);
		cout << "generator " << i << " / " << len << ":" << endl;

		cout << "before:" << endl;
		Int_matrix_print(ith(i), 6, 6);

		cout << "after:" << endl;
		Int_matrix_print(A4, 4, 4);
	}

	FREE_OBJECT(K);
	FREE_OBJECT(O);
	if (f_v) {
		cout << "vector_ge::reverse_isomorphism_exterior_square done" << endl;
	}
}

void vector_ge::matrix_representation(
		induced_actions::action_on_homogeneous_polynomials *A_on_HPD,
		int *&M, int &nb_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int n;

	if (f_v) {
		cout << "vector_ge::matrix_representation" << endl;
	}

	nb_gens = len;

	n = A_on_HPD->dimension;

	M = NEW_int(len * n * n);

	for (i = 0; i < len; i++) {

		A_on_HPD->compute_representation(ith(i),
				M + i * n * n, verbose_level);

	}

	if (f_v) {
		cout << "vector_ge::matrix_representation done" << endl;
	}
}

void vector_ge::stab_BLT_set_from_catalogue(
		actions::action *A,
	field_theory::finite_field *F, int iso,
	std::string &target_go_text,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::stab_BLT_set_from_catalogue" << endl;
		cout << "q=" << F->q << endl;
		cout << "iso=" << iso << endl;
		cout << "A=" << endl;
		A->print_info();
	}

	int *data;
	int nb_gens;
	int data_size;
	//string ascii_target_go;
	//ring_theory::longinteger_object target_go;
	int i;
	knowledge_base::knowledge_base K;

	if (f_v) {
		cout << "vector_ge::stab_BLT_set_from_catalogue "
				"before K.BLT_stab_gens" << endl;
	}
	K.BLT_stab_gens(F->q, iso, data, nb_gens, data_size, target_go_text);

	if (f_v) {
		cout << "vector_ge::stab_BLT_set_from_catalogue "
				"data_size=" << data_size << endl;
		cout << "vector_ge::stab_BLT_set_from_catalogue "
				"nb_gens=" << nb_gens << endl;
		cout << "vector_ge::stab_BLT_set_from_catalogue "
				"target_go_text=" << target_go_text << endl;
	}

	init(A, verbose_level - 2);
	//target_go.create_from_base_10_string(target_go_text);


	allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		A->Group_element->make_element(ith(i), data + i * data_size, 0);
	}

	if (f_v) {
		cout << "vector_ge::stab_BLT_set_from_catalogue "
				"generators are:" << endl;
		print_quick(cout);
	}

	if (f_v) {
		cout << "vector_ge::stab_BLT_set_from_catalogue done" << endl;
	}
}

int vector_ge::test_if_in_set_stabilizer(
		actions::action *A,
		long int *set, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;
	int i, c;

	if (f_v) {
		cout << "vector_ge::test_if_in_set_stabilizer" << endl;
	}

	ret = true;
	for (i = 0; i < len; i++) {
		c = A->Group_element->check_if_in_set_stabilizer(
				ith(i),
				sz, set, 0 /* verbose_level*/);
		if (!c) {
			if (f_v) {
				cout << "vector_ge::test_if_in_set_stabilizer element "
						<< i << " fails to stabilize the given set" << endl;
			}
			ret = false;
			break;
		}
	}
	if (f_v) {
		cout << "vector_ge::test_if_in_set_stabilizer done" << endl;
	}
	return ret;
}
}}}



