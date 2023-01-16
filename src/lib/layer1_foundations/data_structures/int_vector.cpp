// int_vector.cpp
//
// Anton Betten
//
// Aug 12, 2014




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


int_vector::int_vector()
{
	M = NULL;
	m = 0;
	alloc_length = 0;
}

int_vector::~int_vector()
{
	if (M) {
		FREE_lint(M);
	}
}

void int_vector::allocate(int len)
{
	if (M) {
		FREE_lint(M);
	}
	M = NEW_lint(len);
	m = 0;
	alloc_length = len;
}

void int_vector::allocate_and_init(int len, long int *V)
{
	if (M) {
		FREE_lint(M);
	}
	M = NEW_lint(len);
	m = len;
	alloc_length = len;
	Lint_vec_copy(V, M, len);
}

void int_vector::allocate_and_init_int(int len, int *V)
{
	int i;

	if (M) {
		FREE_lint(M);
	}
	M = NEW_lint(len);
	m = len;
	alloc_length = len;
	for (i = 0; i < len; i++) {
		M[i] = V[i];
	}
}

void int_vector::init_permutation_from_string(std::string &s)
{
	int verbose_level = 0;
	int *perm;
	int degree;
	string_tools ST;
	
	ST.scan_permutation_from_string(s, perm, degree, verbose_level);
	allocate_and_init_int(degree, perm);
	FREE_int(perm);
}

void int_vector::read_ascii_file(std::string &fname)
{
	int verbose_level = 0;
	long int *the_set;
	int set_size;
	orbiter_kernel_system::file_io Fio;

	Fio.read_set_from_file(fname, the_set, set_size, verbose_level);
	allocate_and_init(set_size, the_set);
	FREE_lint(the_set);
}

void int_vector::read_binary_file_int4(std::string &fname)
{
	int verbose_level = 0;
	long int *the_set;
	int set_size;
	orbiter_kernel_system::file_io Fio;

	Fio.read_set_from_file_int4(fname, the_set, set_size, verbose_level);
	allocate_and_init(set_size, the_set);
	FREE_lint(the_set);
}

long int &int_vector::s_i(int i)
{
	return M[i];
}

int &int_vector::length()
{
	return m;
}

void int_vector::print(std::ostream &ost)
{
	Lint_vec_print(ost, M, m);
}

void int_vector::zero()
{
	orbiter_kernel_system::Orbiter->Lint_vec->zero(M, m);
}

int int_vector::search(int a, int &idx)
{
	sorting Sorting;

	return Sorting.lint_vec_search(M, m, a, idx, 0);
}

void int_vector::sort()
{
	sorting Sorting;

	Sorting.lint_vec_heapsort(M, m);
}

void int_vector::make_space()
{
	long int *M1;
	int new_alloc_length;

	if (alloc_length) {
		new_alloc_length = alloc_length * 2;
		}
	else {
		new_alloc_length = 1;
		}
	M1 = NEW_lint(new_alloc_length);
	Lint_vec_copy(M, M1, m);
	if (M) {
		FREE_lint(M);
		}
	M = M1;
	alloc_length = new_alloc_length;
}

void int_vector::append(int a)
{
	if (m == alloc_length) {
		make_space();
		}
	M[m] = a;
	m++;
}

void int_vector::insert_at(int a, int idx)
{
	int i;
	
	if (m == alloc_length) {
		make_space();
		}
	for (i = m; i > idx; i--) {
		M[i] = M[i - 1];
		}
	M[idx] = a;
	m++;
}

void int_vector::insert_if_not_yet_there(int a)
{
	int idx;

	if (!search(a, idx)) {
		insert_at(a, idx);
		}
}

void int_vector::sort_and_remove_duplicates()
{
	sorting Sorting;

	Sorting.lint_vec_sort_and_remove_duplicates(M, m);
}

void int_vector::write_to_ascii_file(std::string &fname)
{
	orbiter_kernel_system::file_io Fio;

	Fio.write_set_to_file(fname, M, m, 0 /*verbose_level*/);
}

void int_vector::write_to_binary_file_int4(std::string &fname)
{
	orbiter_kernel_system::file_io Fio;

	Fio.write_set_to_file_as_int4(fname, M, m, 0 /*verbose_level*/);
}

void int_vector::write_to_csv_file(std::string &fname, std::string &label)
{
	orbiter_kernel_system::file_io Fio;

	Fio.lint_vec_write_csv(M, m, fname, label);
}

uint32_t int_vector::hash()
{
	data_structures_global D;

	return D.lint_vec_hash(M, m);
}

int int_vector::minimum()
{
	return orbiter_kernel_system::Orbiter->Lint_vec->minimum(M, m);
}

int int_vector::maximum()
{
	return orbiter_kernel_system::Orbiter->Lint_vec->maximum(M, m);
}




}}}

