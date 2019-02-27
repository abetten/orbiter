// int_vector.C
//
// Anton Betten
//
// Aug 12, 2014




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


int_vector::int_vector()
{
	null();
}

int_vector::~int_vector()
{
	freeself();
}

void int_vector::null()
{
	M = NULL;
	m = 0;
	alloc_length = 0;
}

void int_vector::freeself()
{
	if (M) {
		FREE_int(M);
		}
	null();
}

void int_vector::allocate(int len)
{
	freeself();
	M = NEW_int(len);
	m = 0;
	alloc_length = len;
}

void int_vector::allocate_and_init(int len, int *V)
{
	freeself();
	M = NEW_int(len);
	m = len;
	alloc_length = len;
	int_vec_copy(V, M, len);
}

void int_vector::init_permutation_from_string(const char *s)
{
	int verbose_level = 0;
	int *perm;
	int degree;
	
	scan_permutation_from_string(s, perm, degree, verbose_level);
	allocate_and_init(degree, perm);
	FREE_int(perm);
}

void int_vector::read_ascii_file(const char *fname)
{
	int verbose_level = 0;
	int *the_set;
	int set_size;
	read_set_from_file(fname, the_set, set_size, verbose_level);
	allocate_and_init(set_size, the_set);
	FREE_int(the_set);
}

void int_vector::read_binary_file_int4(const char *fname)
{
	int verbose_level = 0;
	int *the_set;
	int set_size;
	read_set_from_file_int4(fname, the_set, set_size, verbose_level);
	allocate_and_init(set_size, the_set);
	FREE_int(the_set);
}

int &int_vector::s_i(int i)
{
	return M[i];
}

int &int_vector::length()
{
	return m;
}

void int_vector::print(ostream &ost)
{
	int_vec_print(ost, M, m);
}

void int_vector::zero()
{
	int_vec_zero(M, m);
}

int int_vector::search(int a, int &idx)
{
	return int_vec_search(M, m, a, idx);
}

void int_vector::sort()
{
	int_vec_sort(m, M);
}

void int_vector::make_space()
{
	int *M1;
	int new_alloc_length;

	if (alloc_length) {
		new_alloc_length = alloc_length * 2;
		}
	else {
		new_alloc_length = 1;
		}
	M1 = NEW_int(new_alloc_length);
	int_vec_copy(M, M1, m);
	if (M) {
		FREE_int(M);
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
	int_vec_sort_and_remove_duplicates(M, m);
}

void int_vector::write_to_ascii_file(const char *fname)
{
	write_set_to_file(fname, M, m, 0 /*verbose_level*/);
}

void int_vector::write_to_binary_file_int4(const char *fname)
{
	write_set_to_file_as_int4(fname, M, m, 0 /*verbose_level*/);
}

void int_vector::write_to_csv_file(const char *fname, const char *label)
{
	int_vec_write_csv(M, m, fname, label);
}

int int_vector::hash()
{
	return int_vec_hash(M, m);
}

int int_vector::minimum()
{
	return int_vec_minimum(M, m);
}

int int_vector::maximum()
{
	return int_vec_maximum(M, m);
}


}
}

