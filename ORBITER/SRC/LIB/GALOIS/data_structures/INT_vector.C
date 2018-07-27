// INT_vector.C
//
// Anton Betten
//
// Aug 12, 2014




#include "galois.h"

INT_vector::INT_vector()
{
	null();
}

INT_vector::~INT_vector()
{
	freeself();
}

void INT_vector::null()
{
	M = NULL;
	m = 0;
	alloc_length = 0;
}

void INT_vector::freeself()
{
	if (M) {
		FREE_INT(M);
		}
	null();
}

void INT_vector::allocate(INT len)
{
	freeself();
	M = NEW_INT(len);
	m = 0;
	alloc_length = len;
}

void INT_vector::allocate_and_init(INT len, INT *V)
{
	freeself();
	M = NEW_INT(len);
	m = len;
	alloc_length = len;
	INT_vec_copy(V, M, len);
}

void INT_vector::init_permutation_from_string(const char *s)
{
	INT verbose_level = 0;
	INT *perm;
	INT degree;
	
	scan_permutation_from_string(s, perm, degree, verbose_level);
	allocate_and_init(degree, perm);
	FREE_INT(perm);
}

void INT_vector::read_ascii_file(const BYTE *fname)
{
	INT verbose_level = 0;
	INT *the_set;
	INT set_size;
	read_set_from_file(fname, the_set, set_size, verbose_level);
	allocate_and_init(set_size, the_set);
	FREE_INT(the_set);
}

void INT_vector::read_binary_file_INT4(const BYTE *fname)
{
	INT verbose_level = 0;
	INT *the_set;
	INT set_size;
	read_set_from_file_INT4(fname, the_set, set_size, verbose_level);
	allocate_and_init(set_size, the_set);
	FREE_INT(the_set);
}

INT &INT_vector::s_i(INT i)
{
	return M[i];
}

INT &INT_vector::length()
{
	return m;
}

void INT_vector::print(ostream &ost)
{
	INT_vec_print(ost, M, m);
}

void INT_vector::zero()
{
	INT_vec_zero(M, m);
}

INT INT_vector::search(INT a, INT &idx)
{
	return INT_vec_search(M, m, a, idx);
}

void INT_vector::sort()
{
	INT_vec_sort(m, M);
}

void INT_vector::make_space()
{
	INT *M1;
	INT new_alloc_length;

	if (alloc_length) {
		new_alloc_length = alloc_length * 2;
		}
	else {
		new_alloc_length = 1;
		}
	M1 = NEW_INT(new_alloc_length);
	INT_vec_copy(M, M1, m);
	if (M) {
		FREE_INT(M);
		}
	M = M1;
	alloc_length = new_alloc_length;
}

void INT_vector::append(INT a)
{
	if (m == alloc_length) {
		make_space();
		}
	M[m] = a;
	m++;
}

void INT_vector::insert_at(INT a, INT idx)
{
	INT i;
	
	if (m == alloc_length) {
		make_space();
		}
	for (i = m; i > idx; i--) {
		M[i] = M[i - 1];
		}
	M[idx] = a;
	m++;
}

void INT_vector::insert_if_not_yet_there(INT a)
{
	INT idx;

	if (!search(a, idx)) {
		insert_at(a, idx);
		}
}

void INT_vector::sort_and_remove_duplicates()
{
	INT_vec_sort_and_remove_duplicates(M, m);
}

void INT_vector::write_to_ascii_file(const BYTE *fname)
{
	write_set_to_file(fname, M, m, 0 /*verbose_level*/);
}

void INT_vector::write_to_binary_file_INT4(const BYTE *fname)
{
	write_set_to_file_as_INT4(fname, M, m, 0 /*verbose_level*/);
}

void INT_vector::write_to_csv_file(const BYTE *fname, const BYTE *label)
{
	INT_vec_write_csv(M, m, fname, label);
}

INT INT_vector::hash()
{
	return INT_vec_hash(M, m);
}

INT INT_vector::minimum()
{
	return INT_vec_minimum(M, m);
}

INT INT_vector::maximum()
{
	return INT_vec_maximum(M, m);
}



