/*
 * algorithms.cpp
 *
 *  Created on: Jan 14, 2022
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



//#include "pstdint.h" /* Replace with <stdint.h> if appropriate */
#include <stdint.h>


namespace orbiter {
namespace layer1_foundations {
namespace data_structures {




algorithms::algorithms()
{

}

algorithms::~algorithms()
{

}


//#define HASH_PRIME ((int) 1 << 30 - 1)
#define HASH_PRIME 174962718

int algorithms::hashing(
		int hash0, int a)
{
	int h = hash0; // a1 = a;

	do {
		h <<= 1;
		if (ODD(a)){
			h++;
		}
		h = h % HASH_PRIME;	// h %= HASH_PRIME;
		a >>= 1;
	} while (a);
	//cout << "hashing: " << hash0 << " + " << a1 << " = " << h << endl;
	return h;
}

int algorithms::hashing_fixed_width(
		int hash0, int a, int bit_length)
{
	int h = hash0;
	int a1 = a;
	int i;

	for (i = 0; i < bit_length; i++) {
		h <<= 1;
		if (ODD(a)){
			h++;
		}
		h = h % HASH_PRIME;	// h %= HASH_PRIME;
		a >>= 1;
	}
	if (a) {
		cout << "algorithms::hashing_fixed_width a is not zero" << endl;
		cout << "a=" << a1 << endl;
		cout << "bit_length=" << bit_length << endl;
		exit(1);
	}
	//cout << "hashing: " << hash0 << " + " << a1 << " = " << h << endl;
	return h;
}

void algorithms::uchar_print_bitwise(
		std::ostream &ost, unsigned char u)
{
	uchar mask;
	int i;

	for (i = 0; i < 8; i++) {
		mask = ((uchar) 1) << i;
		if (u & mask) {
			ost << "1";
		}
		else {
			ost << "0";
		}
	}
}

void algorithms::uchar_move(
		const unsigned char *p, unsigned char *q, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		*q++ = *p++;
	}
}

void algorithms::uchar_expand_4(
		const unsigned char *p, unsigned char *q, int len)
{
	int i;
	uchar a, b;

	for (i = 0; i < len; i++) {

		a = p[i >> 1];
		if (i % 2) {
			b = a & 15;
		}
		else {
			b = (a & 240) >> 4;
		}
		q[i] = b;
	}
}

void algorithms::uchar_compress_4(
		const unsigned char *p, unsigned char *q, int len)
{
	int i, i_half;
	int f_v = false;
	uchar a, b;

	for (i = 0; i < len; i++) {

		i_half = i >> 1;
		a = p[i];
		if (a >= 16) {
			cout << "algorithms::uchar_compress_4 a >= 16" << endl;
			exit(1);
		}
		if (i % 2) {
			b = a & 15;
			if (f_v) {
				cout << "algorithms::uchar_compress_4 "
						"i=" << i << " i_half=" << i_half
						<< " a=" << (int) a
						<< " b=" << (int) b
						<< " q[i_half]=" << (int) q[i_half] << endl;
			}
			q[i_half] ^= b;
			if (f_v) {
				cout << "algorithms::uchar_compress_4 "
						"q[i_half]=" << (int) q[i_half] << endl;
			}
		}
		else {
			b = a << 4;
			q[i_half] = b;
		}
	}
}

void algorithms::uchar_zero(
		unsigned char *p, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		*p++ = 0;
	}
}

void algorithms::uchar_xor(
		unsigned char *in1, unsigned char *in2, unsigned char *out, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		out[i] = in1[i] ^ in2[i];
	}
}

int algorithms::uchar_compare(
		unsigned char *in1, unsigned char *in2, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		if (in1[i] < in2[i]) {
			return -1;
		}
		else if (in1[i] > in2[i]) {
			return 1;
		}
	}
	return 0;
}

int algorithms::uchar_is_zero(
		unsigned char *in, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		if (in[i]) {
			return false;
		}
	}
	return true;
}


void algorithms::int_swap(
		int& x, int& y)
{
	int z;

	z = x;
	x = y;
	y = z;
}

void algorithms::lint_swap(
		long int & x, long int & y)
{
	long int z;

	z = x;
	x = y;
	y = z;
}

void algorithms::print_pointer_hex(
		std::ostream &ost, void *p)
{
	void *q = p;
	uchar *pp = (uchar *)&q;
	int i, a, low, high;

	ost << "0x";
	for (i = (int)sizeof(pvoid) - 1; i >= 0; i--) {
		a = (int)pp[i];
		//cout << " a=" << a << " ";
		low = a % 16;
		high = a / 16;
		print_hex_digit(ost, high);
		print_hex_digit(ost, low);
	}
}

void algorithms::print_uint32_hex(
		std::ostream &ost, uint32_t val)
{
	uchar *pp = (uchar *)&val;
	int i, a, low, high;

	ost << "0x";
	for (i = (int)sizeof(uint32_t) - 1; i >= 0; i--) {
		a = (int)pp[i];
		//cout << " a=" << a << " ";
		low = a % 16;
		high = a / 16;
		print_hex_digit(ost, high);
		print_hex_digit(ost, low);
	}
}

void algorithms::print_hex(
		std::ostream &ost, unsigned char *p, int len)
{
	int i, j, h, a, low, high;
	int nb_rows;

	nb_rows = (len + 15) / 16; // 16 per row

	for (i = 0; i < nb_rows; i++) {
		print_uint32_hex(ost, i * 16);
		for (h = 0; h < 2; h++) {
			ost << " ";
			for (j = 0; j < 8; j++) {
				a = (int) p[i * 16 + h * 8 + j];
				low = a % 16;
				high = a / 16;
				ost << " ";
				print_hex_digit(ost, high);
				print_hex_digit(ost, low);
			}
		}
		cout << endl;
	}
}

void algorithms::print_binary(
		std::ostream &ost, unsigned char *p, int len)
{
	int i, j, h, a, low;
	int nb_rows;
	int bits[8];

	nb_rows = (len + 3) / 4; // 4 per row

	for (i = 0; i < nb_rows; i++) {
		ost << setw(10) << (i * 4) << " ";
		//print_uint32_hex(ost, i * 4);
		for (h = 0; h < 4; h++) {
			ost << " ";
			a = (int) p[i * 4 + h];
			for (j = 0; j < 8; j++) {
				low = a % 2;
				a >>= 1;
				bits[j] = low;
			}
			for (j = 7; j >= 0; j--) {
				ost << bits[j];
			}
		}
		cout << endl;
	}
}



void algorithms::print_uint32_binary(
		std::ostream &ost, uint32_t val)
{
	//uchar *pp = (uchar *)&val;
	int i, a;
	int bits[32];

	for (i = 0; i < 32; i++) {
		if (val % 2) {
			bits[i] = 1;
		}
		else {
			bits[i] = 0;
		}
		val >>= 1;
	}
	ost << "0b";
	for (i = 31; i >= 0; i--) {
		a = (int)bits[i];
		cout << a;
		if (i > 0 && (i % 8 == 0)) {
			cout << ";";
		}
	}
}


void algorithms::print_hex_digit(
		std::ostream &ost, int digit)
{
	if (digit < 10) {
		ost << (char)('0' + digit);
	}
	else if (digit < 16) {
		ost << (char)('a' + (digit - 10));
	}
	else {
		cout << "algorithms::print_hex_digit illegal digit " << digit << endl;
		exit(1);
	}
}

void algorithms::print_bits(
		std::ostream &ost, char *data, int data_size)
{
	int i, j;
	unsigned char c;

	for (i = 0; i < data_size; i++) {
		c = (unsigned char) data[i];
		for (j = 0; j < 8; j++) {
			if (c % 2) {
				cout << "1";
			}
			else {
				cout << "0";
			}
			c >>= 1;
		}
	}
}

unsigned long int algorithms::make_bitword(
		char *data, int data_size)
{
	int i, c;
	unsigned long int m;

	if (data_size > sizeof(long int) * 8) {
		cout << "algorithms::make_bitword "
				"data_size cannot be larger than " << sizeof(long int) * 8 << endl;
		cout << "data_size = " << data_size << endl;
		exit(1);
	}

	m = 0;
	for (i = 0; i < data_size; i++) {
		m <<= 1;
		c = data[i];
		if (c) {
			m |= 1;
		}
	}
	return m;
}



void algorithms::read_hex_data(
		std::string &str,
		char *&data, int &data_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algorithms::read_hex_data " << str << endl;
	}
	int len, i;
	unsigned char high, low;
	unsigned char val;
	const char *q;

	len = str.length();
	if (f_v) {
		cout << "algorithms::read_hex_data len = " << len << endl;
	}
	data_size = len / 2;

	data = NEW_char(data_size);

	q = str.c_str();

	for (i = 0; i < data_size; i++) {
		if (f_v) {
			cout << "algorithms::read_hex_data i = " << i << endl;
		}
		low = read_hex_digit(q[2 * i + 0]);
		high = read_hex_digit(q[2 * i + 1]);
		val = high * 16 + low;
		if (f_v) {
			cout << "algorithms::read_hex_data high = " << (int) high << endl;
			cout << "algorithms::read_hex_data low = " << (int) low << endl;
			cout << "algorithms::read_hex_data val = " << (int) val << endl;
		}
		data[i] = val;
	}
}

unsigned char algorithms::read_hex_digit(
		char digit)
{
	char val;
	int f_v = true;

	if (f_v) {
		cout << "algorithms::read_hex_data digit = " << digit << endl;
	}
	if (digit >= '0' && digit <= '9') {
		val = (digit - '0');
	}
	else if (digit >= 'A' && digit <= 'F') {
		val = 10 + (digit - 'A');
	}
	else if (digit >= 'a' && digit <= 'f') {
		val = 10 + (digit - 'a');
	}
	else {
		cout << "algorithms::print_hex_digit illegal digit " << digit << endl;
		exit(1);
	}
	return val;
}

void algorithms::print_repeated_character(
		std::ostream &ost, char c, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		ost << c;
	}
}

uint32_t algorithms::root_of_tree_uint32_t(
		uint32_t* S, uint32_t i)
{
	while (S[i] != i) {
		i = S[i];
	}
	return i;
}

void algorithms::solve_diophant(
		int *Inc,
	int nb_rows, int nb_cols, int nb_needed,
	int f_has_Rhs, int *Rhs,
	int *&Solutions, int &nb_sol,
	long int &nb_backtrack, int &dt,
	int f_DLX,
	int verbose_level)
// allocates Solutions[nb_sol * nb_cols]
{
	int f_v = (verbose_level >= 1);
	solvers::diophant *Dio;
	orbiter_kernel_system::os_interface Os;
	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "algorithms::solve_diophant "
				"nb_rows=" << nb_rows
				<< " nb_cols=" << nb_cols
				<< " f_has_Rhs=" << f_has_Rhs
			<< " verbose_level=" << verbose_level << endl;
		cout << "f_DLX=" << f_DLX << endl;
		//int_matrix_print(Inc, nb_rows, nb_cols);
	}
	Dio = NEW_OBJECT(solvers::diophant);

	if (f_has_Rhs) {
		Dio->init_problem_of_Steiner_type_with_RHS(nb_rows,
			nb_cols, Inc, nb_needed,
			Rhs,
			0 /* verbose_level */);
	}
	else {
		Dio->init_problem_of_Steiner_type(nb_rows,
			nb_cols, Inc, nb_needed,
			0 /* verbose_level */);
	}

	if (false /*f_v4*/) {
		Dio->print();
	}

	if (f_DLX && !f_has_Rhs) {
		Dio->solve_all_DLX(0 /* verbose_level*/);
		nb_backtrack = Dio->nb_steps_betten;
	}
	else {
		Dio->solve_all_mckay(nb_backtrack, INT_MAX, verbose_level - 2);
	}

	nb_sol = Dio->_resultanz;
	if (nb_sol) {
		Dio->get_solutions_index_set(Solutions, nb_sol, 1 /* verbose_level */);
		if (false /*f_v4*/) {
			cout << "Solutions:" << endl;
			Int_matrix_print(Solutions, nb_sol, nb_needed);
		}
	}
	else {
		Solutions = NULL;
	}
	FREE_OBJECT(Dio);
	int t1 = Os.os_ticks();
	dt = t1 - t0;
	if (f_v) {
		cout << "algorithms::solve_diophant done nb_sol=" << nb_sol
				<< " nb_backtrack=" << nb_backtrack << " dt=" << dt << endl;
	}
}

#undef get16bits
#if (defined(__GNUC__) && defined(__i386__)) || defined(__WATCOMC__) \
  || defined(_MSC_VER) || defined (__BORLANDC__) || defined (__TURBOC__)
#define get16bits(d) (*((const uint16_t *) (d)))
#endif

#if !defined (get16bits)
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )
#endif


uint32_t algorithms::SuperFastHash(
		const char * data, int len)
{
	uint32_t hash = len, tmp;
	int rem;

    if (len <= 0 || data == 0) return 0;

    rem = len & 3;
    len >>= 2;

    /* Main loop */
    for (;len > 0; len--) {
        hash  += get16bits (data);
        tmp    = (get16bits (data+2) << 11) ^ hash;
        hash   = (hash << 16) ^ tmp;
        data  += 2*sizeof (uint16_t);
        hash  += hash >> 11;
    }

    /* Handle end cases */
    switch (rem) {
        case 3: hash += get16bits (data);
                hash ^= hash << 16;
                hash ^= ((signed char)data[sizeof (uint16_t)]) << 18;
                hash += hash >> 11;
                break;
        case 2: hash += get16bits (data);
                hash ^= hash << 11;
                hash += hash >> 17;
                break;
        case 1: hash += (signed char)*data;
                hash ^= hash << 10;
                hash += hash >> 1;
    }

    /* Force "avalanching" of final 127 bits */
    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash;
}

uint32_t algorithms::SuperFastHash_uint(
		const unsigned int * p, int sz)
{
	uint32_t hash = 0, tmp;
	int rem;

	hash = sz;

    if (sz <= 0 || p == NULL) return 0;

    int len = sizeof(int) * sz;
    const char *data = (const char *) p;

    rem = len & 3;
    len >>= 2;

    /* Main loop */
    for (;len > 0; len--) {
        hash  += get16bits (data);
        tmp    = (get16bits (data+2) << 11) ^ hash;
        hash   = (hash << 16) ^ tmp;
        data  += 2*sizeof (uint16_t);
        hash  += hash >> 11;
    }

    /* Handle end cases */
    switch (rem) {
        case 3: hash += get16bits (data);
                hash ^= hash << 16;
                hash ^= ((signed char)data[sizeof (uint16_t)]) << 18;
                hash += hash >> 11;
                break;
        case 2: hash += get16bits (data);
                hash ^= hash << 11;
                hash += hash >> 17;
                break;
        case 1: hash += (signed char)*data;
                hash ^= hash << 10;
                hash += hash >> 1;
    }

    /* Force "avalanching" of final 127 bits */
    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash;
}



void algorithms::union_of_sets(
		std::string &fname_set_of_sets,
		std::string &fname_input,
		std::string &fname_output, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algorithms::union_of_sets" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	long int *M;
	int m, n;

	Fio.Csv_file_support->lint_matrix_read_csv(
			fname_set_of_sets, M, m, n, verbose_level);

	if (f_v) {
		cout << "algorithms::union_of_sets "
				"the file " << fname_set_of_sets
				<< " contains " << m << " sets of size " << n << endl;
	}

	std::vector<std::vector<long int> > Solutions;
	int solution_size;
	int nb_solutions;


	Fio.count_number_of_solutions_in_file_and_get_solution_size(
			fname_input,
			nb_solutions, solution_size,
			verbose_level);


	if (f_v) {
		cout << "algorithms::union_of_sets "
				"the file " << fname_input
				<< " contains " << nb_solutions << " solutions" << endl;
	}


	Fio.read_solutions_from_file_size_is_known(
			fname_input,
		Solutions, solution_size,
		verbose_level);

	int i, j, h, a, cnt;
	data_structures::sorting Sorting;
	long int *S;
	long int s;
	int len, sz;

	len = Solutions.size();
	if (len != nb_solutions) {
		cout << "algorithms::union_of_sets len != nb_solutions" << endl;
		exit(1);
	}
	sz = solution_size * n;
	S = NEW_lint(len * sz);


	cnt = 0;
	for (i = 0; i < len; i++) {
		vector<long int> U;

		for (j = 0; j < solution_size; j++) {

			a = Solutions[i][j];

			for (h = 0; h < n; h++) {
				s = M[a * n + h];
				S[cnt * sz + j * n + h] = s;
			}
		}

		Sorting.lint_vec_heapsort(S + cnt * sz, sz);

		if (Sorting.lint_vec_test_if_set(S + cnt * sz, sz)) {
			cnt++;
		}
	}


	Fio.Csv_file_support->lint_matrix_write_csv(
			fname_output, S, cnt, sz);
	if (f_v) {
		cout << "algorithms::union_of_sets "
				"written file " << fname_output
				<< " of size " << Fio.file_size(fname_output) << endl;
	}



	FREE_lint(S);
	if (f_v) {
		cout << "algorithms::union_of_sets done" << endl;
	}

}

void algorithms::dot_product_of_columns(
		std::string &label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algorithms::dot_product_of_columns" << endl;
	}

	int *A;
	int m, n;

	Get_matrix(label, A, m, n);

	int *Dot_products;
	int i, j, h, c;

	Dot_products = NEW_int(n * n);

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			c = 0;
			for (h = 0; h < m; h++) {
				c += A[h * n + i] * A[h * n + j];
			}
			Dot_products[i * n + j] = c;
		}
	}

	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = label + "_dot_products_columns.csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Dot_products,
			n, n);

	if (f_v) {
		cout << "Dot_products:" << endl;
		Int_matrix_print(Dot_products, n, n);
	}

	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}



	if (f_v) {
		cout << "algorithms::dot_product_of_columns done" << endl;
	}
}

void algorithms::dot_product_of_rows(
		std::string &label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algorithms::dot_product_of_rows" << endl;
	}

	int *A;
	int m, n;

	Get_matrix(label, A, m, n);

	int *Dot_products;
	int i, j, h, c;

	Dot_products = NEW_int(m * m);

	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			c = 0;
			for (h = 0; h < n; h++) {
				c += A[i * n + h] * A[j * n + h];
			}
			Dot_products[i * m + j] = c;
		}
	}

	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = label + "_dot_products_rows.csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Dot_products,
			m, m);

	if (f_v) {
		cout << "Dot_products:" << endl;
		Int_matrix_print(Dot_products, m, m);
	}

	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}



	if (f_v) {
		cout << "algorithms::dot_product_of_rows done" << endl;
	}
}

void algorithms::matrix_multiply_over_Z(
		std::string &label1, std::string &label2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algorithms::matrix_multiply_over_Z" << endl;
	}

	int *A1;
	int *A2;
	int m1, n1;
	int m2, n2;

	if (f_v) {
		cout << "algorithms::matrix_multiply_over_Z get "
				"matrix " << label1 << endl;
	}
	Get_matrix(label1, A1, m1, n1);

	if (f_v) {
		cout << "algorithms::matrix_multiply_over_Z get "
				"matrix " << label2 << endl;
	}
	Get_matrix(label2, A2, m2, n2);

	if (n1 != m2) {
		cout << "algorithms::matrix_multiply_over_Z "
				"n1 != m2, cannot multiply" << endl;
		exit(1);
	}
	int *A3;
	int m3, n3;


	m3 = m1;
	n3 = n2;
	A3 = NEW_int(m3 * n3);

	algebra::module Mod;

	Mod.matrix_multiply_over_Z_low_level(
			A1, A2, m1, n1, m2, n2,
			A3, verbose_level - 2);

	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = label1 + "_times_" + label2 + ".csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, A3,
			m3, n3);

	if (f_v) {
		cout << "A1 * A2:" << endl;
		Int_matrix_print(A3, m3, n3);
	}

	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "algorithms::matrix_multiply_over_Z done" << endl;
	}
}



void algorithms::matrix_rowspan_over_R(
		std::string &label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algorithms::matrix_rowspan_over_R" << endl;
	}

	int *A;
	int m, n;
	double *D;

	Get_matrix(label, A, m, n);


	int i, j;

	D = new double [m * n];

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			D[i * n + j] = A[i * n + j];
		}
	}

	orbiter_kernel_system::numerics Num;
	int *base_cols;
	int f_complete = true;
	int r;

	base_cols = NEW_int(n);

	r = Num.Gauss_elimination(
				D, m, n,
			base_cols, f_complete,
			verbose_level);





	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = label + "_rref" + ".csv";

	Fio.Csv_file_support->double_matrix_write_csv(
			fname, D, r, n);

	if (f_v) {
		cout << "RREF=" << endl;
		Num.print_matrix(D, r, n);
	}
	if (f_v) {
		cout << "The rank of the matrix is " << r << endl;
	}

	if (f_v) {
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}


	delete [] D;


	if (f_v) {
		cout << "algorithms::matrix_rowspan_over_R done" << endl;
	}
}

int algorithms::binary_logarithm(
		int m)
{
	int i = 0;

	while (m) {
		i++;
		m >>= 1;
	}
	return i;
}

char algorithms::make_single_hex_digit(
		int c)
{
	if (c < 10) {
		return '0' + c;
	}
	else {
		return 'a' + c - 10;
	}
}

void algorithms::process_class_list(
		std::vector<std::vector<std::string> > &Classes_parsed,
		std::string &fname_cross_ref,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algorithms::process_class_list" << endl;
	}

	if (f_v) {
		cout << "algorithms::process_class_list Classes_parsed.size()=" << Classes_parsed.size() << endl;
	}
	int i, j;
	for (i = 0; i < Classes_parsed.size(); i++) {
		cout << setw(3) << i << " : ";
    	for (j = 0; j < Classes_parsed[i].size(); j++) {
    		cout << Classes_parsed[i][j];
    		if (j < Classes_parsed[i].size() - 1) {
    			cout << ", ";
			}
    	}
    	cout << endl;
	}
	int maxdepth = 0;
	for (i = 0; i < Classes_parsed.size(); i++) {
		maxdepth = MAXIMUM(maxdepth, Classes_parsed[i].size());
	}
	//map<string,int> *name;
	//int *Nb;
	//name = new map<string,int> [maxdepth];
	//Nb = NEW_int(maxdepth);
	//Int_vec_zero(Nb, maxdepth);

	vector<vector<string>> Names;

	for (j = 0; j < maxdepth; j++) {

		vector<string> names;
		unordered_set<string> Set;
		for (i = 0; i < Classes_parsed.size(); i++) {
			if (Classes_parsed[i].size() > j) {
				string s;

				s = Classes_parsed[i][j];
				Set.insert(s);
			}
		}

		names.assign( Set.begin(), Set.end() );
		sort( names.begin(), names.end() );
		Names.push_back(names);

	}
	for (j = 0; j < maxdepth; j++) {
		cout << "depth " << j << ":" << endl;
		for (i = 0; i < Names[j].size(); i++) {
			cout << i << " : " << Names[j][i] << endl;
		}
	}

	map<string,int> *map_name;

	map_name = new map<string,int> [maxdepth];
	for (j = 0; j < maxdepth; j++) {
		for (i = 0; i < Names[j].size(); i++) {
			map_name[j].insert(pair<string, int>(Names[j][i], i));
		}
	}

	vector<vector<int>> Tree;

	for (i = 0; i < Classes_parsed.size(); i++) {
		vector<int> path;
    	for (j = 0; j < Classes_parsed[i].size(); j++) {
    		string s;

    		s = Classes_parsed[i][j];
    		path.push_back(map_name[j][s]);
    	}
    	Tree.push_back(path);
	}

	sort(Tree.begin(), Tree.end());

	for (i = 0; i < Tree.size(); i++) {
		cout << setw(3) << i << " : ";
		for (j = 0; j < Tree[i].size(); j++) {
			cout << Tree[i][j];
			if (j < Tree[i].size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}

	data_structures::string_tools String;
	orbiter_kernel_system::file_io Fio;


	std::string *col_label;
	std::string *Table;
	int m, n;

	Fio.Csv_file_support->read_table_of_strings(
			fname_cross_ref, col_label,
			Table, m, n,
			verbose_level);

	if (n != 3) {
		cout << "algorithms::process_class_list n != 3" << endl;
		exit(1);
	}

	for (i = 0; i < m; i++) {
		cout << "row " << i;
		for (j = 0; j < n; j++) {
			cout << " : " << Table[i * 3 + j];
		}
		cout << endl;
	}

	int c;
	for (i = 0; i < Tree.size(); i++) {
		cout << setw(3) << i << " : ";
		for (j = 0; j < Tree[i].size(); j++) {
			c = Tree[i][j];

			string s1, s2, s3;

			s1 = Names[j][c];

			String.make_latex_friendly_string(
					s1, s2, 0 /* verbose_level*/);

			if (j == Tree[i].size() - 1) {

				int h;
				vector<int> Idx;

				for (h = 0; h < m; h++) {
					if (Table[h * 3 + 0] == s1) {
						Idx.push_back(h);
					}
				}

				s3 = "{\\bf " + s2;
				for (h = 0; h < Idx.size(); h++) {
					string s;

					s = " Table~\\ref{" + Table[Idx[h] * 3 + 2] + "}";
					if (h < Idx.size() - 1) {
						s += ", ";
					}
					s3 += s;
				}
				s3 += "}";
			}
			else {
				s3 = s2;
			}
			cout << s3;
			if (j < Tree[i].size() - 1) {
				cout << ", ";
			}
		}
		cout << "\\\\" << endl;
	}


	if (f_v) {
		cout << "algorithms::process_class_list maxdepth = " << maxdepth << endl;
	}



	if (f_v) {
		cout << "algorithms::process_class_list done" << endl;
	}

}

void algorithms::filter_duplicates_and_make_array_of_long_int(
		std::vector<long int> &In, long int *&Out, int &size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algorithms::filter_duplicates_and_make_array_of_long_int" << endl;
	}

	// filter out duplicates:
	unordered_set<long int> Pts_as_set;

	int i;

	for (i = 0; i < In.size(); i++) {
		Pts_as_set.insert(In[i]);
	}

	vector<long int> Pts2;
	Pts2.assign( Pts_as_set.begin(), Pts_as_set.end() );
	sort( Pts2.begin(), Pts2.end() );

	// convert to array of long int:
	size = Pts2.size();
	Out = NEW_lint(size);
	for (i = 0; i < size; i++) {
		Out[i] = Pts2[i];
	}

	if (f_v) {
		cout << "algorithms::filter_duplicates_and_make_array_of_long_int done" << endl;
	}
}


void algorithms::set_minus(
		std::vector<long int> &In, long int *subtract_this, int size,
		std::vector<long int> &Out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algorithms::set_minus" << endl;
	}

	unordered_set<long int> set;

	int i;

	for (i = 0; i < In.size(); i++) {
		set.insert(In[i]);
	}


	for (i = 0; i < size; i++) {
		set.erase(subtract_this[i]);
	}
	Out.assign( set.begin(), set.end() );
	sort( Out.begin(), Out.end() );

	if (f_v) {
		cout << "algorithms::set_minus done" << endl;
	}

}

}}}


