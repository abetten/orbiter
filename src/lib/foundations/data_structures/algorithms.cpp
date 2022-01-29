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

int algorithms::hashing(int hash0, int a)
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

int algorithms::hashing_fixed_width(int hash0, int a, int bit_length)
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

void algorithms::uchar_print_bitwise(std::ostream &ost, uchar u)
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

void algorithms::uchar_move(uchar *p, uchar *q, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		*q++ = *p++;
	}
}

void algorithms::int_swap(int& x, int& y)
{
	int z;

	z = x;
	x = y;
	y = z;
}

void algorithms::print_pointer_hex(std::ostream &ost, void *p)
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

void algorithms::print_hex_digit(std::ostream &ost, int digit)
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

void algorithms::print_repeated_character(std::ostream &ost, char c, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		ost << c;
	}
}

uint32_t algorithms::root_of_tree_uint32_t (uint32_t* S, uint32_t i)
{
	while (S[i] != i) {
		i = S[i];
	}
	return i;
}

void algorithms::solve_diophant(int *Inc,
	int nb_rows, int nb_cols, int nb_needed,
	int f_has_Rhs, int *Rhs,
	long int *&Solutions, int &nb_sol, long int &nb_backtrack, int &dt,
	int f_DLX,
	int verbose_level)
// allocates Solutions[nb_sol * nb_needed]
{
	int f_v = (verbose_level >= 1);
	solvers::diophant *Dio;
	orbiter_kernel_system::os_interface Os;
	int t0 = Os.os_ticks();

	if (f_v) {
		cout << "algorithms::solve_diophant nb_rows=" << nb_rows << " nb_cols="
			<< nb_cols << " f_has_Rhs=" << f_has_Rhs
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

	if (FALSE /*f_v4*/) {
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
		Dio->get_solutions(Solutions, nb_sol, 1 /* verbose_level */);
		if (FALSE /*f_v4*/) {
			cout << "Solutions:" << endl;
			Lint_matrix_print(Solutions, nb_sol, nb_needed);
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


uint32_t algorithms::SuperFastHash (const char * data, int len)
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



}}}

