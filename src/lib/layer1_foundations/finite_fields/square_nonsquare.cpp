/*
 * square_nonsquare.cpp
 *
 *  Created on: Sep 30, 2022
 *      Author: betten
 */






#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace field_theory {


square_nonsquare::square_nonsquare()
{
	F = NULL;

	minus_squares = NULL;
	minus_squares_without = NULL;
	minus_nonsquares = NULL;
	f_is_minus_square = FALSE;
	index_minus_square = NULL;
	index_minus_square_without = NULL;
	index_minus_nonsquare = NULL;

}

square_nonsquare::~square_nonsquare()
{
	if (minus_squares) {
		FREE_int(minus_squares);
	}
	if (minus_squares_without) {
		FREE_int(minus_squares_without);
	}
	if (minus_nonsquares) {
		FREE_int(minus_nonsquares);
	}
	if (f_is_minus_square) {
		FREE_int(f_is_minus_square);
	}
	if (index_minus_square) {
		FREE_int(index_minus_square);
	}
	if (index_minus_square_without) {
		FREE_int(index_minus_square_without);
	}
	if (index_minus_nonsquare) {
		FREE_int(index_minus_nonsquare);
	}




}


void square_nonsquare::init(field_theory::finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = FALSE; //(verbose_level >= 1);

	if (f_v) {
		cout << "square_nonsquare::init" << endl;
	}

	square_nonsquare::F = F;

	int q = F->q;
	int a, b, c;
	int i, j;

	if (f_v) {
		cout << "square_nonsquare::init q=" << q << endl;
	}

	minus_squares = NEW_int((q-1)/2);
	minus_squares_without = NEW_int((q-1)/2 - 1);
	minus_nonsquares = NEW_int((q-1)/2);
	f_is_minus_square = NEW_int(q);
	index_minus_square = NEW_int(q);
	index_minus_square_without = NEW_int(q);
	index_minus_nonsquare = NEW_int(q);
	a = b = c = 0;
	if (f_v) {
		cout << "square_nonsquare::init computing minus_squares:" << endl;
	}
	for (i = 0; i < q; i++) {
		index_minus_square[i] = -1;
		index_minus_square_without[i] = -1;
		index_minus_nonsquare[i] = -1;
		f_is_minus_square[i]= FALSE;
	}
	for (i = 0; i < q - 1; i++) {
		if (f_v) {
			cout << "square_nonsquare::init i=" << i << endl;
		}
		j = F->alpha_power(i);
		if (f_v) {
			cout << "square_nonsquare::init j=" << j << endl;
		}
		if (is_minus_square(i)) {
			if (f_v) {
				cout << "i=" << i << " j=" << j
						<< " is minus a square" << endl;
			}
			f_is_minus_square[j]= TRUE;
			minus_squares[a] = j;
			index_minus_square[j] = a;
			if (j != F->negate(1)) {
				minus_squares_without[b] = j;
				index_minus_square_without[j] = b;
				b++;
			}
			a++;
		}
		else {
			minus_nonsquares[c] = j;
			index_minus_nonsquare[j] = c;
			c++;
		}
	}
	if (f_v) {
		cout << "minus_squares:" << endl;
		for (i = 0; i < a; i++) {
			cout << i << " : " << minus_squares[i] << endl;
		}
		cout << "minus_squares_without:" << endl;
		for (i = 0; i < b; i++) {
			cout << i << " : " << minus_squares_without[i] << endl;
		}
		cout << "minus_nonsquares:" << endl;
		for (i = 0; i < c; i++) {
			cout << i << " : " << minus_nonsquares[i] << endl;
		}
		print_minus_square_tables();
	}

	if (f_v) {
		cout << "square_nonsquare::init done" << endl;
	}
}

int square_nonsquare::is_minus_square(int i)
{
	if (DOUBLYEVEN(F->q - 1)) {
		if (EVEN(i)) {
			return TRUE;
		}
		else {
			return FALSE;
		}
	}
	else {
		if (EVEN(i)) {
			return FALSE;
		}
		else {
			return TRUE;
		}
	}
}

void square_nonsquare::print_minus_square_tables()
{
	int i;

	cout << "field element indices and f_minus_square:" << endl;
	for (i = 0; i < F->q; i++) {
			cout << i << " : "
			<< setw(3) << index_minus_square[i] << ","
			<< setw(3) << index_minus_square_without[i] << ","
			<< setw(3) << index_minus_nonsquare[i] << " : "
			<< setw(3) << f_is_minus_square[i] << endl;
	}
}



}}}

