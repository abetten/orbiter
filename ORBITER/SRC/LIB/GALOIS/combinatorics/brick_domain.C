// brick_domain.C
//
// Anton Betten
//
// Jan 10, 2013

#include "galois.h"


brick_domain::brick_domain()
{
	null();
}

brick_domain::~brick_domain()
{
	freeself();
}

void brick_domain::null()
{
}

void brick_domain::freeself()
{
	//INT i;
	
	null();
}

void brick_domain::init(finite_field *F, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT i;
	
	brick_domain::F = F;
	q = F->q;
	if (F->e > 1) {
		cout << "brick_domain::init field order must be a prime" << endl;
		exit(1);
		}
	nb_bricks = 2 * q * q;
	if (f_v) {
		cout << "brick_domain::init q=" << q << " nb_bricks=" << nb_bricks << endl;
		}
}

void brick_domain::unrank(INT rk, INT &f_vertical, INT &x0, INT &y0, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	f_vertical = rk % 2;
	rk >>= 1;
	x0 = rk % q;
	rk /= q;
	y0 = rk;
	if (f_v) {
		cout << "brick_domain::unrank rk=" << rk << " f_vertical=" << f_vertical << " x0=" << x0 << " y0=" << y0 << endl;
		}
}

INT brick_domain::rank(INT f_vertical, INT x0, INT y0, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rk;
	
	rk = y0;
	rk *= q;
	rk += x0;
	rk *= 2;
	if (f_vertical) {
		rk++;
		}
	if (f_v) {
		cout << "brick_domain::rank rk=" << rk << " f_vertical=" << f_vertical << " x0=" << x0 << " y0=" << y0 << endl;
		}
	return rk;
}

void brick_domain::unrank_coordinates(INT rk, INT &x1, INT &y1, INT &x2, INT &y2, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	INT x0, y0, f_vertical;

	unrank(rk, f_vertical, x0, y0, verbose_level);
	x1 = x0;
	y1 = y0;
	if (f_vertical) {
		x2 = x1;
		y2 = F->add(y1, 1);
		}
	else {
		x2 = F->add(x1, 1);
		y2 = y1;
		}
}

INT brick_domain::rank_coordinates(INT x1, INT y1, INT x2, INT y2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rk;
	INT x0, y0, f_vertical;
	
	if (x1 == x2) {
		f_vertical = TRUE;
		x0 = x1;
		if (y2 == F->add(y1, 1)) {
			y0 = y1;
			}
		else if (y1 == F->add(y2, 1)) {
			y0 = y2;
			}
		else {
			cout << "brick_domain::rank_coordinates y-coordinates apart" << endl;
			exit(1);
			}
		}
	else if (y1 == y2) {
		f_vertical = FALSE;
		y0 = y1;
		if (x2 == F->add(x1, 1)) {
			x0 = x1;
			}
		else if (x1 == F->add(x2, 1)) {
			x0 = x2;
			}
		else {
			cout << "brick_domain::rank_coordinates x-coordinates apart" << endl;
			exit(1);
			}
		}
	else {
		cout << "brick_domain::rank_coordinates neither vertical nor horizontal" << endl;
		exit(1);
		}
	rk = y0;
	rk *= q;
	rk += x0;
	rk *= 2;
	if (f_vertical) {
		rk++;
		}
	if (f_v) {
		cout << "brick_domain::rank_coordinates rk=" << rk << " f_vertical=" << f_vertical << " x0=" << x0 << " y0=" << y0 << endl;
		}
	return rk;
}

void brick_test(INT q, INT verbose_level)
{
	brick_domain B;
	INT i, j;
	INT f_vertical, x, y, x2, y2;
	finite_field F;
	
	F.init(q, 0);
	B.init(&F, verbose_level);
	for (i = 0; i < B.nb_bricks; i++) {
		B.unrank(i, f_vertical, x, y, 0);
		j = B.rank(f_vertical, x, y, 0);
		if (j != i) {
			cout << "brick_test i != j in rank" << endl;
			exit(1);
			}
		B.unrank_coordinates(i, x, y, x2, y2, 0);
		j = B.rank_coordinates(x, y, x2, y2, 0);
		if (j != i) {
			cout << "brick_test i != j in rank_coordinates" << endl;
			cout << "i=" << i << endl;
			cout << "x=" << x << endl;
			cout << "y=" << y << endl;
			cout << "x2=" << x2 << endl;
			cout << "y2=" << y2 << endl;
			exit(1);
			}
		}
	cout << "brick_test: OK" << endl;
}


