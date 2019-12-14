/*
 * tutorial.cpp
 *
 *  Created on: Dec 9, 2019
 *      Author: betten
 */






#include "ginac/ginac.h"

using namespace GiNaC;

#include <iostream>
#include <sstream>
using namespace std;


void t1()
{
	symbol x("x"), y("y"), z("z");
	ex e2 = x*y + x;
	cout << "e2=" << e2 << endl;
	cout << "e2(-2, 4) = " << e2.subs(lst{x, y}, lst{-2, 4}) << endl;
	cout << "e2(z, 4) = " << e2.subs(lst{x, y}, lst{z, 4}) << endl;
}

void t2()
{
	symbol x0("x0"), x1("x1"), x2("x2");
	ex e2 = x0*x1 + x0 + numeric(1,2);
	ostringstream s;
	s << latex << e2;
	cout << "e2=" << s.str() << endl;
	cout << "e2(-2, 4) = " << e2.subs(lst{x0, x1}, lst{-2, 4}) << endl;
	cout << "e2(x2, 4) = " << e2.subs(lst{x0, x1}, lst{x2, 4}) << endl;
}

void t3()
{
	parser reader;
	ex e = reader("2*x+sin(y)");
	ostringstream s;
	s << latex << e;
	cout << "e=" << s.str() << endl;
	symtab table = reader.get_syms();
	symbol x = ex_to<symbol>(table["x"]);
	symbol y = ex_to<symbol>(table["y"]);
}

int main(int argc, char** argv)
{
	t1();
	t2();
	t3();
}


