// eckardt_point.cpp
//
// Anton Betten
// January 12, 2017

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


eckardt_point::eckardt_point()
{
	len = 0;
	pt = 0;
	//int index[3];

}

eckardt_point::~eckardt_point()
{
}

void eckardt_point::print()
{

	latex(cout);

}

void eckardt_point::latex(
		std::ostream &ost)
{
	int t, i, j;
	combinatorics::combinatorics_domain Combi;
	
	string s;

	s = make_label();
	ost << "E_{" << s << "}" << endl;
}

std::string eckardt_point::make_label()
{
	int t, i, j;
	combinatorics::combinatorics_domain Combi;
	string s;

	if (len == 3) {
		s = "";
		for (t = 0; t < 3; t++) {
			Combi.k2ij(index[t], i, j, 6);
			s += std::to_string(i + 1) + std::to_string(j + 1);
			if (t < 2) {
				s += ",";
			}
		}
	}
	else if (len == 2) {
		s = std::to_string(index[0] + 1) + std::to_string(index[1] + 1);
	}
	else {
		cout << "eckardt_point::make_label len is illegal" << endl;
		exit(1);
	}
	return s;
}


void eckardt_point::latex_index_only(
		std::ostream &ost)
{
	string s;

	s = make_label();
	ost << s;

}


std::string eckardt_point::make_symbol()
{
	string s1, s2;

	s1 = make_label();
	s2 = "E_{" + s1 + "}";
	return s2;
}




void eckardt_point::init2(
		int i, int j)
{
	len = 2;
	index[0] = i;
	index[1] = j;
}

void eckardt_point::init3(
		int ij, int kl, int mn)
{
	len = 3;
	index[0] = ij;
	index[1] = kl;
	index[2] = mn;
}

void eckardt_point::init6(
		int i, int j, int k, int l, int m, int n)
{
	combinatorics::combinatorics_domain Combi;

	len = 3;
	index[0] = Combi.ij2k(i, j, 6);
	index[1] = Combi.ij2k(k, l, 6);
	index[2] = Combi.ij2k(m, n, 6);
}

void eckardt_point::init_by_rank(
		int rk)
{
	combinatorics::combinatorics_domain Combi;

	if (rk < 30) {
		len = 2;
		Combi.ordered_pair_unrank(rk, index[0], index[1], 6);
	}
	else {
		int i, j, k, l, m, n;
		
		len = 3;
		rk -= 30;
		Combi.unordered_triple_pair_unrank(rk, i, j, k, l, m, n);
		index[0] = Combi.ij2k(i, j, 6);
		index[1] = Combi.ij2k(k, l, 6);
		index[2] = Combi.ij2k(m, n, 6);
	}
}


void eckardt_point::three_lines(
		surface_domain *S, int *three_lines)
{
	combinatorics::combinatorics_domain Combi;

	if (len == 2) {
		three_lines[0] = S->Schlaefli->line_ai(index[0]);
		three_lines[1] = S->Schlaefli->line_bi(index[1]);
		three_lines[2] = S->Schlaefli->line_cij(index[0], index[1]);
	}
	else if (len == 3) {
		int i, j;

		Combi.k2ij(index[0], i, j, 6);
		three_lines[0] = S->Schlaefli->line_cij(i, j);
		Combi.k2ij(index[1], i, j, 6);
		three_lines[1] = S->Schlaefli->line_cij(i, j);
		Combi.k2ij(index[2], i, j, 6);
		three_lines[2] = S->Schlaefli->line_cij(i, j);
	}
	else {
		cout << "eckardt_point::three_lines len must be 2 or 3" << endl;
		exit(1);
	}
}

int eckardt_point::rank()
{
	int a;
	combinatorics::combinatorics_domain Combi;
	
	if (len == 2) {
		a = Combi.ordered_pair_rank(index[0], index[1], 6);
		return a;
	}
	else if (len == 3) {
		int i, j, k, l, m, n;

		Combi.k2ij(index[0], i, j, 6);
		Combi.k2ij(index[1], k, l, 6);
		Combi.k2ij(index[2], m, n, 6);
		a = Combi.unordered_triple_pair_rank(i, j, k, l, m, n);
		return 30 + a;
	}
	else {
		cout << "eckardt_point::rank len must be 2 or 3" << endl;
		exit(1);
	}
}

void eckardt_point::unrank(
		int rk,
		int &i, int &j, int &k, int &l, int &m, int &n)
{
	combinatorics::combinatorics_domain Combi;

	if (rk < 30) {
		len = 2;
		Combi.ordered_pair_unrank(rk, index[0], index[1], 6);
	}
	else {
		rk -= 30;
		Combi.unordered_triple_pair_unrank(rk, i, j, k, l, m, n);
		index[0] = Combi.ij2k(i, j, 6);
		index[1] = Combi.ij2k(k, l, 6);
		index[2] = Combi.ij2k(m, n, 6);
	}
}


}}}

