/*
 * nauty_output.cpp
 *
 *  Created on: Aug 21, 2021
 *      Author: betten
 */


#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace l1_interfaces {


nauty_output::nauty_output()
{
	N = 0;
	invariant_set_start = 0;
	invariant_set_size = 0;

	Aut = NULL;
	Aut_counter = 0;
	Base = NULL;
	Base_length = 0;
	Base_lint = NULL;
	Transversal_length = NULL;
	Ago = NULL;

	canonical_labeling = NULL;

	nb_firstpathnode = 0;
	nb_othernode = 0;
	nb_processnode = 0;
	nb_firstterminal = 0;
}

nauty_output::~nauty_output()
{
	//cout << "nauty_output::~nauty_output" << endl;
	if (Aut) {
		//cout << "before FREE_int(Aut);" << endl;
		FREE_int(Aut);
	}
	if (Base) {
		//cout << "before FREE_int(Base);" << endl;
		FREE_int(Base);
	}
	if (Base_lint) {
		//cout << "before FREE_lint(Base_lint);" << endl;
		FREE_lint(Base_lint);
	}
	if (Transversal_length) {
		//cout << "before FREE_int(Transversal_length);" << endl;
		FREE_int(Transversal_length);
	}
	if (Ago) {
		//cout << "before FREE_OBJECT(Ago);" << endl;
		FREE_OBJECT(Ago);
	}
	if (canonical_labeling) {
		//cout << "before FREE_int(canonical_labeling);" << endl;
		FREE_int(canonical_labeling);
	}
	//cout << "nauty_output::~nauty_output done" << endl;
}

void nauty_output::nauty_output_allocate(
		int N,
		int invariant_set_start, int invariant_set_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "nauty_output::nauty_output_allocate" << endl;
	}
	nauty_output::N = N;
	nauty_output::invariant_set_start = invariant_set_start;
	nauty_output::invariant_set_size = invariant_set_size;

	Aut = NEW_int(N * N);
	Base = NEW_int(N);
	Base_lint = NEW_lint(N);
	Transversal_length = NEW_int(N);
	Ago = NEW_OBJECT(ring_theory::longinteger_object);
	canonical_labeling = NEW_int(N);

	int i;

	for (i = 0; i < N; i++) {
		canonical_labeling[i] = i;
	}
}

void nauty_output::print()
{
		cout << "nauty_output::print" << endl;
		cout << "N=" << N << endl;
		cout << "invariant_set_start=" << invariant_set_start << endl;
		cout << "invariant_set_size=" << invariant_set_size << endl;
}

void nauty_output::print_stats()
{
	cout << "nb_backtrack1 = " << nb_firstpathnode << endl;
	cout << "nb_backtrack2 = " << nb_othernode << endl;
	cout << "nb_backtrack3 = " << nb_processnode << endl;
	cout << "nb_backtrack4 = " << nb_firstterminal << endl;

}

int nauty_output::belong_to_the_same_orbit(
		int a, int b, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "nauty_output::belong_to_the_same_orbit" << endl;
	}

	int *prev;
	int *orbit;
	int *Q;
	int Q_len;
	int orbit_len;
	int c, d;
	int i, j;
	int nb_gen;

	nb_gen = Aut_counter;
	prev = NEW_int(N);
	orbit = NEW_int(N);
	Q = NEW_int(N);
	Q[0] = a;
	Q_len = 1;
	orbit_len = 0;
	prev[a] = a;

	for (i = 0; i < N; i++) {
		prev[i] = -1;
	}

	while (Q_len) {
		c = Q[0];
		for (i = 1; i < Q_len; i++) {
			Q[i - 1] = Q[i];
		}
		Q_len--;
		orbit[orbit_len++] = c;
		for (j = 0; j < nb_gen; j++) {
			d = Aut[j * N + c];
			if (prev[d] == -1) {
				prev[d] = c;
				Q[Q_len++] = d;
				if (d == b) {
					FREE_int(prev);
					FREE_int(orbit);
					FREE_int(Q);
					return true;
				}
			}
		}
	}

	if (f_v) {
		cout << "nauty_output::belong_to_the_same_orbit done" << endl;
	}
	return false;
}

void nauty_output::stringify_as_vector(
		std::vector<std::string> &V,
		int verbose_level)
{
	std::string s_n;
	std::string s_ago;
	std::string s_base_length;
	std::string s_aut_counter;
	std::string s_base;
	std::string s_tl;
	std::string s_aut;
	std::string s_cl;
	std::string s_stats;

	stringify(s_n,
		s_ago,
		s_base_length,
		s_aut_counter,
		s_base,
		s_tl,
		s_aut,
		s_cl,
		s_stats,
		verbose_level - 1);

	V.push_back(s_n);
	V.push_back(s_ago);
	V.push_back(s_base_length);
	V.push_back(s_aut_counter);
	V.push_back(s_base);
	V.push_back(s_tl);
	V.push_back(s_aut);
	V.push_back(s_cl);
	V.push_back(s_stats);
}

void nauty_output::stringify(
		std::string &s_n, std::string &s_ago,
		std::string &s_base_length, std::string &s_aut_counter,
		std::string &s_base, std::string &s_tl,
		std::string &s_aut, std::string &s_cl,
		std::string &s_stats,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "nauty_output::stringify" << endl;
	}

	s_n = std::to_string(N);
	s_ago = Ago->stringify();
	s_base_length = std::to_string(Base_length);
	s_aut_counter = std::to_string(Aut_counter);
	s_base = Int_vec_stringify(Base, Base_length);
	s_tl = Int_vec_stringify(Transversal_length, Base_length);
	s_aut = Int_vec_stringify(Aut, Aut_counter * N);
	s_cl = Int_vec_stringify(canonical_labeling, N);

	long int stats[4];

	stats[0] = nb_firstpathnode;
	stats[1] = nb_othernode;
	stats[2] = nb_processnode;
	stats[3] = nb_firstterminal;

	s_stats = Lint_vec_stringify(stats, 4);

	if (f_v) {
		cout << "nauty_output::stringify done" << endl;
	}
}

void nauty_output::nauty_output_init_from_string(
		int N,
		int invariant_set_start, int invariant_set_size,
		int idx_start,
		std::vector<std::string> &Carrying_through,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "nauty_output::nauty_output_init_from_string" << endl;
	}
	nauty_output::N = N;
	nauty_output::invariant_set_start = invariant_set_start;
	nauty_output::invariant_set_size = invariant_set_size;

	if (f_v) {
		cout << "nauty_output::nauty_output_init_from_string idx_start = " << idx_start << endl;
		cout << "nauty_output::nauty_output_init_from_string Carrying_through.size() = " << Carrying_through.size() << endl;
	}
	if (idx_start + 9 < Carrying_through.size()) {
		cout << "nauty_output::nauty_output_init_from_string idx_start + 9 < Carrying_through.size()" << endl;
		exit(1);
	}
	string &s_n = Carrying_through[idx_start + 0];
	string &s_ago = Carrying_through[idx_start + 1];
	string &s_base_length = Carrying_through[idx_start + 2];
	string &s_aut_counter = Carrying_through[idx_start + 3];
	string &s_base = Carrying_through[idx_start + 4];
	string &s_tl = Carrying_through[idx_start + 5];
	string &s_aut = Carrying_through[idx_start + 6];
	string &s_cl = Carrying_through[idx_start + 7];
	string &s_stats = Carrying_through[idx_start + 8];

	data_structures::string_tools String;

	std::string s;

	String.drop_quotes(s_n, s);
	s_n = s;
	String.drop_quotes(s_ago, s);
	s_ago = s;
	String.drop_quotes(s_base_length, s);
	s_base_length = s;
	String.drop_quotes(s_aut_counter, s);
	s_aut_counter = s;
	String.drop_quotes(s_base, s);
	s_base = s;
	String.drop_quotes(s_tl, s);
	s_tl = s;
	String.drop_quotes(s_aut, s);
	s_aut = s;
	String.drop_quotes(s_cl, s);
	s_cl = s;
	String.drop_quotes(s_stats, s);
	s_stats = s;

	int len;
	int N1;

	if (f_v) {
		cout << "nauty_output::nauty_output_init_from_string s_n = " << s_n << endl;
	}
	N1 = std::stoi(s_n);
	if (N1 != N) {
		cout << "nauty_output::nauty_output_init_from_string scanning N: N1 != N" << endl;
		exit(1);
	}
	Ago = NEW_OBJECT(ring_theory::longinteger_object);
	Ago->create_from_base_10_string(s_ago);
	if (f_v) {
		cout << "nauty_output::nauty_output_init_from_string s_base_length = " << s_base_length << endl;
	}

	if (Ago->is_one()) {
		Base_length = 0;
		Aut_counter = 0;
		Base = NEW_int(1);
		Transversal_length = NEW_int(1);
		Aut = NEW_int(1);
	}
	else {
		Base_length = std::stoi(s_base_length);

		Aut_counter = std::stoi(s_aut_counter);
		if (f_v) {
			cout << "nauty_output::nauty_output_init_from_string Aut_counter = " << Aut_counter << endl;
		}

		if (f_v) {
			cout << "nauty_output::nauty_output_init_from_string s_base = " << s_base << endl;
		}
		Int_vec_scan(s_base, Base, len);
		if (len != Base_length) {
			cout << "nauty_output::nauty_output_init_from_string scanning Base: len != Base_length" << endl;
			cout << "nauty_output::nauty_output_init_from_string len = " << len << endl;
			cout << "nauty_output::nauty_output_init_from_string Base_length = " << Base_length << endl;
			exit(1);
		}
		Int_vec_scan(s_tl, Transversal_length, len);
		if (len != Base_length) {
			cout << "nauty_output::nauty_output_init_from_string scanning Transversal_length: len != Base_length" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "nauty_output::nauty_output_init_from_string s_aut = " << s_aut << endl;
		}
		Int_vec_scan(s_aut, Aut, len);
		if (len != Aut_counter * N) {
			cout << "nauty_output::nauty_output_init_from_string scanning Aut: len != Aut_counter * N" << endl;
			cout << "nauty_output::nauty_output_init_from_string len = " << len << endl;
			cout << "nauty_output::nauty_output_init_from_string Aut_counter = " << Aut_counter << endl;
			cout << "nauty_output::nauty_output_init_from_string N = " << N << endl;
			exit(1);
		}
	}
	Int_vec_scan(s_cl, canonical_labeling, len);
	if (len != N) {
		cout << "nauty_output::nauty_output_init_from_string scanning canonical_labeling: len != N" << endl;
		exit(1);
	}
	long int *Stats;
	if (f_v) {
		cout << "nauty_output::nauty_output_init_from_string s_stats = " << s_stats << endl;
	}
	Lint_vec_scan(s_stats, Stats, len);
	if (len != 4) {
		cout << "nauty_output::nauty_output_init_from_string scanning Stats: len != 4" << endl;
		exit(1);
	}

	nb_firstpathnode = Stats[0];
	nb_othernode = Stats[1];
	nb_processnode = Stats[2];
	nb_firstterminal = Stats[3];

	FREE_lint(Stats);

#if 0
	s_n = std::to_string(N);
	s_ago = Ago->stringify();
	s_base_length = std::to_string(Base_length);
	s_aut_counter = std::to_string(Aut_counter);
	s_base = Int_vec_stringify(Base, Base_length);
	s_tl = Int_vec_stringify(Transversal_length, Base_length);
	s_aut = Int_vec_stringify(Aut, Aut_counter * N);
	s_cl = Int_vec_stringify(canonical_labeling, N);


	Aut = NEW_int(N * N);
	Base = NEW_int(N);
	Base_lint = NEW_lint(N);
	Transversal_length = NEW_int(N);
	Ago = NEW_OBJECT(ring_theory::longinteger_object);
	canonical_labeling = NEW_int(N);

	int i;

	for (i = 0; i < N; i++) {
		canonical_labeling[i] = i;
	}
#endif

	if (f_v) {
		cout << "nauty_output::nauty_output_init_from_string done" << endl;
	}

}

long int nauty_output::nauty_complexity()
{
	long int c;

	c = nb_firstpathnode + nb_othernode + nb_processnode + nb_firstterminal;
	return c;
}


}}}

