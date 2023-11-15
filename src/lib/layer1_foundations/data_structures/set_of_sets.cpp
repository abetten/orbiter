// set_of_sets.cpp
//
// Anton Betten
//
// November 30, 2012

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace data_structures {

static int set_of_sets_compare_func(
		void *data, int i, int j, void *extra_data);
static void set_of_sets_swap_func(
		void *data, int i, int j, void *extra_data);



set_of_sets::set_of_sets()
{
	underlying_set_size = 0;
	nb_sets = 0;
	Sets = NULL;
	Set_size = NULL;
}

set_of_sets::~set_of_sets()
{
	int i;
	
	if (Sets) {
		for (i = 0; i < nb_sets; i++) {
			if (Sets[i]) {
				FREE_lint(Sets[i]);
			}
		}
		FREE_plint(Sets);
		FREE_lint(Set_size);
	}
}

set_of_sets *set_of_sets::copy()
{
	set_of_sets *SoS;

	SoS = NEW_OBJECT(set_of_sets);

	SoS->init(underlying_set_size,
			nb_sets, Sets, Set_size, 0 /*verbose_level */);
	return SoS;
}

void set_of_sets::init_simple(
		int underlying_set_size,
		int nb_sets, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "set_of_sets::init_simple nb_sets=" << nb_sets
				<< " underlying_set_size=" << underlying_set_size << endl;
	}
	set_of_sets::nb_sets = nb_sets;
	set_of_sets::underlying_set_size = underlying_set_size;
	Sets = NEW_plint(nb_sets);
	Set_size = NEW_lint(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sets[i] = NULL;
	}
	Lint_vec_zero(Set_size, nb_sets);
}

void set_of_sets::init_from_adjacency_matrix(
		int n, int *Adj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	
	if (f_v) {
		cout << "set_of_sets::init_from_adjacency_matrix "
				"n=" << n << endl;
	}
	init_simple(n, n, 0 /* verbose_level*/);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (Adj[i * n + j]) {
				Set_size[i]++;
			}
		}
	}
	for (i = 0; i < n; i++) {
		Sets[i] = NEW_lint(Set_size[i]);
	}
	Lint_vec_zero(Set_size, n);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (Adj[i * n + j]) {
				Sets[i][Set_size[i]++] = j;
			}
		}
	}
	
}

void set_of_sets::init(
		int underlying_set_size,
		int nb_sets, long int **Pts, long int *Sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "set_of_sets::init nb_sets=" << nb_sets
				<< " underlying_set_size=" << underlying_set_size << endl;
	}

	init_basic(underlying_set_size, nb_sets, Sz, verbose_level);

	for (i = 0; i < nb_sets; i++) {
		Lint_vec_copy(Pts[i], Sets[i], Sz[i]);
	}
}

void set_of_sets::init_with_Sz_in_int(
		int underlying_set_size,
		int nb_sets, long int **Pts, int *Sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "set_of_sets::init nb_sets=" << nb_sets
				<< " underlying_set_size=" << underlying_set_size << endl;
	}

	long int *Sz1;

	Sz1 = NEW_lint(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sz1[i] = Sz[i];
	}

	init_basic(underlying_set_size, nb_sets, Sz1, verbose_level);

	for (i = 0; i < nb_sets; i++) {
		Lint_vec_copy(Pts[i], Sets[i], Sz[i]);
	}
	FREE_lint(Sz1);
}

void set_of_sets::init_basic(
		int underlying_set_size,
		int nb_sets, long int *Sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "set_of_sets::init_basic nb_sets=" << nb_sets
				<< " underlying_set_size=" << underlying_set_size << endl;
	}
	set_of_sets::nb_sets = nb_sets;
	set_of_sets::underlying_set_size = underlying_set_size;
	Sets = NEW_plint(nb_sets);
	Set_size = NEW_lint(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sets[i] = NULL;
	}
	for (i = 0; i < nb_sets; i++) {
		Set_size[i] = Sz[i];
		if (false /*f_v*/) {
			cout << "set_of_sets::init_basic allocating set " << i
					<< " of size " << Sz[i] << endl;
		}
		Sets[i] = NEW_lint(Sz[i]);
	}
	if (f_v) {
		cout << "set_of_sets::init_basic done" << endl;
	}
}

void set_of_sets::init_basic_with_Sz_in_int(
		int underlying_set_size,
		int nb_sets, int *Sz, int verbose_level)
{
	//int f_v = (verbose_level >= 1);

	long int *Sz1;
	int i;

	Sz1 = NEW_lint(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sz1[i] = Sz[i];
	}

	init_basic(underlying_set_size, nb_sets, Sz1, verbose_level);

	FREE_lint(Sz1);
}

void set_of_sets::init_basic_constant_size(
	int underlying_set_size,
	int nb_sets, int constant_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "set_of_sets::init_basic_constant_size "
				"nb_sets=" << nb_sets
				<< " underlying_set_size=" << underlying_set_size << endl;
	}
	set_of_sets::nb_sets = nb_sets;
	set_of_sets::underlying_set_size = underlying_set_size;
	Sets = NEW_plint(nb_sets);
	Set_size = NEW_lint(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sets[i] = NULL;
	}
	for (i = 0; i < nb_sets; i++) {
		Set_size[i] = constant_size;
		if (false /*f_v*/) {
			cout << "set_of_sets::init_basic_constant_size "
					"allocating set " << i << " of size "
					<< constant_size << endl;
		}
		Sets[i] = NEW_lint(constant_size);
	}
}

//#define MY_BUFSIZE ONE_MILLION

void set_of_sets::init_from_file(
		int &underlying_set_size,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string_tools ST;
	
	if (f_v) {
		cout << "set_of_sets::init_from_file fname=" << fname << endl;
	}
	if (ST.is_csv_file(fname.c_str())) {
		if (f_v) {
			cout << "set_of_sets::init_from_file "
					"the file is a csv file" << endl;
		}
		init_from_csv_file(underlying_set_size, fname, verbose_level);
	}
	else if (ST.is_inc_file(fname.c_str())) {
		if (f_v) {
			cout << "set_of_sets::init_from_file "
					"the file is an inc file" << endl;
		}
		orbiter_kernel_system::file_io Fio;
		int m, n, nb_flags;
		int h, f;

		std::vector<std::vector<int> > Geos;

		Fio.read_incidence_file(Geos, m, n, nb_flags, fname, verbose_level);
		if (f_v) {
			cout << "set_of_sets::init_from_file "
					"the file contains " << Geos.size()
					<< " incidence geometries" << endl;
			cout << "set_of_sets::init_from_file "
					"m=" << m << " n=" << n << " nb_flags=" << nb_flags << endl;
		}

		underlying_set_size = m * n;

		init_basic_constant_size(underlying_set_size,
				Geos.size() /* nb_sets */,
				nb_flags/* constant_size */,
				0 /* verbose_level */);

		for (h = 0; h < Geos.size(); h++) {
			for (f = 0; f < nb_flags; f++) {
				Sets[h][f] = Geos[h][f];
			}
		}

	}
	else {
		if (f_v) {
			cout << "set_of_sets::init_from_file "
					"assuming the file is an orbiter file" << endl;
		}
		init_from_orbiter_file(underlying_set_size, fname, verbose_level);
	}
	if (f_v) {
		cout << "set_of_sets::init_from_file done" << endl;
	}
}

void set_of_sets::init_from_csv_file(
		int underlying_set_size,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "set_of_sets::init_from_csv_file fname=" << fname << endl;
		}

	long int *Data;
	int m, n;
	orbiter_kernel_system::file_io Fio;

	Fio.Csv_file_support->lint_matrix_read_csv(
			fname, Data, m, n, verbose_level);

	if (f_v) {
		cout << "set_of_sets::init_from_csv_file "
				"m=" << m << " n=" << n << endl;
		}

	init_basic_constant_size(underlying_set_size, 
		m /* nb_sets */, 
		n /* constant_size */, 
		0 /* verbose_level */);

	for (i = 0; i < m; i++) {
		Lint_vec_copy(Data + i * n, Sets[i], n);
		}

	
	FREE_lint(Data);
	if (f_v) {
		cout << "set_of_sets::init_from_csv_file done" << endl;
		}
}

void set_of_sets::init_from_orbiter_file(
		int underlying_set_size,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "set_of_sets::init_from_orbiter_file "
				"fname=" << fname << endl;
		}
	nb_sets = Fio.count_number_of_orbits_in_file(fname, verbose_level - 1);
	if (f_v) {
		cout << "set_of_sets::init_from_orbiter_file "
				"nb_sets=" << nb_sets << endl;
		}
	set_of_sets::underlying_set_size = underlying_set_size;
	Sets = NEW_plint(nb_sets);
	Set_size = NEW_lint(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sets[i] = NULL;
		}
	Lint_vec_zero(Set_size, nb_sets);

	char *buf, *p_buf;
	int sz;

	sz = Fio.file_size(fname);

	buf = NEW_char(sz + 1);

	{
	ifstream fp(fname);
	int len, nb_sol, a, j;
	string_tools ST;

	
	nb_sol = 0;
	while (true) {
		if (fp.eof()) {
			break;
			}
		
		//cout << "set_of_sets::init_from_orbiter_file "
		//"reading line, nb_sol = " << nb_sol << endl;
		fp.getline(buf, sz + 1, '\n');
		if (strlen(buf) == 0) {
			cout << "set_of_sets::init_from_orbiter_file "
					"reading an empty line" << endl;
			break;
			}
		
		// check for comment line:
		if (buf[0] == '#')
			continue;
			
		p_buf = buf;
		ST.s_scan_int(&p_buf, &len);
		if (len == -1) {
			if (f_v) {
				cout << "set_of_sets::init_from_orbiter_file "
						"found a complete file with " << nb_sol
						<< " solutions" << endl;
				}
			break;
			}
		else {
			if (f_v) {
				cout << "set_of_sets::init_from_orbiter_file "
						"reading a set of size " << len << endl;
				}
			}
		Sets[nb_sol] = NEW_lint(len);
		Set_size[nb_sol] = len;
		for (j = 0; j < len; j++) {
			ST.s_scan_int(&p_buf, &a);
			Sets[nb_sol][j] = a;
			}
		nb_sol++;
		}
	if (nb_sol != nb_sets) {
		cout << "set_of_sets::init_from_orbiter_file "
				"nb_sol != nb_sets" << endl;
		exit(1);
		}
	}
	FREE_char(buf);
	
	if (f_v) {
		cout << "set_of_sets::init_from_orbiter_file "
				"done" << endl;
		}
}

void set_of_sets::init_set(
		int idx_of_set,
		int *set, int sz, int verbose_level)
// Stores a copy of the given set.
{
	int f_v = (verbose_level >= 1);
	int j;
	
	if (f_v) {
		cout << "set_of_sets::init_set" << endl;
	}
	if (Sets[idx_of_set]) {
		cout << "set_of_sets::init_set Sets[idx_of_set] "
				"is allocated" << endl;
		exit(1);
	}
	Sets[idx_of_set] = NEW_lint(sz);
	Set_size[idx_of_set] = sz;
	for (j = 0; j < sz; j++) {
		Sets[idx_of_set][j] = set[j];
	}
	
	if (f_v) {
		cout << "set_of_sets::init_set done" << endl;
	}
}

void set_of_sets::init_cycle_structure(
		int *perm,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "set_of_sets::init_cycle_structure" << endl;
	}
	int *have_seen = NULL;
	long int *orbit_length = NULL;
	long int *orbit_length2 = NULL;
	int nb_orbits = 0;
	int nb_orbits2 = 0;
	int i, l, l1, first, next, len, c;
	
	if (f_v) {
		cout << "set_of_sets::init_cycle_structure n=" << n << endl;
	}
	orbit_length = NEW_lint(n);
	orbit_length2 = NEW_lint(n);
	have_seen = NEW_int(n);
	Int_vec_zero(have_seen, n);

	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
		}
		// work on a next cycle, starting at position l:
		first = l;
		//cout << "set_of_sets::init_cycle_structure cyle
		//starting with " << first << endl;
		l1 = l;
		len = 1;
		while (true) {
			if (l1 >= n) {
				cout << "set_of_sets::init_cycle_structure cyle "
						"starting with " << first << endl;
				cout << "l1 = " << l1 << " >= n" << endl;
				exit(1);
			}
			have_seen[l1] = true;
			next = perm[l1];
			if (next >= n) {
				cout << "set_of_sets::init_cycle_structure next = "
						<< next << " >= n = " << n << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "set_of_sets::init_cycle_structure "
						"have_seen[next]" << endl;
				cout << "first=" << first << endl;
				cout << "len=" << len << endl;
				cout << "l1=" << l1 << endl;
				cout << "next=" << next << endl;
				for (i = 0; i < n; i++) {
					cout << i << " : " << perm[i] << endl;
				}
				exit(1);
			}
			l1 = next;
			len++;
		}
		//cout << "set_of_sets::init_cycle_structure cyle starting
		//with " << first << " has length " << len << endl;
		//cout << "nb_orbits=" << nb_orbits << endl;
		orbit_length[nb_orbits++] = len;
#if 0
		// print cycle, beginning with first: 
		l1 = first;
		ost << "(";
		while (true) {
			ost << l1 + offset;
			next = a[l1];
			if (next == first) {
				break;
			}
			ost << ", ";
			l1 = next;
		}
		ost << ")"; //  << endl;
		//cout << "set_of_sets::init_cycle_structure
		//done printing cycle" << endl;
#endif
	}
	if (f_v) {
		cout << "set_of_sets::init_cycle_structure we found "
				"the following cycle structure:";
		Lint_vec_print(cout, orbit_length, nb_orbits);
		cout << endl;
	}

	init_basic(n /* underlying_set_size */,
			nb_orbits, orbit_length, 0 /* verbose_level */);

	Int_vec_zero(have_seen, n);

	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
		}
		// work on the next cycle, starting at position l:
		first = l;
		//cout << "set_of_sets::init_cycle_structure cyle
		//starting with " << first << endl;
		l1 = l;
		len = 1;
		while (true) {
			if (l1 >= n) {
				cout << "set_of_sets::init_cycle_structure cyle "
						"starting with " << first << endl;
				cout << "l1 = " << l1 << " >= n" << endl;
				exit(1);
			}
			have_seen[l1] = true;
			next = perm[l1];
			if (next >= n) {
				cout << "set_of_sets::init_cycle_structure next = "
						<< next << " >= n = " << n << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "set_of_sets::init_cycle_structure "
						"have_seen[next]" << endl;
				cout << "first=" << first << endl;
				cout << "len=" << len << endl;
				cout << "l1=" << l1 << endl;
				cout << "next=" << next << endl;
				for (i = 0; i < n; i++) {
					cout << i << " : " << perm[i] << endl;
				}
				exit(1);
			}
			l1 = next;
			len++;
		}
		//cout << "set_of_sets::init_cycle_structure cycle starting
		//with " << first << " has length " << len << endl;
		//cout << "nb_orbits=" << nb_orbits << endl;
		orbit_length2[nb_orbits2] = len;
		if (orbit_length2[nb_orbits2] != orbit_length[nb_orbits2]) {
			cout << "set_of_sets::init_cycle_structure "
					"orbit_length2[nb_orbits2] != "
					"orbit_length[nb_orbits2]" << endl;
			exit(1);
		}

		// get cycle, beginning with first: 
		l1 = first;
		c = 0;
		while (true) {
			Sets[nb_orbits2][c++] = l1;
			next = perm[l1];
			if (next == first) {
				break;
			}
			l1 = next;
		}
		if (c != orbit_length2[nb_orbits2]) {
			cout << "set_of_sets::init_cycle_structure c != "
					"orbit_length2[nb_orbits2]" << endl;
			exit(1);
		}
		//cout << "set_of_sets::init_cycle_structure
		//done with cycle" << endl;
		nb_orbits2++;
	}
	if (nb_orbits2 != nb_orbits) {
		cout << "set_of_sets::init_cycle_structure "
				"nb_orbits2 != nb_orbits" << endl;
		exit(1);
	}

	FREE_int(have_seen);
	FREE_lint(orbit_length);
	FREE_lint(orbit_length2);
	if (f_v) {
		cout << "set_of_sets::init_cycle_structure done" << endl;
	}
}

int set_of_sets::total_size()
{
	int sz, i;

	sz = 0;
	for (i = 0; i < nb_sets; i++) {
		sz += Set_size[i];
		}
	return sz;
}

long int &set_of_sets::element(int i, int j)
{
	return Sets[i][j];
}

void set_of_sets::add_element(int i, long int a)
{
	Sets[i][Set_size[i]++] = a;
}

void set_of_sets::print()
{
	int i;
	
	cout << "(";
	for (i = 0; i < nb_sets; i++) {
		Lint_vec_print(cout, Sets[i], Set_size[i]);
		if (i < nb_sets - 1) {
			cout << ", ";
		}
	}
	cout << ")" << endl;
}

void set_of_sets::print_table()
{
	int i;
	
	cout << "set of sets with " << nb_sets << " sets :" << endl;
	for (i = 0; i < nb_sets; i++) {
		cout << "set " << i << " has size " << Set_size[i] << " : ";
		Lint_vec_print(cout, Sets[i], Set_size[i]);
		cout << endl;
	}
	cout << "end set of sets" << endl;
}

void set_of_sets::print_table_tex(
		std::ostream &ost)
{
	l1_interfaces::latex_interface L;
	int i;
	
	//cout << "set of sets with " << nb_sets << " sets :" << endl;
	for (i = 0; i < nb_sets; i++) {
		ost << "Set " << i << " has size " << Set_size[i] << " : ";
		L.lint_set_print_tex(ost, Sets[i], Set_size[i]);
		ost << "\\\\" << endl;
	}
	//cout << "end set of sets" << endl;
}

void set_of_sets::print_table_latex_simple(
		std::ostream &ost)
{
	l1_interfaces::latex_interface L;
	int i;

	//cout << "set of sets with " << nb_sets << " sets :" << endl;
	ost << "\\noindent ";
	for (i = 0; i < nb_sets; i++) {
		//ost << "Set " << i << " has size " << Set_size[i] << " : ";
		L.lint_set_print_tex_text_mode(ost, Sets[i], Set_size[i]);
		//L.lint_set_print_tex(ost, Sets[i], Set_size[i]);
		ost << "\\\\" << endl;
	}
	//cout << "end set of sets" << endl;
}


void set_of_sets::print_table_latex_simple_with_selection(
		std::ostream &ost, int *Selection, int nb_sel)
{
	l1_interfaces::latex_interface L;
	int i, h;

	//cout << "set of sets with " << nb_sets << " sets :" << endl;
	ost << "\\noindent ";
	for (h = 0; h < nb_sel; h++) {
		i = Selection[h];
		//ost << "Set " << i << " has size " << Set_size[i] << " : ";
		L.lint_set_print_tex_text_mode(ost, Sets[i], Set_size[i]);
		//L.lint_set_print_tex(ost, Sets[i], Set_size[i]);
		ost << "\\\\" << endl;

	}
	//cout << "end set of sets" << endl;
}


void set_of_sets::dualize(
		set_of_sets *&S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, j;

	if (f_v) {
		cout << "set_of_sets::dualize" << endl;
	}
	S = NEW_OBJECT(set_of_sets);
	S->init_basic_constant_size(nb_sets,
			underlying_set_size, nb_sets, verbose_level - 1);
	Lint_vec_zero(S->Set_size, underlying_set_size);
	for (i = 0; i < nb_sets; i++) {
		for (j = 0; j < Set_size[i]; j++) {
			a = Sets[i][j];
			S->add_element(a, i);
		}
	}
	

	if (f_v) {
		cout << "set_of_sets::dualize done" << endl;
	}
}

void set_of_sets::remove_sets_of_given_size(
		int k,
		set_of_sets &S, int *&Idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, l, a, j;

	if (f_v) {
		cout << "set_of_sets::remove_sets_of_given_size" << endl;
	}
	l = 0;
	for (i = 0; i < nb_sets; i++) {
		if (Set_size[i] != k) {
			l++;
		}
	}
	Idx = NEW_int(l);
	S.init_simple(underlying_set_size, l, verbose_level - 1);
	a = 0;
	for (i = 0; i < nb_sets; i++) {
		if (Set_size[i] != k) {
			S.Sets[a] = NEW_lint(Set_size[i]);
			S.Set_size[a] = Set_size[i];
			for (j = 0; j < Set_size[i]; j++) {
				S.Sets[a][j] = Sets[i][j];
			}
			Idx[a] = i;
			a++;
		}
	}
	if (a != l) {
		cout << "set_of_sets::remove_sets_of_given_size "
				"a != l" << endl;
	}
	
}

void set_of_sets::extract_largest_sets(
		set_of_sets &S, int *&Idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	tally C;
	int f_second = false;
	int f, m, nb_big_sets, i, ii, j;

	if (f_v) {
		cout << "set_of_sets::extract_largest_sets" << endl;
	}
	C.init_lint(Set_size, nb_sets, f_second, 0);
	if (f_v) {
		cout << "set_of_sets::extract_largest_sets set sizes: ";
		C.print(false /* f_backwards*/);
	}
	f = C.type_first[C.nb_types - 1];
	m = C.data_sorted[f];
	nb_big_sets = C.type_len[C.nb_types - 1];
	
	Idx = NEW_int(nb_big_sets);
	S.init_simple(underlying_set_size, nb_big_sets, verbose_level);
	for (i = 0; i < nb_big_sets; i++) {
		ii = C.sorting_perm_inv[f + i];
		Idx[i] = ii;
		S.Sets[i] = NEW_lint(m);
		S.Set_size[i] = m;
		for (j = 0; j < m; j++) {
			S.Sets[i][j] = Sets[ii][j];
		}
	}
	
}

void set_of_sets::intersection_matrix(
	int *&intersection_type, int &highest_intersection_number, 
	int *&intersection_matrix, int &nb_big_sets, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	tally C;
	int f_second = false;
	int f, l, a, i, ii, u, j;

	if (f_v) {
		cout << "set_of_sets::intersection_matrix" << endl;
	}
	C.init_lint(Set_size, nb_sets, f_second, 0);
	if (false /*f_v*/) {
		cout << "set_of_sets::intersection_matrix "
				"plane-intersection type: ";
		C.print(false /* f_backwards*/);
	}

	if (f_v) {
		cout << "The intersection type is (";
		C.print_bare(false /* f_backwards*/);
		cout << ")" << endl << endl;
	}
	f = C.type_first[C.nb_types - 1];
	highest_intersection_number = C.data_sorted[f];
	intersection_type = NEW_int(highest_intersection_number + 1);
	for (i = 0; i <= highest_intersection_number; i++) {
		intersection_type[i] = 0;
	}
	
	for (i = 0; i < C.nb_types; i++) {
		f = C.type_first[i];
		l = C.type_len[i];
		a = C.data_sorted[f];
		intersection_type[a] = l;
	}
	f = C.type_first[C.nb_types - 1];
	nb_big_sets = C.type_len[C.nb_types - 1];

	int *Incma, *Incma_t, *IIt, *ItI;
	
	Incma = NEW_int(underlying_set_size * nb_big_sets);
	Incma_t = NEW_int(nb_big_sets * underlying_set_size);
	IIt = NEW_int(underlying_set_size * underlying_set_size);
	ItI = NEW_int(nb_big_sets * nb_big_sets);


	for (i = 0; i < underlying_set_size * nb_big_sets; i++) {
		Incma[i] = 0;
	}
	for (i = 0; i < nb_big_sets; i++) {
		ii = C.sorting_perm_inv[f + i];
		for (j = 0; j < Set_size[ii]; j++) {
			a = Sets[ii][j];
			Incma[a * nb_big_sets + i] = 1;
		}
	}
	if (false /*f_vv*/) {
		cout << "Incidence matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, Incma,
				underlying_set_size, nb_big_sets, nb_big_sets, 1);
	}
	for (i = 0; i < underlying_set_size; i++) {
		for (j = 0; j < underlying_set_size; j++) {
			a = 0;
			for (u = 0; u < nb_big_sets; u++) {
				a += Incma[i * nb_big_sets + u] *
						Incma_t[u * underlying_set_size + j];
			}
			IIt[i * underlying_set_size + j] = a;
		}
	}
	if (false /*f_vv*/) {
		cout << "I * I^\\top = " << endl;
		Int_vec_print_integer_matrix_width(cout, IIt,
				underlying_set_size, underlying_set_size,
				underlying_set_size, 2);
	}
	for (i = 0; i < nb_big_sets; i++) {
		for (j = 0; j < nb_big_sets; j++) {
			a = 0;
			for (u = 0; u < underlying_set_size; u++) {
				a += Incma[u * nb_big_sets + i] *
						Incma[u * nb_big_sets + j];
			}
			ItI[i * nb_big_sets + j] = a;
		}
	}
	if (false /*f_v*/) {
		cout << "I^\\top * I = " << endl;
		Int_vec_print_integer_matrix_width(cout, ItI,
				nb_big_sets, nb_big_sets, nb_big_sets, 3);
	}
	
	intersection_matrix = NEW_int(nb_big_sets * nb_big_sets);
	for (i = 0; i < nb_big_sets; i++) {
		for (j = 0; j < nb_big_sets; j++) {
			intersection_matrix[i * nb_big_sets + j] =
					ItI[i * nb_big_sets + j];
		}
	}

	FREE_int(Incma);
	FREE_int(Incma_t);
	FREE_int(IIt);
	FREE_int(ItI);
	if (f_v) {
		cout << "set_of_sets::intersection_matrix done" << endl;
	}
}

void set_of_sets::compute_incidence_matrix(
		int *&Inc, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;

	if (f_v) {
		cout << "set_of_sets::compute_and_print_tdo_row_scheme" << endl;
	}
	m = underlying_set_size;
	n = nb_sets;
	Inc = NEW_int(underlying_set_size * nb_sets);
	Int_vec_zero(Inc, m * n);
	for (j = 0; j < nb_sets; j++) {
		for (h = 0; h < Set_size[j]; h++) {
			i = Sets[j][h];
			Inc[i * nb_sets + j] = 1;
		}
	}
}

#if 0
void set_of_sets::compute_and_print_tdo_row_scheme(
		std::ostream &file, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Inc;
	geometry::incidence_structure *I;
	partitionstack *Stack;
	int depth = INT_MAX;
	//int i, j, a;
	int m, n;
		
	if (f_v) {
		cout << "set_of_sets::compute_and_print_tdo_row_scheme" << endl;
	}

	compute_incidence_matrix(Inc, m, n, verbose_level - 2);

#if 0
	Inc = NEW_int(underlying_set_size * nb_sets);
	for (i = 0; i < underlying_set_size * nb_sets; i++) {
		Inc[i] = 0;
		}
	for (i = 0; i < nb_sets; i++) {
		for (j = 0; j < Set_size[i]; j++) {
			a = Sets[i][j];
			Inc[a * nb_sets + i] = 1;
			}
		}
#endif


	int set_size = underlying_set_size;
	int nb_blocks = nb_sets;

	I = NEW_OBJECT(geometry::incidence_structure);
	I->init_by_matrix(set_size, nb_blocks, Inc, 0 /* verbose_level */);
	Stack = NEW_OBJECT(partitionstack);
	Stack->allocate(set_size + nb_blocks, 0 /* verbose_level */);
	Stack->subset_contiguous(set_size, nb_blocks);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();

	I->compute_TDO_safe(*Stack, depth, verbose_level - 2);
		
	I->get_and_print_row_tactical_decomposition_scheme_tex(
		file, false /* f_enter_math */,
		false /* f_print_subscripts */, *Stack);

	FREE_int(Inc);
	FREE_OBJECT(I);
	FREE_OBJECT(Stack);
	if (f_v) {
		cout << "set_of_sets::compute_and_print_tdo_row_scheme done" << endl;
	}
}

void set_of_sets::compute_and_print_tdo_col_scheme(
		std::ostream &file, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Inc;
	int m, n;
	geometry::incidence_structure *I;
	partitionstack *Stack;
	int depth = INT_MAX;
	//int i, j, a;
		
	if (f_v) {
		cout << "set_of_sets::compute_and_print_tdo_col_scheme" << endl;
	}


	compute_incidence_matrix(Inc, m, n, verbose_level - 2);

#if 0
	Inc = NEW_int(underlying_set_size * nb_sets);
	for (i = 0; i < underlying_set_size * nb_sets; i++) {
		Inc[i] = 0;
	}
	for (i = 0; i < nb_sets; i++) {
		for (j = 0; j < Set_size[i]; j++) {
			a = Sets[i][j];
			Inc[a * nb_sets + i] = 1;
		}
	}
#endif

	int set_size = underlying_set_size;
	int nb_blocks = nb_sets;

	I = NEW_OBJECT(geometry::incidence_structure);
	I->init_by_matrix(set_size, nb_blocks, Inc, 0 /* verbose_level */);
	Stack = NEW_OBJECT(partitionstack);
	Stack->allocate(set_size + nb_blocks, 0 /* verbose_level */);
	Stack->subset_contiguous(set_size, nb_blocks);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();

	I->compute_TDO_safe(*Stack, depth, verbose_level - 2);
		
	I->get_and_print_column_tactical_decomposition_scheme_tex(
		file, false /* f_enter_math */,
		false /* f_print_subscripts */, *Stack);

	FREE_int(Inc);
	FREE_OBJECT(I);
	FREE_OBJECT(Stack);
	if (f_v) {
		cout << "set_of_sets::compute_and_print_tdo_col_scheme done" << endl;
	}
}
#endif

void set_of_sets::init_decomposition(
		geometry::decomposition *&D, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Inc;
	int m, n;

	if (f_v) {
		cout << "set_of_sets::init_decomposition" << endl;
	}
	compute_incidence_matrix(Inc, m, n, verbose_level - 2);

	D = NEW_OBJECT(geometry::decomposition);

	D->init_incidence_matrix(underlying_set_size,
			nb_sets, Inc, verbose_level - 1);

	FREE_int(Inc);

	if (f_v) {
		cout << "set_of_sets::init_decomposition done" << endl;
	}
}

void set_of_sets::compute_tdo_decomposition(
		geometry::decomposition &D, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Inc;
	int m, n;
	//incidence_structure *I;
	//partitionstack *Stack;
	//int depth = INT_MAX;
		
	if (f_v) {
		cout << "set_of_sets::compute_tdo_decomposition" << endl;
	}

	compute_incidence_matrix(Inc, m, n, verbose_level - 2);

	if (f_v) {
		cout << "set_of_sets::compute_tdo_decomposition "
				"after compute_incidence_matrix" << endl;
		cout << "underlying_set_size=" << underlying_set_size << endl;
		cout << "nb_sets=" << nb_sets << endl;
	}

	if (f_v) {
		Int_matrix_print(Inc, underlying_set_size, nb_sets);
	}


	if (f_v) {
		cout << "set_of_sets::compute_tdo_decomposition "
				"before D.init_incidence_matrix" << endl;
	}
	D.init_incidence_matrix(underlying_set_size,
			nb_sets, Inc, verbose_level - 1);
	FREE_int(Inc);


	if (f_v) {
		cout << "set_of_sets::compute_tdo_decomposition "
				"before D.setup_default_partition" << endl;
	}
	D.setup_default_partition(verbose_level);

	if (f_v) {
		cout << "set_of_sets::compute_tdo_decomposition "
				"before D.compute_TDO" << endl;
	}
	D.compute_TDO(INT_MAX, verbose_level);

	if (f_v) {
		cout << "set_of_sets::compute_tdo_scheme done" << endl;
	}
}

int set_of_sets::is_member(
		int i, int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret, idx;
	sorting Sorting;
	
	if (f_v) {
		cout << "set_of_sets::is_member" << endl;
	}
	ret = Sorting.lint_vec_search(Sets[i], Set_size[i], a, idx, 0);
	if (f_v) {
		cout << "set_of_sets::is_member done" << endl;
	}
	return ret;
}

void set_of_sets::sort_all(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	sorting Sorting;
	
	if (f_v) {
		cout << "set_of_sets::sort_all" << endl;
	}
	for (i = 0; i < nb_sets; i++) {
		Sorting.lint_vec_heapsort(Sets[i], Set_size[i]);
	}

	if (f_v) {
		cout << "set_of_sets::sort_all done" << endl;
	}
}

void set_of_sets::all_pairwise_intersections(
		set_of_sets *&Intersections, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int N, i, j, k;
	long int *v;
	int l;
	combinatorics::combinatorics_domain Combi;
	sorting Sorting;
	
	if (f_v) {
		cout << "set_of_sets::all_pairwise_intersections" << endl;
	}
	N = Combi.int_n_choose_k(nb_sets, 2);


	Intersections = NEW_OBJECT(set_of_sets);
	Intersections->init_simple(underlying_set_size,
			N, verbose_level - 1);

	v = NEW_lint(underlying_set_size);
	for (i = 0; i < nb_sets; i++) {
		for (j = i + 1; j < nb_sets; j++) {
			k = Combi.ij2k(i, j, nb_sets);
			Sorting.lint_vec_intersect_sorted_vectors(
					Sets[i], Set_size[i], Sets[j], Set_size[j], v, l);
			Intersections->Sets[k] = NEW_lint(l);
			Lint_vec_copy(v, Intersections->Sets[k], l);
			Intersections->Set_size[k] = l;
		}
	}

	FREE_lint(v);
	
	if (f_v) {
		cout << "set_of_sets::all_pairwise_intersections done" << endl;
	}
}

void set_of_sets::pairwise_intersection_matrix(
		int *&M, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int *v;
	int l;
	sorting Sorting;
	
	if (f_v) {
		cout << "set_of_sets::pairwise_intersection_matrix" << endl;
	}


	M = NEW_int(nb_sets * nb_sets);
	Int_vec_zero(M, nb_sets * nb_sets);

	v = NEW_lint(underlying_set_size);
	for (i = 0; i < nb_sets; i++) {
		for (j = i + 1; j < nb_sets; j++) {
			Sorting.lint_vec_intersect_sorted_vectors(Sets[i],
					Set_size[i], Sets[j], Set_size[j], v, l);
			M[i * nb_sets + j] = l;
			M[j * nb_sets + i] = l;
		}
	}

	FREE_lint(v);
	
	if (f_v) {
		cout << "set_of_sets::all_pairwise_intersections done" << endl;
	}
}

void set_of_sets::all_triple_intersections(
		set_of_sets *&Intersections, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N, i, j, k, h;
	long int *v;
	long int *w;
	int l1, l2;
	combinatorics::combinatorics_domain Combi;
	sorting Sorting;
	
	if (f_v) {
		cout << "set_of_sets::all_triple_intersections" << endl;
	}
	N = Combi.int_n_choose_k(nb_sets, 3);


	Intersections = NEW_OBJECT(set_of_sets);
	Intersections->init_simple(underlying_set_size, N, verbose_level - 1);

	v = NEW_lint(underlying_set_size);
	w = NEW_lint(underlying_set_size);
	for (i = 0; i < nb_sets; i++) {
		for (j = i + 1; j < nb_sets; j++) {

			Sorting.lint_vec_intersect_sorted_vectors(Sets[i],
					Set_size[i], Sets[j], Set_size[j], v, l1);

			for (k = j + 1; k < nb_sets; k++) {
			
				h = Combi.ijk2h(i, j, k, nb_sets);
				Sorting.lint_vec_intersect_sorted_vectors(v, l1,
						Sets[k], Set_size[k], w, l2);
				Intersections->Sets[h] = NEW_lint(l2);
				Lint_vec_copy(w, Intersections->Sets[h], l2);
				Intersections->Set_size[h] = l2;
			}
		}
	}

	FREE_lint(v);
	FREE_lint(w);
	
	if (f_v) {
		cout << "set_of_sets::all_triple_intersections done" << endl;
	}
}

int set_of_sets::has_constant_size_property()
{
	int s, i;

	if (nb_sets == 0) {
		cout << "set_of_sets::has_constant_size_property no sets" << endl;
		exit(1);
	}
	s = Set_size[0];
	for (i = 1; i < nb_sets; i++) {
		if (Set_size[i] != s) {
			return false;
		}
	}
	return true;
}

int set_of_sets::get_constant_size()
{
	int s, i;

	if (nb_sets == 0) {
		cout << "set_of_sets::get_constant_size no sets" << endl;
		exit(1);
	}
	s = Set_size[0];
	for (i = 1; i < nb_sets; i++) {
		if (Set_size[i] != s) {
			cout << "set_of_sets::get_constant_size "
					"the size of the sets is not constant" << endl;
			exit(1);
		}
	}
	return s;
}


int set_of_sets::largest_set_size()
{
	int s = INT_MIN;
	int i;
	
	for (i = 0; i < nb_sets; i++) {
		s = MAXIMUM(s, Set_size[i]);
	}
	return s;
}

void set_of_sets::save_csv(
		std::string &fname,
		int f_make_heading, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	spreadsheet *Sp;

	if (f_v) {
		cout << "set_of_sets::save_csv" << endl;
	}
	Sp = NEW_OBJECT(spreadsheet);
	Sp->init_set_of_sets(this, f_make_heading);
	Sp->save(fname, verbose_level);
	if (f_v) {
		cout << "set_of_sets::save_csv "
				"before delete spreadsheet" << endl;
	}
	//FREE_OBJECT(Sp); // ToDo
	if (f_v) {
		cout << "set_of_sets::save_csv done" << endl;
	}
}

void set_of_sets::save_constant_size_csv(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_cols;
	long int *M;
	int i, j;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "set_of_sets::save_constant_size_csv" << endl;
	}
	if (!has_constant_size_property()) {
		cout << "set_of_sets::save_constant_size_csv "
				"!has_constant_size_property" << endl;
		exit(1);
	}
	nb_cols = Set_size[0];
	M = NEW_lint(nb_sets * nb_cols);
	for (i = 0; i < nb_sets; i++) {
		for (j = 0; j < nb_cols; j++) {
			M[i * nb_cols + j] = Sets[i][j];
		}
	}
	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, M, nb_sets, nb_cols);
	FREE_lint(M);
	if (f_v) {
		cout << "set_of_sets::save_constant_size_csv done" << endl;
	}
}

int set_of_sets::find_common_element_in_two_sets(
		int idx1, int idx2, int &common_elt)
{
	int pos1, pos2;
	sorting Sorting;
	
	if (Sorting.lint_vecs_find_common_element(Sets[idx1],
			Set_size[idx1], Sets[idx2], Set_size[idx2], pos1, pos2)) {
		common_elt = Sets[idx1][pos1];
		return true;
	}
	return false;
	
}

void set_of_sets::sort()
{
	int i;
	sorting Sorting;
	
	for (i = 0; i < nb_sets; i++) {
		Sorting.lint_vec_heapsort(Sets[i], Set_size[i]);
	}
}

void set_of_sets::sort_big(int verbose_level)
{
	sorting Sorting;

	Sorting.Heapsort_general(this, nb_sets,
		set_of_sets_compare_func, 
		set_of_sets_swap_func, NULL);
}

void set_of_sets::compute_orbits(
		int &nb_orbits,
	int *&orbit, int *&orbit_inv,
	int *&orbit_first, int *&orbit_len, 
	void (*compute_image_function)(set_of_sets *S,
			void *compute_image_data, int elt_idx,
			int gen_idx, int &idx_of_image, int verbose_level),
	void *compute_image_data, 
	int nb_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, cur, a, b, g, t, l, pos, x;

	if (f_v) {
		cout << "set_of_sets::compute_orbits" << endl;
	}
	orbit = NEW_int(nb_sets);
	orbit_inv = NEW_int(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		orbit[i] = i;
		orbit_inv[i] = i;
	}
	orbit_first = NEW_int(nb_sets);
	orbit_len = NEW_int(nb_sets);
	nb_orbits = 0;
	cur = 0;
	while (cur < nb_sets) {
		l = cur + 1;
		orbit_first[nb_orbits] = cur;
		orbit_len[nb_orbits] = 1;
		if (f_v) {
			cout << "set_of_sets::compute_orbits "
					"New orbit " << nb_orbits
					<< " is orbit of " << orbit[cur] << endl;
		}
		while (cur < l) {
			a = orbit[cur];
			for (g = 0; g < nb_gens; g++) {
				(*compute_image_function)(this,
						compute_image_data, a, g, b, verbose_level - 2);
				if (f_vv) {
					cout << a << " -" << g << "-> " << b << endl;
				}
				pos = orbit_inv[b];
				if (pos >= l) {
					if (pos > l) {
						t = orbit[pos];
						for (i = pos; i > l; i--) {
							x = orbit[i - 1];
							orbit[i] = x;
							orbit_inv[x] = i;
						}
						orbit[l] = t;
						orbit_inv[t] = l;

						//t = orbit[l];
						//orbit[l] = b;
						//orbit[pos] = t;
						//orbit_inv[b] = l;
						//orbit_inv[t] = pos;
					}
					orbit_len[nb_orbits]++;
					l++;
				}
			}
			cur++;
		}
		nb_orbits++;
	}
	if (f_v) {
		cout << "set_of_sets::compute_orbits "
				"we found " << nb_orbits << " orbits" << endl;
	}

	if (f_v) {
		cout << "set_of_sets::compute_orbits done" << endl;
	}
}

int set_of_sets::number_of_eckardt_points(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *E;
	int nb_E;

	if (f_v) {
		cout << "set_of_sets::number_of_eckardt_points" << endl;
	}
	get_eckardt_points(E, nb_E, verbose_level);
	FREE_int(E);
	if (f_v) {
		cout << "set_of_sets::number_of_eckardt_points done" << endl;
	}
	return nb_E;
}

void set_of_sets::get_eckardt_points(
		int *&E, int &nb_E, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_of_sets::get_eckardt_points" << endl;
	}

	geometry::incidence_structure *Inc;

	Inc = NEW_OBJECT(geometry::incidence_structure);

	Inc->init_by_set_of_sets(this, false);

	geometry::decomposition *Decomposition;

	Decomposition = NEW_OBJECT(geometry::decomposition);

	Decomposition->init_incidence_structure(
			Inc,
			verbose_level - 1);

#if 0
	partitionstack *PStack;

	PStack = NEW_OBJECT(partitionstack);
	PStack->allocate(nb_sets + underlying_set_size, 0 /* verbose_level */);
	PStack->subset_contiguous(nb_sets, underlying_set_size);
	PStack->split_cell(0 /* verbose_level */);
#endif
	
	Decomposition->compute_TDO_safe(
			1 /*nb_lines + nb_points_on_surface*/ /* depth */,
			0 /* verbose_level */);

#if 0
	{
	IS->get_and_print_row_tactical_decomposition_scheme_tex(
		cout /*fp_row_scheme */, false /* f_enter_math */, *PStack);
	IS->get_and_print_column_tactical_decomposition_scheme_tex(
		cout /*fp_col_scheme */, false /* f_enter_math */, *PStack);
	}
#endif
	int *row_classes, *row_class_inv, nb_row_classes;
	int *col_classes, *col_class_inv, nb_col_classes;
	int *col_scheme;

	Decomposition->Stack->allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		0/*verbose_level*/);
	
	col_scheme = NEW_int(nb_row_classes * nb_col_classes);

	Decomposition->get_col_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		col_scheme, 0/*verbose_level*/);

	//cout << *this << endl;
	
	if (f_v) {
		cout << "col_scheme:" << endl;
		Decomposition->Stack->print_decomposition_scheme(cout,
			row_classes, nb_row_classes,
			col_classes, nb_col_classes, 
			col_scheme);
	}

	int i, j, c, s, sz;
	
	nb_E = 0;
	for (j = 0; j < nb_col_classes; j++) {
		c = col_classes[j];
		sz = Decomposition->Stack->cellSize[c];
		s = 0;
		for (i = 0; i < nb_row_classes; i++) {
			s += col_scheme[i * nb_col_classes + j];
		}
		if (s == 3) {
			nb_E += sz;
		}
	}
	if (f_v) {
		cout << "set_of_sets::get_eckardt_points nb_E=" << nb_E << endl;
	}

	int h, f, y;
	
	E = NEW_int(nb_E);
	h = 0;
	for (j = 0; j < nb_col_classes; j++) {
		c = col_classes[j];
		sz = Decomposition->Stack->cellSize[c];
		s = 0;
		for (i = 0; i < nb_row_classes; i++) {
			s += col_scheme[i * nb_col_classes + j];
		}
		if (s == 3) {
			f = Decomposition->Stack->startCell[c];
			for (i = 0; i < sz; i++) {
				y = Decomposition->Stack->pointList[f + i] - nb_sets;
				E[h++] = y;
			}
		}
	}

	FREE_int(row_classes);
	FREE_int(row_class_inv);
	FREE_int(col_classes);
	FREE_int(col_class_inv);
	FREE_int(col_scheme);
	//FREE_OBJECT(PStack);
	FREE_OBJECT(Decomposition);
	FREE_OBJECT(Inc);

	if (f_v) {
		cout << "set_of_sets::get_eckardt_points done" << endl;
	}
}

void set_of_sets::evaluate_function_and_store(
		data_structures::set_of_sets *&Function_values,
		int (*evaluate_function)(int a, int i, int j,
				void *evaluate_data, int verbose_level),
		void *evaluate_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, c;

	if (f_v) {
		cout << "set_of_sets::evaluate_function_and_store nb_sets=" << nb_sets
				<< " underlying_set_size=" << underlying_set_size << endl;
	}
	Function_values = NEW_OBJECT(data_structures::set_of_sets);

	Function_values->init_basic(underlying_set_size,
			nb_sets, Set_size, verbose_level);
	for (i = 0; i < nb_sets; i++) {
		for (j = 0; j < Set_size[i]; j++) {
			a = Sets[i][j];
			c = (*evaluate_function)(a, i, j, evaluate_data, verbose_level - 2);
			Function_values->Sets[i][j] = c;
		}
	}
	if (f_v) {
		cout << "set_of_sets::evaluate_function_and_store done" << endl;
	}
}

int set_of_sets::find_smallest_class()
{
	int i;
	int idx = 0;
	int sz = Set_size[idx];

	for (i = 1; i < nb_sets; i++) {
		if (Set_size[i] < sz) {
			idx = i;
			sz = Set_size[i];
		}
	}
	return idx;
}

// #############################################################################
// global functions:
// #############################################################################


static int set_of_sets_compare_func(
		void *data, int i, int j, void *extra_data)
{
	set_of_sets *S = (set_of_sets *) data;
	sorting Sorting;
	int c;

	if (S->Set_size[i] != S->Set_size[j]) {
		cout << "set_of_sets_compare_func sets "
				"must have the same size" << endl;
		exit(1);
	}
	c = Sorting.lint_vec_compare(S->Sets[i], S->Sets[j], S->Set_size[i]);
	return c;
}

static void set_of_sets_swap_func(
		void *data, int i, int j, void *extra_data)
{
	set_of_sets *S = (set_of_sets *) data;
	long int *p;

	if (S->Set_size[i] != S->Set_size[j]) {
		cout << "set_of_sets_swap_func sets "
				"must have the same size" << endl;
		exit(1);
	}
	p = S->Sets[i];
	S->Sets[i] = S->Sets[j];
	S->Sets[j] = p;
}

}}}

