// set_of_sets.C
//
// Anton Betten
//
// November 30, 2012

#include "foundations.h"


set_of_sets::set_of_sets()
{
	null();
}

set_of_sets::~set_of_sets()
{
	freeself();
}

void set_of_sets::null()
{
	underlying_set_size = 0;
	nb_sets = 0;
	Sets = NULL;
	Set_size = NULL;
}

void set_of_sets::freeself()
{
	INT i;
	
	if (Sets) {
		for (i = 0; i < nb_sets; i++) {
			if (Sets[i]) {
				FREE_INT(Sets[i]);
				}
			}
		FREE_PINT(Sets);
		FREE_INT(Set_size);
		}
	null();
}

set_of_sets *set_of_sets::copy()
{
	set_of_sets *SoS;

	SoS = NEW_OBJECT(set_of_sets);

	SoS->init(underlying_set_size, nb_sets, Sets, Set_size, 0 /*verbose_level */);
	return SoS;
}

void set_of_sets::init_simple(INT underlying_set_size, INT nb_sets, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "set_of_sets::init_simple nb_sets=" << nb_sets << " underlying_set_size=" << underlying_set_size << endl;
		}
	set_of_sets::nb_sets = nb_sets;
	set_of_sets::underlying_set_size = underlying_set_size;
	Sets = NEW_PINT(nb_sets);
	Set_size = NEW_INT(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sets[i] = NULL;
		}
	INT_vec_zero(Set_size, nb_sets);
}

void set_of_sets::init_from_adjacency_matrix(INT n, INT *Adj, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	
	if (f_v) {
		cout << "set_of_sets::init_from_adjacency_matrix n=" << n << endl;
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
		Sets[i] = NEW_INT(Set_size[i]);
		}
	INT_vec_zero(Set_size, n);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (Adj[i * n + j]) {
				Sets[i][Set_size[i]++] = j;
				}
			}
		}
	
}

void set_of_sets::init(INT underlying_set_size, INT nb_sets, INT **Pts, INT *Sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "set_of_sets::init nb_sets=" << nb_sets << " underlying_set_size=" << underlying_set_size << endl;
		}

	init_basic(underlying_set_size, nb_sets, Sz, verbose_level);

	for (i = 0; i < nb_sets; i++) {
		INT_vec_copy(Pts[i], Sets[i], Sz[i]);
		}
}

void set_of_sets::init_basic(INT underlying_set_size, INT nb_sets, INT *Sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "set_of_sets::init_basic nb_sets=" << nb_sets << " underlying_set_size=" << underlying_set_size << endl;
		}
	set_of_sets::nb_sets = nb_sets;
	set_of_sets::underlying_set_size = underlying_set_size;
	Sets = NEW_PINT(nb_sets);
	Set_size = NEW_INT(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sets[i] = NULL;
		}
	for (i = 0; i < nb_sets; i++) {
		Set_size[i] = Sz[i];
		if (FALSE /*f_v*/) {
			cout << "set_of_sets::init_basic allocating set " << i << " of size " << Sz[i] << endl;
			}
		Sets[i] = NEW_INT(Sz[i]);
		}
}

void set_of_sets::init_basic_constant_size(INT underlying_set_size, 
	INT nb_sets, INT constant_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "set_of_sets::init_basic_constant_size nb_sets=" << nb_sets << " underlying_set_size=" << underlying_set_size << endl;
		}
	set_of_sets::nb_sets = nb_sets;
	set_of_sets::underlying_set_size = underlying_set_size;
	Sets = NEW_PINT(nb_sets);
	Set_size = NEW_INT(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sets[i] = NULL;
		}
	for (i = 0; i < nb_sets; i++) {
		Set_size[i] = constant_size;
		if (FALSE /*f_v*/) {
			cout << "set_of_sets::init_basic_constant_size allocating set " << i << " of size " << constant_size << endl;
			}
		Sets[i] = NEW_INT(constant_size);
		}
}

#define MY_BUFSIZE ONE_MILLION

void set_of_sets::init_from_file(INT underlying_set_size, const char *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "set_of_sets::init_from_file fname=" << fname << endl;
		}
	if (is_csv_file(fname)) {
		if (f_v) {
			cout << "set_of_sets::init_from_file the file is a csv file" << endl;
			}
		init_from_csv_file(underlying_set_size, fname, verbose_level);
		}
	else {
		if (f_v) {
			cout << "set_of_sets::init_from_file assuming the file is an orbiter file" << endl;
			}
		init_from_orbiter_file(underlying_set_size, fname, verbose_level);
		}
	if (f_v) {
		cout << "set_of_sets::init_from_file done" << endl;
		}
}

void set_of_sets::init_from_csv_file(INT underlying_set_size, const char *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "set_of_sets::init_from_csv_file fname=" << fname << endl;
		}

	INT *Data;
	INT m, n;
	
	INT_matrix_read_csv(fname, Data, m, n, verbose_level);

	if (f_v) {
		cout << "set_of_sets::init_from_csv_file m=" << m << " n=" << n << endl;
		}

	init_basic_constant_size(underlying_set_size, 
		m /* nb_sets */, 
		n /* constant_size */, 
		0 /* verbose_level */);

	for (i = 0; i < m; i++) {
		INT_vec_copy(Data + i * n, Sets[i], n);
		}

	
	FREE_INT(Data);
	if (f_v) {
		cout << "set_of_sets::init_from_csv_file done" << endl;
		}
}

void set_of_sets::init_from_orbiter_file(INT underlying_set_size, const char *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "set_of_sets::init_from_orbiter_file fname=" << fname << endl;
		}
	nb_sets = count_number_of_orbits_in_file(fname, verbose_level - 1);
	if (f_v) {
		cout << "set_of_sets::init_from_orbiter_file nb_sets=" << nb_sets << endl;
		}
	set_of_sets::underlying_set_size = underlying_set_size;
	Sets = NEW_PINT(nb_sets);
	Set_size = NEW_INT(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sets[i] = NULL;
		}
	INT_vec_zero(Set_size, nb_sets);

	char *buf, *p_buf;

	buf = NEW_char(MY_BUFSIZE);

	{
	ifstream fp(fname);
	INT len, nb_sol, a, j;

	
	nb_sol = 0;
	while (TRUE) {
		if (fp.eof()) {
			break;
			}
		
		//cout << "set_of_sets::init_from_orbiter_file reading line, nb_sol = " << nb_sol << endl;
		fp.getline(buf, MY_BUFSIZE, '\n');
		if (strlen(buf) == 0) {
			cout << "set_of_sets::init_from_orbiter_file reading an empty line" << endl;
			break;
			}
		
		// check for comment line:
		if (buf[0] == '#')
			continue;
			
		p_buf = buf;
		s_scan_int(&p_buf, &len);
		if (len == -1) {
			if (f_v) {
				cout << "set_of_sets::init_from_orbiter_file found a complete file with " << nb_sol << " solutions" << endl;
				}
			break;
			}
		else {
			if (f_v) {
				cout << "set_of_sets::init_from_orbiter_file reading a set of size " << len << endl;
				}
			}
		Sets[nb_sol] = NEW_INT(len);
		Set_size[nb_sol] = len;
		for (j = 0; j < len; j++) {
			s_scan_int(&p_buf, &a);
			Sets[nb_sol][j] = a;
			}
		nb_sol++;
		}
	if (nb_sol != nb_sets) {
		cout << "set_of_sets::init_from_orbiter_file nb_sol != nb_sets" << endl;
		exit(1);
		}
	}
	FREE_char(buf);
	
	if (f_v) {
		cout << "set_of_sets::init_from_orbiter_file done" << endl;
		}
}

void set_of_sets::init_set(INT idx_of_set, INT *set, INT sz, INT verbose_level)
// Stores a copy of the given set.
{
	INT f_v = (verbose_level >= 1);
	INT j;
	
	if (f_v) {
		cout << "set_of_sets::init_set" << endl;
		}
	if (Sets[idx_of_set]) {
		cout << "set_of_sets::init_set Sets[idx_of_set] is allocated" << endl;
		exit(1);
		}
	Sets[idx_of_set] = NEW_INT(sz);
	Set_size[idx_of_set] = sz;
	for (j = 0; j < sz; j++) {
		Sets[idx_of_set][j] = set[j];
		}
	
	if (f_v) {
		cout << "set_of_sets::init_set done" << endl;
		}
}

void set_of_sets::init_cycle_structure(INT *perm, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "set_of_sets::init_cycle_structure" << endl;
		}
	INT *have_seen = NULL;
	INT *orbit_length = NULL;
	INT *orbit_length2 = NULL;
	INT nb_orbits = 0;
	INT nb_orbits2 = 0;
	INT i, l, l1, first, next, len, c;
	
	if (f_v) {
		cout << "set_of_sets::init_cycle_structure n=" << n << endl;
		}
	orbit_length = NEW_INT(n);
	orbit_length2 = NEW_INT(n);
	have_seen = NEW_INT(n);
	INT_vec_zero(have_seen, n);

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
		while (TRUE) {
			if (l1 >= n) {
				cout << "set_of_sets::init_cycle_structure cyle "
						"starting with " << first << endl;
				cout << "l1 = " << l1 << " >= n" << endl;
				exit(1);
				}
			have_seen[l1] = TRUE;
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
		while (TRUE) {
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
		INT_vec_print(cout, orbit_length, nb_orbits);
		cout << endl;
		}

	init_basic(n /* underlying_set_size */,
			nb_orbits, orbit_length, 0 /* verbose_level */);

	INT_vec_zero(have_seen, n);

	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
			}
		// work on a n e w cycle, starting at position l:
		first = l;
		//cout << "set_of_sets::init_cycle_structure cyle
		//starting with " << first << endl;
		l1 = l;
		len = 1;
		while (TRUE) {
			if (l1 >= n) {
				cout << "set_of_sets::init_cycle_structure cyle "
						"starting with " << first << endl;
				cout << "l1 = " << l1 << " >= n" << endl;
				exit(1);
				}
			have_seen[l1] = TRUE;
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
		while (TRUE) {
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
	
	FREE_INT(have_seen);
	FREE_INT(orbit_length);
	FREE_INT(orbit_length2);
	if (f_v) {
		cout << "set_of_sets::init_cycle_structure done" << endl;
		}
}

INT set_of_sets::total_size()
{
	INT sz, i;

	sz = 0;
	for (i = 0; i < nb_sets; i++) {
		sz += Set_size[i];
		}
	return sz;
}

INT &set_of_sets::element(INT i, INT j)
{
	return Sets[i][j];
}

void set_of_sets::add_element(INT i, INT a)
{
	Sets[i][Set_size[i]++] = a;
}

void set_of_sets::print()
{
	INT i;
	
	cout << "(";
	for (i = 0; i < nb_sets; i++) {
		INT_vec_print(cout, Sets[i], Set_size[i]);
		if (i < nb_sets - 1) {
			cout << ", ";
			}
		}
	cout << ")" << endl;
}

void set_of_sets::print_table()
{
	INT i;
	
	cout << "set of sets with " << nb_sets << " sets :" << endl;
	for (i = 0; i < nb_sets; i++) {
		cout << "set " << i << " has size " << Set_size[i] << " : ";
		INT_vec_print(cout, Sets[i], Set_size[i]);
		cout << endl;
		}
	cout << "end set of sets" << endl;
}

void set_of_sets::print_table_tex(ostream &ost)
{
	INT i;
	
	//cout << "set of sets with " << nb_sets << " sets :" << endl;
	for (i = 0; i < nb_sets; i++) {
		ost << "Set " << i << " has size " << Set_size[i] << " : $";
		INT_set_print_tex(ost, Sets[i], Set_size[i]);
		ost << "$\\\\" << endl;
		}
	//cout << "end set of sets" << endl;
}


void set_of_sets::dualize(set_of_sets *&S, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, a, j;

	if (f_v) {
		cout << "set_of_sets::dualize" << endl;
		}
	S = NEW_OBJECT(set_of_sets);
	S->init_basic_constant_size(nb_sets,
			underlying_set_size, nb_sets, verbose_level - 1);
	INT_vec_zero(S->Set_size, underlying_set_size);
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

void set_of_sets::remove_sets_of_given_size(INT k,
		set_of_sets &S, INT *&Idx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i, l, a, j;

	if (f_v) {
		cout << "set_of_sets::remove_sets_of_given_size" << endl;
		}
	l = 0;
	for (i = 0; i < nb_sets; i++) {
		if (Set_size[i] != k) {
			l++;
			}
		}
	Idx = NEW_INT(l);
	S.init_simple(underlying_set_size, l, verbose_level - 1);
	a = 0;
	for (i = 0; i < nb_sets; i++) {
		if (Set_size[i] != k) {
			S.Sets[a] = NEW_INT(Set_size[i]);
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
		set_of_sets &S, INT *&Idx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	classify C;
	INT f_second = FALSE;
	INT f, m, nb_big_sets, i, ii, j;

	if (f_v) {
		cout << "set_of_sets::extract_largest_sets" << endl;
		}
	C.init(Set_size, nb_sets, f_second, 0);
	if (f_v) {
		cout << "set_of_sets::extract_largest_sets set sizes: ";
		C.print(FALSE /* f_backwards*/);
		}
	f = C.type_first[C.nb_types - 1];
	m = C.data_sorted[f];
	nb_big_sets = C.type_len[C.nb_types - 1];
	
	Idx = NEW_INT(nb_big_sets);
	S.init_simple(underlying_set_size, nb_big_sets, verbose_level);
	for (i = 0; i < nb_big_sets; i++) {
		ii = C.sorting_perm_inv[f + i];
		Idx[i] = ii;
		S.Sets[i] = NEW_INT(m);
		S.Set_size[i] = m;
		for (j = 0; j < m; j++) {
			S.Sets[i][j] = Sets[ii][j];
			}
		}
	
}

void set_of_sets::intersection_matrix(
	INT *&intersection_type, INT &highest_intersection_number, 
	INT *&intersection_matrix, INT &nb_big_sets, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	classify C;
	INT f_second = FALSE;
	INT f, l, a, i, ii, u, j;

	if (f_v) {
		cout << "set_of_sets::intersection_matrix" << endl;
		}
	C.init(Set_size, nb_sets, f_second, 0);
	if (FALSE /*f_v*/) {
		cout << "set_of_sets::intersection_matrix "
				"plane-intersection type: ";
		C.print(FALSE /* f_backwards*/);
		}

	if (f_v) {
		cout << "The intersection type is (";
		C.print_naked(FALSE /* f_backwards*/);
		cout << ")" << endl << endl;
		}
	f = C.type_first[C.nb_types - 1];
	highest_intersection_number = C.data_sorted[f];
	intersection_type = NEW_INT(highest_intersection_number + 1);
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

	INT *Incma, *Incma_t, *IIt, *ItI;
	
	Incma = NEW_INT(underlying_set_size * nb_big_sets);
	Incma_t = NEW_INT(nb_big_sets * underlying_set_size);
	IIt = NEW_INT(underlying_set_size * underlying_set_size);
	ItI = NEW_INT(nb_big_sets * nb_big_sets);


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
	if (FALSE /*f_vv*/) {
		cout << "Incidence matrix:" << endl;
		print_integer_matrix_width(cout, Incma,
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
	if (FALSE /*f_vv*/) {
		cout << "I * I^\\top = " << endl;
		print_integer_matrix_width(cout, IIt,
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
	if (FALSE /*f_v*/) {
		cout << "I^\\top * I = " << endl;
		print_integer_matrix_width(cout, ItI,
				nb_big_sets, nb_big_sets, nb_big_sets, 3);
		}
	
	intersection_matrix = NEW_INT(nb_big_sets * nb_big_sets);
	for (i = 0; i < nb_big_sets; i++) {
		for (j = 0; j < nb_big_sets; j++) {
			intersection_matrix[i * nb_big_sets + j] =
					ItI[i * nb_big_sets + j];
			}
		}

	FREE_INT(Incma);
	FREE_INT(Incma_t);
	FREE_INT(IIt);
	FREE_INT(ItI);
	if (f_v) {
		cout << "set_of_sets::intersection_matrix done" << endl;
		}
}

void set_of_sets::compute_incidence_matrix(
		INT *&Inc, INT &m, INT &n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, h;

	if (f_v) {
		cout << "set_of_sets::compute_and_print_tdo_row_scheme" << endl;
		}
	m = underlying_set_size;
	n = nb_sets;
	Inc = NEW_INT(underlying_set_size * nb_sets);
	INT_vec_zero(Inc, m * n);
	for (j = 0; j < nb_sets; j++) {
		for (h = 0; h < Set_size[j]; h++) {
			i = Sets[j][h];
			Inc[i * nb_sets + j] = 1;
			}
		}
}

void set_of_sets::compute_and_print_tdo_row_scheme(
		ofstream &file, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Inc;
	incidence_structure *I;
	partitionstack *Stack;
	INT depth = INT_MAX;
	//INT i, j, a;
	INT m, n;
		
	if (f_v) {
		cout << "set_of_sets::compute_and_print_tdo_row_scheme" << endl;
		}

	compute_incidence_matrix(Inc, m, n, verbose_level - 2);

#if 0
	Inc = NEW_INT(underlying_set_size * nb_sets);
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


	INT set_size = underlying_set_size;
	INT nb_blocks = nb_sets;

	I = NEW_OBJECT(incidence_structure);
	I->init_by_matrix(set_size, nb_blocks, Inc, 0 /* verbose_level */);
	Stack = NEW_OBJECT(partitionstack);
	Stack->allocate(set_size + nb_blocks, 0 /* verbose_level */);
	Stack->subset_continguous(set_size, nb_blocks);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();

	I->compute_TDO_safe(*Stack, depth, verbose_level - 2);
		
	I->get_and_print_row_tactical_decomposition_scheme_tex(
		file, FALSE /* f_enter_math */,
		FALSE /* f_print_subscripts */, *Stack);

	FREE_INT(Inc);
	FREE_OBJECT(I);
	FREE_OBJECT(Stack);
	if (f_v) {
		cout << "set_of_sets::compute_and_print_tdo_row_scheme done" << endl;
		}
}

void set_of_sets::compute_and_print_tdo_col_scheme(
		ofstream &file, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Inc;
	INT m, n;
	incidence_structure *I;
	partitionstack *Stack;
	INT depth = INT_MAX;
	//INT i, j, a;
		
	if (f_v) {
		cout << "set_of_sets::compute_and_print_tdo_col_scheme" << endl;
		}


	compute_incidence_matrix(Inc, m, n, verbose_level - 2);

#if 0
	Inc = NEW_INT(underlying_set_size * nb_sets);
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

	INT set_size = underlying_set_size;
	INT nb_blocks = nb_sets;

	I = NEW_OBJECT(incidence_structure);
	I->init_by_matrix(set_size, nb_blocks, Inc, 0 /* verbose_level */);
	Stack = NEW_OBJECT(partitionstack);
	Stack->allocate(set_size + nb_blocks, 0 /* verbose_level */);
	Stack->subset_continguous(set_size, nb_blocks);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();

	I->compute_TDO_safe(*Stack, depth, verbose_level - 2);
		
	I->get_and_print_column_tactical_decomposition_scheme_tex(
		file, FALSE /* f_enter_math */,
		FALSE /* f_print_subscripts */, *Stack);

	FREE_INT(Inc);
	FREE_OBJECT(I);
	FREE_OBJECT(Stack);
	if (f_v) {
		cout << "set_of_sets::compute_and_print_tdo_col_scheme done" << endl;
		}
}

void set_of_sets::init_decomposition(
		decomposition *&D, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Inc;
	INT m, n;

	if (f_v) {
		cout << "set_of_sets::init_decomposition" << endl;
		}
	compute_incidence_matrix(Inc, m, n, verbose_level - 2);

	D = NEW_OBJECT(decomposition);

	D->init_incidence_matrix(underlying_set_size,
			nb_sets, Inc, verbose_level - 1);

	FREE_INT(Inc);

	if (f_v) {
		cout << "set_of_sets::init_decomposition done" << endl;
		}
}

void set_of_sets::compute_tdo_decomposition(
		decomposition &D, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Inc;
	INT m, n;
	//incidence_structure *I;
	//partitionstack *Stack;
	//INT depth = INT_MAX;
		
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
		INT_matrix_print(Inc, underlying_set_size, nb_sets);
		}


	if (f_v) {
		cout << "set_of_sets::compute_tdo_decomposition "
				"before D.init_incidence_matrix" << endl;
		}
	D.init_incidence_matrix(underlying_set_size,
			nb_sets, Inc, verbose_level - 1);
	FREE_INT(Inc);


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

INT set_of_sets::is_member(INT i, INT a, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT ret, idx;
	
	if (f_v) {
		cout << "set_of_sets::is_member" << endl;
		}
	ret = INT_vec_search(Sets[i], Set_size[i], a, idx);
	if (f_v) {
		cout << "set_of_sets::is_member done" << endl;
		}
	return ret;
}

void set_of_sets::sort_all(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "set_of_sets::sort_all" << endl;
		}
	for (i = 0; i < nb_sets; i++) {
		INT_vec_heapsort(Sets[i], Set_size[i]);
		}

	if (f_v) {
		cout << "set_of_sets::sort_all done" << endl;
		}
}

void set_of_sets::all_pairwise_intersections(
		set_of_sets *&Intersections, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT N, i, j, k;
	INT *v;
	INT l;
	
	if (f_v) {
		cout << "set_of_sets::all_pairwise_intersections" << endl;
		}
	N = INT_n_choose_k(nb_sets, 2);


	Intersections = NEW_OBJECT(set_of_sets);
	Intersections->init_simple(underlying_set_size,
			N, verbose_level - 1);

	v = NEW_INT(underlying_set_size);
	for (i = 0; i < nb_sets; i++) {
		for (j = i + 1; j < nb_sets; j++) {
			k = ij2k(i, j, nb_sets);	
			INT_vec_intersect_sorted_vectors(
					Sets[i], Set_size[i], Sets[j], Set_size[j], v, l);
			Intersections->Sets[k] = NEW_INT(l);
			INT_vec_copy(v, Intersections->Sets[k], l);
			Intersections->Set_size[k] = l;
			}
		}

	FREE_INT(v);
	
	if (f_v) {
		cout << "set_of_sets::all_pairwise_intersections done" << endl;
		}
}

void set_of_sets::pairwise_intersection_matrix(INT *&M, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	INT *v;
	INT l;
	
	if (f_v) {
		cout << "set_of_sets::pairwise_intersection_matrix" << endl;
		}


	M = NEW_INT(nb_sets * nb_sets);
	INT_vec_zero(M, nb_sets * nb_sets);

	v = NEW_INT(underlying_set_size);
	for (i = 0; i < nb_sets; i++) {
		for (j = i + 1; j < nb_sets; j++) {
			INT_vec_intersect_sorted_vectors(Sets[i],
					Set_size[i], Sets[j], Set_size[j], v, l);
			M[i * nb_sets + j] = l;
			M[j * nb_sets + i] = l;
			}
		}

	FREE_INT(v);
	
	if (f_v) {
		cout << "set_of_sets::all_pairwise_intersections done" << endl;
		}
}

void set_of_sets::all_triple_intersections(
		set_of_sets *&Intersections, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT N, i, j, k, h;
	INT *v;
	INT *w;
	INT l1, l2;
	
	if (f_v) {
		cout << "set_of_sets::all_triple_intersections" << endl;
		}
	N = INT_n_choose_k(nb_sets, 3);


	Intersections = NEW_OBJECT(set_of_sets);
	Intersections->init_simple(underlying_set_size, N, verbose_level - 1);

	v = NEW_INT(underlying_set_size);
	w = NEW_INT(underlying_set_size);
	for (i = 0; i < nb_sets; i++) {
		for (j = i + 1; j < nb_sets; j++) {

			INT_vec_intersect_sorted_vectors(Sets[i],
					Set_size[i], Sets[j], Set_size[j], v, l1);

			for (k = j + 1; k < nb_sets; k++) {
			
				h = ijk2h(i, j, k, nb_sets);	
				INT_vec_intersect_sorted_vectors(v, l1,
						Sets[k], Set_size[k], w, l2);
				Intersections->Sets[h] = NEW_INT(l2);
				INT_vec_copy(w, Intersections->Sets[h], l2);
				Intersections->Set_size[h] = l2;
				}
			}
		}

	FREE_INT(v);
	FREE_INT(w);
	
	if (f_v) {
		cout << "set_of_sets::all_triple_intersections done" << endl;
		}
}

INT set_of_sets::has_constant_size_property()
{
	INT s, i;

	if (nb_sets == 0) {
		cout << "set_of_sets::has_constant_size_property no sets" << endl;
		exit(1);
		}
	s = Set_size[0];
	for (i = 1; i < nb_sets; i++) {
		if (Set_size[i] != s) {
			return FALSE;
			}
		}
	return TRUE;
}

INT set_of_sets::largest_set_size()
{
	INT s = INT_MIN;
	INT i;
	
	for (i = 0; i < nb_sets; i++) {
		s = MAXIMUM(s, Set_size[i]);
		}
	return s;
}

void set_of_sets::save_csv(const char *fname,
		INT f_make_heading, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
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
	FREE_OBJECT(Sp);
	if (f_v) {
		cout << "set_of_sets::save_csv done" << endl;
		}
}

INT set_of_sets::find_common_element_in_two_sets(
		INT idx1, INT idx2, INT &common_elt)
{
	INT pos1, pos2;
	
	if (INT_vecs_find_common_element(Sets[idx1],
			Set_size[idx1], Sets[idx2], Set_size[idx2], pos1, pos2)) {
		common_elt = Sets[idx1][pos1];
		return TRUE;
		}
	return FALSE;
	
}

void set_of_sets::sort()
{
	INT i;
	
	for (i = 0; i < nb_sets; i++) {
		INT_vec_heapsort(Sets[i], Set_size[i]);
		}
}

void set_of_sets::sort_big(INT verbose_level)
{
	Heapsort_general(this, nb_sets, 
		set_of_sets_compare_func, 
		set_of_sets_swap_func, NULL);
}

void set_of_sets::compute_orbits(INT &nb_orbits,
	INT *&orbit, INT *&orbit_inv,
	INT *&orbit_first, INT *&orbit_len, 
	void (*compute_image_function)(set_of_sets *S,
			void *compute_image_data, INT elt_idx,
			INT gen_idx, INT &idx_of_image, INT verbose_level),
	void *compute_image_data, 
	INT nb_gens, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, cur, a, b, g, t, l, pos, x;

	if (f_v) {
		cout << "set_of_sets::compute_orbits" << endl;
		}
	orbit = NEW_INT(nb_sets);
	orbit_inv = NEW_INT(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		orbit[i] = i;
		orbit_inv[i] = i;
		}
	orbit_first = NEW_INT(nb_sets);
	orbit_len = NEW_INT(nb_sets);
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

INT set_of_sets::number_of_eckardt_points(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *E;
	INT nb_E;

	if (f_v) {
		cout << "set_of_sets::number_of_eckardt_points" << endl;
		}
	get_eckardt_points(E, nb_E, verbose_level);
	FREE_INT(E);
	if (f_v) {
		cout << "set_of_sets::number_of_eckardt_points done" << endl;
		}
	return nb_E;
}

void set_of_sets::get_eckardt_points(
		INT *&E, INT &nb_E, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_of_sets::get_eckardt_points" << endl;
		}

	incidence_structure *IS;
	partitionstack *PStack;

	IS = NEW_OBJECT(incidence_structure);

	IS->init_by_set_of_sets(this, FALSE);

	PStack = NEW_OBJECT(partitionstack);
	PStack->allocate(nb_sets + underlying_set_size, 0 /* verbose_level */);
	PStack->subset_continguous(nb_sets, underlying_set_size);
	PStack->split_cell(0 /* verbose_level */);
	
	IS->compute_TDO_safe(*PStack,
			1 /*nb_lines + nb_points_on_surface*/ /* depth */,
			0 /* verbose_level */);

#if 0
	{
	IS->get_and_print_row_tactical_decomposition_scheme_tex(
		cout /*fp_row_scheme */, FALSE /* f_enter_math */, *PStack);
	IS->get_and_print_column_tactical_decomposition_scheme_tex(
		cout /*fp_col_scheme */, FALSE /* f_enter_math */, *PStack);
	}
#endif
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT *col_scheme;

	PStack->allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		verbose_level);
	
	col_scheme = NEW_INT(nb_row_classes * nb_col_classes);

	IS->get_col_decomposition_scheme(*PStack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		col_scheme, verbose_level);

	//cout << *this << endl;
	
	if (f_v) {
		cout << "col_scheme:" << endl;
		PStack->print_decomposition_scheme(cout, 
			row_classes, nb_row_classes,
			col_classes, nb_col_classes, 
			col_scheme, -1, -1);
		}

	INT i, j, c, s, sz;
	
	nb_E = 0;
	for (j = 0; j < nb_col_classes; j++) {
		c = col_classes[j];
		sz = PStack->cellSize[c];
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

	INT h, f, y;
	
	E = NEW_INT(nb_E);
	h = 0;
	for (j = 0; j < nb_col_classes; j++) {
		c = col_classes[j];
		sz = PStack->cellSize[c];
		s = 0;
		for (i = 0; i < nb_row_classes; i++) {
			s += col_scheme[i * nb_col_classes + j];
			}
		if (s == 3) {
			f = PStack->startCell[c];
			for (i = 0; i < sz; i++) {
				y = PStack->pointList[f + i] - nb_sets;
				E[h++] = y;
				}
			}
		}

	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	FREE_INT(col_scheme);
	FREE_OBJECT(PStack);
	FREE_OBJECT(IS);

	if (f_v) {
		cout << "set_of_sets::get_eckardt_points done" << endl;
		}
}


// #############################################################################
// global functions:
// #############################################################################


INT set_of_sets_compare_func(void *data, INT i, INT j, void *extra_data)
{
	set_of_sets *S = (set_of_sets *) data;
	INT c;

	if (S->Set_size[i] != S->Set_size[j]) {
		cout << "set_of_sets_compare_func sets "
				"must have the same size" << endl;
		exit(1);
		}
	c = INT_vec_compare(S->Sets[i], S->Sets[j], S->Set_size[i]);
	return c;
}

void set_of_sets_swap_func(void *data, INT i, INT j, void *extra_data)
{
	set_of_sets *S = (set_of_sets *) data;
	INT *p;

	if (S->Set_size[i] != S->Set_size[j]) {
		cout << "set_of_sets_swap_func sets "
				"must have the same size" << endl;
		exit(1);
		}
	p = S->Sets[i];
	S->Sets[i] = S->Sets[j];
	S->Sets[j] = p;
}

