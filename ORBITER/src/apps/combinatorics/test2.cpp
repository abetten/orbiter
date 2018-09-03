// test2.C
// 
// Anton Betten
// Aug 26, 2009
//
//
// rank and unrank for subsets
//
//

#include "orbiter.h"


// global data:

int t0; // the system time when the program started
void FortCollins();
void test1();
void test2();
void test3();
void test4();
void test5();
void test6();
void print_set(int *set, int n, int k);
void print_set_as_bitstring(int *set, int n, int k);
void set_to_bitstring(int *set, int n, int k, int *bitstring);
int first_subset(int n, int *set, int &k);
int next_subset(int n, int *set, int &k);
void rank_subset(int *set, int n, int k, longinteger_object &rk);
void unrank_subset(int *set, int n, int &k, longinteger_object &rk);

int main(int argc, char **argv)
{
	//int verbose_level = 2;
	t0 = os_ticks();
#if 0
	if (argc <= 1) {
		usage(argc, argv);
		exit(1);
		}
#endif
	FortCollins();
	//test1();
	//test2();
	//test3();
	//test4();
	//test5();
	//test6();
	
	the_end_quietly(t0);
}

void FortCollins()
{
	const char *alphabet = "CFILNORST";

	int n = 9, i;
	int set[9];
	int k, cnt;
	longinteger_object rk;
	
	first_subset(n, set, k);
	cnt = 0;
	while (TRUE) {

		rank_subset(set, n, k, rk);

		cout << setw(5) << cnt << " & ";
		for (i = 0; i < k; i++) {
			cout << alphabet[set[i]];
			}
		cout << " \\\\" << endl;



		
		if (!next_subset(n, set, k))
			break;
		cnt++;
		}

}

void test1()
{
	int n = 9, i;
	int set[9];
	int bitstring[9];
	int k, cnt, nb;
	longinteger_object rk;
	
	first_subset(n, set, k);
	cnt = 0;
	while (TRUE) {
		rank_subset(set, n, k, rk);

		set_to_bitstring(set, n, k, bitstring);

		nb = 0;
		for (i = 0; i < n; i++) {
			nb <<= 1;
			nb += bitstring[i];
			}
		cout << cnt << " : ";
		print_set(set, n, k);
		cout << " : " << rk;
		if (cnt == nb) {
			cout << "*";
			}
		cout << endl;
		
		if (!next_subset(n, set, k))
			break;
		cnt++;
		}
}

void test2()
{
	int n = 26;
	int set[26];
	int k, cnt;
	longinteger_object rk;
	
	first_subset(n, set, k);
	cnt = 0;
	while (TRUE) {
		if (cnt == 33548164 || (cnt % 1000000) == 0) {
			rank_subset(set, n, k, rk);
			cout << cnt << " : ";
			print_set(set, n, k);
			cout << " : " << rk << endl;
			}
		if (!next_subset(n, set, k))
			break;
		cnt++;
		}
}

void test3()
{
	int n = 5;
	int set[5];
	int k;
	longinteger_object N, rk, rk1;
	longinteger_domain D;
	
	N.create_i_power_j(2, n);
	rk.create(0);
	for (rk.create(0); D.compare(rk, N) < 0; rk.increment()) {
		unrank_subset(set, n, k, rk);
		rank_subset(set, n, k, rk1);
		cout << rk << " : ";
		print_set(set, n, k);
		cout << " : " << rk1 << endl;
		}
}

void test4()
{
	int n = 26;
	int set[26];
	int k;
	longinteger_object rk, rk1;
	
	//rk.create(33548164);
	rk.create(53791024);
	
	unrank_subset(set, n, k, rk);
	rank_subset(set, n, k, rk1);
	cout << rk << " : ";
	print_set(set, n, k);
	cout << " : " << rk1 << endl;
}

void test5()
{
	int n = 10;
	int set[10];
	int k;
	longinteger_object rk, rk1;
	
	//rk.create(33548164);
	//rk.create(9753);
	
	set[0] = 2;
	set[1] = 3;
	set[2] = 4;
	set[3] = 8;
	
	k = 4;
	rank_subset(set, n, k, rk);
	unrank_subset(set, n, k, rk);
	rank_subset(set, n, k, rk1);
	cout << rk << " : ";
	print_set(set, n, k);
	cout << " : " << rk1 << endl;
}

void test6()
{
	const char *Names[] = {
		"francis",
		"peter",
		"cory",
		"nissa",
		"eric",
		"hilary",
		"dean",
		"nathan",
		"nathanial",
		"yang",
		"jamie",
		"patrick",
		"eunju",
		"anton"
		};
	int nb_names = 14;
	int c;
	
	for (c = 0; c < nb_names; c++) {

		const char *name = Names[c];
		int len = strlen(name);
		int *set;
		int i, j;
		
		set = new int[len + 1];
		for (i = 0; i < len; i++) {
			set[i] = (int)(name[i] - 'a');
			}
		int_vec_sort(len, set);
		j = 0;
		for (i = 1; i < len; i++) {
			if (set[i] != set[j]) {
				set[j + 1] = set[i];
				j++;
				}
			}
		len = j + 1;
		cout << "the name " << name << " corresponds to the set ";
		int_vec_print(cout, set, len);
		cout << endl;
		
		
		int n = 26;
		int k;
		longinteger_object rk, rk1;
	
		//rk.create(33548164);
		//rk.create(9753);
	
	
		k = len;
		rank_subset(set, n, k, rk);
		unrank_subset(set, n, k, rk);
		rank_subset(set, n, k, rk1);
		cout << rk << " : ";
		print_set(set, n, k);
		cout << " : " << rk1 << endl;

		cout << endl;
		
		delete [] set;
		
		} // next c
}


void print_set(int *set, int n, int k)
{
	print_set_as_bitstring(set, n, k);
	//int_vec_print(cout, set, k);
}

void print_set_as_bitstring(int *set, int n, int k)
{
	int i, j;
	
	j = 0;
	for (i = 0; i < n; i++) {
		if (j < k && set[j] == i) {
			cout << "1";
			j++;
			}
		else {
			cout << "0";
			}
		}
}

void set_to_bitstring(int *set, int n, int k, int *bitstring)
{
	int i, j;
	
	j = 0;
	for (i = 0; i < n; i++) {
		if (j < k && set[j] == i) {
			bitstring[i] = 1;
			j++;
			}
		else {
			bitstring[i] = 0;
			}
		}
}

int first_subset(int n, int *set, int &k)
{
	
	k = 0;
	return TRUE;
}

int next_subset(int n, int *set, int &k)
{	
	if (k == 0) {
		if (n) {
			set[0] = 0;
			k = 1;
			return TRUE;
			}
		else {
			return FALSE;
			}
		}
	if (k == 1 && set[0] == n - 1) {
		return FALSE;
		}
	if (set[k - 1] == n - 1) {
		set[k - 2]++;
		k--;
		return TRUE;
		}
	else {
		set[k] = set[k - 1] + 1;
		k++;
		return TRUE;
		}
}

void rank_subset(int *set, int n, int k, longinteger_object &rk)
{
	int i, j;
	longinteger_object a, b;
	longinteger_domain D;
	
	rk.create(0);
	j = 0;
	for (i = 0; i < n; i++) {
		if (j == k)
			break;
		if (set[j] > i) {
			a.create_i_power_j(2, n - 1 - i);
			D.add(rk, a, b);
			b.assign_to(rk);
			}
		else {
			j++;
			}
		}
	a.create(k);
	D.add(rk, a, b);
	b.assign_to(rk);
}

void unrank_subset(int *set, int n, int &k, longinteger_object &rk)
{
	int i, c;
	longinteger_object r, a, b;
	longinteger_domain D;
	
	rk.assign_to(r);
	k = 0;
	for (i = 0; i < n; i++) {
		a.create_i_power_j(2, n - 1 - i);
		//cout << "r=" << r << " i=" << i << " a=" << a << endl;
		c = D.compare(r, a);
		if (c <= 0) {
			//cout << "left" << endl;
			if (r.is_zero()) {
				return;
				}
			set[k] = i;
			//cout << "set[ " << k << " ] = " << i << endl;
			k++;
			r.decrement();
			}
		else {
			//cout << "right" << endl;
			a.negate();
			D.add(r, a, b);
			b.assign_to(r);
			}
		}
}


