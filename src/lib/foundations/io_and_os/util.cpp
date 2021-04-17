// util.cpp
//
// Anton Betten
//
// started:  October 23, 2002




#include "foundations.h"





using namespace std;


namespace orbiter {
namespace foundations {




const char *strip_directory(const char *p)
{
	int i, l;

	l = strlen(p);
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '/') {
			return p + i + 1;
		}
	}
	return p;
}


int is_all_whitespace(const char *str)
{
	int i, l;

	l = strlen(str);
	for (i = 0; i < l; i++) {
		if (str[i] == ' ') {
			continue;
		}
		if (str[i] == '\\') {
			i++;
			if (str[i] == 0) {
				return TRUE;
			}
			if (str[i] == 'n') {
				continue;
			}
		}
		return FALSE;
	}
	return TRUE;
}




int is_all_digits(char *p)
{
	int i, l;

	l = strlen(p);
	for (i = 0; i < l; i++) {
		if (!isdigit(p[i])) {
			return FALSE;
		}
	}
	return TRUE;
}



int str2int(string &str)
{
	int i, res, l;

	l = (int) str.length();
	res = 0;
	for (i = 0; i < l; i++) {
		res = (res * 10) + (str[i] - 48);
	}
	return res;
}

void print_longinteger_after_multiplying(
		ostream &ost, int *factors, int len)
{
	longinteger_domain D;
	longinteger_object a;

	D.multiply_up(a, factors, len, 0 /* verbose_level */);
	ost << a;
}

int my_atoi(char *str)
{
	int a;
	if (strlen(str)) {
		sscanf(str, "%d", &a); // don't use atoi, it fails for large numbers.
		return a;
	}
	return 0;
}

long int my_atol(char *str)
{
	long int a;
	if (strlen(str)) {
		sscanf(str, "%ld", &a); // don't use atoi, it fails for large numbers.
		return a;
	}
	return 0;
}

int compare_strings(void *a, void *b, void *data)
{
	char *A = (char *) a;
	char *B = (char *) b;
	return strcmp(A, B);
}

int strcmp_with_or_without(char *p, char *q)
{
	char *str1;
	char *str2;
	int ret;

	if (p[0] == '"') {
		str1 = NEW_char(strlen(p) + 1);
		strcpy(str1, p);
	}
	else {
		str1 = NEW_char(strlen(p) + 3);
		strcpy(str1, "\"");
		strcpy(str1 + strlen(str1), p);
		strcpy(str1 + strlen(str1), "\"");
	}
	if (q[0] == '"') {
		str2 = NEW_char(strlen(q) + 1);
		strcpy(str2, q);
	}
	else {
		str2 = NEW_char(strlen(q) + 3);
		strcpy(str2, "\"");
		strcpy(str2 + strlen(str2), q);
		strcpy(str2 + strlen(str2), "\"");
	}
	ret = strcmp(str1, str2);
	FREE_char(str1);
	FREE_char(str2);
	return ret;
}


uint32_t root_of_tree_uint32_t (uint32_t* S, uint32_t i)
{
	while (S[i] != i) {
		i = S[i];
	}
	return i;
}

int util_compare_func(void *a, void *b, void *data)
{
	int *A = (int *)a;
	int *B = (int *)b;
	int *p = (int *) data;
	int n = *p;
	int i;

	for (i = 0; i < n; i++) {
		if (A[i] < B[i]) {
			return 1;
		}
		if (A[i] > B[i]) {
			return -1;
		}
	}
	return 0;
}

void text_to_three_double(const char *text, double *d)
{
	const char *rotation_axis_custom_text;

	rotation_axis_custom_text = text;
	double *data;
	int data_sz;
	numerics Num;

	Num.vec_scan(rotation_axis_custom_text, data, data_sz);
	if (data_sz != 3) {
		cout << "text_to_three_double; is " << data_sz << endl;
		exit(1);
	}
	d[0] = data[0];
	d[1] = data[1];
	d[2] = data[2];
	delete [] data;

}

void text_to_three_double(std::string &text, double *d)
{
	double *data;
	int data_sz;
	numerics Num;

	Num.vec_scan(text.c_str(), data, data_sz);
	if (data_sz != 3) {
		cout << "text_to_three_double; is " << data_sz << endl;
		exit(1);
	}
	d[0] = data[0];
	d[1] = data[1];
	d[2] = data[2];
	delete [] data;

}



int stringcmp(std::string &str, const char *p)
{
	return strcmp(str.c_str(), p);
}

int strtoi(std::string &str)
{
	int i;

	i = atoi(str.c_str());
	return i;
}

long int strtolint(std::string &str)
{
	long int i;

	i = atol(str.c_str());
	return i;
}

double strtof(std::string &str)
{
	double f;

	f = atof(str.c_str());
	return f;
}

int string_starts_with_a_number(std::string &str)
{
	char c;

	c = str.c_str()[0];
	if (c >= '0' && c <= '9') {
		return TRUE;
	}
	else {
		return FALSE;
	}
}


void dump_memory_chain(void *allocated_objects)
{
	int i;
	void **pp;
	int *pi;
	void **next;

	i = 0;
	next = (void **) allocated_objects;
	while (next) {
		pp = next;
		next = (void **) pp[1];
		pi = (int *) &pp[2];
		cout << i << " : " << *pi << endl;
		i++;
	}
}

void print_vector(ostream &ost, int *v, int size)
{
	int i;

	ost << "(";
	for (i = 0; i < size; i++) {
		ost << v[i];
		if (i < size - 1) {
			ost << ", ";
		}
	}
	ost << ")";
}

void itoa(char *p, int len_of_p, int i)
{
	sprintf(p, "%d", i);
#if 0
	ostrstream os(p, len_of_p);
	os << i << ends;
#endif
}

void char_swap(char *p, char *q, int len)
{
	int i;
	char c;

	for (i = 0; i < len; i++) {
		c = *q;
		*q++ = *p;
		*p++ = c;
	}
}

void int_swap(int& x, int& y)
{
	int z;

	z = x;
	x = y;
	y = z;
}

void uchar_print_bitwise(std::ostream &ost, uchar u)
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

void uchar_move(uchar *p, uchar *q, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		*q++ = *p++;
	}
}

void int_submatrix_all_rows(int *A, int m, int n,
	int nb_cols, int *cols, int *B)
{
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < nb_cols; j++) {
			B[i * nb_cols + j] = A[i * n + cols[j]];
		}
	}
}

void int_submatrix_all_cols(int *A, int m, int n,
	int nb_rows, int *rows, int *B)
{
	int i, j;

	for (j = 0; j < n; j++) {
		for (i = 0; i < nb_rows; i++) {
			B[i * n + j] = A[rows[i] * n + j];
		}
	}
}

void int_submatrix(int *A, int m, int n,
	int nb_rows, int *rows, int nb_cols, int *cols, int *B)
{
	int i, j;

	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			B[i * nb_cols + j] = A[rows[i] * n + cols[j]];
		}
	}
}

void int_matrix_transpose(int n, int *A)
{
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < i; j++) {
			if (i != j) {
				int_swap(A[i * n + j], A[j * n + i]);
			}
		}
	}
}

void int_matrix_transpose(int *M, int m, int n, int *Mt)
// Mt must point to the right amount of memory (n * m int's)
{
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			Mt[j * m + i] = M[i * n + j];
		}
	}
}

void int_matrix_shorten_rows(int *&p, int m, int n)
{
	int *q = NEW_int(m * n);
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			q[i * n + j] = p[i * n + j];
		}
	}
	FREE_int(p);
	p = q;
}



void pint_matrix_shorten_rows(pint *&p, int m, int n)
{
	pint *q = NEW_pint(m * n);
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			q[i * n + j] = p[i * n + j];
		}
	}
	FREE_pint(p);
	p = q;
}



void print_set(ostream &ost, int size, long int *set)
{
	int i;

	ost << "{ ";
	for (i = 0; i < size; i++) {
		ost << set[i];
		if (i < size - 1) {
			ost << ", ";
		}
	}
	ost << " }";
}

void print_set_lint(ostream &ost, int size, long int *set)
{
	int i;

	ost << "{ ";
	for (i = 0; i < size; i++) {
		ost << set[i];
		if (i < size - 1) {
			ost << ", ";
		}
	}
	ost << " }";
}

void print_incidence_structure(ostream &ost,
	int m, int n, int len, int *S)
{
	int *M;
	int h, i, j;

	M = NEW_int(m * n);
	for (i = 0 ; i < m * n; i++) {
		M[i] = 0;
	}

	for (h = 0; h < len; h++) {
		i = S[h] / n;
		j = S[h] % n;
		M[i * n + j] = 1;
	}
	Orbiter->Int_vec.print_integer_matrix(ost, M, m, n);

	FREE_int(M);
}








void print_line_of_number_signs()
{
	cout << "###########################################################"
			"#######################################" << endl;
}

void print_repeated_character(ostream &ost, char c, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		ost << c;
	}
}

void print_pointer_hex(ostream &ost, void *p)
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

void print_hex_digit(ostream &ost, int digit)
{
	if (digit < 10) {
		ost << (char)('0' + digit);
	}
	else if (digit < 16) {
		ost << (char)('a' + (digit - 10));
	}
	else {
		cout << "print_hex_digit illegal digit " << digit << endl;
		exit(1);
	}
}


int test_if_sets_are_disjoint_assuming_sorted(int *set1, int *set2, int sz1, int sz2)
{
	int sz;
	int *p, *q;
	int u, v;

	sz = sz1 + sz2;
	u = v = 0;
	p = set1;
	q = set2;
	while (u + v < sz) {
		if (p[u] == q[v]) {
			return FALSE;
		}
		if (u == sz1) {
			v++;
		}
		else if (v == sz2) {
			u++;
		}
		else if (p[u] < q[v]) {
			u++;
		}
		else {
			v++;
		}
	}
	return TRUE;
}

int test_if_sets_are_disjoint_assuming_sorted_lint(
		long int *set1, long int *set2, int sz1, int sz2)
{
	int sz;
	long int *p, *q;
	int u, v;

	sz = sz1 + sz2;
	u = v = 0;
	p = set1;
	q = set2;
	while (u + v < sz) {
		if (p[u] == q[v]) {
			return FALSE;
		}
		if (u == sz1) {
			v++;
		}
		else if (v == sz2) {
			u++;
		}
		else if (p[u] < q[v]) {
			u++;
		}
		else {
			v++;
		}
	}
	return TRUE;
}

#if 1
//#define HASH_PRIME ((int) 1 << 30 - 1)
#define HASH_PRIME 174962718

int hashing(int hash0, int a)
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

int hashing_fixed_width(int hash0, int a, int bit_length)
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
		cout << "hashing_fixed_width a is not zero" << endl;
		cout << "a=" << a1 << endl;
		cout << "bit_length=" << bit_length << endl;
		exit(1);
	}
	//cout << "hashing: " << hash0 << " + " << a1 << " = " << h << endl;
	return h;
}
#endif





}}

