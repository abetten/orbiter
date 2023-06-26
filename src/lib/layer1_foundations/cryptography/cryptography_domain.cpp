/*
 * cryptography_domain.cpp
 *
 *  Created on: Nov 4, 2020
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace cryptography {


static double letter_probability[] = {
	0.082, // a
	0.015, // b
	0.028, // c
	0.043, // d
	0.127, // e
	0.022, // f
	0.020, // g
	0.061, // h
	0.070, // i
	0.002, // j
	0.008, // k
	0.040, // l
	0.024, // m
	0.067, // n
	0.075, // o
	0.019, // p
	0.001, // q
	0.060, // r
	0.063, // s
	0.091, // t
	0.028, // u
	0.010, // v
	0.001, // x
	0.020, // y
	0.001 // z
};





cryptography_domain::cryptography_domain()
{

}

cryptography_domain::~cryptography_domain()
{

}

void cryptography_domain::affine_cipher(
		std::string &ptext,
		std::string &ctext, int a, int b)
{
	int i, l, x, y;
	char *str;

	cout << "applying key (" << a << "," << b << ")" << endl;
	l = ptext.length();
	str = NEW_char(l + 1);
	for (i = 0; i < l; i++) {
		x = (int)(upper_case(ptext[i]) - 'A');
		y = a * x + b;
		y = y % 26;
		str[i] = 'A' + y;
	}
	str[l] = 0;
	ctext.assign(str);
	FREE_char(str);
}

void cryptography_domain::affine_decipher(
		std::string &ctext,
		std::string &ptext, std::string &guess)
// we have ax_1 + b = y_1
// and     ax_2 + b = y_2
// or equivalently
//         matrix(x_1,1,x_2,1) vector(a,b) = vector(y_1,y_2)
// and hence
//         vector(a,b) = matrix(1,-1,-x_2,x_1) vector(y_1,y_2) * 1/(x_1 - x_2)
{
	int x1, x2, y1, y2, dy, dx, i;
	int a, b, a0, av, c, g, dxv, n, gg;
	number_theory::number_theory_domain NT;

	if (guess.length() != 4) {
		cout << "guess must be 4 characters long!" << endl;
		exit(1);
	}
	cout << "guess=" << guess << endl;
	y1 = lower_case(guess[0]) - 'a';
	x1 = lower_case(guess[1]) - 'a';
	y2 = lower_case(guess[2]) - 'a';
	x2 = lower_case(guess[3]) - 'a';

	cout << "y1=" << y1 << endl;
	cout << "x1=" << x1 << endl;
	cout << "y2=" << y2 << endl;
	cout << "x2=" << x2 << endl;
	dy = remainder(y2 - y1, 26);
	dx = remainder(x2 - x1, 26);
	//cout << "dx = x2 - x1 = " << dx << endl;
	//cout << "dy = y2 - y1 = " << dy << endl;
	cout << "solving:  a * (x2-x1) = y2 - y1 mod 26" << endl;
	cout << "here:     a * " << dx << " = " << dy << " mod 26" << endl;
	n = 26;
	//g = gcd_int(dx, n);
	g = NT.gcd_lint(dx, n);
	if (remainder(dy, g) != 0) {
		cout << "gcd(x2-x1,26) does not divide y2-y1, "
				"hence no solution! try again" << endl;
		exit(1);
	}
	if (g != 1) {
		dy /= g;
		dx /= g;
		n /= g;
		cout << "reducing to:     a * " << dx << " = " << dy << " mod " << n << endl;
	}
	dxv = NT.inverse_mod(dx, n);
	cout << dx << "^-1 mod " << n << " = " << dxv << endl;
	a0 = remainder(dy * dxv, n);
	cout << "solution a = " << a0 << " mod " << n << endl;
	cout << "(ie " << g << " solutions for a mod 26)" << endl;
	for (i = 0; i < g; i++) {
		cout << "trying solution " << i + 1 << ":" << endl;
		a = a0 + i * n;
		b = remainder(y1 - a * x1, 26);
		cout << "yields key (" << a << "," << b << ")" << endl;
		//gg = gcd_int(a, 26);
		gg = NT.gcd_lint(a, 26);
		if (gg != 1) {
			cout << a << " not prime to 26, no solution" << endl;
			continue;
		}
		av = NT.inverse_mod(a, 26);
		cout << "a^-1 mod 26 = " << av << endl;
		c = av * (-b);
		c = remainder(c, 26);
		cout << "a^-1 (-b) = " << c << " mod 26" << endl;
		affine_cipher(ctext, ptext, av, c);
		cout << " " << endl;
		print_on_top(ptext, ctext);
	}
}

void cryptography_domain::vigenere_cipher(
		std::string &ptext,
		std::string &ctext, std::string &key)
{
	int i, j, l, key_len, a, b, c;
	char *str;

	key_len = key.length();
	l = ptext.length();
	str = NEW_char(l + 1);
	for (i = 0, j = 0; i < l; i++, j++) {
		if (j == key_len) {
			j = 0;
		}
		if (is_alnum(ptext[i]) && is_alnum(key[j])) {
			a = (int)(lower_case(ptext[i]) - 'a');
			b = (int)(lower_case(key[j]) - 'a');
			c = a + b;
			c = c % 26;
			str[i] = 'a' + c;
		}
		else {
			str[i] = ptext[i];
		}
	}
	str[l] = 0;
	ctext.assign(str);
	FREE_char(str);
}

void cryptography_domain::vigenere_decipher(
		std::string &ctext,
		std::string &ptext, std::string &key)
{
	int i, j, l, key_len, a, b, c;
	char *str;

	key_len = key.length();
	l = ctext.length();
	str = NEW_char(l + 1);
	for (i = 0, j = 0; i < l; i++, j++) {
		if (j == key_len)
			j = 0;
		if (is_alnum(ctext[i]) && is_alnum(key[j])) {
			a = (int)(lower_case(ctext[i]) - 'a');
			b = (int)(lower_case(key[j]) - 'a');
			c = a - b;
			if (c < 0)
				c += 26;
			str[i] = 'a' + c;
		}
		else {
			str[i] = ctext[i];
		}
	}
	str[l] = 0;
	ptext.assign(str);
	FREE_char(str);
}

void cryptography_domain::vigenere_analysis(
		std::string &ctext)
{
	int stride, l, n, m, j;
	int mult[100];
	double I, II;

	cout.precision(3);
	l = ctext.length();
	cout << "key length : Friedman indices in 1/1000-th : average" << endl;
	for (stride = 1; stride <= MINIMUM(l, 20); stride++) {
		m = l / stride;
		if (m < 2)
			continue;
		II = 0;
		cout << stride << " : ";
		for (j = 0; j < stride; j++) {
			if (j == stride - 1) {
				n = l - (stride - 1) * m;
			}
			else {
				n = m;
			}

			std::string str2;

			str2 = ctext.substr(j, n);
			single_frequencies2(str2, stride, n, mult);

			//print_frequencies(mult);
			I = friedman_index(mult, n);
			if (stride < 20) {
				if (j) {
					cout << ", ";
				}
				cout << (int)(1000. * I);
			}
			II += I;
		}
		II *= 1. / (double) stride;
		cout << " : " << (int)(1000. * II) << endl;
	}
}

void cryptography_domain::vigenere_analysis2(
		std::string &ctext, int key_length)
{
	int i, j, shift, n, m, l, a, h;
	int mult[100];
	int index[100];
	int shift0[100];
	double I;
	char c;

	l = ctext.length();
	m = l / key_length;
	cout << "    :";
	for (shift = 0; shift < 26; shift++) {
		cout << "  ";
		c = 'a' + shift;
		cout << c;
	}
	cout << endl;
	for (j = 0; j < key_length; j++) {
		cout.width(3);
		cout << j << " :";
		if (j == key_length - 1) {
			n = l - (key_length - 1) * m;
		}
		else {
			n = m;
		}

		std::string str = ctext.substr(j, n);
		single_frequencies2(str, key_length, n, mult);

		//print_frequencies(mult);
		for (shift = 0; shift < 26; shift++) {
			I = friedman_index_shifted(mult, n, shift);
			cout.width(3);
			a = (int)(1000. * I);
			cout << a;
			if (shift) {
				for (i = 0; i < shift; i++) {
					if (index[i] < a) {
						for (h = shift; h > i; h--) {
							index[h] = index[h - 1];
							shift0[h] = shift0[h - 1];
							}
						index[i] = a;
						shift0[i] = shift;
						break;
					}
				}
			}
			else {
				index[0] = a;
				shift0[0] = shift;
			}
		}
		cout << " : ";
		for (i = 0; i < 5; i++) {
			c = 'a' + shift0[i];
			if (i) {
				cout << ", ";
			}
			cout << c << " " << index[i];
		}
		cout << endl;
	}
}

int cryptography_domain::kasiski_test(
		std::string &ctext,
		int threshold)
{
	int l, i, j, k, u, h, offset;
	int *candidates, nb_candidates, *Nb_candidates;
	int *f_taken;
	int g = 0, g1;
	number_theory::number_theory_domain NT;

	l = ctext.length();
	candidates = new int[l];
	f_taken = new int[l];
	Nb_candidates = new int[l];
	for (i = 0; i < l; i++) {
		f_taken[i] = false;
	}

	for (i = 0; i < l; i++) {
		//cout << "position " << i << endl;
		nb_candidates = 0;
		for (j = i + 1; j < l; j++) {
			if (ctext[j] == ctext[i]) {
				candidates[nb_candidates++] = j;
			}
		}
		h = 1;
		//print_candidates(ctext, i, h, nb_candidates, candidates);

		//cout << "at position " << i << ", found " << nb_candidates << " matches of length 1" << endl;
		Nb_candidates[h - 1] += nb_candidates;
		while (nb_candidates) {
			for (k = 0; k < nb_candidates; k++) {
				j = candidates[k];
				if (ctext[i + h] != ctext[j + h]) {
					for (u = k + 1; u < nb_candidates; u++) {
						candidates[u - 1] = candidates[u];
					}
					nb_candidates--;
					k--;
				}
			}
			h++;
			Nb_candidates[h - 1] += nb_candidates;
			if (h >= threshold && nb_candidates) {
				print_candidates(ctext, i, h, nb_candidates, candidates);
				g1 = 0;
				for (k = 0; k < nb_candidates; k++) {
					offset = candidates[k] - i;
					if (g1 == 0) {
						g1 = offset;
					}
					else {
						//g1 = gcd_int(g1, offset);
						g1 = NT.gcd_lint(g1, offset);
					}
				}
				//cout << "g1 = " << g1 << endl;
				if (g == 0) {
					g = g1;
				}
				else {
					//g = gcd_int(g, g1);
					g = NT.gcd_lint(g, g1);
				}
				//cout << "g = " << g << endl;
				//break;
			}
		}
	}
	for (i = 0; Nb_candidates[i]; i++) {
		cout << "matches of length " << i + 1 << " : " <<  Nb_candidates[i] << endl;
	}
	delete [] candidates;
	delete [] f_taken;
	delete [] Nb_candidates;
	return g;
}

void cryptography_domain::print_candidates(
		std::string &ctext,
		int i, int h, int nb_candidates, int *candidates)
{
	int k, j, u;

	if (nb_candidates == 0) {
		return;
	}
	cout << "there are " << nb_candidates << " level " << h << " coincidences with position " << i << endl;
	for (k = 0; k < nb_candidates; k++) {
		j = candidates[k];
		cout << "at " << j << " : ";
		for (u = 0; u < h; u++) {
			cout << ctext[i + u];
		}
		cout << " : ";
		for (u = 0; u < h; u++) {
			cout << ctext[j + u];
		}
		cout << " offset " << j - i;
		cout << endl;
	}
}

void cryptography_domain::print_set(int l, int *s)
{
	int i;

	for (i = 0; i < l; i++) {
		cout << s[i] << " ";
	}
	cout << endl;
}

void cryptography_domain::print_on_top(
		std::string &text1,
		std::string &text2)
{
	int i, j, l, l2, lines, line_length;

	l = text1.length();
	l2 = text2.length();
	if (l2 != l) {
		cout << "text lengths do not match" << endl;
		exit(1);
	}
	lines = l / 80;
	for (i = 0; i <= lines; i++) {
		if (i == lines) {
			line_length = l % 80;
		}
		else {
			line_length = 80;
		}
		for (j = 0; j < line_length; j++) {
			cout << text1[i * 80 + j];
		}
		cout << endl;
		for (j = 0; j < line_length; j++) {
			cout << text2[i * 80 + j];
		}
		cout << endl;
		cout << " " << endl;
	}
}

void cryptography_domain::decipher(
		std::string &ctext,
		std::string &ptext, std::string &guess)
{
	int i, j, l;
	char str_key[1000], c1, c2;

	l = guess.length() / 2;
	for (i = 0; i < 26; i++) {
		str_key[i] = '-';
	}
	str_key[26] = 0;
	for (i = 0; i < l; i++) {
		c1 = guess[2 * i + 0];
		c2 = guess[2 * i + 1];
		c1 = lower_case(c1);
		c2 = lower_case(c2);
		cout << c1 << " -> " << c2 << endl;
		j = c1 - 'a';
		//cout << "j=" << j << endl;
		str_key[j] = c2;
	}
	string key;

	key.assign(str_key);
	substition_cipher(ctext, ptext, key);
}

void cryptography_domain::analyze(
		std::string &text)
{
	int mult[100];
	int mult2[100];

	single_frequencies(text, mult);
	cout << "single frequencies:" << endl;
	print_frequencies(mult);
	cout << endl;

	double_frequencies(text, mult2);
	cout << "double frequencies:" << endl;
	print_frequencies(mult2);
	cout << endl;
}

double cryptography_domain::friedman_index(
		int *mult, int n)
{
	int i, a = 0, b = n * (n - 1);
	double d;

	for (i = 0; i < 26; i++) {
		if (mult[i] > 1) {
			a += mult[i] * (mult[i] - 1);
		}
	}
	d = (double) a / (double) b;
	return d;
}

double cryptography_domain::friedman_index_shifted(
		int *mult, int n, int shift)
{
	int i, ii;
	double a = 0., d;

	for (i = 0; i < 26; i++) {
		ii = i + shift;
		ii = ii % 26;
		if (mult[ii]) {
			a += letter_probability[i] * (double) mult[ii];
		}
	}
	d = (double) a / (double) n;
	return d;
}

void cryptography_domain::print_frequencies(int *mult)
{
	int i, j = 0, k = 0, h, l = 0, f_first = true;
	char c;

	for (i = 0; i < 26; i++) {
		if (mult[i]) {
			l++;
		}
	}
	int *mult_val = new int[2 * l];
	for (i = 0; i < 26; i++) {
		if (mult[i]) {
			for (j = 0; j < k; j++) {
				if (mult_val[2 * j + 1] < mult[i]) {
					// insert here:
					for (h = k; h > j; h--) {
						mult_val[2 * h + 1] = mult_val[2 * (h - 1) + 1];
						mult_val[2 * h + 0] = mult_val[2 * (h - 1) + 0];
					}
					mult_val[2 * j + 0] = i;
					mult_val[2 * j + 1] = mult[i];
					break;
				}
			}
			if (j == k) {
				mult_val[2 * k + 0] = i;
				mult_val[2 * k + 1] = mult[i];
			}
			k++;
		}
	}

	for (i = 0; i < l; i++) {
		c = 'a' + mult_val[2 * i + 0];
		j = mult_val[2 * i + 1];
		if (!f_first) {
			cout << ", ";
		}
		cout << c;
		if (j > 1) {
			cout << "^" << j;
		}
		f_first = false;
	}
}

void cryptography_domain::single_frequencies(
		std::string &text, int *mult)
{
	int i, l;

	l = text.length();
	for (i = 0; i < 26; i++) {
		mult[i] = 0;
	}
	for (i = 0; i < l; i++) {
		mult[text[i] - 'a']++;
	}
}

void cryptography_domain::single_frequencies2(
		std::string &text,
		int stride, int n, int *mult)
{
	int i;

	for (i = 0; i < 26; i++) {
		mult[i] = 0;
	}
	for (i = 0; i < n; i++) {
		mult[text[i * stride] - 'a']++;
	}
}

void cryptography_domain::double_frequencies(
		std::string &text, int *mult)
{
	int i, l;

	l = text.length();
	for (i = 0; i < 26; i++) {
		mult[i] = 0;
	}
	for (i = 0; i < l - 1; i++) {
		if (text[i] == text[i + 1]) {
			mult[text[i] - 'a']++;
		}
	}
}

void cryptography_domain::substition_cipher(
		std::string &ptext,
		std::string &ctext, std::string &key)
{
	int i, l;
	char c;
	char *str;

	cout << "cryptography_domain::substition_cipher "
			"applying key:" << endl;
	for (i = 0; i < 26; i++) {
		c = 'a' + i;
		cout << c;
	}
	cout << endl;
	cout << key << endl;
	l = ptext.length();
	str = NEW_char(l + 1);
	for (i = 0; i < l; i++) {
		if (is_alnum(ptext[i])) {
			str[i] = key[lower_case(ptext[i]) - 'a'];
		}
		else {
			str[i] = ptext[i];
		}
	}
	str[l] = 0;
	ctext.assign(str);
	FREE_char(str);
}

char cryptography_domain::lower_case(char c)
{
	if (c >= 'A' && c <= 'Z') {
		c = c - ('A' - 'a');
		return c;
	}
	else if (c >= 'a' && c <= 'z') {
		return c;
	}
	else {
		//cout << "illegal character " << c << endl;
		//exit(1);
		return c;
	}
}

char cryptography_domain::upper_case(char c)
{
	if (c >= 'a' && c <= 'z') {
		c = c + ('A' - 'a');
		return c;
	}
	else if (c >= 'A' && c <= 'Z') {
		return c;
	}
	else {
		//cout << "illegal character " << c << endl;
		//exit(1);
		return c;
	}
}

char cryptography_domain::is_alnum(char c)
{
	if (c >= 'A' && c <= 'Z') {
		return true;
	}
	if (c >= 'a' && c <= 'z') {
		return true;
	}
	return false;
}

void cryptography_domain::get_random_permutation(std::string &p)
{
	char digits[100];
	int i, j, k, l;
	orbiter_kernel_system::os_interface OS;

	for (i = 0; i < 26; i++) {
		digits[i] = 'a' + i;
	}
	digits[26] = 0;
	for (i = 0; i < 26; i++) {
		l = strlen(digits);
		j = OS.random_integer(l);
		p[i] = digits[j];
		for (k = j + 1; k <= l; k++) {
			digits[k - 1] = digits[k];
		}
		l--;
	}
	p.assign(digits);
}




void cryptography_domain::make_affine_sequence(
		int a, int c, int m,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *f_reached;
	int *orbit;
	int x0, x, y, len, cnt;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "cryptography_domain::make_affine_sequence "
				"a=" << a << " c=" << c << " m=" << m << endl;
	}
	f_reached = NEW_int(m);
	orbit = NEW_int(m);
	Int_vec_zero(f_reached, m);
	cnt = 0;
	for (x0 = 0; x0 < m; x0++) {
		if (f_reached[x0]) {
			continue;
		}

		x = x0;
		orbit[0] = x0;
		len = 1;
		while (true) {
			f_reached[x] = true;
			y = NT.mult_mod(a, x, m);
			y = NT.add_mod(y, c, m);

			if (f_reached[y]) {
				break;
			}
			orbit[len++] = y;
			x = y;
		}
		cout << "orbit " << cnt << " of " << x0 << " has length " << len << " : ";
		Int_vec_print(cout, orbit, len);
		cout << endl;

		make_2D_plot(orbit, len, cnt, m, a, c, verbose_level);
		//make_graph(orbit, len, m, verbose_level);
		//list_sequence_in_binary(orbit, len, m, verbose_level);
		cnt++;
	}
	FREE_int(orbit);
	FREE_int(f_reached);
	if (f_v) {
		cout << "cryptography_domain::make_affine_sequence done" << endl;
	}


}

void cryptography_domain::make_2D_plot(
		int *orbit, int orbit_len, int cnt, int m, int a, int c,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::make_2D_plot "
				"m=" << m << " orbit_len=" << orbit_len << endl;
	}
	int *M;
	int h, x, y;

	M = NEW_int(m * m);
	Int_vec_zero(M, m * m);



	for (h = 0; h < orbit_len - 1; h++) {
		x = orbit[h];
		y = orbit[h + 1];
		M[x * m + y] = 1;
	}
	string fname;
	orbiter_kernel_system::file_io Fio;

	fname = "orbit_cnt" + std::to_string(cnt) + "_m" + std::to_string(m)
			+ "_a" + std::to_string(a) + "_c" + std::to_string(c) + ".csv";

	Fio.int_matrix_write_csv(fname, M, m, m);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	FREE_int(M);
	if (f_v) {
		cout << "cryptography_domain::make_2D_plot done" << endl;
	}
}



void cryptography_domain::do_random_last(int random_nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_random_last" << endl;
	}

	int i, r = 0;


	cout << "RAND_MAX=" << RAND_MAX << endl;

	for (i = 0; i < random_nb; i++) {
		r = rand();
	}
	cout << r << endl;

	if (f_v) {
		cout << "cryptography_domain::do_random_last done" << endl;
	}

}

void cryptography_domain::do_random(
		int random_nb,
		std::string &fname_csv, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_random" << endl;
	}

	int i;
	int *R;


	cout << "RAND_MAX=" << RAND_MAX << endl;

	R = NEW_int(random_nb);
	for (i = 0; i < random_nb; i++) {
		R[i] = rand();
	}

	orbiter_kernel_system::file_io Fio;

	string label;

	label.assign("R");
	Fio.int_vec_write_csv(R, random_nb, fname_csv, label);

	cout << "written file " << fname_csv << " of size " << Fio.file_size(fname_csv) << endl;

	if (f_v) {
		cout << "cryptography_domain::do_random done" << endl;
	}

}

void cryptography_domain::do_EC_Koblitz_encoding(
		field_theory::finite_field *F,
		int EC_b, int EC_c, int EC_s,
		std::string &pt_text, std::string &EC_message,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_EC_Koblitz_encoding" << endl;
	}

	int x0, x, y;

	if (f_v) {
		cout << "cryptography_domain::do_EC_Koblitz_encoding b = " << EC_b << endl;
		cout << "cryptography_domain::do_EC_Koblitz_encoding c = " << EC_c << endl;
		cout << "cryptography_domain::do_EC_Koblitz_encoding s = " << EC_s << endl;
	}

	vector<vector<int>> Encoding;
	vector<int> J;

	int u, i, j, r;

	u = F->q / 27;
	if (f_v) {
		cout << "cryptography_domain::do_EC_Koblitz_encoding u = " << u << endl;
	}


	for (i = 1; i <= 26; i++) {
		x0 = i * u;
		for (j = 0; j < u; j++) {
			x = x0 + j;
			r = EC_evaluate_RHS(F, EC_b, EC_c, x);
			if (F->is_square(r)) {
				y = F->square_root(r);
				break;
			}
		}
		if (j < u) {
			{
				vector<int> pt;

				J.push_back(j);
				pt.push_back(x);
				pt.push_back(y);
				pt.push_back(1);
				Encoding.push_back(pt);
			}
		}
		else {
			cout << "cryptography_domain::do_EC_Koblitz_encoding "
					"failure to encode letter " << i << endl;
			exit(1);
		}
	}
	for (i = 0; i < 26; i++) {


		x = (i + 1) * u + J[i];

		r = EC_evaluate_RHS(F, EC_b, EC_c, x);

		y = F->square_root(r);

		cout << (char)('A' + i) << " & " << i + 1 << " & " << J[i] << " & " << x
				<< " & " << r
				<< " & " << y
				<< " & $(" << Encoding[i][0] << "," << Encoding[i][1] << ")$ "
				<< "\\\\" << endl;

	}

	cout << "without j:" << endl;
	for (i = 0; i < 26; i++) {
		cout << (char)('A' + i) << " & $(" << Encoding[i][0] << "," << Encoding[i][1] << ")$ \\\\" << endl;

	}



	vector<vector<int>> Pts;
	int order;
	int *v;
	int len;
	int Gx, Gy, Gz;
	int Mx, My, Mz;
	int Rx, Ry, Rz;
	int Ax, Ay, Az;
	int Cx, Cy, Cz;
	int Tx, Ty, Tz;
	int Dx, Dy, Dz;
	int msRx, msRy, msRz;
	int m, k, plain;
	orbiter_kernel_system::os_interface Os;
	number_theory::number_theory_domain NT;

	Int_vec_scan(pt_text, v, len);
	if (len != 2) {
		cout << "cryptography_domain::do_EC_Koblitz_encoding "
				"point should have just two coordinates" << endl;
		exit(1);
	}
	Gx = v[0];
	Gy = v[1];
	Gz = 1;
	FREE_int(v);
	cout << "G = (" << Gx << "," << Gy << "," << Gz << ")" << endl;


	NT.elliptic_curve_all_point_multiples(F,
			EC_b, EC_c, order,
			Gx, Gy, Gz,
			Pts,
			verbose_level);


	int minus_s;

	minus_s = order - EC_s;

	cout << "order = " << order << endl;
	cout << "minus_s = " << minus_s << endl;

	Ax = Pts[EC_s - 1][0];
	Ay = Pts[EC_s - 1][1];
	Az = 1;
	cout << "A = (" << Ax << "," << Ay << "," << Az << ")" << endl;

	len = EC_message.length();

	//F->nb_calls_to_elliptic_curve_addition() = 0;

	vector<vector<int>> Ciphertext;

	for (i = 0; i < len; i++) {
		if (EC_message[i] < 'A' || EC_message[i] > 'Z') {
			continue;
		}
		m = EC_message[i] - 'A' + 1;
		k = 1 + Os.random_integer(order - 1);

		Mx = Encoding[m - 1][0];
		My = Encoding[m - 1][1];
		Mz = 1;

		// R := k * G
		//cout << "$R=" << k << "*G$\\\\" << endl;

		NT.elliptic_curve_point_multiple /*_with_log*/(F,
					EC_b, EC_c, k,
					Gx, Gy, Gz,
					Rx, Ry, Rz,
					0 /*verbose_level*/);
		//cout << "$R=" << k << "*G=(" << Rx << "," << Ry << "," << Rz << ")$\\\\" << endl;

		// C := k * A
		//cout << "$C=" << k << "*A$\\\\" << endl;
		NT.elliptic_curve_point_multiple /*_with_log*/(F,
					EC_b, EC_c, k,
					Ax, Ay, Az,
					Cx, Cy, Cz,
					0 /*verbose_level*/);
		//cout << "$C=" << k << "*A=(" << Cx << "," << Cy << "," << Cz << ")$\\\\" << endl;

		// T := C + M
		NT.elliptic_curve_addition(F,
				EC_b, EC_c,
				Cx, Cy, Cz,
				Mx, My, Mz,
				Tx, Ty, Tz,
				0 /*verbose_level*/);
		//cout << "$T=C+M=(" << Tx << "," << Ty << "," << Tz << ")$\\\\" << endl;
		{
		vector<int> cipher;

		cipher.push_back(Rx);
		cipher.push_back(Ry);
		cipher.push_back(Tx);
		cipher.push_back(Ty);
		Ciphertext.push_back(cipher);
		}

		cout << setw(4) << i << " & " << EC_message[i] << " & " << setw(4) << m << " & " << setw(4) << k
				<< "& (" << setw(4) << Mx << "," << setw(4) << My << "," << setw(4) << Mz << ") "
				<< "& (" << setw(4) << Rx << "," << setw(4) << Ry << "," << setw(4) << Rz << ") "
				<< "& (" << setw(4) << Cx << "," << setw(4) << Cy << "," << setw(4) << Cz << ") "
				<< "& (" << setw(4) << Tx << "," << setw(4) << Ty << "," << setw(4) << Tz << ") "
				<< "\\\\"
				<< endl;

	}

	cout << "Ciphertext:\\\\" << endl;
	for (i = 0; i < (int) Ciphertext.size(); i++) {
		cout << Ciphertext[i][0] << ",";
		cout << Ciphertext[i][1] << ",";
		cout << Ciphertext[i][2] << ",";
		cout << Ciphertext[i][3] << "\\\\" << endl;
	}

	for (i = 0; i < (int) Ciphertext.size(); i++) {
		Rx = Ciphertext[i][0];
		Ry = Ciphertext[i][1];
		Tx = Ciphertext[i][2];
		Ty = Ciphertext[i][3];

		// msR := -s * R
		NT.elliptic_curve_point_multiple(F,
					EC_b, EC_c, minus_s,
					Rx, Ry, Rz,
					msRx, msRy, msRz,
					0 /*verbose_level*/);

		// D := msR + T
		NT.elliptic_curve_addition(F,
				EC_b, EC_c,
				msRx, msRy, msRz,
				Tx, Ty, Tz,
				Dx, Dy, Dz,
				0 /*verbose_level*/);

		plain = Dx / u;

		cout << setw(4) << i << " & (" << Rx << "," << Ry << "," << Tx << "," << Ty << ") "
				<< "& (" << setw(4) << msRx << "," << setw(4) << msRy << "," << setw(4) << msRz << ") "
				<< "& (" << setw(4) << Dx << "," << setw(4) << Dy << "," << setw(4) << Dz << ") "
				<< " & " << plain << " & " << (char)('A' - 1 + plain)
				<< "\\\\"
				<< endl;

	}

	//cout << "nb_calls_to_elliptic_curve_addition="
	//		<< F->nb_calls_to_elliptic_curve_addition() << endl;


	if (f_v) {
		cout << "cryptography_domain::do_EC_Koblitz_encoding done" << endl;
	}
}

void cryptography_domain::do_EC_points(
		field_theory::finite_field *F,
		std::string &label,
		int EC_b, int EC_c,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_EC_points" << endl;
	}
	int x, y, r, y1, y2;
	vector<vector<int>> Pts;

	for (x = 0; x < F->q; x++) {
		r = EC_evaluate_RHS(F, EC_b, EC_c, x);
		if (r == 0) {

			{
				vector<int> pt;

				pt.push_back(x);
				pt.push_back(0);
				pt.push_back(1);
				Pts.push_back(pt);
			}
		}
		else {
			if (F->is_square(r)) {
				y = F->square_root(r);
				y1 = y;
				y2 = F->negate(y);
				if (y2 == y1) {
					{
						vector<int> pt;

						pt.push_back(x);
						pt.push_back(y1);
						pt.push_back(1);
						Pts.push_back(pt);
					}
				}
				else {
					if (y2 < y1) {
						y1 = y2;
						y2 = y;
					}
					{
						vector<int> pt;

						pt.push_back(x);
						pt.push_back(y1);
						pt.push_back(1);
						Pts.push_back(pt);
					}
					{
						vector<int> pt;

						pt.push_back(x);
						pt.push_back(y2);
						pt.push_back(1);
						Pts.push_back(pt);
					}
				}
			}
			else {
				// no point for this x coordinate
			}

#if 0
			if (p != 2) {
				l = Legendre(r, q, 0);

				if (l == 1) {
					y = sqrt_mod_involved(r, q);
						// DISCRETA/global.cpp

					if (F->mult(y, y) != r) {
						cout << "There is a problem "
								"with the square root" << endl;
						exit(1);
					}
					y1 = y;
					y2 = F->negate(y);
					if (y2 < y1) {
						y1 = y2;
						y2 = y;
					}
					add_point_to_table(x, y1, 1);
					if (nb == bound) {
						cout << "The number of points "
								"exceeds the bound" << endl;
						exit(1);
					}
					add_point_to_table(x, y2, 1);
					if (nb == bound) {
						cout << "The number of points "
								"exceeds the bound" << endl;
						exit(1);
					}
					//cout << nb++ << " : (" << x << ","
					// << y << ",1)" << endl;
					//cout << nb++ << " : (" << x << ","
					// << F.negate(y) << ",1)" << endl;
				}
			}
			else {
				y = F->frobenius_power(r, e - 1);
				add_point_to_table(x, y, 1);
				if (nb == bound) {
					cout << "The number of points exceeds "
							"the bound" << endl;
					exit(1);
				}
				//cout << nb++ << " : (" << x << ","
				// << y << ",1)" << endl;
			}
#endif

		}
	}
	{
		vector<int> pt;

		pt.push_back(0);
		pt.push_back(1);
		pt.push_back(0);
		Pts.push_back(pt);
	}
	int i;
	cout << "We found " << Pts.size() << " points:" << endl;


	for (i = 0; i < (int) Pts.size(); i++) {
		if (i == (int) Pts.size()) {

			cout << i << " : {\\cal O} : 1\\\\" << endl;

		}
		else {
			{
			vector<vector<int>> Multiples;
			int order;
			number_theory::number_theory_domain NT;

			NT.elliptic_curve_all_point_multiples(F,
					EC_b, EC_c, order,
					Pts[i][0], Pts[i][1], 1,
					Multiples,
					0 /*verbose_level*/);

			//cout << "we found that the point has order " << order << endl;

			cout << i << " : $(" << Pts[i][0] << "," << Pts[i][1] << ")$ : " << order << "\\\\" << endl;
			}
		}
	}

	{
		int *M;

		M = NEW_int(F->q * F->q);
		Int_vec_zero(M, F->q * F->q);


		for (i = 0; i < (int) Pts.size(); i++) {
			vector<int> pt;
			int x, y, z;

			pt = Pts[i];
			x = pt[0];
			y = pt[1];
			z = pt[2];
			if (z == 1) {
				M[(F->q - 1 - y) * F->q + x] = 1;
			}
		}
		string fname;
		orbiter_kernel_system::file_io Fio;

		fname.assign(label);
		fname.append("_points_xy.csv");
		Fio.int_matrix_write_csv(fname, M, F->q, F->q);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		FREE_int(M);
	}

	{
		int *M;
		int cnt = 0;

		M = NEW_int((int) Pts.size() * 2);
		Int_vec_zero(M, (int) Pts.size() * 2);


		for (i = 0; i < (int) Pts.size(); i++) {
			vector<int> pt;
			int x, y, z;

			pt = Pts[i];
			x = pt[0];
			y = pt[1];
			z = pt[2];
			if (z == 1) {
				M[cnt * 2 + 0] = x;
				M[cnt * 2 + 1] = y;
				cnt++;
			}
		}
		string fname;
		orbiter_kernel_system::file_io Fio;

		fname.assign(label);
		fname.append("_points_xy_affine_pts.csv");
		Fio.int_matrix_write_csv(fname, M, cnt, 2);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		FREE_int(M);
	}


	if (f_v) {
		cout << "cryptography_domain::do_EC_points done" << endl;
	}
}

int cryptography_domain::EC_evaluate_RHS(
		field_theory::finite_field *F,
		int EC_b, int EC_c, int x)
// evaluates x^3 + bx + c
{
	int x2, x3, t;

	x2 = F->mult(x, x);
	x3 = F->mult(x2, x);
	t = F->add(x3, F->mult(EC_b, x));
	t = F->add(t, EC_c);
	return t;
}


void cryptography_domain::do_EC_add(
		field_theory::finite_field *F,
		int EC_b, int EC_c,
		std::string &pt1_text,
		std::string &pt2_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int x1, y1, z1;
	int x2, y2, z2;
	int x3, y3, z3;
	int *v;
	int len;
	number_theory::number_theory_domain NT;
	//sscanf(p1, "(%d,%d,%d)", &x1, &y1, &z1);

	if (f_v) {
		cout << "cryptography_domain::do_EC_add" << endl;
	}
	vector<vector<int>> Pts;

	Int_vec_scan(pt1_text, v, len);
	if (len != 2) {
		cout << "cryptography_domain::do_EC_add "
				"point should have just two coordinates" << endl;
		exit(1);
	}
	x1 = v[0];
	y1 = v[1];
	z1 = 1;
	FREE_int(v);

	Int_vec_scan(pt2_text, v, len);
	if (len != 2) {
		cout << "cryptography_domain::do_EC_add "
				"point should have just two coordinates" << endl;
		exit(1);
	}
	x2 = v[0];
	y2 = v[1];
	z2 = 1;
	FREE_int(v);


	NT.elliptic_curve_addition(F,
			EC_b, EC_c,
			x1, y1, z1,
			x2, y2, z2,
			x3, y3, z3,
			verbose_level);
	cout << "(" << x1 << "," << y1 << "," << z1 << ")";
	cout << " + ";
	cout << "(" << x2 << "," << y2 << "," << z2 << ")";
	cout << " = ";
	cout << "(" << x3 << "," << y3 << "," << z3 << ")";
	cout << endl;


	//FREE_OBJECT(F);

	if (f_v) {
		cout << "cryptography_domain::do_EC_add done" << endl;
	}
}

void cryptography_domain::do_EC_cyclic_subgroup(
		field_theory::finite_field *F,
		int EC_b, int EC_c, std::string &pt_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_EC_cyclic_subgroup" << endl;
	}

	int x1, y1, z1;
	int *v;
	int len, i;
	number_theory::number_theory_domain NT;
	//sscanf(p1, "(%d,%d,%d)", &x1, &y1, &z1);

	vector<vector<int>> Pts;
	int order;

	Int_vec_scan(pt_text, v, len);
	if (len != 2) {
		cout << "cryptography_domain::do_EC_cyclic_subgroup "
				"point should have just two coordinates" << endl;
		exit(1);
	}
	x1 = v[0];
	y1 = v[1];
	z1 = 1;
	FREE_int(v);


	NT.elliptic_curve_all_point_multiples(F,
			EC_b, EC_c, order,
			x1, y1, z1,
			Pts,
			verbose_level);

	cout << "cryptography_domain::do_EC_cyclic_subgroup "
			"we found that the point has order " << order << endl;
	cout << "The multiples are:" << endl;
	cout << "i : (" << x1 << "," << y1 << ")" << endl;
	for (i = 0; i < (int) Pts.size(); i++) {

		vector<int> pts = Pts[i];

		if (i < (int) Pts.size() - 1) {
			cout << setw(3) << i + 1 << " : ";
			cout << "$(" << pts[0] << "," << pts[1] << ")$";
			cout << "\\\\" << endl;
		}
		else {
			cout << setw(3) << i + 1 << " : ";
			cout << "${\\cal O}$";
			cout << "\\\\" << endl;

		}
	}

	if (f_v) {
		cout << "cryptography_domain::do_EC_cyclic_subgroup done" << endl;
	}
}

void cryptography_domain::do_EC_multiple_of(
		field_theory::finite_field *F,
		int EC_b, int EC_c,
		std::string &pt_text, int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_EC_multiple_of" << endl;
	}

	int x1, y1, z1;
	int x3, y3, z3;
	int *v;
	int len;
	number_theory::number_theory_domain NT;

	Int_vec_scan(pt_text, v, len);
	if (len != 2) {
		cout << "cryptography_domain::do_EC_multiple_of "
				"point should have just two coordinates" << endl;
		exit(1);
	}
	x1 = v[0];
	y1 = v[1];
	z1 = 1;
	FREE_int(v);


	NT.elliptic_curve_point_multiple(F,
			EC_b, EC_c, n,
			x1, y1, z1,
			x3, y3, z3,
			verbose_level);

	cout << "The " << n << "-fold multiple of (" << x1 << "," << y1 << ") is ";
	if (z3 == 0) {

	}
	else {
		if (z3 != 1) {
			cout << "z1 != 1" << endl;
			exit(1);
		}
		cout << "(" << x3 << "," << y3 << ")" << endl;
	}

	if (f_v) {
		cout << "cryptography_domain::do_EC_multiple_of done" << endl;
	}
}

void cryptography_domain::do_EC_discrete_log(
		field_theory::finite_field *F,
		int EC_b, int EC_c,
		std::string &base_pt_text,
		std::string &pt_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_EC_discrete_log" << endl;
	}

	int x1, y1, z1;
	int x3, y3, z3;
	int *v;
	int len;
	int n;
	number_theory::number_theory_domain NT;

	Int_vec_scan(base_pt_text, v, len);
	if (len != 2) {
		cout << "point should have just two coordinates" << endl;
		exit(1);
	}
	x1 = v[0];
	y1 = v[1];
	z1 = 1;
	FREE_int(v);


	Int_vec_scan(pt_text, v, len);
	if (len == 2) {
		x3 = v[0];
		y3 = v[1];
		z3 = 1;
	}
	else if (len == 3) {
		x3 = v[0];
		y3 = v[1];
		z3 = v[2];
	}
	else {
		cout << "the point should have either two or three coordinates" << endl;
		exit(1);
	}
	FREE_int(v);


	n = NT.elliptic_curve_discrete_log(F,
			EC_b, EC_c,
			x1, y1, z1,
			x3, y3, z3,
			verbose_level);


	cout << "The discrete log of (" << x3 << "," << y3 << "," << z3 << ") "
			"w.r.t. (" << x1 << "," << y1 << "," << z1 << ") "
			"is " << n << endl;

	if (f_v) {
		cout << "cryptography_domain::do_EC_discrete_log done" << endl;
	}
}

void cryptography_domain::do_EC_baby_step_giant_step(
		field_theory::finite_field *F,
		int EC_b, int EC_c,
		std::string &EC_bsgs_G, int EC_bsgs_N,
		std::string &EC_bsgs_cipher_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_EC_baby_step_giant_step" << endl;
	}

	int Gx, Gy, Gz;
	int nGx, nGy, nGz;
	int Cx, Cy, Cz;
	int Mx, My, Mz;
	int Ax, Ay, Az;
	int *v;
	int len;
	int n;
	number_theory::number_theory_domain NT;


	Int_vec_scan(EC_bsgs_G, v, len);
	if (len != 2) {
		cout << "point should have just two coordinates" << endl;
		exit(1);
	}
	Gx = v[0];
	Gy = v[1];
	Gz = 1;
	FREE_int(v);

	n = (int) sqrt((double) EC_bsgs_N) + 1;
	if (f_v) {
		cout << "cryptography_domain::do_EC_baby_step_giant_step N = " << EC_bsgs_N << endl;
		cout << "cryptography_domain::do_EC_baby_step_giant_step n = " << n << endl;
	}

	Int_vec_scan(EC_bsgs_cipher_text, v, len);

	int cipher_text_length = len >> 1;
	int h, i;

	if (f_v) {
		cout << "cryptography_domain::do_EC_baby_step_giant_step "
				"cipher_text_length = " << cipher_text_length << endl;
	}

	NT.elliptic_curve_point_multiple(F,
			EC_b, EC_c, n,
			Gx, Gy, Gz,
			nGx, nGy, nGz,
			0 /*verbose_level*/);

	cout << "$" << n << " * G = (" << nGx << "," << nGy << ")$\\\\" << endl;

	cout << " & ";
	for (h = 0; h < cipher_text_length; h++) {
		Cx = v[2 * h + 0];
		Cy = v[2 * h + 1];
		Cz = 1;
		cout << " & (" << Cx << "," << Cy << ")";
	}
	cout << endl;

	for (i = 1; i <= n + 1; i++) {

		NT.elliptic_curve_point_multiple(F,
				EC_b, EC_c, i,
				Gx, Gy, Gz,
				Mx, My, Mz,
				0 /*verbose_level*/);

		cout << i << " & (" << Mx << "," << My << ")";

		for (h = 0; h < cipher_text_length; h++) {
			Cx = v[2 * h + 0];
			Cy = v[2 * h + 1];
			Cz = 1;

			NT.elliptic_curve_point_multiple(F,
					EC_b, EC_c, i,
					nGx, nGy, nGz,
					Mx, My, Mz,
					0 /*verbose_level*/);

			My = F->negate(My);



			NT.elliptic_curve_addition(F,
					EC_b, EC_c,
					Cx, Cy, Cz,
					Mx, My, Mz,
					Ax, Ay, Az,
					0 /*verbose_level*/);

			cout << " & (" << Ax << "," << Ay << ")";

		}
		cout << "\\\\" << endl;
	}



	FREE_int(v);

	if (f_v) {
		cout << "cryptography_domain::do_EC_baby_step_giant_step done" << endl;
	}
}

void cryptography_domain::do_EC_baby_step_giant_step_decode(
		field_theory::finite_field *F,
		int EC_b, int EC_c,
		std::string &EC_bsgs_A, int EC_bsgs_N,
		std::string &EC_bsgs_cipher_text, std::string &EC_bsgs_keys,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_EC_baby_step_giant_step_decode" << endl;
	}

	int Ax, Ay, Az;
	int Tx, Ty, Tz;
	int Cx, Cy, Cz;
	int Mx, My, Mz;
	int *v;
	int len;
	int n;
	int *keys;
	int nb_keys;
	int u, plain;
	number_theory::number_theory_domain NT;


	u = F->q / 27;
	if (f_v) {
		cout << "cryptography_domain::do_EC_baby_step_giant_step_decode u = " << u << endl;
	}


	Int_vec_scan(EC_bsgs_A, v, len);
	if (len != 2) {
		cout << "cryptography_domain::do_EC_baby_step_giant_step_decode "
				"point should have just two coordinates" << endl;
		exit(1);
	}
	Ax = v[0];
	Ay = v[1];
	Az = 1;
	FREE_int(v);

	Int_vec_scan(EC_bsgs_keys, keys, nb_keys);


	n = (int) sqrt((double) EC_bsgs_N) + 1;
	if (f_v) {
		cout << "cryptography_domain::do_EC_baby_step_giant_step_decode N = " << EC_bsgs_N << endl;
		cout << "cryptography_domain::do_EC_baby_step_giant_step_decode n = " << n << endl;
	}

	Int_vec_scan(EC_bsgs_cipher_text, v, len);

	int cipher_text_length = len >> 1;
	int h;

	if (f_v) {
		cout << "cryptography_domain::do_EC_baby_step_giant_step_decode "
				"cipher_text_length = " << cipher_text_length << endl;
		cout << "cryptography_domain::do_EC_baby_step_giant_step_decode "
				"nb_keys = " << nb_keys << endl;
	}
	if (nb_keys != cipher_text_length) {
		cout << "nb_keys != cipher_text_length" << endl;
		exit(1);
	}


	for (h = 0; h < cipher_text_length; h++) {
		Tx = v[2 * h + 0];
		Ty = v[2 * h + 1];
		Tz = 1;
		cout << h << " & (" << Tx << "," << Ty << ")\\\\" << endl;;
	}
	cout << endl;


	for (h = 0; h < cipher_text_length; h++) {



		Tx = v[2 * h + 0];
		Ty = v[2 * h + 1];
		Tz = 1;


		NT.elliptic_curve_point_multiple(F,
				EC_b, EC_c, keys[h],
				Ax, Ay, Az,
				Cx, Cy, Cz,
				0 /*verbose_level*/);

		Cy = F->negate(Cy);


		cout << h << " & " << keys[h]
			<< " & (" << Tx << "," << Ty << ")"
			<< " & (" << Cx << "," << Cy << ")";


		NT.elliptic_curve_addition(F,
				EC_b, EC_c,
				Tx, Ty, Tz,
				Cx, Cy, Cz,
				Mx, My, Mz,
				0 /*verbose_level*/);

		cout << " & (" << Mx << "," << My << ")";

		plain = Mx / u;
		cout << " & " << plain << " & " << (char)('A' - 1 + plain) << "\\\\" << endl;

	}


	FREE_int(v);
	FREE_int(keys);

	if (f_v) {
		cout << "cryptography_domain::do_EC_baby_step_giant_step_decode done" << endl;
	}
}

void cryptography_domain::do_RSA_encrypt_text(
		long int RSA_d, long int RSA_m,
		int RSA_block_size,
		std::string &RSA_encrypt_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_RSA_encrypt_text" << endl;
	}

	int i, j, l, nb_blocks;
	long int a;
	char c;
	long int *Data;

	l = RSA_encrypt_text.length();
	nb_blocks = (l + RSA_block_size - 1) /  RSA_block_size;
	Data = NEW_lint(nb_blocks);
	for (i = 0; i < nb_blocks; i++) {
		a = 0;
		for (j = 0; j < RSA_block_size; j++) {
			c = RSA_encrypt_text[i * RSA_block_size + j];
			if (c >= 'a' && c <= 'z') {
				a *= 100;
				a += (int) (c - 'a') + 1;
			}
		Data[i] = a;
		}
	}

	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object A, M;

	M.create(RSA_m);

	for (i = 0; i < nb_blocks; i++) {
		A.create(Data[i]);
		D.power_int_mod(
				A, RSA_d, M);
		cout << A;
		if (i < nb_blocks - 1) {
			cout << ",";
		}
	}
	cout << endl;
	if (f_v) {
		cout << "cryptography_domain::do_RSA_encrypt_text done" << endl;
	}
}

void cryptography_domain::do_RSA(
		long int RSA_d,
		long int RSA_m, int RSA_block_size,
		std::string &RSA_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *data;
	int data_sz;
	int i;

	if (f_v) {
		cout << "cryptography_domain::do_RSA RSA_d=" << RSA_d << " RSA_m=" << RSA_m << endl;
	}
	Lint_vec_scan(RSA_text, data, data_sz);
	if (f_v) {
		cout << "text: ";
		Lint_vec_print(cout, data, data_sz);
		cout << endl;
	}

	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object A, M;

	M.create(RSA_m);
	for (i = 0; i < data_sz; i++) {
		A.create(data[i]);
		D.power_int_mod(A, RSA_d, M);
		if (f_v) {
			cout << i << " : " << data[i] << " : " << A << endl;
		}
	}
	for (i = 0; i < data_sz; i++) {
		A.create(data[i]);
		D.power_int_mod(A, RSA_d, M);
		if (f_v) {
			cout << A;
			if (i < data_sz - 1) {
				cout << ",";
			}
		}
	}
	if (f_v) {
		cout << endl;
	}

	long int a;
	int b, j;
	char str[1000];

	for (i = 0; i < data_sz; i++) {
		A.create(data[i]);
		D.power_int_mod(A, RSA_d, M);
		//cout << A;
		a = A.as_lint();
		for (j = 0; j < RSA_block_size; j++) {
			b = a % 100;
			if (b > 26 || b == 0) {
				str[RSA_block_size - 1 - j] = ' ';
			}
			else {
				str[RSA_block_size - 1 - j] = 'a' + b - 1;
			}
			a -= b;
			a /= 100;
		}
		str[RSA_block_size] = 0;
		for (j = 0; j < RSA_block_size; j++) {
			if (str[j] != ' ') {
				break;
			}
		}
		if (f_v) {
			cout << str + j;
			if (i < data_sz - 1) {
				cout << ",";
			}
		}
	}
	if (f_v) {
		cout << endl;
	}
}


void cryptography_domain::NTRU_encrypt(
		int N, int p,
		field_theory::finite_field *Fq,
		std::string &H_coeffs,
		std::string &R_coeffs,
		std::string &Msg_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::NTRU_encrypt" << endl;
	}


	int *data_H;
	int *data_R;
	int *data_Msg;
	int sz_H, sz_R, sz_Msg;

	Int_vec_scan(H_coeffs, data_H, sz_H);
	Int_vec_scan(R_coeffs, data_R, sz_R);
	Int_vec_scan(Msg_coeffs, data_Msg, sz_Msg);

	number_theory::number_theory_domain NT;



	ring_theory::unipoly_domain FX(Fq);
	ring_theory::unipoly_object H, R, Msg, M, C, D;


	int dh = sz_H - 1;
	int dr = sz_R - 1;
	int dm = sz_Msg - 1;
	int i;

	FX.create_object_of_degree(H, dh);

	for (i = 0; i <= dh; i++) {
		if (data_H[i] < 0 || data_H[i] >= Fq->q) {
			data_H[i] = NT.mod(data_H[i], Fq->q);
		}
		FX.s_i(H, i) = data_H[i];
	}

	FX.create_object_of_degree(R, dr);

	for (i = 0; i <= dr; i++) {
		if (data_R[i] < 0 || data_R[i] >= Fq->q) {
			data_R[i] = NT.mod(data_R[i], Fq->q);
		}
		FX.s_i(R, i) = data_R[i];
	}

	FX.create_object_of_degree(Msg, dm);

	for (i = 0; i <= dm; i++) {
		if (data_Msg[i] < 0 || data_Msg[i] >= Fq->q) {
			data_Msg[i] = NT.mod(data_Msg[i], Fq->q);
		}
		FX.s_i(Msg, i) = data_Msg[i];
	}

	FX.create_object_of_degree(M, N);
	for (i = 0; i <= N; i++) {
		FX.s_i(M, i) = 0;
	}
	FX.s_i(M, 0) = Fq->negate(1);
	FX.s_i(M, N) = 1;

	if (f_v) {
		cout << "H(X)=";
		FX.print_object(H, cout);
		cout << endl;


		cout << "R(X)=";
		FX.print_object(R, cout);
		cout << endl;

		cout << "Msg(X)=";
		FX.print_object(Msg, cout);
		cout << endl;
	}

	FX.create_object_of_degree(C, dh);

	FX.create_object_of_degree(D, dh);



	if (f_v) {
		cout << "cryptography_domain::NTRU_encrypt before FX.mult_mod" << endl;
	}

	{
		FX.mult_mod(R, H, C, M, verbose_level);
		int d;

		d = FX.degree(C);

		for (i = 0; i <= d; i++) {
			FX.s_i(C, i) = Fq->mult(p, FX.s_i(C, i));
		}

		FX.add(C, Msg, D);

	}

	if (f_v) {
		cout << "cryptography_domain::NTRU_encrypt after FX.mult_mod" << endl;
	}

	if (f_v) {
		cout << "D(X)=";
		FX.print_object(D, cout);
		cout << endl;

		cout << "deg D(X) = " << FX.degree(D) << endl;
	}





	if (f_v) {
		cout << "cryptography_domain::NTRU_encrypt done" << endl;
	}
}


void cryptography_domain::polynomial_center_lift(
		std::string &A_coeffs,
		field_theory::finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::polynomial_center_lift" << endl;
	}


	int *data_A;
	int sz_A;

	Int_vec_scan(A_coeffs, data_A, sz_A);

	number_theory::number_theory_domain NT;



	ring_theory::unipoly_domain FX(F);
	ring_theory::unipoly_object A;


	int da = sz_A - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		if (data_A[i] < 0 || data_A[i] >= F->q) {
			data_A[i] = NT.mod(data_A[i], F->q);
		}
		FX.s_i(A, i) = data_A[i];
	}


	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;
	}



	if (f_v) {
		cout << "cryptography_domain::polynomial_center_lift before FX.mult_mod" << endl;
	}

	{
		FX.center_lift_coordinates(A, F->q);

	}

	if (f_v) {
		cout << "cryptography_domain::polynomial_center_lift after FX.mult_mod" << endl;
	}

	if (f_v) {
		cout << "A(X)=";
		FX.print_object(A, cout);
		cout << endl;
	}



	if (f_v) {
		cout << "cryptography_domain::polynomial_center_lift done" << endl;
	}
}


void cryptography_domain::polynomial_reduce_mod_p(
		std::string &A_coeffs,
		field_theory::finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::polynomial_reduce_mod_p" << endl;
	}


	int *data_A;
	int sz_A;

	Int_vec_scan(A_coeffs, data_A, sz_A);

	number_theory::number_theory_domain NT;



	ring_theory::unipoly_domain FX(F);
	ring_theory::unipoly_object A;


	int da = sz_A - 1;
	int i;

	FX.create_object_of_degree(A, da);

	for (i = 0; i <= da; i++) {
		data_A[i] = NT.mod(data_A[i], F->q);
		FX.s_i(A, i) = data_A[i];
	}


	cout << "A(X)=";
	FX.print_object(A, cout);
	cout << endl;




	if (f_v) {
		cout << "cryptography_domain::polynomial_reduce_mod_p done" << endl;
	}
}




void cryptography_domain::do_solovay_strassen(
		int p, int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_solovay_strassen" << endl;
	}
	string fname;
	string author;
	string title;
	string extra_praeamble;



	fname = "solovay_strassen_" + std::to_string(p) + "_" + std::to_string(a) + ".tex";
	title = "Solovay Strassen " + std::to_string(p) + " with base " + std::to_string(a);



	{
	ofstream f(fname);


	l1_interfaces::latex_interface L;


	L.head(f, false /* f_book*/, true /* f_title */,
		title, author, false /* f_toc */, false /* f_landscape */,
			true /* f_12pt */,
			true /* f_enlarged_page */,
			true /* f_pagenumbers */,
			extra_praeamble /* extra_praeamble */);


	number_theory::number_theory_domain NT;
	//longinteger_domain D;

	ring_theory::longinteger_object P, A;

	P.create(p);

	A.create(a);

	//D.jacobi(A, B, verbose_level);

	solovay_strassen_test_with_latex_key(f,
			P, A,
			verbose_level);


	L.foot(f);
	}

	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "cryptography_domain::do_solovay_strassen done" << endl;
	}

}

void cryptography_domain::do_miller_rabin(
		int p, int nb_times, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_miller_rabin" << endl;
	}
	string fname;
	string author;
	string title;
	string extra_praeamble;


	fname = "miller_rabin_" + std::to_string(p) + ".tex";
	title = "Miller Rabin " + std::to_string(p);



	{
		ofstream f(fname);


		l1_interfaces::latex_interface L;


		L.head(f, false /* f_book*/, true /* f_title */,
			title, author, false /* f_toc */, false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);


		//longinteger_domain D;

		ring_theory::longinteger_object P, A;

		P.create(p);

		int i;

		for (i = 0; i < nb_times; i++) {

			f << "Miller Rabin test no " << i << ":\\\\" << endl;
			if (!miller_rabin_test_with_latex_key(f,
				P, i,
				verbose_level)) {
				break;
			}

		}
		if (i == nb_times) {
			f << "Miller Rabin: The number is probably prime. Miller Rabin is inconclusive.\\\\" << endl;
		}
		else {
			f << "Miller Rabin: The number is not prime.\\\\" << endl;
		}

		L.foot(f);
	}

	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "cryptography_domain::do_miller_rabin done" << endl;
	}

}

void cryptography_domain::do_fermat_test(
		int p, int nb_times, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_fermat_test" << endl;
	}
	string fname;
	string author;
	string title;
	string extra_praeamble;


	fname = "fermat_" + std::to_string(p) + ".tex";
	title = "Fermat test " + std::to_string(p);


	{
		ofstream f(fname);


		l1_interfaces::latex_interface L;


		L.head(f, false /* f_book*/, true /* f_title */,
			title, author, false /* f_toc */, false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);


		//longinteger_domain D;
		ring_theory::longinteger_object P;


		P.create(p);

		if (fermat_test_iterated_with_latex_key(f,
				P, nb_times,
				verbose_level)) {
			f << "Fermat: The number $" << P << "$ is not prime.\\\\" << endl;
		}
		else {
			f << "Fermat: The number $" << P << "$ is probably prime. Fermat test is inconclusive.\\\\" << endl;
		}

		L.foot(f);
	}

	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "cryptography_domain::do_fermat_test done" << endl;
	}


}

void cryptography_domain::do_find_pseudoprime(
		int nb_digits,
		int nb_fermat,
		int nb_miller_rabin,
		int nb_solovay_strassen,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_find_pseudoprime" << endl;
	}
	string fname;
	string author;
	string title;
	string extra_praeamble;


	fname = "pseudoprime_" + std::to_string(nb_digits) + ".tex";
	title = "Pseudoprime " + std::to_string(nb_digits);



	{
		ofstream ost(fname);


		l1_interfaces::latex_interface L;


		L.head(ost, false /* f_book*/, true /* f_title */,
			title, author, false /* f_toc */, false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);


		ring_theory::longinteger_domain D;
		ring_theory::longinteger_object P;


		int cnt = -1;

		//f << "\\begin{multicols}{2}" << endl;
		ost << "\\begin{enumerate}[(1)]" << endl;
		while (true) {

			cnt++;

			D.random_number_with_n_decimals(P, nb_digits, verbose_level);

			ost << "\\item" << endl;
			ost << "Trial " << cnt << ", testing random number " << P << endl;

			if (P.ith(0) != 1 && P.ith(0) != 3 && P.ith(0) != 7 && P.ith(0) != 9) {
				ost << "the number is not prime by looking at the lowest digit." << endl;
				continue;
			}

			ost << "\\begin{enumerate}[(a)]" << endl;
			ost << "\\item" << endl;
			if (fermat_test_iterated_with_latex_key(ost,
					P, nb_fermat,
					verbose_level)) {
				//f << "Fermat: The number $" << P << "$ is not prime.\\\\" << endl;
				ost << "\\end{enumerate}" << endl;
				continue;
			}
			else {
				//f << "Fermat: The number $" << P << "$ is probably prime. Fermat test is inconclusive.\\\\" << endl;
			}


			if (nb_miller_rabin) {
				ost << "\\item" << endl;
				if (miller_rabin_test_iterated_with_latex_key(ost,
						P, nb_miller_rabin,
						verbose_level)) {
					ost << "Miller Rabin: The number $" << P << "$ is not prime.\\\\" << endl;
					ost << "\\end{enumerate}" << endl;
					continue;
				}
				else {
					//ost << "Miller Rabin: The number $" << P << "$ is probably prime. Miller Rabin test is inconclusive.\\\\" << endl;
				}
			}
			else {
				ost << "\\end{enumerate}" << endl;
				break;
			}

			if (nb_solovay_strassen) {
				ost << "\\item" << endl;
				if (solovay_strassen_test_iterated_with_latex_key(ost,
						P, nb_solovay_strassen,
						verbose_level)) {
					//ost << "Solovay-Strassen: The number $" << P << "$ is not prime.\\\\" << endl;
					ost << "\\end{enumerate}" << endl;
					continue;
				}
				else {
					//ost << "Solovay-Strassen: The number $" << P << "$ is probably prime. Solovay-Strassen test is inconclusive.\\\\" << endl;
					ost << "\\end{enumerate}" << endl;
					break;
				}
			}
			else {
				ost << "\\end{enumerate}" << endl;
				break;
			}
			ost << "\\end{enumerate}" << endl;

		}
		ost << "\\end{enumerate}" << endl;
		//ost << "\\end{multicols}" << endl;

		ost << "\\noindent" << endl;
		ost << "The number $" << P << "$ is probably prime. \\\\" << endl;
		ost << "Number of Fermat tests = " << nb_fermat << " \\\\" << endl;
		ost << "Number of Miller Rabin tests = " << nb_miller_rabin << " \\\\" << endl;
		ost << "Number of Solovay-Strassen tests = " << nb_solovay_strassen << " \\\\" << endl;

		L.foot(ost);
	}

	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "cryptography_domain::do_find_pseudoprime done" << endl;
	}

}

void cryptography_domain::do_find_strong_pseudoprime(
		int nb_digits, int nb_fermat, int nb_miller_rabin,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_find_strong_pseudoprime" << endl;
	}
	string fname;
	string author;
	string title;
	string extra_praeamble;


	fname = "strong_pseudoprime_" + std::to_string(nb_digits) + ".tex";
	title = "Strong Pseudoprime " + std::to_string(nb_digits);


	{
		ofstream f(fname);


		l1_interfaces::latex_interface L;


		L.head(f, false /* f_book*/, true /* f_title */,
			title, author, false /* f_toc */, false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);


		ring_theory::longinteger_domain D;
		ring_theory::longinteger_object P;


		int cnt = -1;

		f << "\\begin{multicols}{2}" << endl;
		f << "\\begin{enumerate}[(1)]" << endl;
		while (true) {

			cnt++;

			D.random_number_with_n_decimals(P, nb_digits, verbose_level);

			f << "\\item" << endl;
			f << "Trial " << cnt << ", testing random number " << P << endl;

			if (P.ith(0) != 1 && P.ith(0) != 3 && P.ith(0) != 7 && P.ith(0) != 9) {
				f << "the number is not prime by looking at the lowest digit" << endl;
				continue;
			}

			f << "\\begin{enumerate}[(a)]" << endl;
			f << "\\item" << endl;
			if (fermat_test_iterated_with_latex_key(f,
					P, nb_fermat,
					verbose_level)) {
				//f << "Fermat: The number $" << P << "$ is not prime.\\\\" << endl;
				f << "\\end{enumerate}" << endl;
				continue;
			}
			else {
				//f << "Fermat: The number $" << P << "$ is probably prime. Fermat test is inconclusive.\\\\" << endl;
			}

			f << "\\item" << endl;
			if (miller_rabin_test_iterated_with_latex_key(f,
					P, nb_miller_rabin,
					verbose_level)) {
				//f << "Miller Rabin: The number $" << P << "$ is not prime.\\\\" << endl;
				f << "\\end{enumerate}" << endl;
				continue;
			}
			else {
				//f << "Miller Rabin: The number $" << P << "$ is probably prime. Miller Rabin test is inconclusive.\\\\" << endl;
				f << "\\end{enumerate}" << endl;
				break;
			}


			f << "\\end{enumerate}" << endl;

		}
		f << "\\end{enumerate}" << endl;
		f << "\\end{multicols}" << endl;

		f << "\\noindent" << endl;
		f << "The number $" << P << "$ is probably prime. \\\\" << endl;
		f << "Number of Fermat tests = " << nb_fermat << " \\\\" << endl;
		f << "Number of Miller Rabin tests = " << nb_miller_rabin << " \\\\" << endl;

		L.foot(f);
	}

	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "cryptography_domain::do_find_strong_pseudoprime done" << endl;
	}

}


void cryptography_domain::do_miller_rabin_text(std::string &number_text,
		int nb_miller_rabin, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cryptography_domain::do_miller_rabin_text" << endl;
	}
	string fname;
	string author;
	string title;
	string extra_praeamble;

	fname = "miller_rabin_" + number_text + ".tex";
	title = "Miller Rabin " + number_text;


	{
		ofstream f(fname);


		l1_interfaces::latex_interface L;


		L.head(f, false /* f_book*/, true /* f_title */,
			title, author, false /* f_toc */, false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);


		//longinteger_domain D;
		ring_theory::longinteger_object P;


		f << "\\begin{multicols}{2}" << endl;

		P.create_from_base_10_string(number_text);


		if (P.ith(0) != 1 && P.ith(0) != 3 && P.ith(0) != 7 && P.ith(0) != 9) {
			f << "the number is not prime by looking at the lowest digit" << endl;
		}
		else {

			if (miller_rabin_test_iterated_with_latex_key(f,
					P, nb_miller_rabin,
					verbose_level)) {
				f << "Miller Rabin: The number $" << P << "$ is not prime.\\\\" << endl;
			}
			else {
				f << "The number $" << P << "$ is probably prime. \\\\" << endl;
			}
		}


		f << "\\end{multicols}" << endl;

		f << "\\noindent" << endl;
		f << "Number of Miller Rabin tests = " << nb_miller_rabin << " \\\\" << endl;

		L.foot(f);
	}

	orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "cryptography_domain::do_miller_rabin_text done" << endl;
	}


}

void cryptography_domain::quadratic_sieve(int n,
		int factorbase, int x0, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	vector<int> small_factors, primes, primes_log2, R1, R2;
	ring_theory::longinteger_object M, sqrtM;
	ring_theory::longinteger_domain D;
	number_theory::number_theory_domain NT;
	int f_found_small_factor = false;
	int small_factor;
	int i;

	if (f_v) {
		cout << "cryptography_domain::quadratic_sieve" << endl;
	}
	if (f_v) {
		cout << "cryptography_domain::quadratic_sieve before sieve" << endl;
	}
	NT.sieve(primes, factorbase, verbose_level - 1);
	if (f_v) {
		cout << "cryptography_domain::quadratic_sieve after sieve" << endl;
		cout << "list of primes has length " << primes.size() << endl;
	}
	D.create_Mersenne(M, n);
	if (f_v) {
		cout << "cryptography_domain::quadratic_sieve "
				"Mersenne number M_" << n << "=" << M << " log10=" << M.len() << endl;
	}
	D.square_root(M, sqrtM, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "cryptography_domain::quadratic_sieve "
				"sqrtM=" << sqrtM << " log10=" << sqrtM.len() << endl;
	}

	// N1.mult(sqrtM, sqrtM);
	// sqrtM.inc();
	// N2.mult(sqrtM, sqrtM);
	// sqrtM.dec();
	// cout << "sqrtM^2=" << N1 << endl;
	// cout << "M=" << M << endl;
	// cout << "(sqrtM+1)^2=" << N2 << endl;
	// longinteger_compare_verbose(N1, M);
	// longinteger_compare_verbose(M, N2);

	if (f_v) {
		cout << "cryptography_domain::quadratic_sieve "
				"calling reduce_primes" << endl;
	}
	//small_primes.m_l(0);
	while (true) {
		//int p;
		reduce_primes(primes, M,
				f_found_small_factor, small_factor,
				verbose_level - 1);
		if (!f_found_small_factor) {
			break;
		}
		ring_theory::longinteger_object P, Q;

		if (f_v) {
			cout << "cryptography_domain::quadratic_sieve "
					"dividing out small factor " << small_factor << endl;
		}
		small_factors.push_back(small_factor);
		P.create(small_factor);
		D.integral_division_exact(M, P, Q);
		Q.assign_to(M);
		if (f_v) {
			cout << "cryptography_domain::quadratic_sieve "
					"reduced M=" << M << " log10=" << M.len() << endl;
		}
		D.square_root(M, sqrtM, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "cryptography_domain::quadratic_sieve "
					"sqrtM=" << sqrtM << endl << endl;
		}
	}
	if (f_v) {
		cout << "cryptography_domain::quadratic_sieve "
				"list of small factors has length " << small_factors.size() << endl;
		for (i = 0; i < (int) small_factors.size(); i++) {
			cout << i << " : " << small_factors[i] << endl;
		}
	}

	if (M.is_one()) {
		cout << "cryptography_domain::quadratic_sieve "
				"the number has been completely factored" << endl;
		exit(0);
	}


	calc_roots(M, sqrtM, primes, R1, R2, 0/*verbose_level - 1*/);
	calc_log2(primes, primes_log2, 0 /*verbose_level - 1*/);


	int f_x_file = false;
	vector<int> X;

	if (f_x_file) {
#if 0
		int l = primes.size();
		int ll = l + 10;

		read_x_file(X, x_file, ll);
#endif
		}
	else {
		Quadratic_Sieve(factorbase, false /* f_mod */, 0 /* mod_n */, 0 /* mod_r */, x0,
			n, M, sqrtM,
			primes, primes_log2, R1, R2, X, verbose_level - 1);
		if (false /*f_mod*/) {
			exit(1);
			}
		}


	if (f_v) {
		cout << "cryptography_domain::quadratic_sieve done" << endl;
	}

}

void cryptography_domain::calc_log2(
		std::vector<int> &primes,
		std::vector<int> &primes_log2,
		int verbose_level)
{
	int i, l, k;

	l = primes.size();
	for (i = 0; i < l; i++) {
		k = log2(primes[i]);
		primes_log2.push_back(k);
		}
}

void cryptography_domain::all_square_roots_mod_n_by_exhaustive_search_lint(
		std::string &square_root_a,
		std::string &square_root_mod_n,
		std::vector<long int> &S,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, a, n;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "cryptography_domain::all_square_roots_mod_n_by_exhaustive_search_lint" << endl;
	}

	a = ST.strtoi(square_root_a);
	n = ST.strtoi(square_root_mod_n);
	for (i = 0; i < a; i++) {
		if (((i * i) % n) == a) {
			S.push_back(i);
		}
	}

	if (f_v) {
		cout << "cryptography_domain::all_square_roots_mod_n_by_exhaustive_search_lint done" << endl;
	}
}

void cryptography_domain::square_root(
		std::string &square_root_number, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, b;

	if (f_v) {
		cout << "cryptography_domain::square_root" << endl;
	}
	a.create_from_base_10_string(square_root_number);
	if (f_v) {
		cout << "computing square root of " << a << endl;
	}
	D.square_root(a, b, verbose_level - 4);
	if (f_v) {
		cout << "square root of " << a << " is " << b << endl;
	}

	if (f_v) {
		cout << "cryptography_domain::square_root done" << endl;
	}
}

void cryptography_domain::square_root_mod(
		std::string &square_root_number,
		std::string &mod_number, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, m;
	int b;

	if (f_v) {
		cout << "cryptography_domain::square_root_mod" << endl;
	}
	a.create_from_base_10_string(square_root_number);
	if (f_v) {
		cout << "cryptography_domain::square_root_mod "
				"computing square root of " << a << endl;
	}
	m.create_from_base_10_string(mod_number);
	if (f_v) {
		cout << "cryptography_domain::square_root_mod modulo " << m << endl;
	}
	b = D.square_root_mod(a.as_int(), m.as_int(), verbose_level -1);
	if (f_v) {
		cout << "cryptography_domain::square_root_mod "
				"square root of " << a << " mod " << m << " is " << b << endl;
	}

	if (f_v) {
		cout << "cryptography_domain::square_root_mod done" << endl;
	}
}

void cryptography_domain::reduce_primes(
		std::vector<int> &primes,
		ring_theory::longinteger_object &M,
		int &f_found_small_factor, int &small_factor,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l, r, s, p;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object Q, R, P;

	if (f_v) {
		cout << "cryptography_domain::reduce_primes" << endl;
	}
	f_found_small_factor = false;
	small_factor = 0;
	l = primes.size();
	for (i = 0; i < l; i++) {
		p = primes[i];
		// cout << "i=" << i << " prime=" << p << endl;
		D.integral_division_by_int(M,
				p, Q, r);

		R.create(r);
		P.create(p);

		s = D.jacobi(R, P, 0 /* verbose_level */);
		//s = Legendre(r, p, 0);
		// cout << "i=" << i << " p=" << p << " Mmodp=" << r
		//<< " Legendre(r,p)=" << s << endl;
		if (s == 0) {
			cout << "cryptography_domain::reduce_primes "
					"M is divisible by " << p << endl;
			//exit(1);
			f_found_small_factor = true;
			small_factor = p;
			return;
			}
		if (s == -1) {
			primes.erase(primes.begin()+i);
			l--;
			i--;
			}
		}
	cout << "cryptography_domain::reduce_primes "
			"number of primes remaining = " << primes.size() << endl;
}


void cryptography_domain::do_sift_smooth(
		int sift_smooth_from,
		int sift_smooth_len,
		std::string &sift_smooth_factor_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *B;
	int sz;
	int a, i, j, nb, p, idx, cnt;
	number_theory::number_theory_domain NT;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "do_sift_smooth" << endl;
	}

	Int_vec_scan(sift_smooth_factor_base, B, sz);
	Sorting.int_vec_heapsort(B, sz);

	cnt = 0;
	for (i = 0; i < sift_smooth_len; i++) {
		a = sift_smooth_from + i;

		int *primes;
		int *exponents;

		nb = NT.factor_int(a, primes, exponents);
		for (j = 0; j < nb; j++) {
			p = primes[j];
			if (!Sorting.int_vec_search(B, sz, p, idx)) {
				break;
			}
		}
		if (j == nb) {
			// the number is smooth:

			cout << cnt << " : " << a << " : ";
			NT.print_factorization(nb, primes, exponents);
			cout << endl;

			cnt++;
		}

		FREE_int(primes);
		FREE_int(exponents);

	}
}

void cryptography_domain::do_discrete_log(
		long int y,
		long int a, long int p,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;
	int t0, t1, dt;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "do_discrete_log" << endl;
		cout << "y=" << y << endl;
		cout << "a=" << a << endl;
		cout << "p=" << p << endl;
	}

	t0 = Os.os_ticks();
	//finite_field F;
	long int n, b;

	//F.init(p, 0);
	for (n = 0; n < p - 1; n++) {
		//b = F.power(a, n);
		b = NT.power_mod(a, n, p);
		if (b == y) {
			break;
		}
	}

	if (n == p - 1) {
		cout << "could not solve the discrete log problem." << endl;
	}
	else {
		cout << "The discrete log is " << n << " since ";
		cout << y << " = " << a << "^" << n << " mod " << p << endl;
	}

	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;
}

void cryptography_domain::do_primitive_root(
		long int p, int verbose_level)
{
	number_theory::number_theory_domain NT;
	long int a;
	int t0, t1, dt;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	a = NT.primitive_root_randomized(p, verbose_level);
	cout << "a primitive root modulo " << p << " is " << a << endl;

	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;
}


void cryptography_domain::do_primitive_root_longinteger(
		ring_theory::longinteger_object &p, int verbose_level)
{
	number_theory::number_theory_domain NT;
	long int a;
	int t0, t1, dt;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	a = NT.primitive_root_randomized(p.as_lint(), verbose_level);
	cout << "a primitive root modulo " << p << " is " << a << endl;

	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;
}



void cryptography_domain::do_smallest_primitive_root(
		long int p, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;
	long int a;
	int t0, t1, dt;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();
	if (f_v) {
		cout << "cryptography_domain::do_smallest_primitive_root p=" << p << endl;
	}


	a = NT.primitive_root(p, verbose_level);
	cout << "a primitive root modulo " << p << " is " << a << endl;

	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;
	if (f_v) {
		cout << "cryptography_domain::do_smallest_primitive_root done" << endl;
	}
}

void cryptography_domain::do_smallest_primitive_root_interval(
		long int p_min, long int p_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;
	long int a, p, i;
	long int *T;
	int t0, t1, dt;
	orbiter_kernel_system::os_interface Os;
	orbiter_kernel_system::file_io Fio;
	char str[1000];

	t0 = Os.os_ticks();
	if (f_v) {
		cout << "cryptography_domain::do_smallest_primitive_root_interval p_min=" << p_min << " p_max=" << p_max << endl;
	}

	std::vector<std::pair<long int, long int>> Table;

	for (p = p_min; p < p_max; p++) {

		if (!NT.is_prime(p)) {
			continue;
		}

		std::pair<long int, long int> P;

		a = NT.primitive_root(p, verbose_level);
		cout << "a primitive root modulo " << p << " is " << a << endl;

		P.first = p;
		P.second = a;

		Table.push_back(P);

	}
	T = NEW_lint(2 * Table.size());
	for (i = 0; i < Table.size(); i++) {
		T[2 * i + 0] = Table[i].first;
		T[2 * i + 1] = Table[i].second;
	}
	string fname;

	fname = "primitive_element_table_" + std::to_string(p_min) + "_" + std::to_string(p_max) + ".csv";

	fname.assign(str);
	Fio.lint_matrix_write_csv(fname, T, Table.size(), 2);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;



	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;
	if (f_v) {
		cout << "cryptography_domain::do_smallest_primitive_root_interval done" << endl;
	}
}

void cryptography_domain::do_number_of_primitive_roots_interval(
		long int p_min, long int p_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;
	long int a, p, i;
	long int *T;
	int t0, t1, dt;
	orbiter_kernel_system::os_interface Os;
	orbiter_kernel_system::file_io Fio;
	char str[1000];

	t0 = Os.os_ticks();
	if (f_v) {
		cout << "cryptography_domain::do_number_of_primitive_roots_interval "
				"p_min=" << p_min << " p_max=" << p_max << endl;
	}

	std::vector<std::pair<long int, long int>> Table;

	for (p = p_min; p < p_max; p++) {

		if (!NT.is_prime(p)) {
			continue;
		}

		std::pair<long int, long int> P;

		a = NT.euler_function(p - 1);
		cout << "the number of primitive elements modulo " << p << " is " << a << endl;

		P.first = p;
		P.second = a;

		Table.push_back(P);

	}
	T = NEW_lint(2 * Table.size());
	for (i = 0; i < Table.size(); i++) {
		T[2 * i + 0] = Table[i].first;
		T[2 * i + 1] = Table[i].second;
	}
	string fname;
	fname = "table_number_of_pe_" + std::to_string(p_min) + "_" + std::to_string(p_max) + ".csv";

	fname.assign(str);
	Fio.lint_matrix_write_csv(fname, T, Table.size(), 2);

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;



	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;
	if (f_v) {
		cout << "cryptography_domain::do_number_of_primitive_roots_interval done" << endl;
	}
}


void cryptography_domain::do_inverse_mod(
		long int a, long int n, int verbose_level)
{
	number_theory::number_theory_domain NT;
	long int b;
	int t0, t1, dt;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	b = NT.inverse_mod(a, n);
	cout << "the inverse of " << a << " mod " << n << " is " << b << endl;

	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;
}

void cryptography_domain::do_extended_gcd(
		int a, int b, int verbose_level)
{
	{
		ring_theory::longinteger_domain D;

		ring_theory::longinteger_object A, B, G, U, V;

	A.create(a);
	B.create(b);

	cout << "before D.extended_gcd" << endl;
	D.extended_gcd(A, B,
			G, U, V, verbose_level);
	cout << "after D.extended_gcd" << endl;

	}

}


void cryptography_domain::do_power_mod(
		ring_theory::longinteger_object &a,
		ring_theory::longinteger_object &k,
		ring_theory::longinteger_object &n,
		int verbose_level)
{
	ring_theory::longinteger_domain D;
	//number_theory_domain NT;
	ring_theory::longinteger_object b;
	int t0, t1, dt;
	orbiter_kernel_system::os_interface Os;

	t0 = Os.os_ticks();

	a.assign_to(b);
	D.power_longint_mod(b, k, n, 0 /* verbose_level */);
	//b = NT.power_mod(a, k, n);
	cout << "the power of " << a << " to the " << k << " mod " << n << " is " << b << endl;

	t1 = Os.os_ticks();
	dt = t1 - t0;
	cout << "time: ";
	Os.time_check_delta(cout, dt);
	cout << endl;

}

void cryptography_domain::calc_roots(
		ring_theory::longinteger_object &M,
		ring_theory::longinteger_object &sqrtM,
	std::vector<int> &primes,
	std::vector<int> &R1, std::vector<int> &R2,
	int verbose_level)
// computes the root of the polynomial
// $X^2 + a X + b$ over $GF(p)$
// here, $a = 2 \cdot \lfloor \sqrt{M} \rfloor$
// and $b= {\lfloor \sqrt{M} \rfloor }^2 - M$
// which is equal to
// (X + \lfloor \sqrt{M} \rfloor)^2 - M.
// If $x$ is a root of this polynomial mod p then
// (x + \lfloor \sqrt{M} \rfloor)^2 = M mod p
// and M is a square mod p.
// Due to reduce prime, only such p are considered.
// The polynomial factors as
// $(X - r_1)(X - r_1)= X^2 - (r_1 + r_2) X + r_1 r_2$
// Due to reduce primes, the polynomial factors mod p.
{
	int f_v = (verbose_level >= 1);
	int i, l, p, Mmodp, sqrtMmodp, b;
	int r1, r2, c, c2, s;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object P, l1, l2, l3;

	if (f_v) {
		cout << "cryptography_domain::calc_roots, verbose_level=" << verbose_level << endl;
		cout << "cryptography_domain::calc_roots, M=" << M << endl;
		cout << "cryptography_domain::calc_roots, sqrtM=" << sqrtM << endl;
	}
	l = primes.size();
	for (i = 0; i < l; i++) {
		p = primes[i];
		if (f_v) {
			cout << "cryptography_domain::calc_roots i=" << i << " / " << l << " p=" << p << endl;
		}
		P.create(p);

		if (f_v) {
			cout << "cryptography_domain::calc_roots before remainder_mod_int" << endl;
		}
		Mmodp = D.remainder_mod_int(M, p);
		if (f_v) {
			cout << "cryptography_domain::calc_roots after remainder_mod_int "
					"Mmodp=" << Mmodp << endl;
		}
		if (f_v) {
			cout << "cryptography_domain::calc_roots before remainder_mod_int" << endl;
		}
		sqrtMmodp = D.remainder_mod_int(sqrtM, p);
		if (f_v) {
			cout << "cryptography_domain::calc_roots after remainder_mod_int, "
					"sqrtMmodp=" << sqrtMmodp << endl;
		}

		// a = 2 * sqrtMmodp mod p
		//a = (sqrtMmodp << 1) % p;

		// b = (sqrtMmodp * sqrtMmodp) % p;
		l1.create(sqrtMmodp);
		D.mult_mod(l1, l1, l2, P, 0 /* verbose_level */);
		b = l2.as_int();

		b = b - Mmodp;
		if (b < 0) {
			b += p;
		}
		else {
			b = b % p;
		}

		// use the quadratic formula to compute the roots:
		// sqrtMmodp = a / 2.

		l1.create(sqrtMmodp);
		D.mult_mod(l1, l1, l2, P, 0 /* verbose_level */);
		c2 = l2.as_int();
		c2 -= b;
		while (c2 < 0) {
			c2 += p;
		}
		// c2 = discriminant


		if (f_v) {
			cout << "cryptography_domain::calc_roots computing square root "
					"of discriminant c2=" << c2 << endl;
		}
		s = D.square_root_mod(c2, p, 0 /* verbose_level*/);
		if (f_v) {
			cout << "cryptography_domain::calc_roots c2=" << c2 << " s=" << s << endl;
		}


		c = - sqrtMmodp;
		if (c < 0) {
			c += p;
		}

		r1 = (c + s) % p;

		r2 = c - s;
		if (r2 < 0) {
			r2 += p;
		}
		r2 = r2 % p;


		if (f_v) {
			cout << "cryptography_domain::calc_roots r1=" << r1 << " r2=" << r2 << endl;
		}


		R1.push_back(r1);
		R2.push_back(r2);
		// cout << "i=" << i << " p=" << p
		//<< " r1=" << r1 << " r2=" << r2 << endl;

	} // next i

	if (f_v) {
		cout << "cryptography_domain::calc_roots done" << endl;
	}
}

void cryptography_domain::Quadratic_Sieve(
	int factorbase,
	int f_mod, int mod_n, int mod_r, int x0,
	int n, ring_theory::longinteger_object &M,
	ring_theory::longinteger_object &sqrtM,
	std::vector<int> &primes,
	std::vector<int> &primes_log2,
	std::vector<int> &R1, std::vector<int> &R2,
	std::vector<int> &X,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ostringstream ff;
	string s;

	if (f_v) {
		cout << "cryptography_domain::Quadratic_Sieve" << endl;
	}
	ff << "X_M_" << n << "_FB_" << factorbase;
	if (f_mod) {
		ff << "_mod_" << mod_n << "_" << mod_r;
	}
	ff << ".txt";
	ff << ends;
	s = ff.str();

	int l = primes.size();
	int ll = l + 10;
	int from = x0, to = x0, count = -1, step_size = 50000;

	if (f_mod) {
		ll = ll / mod_n + 1;
	}
	//X.m_l(0);




	while (true) {
		from = to;
		to = from + step_size;
		count++;

		if (f_mod) {
			if (count % mod_n != mod_r) {
				continue;
			}
		}
		if (quadratic_sieve(M, sqrtM,
			primes, primes_log2, R1, R2, from, to, ll, X, verbose_level)) {
			break;
		}
	}

	if (f_v) {
		cout << "found " << ll << " x_i" << endl;
	}

	{
		ofstream f(s.c_str());

#if 1
		int i;

		for (i = 0; i < ll; i++) {
			f << X[i] << " ";
			if ((i + 1) % 10 == 0)
				f << endl;
			}
#endif
		f << endl << "-1" << endl;
	}
}

int cryptography_domain::quadratic_sieve(
		ring_theory::longinteger_object& M,
		ring_theory::longinteger_object& sqrtM,
	std::vector<int> &primes,
	std::vector<int> &primes_log2,
	std::vector<int> &R1, std::vector<int> &R2,
	int from, int to,
	int ll, std::vector<int> &X,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int x, j;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object Z, zero, a, b, c, d;
	int i, l;
	vector<int> factor_idx, factor_exp;

	if (f_v) {
		cout << "cryptography_domain::quadratic_sieve" << endl;
	}
	zero.create(0);
	l = primes.size();
	j = X.size();
	if (f_v) {
		cout << "quadratic sieve from=" << from
				<< " to=" << to << " j=" << j << endl;
		cout << "searching for " << ll << " numbers" << endl;
	}
	for (x = from; x < to; x++) {
		if (x == 0) {
			continue;
		}
		a.create(x);
		D.add(a, sqrtM, c);
		D.mult(c, a, d);
		d.assign_to(a);
		M.assign_to(b);
		b.negate();
		D.add(a, b, c);
		c.assign_to(a);
		if (D.compare_unsigned(a, zero) <= 0) {
			continue;
		}
		a.normalize();

#if 1
		int xmodp, log2a, sumlog2;
		log2a = 3 * (a.len() - 1);
		sumlog2 = 0;
		for (i = 0; i < l; i++) {
			xmodp = x % primes[i];
			if (xmodp == R1[i]) {
				sumlog2 += primes_log2[i] + 0;
			}
			if (xmodp == R2[i]) {
				sumlog2 += primes_log2[i] + 0;
			}
		}
		// cout << "sieve x=" << x << " log2=" << log2a
		//<< " sumlog2=" << sumlog2 << endl;
		if (sumlog2 < log2a)
			continue;
#endif
		if (!factor_over_factor_base(a,
				primes, factor_idx, factor_exp,
				verbose_level - 1)) {
			continue;
		}
		//f << x << endl;
		if (f_v) {
			cout << "found solution " << j << " which is " << x
					<< ", need " << ll - j << " more" << endl;
		}
		X.push_back(x);
		j++;
		if (j >= ll) {
			if (f_v) {
				cout << "sieve: found enough numbers "
						"(enough = " << ll << ")" << endl;
			}
			if (f_v) {
				cout << "cryptography_domain::quadratic_sieve done" << endl;
			}
			return true;
		}
	} // next x
	if (f_v) {
		cout << "cryptography_domain::quadratic_sieve done" << endl;
	}
	return false;
}

int cryptography_domain::factor_over_factor_base(
		ring_theory::longinteger_object &x,
		std::vector<int> &primes,
		std::vector<int> &factor_idx,
		std::vector<int> &factor_exp,
		int verbose_level)
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object y, z1, residue;
	int i, l, n, p;

	x.assign_to(y);
	z1.create(1);
	l = primes.size();
	//factor_idx.m_l(0);
	//factor_exp.m_l(0);
	for (i = 0; i < l; i++) {
		if (D.compare(y, z1) <= 0) {
			break;
		}
		p = primes[i];
		n = D.multiplicity_of_p(y, residue, p);
		residue.assign_to(y);
		if (n) {
			factor_idx.push_back(i);
			factor_exp.push_back(n);
		}
	}
	if (D.compare_unsigned(y, z1) == 0) {
		return true;
	}
	else {
		return false;
	}
}

int cryptography_domain::factor_over_factor_base2(
		ring_theory::longinteger_object &x,
		std::vector<int> &primes,
		std::vector<int> &exponents,
		int verbose_level)
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object y, z1, residue;
	int i, l, n, nn, p;

	x.assign_to(y);
	z1.create(1);
	l = primes.size();
	for (i = 0; i < l; i++) {
		if (D.compare(x, z1) <= 0) {
			break;
		}
		p = primes[i];
		n = D.multiplicity_of_p(x, residue, p);
		residue.assign_to(x);
		//n = x.ny_p(p);
		// cout << "p=" << p << " ny_p=" << n << endl;
		if (n) {
			nn = exponents[i] + n;
			exponents[i] = nn;
		}
	}
	if (D.compare_unsigned(x, z1) == 0) {
		return true;
	}
	else {
		return false;
	}
}


void cryptography_domain::find_probable_prime_above(
		ring_theory::longinteger_object &a,
	int nb_solovay_strassen_tests, int f_miller_rabin_test,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object b, one;
	int i = 0;

	if (f_v) {
		cout << "cryptography_domain::find_probable_prime_above" << endl;
	}
	one.create(1);
	while (true) {
		if (f_vv) {
			cout << "considering " << a << endl;
		}
		if (!miller_rabin_test(a, verbose_level - 2)) {
			if (f_vv) {
				cout << "is not prime because of Miller Rabin" << endl;
			}
			goto loop;
		}
		if (solovay_strassen_is_prime(a,
				nb_solovay_strassen_tests, verbose_level - 2)) {
			if (f_vv) {
				cout << "may be prime" << endl;
			}
			break;
		}
		else {
			if (f_vv) {
				cout << "is not prime because of "
					"Solovay Strassen" << endl;
			}
		}
loop:
		D.add(a, one, b);
		b.assign_to(a);
		i++;
	}
	if (f_v) {
		cout << "cryptography_domain::find_probable_prime_above: probable prime: "
			<< a << " (found after " << i << " tests)" << endl;
	}
}

int cryptography_domain::solovay_strassen_is_prime(
		ring_theory::longinteger_object &n,
		int nb_tests,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "cryptography_domain::solovay_strassen_is_prime for "
			<< n << " with " << nb_tests << " tests:" << endl;
		}
	for (i = 0; i < nb_tests; i++) {
		if (!solovay_strassen_is_prime_single_test(
				n, verbose_level - 2)) {
			if (f_v) {
				cout << "is not prime after "
						<< i + 1 << " tests" << endl;
				}
			return false;
		}
	}
	return true;
}

int cryptography_domain::solovay_strassen_is_prime_single_test(
		ring_theory::longinteger_object &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, one, b, m_one, n_minus_one;
	int r;

	if (f_v) {
		cout << "cryptography_domain::solovay_strassen_is_prime_single_test" << endl;
	}
	one.create(1);
	m_one.create(-1);
	D.add(n, m_one, n_minus_one);
	D.random_number_less_than_n(n_minus_one, a);
	D.add(a, one, b);
	b.assign_to(a);
	if (f_vv) {
		cout << "cryptography_domain::solovay_strassen_is_prime "
				"choosing integer " << a
				<< " less than " << n << endl;
	}

	r = solovay_strassen_test(n, a, verbose_level);
	return r;

}

int cryptography_domain::fermat_test_iterated_with_latex_key(
		std::ostream &ost,
		ring_theory::longinteger_object &P, int nb_times,
		int verbose_level)
// returns true is the test is conclusive, i.e. if the number is not prime.
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object A, B, one, minus_two, n_minus_two;
	int i, ret;

	if (f_v) {
		cout << "cryptography_domain::fermat_test_iterated_with_latex_key" << endl;
	}
	one.create(1);
	minus_two.create(-2);

	D.add(P, minus_two, n_minus_two);

	ost << "We will do " << nb_times << " Fermat tests for $" << P << "$:\\\\" << endl;

	ost << "\\begin{enumerate}[(1)]" << endl;
	for (i = 0; i < nb_times; i++) {


		ost << "\\item" << endl;
		ost << "Fermat test no " << i + 1 << ":\\\\" << endl;

		// choose a random integer a with 1 <= a < n - 1
		D.random_number_less_than_n(n_minus_two, A);
		D.add(A, one, B);
		B.assign_to(A);


		ost << "Choosing base $" << A << ".$\\\\" << endl;

		if (fermat_test_with_latex_key(ost,
			P, A,
			verbose_level)) {
			// test applies, the number is not prime
			break;
		}

	}
	ost << "\\end{enumerate}" << endl;
	if (i == nb_times) {
		//ost << "Fermat: The number $" << P << "$ is probably prime. Fermat test is inconclusive.\\\\" << endl;
		ret = false;
	}
	else {
		//ost << "Fermat: The number $" << P << "$ is not prime.\\\\" << endl;
		ret = true;
	}
	if (f_v) {
		cout << "cryptography_domain::fermat_test_iterated_with_latex_key done" << endl;
	}
	return ret;
}

int cryptography_domain::fermat_test_with_latex_key(
		std::ostream &ost,
		ring_theory::longinteger_object &n,
		ring_theory::longinteger_object &a,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object b, one, m_one, n2, n_minus_one;

	if (f_v) {
		cout << "cryptography_domain::fermat_test_with_latex_key" << endl;
	}
	one.create(1);
	m_one.create(-1);
	D.add(n, m_one, n_minus_one);
	if (f_vv) {
		cout << "cryptography_domain::fermat_test_with_latex_key "
			"a = " << a << endl;
	}
	ost << "Fermat test for $n=" << n << ",$ picking basis $a=" << a << "$\\\\" << endl;
	D.power_longint_mod(a, n_minus_one, n, 0 /*verbose_level - 2*/);
	if (f_vv) {
		cout << "cryptography_domain::fermat_test_with_latex_key "
				"a^((n-1)) = " << a << endl;
	}
	ost << "$a^{" << n_minus_one << "} \\equiv " << a << "$\\\\" << endl;
	if (a.is_one()) {
		if (f_v) {
			cout << "cryptography_domain::fermat_test_with_latex_key "
				"inconclusive" << endl;
		}
		ost << "The Fermat test is inconclusive.\\\\" << endl;
		return false;
	}
	else {
		if (f_v) {
			cout << "cryptography_domain::fermat_test_with_latex_key "
				"not prime (sure)" << endl;
		}
		ost << "The number $" << n << "$ is not prime because of the Fermat test.\\\\" << endl;
		return true;
	}
}

int cryptography_domain::solovay_strassen_test(
		ring_theory::longinteger_object &n,
		ring_theory::longinteger_object &a,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object b, one, m_one, n2, n_minus_one;
	int x, r;

	if (f_v) {
		cout << "cryptography_domain::solovay_strassen_test" << endl;
	}
	one.create(1);
	m_one.create(-1);
	D.add(n, m_one, n_minus_one);
	if (f_vv) {
		cout << "cryptography_domain::solovay_strassen_test "
			"a = " << a << endl;
	}
	x = D.jacobi(a, n, verbose_level - 2);
	if (x == 0) {
		if (f_v) {
			cout << "not prime (sure)" << endl;
		}
		return false;
	}
	D.add(n, m_one, b);
	D.integral_division_by_int(b, 2, n2, r);
	if (f_vv) {
		cout << "cryptography_domain::solovay_strassen_test "
			"raising to the power " << n2 << endl;
	}
	D.power_longint_mod(a, n2, n, 0 /*verbose_level - 2*/);
	if (f_vv) {
		cout << "cryptography_domain::solovay_strassen_test "
				"a^((n-1)/2) = " << a << endl;
	}
	if (x == 1) {
		if (a.is_one()) {
			if (f_v) {
				cout << "cryptography_domain::solovay_strassen_test "
					"inconclusive" << endl;
			}
			return true;
		}
		else {
			if (f_v) {
				cout << "cryptography_domain::solovay_strassen_test "
					"not prime (sure)" << endl;
			}
			return false;
		}
	}
	if (x == -1) {
		if (D.compare_unsigned(a, n_minus_one) == 0) {
			if (f_v) {
				cout << "cryptography_domain::solovay_strassen_test "
					"inconclusive" << endl;
			}
			return true;
		}
		else {
			if (f_v) {
				cout << "cryptography_domain::solovay_strassen_test "
					"not prime (sure)" << endl;
			}
			return false;
		}
	}
	// we should never be here:
	cout << "cryptography_domain::solovay_strassen_test "
			"error" << endl;
	exit(1);
}

int cryptography_domain::solovay_strassen_test_with_latex_key(
		ostream &ost,
		ring_theory::longinteger_object &n,
		ring_theory::longinteger_object &a,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object b, one, m_one, n2, n_minus_one;
	int x, r;

	if (f_v) {
		cout << "cryptography_domain::solovay_strassen_test_with_latex_key" << endl;
	}
	one.create(1);
	m_one.create(-1);
	D.add(n, m_one, n_minus_one);
	if (f_vv) {
		cout << "cryptography_domain::solovay_strassen_test_with_latex_key "
			"a = " << a << endl;
	}
	ost << "Solovay-Strassen pseudoprime test for $n=" << n
			<< ",$ picking basis $a=" << a << "$\\\\" << endl;
	x = D.jacobi(a, n, verbose_level - 2);
	ost << "$\\Big( \\frac{" << a
		<< " }{ " << n << "}\\Big) = " << x << "$\\\\" << endl;
	if (x == 0) {
		if (f_v) {
			cout << "not prime (sure)" << endl;
		}
		return false;
	}
	D.add(n, m_one, b);
	D.integral_division_by_int(b, 2, n2, r);
	if (f_vv) {
		cout << "cryptography_domain::solovay_strassen_test_with_latex_key "
			"raising to the power " << n2 << endl;
	}
	D.power_longint_mod(a, n2, n, 0 /*verbose_level - 2*/);
	if (f_vv) {
		cout << "cryptography_domain::solovay_strassen_test_with_latex_key "
				"a^((n-1)/2) = " << a << endl;
	}


	ost << "$a^{\\frac{" << n << "-1}{2}} \\equiv " << a;
	if (D.compare_unsigned(a, n_minus_one) == 0) {
		ost << " \\equiv -1";
	}
	ost << "$\\\\" << endl;

	if (x == 1) {
		if (a.is_one()) {
			if (f_v) {
				cout << "cryptography_domain::solovay_strassen_test_with_latex_key "
					"inconclusive" << endl;
			}
			ost << "The Solovay-Strassen test is inconclusive.\\\\" << endl;
			return true;
		}
		else {
			if (f_v) {
				cout << "cryptography_domain::solovay_strassen_test_with_latex_key "
					"not prime (sure)" << endl;
			}
			ost << "The number $n$ is not prime by the Solovay-Strassen test.\\\\" << endl;
			return false;
		}
	}
	if (x == -1) {
		if (D.compare_unsigned(a, n_minus_one) == 0) {
			if (f_v) {
				cout << "cryptography_domain::solovay_strassen_test_with_latex_key "
					"inconclusive" << endl;
			}
			ost << "The Solovay-Strassen test is inconclusive.\\\\" << endl;
			return true;
		}
		else {
			if (f_v) {
				cout << "cryptography_domain::solovay_strassen_test_with_latex_key "
					"not prime (sure)" << endl;
			}
			ost << "The number $n$ is not prime by the Solovay-Strassen test.\\\\" << endl;
			return false;
		}
	}
	// we should never be here:
	cout << "cryptography_domain::solovay_strassen_test_with_latex_key "
			"error" << endl;
	exit(1);
}

int cryptography_domain::solovay_strassen_test_iterated_with_latex_key(
		std::ostream &ost,
		ring_theory::longinteger_object &P, int nb_times,
		int verbose_level)
// returns true is the test is conclusive, i.e. if the number is not prime.
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object A, B, one, m_one, m_two, P_minus_one, P_minus_two;
	int i, ret;

	if (f_v) {
		cout << "cryptography_domain::solovay_strassen_test_iterated_with_latex_key" << endl;
	}

	ost << "We will do " << nb_times << " Solovay-Strassen "
			"tests for $" << P << "$:\\\\" << endl;

	one.create(1);
	m_one.create(-1);
	m_two.create(-2);
	D.add(P, m_one, P_minus_one);
	D.add(P, m_two, P_minus_two);

	ost << "\\begin{enumerate}[(1)]" << endl;


	for (i = 0; i < nb_times; i++) {


		ost << "\\item" << endl;
		ost << "Solovay-Strassen test no " << i + 1 << ":\\\\" << endl;

		// choose a random integer a with 1 <= a < n - 1
		D.random_number_less_than_n(P_minus_two, A);
		D.add(A, one, B);
		B.assign_to(A);


		ost << "Choosing base $" << A << ".$\\\\" << endl;

		if (!solovay_strassen_test_with_latex_key(ost,
			P, A,
			verbose_level)) {
			// test applies, the number is not prime
			break;
		}

	}
	ost << "\\end{enumerate}" << endl;

	if (i == nb_times) {
		//ost << "Solovay-Strassen: The number $" << P << "$ is probably prime. Solovay-Strassen test is inconclusive.\\\\" << endl;
		ret = false;
	}
	else {
		//ost << "Solovay-Strassen: The number $" << P << "$ is not prime.\\\\" << endl;
		ret = true;
	}
	if (f_v) {
		cout << "cryptography_domain::solovay_strassen_test_iterated_with_latex_key done" << endl;
	}
	return ret;
}




int cryptography_domain::miller_rabin_test(
		ring_theory::longinteger_object &n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, b, c, one, m_one, n_minus_one, m, mm;
	int k, i;

	if (f_v) {
		cout << "cryptography_domain::miller_rabin_test "
				"for " << n << endl;
	}
	one.create(1);
	m_one.create(-1);
	D.add(n, m_one, n_minus_one);

#if 1
	// choose a random integer a with 1 <= a <= n - 1
	D.random_number_less_than_n(n_minus_one, a);
	D.add(a, one, b);
	b.assign_to(a);
#else
	a.create(2, __FILE__, __LINE__);
#endif
	if (f_vv) {
		cout << "cryptography_domain::miller_rabin_test "
			"choosing integer " << a << " less than " << n << endl;
	}

	k = D.multiplicity_of_p(n_minus_one, m, 2);
	m.assign_to(mm);
	if (f_vv) {
		cout << n_minus_one << " = 2^" << k << " x " << m << endl;
	}

	// compute b := a^m mod n
	a.assign_to(b);
	D.power_longint_mod(b, m, n, false /* f_v */);
	if (f_vv) {
		cout << a << "^" << mm << " = " << b << endl;
	}
	if (b.is_one()) {
		if (f_v) {
			cout << "a^m = 1 mod n, so the test is inconclusive" << endl;
		}
		return true;
	}
	if (D.compare_unsigned(b, n_minus_one) == 0) {
		if (f_v) {
			cout << "is minus one, so the test is inconclusive" << endl;
		}
		return true;
	}
	for (i = 0; i < k; i++) {
		D.mult_mod(b, b, c, n, 0);
		if (f_vv) {
			cout << "b_" << i << "=" << b
					<< " b_" << i + 1 << "=" << c << endl;
		}
		c.assign_to(b);
		if (D.compare_unsigned(b, n_minus_one) == 0) {
			if (f_v) {
				cout << "is minus one, so the test is inconclusive" << endl;
			}
			return true;
		}
		if (D.compare_unsigned(b, one) == 0) {
			if (f_v) {
				cout << "is one, we reject as composite" << endl;
			}
			return false;
		}
		//mult(b, b, c);
	}
	if (f_v) {
		cout << "inconclusive, we accept as probably prime" << endl;
	}
	return true;
}

int cryptography_domain::miller_rabin_test_with_latex_key(
		std::ostream &ost,
		ring_theory::longinteger_object &n, int iteration,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object a, b, c, one, m_one, n_minus_one, m, mm;
	int k, i;

	if (f_v) {
		cout << "cryptography_domain::miller_rabin_test_with_latex_key "
				"for " << n << " iteration=" << iteration << endl;
	}
	ost << "Miller-Rabin pseudoprime test for $n=" << n << "$\\\\" << endl;
	one.create(1);
	m_one.create(-1);
	D.add(n, m_one, n_minus_one);



		if (iteration < 5) {
			int small_prime;
			number_theory::number_theory_domain NT;

			small_prime = NT.get_prime_from_table(iteration);
			a.create(small_prime);
		}
		else {
			// choose a random integer a with 1 <= a <= n - 1
			D.random_number_less_than_n(n_minus_one, a);
			D.add(a, one, b);
			b.assign_to(a);
		}


		if (f_vv) {
			cout << "cryptography_domain::miller_rabin_test_with_latex_key "
				"choosing test base a= " << a << endl;
		}

		ost << "Picking test base $a=" << a << "$\\\\" << endl;


		// do a Fermat test:
		a.assign_to(b);
		D.power_longint_mod(b, n_minus_one, n, false /* f_v */);
		if (f_vv) {
			cout << a << "^{n-1} = " << b << endl;
		}

		ost << "$a^{n-1} = a^{" << n_minus_one << "}=" << b << "$\\\\" << endl;

		if (!b.is_one()) {
			if (f_v) {
				cout << "a^{n-1} != 1 mod n, so the number is not prime by Fermat" << endl;
			}
			ost << "The number is not prime, a=" << a << " is a Fermat witness\\\\" << endl;
			return true;
		}
		else {
			ost << "The number survives the Fermat test\\\\" << endl;

		}


		k = D.multiplicity_of_p(n_minus_one, m, 2);
		m.assign_to(mm);
		if (f_vv) {
			cout << n_minus_one << " = 2^" << k << " x " << m << endl;
		}
		ost << "$n-1=2^s \\cdot m = 2^{" << k << "} \\cdot " << m << "$\\\\" << endl;



		// compute b := a^m mod n
		a.assign_to(b);
		D.power_longint_mod(b, m, n, false /* f_v */);
		if (f_vv) {
			cout << a << "^" << mm << " = " << b << endl;
		}

		ost << "$b_0 = a^m = a^{" << mm << "}=" << b << "$\\\\" << endl;

		if (b.is_one()) {
			if (f_v) {
				cout << "a^m = 1 mod n, so the test is inconclusive" << endl;
			}
			ost << "The Miller-Rabin test is inconclusive\\\\" << endl;
			return false;
		}
		if (D.compare_unsigned(b, n_minus_one) == 0) {
			if (f_v) {
				cout << "is minus one, so the test is inconclusive" << endl;
			}
			ost << "The Miller-Rabin test is inconclusive\\\\" << endl;
			return false;
		}
		ost << "$b_{0} = " << b << "$\\\\" << endl;
		for (i = 0; i < k; i++) {
			D.mult_mod(b, b, c, n, 0);
			if (f_vv) {
				cout << "b_" << i << "=" << b
						<< " b_" << i + 1 << "=" << c << endl;
			}
			ost << "$b_{" << i + 1 << "} = " << c << "$\\\\" << endl;
			c.assign_to(b);
			if (D.compare_unsigned(b, n_minus_one) == 0) {
				if (f_v) {
					cout << "is minus one, so the test is inconclusive" << endl;
				}
				ost << "The Miller-Rabin test is inconclusive.\\\\" << endl;
				return false;
			}
			if (D.compare_unsigned(b, one) == 0) {
				if (f_v) {
					cout << "is one, we reject as composite" << endl;
				}
				ost << "The number is not prime because of the Miller-Rabin test.\\\\" << endl;
				return true;
			}
			//mult(b, b, c);
		}
		if (f_v) {
			cout << "inconclusive, we accept as probably prime" << endl;
		}
		ost << "The Miller-Rabin test is inconclusive.\\\\" << endl;

	if (f_v) {
		cout << "cryptography_domain::miller_rabin_test_with_latex_key "
				"done" << endl;
	}
	return false;
}

int cryptography_domain::miller_rabin_test_iterated_with_latex_key(
		std::ostream &ost,
		ring_theory::longinteger_object &P, int nb_times,
		int verbose_level)
// returns true if the test is conclusive,
// i.e. if the number is not prime.
{
	int f_v = (verbose_level >= 1);
	int i, ret;

	if (f_v) {
		cout << "cryptography_domain::miller_rabin_test_iterated_with_latex_key" << endl;
	}

	ost << "Miller-Rabin test for $" << P << "$:\\\\" << endl;

	ost << "\\begin{enumerate}[(1)]" << endl;
	for (i = 0; i < nb_times; i++) {


		ost << "\\item" << endl;
		ost << "Miller-Rabin test no " << i + 1 << ":\\\\" << endl;

		if (miller_rabin_test_with_latex_key(ost,
			P, i,
			verbose_level)) {
			// test applies, the number is not prime
			break;
		}

	}
	ost << "\\end{enumerate}" << endl;
	if (i == nb_times) {
		//ost << "Miller Rabin: The number $" << P << "$ is probably prime. Miller Rabin test is inconclusive.\\\\" << endl;
		ret = false;
	}
	else {
		//ost << "Miller Rabin: The number $" << P << "$ is not prime.\\\\" << endl;
		ret = true;
	}
	if (f_v) {
		cout << "cryptography_domain::miller_rabin_test_iterated_with_latex_key done" << endl;
	}
	return ret;
}

void cryptography_domain::get_k_bit_random_pseudoprime(
		ring_theory::longinteger_object &n, int k,
	int nb_tests_solovay_strassen,
	int f_miller_rabin_test, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	int kk = (k * 3) / 10;
	ring_theory::longinteger_object a, b;

	if (f_v) {
		cout << "cryptography_domain::get_k_bit_random_pseudoprime "
			"trying to get a " << k << " bit, " << kk
			<< " decimals random pseudoprime" << endl;
	}
	a.create(10);
	D.power_int(a, kk);
	D.random_number_less_than_n(a, b);
	if (f_v) {
		cout << "choosing integer " << b << " less than " << a << endl;
	}
	D.add(a, b, n);
	if (f_v) {
		cout << "the sum is " << n << endl;
	}

	find_probable_prime_above(n,
			nb_tests_solovay_strassen, f_miller_rabin_test,
			verbose_level - 1);

}

void cryptography_domain::RSA_setup(
		ring_theory::longinteger_object &n,
		ring_theory::longinteger_object &p,
		ring_theory::longinteger_object &q,
		ring_theory::longinteger_object &a,
		ring_theory::longinteger_object &b,
	int nb_bits,
	int nb_tests_solovay_strassen, int f_miller_rabin_test,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object m1, pm1, qm1, phi_n, v, g;
	int half_bits = nb_bits >> 1;

	if (f_v) {
		cout << "cryptography_domain::RSA_setup nb_bits=" << nb_bits
			<< " nb_tests_solovay_strassen=" << nb_tests_solovay_strassen
			<< " f_miller_rabin_test=" << f_miller_rabin_test << endl;
	}
	m1.create(-1);
	get_k_bit_random_pseudoprime(p, half_bits,
		nb_tests_solovay_strassen,
		f_miller_rabin_test, verbose_level - 2);
	if (f_vv) {
		cout << "choosing p = " << p << endl;
	}
	get_k_bit_random_pseudoprime(q, half_bits,
		nb_tests_solovay_strassen,
		f_miller_rabin_test, verbose_level - 2);
	if (f_v) {
		cout << "choosing p = " << p << endl;
		cout << "choosing q = " << q << endl;
	}
	D.mult(p, q, n);
	if (f_v) {
		cout << "n = pq = " << n << endl;
	}
	D.add(p, m1, pm1);
	D.add(q, m1, qm1);
	D.mult(pm1, qm1, phi_n);
	if (f_v) {
		cout << "phi(n) = (p - 1)(q - 1) = "
				<< phi_n << endl;
	}

	while (true) {
		D.random_number_less_than_n(n, a);
		if (f_v) {
			cout << "choosing integer " << a
					<< " less than " << n << endl;
		}
		D.extended_gcd(a, phi_n, g, b, v, verbose_level - 2);
		if (g.is_one()) {
			break;
		}
		if (f_v) {
			cout << "non trivial gcd: " << g
					<< " , repeating" << endl;
		}
	}
	if (b.sign()) {
		if (f_v) {
			cout << "making b positive" << endl;
		}
		D.add(b, phi_n, v);
		v.assign_to(b);
	}
	if (f_v) {
		cout << "the public key is (a,n) = " << a << "," << n << endl;
		cout << "the private key is (b,n) = " << b << "," << n << endl;
	}
}


void cryptography_domain::do_babystep_giantstep(
		long int p, long int g, long int h,
		int f_latex, std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int N, n;
	double sqrtN;
	long int *Table1;
	long int *Table2;
	long int *data;
	long int gn, gmn, hgmn;
	long int i, r;
	number_theory::number_theory_domain NT;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "cryptography_domain::do_babystep_giantstep "
				"p=" << p << " g=" << g << " h=" << h << endl;
	}
	r = NT.primitive_root(p, 0 /* verbose_level */);
	if (f_v) {
		cout << "a primitive root modulo " << p << " is " << r << endl;
	}

	N = p - 1;
	sqrtN = sqrt(N);
	n = 1 + (int) sqrtN;
	if (f_v) {
		cout << "do_babystep_giantstep "
				"p=" << p
				<< ", g=" << g
				<< " h=" << h
				<< " n=" << n << endl;
	}
	Table1 = NEW_lint(n);
	Table2 = NEW_lint(n);
	data = NEW_lint(2 * n);
	gn = NT.power_mod(g, n, p);
	if (f_v) {
		cout << "g^n=" << gn << endl;
	}
	gmn = NT.inverse_mod(gn, p);
	if (f_v) {
		cout << "g^-n=" << gmn << endl;
	}
	hgmn = NT.mult_mod(h, gmn, p);
	if (f_v) {
		cout << "h*g^-n=" << hgmn << endl;
	}
	Table1[0] = g;
	Table2[0] = hgmn;
	for (i = 1; i < n; i++) {
		Table1[i] = NT.mult_mod(Table1[i - 1], g, p);
		Table2[i] = NT.mult_mod(Table2[i - 1], gmn, p);
	}
	Lint_vec_copy(Table1, data, n);
	Lint_vec_copy(Table2, data + n, n);
	Sorting.lint_vec_heapsort(data, 2 * n);
	if (f_v) {
		cout << "duplicates:" << endl;
		for (i = 1; i < 2 * n; i++) {
			if (data[i] == data[i - 1]) {
				cout << data[i] << endl;
			}
		}
	}

	if (f_latex) {
		ost << "$$" << endl;
		ost << "\\begin{array}[t]{|r|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "i & T_1[i] & T_2[i] \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;
		//ost << "i : g^i : h*g^{-i*n}" << endl;
		for (i = 0; i < n; i++) {
			ost << i + 1 << " & " << Table1[i] << " & "
					<< Table2[i] << "\\\\" << endl;
			if ((i + 1) % 10 == 0) {
				ost << "\\hline" << endl;
				ost << "\\end{array}" << endl;
				ost << "\\quad" << endl;
				ost << "\\begin{array}[t]{|r|r|r|}" << endl;
				ost << "\\hline" << endl;
				ost << "i & T_1[i] & T_2[i] \\\\" << endl;
				ost << "\\hline" << endl;
				ost << "\\hline" << endl;
			}
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
	}
}


}}}


