/*
 * string_tools.cpp
 *
 *  Created on: Apr 15, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace data_structures {



string_tools::string_tools()
{

}

string_tools::~string_tools()
{

}


int string_tools::is_csv_file(const char *fname)
{
	char ext[1000];

	get_extension_if_present(fname, ext);
	if (strcmp(ext, ".csv") == 0) {
		return true;
	}
	else {
		return false;
	}
}

int string_tools::is_inc_file(const char *fname)
{
	char ext[1000];

	get_extension_if_present(fname, ext);
	if (strcmp(ext, ".inc") == 0) {
		return true;
	}
	else {
		return false;
	}
}

int string_tools::is_xml_file(const char *fname)
{
	char ext[1000];

	get_extension_if_present(fname, ext);
	if (strcmp(ext, ".xml") == 0) {
		return true;
	}
	else {
		return false;
	}
}

int string_tools::s_scan_int(char **s, int *i)
{
	char str1[512];

	if (!s_scan_token(s, str1)) {
		return false;
	}
	if (strcmp(str1, ",") == 0) {
		if (!s_scan_token(s, str1)) {
			return false;
		}
	}
	//*i = atoi(str1);
	sscanf(str1, "%d", i);
	return true;
}

int string_tools::s_scan_lint(char **s, long int *i)
{
	char str1[512];

	if (!s_scan_token(s, str1)) {
		return false;
	}
	if (strcmp(str1, ",") == 0) {
		if (!s_scan_token(s, str1)) {
			return false;
		}
	}
	//*i = atoi(str1);
	sscanf(str1, "%ld", i);
	return true;
}

int string_tools::s_scan_double(char **s, double *d)
{
	char str1[512];
	char c;
	int len;

	//cout << "s_scan_double input='" << *s << "'" << endl;
	while (true) {
		c = **s;
		if (c == 0) {
			return(false);
		}
		if (c == ' ' || c == '\t' ||
			c == '\r' || c == 10 || c == 13) {
			(*s)++;
			continue;
		}
		break;
	}
	len = 0;
	c = **s;
	if (isdigit(c) || c == '-') {
		//cout << "s_scan_double character '" << c << "'" << endl;
		while (isdigit(c) || c == '.' || c == 'e' || c == '-') {
			str1[len] = c;
			len++;
			(*s)++;
			c = **s;
			//cout << "character '" << c << "'" << endl;
			//<< *s << "'" << endl;
		}
		str1[len] = 0;
	}
	//cout << "s_scan_double token = " << str1 << endl;
	sscanf(str1, "%lf", d);
	return true;
}

int string_tools::s_scan_token(char **s, char *str)
{
	char c;
	int len;

	while (true) {
		c = **s;
		if (c == 0) {
			return(false);
		}
		if (c == ' ' || c == '\t' ||
			c == '\r' || c == 10 || c == 13) {
			(*s)++;
			continue;
		}
		break;
	}
	len = 0;
	c = **s;
	if (isalpha(c)) {
		//cout << "character '" << c << "', remainder '"
		//<< *s << "'" << endl;
		while (isalnum(c) || c == '_') {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
			//cout << "character '" << c << "', remainder '"
			//<< *s << "'" << endl;
		}
		str[len] = 0;
	}
	else if (isdigit(c) || c == '-') {
		str[len++] = c;
		(*s)++;
		//cout << "character '" << c << "', remainder '"
		//<< *s << "'" << endl;
		//printf("\"%s\"\n", *s);
		c = **s;
		while (isdigit(c)) {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
		}
		str[len] = 0;
	}
	else {
		str[0] = c;
		str[1] = 0;
		(*s)++;
	}
	// printf("token = \"%s\"\n", str);
	return true;
}

int string_tools::s_scan_token_arbitrary(char **s, char *str)
{
	char c;
	int len;

	while (true) {
		c = **s;
		if (c == 0) {
			return(false);
		}
		if (c == ' ' || c == '\t' ||
			c == '\r' || c == 10 || c == 13) {
			(*s)++;
			continue;
		}
		break;
	}
	len = 0;
	c = **s;
	while (c != 0 && c != ' ' && c != '\t' &&
		c != '\r' && c != 10 && c != 13) {
		//cout << "s_scan_token_arbitrary len=" << len
		//<< " reading " << c << endl;
		str[len] = c;
		len++;
		(*s)++;
		c = **s;
	}
	str[len] = 0;
	//printf("token = \"%s\"\n", str);
	return true;
}


int string_tools::s_scan_str(char **s, char *str)
{
	char c;
	int len, f_break;

	while (true) {
		c = **s;
		if (c == 0) {
			return(false);
		}
		if (c == ' ' || c == '\t' ||
			c == '\r' || c == 10 || c == 13) {
			(*s)++;
			continue;
		}
		break;
	}
	if (c != '\"') {
		cout << "s_scan_str() error: c != '\"'" << endl;
		return(false);
	}
	(*s)++;
	len = 0;
	f_break = false;
	while (true) {
		c = **s;
		if (c == 0) {
			break;
		}
		if (c == '\\') {
			(*s)++;
			c = **s;
			str[len] = c;
			len++;
		}
		else if (c == '\"') {
			f_break = true;
		}
		else {
			str[len] = c;
			len++;
		}
		(*s)++;
		if (f_break) {
			break;
		}
	}
	str[len] = 0;
	return true;
}

int string_tools::s_scan_token_comma_separated(
		const char **s, char *str,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char c;
	int len;

	len = 0;
	c = **s;
	if (c == 0 || c == 13 || c == 10) {
		return false;
	}
	if (f_v) {
		cout << "string_tools::s_scan_token_comma_separated seeing character " << (int) c << endl;
	}
#if 0
	if (c == 10 || c == 13) {
		(*s)++;
		//sprintf(str, "END_OF_LINE");
		return false;
	}
#endif
	if (c == ',') {
		(*s)++;
		str[0] = 0;
		//sprintf(str, "");
		return true;
	}
	while (c != 13 && c != ',') {
		if (c == 0) {
			break;
		}
		if (c == '"') {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
			while (true) {
				//cout << "read '" << c << "'" << endl;
				if (c == 0) {
					str[len] = 0;
					cout << "s_scan_token_comma_separated: "
							"end of line inside string" << endl;
					cout << "while scanning '" << str << "'" << endl;
					exit(1);
					break;
				}
				str[len] = c;
				len++;
				if (c == '"') {
					//cout << "end of string" << endl;
					(*s)++;
					c = **s;
					break;
				}
				(*s)++;
				c = **s;
			}
		}
		else {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
		}
	}
	str[len] = 0;
	if (c == ',') {
		(*s)++;
	}
	// printf("token = \"%s\"\n", str);
	return true;
}

void string_tools::scan_permutation_from_string(std::string &s,
	int *&perm, int &degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "string_tools::scan_permutation_from_string s = " << s << endl;
	}
	istringstream ins(s);
	if (f_v) {
		cout << "string_tools::scan_permutation_from_string before scan_permutation_from_stream" << endl;
	}
	scan_permutation_from_stream(ins, perm, degree, verbose_level);
	if (f_v) {
		cout << "string_tools::scan_permutation_from_string done" << endl;
	}
}

void string_tools::scan_permutation_from_stream(std::istream & is,
	int *&perm, int &degree, int verbose_level)
// Scans a permutation from a stream.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "string_tools::scan_permutation_from_string" << endl;
	}

	int l = 20;
	int *cycle; // [l]
	//int *perm; // [l]
	int i, a_last, a, dig, ci;
	char s[1000], c;
	int si, largest_point = 0;
	combinatorics::combinatorics_domain Combi;

	cycle = NEW_int(l);
	perm = NEW_int(l);
	degree = l;
	//l = s_l();
	//perm.m_l(l);
	//cycle.m_l_n(l);
	Combi.perm_identity(perm, l);
	//perm.one();
	while (true) {
		c = get_character(is, verbose_level - 2);
		while (c == ' ' || c == '\t') {
			c = get_character(is, verbose_level - 2);
		}
		ci = 0;
		if (c != '(') {
			break;
		}
		if (f_v) {
			cout << "opening parenthesis" << endl;
		}
		c = get_character(is, verbose_level - 2);
		while (true) {
			while (c == ' ' || c == '\t') {
				c = get_character(is, verbose_level - 2);
			}

			si = 0;
			// read digits:
			while (c >= '0' && c <= '9') {
				s[si++] = c;
				c = get_character(is, verbose_level - 2);
			}
			while (c == ' ' || c == '\t') {
				c = get_character(is, verbose_level - 2);
			}
			if (c == ',') {
				c = get_character(is, verbose_level - 2);
			}
			s[si] = 0;
			dig = atoi(s);
			if (dig > largest_point) {
				largest_point = dig;
			}
			if (f_v) {
				cout << "digit as string: " << s << ", numeric: " << dig << endl;
			}
			if (dig < 0) {
				cout << "string_tools::scan_permutation_from_stream digit < 0" << endl;
				exit(1);
			}
			if (dig >= l) {
				int *perm1;
				int *cycle1;
				//permutation perm1;
				//vector cycle1;
				int l1, i;

				l1 = MAXIMUM(l + (l >> 1), largest_point + 1);
				if (f_v) {
					cout << "string_tools::scan_permutation_from_stream digit = "
						<< dig << " >= " << l
						<< ", extending permutation degree to "
						<< l1 << endl;
				}
				perm1 = NEW_int(l1);
				cycle1 = NEW_int(l1);

				//perm1.m_l(l1);
				for (i = 0; i < l; i++) {
					//perm1.m_ii(i, perm.s_i(i));
					perm1[i] = perm[i];
				}
				for (i = l; i < l1; i++) {
					perm1[i] = i;
				}
				FREE_int(perm);
				perm = perm1;
				degree = l1;
				//perm.swap(perm1);

				//cycle1.m_l_n(l1);
				for (i = 0; i < l; i++) {
					//cycle1.m_ii(i, cycle.s_ii(i));
					cycle1[i] = cycle[i];
				}
				FREE_int(cycle);
				cycle = cycle1;
				//cycle.swap(cycle1);
				l = l1;
			}
			si = 0;
			//cycle.m_ii(ci, dig + 1);
			cycle[ci] = dig;
			ci++;
			if (c == ')') {
				if (f_v) {
					cout << "closing parenthesis, cycle = ";
					for (i = 0; i < ci; i++)
						cout << cycle[i] << " ";
					cout << endl;
				}
				for (i = 1; i < ci; i++) {
					a_last = cycle[i - 1];
					a = cycle[i];
					perm[a_last] = a;
				}
				if (ci > 1) {
					a_last = cycle[ci - 1];
					a = cycle[0];
					perm[a_last] = a;
				}
				ci = 0;
				if (!is) {
					break;
				}
				//c = get_character(is, verbose_level - 2);
				break;
			}
		} // loop for one cycle
		if (!is) {
			break;
		}
		while (c == ' ' || c == '\t') {
			c = get_character(is, verbose_level - 2);
		}
		ci = 0;
	} // end of loop over all cycles
#if 0
	{
		permutation perm1;
		int i;

		perm1.m_l(largest_point + 1);
		for (i = 0; i <= largest_point; i++) {
			perm1.m_ii(i, perm.s_i(i));
		}
		perm.swap(perm1);
	}
#endif
	degree = largest_point + 1;
	if (f_v) {
		cout << "read permutation: ";
		Combi.perm_print(cout, perm, degree);
		cout << endl;
	}
	FREE_int(cycle);
}

void string_tools::chop_string(const char *str, int &argc, char **&argv)
{
	int l, i, len;
	char *s;
	char *buf;
	char *p_buf;

	l = strlen(str);
	s = NEW_char(l + 1);
	buf = NEW_char(l + 1);

	strcpy(s, str);
	p_buf = s;
	i = 0;
	while (true) {
		if (*p_buf == 0) {
			break;
		}
		s_scan_token_arbitrary(&p_buf, buf);

		if (false) {
			cout << "Token " << setw(6) << i << " is '"
					<< buf << "'" << endl;
		}
		i++;
	}
	argc = i;
	argv = NEW_pchar(argc);
	i = 0;
	p_buf = s;
	while (true) {
		if (*p_buf == 0) {
			break;
		}
		s_scan_token_arbitrary(&p_buf, buf);

		if (false) {
			cout << "Token " << setw(6) << i << " is '"
					<< buf << "'" << endl;
		}
		len = strlen(buf);
		argv[i] = NEW_char(len + 1);
		strcpy(argv[i], buf);
		i++;
	}

#if 0
	cout << "argv:" << endl;
	for (i = 0; i < argc; i++) {
		cout << i << " : " << argv[i] << endl;
	}
#endif


	FREE_char(s);
	FREE_char(buf);
#if 0
	for (i = 0; i < argc; i++) {
		FREE_char(argv[i]);
	}
	FREE_pchar(argv);
#endif
}


void string_tools::chop_string_comma_separated(
		const char *str, int &argc, char **&argv)
{
	int l, i, len;
	char *s;
	char *buf;
	const char *p_buf;

	l = strlen(str);
	s = NEW_char(l + 1);
	buf = NEW_char(l + 1);

	strcpy(s, str);
	p_buf = s;
	i = 0;
	while (true) {
		if (*p_buf == 0) {
			break;
		}
		s_scan_token_comma_separated(&p_buf, buf, 0 /* verbose_level */);

		if (false) {
			cout << "Token " << setw(6) << i << " is '"
					<< buf << "'" << endl;
		}
		i++;
	}
	argc = i;
	argv = NEW_pchar(argc);
	i = 0;
	p_buf = s;
	while (true) {
		if (*p_buf == 0) {
			break;
		}
		s_scan_token_comma_separated(&p_buf, buf, 0 /* verbose_level */);

		if (false) {
			cout << "Token " << setw(6) << i << " is '"
					<< buf << "'" << endl;
		}
		len = strlen(buf);
		argv[i] = NEW_char(len + 1);
		strcpy(argv[i], buf);
		i++;
	}

#if 0
	cout << "argv:" << endl;
	for (i = 0; i < argc; i++) {
		cout << i << " : " << argv[i] << endl;
	}
#endif


	FREE_char(s);
	FREE_char(buf);
#if 0
	for (i = 0; i < argc; i++) {
		FREE_char(argv[i]);
	}
	FREE_pchar(argv);
#endif
}



void string_tools::convert_arguments(
		int &argc, const char **argv, std::string *&Argv)
{
	int i;
	vector<string> Arg_vec;

	for (i = 0; i < argc; i++) {

		if (false /*strcmp(argv[i], "-repeat") == 0*/) {
			string variable_name;
			int loop_from;
			int loop_upper_bound;
			int loop_increment;
			int index_of_repeat_start;
			int index_of_repeat_end;

			variable_name.assign(argv[++i]);
			loop_from = atoi(argv[++i]);
			loop_upper_bound = atoi(argv[++i]);
			loop_increment = atoi(argv[++i]);
			i++;
			index_of_repeat_start = i;
			while (i < argc) {
				if (strcmp(argv[i], "-repeat_end") == 0) {
					index_of_repeat_end = i;
					break;
				}
				i++;
			}
			int loop_var;
			int h;
			string variable;

			variable = "%" + variable_name;

			for (loop_var = loop_from; loop_var < loop_upper_bound; loop_var += loop_increment) {
				for (h = index_of_repeat_start; h < index_of_repeat_end; h++) {
					string arg;
					string value_L;
					char str[1000];

					snprintf(str, sizeof(str), "%d", loop_var);
					value_L.assign(str);

					arg.assign(argv[h]);

					while (arg.find(variable) != std::string::npos) {
						arg.replace(arg.find(variable), variable.length(), value_L);
					}


					Arg_vec.push_back(arg);
				}
			}
		}
		else {
			string str;

			str.assign(argv[i]);
			Arg_vec.push_back(str);
		}
	}
	argc = Arg_vec.size();
	Argv = new string[argc];
	for (i = 0; i < argc; i++) {
		Argv[i].assign(Arg_vec[i]);
	}

}

char string_tools::get_character(istream & is, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char c;

	if (!is) {
		cout << "string_tools::get_character at end" << endl;
		exit(1);
	}
	is >> c;
	if (f_v) {
		cout << "string_tools::get_character: \"" << c
			<< "\", ascii=" << (int)c << endl;
	}
	return c;
}

void string_tools::replace_extension_with(char *p, const char *new_ext)
{
	int i, l;

	l = strlen(p);
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '.') {
			p[i] = 0;
			break;
		}
		else if (p[i] == '/') {
			break;
		}
	}
	strcat(p, new_ext);
}

void string_tools::replace_extension_with(std::string &p, const char *new_ext)
{
	int i, l;
	string q;

	l = p.length();
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '.') {
			q = p.substr(0, i);
			//p[i] = 0;
			break;
		}
		else if (p[i] == '/') {
			q = p.substr(0, i);
			break;
		}
	}
	if (i == -1) {
		q = p;
	}
	q += new_ext;
	p = q;
}

void string_tools::chop_off_extension(char *p)
{
	int len = strlen(p);
	int i;

	for (i = len - 1; i >= 0; i--) {
		if (p[i] == '/') {
			break;
		}
		if (p[i] == '.') {
			p[i] = 0;
			break;
		}
	}
}

void string_tools::chop_off_extension_and_path(std::string &p)
{
	chop_off_path(p);
	chop_off_extension(p);
}

void string_tools::chop_off_extension(std::string &p)
{
#if 0
	int l;
	int i;
	string q;

	l = p.length();
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '/') {
			q = p.substr(0, i);
			break;
		}
		if (p[i] == '.') {
			q = p.substr(0, i);
			break;
		}
	}
	if (i == -1) {
		q = p;
	}
	p = q;
#else

	std::string ext;
	std::string q;

	get_extension(p, ext);
	q = p.substr(0, p.length() - ext.length());
	p = q;
#endif
}

void string_tools::chop_off_path(std::string &p)
{
	int l;
	int i;
	string q;

	l = p.length();
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '/') {
			q = p.substr(i + 1, l - i - 1);
			break;
		}
	}
	if (i == -1) {
		q = p;
	}
	p = q;
}


void string_tools::chop_off_extension_if_present(
		std::string &p, const char *ext)
{
	int l1 = p.length();
	int l2 = strlen(ext);

	if (l1 > l2) {
		string q;
		q = p.substr(l1 - l2, l2);
		if (strcmp(p.c_str(), ext) == 0) {
			string r;
			r = q.substr(0, l1 - l2);
			p = r;
		}
	}
}


void string_tools::chop_off_extension_if_present(
		char *p, const char *ext)
{
	int l1 = strlen(p);
	int l2 = strlen(ext);

	if (l1 > l2 && strcmp(p + l1 - l2, ext) == 0) {
		p[l1 - l2] = 0;
	}
}

void string_tools::get_fname_base(const char *p, char *fname_base)
{
	int i, l = strlen(p);

	strcpy(fname_base, p);
	for (i = l - 1; i >= 0; i--) {
		if (fname_base[i] == '.') {
			//cout << "p[" << i << "] is dot" << endl;
			fname_base[i] = 0;
			return;
		}
	}
}

void string_tools::get_extension(std::string &p, std::string &ext)
{
	int i, l = p.length();

	//cout << "string_tools::get_extension " << p << " l=" << l << endl;
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '.') {
			//cout << "p[" << i << "] is dot" << endl;
			ext = p.substr(i, l - i);
			return;
		}
		if (p[i] == '/') {
			break;
		}
	}
	ext.assign("");
}


void string_tools::get_extension_if_present(
		const char *p, char *ext)
{
	int i, l = strlen(p);

	//cout << "string_tools::get_extension_if_present " << p << " l=" << l << endl;
	ext[0] = 0;
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '.') {
			//cout << "p[" << i << "] is dot" << endl;
			strcpy(ext, p + i);
			return;
		}
	}
}

void string_tools::get_extension_if_present_and_chop_off(
		char *p, char *ext)
{
	int i, l = strlen(p);

	//cout << "string_tools::get_extension_if_present " << p << " l=" << l << endl;
	ext[0] = 0;
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '.') {
			//cout << "p[" << i << "] is dot" << endl;
			strcpy(ext, p + i);
			p[i] = 0;
			return;
		}
	}
}

void string_tools::string_fix_escape_characters(std::string &str)
{
	string str_t, str_D, str_B, str_n;

	str_t.assign("\\t");
	str_D.assign("\\D");
	str_B.assign("\\B");
	str_n.assign("\\n");


	while (str.find(str_t) != std::string::npos) {
		str.replace(str.find(str_t),str_t.length(),"\t");
	}

	while (str.find(str_D) != std::string::npos) {
		str.replace(str.find(str_D),str_D.length(),"$");
	}

	while (str.find(str_B) != std::string::npos) {
		str.replace(str.find(str_B),str_B.length(),"\\");
	}

	while (str.find(str_n) != std::string::npos) {
		str.replace(str.find(str_n),str_n.length(),"\n");
	}

}

void string_tools::remove_specific_character(std::string &str, char c)
{
	char st[1000];

	st[0] = c;
	st[1] = 0;
	string str_t;

	str_t.assign(st);


	while (str.find(str_t) != std::string::npos) {
		str.replace(str.find(str_t),str_t.length(),"");
	}


}

void string_tools::create_comma_separated_list(
		std::string &output,
		long int *input, int input_sz)
{
	int i;

	output = std::to_string(input[0]);
	for (i = 1; i < input_sz; i++) {
		output += "," + std::to_string(input[i]);
	}

}

void string_tools::parse_comma_separated_list(
		std::string &input_text, std::vector<std::string> &output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "string_tools::parse_comma_separated_list" << endl;
	}

	string s;

	s.assign(input_text);

	while (true) {

		std::size_t found;
		int len;

		len = s.length();

		found = s.find(',');
		if (found == std::string::npos) {
			if (f_v) {
				cout << "string_tools::parse_comma_separated_list "
						"adding symbol " << s << endl;
			}

			output.push_back(s);
			break;
		}
		else {
			std::string symb = s.substr (0, found);
			std::string rest = s.substr (found + 1, len - found - 1);



			if (f_v) {
				cout << "string_tools::parse_comma_separated_list "
						"adding symbol " << symb << endl;
			}

			output.push_back(symb);


			s = rest;

		}
	}

	if (f_v) {
		cout << "string_tools::parse_value_pairs done" << endl;
	}

}


int string_tools::is_all_whitespace(const char *str)
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
				return true;
			}
			if (str[i] == 'n') {
				continue;
			}
		}
		return false;
	}
	return true;
}

void string_tools::text_to_three_double(std::string &text, double *d)
{
	double *data;
	int data_sz;
	orbiter_kernel_system::numerics Num;

	Num.vec_scan(text.c_str(), data, data_sz);
	if (data_sz != 3) {
		cout << "string_tools::text_to_three_double "
				"data_sz != 3, data_sz = " << data_sz << endl;
		exit(1);
	}
	d[0] = data[0];
	d[1] = data[1];
	d[2] = data[2];
	delete [] data;

}

int string_tools::strcmp_with_or_without(char *p, char *q)
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

int string_tools::starts_with_a_number(std::string &str)
{
	char c;

	c = str.c_str()[0];
	if (c >= '0' && c <= '9') {
		return true;
	}
	else {
		return false;
	}
}

int string_tools::compare_string_string(std::string &str1, std::string &str2)
{
	return strcmp(str1.c_str(), str2.c_str());
}


int string_tools::stringcmp(std::string &str, const char *p)
{
	return strcmp(str.c_str(), p);
}

int string_tools::strtoi(std::string &str)
{
	int i;

	i = atoi(str.c_str());
	return i;
}

int string_tools::str2int(std::string &str)
{
	int i, res, l;

	l = (int) str.length();
	res = 0;
	for (i = 0; i < l; i++) {
		res = (res * 10) + (str[i] - 48);
	}
	return res;
}

long int string_tools::strtolint(std::string &str)
{
	long int i;

	i = atol(str.c_str());
	return i;
}

double string_tools::strtof(std::string &str)
{
	double f;

	f = atof(str.c_str());
	return f;
}

void string_tools::parse_value_pairs(
		std::map<std::string, std::string> &symbol_table,
		std::string &evaluate_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "string_tools::parse_value_pairs" << endl;
	}

	std::vector<std::string> values;

	parse_comma_separated_list(
			evaluate_text, values,
			0 /*verbose_level*/);


	int i;
	std::size_t found;

	for (i = 0; i < values.size(); i++) {

		found = values[i].find('=');
		if (found == std::string::npos) {
			cout << "did not find '=' in variable assignment" << endl;
			exit(1);
		}
		std::string symb = values[i].substr (0, found);
		std::string val = values[i].substr (found + 1, values[i].length() - found - 1);



		if (f_v) {
			cout << "adding symbol " << symb << " = " << val << endl;
		}

		symbol_table[symb] = val;


	}


	if (f_v) {
		cout << "string_tools::parse_value_pairs done" << endl;
	}

}


void string_tools::parse_comma_separated_values(
		std::vector<std::string> &symbol_table,
		std::string &evaluate_text, int verbose_level)
// calls parse_comma_separated_list
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "string_tools::parse_comma_separated_values" << endl;
	}


	parse_comma_separated_list(
			evaluate_text, symbol_table,
			0 /*verbose_level*/);


	if (f_v) {
		cout << "string_tools::parse_comma_separated_values done" << endl;
	}
}

void string_tools::drop_quotes(std::string &in, std::string &out)
{
	int len;

	len = in.length();

	if (len == 0) {
		out = in;
		return;
	}

	if (in[0] == '\"' && in[len - 1] == '\"') {
		out = in.substr(1, len - 2);
	}
	else {
		out = in;
	}
}


void string_tools::parse_comma_separated_strings(
		std::string &in, std::vector<std::string> &out)
{
	const char *p = in.c_str();
	char str[1000];

	while (true) {
		if (!s_scan_token_comma_separated(&p, str, 0 /* verbose_level */)) {
			break;
		}
		string var;

		var.assign(str);

		out.push_back(var);

	}

}


int string_tools::read_schlaefli_label(const char *p)
{
	if (strcmp(p, "a1") == 0) {
		return 0;
	}
	else if (strcmp(p, "a2") == 0) {
		return 1;
	}
	else if (strcmp(p, "a3") == 0) {
		return 2;
	}
	else if (strcmp(p, "a4") == 0) {
		return 3;
	}
	else if (strcmp(p, "a5") == 0) {
		return 4;
	}
	else if (strcmp(p, "a6") == 0) {
		return 5;
	}
	else if (strcmp(p, "b1") == 0) {
		return 6;
	}
	else if (strcmp(p, "b2") == 0) {
		return 7;
	}
	else if (strcmp(p, "b3") == 0) {
		return 8;
	}
	else if (strcmp(p, "b4") == 0) {
		return 9;
	}
	else if (strcmp(p, "b5") == 0) {
		return 10;
	}
	else if (strcmp(p, "b6") == 0) {
		return 11;
	}
	else if (strcmp(p, "c12") == 0) {
		return 12;
	}
	else if (strcmp(p, "c13") == 0) {
		return 13;
	}
	else if (strcmp(p, "c14") == 0) {
		return 14;
	}
	else if (strcmp(p, "c15") == 0) {
		return 15;
	}
	else if (strcmp(p, "c16") == 0) {
		return 16;
	}
	else if (strcmp(p, "c23") == 0) {
		return 17;
	}
	else if (strcmp(p, "c24") == 0) {
		return 18;
	}
	else if (strcmp(p, "c25") == 0) {
		return 19;
	}
	else if (strcmp(p, "c26") == 0) {
		return 20;
	}
	else if (strcmp(p, "c34") == 0) {
		return 21;
	}
	else if (strcmp(p, "c35") == 0) {
		return 22;
	}
	else if (strcmp(p, "c36") == 0) {
		return 23;
	}
	else if (strcmp(p, "c45") == 0) {
		return 24;
	}
	else if (strcmp(p, "c46") == 0) {
		return 25;
	}
	else if (strcmp(p, "c56") == 0) {
		return 26;
	}
	else {
		cout << "string_tools::read_schlaefli_label unknown schlaefli label: " << p << endl;
		exit(1);
	}
}

void string_tools::read_string_of_schlaefli_labels(
		std::string &str, int *&v, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char **argv;
	int i;

	if (f_v) {
		cout << "string_tools::read_string_of_schlaefli_labels" << endl;
	}

	chop_string_comma_separated(str.c_str(), sz, argv);

	if (f_v) {
		cout << "string_tools::read_string_of_schlaefli_labels reading:" << endl;
		for (i = 0; i < sz; i++) {
			cout << i << " : " << argv[i] << endl;
		}
	}

	v = NEW_int(sz);
	for (i = 0; i < sz; i++) {
		v[i] = read_schlaefli_label(argv[i]);
	}


	if (f_v) {
		cout << "string_tools::read_string_of_schlaefli_labels done" << endl;
	}
}

void string_tools::name_of_group_projective(
		std::string &label_txt,
		std::string &label_tex,
		int n, int q, int f_semilinear, int f_special,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "string_tools::name_of_group_projective" << endl;
	}

	if (f_special) {
		if (f_semilinear) {
			label_txt = "PSSL_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm P}\\Sigma {\\rm L}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
		else {
			label_txt = "PSL_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm PSL}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}

	}
	else {
		if (f_semilinear) {
			label_txt = "PGGL_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm P}\\Gamma {\\rm L}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
		else {
			label_txt = "PGL_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm PGL}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
	}

	if (f_v) {
		cout << "string_tools::name_of_group_projective done" << endl;
	}
}


void string_tools::name_of_group_affine(
		std::string &label_txt,
		std::string &label_tex,
		int n, int q, int f_semilinear, int f_special,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "string_tools::name_of_group_affine" << endl;
	}

	if (f_special) {
		if (f_semilinear) {
			label_txt = "ASSL_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm A}\\Sigma {\\rm L}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
		else {
			label_txt = "ASL_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm ASL}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}

	}
	else {
		if (f_semilinear) {
			label_txt = "AGGL_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm A}\\Gamma {\\rm L}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
		else {
			label_txt = "GL_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm AGL}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
	}
	if (f_v) {
		cout << "string_tools::name_of_group_affine done" << endl;
	}

}

void string_tools::name_of_group_general_linear(
		std::string &label_txt,
		std::string &label_tex,
		int n, int q, int f_semilinear, int f_special,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "string_tools::name_of_group_general_linear" << endl;
	}

	if (f_special) {
		if (f_semilinear) {
			label_txt = "SSL_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "\\Sigma {\\rm L}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
		else {
			label_txt = "SL_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm SL}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}

	}
	else {
		if (f_semilinear) {
			label_txt = "GGL_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "\\Gamma {\\rm L}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
		else {
			label_txt = "GL_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm GL}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
	}

	if (f_v) {
		cout << "string_tools::name_of_group_general_linear done" << endl;
	}

}

void string_tools::name_of_orthogonal_group(
		std::string &label_txt,
		std::string &label_tex,
		int epsilon, int n, int q, int f_semilinear, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "string_tools::name_of_orthogonal_group" << endl;
	}

	if (f_semilinear) {
		if (epsilon == 1) {
			label_txt = "PGGOp_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm P}\\Gamma{\\rm O}^+(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
		else if (epsilon == -1) {
			label_txt = "PGGOm_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm P}\\Gamma{\\rm O}^-(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
		else {
			label_txt = "PGGO_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm P}\\Gamma{\\rm O}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
	}
	else {
		if (epsilon == 1) {
			label_txt = "PGOp_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm PGO}^+(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
		else if (epsilon == -1) {
			label_txt = "PGOm_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm PGO}^-(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
		else {
			label_txt = "PGO_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex = "{\\rm PGO}(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
	}

	if (f_v) {
		cout << "string_tools::name_of_orthogonal_group done" << endl;
	}

}

void string_tools::name_of_orthogonal_space(
		std::string &label_txt,
		std::string &label_tex,
		int epsilon, int n, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "string_tools::name_of_orthogonal_space" << endl;
	}

	label_txt = "O_"+ std::to_string(epsilon) + "_"+ std::to_string(n) + "_"+ std::to_string(q);

	if (epsilon == 1) {
		label_tex = "O^+("+ std::to_string(n) + ","+ std::to_string(q) + ")";
	}
	else if (epsilon == 0) {
		label_tex = "O("+ std::to_string(n) + ","+ std::to_string(q) + ")";
	}
	else if (epsilon == -1) {
		label_tex = "O^-("+ std::to_string(n) + ","+ std::to_string(q) + ")";
	}
	else {
		cout << "string_tools::name_of_orthogonal_space illegal value of epsilon" << endl;
		exit(1);
	}


}


void string_tools::name_of_BLT_set(
		std::string &label_txt,
		std::string &label_tex,
		int q, int ocn, int f_embedded, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "string_tools::name_of_BLT_set" << endl;
	}

	if (f_embedded) {
		label_txt = "BLT_"+ std::to_string(q) + "_"+ std::to_string(ocn) + "_embedded";

		label_tex = "BLT_"+ std::to_string(q) + "\\_"+ std::to_string(ocn) + "\\_embedded";
	}
	else {
		label_txt = "BLT_"+ std::to_string(q) + "_"+ std::to_string(ocn);

		label_tex = "BLT\\_"+ std::to_string(q) + "\\_"+ std::to_string(ocn);
	}

}

void string_tools::make_latex_friendly_vector(
		std::vector<std::string> &S, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "string_tools::make_latex_friendly_vector" << endl;
	}

	int i;

	for (i = 0; i < S.size(); i++) {
		string s, s2;

		s = S[i];
		make_latex_friendly_string(s, s2, verbose_level);
		S[i] = s2;
	}
	if (f_v) {
		cout << "string_tools::make_latex_friendly_vector output" << endl;
		for (i = 0; i < S.size(); i++) {
			cout << i << " : " << S[i] << endl;
		}
	}

	if (f_v) {
		cout << "string_tools::make_latex_friendly_vector done" << endl;
	}

}

void string_tools::make_latex_friendly_string(
		std::string &in, std::string &out, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "string_tools::make_latex_friendly_string" << endl;
	}

	string s;
	string s2;
	int len, line, l, j;
	char c;
	int f_tab_no_triangle = true;
	int f_numbers = false;
	int MAX_LEN = 75;

	s = in;
	s2 = "";
	len = 0;
	line = 0;
	l = s.length();
	for (j = 0; j < l; j++) {
		c = s[j];
		if (c == '\t') {
			if (f_tab_no_triangle) {
				s2 += "\\>";
			}
			else {
				s2 += "$\\triangleright$\\>";
			}
			len += 2;
		}
		else if (c == ' ') {
			s2 += "\\ ";
			len++;
		}
		else if (c == '<') {
			s2 += "{\\textless}";
			len++;
		}
		else if (c == '>') {
			s2 += "{\\textgreater}";
			len++;
		}
		else if (c == '|') {
			s2 += "{\\textbar}";
			len++;
		}
		else if (c == '\\') {
			s2 += "\\symbol{92}";
			len++;
		}
		else if (c == '\'') { /* Akut */
			s2 += "\\symbol{19}";
			len++;
		}
		else if (c == ',') {
			s2 += "\\symbol{44}";
			len++;
		}
		else if (c == '!') {
			s2 += "\\symbol{33}";
			len++;
		}
		else if (c == '\"') {
			s2 += "\\symbol{34}";
			len++;
		}
		else if (c == '.') {
			s2 += "\\symbol{46}";
			len++;
		}
		else if (c == '-') {
			s2 += "\\symbol{45}";
			len++;
		}
		else if (c == '#') {
			s2 += "\\symbol{35}";
			len++;
		}
		else if (c == '$') {
			s2 += "\\symbol{36}";
			len++;
		}
		else if (c == '&') {
			s2 += "\\symbol{38}";
			len++;
		}
		else if (c == '~') {
			s2 += "\\symbol{126}";
			len++;
		}
		else if (c == '_') {
			s2 += "\\_";
			len++;
		}
		else if (c == '^') {
			s2 += "\\symbol{94}";
			len++;
		}
		else if (c == '%') {
			s2 += "\\symbol{37}";
			len++;
		}
		else if (c == '{') {
			s2 += "\\symbol{123}";
			len++;
		}
		else if (c == '}') {
			s2 += "\\symbol{125}";
			len++;
		}
#if 1
		else if (c == '\n') {
			s2 += "\\\\\n";
			len = 0;
			line++;
			if (f_numbers) {
				s2 += std::to_string(line) + "\\>";
				}
			}
#endif
		else {
			s2 += c;
			len++;
		}
#if 1
		if (len > MAX_LEN) {
			//printf("len=%d", len);
			s2 += "\\\\\n";
			len = 0;
			if (f_numbers) {
				s2 += "\\>";
			}
		}
#endif

	} // next j

	out = s2;
	if (f_v) {
		cout << "string_tools::make_latex_friendly_string  in:" << in << endl;
		cout << "string_tools::make_latex_friendly_string out:" << out << endl;
	}

	if (f_v) {
		cout << "string_tools::make_latex_friendly_string done" << endl;
	}
}



//#############################################################################


int string_tools_compare_strings(void *a, void *b, void *data)
{
	char *A = (char *) a;
	char *B = (char *) b;
	return strcmp(A, B);
}


}}}
