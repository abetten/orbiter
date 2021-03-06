/*
 * string_tools.cpp
 *
 *  Created on: Apr 15, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {



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
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int string_tools::is_xml_file(const char *fname)
{
	char ext[1000];

	get_extension_if_present(fname, ext);
	if (strcmp(ext, ".xml") == 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int string_tools::s_scan_int(char **s, int *i)
{
	char str1[512];

	if (!s_scan_token(s, str1)) {
		return FALSE;
	}
	if (strcmp(str1, ",") == 0) {
		if (!s_scan_token(s, str1)) {
			return FALSE;
		}
	}
	//*i = atoi(str1);
	sscanf(str1, "%d", i);
	return TRUE;
}

int string_tools::s_scan_lint(char **s, long int *i)
{
	char str1[512];

	if (!s_scan_token(s, str1)) {
		return FALSE;
	}
	if (strcmp(str1, ",") == 0) {
		if (!s_scan_token(s, str1)) {
			return FALSE;
		}
	}
	//*i = atoi(str1);
	sscanf(str1, "%ld", i);
	return TRUE;
}

int string_tools::s_scan_double(char **s, double *d)
{
	char str1[512];
	char c;
	int len;

	//cout << "s_scan_double input='" << *s << "'" << endl;
	while (TRUE) {
		c = **s;
		if (c == 0) {
			return(FALSE);
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
	return TRUE;
}

int string_tools::s_scan_token(char **s, char *str)
{
	char c;
	int len;

	while (TRUE) {
		c = **s;
		if (c == 0) {
			return(FALSE);
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
	return TRUE;
}

int string_tools::s_scan_token_arbitrary(char **s, char *str)
{
	char c;
	int len;

	while (TRUE) {
		c = **s;
		if (c == 0) {
			return(FALSE);
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
	return TRUE;
}

int string_tools::s_scan_str(char **s, char *str)
{
	char c;
	int len, f_break;

	while (TRUE) {
		c = **s;
		if (c == 0) {
			return(FALSE);
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
		return(FALSE);
	}
	(*s)++;
	len = 0;
	f_break = FALSE;
	while (TRUE) {
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
			f_break = TRUE;
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
	return TRUE;
}

int string_tools::s_scan_token_comma_separated(const char **s, char *str)
{
	char c;
	int len;

	len = 0;
	c = **s;
	if (c == 0) {
		return false;
	}
#if 0
	if (c == 10 || c == 13) {
		(*s)++;
		sprintf(str, "END_OF_LINE");
		return FALSE;
	}
#endif
	if (c == ',') {
		(*s)++;
		str[0] = 0;
		//sprintf(str, "");
		return TRUE;
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
			while (TRUE) {
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
	return TRUE;
}

void string_tools::scan_permutation_from_string(const char *s,
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
	combinatorics_domain Combi;

	cycle = NEW_int(l);
	perm = NEW_int(l);
	degree = l;
	//l = s_l();
	//perm.m_l(l);
	//cycle.m_l_n(l);
	Combi.perm_identity(perm, l);
	//perm.one();
	while (TRUE) {
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
		while (TRUE) {
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
	while (TRUE) {
		if (*p_buf == 0) {
			break;
		}
		s_scan_token_arbitrary(&p_buf, buf);

		if (FALSE) {
			cout << "Token " << setw(6) << i << " is '"
					<< buf << "'" << endl;
		}
		i++;
	}
	argc = i;
	argv = NEW_pchar(argc);
	i = 0;
	p_buf = s;
	while (TRUE) {
		if (*p_buf == 0) {
			break;
		}
		s_scan_token_arbitrary(&p_buf, buf);

		if (FALSE) {
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


void string_tools::convert_arguments(int &argc, const char **argv, std::string *&Argv)
{
	int i;
	vector<string> Arg_vec;

	for (i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-repeat") == 0) {
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

			variable.assign("%");
			variable.append(variable_name);

			for (loop_var = loop_from; loop_var < loop_upper_bound; loop_var += loop_increment) {
				for (h = index_of_repeat_start; h < index_of_repeat_end; h++) {
					string arg;
					string value_L;
					char str[1000];

					sprintf(str, "%d", loop_var);
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
	q.append(new_ext);
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

void string_tools::chop_off_extension(std::string &p)
{
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
}

void string_tools::chop_off_extension_if_present(std::string &p, const char *ext)
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


void string_tools::chop_off_extension_if_present(char *p, const char *ext)
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

void string_tools::get_extension_if_present(const char *p, char *ext)
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

void string_tools::get_extension_if_present_and_chop_off(char *p, char *ext)
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
	string str_t("\\t");
	string str_D("\\D");
	string str_B("\\B");
	string str_n("\\n");


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
	string str_t(st);


	while (str.find(str_t) != std::string::npos) {
		str.replace(str.find(str_t),str_t.length(),"");
	}


}


}}
