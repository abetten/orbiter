/* 
 * readline.cpp
 * 
 * readline boilerplate downloaded from 
 * https://tiswww.case.edu/php/chet/readline/readline.html
 *
 * Anton Betten
 * November 18, 2018
 */

/* Standard include files. stdio.h is required. */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <locale.h>

/* Used for select(2) */
#include <sys/types.h>
#include <sys/select.h>

#include <signal.h>

#include <stdio.h>

/* Standard readline include files. */
#include <readline/readline.h>
#include <readline/history.h>

#include "orbiter.h"


//typedef enum orbiter_token_type orbiter_token_type;

enum orbiter_token_type {
		t_error,
		t_label,
		t_int,
		t_symbol,
};

enum orbiter_token_symbol_type {
		t_unknown,
		t_assign,
		t_plus,
		t_minus,
		t_times,
		t_divide,
		t_equal,
		t_bang,
		t_lt,
		t_le,
		t_gt,
		t_ge,
		t_at,
		t_hash,
		t_dollar,
		t_percent,
		t_exp,
		t_amp,
		t_left_paren,
		t_right_paren,
		t_left_bracket,
		t_right_bracket,
		t_left_curly,
		t_right_curly,
		t_vertical_bar,
		t_comma,
		t_dot,
		t_question_mark,
		t_colon,
		t_semicolon,
		t_single_quotes,
		t_double_quotes,
		t_tilda,
		t_acute,
};



class orbiter_token {
public:
	char *str;
	orbiter_token();
	~orbiter_token();
	void null();
	void freeself();
	void init(char *s);
	enum orbiter_token_type type();
	int int_value();
	orbiter_token_symbol_type symbol_type();
};

enum symbol_table_entry_type {
	t_nothing,
	t_intvec,
	t_object,
	t_string,
};

enum symbol_table_object_type {
	t_nothing_object,
	t_finite_field,
	t_action,
};

class orbiter_symbol_table_entry {
public:
	orbiter_token *label;
	enum symbol_table_entry_type type;
	enum symbol_table_object_type object_type ;
	int *vec;
	int vec_len;
	char *str;
	void *ptr;

	orbiter_symbol_table_entry();
	~orbiter_symbol_table_entry();
	void null();
	void freeself();
	void init(char *str_label);
	void print();
};

orbiter_symbol_table_entry::orbiter_symbol_table_entry()
{
	null();
}

orbiter_symbol_table_entry::~orbiter_symbol_table_entry()
{
	freeself();
}

void orbiter_symbol_table_entry::null()
{
	label = NULL;
	type = t_nothing;
	object_type = t_nothing_object;
	vec = NULL;
	vec_len = 0;
	str = NULL;
	ptr = NULL;
}

void orbiter_symbol_table_entry::freeself()
{
	if (label) {
		delete label;
	}
	if (type == t_intvec && vec) {
		FREE_int(vec);
	}
	if (type == t_string && str) {
		FREE_char(str);
	}
	null();
}

void orbiter_symbol_table_entry::init(char *str_label)
{
	label = new orbiter_token;
	label->init(str_label);
}

void orbiter_symbol_table_entry::print()
{
	if (type == t_intvec) {
		int_vec_print(cout, vec, vec_len);
		cout << endl;
	}
	else if (type == t_object) {
		if (object_type == t_finite_field) {
			finite_field *F;

			F = (finite_field *) ptr;
			F->print();
		}
		else if (object_type == t_action) {
			action *A;

			A = (action *) ptr;
			A->print_info();
		}
	}
}


class orbiter_session {
public:
	orbiter_symbol_table_entry *Table;
	int nb_symb;

	orbiter_session();
	~orbiter_session();
	void null();
	void freeself();
	void init();
	int find_symbol(char *str);
	void add_symbol_table_entry(char *str,
			orbiter_symbol_table_entry *Symb);
	void print_symbol_table();
};

orbiter_session::orbiter_session()
{
	null();
}

orbiter_session::~orbiter_session()
{
	freeself();
}

void orbiter_session::null()
{
	Table = NULL;
	nb_symb = 0;
}

void orbiter_session::freeself()
{
	if (Table) {
		delete [] Table;
	}
	null();
}

void orbiter_session::init()
{
}

int orbiter_session::find_symbol(char *str)
{
	int i;

	for (i = 0; i < nb_symb; i++) {
		if (strcmp(str, Table[i].label->str) == 0) {
			return i;
		}
	}
	return -1;
}

void orbiter_session::add_symbol_table_entry(char *str,
		orbiter_symbol_table_entry *Symb)
{
	int idx;
	idx = find_symbol(str);
	if (idx >= 0) {
		cout << "Overriding symbol " << idx << endl;
		Symb[idx].freeself();
		Symb[idx] = *Symb;
	}
	else {
		int i;

		orbiter_symbol_table_entry *old_table = Table;
		Table = new orbiter_symbol_table_entry [nb_symb + 1];
		for (i = 0; i < nb_symb; i++) {
			Table[i] = old_table[i];
			old_table[i].null();
		}
		Table[nb_symb] = *Symb;
		Symb->null();
		nb_symb++;
		delete [] old_table;
	}
}

void orbiter_session::print_symbol_table()
{
	int i;

	if (nb_symb) {
		for (i = 0; i < nb_symb; i++) {
			cout << i << " : " << Table[i].label->str << " : ";
			Table[i].print();
			cout << endl;
		}
	}
	else {
		cout << "symbol table is empty" << endl;
	}
}

	orbiter_session *Session = NULL;



static void cb_linehandler (char *);
static void sighandler (int);
void do_something(char *input_line, int verbose_level);
void execute(orbiter_token *T, int nb_T, int verbose_level);
void execute_assignment(orbiter_token *T, int nb_T, int verbose_level);
void refresh_prompt();
void init_prompt();
void tokenize(char *line, orbiter_token *&T, int &nb_tokens);
int is_whitespace(char c);
int is_special_symbol(char c);
int s_scan_token_orbiter(char **s, char *str);

int running;
int sigwinch_received;
char prompt[1000];
int prompt_counter = 0;
#define PROMPT_MASK "Orbiter %d > "

//const char *prompt = "Orbiter > ";

/* Handle SIGWINCH and window size changes when readline is not active and
   reading a character. */
static void
sighandler (int sig)
{
  sigwinch_received = 1;
}

/* Callback function called for each line when accept-line executed, EOF
   seen, or EOF character read.  This sets a flag and returns; it could
   also call exit(3). */
static void
cb_linehandler (char *line)
{
  /* Can use ^D (stty eof) or `exit' to exit. */
  if (line == NULL || strcmp (line, "exit") == 0)
    {
      if (line == 0)
        printf ("\n");
      printf ("exit\n");
      /* This function needs to be called to reset the terminal settings,
         and calling it from the line handler keeps one extra prompt from
         being displayed. */
      rl_callback_handler_remove ();

      running = 0;
    }
  else
    {
      if (*line)
        add_history (line);
      do_something(line, 2 /* verbose_level */);
      //printf ("input line: %s\n", line);
      free (line);
    }
}

int
main (int c, char **v)
{
  fd_set fds;
  int r;


	Session = new orbiter_session;
	Session->init();

	//refresh_prompt();

  /* Set the default locale values according to environment variables. */
  setlocale (LC_ALL, "");

  /* Handle window size changes when readline is not active and reading
     characters. */
  signal (SIGWINCH, sighandler);

	/* Install the line handler. */
	//rl_callback_handler_install (prompt, cb_linehandler);
	init_prompt();

  /* Enter a simple event loop.  This waits until something is available
     to read on readline's input stream (defaults to standard input) and
     calls the builtin character read callback to read it.  It does not
     have to modify the user's terminal settings. */
  running = 1;
  while (running)
    {
      FD_ZERO (&fds);
      FD_SET (fileno (rl_instream), &fds);    

      r = select (FD_SETSIZE, &fds, NULL, NULL, NULL);
      if (r < 0 /*&& errno != EINTR*/)
        {
          perror ("rltest: select");
          rl_callback_handler_remove ();
          break;
        }
      if (sigwinch_received)
	{
	  rl_resize_terminal ();
	  sigwinch_received = 0;
	}
      if (r < 0)
	continue;     

      if (FD_ISSET (fileno (rl_instream), &fds))
        rl_callback_read_char ();
    }

  //printf ("rltest: Event loop has exited\n");
  printf ("Good-bye!\n");
  return 0;
}

void do_something(char *input_line, int verbose_level)
{
	orbiter_token *T;
	int nb_T;
	int i;

	prompt_counter++;
	printf ("input line: %s\n", input_line);
	tokenize(input_line, T, nb_T);

	cout << "we found " << nb_T << " tokens" << endl;
	for (i = 0; i < nb_T; i++) {
		enum orbiter_token_type t;

		t = T[i].type();
		cout << i << " : " << T[i].str << " : ";
		if (t == t_label) {
			cout << "label: " << T[i].str;
		} else if (t == t_int) {
			cout << "int: " << T[i].int_value();
		} else if (t == t_symbol) {
			cout << "symbol: " << T[i].str << " : " << T[i].symbol_type();
		}
		cout << endl;
	}
	execute(T, nb_T, verbose_level);
	delete [] T;
	refresh_prompt();
}

void execute(orbiter_token *T, int nb_T, int verbose_level)
{
	if (nb_T >= 3 &&
			T[0].type() == t_label &&
			T[1].type() == t_symbol &&
			T[1].symbol_type() == t_assign) {
		execute_assignment(T, nb_T, verbose_level);
	}
	else if (T[0].type() == t_label) {
		cout << "trying to execute command " << T[0].str << endl;
		if (strcmp(T[0].str, "print") == 0) {
			int i;
			for (i = 1; i < nb_T; i++) {
				int idx;
				idx = Session->find_symbol(T[i].str);
				if (idx >= 0) {
					cout << T[i].str << " = ";
					Session->Table[idx].print();
				}
				else {
					cout << "cound not find symbol " << T[i].str << endl;
				}
			}
		}
		else if (strcmp(T[0].str, "printtable") == 0) {
#if 0
			int idx;
			idx = Session->find_symbol(T[0].str);
			if (idx >= 0) {
				cout << "Symbol " << idx << " has label " << T[0].str << endl;
			} else {
				cout << "Symbol " << T[0].str << " does not exist" << endl;
			}
#endif
			Session->print_symbol_table();
		}
		else {
			cout << "unknown operation" << endl;
		}
	}
}

void execute_assignment(orbiter_token *T, int nb_T, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	cout << "assignment to " << T[0].str << endl;

	orbiter_symbol_table_entry *Symb;

	Symb = new orbiter_symbol_table_entry;
	Symb->init(T[0].str);
	int idx;
	idx = Session->find_symbol(T[0].str);
	if (idx >= 0) {
		cout << "Overriding symbol " << idx << endl;
	}
	if (T[2].type() == t_int) {
		int *v;
		int len;
		int i;

		len = 0;
		v = NEW_int(nb_T);
		for (i = 2; i < nb_T; i++) {
			if (T[i].type() == t_int) {
				v[len++] = T[i].int_value();
			}
		}
		cout << "RHS is int vec of length " << len << " : ";
		int_vec_print(cout, v, len);
		cout << endl;
		Symb->type = t_intvec;
		Symb->vec = v;
		Symb->vec_len = len;
		Session->add_symbol_table_entry(T[0].str, Symb);
	}
	else if (T[2].type() == t_label) {
		if (strcmp(T[2].str, "finite_field") == 0) {
			if (nb_T <= 2) {
				cout << "need another argument to specify the size of the field" << endl;
			}
			else {
				if (T[3].type() == t_int) {
					int q;

					q = T[3].int_value();
					finite_field *F;

					F = NEW_OBJECT(finite_field);
					F->init(q);
					Symb->type = t_object;
					Symb->object_type = t_finite_field;
					Symb->ptr = F;
					Session->add_symbol_table_entry(T[0].str, Symb);
				}
			}
		}
		else if (strcmp(T[2].str, "group") == 0) {
			if (nb_T <= 2) {
				cout << "need another argument to specify the parameters of the group" << endl;
			}
			else {
				char arguments[1000];
				int i;
				finite_field *F = NULL;

				if (T[3].type() != t_label) {
					cout << "the first argument after group must be label of the finite field" << endl;
				}
				else {
					int idx;
					idx = Session->find_symbol(T[3].str);

					if (Session->Table[idx].type != t_object) {
						cout << "object with label " << T[3].str << " not of type t_object" << endl;
					}
					if (Session->Table[idx].object_type == t_finite_field) {
						F = (finite_field *) Session->Table[idx].ptr;
						F->print();
					}
					arguments[0] = 0;
					int f_next_symbol_is_keyword;
					int f_this_symbol_is_keyword;

					f_next_symbol_is_keyword = FALSE;
					for (i = 4; i < nb_T; i++) {
						f_this_symbol_is_keyword = f_next_symbol_is_keyword;
						f_next_symbol_is_keyword = FALSE;
						if (T[i].type() == t_label) {
							if (f_this_symbol_is_keyword) {
								sprintf(arguments + strlen(arguments), "-%s ", T[i].str);
							}
							else {
								int idx;
								idx = Session->find_symbol(T[i].str);
								if (idx >= 0) {
									cout << T[i].str << " = ";
									Session->Table[idx].print();
								}
								if (Session->Table[idx].type != t_intvec) {
									cout << "label " << T[i].str << " not of type int" << endl;
								}
								else {
									sprintf(arguments + strlen(arguments), "%d ", Session->Table[idx].vec[0]);
								}
							}
						}
						else if (T[i].type() == t_int) {
							sprintf(arguments + strlen(arguments), "%s ", T[i].str);
						}
						else if (T[i].type() == t_symbol) {
							if (T[i].symbol_type() == t_minus) {
								f_next_symbol_is_keyword = TRUE;
							}
						}
					}
					sprintf(arguments + strlen(arguments), " -end");
					cout << "creating group from arguments: " << arguments << endl;
					linear_group_description *Descr;
					linear_group *LG;
					Descr = NEW_OBJECT(linear_group_description);
					cout << "before Descr->read_arguments_from_string" << endl;
					Descr->read_arguments_from_string(
							arguments, 2 /*verbose_level*/);
					cout << "after Descr->read_arguments_from_string" << endl;
					if (F == NULL) {
						cout << "finite field is missing" << endl;
					}
					else {
						Descr->F = F;

						LG = NEW_OBJECT(linear_group);
						if (f_v) {
							cout << "linear_group before LG->init, "
									"creating the group" << endl;
							}

						cout << "before LG->init" << endl;
						LG->init(Descr, verbose_level - 1);
						cout << "after LG->init" << endl;

						if (f_v) {
							cout << "linear_group after LG->init" << endl;
							}

						action *A;

						A = LG->A2;

						cout << "created group " << LG->prefix << endl;

						Symb->type = t_object;
						Symb->object_type = t_action;
						Symb->ptr = A;
						Session->add_symbol_table_entry(T[0].str, Symb);
					}
				}
			}
		}
		else {
			cout << "unknown label " << T[2].str << endl;
		}
	}
}

void refresh_prompt()
{
	sprintf(prompt, PROMPT_MASK, prompt_counter);
	rl_callback_handler_remove ();
	rl_callback_handler_install (prompt, cb_linehandler);
}

void init_prompt()
{
	sprintf(prompt, PROMPT_MASK, prompt_counter);
	rl_callback_handler_install (prompt, cb_linehandler);
}

void tokenize(char *line, orbiter_token *&T, int &nb_tokens)
{
	int f_vv = TRUE;
	char *p_buf = line;
	char str[100000];
	int i;

	i = 0;
	while (TRUE) {
		if (*p_buf == 0) {
			break;
			}
		//s_scan_token(&p_buf, str);
		//s_scan_token(&p_buf, str);
		/* r =*/ s_scan_token_orbiter(&p_buf, str);

		if (f_vv) {
			cout << "Token " << setw(6) << i << " is '"
					<< str << "'" << endl;
			}
#if 0
		if (strcmp(str, ",") == 0) {
			continue;
			}
#endif
		i++;
		}
	nb_tokens = i;
	cout << "found " << nb_tokens << " tokens " << endl;

	T = new orbiter_token[nb_tokens];
	p_buf = line;
	i = 0;
	while (TRUE) {
		if (*p_buf == 0) {
			break;
			}
		//s_scan_token(&p_buf, str);
		//s_scan_token(&p_buf, str);
		/* r =*/ s_scan_token_orbiter(&p_buf, str);
		T[i].init(str);

		if (f_vv) {
			cout << "Token " << setw(6) << i << " is '"
					<< str << "'" << endl;
			}
#if 0
		if (strcmp(str, ",") == 0) {
			continue;
			}
#endif
		i++;
		}

}

int is_whitespace(char c)
{
	if (c == ' ' || c == '\t' ||
		c == '\r' || c == 10 || c == 13) {
		return TRUE;
	} else {
		return FALSE;
	}

}

int is_special(char c)
{
	if (c == ' ' || c == '\t' ||
		c == '\r' || c == 10 || c == 13) {
		return TRUE;
	} else {
		return FALSE;
	}

}

int s_scan_token_orbiter(char **s, char *str)
{
	char c;
	int len;

	while (TRUE) {
		c = **s;
		if (c == 0) {
			return(FALSE);
			}
		if (is_whitespace(c)) {
			(*s)++;
			continue;
			}
		break;
		}
	len = 0;
	c = **s;
	if (c == 0) {
		str[len] = 0;
		return TRUE;
	}
	if (isalpha(c)) {
		str[len] = c;
		len++;
		(*s)++;
		c = **s;
		while (c != 0 && c != ' ' && c != '\t' &&
				c != '\r' && c != 10 && c != 13 && !is_special_symbol(c)) {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
		}
		str[len] = 0;
		return TRUE;
	}
	else if (isdigit(c)) {
		while (c != 0 && c != ' ' && c != '\t' &&
				c != '\r' && c != 10 && c != 13 && !is_special_symbol(c)) {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
		}
		str[len] = 0;
		return TRUE;
	}
	else if (is_special_symbol(c)) {
		if (c == '<') {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
			if (c == '-') {
				str[len] = c;
				len++;
				(*s)++;
				c = **s;
				str[len] = 0;
				return TRUE;
			}
			else if (c == '=') {
				str[len] = c;
				len++;
				(*s)++;
				c = **s;
				str[len] = 0;
				return TRUE;
			}
			str[len] = 0;
			return TRUE;
		}
		else if (c == '>') {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
			if (c == '=') {
				str[len] = c;
				len++;
				(*s)++;
				c = **s;
				str[len] = 0;
				return TRUE;
			}
			str[len] = 0;
			return TRUE;
		}
		else {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
			str[len] = 0;
			return TRUE;
		}
	}
	cout << "unknown symbol: \'" << c << "'" << endl;
	return FALSE;
}


orbiter_token::orbiter_token()
{
	null();
}

orbiter_token::~orbiter_token()
{
	freeself();
}

void orbiter_token::null()
{
	str = NULL;
}

void orbiter_token::freeself()
{
	if (str) {
		FREE_char(str);
	}
	null();
}

void orbiter_token::init(char *s)
{
	int l;

	l = strlen(s);
	str = NEW_char(l + 1);
	strcpy(str, s);
}

orbiter_token_type orbiter_token::type()
{
	int l = strlen(str);

	if (l == 0) {
		return t_error;
	}
	if (isalpha(str[0])) {
		return t_label;
	}
	if (isdigit(str[0])) {
		return t_int;
	}
	return t_symbol;
}

int orbiter_token::int_value()
{
	int a;

	a = atoi(str);
	return a;
}

orbiter_token_symbol_type orbiter_token::symbol_type()
{
	if (strcmp(str, "<-") == 0) {
		return t_assign;
	}
	else if (strcmp(str, "+") == 0) {
		return t_plus;
	}
	else if (strcmp(str, "-") == 0) {
		return t_minus;
	}
	else if (strcmp(str, "*") == 0) {
		return t_times;
	}
	else if (strcmp(str, "/") == 0) {
		return t_divide;
	}
	else if (strcmp(str, "=") == 0) {
		return t_equal;
	}
	else if (strcmp(str, "!") == 0) {
		return t_bang;
	}
	else if (strcmp(str, "<") == 0) {
		return t_lt;
	}
	else if (strcmp(str, "<=") == 0) {
		return t_le;
	}
	else if (strcmp(str, ">") == 0) {
		return t_gt;
	}
	else if (strcmp(str, ">=") == 0) {
		return t_ge;
	}
	else if (strcmp(str, "@") == 0) {
		return t_at;
	}
	else if (strcmp(str, "#") == 0) {
		return t_hash;
	}
	else if (strcmp(str, "$") == 0) {
		return t_dollar;
	}
	else if (strcmp(str, "%") == 0) {
		return t_percent;
	}
	else if (strcmp(str, "^") == 0) {
		return t_exp;
	}
	else if (strcmp(str, "&") == 0) {
		return t_amp;
	}
	else if (strcmp(str, "(") == 0) {
		return t_left_paren;
	}
	else if (strcmp(str, ")") == 0) {
		return t_right_paren;
	}
	else if (strcmp(str, "[") == 0) {
		return t_left_bracket;
	}
	else if (strcmp(str, "]") == 0) {
		return t_right_bracket;
	}
	else if (strcmp(str, "{") == 0) {
		return t_left_curly;
	}
	else if (strcmp(str, "}") == 0) {
		return t_right_curly;
	}
	else if (strcmp(str, "|") == 0) {
		return t_vertical_bar;
	}
	else if (strcmp(str, ",") == 0) {
		return t_comma;
	}
	else if (strcmp(str, ".") == 0) {
		return t_dot;
	}
	else if (strcmp(str, "?") == 0) {
		return t_question_mark;
	}
	else if (strcmp(str, ":") == 0) {
		return t_colon;
	}
	else if (strcmp(str, ";") == 0) {
		return t_semicolon;
	}
	else if (strcmp(str, "\'") == 0) {
		return t_single_quotes;
	}
	else if (strcmp(str, "\"") == 0) {
		return t_double_quotes;
	}
	else if (strcmp(str, "~") == 0) {
		return t_tilda;
	}
	else if (strcmp(str, "`") == 0) {
		return t_acute;
	}
	else
		return t_unknown;
}

int is_special_symbol(char c)
{
	if (c == '+') {
		return t_plus;
	}
	else if (c == '-') {
		return t_minus;
	}
	else if (c == '*') {
		return t_times;
	}
	else if (c == '/') {
		return t_divide;
	}
	else if (c == '=') {
		return t_equal;
	}
	else if (c == '!') {
		return t_bang;
	}
	else if (c == '<') {
		return t_lt;
	}
	else if (c == '>') {
		return t_gt;
	}
	else if (c == '@') {
		return t_at;
	}
	else if (c == '#') {
		return t_hash;
	}
	else if (c == '$') {
		return t_dollar;
	}
	else if (c == '%') {
		return t_percent;
	}
	else if (c == '^') {
		return t_exp;
	}
	else if (c == '&') {
		return t_amp;
	}
	else if (c == '(') {
		return t_left_paren;
	}
	else if (c == ')') {
		return t_right_paren;
	}
	else if (c == '[') {
		return t_left_bracket;
	}
	else if (c == ']') {
		return t_right_bracket;
	}
	else if (c == '{') {
		return t_left_curly;
	}
	else if (c == '}') {
		return t_right_curly;
	}
	else if (c == '|') {
		return t_vertical_bar;
	}
	else if (c == ',') {
		return t_comma;
	}
	else if (c == '.') {
		return t_dot;
	}
	else if (c == '?') {
		return t_question_mark;
	}
	else if (c == ':') {
		return t_colon;
	}
	else if (c == ';') {
		return t_semicolon;
	}
	else if (c == '\'') {
		return t_single_quotes;
	}
	else if (c == '\"') {
		return t_double_quotes;
	}
	else if (c == '~') {
		return t_tilda;
	}
	else if (c == '`') {
		return t_acute;
	}
	else
		return t_unknown;

}
