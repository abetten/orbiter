// homogeneous_polynomial_domain.cpp
//
// Anton Betten
//
// September 9, 2016



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


homogeneous_polynomial_domain::homogeneous_polynomial_domain()
{
	Monomial_ordering_type = t_LEX;
	F = NULL;
	nb_monomials = 0;
	Monomials = NULL;
	symbols = NULL;
	symbols_latex = NULL;
	monomial_symbols = NULL;
	monomial_symbols_latex = NULL;
	monomial_symbols_easy = NULL;
	Variables = NULL;
	nb_affine = 0;
	Affine = NULL;
	v = NULL;
	Affine_to_monomial = NULL;
	coeff2 = NULL;
	coeff3 = NULL;
	coeff4 = NULL;
	factors = NULL;
	my_affine = NULL;
	P = NULL;
	base_cols = NULL;
	type1 = NULL;
	type2 = NULL;


	q = 0;
	nb_variables = 0;
	degree = 0;


	//null();
}

homogeneous_polynomial_domain::~homogeneous_polynomial_domain()
{
	freeself();
}

void homogeneous_polynomial_domain::freeself()
{
	int i;
	
	if (v) {
		FREE_int(v);
	}
	if (Monomials) {
		FREE_int(Monomials);
	}
	if (symbols) {
		delete [] symbols;
	}
	if (symbols_latex) {
		delete [] symbols_latex;
	}
	if (monomial_symbols) {
		for (i = 0; i < nb_monomials; i++) {
			FREE_char(monomial_symbols[i]);
		}
		FREE_pchar(monomial_symbols);
	}
	if (monomial_symbols_latex) {
		for (i = 0; i < nb_monomials; i++) {
			FREE_char(monomial_symbols_latex[i]);
		}
		FREE_pchar(monomial_symbols_latex);
	}
	if (monomial_symbols_easy) {
		for (i = 0; i < nb_monomials; i++) {
			FREE_char(monomial_symbols_easy[i]);
		}
		FREE_pchar(monomial_symbols_easy);
	}
	if (Variables) {
		FREE_int(Variables);
	}
	if (Affine) {
		FREE_int(Affine);
	}
	if (Affine_to_monomial) {
		FREE_int(Affine_to_monomial);
	}
	if (coeff2) {
		FREE_int(coeff2);
	}
	if (coeff3) {
		FREE_int(coeff3);
	}
	if (coeff4) {
		FREE_int(coeff4);
	}
	if (factors) {
		FREE_int(factors);
	}
	if (my_affine) {
		FREE_int(my_affine);
	}
	if (P) {
		FREE_OBJECT(P);
	}
	if (base_cols) {
		FREE_int(base_cols);
	}
	if (type1) {
		FREE_int(type1);
	}
	if (type2) {
		FREE_int(type2);
	}
	null();
}

void homogeneous_polynomial_domain::null()
{
}

void homogeneous_polynomial_domain::init(finite_field *F,
		int nb_vars, int degree, int f_init_incidence_structure,
		monomial_ordering_type Monomial_ordering_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init" << endl;
	}
	homogeneous_polynomial_domain::F = F;
	q = F->q;
	homogeneous_polynomial_domain::nb_variables = nb_vars;
	homogeneous_polynomial_domain::degree = degree;
	homogeneous_polynomial_domain::Monomial_ordering_type = Monomial_ordering_type;
	
	v = NEW_int(nb_variables);
	type1 = NEW_int(degree + 1);
	type2 = NEW_int(degree + 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::init before make_monomials" << endl;
	}
	make_monomials(Monomial_ordering_type, verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::init after make_monomials" << endl;
	}
	
	m = MAXIMUM(nb_monomials, degree + 1);
		// substitute_semilinear needs [nb_monomials]
		// substitute_line needs [degree + 1]

	coeff2 = NEW_int(m);
	coeff3 = NEW_int(m);
	coeff4 = NEW_int(m);
	factors = NEW_int(degree);

	my_affine = NEW_int(degree);



	P = NEW_OBJECT(projective_space);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::init before P->init" << endl;
	}
	P->init(nb_variables - 1, F,
		f_init_incidence_structure, 
		verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::init after P->init" << endl;
	}
	base_cols = NEW_int(nb_monomials);
	
	if (f_v) {
		cout << "homogeneous_polynomial_domain::init done" << endl;
	}
	
}

int homogeneous_polynomial_domain::get_nb_monomials()
{
	return nb_monomials;
}

projective_space *homogeneous_polynomial_domain::get_P()
{
	return P;
}

finite_field *homogeneous_polynomial_domain::get_F()
{
	return F;
}

int homogeneous_polynomial_domain::get_monomial(int i, int j)
{
	if (j > nb_variables) {
		cout << "homogeneous_polynomial_domain::get_monomial j > nb_variables" << endl;
		exit(1);
	}
	return Monomials[i * nb_variables + j];
}

char *homogeneous_polynomial_domain::get_monomial_symbol_easy(int i)
{
	return monomial_symbols_easy[i];
}

int *homogeneous_polynomial_domain::get_monomial_pointer(int i)
{
	return Monomials + i * nb_variables;
}

int homogeneous_polynomial_domain::evaluate_monomial(int idx_of_monomial, int *coords)
{
	int r;

	r = F->evaluate_monomial(
			Monomials + idx_of_monomial * nb_variables,
			coords, nb_variables);
	return r;
}

void homogeneous_polynomial_domain::remake_symbols(int symbol_offset,
		const char *symbol_mask, const char *symbol_mask_latex,
		int verbose_level)
{
	int i; //, l;
	char label[1000];

	if (symbols) {
		delete [] symbols;
	}
	if (symbols_latex) {
		delete [] symbols_latex;
	}
	symbols = new string [nb_variables];
	symbols_latex = new string [nb_variables];
	for (i = 0; i < nb_variables; i++) {
		snprintf(label, 1000, symbol_mask, i + symbol_offset);
		symbols[i].assign(label);
	}
	for (i = 0; i < nb_variables; i++) {
		snprintf(label, 1000, symbol_mask_latex, i + symbol_offset);
		symbols_latex[i].assign(label);
	}
}

void homogeneous_polynomial_domain::remake_symbols_interval(int symbol_offset,
		int from, int len,
		const char *symbol_mask, const char *symbol_mask_latex,
		int verbose_level)
{
	int i, j; //, l;
	char label[1000];

	for (j = 0; j < len; j++) {
		i = from + j;
		snprintf(label, 1000, symbol_mask, i + symbol_offset);
		symbols[i].assign(label);
	}
	for (j = 0; j < len; j++) {
		i = from + j;
		snprintf(label, 1000, symbol_mask_latex, i + symbol_offset);
		symbols_latex[i].assign(label);
	}

}

void homogeneous_polynomial_domain::make_monomials(
		monomial_ordering_type Monomial_ordering_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, h, idx, t;
	number_theory_domain NT;
	geometry_global Gg;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials" << endl;
	}
	
	nb_monomials = Combi.int_n_choose_k(nb_variables + degree - 1, nb_variables - 1);

	diophant *D;

	D = NEW_OBJECT(diophant);

	D->open(1, nb_variables);
	D->fill_coefficient_matrix_with(1);
	D->RHSi(0) = degree;
	D->type[0] = t_EQ;
	D->set_x_min_constant(0);
	D->set_x_max_constant(degree);
	D->f_has_sum = TRUE;
	D->sum = degree;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials before D->solve_all_betten" << endl;
	}
	D->solve_all_betten(0 /* verbose_level */);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials after D->solve_all_betten" << endl;
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"We found " << D->_resultanz << " monomials" << endl;
	}

	int nb_sol;

	D->get_solutions_full_length(Monomials, nb_sol, 0 /* verbose_level */);
	if (f_v) {
		cout << "There are " << nb_sol << " monomials." << endl;

		if (nb_sol < 100) {
			int_matrix_print(Monomials, nb_sol, nb_variables);
		}
		else {
			cout << "too many to print" << endl;
		}
	}

	if (nb_sol != nb_monomials) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"nb_sol != nb_monomials" << endl;
		cout << "nb_sol=" << nb_sol << endl;
		cout << "nb_monomials=" << nb_monomials << endl;
		exit(1);
	}

	FREE_OBJECT(D);
	
	if (Monomial_ordering_type == t_PART) {

		if (f_v) {
			cout << "homogeneous_polynomial_domain::make_monomials rearranging by partition type:" << endl;
		}
		rearrange_monomials_by_partition_type(verbose_level);

	}

	if (f_v) {
		cout << "After rearranging by type:" << endl;
		if (nb_monomials < 100) {
			int_matrix_print(Monomials, nb_monomials, nb_variables);
		}
		else {
			cout << "too many to print" << endl;
		}
	}

	char label[1000];
	int l;

	symbols = new string[nb_variables];
	for (i = 0; i < nb_variables; i++) {

		
		if (TRUE) {
			label[0] = 'X';
			label[1] = '0' + i;
			label[2] = 0;
		}
		else {
			label[0] = 'A' + i;
			label[1] = 0;
		}
		symbols[i].assign(label);
	}
	symbols_latex = new string[nb_variables];
	for (i = 0; i < nb_variables; i++) {

		//int l;
		
		if (TRUE) {
			label[0] = 'X';
			label[1] = '_';
			label[2] = '0' + i;
			label[3] = 0;
		}
		else {
			label[0] = 'A' + i;
			label[1] = 0;
		}
		symbols_latex[i].assign(label);
	}

	int f_first = FALSE;

	monomial_symbols = NEW_pchar(nb_monomials);
	for (i = 0; i < nb_monomials; i++) {
		label[0] = 0;
		f_first = TRUE;
		for (j = 0; j < nb_variables; j++) {
			a = Monomials[i * nb_variables + j];
			if (a) {
				if (!f_first) {
					strcat(label, "*");
				}
				else {
					f_first = FALSE;
				}
				strcat(label, symbols[j].c_str());
				if (a > 1) {
					sprintf(label + strlen(label), "^%d", a);
				}
			}
		}
		l = strlen(label);
		monomial_symbols[i] = NEW_char(l + 1);
		strcpy(monomial_symbols[i], label);
	}

	monomial_symbols_latex = NEW_pchar(nb_monomials);
	for (i = 0; i < nb_monomials; i++) {
		label[0] = 0;
		for (j = 0; j < nb_variables; j++) {
			a = Monomials[i * nb_variables + j];
			if (a) {
				strcat(label, symbols_latex[j].c_str());
				if (a > 1) {
					if (a >= 10) {
						sprintf(label + strlen(label), "^{%d}", a);
					}
					else {
						sprintf(label + strlen(label), "^%d", a);
					}
				}
			}
		}
		l = strlen(label);
		monomial_symbols_latex[i] = NEW_char(l + 1);
		strcpy(monomial_symbols_latex[i], label);
	}

	Variables = NEW_int(nb_monomials * degree);
	for (i = 0; i < nb_monomials; i++) {
		h = 0;
		for (j = 0; j < nb_variables; j++) {
			a = Monomials[i * nb_variables + j];
			for (t = 0; t < a; t++) {
				Variables[i * degree + h] = j;
				h++;
			}
		}
		if (h != degree) {
			cout << "homogeneous_polynomial_domain::make_monomials "
					"h != degree" << endl;
			exit(1);
		}
	}


	monomial_symbols_easy = NEW_pchar(nb_monomials);
	for (i = 0; i < nb_monomials; i++) {
		label[0] = 'X';
		label[1] = 0;
		for (j = 0; j < degree; j++) {
			a = Variables[i * degree + j];
			sprintf(label + strlen(label), "%d", a);
		}
		l = strlen(label);
		monomial_symbols_easy[i] = NEW_char(l + 1);
		strcpy(monomial_symbols_easy[i], label);
	}


	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials the "
				"variable lists are:" << endl;
		if (nb_monomials < 100) {
			for (i = 0; i < nb_monomials; i++) {
				cout << i << " : " << monomial_symbols[i] << " : ";
				int_vec_print(cout, Variables + i * degree, degree);
				cout << endl;
			}
		}
		else {
			cout << "too many to print" << endl;
		}
	}




	nb_affine = NT.i_power_j(nb_variables, degree);

	if (nb_affine < ONE_MILLION) {
		Affine = NEW_int(nb_affine * degree);
		if (f_v) {
			cout << "homogeneous_polynomial_domain::make_monomials  "
					"Affine, nb_affine=" << nb_affine << endl;
		}
		for (h = 0; h < nb_affine; h++) {
			Gg.AG_element_unrank(nb_variables /* q */, Affine + h * degree, 1, degree, h);
		}
		if (FALSE) {
			cout << "homogeneous_polynomial_domain::make_monomials  "
					"Affine" << endl;
			int_matrix_print(Affine, nb_affine, degree);
		}
		Affine_to_monomial = NEW_int(nb_affine);
		for (i = 0; i < nb_affine; i++) {
			if (i > 0 && (i & ((1 << 20) - 1)) == 0) {
				cout << "homogeneous_polynomial_domain::make_monomials "
						"i = " << i << " / " << nb_affine << endl;
			}
			int_vec_zero(v, nb_variables);
			for (j = 0; j < degree; j++) {
				a = Affine[i * degree + j];
				v[a]++;
			}
			idx = index_of_monomial(v);
			Affine_to_monomial[i] = idx;
		}
	}
	else {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"nb_affine is too big, skipping Affine_to_monomial" << endl;
		Affine = NULL;
		Affine_to_monomial = NULL;
	}

	if (FALSE) {
		cout << "homogeneous_polynomial_domain::make_monomials "
				"Affine : idx:" << endl;
		for (i = 0; i < nb_affine; i++) {
			cout << i << " : ";
			int_vec_print(cout, Affine + i * degree, degree);
			cout << " : " << Affine_to_monomial[i] << endl;
		}
	}
	

	if (f_v) {
		cout << "homogeneous_polynomial_domain::make_monomials done" << endl;
	}
}

void homogeneous_polynomial_domain::rearrange_monomials_by_partition_type(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sorting Sorting;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::rearrange_monomials_by_partition_type" << endl;
	}

	Sorting.Heapsort_general(Monomials, nb_monomials,
		homogeneous_polynomial_domain_compare_monomial, 
		homogeneous_polynomial_domain_swap_monomial, 
		this);


	if (f_v) {
		cout << "homogeneous_polynomial_domain::rearrange_monomials_by_partition_type done" << endl;
	}
}

int homogeneous_polynomial_domain::index_of_monomial(int *v)
{
	sorting Sorting;

#if 0
	int i, j;
	
	for (i = 0; i < nb_monomials; i++) {
		for (j = 0; j < n; j++) {
			if (v[j] != Monomials[i * n + j]) {
				break;
				}
			}
		if (j == n) {
			return i;
			}
		}
#endif
	int idx;
	
	if (!Sorting.search_general(Monomials, nb_monomials, v, idx,
		homogeneous_polynomial_domain_compare_monomial_with, 
		this /* extra_data */, 0 /* verbose_level */)) {

		cout << "homogeneous_polynomial_domain::index_of_monomial "
				"Did not find the monomial v=";
		int_vec_print(cout, v, nb_variables);
		cout << endl;
		cout << "Monomials:" << endl;
		//int_matrix_print(Monomials, nb_monomials, n);
		int i;
		for (i = 0; i < nb_monomials; i++) {
			cout << setw(3) << i << " : ";
			int_vec_print(cout, Monomials + i * nb_variables, nb_variables);
			cout << endl;
		}
		cout << "homogeneous_polynomial_domain::index_of_monomial "
				"Did not find the monomial v=";
		int_vec_print(cout, v, nb_variables);
		cout << endl;
		Sorting.search_general(Monomials, nb_monomials, v, idx,
				homogeneous_polynomial_domain_compare_monomial_with,
				this /* extra_data */, 3);
		exit(1);
	}
	return idx;
}

void homogeneous_polynomial_domain::affine_evaluation_kernel(
		int *&Kernel, int &dim_kernel, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h, a, b, c, idx, f_kernel;
	int *mon;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::affine_evaluation_kernel" << endl;
	}
	dim_kernel = 0;
	mon = NEW_int(nb_variables);
	for (i = 0; i < nb_monomials; i++) {
		int_vec_copy(Monomials + i * nb_variables, mon, nb_variables);
		f_kernel = FALSE;
		for (j = 0; j < nb_variables - 1; j++) {
			a = mon[j];
			if (a >= q) {
				b = a % (q - 1);
				if (b == 0) {
					b += (q - 1);
				}
				c = a - b;
				mon[j] = b;
				mon[nb_variables - 1] += c;
				f_kernel = TRUE;
			}
		}
		if (f_kernel) {
			if (f_v) {
				cout << "homogeneous_polynomial_domain::affine_evaluation_kernel monomial ";
				int_vec_print(cout, Monomials + i * nb_variables, nb_variables);
				cout << " = ";
				int_vec_print(cout, mon, nb_variables);
				cout << endl;
			}
			dim_kernel++;
		}
	}
	if (f_v) {
		cout << "homogeneous_polynomial_domain::affine_evaluation_kernel dim_kernel = " << dim_kernel << endl;
	}
	Kernel = NEW_int(dim_kernel * 2);
	h = 0;
	for (i = 0; i < nb_monomials; i++) {
		int_vec_copy(Monomials + i * nb_variables, mon, nb_variables);
		f_kernel = FALSE;
		for (j = 0; j < nb_variables - 1; j++) {
			a = mon[j];
			if (a >= q) {
				b = a % (q - 1);
				if (b == 0) {
					b += (q - 1);
				}
				c = a - b;
				mon[j] = b;
				mon[nb_variables - 1] += c;
				f_kernel = TRUE;
			}
		}
		if (f_kernel) {
			if (f_v) {
				cout << "homogeneous_polynomial_domain::affine_evaluation_kernel monomial ";
				int_vec_print(cout, Monomials + i * nb_variables, nb_variables);
				cout << " = ";
				int_vec_print(cout, mon, nb_variables);
				cout << endl;
			}
			idx = index_of_monomial(mon);
			Kernel[h * 2 + 0] = i;
			Kernel[h * 2 + 1] = idx;
			h++;
		}
	}
	FREE_int(mon);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::affine_evaluation_kernel done" << endl;
	}
}

void homogeneous_polynomial_domain::print_monomial(ostream &ost, int i)
{
	int j, a, f_first = TRUE;
	
	for (j = 0; j < nb_variables; j++) {
		a = Monomials[i * nb_variables + j];
		if (a == 0) {
			continue;
		}
		if (!f_first) {
			ost << "*";
		}
		else {
			f_first = FALSE;
		}
		ost << symbols[j];
		if (a > 1) {
			ost << "^" << a;
		}
	}
}

#if 0
void homogeneous_polynomial_domain::print_monomial_latex(ostream &ost, int i)
{
	int j, a;

	for (j = 0; j < nb_variables; j++) {
		a = Monomials[i * nb_variables + j];
		if (a == 0) {
			continue;
		}
		ost << symbols_latex[j];
		if (a > 1) {
			ost << "^" << a;
		}
	}
}
#endif

void homogeneous_polynomial_domain::print_monomial(ostream &ost, int *mon)
{
	int j, a, f_first = TRUE;
	
	for (j = 0; j < nb_variables; j++) {
		a = mon[j];
		if (a == 0) {
			continue;
		}
		if (!f_first) {
			ost << "*";
		}
		else {
			f_first = FALSE;
		}
		ost << symbols[j];
		if (a > 1) {
			ost << "^" << a;
		}
	}
}

void homogeneous_polynomial_domain::print_monomial_latex(std::ostream &ost, int *mon)
{
	int j, a;

	for (j = 0; j < nb_variables; j++) {
		a = mon[j];
		if (a == 0) {
			continue;
		}
		ost << symbols_latex[j];
		if (a >= 10) {
			ost << "^{" << a << "}";
		}
		else if (a > 1) {
			ost << "^" << a;
		}
	}
}

void homogeneous_polynomial_domain::print_monomial_latex(std::ostream &ost, int i)
{
	int *mon;

	mon = Monomials + i * nb_variables;
	print_monomial_latex(ost, mon);
}

void homogeneous_polynomial_domain::print_monomial_latex(std::string &s, int *mon)
{
	int j, a;

	for (j = 0; j < nb_variables; j++) {
		a = mon[j];
		if (a == 0) {
			continue;
		}
		s.append(symbols_latex[j]);

		char str[1000];


		if (a >= 10) {
			sprintf(str, "^{%d}", a);
		}
		else if (a > 1) {
			sprintf(str, "^%d", a);
		}
		s.append(str);
	}
}

void homogeneous_polynomial_domain::print_monomial_latex(std::string &s, int i)
{
	int *mon;

	mon = Monomials + i * nb_variables;
	print_monomial_latex(s, mon);
}


#if 0
void homogeneous_polynomial_domain::print_monomial(char *str, int i)
{
	int j, a;
	
	str[0] = 0;
	for (j = 0; j < nb_variables; j++) {
		a = Monomials[i * nb_variables + j];
		if (a == 0) {
			continue;
		}
		strcat(str + strlen(str), symbols_latex[j].c_str());
		if (a >= 10) {
			sprintf(str + strlen(str), "^{%d}", a);
		}
		else if (a > 1) {
			sprintf(str + strlen(str), "^%d", a);
		}
	}
}
#endif

void homogeneous_polynomial_domain::print_monomial_str(stringstream &ost, int i)
{
	int j, a, f_first = TRUE;

	for (j = 0; j < nb_variables; j++) {
		a = Monomials[i * nb_variables + j];
		if (a == 0) {
			continue;
		}
		if (!f_first) {
			ost << "*";
		}
		else {
			f_first = FALSE;
		}
		ost << symbols[j];
		if (a > 1) {
			ost << "^" << a;
		}
	}
}


void homogeneous_polynomial_domain::print_monomial_latex_str(stringstream &ost, int i)
{
	int j, a;

	for (j = 0; j < nb_variables; j++) {
		a = Monomials[i * nb_variables + j];
		if (a == 0) {
			continue;
		}
		ost << symbols_latex[j];
		if (a > 1) {
			ost << "^" << a;
		}
	}
}

void homogeneous_polynomial_domain::print_equation(ostream &ost, int *coeffs)
{
	int i, c;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element(ost, c);
			//ost << c;
		}
		print_monomial(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_tex(ostream &ost, int *coeffs)
{
	int i, c;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element(ost, c);
			//ost << c;
		}
		print_monomial_latex(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_numerical(std::ostream &ost, int *coeffs)
{
	int i, c;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			//F->print_element(ost, c);
			ost << c;
		}
		print_monomial(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_lint(ostream &ost, long int *coeffs)
{
	int i, c;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element(ost, c);
			//ost << c;
		}
		print_monomial(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_lint_tex(ostream &ost, long int *coeffs)
{
	int i, c;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element(ost, c);
			//ost << c;
		}
		print_monomial_latex(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_str(stringstream &ost, int *coeffs)
{
	int i, c;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element_str(ost, c);
			//ost << c;
		}
		print_monomial_str(ost, i);
	}
}

void homogeneous_polynomial_domain::print_equation_with_line_breaks_tex(
	ostream &ost, int *coeffs, int nb_terms_per_line,
	const char *new_line_text)
{
	int i, c, cnt = 0;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}

		if ((cnt % nb_terms_per_line) == 0 && cnt) {
			ost << new_line_text;
		}


		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element(ost, c);
			//ost << c;
		}
		print_monomial_latex(ost, i);
		cnt++;
	}
}

void homogeneous_polynomial_domain::print_equation_with_line_breaks_tex_lint(
	ostream &ost, long int *coeffs, int nb_terms_per_line,
	const char *new_line_text)
{
	int i, c, cnt = 0;
	int f_first = TRUE;


	for (i = 0; i < nb_monomials; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}

		if ((cnt % nb_terms_per_line) == 0 && cnt) {
			ost << new_line_text;
		}


		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << " + ";
		}
		if (c > 1) {
			F->print_element(ost, c);
			//ost << c;
		}
		print_monomial_latex(ost, i);
		cnt++;
	}
}

void homogeneous_polynomial_domain::algebraic_set(int *Eqns, int nb_eqns,
		long int *Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk, a, i;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::algebraic_set "
				"P->N_points=" << P->N_points << endl;
	}
	nb_pts = 0;
	for (rk = 0; rk < P->N_points; rk++) {
		unrank_point(v, rk);
		for (i = 0; i < nb_eqns; i++) {
			a = evaluate_at_a_point(Eqns + i * nb_monomials, v);
			if (a) {
				break;
			}
		}
		if (i == nb_eqns) {
			Pts[nb_pts++] = rk;
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::algebraic_set "
				"done" << endl;
	}
}

void homogeneous_polynomial_domain::polynomial_function(int *coeff, int *f, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk, a;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::polynomial_function "
				"P->N_points=" << P->N_points << endl;
	}
	for (rk = 0; rk < P->N_points; rk++) {
		unrank_point(v, rk);
		a = evaluate_at_a_point(coeff, v);
		f[rk] = a;
	}
}
void homogeneous_polynomial_domain::enumerate_points(int *coeff,
		std::vector<long int> &Pts,
		//long int *Pts, int &nb_pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int rk;
	int a;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points" << endl;
	}
	if (f_vv) {
		cout << "homogeneous_polynomial_domain::enumerate_points P->N_points=" << P->N_points << endl;
#if 0
		print_equation_with_line_breaks_tex(cout,
				coeff, 8 /* nb_terms_per_line*/,
				"\\\\\n");
		cout << endl;
#endif
	}
	//nb_pts = 0;
	for (rk = 0; rk < P->N_points; rk++) {
		unrank_point(v, rk);
		a = evaluate_at_a_point(coeff, v);
		if (a == 0) {
			Pts.push_back(rk);
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::enumerate_points "
				"done" << endl;
	}
}

int homogeneous_polynomial_domain::evaluate_at_a_point_by_rank(
		int *coeff, int pt)
{
	int a;
	
	unrank_point(v, pt);
	a = evaluate_at_a_point(coeff, v);
	return a;
}

int homogeneous_polynomial_domain::evaluate_at_a_point(
		int *coeff, int *pt_vec)
{
	int i, a, b, c;
	
	a = 0;
	for (i = 0; i < nb_monomials; i++) {
		if (coeff[i] == 0) {
			continue;
		}
		b = F->evaluate_monomial(Monomials + i * nb_variables, pt_vec, nb_variables);
		c = F->mult(coeff[i], b);
		a = F->add(a, c);
	}
	return a;
}

void homogeneous_polynomial_domain::substitute_linear(
	int *coeff_in, int *coeff_out,
	int *Mtx_inv, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::substitute_linear" << endl;
		}

	substitute_semilinear(coeff_in, coeff_out, 
		FALSE /* f_semilinear */, 0 /* frob_power */,
		Mtx_inv, verbose_level);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::substitute_linear "
				"done" << endl;
		}
}

void homogeneous_polynomial_domain::substitute_semilinear(
	int *coeff_in, int *coeff_out,
	int f_semilinear, int frob_power, int *Mtx_inv,
	int verbose_level)
// applies frob_power field automorphisms and then performs substitution
{
	int f_v = (verbose_level >= 1);
	int a, b, c, i, j, idx;
	int *A;
	int *V;
	geometry_global Gg;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::substitute_semilinear" << endl;
	}


	if (f_semilinear) {
		F->frobenius_power_vec_to_vec(coeff_in, coeff4, nb_monomials, frob_power);
	}
	else {
		int_vec_copy(coeff_in, coeff4, nb_monomials);
	}


	int_vec_zero(coeff3, nb_monomials);
	for (i = 0; i < nb_monomials; i++) {
		c = coeff4[i];
		if (c == 0) {
			continue;
		}

#if 0
		cout << "homogeneous_polynomial_domain::substitute_semilinear monomial " << c << " * ";
		print_monomial(cout, i);
		cout << endl;
#endif
		
		V = Variables + i * degree;
			// a list of the indices of the variables
			// which appear in the monomial
			// (possibly with repeats)
			// Example: the monomial x_0^3 becomes 0,0,0

#if 0
		cout << "variables: ";
		int_vec_print(cout, V, degree);
		cout << endl;

		cout << "Mtx:" << endl;
		int_matrix_print(Mtx_inv, n, n);
#endif

		int_vec_zero(coeff2, nb_monomials);
		for (a = 0; a < nb_affine; a++) {
			if (Affine) {
				A = Affine + a * degree;
			}
			else {
				A = my_affine;
				Gg.AG_element_unrank(nb_variables /* q */, my_affine, 1, degree, a);
					// sequence of length degree over the alphabet  0,...,n-1.
			}
			for (j = 0; j < degree; j++) {
				//factors[j] = Mtx_inv[V[j] * n + A[j]];
				factors[j] = Mtx_inv[A[j] * nb_variables + V[j]];
			}

			b = F->product_n(factors, degree);
			if (Affine_to_monomial) {
				idx = Affine_to_monomial[a];
			}
			else {
				int_vec_zero(v, nb_variables);
				for (j = 0; j < degree; j++) {
					a = Affine[i * degree + j];
					v[a]++;
				}
				idx = index_of_monomial(v);
			}

#if 0
			cout << "affine " << a << " / " << nb_affine << " : ";
			int_vec_print(cout, A, 3);
			cout << " factors ";
			int_vec_print(cout, factors, 3);
			cout << " b=" << b << " idx=" << idx << endl;
#endif
			coeff2[idx] = F->add(coeff2[idx], b);
		}
		for (j = 0; j < nb_monomials; j++) {
			coeff2[j] = F->mult(coeff2[j], c);
		}

#if 0
		cout << "homogeneous_polynomial_domain::substitute_semilinear "
				"monomial " << c << " * ";
		print_monomial(cout, i);
		cout << " yields:" << endl;
		int_vec_print(cout, coeff2, nb_monomials);
		cout << endl;
#endif
		
		for (j = 0; j < nb_monomials; j++) {
			coeff3[j] = F->add(coeff2[j], coeff3[j]);
		}
	}
#if 0
	cout << "homogeneous_polynomial_domain::substitute_semilinear "
			"input:" << endl;
	int_vec_print(cout, coeff_in, nb_monomials);
	cout << endl;
	cout << "homogeneous_polynomial_domain::substitute_semilinear "
			"output:" << endl;
	int_vec_print(cout, coeff3, nb_monomials);
	cout << endl;
#endif





	int_vec_copy(coeff3, coeff_out, nb_monomials);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::substitute_semilinear "
				"done" << endl;
	}
}

void homogeneous_polynomial_domain::substitute_line(
	int *coeff_in, int *coeff_out,
	int *Pt1_coeff, int *Pt2_coeff,
	int verbose_level)
// coeff_in[nb_monomials], coeff_out[degree + 1]
{
	int f_v = (verbose_level >= 1);
	int rk, b, c, i, j, idx;
	int *A;
	int *V;
	int *Mtx;
	int my_nb_affine, wt;
	number_theory_domain NT;
	geometry_global Gg;


	if (f_v) {
		cout << "homogeneous_polynomial_domain::substitute_line" << endl;
	}

	my_nb_affine = NT.i_power_j(2, degree);

	Mtx = NEW_int(nb_variables * 2);

	for (i = 0; i < nb_variables; i++) {
		Mtx[i * 2 + 0] = Pt1_coeff[i];
		Mtx[i * 2 + 1] = Pt2_coeff[i];
	}

	int_vec_copy(coeff_in, coeff4, nb_monomials);


	int_vec_zero(coeff3, degree + 1);

	for (i = 0; i < nb_monomials; i++) {
		c = coeff4[i];
		if (c == 0) {
			continue;
		}

#if 0
		cout << "homogeneous_polynomial_domain::substitute_line monomial " << c << " * ";
		print_monomial(cout, i);
		cout << endl;
#endif

		V = Variables + i * degree;
			// a list of the indices of the variables
			// which appear in the monomial
			// (possibly with repeats)
			// Example: the monomial x_0^3 becomes 0,0,0

#if 0
		cout << "variables: ";
		int_vec_print(cout, V, degree);
		cout << endl;

		cout << "Mtx:" << endl;
		int_matrix_print(Mtx, n, 2);
#endif

		int_vec_zero(coeff2, degree + 1);
		for (rk = 0; rk < my_nb_affine; rk++) {

			A = my_affine;
			Gg.AG_element_unrank(2 /* q */, my_affine, 1, degree, rk);
					// sequence of length degree over the alphabet  0,1.

			wt = 0;
			for (j = 0; j < degree; j++) {
				if (my_affine[j]) {
					wt++;
				}
			}
			for (j = 0; j < degree; j++) {
				//factors[j] = Mtx_inv[V[j] * n + A[j]];
				factors[j] = Mtx[V[j] * 2 + A[j]];
			}

			b = F->product_n(factors, degree);

#if 0
			if (Affine_to_monomial) {
				idx = Affine_to_monomial[a];
			}
			else {
				int_vec_zero(v, n);
				for (j = 0; j < degree; j++) {
					a = Affine[i * degree + j];
					v[a]++;
				}
				idx = index_of_monomial(v);
			}
#else
			idx = wt;
#endif

#if 0
			cout << "affine " << a << " / " << nb_affine << " : ";
			int_vec_print(cout, A, 3);
			cout << " factors ";
			int_vec_print(cout, factors, 3);
			cout << " b=" << b << " idx=" << idx << endl;
#endif
			coeff2[idx] = F->add(coeff2[idx], b);
		}
		for (j = 0; j <= degree; j++) {
			coeff2[j] = F->mult(coeff2[j], c);
		}

#if 0
		cout << "homogeneous_polynomial_domain::substitute_line "
				"monomial " << c << " * ";
		print_monomial(cout, i);
		cout << " yields:" << endl;
		int_vec_print(cout, coeff2, nb_monomials);
		cout << endl;
#endif

		for (j = 0; j <= degree; j++) {
			coeff3[j] = F->add(coeff2[j], coeff3[j]);
		}
	}
#if 0
	cout << "homogeneous_polynomial_domain::substitute_line "
			"input:" << endl;
	int_vec_print(cout, coeff_in, nb_monomials);
	cout << endl;
	cout << "homogeneous_polynomial_domain::substitute_line "
			"output:" << endl;
	int_vec_print(cout, coeff3, nb_monomials);
	cout << endl;
#endif





	int_vec_copy(coeff3, coeff_out, degree + 1);

	FREE_int(Mtx);

	if (f_v) {
		cout << "homogeneous_polynomial_domain::substitute_line "
				"done" << endl;
	}
}

void homogeneous_polynomial_domain::multiply_mod(
	int *coeff1, int *coeff2, int *coeff3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::multiply_mod" << endl;
	}

	int_vec_zero(coeff3, nb_monomials);
	for (i = 0; i < nb_monomials; i++) {
		a = coeff1[i];
		if (a == 0) {
			continue;
		}
		if (f_v) {
			cout << "coeff1[" << i << "] = " << a << endl;
		}
		for (j = 0; j < nb_monomials; j++) {
			b = coeff2[j];
			if (b == 0) {
				continue;
			}
			if (f_v) {
				cout << "coeff2[" << j << "] = " << b << endl;
			}
			c = F->mult(a, b);
			idx = (i + j) % nb_monomials;
			coeff3[idx] = F->add(coeff3[idx], c);
			if (f_v) {
				cout << "coeff3[" << idx << "] += " << c << endl;
			}
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::multiply_mod done" << endl;
	}
}

void homogeneous_polynomial_domain::multiply_mod_negatively_wrapped(
	int *coeff1, int *coeff2, int *coeff3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::multiply_mod_negatively_wrapped" << endl;
	}

	int_vec_zero(coeff3, nb_monomials);
	for (i = 0; i < nb_monomials; i++) {
		a = coeff1[i];
		if (a == 0) {
			continue;
		}
		if (f_v) {
			cout << "coeff1[" << i << "] = " << a << endl;
		}
		for (j = 0; j < nb_monomials; j++) {
			b = coeff2[j];
			if (b == 0) {
				continue;
			}
			if (f_v) {
				cout << "coeff2[" << j << "] = " << b << endl;
			}
			c = F->mult(a, b);
			idx = i + j;
			if (idx < nb_monomials) {
				coeff3[idx] = F->add(coeff3[idx], c);
			}
			else {
				idx = idx % nb_monomials;
				coeff3[idx] = F->add(coeff3[idx], F->negate(c));
			}
			if (f_v) {
				cout << "coeff3[" << idx << "] += " << c << endl;
			}
		}
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::multiply_mod_negatively_wrapped done" << endl;
	}
}


int homogeneous_polynomial_domain::is_zero(int *coeff)
{
	int i;
	
	for (i = 0; i < nb_monomials; i++) {
		if (coeff[i]) {
			return FALSE;
		}
	}
	return TRUE;
}

void homogeneous_polynomial_domain::unrank_point(int *v, int rk)
{
	P->unrank_point(v, rk);
}

int homogeneous_polynomial_domain::rank_point(int *v)
{
	int rk;

	rk = P->rank_point(v);
	return rk;
}

void homogeneous_polynomial_domain::unrank_coeff_vector(int *v, long int rk)
{
	F->PG_element_unrank_modified_lint(v, 1, nb_monomials, rk);
}

long int homogeneous_polynomial_domain::rank_coeff_vector(int *v)
{
	long int rk;

	F->PG_element_rank_modified_lint(v, 1, nb_monomials, rk);
	return rk;
}

int homogeneous_polynomial_domain::test_weierstrass_form(int rk, 
	int &a1, int &a2, int &a3, int &a4, int &a6, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m_one;

	unrank_coeff_vector(coeff2, rk);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::test_weierstrass_form"
				<< endl;
	}
	if (nb_variables != 3) {
		cout << "homogeneous_polynomial_domain::test_weierstrass_form "
				"nb_variables != 3" << endl;
		exit(1);
	}
	if (degree != 3) {
		cout << "homogeneous_polynomial_domain::test_weierstrass_form "
				"degree != 3" << endl;
		exit(1);
	}
	if (coeff2[1] || coeff2[3] || coeff2[6]) {
		return FALSE;
	}
	F->PG_element_normalize_from_front(coeff2, 1, nb_monomials);
	if (coeff2[0] != 1) {
		return FALSE;
	}
	m_one = F->negate(1);
	if (coeff2[7] != m_one) {
		return FALSE;
	}
	a1 = F->negate(coeff2[4]);
	a2 = coeff2[2];
	a3 = F->negate(coeff2[8]);
	a4 = coeff2[5];
	a6 = coeff2[9];
	return TRUE;
}

void homogeneous_polynomial_domain::vanishing_ideal(long int *Pts,
		int nb_pts, int &r, int *Kernel, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int *System;

	if (f_v) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal" << endl;
	}
	System = NEW_int(MAX(nb_pts, nb_monomials) * nb_monomials);
	for (i = 0; i < nb_pts; i++) {
		unrank_point(v, Pts[i]);
		for (j = 0; j < nb_monomials; j++) {
			System[i * nb_monomials + j] =
					F->evaluate_monomial(Monomials + j * nb_variables, v, nb_variables);
		}
	}
	if (f_v && FALSE) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"The system:" << endl;
		int_matrix_print(System, nb_pts, nb_monomials);
	}

	if (f_v) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"before RREF_and_kernel" << endl;
	}
	r = F->RREF_and_kernel(nb_monomials,
			nb_pts, System, 0 /* verbose_level */);
	if (f_v) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"The system has rank " << r << endl;
	}
	if (TRUE) {
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"The system in RREF:" << endl;
		int_matrix_print(System, r, nb_monomials);
		cout << "homogeneous_polynomial_domain::vanishing_ideal "
				"The kernel:" << endl;
		int_matrix_print(System + r * nb_monomials,
				nb_monomials - r, nb_monomials);
	}
	int_vec_copy(System + r * nb_monomials, Kernel,
			(nb_monomials - r) * nb_monomials);
	FREE_int(System);
}

int homogeneous_polynomial_domain::compare_monomials(int *M1, int *M2)
{
	if (Monomial_ordering_type == t_PART) {
		return compare_monomials_PART(M1, M2);
	}
	if (Monomial_ordering_type == t_LEX) {
		return int_vec_compare(M1, M2, nb_variables) * -1;
	}
	else {
		cout << "homogeneous_polynomial_domain::compare_monomials "
				"monomial ordering unrecognized" << endl;
		exit(1);
	}
}

int homogeneous_polynomial_domain::compare_monomials_PART(int *M1, int *M2)
{
	int h, a;
	int ret = 0;
	
	int_vec_zero(type1, degree + 1);
	int_vec_zero(type2, degree + 1);

	for (h = 0; h < nb_variables; h++) {
		a = M1[h];
		type1[a]++;
	}
	for (h = 0; h < nb_variables; h++) {
		a = M2[h];
		type2[a]++;
	}
	for (h = degree; h >= 0; h--) {
		if (type2[h] > type1[h]) {
			//cout << "type2[h] > type1[h] h=" << h << ", needs swap" << endl;
			ret = 1;
			goto the_end;
		}
		if (type2[h] < type1[h]) {
			ret = -1;
			goto the_end;
		}
	}
	
	for (a = degree; a >= 1; a--) {
		for (h = 0; h < nb_variables; h++) {
			if ((M1[h] != a) && (M2[h] != a)) {
				continue;
			}
			if (M1[h] > M2[h]) {
				ret = -1;
				goto the_end;
			}
			if (M1[h] < M2[h]) {
				//cout << "M1[h] < M2[h] h=" << h << ", needs swap" << endl;
				ret = 1;
				goto the_end;
			}
		}
	}

the_end:
	return ret;

}


void homogeneous_polynomial_domain::print_monomial_ordering(ostream &ost)
{
	int h;
	
	//ost << "The ordering of monomials is:\\\\" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "h &  \\mbox{monomial} & \\mbox{vector} \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (h = 0; h < nb_monomials; h++) {
		ost << h << " & ";
		print_monomial_latex(ost, h);
		ost << " & ";
		int_vec_print(ost, Monomials + h * nb_variables, nb_variables);
		ost << "\\\\" << endl; 
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
}




int homogeneous_polynomial_domain_compare_monomial_with(
		void *data, int i, void *data2, void *extra_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) extra_data;
	int *Data;
	int ret, nb_variables;

	Data = (int *) data;
	nb_variables = HPD->nb_variables;
	ret = HPD->compare_monomials(Data + i * nb_variables, (int *) data2);
	return ret;
}

int homogeneous_polynomial_domain_compare_monomial(
		void *data, int i, int j, void *extra_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) extra_data;
	int *Data;
	int ret, nb_variables;

	Data = (int *) data;
	nb_variables = HPD->nb_variables;
	ret = HPD->compare_monomials(Data + i * nb_variables, Data + j * nb_variables);
	return ret;
}

void homogeneous_polynomial_domain_swap_monomial(
		void *data, int i, int j, void *extra_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) extra_data;
	int *Data;
	int h, a, nb_variables;

	Data = (int *) data;
	nb_variables = HPD->nb_variables;

	for (h = 0; h < nb_variables; h++) {
		a = Data[i * nb_variables + h];
		Data[i * nb_variables + h] = Data[j * nb_variables + h];
		Data[j * nb_variables + h] = a;
	}
	
}



void HPD_callback_print_function(
		stringstream &ost, void *data, void *callback_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) callback_data;

	int *coeff;
	int *i_data = (int *) data;

	coeff = NEW_int(HPD->get_nb_monomials());
	HPD->unrank_coeff_vector(coeff, i_data[0]);
	//int_vec_print(cout, coeff, HPD->nb_monomials);
	//cout << " = ";
	HPD->print_equation_str(ost, coeff);
	//ost << endl;
	FREE_int(coeff);
}

void HPD_callback_print_function2(
		stringstream &ost, void *data, void *callback_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) callback_data;

	int *coeff;
	int *i_data = (int *) data;
	//long int *Pts;
	//int nb_pts;
	vector<long int> Points;

	//Pts = NEW_lint(HPD->get_P()->N_points);
	coeff = NEW_int(HPD->get_nb_monomials());
	HPD->unrank_coeff_vector(coeff, i_data[0]);
	HPD->enumerate_points(coeff, Points,  0 /*verbose_level*/);
	ost << Points.size();
	//int_vec_print(cout, coeff, HPD->nb_monomials);
	//cout << " = ";
	//HPD->print_equation_str(ost, coeff);
	//ost << endl;
	FREE_int(coeff);
	//FREE_lint(Pts);
}




}}


