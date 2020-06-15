/*
 * ginac_linear_algebra.cpp
 *
 *  Created on: Jan 3, 2020
 *      Author: betten
 */



// local functions copied from ginac:

ex my_lsolve(ostream &ost, const ex &eqns, const ex &symbols);
matrix solve(matrix *M, const matrix & vars,
                     const matrix & rhs);
void row_operation_add(matrix *M, int r1, int r2, ex factor);
// Row[r2] += Row[r1] * factor
void row_operation_multiply(matrix *M, int r1, ex factor);
// Row[r1] = Row[r1] * factor
void right_normalize_row(matrix *M, int r);
matrix right_nullspace(matrix *M);
int pivot(matrix *M, unsigned ro, unsigned co, bool symbolic);


// ginac stuff:
class my_symbolset {
	exset s;
	void insert_symbols(const ex &e)
	{
		if (is_a<symbol>(e)) {
			s.insert(e);
		}
		else {
			for (const ex &sube : e) {
				insert_symbols(sube);
			}
		}
	}
public:
	explicit my_symbolset(const ex &e)
	{
		insert_symbols(e);
	}
	bool has(const ex &e) const
	{
		return s.find(e) != s.end();
	}
};



ex my_lsolve(ostream &ost, const ex &eqns, const ex &symbols)
{
	// solve a system of linear equations
	if (eqns.info(info_flags::relation_equal)) {
		if (!symbols.info(info_flags::symbol))
			throw(std::invalid_argument("lsolve(): 2nd argument must be a symbol"));
		const ex sol = lsolve(lst{eqns}, lst{symbols});

		GINAC_ASSERT(sol.nops()==1);
		GINAC_ASSERT(is_exactly_a<relational>(sol.op(0)));

		return sol.op(0).op(1); // return rhs of first solution
	}

	// syntax checks
	if (!(eqns.info(info_flags::list) || eqns.info(info_flags::exprseq))) {
		throw(std::invalid_argument("lsolve(): 1st argument must be a list, a sequence, or an equation"));
	}
	for (size_t i=0; i<eqns.nops(); i++) {
		if (!eqns.op(i).info(info_flags::relation_equal)) {
			throw(std::invalid_argument("lsolve(): 1st argument must be a list of equations"));
		}
	}
	if (!(symbols.info(info_flags::list) || symbols.info(info_flags::exprseq))) {
		throw(std::invalid_argument("lsolve(): 2nd argument must be a list, a sequence, or a symbol"));
	}
	for (size_t i=0; i<symbols.nops(); i++) {
		if (!symbols.op(i).info(info_flags::symbol)) {
			throw(std::invalid_argument("lsolve(): 2nd argument must be a list or a sequence of symbols"));
		}
	}

	// build matrix from equation system
	matrix sys(eqns.nops(),symbols.nops());
	matrix rhs(eqns.nops(),1);
	matrix vars(symbols.nops(),1);

	for (size_t r=0; r<eqns.nops(); r++) {
		const ex eq = eqns.op(r).op(0)-eqns.op(r).op(1); // lhs-rhs==0
		const my_symbolset syms(eq);
		ex linpart = eq;
		for (size_t c=0; c<symbols.nops(); c++) {
			if (!syms.has(symbols.op(c)))
				continue;
			const ex co = eq.coeff(ex_to<symbol>(symbols.op(c)),1);
			linpart -= co*symbols.op(c);
			sys(r,c) = co;
		}
		linpart = linpart.expand();
		rhs(r,0) = -linpart;
	}

#if 0
	ost << "\\begin{sidewaysfigure}" << endl;
	ost << "$$" << endl << sys << endl << "$$" << endl;
	ost << "\\end{sidewaysfigure}" << endl;
	ost << "\\begin{figure}" << endl;
	ost << "$$" << endl << rhs << endl << "$$" << endl;
	ost << "\\end{figure}" << endl;
#endif


	// test if system is linear and fill vars matrix
	const my_symbolset sys_syms(sys);
	const my_symbolset rhs_syms(rhs);
	for (size_t i=0; i<symbols.nops(); i++) {
		vars(i,0) = symbols.op(i);
		if (sys_syms.has(symbols.op(i)))
			throw(std::logic_error("lsolve: system is not linear"));
		if (rhs_syms.has(symbols.op(i)))
			throw(std::logic_error("lsolve: system is not linear"));
	}

	matrix solution;
	try {
		solution = solve(&sys, vars, rhs);
	} catch (const std::runtime_error & e) {
		// Probably singular matrix or otherwise overdetermined system:
		// It is consistent to return an empty list
		return lst{};
	}
	GINAC_ASSERT(solution.cols()==1);
	GINAC_ASSERT(solution.rows()==symbols.nops());

#if 0
	// return list of equations of the form lst{var1==sol1,var2==sol2,...}
	lst sollist;
	for (size_t i=0; i<symbols.nops(); i++)
		sollist.append(symbols.op(i)==solution(i,0));

	return sollist;
#endif
	return solution;
}

matrix solve(matrix *M, const matrix & vars,
                     const matrix & rhs)
{
	const unsigned m = M->rows();
	const unsigned n = M->cols();
	const unsigned p = rhs.cols();

	cout << "solve M=" << endl;
	//cout << *M << endl;

	cout << "solve rhs=" << endl;
	//cout << rhs << endl;

	// syntax checks
	if ((rhs.rows() != m) || (vars.rows() != n) || (vars.cols() != p))
		throw (std::logic_error("matrix::solve(): incompatible matrices"));
	for (unsigned ro=0; ro<n; ++ro)
		for (unsigned co=0; co<p; ++co)
			if (!vars(ro,co).info(info_flags::symbol))
				throw (std::invalid_argument("matrix::solve(): 1st argument must be matrix of symbols"));

#if 0
	// build the augmented matrix of *this with rhs attached to the right
	matrix aug(m,n+p);
	for (unsigned r=0; r<m; ++r) {
		for (unsigned c=0; c<n; ++c)
			aug.m[r*(n+p)+c] = M->m[r*n+c];
		for (unsigned c=0; c<p; ++c)
			aug.m[r*(n+p)+c+n] = rhs.m[r*p+c];
	}
#endif

	//cout << "solve before gauss_elimination, M=" << endl;
	//cout << *M << endl;


	// Eliminate the augmented matrix:
	//auto colid = aug.echelon_form(algo, n);

	matrix K;

	K = right_nullspace(M);

	//cout << "solve after gauss_elimination, M=" << endl;
	//cout << *M << endl;
	cout << "solve after gauss_elimination, K=" << endl;
	cout << K << endl;

	return K;
#if 0
	// assemble the solution matrix:
	matrix sol(n,p);
	for (unsigned co=0; co<p; ++co) {
		unsigned last_assigned_sol = n+1;
		for (int r=m-1; r>=0; --r) {
			unsigned fnz = 1;    // first non-zero in row
			while ((fnz<=n) && (aug.m[r*(n+p)+(fnz-1)].normal().is_zero()))
				++fnz;
			if (fnz>n) {
				// row consists only of zeros, corresponding rhs must be 0, too
				if (!aug.m[r*(n+p)+n+co].normal().is_zero()) {
					throw (std::runtime_error("matrix::solve(): inconsistent linear system"));
				}
			}
			else {
				// assign solutions for vars between fnz+1 and
				// last_assigned_sol-1: free parameters
				for (unsigned c=fnz; c<last_assigned_sol-1; ++c)
					sol(colid[c],co) = vars.m[colid[c]*p+co];
				ex e = aug.m[r*(n+p)+n+co];
				for (unsigned c=fnz; c<n; ++c)
					e -= aug.m[r*(n+p)+c]*sol.m[colid[c]*p+co];
				sol(colid[fnz-1],co) = (e/(aug.m[r*(n+p)+fnz-1])).normal();
				last_assigned_sol = fnz;
			}
		}
		// assign solutions for vars between 1 and
		// last_assigned_sol-1: free parameters
		for (unsigned ro=0; ro<last_assigned_sol-1; ++ro)
			sol(colid[ro],co) = vars(colid[ro],co);
	}

	return sol;
#endif
}

void row_operation_add(matrix *M, int r1, int r2, ex factor)
// Row[r2] += Row[r1] * factor
{
	//const unsigned m = M->rows();
	const unsigned n = M->cols();
	for (unsigned c=0; c<n; ++c) {
		M->m[r2*n+c] += factor * M->m[r1*n+c];
	}

}

void row_operation_multiply(matrix *M, int r1, ex factor)
// Row[r1] = Row[r1] * factor
{
	//const unsigned m = M->rows();
	const unsigned n = M->cols();
	for (unsigned c=0; c<n; ++c) {
		M->m[r1*n+c] *= factor;
	}

}

void right_normalize_row(matrix *M, int r)
{
	//const unsigned m = M->rows();
	const unsigned n = M->cols();
	int j;
	for (j = n - 1; j >= 0; j--) {
		if (!M->m[r*n+j].expand().is_zero()) {
			break;
		}
	}
	if (j == -1) {
		cout << "right_normalize_row zero row" << endl;
		exit(1);
	}
	ex factor = 1 / M->m[r*n+j];
	for (unsigned c = 0; c < n; ++c) {
		M->m[r*n+c] *= factor;
		//M->m[r*n+c].expand(1);
	}

}

matrix right_nullspace(matrix *M)
{
	//ensure_if_modifiable();
	const unsigned m = M->rows();
	const unsigned n = M->cols();
	GINAC_ASSERT(!det || n==m);
	int sign = 1;
	int *base_cols;
	int *kernel_cols;

	base_cols = NEW_int(n);
	kernel_cols = NEW_int(n);

	unsigned r0 = 0;
	for (unsigned c0=0; c0<n; ++c0) {


		cout << "right_nullspace, c0=" << c0 << " / " << n << " r0=" << r0 << endl;
		//cout << "right_nullspace, M=" << endl;
		//cout << *M << endl;



		int indx = pivot(M, r0, c0, true);

		cout << "right_nullspace, after pivot, c0=" << c0 << " r0=" << r0 << " indx=" << indx << endl;
		//cout << "right_nullspace, M=" << endl;
		//cout << *M << endl;


		if (indx == -1) {
			cout << "right_nullspace, did not find pivot element" << endl;
			sign = 0;
		}
		if (indx>=0) {

			base_cols[r0] = c0;

			if (indx > 0)
				sign = -sign;
			{
				ex piv = 1 / M->m[r0*n+c0];
				cout << "piv=" << piv << endl;
				for (unsigned c=c0/*+1*/; c<n; ++c) {
					M->m[r0*n+c] *= piv;

				}
			}
			for (unsigned r2=r0+1; r2<m; ++r2) {
				if (!M->m[r2*n+c0].is_zero()) {
					cout << "right_nullspace, cleaning c0=" << c0 << " r0=" << r0 << " r2=" << r2 << endl;
					// yes, there is something to do in this row
					ex piv = M->m[r2*n+c0];
					cout << "piv=" << piv << endl;
					for (unsigned c=c0/*+1*/; c<n; ++c) {
						M->m[r2*n+c] -= piv * M->m[r0*n+c];
					}
				}

			}
			cout << "right_nullspace after cleaning column " << c0 << ", M=" << endl;
			//cout << *M << endl;
			++r0;
		}
	} // next c0

	cout << "right_nullspace, rank = " << r0 << endl;
	cout << "right_nullspace, base_cols=";
	int_vec_print(cout, base_cols, r0);
	cout << endl;
	//cout << "right_nullspace, M=" << endl;
	//cout << *M << endl;
	cout << "right_nullspace, second part" << endl;

	int i, j, c;
	for (i = r0 - 1; i >= 0; i--) {
		c = base_cols[i];
		cout << "i=" << i << " c=" << c << " piv=" << M->m[i*n+c] << endl;
		for (int r2 = i - 1; r2 >= 0; r2--) {
			ex entry = M->m[r2*n+c];
			cout << "cleaning row " << r2 << " entry=" << entry << endl;
			if (!M->m[r2*n+c].is_zero()) {
				for (j = c; j < n; j++) {
					cout << "j=" << j << endl;
					M->m[r2*n+j] -= entry * M->m[i*n+j];
					cout << "done" << endl;
				}
			}
		}
		//cout << "right_nullspace after cleaning column " << c << ", M=" << endl;
		//cout << *M << endl;

	}
	cout << "right_nullspace finished, M=" << endl;
	//cout << *M << endl;

	int_vec_complement(base_cols, kernel_cols, n, r0 /* nb_base_cols */);

	int r, k, ii, b, iii, a;
	int kernel_n, kernel_m;
	r = r0;
	k = n - r;

	cout << "right_nullspace, kernel_cols=";
	int_vec_print(cout, kernel_cols, k);
	cout << endl;

	kernel_m = n;
	kernel_n = k;

	matrix K(kernel_m, kernel_n);


	ii = 0;
	j = 0;
	if (j < r) {
		b = base_cols[j];
		}
	else {
		b = -1;
		}
	for (i = 0; i < n; i++) {
		cout << "i=" << i << " / " << n << endl;
		if (i == b) {
			for (iii = 0; iii < k; iii++) {
				a = kernel_cols[iii];
				K(i, iii) = M->m[j * n + a];
				}
			j++;
			if (j < r) {
				b = base_cols[j];
				}
			else {
				b = -1;
				}
			}
		else {
			for (iii = 0; iii < k; iii++) {
				if (iii == ii) {
					K(i, iii) = -1;
					}
				else {
					K(i, iii) = 0;
					}
				}
			ii++;
			}
		}

	cout << "right_nullspace finished, K=" << endl;
	cout << K << endl;

	FREE_int(base_cols);
	FREE_int(kernel_cols);
#if 0
	// clear remaining rows
	for (unsigned r=r0+1; r<m; ++r) {
		for (unsigned c=0; c<n; ++c)
			M->m[r*n+c] = _ex0;
	}
#endif

	//return sign;
	return K;
}

int pivot(matrix *M, unsigned ro, unsigned co, bool symbolic)
{
	unsigned k = ro;
	if (symbolic) {
		// search first non-zero element in column co beginning at row ro
		while ((k<M->row) && (M->m[k*M->col+co].expand().is_zero()))
			++k;
	}
	else {
		// search largest element in column co beginning at row ro
		GINAC_ASSERT(is_exactly_a<numeric>(this->m[k*col+co]));
		unsigned kmax = k+1;
		numeric mmax = abs(ex_to<numeric>(M->m[kmax*M->col+co]));
		while (kmax<M->row) {
			GINAC_ASSERT(is_exactly_a<numeric>(M->m[kmax*col+co]));
			numeric tmp = ex_to<numeric>(M->m[kmax*M->col+co]);
			if (abs(tmp) > mmax) {
				mmax = tmp;
				k = kmax;
			}
			++kmax;
		}
		if (!mmax.is_zero())
			k = kmax;
	}
	if (k==M->row)
		// all elements in column co below row ro vanish
		return -1;
	if (k==ro)
		// matrix needs no pivoting
		return 0;
	// matrix needs pivoting, so swap rows k and ro
	//ensure_if_modifiable();
	for (unsigned c=0; c<M->col; ++c)
		M->m[k*M->col+c].swap(M->m[ro*M->col+c]);

	return k;
}
