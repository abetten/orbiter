// bitmatrix.C
//
// Anton Betten
// 20.11.1999
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"


#define BITMATRIX_COPY_VERBOSE

/* ANCHOR bitmatrix */

bitmatrix::bitmatrix()
{
	k = BITMATRIX;
	self.bitmatrix_rep = NULL;
}

bitmatrix::bitmatrix(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "bitmatrix::copy constructor for object: " << const_cast<discreta_base &>(x) << "\n";
	clearself();
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

bitmatrix& bitmatrix::operator = (const discreta_base &x)
	// copy assignment
{
	cout << "bitmatrix::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void bitmatrix::settype_bitmatrix()
{
	OBJECTSELF s;
	
	s = self;
	new(this) bitmatrix;
	k = BITMATRIX;
	self = s;
}

bitmatrix::~bitmatrix()
{
	freeself_bitmatrix();
}

void bitmatrix::freeself_bitmatrix()
{
	if (self.bitmatrix_rep == NULL)
		return;
	delete self.bitmatrix_rep;
	self.bitmatrix_rep = NULL;
}

kind bitmatrix::s_virtual_kind()
{
	return BITMATRIX;
}

void bitmatrix::copyobject_to(discreta_base &x)
{
	INT i, m, n, N, l;
	
#ifdef BITMATRIX_COPY_VERBOSE
	cout << "in bitmatrix::copyobject_to()\n";
#endif
	x.freeself();
	bitmatrix & xx = x.change_to_bitmatrix();
	if (self.bitmatrix_rep == NULL)
		return;
	m = s_m();
	n = s_n();
	N = s_N();
	l = sizeof(BITMATRIX_REPRESENTATION) + m * N * 4;
	BITMATRIX_REPRESENTATION * rep = (BITMATRIX_REPRESENTATION *) new char[l];
	xx.self.bitmatrix_rep = rep;
	rep->m = m;
	rep->n = n;
	rep->N = N;
	l = m * N;
	for (i = 0; i < l; i++) {
		rep->p[i] = self.bitmatrix_rep->p[i];
		}
}

ostream& bitmatrix::print(ostream& ost)
{
	INT i, j, x, m, n;
	
	m = s_m();
	n = s_n();
#ifdef PRINT_WITH_TYPE
	ost << "(BITMATRIX of size " << m << " x " << n << ", \n";
#endif
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			x = s_ij(i, j);
			if (x)
				ost << "1";
			else
				ost << "0";
			}
		ost << endl;
		}
#ifdef PRINT_WITH_TYPE
	ost << ")";
#endif
	ost << "\n";
	return ost;
}

bitmatrix& bitmatrix::m_mn(INT m, INT n)
{
	INT N, size, n1;
	
	freeself();
	N = n >> 5; // 4 chars = 32 bits = 2^5
	n1 = N << 5;
	if (n > n1)
		N++;
	size = sizeof(BITMATRIX_REPRESENTATION) + m * N * 4;
	cout << "allocating BITMATRIX of size " << size << endl;
	self.bitmatrix_rep = (BITMATRIX_REPRESENTATION *) new char[size];
	self.bitmatrix_rep->m = m;
	self.bitmatrix_rep->n = n;
	self.bitmatrix_rep->N = N;
	return *this;
}

bitmatrix& bitmatrix::m_mn_n(INT m, INT n)
{
	INT i, l;
	
	m_mn(m, n);
	l = self.bitmatrix_rep->m * self.bitmatrix_rep->N;
	for (i = 0; i < l; i++) {
		self.bitmatrix_rep->p[i] = (UINT4) 0;
		}
	return *this;
}

INT bitmatrix::s_m()
{
	if (self.bitmatrix_rep == NULL) {
		cout << "bitmatrix::s_m() not allocated\n";
		exit(1);
		}
	return self.bitmatrix_rep->m;
}

INT bitmatrix::s_n()
{
	if (self.bitmatrix_rep == NULL) {
		cout << "bitmatrix::s_n() not allocated\n";
		exit(1);
		}
	return self.bitmatrix_rep->n;
}

INT bitmatrix::s_N()
{
	if (self.bitmatrix_rep == NULL) {
		cout << "bitmatrix::s_N() not allocated\n";
		exit(1);
		}
	return self.bitmatrix_rep->N;
}

UINT4& bitmatrix::s_i(INT i)
{
	if (self.bitmatrix_rep == NULL) {
		cout << "bitmatrix::s_i() not allocated\n";
		exit(1);
		}
	return self.bitmatrix_rep->p[i];
}

INT bitmatrix::s_ij(INT i, INT j)
{
	INT m, n, N, jj, bit;
	UINT4 mask;
	
	if (self.bitmatrix_rep == NULL) {
		cout << "bitmatrix::s_ij() bitmatrix_rep == NULL\n";
		exit(1);
		}
	m = s_m();
	n = s_n();
	N = s_N();
	if ( i < 0 || i >= m ) {
		cout << "bitmatrix::s_ij() addressing error, i = " << i << ", m = " << m << "\n";
		exit(1);		
		}
	if ( j < 0 || j >= n ) {
		cout << "bitmatrix::s_ij() addressing error, j = " << j << ", n = " << n << "\n";
		exit(1);		
		}
	jj = j >> 5;
	bit = j & 31;
	mask = ((UINT4) 1) << bit;
	UINT4 &x = s_i(i * N + jj);
	if (x & mask)
		return 1;
	else
		return 0;
}

void bitmatrix::m_iji(INT i, INT j, INT a)
{
	INT m, n, N, jj, bit;
	UINT4 mask;
	
	if (self.bitmatrix_rep == NULL) {
		cout << "bitmatrix::m_iji() bitmatrix_rep == NULL\n";
		exit(1);
		}
	m = s_m();
	n = s_n();
	N = s_N();
	if ( i < 0 || i >= m ) {
		cout << "bitmatrix::m_iji() addressing error, i = " << i << ", m = " << m << "\n";
		exit(1);		
		}
	if ( j < 0 || j >= n ) {
		cout << "bitmatrix::m_iji() addressing error, j = " << j << ", n = " << n << "\n";
		exit(1);		
		}
	jj = j >> 5;
	bit = j & 31;
	mask = ((UINT4) 1) << bit;
	UINT4 &x = s_i(i * N + jj);
	if (a == 0) {
		UINT4 not_mask = ~mask;
		x &= not_mask;
		}
	else {
		x |= mask;
		}
}

void bitmatrix::mult_to(discreta_base &x, discreta_base &y)
// multiply two bitmatrices over GF(2)
{
	if (x.s_kind() == BITMATRIX) {
		bitmatrix& px = x.as_bitmatrix();
		bitmatrix_mult_to(px, y);
		}
	else {
		cout << "bitmatrix::mult_to() object x is of bad type\n";
		exit(1);
		}
}

void bitmatrix::bitmatrix_mult_to(bitmatrix &x, discreta_base &y)
{
	bitmatrix py;
	INT i, j, k, m, n, l;
	
	if (s_kind() != BITMATRIX) {
		cout << "bitmatrix::bitmatrix_mult_to() this is not a bitmatrix\n";
		exit(1);
		}
	if (x.s_kind() != BITMATRIX) {
		cout << "bitmatrix::bitmatrix_mult_to() x is not a bitmatrix\n";
		exit(1);
		}
	m = s_m();
	l = s_n();
	if (l != x.s_m()) {
		cout << "bitmatrix::bitmatrix_mult_to() l != x.s_m(), cannot multiply\n";
		exit(1);
		}
	n = x.s_n();
	
	py.m_mn_n(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			INT a, b;
            a = 0;
			for (k = 0; k < l; k++) {
                b = s_ij(i, k) * x.s_ij(k, j);
                a += b;
                a %= 2;
				}
            if (a) {
				py.m_iji(i, j, 1);
                }
			}
		}
	y.swap(py);
}

INT bitmatrix::gauss(INT f_complete, Vector& base_cols, INT f_v)
// returns the rank
{
	INT rank, i, j, k, jj, m, n, N;
	
	base_cols.m_l(0);
	m = s_m();
	n = s_n();
	N = s_N();
	
	i = 0;
	for (j = 0; j < n; j++) {
	
		/* search for pivot element: */
		for (k = i; k < m; k++) {
			if (s_ij(k, j)) {
				// pivot element found: 
				if (k != i) {
					UINT4 *p, *q;
					INT j1;
					
					j1 = j >> 5;
					p = &s_i(i * N);
					q = &s_i(k * N);
					for (jj = j1; jj < N; jj++) {
						UINT4_swap(p[jj], q[jj]);
						}
					}
				break;
				} // if != 0 
			} // next k
		
		if (k == m) // no pivot found 
			continue; // increase j, leave i constant
		
		if (f_v) {
			cout << "row " << i << " pivot in row " << k << " colum " << j << endl;
			// cout << *this << endl;
			}
		
		base_cols.append_integer(j);

		/* do the gaussian elimination: */
		for (k = i + 1; k < m; k++) {
			if (s_ij(k, j) == 0)
				continue;
			INT j1;
			UINT4 *p, *q;
			
			j1 = j >> 5;
			p = &s_i(i * N);
			q = &s_i(k * N);
			for (jj = j1; jj < N; jj++) {
				q[jj] ^= p[jj];
				}
			// cout << "elimination in row " << k << "done\n";
			// cout << *this << endl;
			}
		i++;
		} // next j 
	rank = i;

	if (f_complete) {
		for (i = rank - 1; i >= 0; i--) {
			j = base_cols.s_ii(i);
			// do the gaussian elimination in the upper part: 
			for (k = i - 1; k >= 0; k--) {
				if (s_ij(k, j) == 0)
					continue;
				INT j1;
				UINT4 *p, *q;
				
				j1 = j >> 5;
				p = &s_i(i * N);
				q = &s_i(k * N);
				for (jj = j1; jj < N; jj++) {
					q[jj] ^= p[jj];
					}
				}
			}
		}
	return rank;
}

INT bitmatrix::get_kernel(Vector& base_cols, bitmatrix& kernel)
{
	INT r, m, n, k, i, j, ii, iii, a, b, x;
	Vector kcol;
	
	m = s_m();
	n = s_n();
	r = base_cols.s_l();
	k = n - r;
	kernel.m_mn_n(n, k);
	kcol.m_l(k);
	ii = 0;
	j = 0;
	if (j < r)
		b = base_cols.s_ii(j);
	else
		b = -1;
	for (i = 0; i < n; i++) {
		if (i == b) {
			j++;
			if (j < r)
				b = base_cols.s_ii(j);
			else
				b = -1;
			}
		else {
			kcol.m_ii(ii, i);
			ii++;
			}
		}
	if (ii != k) {
		cout << "bitmatrix::get_kernel() ii != k" << endl;
		exit(1);
		}
	// cout << "kcol = " << kcol << endl;
	ii = 0;
	j = 0;
	if (j < r)
		b = base_cols.s_ii(j);
	else
		b = -1;
	for (i = 0; i < n; i++) {
		if (i == b) {
			for (iii = 0; iii < k; iii++) {
				a = kcol.s_ii(iii);
				x = s_ij(j, a);
				if (x)
					kernel.m_iji(i, iii, 1);
				}
			j++;
			if (j < r)
				b = base_cols.s_ii(j);
			else
				b = -1;
			}
		else {
			for (iii = 0; iii < k; iii++) {
				if (iii == ii) {
					kernel.m_iji(i, iii, 1);
					// kernel.s_ij(i, iii).m_one();
					}
				else {
					// kernel.s_ij(i, iii).zero();
					}
				}
			ii++;
			}
		}
	return TRUE;
}

void bitmatrix::write_mem(memory & M, INT debug_depth)
{
	INT i, m, n, N, a;

	m = self.bitmatrix_rep->m;
	n = self.bitmatrix_rep->n;
	N = self.bitmatrix_rep->N;
	M.write_int(m);
	M.write_int(n);
	M.write_int(N);

	for (i = 0; i < m * N; i++) {
		a = self.bitmatrix_rep->p[i];
		M.write_int(a);
		}
}

void bitmatrix::read_mem(memory & M, INT debug_depth)
{
	INT i, m, n, N, a;
	
	M.read_int(&m);
	M.read_int(&n);
	M.read_int(&N);
	
	m_mn(m, n);
	if (N != self.bitmatrix_rep->N) {
		cout << "bitmatrix::read_mem N != self.bitmatrix_rep->N" << endl;
		exit(1);
		}
	for (i = 0; i < m; i++) {
		M.read_int(&a);
		self.bitmatrix_rep->p[i] = (UINT4) a;
		}
}

INT bitmatrix::csf()
{
	INT size = 0;
	INT m, n, N;
	
	m = self.bitmatrix_rep->m;
	n = self.bitmatrix_rep->n;
	N = self.bitmatrix_rep->N;
	size += 12; /* m, n, N */
	size += m * N * 4;
	return size;
}



