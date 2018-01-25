// finitefield_representations.C
//
// Anton Betten
//
// started:  October 23, 2002
// pulled out of finitefield:  July 5 2007




#include "galois.h"

void finite_field::representing_matrix8_R(INT *A, INT q, INT a, INT b, INT c, INT d)
{
	INT i;	
	
	for (i = 0; i < 8 * 8; i++) {
		A[i] = 0;
		}
	A[0 * 8 + 0] = m_term(q, d, d, d);
	A[0 * 8 + 1] = m_term(q, c, c, c);
	A[0 * 8 + 2] = m_term(q, d, d, c);
	A[0 * 8 + 3] = m_term(q, d, c, d);
	A[0 * 8 + 4] = m_term(q, c, d, d);
	A[0 * 8 + 5] = m_term(q, d, c, c);
	A[0 * 8 + 6] = m_term(q, c, c, d);
	A[0 * 8 + 7] = m_term(q, c, d, c);
	
	A[1 * 8 + 0] = m_term(q, b, b, b);
	A[1 * 8 + 1] = m_term(q, a, a, a);
	A[1 * 8 + 2] = m_term(q, b, b, a);
	A[1 * 8 + 3] = m_term(q, b, a, b);
	A[1 * 8 + 4] = m_term(q, a, b, b);
	A[1 * 8 + 5] = m_term(q, b, a, a);
	A[1 * 8 + 6] = m_term(q, a, a, b);
	A[1 * 8 + 7] = m_term(q, a, b, a);
	
	A[2 * 8 + 0] = m_term(q, d, d, b);
	A[2 * 8 + 1] = m_term(q, c, c, a);
	A[2 * 8 + 2] = m_term(q, d, d, a);
	A[2 * 8 + 3] = m_term(q, d, c, b);
	A[2 * 8 + 4] = m_term(q, c, d, b);
	A[2 * 8 + 5] = m_term(q, d, c, a);
	A[2 * 8 + 6] = m_term(q, c, c, b);
	A[2 * 8 + 7] = m_term(q, c, d, a);
	
	A[3 * 8 + 0] = m_term(q, d, b, d);
	A[3 * 8 + 1] = m_term(q, c, a, c);
	A[3 * 8 + 2] = m_term(q, d, b, c);
	A[3 * 8 + 3] = m_term(q, d, a, d);
	A[3 * 8 + 4] = m_term(q, c, b, d);
	A[3 * 8 + 5] = m_term(q, d, a, c);
	A[3 * 8 + 6] = m_term(q, c, a, d);
	A[3 * 8 + 7] = m_term(q, c, b, c);
	
	A[4 * 8 + 0] = m_term(q, b, d, d);
	A[4 * 8 + 1] = m_term(q, a, c, c);
	A[4 * 8 + 2] = m_term(q, b, d, c);
	A[4 * 8 + 3] = m_term(q, b, c, d);
	A[4 * 8 + 4] = m_term(q, a, d, d);
	A[4 * 8 + 5] = m_term(q, b, c, c);
	A[4 * 8 + 6] = m_term(q, a, c, d);
	A[4 * 8 + 7] = m_term(q, a, d, c);
	
	A[5 * 8 + 0] = m_term(q, d, b, b);
	A[5 * 8 + 1] = m_term(q, c, a, a);
	A[5 * 8 + 2] = m_term(q, d, b, a);
	A[5 * 8 + 3] = m_term(q, d, a, b);
	A[5 * 8 + 4] = m_term(q, c, b, b);
	A[5 * 8 + 5] = m_term(q, d, a, a);
	A[5 * 8 + 6] = m_term(q, c, a, b);
	A[5 * 8 + 7] = m_term(q, c, b, a);
	
	A[6 * 8 + 0] = m_term(q, b, b, d);
	A[6 * 8 + 1] = m_term(q, a, a, c);
	A[6 * 8 + 2] = m_term(q, b, b, c);
	A[6 * 8 + 3] = m_term(q, b, a, d);
	A[6 * 8 + 4] = m_term(q, a, b, d);
	A[6 * 8 + 5] = m_term(q, b, a, c);
	A[6 * 8 + 6] = m_term(q, a, a, d);
	A[6 * 8 + 7] = m_term(q, a, b, c);
	
	A[7 * 8 + 0] = m_term(q, b, d, b);
	A[7 * 8 + 1] = m_term(q, a, c, a);
	A[7 * 8 + 2] = m_term(q, b, d, a);
	A[7 * 8 + 3] = m_term(q, b, c, b);
	A[7 * 8 + 4] = m_term(q, a, d, b);
	A[7 * 8 + 5] = m_term(q, b, c, a);
	A[7 * 8 + 6] = m_term(q, a, c, b);
	A[7 * 8 + 7] = m_term(q, a, d, a);

	transpose_matrix_in_place(A, 8);
}

void finite_field::representing_matrix9_R(INT *A, INT q, INT a, INT b, INT c, INT d)
{
	INT i;
	INT tq = 2 * q;
	INT tq1 = 2 * q + 1;
	INT tq2 = 2 * q + 2;
	INT q1 = q + 1;
	INT q2 = q + 2;
	
	
	for (i = 0; i < 9 * 9; i++) {
		A[i] = 0;
		}
	A[0 * 9 + 0] = term1(d,tq2);
	A[0 * 9 + 1] = term2(c,d,q1,q1);
	A[0 * 9 + 2] = term1(c,tq2);
	A[0 * 9 + 3] = term2(c,d,q,q2);
	A[0 * 9 + 4] = term2(c,d,1,tq1);
	A[0 * 9 + 5] = term2(c,d,tq,2);
	A[0 * 9 + 6] = term2(c,d,2,tq);
	A[0 * 9 + 7] = term2(c,d,tq1,1);
	A[0 * 9 + 8] = term2(c,d,q2,q);

	A[1 * 9 + 0] = four_times(term2(b,d,q1,q1));
	A[1 * 9 + 1] = add(add(add(term2(a,d,q1,q1),term4(a,b,c,d,q,1,1,q)),term4(a,b,c,d,1,q,q,1)),term2(b,c,q1,q1));
	A[1 * 9 + 2] = four_times(term2(a,c,q1,q1));
	A[1 * 9 + 3] = twice(add(term3(a,b,d,q,1,q1),term3(b,c,d,q1,q,1)));
	A[1 * 9 + 4] = twice(add(term3(a,b,d,1,q,q1),term3(b,c,d,q1,1,q)));
	A[1 * 9 + 5] = four_times(term4(a,b,c,d,q,1,q,1));
	A[1 * 9 + 6] = four_times(term4(a,b,c,d,1,q,1,q));
	A[1 * 9 + 7] = twice(add(term3(a,b,c,q,1,q1),term3(a,c,d,q1,q,1)));
	A[1 * 9 + 8] = twice(add(term3(a,c,d,q1,1,q),term3(a,b,c,1,q,q1)));

	A[2 * 9 + 0] = term1(b,tq2);
	A[2 * 9 + 1] = term2(a,b,q1,q1);
	A[2 * 9 + 2] = term1(a,tq2);
	A[2 * 9 + 3] = term2(a,b,q,q2);
	A[2 * 9 + 4] = term2(a,b,1,tq1);
	A[2 * 9 + 5] = term2(a,b,tq,2);
	A[2 * 9 + 6] = term2(a,b,2,tq);
	A[2 * 9 + 7] = term2(a,b,tq1,1);
	A[2 * 9 + 8] = term2(a,b,q2,q);
	
	A[3 * 9 + 0] = twice(term2(b,d,q,q2));
	A[3 * 9 + 1] = add(term3(a,c,d,q,1,q1),term3(b,c,d,q,q1,1));
	A[3 * 9 + 2] = twice(term2(a,c,q,q2));
	A[3 * 9 + 3] = add(term2(a,d,q,q2),term3(b,c,d,q,q,2));
	A[3 * 9 + 4] = twice(term3(b,c,d,q,1,q1));
	A[3 * 9 + 5] = twice(term3(a,c,d,q,q,2));
	A[3 * 9 + 6] = twice(term3(b,c,d,q,2,q));
	A[3 * 9 + 7] = twice(term3(a,c,d,q,q1,1));
	A[3 * 9 + 8] = add(term3(a,c,d,q,2,q),term2(b,c,q,q2));
	
	A[4 * 9 + 0] = twice(term2(b,d,1,tq1));
	A[4 * 9 + 1] = add(term3(a,c,d,1,q,q1),term3(b,c,d,1,q1,q));
	A[4 * 9 + 2] = twice(term2(a,c,1,tq1));
	A[4 * 9 + 3] = twice(term3(b,c,d,1,q,q1));
	A[4 * 9 + 4] = add(term2(a,d,1,tq1),term3(b,c,d,1,1,tq));
	A[4 * 9 + 5] = twice(term3(b,c,d,1,tq,1));
	A[4 * 9 + 6] = twice(term3(a,c,d,1,1,tq));
	A[4 * 9 + 7] = add(term3(a,c,d,1,tq,1),term2(b,c,1,tq1));
	A[4 * 9 + 8] = twice(term3(a,c,d,1,q1,q));
	
	A[5 * 9 + 0] = term2(b,d,tq,2);
	A[5 * 9 + 1] = term4(a,b,c,d,q,q,1,1);
	A[5 * 9 + 2] = term2(a,c,tq,2);
	A[5 * 9 + 3] = term3(a,b,d,q,q,2);
	A[5 * 9 + 4] = term3(b,c,d,tq,1,1);
	A[5 * 9 + 5] = term2(a,d,tq,2);
	A[5 * 9 + 6] = term2(b,c,tq,2);
	A[5 * 9 + 7] = term3(a,c,d,tq,1,1);
	A[5 * 9 + 8] = term3(a,b,c,q,q,2);
	
	A[6 * 9 + 0] = term2(b,d,2,tq);
	A[6 * 9 + 1] = term4(a,b,c,d,1,1,q,q);
	A[6 * 9 + 2] = term2(a,c,2,tq);
	A[6 * 9 + 3] = term3(b,c,d,2,q,q);
	A[6 * 9 + 4] = term3(a,b,d,1,1,tq);
	A[6 * 9 + 5] = term2(b,c,2,tq);
	A[6 * 9 + 6] = term2(a,d,2,tq);
	A[6 * 9 + 7] = term3(a,b,c,1,1,tq);
	A[6 * 9 + 8] = term3(a,c,d,2,q,q);
	
	A[7 * 9 + 0] = twice(term2(b,d,tq1,1));
	A[7 * 9 + 1] = add(term3(a,b,d,q1,q,1),term3(a,b,c,q,q1,1));
	A[7 * 9 + 2] = twice(term2(a,c,tq1,1));
	A[7 * 9 + 3] = twice(term3(a,b,d,q,q1,1));
	A[7 * 9 + 4] = add(term3(a,b,d,1,tq,1),term2(b,c,tq1,1));
	A[7 * 9 + 5] = twice(term3(a,b,d,tq,1,1));
	A[7 * 9 + 6] = twice(term3(a,b,c,1,tq,1));
	A[7 * 9 + 7] = add(term2(a,d,tq1,1),term3(a,b,c,tq,1,1));
	A[7 * 9 + 8] = twice(term3(a,b,c,q1,q,1));
	
	A[8 * 9 + 0] = twice(term2(b,d,q2,q));
	A[8 * 9 + 1] = add(term3(a,b,d,q1,1,q),term3(a,b,c,1,q1,q));
	A[8 * 9 + 2] = twice(term2(a,c,q2,q));
	A[8 * 9 + 3] = add(term3(a,b,d,q,2,q),term2(b,c,q2,q));
	A[8 * 9 + 4] = twice(term3(a,b,d,1,q1,q));
	A[8 * 9 + 5] = twice(term3(a,b,c,q,2,q));
	A[8 * 9 + 6] = twice(term3(a,b,d,2,q,q));
	A[8 * 9 + 7] = twice(term3(a,b,c,q1,1,q));
	A[8 * 9 + 8] = add(term2(a,d,q2,q),term3(a,b,c,2,q,q));
	transpose_matrix_in_place(A, 9);
	
}

void finite_field::representing_matrix9_U(INT *A, INT a, INT b, INT c, INT d, INT beta)
{
	INT beta_q, delta, gamma;
	INT r, q, q1,q2,tq, tq1, tq2;
	
	r = e >> 1;
	if (e != 2 * r) {
		cout << "finite_field::representing_matrix9_U field does not have a quadratic subfield" << endl;
		exit(1);
		}
	q = i_power_j(p, r);
	q1 = q + 1;
	q2 = q + 2;
	tq = 2 * q;
	tq1 = tq + 1;
	tq2 = tq + 2;
	beta_q = frobenius_power(beta, r);
	delta = inverse(add(beta, negate(beta_q)));
	gamma = mult(delta, beta);

	A[0 * 9 + 0] = N2(square(d));
	A[0 * 9 + 1] = four_times(N2(mult(b, d)));
	A[0 * 9 + 2] = N2(square(b));
	A[0 * 9 + 3] = twice(Term3(b,d,gamma,q,q2,1));
	A[0 * 9 + 4] = twice(Term3(b,d,delta,1,tq1,1));
	A[0 * 9 + 5] = Term3(b,d,gamma,tq,2,1);
	A[0 * 9 + 6] = Term3(b,d,delta,2,tq,1);
	A[0 * 9 + 7] = twice(Term3(b,d,gamma,tq1,1,1));
	A[0 * 9 + 8] = twice(Term3(b,d,delta,q2,q,1));

	A[1 * 9 + 0] = N2(mult(c,d));
	A[1 * 9 + 1] = add3(N2(mult(a,d)),N2(mult(b,c)),T2(term4(a,b,c,d,q,1,1,q)));
	A[1 * 9 + 2] = N2(mult(a,b));
	A[1 * 9 + 3] = add(Term4(a,c,d,gamma,q,1,q1,1),Term4(b,c,d,gamma,q,q1,1,1));
	A[1 * 9 + 4] = add(Term4(a,c,d,delta,1,q,q1,1),Term4(b,c,d,delta,1,q1,q,1));
	A[1 * 9 + 5] = Term5(a,b,c,d,gamma,q,q,1,1,1);
	A[1 * 9 + 6] = Term5(a,b,c,d,delta,1,1,q,q,1);
	A[1 * 9 + 7] = add(Term4(a,b,d,gamma,q1,q,1,1),Term4(a,b,c,gamma,q,q1,1,1));
	A[1 * 9 + 8] = add(Term4(a,b,c,delta,1,q1,q,1),Term4(a,b,d,delta,q1,1,q,1));
	
	A[2 * 9 + 0] = N2(square(c));
	A[2 * 9 + 1] = four_times(N2(mult(a,c)));
	A[2 * 9 + 2] = N2(square(a));
	A[2 * 9 + 3] = twice(Term3(a,c,gamma,q,q2,1));
	A[2 * 9 + 4] = twice(Term3(a,c,delta,1,tq1,1));
	A[2 * 9 + 5] = Term3(a,c,gamma,tq,2,1);
	A[2 * 9 + 6] = Term3(a,c,delta,2,tq,1);
	A[2 * 9 + 7] = twice(Term3(a,c,gamma,tq1,1,1));
	A[2 * 9 + 8] = twice(Term3(a,c,delta,q2,q,1));

	A[3 * 9 + 0] = Term2(c,d,1,tq1);
	A[3 * 9 + 1] = twice(add(Term3(a,b,d,1,q,q1),Term3(b,c,d,q1,1,q)));
	A[3 * 9 + 2] = Term2(a,b,1,tq1);
	A[3 * 9 + 3] = add3(twice(Term4(b,c,d,gamma,q,1,q1,1)),Term3(a,d,gamma,q,q2,1),Term4(b,c,d,gamma,q,q,2,1));
	A[3 * 9 + 4] = add3(twice(Term4(b,c,d,delta,1,q,q1,1)),Term3(a,d,delta,1,tq1,1),Term4(b,c,d,delta,1,1,tq,1));
	A[3 * 9 + 5] = add(Term4(a,b,d,gamma,q,q,2,1),Term4(b,c,d,gamma,tq,1,1,1));
	A[3 * 9 + 6] = add(Term4(b,c,d,delta,2,q,q,1),Term4(a,b,d,delta,1,1,tq,1));
	A[3 * 9 + 7] = add3(twice(Term4(a,b,d,gamma,q,q1,1,1)),Term4(a,b,d,gamma,1,tq,1,1),Term3(b,c,gamma,tq1,1,1));
	A[3 * 9 + 8] = add3(twice(Term4(a,b,d,delta,1,q1,q,1)),Term4(a,b,d,delta,q,2,q,1),Term3(b,c,delta,q2,q,1));
	
	A[4 * 9 + 0] = Term3(c,d,beta,1,tq1,1);
	A[4 * 9 + 1] = twice(add(Term4(a,b,d,beta,1,q,q1,1),Term4(b,c,d,beta,q1,1,q,1)));
	A[4 * 9 + 2] = Term3(a,b,beta,1,tq1,1);
	A[4 * 9 + 3] = add3(twice(Term5(b,c,d,beta,gamma,q,1,q1,1,1)),Term4(a,d,beta,gamma,q,q2,q,1),Term5(b,c,d,beta,gamma,q,q,2,q,1));
	A[4 * 9 + 4] = add3(twice(Term5(b,c,d,beta,delta,1,q,q1,q,1)),Term4(a,d,beta,delta,1,tq1,1,1),Term5(b,c,d,beta,delta,1,1,tq,1,1));
	A[4 * 9 + 5] = add(Term5(a,b,d,beta,gamma,q,q,2,q,1),Term5(b,c,d,beta,gamma,tq,1,1,1,1));
	A[4 * 9 + 6] = add(Term5(b,c,d,beta,delta,2,q,q,q,1),Term5(a,b,d,beta,delta,1,1,tq,1,1));
	A[4 * 9 + 7] = add3(twice(Term5(a,b,d,beta,gamma,q,q1,1,q,1)),Term5(a,b,d,beta,gamma,1,tq,1,1,1),Term4(b,c,beta,gamma,tq1,1,1,1));
	A[4 * 9 + 8] = add3(twice(Term5(a,b,d,beta,delta,1,q1,q,1,1)),Term5(a,b,d,beta,delta,q,2,q,q,1),Term4(b,c,beta,delta,q2,q,q,1));
	
	A[5 * 9 + 0] = Term2(c,d,2,tq);
	A[5 * 9 + 1] = four_times(Term4(a,b,c,d,1,q,1,q));
	A[5 * 9 + 2] = Term2(a,b,2,tq);
	A[5 * 9 + 3] = twice(add(Term4(a,c,d,gamma,q,q,2,1),Term4(b,c,d,gamma,q,2,q,1)));
	A[5 * 9 + 4] = twice(add(Term4(b,c,d,delta,1,tq,1,1),Term4(a,c,d,delta,1,1,tq,1)));
	A[5 * 9 + 5] = add(Term3(a,d,gamma,tq,2,1),Term3(b,c,gamma,tq,2,1));
	A[5 * 9 + 6] = add(Term3(a,d,delta,2,tq,1),Term3(b,c,delta,2,tq,1));
	A[5 * 9 + 7] = twice(add(Term4(a,b,d,gamma,tq,1,1,1),Term4(a,b,c,gamma,1,tq,1,1)));
	A[5 * 9 + 8] = twice(add(Term4(a,b,c,delta,q,2,q,1),Term4(a,b,d,delta,2,q,q,1)));
	
	A[6 * 9 + 0] = Term3(c,d,beta,2,tq,1);
	A[6 * 9 + 1] = four_times(Term5(a,b,c,d,beta,1,q,1,q,1));
	A[6 * 9 + 2] = Term3(a,b,beta,2,tq,1);
	A[6 * 9 + 3] = twice(add(Term5(a,c,d,beta,gamma,q,q,2,q,1),Term5(b,c,d,beta,gamma,q,2,q,1,1)));
	A[6 * 9 + 4] = twice(add(Term5(b,c,d,beta,delta,1,tq,1,q,1),Term5(a,c,d,beta,delta,1,1,tq,1,1)));
	A[6 * 9 + 5] = add(Term4(a,d,beta,gamma,tq,2,q,1),Term4(b,c,beta,gamma,tq,2,1,1));
	A[6 * 9 + 6] = add(Term4(a,d,beta,delta,2,tq,1,1),Term4(b,c,beta,delta,2,tq,q,1));
	A[6 * 9 + 7] = twice(add(Term5(a,b,d,beta,gamma,tq,1,1,q,1),Term5(a,b,c,beta,gamma,1,tq,1,1,1)));
	A[6 * 9 + 8] = twice(add(Term5(a,b,c,beta,delta,q,2,q,q,1),Term5(a,b,d,beta,delta,2,q,q,1,1)));
	
	A[7 * 9 + 0] = Term2(c,d,q2,q);
	A[7 * 9 + 1] = twice(add(Term3(a,c,d,q1,1,q),Term3(a,b,c,1,q,q1)));
	A[7 * 9 + 2] = Term2(a,b,q2,q);
	A[7 * 9 + 3] = add3(twice(Term4(a,c,d,gamma,q,q1,1,1)),Term4(a,c,d,gamma,q,2,q,1),Term3(b,c,gamma,q,q2,1));
	A[7 * 9 + 4] = add3(twice(Term4(a,c,d,delta,1,q1,q,1)),Term4(a,c,d,delta,1,tq,1,1),Term3(b,c,delta,1,tq1,1));
	A[7 * 9 + 5] = add(Term4(a,c,d,gamma,tq,1,1,1),Term4(a,b,c,gamma,q,q,2,1));
	A[7 * 9 + 6] = add(Term4(a,b,c,delta,1,1,tq,1),Term4(a,c,d,delta,2,q,q,1));
	A[7 * 9 + 7] = add3(twice(Term4(a,b,c,gamma,q1,q,1,1)),Term3(a,d,gamma,tq1,1,1),Term4(a,b,c,gamma,tq,1,1,1));
	A[7 * 9 + 8] = add3(twice(Term4(a,b,c,delta,q1,1,q,1)),Term3(a,d,delta,q2,q,1),Term4(a,b,c,delta,2,q,q,1));
	
	A[8 * 9 + 0] = Term3(c,d,beta,q2,q,1);
	A[8 * 9 + 1] = twice(add(Term4(a,c,d,beta,q1,1,q,1),Term4(a,b,c,beta,1,q,q1,1)));
	A[8 * 9 + 2] = Term3(a,b,beta,q2,q,1);
	A[8 * 9 + 3] = add3(twice(Term5(a,c,d,beta,gamma,q,q1,1,q,1)),Term5(a,c,d,beta,gamma,q,2,q,1,1),Term4(b,c,beta,gamma,q,q2,1,1));
	A[8 * 9 + 4] = add3(twice(Term5(a,c,d,beta,delta,1,q1,q,1,1)),Term5(a,c,d,beta,delta,1,tq,1,q,1),Term4(b,c,beta,delta,1,tq1,q,1));
	A[8 * 9 + 5] = add(Term5(a,c,d,beta,gamma,tq,1,1,q,1),Term5(a,b,c,beta,gamma,q,q,2,1,1));
	A[8 * 9 + 6] = add(Term5(a,b,c,beta,delta,1,1,tq,q,1),Term5(a,c,d,beta,delta,2,q,q,1,1));
	A[8 * 9 + 7] = add3(twice(Term5(a,b,c,beta,gamma,q1,q,1,1,1)),Term4(a,d,beta,gamma,tq1,1,q,1),Term5(a,b,c,beta,gamma,tq,1,1,q,1));
	A[8 * 9 + 8] = add3(twice(Term5(a,b,c,beta,delta,q1,1,q,q,1)),Term4(a,d,beta,delta,q2,q,1,1),Term5(a,b,c,beta,delta,2,q,q,1,1));
	
	
}

void finite_field::representing_matrix8_U(INT *A, INT a, INT b, INT c, INT d, INT beta)
{
	INT delta1, delta2;
	INT r, q, i, j;
	INT beta_2, beta_11, beta_22, beta_21, beta_12, beta_123, beta_132;
	INT gamma3, gamma4, gamma5, gamma6, gamma7, gamma8;
	INT *eta, *M1, *B1;
	INT *zeta, *M2, *B2;
	
	
	
	r = e / 3;
	if (e != 3 * r) {
		cout << "finite_field::representing_matrix8_U field does not have a cubic subfield" << endl;
		exit(1);
		}
	//cout << "a=" << a << " b=" << b << " c=" << c << " d=" << d << endl;
	
	q = i_power_j(p, r);
	beta_2 = beta_trinomial(q, beta, 0, 0, 2);
	beta_11 = beta_trinomial(q, beta, 0, 1, 1);
	beta_22 = beta_trinomial(q, beta, 0, 2, 2);
	beta_21 = beta_trinomial(q, beta, 0, 2, 1);
	beta_12 = beta_trinomial(q, beta, 0, 1, 2);
	beta_123 = beta_trinomial(q, beta, 1, 2, 3);
	beta_132 = beta_trinomial(q, beta, 1, 3, 2);
	delta1 = inverse(T3(add(beta_21, negate(beta_12))));
	delta2 = inverse(T3(add(beta_123, negate(beta_132))));
	gamma3 = add(beta_trinomial(q, beta, 1, 0, 2), negate(beta_trinomial(q, beta, 2, 0, 1)));
	gamma4 = add(beta_trinomial(q, beta, 2, 0, 0), negate(beta_trinomial(q, beta, 0, 0, 2)));
	gamma5 = add(beta_trinomial(q, beta, 0, 0, 1), negate(beta_trinomial(q, beta, 1, 0, 0)));
	gamma6 = add(beta_trinomial(q, beta, 1, 2, 3), negate(beta_trinomial(q, beta, 2, 1, 3)));
	gamma7 = add(beta_trinomial(q, beta, 2, 0, 2), negate(beta_trinomial(q, beta, 0, 2, 2)));
	gamma8 = add(beta_trinomial(q, beta, 0, 1, 1), negate(beta_trinomial(q, beta, 1, 0, 1)));
	//cout << "delta1=" << delta1 << endl;
	//cout << "delta2=" << delta2 << endl;
	//cout << "gamma8=" << gamma8 << endl;
	//cout << "beta_trinomial(q, beta, 1, 2, 3)=" << beta_trinomial(q, beta, 1, 2, 3) << endl;
	//cout << "beta_trinomial(q, beta, 2, 1, 3)=" << beta_trinomial(q, beta, 2, 1, 3) << endl;
	
	eta = NEW_INT(2 * 3);
	zeta = NEW_INT(2 * 3);
	M1 = NEW_INT(2 * 3);
	M2 = NEW_INT(2 * 3);
	B1 = NEW_INT(3 * 3);
	B2 = NEW_INT(3 * 3);
	
	M1[0 * 3 + 0] = m_term(q, d, b, c);
	M1[0 * 3 + 1] = m_term(q, d, a, d);
	M1[0 * 3 + 2] = m_term(q, c, b, d);
	M1[1 * 3 + 0] = m_term(q, b, b, c);
	M1[1 * 3 + 1] = m_term(q, b, a, d);
	M1[1 * 3 + 2] = m_term(q, a, b, d);
	
	M2[0 * 3 + 0] = m_term(q, d, a, c);
	M2[0 * 3 + 1] = m_term(q, c, a, d);
	M2[0 * 3 + 2] = m_term(q, c, b, c);
	M2[1 * 3 + 0] = m_term(q, b, a, c);
	M2[1 * 3 + 1] = m_term(q, a, a, d);
	M2[1 * 3 + 2] = m_term(q, a, b, c);
	
	B1[0 * 3 + 0] = 1;
	B1[0 * 3 + 1] = beta_trinomial(q, beta, 0, 0, 1);
	B1[0 * 3 + 2] = beta_trinomial(q, beta, 0, 0, 2);
	B1[1 * 3 + 0] = 1;
	B1[1 * 3 + 1] = beta_trinomial(q, beta, 0, 1, 0);
	B1[1 * 3 + 2] = beta_trinomial(q, beta, 0, 2, 0);
	B1[2 * 3 + 0] = 1;
	B1[2 * 3 + 1] = beta_trinomial(q, beta, 1, 0, 0);
	B1[2 * 3 + 2] = beta_trinomial(q, beta, 2, 0, 0);
	
	B2[0 * 3 + 0] = 1;
	B2[0 * 3 + 1] = beta_trinomial(q, beta, 0, 1, 1);
	B2[0 * 3 + 2] = beta_trinomial(q, beta, 0, 2, 2);
	B2[1 * 3 + 0] = 1;
	B2[1 * 3 + 1] = beta_trinomial(q, beta, 1, 1, 0);
	B2[1 * 3 + 2] = beta_trinomial(q, beta, 2, 2, 0);
	B2[2 * 3 + 0] = 1;
	B2[2 * 3 + 1] = beta_trinomial(q, beta, 1, 0, 1);
	B2[2 * 3 + 2] = beta_trinomial(q, beta, 2, 0, 2);
	
	mult_matrix(M1, B1, eta, 2, 3, 3);
	mult_matrix(M2, B2, zeta, 2, 3, 3);
	INT eta11, eta12, eta13;
	INT eta21, eta22, eta23;
	INT zeta11, zeta12, zeta13;
	INT zeta21, zeta22, zeta23;
	eta11 = eta[0*3+0];
	eta12 = eta[0*3+1];
	eta13 = eta[0*3+2];
	eta21 = eta[1*3+0];
	eta22 = eta[1*3+1];
	eta23 = eta[1*3+2];
	zeta11 = zeta[0*3+0];
	zeta12 = zeta[0*3+1];
	zeta13 = zeta[0*3+2];
	zeta21 = zeta[1*3+0];
	zeta22 = zeta[1*3+1];
	zeta23 = zeta[1*3+2];

	//cout << "eta22=" << eta22 << endl;
	//cout << "eta22 gamma8=" << mult(eta22,gamma8) << endl;
	
	A[0 * 8 + 0] = N3(d);
	A[0 * 8 + 1] = N3(b);
	A[0 * 8 + 2] = T3product2(m_term(q, d, b, d), gamma3);
	A[0 * 8 + 3] = T3product2(m_term(q, d, b, d), gamma4);
	A[0 * 8 + 4] = T3product2(m_term(q, d, b, d), gamma5);
	A[0 * 8 + 5] = T3product2(m_term(q, b, b, d), gamma6);
	A[0 * 8 + 6] = T3product2(m_term(q, b, b, d), gamma7);
	A[0 * 8 + 7] = T3product2(m_term(q, b, b, d), gamma8);

	A[1 * 8 + 0] = N3(c);
	A[1 * 8 + 1] = N3(a);
	A[1 * 8 + 2] = T3product2(m_term(q, c, a, c), gamma3);
	A[1 * 8 + 3] = T3product2(m_term(q, c, a, c), gamma4);
	A[1 * 8 + 4] = T3product2(m_term(q, c, a, c), gamma5);
	A[1 * 8 + 5] = T3product2(m_term(q, a, a, c), gamma6);
	A[1 * 8 + 6] = T3product2(m_term(q, a, a, c), gamma7);
	A[1 * 8 + 7] = T3product2(m_term(q, a, a, c), gamma8);
	
	A[2 * 8 + 0] = T3(m_term(q, d, d, c));
	A[2 * 8 + 1] = T3(m_term(q, b, b, a));
	A[2 * 8 + 2] = T3product2(eta11, gamma3);
	A[2 * 8 + 3] = T3product2(eta11, gamma4);
	A[2 * 8 + 4] = T3product2(eta11, gamma5);
	A[2 * 8 + 5] = T3product2(eta21, gamma6);
	A[2 * 8 + 6] = T3product2(eta21, gamma7);
	A[2 * 8 + 7] = T3product2(eta21, gamma8);
	
	A[3 * 8 + 0] = T3product2(m_term(q, d, d, c), beta);
	A[3 * 8 + 1] = T3product2(m_term(q, b, b, a), beta);
	A[3 * 8 + 2] = T3product2(eta12, gamma3);
	A[3 * 8 + 3] = T3product2(eta12, gamma4);
	A[3 * 8 + 4] = T3product2(eta12, gamma5);
	A[3 * 8 + 5] = T3product2(eta22, gamma6);
	A[3 * 8 + 6] = T3product2(eta22, gamma7);
	A[3 * 8 + 7] = T3product2(eta22, gamma8);
	
	A[4 * 8 + 0] = T3product2(m_term(q, d, d, c), beta_2);
	A[4 * 8 + 1] = T3product2(m_term(q, b, b, a), beta_2);
	A[4 * 8 + 2] = T3product2(eta13, gamma3);
	A[4 * 8 + 3] = T3product2(eta13, gamma4);
	A[4 * 8 + 4] = T3product2(eta13, gamma5);
	A[4 * 8 + 5] = T3product2(eta23, gamma6);
	A[4 * 8 + 6] = T3product2(eta23, gamma7);
	A[4 * 8 + 7] = T3product2(eta23, gamma8);
	
	A[5 * 8 + 0] = T3product2(m_term(q, d, c, c), 1);
	A[5 * 8 + 1] = T3product2(m_term(q, b, a, a), 1);
	A[5 * 8 + 2] = T3product2(zeta11, gamma3);
	A[5 * 8 + 3] = T3product2(zeta11, gamma4);
	A[5 * 8 + 4] = T3product2(zeta11, gamma5);
	A[5 * 8 + 5] = T3product2(zeta21, gamma6);
	A[5 * 8 + 6] = T3product2(zeta21, gamma7);
	A[5 * 8 + 7] = T3product2(zeta21, gamma8);
	
	A[6 * 8 + 0] = T3product2(m_term(q, d, c, c), beta_11);
	A[6 * 8 + 1] = T3product2(m_term(q, b, a, a), beta_11);
	A[6 * 8 + 2] = T3product2(zeta12, gamma3);
	A[6 * 8 + 3] = T3product2(zeta12, gamma4);
	A[6 * 8 + 4] = T3product2(zeta12, gamma5);
	A[6 * 8 + 5] = T3product2(zeta22, gamma6);
	A[6 * 8 + 6] = T3product2(zeta22, gamma7);
	A[6 * 8 + 7] = T3product2(zeta22, gamma8);
	
	A[7 * 8 + 0] = T3product2(m_term(q, d, c, c), beta_22);
	A[7 * 8 + 1] = T3product2(m_term(q, b, a, a), beta_22);
	A[7 * 8 + 2] = T3product2(zeta13, gamma3);
	A[7 * 8 + 3] = T3product2(zeta13, gamma4);
	A[7 * 8 + 4] = T3product2(zeta13, gamma5);
	A[7 * 8 + 5] = T3product2(zeta23, gamma6);
	A[7 * 8 + 6] = T3product2(zeta23, gamma7);
	A[7 * 8 + 7] = T3product2(zeta23, gamma8);
	
	for (j = 2; j <= 4; j++) {
		for (i = 0; i < 8; i++) {
			A[i * 8 + j] = mult(A[i * 8 + j], delta1);
			}
		}
	for (j = 5; j <= 7; j++) {
		for (i = 0; i < 8; i++) {
			A[i * 8 + j] = mult(A[i * 8 + j], delta2);
			}
		}
	FREE_INT(eta);
	FREE_INT(zeta);
	FREE_INT(M1);
	FREE_INT(M2);
	FREE_INT(B1);
	FREE_INT(B2);
}

void finite_field::representing_matrix8_V(INT *A, INT beta)
{
	INT delta1, delta2;
	INT beta_21, beta_12, beta_123, beta_132;
	INT r, q, i, j;
	
	
	
	r = e / 3;
	if (e != 3 * r) {
		cout << "finite_field::representing_matrix8_V field does not have a cubic subfield" << endl;
		exit(1);
		}
	//cout << "a=" << a << " b=" << b << " c=" << c << " d=" << d << endl;
	
	q = i_power_j(p, r);
	beta_21 = beta_trinomial(q, beta, 0, 2, 1);
	beta_12 = beta_trinomial(q, beta, 0, 1, 2);
	beta_123 = beta_trinomial(q, beta, 1, 2, 3);
	beta_132 = beta_trinomial(q, beta, 1, 3, 2);
	delta1 = inverse(T3(add(beta_21, negate(beta_12))));
	delta2 = inverse(T3(add(beta_123, negate(beta_132))));

	for (i = 0; i < 8 * 8; i++)
		A[i] = 0;
	
	A[0 * 8 + 0] = 1;
	A[1 * 8 + 1] = 1;
	A[2 * 8 + 2] = T3(add(beta_trinomial(q, beta, 0, 2, 1), negate(beta_trinomial(q, beta, 0, 1, 2))));
	A[2 * 8 + 3] = T3(add(beta_trinomial(q, beta, 0, 0, 2), negate(beta_trinomial(q, beta, 0, 2, 0))));
	A[2 * 8 + 4] = T3(add(beta_trinomial(q, beta, 0, 1, 0), negate(beta_trinomial(q, beta, 0, 0, 1))));
	A[3 * 8 + 2] = T3(add(beta_trinomial(q, beta, 0, 3, 1), negate(beta_trinomial(q, beta, 0, 2, 2))));
	A[3 * 8 + 3] = T3(add(beta_trinomial(q, beta, 0, 1, 2), negate(beta_trinomial(q, beta, 0, 3, 0))));
	A[3 * 8 + 4] = T3(add(beta_trinomial(q, beta, 0, 2, 0), negate(beta_trinomial(q, beta, 0, 1, 1))));
	A[4 * 8 + 2] = T3(add(beta_trinomial(q, beta, 0, 4, 1), negate(beta_trinomial(q, beta, 0, 3, 2))));
	A[4 * 8 + 3] = T3(add(beta_trinomial(q, beta, 0, 2, 2), negate(beta_trinomial(q, beta, 0, 4, 0))));
	A[4 * 8 + 4] = T3(add(beta_trinomial(q, beta, 0, 3, 0), negate(beta_trinomial(q, beta, 0, 2, 1))));

	A[5 * 8 + 5] = T3(add(beta_trinomial(q, beta, 2, 3, 1), negate(beta_trinomial(q, beta, 1, 3, 2))));
	A[5 * 8 + 6] = T3(add(beta_trinomial(q, beta, 0, 2, 2), negate(beta_trinomial(q, beta, 2, 2, 0))));
	A[5 * 8 + 7] = T3(add(beta_trinomial(q, beta, 1, 1, 0), negate(beta_trinomial(q, beta, 0, 1, 1))));
	A[6 * 8 + 5] = T3(add(beta_trinomial(q, beta, 3, 4, 1), negate(beta_trinomial(q, beta, 2, 4, 2))));
	A[6 * 8 + 6] = T3(add(beta_trinomial(q, beta, 1, 3, 2), negate(beta_trinomial(q, beta, 3, 3, 0))));
	A[6 * 8 + 7] = T3(add(beta_trinomial(q, beta, 2, 2, 0), negate(beta_trinomial(q, beta, 1, 2, 1))));
	A[7 * 8 + 5] = T3(add(beta_trinomial(q, beta, 4, 5, 1), negate(beta_trinomial(q, beta, 3, 5, 2))));
	A[7 * 8 + 6] = T3(add(beta_trinomial(q, beta, 2, 4, 2), negate(beta_trinomial(q, beta, 4, 4, 0))));
	A[7 * 8 + 7] = T3(add(beta_trinomial(q, beta, 3, 3, 0), negate(beta_trinomial(q, beta, 2, 3, 1))));

	for (j = 2; j <= 4; j++) {
		for (i = 2; i <= 4; i++) {
			A[i * 8 + j] = mult(A[i * 8 + j], delta1);
			}
		}
	for (j = 5; j <= 7; j++) {
		for (i = 5; i <= 7; i++) {
			A[i * 8 + j] = mult(A[i * 8 + j], delta2);
			}
		}
}


void finite_field::representing_matrix9b(INT *A, INT beta)
{
	INT r, q, beta_q, delta, minus_one, i; // gamma, betagamma, i, Tgamma, Tbetagamma, nTgamma;
	
	r = e / 2;
	if (e != 2 * r) {
		cout << "finite_field::representing_matrix9b field does not have a quadratic subfield" << endl;
		exit(1);
		}
	q = i_power_j(p, r);
	minus_one = negate(1);
	beta_q = frobenius_power(beta, r);
	delta = add(beta_q, beta);
	//gamma = mult(delta, beta);
	//betagamma = mult(beta, gamma);
	//Tgamma = T2(gamma);
	//Tbetagamma = T2(betagamma);
	//nTgamma = negate(Tgamma);
	//cout << "gamma=" << gamma << endl;
	//cout << "betagamma=" << betagamma << endl;
	//cout << "Tgamma=" << Tgamma << endl;
	//cout << "nTgamma=" << nTgamma << endl;
	//cout << "Tbetagamma=" << Tbetagamma << endl;
	
	for (i = 0; i < 81; i++)
		A[i] = 0;
	
	// changed to new base:
	// attention, now transposed!
	A[0 * 9 + 0] = 1;
	A[1 * 9 + 1] = 1;
	A[2 * 9 + 2] = 1;
	A[3 * 9 + 3] = 1;
	A[4 * 9 + 3] = delta;
	A[4 * 9 + 4] = minus_one;
	A[5 * 9 + 5] = 1;
	A[6 * 9 + 5] = delta;
	A[6 * 9 + 6] = minus_one;
	A[7 * 9 + 7] = 1;
	A[8 * 9 + 7] = delta;
	A[8 * 9 + 8] = minus_one;
#if 0
	// changed to new base:
	A[0 * 9 + 0] = 1;
	A[1 * 9 + 1] = 1;
	A[2 * 9 + 2] = 1;
	A[4 * 9 + 4] = Tgamma;
	A[4 * 9 + 3] = Tbetagamma;
	A[6 * 9 + 6] = Tgamma;
	A[6 * 9 + 5] = Tbetagamma;
	A[3 * 9 + 3] = nTgamma;
	A[8 * 9 + 8] = Tgamma;
	A[8 * 9 + 7] = Tbetagamma;
	A[5 * 9 + 5] = nTgamma;
	A[7 * 9 + 7] = nTgamma;
#endif
}

void finite_field::representing_matrix8a(INT *A, INT a, INT b, INT c, INT d, INT beta)
{


#if 0
	INT delta, omega, gamma, eta, zeta, epsilon, xi, tau;
	INT r, q;
	
	r = e / 3;
	if (e != 3 * r) {
		cout << "finite_field::representing_matrix8a field does not have a cubic subfield" << endl;
		exit(1);
		}
	q = i_power_j(p, r);
	
	delta = inverse(add(T3(power(beta, 2*q+1)),negate(T3(power(beta,q+2)))));
	omega = inverse(add(T3(power(beta,q*q+2*q+3)),negate(T3(power(beta,q*q+3*q+2)))));
	gamma = add(power(beta,2*q),negate(power(beta,2*q*q)));
	eta = add(power(beta,2*q*q+q),negate(power(beta,q*q+2*q)));
	zeta = add(power(beta,q),negate(power(beta,1)));
	epsilon = add(power(beta,3*q*q+q+2),negate(power(beta,3*q*q+2*q+1)));
	xi = add(power(beta,2*q*q+2*q),negate(power(beta,2*q*q+2)));
	tau = add(power(beta,q*q+1),negate(power(beta,q*q+q)));
	
	//cout << "delta=" << delta << endl;
	//cout << "omega=" << omega << endl;
	//cout << "gamma=" << gamma << endl;
	//cout << "eta=" << eta << endl;
	//cout << "zeta=" << zeta << endl;
	//cout << "epsilon=" << epsilon << endl;
	//cout << "xi=" << xi << endl;
	//cout << "tau=" << tau << endl;
	
A[0 * 8 + 0] = N3(d);
A[0 * 8 + 1] = T3(term3(beta,c,d,1,1,q*q+q));
A[0 * 8 + 2] = T3(term2(c,d,1,q*q+q));
A[0 * 8 + 3] = T3(term2(c,d,q+1,q*q));
A[0 * 8 + 4] = T3(term3(beta,c,d,2,1,q*q+q));
A[0 * 8 + 5] = T3(term3(beta,c,d,q+1,q+1,q*q));
A[0 * 8 + 6] = T3(term3(beta,c,d,2*q+2,q+1,q*q));
A[0 * 8 + 7] = N3(c);
	
A[1*8+0]=mult(delta,T3(term3(gamma,b,d,1,1,q*q+q)));
A[1*8+1]=mult(delta,T3(mult(beta,add(add(term3(gamma,a,d,1,1,q*q+q),term4(gamma,b,c,d,q,q,1,q*q)),term4(gamma,b,c,d,q*q,q*q,1,q)))));
A[1*8+2]=mult(delta,T3(mult(gamma,add(add(term2(a,d,1,q*q+q),term3(b,c,d,1,q,q*q)),term3(b,c,d,1,q*q,q)))));
A[1*8+3]=mult(delta,T3(mult(gamma,add(add(term3(a,c,d,1,q,q*q),term2(b,c,1,q*q+q)),term3(a,c,d,1,q*q,q)))));
A[1*8+4]=mult(delta,T3(mult(mult(beta,beta),add(add(term3(gamma,a,d,1,1,q*q+q),term4(gamma,b,c,d,q,q,1,q*q)),term4(gamma,b,c,d,q*q,q*q,1,q)))));
A[1*8+5]=mult(delta,T3(mult(power(beta,q+1),add(add(term4(gamma,a,c,d,1,1,q,q*q),term4(gamma,a,c,d,q,q,1,q*q)),term3(gamma,b,c,q*q,q*q,q+1)))));
A[1*8+6]=mult(delta,T3(mult(power(beta,2*q+2),add(add(term4(gamma,a,c,d,1,1,q,q*q),term4(gamma,a,c,d,q,q,1,q*q)),term3(gamma,b,c,q*q,q*q,q+1)))));
A[1*8+7]=mult(delta,T3(term3(gamma,a,c,1,1,q*q+q)));

A[2*8+0]=mult(delta,T3(term3(eta,b,d,1,1,q*q+q)));
A[2*8+1]=mult(delta,T3(mult(beta,add(add(term3(eta,a,d,1,1,q*q+q),term4(eta,b,c,d,q,q,1,q*q)),term4(eta,b,c,d,q*q,q*q,1,q)))));
A[2*8+2]=mult(delta,T3(mult(eta,add(add(term3(b,c,d,1,q,q*q),term2(a,d,1,q*q+q)),term3(b,c,d,1,q*q,q)))));
A[2*8+3]=mult(delta,T3(mult(eta,add(add(term3(a,c,d,1,q,q*q),term2(b,c,1,q*q+q)),term3(a,c,d,1,q*q,q)))));
A[2*8+4]=mult(delta,T3(mult(power(beta,2),add(add(term3(eta,a,d,1,1,q*q+q),term4(eta,b,c,d,q,q,1,q*q)),term4(eta,b,c,d,q*q,q*q,1,q)))));
A[2*8+5]=mult(delta,T3(mult(power(beta,q+1),add(add(term4(eta,a,c,d,1,1,q,q*q),term4(eta,a,c,d,q,q,1,q*q)),term3(eta,b,c,q*q,q*q,q+1)))));
A[2*8+6]=mult(delta,T3(mult(power(beta,2*q+2),add(add(term4(eta,a,c,d,1,1,q,q*q),term4(eta,a,c,d,q,q,1,q*q)),term3(eta,b,c,q*q,q*q,q+1)))));
A[2*8+7]=mult(delta,T3(term3(eta,a,c,1,1,q*q+q)));

A[3*8+0]=mult(omega,T3(term3(epsilon,b,d,1,q+1,q*q)));
A[3*8+1]=mult(omega,T3(mult(beta,add(add(term4(epsilon,a,b,d,1,1,q,q*q),term3(epsilon,b,c,q,q*q+q,1)),term4(epsilon,a,b,d,q*q,1,q*q,q)))));
A[3*8+2]=mult(omega,T3(mult(epsilon,add(add(term3(a,b,d,q,1,q*q),term3(a,b,d,1,q,q*q)),term2(b,c,q+1,q*q)))));
A[3*8+3]=mult(omega,T3(mult(epsilon,add(add(term2(a,d,q+1,q*q),term3(a,b,c,q,1,q*q)),term3(a,b,c,1,q,q*q)))));
A[3*8+4]=mult(omega,T3(mult(power(beta,2),add(add(term4(epsilon,a,b,d,1,1,q,q*q),term3(epsilon,b,c,q,q*q+q,1)),term4(epsilon,a,b,d,q*q,1,q*q,q)))));
A[3*8+5]=mult(omega,T3(mult(power(beta,q+1),add(add(term3(epsilon,a,d,1,q+1,q*q),term4(epsilon,a,b,c,q,q,q*q,1)),term4(epsilon,a,b,c,q*q,1,q*q,q)))));
A[3*8+6]=mult(omega,T3(mult(power(beta,2*q+2),add(add(term3(epsilon,a,d,1,q+1,q*q),term4(epsilon,a,b,c,q,q,q*q,1)),term4(epsilon,a,b,c,q*q,1,q*q,q)))));
A[3*8+7]=mult(omega,T3(term3(epsilon,a,c,1,q+1,q*q)));


A[4*8+0]=mult(delta,T3(term3(zeta,b,d,1,q*q,q+1)));
A[4*8+1]=mult(delta,T3(mult(beta,add(add(term4(zeta,b,c,d,1,q*q,1,q),term3(zeta,a,d,q,1,q*q+q)),term4(zeta,b,c,d,q*q,q,1,q*q)))));
A[4*8+2]=mult(delta,T3(mult(zeta,add(add(term3(b,c,d,q*q,q,1),term3(b,c,d,q*q,1,q)),term2(a,d,q*q,q+1)))));
A[4*8+3]=mult(delta,T3(mult(zeta,add(add(term2(b,c,q*q,q+1),term3(a,c,d,q*q,q,1)),term3(a,c,d,q*q,1,q)))));
A[4*8+4]=mult(delta,T3(mult(power(beta,2),add(add(term4(zeta,b,c,d,1,q*q,1,q),term3(zeta,a,d,q,1,q*q+q)),term4(zeta,b,c,d,q*q,q,1,q*q)))));
A[4*8+5]=mult(delta,T3(mult(power(beta,q+1),add(add(term3(zeta,b,c,1,q*q,q+1),term4(zeta,a,c,d,q,1,q,q*q)),term4(zeta,a,c,d,q*q,q,1,q*q)))));
A[4*8+6]=mult(delta,T3(mult(power(beta,2*q+2),add(add(term3(zeta,b,c,1,q*q,q+1),term4(zeta,a,c,d,q,1,q,q*q)),term4(zeta,a,c,d,q*q,q,1,q*q)))));
A[4*8+7]=mult(delta,T3(term3(zeta,a,c,1,q*q,q+1)));

A[5*8+0]=mult(omega,T3(term3(xi,b,d,1,q+1,q*q)));
A[5*8+1]=mult(omega,T3(mult(beta,add(add(term4(xi,a,b,d,1,1,q,q*q),term3(xi,b,c,q,q*q+q,1)),term4(xi,a,b,d,q*q,1,q*q,q)))));
A[5*8+2]=mult(omega,T3(mult(xi,add(add(term3(a,b,d,q,1,q*q),term3(a,b,d,1,q,q*q)),term2(b,c,q+1,q*q)))));
A[5*8+3]=mult(omega,T3(mult(xi,add(add(term2(a,d,q+1,q*q),term3(a,b,c,q,1,q*q)),term3(a,b,c,1,q,q*q)))));
A[5*8+4]=mult(omega,T3(mult(power(beta,2),add(add(term4(xi,a,b,d,1,1,q,q*q),term3(xi,b,c,q,q*q+q,1)),term4(xi,a,b,d,q*q,1,q*q,q)))));
A[5*8+5]=mult(omega,T3(mult(power(beta,q+1),add(add(term3(xi,a,d,1,q+1,q*q),term4(xi,a,b,c,q,q,q*q,1)),term4(xi,a,b,c,q*q,1,q*q,q)))));
A[5*8+6]=mult(omega,T3(mult(power(beta,2*q+2),add(add(term3(xi,a,d,1,q+1,q*q),term4(xi,a,b,c,q,q,q*q,1)),term4(xi,a,b,c,q*q,1,q*q,q)))));
A[5*8+7]=mult(omega,T3(term3(xi,a,c,1,q+1,q*q)));

A[6*8+0]=mult(omega,T3(term3(tau,b,d,1,q+1,q*q)));
A[6*8+1]=mult(omega,T3(mult(beta,add(add(term4(tau,a,b,d,1,1,q,q*q),term3(tau,b,c,q,q*q+q,1)),term4(tau,a,b,d,q*q,1,q*q,q)))));
A[6*8+2]=mult(omega,T3(mult(tau,add(add(term3(a,b,d,q,1,q*q),term3(a,b,d,1,q,q*q)),term2(b,c,q+1,q*q)))));
A[6*8+3]=mult(omega,T3(mult(tau,add(add(term2(a,d,q+1,q*q),term3(a,b,c,q,1,q*q)),term3(a,b,c,1,q,q*q)))));
A[6*8+4]=mult(omega,T3(mult(power(beta,2),add(add(term4(tau,a,b,d,1,1,q,q*q),term3(tau,b,c,q,q*q+q,1)),term4(tau,a,b,d,q*q,1,q*q,q)))));
A[6*8+5]=mult(omega,T3(mult(power(beta,q+1),add(add(term3(tau,a,d,1,q+1,q*q),term4(tau,a,b,c,q,q,q*q,1)),term4(tau,a,b,c,q*q,1,q*q,q)))));
A[6*8+6]=mult(omega,T3(mult(power(beta,2*q+2),add(add(term3(tau,a,d,1,q+1,q*q),term4(tau,a,b,c,q,q,q*q,1)),term4(tau,a,b,c,q*q,1,q*q,q)))));
A[6*8+7]=mult(omega,T3(term3(tau,a,c,1,q+1,q*q)));

A[7 * 8 + 0] = N3(b);
A[7 * 8 + 1] = T3(term3(beta,a,b,1,1,q*q+q));
A[7 * 8 + 2] = T3(term2(a,b,1,q*q+q));
A[7 * 8 + 3] = T3(term2(a,b,q+1,q*q));
A[7 * 8 + 4] = T3(term3(beta,a,b,2,1,q*q+q));
A[7 * 8 + 5] = T3(term3(beta,a,b,q+1,q+1,q*q));
A[7 * 8 + 6] = T3(term3(beta,a,b,2*q+2,q+1,q*q));
A[7 * 8 + 7] = N3(a);
#endif
}

void finite_field::representing_matrix8b(INT *A, INT beta)
{
	INT r, q, delta, omega, i;
	
	r = e / 3;
	if (e != 3 * r) {
		cout << "finite_field::representing_matrix8b field does not have a cubic subfield" << endl;
		exit(1);
		}
	q = i_power_j(p, r);

	delta = inverse(add(T3(power(beta, 2*q+1)),negate(T3(power(beta,q+2)))));
	omega = inverse(add(T3(power(beta,q*q+2*q+3)),negate(T3(power(beta,q*q+3*q+2)))));
	cout << "delta=" << delta << endl;
	cout << "omega=" << omega << endl;
	
	for (i = 0; i < 64; i++)
		A[i] = 0;
	
	A[0 * 8 + 0] = 1;
	A[7 * 8 + 7] = 1;
#if 1
	A[1 * 8 + 1] = mult(delta,T3(add(power(beta,3),negate(power(beta,2*q+1)))));
	A[1 * 8 + 2] = mult(delta,T3(add(power(beta,2),negate(power(beta,2*q)))));
	A[1 * 8 + 4] = mult(delta,T3(add(power(beta,4),negate(power(beta,2*q+2)))));
	A[2 * 8 + 1] = mult(delta,T3(add(power(beta,2*q+2),negate(power(beta,q+3)))));
	A[2 * 8 + 2] = mult(delta,T3(add(power(beta,2*q+1),negate(power(beta,q+2)))));
	A[2 * 8 + 4] = mult(delta,T3(add(power(beta,2*q+3),negate(power(beta,q+4)))));
	A[3 * 8 + 3] = mult(omega,T3(add(power(beta,q*q+2*q+3),negate(power(beta,2*q*q+q+3)))));
	A[3 * 8 + 5] = mult(omega,T3(add(power(beta,2*q*q+4*q+2),negate(power(beta,q*q+4*q+3)))));
	A[3 * 8 + 6] = mult(omega,T3(add(power(beta,2*q*q+5*q+3),negate(power(beta,q*q+5*q+4)))));
	A[4 * 8 + 1] = mult(delta,T3(add(power(beta,q+1),negate(power(beta,2)))));
	A[4 * 8 + 2] = mult(delta,T3(add(power(beta,q),negate(power(beta,1)))));
	A[4 * 8 + 4] = mult(delta,T3(add(power(beta,q+2),negate(power(beta,3)))));
	A[5 * 8 + 3] = mult(omega,T3(add(power(beta,2*q*q+2),negate(power(beta,2*q+2)))));
	A[5 * 8 + 5] = mult(omega,T3(add(power(beta,3*q+3),negate(power(beta,2*q*q+3*q+1)))));
	A[5 * 8 + 6] = mult(omega,T3(add(power(beta,4*q+4),negate(power(beta,2*q*q+4*q+2)))));
	A[6 * 8 + 3] = mult(omega,T3(add(power(beta,q+1),negate(power(beta,q*q+1)))));
	A[6 * 8 + 5] = mult(omega,T3(add(power(beta,q*q+2*q+1),negate(power(beta,2*q+2)))));
	A[6 * 8 + 6] = mult(omega,T3(add(power(beta,q*q+3*q+2),negate(power(beta,3*q+3)))));
#else
	A[1 * 8 + 1] = mult(delta,T3(add(power(beta,q+2),negate(power(beta,3*q)))));
	A[1 * 8 + 2] = mult(delta,T3(add(power(beta,2),negate(power(beta,2*q)))));
	A[1 * 8 + 4] = mult(delta,T3(add(power(beta,2*q+2),negate(power(beta,4*q)))));
	A[2 * 8 + 1] = mult(delta,T3(add(power(beta,2*q+1),negate(power(beta,2*q+2)))));
	A[2 * 8 + 2] = mult(delta,T3(add(power(beta,2*q+1),negate(power(beta,q+2)))));
	A[2 * 8 + 4] = mult(delta,T3(add(power(beta,4*q+1),negate(power(beta,3*q+2)))));
	A[4 * 8 + 1] = mult(delta,T3(add(power(beta,2),negate(power(beta,q*q+1)))));
	A[4 * 8 + 2] = mult(delta,T3(add(power(beta,1),negate(power(beta,q*q)))));
	A[4 * 8 + 4] = mult(delta,T3(add(power(beta,3),negate(power(beta,q*q+2)))));
	A[3 * 8 + 3] = mult(omega,T3(add(power(beta,q*q+2*q+3),negate(power(beta,2*q*q+q+3)))));
	A[3 * 8 + 5] = mult(omega,T3(add(power(beta,q*q+3*q+4),negate(power(beta,2*q*q+2*q+4)))));
	A[3 * 8 + 6] = mult(omega,T3(add(power(beta,q*q+4*q+5),negate(power(beta,2*q*q+3*q+5)))));
	A[5 * 8 + 3] = mult(omega,T3(add(power(beta,2*q*q+2),negate(power(beta,2*q+2)))));
	A[5 * 8 + 5] = mult(omega,T3(add(power(beta,2*q*q+q+3),negate(power(beta,3*q+3)))));
	A[5 * 8 + 6] = mult(omega,T3(add(power(beta,2*q*q+2*q+4),negate(power(beta,4*q+4)))));
	A[6 * 8 + 3] = mult(omega,T3(add(power(beta,q+1),negate(power(beta,q*q+1)))));
	A[6 * 8 + 5] = mult(omega,T3(add(power(beta,2*q+2),negate(power(beta,q*q+q+2)))));
	A[6 * 8 + 6] = mult(omega,T3(add(power(beta,3*q+3),negate(power(beta,q*q+2*q+3)))));
	//transpose_matrix_in_place(A, 8);
#endif
}

INT finite_field::Term1(INT a1, INT e1)
{
	INT x;
	
	x = term1(a1, e1);
	return T2(x);
}

INT finite_field::Term2(INT a1, INT a2, INT e1, INT e2)
{
	INT x;
	
	x = term2(a1, a2, e1, e2);
	return T2(x);
}

INT finite_field::Term3(INT a1, INT a2, INT a3, INT e1, INT e2, INT e3)
{
	INT x;
	
	x = term3(a1, a2, a3, e1, e2, e3);
	return T2(x);
}

INT finite_field::Term4(INT a1, INT a2, INT a3, INT a4, INT e1, INT e2, INT e3, INT e4)
{
	INT x;
	
	x = term4(a1, a2, a3, a4, e1, e2, e3, e4);
	return T2(x);
}

INT finite_field::Term5(INT a1, INT a2, INT a3, INT a4, INT a5, INT e1, INT e2, INT e3, INT e4, INT e5)
{
	INT x;
	
	x = term5(a1, a2, a3, a4, a5, e1, e2, e3, e4, e5);
	return T2(x);
}

INT finite_field::term1(INT a1, INT e1)
{
	INT x;
	
	x = 1;
	if (e1) {
		x = mult(x, power(a1, e1));
		}
	return x;
}

INT finite_field::term2(INT a1, INT a2, INT e1, INT e2)
{
	INT x;
	
	x = 1;
	if (e1) {
		x = mult(x, power(a1, e1));
		}
	if (e2) {
		x = mult(x, power(a2, e2));
		}
	return x;
}

INT finite_field::term3(INT a1, INT a2, INT a3, INT e1, INT e2, INT e3)
{
	INT x;
	
	x = 1;
	if (e1) {
		x = mult(x, power(a1, e1));
		}
	if (e2) {
		x = mult(x, power(a2, e2));
		}
	if (e3) {
		x = mult(x, power(a3, e3));
		}
	return x;
}

INT finite_field::term4(INT a1, INT a2, INT a3, INT a4, INT e1, INT e2, INT e3, INT e4)
{
	INT x;
	
	x = 1;
	if (e1) {
		x = mult(x, power(a1, e1));
		}
	if (e2) {
		x = mult(x, power(a2, e2));
		}
	if (e3) {
		x = mult(x, power(a3, e3));
		}
	if (e4) {
		x = mult(x, power(a4, e4));
		}
	return x;
}

INT finite_field::term5(INT a1, INT a2, INT a3, INT a4, INT a5, INT e1, INT e2, INT e3, INT e4, INT e5)
{
	INT x;
	
	x = 1;
	if (e1) {
		x = mult(x, power(a1, e1));
		}
	if (e2) {
		x = mult(x, power(a2, e2));
		}
	if (e3) {
		x = mult(x, power(a3, e3));
		}
	if (e4) {
		x = mult(x, power(a4, e4));
		}
	if (e5) {
		x = mult(x, power(a5, e5));
		}
	return x;
}

#if 0
INT finite_field::product2(INT a1, INT a2)
{
	INT x;
	
	x = mult(x, power(a1, a2));
	return x;
}
#endif

INT finite_field::m_term(INT q, INT a1, INT a2, INT a3)
{
	INT x;
	
	x = 1;
	x = mult(x, power(a1, q * q));
	x = mult(x, power(a2, q));
	x = mult(x, a3);
	return x;
}

INT finite_field::beta_trinomial(INT q, INT beta, INT a1, INT a2, INT a3)
{
	INT x;
	
	x = 1;
	x = mult(x, power(beta, a1 * q * q));
	x = mult(x, power(beta, a2 * q));
	x = mult(x, power(beta, a3));
	return x;
}

INT finite_field::T3product2(INT a1, INT a2)
{
	INT x;
	
	x = mult(a1, a2);
	return T3(x);
}

