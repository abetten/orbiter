/*
 * PG.h
 *
 *  Created on: Nov 30, 2018
 *      Author: sajeeb
 */

#ifndef PG_H_
#define PG_H_

#include "Vector.h"
#include "FiniteField.h"


	template <typename T>
	__host__
	void make_num_rep(Vector<T>& v, unsigned int q) {
		// This function assumes that the vector is already normalized

		int i, j, q_power_j, b, sqj;
		int f_v = false;

		int stride = 1, len = v.size();

		if (len <= 0) {
			cout << "PG_element_rank_modified len <= 0" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "the vector before normalization is ";
			for (i = 0; i < len; i++) {
				cout << v[i * stride] << " ";
			}
			cout << endl;
		}


		if (f_v) {
			cout << "the vector after normalization is ";
			for (i = 0; i < len; i++) {
				cout << v[i * stride] << " ";
			}
			cout << endl;
		}

		for (i = 0; i < len; i++) {
			if (v[i * stride])
				break;
		}

		if (i == len) {
			cout << "PG_element_rank_modified zero vector" << endl;
			exit(1);
		}

		for (j = i + 1; j < len; j++) {
			if (v[j * stride])
				break;
		}

		if (j == len) {
			// we have the unit vector vector e_i
			v.num_rep_ = i;
			return;
		}

		// test for the all one vector:
		if (i == 0 && v[i * stride] == 1) {
			for (j = i + 1; j < len; j++) {
				if (v[j * stride] != 1)
					break;
			}
			if (j == len) {
				v.num_rep_ = len;
				return;
			}
		}

		for (i = len - 1; i >= 0; i--) {
			if (v[i * stride])
				break;
		}

		if (i < 0) {
			cout << "PG_element_rank_modified zero vector" << endl;
			exit(1);
		}

		if (v[i * stride] != 1) {
			cout << "PG_element_rank_modified vector not normalized" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "i=" << i << endl;
		}

		b = 0;
		q_power_j = 1;
		sqj = 0;

		for (j = 0; j < i; j++) {
			b += q_power_j - 1;
			sqj += q_power_j;
			q_power_j *= q;
		}

		if (f_v) {
			cout << "b=" << b << endl;
			cout << "sqj=" << sqj << endl;
		}

		v.num_rep_ = 0;

		for (j = i - 1; j >= 0; j--) {
			v.num_rep_ += v[j * stride];
			if (j > 0)
				v.num_rep_ *= q;
			if (f_v) {
				cout << "j=" << j << ", a=" << v.num_rep_ << endl;
			}
		}

		if (f_v) {
			cout << "a=" << v.num_rep_ << endl;
		}

		// take care of 1111 vector being left out
		if (i == len - 1) {
			//cout << "sqj=" << sqj << endl;
			if (v.num_rep_ >= sqj)
				v.num_rep_--;
		}

		v.num_rep_ += b;
		v.num_rep_ += len;
	}

	template <typename T>
	__host__
	void make_vector_from_number (Vector<T>& vec, unsigned int number, int q) {
		// Create a new in the heap from the number n
		// and return a pointer to it.

		int len = vec.size_;

		int a = number;
		int stride = 1;
		int n, l, ql, sql, k, j, r, a1 = a;

		n = len;

		if (a < n) {
			// unit vector:
			for (k = 0; k < n; k++) {
				if (k == a) {
					vec.vec_[k * stride] = 1;
				}
				else {
					vec.vec_[k * stride] = 0;
				}
			}
			return;
		}
		a -= n;
		if (a == 0) {
			// all one vector
			for (k = 0; k < n; k++) {
				vec.vec_[k * stride] = 1;
			}
			return;
		}
		a--;

		l = 1;
		ql = q;
		sql = 1;
		// sql = q^0 + q^1 + \cdots + q^{l-1}
		while (l < n) {
			if (a >= ql - 1) {
				a -= (ql - 1);
				sql += ql;
				ql *= q;
				l++;
				continue;
			}
			vec.vec_[l * stride] = 1;
			for (k = l + 1; k < n; k++) {
				vec.vec_[k * stride] = 0;
			}
			a++; // take into account that we do not want 00001000
			if (l == n - 1 && a >= sql) {
				a++;
				// take int account that the
				// vector 11111 has already been listed
			}
			j = 0;
			while (a != 0) {
				r = a % q;
				vec.vec_[j * stride] = r;
				j++;
				a -= r;
				a /= q;
			}
			for (; j < l; j++) {
				vec.vec_[j * stride] = 0;
			}
			return;
		}
		cout << __FILE__ << ":" << __LINE__ << endl;
		cout << "PG_element_unrank_modified a too large" << endl;
		cout << "len = " << len << endl;
		cout << "a = " << a1 << endl;
		exit(1);
	}



#endif /* PG_H_ */
