/*
 * cra.cpp
 *
 *  Created on: Dec 8, 2019
 *      Author: betten
 *
 *      from the ginac project check directory
 */




//#include "ginac/polynomial/cra_garner.h"
#include <cln/integer.h>
#include <vector>

namespace cln {

extern cl_I integer_cra(const std::vector<cl_I>& residues,
	                const std::vector<cl_I>& moduli);

} // namespace cln

#include <cln/integer.h>
#include <cln/integer_io.h>
#include <cln/random.h>
#include <cln/numtheory.h>
using namespace cln;
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>
#include <algorithm>
using namespace std;

/// Generate a sequences of primes p_i such that \prod_i p_i < limit
static std::vector<cln::cl_I>
make_random_moduli(const cln::cl_I& limit);

static std::vector<cln::cl_I>
calc_residues(const cln::cl_I& x, const std::vector<cln::cl_I>& moduli);

static void dump(const std::vector<cln::cl_I>& v);

/// Make @a n random relatively prime moduli, each < limit, make a
/// random number x < \prod_{i=0}{n-1}, calculate residues, and
/// compute x' by chinese remainder algorithm. Check if the result
/// of computation matches the original value x.
static void run_test_once(const cln::cl_I& lim)
{
	std::vector<cln::cl_I> moduli = make_random_moduli(lim);
	cln::cl_I x = random_I(lim) + 1;

	if (x > (lim >> 1))
		x = x - lim;

	std::vector<cln::cl_I> residues = calc_residues(x, moduli);
	cln::cl_I x_test;

	bool error = false;
	try {
		cout << "residues=";
		dump(residues);
		cout << ", ";
		cout << "moduli=";
		dump(moduli);
		cout << endl;
		x_test = integer_cra(residues, moduli);
		cout << "result = " << x_test << endl;
	} catch (std::exception& oops) {
		std::cerr << "Oops: " << oops.what() << std::endl;
		error = true;
	}

	if (x != x_test)
		error = true;

	if (error) {
		std::cerr << "Expected x = " << x << ", got " <<
			x_test << " instead" << std::endl;
		std::cerr << "moduli = ";
		dump(moduli);
		std::cerr << std::endl;
		std::cerr << "residues = ";
		dump(residues);
		std::cerr << std::endl;
		throw std::logic_error("bug in integer_cra?");
	}
}

static void run_test(const cln::cl_I& limit, const std::size_t ntimes)
{
	for (std::size_t i = 0; i < ntimes; ++i) {
		cout << "run_test " << i << " / " << ntimes << endl;
		run_test_once(limit);
	}
}

int main(int argc, char** argv)
{
	typedef std::map<cln::cl_I, std::size_t> map_t;
	map_t the_map;
	// Run 1024 tests with native 32-bit numbers
	the_map[cln::cl_I(std::numeric_limits<int>::max())] = 1024;

	// Run 512 tests with native 64-bit integers
	if (sizeof(long) > sizeof(int))
		the_map[cln::cl_I(std::numeric_limits<long>::max())] = 512;

	// Run 32 tests with a bit bigger numbers
	the_map[cln::cl_I("987654321098765432109876543210")] = 32;

	std::cout << "examining Garner's integer chinese remainder algorithm " << std::flush;

	for (map_t::const_iterator i = the_map.begin(); i != the_map.end(); ++i)
		run_test(i->first, i->second);

	return 0;
}

static std::vector<cln::cl_I>
calc_residues(const cln::cl_I& x, const std::vector<cln::cl_I>& moduli)
{
	std::vector<cln::cl_I> residues(moduli.size());
	for (std::size_t i = moduli.size(); i-- != 0; )
		residues[i] = mod(x, moduli[i]);
	return residues;
}

static std::vector<cln::cl_I>
make_random_moduli(const cln::cl_I& limit)
{
	std::vector<cln::cl_I> moduli;
	cln::cl_I prod(1);
	cln::cl_I next = random_I(std::min(limit >> 1, cln::cl_I(128)));
	unsigned count = 0;
	do {
		cln::cl_I tmp = nextprobprime(next);
		next = tmp + random_I(cln::cl_I(10)) + 1;
		prod = prod*tmp;
		moduli.push_back(tmp);
		++count;
	} while (prod < limit || (count < 2));
	return moduli;
}

static void dump(const std::vector<cln::cl_I>& v)
{
	std::cerr << "[ ";
	for (std::size_t i = 0; i < v.size(); ++i)
		std::cerr << v[i] << " ";
	std::cerr << "]";
}


// #############################################################################
//
// #############################################################################

//#include "compiler.h"
#ifdef __GNUC__
#define unlikely(cond) __builtin_expect((cond), 0)
#define likely(cond) __builtin_expect((cond), 1)
#define attribute_deprecated __attribute__ ((deprecated))
#else
#define unlikely(cond) (cond)
#define likely(cond) (cond)
#define attribute_deprecated
#endif

#include <cln/integer.h>
#include <cln/modinteger.h>
#include <cstddef>
#include <vector>

namespace cln {

using std::vector;
using std::size_t;

static cl_I
retract_symm(const cl_MI& x, const cl_modint_ring& R,
	     const cl_I& modulus)
{
	cl_I result = R->retract(x);
	if (result > (modulus >> 1))
		result = result - modulus;
	return result;
}

static void
compute_recips(vector<cl_MI>& dst,
	       const vector<cl_I>& moduli)
{
	for (size_t k = 1; k < moduli.size(); ++k) {
		cl_modint_ring R = find_modint_ring(moduli[k]);
		cl_MI product = R->canonhom(moduli[0]);
		for (size_t i = 1; i < k; ++i)
			product = product*moduli[i];
		dst[k-1] = recip(product);
	}
}

static void
compute_mix_radix_coeffs(vector<cl_I>& dst,
	                 const vector<cl_I>& residues,
	                 const vector<cl_I>& moduli,
			 const vector<cl_MI>& recips)
{
	dst[0] = residues[0];

	do {
		cl_modint_ring R = find_modint_ring(moduli[1]);
		cl_MI tmp = R->canonhom(residues[0]);
		cl_MI next = (R->canonhom(residues[1]) - tmp)*recips[0];
		dst[1] = retract_symm(next, R, moduli[1]);
	} while (0);

	for (size_t k = 2; k < residues.size(); ++k) {
		cl_modint_ring R = find_modint_ring(moduli[k]);
		cl_MI tmp = R->canonhom(dst[k-1]);

		for (size_t j = k - 1 /* NOT k - 2 */; j-- != 0; )
			tmp = tmp*moduli[j] + R->canonhom(dst[j]);

		cl_MI next = (R->canonhom(residues[k]) - tmp)*recips[k-1];
		dst[k] = retract_symm(next, R, moduli[k]);
	}
}

static cl_I
mixed_radix_2_ordinary(const vector<cl_I>& mixed_radix_coeffs,
	               const vector<cl_I>& moduli)
{
	size_t k = mixed_radix_coeffs.size() - 1;
	cl_I u = mixed_radix_coeffs[k];
	for (; k-- != 0; )
		u = u*moduli[k] + mixed_radix_coeffs[k];
	return u;
}

cl_I integer_cra(const vector<cl_I>& residues,
	         const vector<cl_I>& moduli)
{
	if (unlikely(moduli.size() < 2))
		throw std::invalid_argument("integer_cra: need at least 2 moduli");

	vector<cl_MI> recips(moduli.size() - 1);
	compute_recips(recips, moduli);

	vector<cl_I> coeffs(moduli.size());
	compute_mix_radix_coeffs(coeffs, residues, moduli, recips);
	cl_I result = mixed_radix_2_ordinary(coeffs, moduli);

	return result;
}

} // namespace cln


