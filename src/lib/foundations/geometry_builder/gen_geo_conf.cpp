/*
 * gen_geo_conf.cpp
 *
 *  Created on: Aug 14, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry_builder {


gen_geo_conf::gen_geo_conf()
{
	fuse_idx = -1;

	v = 0;
	b = 0;
	r = 0;
	// int k;

	r0 = 0;
	// int k0;
	// int k1;
	i0 = 0;
	j0 = 0;
	f_last_non_zero_in_fuse = FALSE;

}

gen_geo_conf::~gen_geo_conf()
{

}

void gen_geo_conf::print(std::ostream &ost)
{
	ost << "v=" << v << " b=" << b << " r=" << r
			<< " r0=" << r0 << " i0=" << i0 << " j0=" << j0 << endl;
}

}}}


