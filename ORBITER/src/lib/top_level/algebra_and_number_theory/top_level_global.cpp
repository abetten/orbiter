// top_level_global.C
//
// Anton Betten
// September 2, 2016

#include "orbiter.h"



int callback_partial_ovoid_test(int len, int *S,
		void *data, int verbose_level)
{
	classify_double_sixes *Classify_double_sixes =
			(classify_double_sixes *) data;
	//surface_classify_wedge *SCW = (surface_classify_wedge *) data;
	
	return Classify_double_sixes->partial_ovoid_test(
			S, len, verbose_level);
}



