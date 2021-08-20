/*
 * iso_info.cpp
 *
 *  Created on: Aug 16, 2021
 *      Author: betten
 */




#include <stdlib.h>
#include "geo.h"

using namespace std;



iso_info::iso_info()
{
	AtheX = NULL;
	/* v x max_r;
	 * dimension v x max_r or
	 * v x MAX_R */
	BtheX = NULL;
	Af_full = FALSE;
	Bf_full = FALSE;

	v = 0;
	b = 0;
	max_r = 0;

	R = NULL; /* [MAX_V] */

	tdo_m = 0;
	//int tdo_V[MAX_V];
	tdo_n = 0;
	//int tdo_B[MAX_B];

	nb_isomorphisms = 0;
	f_break_after_fst = FALSE;
	f_verbose = FALSE;
	f_very_verbose = FALSE;
	f_use_d = FALSE;
	f_use_ddp = FALSE;
	f_use_ddb = FALSE;
	f_transpose_it = FALSE;

	/* optionally: */
	Ar = NULL; /* v entries */
	Br = NULL;
	Ad = NULL; /* v entries */
	Bd = NULL;
	Addp = NULL; /* (v \atop 2) entries */
	Bddp = NULL;
	Addb = NULL; /* (b \atop 2) entries */
	Bddb = NULL;

	//f_igd = FALSE; /* use igd instead */
	//iso_geo_data A, B;

}

iso_info::~iso_info()
{

}

