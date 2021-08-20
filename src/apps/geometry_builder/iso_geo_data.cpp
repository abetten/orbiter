/*
 * iso_geo_data.cpp
 *
 *  Created on: Aug 16, 2021
 *      Author: betten
 */




#include "geo.h"

using namespace std;



#if 0
iso_geo_data::iso_geo_data()
{
	dim_n = 0;
	theX = NULL;

	f_R_allocated = FALSE;
	R = NULL;
	/* R[i] is number of incidences in row i */

	v = 0;
	b = 0;
	V = 0;
	B = 0; /* the size of the larger geometry (B) */
	/* int max_r; */

	/* second derivatives
	 * (a map from the unordered pairs
	 * of points (p) or blocks (b)
	 * into integers,
	 * ij2k() used for indexing) */
	f_use_ddp = FALSE;
	f_use_ddb = FALSE;
	ddp = NULL;
	ddb = NULL;

	/* tdo data: */
	f_tdo = FALSE;
	tdo_m = 0;
	//int tdo_V[MAX_V];
	tdo_n = 0;
	//int tdo_B[MAX_B];

	/* additional coloring of
	 * the points (p) / blocks (b)
	 * (multiple color fields allowed) */
	f_colors = FALSE;
	nb_bcol = 0;
	nb_pcol = 0;
	bcol = NULL;
	pcol = NULL;

	/* current row-permutation (degree V),
	 * used in ISO2;
	 * pv = p^{-1} */
	//cperm p, pv;
	/* current column-permutation (degree B),
	 * qv = q^{-1} */
	//cperm q, qv;

	f_transpose_it = FALSE;

	//int hbar[MAX_VB];
	//int hlen[MAX_VB];
	//int hlen01[MAX_VB];
	//int hlen1[MAX_VB];
	//int grid_entry[MAX_VB];
	G_max = 0;

}

iso_geo_data::~iso_geo_data()
{
}
#endif


