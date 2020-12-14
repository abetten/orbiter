MY_PATH=~/DEV.20/GITHUB/orbiter
#MY_PATH=../../..
#MY_PATH=/scratch/betten/COMPILE/orbiter

GINAC_PATH=$(MY_PATH)/src/apps/ginac
ORBITER_PATH=$(MY_PATH)/src/apps/orbiter

update:
	cd $(ORBITER_PATH); make clean;
	cd $(MY_PATH); make cleana; git pull; make 


update_all:
	cd $(MY_PATH); make clean; git pull; make 


test:
	$(ORBITER_PATH)/orbiter.out


#############################################################################################################
# Section 4: Orbiter Session
	
	
ORBITER_SESSION:
	
test:
	$(ORBITER_PATH)/orbiter.out

		

#############################################################################################################
# Section 5: Set Builder

set_builder_examples:
	$(ORBITER_PATH)/orbiter.out -v 2 -long_code 64 7 \
		-set_builder -loop 0 64 1 -end \
		-set_builder -loop 0 32 1 -affine_function 2 1 -end \
		-set_builder -loop 0 16 1 -affine_function 4 2 -clone_with_affine_function 4 3 -end \
		-set_builder -set_builder -set_builder -loop 0 4 1 -affine_function 1 4 \
			-clone_with_affine_function 1 12 -end -clone_with_affine_function 1 16 \
			-end -clone_with_affine_function 1 32 -end\
		-set_builder -set_builder -loop 0 8 1 -affine_function 1 8 \
			-clone_with_affine_function 1 24 -end -clone_with_affine_function 1 32 -end\
		-set_builder -loop 0 16 1 -affine_function 1 16 -clone_with_affine_function 1 48 -end \
		-set_builder -loop 0 32 1 -affine_function 1 32 -end 



#############################################################################################################
# Section 6: Basic Number Theory

BASIC_NUMBER_THEORY:



PR2:
		$(ORBITER_PATH)/orbiter.out -v 5 -primitive_root 915839

#a primitive root modulo 915839 is 43085

PM2:
		$(ORBITER_PATH)/orbiter.out -v 5 -power_mod 43085 49842 915839

#the power of 43085 to the 49842 mod 915839 is 487320
	
DL2:
		$(ORBITER_PATH)/orbiter.out -v 5 -discrete_log 487320 43085 915839


#The discrete log is 49842 since 487320 = 43085^49842 mod 915839, time: 0:22




IM:
	$(ORBITER_PATH)/orbiter.out -v 5 -inverse_mod 1865025205 2147483647

IM_gcd:
	$(ORBITER_PATH)/orbiter.out -v 5 -extended_gcd 1865025205 2147483647


PM3a:
		$(ORBITER_PATH)/orbiter.out -v 5 -power_mod 16807 1073741823 2147483647

sqrt_mod:
		$(ORBITER_PATH)/orbiter.out -v 2 -square_root_mod 33 41


draw_mod_13:
		$(ORBITER_PATH)/orbiter.out -v 2 -draw_options -embedded -end \
			-draw_mod_n 13 mod_13 -draw_mod_n_inverse
			pdflatex mod_13_draw.tex
			open mod_13_draw.pdf


draw_mod_13_power_2:
		$(ORBITER_PATH)/orbiter.out -v 2 -draw_options -scale 0.1  -end \
			-draw_mod_n 13 mod_13_powers_of_2 -draw_mod_n_power_cycle 2 -end
			#pdflatex mod_13_draw.tex
			#open mod_13_draw.pdf

draw_mod_13_power_3:
		$(ORBITER_PATH)/orbiter.out -v 2 -draw_options -scale 0.1  -end \
			-draw_mod_n 13 mod_13_powers_of_3 -draw_mod_n_power_cycle 3 -end
			#pdflatex mod_13_draw.tex
			#open mod_13_draw.pdf

draw_mod_13_power_4:
		$(ORBITER_PATH)/orbiter.out -v 2 -draw_options -scale 0.1  -end \
			-draw_mod_n 13 mod_13_powers_of_4 -draw_mod_n_power_cycle 4 -end
			#pdflatex mod_13_draw.tex
			#open mod_13_draw.pdf

draw_mod_13_power_5:
		$(ORBITER_PATH)/orbiter.out -v 2 -draw_options -scale 0.1  -end \
			-draw_mod_n 13 mod_13_powers_of_5 -draw_mod_n_power_cycle 5 -end
			#pdflatex mod_13_draw.tex
			#open mod_13_draw.pdf

draw_mod_13_power_6:
		$(ORBITER_PATH)/orbiter.out -v 2 -draw_options -scale 0.1  -end \
			-draw_mod_n 13 mod_13_powers_of_6 -draw_mod_n_power_cycle 6 -end


PR127:
		$(ORBITER_PATH)/orbiter.out -v 5 -primitive_root 127


draw_mod_127_power:
		$(ORBITER_PATH)/orbiter.out -v 2 -draw_options -scale 0.4 -embedded -end \
			-draw_mod_n 127 mod_127 -draw_mod_n_power_cycle 3 
		pdflatex mod_127_draw.tex
		open mod_127_draw.pdf



#############################################################################################################
# Section 7: Prime Fields

PRIME_FIELDS:
	

F_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_2.tex
	open GF_2.pdf


F_3:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 3 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q3_addition_table.csv 40 24 -draw_matrix_partition 4 3 3
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q3_multiplication_table.csv 40 24 -draw_matrix_partition 4 2 2
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q3_addition_table_reordered.csv 40 24 -draw_matrix_partition 3 3 3
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q3_multiplication_table_reordered.csv 40 24 -draw_matrix_partition 3 2 2
	pdflatex GF_3.tex
	open GF_3.pdf


F_7:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 7 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q7_addition_table.csv 40 24 -draw_matrix_partition 3 7 7 
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q7_multiplication_table.csv 40 24 -draw_matrix_partition 3 6 6
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q7_addition_table_reordered.csv 40 24 -draw_matrix_partition 3 7 7 
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q7_multiplication_table_reordered.csv 40 24 -draw_matrix_partition 3 6 6
	pdflatex GF_7.tex
	open GF_7.pdf

F_13:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 13 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_13.tex
	open GF_13.pdf




F_17:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 17 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q17_addition_table.csv 40 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q17_multiplication_table.csv 40 8
	pdflatex GF_17.tex
	open GF_17.pdf






F_19:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 19 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_19.tex
	open GF_19.pdf



#############################################################################################################
# Section 8: Polynomials over Finite Fields

POLYNOMIALS:





poly_division:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 2 \
		-polynomial_division "1,0,0,0,0,0,0,0,0,0,1" "1,0,1,1" -end
	
poly_gcd:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 2  \
		-extended_gcd_for_polynomials "1,0,0,0,0,0,0,0,0,0,1" "1,0,1,1" -end

poly_mult_mod1:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 7 \
		-polynomial_mult_mod "1,2,3" "3,4,5" "6,0,0,1" -end

poly_mult_mod2:
	$(ORBITER_PATH)/orbiter.out -v 2  -finite_field_activity -q 7 \
		-polynomial_mult_mod "3,1,2" "5,3,4" "6,0,0,1" -end

Berlekamp_matrix_2_3:
	$(ORBITER_PATH)/orbiter.out -v 2  -finite_field_activity -q 2 \
		-Berlekamp_matrix  "1,1,0,1" -end

# the polynomial X^3+X+1 is irreducible over GF(2) because the rank of the Berlekamp matrix is 2.

Berlekamp_matrix_2_4:
	$(ORBITER_PATH)/orbiter.out -v 2  -finite_field_activity -q 2 \
		-Berlekamp_matrix "1,1,0,0,1" -end

# the polynomial X^4+X+1 is irreducible over GF(2) because the rank of the Berlekamp matrix is 3.


Berlekamp_matrix_4_3a:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 4  \
		-Berlekamp_matrix  "1,3,0,1" -end

Berlekamp_matrix_4_3b:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 4  \
		-Berlekamp_matrix "1,3,1,1" -end




find_roots_a:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 19 \
		-polynomial_find_roots "18,1,1" -end

find_roots_b:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 19 \
		-polynomial_find_roots "1,3,1" -end

find_roots_c:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 19 \
		-polynomial_find_roots "1,16,1" -end

find_roots_d:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 19 \
		-polynomial_find_roots "1,18,1" -end

find_roots_e:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 19 \
		-polynomial_find_roots "1,16,3" -end



# degree and then order of the field of coefficients:

irred_4_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -finite_field_activity -q 2 -make_table_of_irreducible_polynomials 4 -end

# 3 polys

irred_5_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -finite_field_activity -q 2 -make_table_of_irreducible_polynomials 5 -end

# 6 polys

irred_6_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -finite_field_activity -q 2 -make_table_of_irreducible_polynomials 6 -end

# 9 polys

irred_7_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -finite_field_activity -q 2 -make_table_of_irreducible_polynomials 7 -end

# 18 polys

irred_8_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -finite_field_activity -q 2 -make_table_of_irreducible_polynomials 8 -end

# 30 polys

irred_9_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -finite_field_activity -q 2 -make_table_of_irreducible_polynomials 9 -end

# 56 polys

irred_10_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -finite_field_activity -q 2 -make_table_of_irreducible_polynomials 10 -end

# 99 polys

irred_2_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -finite_field_activity -q 4 -make_table_of_irreducible_polynomials 2 -end

# 6 polys


irred_3_4:
	$(ORBITER_PATH)/orbiter.out -v 6 -finite_field_activity -q 4 \
		-make_table_of_irreducible_polynomials 3 -end

# 20 polys


search_primitive_poly_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -search_for_primitive_polynomial_in_range 2 2 2 10

# stuck in factoring 2^61-1

search_primitive_poly_3:
	$(ORBITER_PATH)/orbiter.out -v 6 -search_for_primitive_polynomial_in_range 3 3 2 60
	

search_primitive_poly_4:
	$(ORBITER_PATH)/orbiter.out -v 6 -search_for_primitive_polynomial_in_range 4 4 2 30
	
search_primitive_poly_5:
	$(ORBITER_PATH)/orbiter.out -v 6 -search_for_primitive_polynomial_in_range 5 5 2 30
	

search_primitive_poly_7:
	$(ORBITER_PATH)/orbiter.out -v 6 -search_for_primitive_polynomial_in_range 7 7 2 20
	

search_primitive_poly_8:
	$(ORBITER_PATH)/orbiter.out -v 6 -search_for_primitive_polynomial_in_range 8 8 2 20
	
search_primitive_poly_9:
	$(ORBITER_PATH)/orbiter.out -v 6 -search_for_primitive_polynomial_in_range 9 9 2 15
	
search_primitive_poly_11:
	$(ORBITER_PATH)/orbiter.out -v 6 -search_for_primitive_polynomial_in_range 11 11 2 15
	
search_primitive_poly_13:
	$(ORBITER_PATH)/orbiter.out -v 6 -search_for_primitive_polynomial_in_range 13 13 2 15
	
search_primitive_poly_16:
	$(ORBITER_PATH)/orbiter.out -v 6 -search_for_primitive_polynomial_in_range 16 16 2 20
	
search_primitive_poly_32:
	$(ORBITER_PATH)/orbiter.out -v 6 -search_for_primitive_polynomial_in_range 32 32 2 10
	


poly_mult_mod_F4:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 2 \
		-polynomial_mult_mod "1,1" "1,1" "1,1,1" -end
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 2 \
		-polynomial_mult_mod "0,1" "1,1" "1,1,1" -end
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 2 \
		-polynomial_mult_mod "0,1" "0,1" "1,1,1" -end


CRC_poly_3_128_10:
	$(ORBITER_PATH)/orbiter.out -v 1 -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 128 10 -end



#############################################################################################################
# Section 9: Extension Fields

EXTENSION_FIELDS:
	



F_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 4 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q4_addition_table.csv 40 24 -draw_matrix_partition 3 4 4 
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q4_multiplication_table.csv 40 24 -draw_matrix_partition 3 3 3
	pdflatex GF_4.tex
	open GF_4.pdf



F_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 8 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_8.tex
	open GF_8.pdf



F_9:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 9 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q9_addition_table.csv 40 24 -draw_matrix_partition 4 "3,6" "3,6"
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q9_multiplication_table.csv 40 24 -draw_matrix_partition 4 "2,6" "2,6"
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q9_addition_table_reordered.csv 40 24 -draw_matrix_partition 3 9 9
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q9_multiplication_table_reordered.csv 40 24 -draw_matrix_partition 3 8 8
	pdflatex GF_9.tex
	open GF_9.pdf





F_16:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 16 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q16_addition_table.csv 40 24 -draw_matrix_partition 3 16 16 
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q16_multiplication_table.csv 40 24 -draw_matrix_partition 3 15 15
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q16_addition_table_reordered.csv 40 24 -draw_matrix_partition 3 16 16 
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q16_multiplication_table_reordered.csv 40 24 -draw_matrix_partition 3 15 15
	pdflatex GF_16.tex
	open GF_16.pdf



F_25:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 25 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q25_addition_table.csv 40 24 -draw_matrix_partition 3 25 25
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q25_multiplication_table.csv 40 24 -draw_matrix_partition 3 24 24
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q25_addition_table_reordered.csv 40 24 -draw_matrix_partition 3 25 25
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q25_multiplication_table_reordered.csv 40 24 -draw_matrix_partition 3 24 24



F_32:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 32 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_32.tex
	open GF_32.pdf



F_49:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 49 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q49_addition_table.csv 40 24  -draw_matrix_partition 3 49 49
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q49_multiplication_table.csv 40 24 -draw_matrix_partition 3 48 48
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q49_addition_table_reordered.csv 40 24 -draw_matrix_partition 3 49 49
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q49_multiplication_table_reordered.csv 40 24 -draw_matrix_partition 3 48 48
	#pdflatex GF_49.tex
	#open GF_49.pdf



F_64:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 64 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q64_addition_table.csv 40 24  -draw_matrix_partition 3 64 64
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q64_multiplication_table.csv 40 24 -draw_matrix_partition 3 63 63
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q64_addition_table_reordered.csv 40 24 -draw_matrix_partition 3 64 64
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q64_multiplication_table_reordered.csv 40 24 -draw_matrix_partition 3 63 63
	pdflatex GF_64.tex
	open GF_64.pdf


F_125:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 125 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_125.tex
	open GF_125.pdf



F_256:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 256 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_256.tex
	open GF_256.pdf


F_289:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 289 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_289.tex
	open GF_289.pdf



normal_basis_2_3:
	$(ORBITER_PATH)/orbiter.out -v 2 -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity \
		-normal_basis 3 -end



#############################################################################################################
# Section 10: Linear Algebra over Finite Fields


LINEAR_ALGEBRA:


RREF:
	$(ORBITER_PATH)/orbiter.out -v 2 -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity \
		-RREF 2 5 "1,1,1,1,0,1,1,0,0,1" -normalize_from_the_right \
		-end

	
	
RREF_old:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 2 \
		-RREF 2 5 "1,1,1,1,0,1,1,0,0,1" -normalize_from_the_right \
		-end

nullspace:
	$(ORBITER_PATH)/orbiter.out -v 2 -define F2 -finite_field -q 2 -end -end \
		-with F2 -do -finite_field_activity  \
		-nullspace 2 5 "1,1,1,1,0,1,1,0,0,1" \
		-normalize_from_the_right \
		-end

eigen:
	$(ORBITER_PATH)/orbiter.out -v 6 -finite_field_activity -q 5 \
		-eigenstuff 4 "0,1,0,2,0,1,2,1,4,2,3,1,2,0,4,3" \
		-end

RNF_GL_3_2:
	$(ORBITER_PATH)/orbiter.out -v 7 -finite_field_activity -q 2 \
		-all_rational_normal_forms 3 -end
	pdflatex Class_reps_GL_3_2.tex
	open Class_reps_GL_3_2.pdf


RNF_GL_4_2:
	$(ORBITER_PATH)/orbiter.out -v 7 -finite_field_activity -q 2 \
		-all_rational_normal_forms 4 -end
	pdflatex Class_reps_GL_4_2.tex
	open Class_reps_GL_4_2.pdf


#############################################################################################################
# Section 11: Representation Theory

REPRESENTATION_THEORY:


Char_Sym_4:
	$(ORBITER_PATH)/orbiter.out -v 2 -character_table_symmetric_group 4
	
Char_Sym_5:
	$(ORBITER_PATH)/orbiter.out -v 2 -character_table_symmetric_group 5
	
Char_Sym_6:
	$(ORBITER_PATH)/orbiter.out -v 2 -character_table_symmetric_group 6
	


representation_on_polynomials_of_degree_3:
	$(ORBITER_PATH)/orbiter.out -v 4 -define G -linear_group -PGL 4 3 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-representation_on_polynomials 3 \
		-end 


representation_on_polynomials_of_degree_3_old:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-linear_group -PGL 4 3 \
		-end \
		-group_theoretic_activities \
		-representation_on_polynomials 3 \
		-end 
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGL_4_3_rep_3_0.csv 20 3 -draw_matrix_partition 3 20 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGL_4_3_rep_3_1.csv 20 3 -draw_matrix_partition 3 20 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGL_4_3_rep_3_2.csv 20 3 -draw_matrix_partition 3 20 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGL_4_3_rep_3_3.csv 20 3 -draw_matrix_partition 3 20 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGL_4_3_rep_3_4.csv 20 3 -draw_matrix_partition 3 20 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGL_4_3_rep_3_5.csv 20 3 -draw_matrix_partition 3 20 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGL_4_3_rep_3_6.csv 20 3 -draw_matrix_partition 3 20 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGL_4_3_rep_3_7.csv 20 3 -draw_matrix_partition 3 20 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGL_4_3_rep_3_8.csv 20 3 -draw_matrix_partition 3 20 20
	open PGL_4_3_rep_3_0_draw.bmp
	open PGL_4_3_rep_3_1_draw.bmp
	open PGL_4_3_rep_3_2_draw.bmp
	open PGL_4_3_rep_3_3_draw.bmp
	open PGL_4_3_rep_3_4_draw.bmp
	open PGL_4_3_rep_3_5_draw.bmp
	open PGL_4_3_rep_3_6_draw.bmp
	open PGL_4_3_rep_3_7_draw.bmp
	open PGL_4_3_rep_3_8_draw.bmp

# write PGL_4_3_rep_3_0.csv, ... PGL_4_3_rep_3_8.csv


representation_tetrahedral_group_on_polynomials_of_degree_3:
	$(ORBITER_PATH)/orbiter.out -v 4 -linear_group -GL 3 3 \
		-subgroup_by_generators "tetra" "12" 2 \
		"0,1,0,0,0,1,1,0,0" \
		"0,0,1,2,0,0,0,2,0" \
		-end \
		-group_theoretic_activities \
		-representation_on_polynomials 3 \
		-end 
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GL_3_3_Subgroup_tetra_12_rep_3_0.csv 20 3 -draw_matrix_partition 3 10 10
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GL_3_3_Subgroup_tetra_12_rep_3_1.csv 20 3 -draw_matrix_partition 3 10 10
	open GL_3_3_Subgroup_tetra_12_rep_3_0_draw.bmp
	open GL_3_3_Subgroup_tetra_12_rep_3_1_draw.bmp

# write GL_3_3_Subgroup_tetra_12_rep_3_0.csv


#############################################################################################################
# Section 12: Finite Projective Spaces

FINITE_PROJECTIVE_SPACES:

PG_1_16:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 16 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 1 -end
	pdflatex PG_1_16.tex 
	open PG_1_16.pdf




PG_3_2:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 3 -end
	pdflatex PG_3_2.tex 
	open PG_3_2.pdf

PG_3_4:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 4 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 3 -end
	pdflatex PG_3_4.tex 
	open PG_3_4.pdf


PG_3_5:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 5 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 3 -end
	pdflatex PG_3_5.tex 
	open PG_3_5.pdf


PG_3_8:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 8 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 3 -end
	pdflatex PG_3_8.tex 
	open PG_3_8.pdf


PG_3_16:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 16 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 3 -end
	pdflatex PG_3_16.tex 
	open PG_3_16.pdf


PG_4_3:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 3 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 4 -end
	pdflatex PG_4_3.tex 
	open PG_4_3.pdf





PG_4_3:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 3 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 4 -end
	pdflatex PG_4_3.tex 
	open PG_4_3.pdf



#############################################################################################################
# Section 13: Projective Planes


PROJECTIVE_PLANES:


PG_2_4:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 4 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 2 -end
	pdflatex PG_2_4.tex 
	open PG_2_4.pdf


PG_2_4_with_decomposition:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 4 -end -end \
		-with F -do -finite_field_activity \
		-decomposition_by_element 2 1 "0,1,0, 0,0,1, 2,1,1, 0" "PG_2_4_singer" -end
	pdflatex PG_2_4.tex 
	open PG_2_4.pdf

#PG_2_4_singer_incma_transitive.csv


PG_2_4_sub3:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 4 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 2  \
		-decomposition_by_element 3 "0,1,0, 0,0,1, 2,1,1, 0" "PG_2_4_singer" -end
	pdflatex PG_2_4.tex 
	open PG_2_4.pdf

PG_2_4_sub7:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 4 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 2  \
		-decomposition_by_element 7 "0,1,0, 0,0,1, 2,1,1, 0" "PG_2_4_singer" -end
	pdflatex PG_2_4.tex 
	open PG_2_4.pdf


PG_2_4_incma_singer:
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PG_2_4_singer_incma_transitive.csv 20 3 \
		-draw_matrix_partition 3 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
	open PG_2_4_singer_incma_transitive_draw.bmp


PG_2_4_incma_singer_sub_3:
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PG_2_4_singer_incma_subgroup_index_3.csv 20 3 \
		-draw_matrix_partition 3 7,7,7 7,7,7
	open PG_2_4_singer_incma_subgroup_index_3_draw.bmp

PG_2_4_incma_singer_sub_7:
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PG_2_4_singer_incma_subgroup_index_7.csv 20 3 \
		-draw_matrix_partition 3 3,3,3,3,3,3,3 3,3,3,3,3,3,3
	open PG_2_4_singer_incma_subgroup_index_7_draw.bmp


PG_2_4_incma:
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PG_n2_q4_incidence_matrix.csv 20 8 \
		-draw_matrix_partition 3 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
	open PG_n2_q4_incidence_matrix_draw.bmp


#PG_n2_q4_incidence_matrix.csv


PG_2_7:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 7 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 2 -end
	pdflatex PG_2_7.tex 
	open PG_2_7.pdf



PG_2_8:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 8 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 2 -end
	pdflatex PG_2_8.tex 
	open PG_2_8.pdf

PG_2_16:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 16 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 2 -end
	pdflatex PG_2_16.tex 
	open PG_2_16.pdf


PG_2_128:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 128 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_PG 2 -end
	pdflatex PG_2_128.tex 
	open PG_2_128.pdf





#############################################################################################################
# Section 14: The Grassmannian


GRASSMANNIAN:

GR_3_2_2:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_Gr 3 2 -end
	pdflatex Gr_3_2_2.tex 
	open Gr_3_2_2.pdf


#############################################################################################################
# Section 15: Algebraic Sets


ALGEBRAIC_SETS:

EC1:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -q 11 -n 2 \
		-projective_variety "EC" 3 "1,0,3,2,10,6,1,7"	-monomial_type_PART -end -save "./"


Frame_complement:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -q 11 -n 2 \
		-intersection_of_zariski_open_sets "frame_complement" 1 4 \
		"1,0" "1,1" "1,2" "1,0,1,1,1,2" -monomial_type_PART -end -save "./"



Hirschfeld4:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -n 3 -q 4 \
		-projective_variety "Hirschfeld_surface_q4" 3 "1,6,1,8,1,11,1,13" \
		-monomial_type_PART -end -save "./"

# creates Hirschfeld_surface_q4.txt

Endrass_7:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -n 3 -q 7 \
		-projective_variety "Endrass_F7" 8 \
		-long_string \
		"6,0,4,4,2,7,5,9,6,20,6,23,1,25,3,30,1,32,3,34,4,56,6,59,1,61,6,66," \
		"2,68,6,70,3,77,2,79,6,83,6,120,2,123,5,125,3,130,1,132,3,134,3,141," \
		"2,143,6,147,3,156" \
		-end_string \
		-monomial_type_LEX -end -save "./"

# we created a set of 33 points, called Endrass_F7.txt



octic_prepare:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -label octic_monomials  \
		-coefficient_matrix 1 4 "1,1,1,1" -RHS "8,8,1" -x_min_global 0 -x_max_global 8
	$(ORBITER_PATH)/orbiter.out -v 3 -diophant_activity -input_file octic_monomials.diophant -solve_mckay 
	sort -r octic_monomials.sol >octic_monomials_sorted.txt

#Found 165 solutions with 210 backtrack steps
# 165=binomial(11,3)



#############################################################################################################
# Section 16: The Klein Quadric and Pluecker coordinates


KLEIN_QUADRIC_AND_PLUECKER_COORDINATES:



GR_4_2_2:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_Gr 4 2 -end
	pdflatex Gr_4_2_2.tex 
	open Gr_4_2_2.pdf


#############################################################################################################
# Section 17: Orthogonal spaces


ORTHOGONAL_SPACES:



Op_4_2:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_orthogonal 1 4 -end
	pdflatex O_1_4_2.tex 
	open O_1_4_2.pdf


Op_6_2:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_orthogonal 1 6 -end
	pdflatex O_1_6_2.tex 
	open O_1_6_2.pdf


#############################################################################################################
# Section 18: Hermitian varieties


HERMITIAN_VARIETIES:



H_2_4:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 4 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_hermitian 2 -end
	pdflatex H_2_4.tex 
	open H_2_4.pdf

H_3_4:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 4 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_hermitian 3 -end
	pdflatex H_3_4.tex 
	open H_3_4.pdf


# The Hirschfeld surface



#############################################################################################################
# Section 19: Linear Groups


LINEAR_GROUPS:

PGL_4_2:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-draw_options -rad 200 -end \
		-define F2 -finite_field -q 2 -end -end \
		-define G -linear_group -PGL 4 F2 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex PGL_4_2_report.tex
	open PGL_4_2_report.pdf


PGL_4_2_old:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-draw_options -rad 200 -end \
		-define G -linear_group -PGL 4 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end





PGL_4_2_wd:
	$(ORBITER_PATH)/orbiter.out -v 12 \
		-draw_options -rad 200 -end \
		-define G -linear_group -PGL 4 2 -wedge_detached -end -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex PGL_4_2_Wedge_4_0_detached_report.tex
	open PGL_4_2_Wedge_4_0_detached_report.pdf


PGL_4_2_wd_ideal:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-process_combinatorial_objects -n 5 -q 2 \
			-input \
			-set_of_points \
				-long_string \
				"0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 18, 21, " \
				"22, 25, 26, 31, 32, 34, 36, 38, 40, 42, 45, 47, 48, " \
				"51, 52, 55, 56, 59, 61, 62" \
				-end_string \
			-end \
		-ideal_PART 2 \
		-fname_base_out "form" \
		-end
	#pdflatex form.tex
	#open form.pdf

#invariant form:
#X_0X_5 + X_1X_4 + X_2X_3


PGL_4_2_wd_reverse:
	$(ORBITER_PATH)/orbiter.out -v 12 -linear_group -PGL 4 2 -wedge_detached -end \
		-group_theoretic_activities \
		-reverse_isomorphism_exterior_square \
		-end

	
PGGL_3_4:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 3 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-sylow -classes \
		-end
	pdflatex PGGL_3_4_report.tex
	open PGGL_3_4_report.pdf

#-sylow  
#		-export_schreier_trees \
#		-tools_path $(GRAPH_THEORY_PATH) \
#		-report \
#		-report_schreier_trees

J1:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 7 11 -Janko1 -end -end \
		-with G -do \
		-group_theoretic_activities -report \
		-end
	pdflatex PGL_7_11_Subgroup_Janko1_report.tex
	open PGL_7_11_Subgroup_Janko1_report.pdf

PGL_3_11_singer:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 3 11 -singer_and_frobenius 19 -end -end \
		-with G -do \
		-group_theoretic_activities -report \
		-end
		pdflatex PGL_3_11_Singer_and_Frob3_11_19_report.tex
		open PGL_3_11_Singer_and_Frob3_11_19_report.pdf

PG_2_4_order_21:
	$(ORBITER_PATH)/orbiter.out -v 5 -define G -linear_group -PGL 3 4  -end -end \
		-with G -do \
		-group_theoretic_activities -search_element_of_order 21




quaternion:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -SL 2 3 \
		-subgroup_by_generators "quaternion" "8" 3 \
		"1,1,1,2" \
		"2,1,1,1" \
		"0,2,1,0" \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-print_elements_tex \
		-group_table \
		-report \
		-end
	pdflatex GL_2_3_Subgroup_quaternion_8_elements.tex
	open GL_2_3_Subgroup_quaternion_8_elements.pdf



#############################################################################################################
# Section 20:  Linear Groups, Advanced Topics


LINEAR_GROUPS_ADVANCED_TOPICS:


U_3_2:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 9 -override_polynomial "17"  -end -end \
		-define G -linear_group -PGL 3 F \
		-subgroup_by_generators "U_3_3" "6048" 2 \
		"1,6,4, 5,0,6, 8,5,1" \
		"6,2,1, 7,8,4, 0,6,6" \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-report -end
	pdflatex PGL_3_9_Subgroup_U_3_3_6048_report.tex
	open PGL_3_9_Subgroup_U_3_3_6048_report.pdf


cube_group:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -GL 3 3 \
		-subgroup_by_generators "cube" "48" 3 \
		"0,1,0,2,0,0,0,0,1" \
		"0,0,1,0,1,0,2,0,0" \
		"2,0,0,0,1,0,0,0,1" \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-print_elements_tex \
		-report
	pdflatex GL_3_3_Subgroup_cube_48_report.tex
	open GL_3_3_Subgroup_cube_48_report.pdf
	pdflatex GL_3_3_Subgroup_cube_48_elements.tex
	open GL_3_3_Subgroup_cube_48_elements.pdf


PGL_2_3:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 2 3 -end -end \
		-with G -do \
		-group_theoretic_activities -report \
		-group_table \
		-end
	pdflatex PGL_2_3_group_table_order_24.tex
	pdflatex PGL_2_3_report.tex
	open PGL_2_3_group_table_order_24.pdf
	open PGL_2_3_report.pdf



PG_3_2_again:
	$(ORBITER_PATH)/orbiter.out -finite_field_activity -q 3 -cheat_sheet_PG 2 -end
	pdflatex PG_2_3.tex 
	open PG_2_3.pdf


tetra_group:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -GL 3 3 \
		-subgroup_by_generators "tetra" "12" 2 \
		"0,1,0,0,0,1,1,0,0" \
		"0,0,1,2,0,0,0,2,0" \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-print_elements_tex \
		-report
	pdflatex GL_3_3_Subgroup_tetra_12_report.tex
	open GL_3_3_Subgroup_tetra_12_report.pdf
	pdflatex GL_3_3_Subgroup_tetra_12_elements.tex
	open GL_3_3_Subgroup_tetra_12_elements.pdf

Hesse_group:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 3 4 \
		-subgroup_by_generators "Hesse" "432" 7 \
		"3,0,0,0,3,0,0,0,3,0" \
		"2,0,0,0,2,0,1,2,3,0" \
		"1,0,0,0,1,0,0,1,1,1" \
		"1,0,0,0,2,2,0,2,0,0" \
		"1,0,0,2,3,1,2,0,1,0" \
		"0,3,3,1,0,0,3,2,1,1" \
		"2,2,0,0,0,1,1,3,3,1" \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-print_elements_tex \
		-report
	pdflatex PGGL_3_4_Subgroup_Hesse_432_report.tex
	open PGGL_3_4_Subgroup_Hesse_432_report.pdf

#Hesse group:
#1,0,0,0,0,1,0,0,0,0,1,0,3,2,3,2,0, 
#1,0,0,0,0,1,0,0,3,1,2,0,1,0,1,3,0, 
#1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1, 
#1,0,0,0,0,2,2,0,0,2,0,0,0,0,0,1,0, 
#1,0,0,0,2,3,1,0,2,0,1,0,3,1,3,1,0, 
#0,1,1,0,2,0,0,0,1,3,2,0,2,1,1,2,1, 
#1,1,0,0,0,0,3,0,3,2,2,0,1,2,3,3,1,







#Co3 from Conway et al., 1985 (ATLAS)
#order = 495766656000
#Co3 from the paper by Suleiman and Wilson 1997

#1101110001000001010000
#1111010111110100001011
#0000001000000100010101
#1111100110110001001110
#0101010000000010011101
#0000010000000100010101
#0010000000000100010101
#0001000011000000111111
#1110100100110100010011
#0000000000000110010101
#0000000000100100010101
#0110111111010011101111
#0000000000001100010101
#0000000000000100000101
#0000000001000100010101
#0000000000000100011101
#0001000110000010011010
#0000000000000000010101
#0000000000000101010101
#0000000000000100010100
#0000000000000100010111
#0000000000000100010001


#0101000010111010111111
#0110010100011110110000
#0011010000111111010111
#0001101110001011010011
#1010010000100001011110
#1101000000001010100011
#1100101010001111010101
#1000110100110101010101
#0100110001010000000111
#1100000010100101010010
#0101110110011100000101
#0101111101010011111001
#1000010101010101010001
#0001010000111100100111
#0011010010111011001111
#0100110010110011111010
#1101011001111101100011
#0100101001001000100001
#1100101100001001110011
#0101110110010100000001
#0000001101111000101110
#1101101010101110000101

Co3:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-draw_options -nodes_empty -end \
		-define G -linear_group -PGL 22 2 \
		-subgroup_by_generators "Co3" "495766656000" 2 \
			-long_string \
			"1,1,0,1,1,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0, " \
			"1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,0,0,0,1,0,1,1, " \
			"0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1, " \
			"1,1,1,1,1,0,0,1,1,0,1,1,0,0,0,1,0,0,1,1,1,0, " \
			"0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1, " \
			"0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1, " \
			"0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1, " \
			"0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1, " \
			"1,1,1,0,1,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,0,1, " \
			"0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,1, " \
			"0,1,1,0,1,1,1,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1, " \
			"0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,1, " \
			"0,0,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,1,1,0,1,0, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1," \
			-end_string \
			-long_string \
			"0,1,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1, " \
			"0,1,1,0,0,1,0,1,0,0,0,1,1,1,1,0,1,1,0,0,0,0, " \
			"0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,0,1,1,1, " \
			"0,0,0,1,1,0,1,1,1,0,0,0,1,0,1,1,0,1,0,0,1,1, " \
			"1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0, " \
			"1,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,1, " \
			"1,1,0,0,1,0,1,0,1,0,0,0,1,1,1,1,0,1,0,1,0,1, " \
			"1,0,0,0,1,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1, " \
			"0,1,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,1, " \
			"1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,1,0, " \
			"0,1,0,1,1,1,0,1,1,0,0,1,1,1,0,0,0,0,0,1,0,1, " \
			"0,1,0,1,1,1,1,1,0,1,0,1,0,0,1,1,1,1,1,0,0,1, " \
			"1,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1, " \
			"0,0,0,1,0,1,0,0,0,0,1,1,1,1,0,0,1,0,0,1,1,1, " \
			"0,0,1,1,0,1,0,0,1,0,1,1,1,0,1,1,0,0,1,1,1,1, " \
			"0,1,0,0,1,1,0,0,1,0,1,1,0,0,1,1,1,1,1,0,1,0, " \
			"1,1,0,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,0,0,1,1, " \
			"0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1, " \
			"1,1,0,0,1,0,1,1,0,0,0,0,1,0,0,1,1,1,0,0,1,1, " \
			"0,1,0,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,0,0,0,1, " \
			"0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,0,1,0,1,1,1,0, " \
			"1,1,0,1,1,0,1,0,1,0,1,0,1,1,1,0,0,0,0,1,0,1," \
			-end_string \
			-end -end \
		-with G -do \
		-group_theoretic_activities \
		-report
	pdflatex PGL_22_2_Subgroup_Co3_495766656000_report.tex
	open PGL_22_2_Subgroup_Co3_495766656000_report.pdf



Ree_27:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-draw_options -end \
		-define F -finite_field -q 27 -override_polynomial "34" -end -end \
		-define G -linear_group -PGL 7 F \
			-subgroup_by_generators "Ree_27" "10073444472" 2 \
			-long_string \
			"21,5,1,6,17,1,1, 3,13,5,21,6,6,18, 21,3,21,21,22,6,14, " \
			"14,18,1,5,13,6,7, 3,3,2,1,24,16,3, 17,3,22,10,16,24,26, " \
			"21,21,6,18,20,2,5" \
			-end_string \
			-long_string \
			"16,3,11,5,16,22,20, 24,6,18,24,7,1,26, 9,23,17,18,23,20,13, " \
			"9,7,2,15,17,5,11, 3,3,6,21,4,24,16, 25,8,6,24,21,12,7, " \
			"24,15,2,13,11,14,24" \
			-end_string \
			-end -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end


#############################################################################################################
# Section 21:  Induced Actions


INDUCED_ACTIONS:



T4:
	$(ORBITER_PATH)/orbiter.out -v 4 -define G \
		-linear_group -GL_d_q_wr_Sym_n 2 2 4 -on_tensors -end -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_points
	pdflatex GL_2_2_wreath_Sym4_res65535_orbits.tex
	open GL_2_2_wreath_Sym4_res65535_orbits.pdf

T4r1:
	$(ORBITER_PATH)/orbiter.out -v 4 -define G \
		-linear_group -GL_d_q_wr_Sym_n 2 2 4 -on_rank_one_tensors -end -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_points
	pdflatex GL_2_2_wreath_Sym4_res81_orbits.tex
	open GL_2_2_wreath_Sym4_res81_orbits.pdf

T4r1_orbits:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -GL_d_q_wr_Sym_n 2 2 4 \
			-on_rank_one_tensors -end -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_subsets 4 \
		-report




#############################################################################################################
# Section 22:  Group Theoretic Activities


GROUP_THEORETIC_ACTIVITIES:

PG_2_4_singer:
	$(ORBITER_PATH)/orbiter.out -v 5 -define G -linear_group -PGL 3 4 -end -edn \
		-with G -do \
		-group_theoretic_activities -find_singer_cycle




PSL_3_2_classes:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PSL 3 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-classes_based_on_normal_form
	pdflatex PGL_3_2_classes_normal_form.tex
	open PGL_3_2_classes_normal_form.pdf
	#pdflatex PGL_3_2_classes_out.tex
	#open PGL_3_2_classes_out.pdf

#		-classes \



	
			




PGL_3_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGL 3 4 -end

PGL_3_4_singer:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGL 3 4 -singer 1 -end




PGGL_3_4_classes:
	$(ORBITER_PATH)/orbiter.out -v 10 \
		-define G \
		-linear_group -PGGL 3 4 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-classes
	pdflatex PGGL_3_4_classes_out.tex
	open PGGL_3_4_classes_out.pdf


#1,3,3,1,3,2,3,0,3,0,
# is an element of order 21


#		-subgroup_by_generators "singer" "21" 1 \
#		"0,1,0, 0,0,1, 2,1,1" \

normal_forms_PGL_4_4:
	$(ORBITER_PATH)/orbiter.out -v 7 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-classes_based_on_normal_form


classes_PGL_4_4:
	$(ORBITER_PATH)/orbiter.out -v 7 \
		-all_rational_normal_forms 4 4


# 252 classes


classes_PGGL_4_4:
	$(ORBITER_PATH)/orbiter.out \
		-magma_path /usr/local/magma/ \
		-v 6 \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-classes

# group order 1974067200 = 2^13 * 3^4 * 5^2 * 7 * 17

#	pdflatex PGGL_4_4_classes_out.tex
#	open PGGL_4_4_classes_out.pdf


# elements of order 2:
# conjugacy class reps:
# elt order, class size, centralizer order 

#2 48960 40320 Baer involution

#2 5355 368640  one block of 10,11

#2 64260 30720 two blocks of 10,11 (problem group)


CLASS_2A=-centralizer_of_element "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1" -label "2A" 
# Baer involution

CLASS_2B=-centralizer_of_element "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,0,1, 0" -label "2B"

CLASS_2C=-centralizer_of_element "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0" -label "2C"  
# problem group


# 3 classes of elements of order 3
# 4 classes of elements of order 4

PGGL_4_4_cent_2A:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-draw_options -end \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-centralizer_of_element "2A" "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1" \
		-report

#Strong generators for a group of order 40320:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0,\\
#1,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,\\
#1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,0,\\
#1,0,0,0,1,1,1,1,1,0,1,0,1,1,1,0,0,\\
#1,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,0,\\
#0,1,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,\\


PGL_4_4_2A_rank:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-element_rank "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1"
		
PGL_4_4_2A_unrank:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-element_unrank "1"
		
PGL_4_4_2A_class:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-conjugacy_class_of "1"





PGGL_4_4_cent_2B:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-centralizer_of_element "2B" "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,0,1, 0" \
		-report

#Strong generators for a group of order 368640:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,3,1,\\
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,2,0,0,3,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,3,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,3,0,1,3,1,\\
#1,0,0,0,0,1,0,0,1,0,1,0,3,0,0,1,0,\\
#1,0,0,0,0,1,0,0,3,0,1,0,0,0,0,1,0,\\
#1,0,0,0,0,1,0,0,3,0,1,3,0,0,0,1,0,\\
#1,0,0,0,0,1,0,0,3,0,1,3,2,0,0,3,1,\\
#1,0,0,0,0,1,0,0,1,0,0,3,3,0,2,3,0,\\
#1,0,0,0,1,1,0,0,2,0,1,0,2,0,0,1,0,\\
#1,0,0,0,1,1,0,2,2,0,1,3,2,0,0,2,0,\\
#1,0,0,0,2,1,1,2,1,0,2,1,3,0,2,2,0,\\
#1,0,0,0,2,1,0,2,0,0,1,0,2,0,0,3,1,\\

PGGL_4_4_cent_2C:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-centralizer_of_element "2C" "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0"


#Strong generators for a group of order 30720:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,2,0,3,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,3,1,0,\\
#1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1,\\
#1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,\\
#1,0,0,0,2,1,0,0,0,0,1,0,1,0,0,1,0,\\
#1,0,0,0,1,1,2,0,0,0,1,0,0,0,0,1,0,\\
#1,0,3,0,1,1,1,3,0,0,2,0,0,0,0,2,1,\\



PGGL_4_4_norm_4A:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "4A" "1,0,0,0,1,1,0,0, 0,1,1,0, 0,0,1,1, 0"


# normalizer has order 256:
#Strong generators for a group of order 256:\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,3,0,0,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,\\
#1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,\\
#1,0,0,0,0,1,0,0,3,1,1,0,1,2,0,1,0,\\
#1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,1,0,\\
#1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,\\
#1,0,0,0,2,1,0,0,1,3,1,0,2,2,2,1,0,\\



PGGL_4_4_norm_4B:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "4B" "1,0,0,0,0,1,0,0, 0,1,1,0, 0,0,1,1, 0"

#Strong generators for a group of order 3072:\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,2,0,0,0,0,2,0,0,0,0,2,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,2,1,1,0,\\
#1,0,0,0,0,2,0,0,0,2,2,0,0,3,0,2,0,\\
#1,0,0,0,0,3,0,0,0,1,3,0,0,2,1,3,1,\\
#1,2,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,\\




PGGL_4_4_norm_4C:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "4C" "3,0,0,0, 1,3,0,0, 0,0,3,0, 0,0,1,3, 1"



#Strong generators for a group of order 384:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,1,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,\\
#1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1,\\
#1,0,0,0,2,1,0,0,0,0,1,0,1,0,2,1,0,\\
#1,0,0,0,3,1,1,0,0,0,1,0,1,0,2,1,0,\\
#0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0,1,\\



PGGL_4_4_norm_4D:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "4D" "3,0,0,0, 0,3,0,0, 0,0,3,0, 0,0,1,3, 1"


#Strong generators for a group of order 768:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,2,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,2,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,1,2,1,0,\\
#1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0,\\
#1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,\\
#0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,\\



PGGL_4_4_cent_5A:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-centralizer_of_element "5A" "0,2,0,0, 1,1,0,0, 0,0,3,0, 0,0,0,3, 0"

#centralizer has order 900
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,3,0,\\
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,3,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,2,0,0,0,2,0,\\
#0,1,0,0,3,3,0,0,0,0,2,0,0,0,0,2,0,\\

PGGL_4_4_norm_5A:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "5A" "0,2,0,0, 1,1,0,0, 0,0,3,0, 0,0,0,3, 0"

# normalizer has order 3600
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,2,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,\\
#1,0,0,0,0,1,0,0,0,0,0,2,0,0,2,0,0,\\
#1,0,0,0,2,2,0,0,0,0,1,0,0,0,0,1,1,\\
#0,1,0,0,3,3,0,0,0,0,2,0,0,0,0,2,0,\\


PGGL_4_4_norm_5B:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "5B" "0,2,0,0,1,1,0,0,0,0,0,2,0,0,1,1,0"

# normalizer has order 81600:
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0,\\
#1,0,0,0,0,1,0,0,2,2,2,2,1,3,1,3,0,\\
#1,0,0,0,0,1,0,0,0,0,1,1,0,0,3,2,0,\\
#1,0,0,0,2,2,0,0,0,0,3,0,0,0,1,1,1,\\
#0,1,0,0,3,3,0,0,0,0,0,1,0,0,3,3,0,\\
#0,0,1,0,0,0,1,2,2,0,0,0,2,3,0,0,1,\\

PGGL_4_4_norm_5C:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "5C" "0,1,0,0,3,3,0,0,0,0,1,2,0,0,2,0,0"


# normalizer has order 600:
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0,\\
#1,0,0,0,0,1,0,0,0,0,1,2,0,0,2,0,0,\\
#1,0,0,0,2,2,0,0,0,0,3,0,0,0,3,1,1,\\
#0,1,0,0,3,3,0,0,0,0,1,2,0,0,2,0,0,\\
#0,0,1,0,0,0,3,2,1,0,0,0,3,2,0,0,1,\\


subgroups_PGGL_4_4:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-find_subgroup 4 
	pdflatex PGGL_4_4_report.tex
	open PGGL_4_4_report.pdf

no:
	pdflatex PGGL_4_4_classes_out.tex
	open PGGL_4_4_classes_out.pdf





no:
	pdflatex PGGL_4_4_classes_out.tex
	open PGGL_4_4_classes_out.pdf



cent_2A:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 \
		-subgroup_by_generators "centralizer_2A" "40320" 10 \
			"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0," \
			"1,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0," \
			"1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,0," \
			"1,0,0,0,1,1,1,1,1,0,1,0,1,1,1,0,0," \
			"1,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,0," \
			"0,1,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1," \
		-end



subgroups_PGL_4_5:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGL 4 5 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-find_subgroup 3 
	pdflatex PGL_4_5_report.tex
	open PGL_4_5_report.pdf


classes_PGL_4_5:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGL 4 5 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-classes
	pdflatex PGL_4_5_classes_out.tex
	open PGL_4_5_classes_out.pdf
	
# 163 classes


# two classes of elements of order 3
#Order of element = 3 Class size = 310000 Centralizer order = 93600 Normalizer order = 187200
#of order 3 and with 0 fixed points.
#0,1,0,2,0,1,2,1,4,2,3,1,2,0,4,3,

#Class size = 10075000 Centralizer order = 2880 Normalizer order = 5760
#of order 3 and with 6 fixed points.
#0,0,0,1,2,3,0,1,0,3,4,4,0,1,2,1,

PGL_4_5_3B_rank:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGL 4 5 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-element_rank "0,0,0,1, 2,3,0,1, 0,3,4,4, 0,1,2,1"
		
PGL_4_5_3B_unrank:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGL 4 5 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-element_unrank "701459351"
		
PGL_4_5_3B_class:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGL 4 5 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-conjugacy_class_of "0,0,0,1, 2,3,0,1, 0,3,4,4, 0,1,2,1"
		


eigen_3A:
	$(ORBITER_PATH)/orbiter.out -v 6  -finite_field_activity -q 5 -eigenstuff 4 "0,1,0,2,0,1,2,1,4,2,3,1,2,0,4,3" -end
	
eigen_3B:
	$(ORBITER_PATH)/orbiter.out -v 6  -finite_field_activity -q 5 -eigenstuff 4 "0,0,0,1,2,3,0,1,0,3,4,4,0,1,2,1" -end
	

# element of order 31 in PGL(4,5):
#		int data[] = {1,0,0,0, 0,0,1,0, 0,0,0,1, 0,1,1,3, 0 };


normal_forms_PGL_4_5:
	$(ORBITER_PATH)/orbiter.out -v 7 -define G \
		-linear_group -PGL 4 5 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-classes_based_on_normal_form
	pdflatex PGL_4_5_classes_normal_form.tex
	open PGL_4_5_classes_normal_form.pdf



PGL_4_5_classes:
	$(ORBITER_PATH)/orbiter.out -v 3 -all_rational_normal_forms 4 5
	pdflatex Class_reps_GL_4_5.tex
	open Class_reps_GL_4_5.pdf

PGL_4_5_norm_31:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGL 4 5 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "31" "1,0,0,0, 0,0,1,0, 0,0,0,1, 0,1,1,3"


# normalizer has order 372
#1,0,0,0,0,4,0,0,0,0,4,0,0,0,0,4,\\
#1,0,0,0,0,3,0,0,0,0,3,0,0,0,0,3,\\
#1,0,0,0,0,1,0,0,0,3,1,1,0,4,2,3,\\
#1,0,0,0,0,0,1,0,0,0,0,1,0,1,1,3,\\




#############################################################################################################
# Section 23:  Orbit Algorithms


ORBIT_ALGORITHMS:


orbits_cubic_surfaces_q2:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -yout 500000 -rad 15 -nodes_empty -line_width 0.5 -y_stretch 0.25 -embedded -end \
		-define G -linear_group -PGL 4 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 3 \
		-orbits_on_polynomials_draw_tree 6 \
		-end 
	pdflatex poly_orbits_d3_n3_q2.tex
	open poly_orbits_d3_n3_q2.pdf






#############################################################################################################
# Section 24: Poset Classification


POSET_CLASSIFICATION:
	
poset_of_4subsets:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options  -end \
		-define G -linear_group -PGL 2 3 -identity_group -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label poset_4 -W -depth 4 -draw_options -end  -draw_poset -end \
		-orbits_on_subsets 4 \
		-report \
		-end

poset_of_4subsets_draw:
	$(ORBITER_PATH)/orbiter.out -v 3 -draw_layered_graph poset_4_poset_lvl_4.layered_graph \
		-rad 300 -embedded -line_width 1.1  -y_stretch 0.9 -scale 0.25 \
		-end
	pdflatex poset_4_poset_lvl_4_draw.tex
	open poset_4_poset_lvl_4_draw.pdf


poset_of_5subsets:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options  -end \
		-define G -linear_group -PGL 2 4 -identity_group -end -end \
		-with G -do \
			-group_theoretic_activities \
			-poset_classification_control -problem_label poset_5 -W -depth 5 -draw_options -end  -draw_poset -end \
			-orbits_on_subsets 5 \
			-report \
			-end

poset_of_5subsets_draw:
	$(ORBITER_PATH)/orbiter.out -v 3 -draw_layered_graph poset_5_poset_lvl_5.layered_graph \
		-rad 300 -embedded -line_width 1.1  -y_stretch 0.9 -scale 0.25 \
		-end
	pdflatex poset_5_poset_lvl_5_draw.tex
	open poset_5_poset_lvl_5_draw.pdf



V_3_2_trivial:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGL 3 2 -identity_group -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -node_label_is_element -draw_options -rad 200 -embedded  -end -draw_poset -problem_label V_3_2 -W -depth 3 -end \
		-orbits_on_subspaces 3 \
		-report \
		-end
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_layered_graph PGL_3_2_Identity_3_2_poset_lvl_3.layered_graph \
				-rad 300 -embedded -line_width 1.1  -y_stretch 0.9 -scale 0.25 \
				-end
	#pdflatex PGL_4_2_Identity_4_2_report.tex
	#open PGL_4_2_Identity_4_2_report.pdf
	pdflatex PGL_3_2_Identity_3_2_poset_lvl_3_draw.tex
	open PGL_3_2_Identity_3_2_poset_lvl_3_draw.pdf




V_4_2_trivial:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGL 4 2 -identity_group -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -node_label_is_element -draw_options -embedded  -end -draw_poset -problem_label V_4_2 -W -depth 4 -end \
		-orbits_on_subspaces 4 \
		-report 
	#pdflatex PGL_4_2_Identity_4_2_report.tex
	#open PGL_4_2_Identity_4_2_report.pdf
	pdflatex PGL_4_2_Identity_4_2_tree_lvl_4.tex
	open PGL_4_2_Identity_4_2_tree_lvl_4.pdf




#############################################################################################################
# Section 25: Orbits on Subsets


ORBITS_ON_SUBSETS:

PG_2_2_subsets:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 3 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label PGL_3_2 \
			-depth 7 -draw_poset -draw_options -rad 200 -embedded -end -end \
			-orbits_on_subsets 7 \
			-report \
		-end
	pdflatex PGL_3_2_poset_lvl_7.tex
	open PGL_3_2_poset_lvl_7.pdf
	#pdflatex PGL_3_2_poset_detailed_lvl_7.tex
	#open PGL_3_2_poset_detailed_lvl_7.pdf


# PG(3,2) has 2^3+2^2+2^1+1 = 15 points:
# PG(3,3) has 3^3+3^2+3^1+1 = 27 + 9 + 3 + 1 = 40 points.


PG_3_2_subsets:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define F2 -finite_field -q 2 -end -end \
		-define G -linear_group -PGL 4 F2 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label PGL_4_2 \
			-depth 15 -draw_poset -draw_options -rad 200 -embedded -end -end \
			-orbits_on_subsets 15 \
			-report \
		-end
	pdflatex PGL_4_2_poset_lvl_15.tex
	open PGL_4_2_poset_lvl_15.pdf
	#pdflatex PGL_4_2_poset_detailed_lvl_15.tex
	#open PGL_4_2_poset_detailed_lvl_15.pdf


PG_3_3_subsets:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 4 3 -end -end \
		-with G -do \
			-group_theoretic_activities \
			-poset_classification_control -problem_label PGL_4_3 \
			-depth 40 -draw_poset -draw_options -rad 200 -embedded -end -end \
			-orbits_on_subsets 40 \
			-draw_poset \
			-report \
		-end
	pdflatex PGL_4_3_poset_lvl_40.tex
	open PGL_4_3_poset_lvl_40.pdf



	

orb:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -GL 2 2 -singer 1 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label GL_2_2_singer -W -depth 4 -end \
			-orbits_on_subsets 4 \
			-draw_poset \
			-report \
		-end
	pdflatex GL_2_2_Singer_2_2_1_report.tex
	open GL_2_2_Singer_2_2_1_report.pdf


orb16:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 2 16 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label PGGL_2_16 -W -depth 4 -end \
			-orbits_on_subsets 10 \
			-report \
		-end
	pdflatex PGGL_2_16_report.tex
	open PGGL_2_16_report.pdf







PG_3_4_subsets:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label PGGL_4_4 \
			-depth 5 -draw_poset -draw_options -rad 200 -embedded -end -end \
			-orbits_on_subsets 5 \
			-draw_poset \
			-report \
		-end





#############################################################################################################
# Section 26: Orbits on Subspaces


ORBITS_ON_SUBSPACES:
	

subspaces_Op_4_2:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGL 4 2 -orthogonal 1 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -node_label_is_element \
			-draw_poset -draw_options -end -problem_label Op_4_2 -W -depth 4 -end \
		-orbits_on_subspaces 4 \
		-report \
		-end
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_layered_graph PGL_4_2_Orthogonal_plus_4_2_poset_lvl_4.layered_graph \
			-rad 300 -embedded -line_width 1.1  -y_stretch 0.9 -scale 0.25 \
			-end
	pdflatex PGL_4_2_Orthogonal_plus_4_2_poset_lvl_4_draw.tex
	open PGL_4_2_Orthogonal_plus_4_2_poset_lvl_4_draw.pdf


sub:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGL 4 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label PGL_4_2 -W -depth 4 -end \
		-orbits_on_subspaces 4 -\
		-report 
	pdflatex PGL_4_2_report.tex
	open PGL_4_2_report.pdf

sub_4_2_singer:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGL 4 2 -singer 1 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label PGL_4_2_singer -W -depth 4 -end \
		-orbits_on_subspaces 4 -draw_poset \
		-report 
	pdflatex PGL_4_2_Singer_4_2_1_report.tex
	open PGL_4_2_Singer_4_2_1_report.pdf



sub_8_2_singer:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGL 8 2 -singer 1 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label PGL_8_2_singer -W -depth 8 -end \
		-orbits_on_subspaces 8 \
		-report
	pdflatex PGL_8_2_Singer_8_2_1_report.tex
	open PGL_8_2_Singer_8_2_1_report.pdf

# May 7, 2020: 16 sec on Mac


Op_6_2_orbits:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGL 6 2 -orthogonal 1 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label Op_6_2 -W -depth 8 -end \
		-orbits_on_subspaces 3 \
		-report \
		-end
	pdflatex PGL_6_2_Orthogonal_plus_6_2_report.tex
	open PGL_6_2_Orthogonal_plus_6_2_report.pdf



Op_6_3_orbits:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGL 6 3 -orthogonal 1 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label Op_6_3 -W -depth 8 -end \
		-orbits_on_subspaces 3 \
		-report \
		-end 
	pdflatex PGL_6_3_Orthogonal_plus_6_3_report.tex
	open PGL_6_3_Orthogonal_plus_6_3_report.pdf

# June 3, 2020 on Mac: 0 sec



Op_6_11_orbits:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -nodes_empty -end \
		-define G -linear_group -PGL 6 11 -orthogonal 1 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label Op_6_11 -W -depth 8 -end \
		-orbits_on_subspaces 3 \
		-report \
		-end
	pdflatex PGL_6_11_Orthogonal_plus_6_11_report.tex
	open PGL_6_11_Orthogonal_plus_6_11_report.pdf


# June 3, 2020 on Mac: 12 sec




#############################################################################################################
# Section 27: Creating Objects in Projective Geometry

CREATING_OBJECTS_IN_PG:


EC:
	$(ORBITER_PATH)/orbiter.out -v 5 -create_combinatorial_object -q 11 \
		-elliptic_curve 1 3 -end \
		-save "./"


arcs_4_2:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-linear_group -PGGL 3 8 -end \
		-group_theoretic_activities \
				-poset_classification_control -problem_label arcs_q8_d2 -w -depth 4 -end \
			-classify_arcs \
				-target_size 4 \
				-q 8 \
				-n 3 \
				-d 2 \
			-end \
		-end

arcs_4_2_lift:
	$(ORBITER_PATH)/orbiter.out -v 2 -process_combinatorial_objects -q 8 -n 2  \
		-arc_with_given_set_as_s_lines_after_dualizing \
			33 5 0 0 \
		-fname_base_out arcs_q8_d2_lvl_4_lifted \
		-input \
			-file_of_points arcs_q8_d2_lvl_4 \
		-end \
		-end 
	mv system_0.solutions arcs_4_2_lift.solutions

# 2 minutes




cuspidal_cubic_q8:
	$(ORBITER_PATH)/orbiter.out -v 5 -create_combinatorial_object -q 8 -n 2 \
		-cuspidal_cubic -end \
		-save "./"



cuspidal_group:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 3 8 \
		-subgroup_by_generators "cuspidal_group" "21" 2 \
		"1,0,0,0,1,0,0,0,1,1" \
		"1,0,0,0,6,0,0,0,7,0" \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-print_elements_tex \
		-report \
		-end
	pdflatex PGGL_3_8_Subgroup_cuspidal_group_21_elements.tex
	open PGGL_3_8_Subgroup_cuspidal_group_21_elements.pdf



cuspidal_group_orbits:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 3 8 \
		-subgroup_by_generators "cuspidal_group" "21" 2 \
		"1,0,0,0,1,0,0,0,1,1" \
		"1,0,0,0,6,0,0,0,7,0" \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label cuspidal_group -W -depth 6 -end \
			-orbits_on_subsets 6 \
			-draw_poset \
			-report \
		-end






cuspidal_cubic_q16:
	$(ORBITER_PATH)/orbiter.out -v 5 -create_combinatorial_object -q 16 -n 2 \
		-cuspidal_cubic -end \
		-save "./"

#cuspidal_cubic_16.txt
#cuspidal_cubic_16_subsets_6.txt




Hirschfeld16:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -n 3  -q 16 \
		-projective_variety "Hirschfeld_surface_q16" 3 "1,6,1,8,1,11,1,13" \
		-end \
		-save ""


#############################################################################################################
# Section 28: Canonical Forms of Objects in Finite Projective Spaces


CANONICAL_FORMS:

EC_canon:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 11 \
		-canonical_form_PG  2  \
		-input -file_of_points elliptic_curve_b1_c3_q11.txt \
		-end \
		-classification_prefix elliptic_curve_b1_c3_q11 \
		-report \
		-end \
		-end
	pdflatex elliptic_curve_b1_c3_q11_classification.tex
	open elliptic_curve_b1_c3_q11_classification.pdf


EC_draw:
	$(ORBITER_PATH)/orbiter.out -v 2 -process_combinatorial_objects -q 11 -n 2  \
		-draw_points_in_plane EC_11_1_3 \
		-fname_base_out EC_11_1_3 -embedded \
		-input -file_of_points elliptic_curve_b1_c3_q11.txt -end \
		-end


arcs_4_2_classify:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 8 \
		-canonical_form_PG  2  \
		-input -file_of_points arcs_4_2_lift.solutions \
		-end \
		-classification_prefix arcs_4_2_lift \
		-report \
		-end \
		-end
	pdflatex arcs_4_2_lift_classification.tex
	open arcs_4_2_lift_classification.pdf



cuspidal_cubic_q8_canonical:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 8 \
		-canonical_form_PG  2  \
		-input -file_of_points cuspidal_cubic_8.txt \
		-end \
		-classification_prefix cuspidal_cubic_q8 \
		-report \
		-end \
		-end
	pdflatex cuspidal_cubic_q8_classification.tex
	open cuspidal_cubic_q8_classification.pdf


# group of order 21:
#1,0,0,0,1,0,0,0,1,1, 
#1,0,0,0,6,0,0,0,7,0,
# acting on set of size 9: 0, 2, 3, 46, 51, 71, 61, 40, 28


cuspidal_cubic_q8_subsets_canonical:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 8 
		-canonical_form_PG  2  \
		-input -file_of_points cuspidal_cubic_8_subsets_6.txt \
		-end \
		-classification_prefix cuspidal_cubic_q8 \
		-report \
		-end \
		-end
	pdflatex cuspidal_cubic_q8_classification.tex
	open cuspidal_cubic_q8_classification.pdf


Hirschfeld4c:
	$(ORBITER_PATH)/orbiter.out -v 10 \
		-finite_field_activity -q 4 \
			-canonical_form_PG  3  \
				-input -file_of_points Hirschfeld_surface_q4.txt -end \
				-classification_prefix Hirschfeld_surface_q4 \
				-report \
			-end \
		-end
	pdflatex Hirschfeld_surface_q4_classification.tex
	open Hirschfeld_surface_q4_classification.pdf


# group order is 51840



Hirschfeld4c_set:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-finite_field_activity -q 4 \
			-canonical_form_PG  3 \
				-input \
					-set_of_points "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,23,26,27,30,31,34,35,38,39,42,47,48,51,52,53,54,59,60,61,62,67,68,69,70,75,76,79,80,81,82" \
				-end \
				-classification_prefix Hirschfeld_surface_q4 \
				-save_canonical_labeling \
				-save_ago \
				-save_cumulative_canonical_labeling \
				-save_cumulative_data \
				-report \
			-end \
		-end 
	pdflatex Hirschfeld_surface_q4_classification.tex
	open Hirschfeld_surface_q4_classification.pdf



Dickson_set_stabilizer:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-finite_field_activity -q 2 \
			-canonical_form_PG  3 \
				-input \
					-set_of_points "0,1,2,5,6" \
					-set_of_points "0,1,2,3,6" \
					-set_of_points "0,1,2,3,4" \
					-set_of_points "0,1,2,3,8" \
					-set_of_points "0,1,2,5,6,7,8" \
					-set_of_points "0,1,2,3,5,6,7" \
					-set_of_points "0,1,2,3,5,6,9" \
					-set_of_points "0,1,2,3,5,6,10" \
					-set_of_points "0,1,2,3,5,6,4" \
					-set_of_points "0,1,2,3,8,11,13" \
					-set_of_points "3,6,9,7,10,12,8,11,13,14,4" \
					-set_of_points "3,5,6,9,7,10,12,11,13,14,4" \
					-set_of_points "0,1,2,3,5,6,9,7,10,12,4" \
				-end \
				-classification_prefix Dickson_set \
				-save_canonical_labeling \
				-save_ago \
				-save_cumulative_canonical_labeling \
				-save_cumulative_data \
				-report \
			-end \
		-end 
	pdflatex Dickson_set_classification.tex
	open Dickson_set_classification.pdf



Endrass_7c:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 7 \
		-canonical_form_PG  3  \
		-input -file_of_points Endrass_F7.txt -end \
		-classification_prefix Endrass_F7 \
		-report \
		-end \
		-end
	pdflatex Endrass_F7_classification.tex
	open Endrass_F7_classification.pdf


# group order is 32


Hirschfeld16c:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 16 \
		-canonical_form_PG  3 \
		-input -file_of_points \
			Hirschfeld_surface_q16.txt -end \
		-classification_prefix Hirschfeld_surface_q16 \
		-report \
		-end \
		-end
	pdflatex Hirschfeld_surface_q16_classification.tex
	open Hirschfeld_surface_q16_classification.pdf

# nauty_interface_matrix_int n >= MAXN
# nauty_interface_matrix_int n = 74532
# nauty_interface_matrix_int MAXN = 22000



cuspidal_cubic_q16_subsets_canonical:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 8 \
		-canonical_form_PG  2  \
		-input -file_of_points cuspidal_cubic_16_subsets_6.txt \
		-end \
		-classification_prefix cuspidal_cubic_q16 \
		-report \
		-end \
		-end
	pdflatex cuspidal_cubic_q16_classification.tex
	open cuspidal_cubic_q16_classification.pdf




#############################################################################################################
# Section 29: Arcs


ARCS:


arcs_q7_6:
	$(ORBITER_PATH)/orbiter.out -v 7 \
		-orbiter_path $(ORBITER_PATH) \
		-define G \
		-linear_group -PGL 3 7 -end -end \
		-with G -do \
		-group_theoretic_activities \
				-poset_classification_control -problem_label arcs_q7_d2 -W -depth 6 -report -draw_poset -draw_options -end -end \
			-classify_arcs \
				-target_size 6 \
				-q 7 \
				-n 3 \
				-d 2 \
			-end \
		-end
	pdflatex arcs_q7_d2_poset.tex
	open arcs_q7_d2_poset.pdf


arcs_4_q8:
	$(ORBITER_PATH)/orbiter.out -v 8 \
		-orbiter_path $(ORBITER_PATH) \
		-define G \
		-linear_group -PGGL 3 8 -end -end \
		-with G -do \
		-group_theoretic_activities \
				-poset_classification_control -problem_label arcs_q8_d2 -W -depth 5 -report -draw_poset -draw_options -end -end \
			-classify_arcs \
				-target_size 5 \
				-q 8 \
				-n 3 \
				-d 2 \
			-end \
		-end
	pdflatex arcs_q8_d2_poset.tex
	open arcs_q8_d2_poset.pdf



AGL_4_3:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G \
		-linear_group -PGL 5 3 -subgroup_by_generators "AGL_4_3" "1965150720" 14 \
			"1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,2," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0," \
			"1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1," \
			"1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1," \
			-end -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex PGL_5_3_Subgroup_AGL_4_3_1965150720_report.tex
	open PGL_5_3_Subgroup_AGL_4_3_1965150720_report.pdf


arcs_AG_4_3:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define G \
		-linear_group -PGL 5 3 -subgroup_by_generators "AGL_4_3" "1965150720" 14 \
			"1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,2," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0," \
			"1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1," \
			"1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1," \
			-end -end \
		-with G -do \
		-group_theoretic_activities \
				-poset_classification_control -problem_label arcs_AG_4_3_d2 -W -depth 25 -end \
			-classify_arcs \
				-affine \
				-target_size 25 \
				-q 3 \
				-n 5 \
				-d 2 \
				-forbidden_point_set -long_string \
				"1,2,3,4,10,13,18,21,24,27,30,33,36,39,44,47,50,53,56,59,62," \
				"65,68,71,74,77,80,82,85,88,91,94,97,100,103,106,109,112,115,118" \
				-end_string \
			-end \
		-end \
		
		
#	pdflatex PGL_5_3_Subgroup_AGL_4_3_1965150720_report.tex
#	open PGL_5_3_Subgroup_AGL_4_3_1965150720_report.pdf

# 20
#20 0 5 6 8 11 16 19 25 28 43 45 51 57 63 70 99 102 105 107 114 2880 agaaaaaaagaaaaaaaaaaaaaaabaaaaaaacaaaaaaadaaaaaaaeaaaaaaafaaaaaabeaaaaaabcaaaaaaaeaaaaaaabaaaaaaacaaaaaaabaaaaaaabbaaaabbagiacabbaeaceegaaababgaeeceabciaaabbaaeafiaiiabfbkaeiacgfiiacfbjcaabieafaac
#-1 1 9907 in 0:00(2880) average is 2880 + 0 / 1
# time: 0:9


arcs_AG_4_3c:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 3 \
		-canonical_form_PG  4  \
		-input -set_of_points "0,5,6,8,11,16,19,25,28,43,45,51,57,63,70,99,102,105,107,114" -end \
		-classification_prefix arc_AG_4_3 \
		-report \
		-end \
		-end
	pdflatex arc_AG_4_3_classification.tex
	open arc_AG_4_3_classification.pdf



arcs_AG_4_3_aut:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G \
		-linear_group -PGL 5 3 -subgroup_by_generators "AGL_4_3_arc_aut" "2880" 6 \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,2,2,1,2," \
			"1,0,0,0,0,0,1,0,0,0,0,2,1,0,1,0,1,0,1,2,0,2,0,0,2," \
			"1,0,0,0,0,0,2,0,1,2,0,1,1,0,0,0,1,0,0,1,0,2,0,0,0," \
			"1,1,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1," \
			"1,0,1,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,2,0,1,2," \
			"1,0,1,0,1,0,2,0,0,0,0,1,2,0,2,0,1,0,1,2,0,2,0,1,0," \
			-end -end \
		-with G -do \
		-group_theoretic_activities \
		-export_gap \
		-export_magma \
		-report
	pdflatex PGL_5_3_Subgroup_AGL_4_3_arc_aut_2880_report.tex
	open PGL_5_3_Subgroup_AGL_4_3_arc_aut_2880_report.pdf


#strong_generators::print_generators_gap(ostream &ost)

hyperoval_4:
	$(ORBITER_PATH)/orbiter.out -v 8 \
		-draw_options -end \
		-define G \
		-linear_group -PGGL 3 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
				-poset_classification_control -problem_label arcs_q4_d2 -W -depth 6 -end \
			-classify_arcs \
				-target_size 6 \
				-q 4 \
				-n 3 \
				-d 2 \
			-end \
		-report \
		-end 
	pdflatex PGGL_3_4_report.tex
	open PGGL_3_4_report.pdf




hyperoval_8:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 3 8 -end -end \
		-with G -do \
		-group_theoretic_activities \
				-poset_classification_control -problem_label arcs_q8_d2 -W -depth 10 -end \
			-classify_arcs \
				-target_size 10 \
				-q 8 \
				-n 3 \
				-d 2 \
			-end \
		-end


frame_stabilizer_PGGL:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 3 8 -end -end \
		-with G -do \
		-group_theoretic_activities \
				-poset_classification_control -problem_label frame_q8 -W -depth 4 \
				-draw_options -end \
				-report \
				-end \
			-classify_arcs \
				-target_size 4 \
				-q 8 \
				-n 3 \
				-d 2 \
			-end \
		-end
	pdflatex frame_q8_poset.tex
	open frame_q8_poset.pdf

frame_stabilizer_PGL:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGL 3 8 -end -end \
		-with G -do \
		-group_theoretic_activities \
				-poset_classification_control -problem_label frame_q8 -W -depth 4 \
				-draw_options -end \
				-report \
				-end \
			-classify_arcs \
				-target_size 4 \
				-q 8 \
				-n 3 \
				-d 2 \
			-end \
		-end
	pdflatex frame_q8_poset.tex
	open frame_q8_poset.pdf



hyperoval_16:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define G \
		-linear_group -PGGL 3 16 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label arcs_q16_d2 -W -depth 18 -end \
			-classify_arcs \
				-target_size 18 \
				-q 16 \
				-n 3 \
				-d 2 \
			-end \
		-end




# June 3 2020: 1 sec on Mac
 

hyperoval_32:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 3 32 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label arcs_q32_d2 -W -depth 6 -end \
			-classify_arcs \
				-target_size 34 \
				-q 32 \
				-n 3 \
				-d 2 \
			-end \
		-end


# nonconical 6-arcs are used for classifying cubic surfaces:


nc_arcs_4:
	$(ORBITER_PATH)/orbiter.out -v 8 \
		-define G \
		-linear_group -PGGL 3 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label nc_arcs_q4_d2 -W -depth 4 -list -end \
			-classify_arcs \
				-target_size 4 \
				-q 4 \
				-n 3 \
				-d 2 \
				-conic_test \
			-end \
		-end



nc_arcs_8:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G \
		-linear_group -PGGL 3 8 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label nc_arcs_q8_d2 \
			-W -depth 6 -recognize "1,6,2,13,23,22" -report -end \
			-classify_arcs \
				-target_size 6 \
				-q 8 \
				-n 3 \
				-d 2 \
				-conic_test \
			-end \
		-report \
		-end
	pdflatex nc_arcs_q8_d2_poset.tex
	open nc_arcs_q8_d2_poset.pdf
	

# finds 2 nonconical 6-arcs:
## 6
#6 0 1 2 3 28 35 24 afaaaaaaafaaaaaaaaaaaaaaabaaaaaaacaaaaaaadaaaaaaafaaaaaaagaaaaaaabaaaaaaaeaaaaaaabaaaaaaabaaaaaaaccajiacabbaceabaiacceabeajcaeaaodljodaa
#6 0 1 2 3 28 38 12 afaaaaaaadaaaaaaaaaaaaaaabaaaaaaacaaaaaaadaaaaaaafaaaaaaaeaaaaaaabaaaaaaadaaaaaaabaaaaaaabaaaaaaafhapealgnfbbebamnbbaabh


nc_arcs_13:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGL 3 13 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label nc_arcs_q13_d2 -W -depth 6 -end \
			-classify_arcs \
				-target_size 6 \
				-q 13 \
				-n 3 \
				-d 2 \
				-conic_test \
			-end \
		-end

# finds 21 nonconical 6-arcs


nc_arcs_16:
	$(ORBITER_PATH)/orbiter.out -v 10 \
		-define G \
		-linear_group -PGGL 3 16 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label nc_arcs_q16_d2 -W -depth 6 -recognize "0,1,2,70,148,40" -end \
			-classify_arcs \
				-target_size 6 \
				-q 16 \
				-n 3 \
				-d 2 \
				-conic_test \
			-end \
		-end

nc_arcs_32:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 3 32 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label nc_arcs_q32_d2 -W -depth 6 -end \
			-classify_arcs \
				-target_size 6 \
				-q 32 \
				-n 3 \
				-d 2 \
				-conic_test \
			-end \
		-end


nc_arcs_64:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 3 64 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label nc_arcs_q64_d2 -W -depth 6 -end \
			-classify_arcs \
				-target_size 6 \
				-q 64 \
				-n 3 \
				-d 2 \
				-conic_test \
			-end \
		-end

nc_arcs_128:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 3 128 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label nc_arcs_q128_d2 -W -depth 6 -end \
			-classify_arcs \
				-target_size 6 \
				-q 128 \
				-n 3 \
				-d 2 \
				-conic_test \
			-end \
		-end

nc_arcs_256:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 3 256 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label nc_arcs_q256_d2 -W -depth 6 -end \
			-classify_arcs \
				-target_size 6 \
				-q 256 \
				-n 3 \
				-d 2 \
				-conic_test \
			-end \
		-end

six_arcs_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G \
		-linear_group -PGGL 3 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q4 -end \
		-six_arcs -filter_by_nb_Eckardt_points 13 -end

six_arcs_8_PGL:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G \
		-linear_group -PGL 3 8 -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q8 -end \
		-six_arcs -filter_by_nb_Eckardt_points 13 -end

six_arcs_8_PGGL:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G \
		-linear_group -PGGL 3 8 -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q8 -end \
		-six_arcs -filter_by_nb_Eckardt_points 13 -end

six_arcs_16_PGGL:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G \
		-linear_group -PGGL 3 16 -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q16 -end \
		-six_arcs -filter_by_nb_Eckardt_points 13 -end

six_arcs_32:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G \
		-linear_group -PGGL 3 32 -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q32 -end \
		-six_arcs -filter_by_nb_Eckardt_points 13 -end

six_arcs_64:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G \
		-linear_group -PGGL 3 64 -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q64 -end \
		-six_arcs -filter_by_nb_Eckardt_points 13 -end

six_arcs_128:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G \
		-linear_group -PGGL 3 128 -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q128 -end \
		-six_arcs -filter_by_nb_Eckardt_points 13 -end

# 1 min 39 sec



six_arcs_256:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G \
		-linear_group -PGGL 3 256 -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q256 -W -end \
		-six_arcs -filter_by_nb_Eckardt_points 13 -end


# 27 minutes on ripoff


awss:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G \
		-linear_group -PGL 3 13 -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label arcs_13_3 -W -depth 13 -end \
		-classify_arcs -target_size 13 -q 13 -n 3 -d 3 -end -end






#############################################################################################################
# Section 30: Cubic Curves

CUBIC_CURVES:


cubic_curves_PG_2_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G \
		-linear_group -PGGL 3 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-classify_cubic_curves -q 4 -target_size 9 -n 3 -d 3 \
			-poset_classification_control -problem_label cc_4 -W -depth 9 \
				-draw_options -rad 200 -embedded -end \
				-draw_poset \
			-end \
		-end
	pdflatex Cubic_curves_q4.tex
	open Cubic_curves_q4.pdf
	pdflatex cc_4_poset_lvl_9.tex
	#open cc_4_poset_lvl_9.pdf

cubic_curves_PG_2_4_draw:
	$(ORBITER_PATH)/orbiter.out -v 3 -draw_layered_graph cc_4_poset_lvl_9.layered_graph \
		-rad 300 -embedded -line_width 1.1  -y_stretch 0.9 -scale 0.25 \
		-paths_in_between 6 4 9 0 \
		-end
	pdflatex cc_4_poset_lvl_9_draw.tex
	open cc_4_poset_lvl_9_draw.pdf


cubic_curves_PG_2_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G \
		-linear_group -PGGL 3 8 -end \
		-with G -do \
		-group_theoretic_activities \
		-classify_cubic_curves -q 8 -target_size 9 -n 3 -d 3 \
			-poset_classification_control -problem_label cc_8 -W -depth 9 \
				-draw_options -rad 200 -embedded -end \
				-recognize "0,1,2,3,35,28" \
				-recognize "1,2,3,51,28,61,46,71,40" \
				-draw_poset \
				-Kramer_Mesner_matrix 6 9 \
			-end \
		-end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix cc_8_KM_6_9.csv 50 3
	pdflatex Cubic_curves_q8.tex
	open Cubic_curves_q8.pdf
	#pdflatex cc_8_tree_lvl_9.tex
	#open cc_8_tree_lvl_9.pdf
	
# the 6-set is orbit 7
# the 9-set is orbit 1


cubic_curves_PG_2_8_draw:
	$(ORBITER_PATH)/orbiter.out -v 3 -draw_layered_graph cc_8_poset_lvl_9.layered_graph \
		-rad 2 -embedded -line_width 0.01  -y_stretch 1.3 -scale 0.5 \
		-paths_in_between 6 7 9 1 \
		-end
	pdflatex cc_8_poset_lvl_9_draw.tex
	open cc_8_poset_lvl_9_draw.pdf

#cc_8_poset_lvl_9.layered_graph
#cc_8_poset_detailed_lvl_9.layered_graph
	
cubic_curves_PG_2_8_canon:
	$(ORBITER_PATH)/orbiter.out -v 6 -finite_field_activity -q 8 \
		-canonical_form_PG  2   \
		-input -set_of_points "2,3,28,46,51,61,40,71" -end \
		-classification_prefix cc_8 \
		-report \
		-end \
		-end

arcs_q8_6_3:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G \
		-linear_group -PGGL 3 8 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label arcs_q8_d3 \
			-W -depth 6 -recognize "0,1,2,3,35,28" -end \
			-classify_arcs \
				-target_size 6 \
				-q 8 \
				-n 3 \
				-d 3 \
			-end \
		-end
		#-report \
	#pdflatex nc_arcs_q8_d2_poset.tex
	#open nc_arcs_q8_d2_poset.pdf
	
arcs_q8_8_3:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G \
		-linear_group -PGGL 3 8 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label arcs_q8_d3 \
			-W -depth 8  \
				-draw_options -rad 200 -embedded -end \
				-draw_poset \
			-end \
			-classify_arcs \
				-target_size 8 \
				-q 8 \
				-n 3 \
				-d 3 \
			-end \
		-end
	pdflatex arcs_q8_d3_tree_lvl_8.tex
	open arcs_q8_d3_tree_lvl_8.pdf




cuspidal_cubic_q11:
	$(ORBITER_PATH)/orbiter.out -v 5 -create_combinatorial_object -q 11 -n 2 \
		-cuspidal_cubic -end \
		-save "./"




cubic_curves_PG_2_11:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G \
		-linear_group -PGL 3 11 -end \
		-with G -do \
		-group_theoretic_activities \
		-classify_cubic_curves -q 11 -target_size 9 -n 3 -d 3 \
			-poset_classification_control -problem_label cc_11 -W -depth 9 \
				-draw_options -rad 200 -embedded -end \
				-recognize "0,1,2,3,37,47" \
				-recognize "0, 1, 2, 3, 4, 16, 26, 88, 90" \
				-draw_poset \
			-end \
		-end
	pdflatex Cubic_curves_q11.tex
	open Cubic_curves_q11.pdf
	#pdflatex cc_8_tree_lvl_9.tex
	#open cc_8_tree_lvl_9.pdf



cubic_curves_PG_2_11_draw:
	$(ORBITER_PATH)/orbiter.out -v 3 -draw_layered_graph cc_11_poset_lvl_9.layered_graph \
		-rad 200 -embedded -line_width 0.3  -y_stretch 1.3 -scale 0.5 \
		-paths_in_between 6 38 9 32307 \
		-end
	pdflatex cc_11_poset_lvl_9_draw.tex
	open cc_11_poset_lvl_9_draw.pdf



cuspidal_cubic_q11:
	$(ORBITER_PATH)/orbiter.out -v 5 -create_combinatorial_object -q 11 -n 2 \
		-cuspidal_cubic -end \
		-save "./"

#cuspidal_cubic_11_subsets_6.txt



cuspidal_cubic_q11_subsets_canonical:
	$(ORBITER_PATH)/orbiter.out -v 10 -define F11 -finite_field -q 11 -end -end \
		-with F11 -do \
		-finite_field_activity \
		-canonical_form_PG  2 \
		-input \
			-set_of_points "0,1,2,3,37,47"  \
			-file_of_points cuspidal_cubic_11_subsets_6.txt \
		-end \
		-classification_prefix cuspidal_cubic_q11 \
		-report \
		-end \
		-end
	pdflatex cuspidal_cubic_q11_classification.tex
	open cuspidal_cubic_q11_classification.pdf





#############################################################################################################
# Section 31: Cubic Surfaces Creation


CUBIC_SURFACES_CREATION:



surface_table:
	$(ORBITER_PATH)/orbiter.out -v 3 -make_table_of_surfaces
	pdflatex surfaces_report.tex
	open surfaces_report.pdf





# Dickson:

D1="1,4,1,7,1,12,1,15"
D2=	"1,1,1,2,1,3,1,4,1,7,1,11,1,14,1,15,1,19"
D3="1,4,1,7,1,12,1,14,1,15,1,19"
D4="1,3,1,4,1,8,1,9,1,10,1,14,1,17,1,18,1,19"
D5="1,2,1,3,1,4,1,7,1,15"
D6="1,2,1,3,1,5,1,9,1,10,1,14,1,15"
D7="1,3,1,4,1,5,1,7,1,10,1,11,1,19"
D8="1,3,1,4,1,7,1,11,1,12,1,15,1,19"
D9="1,4,1,7,1,8,1,9,1,10,1,13,1,15,1,16,1,17"
D10:="1,4,1,7,1,8,1,10,1,13,1,14,1,15,1,16,1,17"
D11="1,4,1,6,1,7,1,8,1,9,1,10,1,15,1,16,1,17"
D12="1,6,1,8,1,9,1,10,1,15,1,16,1,17"
D13="1,6,1,7,1,10,1,11,1,14,1,15"
D14="1,4,1,5,1,6,1,8,1,12,1,14,1,16"
D15="1,4,1,5,1,8,1,12,1,13,1,14,1,16"
D16="1,4,1,5,1,8,1,9,1,12,1,13,1,16"
D17=""
D18=""
D19="1,4,1,5,1,6,1,7,1,8,1,10,1,13,1,14,1,15"
D20="1,5,1,6,1,8,1,10,1,13,1,14,1,15"
D21="1,4,1,5,1,7,1,9,1,10,1,11,1,13,1,14,1,15,1,18,1,19"
D22="1,4,1,7,1,9,1,11,1,13,1,14,1,15,1,18,1,19"
D23="1,5,1,8,1,9,1,10,1,13,1,14,1,15,1,18,1,19"
D24="1,4,1,5,1,7,1,10,1,11,1,12,1,13,1,14,1,17,1,18"
D25="1,4,1,5,1,7,1,10,1,11,1,13,1,14,1,15,1,17,1,18"
D26="1,4,1,5,1,6,1,7,1,10,1,11,1,14,1,15,1,17,1,18"
D27="1,4,1,7,1,5,1,10,1,6,1,11,1,9,1,15,1,17,1,18"
D28="1,4,1,7,1,13,1,11,1,9,1,15,1,17,1,18"
D29="1,0,1,1,1,2,1,5,1,7,1,11,1,13,1,14,1,15,1,17,1,18,1,19"
D30="1,0,1,1,1,2,1,3,1,7,1,8,1,10,1,11,1,12,1,15,1,16,1,19"
D31="1,0,1,1,1,2,1,3,1,6,1,7,1,10,1,12,1,13,1,15,1,16,1,19"
D32="1,0,1,1,1,2,1,3,1,6,1,7,1,8,1,11,1,13,1,15"
D33="1,0,1,1,1,2,1,3,1,7,1,15"
D34="1,0,1,1,1,2,1,3,1,7,1,12,1,15,1,19"
D35="1,0,1,1,1,2,1,3,1,6,1,7,1,12,1,13,1,15,1,19"
D36="1,0,1,1,1,2,1,3,1,5,1,7,1,10,1,12,1,15,1,19"
D37="1,0,1,1,1,2,1,3,1,7,1,8,1,11,1,19"
D38="1,0,1,1,1,2,1,3,1,6,1,7,1,8,1,11,1,13,1,19"
D39="1,0,1,1,1,2,1,3,1,5,1,7,1,8,1,10,1,11,1,19"
D40="1,0,1,1,1,2,1,3,1,7,1,8,1,11,1,19"
D41="1,0,1,1,1,2,1,3,1,7,1,12,1,15,1,19"
D42="1,2,1,4,1,7,1,5,1,10,1,11,1,14,1,19"
D43="1,2,1,4,1,7,1,11,1,12,1,14,1,15,1,19"
D44="1,0,1,1,1,2,1,6,1,8,1,11,1,12,1,14"
D45="1,6,1,8,1,9,1,11,1,12,1,13,1,14,1,15,1,16,1,17,1,18,1,19"


D6_q2:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGL 4 2 -wedge -end \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -by_coefficients $(D6) -q 2 \
		-end \
		-end 
	pdflatex surface_by_coefficients_q2_report.tex
	open surface_by_coefficients_q2_report.pdf
	mv surface_by_coefficients_q2_points.txt surface_by_coefficients_q2_D6_points.txt


# 1 line over GF(2)

D3_q4:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 4 4 -wedge -end \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -by_coefficients $(D3) -q 4 \
		-end \
		-end 
	pdflatex surface_by_coefficients_q4_report.tex
	open surface_by_coefficients_q4_report.pdf
	mv surface_by_coefficients_q4_points.txt surface_by_coefficients_q4_D3_points.txt

#surface_by_coefficients_q4_points.txt


D3_q4_canonical:
	$(ORBITER_PATH)/orbiter.out -v 10 -finite_field_activity -q 4 \
		-canonical_form_PG  3  \
		-input -file_of_points surface_by_coefficients_q4_points.txt -end \
		-classification_prefix surface_by_coefficients_q4_points \
		-report \
		-end \
		-end
	pdflatex surface_by_coefficients_q4_points_classification.tex
	open surface_by_coefficients_q4_points_classification.pdf

# group order 24




D4_q8:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 4 8 -wedge -end \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -by_coefficients $(D4) -q 8 \
		-end \
		-end 
	pdflatex surface_by_coefficients_q8_report.tex
	open surface_by_coefficients_q8_report.pdf
	mv surface_by_coefficients_q8_points.txt surface_by_coefficients_q8_D4_points.txt





D6_q4:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 4 4 -wedge -end \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -by_coefficients $(D6) -q 4 \
		-end \
		-end 
	pdflatex surface_by_coefficients_q4_report.tex
	open surface_by_coefficients_q4_report.pdf
	mv surface_by_coefficients_q4_points.txt surface_by_coefficients_q4_D6_points.txt

# D6 has 7 lines over GF(4)



D6_q4_canonical:
	$(ORBITER_PATH)/orbiter.out -v 10 -finite_field_activity -q 4 \
		-canonical_form_PG  3  \
		-input -file_of_points surface_by_coefficients_q4_D6_points.txt -end \
		-classification_prefix surface_by_coefficients_q4_points \
		-report \
		-end \
		-end
	pdflatex surface_by_coefficients_q4_points_classification.tex
	open surface_by_coefficients_q4_points_classification.pdf


# group order 384 = 192 * 2



D8_q4:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 4 4 -wedge -end \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -by_coefficients $(D8) -q 4 \
		-end \
		-end 
	pdflatex surface_by_coefficients_q4_report.tex
	open surface_by_coefficients_q4_report.pdf
	mv surface_by_coefficients_q4_points.txt surface_by_coefficients_q4_D8_points.txt


D_3_8_q4_canonical:
	$(ORBITER_PATH)/orbiter.out -v 10 -finite_field_activity -q 4 \
		-canonical_form_PG  3  \
		-input \
			-file_of_points surface_by_coefficients_q4_D3_points.txt \
			-file_of_points surface_by_coefficients_q4_D8_points.txt \
		-end \
		-classification_prefix surface_by_coefficients_q4_points \
		-report \
		-end \
		-end
	pdflatex surface_by_coefficients_q4_points_classification.tex
	open surface_by_coefficients_q4_points_classification.pdf

# D3 and D8 are isomorphic over GF(4)


D1_q8:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 4 8 -wedge -end \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -by_coefficients $(D1) -q 8 \
		-end \
		-end 
	pdflatex surface_by_coefficients_q8_report.tex
	open surface_by_coefficients_q8_report.pdf

#surface_by_coefficients_q8_points.txt


D1_q8_canonical:
	$(ORBITER_PATH)/orbiter.out -v 10 -finite_field_activity -q 8 \
		-canonical_form_PG  3  \
		-input -file_of_points surface_by_coefficients_q8_points.txt -end \
		-classification_prefix surface_by_coefficients_q8_points \
		-report \
		-end \
		-end
	pdflatex surface_by_coefficients_q8_points_classification.tex
	open surface_by_coefficients_q8_points_classification.pdf

# group order 2160




surface_4_0:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end
		-define G -linear_group -PGGL 4 4 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 4 -catalogue 0 -end \
		-end
	pdflatex surface_catalogue_q4_iso0_with_group.tex
	open surface_catalogue_q4_iso0_with_group.pdf


# this gives good Eckardt points:
# absolutely clean, meaning we have the lines b1 and b2 right.
# b1 = [1,0,0,0,0,0,1,1]
# b2 = [0,1,0,0,0,0,0,1]
# the last transformation cleans up the lines b1 and b2.



Family_general_F7:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 7 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 2 3 3 4 -q 7 -end 
		pdflatex surface_family_general_abcd_q7_a2_b3_c3_d4_report.tex
		open surface_family_general_abcd_q7_a2_b3_c3_d4_report.pdf

# Fermat with 18 Eckardt points





Family_general_F7_tangents:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-process_combinatorial_objects -n 3 -q 7 \
			-input \
			-set_of_points "56, 0, 3, 156, 350, 23, 212, 216, 284, 321, 70, 261, 384, 149, 381, 297, 234, 63" \
			-end \
		-ideal_PART 3 \
		-fname_base_out "form" \
		-end



surface_recognize_q7_abcd_2_3_3_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 7 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
			-surface_recognize -q 7 \
			-family_general_abcd 2 3 3 4 \
			-end \
		-end




Family_general_7_rec:
	$(ORBITER_PATH)/orbiter.out -v 5 -define G -linear_group -PGL 4 7 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities -surface_recognize -q 7 \
		-by_coefficients "3,14,1,17,3,18,3,19" -end -end

# has 73 lines



Family_general_q8_2453:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 8 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 2 4 5 3 -q 8 \
		-end \
		-end 
	pdflatex surface_family_general_abcd_q8_a2_b4_c5_d3_report.tex
	open surface_family_general_abcd_q8_a2_b4_c5_d3_report.pdf

#( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 6, 1, 7 )
# has 91 lines



surface_8_0_unclean:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 8 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 8 -catalogue 0 -end \
		-end
	pdflatex surface_catalogue_q8_iso0_report.tex
	open surface_catalogue_q8_iso0_report.pdf



surface_8_0:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 8 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 8 -catalogue 0 \
			-select_double_six "15,11,22,19,24,5,16,10,23,20,25,4" \
			-select_double_six "3,2,1,0,5,4,9,8,7,6,11,10" \
			-transform_inverse "1,4,4,0,6,0,0,0,6,2,0,1,7,0,4,0,0" \
			-transform "4,4,0,0, 0,0,1,0, 1,0,0,0, 0,0,0,1, 0" \
			-transform_inverse "2,0,0,0,0,2,0,0,0,0,2,0,1,1,2,3,0" \
		-end
	pdflatex surface_catalogue_q8_iso0_report.tex
	open surface_catalogue_q8_iso0_report.pdf



# clean equation for Tekirdag-1:

surface_8_0b:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 8 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 8 -catalogue 0 \
			-select_double_six "15,11,22,19,24,5,16,10,23,20,25,4" \
			-select_double_six "3,2,1,0,5,4,9,8,7,6,11,10" \
			-transform "1,0,0,0,0,1,0,6,0,0,1,6,0,0,0,1,0" \
			-transform_inverse "3,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0" \
			-transform_inverse "2,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0" \
		-end 
	pdflatex surface_catalogue_q8_0.tex
	open surface_catalogue_q8_0.pdf

surface_recognize_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 8 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities -surface_recognize -q 8 \
		-by_coefficients "1,6,1,8,1,11,1,13,1,19" -end -end




surface_8_from_arc_old:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 8  -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 8 -arc_lifting "0,1,2,3,28,35"  \
		-end
	pdflatex surface_arc_q8_0_1_2_3_28_35.tex
	open surface_arc_q8_0_1_2_3_28_35.pdf



surface_8_from_arc_new:
	$(ORBITER_PATH)/orbiter.out -v 10 -define G -linear_group -PGGL 4 8  -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 8 -arc_lifting_with_two_lines "0,1,2,20,30,37" "24,1898" \
		-end
	pdflatex surface_arc_lifting_with_two_lines_q8_lines24_1898_arc0_1_2_20_30_37_report.tex
	open surface_arc_lifting_with_two_lines_q8_lines24_1898_arc0_1_2_20_30_37_report.pdf

arc_lifting_with_lines_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 8  -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 8 -arc_lifting_with_two_lines "0,1,2,20,30,37" "0,0" \
		-end


surface_isomorph_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 8 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities -surface_isomorphism_testing \
			-q 8 -by_coefficients "1,5,1,8,1,9,1,10,1,11,1,12,6,14,6,15,7,18,7,19" -end \
			-q 8 -by_coefficients "1,6,1,8,1,11,1,13,1,19" -end 








HCV_13:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGL 4 13 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_HCV 3 1 -q 13 -end \
		-end
	pdflatex surface_family_HCV_q13_a3_b1_with_group.tex
	open surface_family_HCV_q13_a3_b1_with_group.pdf

# writes tangents.txt

HCV_tangents_ideal:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-process_combinatorial_objects -n 3 -q 13 \
			-input \
			-file_of_points "tangents.txt" \
			-end \
		-ideal_PART 4 \
		-fname_base_out "form" \
		-end








surface_isomorph_16:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities -surface_isomorphism_testing \
			-q 16 -by_coefficients "1,5,1,8,1,9,1,10,1,11,1,12,6,14,6,15,7,18,7,19" -end \
			-q 16 -by_coefficients "13,6,3,8,3,11,13,13,1,19" -end 

# 1 min 8 sec on Mac from scratch (with all data files removed)


# clean equation for Tekirdag-2:


surface_16_0:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 0 \
		-transform "1,0,0,0,0,1,0,12,0,0,1,12,0,0,0,1,0" \
		-transform "15,11,4,0,0,0,12,0,0,12,0,0,0,0,0,1,3" \
		-end
	pdflatex surface_catalogue_q16_0.tex
	open surface_catalogue_q16_0.pdf


#		-transform_inverse "3,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0" \
#		-transform_inverse "13,12,1,0,12,13,1,0,0,0,1,0,0,0,0,1,0" \
#		-transform_inverse "1,0,0,0,0,1,0,0,12,12,1,0,0,0,0,1,0" \
#		-transform_inverse "12,0,0,0,0,12,0,0,0,0,1,0,0,0,0,1,0" 



identify_F13_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 4 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-surface_identify_F13


identify_F13_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 8 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-surface_identify_F13

identify_F13_16:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-surface_identify_F13

identify_F13_32:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 32 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-surface_identify_F13

identify_F13_64:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 64 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-surface_identify_F13



# Tekirdag-1:

G13_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 8 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_G13 2 -q 8 -end
	pdflatex surface_family_G13_q8_a2.tex
	open surface_family_G13_q8_a2.pdf


F13_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 8 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_F13 2 -q 8 -end
	pdflatex surface_family_F13_q8_a2.tex
	open surface_family_F13_q8_a2.pdf


# Tekirdag-2:

F13_16:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_F13 2 -q 16 -end
	pdflatex surface_family_G13_q16_a2.tex
	open surface_family_G13_q16_a2.pdf


# Tekirdag-3:

F13_32:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 32 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_F13 2 -q 32 -end

# Kapadokya-1:

F13_64a:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 64 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_F13 2 -q 64 -end

# Kapadokya-2:

F13_64b:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 64 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_F13 18 -q 64 -end


Colorado1:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 128 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -q 128 -catalogue 0 \
			-transform_inverse "1,0,0,0,0,1,0,96,0,0,1,96,0,0,0,1,0" \
		-end 
	pdflatex surface_catalogue_q128_0.tex
	open surface_catalogue_q128_0.pdf


# recognize the arcs from Colorado-1,2,3:

nc_arcs_128:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 3 128 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label nc_arcs_q128_d2 -W -depth 6 \
				-recognize "0,1,2,262,516,8322" \
				-recognize "0,1,2,278,1032,15234" \
				-recognize "0,1,2,342,2064,13698" \
			-end \
			-classify_arcs \
				-target_size 6 \
				-q 128 \
				-n 3 \
				-d 2 \
				-conic_test \
			-end \
		-end



Colorado2:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 128 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -q 128 -catalogue 926 \
			-transform_inverse "1,0,0,0,0,1,0,32,0,0,1,32,0,0,0,1,0" \
		-end 
	pdflatex surface_catalogue_q128_926.tex
	open surface_catalogue_q128_926.pdf

Colorado3:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 128 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -q 128 -catalogue 928 \
			-transform_inverse "1,0,0,0,0,1,0,59,0,0,1,59,0,0,0,1,0" \
		-end 
	pdflatex surface_catalogue_q128_928.tex
	open surface_catalogue_q128_928.pdf


# Colorado-1:

F13_128a:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 128 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_F13 2 -q 128 -end

# Colorado-2:

F13_128b:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 128 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_F13 6 -q 128 -end

# Colorado-3:

F13_128c:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 128 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_F13 14 -q 128 -end


move_two_lines:
	$(ORBITER_PATH)/orbiter.out -v 5 -define F -finite_field -q 8 -end -end \
		-with F -do -finite_field_activity \
		-move_two_lines_in_hyperplane_stabilizer 65 4680 72 657 -end




test_432:
	$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGGL 4 4 \
		-subgroup_by_generators "trihedra" "432" 8 \
		"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0" \
		"1,0,0,0,0,3,0,0,0,0,3,0,0,0,0,2,1" \
		"1,0,0,0,0,1,0,0,0,0,3,0,0,0,1,3,0" \
		"1,0,0,0,0,1,0,0,0,0,3,2,0,0,0,3,0" \
		"1,0,0,0,0,1,0,0,0,0,0,2,0,0,1,0,0" \
		"1,0,0,0,2,3,0,0,0,0,3,0,0,0,0,2,1" \
		"1,3,0,0,2,0,0,0,0,0,2,0,0,0,0,2,0" \
		"0,0,1,0,0,0,2,1,3,0,0,0,0,3,0,0,0" \
		-end \

test_432_t:
	$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGGL 4 4 \
		-subgroup_by_generators "trihedra" "432" 8 \
		"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0" \
		"1,0,0,0,0,3,0,0,0,0,3,0,0,0,0,2,1" \
		"1,0,0,0,0,1,0,0,0,0,3,1,0,0,0,3,0" \
		"1,0,0,0,0,1,0,0,0,0,3,0,0,0,2,3,0" \
		"1,0,0,0,0,1,0,0,0,0,0,1,0,0,2,0,0" \
		"1,2,0,0,0,3,0,0,0,0,3,0,0,0,0,2,1" \
		"1,2,0,0,3,0,0,0,0,0,2,0,0,0,0,2,0" \
		"0,0,3,0,0,0,0,3,1,2,0,0,0,1,0,0,0" \
		-end \
		
		
# 5 Eckardt points:

surface_16_2a:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
		-end 
	- rm surface_catalogue_q16_iso2.pdf
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf


surface_16_2b:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
			-transform "1,0,0,10, 0,1,0,0, 0,0,0,1, 0,0,1,0, 0" \
		-end 
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf


surface_16_2c:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
			-transform "1,0,0,10, 0,1,0,0, 0,0,0,1, 0,0,1,0, 0" \
			-transform "5,0,0,0, 0,15,0,0, 1,1,1,0, 0,0,0,1, 0" \
		-end \
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf


surface_16_2d:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
			-select_double_six "17,20,3,4,0,23, 11,8,15,14,24,7" \
			-transform "1,0,0,10, 0,1,0,0, 0,0,0,1, 0,0,1,0, 0" \
			-transform "5,0,0,0, 0,15,0,0, 1,1,1,0, 0,0,0,1, 0" \
		-end
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf

surface_16_2_move_two_lines:
	$(ORBITER_PATH)/orbiter.out -v 5 -define F -finite_field -q 16 -end -end \
		-with F -do -finite_field_activity \
		-move_two_lines_in_hyperplane_stabilizer_text  \
		"1,0,0,0, 0,0,0,1"  "1,1,0,2, 0,0,1,0" \
		"1,0,0,0, 0,0,0,1" "0,1,0,1, 0,0,1,0" \
		-end

surface_16_2e:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
			-select_double_six "17,20,3,4,0,23, 11,8,15,14,24,7" \
			-transform "1,0,0,10, 0,1,0,0, 0,0,0,1, 0,0,1,0, 0" \
			-transform "5,0,0,0, 0,15,0,0, 1,1,1,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 0,1,0,0, 0,0,1,0, 12,0,0,12, 0" \
		-end \
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf



### BAD:
bad_surface_16_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
			-transform "1,0,0,10,0,1,0,0,0,0,0,1,0,0,1,0,0" \
			-transform "5,0,0,0, 14,14,1,0, 1,1,0,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 1,1,1,0, 0,0,1,0, 0,0,0,1, 0" \
		-end 
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf

#		-transform_inverse "1,0,5,0,0,1,15,0,0,0,1,0,0,0,0,1,0" 



PGGL_2_16:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -rad 200 -end \
		-define G \
		-linear_group -PGGL 2 16 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
			-orbits_on_points \
			-stabilizer \ 
		-end 
	pdflatex PGGL_2_16_report.tex
	open PGGL_2_16_report.pdf
		

#stabilizer of point 0 has order 960:
#1,0,0,1,2,\\
#1,0,0,1,3,\\
#1,0,0,8,2,\\
#1,0,0,7,2,\\
#1,0,3,5,0,\\


PGGL_2_16_pt_stab:
	$(ORBITER_PATH)/orbiter.out -v 10 \
		-draw_options -rad 200 -end \
		-define G -linear_group -PGGL 2 16 \
			-subgroup_by_generators "stab_of_0" "960" 5 \
				"1,0,0,1,2" \
				"1,0,0,1,3" \
				"1,0,0,8,2" \
				"1,0,0,7,2" \
				"1,0,3,5,0" \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
			-orbits_on_subsets 4 \
			-poset_classification_control -recognize "2,5,7,15" -end \
		-end 
		


bad_surface_16_2d:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
			-select_double_six "24,25,8,18,2,14, 22,23,9,17,3,13" \
			-transform "1,0,0,10,0,1,0,0,0,0,0,1,0,0,1,0,0" \
			-transform "5,0,0,0, 14,14,1,0, 1,1,0,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 1,1,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "7,1,0,0, 0,1,0,0, 0,1,7,0, 0,0,0,1, 3" \
		-end 
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf



arc_lifting_with_lines_16:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16  -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -arc_lifting_with_two_lines "164,148,14,11,35,83" "31526,41802" \
		-end

#		-create_surface -q 16 -arc_lifting_with_two_lines "162,146,13,10,34,81" "31526,41802" \



nc_arcs_16_again:
	$(ORBITER_PATH)/orbiter.out -v 8 \
		-define G \
		-linear_group -PGGL 3 16 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label nc_arcs_q16_d2 -W -depth 6 \
				-recognize "162,146,13,10,34,81" \
				-recognize "0,1,2,3,52,101" \
			-end \
			-classify_arcs \
				-target_size 6 \
				-q 16 \
				-n 3 \
				-d 2 \
				-conic_test \
			-end \
		-end
		
# stabilizer of the arc has order 4:
#0,1,0,1,0,0,1,1,1,0,\\
#1,1,1,0,0,1,0,1,0,0,\\



arc_lifting_with_lines_16_again:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -embedded -rad 250 -end \
		-define G -linear_group -PGGL 4 16  -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -arc_lifting_with_two_lines "0,2,36,1,54,103" "266,60589" -end \
		-surface_report \
		-end




bad_surface_16_2e:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
			-select_double_six "24,25,8,18,2,14, 22,23,9,17,3,13" \
			-transform "1,0,0,10,0,1,0,0,0,0,0,1,0,0,1,0,0" \
			-transform "5,0,0,0, 14,14,1,0, 1,1,0,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 1,1,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "7,1,0,0, 0,1,0,0, 0,1,7,0, 0,0,0,1, 3" \
			-transform "12,1,12,0, 15,0,15,0, 15,0,14,0, 0,0,0,1, 1" \
		-end 
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf



bad_surface_16_2ef:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
			-select_double_six "24,25,8,18,2,14, 22,23,9,17,3,13" \
			-select_double_six "11,10,8,9,6,7, 5,4,2,3,0,1" \
			-transform "1,0,0,10,0,1,0,0,0,0,0,1,0,0,1,0,0" \
			-transform "5,0,0,0, 14,14,1,0, 1,1,0,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 1,1,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "7,1,0,0, 0,1,0,0, 0,1,7,0, 0,0,0,1, 3" \
			-transform "12,1,12,0, 15,0,15,0, 15,0,14,0, 0,0,0,1, 1" \
		-end 
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf




bad_surface_16_2efg:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
			-select_double_six "24,25,8,18,2,14, 22,23,9,17,3,13" \
			-select_double_six "11,10,8,9,6,7, 5,4,2,3,0,1" \
			-transform "1,0,0,10,0,1,0,0,0,0,0,1,0,0,1,0,0" \
			-transform "5,0,0,0, 14,14,1,0, 1,1,0,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 1,1,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "7,1,0,0, 0,1,0,0, 0,1,7,0, 0,0,0,1, 3" \
			-transform "12,1,12,0, 15,0,15,0, 15,0,14,0, 0,0,0,1, 1" \
			-transform "1,0,0,0, 0,1,0,0, 0,0,1,0, 7,0,6,13, 0" \
		-end 
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf



surface_16_2efgh:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
			-select_double_six "24,25,8,18,2,14, 22,23,9,17,3,13" \
			-select_double_six "11,10,8,9,6,7, 5,4,2,3,0,1" \
			-transform "1,0,0,10,0,1,0,0,0,0,0,1,0,0,1,0,0" \
			-transform "5,0,0,0, 14,14,1,0, 1,1,0,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 1,1,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "7,1,0,0, 0,1,0,0, 0,1,7,0, 0,0,0,1, 3" \
			-transform "12,1,12,0, 15,0,15,0, 15,0,14,0, 0,0,0,1, 1" \
			-transform "1,0,0,0, 0,1,0,0, 0,0,1,0, 7,0,6,13, 0" \
			-transform "0,0,1,0, 1,1,1,0, 1,0,0,0, 0,1,1,1, 0" \
		-end 
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf


########
#old, bad stuff:

surface_16_2f:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
			-select_double_six "24,25,8,18,2,14, 22,23,9,17,3,13" \
			-select_double_six "4,5,2,3,0,1, 10,11,8,9,6,7" \
			-transform "1,0,0,10,0,1,0,0,0,0,0,1,0,0,1,0,0" \
			-transform "5,0,0,0, 14,14,1,0, 1,1,0,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 1,1,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "7,1,0,0, 0,1,0,0, 0,1,7,0, 0,0,0,1, 3" \
			-transform "12,1,12,0, 15,0,15,0, 15,0,14,0, 0,0,0,1, 1" \
		-end 
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf





surface_16_2g:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -catalogue 2 \
			-select_double_six "24,25,8,18,2,14, 22,23,9,17,3,13" \
			-select_double_six "4,5,2,3,0,1, 10,11,8,9,6,7" \
			-transform "1,0,0,10,0,1,0,0,0,0,0,1,0,0,1,0,0" \
			-transform "5,0,0,0, 14,14,1,0, 1,1,0,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 1,1,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "7,1,0,0, 0,1,0,0, 0,1,7,0, 0,0,0,1, 3" \
			-transform "12,1,12,0, 15,0,15,0, 15,0,14,0, 0,0,0,1, 1" \
			-transform "1,0,0,0, 0,1,0,0, 0,0,1,0, 6,0,4,13, 0" \
		-end 
	pdflatex surface_catalogue_q16_iso2.tex
	open surface_catalogue_q16_iso2.pdf






Family_bes_16_a3_c4:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_bes 3 4 -q 16 -end 
		



identify_Bes_16:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-surface_identify_Bes \
		-end

# 17 minutes


arc_lifting_with_lines_Bes_16:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16  -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs  -end \
		-create_surface -q 16 -arc_lifting_with_two_lines "0,1,2,36,54,103" "0,0" \
		-end


arc_lifting_two_lines_16_E5:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 16 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q16 -end \
		-classify_surfaces_through_arcs_and_two_lines \
		-test_nb_Eckardt_points 5 \
		-report \
		-end
	pdflatex surfaces_arc_lifting_16.tex
	open surfaces_arc_lifting_16.pdf



Family_general_T1:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 8 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 2 3 3 2 -q 8 -end 

# 13 Eckardt points


surface_atlas:
	$(ORBITER_PATH)/orbiter.out -v 3 -create_surface_atlas 17
	~/bin/tth surface_atlas.tex


surface_reports:
	$(ORBITER_PATH)/orbiter.out -v 3 -orbiter_path $(ORBITER_PATH) -create_surface_reports 17



#############################################################################################################
# Section 32: Cubic Surfaces Classification


CUBIC_SURFACES_CLASSIFICATION:


surface_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 4 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -W -end \
		-surface_classify -end

# time: 0:00


surface_4_report:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 4 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -W -end \
		-surface_classify -end \
		-report \
		-end

# -report does not do anything




arc_lifting_4:
	$(ORBITER_PATH)/orbiter.out -v 4 -define G -linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-trihedra1_control -problem_label tri1_q4 -end \
		-trihedra2_control -problem_label tri2_q4 -end \
		-control_six_arcs -problem_label sixarcs_q4 -end \
		-classify_surfaces_through_arcs_and_trihedral_pairs \
		-end
	pdflatex arc_lifting_q4.tex
	open arc_lifting_q4.pdf
	#$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix surfaces_q4_decomposition_matrix.csv 20 8

# writes surfaces_q4_decomposition_matrix.csv


arc_lifting_two_lines_4_PGL:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q4 -end \
		-classify_surfaces_through_arcs_and_two_lines \
		-report \
		-end
	pdflatex surfaces_arc_lifting_4.tex
	open surfaces_arc_lifting_4.pdf



arc_lifting_two_lines_4_PGGL:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q4 -end \
		-classify_surfaces_through_arcs_and_two_lines \
		-report \
		-end
	pdflatex surfaces_arc_lifting_4.tex
	open surfaces_arc_lifting_4.pdf





surface_7:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 7 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -W -end \
		-surface_classify -end
	




surface_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 8 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -W -end \
		-surface_classify -end

# time: 0:00



arc_lifting_8:
	$(ORBITER_PATH)/orbiter.out -v 4 -define G -linear_group -PGGL 4 8 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-trihedra1_control -problem_label tri1_q8 -end \
		-trihedra2_control -problem_label tri2_q8 -end \
		-control_six_arcs -problem_label sixarcs_q8 -end \
		-classify_surfaces_through_arcs_and_trihedral_pairs \
		-end
	#pdflatex arc_lifting_q8.tex
	#open arc_lifting_q8.pdf
	#$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix surfaces_q4_decomposition_matrix.csv 20 8

# time: 0:01


arc_lifting_two_lines_8_PGL:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGL 4 8 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q8 -end \
		-classify_surfaces_through_arcs_and_two_lines \
		-report \
		-end
	pdflatex surfaces_arc_lifting_8.tex
	open surfaces_arc_lifting_8.pdf

arc_lifting_two_lines_8_PGGL:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 8 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q8 -end \
		-classify_surfaces_through_arcs_and_two_lines \
		-report \
		-end
	pdflatex surfaces_arc_lifting_8.tex
	open surfaces_arc_lifting_8.pdf



arc_lifting_11:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 11 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-trihedra1_control -problem_label tri1_q11 -end \
		-trihedra2_control -problem_label tri2_q11 -end \
		-control_six_arcs -problem_label sixarcs_q11 -end \
		-classify_surfaces_through_arcs_and_trihedral_pairs 11 \
		-end
	pdflatex arc_lifting_q11.tex
	open arc_lifting_q11.pdf

surface_11:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 11 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -W -end \
		-surface_classify -end



arc_lifting_two_lines_11:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGL 4 11 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q11 -end \
		-classify_surfaces_through_arcs_and_two_lines \
		-report \
		-end
	pdflatex surfaces_arc_lifting_11.tex
	open surfaces_arc_lifting_11.pdf



arc_lifting_13:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 13 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-trihedra1_control -problem_label tri1_q13 -end \
		-trihedra2_control -problem_label tri2_q13 -end \
		-control_six_arcs -problem_label sixarcs_q13 -end \
		-classify_surfaces_through_arcs_and_trihedral_pairs 13 \
		-end
	pdflatex arc_lifting_q13.tex
	open arc_lifting_q13.pdf
	#$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix surfaces_q13_decomposition_matrix.csv 20 8



surface_16:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -W -problem_label "surf_16" -end \
		-surface_classify -end

# time: 0:00




arc_lifting_16:
	$(ORBITER_PATH)/orbiter.out -v 4 -define G -linear_group -PGGL 4 16 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-trihedra1_control -problem_label tri1_q16 -end \
		-trihedra2_control -problem_label tri2_q16 -end \
		-control_six_arcs -problem_label sixarcs_q16 -end \
		-classify_surfaces_through_arcs_and_trihedral_pairs \
		-end
	pdflatex arc_lifting_q16.tex
	open arc_lifting_q16.pdf
	#$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix surfaces_q4_decomposition_matrix.csv 20 8

# 5 surfaces in 0 min 17 sec on Mac 7/20/2020



arc_lifting_two_lines_16_E13:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 16 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q16 -end \
		-classify_surfaces_through_arcs_and_two_lines \
		-test_nb_Eckardt_points 13 \
		-report \
		-end
	pdflatex surfaces_arc_lifting_16.tex
	open surfaces_arc_lifting_16.pdf




surface_32:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 32 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -W -problem_label surf_32 -end \
		-surface_classify -end

# time: 0:31





arc_lifting_32:
	$(ORBITER_PATH)/orbiter.out -v 6 -define G -linear_group -PGGL 4 32 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-trihedra1_control -problem_label tri1_q32 -end \
		-trihedra2_control -problem_label tri2_q32 -end \
		-control_six_arcs --problem_label sixarcs_q32 -end \
		-classify_surfaces_through_arcs_and_trihedral_pairs \
		-end
	pdflatex arc_lifting_q32.tex
	open arc_lifting_q32.pdf
	#$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix surfaces_q4_decomposition_matrix.csv 20 8

# 9 min 25 sec on slipslide (Oct 9, 2020)
# 13 min 32 sec on Mac 
# 11 surfaces, 200 nonconical 6-arcs

# 145 double triples of type 1
# 4148 double triples of type 2
# 2184 double triples

# surface 0: 13 Eckardt points, ago = 192, orbits on single sixes 4 (Tekirdag 3)
# surface 1: 1 Eckardt points, ago = 2, orbits on single sixes 36
# surface 2: 3 Eckardt points, ago = 6, orbits on single sixes 14
# surface 3: 3 Eckardt points, ago = 6, orbits on single sixes 14
# surface 4: 1 Eckardt points, ago = 2, orbits on single sixes 36
# surface 5: 1 Eckardt points, ago = 2, orbits on single sixes 36
# surface 6: 3 Eckardt points, ago = 6, orbits on single sixes 14
# surface 7: 5 Eckardt points, ago = 80, orbits on single sixes 4
# surface 8: 0 Eckardt points, ago = 5, orbits on single sixes 16
# surface 9: 5 Eckardt points, ago = 16, orbits on single sixes 12
# surface 10: 3 Eckardt points, ago = 6, orbits on single sixes 14



arc_lifting_two_lines_32_E13:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 32 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q32 -end \
		-classify_surfaces_through_arcs_and_two_lines \
		-test_nb_Eckardt_points 13 \
		-report \
		-end
	pdflatex surfaces_arc_lifting_32.tex
	open surfaces_arc_lifting_32.pdf




surface_64:
	$(ORBITER_PATH)/orbiter.out -v 8 -define G -linear_group -PGGL 4 64 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -W -problem_label surf_64 -end \
		-surface_classify -end

# time: 38 min 10 sec
# writes surface_64.cpp
# 101 surfaces
#// the number of Eckardt points:
#static int surface_64_nb_E[] = { 
#	13, 3, 1, 1, 1, 1, 1, 1, 1, 1, 
#	1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 
#	1, 1, 1, 1, 5, 0, 0, 0, 0, 0, 
#	0, 0, 1, 0, 0, 0, 0, 1, 5, 0, 
#	0, 0, 0, 1, 1, 5, 1, 5, 5, 5, 
#	5, 1, 3, 3, 1, 1, 3, 1, 3, 1, 
#	1, 3, 1, 1, 1, 1, 1, 0, 1, 3, 
#	1, 1, 1, 0, 1, 1, 3, 0, 0, 1, 
#	3, 1, 0, 1, 1, 3, 3, 1, 3, 3, 
#	1, 3, 3, 1, 1, 3, 9, 9, 13, 13, 
#	45};

# 3 surfaces with 13 Eckardt points (Tekirdag 1 over q=8 and Kapadokya 1 and 2)
# 1 Hirschfeld with 45 Eckardt points




arc_lifting_two_lines_64_E13:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 64 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q64 -end \
		-classify_surfaces_through_arcs_and_two_lines \
		-test_nb_Eckardt_points 13 \
		-report \
		-end
	pdflatex surfaces_arc_lifting_64.tex
	open surfaces_arc_lifting_64.pdf


# q=128
# 3 surfaces with 13 Eckardt points
# 929 surfaces total


# the classification of cubic surfaces for q=128 takes 3 days on a big machine.


identify_general_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 4 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-surface_identify_general_abcd \
		-end


identify_general_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 8 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-surface_identify_general_abcd \
		-end



identify_general_16:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-surface_identify_general_abcd \
		-end


identify_general_19:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 19 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-surface_identify_general_abcd \
		-end


identify_general_29:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 29 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-surface_identify_general_abcd \
		-end



identify_general_32:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 32 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-surface_identify_general_abcd \
		-end


example_general_q16_10_11_11_10:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 10 11 11 10 -q 16 \
		-end \
		-end 
	pdflatex surface_family_general_abcd_q16_a10_b11_c11_d10_report.tex
	open surface_family_general_abcd_q16_a10_b11_c11_d10_report.pdf



example_q16_iso0:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -catalogue 0 -q 16 \
		-end \
		-end 
	pdflatex surface_catalogue_q16_iso0_report.tex
	open surface_catalogue_q16_iso0_report.pdf

example_q16_iso3:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -catalogue 3 -q 16 \
		-end \
		-end 
	pdflatex surface_catalogue_q16_iso3_report.tex
	open surface_catalogue_q16_iso3_report.pdf

# Hirschfeld in F_4:

example_q4_iso0:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 4 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -catalogue 0 -q 4 \
		-end \
		-end 
	pdflatex surface_catalogue_q4_iso0_report.tex
	open surface_catalogue_q4_iso0_report.pdf
	pdflatex surface_catalogue_q4_iso0_with_group.tex
	open surface_catalogue_q4_iso0_with_group.pdf


example_q4_iso0_relabel_E42:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 4 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -catalogue 0 -q 4 \
		-select_double_six "23,26,6,25,0,20,13,15,11,14,5,12" \
		-end \
		-end 
	pdflatex surface_catalogue_q4_iso0_report.tex
	open surface_catalogue_q4_iso0_report.pdf
	pdflatex surface_catalogue_q4_iso0_with_group.tex
	open surface_catalogue_q4_iso0_with_group.pdf

example_q4_iso0_relabel_E42_transform:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 4 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -catalogue 0 -q 4 \
		-select_double_six "23,26,6,25,0,20,13,15,11,14,5,12" \
		-transform "1,0,0,1, 0,1,0,1, 0,0,1,2, 0,0,0,1, 0" \
		-end \
		-end 
	pdflatex surface_catalogue_q4_iso0_report.tex
	open surface_catalogue_q4_iso0_report.pdf
	pdflatex surface_catalogue_q4_iso0_with_group.tex
	open surface_catalogue_q4_iso0_with_group.pdf


#Hesse group:
#1,0,0,0,0,1,0,0,0,0,1,0,3,2,3,2,0, 
#1,0,0,0,0,1,0,0,3,1,2,0,1,0,1,3,0, 
#1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1, 
#1,0,0,0,0,2,2,0,0,2,0,0,0,0,0,1,0, 
#1,0,0,0,2,3,1,0,2,0,1,0,3,1,3,1,0, 
#0,1,1,0,2,0,0,0,1,3,2,0,2,1,1,2,1, 
#1,1,0,0,0,0,3,0,3,2,2,0,1,2,3,3,1,

# Hirschfeld in F_16:

example_q16_iso4:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -catalogue 4 -q 16 \
		-end \
		-end 
	pdflatex surface_catalogue_q16_iso4_report.tex
	open surface_catalogue_q16_iso4_report.pdf
	pdflatex surface_catalogue_q16_iso4_with_group.tex
	open surface_catalogue_q16_iso4_with_group.pdf


example_q7_iso0:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 4 7 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -catalogue 0 -q 7 -end \
		-end 


example_general_q16_2_3_3_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 2 3 3 2 -q 16 \
		-end \
		-end 
	pdflatex surface_family_general_abcd_q16_a2_b3_c3_d2_report.tex
	open surface_family_general_abcd_q16_a2_b3_c3_d2_report.pdf

example_general_q16_2_7_5_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 2 7 5 2 -q 16 \
		-end \
		-end 
	pdflatex surface_family_general_abcd_q16_a2_b7_c5_d2_report.tex
	open surface_family_general_abcd_q16_a2_b7_c5_d2_report.pdf

example_general_q16_2_5_5_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 2 5 5 2 -q 16 \
		-end \
		-end 
	pdflatex surface_family_general_abcd_q16_a2_b5_c5_d2_report.tex
	open surface_family_general_abcd_q16_a2_b5_c5_d2_report.pdf

example_general_q16_2_3_9_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 16 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 2 3 9 2 -q 16 \
		-end \
		-end 
	pdflatex surface_family_general_abcd_q16_a2_b3_c9_d2_report.tex
	open surface_family_general_abcd_q16_a2_b3_c9_d2_report.pdf

example_general_q32_2_3_5_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 32 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 2 3 5 2 -q 32 \
		-end \
		-end 
	pdflatex surface_family_general_abcd_q32_a2_b3_c5_d2_report.tex
	open surface_family_general_abcd_q32_a2_b3_c5_d2_report.pdf

example_general_q32_2_3_5_18:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 4 32 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 2 3 5 18 -q 32 \
		-end \
		-end 
	pdflatex surface_family_general_abcd_q32_a2_b3_c5_d18_report.tex
	open surface_family_general_abcd_q32_a2_b3_c5_d18_report.pdf



#############################################################################################################
# Section 33:  Number Theory


NUMBER_THEORY:
	

inverse_mod:
	$(ORBITER_PATH)/orbiter.out -v 2 -inverse_mod 18059241 58014043

jacobi:
	$(ORBITER_PATH)/orbiter.out -v 5 -jacobi 2221 7817

sqrt_mod_7817:
		$(ORBITER_PATH)/orbiter.out -v 2 -square_root_mod 2221 7817

trace:
		$(ORBITER_PATH)/orbiter.out -v 2 -trace 16

equivalence_class_of_fractions:
		$(ORBITER_PATH)/orbiter.out -v 2 -equivalence_class_of_fractions 10

equivalence_class_of_fractions_draw:
	$(ORBITER_PATH)/orbiter.out -draw_matrix table_fractions_N10.csv 20 8


#############################################################################################################
# Section 34: Cryptography 


CRYPTOGRAPHY:


EC_add:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 11 -EC_add 1 3 "1,4" "1,4" -end

EC_cyclic_subgroup:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 11 -EC_cyclic_subgroup 1 3 "1,4" -end

EC_points:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 199 -EC_points 5 7 -end

EC_Koblitz_encoding:
	$(ORBITER_PATH)/orbiter.out -v 6 -seed 17 -finite_field_activity -q 199 \
		-EC_Koblitz_encoding 5 7 67 "147,164" "DEADBEEF" \
		-end

EC_bsgs:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 199 -EC_bsgs 5 7 "147,164" 212 \
		"172,158,45,195,50,22,10,103,55,33,50,22,145,105,31,74,73,155,67,60,25,6" \
		-end

EC_bsgs_decode:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 199 -EC_bsgs_decode 5 7 "129,176" 212 \
		"127,188,51,141,85,29,106,90,41,105,179,71,171,2,16,197,183,72,27,129,37,10" \
		"50,179,169,13,153,169,115,116,188,110,176" \
		-end





NTRU_N=7
NTRU_P=3
NTRU_Q=41
NTRU_D=2
NTRUE_XN1="-1,0,0,0,0,0,0,1,"
# D + 1 plus ones and D minus ones
ALICE_PRIVATE_F="-1,0,1,1,-1,0,1"
# D plus ones and D minus ones
ALICE_PRIVATE_G="0,-1,-1,0,1,0,1"

NTRU_Alice1:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q $(NTRU_Q) \
		-extended_gcd_for_polynomials $(NTRUE_XN1) $(ALICE_PRIVATE_F) -end

#F_q(x) = 8X^{6} + 26X^{5} + 31X^{4} + 21X^{3} + 40X^{2} + 2X + 37
ALICE_PRIVATE_FQ="37,2,40,21,31,26,8"


NTRU_Alice2:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q $(NTRU_P) \
		-extended_gcd_for_polynomials $(NTRUE_XN1) $(ALICE_PRIVATE_F) -end

#F_p(x) = X^{6} + 2X^{5} + X^{3} + X^{2} + X + 1
ALICE_PRIVATE_FP="1,1,1,1,0,2,1"

NTRU_Alice_public_key:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q $(NTRU_Q) \
		-polynomial_mult_mod $(ALICE_PRIVATE_F) $(ALICE_PRIVATE_G) $(NTRUE_XN1) \
		-end

#C(X)=20X^{6} + 40X^{5} + 2X^{4} + 38X^{3} + 8X^{2} + 26X + 30
ALICE_PUBLIC_KEY="30,26,8,38,2,40,20"

BOB_MESSAGE="1,-1,1,1,0,-1"
BOB_ONE_TIME_KEY="-1,1,0,0,0,-1,1"


NTRU_encrypt:
	$(ORBITER_PATH)/orbiter.out -v 2  -finite_field_activity -q $(NTRU_Q) \
		-NTRU_encrypt $(NTRU_N) $(NTRU_P)  $(ALICE_PUBLIC_KEY) \
		$(BOB_ONE_TIME_KEY) $(BOB_MESSAGE) -end

# E(X) = 31X^{6} + 19X^{5} + 4X^{4} + 2X^{3} + 40X^{2} + 3X + 25
BOB_ENCRYPT = "25,3,40,2,4,19,31"


NTRU_decrypt1:
	$(ORBITER_PATH)/orbiter.out -v 2  -finite_field_activity -q $(NTRU_Q) \
		-polynomial_mult_mod $(ALICE_PRIVATE_F) $(BOB_ENCRYPT) $(NTRUE_XN1) \
		-end

#C(X)=X^{6} + 10X^{5} + 33X^{4} + 40X^{3} + 40X^{2} + X + 40
ALICE_C1="40,1,40,40,33,10,1"

NTRU_decrypt2:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q $(NTRU_Q) \
		-polynomial_center_lift $(ALICE_C1) -end

#A(X)=X^{6} + 10X^{5} - 8X^{4} - X^{3} - X^{2} + X - 1
ALICE_C2="-1,1,-1,-1,-8,10,1"

NTRU_decrypt3:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q $(NTRU_P) \
		-polynomial_reduce_mod_p $(ALICE_C2) -end

#A(X)=X^{6} + X^{5} + X^{4} + 2X^{3} + 2X^{2} + X + 2
ALICE_C3="2,1,2,2,1,1,1"

NTRU_decrypt4:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q $(NTRU_Q) \
		-polynomial_mult_mod $(ALICE_PRIVATE_FP) $(ALICE_C3) $(NTRUE_XN1) \
		-end

#C(X)=2X^{5} + X^{3} + X^{2} + 2X + 1
ALICE_C4="1,2,1,1,0,2"


NTRU_decrypt5:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q $(NTRU_P) \
		-polynomial_center_lift $(ALICE_C4) -end

#A(X)= - X^{5} + X^{3} + X^{2} - X + 1
#plaintext BOB_MESSAGE




#############################################################################################################
# Section 35: Coding Theory

CODING_THEORY:

BCH15:
	$(ORBITER_PATH)/orbiter.out -BCH 15 2 3 
	$(ORBITER_PATH)/orbiter.out -BCH 15 2 5 
	$(ORBITER_PATH)/orbiter.out -BCH 15 2 7 
	$(ORBITER_PATH)/orbiter.out -BCH 15 2 9 


BCH11:
	$(ORBITER_PATH)/orbiter.out -BCH 11 2 3 
	$(ORBITER_PATH)/orbiter.out -BCH 11 2 5 


BCH13:
	$(ORBITER_PATH)/orbiter.out -BCH 13 2 3 
	$(ORBITER_PATH)/orbiter.out -BCH 13 2 5 


BCH7:
	$(ORBITER_PATH)/orbiter.out -BCH 7 2 3 
	$(ORBITER_PATH)/orbiter.out -BCH 7 2 5 


BCH21:
	$(ORBITER_PATH)/orbiter.out -BCH 21 2 3 
	$(ORBITER_PATH)/orbiter.out -BCH 21 2 5 
	$(ORBITER_PATH)/orbiter.out -BCH 21 2 7 
	$(ORBITER_PATH)/orbiter.out -BCH_dual 21 2 7 



Hamming_4_2:
	$(ORBITER_PATH)/orbiter.out -Hamming_graph 4 2
	$(ORBITER_PATH)/orbiter.out -draw_matrix Hamming_n4_q2.csv 20 8
	open Hamming_n4_q2_draw.bmp



codes_8_4_4:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-orbiter_path $(ORBITER_PATH) \
		-define G \
		-linear_group -PGL 4 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label codes_8_4_4 \
			-draw_poset \
			-draw_options -embedded -rad 250 -line_width 1.0 -spanning_tree -end \
			-report \
		-end \
		-linear_codes 3 8 \
		-end
	pdflatex codes_8_4_4_poset_lvl_8.tex
	open codes_8_4_4_poset_lvl_8.pdf
	pdflatex codes_8_4_4_poset.tex
	open codes_8_4_4_poset.pdf

codes_8_4_4_draw:
	$(ORBITER_PATH)/orbiter.out -v 3 -draw_layered_graph codes_8_4_4_poset_lvl_8.layered_graph \
		-rad 250 -embedded -line_width 1.0  -y_stretch 1.0 -scale 0.5 \
		-end
	pdflatex codes_8_4_4_poset_lvl_8_draw.tex
	open codes_8_4_4_poset_lvl_8_draw.pdf



codes_14_4_9_3:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-define G \
		-linear_group -PGL 10 3 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label codes_14_4_9_3 \
			-draw_poset -draw_options -embedded -rad 250 \
			-end \
		-end \
		-linear_codes 9 14 \
		-end
	pdflatex codes_14_4_9_3_poset_lvl_13.tex
	open codes_14_4_9_3_poset_lvl_13.pdf 




codes_16_5_9_3:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-draw_options -end \
		-codes_classify -n 16 -k 5 -q 3 -d 9 -w -W -lex \
		-draw_poset \
		-end
		

# 5/31/2020: 28 min 22 sec on Mac

#0 : 1 orbits
#1 : 1 orbits
#2 : 1 orbits
#3 : 1 orbits
#4 : 1 orbits
#5 : 1 orbits
#6 : 1 orbits
#7 : 1 orbits
#8 : 1 orbits
#9 : 2 orbits
#10 : 3 orbits
#11 : 4 orbits
#12 : 5 orbits
#13 : 5 orbits
#14 : 4 orbits
#15 : 3 orbits
#16 : 1 orbits
#total: 36


codes_d4:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 4 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -W \
		-problem_label codes_r4_d4 -draw_poset \
		-embedded -end -linear_codes 4 100 \
		-end \
		-end


codes_24_12_8:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-orbiter_path $(ORBITER_PATH) \
		-define G \
		-linear_group -PGL 12 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label codes_24_12_8 \
			-draw_poset \
			-draw_options -embedded -rad 250 -line_width 1.0 -spanning_tree -end \
			-report \
		-end \
		-linear_codes 8 24 \
		-end
	pdflatex codes_24_12_8_poset.tex
	open codes_24_12_8_poset.pdf


code_6:
	$(ORBITER_PATH)/orbiter.out -v 2 -general_code_binary 6 "0,60,50,41,14,21,27,39"
	$(ORBITER_PATH)/orbiter.out -draw_matrix code_matrix_8_8.csv 20 25 \
		-draw_matrix_partition 2 "1,1,1,1,1,1,1,1" "1,1,1,1,1,1,1,1" 
	pdflatex code_6_8.tex
	open code_6_8.pdf
	open code_matrix_8_8_draw.bmp


# linear code with generator matrix
# 111100
# 110010
# 101001




code_Hamming:
	$(ORBITER_PATH)/orbiter.out -v 2 -linear_code_through_basis 7  "67,37,22,15"
	$(ORBITER_PATH)/orbiter.out -draw_matrix code_matrix_16_8.csv 25 8 -draw_matrix_partition 2 16 8
	open code_matrix_16_8_draw.bmp
	pdflatex code_7_16.tex
	open code_7_16.pdf


#	$(DRAW_PATH)/draw_matrix.out -read code_matrix_256_256.csv -box_width 10 

# 1 0 0 0 0 1 1 = 67
# 0 1 0 0 1 0 1 = 37
# 0 0 1 0 1 1 0 = 22
# 0 0 0 1 1 1 1 = 15


RM_6:
	$(ORBITER_PATH)/orbiter.out -v 2 -long_code 64 7 \
		"0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63" \
		"1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63" \
		"2,3,6,7,18,19,22,23,10,11,14,15,26,27,30,31,34,35,38,39,42,43,46,47,50,51,54,55,58,59,62,63" \
		"4,6,12,14,36,38,52,54,5,7,13,15,37,39,53,55,20,22,28,30,44,46,60,62,21,23,29,31,45,47,61,63" \
		"8,9,12,13,24,25,28,29,10,11,14,15,26,27,30,31,40,41,44,45,56,57,60,61,42,43,46,47,58,59,62,63" \
		"16,18,24,26,48,50,56,58,17,19,25,27,49,51,57,59,20,22,28,30,52,54,60,62,21,23,29,31,53,55,61,63" \
		"32,34,48,50,33,35,49,51,36,38,52,54,37,39,53,55,40,42,56,58,41,43,57,59,44,46,60,62,45,47,61,63" 
	$(ORBITER_PATH)/orbiter.out -draw_matrix long_code_genma_n64_k7.csv 25 8 -draw_matrix_partition 3 7 64
	$(ORBITER_PATH)/orbiter.out -draw_matrix long_code_genma_n64_k7_codeword_0.csv 25 8 -draw_matrix_partition 3 8 8
	$(ORBITER_PATH)/orbiter.out -draw_matrix long_code_genma_n64_k7_codeword_1.csv 25 8 -draw_matrix_partition 3 8 8
	$(ORBITER_PATH)/orbiter.out -draw_matrix long_code_genma_n64_k7_codeword_2.csv 25 8 -draw_matrix_partition 3 8 8		


Reed_Muller_6:
	$(ORBITER_PATH)/orbiter.out -v 2 -long_code 64 7 \
		-set_builder -loop 0 64 1 -end \
		-set_builder -loop 0 32 1 -affine_function 2 1 -end \
		-set_builder -loop 0 16 1 -affine_function 4 2 -clone_with_affine_function 4 3 -end \
		-set_builder -set_builder -set_builder -loop 0 4 1 -affine_function 1 4 \
			-clone_with_affine_function 1 12 -end -clone_with_affine_function 1 16 \
			-end -clone_with_affine_function 1 32 -end\
		-set_builder -set_builder -loop 0 8 1 -affine_function 1 8 \
			-clone_with_affine_function 1 24 -end -clone_with_affine_function 1 32 -end\
		-set_builder -loop 0 16 1 -affine_function 1 16 -clone_with_affine_function 1 48 -end \
		-set_builder -loop 0 32 1 -affine_function 1 32 -end 
	$(ORBITER_PATH)/orbiter.out -draw_matrix long_code_genma_n64_k7.csv 25 8 -draw_matrix_partition 3 7 64
	open long_code_genma_n64_k7_draw.bmp




#-nearest_codeword "8,16,32,24,40,48,56,1,2,4,3,5,6,7,9,18,36,27,45,54,63,33,42,20,11,53,62,31"




########################################################################################################
# Section 36: Diophantine Systems

DIOPHANT:
	



part10:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -label part10  \
		-coefficient_matrix 1 10 "10,9,8,7,6,5,4,3,2,1" -RHS "10,10,1" -x_min_global 0 -x_max_global 10

part10_again:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -from_scratch part10 1 10 \
		-coefficient_matrix_csv part10_coeff_matrix.csv \
		-RHS_csv part10_RHS.csv \
		-x_bounds_csv part10_x_bounds.csv
	

part10_solve:
	$(ORBITER_PATH)/orbiter.out -v 10 -diophant_activity -input_file part10.diophant -solve_mckay 

# Found 42 solutions with 67 backtrack steps


octic_monomials:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -label octic_monomials  \
		-coefficient_matrix 1 4 "1,1,1,1" -RHS "8,8,1" -x_min_global 0 -x_max_global 8


octic_monomials_solve:
	$(ORBITER_PATH)/orbiter.out -v 3 -diophant_activity -input_file octic_monomials.diophant -solve_mckay 
	sort -r octic_monomials.sol >octic_monomials_sorted.txt

#Found 165 solutions with 210 backtrack steps
# 165=binomial(11,3)






########################################################################################################
# Section 37: Combinatorial Linear Spaces

COMBINATORIAL_LINEAR_SPACES:
	

linsp7:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -label linsp7  \
		-coefficient_matrix 1 6 "21,15,10,6,3,1" -RHS "21,21,1" -x_min_global 0 -x_max_global 21 
	
linsp7_solve:
	$(ORBITER_PATH)/orbiter.out -v 10 -diophant_activity -input_file linsp7.diophant -solve_mckay 

# 32 solutions in 45 backtrack steps

linsp7_again:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -label linsp7  \
		-coefficient_matrix_csv linsp7_coeff_matrix.csv \
		-RHS_csv linsp7_RHS.csv \
		-x_bounds_csv linsp7_x_bounds.csv
	


linsp6:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -label linsp6  \
		-coefficient_matrix 1 5 "15,10,6,3,1" -RHS "15,15,1" -x_min_global 0 -x_max_global 15
	
linsp6_solve:
	$(ORBITER_PATH)/orbiter.out -v 3 -diophant_activity -input_file linsp6.diophant -solve_mckay 

# Found 15 solutions with 22 backtrack steps





linsp30_pt_types:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -label linsp30_pt_types \
		-coefficient_matrix 1 3 "6,4,3" -RHS "29,29,1" -x_bounds "0,1,0,27,0,24"
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file linsp30_pt_types.diophant -solve_mckay 

linsp30_pt_distribution:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -label linsp30_pt_distribution  \
		-coefficient_matrix 6 4 \
			"1,1,1,1,1,1,0,0,5,2,5,2,1,5,3,7,10,1,10,1,0,10,3,21" \
			-RHS "30,30,1,7,7,1,135,135,1,96,96,1,0,351,2,0,276,2" \
			-x_min_global 0 -x_max_global 30
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file \
		linsp30_pt_distribution.diophant -solve_mckay 
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file \
		linsp30_pt_distribution.diophant -draw_as_bitmap 20 8


#############################################################################################################
# Section 38: Design Theory

DESIGN_THEORY:


design_PG_2_3:
	$(ORBITER_PATH)/orbiter.out -v 8  \
		-create_design -q 3 -family PG_2_q  -end \

design_PG_2_4:
	$(ORBITER_PATH)/orbiter.out -v 8  \
		-create_design -q 4 -family PG_2_q  -end \



# to create simple 7-designs on 33 points with block size 8 and lambda = 10 invariant under PGGL(2,32):

KM_PGGL_2_32:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGGL 2 32 -end -end \
			-with G -do \
			-group_theoretic_activities \
			-poset_classification_control -problem_label KM_PGGL_2_32 -W -depth 8 \
				-Kramer_Mesner_matrix 7 8 \
				-draw_poset  \
				-draw_options -embedded -sideways -rad 50 -scale 0.5 -line_width 0.3 -end \
			-end \
			-orbits_on_subsets 8 \
			-end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix KM_PGGL_2_32_KM_7_8.csv 20 8 \
		-draw_matrix_partition 3 32 97
	pdflatex KM_PGGL_2_32_poset_lvl_8.tex
	open KM_PGGL_2_32_poset_lvl_8.pdf
	open KM_PGGL_2_32_KM_7_8_draw.bmp
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant \
		-label "KM_PGGL_2_32_KM_7_8_system" -coefficient_matrix_csv KM_PGGL_2_32_KM_7_8.csv  \
		-RHS_constant "10,10,1" \
		-x_min_global 0 -x_max_global 1
		
		
KM_PGGL_2_32_solve:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file \
		KM_PGGL_2_32_KM_7_8_system.diophant -solve_mckay 

#			-report



KM_PSL_3_5:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PSL 3 5 -end -end \
			-with G -do \
			-group_theoretic_activities \
			-poset_classification_control -problem_label KM_PSL_3_5 -W -depth 10 \
				-Kramer_Mesner_matrix 8 10 \
				-draw_poset \
				-draw_options -embedded -sideways -rad 50 -scale 0.5 -line_width 0.3 \
			-end \
			-orbits_on_subsets 10
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix KM_PSL_3_5_KM_8_10.csv 10 8 \
		-draw_matrix_partition 3 42 174
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant \
		-label "KM_PSL_3_5_KM_8_10_system" -coefficient_matrix_csv KM_PSL_3_5_KM_8_10.csv  \
		-RHS_constant "93,93,1" \
		-x_min_global 0 -x_max_global 1
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file \
		KM_PSL_3_5_KM_8_10_system.diophant -solve_mckay 





conf_10_3:
	$(ORBITER_PATH)/orbiter.out -v 2 -regular_linear_space_classify -m 10 -n 10 -r 3 -k 3 -target_size 10  \
		-control -problem_label 10_3 -W -end -end
	
conf_11_3:
	$(ORBITER_PATH)/orbiter.out -v 2 -regular_linear_space_classify -m 11 -n 11 -r 3 -k 3 -target_size 11  \
		-control -problem_label 11_3 -w -end -end
	
conf_12_3:
	$(ORBITER_PATH)/orbiter.out -v 2 -regular_linear_space_classify -m 12 -n 12 -r 3 -k 3 -target_size 12  \
		-control -problem_label 12_3 -w -end -end
	
conf_18_3:
	$(ORBITER_PATH)/orbiter.out -v 2 -regular_linear_space_classify -m 18 -n 18 -r 3 -k 3 -target_size 18  \
		-control -problem_label 18_3 -W -depth 8 -end -end
	
	
	
STS13:
	$(ORBITER_PATH)/orbiter.out -v 2 -regular_linear_space_classify -m 13 -n 26 -r 6 -k 3 -target_size 26  \
		-control -problem_label STS13 -w -depth 7 -end -end
	
	




#############################################################################################################
# Section 39: Tactical Decompositions

TACTICAL_DECOMPOSITIONS:

max_arc_16_4_start:
	$(ORBITER_PATH)/orbiter.out -v 4 -maximal_arc_parameters 16 4


max_arc_16_4_convert_stack_tdo:
	$(ORBITER_PATH)/orbiter.out -v 4 -convert_stack_to_tdo max_arc_q16_r4.stack

max_arc_16_4_refine:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_refinement -input_file max_arc_q16_r4.tdo -dual_is_linear_space -end
	
max_arc_16_4r_print:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_print max_arc_q16_r4r.tdo 

max_arc_16_4b_convert_stack_tdo:
	$(ORBITER_PATH)/orbiter.out -v 4 -convert_stack_to_tdo max_arc_q16_r4b.stack

max_arc_16_4b_refine:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_refinement -input_file max_arc_q16_r4b.tdo -dual_is_linear_space -end
	
max_arc_16_4br_print:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_print max_arc_q16_r4br.tdo 


max_arc_16_4bd_convert_stack_tdo:
	$(ORBITER_PATH)/orbiter.out -v 4 -convert_stack_to_tdo max_arc_q16_r4bd.stack

max_arc_16_4bd_refine:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_refinement -input_file max_arc_q16_r4bd.tdo -dual_is_linear_space -end
	
max_arc_16_4bdr_print:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_print max_arc_q16_r4bdr.tdo 

max_arc_16_4bdr_refine:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_refinement -input_file max_arc_q16_r4bdr.tdo -dual_is_linear_space -end
	
max_arc_16_4bdrr_print:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_print max_arc_q16_r4bdrr.tdo 

max_arc_16_4bdrr_refine:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_refinement -input_file max_arc_q16_r4bdrr.tdo -dual_is_linear_space -end
	


max_arc_flag_stabilizer:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGGL 3 16 -subgroup_by_generators "flag_stab" 3686400 16 \
		"-1,0,0,0,1,0,0,0,1,0" \
		"1,0,0,0,-1,0,0,0,1,0" \
		"1,0,0,0,1,0,0,0,-1,0" \
		"1,0,0,1,1,0,0,0,1,0" \
		"1,0,0,2,1,0,0,0,1,0" \
		"1,0,0,4,1,0,0,0,1,0" \
		"1,0,0,8,1,0,0,0,1,0" \
		"1,0,0,0,1,0,1,0,1,0" \
		"1,0,0,0,1,0,2,0,1,0" \
		"1,0,0,0,1,0,4,0,1,0" \
		"1,0,0,0,1,0,8,0,1,0" \
		"1,0,0,0,1,0,0,1,1,0" \
		"1,0,0,0,1,0,0,2,1,0" \
		"1,0,0,0,1,0,0,4,1,0" \
		"1,0,0,0,1,0,0,8,1,0" \
		"1,0,0,0,1,0,0,0,1,1" \
		-end -end \
		-with G -do \
		-group_theoretic_activities -report \
		-end
	pdflatex PGGL_3_16_Subgroup_flag_stab_3686400_report.tex
	open PGGL_3_16_Subgroup_flag_stab_3686400_report.pdf
	


max_arc_flag_stabilizer_on_pencil:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGGL 3 16 -subgroup_by_generators "flag_stab" 3686400 16 \
		"-1,0,0,0,1,0,0,0,1,0" \
		"1,0,0,0,-1,0,0,0,1,0" \
		"1,0,0,0,1,0,0,0,-1,0" \
		"1,0,0,1,1,0,0,0,1,0" \
		"1,0,0,2,1,0,0,0,1,0" \
		"1,0,0,4,1,0,0,0,1,0" \
		"1,0,0,8,1,0,0,0,1,0" \
		"1,0,0,0,1,0,1,0,1,0" \
		"1,0,0,0,1,0,2,0,1,0" \
		"1,0,0,0,1,0,4,0,1,0" \
		"1,0,0,0,1,0,8,0,1,0" \
		"1,0,0,0,1,0,0,1,1,0" \
		"1,0,0,0,1,0,0,2,1,0" \
		"1,0,0,0,1,0,0,4,1,0" \
		"1,0,0,0,1,0,0,8,1,0" \
		"1,0,0,0,1,0,0,0,1,1" \
		-on_k_subspaces 2 \
		-restricted_action "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16" \
		-end -end \
		-with G -do \
		-group_theoretic_activities -report \
		-orbits_on_subsets 3 -end 
	


# lines through 0:
#0 & 0 & 1 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 & 14 & 15 & 16 & 17 & 18\\
#1 & 0 & 34 & 3 & 35 & 36 & 37 & 38 & 39 & 40 & 41 & 42 & 43 & 44 & 45 & 46 & 47 & 48\\
#2 & 0 & 209 & 221 & 210 & 222 & 211 & 223 & 212 & 224 & 213 & 217 & 214 & 218 & 215 & 219 & 216 & 220\\
#3 & 0 & 145 & 153 & 154 & 146 & 156 & 148 & 147 & 155 & 160 & 152 & 151 & 159 & 149 & 157 & 158 & 150\\
#4 & 0 & 113 & 119 & 125 & 123 & 114 & 120 & 126 & 124 & 115 & 117 & 127 & 121 & 116 & 118 & 128 & 122\\
#5 & 0 & 257 & 272 & 264 & 265 & 271 & 258 & 266 & 263 & 262 & 267 & 259 & 270 & 268 & 261 & 269 & 260\\
#6 & 0 & 81 & 85 & 89 & 93 & 90 & 94 & 82 & 86 & 92 & 96 & 84 & 88 & 83 & 87 & 91 & 95\\
#7 & 0 & 241 & 255 & 246 & 252 & 251 & 245 & 256 & 242 & 254 & 244 & 249 & 247 & 248 & 250 & 243 & 253\\
#8 & 0 & 65 & 68 & 71 & 70 & 77 & 80 & 75 & 74 & 66 & 67 & 72 & 69 & 78 & 79 & 76 & 73\\
#9 & 0 & 225 & 238 & 228 & 239 & 231 & 236 & 230 & 233 & 237 & 226 & 240 & 227 & 235 & 232 & 234 & 229\\
#10 & 0 & 193 & 204 & 208 & 197 & 200 & 205 & 201 & 196 & 207 & 198 & 194 & 203 & 202 & 195 & 199 & 206\\
#11 & 0 & 177 & 187 & 190 & 184 & 180 & 186 & 191 & 181 & 183 & 189 & 188 & 178 & 182 & 192 & 185 & 179\\
#12 & 0 & 49 & 51 & 53 & 55 & 57 & 59 & 61 & 63 & 58 & 60 & 62 & 64 & 50 & 52 & 54 & 56\\
#13 & 0 & 161 & 170 & 172 & 163 & 176 & 167 & 165 & 174 & 168 & 175 & 173 & 166 & 169 & 162 & 164 & 171\\
#14 & 0 & 129 & 136 & 143 & 138 & 134 & 131 & 140 & 141 & 139 & 142 & 133 & 132 & 144 & 137 & 130 & 135\\
#15 & 0 & 97 & 102 & 107 & 112 & 110 & 105 & 104 & 99 & 100 & 103 & 106 & 109 & 111 & 108 & 101 & 98\\
#16 & 0 & 2 & 19 & 20 & 21 & 22 & 23 & 24 & 25 & 26 & 27 & 28 & 29 & 30 & 31 & 32 & 33\\


max_arc_flag_stabilizer_pencil1:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGGL 3 16 -subgroup_by_generators "pencil1_group" 7680 5 \
		"1,0,0,0,9,0,0,0,9,0," \
		"1,0,0,0,1,0,0,0,8,2," \
		"1,0,0,0,1,0,3,0,1,0," \
		"1,0,0,0,7,0,2,0,7,0," \
		"1,0,0,4,1,0,13,0,1,0," \
		-restricted_action "113,119,125,123,114,120,126,124,115,117,127,121,116,118,128,122,257,272,264,265,271,258,266,263,262,267,259,270,268,261,269,260,81,85,89,93,90,94,82,86,92,96,84,88,83,87,91,95,241,255,246,252,251,245,256,242,254,244,249,247,248,250,243,253,65,68,71,70,77,80,75,74,66,67,72,69,78,79,76,73,225,238,228,239,231,236,230,233,237,226,240,227,235,232,234,229,193,204,208,197,200,205,201,196,207,198,194,203,202,195,199,206,177,187,190,184,180,186,191,181,183,189,188,178,182,192,185,179,49,51,53,55,57,59,61,63,58,60,62,64,50,52,54,56,161,170,172,163,176,167,165,174,168,175,173,166,169,162,164,171,129,136,143,138,134,131,140,141,139,142,133,132,144,137,130,135,97,102,107,112,110,105,104,99,100,103,106,109,111,108,101,98,2,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33" \
		-end -end \
		-with G -do \
		-group_theoretic_activities -report \
		-orbits_on_subsets 5 -end 


max_arc_flag_stabilizer_of_two_pencils:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -PGGL 3 16 -subgroup_by_generators "flags_stab" 460800 13 \
		"-1,0,0,0,1,0,0,0,1,0" \
		"1,0,0,0,-1,0,0,0,1,0" \
		"1,0,0,0,1,0,0,0,-1,0" \
		"1,0,0,0,1,0,1,0,1,0" \
		"1,0,0,0,1,0,2,0,1,0" \
		"1,0,0,0,1,0,4,0,1,0" \
		"1,0,0,0,1,0,8,0,1,0" \
		"1,0,0,0,1,0,0,1,1,0" \
		"1,0,0,0,1,0,0,2,1,0" \
		"1,0,0,0,1,0,0,4,1,0" \
		"1,0,0,0,1,0,0,8,1,0" \
		"1,0,0,0,1,0,0,0,1,1" \
		"0,1,0,1,0,0,0,0,1,0" \
		-on_k_subspaces 2 \
		-restricted_action "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255,272" \
		-end -end \
		-with G -do \
		-group_theoretic_activities -report \
		-orbits_on_subsets 6 -end 

max_arc_system_1:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -q 16 -maximal_arc 52 4 \
		"1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255,272,0" \
		"0,1,2,16,17,18,32"

max_arc_system_1_solve:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -solve_mckay 


max_arc_system_1_draw:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -draw 
	mpost max_arc_16_52_4_drawing.mp
	latex view.tex
	dvips view.dvi -o
	open view.ps

max_arc_system_1_red:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -perform_column_reductions 

max_arc_system_1_red_draw:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4_red.diophant -draw 
	mpost max_arc_16_52_4_red_drawing.mp
	latex view2.tex
	dvips view2.dvi -o
	open view2.ps
	
max_arc_system_1_red_draw_bitmap:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4_red.diophant -draw_as_bitmap 5 9


max_arc_system_1_red_solve:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4_red.diophant -solve_mckay 


max_arc_system_1_equations:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -test_single_equation 20 

max_arc_system_1_split273:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -project_to_single_equation_and_solve 273 0 

max_arc_system_1_split0:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -project_to_single_equation_and_solve 0 1
	
max_arc_system_1_split0_1:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -project_to_two_equations_and_solve 0 1 0 100000





##########################









max_arc_16_4:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 3 16 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
			-end \
			-classify_arcs 52 \
			-classify_arcs_d 4  \
			-exact_cover \
				-input_prefix ./ARCS/ \
				-output_prefix ./SYSTEMS/ \
				-solution_prefix ./SOLUTIONS/ \
				-starter_size 52 \
				-base_fname maxarcs_q16_d4 \
				-lex \
			-end \
		-end

max_arc_lift:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-process_combinatorial_objects -q 16 -n 2 \
		-arc_with_given_set_as_s_lines_after_dualizing \
			52 4 4 0 \
		-fname_base_out maxarcs_q16_d4_lvl_9_lifted \
		-input \
			-file_of_points maxarcs_q16_d4_lvl_9 \
		-end \
		-end 



#############################################################################################################
# Section 40: Spreads

SPREADS:
	
desarguesian_spread_in_PG_3_2:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-define FQ -finite_field -q 4 -end -end \
		-define Fq -finite_field -q 2 -end -end \
		-with FQ -and Fq -do -finite_field_activity -cheat_sheet_desarguesian_spread 2 -end
	pdflatex Desarguesian_Spread_3_2.tex
	open Desarguesian_Spread_3_2.pdf

desarguesian_spread_in_PG_5_2:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-define FQ -finite_field -q 8 -end -end \
		-define Fq -finite_field -q 2 -end -end \
		-with FQ -and Fq -do -finite_field_activity -cheat_sheet_desarguesian_spread 2 -end
	pdflatex Desarguesian_Spread_5_2.tex
	open Desarguesian_Spread_5_2.pdf

desarguesian_spread_in_PG_3_4:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-define FQ -finite_field -q 16 -end -end \
		-define Fq -finite_field -q 4 -end -end \
		-with FQ -and Fq -do -finite_field_activity -cheat_sheet_desarguesian_spread 2 -end
	pdflatex Desarguesian_Spread_3_4.tex
	open Desarguesian_Spread_3_4.pdf


spreads4:
	- mkdir SPREADS_4
	- rm live_points.txt
	$(ORBITER_PATH)/orbiter.out -v 10 \
		-define G \
		-linear_group -PGL 4 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_4 \
			-end \
			-spread_classify 2 \
		-end

spreads16_4:
	- mkdir SPREADS_16_4
	- rm live_points.txt
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_16_4 \
			-end \
			-spread_classify 2 \
		-end

#17 : 3 orbits
#total: 1129
#(81600, 1200, 576) average is 27792 + 0 / 3


spreads64_4:
	- mkdir SPREADS_64_4
	- rm live_points.txt
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-define G \
		-linear_group -PGGL 6 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_64_4 \
			-end \
			-spread_classify 3 \
		-end


# 853 orbits at level 5, each with about 60k live points

spreads64_8:
	- mkdir SPREADS_64_8
	- rm live_points.txt
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-define G \
		-linear_group -PGGL 4 8 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_64_8 \
			-end \
			-spread_classify 2 \
		-end


# 16766 orbits at level 6, each with about 2k live points


spreads16:
	- mkdir SPREADS_16
	- rm live_points.txt
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGL 8 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_16 \
			-end \
			-spread_classify 4 \
		-end

# 8 spreads of order 16
# May 14, 2020: 52 sec on Mac


line_spreads_PG5_2:
	- rm live_points.txt
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-define G \
		-linear_group -PGL 6 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label line_spreads_PG_5_2 \
				-depth 21 \
			-end \
			-spread_classify 2 \
		-end


# 16167 nodes at level 7
# 357991 nodes at levl 8 with about 150 live points each; 52371304 extension nodes





#############################################################################################################
# Section 41: Packings


PACKINGS:


packing:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0,1,2" "SPREAD_TABLES_4/" \
		-end


# elements of order 2:
# conjugacy class reps:
# elt order, class size, centralizer order 

#2A: 2 48960 40320 Baer involution

#2B: 2 5355 368640  one block of 10,11

#2C: 2 64260 30720 two blocks of 10,11 (problem group)



# Baer involution:

packing_2A:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0,1,2" "SPREAD_TABLES_4/" \
			-packing_with_assumed_symmetry \
				-H -PGGL 4 4 -subgroup_by_generators "2A" 2 1 \
					"1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1" \
				-end \
				-N -PGGL 4 4 -subgroup_by_generators "centralizer_2A" "40320" 10 \
					"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
					"1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1," \
					"1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1," \
					"1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0," \
					"1,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0," \
					"1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1," \
					"1,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,0," \
					"1,0,0,0,1,1,1,1,1,0,1,0,1,1,1,0,0," \
					"1,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,0," \
					"0,1,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1," \
				-end \
				-cliques_on_fixpoint_graph 7 \
				-cliques_on_fixpoint_graph_control -W -problem_label PGGL_4_4_Subgroup_2A_2_fixp_graph_cliques -end \
				-process_long_orbits 7 0 1000 2 7 \
			-end \
		-end

# creates PGGL_4_4_Subgroup_2A_2_fpc0_graph and many more


packing_2Ac0_draw:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_2A_2_fpc0_graph -end \
		-graph_theoretic_activity -export_csv PGGL_4_4_Subgroup_2A_2_fpc0_graph.csv -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGGL_4_4_Subgroup_2A_2_fpc0_graph.csv 2 8


packing_2Ac0_search:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_2A_2_fpc0_graph -end \
		-graph_theoretic_activity -find_cliques -target_size 7 -end -end

# 578 solutions in 0:26 on Mac


packing_2Ac0_search_Sajeeb:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_2A_2_fpc0_graph -end \
		-graph_theoretic_activity -find_cliques -Sajeeb -target_size 7 -end -end




packing_2A_draw:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_2A_2_fixp_graph.bin -end \
		-graph_theoretic_activity -export_csv PGGL_4_4_Subgroup_2A_2_fixp_graph.csv -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGGL_4_4_Subgroup_2A_2_fixp_graph.bin.csv 2 8

# 440 orbits on cliques of size 7 on the fixpoint graph

#orbit_transversal::read_from_file fname = G PGGL_4_4_Subgroup_2A_2_fixp_graph_cliques_lvl_7 
#		reduced_spread_orbits_under_H->Orbits_classified->Set_size[fixpoints_idx],
#		reduced_spread_orbits_under_H->Orbits_classified->Sets[fixpoints_idx],


#orbits_on_something::classify_orbits_by_length The distribution of orbit lengths is: 1^1680, 2^90720
#1:17
# writes PGGL_4_4_Subgroup_2A_2_fixp_graph.bin
#-rw-r--r--   1 betten  staff       5886 Jun 11 00:31 PGGL_4_4_Subgroup_2A_2_line_orbits_orbits.bin
#-rw-r--r--   1 betten  staff   81549922 Jun 11 01:52 PGGL_4_4_Subgroup_2A_2_spread_orbits_orbits.bin
#-rw-r--r--   1 betten  staff    1440222 Jun 11 01:52 PGGL_4_4_Subgroup_2A_2_good_orbits
#-rw-r--r--   1 betten  staff    2936674 Jun 11 01:52 PGGL_4_4_Subgroup_2A_2_reduced_spread_orbits_orbits.bin
#-rw-r--r--   1 betten  staff     196479 Jun 11 01:52 PGGL_4_4_Subgroup_2A_2_fixp_graph.bin
#-rw-r--r--@  1 betten  staff   35192935 Jun 11 01:52 log




# the problem group:

packing_2C_prepare:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0,1,2" "SPREAD_TABLES_4/" \
			-packing_with_assumed_symmetry \
				-H -PGGL 4 4 -subgroup_by_generators "2C" 2 1 \
					"1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0" \
				-end \
				-N -PGGL 4 4 -subgroup_by_generators "centralizer_2C" "30720" 9 \
					"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
					"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,1," \
					"1,0,0,0,0,1,0,0,0,0,1,0,2,0,3,1,0," \
					"1,0,0,0,0,1,0,0,0,0,1,0,1,0,3,1,0," \
					"1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1," \
					"1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0," \
					"1,0,0,0,2,1,0,0,0,0,1,0,1,0,0,1,0," \
					"1,0,0,0,1,1,2,0,0,0,1,0,0,0,0,1,0," \
					"1,0,3,0,1,1,1,3,0,0,2,0,0,0,0,2,1," \
				-end \
				-cliques_on_fixpoint_graph 5 \
				-cliques_on_fixpoint_graph_control -W -problem_label PGGL_4_4_Subgroup_2C_2_fixp_graph_cliques -end \
				-fixp_clique_types_save_individually \
			-end \
		-end

# computed on ripoff:
#depth : number of orbits
#0 : 1
#1 : 8
#2 : 426
#3 : 70315
#4 : 1557542
#5 : 776928
#6 : 0
#7 : 0
# creates the file PGGL_4_4_Subgroup_2C_2_fixp_graph_cliques.csv
# creates PGGL_4_4_Subgroup_2C_2_fpc0_graph


# 8268 cases of type 5,0,0

# the next command creates the graphs for a slice 
# of the problem of extending the fixpoint cliques by long orbits:


packing_2C_500_make_graphs:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0,1,2" "SPREAD_TABLES_4/" \
			-packing_with_assumed_symmetry \
				-H -PGGL 4 4 -subgroup_by_generators "2C" 2 1 \
					"1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0" \
				-end \
				-N -PGGL 4 4 -subgroup_by_generators "centralizer_2C" "30720" 9 \
					"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
					"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,1," \
					"1,0,0,0,0,1,0,0,0,0,1,0,2,0,3,1,0," \
					"1,0,0,0,0,1,0,0,0,0,1,0,1,0,3,1,0," \
					"1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1," \
					"1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0," \
					"1,0,0,0,2,1,0,0,0,0,1,0,1,0,0,1,0," \
					"1,0,0,0,1,1,2,0,0,0,1,0,0,0,0,1,0," \
					"1,0,3,0,1,1,1,3,0,0,2,0,0,0,0,2,1," \
				-end \
				-cliques_on_fixpoint_graph 5 \
				-cliques_on_fixpoint_graph_control -W -problem_label PGGL_4_4_Subgroup_2C_2_fixp_graph_cliques -end \
				-process_long_orbits_by_list_of_cases_from_file \
					PGGL_4_4_Subgroup_2C_2_fixp_graph_cliques_by_type_500.csv \
					5 0 16 2 8 \
			-end \
		-end

packing_2C_500_read:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0,1,2" "SPREAD_TABLES_4/" \
			-packing_with_assumed_symmetry \
				-H -PGGL 4 4 -subgroup_by_generators "2C" 2 1 \
					"1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0" \
				-end \
				-N -PGGL 4 4 -subgroup_by_generators "centralizer_2C" "30720" 9 \
					"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
					"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,1," \
					"1,0,0,0,0,1,0,0,0,0,1,0,2,0,3,1,0," \
					"1,0,0,0,0,1,0,0,0,0,1,0,1,0,3,1,0," \
					"1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1," \
					"1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0," \
					"1,0,0,0,2,1,0,0,0,0,1,0,1,0,0,1,0," \
					"1,0,0,0,1,1,2,0,0,0,1,0,0,0,0,1,0," \
					"1,0,3,0,1,1,1,3,0,0,2,0,0,0,0,2,1," \
				-end \
				-cliques_on_fixpoint_graph 5 \
				-cliques_on_fixpoint_graph_control -W -problem_label PGGL_4_4_Subgroup_2C_2_fixp_graph_cliques -end \
				-process_long_orbits_by_list_of_cases_from_file \
					PGGL_4_4_Subgroup_2C_2_fixp_graph_cliques_by_type_500.csv \
					5 0 1 2 8 \
				-process_long_orbits_solution_path DATA_FROM_SUMMIT/SOLUTIONS/ \
			-end \
		-end


# 8147 packings of type 21,0,0 stored in PGGL_4_4_Subgroup_2C_2_fixp_graph_cliques_by_type_500_packings.csv

packing_2C_iso:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-draw_options -end \
		-finite_field_activity -q 4 \
		-canonical_form_PG  3  \
		-input -file_of_packings_through_spread_table \
			PGGL_4_4_Subgroup_2C_2_fixp_graph_cliques_by_type_500_packings.csv SPREAD_TABLES_4/spread_16_spreads.csv -end \
		-classification_prefix packings_2C \
		-report \
		-end
	pdflatex packings_2C_classification.tex
	open packings_2C_classification.pdf



packing_2C_clique0:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_2C_2_fpc0_graph -end \
		-graph_theoretic_activity -find_cliques -target_size 8 -end -end

# ripoff:
# nb colors = 1
# nb vertices = 29901
#nb_sol = 7442
#8:6:41


packing_2C_clique0_Sajeeb:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_2C_2_fpc0_graph -end \
		-graph_theoretic_activity -find_cliques -Sajeeb -target_size 8 -end -end



c700700:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file G_0_16/PGGL_4_4_Subgroup_2C_2_fpc700700_graph -end \
			-graph_theoretic_activity -find_cliques -Sajeeb -target_size 8 -end -end


# completed cases:
# 0 done
# 1 HPC finished
# 2 done
# 3 ripoff finished; HPC finished
# 4 done
# 5 ripoff finished; uploaded; job scripts uploaded; HPC running; finished
# 6 done
# 7 ripoff running; uploaded; job scripts uploaded; HPC running; finished
# 8 done
# 9 done
# 10 done
# 11 ripoff finished; files uploaded; HPC running; HPC finished
# 12 done
# 13 ripoff finished; files uploaded; HPC running; HPC finished
# 14 done
# 15 ripoff running; files uploaded; job scripts uploaded; HPC running; ***

create_files:
	$(ORBITER_PATH)/orbiter.out -create_files -file_mask job%03d \
		-read_cases ls15.txt  \
		-line "#!/bin/bash" \
		-line_numeric "#SBATCH --job-name=job%03d" \
		-line "#SBATCH --account=csu29_summit1" \
		-line "#SBATCH --nodes=1" \
		-line "#SBATCH --time=24:00:00" \
		-line "#SBATCH --ntasks=24" \
		-line "#SBATCH --partition=shas" \
		-line_numeric "#SBATCH --output=job%03d.out" \
		-line "#SBATCH --qos=normal" \
		-line "#SBATCH --mem=20000" \
		-line "/projects/abetten@colostate.edu/orbiter/src/apps/orbiter/orbiter.out -v 2 -create_graph -load_from_file /projects/abetten@colostate.edu/PACKING.20/G_15_16/%s -end -graph_theoretic_activity -find_cliques -Sajeeb -target_size 8 -end -end" \
		-end


cp_solutions:
	cp DATA_FROM_SUMMIT/G_0_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_1_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_2_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_3_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_4_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_5_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_6_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_7_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_8_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_9_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_10_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_11_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_12_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_13_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_14_16/* DATA_FROM_SUMMIT/SOLUTIONS
	cp DATA_FROM_SUMMIT/G_15_16/* DATA_FROM_SUMMIT/SOLUTIONS
	

		#-v 2 -BLT -q $(ORDER) -starter_size 5 -input_prefix /projects/abetten@colostate.edu/BLT.18/Q$(ORDER)/STARTER_DIR/ -output_prefix /scratch/summit/abetten@colostate.edu/BLT.18/$(ORDER)/%03ld/SYSTEMS/ -solution_prefix /scratch/summit/abetten@colostate.edu/BLT.18/$(ORDER)/%03ld/SOLUTIONS/ -base_fname BLT_$(ORDER) -create_graphs %ld 256 4 -lex -output_prefix /scratch/summit/abetten@colostate.edu/BLT.18/$(ORDER)/%03ld/G/" 
		#-line "mkdir /scratch/summit/abetten@colostate.edu/BLT.18/$(ORDER)/%03ld" \
		#-line "mkdir /scratch/summit/abetten@colostate.edu/BLT.18/$(ORDER)/%03ld/G" \





packing_Hall:
	- mkdir SPREAD_TABLES_4_HALL
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0" "SPREAD_TABLES_4_HALL/" \
		-end


# 9 min 16 sec on Mac




packing_2C_Hall_prepare:
	- mkdir SPREAD_TABLES_4_HALL
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0" "SPREAD_TABLES_4_HALL/" \
			-packing_with_assumed_symmetry \
				-problem_label _uniform_Hall \
				-H -PGGL 4 4 -subgroup_by_generators "2C" 2 1 \
					"1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0" \
				-end \
				-N -PGGL 4 4 -subgroup_by_generators "centralizer_2C" "30720" 9 \
					"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
					"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,1," \
					"1,0,0,0,0,1,0,0,0,0,1,0,2,0,3,1,0," \
					"1,0,0,0,0,1,0,0,0,0,1,0,1,0,3,1,0," \
					"1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1," \
					"1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0," \
					"1,0,0,0,2,1,0,0,0,0,1,0,1,0,0,1,0," \
					"1,0,0,0,1,1,2,0,0,0,1,0,0,0,0,1,0," \
					"1,0,3,0,1,1,1,3,0,0,2,0,0,0,0,2,1," \
				-end \
				-cliques_on_fixpoint_graph 5 \
				-cliques_on_fixpoint_graph_control -W -problem_label PGGL_4_4_Subgroup_2C_2_uniform_Hall_fixp_graph_cliques -end \
				-fixp_clique_types_save_individually \
			-end \
		-end

# 1 min 12 sec on Mac

#8268 x ( 5, 0, 0 )
#Written file PGGL_4_4_Subgroup_2C_2_uniform_Hall_graph_cliques_by_type_500.csv of size 80474



packing_2C_Hall_make_graphs:
	- mkdir SPREAD_TABLES_4_HALL
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0" "SPREAD_TABLES_4_HALL/" \
			-packing_with_assumed_symmetry \
				-problem_label _uniform_Hall \
				-H -PGGL 4 4 -subgroup_by_generators "2C" 2 1 \
					"1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0" \
				-end \
				-N -PGGL 4 4 -subgroup_by_generators "centralizer_2C" "30720" 9 \
					"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
					"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,1," \
					"1,0,0,0,0,1,0,0,0,0,1,0,2,0,3,1,0," \
					"1,0,0,0,0,1,0,0,0,0,1,0,1,0,3,1,0," \
					"1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1," \
					"1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0," \
					"1,0,0,0,2,1,0,0,0,0,1,0,1,0,0,1,0," \
					"1,0,0,0,1,1,2,0,0,0,1,0,0,0,0,1,0," \
					"1,0,3,0,1,1,1,3,0,0,2,0,0,0,0,2,1," \
				-end \
				-cliques_on_fixpoint_graph 5 \
				-cliques_on_fixpoint_graph_control -W -problem_label PGGL_4_4_Subgroup_2C_2_uniform_Hall_fixp_graph_cliques -end \
				-process_long_orbits_by_list_of_cases_from_file \
					PGGL_4_4_Subgroup_2C_2_uniform_Hall_graph_cliques_by_type_500.csv \
					5 0 1 2 8 \
			-end \
		-end




#CLASS_5A=-centralizer_of_element "0,2,0,0, 1,1,0,0, 0,0,3,0, 0,0,0,3, 0" -label "5A"

#		// 0 2 0 0
#		// 1 1 0 0
#		// 0 0 3 0
#		// 0 0 0 3
#		// normalizer has order 3600,



#CLASS_5B=-centralizer_of_element "0,2,0,0,1,1,0,0,0,0,0,2,0,0,1,1,0" -label "5B"

#Generator 0 / 1 is:
#0 2 0 0 
#1 1 0 0 
#0 0 0 2 
#0 0 1 1 
#, 0
#prime_at_a_time::make_normalizer_of_H_of_order_5 created generators for the normalizer
#Strong generators for a group of order 81600 tl=( 85, 4, 80, 1, 3, 1 )

#CLASS_5C=-centralizer_of_element "0,1,0,0,3,3,0,0,0,0,1,2,0,0,2,0,0" -label "5C"

#Generator 0 / 1 is:
#0 1 0 0
#3 3 0 0
#0 0 1 2
#0 0 2 0
#, 0
#
#Normalizer has order 600
#Strong generators for a group of order 600 tl=( 10, 4, 5, 1, 3, 1 )



packing_5A_prepare:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0,1,2" "SPREAD_TABLES_4/" \
			-packing_with_assumed_symmetry \
				-problem_label _5 \
				-H -PGGL 4 4 -subgroup_by_generators "5A" 5 1 \
					"0,2,0,0, 1,1,0,0, 0,0,3,0, 0,0,0,3, 0" \
				-end \
				-N -PGGL 4 4 -subgroup_by_generators "normalizer_5A" "3600" 6 \
					"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0," \
					"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,2,0," \
					"1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0," \
					"1,0,0,0,0,1,0,0,0,0,0,2,0,0,2,0,0," \
					"1,0,0,0,2,2,0,0,0,0,1,0,0,0,0,1,1," \
					"0,1,0,0,3,3,0,0,0,0,2,0,0,0,0,2,0," \
				-end \
				-cliques_on_fixpoint_graph 1 \
				-cliques_on_fixpoint_graph_control -W -problem_label PGGL_4_4_Subgroup_5A_5_5_fixp_graph_cliques -end \
				-fixp_clique_types_save_individually \
			-end \
		-end



#Written file PGGL_4_4_Subgroup_5A_5_5_graph_cliques_by_ago_100.csv of size 18
#Written file PGGL_4_4_Subgroup_5A_5_5_graph_cliques_by_ago_300.csv of size 18


packing_5B_prepare:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0,1,2" "SPREAD_TABLES_4/" \
			-packing_with_assumed_symmetry \
				-problem_label _5 \
				-H -PGGL 4 4 -subgroup_by_generators "5B" 5 1 \
					"0,2,0,0,1,1,0,0,0,0,0,2,0,0,1,1,0" \
				-end \
				-N -PGGL 4 4 -subgroup_by_generators "normalizer_5B" "81600" 6 \
					"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0," \
					"1,0,0,0,0,1,0,0,2,2,2,2,1,3,1,3,0," \
					"1,0,0,0,0,1,0,0,0,0,1,1,0,0,3,2,0," \
					"1,0,0,0,2,2,0,0,0,0,3,0,0,0,1,1,1," \
					"0,1,0,0,3,3,0,0,0,0,0,1,0,0,3,3,0," \
					"0,0,1,0,0,0,1,2,2,0,0,0,2,3,0,0,1," \
				-end \
				-cliques_on_fixpoint_graph 1 \
				-cliques_on_fixpoint_graph_control -W -problem_label PGGL_4_4_Subgroup_5B_5_5_fixp_graph_cliques -end \
				-fixp_clique_types_save_individually \
			-end \
		-end


# normalizer has order 81600:
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0,\\
#1,0,0,0,0,1,0,0,2,2,2,2,1,3,1,3,0,\\
#1,0,0,0,0,1,0,0,0,0,1,1,0,0,3,2,0,\\
#1,0,0,0,2,2,0,0,0,0,3,0,0,0,1,1,1,\\
#0,1,0,0,3,3,0,0,0,0,0,1,0,0,3,3,0,\\
#0,0,1,0,0,0,1,2,2,0,0,0,2,3,0,0,1,\\


#orbits on fixpoints:
#depth : number of orbits
#0 : 1
#1 : 4
#2 : 6
#3 : 29
#4 : 136
#5 : 336
#6 : 482
#7 : 307
#8 : 62
#9 : 0
#10 : 0
#11 : 0


#Written file PGGL_4_4_Subgroup_5B_5_5_graph_cliques_by_ago_200.csv of size 18
#Written file PGGL_4_4_Subgroup_5B_5_5_graph_cliques_by_ago_600.csv of size 18
#Written file PGGL_4_4_Subgroup_5B_5_5_graph_cliques_by_ago_1200.csv of size 18
#Written file PGGL_4_4_Subgroup_5B_5_5_graph_cliques_by_ago_81600.csv of size 18

# 81600 = 85 * 960

packing_5B_search_4:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)/orbiter.out \
		-fork CASE_NUMBER log_%d 4 256 8 \
		-v 5 \
		-list_arguments \
		-orbiter_path $(ORBITER_PATH)/ \
		-define G \
		-linear_group -PGGL 4 4 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0,1,2" "SPREAD_TABLES_4/" \
			-packing_with_assumed_symmetry \
				-problem_label _5 \
				-H -PGGL 4 4 -subgroup_by_generators "5B" 5 1 \
					"0,2,0,0,1,1,0,0,0,0,0,2,0,0,1,1,0" \
				-end \
				-N -PGGL 4 4 -subgroup_by_generators "normalizer_5B" "81600" 6 \
					"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0," \
					"1,0,0,0,0,1,0,0,2,2,2,2,1,3,1,3,0," \
					"1,0,0,0,0,1,0,0,0,0,1,1,0,0,3,2,0," \
					"1,0,0,0,2,2,0,0,0,0,3,0,0,0,1,1,1," \
					"0,1,0,0,3,3,0,0,0,0,0,1,0,0,3,3,0," \
					"0,0,1,0,0,0,1,2,2,0,0,0,2,3,0,0,1," \
				-end \
				-cliques_on_fixpoint_graph 1 \
				-cliques_on_fixpoint_graph_control -W -problem_label PGGL_4_4_Subgroup_5B_5_5_fixp_graph_cliques -end \
				-fixp_clique_types_save_individually \
				-process_long_orbits \
					-split CASE_NUMBER 256 \
					-solve \
					-read_solutions \
					-list_of_cases_from_file PGGL_4_4_Subgroup_5B_5_5_graph_cliques_by_type__all_in_one.csv \
					-clique_size 4 \
					-orbit_length 5 \
				-end \
			-end \
		-end

#					-create_graphs \


packing_5B_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_5B_5_5_fpc3.graph -end \
		-graph_theoretic_activity -find_cliques -target_size 4 -end -end

# only the last of the 4 cases gives solutions




packing_5_reg:
	- mkdir SPREAD_TABLES_5_REG
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-define G \
		-linear_group -PGL 4 5 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_5 \
			-end \
			-packing_classify 2 "12" "SPREAD_TABLES_5_REG/" \
		-end


# number of spreads: 155000

#PGL_4_5_norm_31:
#	$(ORBITER_PATH)/orbiter.out -v 6 \
#		-linear_group -PGL 4 5 -end \
#		-group_theoretic_activities \
#		-normalizer_of_cyclic_subgroup "31" "1,0,0,0, 0,0,1,0, 0,0,0,1, 0,1,1,3"


# normalizer has order 372
#1,0,0,0,0,4,0,0,0,0,4,0,0,0,0,4,\\
#1,0,0,0,0,3,0,0,0,0,3,0,0,0,0,3,\\
#1,0,0,0,0,1,0,0,0,3,1,1,0,4,2,3,\\
#1,0,0,0,0,0,1,0,0,0,0,1,0,1,1,3,\\



packing_5_reg_31_prepare:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define G \
		-linear_group -PGL 4 5 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_5 \
			-end \
			-packing_classify 2 "12" "SPREAD_TABLES_5_REG/" \
			-packing_with_assumed_symmetry \
				-problem_label _uniform_reg \
				-H -PGL 4 5 -subgroup_by_generators "31" 31 1 \
					"1,0,0,0, 0,0,1,0, 0,0,0,1, 0,1,1,3" \
				-end \
				-N -PGL 4 5 -subgroup_by_generators "normalizer_31" "372" 4 \
					"1,0,0,0,0,4,0,0,0,0,4,0,0,0,0,4," \
					"1,0,0,0,0,3,0,0,0,0,3,0,0,0,0,3," \
					"1,0,0,0,0,1,0,0,0,3,1,1,0,4,2,3," \
					"1,0,0,0,0,0,1,0,0,0,0,1,0,1,1,3," \
				-end \
			-end \
		-end

#				-cliques_on_fixpoint_graph 5 \
#				-cliques_on_fixpoint_graph_control -W -problem_label PGL_4_31_Subgroup_31_uniform_reg_fixp_graph_cliques -end \
#				-fixp_clique_types_save_individually \


packing_solution_tally:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-read_solutions_and_tally PGL_4_5_Subgroup_3B_3_reg_spreads_fpc0_sol.txt 10
		
#############################################################################################################
# Section 42: Graph Theory


GRAPH_THEORY:
	

petersen:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -Johnson 5 2 0 -end \
		-graph_theoretic_activity -save  -end

# creates Johnson_5_2_0.colored_graph


petersen_csv:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -Johnson 5 2 0 -end \
		-graph_theoretic_activity -export_csv  -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix Johnson_5_2_0.csv 20 8
	open Johnson_5_2_0_draw.bmp



Paley13:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -Paley 13 -end \
		-graph_theoretic_activity -export_csv  -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix Paley_13.csv 20 8
	open Paley_13_draw.bmp


trihedal_pair_graph:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -trihedral_pair_disjointness_graph  -end \
		-graph_theoretic_activity -export_csv  -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix trihedral_pair_disjointness.csv 20 8
	open trihedral_pair_disjointness_draw.bmp



graph_classify_4:
	$(ORBITER_PATH)/orbiter.out -v 2 -graph_classify -n 4

tournament_classify_4:
	$(ORBITER_PATH)/orbiter.out -v 2 -graph_classify \
	-n 4 -v 2 -tournament \
	-poset_classification_control -problem_label tournament_4  -depth 6 -end \
	-draw_graphs_at_level 6

# problem!

small_graph:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -edges_as_pairs 5 "0,1,0,2,0,3,0,4,1,3,1,4,2,4" -end \
		-graph_theoretic_activity -export_csv -save -end

# creates graph_v5_e7.csv
# creates graph_v5_e7.colored_graph

small_graph_cliques:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file graph_v5_e7.colored_graph -end \
		-graph_theoretic_activity -find_cliques  -target_size 3 -end -end

# nb_sol = 3

small_graph_cliques_Sajeeb:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file graph_v5_e7.colored_graph -end \
		-graph_theoretic_activity -find_cliques -Sajeeb -target_size 3 -end -end

# nb_sol = 3




#############################################################################################################
# Section 43: Graphical Output

GRAPHICAL_OUTPUT:
	

addition_table_F7:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 7 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q7_addition_table.csv 40 24 \
		-draw_matrix_partition 3 7 7
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q7_multiplication_table.csv 40 24 \
		-draw_matrix_partition 3 7 7
	open GF_q7_addition_table_draw.bmp
	open GF_q7_multiplication_table_draw.bmp

tikz:
	$(ORBITER_PATH)/orbiter.out -v 12 -define G -linear_group -PGL 4 2 -wedge_detached -end -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-draw_options -rad 200 -end \
		-end
	pdflatex PGL_4_2_Wedge_4_0_detached_report.tex 
	open PGL_4_2_Wedge_4_0_detached_report.pdf


schreier_tree:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G -linear_group -PGL 4 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 3 \
		-orbits_on_polynomials_draw_tree 6 \
		-draw_options -yout 500000 -rad 15 -nodes_empty -line_width 0.5 -y_stretch 0.25 -embedded -end \
		-end 
	pdflatex poly_orbits_d3_n3_q2.tex
	open poly_orbits_d3_n3_q2.pdf




#############################################################################################################
# Section 44: The Povray Interface


POVRAY:


cube:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask cube_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.5 -default_angle 75 -clipping_radius 2.7 \
		-end \
		-scene_objects \
			-obj_file cube_centered.obj \
			-edge "0, 1" \
			-edge "0, 2" \
			-edge "0, 4" \
			-edge "1, 3" \
			-edge "1, 5" \
			-edge "2, 3" \
			-edge "2, 6" \
			-edge "3, 7" \
			-edge "4, 5" \
			-edge "4, 6" \
			-edge "5, 7" \
			-edge "6, 7" \
			-group_of_things_as_interval 0 8 \
			-spheres 0 0.3 "texture{ Polished_Chrome pigment{quick_color White} }" \
			-group_of_things_as_interval 0 6 \
			-prisms 1 0.05 "texture{ pigment{ color Yellow transmit 0.7 } finish {diffuse 0.9 phong 0.6} }" \
			-group_of_things_as_interval 0 12 \
			-cylinders 2 0.15 "texture{ pigment{ color Red  } finish {diffuse 0.9 phong 0.6} }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv cube_0_*.pov POV
	mv makefile_animation POV
	
monkey:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask monkey_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.8 -default_angle 75 -clipping_radius 0.8 \
		-camera 0 "0,0,1" "1,1,0.5" "0,0,0"  \
		-rotate_about_z_axis \
		-end \
		-scene_objects \
			-cubic_lex "1,0,0,0,-3,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0" \
			-plane_by_dual_coordinates "0,0,1,0" \
			-group_of_things "0" \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ Gold } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-planes 1 "texture{ pigment{ color Blue transmit 0.5 } finish { diffuse 0.9 phong 0.2}}" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv monkey_0_*.pov POV
	mv makefile_animation POV





HCV:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask HCV_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.9 -default_angle 75 -clipping_radius 2.4 \
		-camera 0 "1,1,1" "-3,1,3" "0.12,0.12,0.12"  \
		-end \
		-scene_objects \
			-Hilbert_Cohn_Vossen_surface \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-group_of_things_as_interval 0 6 \
			-group_of_things_as_interval 6 6 \
			-group_of_things_as_interval_with_exceptions 12 15 "14,19,23" \
			-lines 1 0.02 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-lines 2 0.02 "texture{ pigment{ color Blue } finish { diffuse 0.9 phong 1}}" \
			-lines 3 0.02 "texture{ pigment{ color Yellow } finish { diffuse 0.9 phong 1}}" \
			-label 0 "a1" \
			-label 2 "a2" \
			-label 4 "a3" \
			-label 6 "a4" \
			-label 8 "a5" \
			-label 10 "a6" \
			-label 12 "b1" \
			-label 14 "b2" \
			-label 16 "b3" \
			-label 18 "b4" \
			-label 20 "b5" \
			-label 22 "b6" \
			-label 24 "c12" \
			-label 26 "c13" \
			-label 30 "c15" \
			-label 32 "c16" \
			-label 34 "c23" \
			-label 36 "c24" \
			-label 40 "c26" \
			-label 42 "c34" \
			-label 44 "c35" \
			-label 48 "c45" \
			-label 50 "c46" \
			-label 52 "c56" \
			-group_of_things_as_interval 0 6 \
			-texts 4 0.2 0.15 "texture{ pigment{Black} } no_shadow" \
			-group_of_things_as_interval 6 6 \
			-texts 5 0.2 0.15 "texture{ pigment{Black} } no_shadow" \
			-group_of_things_as_interval 12 12 \
			-texts 6 0.2 0.15 "texture{ pigment{Black} } no_shadow" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv HCV_0_*.pov POV
	mv makefile_animation POV

#"-3,2.333,4"  * 1.5 = "-4.5,3.5,6"
#M := Matrix([[-4.5, 3.5, 6], [1, 1, 1]])
#NullSpace(M)
#=0.186080731891197,-0.781539073943026,0.595458342051830
#		-rotate_about_custom_axis "0.186080731891197,-0.781539073943026,0.595458342051830" \
#-W 1024 -H 768 
#-W 2560 -H 1920 
#-W 4096 -H 3072 


HCV_deform:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 93 -output_mask HCV_deform_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.9 -default_angle 75 -clipping_radius 2.4 \
		-camera 0 "1,1,1" "-3,1,3" "0.12,0.12,0.12"  \
		-end \
		-scene_objects \
			-Hilbert_Cohn_Vossen_surface \
			-group_of_things "0" \
			-deformation_of_cubic_lex 93 1.107148718 1.570796327 0 \
				"0, 10, 0, -8, 10, 25, 2, 0, -20, -8, -20, -10, -24, 10, -2, 12, 0, -8, 8, 16" \
				"0, -5, 0, -5, -5, 10, -1, 0, 10, 4, 10, 5, 3, -5, 1, -6, 0, -5, -4, 1" \
			-group_of_things_as_interval 0 93 \
			-group_is_animated 1 \
			-cubics 1 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv HCV_deform_0_*.pov POV
	mv makefile_animation POV





HCV_deform_2:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask HCV_deform_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.9 -default_angle 75 -clipping_radius 2.4 \
		-camera 0 "1,1,1" "-3,1,3" "0.12,0.12,0.12"  \
		-end \
		-scene_objects \
			-Hilbert_Cohn_Vossen_surface \
			-group_of_things "0" \
			-deformation_of_cubic_lex 93 1.107148718 1.570796327 0 \
				"0, 10, 0, -8, 10, 25, 2, 0, -20, -8, -20, -10, -24, 10, -2, 12, 0, -8, 8, 16" \
				"0, -5, 0, -5, -5, 10, -1, 0, 10, 4, 10, 5, 3, -5, 1, -6, 0, -5, -4, 1" \
			--group_of_things_as_interval 0 93 \
			--group_is_animated 1 \
			-group_of_things "0" \
			-cubics 1 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-group_of_things "24" \
			-cubics 2 "texture{ pigment{ Red*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-group_of_things "70" \
			-cubics 3 "texture{ pigment{ Blue*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv HCV_deform_0_*.pov POV
	mv makefile_animation POV




	


Clebsch:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask Clebsch_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.9 -default_angle 80 -clipping_radius 2.4 \
		-camera 0 "1,1,1"  "-4.5,3.5,6" "0,0,0"  \
		-end \
		-scene_objects \
			-Clebsch_surface \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ White*0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-group_of_things_as_interval 0 6 \
			-group_of_things_as_interval 6 6 \
			-group_of_things_as_interval 12 15  \
			-lines 1 0.02 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-lines 2 0.02 "texture{ pigment{ color Blue } finish { diffuse 0.9 phong 1}}" \
			-lines 3 0.02 "texture{ pigment{ color Yellow } finish { diffuse 0.9 phong 1}}" \
			-group_of_things_as_interval 0 12 \
			-spheres 4 0.08 "texture{ pigment{Cyan*1.3} finish {ambient 0.4 diffuse 0.6 roughness 0.001 reflection 0 specular .8}  }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv Clebsch_0_*.pov POV
	mv makefile_animation POV




endrass8:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask endrass_octic_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.75 \
		-default_angle 75 -clipping_radius 3.7 -no_bottom_plane \
		-camera 0 "1,1,1" "6,6,3" "0,0,0"  \
		-rotate_about_111 \
		-end \
		-scene_objects \
			-line_through_two_points_recentered_from_csv_file coordinate_grid.csv \
			-group_of_things "0" \
			-group_of_things "1" \
			-group_of_things "2" \
			-group_of_things_as_interval 3 39 \
			-lines 0 0.15 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-lines 1 0.15 "texture{ pigment{ color Green } finish { diffuse 0.9 phong 1}}" \
			-lines 2 0.15 "texture{ pigment{ color Blue } finish { diffuse 0.9 phong 1}}" \
			-lines 3 0.05 "texture{ pigment{ color Black } finish { diffuse 0.9 phong 1}}" \
			-octic_lex_165 "-93.2548,0,0,0,-309.019,0,0,527.529,0,395.647,0,0,0,0,0,0,0,0,0,0,-687.529,0,0,1582.59,0,1186.94,0,0,0,0,-1055.06,0,-1582.59,0,-593.47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-309.019,0,0,1582.59,0,1186.94,0,0,0,0,-2110.12,0,-3165.17,0,-1186.94,0,0,0,0,0,0,874.039,0,1560.63,0,1677.92,0,343.362,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-93.2548,0,0,527.529,0,395.647,0,0,0,0,-1055.06,0,-1582.59,0,-593.47,0,0,0,0,0,0,874.039,0,1560.63,0,1677.92,0,343.362,0,0,0,0,0,0,0,0,-256,0,-468.077,0,-789.019,0,-525.726,0,0.941125" \
			-plane_by_dual_coordinates "0,0,1,0" \
			-group_of_things "0" \
			-group_of_things "0" \
			-octics 4 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-planes 5 "texture{ pigment{ color Blue transmit 0.5 } finish { diffuse 0.9 phong 1}}" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv endrass_octic_0_*.pov POV
	mv makefile_animation POV

#############################################################################################################
# Section 45: Creating a Movie Using Ffmpeg

ANIMATIONS:


monkey_video:
	- rm -r FRAMES
	- mkdir FRAMES
	- rm monkey.mp4
	$(ORBITER_PATH)/orbiter.out \
		-prepare_frames \
			-i 0 30 monkey_0_%03d.png \
			-output_starts_at 0 \
			-o FRAMES/frame%04d.png \
		-end
	ffmpeg -r 5 -f image2 -i FRAMES/frame%04d.png -f mp4 -q:v 0 -vcodec mpeg4  monkey.mp4

HCV_deform_video:
	- rm -r FRAMES
	- mkdir FRAMES
	- rm HCV_deform.mp4
	$(ORBITER_PATH)/orbiter.out \
		-prepare_frames \
			-i 0 93 HCV_deform_0/HCV_deform_0_%03d.png \
			-output_starts_at 0 \
			-o FRAMES/frame%04d.png \
		-end
	ffmpeg -r 5 -f image2 -i FRAMES/frame%04d.png -f mp4 -q:v 0 -vcodec mpeg4  HCV_deform.mp4




#############################################################################################################
# Section 46: Continuous Function Plotter



CONTINUOUS_FUNCTION_PLOTTER:


lissajous:
	$(ORBITER_PATH)/orbiter.out -v 2 -smooth_curve "lissajous" 0.07 2000 15 0 18.85 \
		-const a 3 b 2 c 1.57 r 7 -const_end \
		-var t -var_end \
		-code \
			push t push a mult push c add sin push r mult return \
			push t push b mult sin push r mult  return \
		-code_end \

#function_lissajous_N2000_points.csv

lissajous_plot:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 1 -output_mask lissajous_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.40 \
		-default_angle 45 -clipping_radius 5 -omit_bottom_plane  \
		-camera 0 "0,-1,0" "0,0,12" "0,0,0"  \
		-rotate_about_z_axis \
		-end \
		-scene_objects \
			-line_through_two_points_recentered_from_csv_file coordinate_grid.csv \
			-group_of_things "0" \
			-group_of_things "1" \
			-group_of_things "2" \
			-lines 0 0.09 "texture{ pigment{ color Yellow } }" \
			-lines 1 0.09 "texture{ pigment{ color Yellow } }" \
			-lines 2 0.09 "texture{ pigment{ color Yellow } }" \
			-group_of_things_as_interval 3 39 \
			-lines 3 0.02 "texture{ pigment{ color Black } }" \
			-point_list_from_csv_file function_lissajous_N2000_points.csv \
			-group_of_things_as_interval 0 6524\
			-spheres 4 0.1 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-plane_by_dual_coordinates "0,0,1,0" \
			-group_of_things "0" \
			-planes 5 "texture{ pigment{ color Blue*0.5 transmit 0.5 } }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv lissajous_0_*.pov POV
	mv makefile_animation POV

lissajous_3d:
	$(ORBITER_PATH)/orbiter.out -v 2 -smooth_curve "lissajous_3d" 0.07 2000 50 0 18.85 \
		-const a 3 b 2 c 1.57 r 7 -const_end \
		-var t -var_end \
		-code \
			push t push a mult push c add sin push r mult return \
			push t push b mult sin push r mult  return \
			push t return \
		-code_end \

lissajous_3d_plot:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask lissajous_3d_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.40 \
		-default_angle 45 -clipping_radius 5 -omit_bottom_plane  \
		-camera 0 "0,0,1" "7,7,5" "0,0,1"  \
		-rotate_about_z_axis \
		-end \
		-scene_objects \
			-line_through_two_points_recentered_from_csv_file coordinate_grid.csv \
			-group_of_things "0" \
			-group_of_things "1" \
			-group_of_things "2" \
			-lines 0 0.09 "texture{ pigment{ color Yellow } }" \
			-lines 1 0.09 "texture{ pigment{ color Yellow } }" \
			-lines 2 0.09 "texture{ pigment{ color Yellow } }" \
			-group_of_things_as_interval 3 39 \
			-lines 3 0.02 "texture{ pigment{ color Black } }" \
			-point_list_from_csv_file function_lissajous_3d_N2000_points.csv \
			-group_of_things_as_interval 0 6538\
			-spheres 4 0.1 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-plane_by_dual_coordinates "0,0,1,0" \
			-group_of_things "0" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv lissajous_3d_0_*.pov POV
	mv makefile_animation POV

#			-planes 5 "texture{ pigment{ color Blue*0.5 transmit 0.5 } }" \






#############################################################################################################
# Section 47: Miscellaneous

MISCELLANEOUS:


#	$(ORBITER_PATH)/orbiter.out -v 4 -csv_file_select_rows poly_orbits_d3_n3_q2.csv \
#		"28,113,130,80,84,17,21,110,101,91,50,52,59,26,117,6,38,39,61,11,15,122,121,138,96,134,76,73,75,65,112,136,118,140,27,24,57,49,87,92,131,102,79,20,137,135,129"


misc_select:
	$(ORBITER_PATH)/orbiter.out -v 3 -define F -finite_field -q 7 -end -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)/orbiter.out -v 4 -csv_file_select_rows_and_cols \
		GF_q7_multiplication_table_reordered.csv \
		"0,2,4" "0,2,4" 
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q7_multiplication_table_reordered.csv 40 24 -draw_matrix_partition 3 6 6
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q7_multiplication_table_reordered_select.csv 40 24 -draw_matrix_partition 3 3 3


misc_join:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-csv_file_join poly_orbits_d3_n3_q2_select_F2.csv Orbit_idx \
		-csv_file_join poly_orbits_d3_n3_q2_select_F4.csv Orbit_idx \
		-csv_file_join poly_orbits_d3_n3_q2_select_F8.csv Orbit_idx \
		-csv_file_join poly_orbits_d3_n3_q2_select_F16.csv Orbit_idx \
		-csv_file_join poly_orbits_d3_n3_q2_select_F32.csv Orbit_idx 


#############

LEFTOVER:





PROBLEM_27_53 = -d1 1 -q1 27 -d2 1 -q2 53 -K 11 -DDx 2 -DDy 1 -search_control -W -end 
PROBLEM_27_53_GROUP1=-subgroup "1,1,1,0, 1,3,1,0, 1,9,1,0, 1,0,1,1, -2,0,-4,0" "18603" -group_label "group1"

# mask 1:
#  XX.
#  X.X

PROBLEM_27_53_MASK1=		-masktest 3 x ge 1 \
		-masktest 4 x+y ge 3 \
		-depth 4 \
		-mask_label "mask1"

# 18603 = 27 * 53 * 13

DD_M1_G1:
	$(ORBITER_PATH)/orbiter.out -v 4  \
		-Delandtsheer_Doyen  $(PROBLEM_27_53) $(PROBLEM_27_53_GROUP1) $(PROBLEM_27_53_MASK1) \
			-end 

DD_M1_G1_S:
	$(ORBITER_PATH)/orbiter.out -v 4  \
		-Delandtsheer_Doyen $(PROBLEM_27_53) $(PROBLEM_27_53_GROUP1) $(PROBLEM_27_53_MASK1) \
				-singletons \
			-end 

PROBLEM_3_7 = -d1 1 -q1 3 -d2 1 -q2 7 -K 5 -DDx 3 -DDy 1 -search_control -W -end 
PROBLEM_3_7_GROUP1=-subgroup "1,1,1,0, 1,0,1,1 " "21" -group_label "group_cyclic"
PROBLEM_3_7_MASK1= -mask_label "mask1" -depth 5

DD_PG_2_4_M1_G1:
	$(ORBITER_PATH)/orbiter.out -v 4  \
		-Delandtsheer_Doyen  $(PROBLEM_3_7) $(PROBLEM_3_7_GROUP1) $(PROBLEM_3_7_MASK1) \
			-end 

PG_2_27:
	$(ORBITER_PATH)/orbiter.out -finite_field_activity -q 27 -override_polynomial 46 -cheat_sheet_PG 2 -end
	pdflatex PG_2_27.tex 
	open PG_2_27.pdf



canon_design:
	$(ORBITER_PATH)/orbiter.out -v 4 -finite_field_activity -q 27 -override_polynomial 46 \
		-canonical_form_PG  2  \
		-input \
		-set_of_points "2,56,30,112,253,90,440,508" \
		-end -end \
		-prefix design \
		-save design_classified \
		-report


design_line_type:
	$(ORBITER_PATH)/orbiter.out -v 4 -override_polynomial 46 \
		-process_combinatorial_objects -n 2 -q 27 -poly 46 \
		-input \
		-set_of_points "2,56,30,112,253,90,440,508" \
		-end \
		-line_type \
		-fname_base_out design \
		-end
		
design_conic_type:
	$(ORBITER_PATH)/orbiter.out -v 4 -override_polynomial 46 \
		-process_combinatorial_objects -n 2 -q 27 -poly 46 \
		-input \
		-set_of_points "2,56,30,112,253,90,440,508" \
		-end \
		-conic_type \
		-fname_base_out design \
		-end


convert_stack_tdo_32_der:
	$(ORBITER_PATH)/orbiter.out -v 4 -convert_stack_to_tdo 32_der.widor
	
	
#<HTDO id=32_der type=pt ptanz=3 btanz=2 fuse=simple>
#      561 496
#   4   33   0
#  30   33   0
#1023   17  16
#
#1 1 2
#</HTDO>


ntt:
	$(ORBITER_PATH)/orbiter.out -v 2 -NTT 4 17 ntt4.cpp


FGDTP:
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_F_k4.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_Fv_k4.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_AAv_k4.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_G_k4.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_D_k4.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_T_k4.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_Tv_k4.csv 20 8
	
no:
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_P_k4.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_F_k3.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_Gr_k3.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_Dr_k3.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_Tr_k3.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_Pr_k3.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_G_k3.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_D_k3.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_T_k3.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_P_k3.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_G_k2.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_D_k2.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_T_k2.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_P_k2.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_the_P_k4.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_L_k4.csv 20 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix Fourier_q17_N16.csv 40 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix Fourier_q17_N8.csv 40 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix Fourier_q17_N4.csv 40 8
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix Fourier_q17_N2.csv 40 8
	
SRC=../../DEV.20/GITHUB/orbiter/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib -std=c++14
LIB = $(SRC)/lib/liborbiter.a -lpthread
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64




ntt4.out:  ntt4.o ;
	$(DISCRETA_CPP)  ntt4.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



PGL_3_16_singer:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-draw_options -rad 200 -end \
		-define G -linear_group -PGGL 3 16 -singer 13 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-orbits_on_points \
		-end
	pdflatex PGGL_3_16_Singer_3_16_13_report.tex
	open PGGL_3_16_Singer_3_16_13_report.pdf
		





#############################################################################################################
# 

DICKSON:


#




Dickson_13_S:
	$(ORBITER_PATH)/orbiter.out -v 4 -finite_field_activity -q 19 -rank_point_in_PG 2 \
		"1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,1,1" \
		-end

Dickson_13_3:
	$(ORBITER_PATH)/orbiter.out -v 4 -finite_field_activity -q 19 -rank_point_in_PG 2 \
		"0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,0,0,0,1" \
		-end

Dickson_13_4:
	$(ORBITER_PATH)/orbiter.out -v 4 -finite_field_activity -q 19 -rank_point_in_PG 2 \
		"1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,1,0" \
		-end

Dickson_13_1:
	$(ORBITER_PATH)/orbiter.out -v 4 -finite_field_activity -q 19 -rank_point_in_PG 2 \
		"1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0,1,0" \
		-end

Dickson_13_2:
	$(ORBITER_PATH)/orbiter.out -v 4 -finite_field_activity -q 19 -rank_point_in_PG 2 \
		"1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0" \
		-end


Dickson_13_ideal:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-process_combinatorial_objects -n 3 -q 2 \
			-input \
			-set_of_points \
				-long_string \
				"2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14" \
				-end_string \
			-end \
		-ideal_PART 3 \
		-fname_base_out "form" \
		-end



orbits_cubic_surfaces_q3:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G -linear_group -PGL 4 3 -end -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 3 \
		-end 
	pdflatex poly_orbits_d3_n3_q3.tex
	open poly_orbits_d3_n3_q3.pdf
	
# this takes 3 days and about 150 GB memory on ripoff

orbits_cubic_curves_q2:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGL 3 2 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 3 \
		-end 
	pdflatex poly_orbits_d3_n2_q2.tex
	open poly_orbits_d3_n2_q2.pdf

orbits_cubic_curves_q3:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGL 3 3 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 3 \
		-end 
	pdflatex poly_orbits_d3_n2_q3.tex
	open poly_orbits_d3_n2_q3.pdf




Dickson_q2:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGL 4 2 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-cubic_surface_properties poly_orbits_d3_n3_q2.csv 2 \
		-end 

Dickson_q2_analyze:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G \
		-linear_group -PGL 4 2 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-cubic_surface_properties_analyze poly_orbits_d3_n3_q2_F2.csv 2 \
		-end 
	pdflatex poly_orbits_d3_n3_q2_F2_report.tex
	open poly_orbits_d3_n3_q2_F2_report.pdf

Dickson_q4:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 4 4 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-cubic_surface_properties poly_orbits_d3_n3_q2.csv 2 \
		-end 

Dickson_q4_analyze:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 4 4 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-cubic_surface_properties_analyze poly_orbits_d3_n3_q2_F4.csv 2 \
		-end 
	pdflatex poly_orbits_d3_n3_q2_F4_report.tex
	open poly_orbits_d3_n3_q2_F4_report.pdf

Dickson_q8:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 4 8 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-cubic_surface_properties poly_orbits_d3_n3_q2.csv 2 \
		-end 

Dickson_q8_analyze:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 4 8 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-cubic_surface_properties_analyze poly_orbits_d3_n3_q2_F8.csv 2 \
		-end 
	pdflatex poly_orbits_d3_n3_q2_F8_report.tex
	open poly_orbits_d3_n3_q2_F8_report.pdf

Dickson_q8_surface_canonical:
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-finite_field_activity -q 8 \
			-canonical_form_PG  3 \
				-input \
					-set_of_points "1,2,3,19,27,35,43,51,59,67,82,90,98,106,114,122,130,138,146,153,161,169,177,185,193,201,209,217,225,233,241,249,257,265,273,281,289,297,305,313,321,329,337,345,353,361,369,377,385,393,401,409,417,425,433,441,449,457,465,473,481,489,497,505,513,521,529,537,545,553,561,569,577" \
				-end \
				-classification_prefix design \
				-save_canonical_labeling \
				-save_ago \
				-save_cumulative_canonical_labeling \
				-save_cumulative_data \
				-report \
			-end \
		-end 


# problem because the group order is too large for int



Dickson_q16:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 4 16 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-cubic_surface_properties poly_orbits_d3_n3_q2.csv 2 \
		-end 

# too big for nauty


Surfaces_q3:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGL 4 3 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-cubic_surface_properties poly_orbits_d3_n3_q3.csv 3 \
		-end 

Surfaces_q3_analyze:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define G \
		-linear_group -PGL 4 3 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-cubic_surface_properties_analyze poly_orbits_d3_n3_q3_F3.csv 2 \
		-end 
	pdflatex poly_orbits_d3_n3_q3_F3_report.tex
	open poly_orbits_d3_n3_q3_F3_report.pdf


Surfaces_q9:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 4 9 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-cubic_surface_properties poly_orbits_d3_n3_q3.csv 3 \
		-end 

Surfaces_q9_analyze:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-draw_options -end \
		-define  G \
		-linear_group -PGGL 4 9 \
		-end -end \
		-with G -do \
		-group_theoretic_activities \
		-cubic_surface_properties_analyze poly_orbits_d3_n3_q3_F9.csv 2 \
		-end 
	pdflatex poly_orbits_d3_n3_q3_F9_report.tex
	open poly_orbits_d3_n3_q3_F9_report.pdf



####




# odd characteristic
# q=19

# E = 2

Family_general_q19_2_3_3_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 19 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 2 3 3 2 -q 19 -end 
		pdflatex surface_family_general_abcd_q19_a2_b3_c3_d2_report.tex
		open surface_family_general_abcd_q19_a2_b3_c3_d2_report.pdf



# E = 4:

Family_general_q19_4_2_2_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 19 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 4 2 2 4 -q 19 -end 
		pdflatex surface_family_general_abcd_q19_a4_b2_c2_d4_report.tex
		open surface_family_general_abcd_q19_a4_b2_c2_d4_report.pdf


# E = 6


Family_general_q19_11_2_15_11:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 19 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 11 2 15 11 -q 19 -end 
		pdflatex surface_family_general_abcd_q19_a11_b2_c15_d11_report.tex
		open surface_family_general_abcd_q19_a11_b2_c15_d11_report.pdf


# E = 9:

Family_general_q19_6_2_3_6:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 19 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 6 2 3 6 -q 19 -end 
		pdflatex surface_family_general_abcd_q19_a6_b2_c3_d6_report.tex
		open surface_family_general_abcd_q19_a6_b2_c3_d6_report.pdf



# E = 10:

Family_general_q19_6_5_5_6:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 19 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 6 5 5 6 -q 19 -end 
		pdflatex surface_family_general_abcd_q19_a6_b5_c5_d6_report.tex
		open surface_family_general_abcd_q19_a6_b5_c5_d6_report.pdf

# E = 18:

Family_general_q19_5_2_12_5:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 19 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 5 2 12 5 -q 19 -end 
		pdflatex surface_family_general_abcd_q19_a5_b2_c12_d5_report.tex
		open surface_family_general_abcd_q19_a5_b2_c12_d5_report.pdf


# q=29

Family_general_q29_3_7_4_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -define G -linear_group -PGL 4 29 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -family_general_abcd 3 7 4 2 -q 29 -end 
		pdflatex surface_family_general_abcd_q29_a3_b7_c4_d2_report.tex
		open surface_family_general_abcd_q29_a3_b7_c4_d2_report.pdf



arc_lifting_two_lines_29_E1:
	$(ORBITER_PATH)/orbiter.out -v 4 -define G -linear_group -PGL 4 29 -end -end \
		-draw_options -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -problem_label sixarcs_q29 -end \
		-classify_surfaces_through_arcs_and_two_lines \
		-test_nb_Eckardt_points 1 \
		-report \
		-end
	pdflatex surfaces_arc_lifting_29.tex
	open surfaces_arc_lifting_29.pdf



E0="-1,55,1,45,-1,46,1,57,1,61,-1,62"
E1="1,39,-1,40,-1,49,1,51,1,61,-1,62"
E2="-1,67,1,69,1,59,-1,64,-1,22,1,24"
E3="-1,38,1,47,1,40,-1,67,-1,51,1,69"
E4="1,61,-1,62,-1,67,1,69"
E5="1,38,-1,47,-1,39,2,66,-1,67,1,49,-2,68,1,69,1,61,-1,62"
E6="1,62,-1,63,-1,64,1,24"
E7="1,35,-1,36,-1,38,2,65,-1,29,1,40,-2,66,1,45,1,51,-1,69"
E8="1,61,-1,63,-1,59,1,22"
E9="1,27,1,47,-2,65,-1,48,1,50,-1,40,1,67,2,68,-1,55,-1,51"
E10="1,66,-1,68,-1,59,1,64,1,21,-1,23"
E11="-1,53,1,44,1,66,-1,68,1,55,-1,45"
E12="1,61,-1,62,1,66,-1,68"
E13="1,53,-1,44,1,66,-2,67,-1,68,1,46,2,69,-1,57,1,61,-1,62"
E14="1,60,-1,61,1,64,-1,23"
E15="1,27,1,41,-1,42,-2,65,1,44,-1,40,2,67,1,68,-1,55,-1,45"
E16="1,60,-1,62,1,59,-1,21"
E17="2,65,-1,53,-1,54,-1,29,1,56,-1,66,1,55,1,45,1,51,-2,69"
E18="-1,67,1,69,1,66,-1,68"
E19="-1,66,1,67,1,68,-1,69,-2,61,2,62,1,21,-1,22,-1,23,1,24"
E20="-1,68,1,61,1,69,-1,62"
E21="1,36,-1,41,-1,37,1,43,2,66,-2,67,-1,68,1,69,1,61,-1,62"
E22="-1,66,1,62,1,67,-1,61"
E23="1,54,-1,50,1,66,-1,67,-2,68,2,69,1,52,-1,58,1,61,-1,62"
E24="1,68,-1,69,-1,60,1,63,1,23,-1,24"
E25="1,36,-1,41,-1,40,1,68,1,45,-1,69"
E26="1,66,-1,67,-1,60,1,63,1,21,-1,22"
E27="-1,54,1,50,1,66,-1,67,1,55,-1,51"
E28="1,37,-1,40,-1,43,1,45,1,61,-1,62"
E29="1,55,-1,51,1,52,-1,58,-1,61,1,62"
E30="1,21,-1,23,-1,22,1,24"
E31="1,61,-1,21,-1,62,1,22"
E32="1,61,-1,24,-1,62,1,23"
E33="1,61,-1,24,-1,62,1,22"
E34="1,61,-1,21,-1,24,1,23"
E35="1,62,-1,22,-1,23,1,21"
E36="1,61,-1,21,-1,62,1,23"
E37="1,61,-1,23"
E38="1,21,-1,62"
E39="1,62,-1,22,-1,23,1,24"
E40="1,62,-1,24"
E41="1,22,-1,23"
E42="1,61,-1,21,-1,24,1,22"
E43="1,61,-1,22"
E44="1,21,-1,24"
H0="1,21"
H1="1,21,-1,4"
H2="1,22"
H3="1,22,-1,4"
H4="1,23"
H5="1,23,-1,4"
H6="1,24"
H7="1,24,-1,4"
H8="1,21,-1,22"
H9="1,21,-1,23"
H10="1,22,-1,24"
H11="1,23,-1,24"
H12="1,61,-1,62"
H13="1,61,-1,21,-1,24,-1,62,1,22,1,23"
H14="-1,67,-1,68,1,61,1,66,1,69,-1,62"


E_complement_19:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -q 19 -n 4 \
		-intersection_of_zariski_open_sets "E_complement" 4 60 \
		$(E0) $(E1) $(E2) $(E3) $(E4) $(E5) $(E6) $(E7) $(E8) $(E9) \
		$(E10) $(E11) $(E12) $(E13) $(E14) $(E15) $(E16) $(E17) $(E18) $(E19) \
		$(E20) $(E21) $(E22) $(E23) $(E24) $(E25) $(E26) $(E27) $(E28) $(E29) \
		$(E30) $(E31) $(E32) $(E33) $(E34) $(E35) $(E36) $(E37) $(E38) $(E39) \
		$(E40) $(E41) $(E42) $(E43) $(E44) \
		$(H0) $(H1) $(H2) $(H3) $(H4) $(H5) $(H6) $(H7) $(H8) $(H9) \
		$(H10) $(H11) $(H12) $(H13) $(H14) \
		 -monomial_type_PART -end

Eckardt_zero_29:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -q 29 -n 4 \
		-intersection_of_zariski_open_sets "E_complement" 4 60 \
		$(E0) $(E1) $(E2) $(E3) $(E4) $(E5) $(E6) $(E7) $(E8) $(E9) \
		$(E10) $(E11) $(E12) $(E13) $(E14) $(E15) $(E16) $(E17) $(E18) $(E19) \
		$(E20) $(E21) $(E22) $(E23) $(E24) $(E25) $(E26) $(E27) $(E28) $(E29) \
		$(E30) $(E31) $(E32) $(E33) $(E34) $(E35) $(E36) $(E37) $(E38) $(E39) \
		$(E40) $(E41) $(E42) $(E43) $(E44) \
		$(H0) $(H1) $(H2) $(H3) $(H4) $(H5) $(H6) $(H7) $(H8) $(H9) \
		$(H10) $(H11) $(H12) $(H13) $(H14) \
		 -monomial_type_PART -end

nonconical_29:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -q 29 -n 4 \
		-intersection_of_zariski_open_sets "points" 4 15 \
		$(H0) $(H1) $(H2) $(H3) $(H4) $(H5) $(H6) $(H7) $(H8) $(H9) \
		$(H10) $(H11) $(H12) $(H13) $(H14) \
		 -monomial_type_PART  \
		 -end -save "nonconical_q29_"

Eckardt_conditions_29:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -q 29 -n 4 \
		-number_of_conditions_satisfied "points" 4 nonconical_q29_points.txt 45 \
		$(E0) $(E1) $(E2) $(E3) $(E4) $(E5) $(E6) $(E7) $(E8) $(E9) \
		$(E10) $(E11) $(E12) $(E13) $(E14) $(E15) $(E16) $(E17) $(E18) $(E19) \
		$(E20) $(E21) $(E22) $(E23) $(E24) $(E25) $(E26) $(E27) $(E28) $(E29) \
		$(E30) $(E31) $(E32) $(E33) $(E34) $(E35) $(E36) $(E37) $(E38) $(E39) \
		$(E40) $(E41) $(E42) $(E43) $(E44) \
		 -monomial_type_PART -end



surface_q9_Hirschfeld_book:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 9 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -q 9 -by_coefficients "1,1,1,4,2,6,2,10,2,12,1,13" -end \
		-end
	pdflatex surface_by_coefficients_q9_report.tex
	open surface_by_coefficients_q9_report.pdf


surface_q9_Hirschfeld_book_a:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 9 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -q 9 -by_coefficients "1,1,1,4,2,6,2,10,2,12,1,13" \
			-select_double_six "0,2,4,5,3,1,6,8,10,11,9,7" \
		-end \
		-end
	pdflatex surface_by_coefficients_q9_report.tex
	open surface_by_coefficients_q9_report.pdf

surface_q9_Hirschfeld_book_ab:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 9 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -q 9 -by_coefficients "1,1,1,4,2,6,2,10,2,12,1,13" \
			-select_double_six "0,2,4,5,3,1,6,8,10,11,9,7" \
			-transform_inverse "1,0,0,5, 0,1,0,2, 0,0,1,1, 0,0,0,1, 0" \
		-end \
		-end
	pdflatex surface_by_coefficients_q9_report.tex
	open surface_by_coefficients_q9_report.pdf


surface_q9_Hirschfeld_book_ab_nullspace:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 9 \
		-nullspace 3 4 "6,0,1,6, 7,2,0,5, 1,1,1,1" \
		-normalize_from_the_right \
		-end



surface_q9_Hirschfeld_book_abc:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 9 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -q 9 -by_coefficients "1,1,1,4,2,6,2,10,2,12,1,13" \
			-select_double_six "0,2,4,5,3,1,6,8,10,11,9,7" \
			-transform_inverse "1,0,0,5, 0,1,0,2, 0,0,1,1, 0,0,0,1, 0" \
			-transform_inverse "7,4,6,0, 0,3,6,0, 8,0,8,0, 0,0,0,1, 0" \
		-end \
		-end
	pdflatex surface_by_coefficients_q9_report.tex
	open surface_by_coefficients_q9_report.pdf


surface_q9_Hirschfeld_book_abc_move_two_lines:
	$(ORBITER_PATH)/orbiter.out -v 5 -finite_field_activity -q 9 \
		-move_two_lines_in_hyperplane_stabilizer_text  \
		"1,0,0,0, 0,1,4,5"  "1,0,2,7, 0,1,0,0" "1,0,0,0, 0,0,0,1" "0,1,0,0, 0,0,2,1" \
		-end


surface_q9_Hirschfeld_book_abcd:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 9 -wedge -end -end \
		-with G -do \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-create_surface -q 9 -by_coefficients "1,1,1,4,2,6,2,10,2,12,1,13" \
			-select_double_six "0,2,4,5,3,1,6,8,10,11,9,7" \
			-transform_inverse "1,0,0,5, 0,1,0,2, 0,0,1,1, 0,0,0,1, 0" \
			-transform_inverse "7,4,6,0, 0,3,6,0, 8,0,8,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 0,1,0,0, 0,0,1,0, 7,5,6,8, 0" \
		-end \
		-end
	pdflatex surface_by_coefficients_q9_report.tex
	open surface_by_coefficients_q9_report.pdf



#############################################################################################################
# Cheon (33,5) arcs in PG(2,8)


point_types_A:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -label pt_A \
		-coefficient_matrix 2 6 "4,3,2,1,0,0, 1,1,1,1,1,1" \
		-RHS "32,32,1,9,9,1" -x_min_global 0 -x_max_global 9
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file \
		pt_A.diophant -solve_mckay


# writes file pt_A.sol containing 5 solutions

point_types_B:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -label pt_B \
		-coefficient_matrix 2 6 "3,4,5,6,7,8, 1,1,1,1,1,1" \
		-RHS "39,39,1,9,9,1" -x_min_global 0 -x_max_global 9
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file \
		pt_B.diophant -solve_mckay

# writes file pt_B.sol containing 43 solutions

arc_33_5:
	$(ORBITER_PATH)/orbiter.out -v 4 -arc_parameters 8 33 5
	$(ORBITER_PATH)/orbiter.out -v 4 -convert_stack_to_tdo arc_q8_s33_r5.stack
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_refinement -input_file arc_q8_s33_r5.tdo \
		-dual_is_linear_space -D1_upper_bound_x0 5 -end

# writes arc_q8_s33_r5.stack
# writes arc_q8_s33_r5.tdo
# finds 197 distributions


arcs_4_2:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 3 8 -end -end \
		-with G -do \
		-group_theoretic_activities \
				-poset_classification_control -problem_label arcs_q8_d2 -W -depth 4 -end \
			-classify_arcs \
				-target_size 4 \
				-q 8 \
				-n 3 \
				-d 2 \
			-end \
		-end


arcs_4_2_lift:
	$(ORBITER_PATH)/orbiter.out -v 2 -process_combinatorial_objects -q 8 -n 2  \
		-arc_with_given_set_as_s_lines_after_dualizing \
			33 5 0 0 \
		-fname_base_out arcs_q8_d2_lvl_4_lifted \
		-input \
			-file_of_points arcs_q8_d2_lvl_4 \
		-end \
		-end 
	mv system_0.solutions arcs_4_2_lift.solutions

arcs_4_2_classify:
	$(ORBITER_PATH)/orbiter.out -v 2 -finite_field_activity -q 8 \
		-canonical_form_PG  2  \
		-input -file_of_points arcs_4_2_lift.solutions \
		-end \
		-classification_prefix arcs_4_2_lift \
		-report \
		-end \
		-end
	pdflatex arcs_4_2_lift_classification.tex
	open arcs_4_2_lift_classification.pdf



semifields_32:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G \
		-linear_group -PGL 10 2 -end -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label semi_32 \
			-end \
			-classify_semifields  \
				-order 32 \
				-level_two_prefix L2 \
				-level_three_prefix L3 \
			-end \
		-end


CRC_2_4_4:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 4 4 -end

CRC_2_5_4:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 5 4 -end

CRC_2_6_4:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 6 4 -end

CRC_2_7_4:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 7 4 -end

CRC_2_8_4:
	$(ORBITER_PATH)/orbiter.out -v 2 -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 8 4 -end

CRC_2_9_4:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 9 4 -end

CRC_2_9_5:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 9 5 -end

CRC_2_10_5:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 10 5 -end

CRC_2_32_6:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 32 6 -end

CRC_2_64_8:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 64 8 -end

CRC_2_128_8:
	$(ORBITER_PATH)/orbiter.out -v 1 -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 128 8 -end

CRC_3_128_8:
	$(ORBITER_PATH)/orbiter.out -v 1 -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 128 8 -end

CRC_3_128_10:
	$(ORBITER_PATH)/orbiter.out -v 1 -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 128 10 -end



CRC_2_256_8:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 256 8 -end


CRC_2_256_12:
	$(ORBITER_PATH)/orbiter.out -v 1 -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 256 12 -end

CRC_3_256_8:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 256 8 -end

CRC_3_256_16:
	$(ORBITER_PATH)/orbiter.out -v 1 -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 256 16 -end



CRC_3_8_4:
	$(ORBITER_PATH)/orbiter.out -v 2 -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 8 4 -end



CRC_3_64_8:
	$(ORBITER_PATH)/orbiter.out -define F -finite_field -q 2 -end -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 64 8 -end









