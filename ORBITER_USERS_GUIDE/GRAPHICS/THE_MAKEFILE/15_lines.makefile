
#MY_PATH=../orbiter
MY_PATH=~/DEV.21/GITHUB/orbiter
#MY_PATH=/scratch/betten/COMPILE/orbiter


# uncomment exactly one of the following two lines.
# uncomment the first if you want to run orbiter through docker.
# uncomment the second if you have an installed 
#copy of orbiter and you want to run it directly 

#ORBITER_PATH=docker run -it --volume ${PWD}:/mnt -w /mnt abetten/orbiter 
ORBITER_PATH=$(MY_PATH)/src/apps/orbiter/


###############################################################################
# End of configuration part
###############################################################################

F_ALPHA_BETA_GAMMA_DELTA="beta*(gamma + 1)*x0*x0*x2 \
+ (alpha*delta - beta*gamma + alpha - beta - delta - 1)*x0*x1*x2 \
-1*(alpha*beta -alpha*delta + delta)*(gamma + 1)*x0*x1*x3 \
+ (0-alpha*delta + alpha*gamma -beta*gamma -beta + delta -gamma)*x0*x2*x2 \
-(alpha*delta +beta -delta)*(gamma +1)*x0*x2*x3 \
-(delta + 1)*(alpha - 1)*x1*x1*x2 \
- (delta + 1)*(alpha - 1)*x1*x1*x3 \
+ (alpha*delta - alpha*gamma + beta*gamma + beta - delta + gamma)*x1*x2*x2 \
+ (alpha*beta*gamma + alpha*beta + alpha*delta \
- alpha*gamma + beta*gamma + beta - delta + gamma)*x1*x2*x3 \
+ alpha*beta*(gamma + 1)*x1*x3*x3"


F_alpha_beta_gamma_delta_sweep_4_q7:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-sweep_4 sweep_4_q7 -q 7 -by_equation "F_alpha_beta_gamma_delta" \
			"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
			$(F_ALPHA_BETA_GAMMA_DELTA) \
			"alpha=2,beta=3,gamma=4,delta=5" \
			"\D\alpha=2,\beta=3,\gamma=4,\delta=5\D" \
		-end


#User time: 0:44
# 1512 parameter sets


F_alpha_beta_gamma_delta_sweep_4_classify_q7_nauty:
	$(ORBITER_PATH)/orbiter.out -v 2 \
	-define F -finite_field -q 7 -end \
	-define P -projective_space 3 F -end \
	-with P -do \
	-projective_space_activity \
		-canonical_form_PG \
		-input \
		-file_of_points F_alpha_beta_gamma_delta_q7_points.txt \
		-end \
		-classification_prefix surface_15_lines_q7 \
		-report \
		-end \
	-end
	pdflatex surface_15_lines_q7_classification.tex
	open surface_15_lines_q7_classification.pdf


#User time: 19:35
# 18 orbits