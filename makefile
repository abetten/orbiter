all: ;
	cd src; $(MAKE) all

clean: ;
	cd src; $(MAKE) clean
	#cd RUN; $(MAKE) clean
	- rm bin/*.out

install: ;
	cd src; $(MAKE) install
