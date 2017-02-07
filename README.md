# tdscf_pyscf
a TDSCF extension of PYSCF incorporating TCL code etc.... 

![Alt text](/misc/spectrum.jpg?raw=false "Realtime Spectra with PySCF")

## Requirements 
- Armadillo (download the lib and use CMake to install)

```	cd /PATHTOARMA/
	mkdir build 
	cd build 
	cmake .. 
	make install 
```

- To install the C-routines, do the same in /lib 

## To use. 
- As of now have installed pyscf and python test.py  

## Goals: 
- Pure Python TDSCF working. 
- BO TDSCF (jkoh). 
- Propagations which call a C/C++ step routine provided by /lib (tnguye23)
- Periodic? 

