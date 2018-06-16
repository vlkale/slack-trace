---

# Description of the Library

The code in this repository generates a library that predicts the time that a process of an MPI program spends waiting for 
other processes to complete their contribution to an MPI communication function during application execution.

The library measures *MPI process slack* of a process of an MPI application running on a supercomputer. The term *MPI process slack* refers to the time that an MPI process spends waiting on other MPI processes to contribute their portion of messages to an interprocess communication function in MPI application run on a supercomputer.


---

# Installation and Configuration of the Library

To install the library on your machine, i.e., generate the library header and library source code, you need to manually run the wrap.py program with the arguments slack-trace.h.w and slack-trace.w or use cmake. 

    ./wrap/wrap.py -c mpiCC slack-trace.h.w > slack-trace.h;
    
    
    ./wrap/wrap.py -c mpiCC slack-trace.w > slack-trace.C; 


Note that mpiCC is the command for a MPI C++ compiler. The command for the MPI C++ compiler may be different on your machine.

---

# Compilation and Usage of the Library

To compile the code files and generate a library file to link to your application program, you have three options: 1) use the Makefile; 2) manually compile the files; or 3) use cmake using the files CMakeLists.txt, and one of the available \*.cmake files in this directory. 

The information below has more information about the \*.cmake files.


## Using cmake

If you want to use cmake, you simply type: 

     cmake; 
 
 
Note that you may need to remove files that CMake has generated and cached in the top-level of the directory where this repository is stored on your computer or machine each type you type 'cmake'.

## Using the Makefile

If you want to use the Makefile, you need to simply type the following on the command-prompt in the directory where this repository is stored on your computer or machine:

     make clean; make;


## Manually Compiling the Files

If you want to manually compile the files, you need to type the following on the command prompt in the directory that the files are stored in on your computer or machine:                                 
          
     mpiCC -c slack-trace.C;
     
     
     ar rcs libslackconscious.a ./slack-trace.o;    


Then, you link the library to your application program as follows: 


     mpiCC -o myApp.C myApp.o -L. -lslackconscious <FLAGS_FOR_OTHER_LIBRARIES>
     

where $(LFLAGS) is a variable that stores flags for any other libraries needed for your application. Examples of 
such flags are -lm, which is the flag for a math library, and -fopenmp, which is a flag for a library for shared memory parallel programming named OpenMP. 


---

# Acknowledgements: 

This library is developed building upon the slack prediction techniques and code of LIBRA. See http://github.com/tgamblin/libra for more information.  Its license file is LIBRA_LICENSE in this repository. 


---
