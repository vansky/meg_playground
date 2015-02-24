meg_playground
=========

A workspace to play with MEG data  

There are a number of dependencies:

    ipython-notebook
    numpy
    scipy
    matplotlib
    pylab
    scikit-learn
    pandas
    statsmodels
    PyMVPA2 (from source [repos are out of date]: https://github.com/PyMVPA/PyMVPA)

You'll need to create a subdirectory named 'MEG_data' and store the relevant data files there for this code to work. Within MEG_data, you'll need:

    paired .set and .fdt files containing MEG data  
    a space-delimited .tab file with word-level predictive features  

Once your file structure looks good, if you want to run an interactive notebook session (outdated), you'll need to go to the 'notebooks' subdirectory and type  

    ipython notebook
    
or you can do non-notebook-based analyses by going to the 'scripts' directory and typing:    

    python meg_stats_runner.py

Easy command-line querying of the output from stats_runner can be obtained by typing:  

    python query_stats.py
    
Visualization of stats_runner output is available from:  

    notebooks/Topomap_real

To obtain the V7.tab file with feature annotations:

    Begin with the V3.tab file from the MEG HOD corpus.  
    Obtain the gcgbadwords, gcgparseops, and totsurp files from the authors (or from /home/corpora)  
    python ../scripts/buildtab.py hod_V3.tab hod.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.gcgbadwords > hod_V4.tab  
    python ../scripts/addsentid.py hod_V4.tab > hod_V5.tab  
    python ../scripts/addparserops.py hod_V5.tab hod.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.gcgparseops > hod_V6.tab  
    python ../scripts/addtotsurp.py hod_V6.tab hod.totsurp > hod_V7.tab
    Set tokenPropsFile = <.tab filename>  