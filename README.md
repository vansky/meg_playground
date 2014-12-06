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
