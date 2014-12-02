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

Once your file structure looks good, if you want to run an interactive notebook session, you'll need to go to the 'notebooks' subdirectory and type  

    ipython notebook

If you want to run the notebook analyses over each individual channel (and multiple frequency bands) in a single command, you'll need to go to the 'scripts' directory and type  

    python meg_frequency_scanner.py
    
or you can get stats by going to the 'scripts' directory and typing  

    python meg_stats_runner.py
