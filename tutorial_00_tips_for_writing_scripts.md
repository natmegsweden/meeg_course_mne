# Tips for writing useful analysis scripts
Writing data analysis scripts can quickly become a mess! There are many steps to go from raw MEG/EEG data to the final results. It is essential to keep track of what processing step that goes before and after another. Know what data that should be read in one step, saved to memory, and then read in the next step. If we mess this up, we might end up with invalid results. And it is easy to  make errors: read the wrong data files, using different versions of toolboxes, working in the wrong directory, etc., especially in MEG/EEG data processing where there are several manual steps and we often have to go back to re-run analysis.

Below you find a quick list of recommendations to make it easier for you to write useful analysis scripts. The recommendations are based on van Vliet (2019)[^1] and the MEG-BIDS guidelines[^2]. I recommend that you take a look at these when you have to write your own analysis scripts.

Here the scripts are written in [Spyder](https://www.spyder-ide.org), an envirornmet that is well suited for our purposes and will make any transition from MATLAB to Python easier.

## Comment your code
In Python you write comments with the hash symbol `#`. Use this to write short explanations of what section or even single lines of code in your scripts do.

```python

# like this

"""
You can also add longer segments of text that should not be treated as code, but as plain text like this.
"""
```
There are several reasons why you should comment your scripts. The first reason is that it makes it much easier to go back to your old scripts and know what they are supposed to do. What is self-evident when you first write your code might not be evident years later. The time you spent on writing comments in your code will come back later. The second reason for commenting your code is the usefulness if you are part of collaboration where you have to share data and scripts. What is self-evident for you might not be evident for other people. The third reason is that there is an increase in demand for sharing analysis scripts when publishing scientific articles, either for review purposes or demand by publishers that it has to be made available upon publication. Make it easier for the reviewers to understand what you are doing with your data. And finally, writing what the code is supposed to do helps you identify code that is not working correctly.

## Python modules
Python you need to import the modules and/or functions that you will use. Many are included by default, but you will need to install some additional modules. How to do this will not be covered in this tutorial. Make a habit of importing the libraries in the begging of your script.

```python 

import numpy as np
import matplotlib.pyplot as plt

```
For this course, you will use the module MNE-python and its dependencies to analyses MEG/EEG data. It is recommended that you also install Anaconda, a virtual envorinment. 

Follow the instructions on how to install Anaconda and MNE [here](https://mne.tools/stable/install/install_python.html)

## Use section breaks when testing code
When writing code you often want to run only a small snip of code, e.g. when you test your code while scripting.

If you only want to run parts of your script, you can mark code and select `run selection` or press `F9`. However, constantly marking code manually becomes annoying, really fast. Instead, use section breaks. You start a section with `# %%` (hash and two percentage signs). The line is commented, and you will notice that the section is grey (with defualt settings). If you now press `ctrl+enter / cmd+enter` you will run the code in the highlighted section.

```python
# %% Make a section
x = np.arange(1, 11)
y = np.random.random(len(x))

# %% Make a new section
plt.scatter(x, y)
```

## Define the paths and import modules at the beginning of the script
The start of my script may look like this:

```python

import mne
import numpy as np
import matplotlib.pyplot as plt

home_path = '/Users/andger/'
project_path = home_path + 'meeg_course_mne/'

```

## Run all analysis with one version of the software
Modules for data analysis gets updated regularly. However, do not update MNE daily! When you begin a project, make sure that all data is processed with the same version of the software that you use. If you need to update, which you sometimes need to do, make sure that your code is backwards compatible with the updated modules. If you need to update, better re-run everything. 

If you have many ongoing projects, it is useful to have several versions to make sure that you use the same versions for each project.

Some advantages with using Anaconda is that you can create a new environment with updated modules if you are starting a new project. You can also export a list of you moduls and versions to recreate environments.

## One script does one data processing step
It might seem like a good idea to have one big script that you only have to run once to go from raw data to the finished result. It is not! It only makes it difficult to find bugs and errors. Instead, try to follow the principle:

> **one script = one analysis step**

For example, one script that import raw data from disc, does the pre-processing, and then save the processed data to script. You can then easily name your scripts in the order they should run and call them from master script, e.g.:

````bash
python S01_import_data.py
python S02_run_ica.py
python S03_evoked_analysis.py
python S04_source_analysis.py
python S05_statistics.py
python ...
````

## Save intermediate results
Save data after each processing step. Especially before and after steps that require manual intervention. This makes it easier to go back and redo individual steps without the need to re-run everything again.

## Visualize results of intermediate processing steps
Thought the tutorial you will be asked a lot to plot data and inspect data structures. This is not just to keep you occupied! It is good practice to visuale the outcome of each processing step (when you also would save the data). When you visualize the output of each processing step, it is easy to spot errors as they occur rahter than only noticing them in the end results—if you even notice them at all by then.

## Use consistent filenames
Do not rename files each time you run the analysis. Use a consistent way to easy read what subject, session, and processing step the data belongs to. For example, output files from an analysis might look like this:

```bash
sub01-raw-tsss.fif
sub01-raw-downsampled.fif
sub01-epochs.fif
sub01-tfr.fif
...
```

Note that each file has the id of the subject (`sub01`) in all filenames and a string indicating what analysis step it belongs to.

## Store data separate by subject and session
When you have data from multiple subjects resist the temptation to throw all data into one folder. Instead, create a project folder where you have one folder per subjects. And if you have more than one session per subject, you should then have separate sub-folders in the subject folder:

```shell
/home/mikkel/my_project/data/ ...
    ./sub01 ...
        ./session1 ...
            ./sub01-ses1-raw-tsss.fif
            ./sub01-ses1-raw-downsampled.fif
            ./sub01-ses1-epochs.fif
            ./sub01-ses1-tfr.fif
            ... etc.
        ./session2 ...
            ./sub01-ses2-raw-tsss.fif
            ./sub01-ses2-raw-downsampled.fif
            ./sub01-ses2-epochs.fif
            ./sub01-ses2-tfr.fif
            ... etc.
    ./sub02 ...
    ... etc.
```

When you keep this structure, it is  easy to set up subject specific paths to loop though when processing multiple subjects.

In the tutorial data, you will find one subject called "NatMEG_0177" with one session called "170424" (the recording date; not the best session name). 

We can then setup subject and recording specific paths as below. The cell array `subjects_and_dates` can be expanded with more subjects and sessions when needed. You will see these lines of code several times throughout the tutorials.

```python
# %% Define subjects and sessions in lists

subjects_and_dates = [
    'NatMEG_0177/170424/'
    ]

subjects = [
    'NatMEG_0177'
    ]
                
filenames = [
    'tactile_stim_raw_tsss_mc.fif',
    'tactile_stim_raw_tsss_mc-1.fif',
    'tactile_stim_raw_tsss_mc-2.fif'
    ]
```

In your scripts, you can then easily loop though several subjects and run the same processing step on all subjects. You also make sure that you always read and save data to the correct folder by generating the paths within the loops, rather than specifying it manually in each script; e.g., like this:

````python

from os import path

meg_path = path.join(project_path, 'MEG')

output_path = path.join(meg_path, subjects_and_dates[0])  # Note that Python starts counting at 0
    
````

## Specify paths and subject names once
When you run multiple scripts on several subjects and have to go back to redo steps of the analysis you can easily loose track of what files belong to which and when. You might end up with several versions of the same file processed with slightly different options. You want to make sure that when you go on to the next step, you read the correct files. If you change the filenames along the way, it is easy to loose track of which files that are new and old and you are prone to make critical errors

The best way to avoid such errors is to specify subject id and filenames as few places as possible; ideally only once! This can be done by making a meta-script where you specify filenames and subject id (such as the code snippet above) that you then run in the beginning of each script.

## Ask for help
If you encounter an error or problem that you do not know how to solve, there is a high likelihood that someone else have encountered the exact same problem before you. You might find that the answer has already been answered in an online forum or on one of the many MEG/EEG mailing list. Simply starting with googling the error message or the issue you are uncertain about often gives you the solution you need.

There are also a lot of good resources for tips and tricks on MEG/EEG analysis. The MEG/EEG community is very open and helpful. Not only are the major analysis toolboxes open-source (which mean you can use it freely and even contribute yourself), they all have mailing lists that you can sign up for and ask for help from MEG/EEG scientists around the world. They tend to be quick to reply and friendly. Do not be afraid to ask for help!

***

[^1]: van Vliet, M. (2019). [Guidelines for data analysis scripts](https://arxiv.org/pdf/1904.06163.pdf). *ArXiv:1904.06163*. 

[^2]: [The Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/) is an initiative to standardize how neuroimaging data is stored to allow easy sharing of data across research groups and sites. Originally BIDS was for sharing MRI data but has since been expanded for MEG (and by extension EEG). Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A., Henson, R. N., Jas, M., Litvak, V., T. Moreau, J., Oostenveld, R., Schoffelen, J.-M., Tadel, F., Wexler, J., & Baillet, S. (2018). [MEG-BIDS, the brain imaging data structure extended to magnetoencephalography](https://doi.org/10.1038/sdata.2018.110). *Scientific Data, 5(1)*.
