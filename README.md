# Imaging in Neuroscience: with a focus on MEG and EEG Methods

Here you find the tutorials for the doctoral course *Imaging in Neuroscience: with a focus on MEG and EEG Methods* at Karolinska Institutet from April 14th to May 3rd.

This is an attempt to port from MATLAB to MNE-python.

All tutorials are example of how to analyse MEG/EEG signals. This is not meant as an exhaustive account of how to analyse MEG/EEG data. The material is design to give an overview of different aspects of MEG/EEG data analysis and useable example code on how to do the analysis. 

To run the tutorial you need MNE-Python (https://mne.tools/stable/index.html), for that you first need to install Python and some modules, preferably installed in an Anaconda virtual environment.

For this course, you will use the module MNE-python and its dependencies to analyses MEG/EEG data. It is recommended that you also install Anaconda, a virtual envorinment. 

Follow the instructions on how to install Anaconda and MNE [here](https://mne.tools/stable/install/install_python.html)

You also need the tutorial data. All tutorials use MNE-Python. To learn more about MNE-Python, we recommend looking at the MNE-Python [webpage](https://mne.tools/stable/index.html) for documentation and many more tutorial examples.

Here the scripts are written in [Spyder](https://www.spyder-ide.org), an envirornmet that is well suited for our purposes and will make any transition from MATLAB to Python easier.

## Overview

The tutorial for the course is divided in seven parts numbered 01-07. Each part consist of one or more pages:

* tutorial_01a_preprocessing
* tutorial_01b_evoked responses

* tutorial_02_frequencyanalysis
* tutorial_03_prepare_mri
* tutorial_04a_dipole analysis
* tutorial_04b_mne
* tutorial_05_beamformer
* tutorial_06_connectivity
* tutorial_07_statistics

In addition, there is extra tutorials that are not part of the main assignment. They contain tips for writing analysis scripts and example on how to create surface-based source models with Freesurfer (www.freesurfer.net) and import them to FieldTrip

* tutorial_00_tips_for_writing_scripts
* tutorial_99_prepare_mne_sourcespace

## Exam and questions

Thought the tutorials you will find questions like this:

> **Question X.X:** what is an ERP?

For the course  *Imaging in Neuroscience: with a focus on MEG and EEG Methods* your assignment is to answer the questions in a separate text document and submit through Canvas. When you answer the questions, use examples, code, and figures from the exercises.

## Credits

All material is free to use under general license. 

The tutorial material was originally created for FieldTrip 2018-2020 by

* Mikkel C. Vinding, NatMEG, Karolinska Insitutet.
* Lau M. Andersen, CFIN, Aarhus University (www.laumollerandersen.org/).
* Robert Oostenveld, Donders Institute for Brain, Cognition and Behaviour, Radboud University.

The attempt to port the tutorials to MNE-Python was done by Andreas Gerhardsson

If you discover errors in the tutorials, code that won't run, or similar, please post your errors under issues.


