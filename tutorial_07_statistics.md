# Statistical analysis of MEG/EEG data

Statistical comparison and testing with MEG/EEG signals require considerable considerations on the design of the study and how to control for multiple comparisons. In this tutorial we shall look at different approaches to comparing MEG/EEG signals from two conditions;

1. Analysis based on pre-specified data feature.
2. Non-parametric cluster-based permutation tests.
3. Multivariate pattern analysis (MVPA).

In this tutorial you will do a within-subject analysis. That means that you compare data across repetitions from only a single subject. The unit of observation is the single trial. In a real dataset, you would, in most cases, have data from many subjects, and the unit of observation is data from each subject. Data from each subject would first be processed to get the ERF/ERP, TFR response, etc., as you have done in the previous tutorials. The within-subject averaged data will be the unit of observation. The principles of the statistics are, however, the same as presented in this tutorial, but be aware that the unit of observation will change when doing analysis "for real".

This tutorial will utilize functions from other modules other than what we have been using previously, namely SciPy and SciKit Learn. 

Importing Error? If scipy or sklearn does not import, install them with `python -m pip install scipy scikit-learn` in a terminal, then restart the kernel.

## Set up paths
```{python}
# Import Modules and setting up paths
import mne
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind ### NEW IMPORT FROM PREVIOUS TUTORIALS
from mne.stats import permutation_cluster_test
from mne.stats import spatio_temporal_cluster_test
from mne.channels import find_ch_adjacency
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# define paths 

project_path = "/Users/erin.noelle.mahan/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Documents/MEG_Course_MNE"
meg_path = join(project_path, 'TutorialDataset') 

#%% Define subject paths and list of all subjects/session

subjects_and_dates = [
    'NatMEG_0177/170424/'  # Add more subjects as you like, separate with comma    
    ]
           
# Define where to put output data
output_path = join(meg_path, subjects_and_dates[0], 'MEG')
```
## Load data
Load the data that contains the pre-processed single-trial data. For a group-level comparison you would load the averaged data for each participant.

```{python}
epo_path= join(output_path, 'tactile_stim_lp70Hz_ds200Hz-clean-ica-epo.fif')
epo = mne.read_epochs(epo_path)
```
## Statistics on data feature
The first option to compare conditions is to select pre-specified features in the data. Say we have a hypothesis that there is a difference in the amplitude of ERF that occurs around 130-160 ms after stimulation between conditions. To test this hypothesis, we only need to select the amplitude of the ERF. That can be represented as a single scalar number per unit of observation.

To compare the amplitude of the ERF component between conditions, we first need to extract this value from the data. We find the channel that shows the largest effect in the time-window of interest and then take the average value withing this time-window to compare between conditions.

In this example, we compare the difference between the tactile stimulation of the thumb (condition A) and the little finger (condition B).

The first step is to select our trials of interest for the two conditions. In this step, we also create a joint data structure containing both conditions to create an unbiased "localizer" condition.

We want to make sure that the different conditions have the same length and baseline period.

```{python}
data_thumb = epo['Thumb'].copy().crop(-.200, .600).apply_baseline((-.200, 0))
data_little = epo['Little finger'].copy().crop(-.200, .600).apply_baseline((-.200, 0))
data_both = mne.concatenate_epochs([data_thumb, data_little])
```
You can verify that they're all the same baseline after. 
```{python}
#verify baseline 
print(data_both.baseline)
print(data_thumb.baseline)
print(data_little.baseline)
```
Calculate the evoked responses for each condition A, B, and A+B and do a visual inspection of the data. 

```{python}
# make evoked
evo_thumb = data_thumb.average()
evo_little = data_little.average()
evo_both = data_both.average()

evo_thumb.plot()
evo_little.plot()
evo_both.plot()
```
Here we could manually look for the channel with the largest response, but this is a sub-optimal procedure for many reasons. Instead, we use the evoked response for the combined data to define the channel where we extract the peak value.

Since the values for magnetometers, gradiometers, and electrodes are in different units, it only makes sense to look at one of the sensor types at the time. For now, we select the magnetometers.

We will find the magnetometer that has the largest peak in the specified time-window (t_win) in the combined data.

```{python}
#find peak channel
t_win = (0.130, 0.160)
evo_mag = evo_both.copy().pick("mag")

# Get time indices
toi = (evo_mag.times >= t_win[0]) & (evo_mag.times <= t_win[1])

# Compute mean over time window for each channel
mean_vals = evo_mag.data[:, toi].mean(axis=1)

# Find peak channel
idx = mean_vals.argmax()
pk_chan = evo_mag.ch_names[idx]
```

We then use the name of the channel that we found (stored in pk_chan) to select data from the thumb and little finger data separately with `.pick`. We can also specify the time-window with `.crop`, and we can average data over the time-window with `.flatten`.

```{python}
thumb_data = data_thumb.copy().pick(pk_chan).crop(*t_win).get_data().mean(axis=2)
little_data = data_little.copy().pick(pk_chan).crop(*t_win).get_data().mean(axis=2)

thumb_data = thumb_data.flatten()
little_data = little_data.flatten()
```

If you look at the trial field in the either of two data structures, you should see that they still have the same number of repetitions as before, but that for each trial there is now only a single value rather than time-series data. These values are the data that we want to compare between conditions.

Let's take a look at the data we have now:
```{python}
plt.scatter(np.ones_like(thumb_data), thumb_data)
plt.scatter(np.ones_like(little_data)*2, little_data, color='r')
plt.xlim(0.5, 2.5)
plt.show()
```
Finally, let us make the statistical comparison. Though data in the two conditions come from the same subject, each unit of observation is independent of the other condition, so we use a two-sample t-test (i.e., there is no logical pairing of data on the single-trial level). To do the test between conditions, we use Matlab's built-in function for a two-sample t-test (ttest2):

```{python}


