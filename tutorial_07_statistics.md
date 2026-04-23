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
tval, pval = ttest_ind(thumb_data, little_data)
print("t =", tval)
print("p =", pval)
```

> **Question 7.1:** In this analysis, we compared values averaged over a fixed time-window. Another approach to get a summary value of the same component could be to find the maximum value of the peak in the peak channel. However, this approach is discouraged. Reflect on why this might be the case, and give your answer.

## Intermediate Summary
The feature summary approach to comparing features of MEG/EEG signals, like what you have just done, is easy to apply and does not require more statistical knowledge compared to any other comparison between two samples. However, the step of selecting which features to compare, requires careful consideration and that you are specific about the hypothesis that you want to test. More exactly, what part of the MEG/EEG signal that you want to compare. This approach open for easy ways to cheat at statistics, by first looking at your data and then select the feature that you want to compare based on how the data looks. This is known as hypothesising after the results are known, or HARKing, and is wrong. The features of the MEG/EEG signals that you compare should be defined in advance before you look at your data and even before you do the experiment.

## Non-parametric cluster-based permutation tests on single-channel data
Non-parametric cluster-based permutation tests is a way to analyse more than just a single feature of the data signals while at the same time, correcting for multiple comparisons. The principles of non-parametric cluster-based permutation tests are, in short, that we test for differences across all time points (or frequency points, or whatever type of data that we test) and then take the sum data points that are connected. In this example, it will be time-points that are connected. The test is then repeated N-times where the data labels are randomly assigned. The cluster sum of the largest cluster in the real data is compared to the distribution of the largest cluster values for all permutation. If the real largest cluster sum is higher than 95% of the permutated largest cluster sums, we can with reject the null-hypothesis (for a detailed description see Maris & Oostenveld, 2007).

Now let us test for differences between the ERFs for the tactile stimulation on the thumb and the little finger, i.e. the null-hypothesis that there is no difference.

We select the channel that showed the largest ERF peak (as above) as the best representation fo the ERF time-course.

```{Python}
X1 = data_thumb.copy().pick(pk_chan).get_data()[:, 0, :]
X2 = data_little.copy().pick(pk_chan).get_data()[:, 0, :]
```
Let's inspect the ERF's before we continue.

```{python}
times = data_thumb.times

plt.plot(times, X1.mean(axis=0), label="Thumb")
plt.plot(times, X2.mean(axis=0), label="Little finger")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title(f"Channel: {pk_chan}")
plt.show()
```
To do the non-parametric cluster-based permutation test on evoked signals, we use the function `permutation_cluster_test`. This is a built-in MNE-Python statistical test that is used to determine if your data differs between conditions across timepoints, while controlling for multiple comparisons. Many things that need to be specified in the Matlab/Fieldtrip version of this statistical test are default or implied in MNE-Python, so don't worry if you are comparing between them and it looks like you're missing something.

The `permutation_cluster_test` function performs a statistical analysis that looks for clusters of adjacent time points that show consistent differences between conditions, reducing false positives that would occur if you performed a separate test at each time point.

The observed data is clustered and assigned cluster-level test statistics. Then, because it's a permutation test, we generate n amount of null distributions for comparison to the observed data. These null distributions are made by shuffling the condition labels across trials (epochs) and performing the clustering again. Basically saying "what would the clustering look like if there was no difference between the conditions?". The observed clusters are compared to these null clusters and they are statistically significant if they are larger than most clusters made during permutation.

In our function call, the parameters we use help explain what it is that we're doing behind the scenes. `n_permutations=1000` means we will randomly permute the data to form clusters for the null distribution 1000 times. The higher this number, the more precise your p-value can be. `tail=0` is a default parameter that we have stated explicitly for clarity; it means that we are performing two-tailed tests. We aren't declaring a specific statistical test that should be done at each cluster, which means that we will be using the default test. In MNE-Python, this is a one-way ANOVA at each time point. It performs a function similar to a t-test in this context. And finally, `threshold=None` indicates that MNE should automatically determine the threshold for forming clusters in our data.

Our data is given as a list `[X1, X2]` where each item on the list is (observations, time points). 

```{python}
T_obs, clusters, p_values, _ = permutation_cluster_test(
    [X1, X2], 
    n_permutations=1000,
    tail=0,
    threshold=None
    )
```
Explore what's inside each of the values that `permutation_cluster_test` returned. The test statistics are in `T_obs`, indices of the identified clusters in `clusters`, and the p-values of each cluster in `p_values`. 

We can plot the results of the test directly on the evoked response to see what sections are significant. The shaded areas are the clusters with significant differences.

```{python}
# Single-channel evoked responses
evo_thumb = data_thumb.copy().pick(pk_chan).average()
evo_little = data_little.copy().pick(pk_chan).average()

times = evo_thumb.times
thumb_mean = evo_thumb.data[0]
little_mean = evo_little.data[0]

fig, ax = plt.subplots()

# Plot the two condition averages
ax.plot(times, thumb_mean, label="Thumb")
ax.plot(times, little_mean, label="Little finger")

# Shade significant clusters
for cluster, p_val in zip(clusters, p_values):
    if p_val < 0.05:
        inds = cluster[0] if isinstance(cluster, tuple) else cluster
        inds = np.asarray(inds)

        if inds.dtype == bool:
            sig_times = times[inds]
            if len(sig_times) > 0:
                ax.axvspan(sig_times[0], sig_times[-1], alpha=0.5)
        else:
            ax.axvspan(times[inds[0]], times[inds[-1]], alpha=0.5)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title(f"Cluster permutation test at {pk_chan}")
ax.legend()
plt.show()
```
> **Question 7.2:** Write a summary of the single-channel non-parametric cluster-based permutation test as if you were to report the result of the test and interpretation thereof in the results section of a scientific paper.

## Non-parametric cluster-based permutation tests on all channel data
