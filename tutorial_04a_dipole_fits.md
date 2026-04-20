# Tutorial 4a: Dipole fitting
In this tutorial, you will do a dipole fit to evoked responses. For the dipole fits, we assume that the dipolar patters can be adequately explained by a few dipolar sources in the brain. We do not need to model activity in the entire brain with this method. What we will do is to create a source model of evenly distributed sources across the entire brain and then scan the sources for the best explanation of the observed scalp potentials and magnetic fields. The steps are as follows:

Create source space and leadfield
Do dipole fits
Evaluate outcome
To do this, you need to have completed three preceding steps:

Calculate time-locked data for the MEG and EEG data (tutorial 1B).
Create the head model for the MEG data (tutorial 3).
Create the head model(s) for the EEG data (tutorial 3).

## Set up paths
Change these to appropriate paths for your operating system and setup
```{python}
# Import Modules and setting up paths
import mne
import mne.bem
import os
from os.path import join, exists
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# define paths

project_path = "/Users/erin.noelle.mahan/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Documents/MEG_Course_MNE"
meg_path = join(project_path, 'TutorialDataset') 
figs_path = join(project_path, 'figs')

show_plots = True # Change to True to open plots in browser

#%% Define subject paths and list of all subjects/session

subjects_and_dates = [
    'NatMEG_0177/170424/'  # Add more subjects as you like, separate with comma    
    ]
           
# Define where to put output data
output_path = join(meg_path, subjects_and_dates[0], 'MEG')
mri_path = join(meg_path, subjects_and_dates[0], 'MRI')
subjects_dir_path = join(meg_path, subjects_and_dates[0], 'freesurfer_subjects')
subject= '170424'
```
## Load relevant files
Load the required data files for dipole fits:
```{python}
#load relevant files
#evokeds
evo_path = join(output_path, 'tactile_stim_ds200Hz-clean-ica-ave.fif')
evo = mne.read_evokeds(evo_path)
# when you import evoked objects without specifying which event type you want, it imports them all as a list

epo_path= join(output_path, 'tactile_stim_hp1Hz_lp95Hz_ds200Hz-clean-ica-epo.fif')
epo = mne.read_epochs(epo_path)

#head models
eeg_bem_path = join(output_path, '170424-eeg-bem-sol.fif')
eeg_head_model = mne.read_bem_solution(eeg_bem_path)

meg_bem_path= join(output_path, '170424-meg-bem-sol.fif')
meg_head_model = mne.read_bem_solution(meg_bem_path)

# transform file
trans_file = join(output_path, "tactile_stim_ds200Hz-clean-ica-epo-trans.fif") 
trans = mne.read_trans(trans_file)
```
## Identify ERP/ERF components of interest
Before doing the dipole fits, take a look at the sensor-level ERF/ERPs to look for dipolar patters in the scalp. Try to do a "manual" source localisation using the right-hand rule.

You can plot the evoked all together...

```{python}
mne.viz.plot_evoked(evo[3], picks='mag', spatial_colors=True)
mne.viz.plot_evoked(evo[3], picks='grad', spatial_colors=True)
```
![evo_butterfly_mag](figures/evo_butterfly_mag.png)

or separate them out by channel.

```{python}
mne.viz.plot_evoked_topo(evo[3], merge_grads=True)
```
![evo_grad_topo](figures/evo_grad_topo.png)

Once you've found a time period that looks interesting, you can plot the topomap for the time periods of interest. 

Let's take a look at the peaks around 55 ms and 135 ms.

```{python}
mne.viz.plot_evoked_topomap(evo[3], times= (0.055), average=.005, ch_type='mag')
mne.viz.plot_evoked_topomap(evo[3], times= 0.055, average=.005, ch_type='grad' )

mne.viz.plot_evoked_topomap(evo[3], times= 0.135, average=.005, ch_type='mag')
mne.viz.plot_evoked_topomap(evo[3], times= 0.135, average=.005, ch_type='grad')
```
Topographical plot of the 55 ms ERF component for magnetometers and combined gradiometers:
![evo_mag_topo_55](figures/evo_mag_topo_55.png)
![evo_grad_topo_55](figures/evo_grad_topo_55.png)

Topographical plot of the 135 ms ERF component for magnetometers and combined gradiometers:
![evo_mag_topo_135](figures/evo_mag_topo_135.png)
![evo_grad_topo_135](figures/evo_grad_topo_135.png)

Notice how you on the topographies can make a qualified guess about the number of equivalent current dipoles and their approximate location.

## Calculate noise covariance
When doing source reconstruction in MNE-Python, you will need to provide a template of what 'no signal of interest' looks like for some of the algorithms employed in the analysis. We call this a noise covariance matrix. In MNE there are a few options of what kind of noise covariance matrix you want to make. 

One option is data driven. You will select a time period within your data that does not contain relevant signal, such as a pre-stimulus baseline period. You can use the function `compute_covariance` for this. If you know where the signal is relative to the noise, this can be a good option. If your data has been MaxFiltered, you must either calculate the `rank` parameter separately, or set `rank='info'` in your function call. This is because in the process of MaxFiltering, the rank is reduced due to the noise and artifact reduction.

> Rank is essentially the number of independent signals in your data. It can be reduced from the total number of sensors by things like MaxFilter, average referencing, bad channel interpolation, and other preprocessing steps.

Another option is called an ad hoc covariance. In this noise covariance estimation, MNE assumes a certain level of noise across all channels equally. It is not driven by the actual data. 

In this tutorial, we will use an ad hoc covariance matrix, but in future tutorials, we will try computing our own. Since we'll only be looking at magnetometers in this tutorial, we can select the channels in `epo` that contain what we want.

```{python}
mag_epo = epo.copy().pick('mag')

ad_hoc_cov = mne.make_ad_hoc_cov(mag_epo.info)
```

## Fit a single dipole
We fit single dipoles separately for the three kinds of sensors (magnetometers, gradiometers and electrodes). We do it for two latencies identified in the (sensor-level) evoked responses above:

1. Early sensory response at 45-65 msec
2. Late sensory response at 115-155 msec

We'll start our analysis with just the magnetometers during the early response.

```{python}
mag_evo = evo[3].copy().pick('mag')
mag_evo_early = mag_evo.crop(0.050, 0.065)
```
Our optimization parameter is residual variance, i.e. we will try to minimize the residual variance between fitted values and actutal measures data. This is how much data that is left unexplained by the dipole model. For this, we call `mne.fit_dipole`.
```{python}
dipole_mag_early, res_var = mne.fit_dipole(mag_evo_early, cov=ad_hoc_cov, bem=meg_head_model, trans=trans) 
```
If you're near someone following the Fieldtrip/MatLab version of the tutorial, you might be wondering about the leadfield/forward model that they have to specify for their dipole fit. MNE-Python doesn't allow for any custom forward models in standard dipole fitting with `fit_dipole`. We will be using one in the two-dipole fitting though.
## Dipole fit diagnostics
Let's take a look at this new Dipole object that has been created. 

In MNE, the Dipole class has some attributes that describe the size, direction, location, etc. of the dipole that was calculated. The attributes are times, pos, amplitude, ori, gof, name, conf, khi2, and nfree.

> **Question 4.1:** Explain what the Dipole attributes times, pos, and amplitude mean in our dipole_mag_early. 

> Hint: It might help to do the following plots first.

For inspection and diagnostics of the dipole fit, let's take a look at where the dipole location is. For this, we're using the MRI and the alignment that we created in the previous tutorial.
```{python}
dipole_mag_early.plot_locations(trans, subject, subjects_dir_path, mode="orthoview")
```
Now, look at the values related to the dipole itself. You can use the code below to inspect the dipole moment, strength, and residual variance.

```{python}
dip = dipole_mag_early  # your Dipole object

# time window in seconds
tmin, tmax = 0.050, 0.065
mask = (dip.times >= tmin) & (dip.times <= tmax)

times_ms = dip.times[mask] * 1e3
gof = dip.gof[mask]
rv = 100 - gof

amp = dip.amplitude[mask]              # Am
moment = amp[:, None] * dip.ori[mask]  # x,y,z moment components
strength = np.abs(amp)                 # same idea as moment magnitude

fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
fig.suptitle("Dipole Fit Metrics (45–65 ms)")
# 1) Goodness of fit
axes[0].plot(times_ms, gof)
axes[0].set_ylabel("GOF (%)")
axes[0].set_title("Goodness of Fit (%)")

# 2) Residual variance
axes[1].plot(times_ms, rv)
axes[1].set_ylabel("Residual variance (%)")
axes[1].set_title("Residual variance")

# 3) Moment components
axes[2].plot(times_ms, moment[:, 0], label="Mx")
axes[2].plot(times_ms, moment[:, 1], label="My")
axes[2].plot(times_ms, moment[:, 2], label="Mz")
axes[2].set_ylabel("Moment (Am)")
axes[2].set_title('Dipole Moment Components')
axes[2].legend(loc='center left')

# 4) Strength / magnitude
axes[3].plot(times_ms, strength)
axes[3].set_ylabel("Strength (Am)")
axes[3].set_xlabel("Time (ms)")
axes[3].set_title('Dipole Strength')

plt.tight_layout()
plt.show()
```
Another useful diagnostic is to assess how well the predicted activity pattern at the scalp generated by the dipole model corresponds to the actual data. 

The code below will calculate that for you. 
```{python}
fwd, stc = mne.make_forward_dipole(
    dipole_mag_early,
    meg_head_model,
    mag_evo_early.info,
    trans,
)

pred_evoked = mne.simulation.simulate_evoked(
    fwd,
    stc,
    mag_evo_early.info,
    cov=ad_hoc_cov,
    nave=np.inf,
)

best_idx = np.argmax(dipole_mag_early.gof)
best_time = dipole_mag_early.times[best_idx]

diff = mne.combine_evoked([mag_evo_early, pred_evoked], weights=[1, -1])

fig, axes = plt.subplots(
    nrows=1,
    ncols=4,
    figsize=[10.0, 3.4],
    gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1], top=0.85),
    layout="constrained",
)

plot_params = dict(
    times=best_time,
    ch_type="mag",
    outlines="head",
    colorbar=False,
    show=False,
)

mag_evo_early.plot_topomap(
    time_format="Measured field",
    axes=axes[0],
    **plot_params,
)

pred_evoked.plot_topomap(
    time_format="Predicted field",
    axes=axes[1],
    **plot_params,
)

plot_params["colorbar"] = True
diff.plot_topomap(
    time_format="Difference",
    axes=axes[2:],
    **plot_params,
)

fig.suptitle(
    f"Measured vs predicted field at {best_time * 1000:.1f} ms",
    fontsize=14,
)

plt.show()
```
![measured_vs_predicted_field_d1](figures/measured_vs_predicted_field_d1.png)
