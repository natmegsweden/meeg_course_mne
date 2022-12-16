---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Tutorial 1A: From raw data to evoked responses: pre-process MEG & EEG data

In this tutorial, you will go through the initial steps that take you from raw data to evoked responses. These steps are:
1. Import the raw MEG and EEG data.
2. Inspect the data and metadata.
3. Select timelocked data
4. Clean the data by removing artefacts, filter data, and re-reference EEG.

The raw data is stored in the three `fif` files that you have downloaded before beginning the tutorials:

    'tactile_stim_raw_tsss_mc.fif'
    'tactile_stim_raw_tsss_mc-1.fif'
    'tactile_stim_raw_tsss_mc-2.fif'

Note that though there are three data files, they are all part of one single recording session. But because the `fif` format (the file format of the Neuromag MEG/EEG system that we used to record data) only allows files to have a size of up to 2GB, the recording has been split into separate files. We should, therefore, think of these as one single file when we continue to process the data despite there being three files.

This is an important thing to notice when recording data. It illustrates the need for consistent file naming. For the example data, you can see that all are called `tactile_stim`, indicating that it is the same task.

## Import libraries and setup paths
The first step is to point to the path where we have the data and setup FieldTrip. Change these to appropriate paths for your operating system and setup.

```python
import mne
import os
from os.path import join, exists
import numpy as np


home_path = '/home/andger' # Change to match your home path
project_path = join(home_path, 'courses/meeg_course_mne') # Change to match your project path
meg_path = join(project_path, '../data')   # Change to match your data path
meg_path = '../data'

figs_path = join(project_path, 'figs')

figs_path = 'figures'

print(os.listdir(meg_path))
print(os.listdir(figs_path))
```

Then define the subject and recording specific paths. For now, we only have one subject and session. In principle, we could just define the path as one string variable when we only have one subject. But we introduce this already now as it is a good ´way to organize your data when you have multiple subjects or session. In that case, the cell array `subjects_and_dates` can be expanded to include more subjects, simply by adding the subject ids and session names.

```python
# %% Define subject paths
# List of all subjects/session

subjects_and_dates = [
    'NatMEG_0177/170424/'  # Add more subjects as you like, separate with comma    
    ]
           
# List of all filenames that we will import                
filenames = [
    'tactile_stim_raw_tsss_mc.fif',
    'tactile_stim_raw_tsss_mc-1.fif',
    'tactile_stim_raw_tsss_mc-2.fif'
            ]

# Define where to put output data
output_path = join(meg_path, subjects_and_dates[0], 'MEG')
```

## First look at what is in the data files
The `fif` files contain everything that was recorded during the recording data, including MEG data, EEG data, triggers, and various metadata. Before you import everything, take a look at what is in the files. This is especially a good idea if you are dealing with large files to avoid that you confidentially read in more data than what your computer can handle.

Now it is time to use the first MNE-Python function: use `mne.io.read_info` to read metadata from the `fif` files. Note that this will not read the data yet.

```python
infile = join(output_path, filenames[0])
info = mne.io.read_info(infile)
print(info)
```

The info object is like a python dictionary and contains information about the data. Explore the struct to find out what is in the data file.

```python
info.keys()
info['ch_names']
```

## Read raw data
Read the raw file, the file data is not automatically loaded into memory to save data. Note that MNE automatically read all split files. If you have files that are split but are not regarded as such by MNE-python, for example if you have stopped a recording within the same condition, then it is probably easiest if to read in the files separatly, concatenate and save as new files. After that MNE-python will treat the files as split files. Since the consequtive split-file is automatically read in this case, I will not put the code in the code field. If you concatenate files that are part of the split-files chain, you risk creating an adding copies of the same data.

raws = []
for file in ['file.fif', 'file-b.fif', 'file-c.fif']:
    raws.append(mne.io.read_raw_fif(join(output_path, file)))
    raw = mne.concatenate_raws(raws)
raw.save('filename')

```python
# %%
raw = mne.io.read_raw_fif(infile)
raw # check raw file
raw.info # Check info, same as the info object above
```

> **Question 1.1:**
>
> * What type of data channels is in the data?  
>
> * What is the sampling frequency? 
>
> * How many samples are there in the entire recording? 
> * How long is the time of the recording?
>
> Hint: look at len(raw) and raw.info['sfreq'] to calculate the total duration of the recording session.


## Read trigger values
The data consists of tactile stimulation to all five fingers of the right hand. When each stimulation to a finger occurred is marked by a trigger in the data. We will use these triggers to select the parts of the data that we will analyze later on.

What the values of the triggers represent is something you always want to write down in a trigger manual or protocol so that you always know what the values represent. You can read the values from the data, but their meaning is something you should know. 

For this data, I know from my recording notes the trigger values represent the following:

    1  = Little finger tactile stimulation
    2  = Ring finger tactile stimulation
    4  = Middle finger tactile stimulation
    8  = Index finger tactile stimulation
    16 = Thumb tactile stimulation
    32 = New block begins
    64 = End of experiment

But knowing what the values represent is one thing. Another is to see how they actually look in the data. It is a good quality check to inspect how the trigger values appear in the data. For example, if we are to pilot a newly designed experiment, we want to make sure that the value and the order of the triggers appear correct.

To inspect trigger values we use `mne.find_events()` in the raw-file. Now use `mne.find_events()` to read the events (i.e. triggers) in the file you specified above:

```python
eve = mne.find_events(raw)
```

Look at the `eve` structure:

> **Question 1.2:** What are the values and the types of events in `eve` and how many events are there in total?

Because there are several trigger channels in the data, MNE-Python automatically finds the composite channel 'STI101'. If you bump into another configuration or you need other event channels you can specify which stim channel to read data from:

```python
# Find only the relevant channel
eve = mne.find_events(raw, stim_channel = 'STI101')
```

The event data structure is an array consisting of:
[sample number, event on, event code]

The middle column is almost always 0, basically indicating that the event is not overlapping more than one sample.

```python
# %% See unique events
np.unique(eve[:,-1])

# %% Make a summary table of event count

np.unique(eve[:,-1], return_counts=True)
```

In addition to knowing how many trials we have of each type, we also want to know how the trials are distributed over time. The sample the trigger occurred is stored in the first column `eve[:,0]`.

Since the trigger values might tell you nothing you can add event ID's to the trigger values by defining them in a dictionary. This is useful later when plotting and creating epochs. We will comment out events that are not fingers.

```python
event_id = {'Little finger': 1,
            'Ring finger': 2,
            'Middle finger': 4,
            'Index finger': 8,
            'Thumb': 16
            # 'New block begins': 32,
            # 'End of experiment': 64
            }    
```

Now plot the triggers across time:

```python
# %% Plot
fig = mne.viz.plot_events(eve, event_id=event_id)
figname = join(figs_path, 'triggers.png')
if not exists(figname):
    fig.savefig(figname)
```

![triggers](figures/triggers.png "Triggers over time")

## Inspect raw data
Now that you have a sense about what is in the data files, it is time to take a look at the actual data. Always start by visually inspecting raw data.

There are several parameters which you can change in the plot functtion. First, plot all channels and show the events. Then pick only a subset of the channels. when using `pick` you have to do this on a copy of the object.

```python
# %% Inspect raw data
raw.plot(events=eve, event_id=event_id)  # Note that all channels are plotted

figname = join(figs_path, 'raw_data.png')
if not exists(figname):
    fig = raw.copy().pick(['mag']).plot(events=eve, event_id=event_id, start = 120)  
    fig.savefig(figname)
```

![raw_data](figures/raw_data.png "raw_data")

Browse through the data. Find browsing functions under the Help button.

You can also change visalization option. Try, for example, to add a low-pass filter to the data`:

```python
raw.plot(eve, lowpass=40)
```

## Create trials from raw data
Now that you have a sense about what is in the data files, it is time to cut out the events of interest.

This is an event-related study, so we want to import data around the events of interest. You do this with `mne.Epochs`. You need to specify the time around the triggers that you want to import, which trigger events, and preferably the event name of thouse triggers. You can define a lot of other parameters but we keep it simple for now. Note that the data will not be loaded in memory.

```python
# %% Create epochs

tmin = -2  # seconds before trigger
tmax = 2  # seconds after trigger

epochs = mne.Epochs(raw, events=eve, event_id=event_id, tmin=tmin, tmax=tmax, )
```

You can have a look at the events in the epochs by calling `epochs.events`.
You can also get a summary of the epochs object by callling `epochs`

The epochs were created for all events, but if you for some reason only want epochs from one trigger you can define a new event_id dictonary.

```python
only_index_id = {'Index finger': 8}
index = mne.Epochs(raw, events=eve, event_id=only_index_id, tmin=tmin, tmax=tmax)
```

Now let's work a bit with the data.
> A large dataset in your computers memory and can cause it become very slow to the point that it will crash.

Since data is pretty big let's start by downsampling the data to 200Hz and save. Now data needs to be loaded into memory. 
The resampling will effectivly create a lowpass filter at 100Hz.

```python
raw.load_data()
raw.resample(200)
raw_ds_name = infile.replace('raw_tsss_mc.fif', 'ds200Hz-raw.fif')
raw.save(raw_ds_name)
```

Then we create epochs, now with the downsampled data.

```python
# %% Create epochs

tmin = -2.0  # seconds before trigger
tmax = 2.0  # seconds after trigger

epochs = mne.Epochs(raw, events=eve, event_id=event_id, tmin=tmin, tmax=tmax)
```

The epochs object contain all the data from all channels and have multiple attributes and methods which can be applied on the epochs.

Explore the different methods by typing `epochs.` followed by tab in the console to see all methods options.

> **Question 1.3:** Explain how the MEG/EEG is data stored in the `epochs` struct.

In MNE-python data from all split-files are automatically read. This is conserved in the epochs so saved epochs files are automatically read as one instance. At this point we do not split the data into MEG or EEG channels.

Lets save the `epochs`. 

```python
epo_name = raw_ds_name.replace('raw.fif', 'epo.fif')
epochs.save(epo_name)

epochs.load_data() # load the data
```

## Inspect trials
You can use plots to visually inspect data after we have segmented it into trials. This time we make a "butterflyplot" by averaging, but there are several other plot options like eg. `plot_image()`. Define `picks` in as argument if you want specific channel types, otherwise all MEG and EEG types are plotted. `spatial colors` gives you an indication of the distribution over the head.

```python
# Butterflyplots
# Plot MEG and EEG

# Without any arguments into plot(), both MEG types and EEG will be plotted. You can select by adding 'mag', 'grad' or 'eeg' to picks, eg. epochs.average().plot(picks='eeg')
figname = join(figs_path, 'butterfly.png')
if not exists(figname):
    fig, ax = plt.subplots(3, 1, figsize=(6, 10))
    epochs.average().plot(spatial_colors=True, axes=ax)
    plt.tight_layout()
    fig.savefig(figname)

# Visualize with an image plot:
# epochs.plot_image()  # image
figname = join(figs_path, 'epochs_image.png')
if not exists(figname):
    fig = epochs.plot_image(picks='mag')[0]
    fig.savefig(figname)

```
![butterfly](figures/butterfly.png "butterfly")
![epochs_image](figures/epochs_image.png "epochs_image")

You can add visual pre-processing options using the `plot` as before.

Try inspect the data and see if you can identify some bad electrodes (and MEG sensors). If you want to note bad channels you can edit the `epochs.info['bads']` object, which is a list, so be careful not to replace previous information, instead you can use `epochs.info['bads'].extend()`.

```python
epochs.plot()
# While in the plot, show inspection alternatives by pressing '?'
# eg. press 'h' to see the peak-to-peak amplitude

```

## EEG specific pre-processing
These steps are specific for EEG:
* The data for the removed channels are interpolated back in based on their neighbours
* The data is re-referenced to the average of all the channels

### Identify bad channels
The bad electrodes will mess up our analysis if left in. There are many ways to detect and deal with artifacts. You can use `epochs.plot()` and manually mark channels or epochs as bad by pressing on them. You can set a rejection thresholds based on the peak-to-peak amplitude.

```python
eeg = epochs.copy().pick_types(eeg=True)

eeg.average().plot_image()
eeg.average().plot_topo()

eeg.average().plot()

# Add the bad channels to the list
epochs.info['bads'].extend(['EEG027', 'EEG003', 'EEG008', 'EEG034'])
```

> **Question 1.4:** Which channels did you mark as bad and why (you can use figures to illustrate)?

















### Interpolate bad channels

The bad electrodes will be interpolated based on a combination of neighbouring electrodes. First, define the neighbours of each electrode and plot the neighbours.

```python

epochs.interpolate_bads()
epochs.average().plot()
```


```matlab
% Interpolate bad channels: Find neighbours
cfg             = [];
cfg.method      = 'triangulation';
cfg.senstype    = 'EEG'; % 
neighbours_EEG = ft_prepare_neighbours(cfg, preprocessed_data_EEG);

% plotting neighbours for inspection
cfg            = [];
cfg.neighbours = neighbours_EEG;
cfg.senstype   = 'EEG';
ft_neighbourplot(cfg, preprocessed_data_EEG);
```

![neighbours](figures/neighbours.jpg "EEG neighbours")

```python
%% Interpolate bad channels: the interpolation
cfg = [];
cfg.method          = 'spline';
cfg.neighbours      = neighbours_EEG;
cfg.badchannel      = badchannels;
cfg.senstype        = 'EEG';
interpolated_data = ft_channelrepair(cfg, preprocessed_data_EEG);
```

Plot the interpolated and non-interpolated EEG data. Can you spot the differences?

```python
cfg = [];
cfg.viewmode = 'butterfly';

ft_databrowser(cfg, preprocessed_data_EEG);     % not interpolated
ft_databrowser(cfg, interpolated_data);         % interpolated
```

### Re-reference to common average
The reference for the EEG signals has a great impact on how the signals appear. What reference you should use depends on several factors, such as what signals you are looking at, what is conventionally done, etc..

In the tutorial data, EEG was recorded with the FCz electrode as reference. This is almost above the sensorimotor areas so not ideal for looking at sensory potentials as we will in this tutorial. We, therefore, re-reference the EEG data. We will use a reference consisting of the average of all channels with `ft_preprocessing`. If we had only EEG data, we could have done this step already when we first imported the data above.

```python
%% Re-reference EEG
cfg = [];
cfg.reref            = 'yes';
cfg.refchannel       = 'all';

rereferenced_interpolated_data = ft_preprocessing(cfg, interpolated_data);
```

Use `ft_databrowser` to plot the re-referenced data to compare before and after.

## Append MEG and EEG data
From here on, the data can be combined into a single dataset and handled together. Use `ft_appenddata` to merge the MEG data, EEG data, and extra channels.

```python
%% Append all data
cfg = [];

preprocessed_data = ft_appenddata(cfg, preprocessed_data_MEG, rereferenced_interpolated_data, preprocessed_data_ExG);
```

## Remove bad trials
In this step, we remove bad trials. We will move the same bad trials from both MEG and EEG data. This is not necessary if we were analysis MEG and EEG separately. But in this tutorial we also want to compare MEG and EEG, so we want to have the exact same trials in both datasets.

If you have several experimental conditions, always collapse all conditions before summarizing and rejecting to avoid subjective bias. You have already done this if you imported trials all five triggers [1, 2, 4, 8, 16].

We will again use `ft_rejectvisual` and this time focus on the summary statistics over trials (bottom left plot). Mark the **trials** that you think are bad and ought to be removed. Drag across the "bad points" in the figure. Try to plot a few trials to see how the channels look. 
There are 

This is visually guided way to reject trials by removing those showing high variance. You can also remove specific artefacts by using the the `ft_artifact_xxx` and `ft_rejectartifact` functions. This is a more automated way to do it.


Find thresholds that is suitable for your data. You might have to play around a little to find reasonable thresholds.
For now, only focus on the channels.

```python

reject_criteria = dict(mag=5000e-15,    # 6000 fT
                       grad=4000e-13,   # 4000 fT/cm
                       eeg=600e-6,      # 600 µV
                    #  eog=300e-6       # 300 µV
                       ) 
flat_criteria = dict(mag=1e-15,         # 1 fT
                     grad=1e-13,        # 1 fT/cm
                     eeg=1e-6)          # 1 µV

epochs_clean = epochs.copy().

```

An another way to inspect data and remove artifacts is to use the `autoreject` package, which is installed separate from `mne`. Read more [here](https://autoreject.github.io/stable/index.html).


```Matlab
%% Remove bad trials
cfg = [];
cfg.method      = 'summary';
cfg.keepchannel = 'yes';
cfg.layout      = 'neuromag306all.lay';             % MEG layout for plotting

% Magnetometers
cfg.channel     = 'MEGMAG';
cleaned_data = ft_rejectvisual(cfg, preprocessed_data);

% Gradiomenters
cfg.channel     = 'MEGGRAD';
cleaned_data = ft_rejectvisual(cfg, cleaned_data);

% Electrodes
cfg.channel     = 'EEG*';
cfg.layout      = 'natmeg_customized_eeg1005.lay';  % Change to EEG layout

cleaned_data = ft_rejectvisual(cfg, cleaned_data);
```

## Adjust for the offset between trigger and the actual delivery of the stimulation

The trigger for the delivery of the tactile stimulation is sent with millisecond precision to the stimulation device and the MEG data acquisition software. However, because we cannot have electrical parts within the magnetically shield room, the stimulus is powered by pressurized air. This means that there is a delay from the device that received the trigger to the actual delivery of the sensory stimulation. The delay has been measured with an accelerometer to 41 ms. There is no way to know this from the data, and if we did not know, we might think that this subject had oddly slow event-related activity.

As a final step, we correct the onset of the trial, i.e. move the offset (0 ms) 41 ms forward in time. This is a processing step specific to this data due to the specific stimulation equipment we used in the experiment. **Warning: do not uncritically copy/paste this step into your analysis scripts for your own data**.

```python
%% Adjust offset
cfg = [];
cfg.offset = -41;   % Number of samples

cleaned_data_adjust = ft_redefinetrial(cfg, cleaned_data);
```

## Downsample the data

Downsample from 1000 Hz to 200 Hz. This speeds up processing time but decreases the number of frequencies we can look at. Note that if you already did this before, you should not do this again.

```Matlab
%% Downsample
cfg = [];
cfg.resamplefs = 200;

cleaned_downsampled_data = ft_resampledata(cfg, cleaned_data_adjust);
```

## Save
Now we are done with the basic pre-processing, so this is a good time to save the data. You will use this dataset in the following tutorials.

```Matlab
%% Save data
save(fullfile(output_path, 'cleaned_downsampled_data'), 'cleaned_downsampled_data', '-v7.3'); disp('done');
```

## Advanced pre-processing: independent component analysis (ICA)
Independent component analysis (ICA) is a decomposition method that breaks data into statistically independent components. ICA is useful for identifying patterns of activity that occur regularly in the data. ICA has many applications in MEG/EEG analysis. The most common use is to identify activity related to eye-blinks and heart-beats. These are part of the signal that we (usually) do not want.

The code below shows how to remove eye-blinks and heart-beats from the MEG data.

```python
%% Run ICA
cfg = [];
cfg.channel     = 'meg';
cfg.method      = 'fastica';    % Use "fastica" algorithm
cfg.numcomponent = 40;          % Can be up to N channels
comp           = ft_componentanalysis(cfg, cleaned_downsampled_data);

%% Save comp (good for bookkeeping)
save(fullfile(meg_dir, 'comp.mat'), 'comp')
```

Plot the components:

```python
%% Plot ICA
% Topography view (split in two figures)
cfg = [];
cfg.layout      = 'neuromag306all.lay';
cfg.marker      = 'off';
cfg.component   = [1:20];               
figure; ft_topoplotIC(cfg, comp)
cfg.component   = [21:40];      
figure; ft_topoplotIC(cfg, comp)

% Time-series view (split in two figures)
cfg = [];
cfg.viewmode    = 'component';
cfg.layout      = 'neuromag306all.lay';
cfg.blocksize   = 10;
cfg.channel     = [1:20];
ft_databrowser(cfg, comp)
cfg.channel     = [21:40];
ft_databrowser(cfg, comp)
```

See if you can find components that correspond to eye-blinks and heart-beats from the component topographies and component time-series? When you have found the components that correspond to eye-blinks and heart-beats, you can remove them with `ft_rejectcomponets`:

```python
%% Remove components
reject_comp = [1, 3, 7, 11];    % Write the index of the components you want to remove

% Remove components
cfg = [];
cfg.component   = reject_comp;
cfg.channel     = 'MEG';
cfg.updatesens  = 'yes';
icacleaned_downsampled_data = ft_rejectcomponent(cfg, comp, cleaned_downsampled_data);
```

### Semi-automatic detection of ECG components
The following code will find the components that show similarity to the ECG signal. It uses FieldTrips automatic detection of ECG artefacts, then makes epochs around the artefacts. It requires manual input for defining the duration of the QRS-complex in the ECG.

```python
%% Find ECG components
% Find ECG artifacts
cfg = [];
cfg.continuous            = 'no';
cfg.artfctdef.ecg.pretim  = 0.25;
cfg.artfctdef.ecg.psttim  = 0.50;
cfg.channel               = 'ECG';
cfg.artfctdef.ecg.inspect = 'ECG';
[~, artifact] = ft_artifact_ecg(cfg, cleaned_downsampled_data);
```

FieldTrip first asks if you want to `keep the current value (y/n) ?` In the a new figure you can see peaks in the ECG signal and the threshold. Accept (`y`) or provide a new cut-off value.

Next, FieldTrip asks you to define the duration of the QRS-complex. Change values, so the red shading covers the QRS-complex.

When it is done, continue with the following:

```python
% Make artifact epochs
cfg = [];
cfg.dftfilter  = 'yes';
cfg.demean     = 'yes';
cfg.trl        = [artifact zeros(size(artifact,1), 1)];
temp = ft_redefinetrial(cfg, cleaned_downsampled_data);

% Re-arrange data
cfg.channel    = 'MEG*';
data_ecg = ft_selectdata(cfg, temp);
cfg.channel    = 'ECG';
ecg = ft_selectdata(cfg, temp);
ecg.channel{:} = 'ECG';         % renaming for bookkeeping

% Remove residual linenoise in electric channel.
cfg = [];
cfg.dftfilter       = 'yes';
cfg.dftfreq         = [50, 100, 150];
ecg = ft_preprocessing(cfg, ecg);

% decompose the ECG-locked datasegments (using the únmixing matrix from comp)
cfg = [];
cfg.unmixing  = comp.unmixing;
cfg.topolabel = comp.topolabel;
comp_ecg = ft_componentanalysis(cfg, data_ecg);

% append the ecg channel to the data structure;
comp_ecg = ft_appenddata([], ecg, comp_ecg);

% average the components timelocked to the QRS-complex
cfg = [];
timelock = ft_timelockanalysis(cfg, comp_ecg);

% Plot
figure
subplot(2,1,1); plot(timelock.time, timelock.avg(1,:)); title('ECG')
subplot(2,1,2); plot(timelock.time, timelock.avg(2:end,:));  title('ICA comp')
```

Can you see a hint of components that might be correlated with the ECG?

![](figures/ica_ecg.png "ECG and ICA")

Now we find the components most like the ECG:

```Matlab
%% Find ECG components
% Define cutoff
cutoff = 0.5;           % Between 0-1 (analogue to a correlation coefficient)

% compute a frequency decomposition of all components and the ECG
cfg = [];
cfg.method     = 'mtmfft';
cfg.output     = 'fourier';
cfg.foilim     = [0 100];
cfg.taper      = 'hanning';
cfg.pad        = 'maxperlen';
freq = ft_freqanalysis(cfg, comp_ecg);

% compute coherence between all components and the ECG
cfg = [];
cfg.channelcmb = {'all' 'ECG'};
cfg.method     = 'coh';
fdcomp = ft_connectivityanalysis(cfg, freq);

% Find ECG components
maxcoh = max(fdcomp.cohspctrm, [], 2);
ecg_comp_idx = find(maxcoh > cutoff);
```
The variable `ecg_comp_idx` is the indexes of the components. Take a look at them:

```python
% look at the coherence spectrum between all components and the ECG
figure;
subplot(3,1,1); plot(fdcomp.freq, abs(fdcomp.cohspctrm)); hold on
plot([min(fdcomp.freq),max(fdcomp.freq)],[cutoff, cutoff], 'k--')
title('ECG'); xlabel('freq'); ylabel('coh');
subplot(3,1,2); imagesc(abs(fdcomp.cohspctrm));
xlabel('freq'); ylabel('comp');
subplot(3,1,3);
maxcoh = max(fdcomp.cohspctrm, [], 2);
foo = find(~(maxcoh > cutoff));
bp = bar(1:length(maxcoh), diag(maxcoh), 'stacked');
set(bp(foo),'facecolor','w'); set(bp(ecg_comp_idx),'facecolor','r')
axis([0.5, length(maxcoh)+0.5, 0, 1]); xlabel('comp'); ylabel('coh');

% View marked component(s)
cfg = [];
cfg.channel     = ecg_comp_idx; % components to be plotted
cfg.viewmode    = 'component';
cfg.layout      = 'neuromag306all.lay'; % specify the layout file that should be used for plotting
ft_databrowser(cfg, comp)
```

Summary:

![](figures/ica_ecg2.png "selected ECG components")

Topography and time-series view with `ft_databrowser` (use arrows to scroll though trials):

![](figures/ica_ecg3.png "selected ECG components topography")

### semi-automatic detection of EOG components
The following code will find the components that show similarity to the EOG signal to remove eye-blinks. It uses Fieldtrip's automatic detection of EOG artefacts. It is similar to the semi-automatic detection of ECG components, but for EOG, we do the procedure twice to for each EOG channel (H/VEOG).

```python
%% Find EOG components
% Find EOG artifacts
cfg = [];
cfg.continuous            = 'no';
cfg.channel               = 'EOG';
[~, artifact] = ft_artifact_eog(cfg, cleaned_downsampled_data);

% Make artifact epochs
cfg = [];
cfg.dftfilter  = 'yes';
cfg.demean     = 'yes';
cfg.trl        = [artifact zeros(size(artifact,1), 1)];
temp = ft_redefinetrial(cfg, cleaned_downsampled_data);
    
% Re-arrange data    
cfg.channel    = 'MEG*';
data_eog = ft_selectdata(cfg, temp);
cfg.channel    = 'EOG';
eog = ft_selectdata(cfg, temp);
eog.channel{:} = 'EOG';         % renaming for bookkeeping
    
 % Remove residual linenoise in electric channel.
cfg = [];
cfg.dftfilter  = 'yes';
cfg.dftfreq    = [50, 100, 150];
eog = ft_preprocessing(cfg, eog);

% decompose the EOG epochs into components
cfg = [];
cfg.unmixing  = comp.unmixing;
cfg.topolabel = comp.topolabel;
comp_eog = ft_componentanalysis(cfg, data_eog);

% append the EOG channel to the data structure
comp_eog = ft_appenddata([], eog, comp_eog);

% Define cutoff
cutoff = 0.5;           % Between 0-1 (analogue to a correlation coefficient)

% compute a frequency decomposition of all components and the EOG
cfg = [];
cfg.method     = 'mtmfft';
cfg.output     = 'fourier';
cfg.foilim     = [0 100];
cfg.taper      = 'hanning';
cfg.pad        = 'maxperlen';
freq = ft_freqanalysis(cfg, comp_eog);

% compute coherence between all components and the E0G
cfg = [];
cfg.method     = 'coh';
cfg.channelcmb = {'comp*' 'EOG001'};
fdcomp_eog1 = ft_connectivityanalysis(cfg, freq);
cfg.channelcmb = {'comp*' 'EOG002'};
fdcomp_eog2 = ft_connectivityanalysis(cfg, freq);

% Find EOG components
maxcoh = max(fdcomp_eog1.cohspctrm, [], 2);
eog1_comp_idx = find(maxcoh > cutoff);
maxcoh = max(fdcomp_eog2.cohspctrm, [], 2);
eog2_comp_idx = find(maxcoh > cutoff);
```

The variables `eog1_comp_idx` and `eog2_comp_idx` are the indexes of the components for respectively H/VEOG. Take a look at them:

```python
% look at the coherence spectrum between all components and the EOG
figure;
subplot(3,2,1); title('EOG001'); xlabel('freq'); ylabel('coh');
plot(fdcomp_eog1.freq, abs(fdcomp.cohspctrm)); hold on
plot([min(fdcomp.freq),max(fdcomp.freq)],[cutoff, cutoff], 'k--');
subplot(3,2,2); title('EOG002'); xlabel('freq'); ylabel('coh');
plot(fdcomp_eog2.freq, abs(fdcomp.cohspctrm)); hold on
plot([min(fdcomp.freq),max(fdcomp.freq)],[cutoff, cutoff], 'k--');
subplot(3,2,3); xlabel('freq'); ylabel('comp');
imagesc(abs(fdcomp_eog1.cohspctrm));
subplot(3,2,4); xlabel('freq'); ylabel('comp');
imagesc(abs(fdcomp_eog2.cohspctrm));
subplot(3,2,5); xlabel('comp'); ylabel('coh');
maxcoh = max(fdcomp_eog1.cohspctrm, [], 2);
foo = find(~(maxcoh > cutoff));
bp = bar(1:length(maxcoh), diag(maxcoh), 'stacked');
set(bp(foo),'facecolor','w'); set(bp(eog1_comp_idx),'facecolor','r');
axis([0.5, length(maxcoh)+0.5, 0, 1]);
subplot(3,2,6); xlabel('comp'); ylabel('coh');
maxcoh = max(fdcomp_eog2.cohspctrm, [], 2);
foo = find(~(maxcoh > cutoff));
bp = bar(1:length(maxcoh), diag(maxcoh), 'stacked');
set(bp(foo),'facecolor','w'); set(bp(eog2_comp_idx),'facecolor','r'); 
axis([0.5, length(maxcoh)+0.5, 0, 1]);

% View marked component(s)
cfg = [];
cfg.channel     = ecg_comp_idx; 	   % components to be plotted
cfg.viewmode    = 'component';
cfg.layout      = 'neuromag306all.lay'; % specify the layout file that should be used for plotting
ft_databrowser(cfg, comp)
```

Summary:

![EOG_comp](figures/ica_eog1.png "EOG components")

Topography and time-series view with `ft_databrowser` (use arrows to scroll though trials):

![EOG_comp_topo](figures/ica_eog2.png "EOG components topography and time-series")

### Reject the marked components
Finally we remove the ECG and EOG componets:

```python
%% Remove components
% Make a list of all "bad" components
reject_comp = unique([ecg_comp_idx; eog1_comp_idx; eog2_comp_idx]);
    
% Remove components
cfg = [];
cfg.component   = reject_comp;
cfg.channel     = 'MEG';
cfg.updatesens  = 'yes';
icacleaned_downsampled_data = ft_rejectcomponent(cfg, comp, raw_flt);
```

Use `ft_databrowser` to inspect the difference between `cleaned_downsampled_data` and `icacleaned_downsampled_data` to see what difference this made.

### Save data
Finally, save the data:

```python
%% Save
save(fullfile(output_path, 'icacleaned_downsampled_data'), 'icacleaned_downsampled_data'); disp('done');
```

## End of Tutorial 1a...
Congratulations, you have now imported and prepared MEG and EEG. The tutorial continues in **Tutorial 1B: evoked responses**, where you will continue the processing on the data to get the evoked responses.

![teaser](figures/timelockeds_multiplot3.png)
